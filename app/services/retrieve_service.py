"""检索层：查询向量库，产出带来源信息的候选上下文。

职责边界：
- 只负责「找得到什么」，不做任何生成；
- 召回策略集中在这里，方便后续扩展混合检索、重排而不动生成层。

策略：
- similarity（默认）：余弦相似度排序 + 阈值过滤，相关性优先；
- mmr（备选）：最大边际相关性，先多召回 fetch_k 个候选，再挑出彼此不重复的 k 个，
  适合"同一段内容被反复切碎、召回结果高度雷同"的场景。MMR 走多样性打分，
  不返回距离，因此不套相似度阈值。
"""

import time
from typing import List, Optional

from langchain_core.documents import Document

from app.core.error_codes import CODE_ERROR, ERR_KB_NOT_FOUND
from app.core.exceptions import KbNotFoundError, ParamError
from app.services.index_service import IndexService
from utils.logger import get_logger

logger = get_logger("app.services.retrieve_service")

# 余弦相似度阈值：Chroma 的 cosine 空间返回的是距离（1 - 相似度），
# 这里按「相似度」设定门槛，越小越宽松。
DEFAULT_SCORE_THRESHOLD = 0.5
MAX_TOP_K = 20
MAX_FETCH_K = 100

MODE_SIMILARITY = "similarity"
MODE_MMR = "mmr"
SUPPORTED_MODES = {MODE_SIMILARITY, MODE_MMR}


class RetrieveService:
    @staticmethod
    def _normalize_top_k(top_k: Optional[int]) -> int:
        if not top_k or top_k < 1:
            return 4
        return min(int(top_k), MAX_TOP_K)

    @staticmethod
    def _normalize_fetch_k(fetch_k: Optional[int], top_k: int) -> int:
        """MMR 的候选池必须大于 k，否则没有多样性可言"""
        base = int(fetch_k) if fetch_k else top_k * 4
        base = max(base, top_k + 1)
        return min(base, MAX_FETCH_K)

    @staticmethod
    def _validate_mode(mode: str) -> str:
        mode = (mode or MODE_SIMILARITY).lower().strip()
        if mode not in SUPPORTED_MODES:
            raise ParamError(
                f"不支持的检索策略 {mode!r}，可选：{sorted(SUPPORTED_MODES)}", code=CODE_ERROR
            )
        return mode

    @classmethod
    def ensure_knowledge_base_exists(cls, knowledge_base: str) -> None:
        """库不存在直接报错，避免 Chroma 静默建空库、把问题拖到生成阶段。

        这里必须用 has_collection（只读 collection 列表）而不是 list_knowledge_bases：
        后者会拉全库所有分片的 metadata，等于每次提问都做一次全库扫描。
        """
        if not IndexService.has_collection(knowledge_base):
            raise KbNotFoundError(
                f"知识库 {knowledge_base!r} 不存在，请先调用 /api/rag/upload 上传文档。",
                code=ERR_KB_NOT_FOUND,
            )

    @classmethod
    def _search_similarity(
        cls,
        vector_store,
        question: str,
        k: int,
        score_threshold: float,
    ) -> List[Document]:
        # 注意：similarity_search_with_score 返回的是「距离」，cosine 空间下 0 最近、2 最远
        hits = vector_store.similarity_search_with_score(question, k=k)

        contexts: List[Document] = []
        for doc, distance in hits:
            similarity = 1.0 - float(distance)
            if similarity < score_threshold:
                continue  # 明显不相关，宁可让生成层兜底也不喂噪声
            doc.metadata = dict(doc.metadata)
            doc.metadata["retrieval_score"] = round(similarity, 4)
            doc.metadata["retrieval_mode"] = MODE_SIMILARITY
            contexts.append(doc)
        return contexts

    @classmethod
    def _search_mmr(
        cls,
        vector_store,
        question: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
    ) -> List[Document]:
        docs = vector_store.max_marginal_relevance_search(
            question, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
        contexts: List[Document] = []
        for doc in docs:
            doc.metadata = dict(doc.metadata)
            doc.metadata["retrieval_mode"] = MODE_MMR
            doc.metadata["mmr_lambda"] = lambda_mult
            contexts.append(doc)
        return contexts

    @classmethod
    def get_contexts(
        cls,
        question: str,
        knowledge_base: str,
        top_k: Optional[int] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        mode: str = MODE_SIMILARITY,
        fetch_k: Optional[int] = None,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        """召回上下文片段。

        mode="similarity"（默认）：按相似度降序，低于阈值直接丢弃，分数写进 metadata；
        mode="mmr"（备选）：先取 fetch_k 个候选，再按 lambda_mult 平衡相关性与多样性，
                            不做阈值过滤，返回结果不含 retrieval_score。
        """
        question = (question or "").strip()
        if not question:
            raise ParamError("question 不能为空", code=CODE_ERROR)

        mode = cls._validate_mode(mode)
        if not 0.0 <= float(lambda_mult) <= 1.0:
            raise ParamError(
                "lambda_mult 需在 0~1 之间（越大越偏相关性，越小越偏多样性）", code=CODE_ERROR
            )

        cls.ensure_knowledge_base_exists(knowledge_base)
        k = cls._normalize_top_k(top_k)
        vector_store = IndexService.get_vector_store(knowledge_base)

        start = time.perf_counter()
        if mode == MODE_MMR:
            contexts = cls._search_mmr(
                vector_store, question, k,
                fetch_k=cls._normalize_fetch_k(fetch_k, k),
                lambda_mult=float(lambda_mult),
            )
        else:
            contexts = cls._search_similarity(vector_store, question, k, score_threshold)
        cls._log_retrieval(knowledge_base, question, mode, k, contexts, start)
        return contexts

    @staticmethod
    def _log_retrieval(
        knowledge_base: str,
        question: str,
        mode: str,
        k: int,
        contexts: List[Document],
        start: float,
    ) -> None:
        """检索结果留痕：命中数与最高分是判断「阈值/分片大小调得对不对」的唯一依据。

        零命中单独走 WARNING——它会直接让生成层走兜底话术，是用户看到「答不上来」的
        最常见原因，不该淹没在 INFO 里。
        """
        cost_ms = (time.perf_counter() - start) * 1000
        # 问题文本截断：日志要能认出「问的是什么」，但不必整段入库（也避免日志暴涨）
        q = question if len(question) <= 40 else question[:40] + "…"
        if not contexts:
            logger.warning(
                "检索无命中 | kb=%s mode=%s k=%d | %.1fms | q=%s",
                knowledge_base, mode, k, cost_ms, q,
            )
            return
        # 结果按相关性降序，第一条即最高分；MMR 不返回距离，故可能取不到
        best = contexts[0].metadata.get("retrieval_score", "-")
        logger.info(
            "检索完成 | kb=%s mode=%s k=%d 命中=%d 最高分=%s | %.1fms | q=%s",
            knowledge_base, mode, k, len(contexts), best, cost_ms, q,
        )
