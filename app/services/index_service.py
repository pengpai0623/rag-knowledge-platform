import errno
import os
import time
from contextlib import contextmanager
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.error_codes import (
    ERR_KB_DIM_MISMATCH,
    ERR_KB_INDEX_BROKEN,
    ERR_KB_NOT_FOUND,
    ERR_KB_STORAGE,
    ERR_KB_STORAGE_FULL,
    ERR_KB_VERSION_MISMATCH,
)
from app.core.exceptions import KbNotFoundError, KbStorageError, RAGBaseError
from app.services.model_factory import ModelFactory
from utils.logger import get_logger

# 向量库持久化目录：默认与 examples/ 保持同一份库，避免出现两套库
# __file__ = app/services/index_service.py -> 上三级就是项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Docker 部署时用 CHROMA_PERSIST_DIR 指向挂载卷（docker-compose 里
# ./chroma_data:/app/chroma_data 对应 CHROMA_PERSIST_DIR=/app/chroma_data），
# 容器重启/删除重建后向量数据仍在宿主机；本地开发不设该变量，走默认路径。
# （model_factory 在本模块顶部被导入时已执行 load_dotenv，写在 .env 里也生效）
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR") or os.path.join(
    PROJECT_ROOT, "vector_db", "chroma_rag_db"
)

# 巡检接口扫描 metadata 时的分页大小：只累积统计量、不保留原始行，避免大库一次性拉全量
_INSPECT_PAGE = 1000

logger = get_logger("app.services.index_service")


class IndexService:
    """向量库（Chroma）访问层。

    异常策略（三类，按异常类型分，不按调用位置分）：
    1. 拿不到客户端（依赖缺失 / 目录无权 / 库文件损坏） → _persistent_client 统一 2002；
    2. collection 名不存在（正常业务状态）                → 2001，由 has_collection 或
       _collection 捕获 chromadb.errors.NotFoundError 给出，含并发删除的窗口期；
    3. 其它操作期故障                                    → _translate_chroma 只翻译
       「调用方有补救动作」的几类（2003 维度 / 2004 索引 / 2005 磁盘满 / 2006 版本），
       识别不了的一律裸抛 → 全局兜底 500（固定文案 + 堆栈 + rid）。
    """

    _embed_model = None

    @classmethod
    def get_embedding(cls):
        """单例加载 Embedding 模型。

        必须与库内已有向量同源：现有 vector_db/chroma_rag_db 是 text-embedding-v3（1024 维），
        换成别的后端（如 BGE-small-zh 512 维）会导致写入维度报错、检索结果失真。
        """
        if cls._embed_model is None:
            cls._embed_model = ModelFactory.get_embedding()
        return cls._embed_model

    @classmethod
    def get_vector_store(cls, knowledge_base: str) -> Chroma:
        """获取指定知识库的向量库实例（新建集合时统一用 cosine 距离，与既有库一致）"""
        return Chroma(
            collection_name=knowledge_base,
            embedding_function=cls.get_embedding(),
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def add_documents(cls, knowledge_base: str, docs: List[Document]) -> int:
        """文档批量入库，返回分片数量"""
        start = time.perf_counter()
        with cls._guard(f"写入知识库 {knowledge_base}"):
            vector_store = cls.get_vector_store(knowledge_base)
            vector_store.add_documents(docs)
        # 逐批调 embedding，是上传链路里最慢的一步；单独记耗时才能和解析耗时区分瓶颈
        logger.info(
            "向量写入完成 | kb=%s 分片=%d | %.1fms",
            knowledge_base, len(docs), (time.perf_counter() - start) * 1000,
        )
        return len(docs)

    # ------------------------------------------------------- 异常翻译（唯一出口）
    @classmethod
    def _persistent_client(cls):
        """打开向量库客户端。这是「存储不可用」（2002）唯一的产生点。

        import 失败（依赖缺失 / protobuf 不匹配）与打开失败（目录无权、库文件损坏、
        被其它进程占用）对调用方是同一件事：服务端存储用不了。明细只进日志，
        响应体给固定文案，避免把 protobuf 版本号、绝对路径带出去。
        """
        try:
            import chromadb
            return chromadb.PersistentClient(path=PERSIST_DIR)
        except Exception as e:
            logger.error("向量库不可用：%s", PERSIST_DIR, exc_info=True)
            wrapped = cls._translate_chroma(e, "打开向量库") or KbStorageError(
                "向量库暂时不可用，请稍后重试", code=ERR_KB_STORAGE
            )
            raise wrapped from e

    @staticmethod
    def _translate_chroma(e: Exception, where: str) -> Optional[RAGBaseError]:
        """把可识别的向量库故障翻成业务异常；识别不了返回 None（由调用方裸抛）。

        只翻译「调用方有补救动作」的那几类：
        - 维度不匹配   → 换回原 embedding 模型，或清库重灌
        - 版本不兼容   → 清库重灌
        - 索引/内部错  → 清库重灌
        - 磁盘满       → 清磁盘后重试
        InvalidUUIDError / InvalidArgumentError / DuplicateIDError 属于我们传参写错或
        并发写重，是 bug，保持裸抛——用业务码会把 bug 掩盖成「预期内的业务失败」。
        """
        # 磁盘满没有独立类型，且可能被 chromadb 包成 InternalError，所以先看 message
        text = str(e).lower()
        if (isinstance(e, OSError) and e.errno == errno.ENOSPC) or \
                "disk is full" in text or "no space left" in text:
            return KbStorageError(
                f"磁盘空间不足，写入失败：{where}", code=ERR_KB_STORAGE_FULL
            )

        try:
            from chromadb import errors as chroma_errors
        except ImportError:
            return None  # chromadb 本身导入失败 → 由调用方兜成 2002

        if isinstance(e, chroma_errors.InvalidDimensionException):
            return KbStorageError(
                "向量维度与库内已有数据不一致，请使用原 embedding 模型或清空知识库后重灌",
                code=ERR_KB_DIM_MISMATCH,
            )
        if isinstance(e, chroma_errors.VersionMismatchError):
            return KbStorageError(
                "向量库版本与当前服务不兼容，需要清空知识库后重灌",
                code=ERR_KB_VERSION_MISMATCH,
            )
        if isinstance(e, chroma_errors.InternalError):
            return KbStorageError(
                f"向量索引异常，建议重建知识库：{where}", code=ERR_KB_INDEX_BROKEN
            )
        return None

    @classmethod
    @contextmanager
    def _guard(cls, where: str):
        """向量库操作的统一翻译层。

        未识别异常**不在这里重复打日志**：它会走到 handlers 的兜底 handler，
        那里已经记录了 `未捕获异常 | 类型: 消息` + 完整堆栈 + rid。
        """
        try:
            yield
        except RAGBaseError:
            raise  # 已经翻译过（2001/2002/...），不要二次包装
        except Exception as e:
            wrapped = cls._translate_chroma(e, where)
            if wrapped is None:
                raise
            raise wrapped from e

    @classmethod
    def _collection(cls, client, knowledge_base: str):
        """按名取 collection：不存在 → 2001，其余交给 _translate_chroma / 裸抛。

        用它替代散落的 get_collection：has_collection 返回 True 之后、真正取用时
        collection 可能已被并发删除（窗口期），那时的 NotFoundError 语义仍是「不存在」，
        不该变成 500。
        """
        try:
            return client.get_collection(knowledge_base)
        except Exception as e:
            from chromadb import errors as chroma_errors
            if isinstance(e, chroma_errors.NotFoundError):
                raise KbNotFoundError(
                    f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND
                ) from e
            wrapped = cls._translate_chroma(e, knowledge_base)
            if wrapped is None:
                raise
            raise wrapped from e

    # ------------------------------------------------------------ 查询
    @classmethod
    def list_knowledge_bases(cls) -> List[dict]:
        """列出所有知识库。

        约定：一个 Chroma collection 就是一个知识库，collection 名即接口里的 knowledge_base，
        与 get_vector_store() 的入参保持一致。

        只读 collection 的统计信息与文档 metadata，不加载 Embedding 模型：
        - doc_count  ：去重后的来源文档数（按 metadata.source 统计）
        - chunk_count：该库的向量分片总数（collection.count() 是整个 collection 的记录数）

        单个库读失败不再「跳过并当成 0」：那是运维入口，宁可整页报错，
        也不要给出「库里少了一个」的错觉。
        """
        if not os.path.exists(PERSIST_DIR):
            return []

        client = cls._persistent_client()
        collections = []
        with cls._guard("列出知识库"):
            for item in client.list_collections():
                # chromadb 1.x list_collections() 返回 Collection 对象，旧版返回名称字符串
                name = getattr(item, "name", item)
                if not isinstance(name, str):
                    continue
                collection = client.get_collection(name)
                chunk_count = collection.count()
                metadatas = collection.get(include=["metadatas"])["metadatas"] or []
                # 缺少 source 的旧数据统一归到未知来源，避免把"有内容"误报成 0 个文档
                sources = {(m or {}).get("source") or "<未知来源>" for m in metadatas}
                collections.append({
                    "name": name,
                    "doc_count": len(sources) if metadatas else 0,
                    "chunk_count": chunk_count,
                })
        return collections

    @classmethod
    def inspect_collections(cls, knowledge_base: str = "") -> List[dict]:
        """巡检向量库：列出 collection 及其内部的来源（source）分布。

        与 list_knowledge_bases 的分工：
        - list_knowledge_bases 是业务视图（库名 / 文档数 / 分片数）；
        - inspect_collections 是排障视图（collection 级 metadata + 每个 source 的分片数
          与解析版本），用来回答「这个库里到底有哪些文档、还有没有 v1 老数据残留」。

        只读 metadata、不加载 Embedding；按 _INSPECT_PAGE 条一页滚动扫描，只累积统计量，
        内存占用与分片总数无关（代价是耗时随分片数线性增长，属排障接口的预期成本）。
        parser_version 缺失的老数据显示为 "<未标记>"，source 缺失显示为 "<未知来源>"。
        """
        if not os.path.exists(PERSIST_DIR):
            return []

        client = cls._persistent_client()
        result: List[dict] = []
        with cls._guard("巡检向量库"):
            for item in client.list_collections():
                # chromadb 1.x 返回 Collection 对象，旧版返回名称字符串
                name = getattr(item, "name", item)
                if not isinstance(name, str):
                    continue
                if knowledge_base and name != knowledge_base:
                    continue
                collection = client.get_collection(name)

                stats: dict = {}
                offset = 0
                while True:
                    page = collection.get(
                        include=["metadatas"], limit=_INSPECT_PAGE, offset=offset
                    )
                    ids = page.get("ids") or []
                    if not ids:
                        break
                    for meta in page.get("metadatas") or []:
                        meta = meta or {}
                        source = meta.get("source") or "<未知来源>"
                        entry = stats.setdefault(
                            source,
                            {"source": source, "chunk_count": 0, "parser_versions": set()},
                        )
                        entry["chunk_count"] += 1
                        entry["parser_versions"].add(meta.get("parser_version") or "<未标记>")
                    offset += len(ids)
                    if len(ids) < _INSPECT_PAGE:
                        break

                result.append({
                    "name": name,
                    "chunk_count": collection.count(),
                    "doc_count": len(stats),
                    "metadata": {str(k): v for k, v in (collection.metadata or {}).items()},
                    "sources": [
                        {
                            "source": entry["source"],
                            "chunk_count": entry["chunk_count"],
                            "parser_versions": sorted(entry["parser_versions"]),
                        }
                        for entry in sorted(
                            stats.values(), key=lambda s: (-s["chunk_count"], s["source"])
                        )
                    ],
                })
        return result

    @classmethod
    def has_collection(cls, knowledge_base: str) -> bool:
        """知识库（collection）是否存在。只查元数据，不加载 Embedding 模型。

        返回 False 只有一种含义：collection 列表里确实没有它（含首次部署、目录不存在）。
        向量库读不出来时抛业务异常，不再吞成 False——否则故障会伪装成
        2001「知识库不存在」，前端和日志都看不出真相。
        """
        if not os.path.exists(PERSIST_DIR):
            return False
        client = cls._persistent_client()
        with cls._guard("读取知识库列表"):
            names = [getattr(c, "name", c) for c in client.list_collections()]
        return knowledge_base in names

    @classmethod
    def has_source(
        cls,
        knowledge_base: str,
        file_sha256: str,
        parser_version: str,
        clean_policy: str = "",
        splitter_version: str = "",
    ) -> bool:
        """库里是否已存在「同内容 + 同解析版本 + 同清洗策略 + 同切分版本」的分片。

        必须带 parser_version：解析策略升级后（v1 → v2），同一份文件虽然 sha 相同，
        但需要用新解析器重灌，不能因为「内容没变」就跳过。
        同理必须带 clean_policy 与 splitter_version：换清洗策略或换切分逻辑后，
        分片文本/边界完全不同，skip 掉就等于改动没生效。
        v2/v3 之前的老数据没有这些字段，$eq 对缺失字段不匹配 → 不跳过、重新入库，
        宁可多算一次 embedding，也不要静默沿用旧策略的切片。
        只查一条记录，不拉全量 metadata。
        """
        if not file_sha256 or not cls.has_collection(knowledge_base):
            return False
        conditions = [
            {"file_sha256": {"$eq": file_sha256}},
            {"parser_version": {"$eq": parser_version}},
        ]
        if clean_policy:
            conditions.append({"clean_policy": {"$eq": clean_policy}})
        if splitter_version:
            conditions.append({"splitter_version": {"$eq": splitter_version}})
        client = cls._persistent_client()
        with cls._guard(f"查询 {knowledge_base} 已入库内容"):
            collection = cls._collection(client, knowledge_base)
            result = collection.get(where={"$and": conditions}, limit=1)
        return bool(result.get("ids"))

    @classmethod
    def source_chunk_ids(
        cls, knowledge_base: str, source: str, file_sha256: str = ""
    ) -> list:
        """列出待覆盖文档的旧分片 id（双身份：source 或 file_sha256，不分版本）。

        用它做「先写后删」：先新增分片，成功后再按 id 精确删掉旧版本的，
        写失败时旧数据仍完整，不像「先删后写」那样一失败就丢。

        为什么除了 source 还要按 sha256 匹配：source 是文件名，改个名重传就变
        （`报告.pdf` → `报告-最终版.pdf`），只认 source 会查不到旧 id，旧分片永久
        残留，检索时同一段话被命中两次、doc_count 也多算一个文档。
        v2 之前的老数据没有 file_sha256 字段，$eq 对缺失字段不匹配，自动被排除。
        """
        if not cls.has_collection(knowledge_base):
            return []
        conditions = [{"source": {"$eq": source}}]
        if file_sha256:
            conditions.append({"file_sha256": {"$eq": file_sha256}})
        where = conditions[0] if len(conditions) == 1 else {"$or": conditions}
        client = cls._persistent_client()
        with cls._guard(f"读取 {knowledge_base}/{source} 的已入库分片"):
            collection = cls._collection(client, knowledge_base)
            result = collection.get(where=where, include=["metadatas"])
        ids = list(result.get("ids") or [])
        # 命中了别的来源，说明是「同内容改名重传」，删掉这些分片属预期行为，但要留痕
        stale_sources = {
            (m or {}).get("source")
            for m in (result.get("metadatas") or [])
        } - {source, None}
        if stale_sources:
            logger.warning(
                "同内容旧分片将被覆盖：%s 命中其他来源 %s（共 %d 个分片）",
                source, sorted(stale_sources), len(ids),
            )
        return ids

    # ------------------------------------------------------------ 删除相关
    @classmethod
    def delete_ids(cls, knowledge_base: str, ids: list) -> int:
        """按 id 删除分片，返回实际删除数量。

        直接用 ids 长度，不再用 collection.count() 前后差：count() 数的是整个
        collection 的记录数，并发删别的文档时两个请求会把对方的删除量算进自己头上。
        """
        if not ids:
            return 0
        client = cls._persistent_client()
        with cls._guard(f"删除 {knowledge_base} 的旧分片"):
            cls._collection(client, knowledge_base).delete(ids=ids)
        return len(ids)

    @classmethod
    def delete_collection(cls, knowledge_base: str) -> bool:
        """清空整个知识库，返回是否真的删掉了一个 collection。

        必要性：按 source 删除只能清「身份对得上」的分片，解析策略升级后
        需要整库重灌，没有清库入口就永远清不干净。
        """
        if not cls.has_collection(knowledge_base):
            return False
        client = cls._persistent_client()
        with cls._guard(f"删除知识库 {knowledge_base}"):
            client.delete_collection(knowledge_base)
        # 破坏性操作留痕：事后追「这个库是谁什么时候清掉的」只能靠这行
        logger.warning("整库删除 | kb=%s", knowledge_base)
        return True

    @classmethod
    def delete_by_source(cls, knowledge_base: str, source: str) -> int:
        """按来源文件名删除该文档的全部向量分片，返回实际删除数量。

        - 知识库不存在 → KbNotFoundError（code 2001）
        - 存储层故障   → KbStorageError（2002/2003/2004/2005/2006），识别不了的落 500
        - 匹配不到分片 → 返回 0，由调用方如实上报，不谎报成功

        先按 source 取 id 再按 id 删：返回值就是「真正命中的分片数」，
        不依赖 collection.count() 前后差（那是全库记录数，并发下会串）。
        """
        if not cls.has_collection(knowledge_base):
            raise KbNotFoundError(f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND)

        client = cls._persistent_client()
        with cls._guard(f"删除 {knowledge_base}/{source} 的分片"):
            collection = cls._collection(client, knowledge_base)
            # include=[] 只取 id：不需要 document/metadata，大文档能省一大段序列化
            ids = list(collection.get(where={"source": {"$eq": source}}, include=[])["ids"] or [])
            if ids:
                collection.delete(ids=ids)
        # 同样留痕：deleted=0 是「名字对不上」的常见现象，有这行才不用猜
        logger.info("按来源删除完成 | kb=%s source=%s 分片=%d", knowledge_base, source, len(ids))
        return len(ids)
