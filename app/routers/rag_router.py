import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Literal, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.core.error_codes import (
    CODE_ERROR,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_SERVER_ERROR,
    ERR_GENERATE_FAILED,
    ERR_KB_NOT_FOUND,
    ERR_PARSE_EMPTY_CONTENT,
    ERR_PARSE_UNSUPPORTED,
)
from app.core.exceptions import (
    EmptyContentError,
    GenerateError,
    KbNotFoundError,
    ParamError,
    RAGBaseError,
    UnsupportedFileTypeError,
)
from app.core.response import ApiResponse, ok
from app.schemas.rag_schema import (
    UploadResponse,
    ChatRequest,
    ChatResponse,
    KnowledgeBaseItem,
    CollectionItem,
    DeleteDocumentsResponse,
    DeleteKnowledgeBaseResponse,
    ParsePreviewResponse,
    PreviewChunk,
    PreviewStage,
)
from app.services.parse_service import ParseService
from app.services.index_service import IndexService
from app.services.retrieve_service import RetrieveService
from app.services.generate_service import GenerateService
from utils.logger import get_logger
from utils.structural_splitter import SPLITTER_VERSION
from utils.text_cleaner import CLEANER_VERSION
from utils.universal_parser import PARSER_VERSION, UniversalDocParser

router = APIRouter(prefix="/api/rag", tags=["RAG 知识库"])

logger = get_logger("app.routers.rag_router")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"}
# 上传体积上限：限制单文件大小，落盘用分块拷贝，避免大文件把内存打满
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
_COPY_CHUNK = 1024 * 1024
# 解析预览里 raw / cleaned 的截断长度：够看出清洗差异，又不至于把响应撑爆
_PREVIEW_CHARS = 500

# 知识库名 = Chroma 的 collection 名，规则（实测）：3-512 位，只允许 [a-zA-Z0-9._-]，
# 且必须由字母或数字开头、结尾。校验的动机不是「好看」：名字不合规时 chromadb 抛
# InvalidArgumentError，那是用户传参错，若不在这里拦下就会落成 500 兜底。
_KB_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")


def _ensure_valid_kb_name(knowledge_base: str, *, allow_empty: bool = False) -> str:
    """校验知识库名，不合规抛 ParamError（400 参数错误）。

    架构约定：外部来的数据不合法 → 业务异常，所以这里给 400，不是 500。
    allow_empty 只给「不传表示全部」的查询接口用（GET /collections）。
    """
    if allow_empty and knowledge_base == "":
        return knowledge_base
    if _KB_NAME_RE.match(knowledge_base or ""):
        return knowledge_base
    raise ParamError(
        f"知识库名不合法：{knowledge_base!r}。"
        "需为 3-512 位的字母、数字、点、下划线或短横线，且以字母或数字开头结尾"
        "（例如 product-docs、kb_2024）",
        code=CODE_ERROR,
    )


# 同一个库的并发上传串行化（详见 upload_document）。
# 键是 collection 名（内容 sha），条目数等于历史上传过的文件数，测试阶段可忽略；
# 若将来文件量级很大，可以在释放时清理未被等待的锁。
_upload_locks: dict = {}
_upload_locks_guard = threading.Lock()


@contextmanager
def _upload_lock(knowledge_base: str):
    """按库名加进程内互斥锁，只保护「判定 → 写入」这段临界区。"""
    with _upload_locks_guard:
        lock = _upload_locks.setdefault(knowledge_base, threading.Lock())
    with lock:
        yield


@router.post("/upload", response_model=ApiResponse[UploadResponse], summary="文档上传入库")
async def upload_document(
    clean_policy: Literal["full", "safe", "raw"] = Query(
        "full",
        description=(
            "清洗策略：full=全量清洗（默认，会合并空格、去空行、去缩进）；"
            "safe=保留缩进与段落空行，其余照做（markdown/含排版文档用）；"
            "raw=仅移除不可见字符（代码块/公式/ASCII 图表用）"
        ),
    ),
    file: UploadFile = File(..., description="待入库的文档文件"),
):
    """上传 → 解析 → 清洗 → 分片 → 入库，由后端判定该不该写。

    调用方不再传库名，也不再选 replace/skip：
    - collection 名 = `doc_` + sha256(文件原始字节)[:16]，内容定名、不可修改；
    - knowledge_base_id 在「库不存在」时由后端生成，库已存在则复用库里的值；
    - 状态三选一，见 IndexService.decide_upload_status：
      created=首次入库 / skipped=同内容同策略已入库 / replaced=同内容换了策略。
    """
    # 校验文件格式
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"不支持的文件格式：{ext}，支持：{sorted(SUPPORTED_EXTENSIONS)}",
            code=ERR_PARSE_UNSUPPORTED,
        )

    start = time.perf_counter()
    # 临时保存文件：分块落盘 + 体积上限
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(_COPY_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                break
            tmp.write(chunk)

    # 先落一条开始日志：大文件解析可能几十秒，只有结束日志的话，中途卡住无从判断。
    # 超限时 total 只累加到刚过线，所以这里用 ≥ 语义（后面会单独报 413）。
    logger.info(
        "上传开始 | file=%s size=%s policy=%s",
        file.filename,
        f"{total / 1024 / 1024:.2f}MB" if total <= max_bytes else f">{MAX_UPLOAD_MB}MB",
        clean_policy,
    )

    try:
        if total > max_bytes:
            # 保留 HTTP 413：反向代理超限时返回的是裸 413（没有我们的 body），
            # 前端按状态码统一处理才不会漏（架构文档 §5）
            raise HTTPException(
                status_code=CODE_PAYLOAD_TOO_LARGE,
                detail=f"文件超过 {MAX_UPLOAD_MB}MB 上限（当前 {total / 1024 / 1024:.1f}MB），请拆分后上传",
            )

        sha16 = UniversalDocParser.file_sha256(tmp_path)
        knowledge_base = f"doc_{sha16}"

        # 同一个库的并发上传串行化：两个请求同时判定为 created 会各生成一个 kb_id，
        # 后写入的分片把前一个的 metadata 覆盖掉，同一个库里就会混着两个 kb_id。
        # 单进程 + 进程内锁就够；将来多副本部署需要分布式锁或 MySQL 唯一键兜底。
        with _upload_lock(knowledge_base):
            profile = IndexService.collection_profile(knowledge_base)
            status, reason = IndexService.decide_upload_status(
                profile,
                parser_version=PARSER_VERSION,
                clean_policy=clean_policy,
                cleaner_version=CLEANER_VERSION,
                splitter_version=SPLITTER_VERSION,
            )

            if status == "skipped":
                # 改名重传：内容没变 → skipped（不解析、不 re-embed），但源文件名要跟着刷新，
                # 否则列表与引用里永远停在第一次入库时的名字。只改 metadata，不碰向量。
                new_source = os.path.basename(file.filename or "")
                stored_source = profile.get("source") or ""
                if new_source and new_source != stored_source:
                    refreshed = IndexService.update_source(knowledge_base, new_source)
                    reason = (
                        f"{reason}；来源名 {stored_source or '<未标记>'} → {new_source}"
                        f"（已刷新 {refreshed} 个分片，未重算向量）"
                    )
                    logger.info(
                        "上传跳过并刷新来源名 | kb=%s %s → %s 分片=%d",
                        knowledge_base, stored_source, new_source, refreshed,
                    )
                # 跳过是「什么都没做」，必须留痕，否则前端看到 chunk_count 不为 0 会困惑
                logger.info(
                    "上传跳过 | kb=%s file=%s 分片=%d reason=%s",
                    knowledge_base, file.filename, profile["chunk_count"], reason,
                )
                return ok(UploadResponse(
                    knowledge_base=knowledge_base,
                    knowledge_base_id=profile.get("knowledge_base_id"),
                    file_name=file.filename,
                    source=new_source or stored_source,
                    chunk_count=profile["chunk_count"],
                    status=status,
                    deleted_chunks=0,
                    # 回显库内实际生效的值，不是本次请求参数——否则「我传了 safe 却被跳过」
                    # 在响应里看不出来
                    clean_policy=profile.get("clean_policy") or clean_policy,
                    cleaner_version=profile.get("cleaner_version") or CLEANER_VERSION,
                    parser_version=profile.get("parser_version") or PARSER_VERSION,
                    splitter_version=profile.get("splitter_version") or SPLITTER_VERSION,
                    reason=reason,
                ))

            if status == "replaced":
                # 复用 kb_id：同一个业务知识库只是换了切片策略，不该换身份。
                # 老数据可能没有 kb_id（v3 之前灌的），那时补一个新的。
                knowledge_base_id = profile.get("knowledge_base_id") \
                    or IndexService.next_knowledge_base_id()
                # 先删后写：同库同内容，旧分片能从原文件重建；而「先写后删」中途失败
                # 会让新旧策略的分片混在一个库里，下次判定命中新分片直接 skipped，
                # 脏数据永久留着。删完失败则退化成空库，下次上传自动 created 重灌。
                deleted = IndexService.delete_collection(knowledge_base)["chunk_count"]
            else:
                knowledge_base_id = IndexService.next_knowledge_base_id()
                deleted = 0

            # 解析清洗分片；source 由解析层按原始文件名归一化，引用不会变成 tmpXXXX.md
            splits = ParseService.parse_and_split(
                tmp_path, source_name=file.filename, clean_policy=clean_policy
            )
            if not splits:
                raise EmptyContentError(
                    "解析后没有可入库的内容（文档可能全为空白页）",
                    code=ERR_PARSE_EMPTY_CONTENT,
                )

            chunk_count = IndexService.add_documents(
                knowledge_base, splits, knowledge_base_id
            )

            # 新建库时顺手查一下「同文件名、不同内容」的历史库：内容一改就是新库新 ID，
            # 旧库会变成孤儿。这里只提示不删（按文件名自动删会误伤别人的同名文件）。
            same_source_kbs = []
            if status == "created":
                same_source_kbs = IndexService.collections_with_source(
                    os.path.basename(file.filename or ""), exclude=knowledge_base
                )
    finally:
        os.unlink(tmp_path)

    # 异常不再在路由层包装：DocumentParseError 等业务异常直接冒泡给全局 handler
    # 一行收口整条上传链路（解析、清洗、分片、embedding、旧库重建的耗时之和）
    logger.info(
        "上传完成 | status=%s kb=%s kb_id=%s file=%s 分片=%d 重建删除=%d 同名历史库=%d | %.1fms",
        status, knowledge_base, knowledge_base_id, file.filename,
        chunk_count, deleted, len(same_source_kbs),
        (time.perf_counter() - start) * 1000,
    )
    return ok(UploadResponse(
        knowledge_base=knowledge_base,
        knowledge_base_id=knowledge_base_id,
        file_name=file.filename,
        source=os.path.basename(file.filename or ""),
        chunk_count=chunk_count,
        status=status,
        deleted_chunks=deleted,
        clean_policy=clean_policy,
        cleaner_version=CLEANER_VERSION,
        parser_version=PARSER_VERSION,
        splitter_version=SPLITTER_VERSION,
        reason=reason,
        same_source_kbs=same_source_kbs,
    ))


@router.post(
    "/parse/preview",
    response_model=ApiResponse[ParsePreviewResponse],
    summary="解析预览（清洗 + 分片，不入库）",
)
async def preview_parse(
    file: UploadFile = File(..., description="待观察的文档"),
    clean_policy: Literal["full", "safe", "raw"] = Query(
        "full", description="清洗策略，与 /upload 的同名参数完全一致"
    ),
    chunk_size: Optional[int] = Query(
        None, ge=1, le=4000, description="默认 800（解析层默认值），只影响本次"
    ),
    chunk_overlap: Optional[int] = Query(
        None, ge=0, le=2000, description="默认 100，必须小于 chunk_size"
    ),
):
    """走一遍「解析 → 清洗 → 分片」，把中间结果摊开，**不写向量库**。

    与 `/upload` 的唯一区别是最后一步：这里不调用 IndexService，
    所以不产生向量、不落库，换参数反复试没有任何残留。

    只返回三段：
    - raw      解析器吐出的文本（PDF 是逐页 Document），清洗前
    - cleaned  清洗后的文本，和 raw 对比就知道策略改动了什么
    - chunks   真正会被入库的分片（带完整 metadata）
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"不支持的文件格式：{ext}，支持：{sorted(SUPPORTED_EXTENSIONS)}",
            code=ERR_PARSE_UNSUPPORTED,
        )

    # 先算实际生效的分片参数（只传 overlap 时 size 会退回默认值，同样要判），
    # 放在落盘之前——参数不合法没必要先写一遍磁盘
    size = ParseService.DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = ParseService.DEFAULT_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if overlap >= size:
        raise ParamError(
            f"chunk_overlap({overlap}) 必须小于 chunk_size({size})，否则切片无法向前推进",
            code=CODE_ERROR,
        )

    # 落盘策略与 upload 一致：分块写、超限即停，避免大文件把内存打满
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        while True:
            block = await file.read(_COPY_CHUNK)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                break
            tmp.write(block)

    try:
        if total > max_bytes:
            raise HTTPException(
                status_code=CODE_PAYLOAD_TOO_LARGE,
                detail=f"文件超过 {MAX_UPLOAD_MB}MB 上限，与上传接口同一限制",
            )

        # 清洗器与分片器都必须从 ParseService 取，不允许在此处直接构造，
        # 否则预览的清洗/切分行为可能与上传链路悄悄分叉（详见 ParseService.get_cleaner）
        cleaner = ParseService.get_cleaner(clean_policy)

        logger.info(
            "解析预览 | file=%s policy=%s chunk_size=%d overlap=%d",
            file.filename, clean_policy, size, overlap,
        )

        raw_docs = UniversalDocParser.parse(tmp_path, source_name=file.filename)
        cleaned_docs = cleaner.clean_documents(raw_docs)
        chunks, strategy = ParseService.split_cleaned(cleaned_docs, chunk_size, chunk_overlap)

        logger.info(
            "预览分片完成 | file=%s strategy=%s 分片=%d", file.filename, strategy, len(chunks)
        )

        def stage(docs) -> PreviewStage:
            text = "\n\n".join(d.page_content for d in docs)
            return PreviewStage(
                docs=len(docs),
                chars=sum(len(d.page_content) for d in docs),
                preview=text[:_PREVIEW_CHARS],
                truncated=len(text) > _PREVIEW_CHARS,
            )

        return ok(ParsePreviewResponse(
            raw=stage(raw_docs),
            cleaned=stage(cleaned_docs),
            chunks=[
                PreviewChunk(
                    index=i,
                    chars=len(d.page_content),
                    content=d.page_content,
                    metadata=dict(d.metadata),
                )
                for i, d in enumerate(chunks)
            ],
        ))
    finally:
        os.unlink(tmp_path)


@router.delete("/documents", response_model=ApiResponse[DeleteDocumentsResponse], summary="按来源删除向量")
def delete_documents(knowledge_base: str, source: str):
    """删除某个知识库(collection)里指定来源文档的全部向量分片。

        现在的层级是「一个 collection 就装一份文件」，所以这个接口基本等价于整库删除：

        vector_db/chroma_rag_db/            ← 一个 Chroma 持久化目录（PERSIST_DIR）
        ├── collection "doc_a3f9c2e1b7d4abcd"  ← 知识库，名字由内容 sha256 前 16 位决定
        │   ├── source "产品手册.pdf"            ← 它装的那份文件（metadata.source）
        │   └── chunk 0..11                     ← 分片（真正的存储/检索单元）
        └── collection "doc_7b21e0c4d9a83315"  ← 另一份文件 = 另一个知识库

    保留它是为了不破坏既有调用方；新前端直接用 DELETE /knowledge-bases/{name} 更准确
    （那里还会带回 knowledge_base_id，便于同步删 MySQL 记录）。

    - 200 + code 2001：知识库不存在
    - 200 + deleted_chunks=0：没匹配到该来源（如实上报，不谎报成功）
    """
    _ensure_valid_kb_name(knowledge_base)
    if not source.strip():
        raise ParamError("source 不能为空", code=CODE_ERROR)
    if not IndexService.has_collection(knowledge_base):
        raise KbNotFoundError(f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND)

    deleted = IndexService.delete_by_source(knowledge_base, source)
    return ok(DeleteDocumentsResponse(
        knowledge_base=knowledge_base,
        source=source,
        deleted_chunks=deleted,
    ))


@router.delete(
    "/knowledge-bases/{knowledge_base}",
    response_model=ApiResponse[DeleteKnowledgeBaseResponse],
    summary="清空整个知识库",
)
def delete_knowledge_base(knowledge_base: str):
    """整库删除（一个 collection 就是一份文件的内容）。

    响应带回 knowledge_base_id：MySQL 接入后，业务侧要按这个 ID 删掉自己的
    knowledge_base 记录，否则业务表里会留下指向空库的悬空行（顺序见
    IndexService.delete_collection 的注释：先删 Chroma，再删 MySQL）。
    """
    _ensure_valid_kb_name(knowledge_base)
    if not IndexService.has_collection(knowledge_base):
        raise KbNotFoundError(f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND)
    # 删除失败属于服务端故障：不再在路由层拼 500，交给兜底 handler（固定文案 + 日志留全量）
    result = IndexService.delete_collection(knowledge_base)
    return ok(DeleteKnowledgeBaseResponse(
        knowledge_base=knowledge_base,
        knowledge_base_id=result["knowledge_base_id"],
        deleted_collections=1 if result["deleted"] else 0,
        deleted_chunks=result["chunk_count"],
    ))


@router.get("/knowledge-bases", response_model=ApiResponse[list[KnowledgeBaseItem]], summary="获取知识库列表")
def get_knowledge_bases():
    bases = IndexService.list_knowledge_bases()
    return ok([KnowledgeBaseItem(**item) for item in bases])


@router.get(
    "/collections",
    response_model=ApiResponse[list[CollectionItem]],
    summary="巡检向量库：collection 及其来源",
)
def get_collections(knowledge_base: str = ""):
    """排障用只读接口：直接看 Chroma 里的 collection，以及每个 collection 内的 source 分布。

    与 `/knowledge-bases` 的区别：那边是业务视图（只有库名和两个计数），这里是
    collection 级细节——metadata（如距离度量）、每个 source 的分片数和解析版本。
    典型用途：确认某份文档是否真的入库、库里是否还有 v1（`<未标记>`）老数据残留。

    - 不传 knowledge_base：返回全部 collection
    - 传了但不存在：code 2001
    - 只读 metadata，不加载 Embedding；耗时随分片总数线性增长
    """
    # 空串表示「不筛选、返回全部」，所以放行空值；非空才校验名字。
    # 校验必须在 has_collection 之前：非法名字到了 Chroma 才会炸，那就成 500 了。
    _ensure_valid_kb_name(knowledge_base, allow_empty=True)
    if knowledge_base and not IndexService.has_collection(knowledge_base):
        raise KbNotFoundError(f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND)
    return ok([CollectionItem(**item) for item in IndexService.inspect_collections(knowledge_base)])


@router.post("/chat", response_model=ApiResponse[ChatResponse], summary="RAG 智能问答")
def rag_chat(req: ChatRequest):
    """检索 → 生成；stream=true 时以 SSE 逐段返回（流式响应不包统一结构体）。

    检索策略由 retrieval_mode 控制：similarity（默认）/ mmr（备选，多样性优先）。
    - code 400：参数非法（检索策略 / lambda_mult / question 为空）
    - code 2001：知识库不存在
    - code 3001：模型端点异常（流式接口流内错误以 error 事件下发）

    业务异常一律直接冒泡给全局 handler，路由层不做任何异常翻译。
    """
    # 检索阶段同时完成参数校验（库名 / question 空 / 策略非法 / lambda 越界）与库存在性检查，
    # 保证这些错误发生在建流之前——流已开始后无法再返回 JSON 错误体。
    _ensure_valid_kb_name(req.knowledge_base)
    contexts = RetrieveService.get_contexts(
        req.question,
        req.knowledge_base,
        req.top_k,
        mode=req.retrieval_mode,
        fetch_k=req.fetch_k,
        lambda_mult=req.mmr_lambda,
    )

    if req.stream:
        sources = GenerateService.format_sources(contexts)

        def event_stream():
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
            try:
                for delta in GenerateService.generate_stream(req.question, contexts):
                    yield f"data: {json.dumps({'type': 'delta', 'content': delta}, ensure_ascii=False)}\n\n"
            except Exception as e:
                # 流已开始，无法再改状态码，只能把错误作为事件下发（与 JSON 错误体共享码表）。
                # 业务异常用自身 code，未预料的异常统一 500；都不带原始报文。
                if isinstance(e, RAGBaseError):
                    code, message = e.code, e.msg
                else:
                    code, message = CODE_SERVER_ERROR, "生成过程发生错误，请稍后重试"
                logger.warning("流式生成失败：%s: %s", type(e).__name__, e)
                yield f"data: {json.dumps({'type': 'error', 'code': code, 'message': message}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    try:
        answer, sources = GenerateService.generate(req.question, contexts)
    except RAGBaseError:
        # 业务异常原样冒泡，不能被下面的宽异常包成「模型调用失败」
        raise
    except Exception as e:
        # 第三方模型端点故障 → 封装业务异常；msg 用固定文案，原始报文只进日志
        logger.warning("模型调用失败：%s: %s", type(e).__name__, e)
        raise GenerateError("模型服务调用失败，请稍后重试", code=ERR_GENERATE_FAILED) from e

    return ok(ChatResponse(
        answer=answer,
        sources=sources,
        knowledge_base=req.knowledge_base
    ))
