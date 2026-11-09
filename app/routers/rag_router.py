import json
import os
import re
import tempfile
import time
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


@router.post("/upload", response_model=ApiResponse[UploadResponse], summary="文档上传入库")
async def upload_document(
    knowledge_base: str = "default",
    mode: Literal["replace", "skip"] = Query(
        "replace",
        description=(
            "replace=同来源覆盖写入（先写新分片再删旧分片）；"
            "skip=库里已有同内容同解析版本(sha256+parser_version)则跳过"
        ),
    ),
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
    # 先校验库名再落盘：名字非法时没必要把几十 MB 文件写到磁盘再报错
    _ensure_valid_kb_name(knowledge_base)

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
        "上传开始 | kb=%s file=%s size=%s mode=%s policy=%s",
        knowledge_base, file.filename,
        f"{total / 1024 / 1024:.2f}MB" if total <= max_bytes
        else f">{MAX_UPLOAD_MB}MB",
        mode, clean_policy,
    )

    try:
        if total > max_bytes:
            # 保留 HTTP 413：反向代理超限时返回的是裸 413（没有我们的 body），
            # 前端按状态码统一处理才不会漏（架构文档 §5）
            raise HTTPException(
                status_code=CODE_PAYLOAD_TOO_LARGE,
                detail=f"文件超过 {MAX_UPLOAD_MB}MB 上限（当前 {total / 1024 / 1024:.1f}MB），请拆分后上传",
            )

        sha256 = UniversalDocParser.file_sha256(tmp_path)

        # 幂等（仅 skip 模式）：同内容 + 同解析版本 + 同清洗策略 + 同切分版本已入库，
        # 则不重复写向量。这四项必须全带，否则任一策略变更后同一份文件永远无法重灌。
        if mode == "skip" and IndexService.has_source(
            knowledge_base, sha256, PARSER_VERSION, clean_policy, SPLITTER_VERSION
        ):
            # 跳过是「什么都没做」，必须留痕，否则前端看到 chunk_count=0 会怀疑上传失败
            logger.info(
                "上传跳过：同内容同解析版本已入库 | kb=%s file=%s policy=%s",
                knowledge_base, file.filename, clean_policy,
            )
            return ok(UploadResponse(
                knowledge_base=knowledge_base,
                file_name=file.filename,
                chunk_count=0,
                status="skipped",
                deleted_chunks=0,
                mode=mode,
                clean_policy=clean_policy,
            ))

        # 解析清洗分片；source 由解析层按原始文件名归一化，引用不会变成 tmpXXXX.md
        splits = ParseService.parse_and_split(
            tmp_path, source_name=file.filename, clean_policy=clean_policy
        )
        if not splits:
            raise EmptyContentError(
                "解析后没有可入库的内容（文档可能全为空白页）",
                code=ERR_PARSE_EMPTY_CONTENT,
            )

        source = str(splits[0].metadata.get("source") or file.filename)

        # 先写新分片、再删旧分片：写失败时旧数据仍完整（「先删后写」一失败就丢数据）
        # 旧分片按「同 source 或同 sha256」匹配，改名重传也不会留下重复向量
        old_ids = IndexService.source_chunk_ids(knowledge_base, source, sha256)
        chunk_count = IndexService.add_documents(knowledge_base, splits)
        deleted = IndexService.delete_ids(knowledge_base, old_ids)
    finally:
        os.unlink(tmp_path)

    # 异常不再在路由层包装：DocumentParseError 等业务异常直接冒泡给全局 handler
    # 一行收口整条上传链路（解析、清洗、分片、embedding、覆盖删除的耗时之和）
    logger.info(
        "上传完成 | kb=%s file=%s 分片=%d 覆盖删除=%d | %.1fms",
        knowledge_base, file.filename, chunk_count, deleted,
        (time.perf_counter() - start) * 1000,
    )
    return ok(UploadResponse(
        knowledge_base=knowledge_base,
        file_name=file.filename,
        chunk_count=chunk_count,
        status="success",
        deleted_chunks=deleted,
        mode=mode,
        clean_policy=clean_policy,
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
        vector_db/chroma_rag_db/          ← 一个 Chroma 持久化目录（PERSIST_DIR）
        ├── collection "default"          ← 知识库 default
            ├── source "产品手册.pdf"               ← 文档（metadata.source）
            │   ├── chunk 0..41                     ← 分片（真正的存储/检索单元）
            ├── source "接口文档.md"
            │   └── chunk 0..17
        ├── collection "rag_knowledge_base" ← 知识库 rag_knowledge_base
        └── collection "xxx"              ← 知识库 xxx

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
    """整库删除(单个collection)。按 source 删只能清身份对得上的分片，解析策略升级需要整库重灌。"""
    _ensure_valid_kb_name(knowledge_base)
    if not IndexService.has_collection(knowledge_base):
        raise KbNotFoundError(f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND)
    # 删除失败属于服务端故障：不再在路由层拼 500，交给兜底 handler（固定文案 + 日志留全量）
    removed = IndexService.delete_collection(knowledge_base)
    return ok(DeleteKnowledgeBaseResponse(
        knowledge_base=knowledge_base,
        deleted_collections=1 if removed else 0,
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
