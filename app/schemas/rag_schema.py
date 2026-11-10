from pydantic import BaseModel
from typing import List, Optional


class SameSourceKnowledgeBase(BaseModel):
    """与本次上传同文件名、但不是同一份内容的历史库（孤儿提示，不自动删除）。"""

    name: str
    knowledge_base_id: Optional[int] = None
    chunk_count: int = 0


class UploadResponse(BaseModel):
    knowledge_base: str                       # 后端生成：doc_<sha256 前 16 位>，不可修改
    knowledge_base_id: Optional[int] = None   # 首次入库时生成，之后恒复用
    file_name: str
    source: str = ""                          # 库内实际生效的来源名（改名重传时会被刷新）
    chunk_count: int                          # 库内当前分片数（skipped 时是库里的真实值）
    status: str                               # created / skipped / replaced
    deleted_chunks: int = 0                   # replaced 时被删掉的旧分片数
    # 以下四项回显「库里实际生效的值」，不是本次请求参数——传了 safe 却被 skipped 时才看得出来
    clean_policy: str = "full"
    cleaner_version: str = ""                 # 清洗实现代数（c1/c2…），与 clean_policy 分工不同
    parser_version: str = ""
    splitter_version: str = ""
    reason: str = ""                          # 中文说明为什么是这个状态
    same_source_kbs: List[SameSourceKnowledgeBase] = []


class ChatRequest(BaseModel):
    question: str
    knowledge_base: str
    top_k: int = 4
    stream: bool = False
    # 检索策略：similarity（默认）| mmr（备选，多样性优先）
    retrieval_mode: str = "similarity"
    fetch_k: Optional[int] = None       # 仅 mmr 生效：候选池大小，默认 top_k * 4
    mmr_lambda: float = 0.5             # 仅 mmr 生效：0 最偏多样性，1 最偏相关性


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    knowledge_base: str


class KnowledgeBaseItem(BaseModel):
    name: str                                 # collection 名 doc_<sha16>
    source: Optional[str] = None              # 这个库装的是哪份文件
    knowledge_base_id: Optional[int] = None   # 业务侧主键，MySQL 接入后由它关联
    doc_count: int      # 恒为 1（有分片）/ 0（空库），一个库只放一份文件
    chunk_count: int = 0  # 向量分片总数


class DeleteDocumentsResponse(BaseModel):
    knowledge_base: str
    source: str
    deleted_chunks: int  # 实际删除的分片数；0 表示没有匹配到该来源


class DeleteKnowledgeBaseResponse(BaseModel):
    knowledge_base: str
    # 带回 kb_id：业务侧要按它删掉 MySQL 里的 knowledge_base 记录，否则会留下悬空行
    knowledge_base_id: Optional[int] = None
    deleted_collections: int  # 1=已删除；0=本来就不存在（路由层会拦成 404 / code 2001）
    deleted_chunks: int = 0   # 实际删掉的分片数


class SourceItem(BaseModel):
    source: str
    chunk_count: int                        # 该来源的分片数
    parser_versions: List[str] = []         # 出现过的解析版本；旧数据为 "<未标记>"


class CollectionItem(BaseModel):
    name: str
    chunk_count: int                        # collection.count()，全库分片总数
    doc_count: int                          # 去重后的来源数
    metadata: dict = {}                     # collection 级 metadata（如 hnsw:space）
    sources: List[SourceItem] = []          # 按分片数降序


# ------------------------------------------------ 解析预览（不入库，debug 用）
class PreviewChunk(BaseModel):
    index: int
    chars: int
    content: str
    metadata: dict


class PreviewStage(BaseModel):
    """某个阶段的整体样貌：多少块、多少字、前 N 字预览"""

    docs: int
    chars: int
    preview: str
    truncated: bool


class ParsePreviewResponse(BaseModel):
    """解析预览：只给三段结果，不返回文件信息、参数回显与耗时统计"""

    raw: PreviewStage         # 解析后、清洗前
    cleaned: PreviewStage     # 清洗后
    chunks: List[PreviewChunk]  # 真正会被入库的分片
