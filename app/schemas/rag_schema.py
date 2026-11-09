from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    knowledge_base: str
    file_name: str
    chunk_count: int
    status: str                  # success / skipped
    deleted_chunks: int = 0      # replace 模式下被覆盖删除的旧分片数
    mode: str = "replace"
    clean_policy: str = "full"   # 回显实际生效的清洗策略（传错位置时便于发现）


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
    name: str
    doc_count: int      # 去重后的来源文档数
    chunk_count: int = 0  # 向量分片总数


class DeleteDocumentsResponse(BaseModel):
    knowledge_base: str
    source: str
    deleted_chunks: int  # 实际删除的分片数；0 表示没有匹配到该来源


class DeleteKnowledgeBaseResponse(BaseModel):
    knowledge_base: str
    deleted_collections: int  # 1=已删除；0=本来就不存在（路由层会拦成 404 / code 2001）


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
