"""结构感知分片：先按文档自身的结构边界分组，再对分组做长度校形。

为什么需要它
------------
原来的做法是 `RecursiveCharacterTextSplitter.split_documents()` 直接切解析结果。
LangChain 会**逐个 Document 独立切、不跨块**，所以「按页切」这件事一直都在；
问题在于它只认字符、不认结构，于是：

- docx 逐段落输出 → 一段 20 字也独占一片，标题（4~10 字）孤零零成一片；
- 一个正常小节被切在句号、逗号处，语义边界丢失；
- 表格、标题、正文之间的从属关系完全没有体现。

这里改成三段式：
    1. 探测 → 这份文档有哪种结构可用（标题 / 页 / 幻灯片 / 无）
    2. 归组 → 把解析出的 Document 按结构聚成语义单元（一个小节、一页）
    3. 校形 → 远超 chunk_size 的单元再递归切；连续的超短页合并

关键不在于「片更小」，而在于「片的边界落在语义边界上」。800 字的完整小节
比 300 字的半句话更容易被检索到，也更容易被模型用上。

降级原则
--------
探测不到结构（md / txt / xlsx / 通篇不用样式的 docx）就退回原来的纯递归切分，
行为与改造前一致。不做「格式是否完整」这类判断——那无法定义，也测不出来。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 切分逻辑的版本号。改这里的分组/校形规则就要 +1：
# 它写进每个分片的 metadata 并参与 mode=skip 的幂等判断，
# 否则同一份文件会因「内容没变」被跳过，新切法永远不生效。
SPLITTER_VERSION = "s1"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

STRATEGY_HEADING = "heading"
STRATEGY_PAGE = "page"
STRATEGY_SLIDE = "slide"
STRATEGY_FLAT = "flat"

# 低于这个字数的页视为「封面 / 图页 / 章节隔页」这类空白页。
# 一行中文约 40 字，两行 80 字——正常正文页远不止这个长度。
MIN_PAGE_CHARS = 80

# 按语义边界由粗到细切：段落 > 行 > 句号 > 叹号 > 问号 > 分号 > 逗号 > 空格 > 字符
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass
class _Block:
    """一个语义单元：一个小节、一页幻灯片，或一段无结构文本。"""

    text: str
    meta: Dict = field(default_factory=dict)
    heading_path: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class StructuralSplitter:
    _splitters: Dict[Tuple[int, int], RecursiveCharacterTextSplitter] = {}

    # ------------------------------------------------------------------ 入口
    @classmethod
    def split(
        cls,
        docs: List[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> Tuple[List[Document], str]:
        """返回 (分片列表, 实际采用的结构策略)。"""
        strategy = cls.detect_strategy(docs)

        if strategy == STRATEGY_HEADING:
            blocks = cls._group_by_heading(docs)
        elif strategy == STRATEGY_SLIDE:
            blocks = cls._group_by_page_key(docs, "slide_index")
        elif strategy == STRATEGY_PAGE:
            blocks = cls._merge_short_pages(
                cls._group_by_page_key(docs, "page_number"), chunk_size
            )
        else:
            blocks = [_Block(text=d.page_content, meta=dict(d.metadata or {})) for d in docs]

        return cls._shape(blocks, chunk_size, chunk_overlap, strategy), strategy

    @classmethod
    def detect_strategy(cls, docs: List[Document]) -> str:
        """看解析结果里带了哪种结构标记，决定用哪条分组路径。

        优先级 heading > slide > page：docx 段落带 heading_level，pptx 带
        slide_index，pdf 带 page_number，三者互斥，但按序判断能避免误判。
        """
        has_heading = has_slide = has_page = False
        for doc in docs:
            meta = doc.metadata or {}
            has_heading = has_heading or bool(meta.get("heading_level"))
            has_slide = has_slide or bool(meta.get("slide_index"))
            has_page = has_page or bool(meta.get("page_number"))
        if has_heading:
            return STRATEGY_HEADING
        if has_slide:
            return STRATEGY_SLIDE
        if has_page:
            return STRATEGY_PAGE
        return STRATEGY_FLAT

    # -------------------------------------------------------------- 分组逻辑
    @classmethod
    def _group_by_heading(cls, docs: List[Document]) -> List[_Block]:
        """按标题层级把段落聚成小节。

        维护一个标题栈：遇到 H2 就弹掉栈里所有 >= 2 级，再压入自己。
        栈上的文字链就是这一节的 heading_path（如「产品手册 > 第1章 入职」）。

        标题本身也并入正文——分片脱离标题后，「打卡规则」这段文字就不知道
        自己属于哪个制度了，检索命中时模型看不懂上下文。
        """
        blocks: List[_Block] = []
        stack: List[Tuple[int, str]] = []
        buffer: List[Document] = []
        buf_path = ""

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            text = "\n".join(d.page_content.strip() for d in buffer).strip()
            first = buffer[0]
            if text:
                blocks.append(_Block(
                    text=text,
                    meta=dict(first.metadata or {}),
                    heading_path=buf_path,
                ))
            buffer = []

        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue
            level = (doc.metadata or {}).get("heading_level")
            if level:
                # 新标题出现 → 上一节收尾
                flush()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((int(level), text))
                buf_path = " > ".join(t for _, t in stack)
            buffer.append(doc)

        flush()
        return blocks

    @classmethod
    def _group_by_page_key(cls, docs: List[Document], key: str) -> List[_Block]:
        """按页 / 按幻灯片：一个 Document 天然就是一个单元，只需带上页码。"""
        blocks: List[_Block] = []
        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue
            meta = dict(doc.metadata or {})
            page = meta.get(key)
            blocks.append(_Block(
                text=text,
                meta=meta,
                page_start=page,
                page_end=page,
            ))
        return blocks

    @classmethod
    def _merge_short_pages(cls, blocks: List[_Block], chunk_size: int) -> List[_Block]:
        """合并**连续**的超短页（封面、目录、图页、章节隔页）。

        只合并连续的短页：一短一长时短页单独成片——把封面粘到正文第一章，
        除了给检索结果添噪声没有任何好处。
        """
        merged: List[_Block] = []
        pending: List[_Block] = []
        pending_chars = 0

        def flush() -> None:
            nonlocal pending, pending_chars
            if not pending:
                return
            merged.append(_Block(
                text="\n".join(b.text for b in pending),
                meta=dict(pending[0].meta),
                page_start=pending[0].page_start,
                page_end=pending[-1].page_end,
            ))
            pending = []
            pending_chars = 0

        for block in blocks:
            if len(block.text) >= MIN_PAGE_CHARS:
                flush()
                merged.append(block)
                continue
            if pending and pending_chars + len(block.text) > chunk_size:
                flush()
            pending.append(block)
            pending_chars += len(block.text)

        flush()
        return merged

    # -------------------------------------------------------------- 长度校形
    @classmethod
    def _shape(
        cls,
        blocks: List[_Block],
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
    ) -> List[Document]:
        """超过 chunk_size 的单元再递归切；其余原样成片。"""
        splitter = cls._get_splitter(chunk_size, chunk_overlap)
        chunks: List[Document] = []

        for block in blocks:
            pieces = (
                [block.text] if len(block.text) <= chunk_size
                else cls._coalesce(splitter.split_text(block.text), chunk_size)
            )
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(Document(
                    page_content=piece,
                    metadata=cls._build_meta(block, strategy, len(chunks)),
                ))
        return chunks

    @staticmethod
    def _coalesce(pieces: List[str], chunk_size: int) -> List[str]:
        """把切分产生的过短碎片并回相邻片。

        超长页按行切时，页首的标题行（「第一章 概述」六个字）会被单独切出来，
        变成一片毫无检索价值的碎片；页尾的收尾句同理。规则是：短于
        MIN_PAGE_CHARS 的片向后并，末尾的向前并，前提是合并后不超上限。
        """
        merged: List[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if merged and len(merged[-1]) < MIN_PAGE_CHARS:
                candidate = merged[-1] + "\n" + piece
                if len(candidate) <= chunk_size:
                    merged[-1] = candidate
                    continue
            merged.append(piece)

        # 末尾残留的短尾片：前面有片就并进去
        if len(merged) >= 2 and len(merged[-1]) < MIN_PAGE_CHARS:
            candidate = merged[-2] + "\n" + merged[-1]
            if len(candidate) <= chunk_size:
                merged[-2] = candidate
                merged.pop()
        return merged

    @staticmethod
    def _build_meta(block: _Block, strategy: str, index: int) -> Dict:
        """分片元数据：继承原块身份，再补结构标记。

        page_number 仍写起始页——generate_service 用它渲染「第 X 页」；
        page_start / page_end 给出真实区间，跨页合并的片靠这两个字段表达。
        """
        meta = dict(block.meta)
        if strategy == STRATEGY_HEADING:
            # 这几个字段描述的是「源段落」，不是分片本身。聚合之后留着会让分片
            # 看起来像个标题（is_heading=True），结构信息已由 heading_path 表达。
            for key in ("position", "is_heading", "heading_level"):
                meta.pop(key, None)
        if block.page_start is not None:
            meta["page_start"] = block.page_start
            meta["page_end"] = block.page_end
            meta["page_number"] = block.page_start
        if block.heading_path:
            meta["heading_path"] = block.heading_path
        meta["split_strategy"] = strategy
        meta["splitter_version"] = SPLITTER_VERSION
        meta["chunk_index"] = index
        return meta

    @classmethod
    def _get_splitter(cls, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        key = (chunk_size, chunk_overlap)
        splitter = cls._splitters.get(key)
        if splitter is None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=SEPARATORS,
            )
            cls._splitters[key] = splitter
        return splitter
