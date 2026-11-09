"""结构感知分片的契约测试。

改造前：docx 逐段落输出 + split_documents 逐块切、不合并，一份 900 字的
三章文档会切成 19 片、平均 49 字，标题（4~10 字）各自独占一片。

改造后要求：
1. docx 按标题聚合成小节，标题不再孤片；
2. 表格回到正文中的真实位置，能归入所属章节；
3. PDF 超长页递归切时，页首标题不被切成碎片；连续超短页（封面/目录）合并；
4. 无结构文档退回纯递归切分，行为不变；
5. 每个分片带 split_strategy / splitter_version，后者参与 skip 幂等。

运行：
    python tests/test_structural_split.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from app.services.parse_service import ParseService
from utils import UniversalDocParser
from utils.structural_splitter import (
    MIN_PAGE_CHARS,
    SPLITTER_VERSION,
    STRATEGY_FLAT,
    STRATEGY_HEADING,
    STRATEGY_PAGE,
    STRATEGY_SLIDE,
    StructuralSplitter,
)

_tmp_files = []


def _squash(text) -> str:
    """去掉所有空白再比较。

    full 清洗策略会在中文与数字之间插空格（「第1章」→「第 1 章」），
    直接比字符串会误判成内容不对。
    """
    return "".join(str(text).split())


def _make_docx(blocks):
    """blocks: [(kind, text)]，kind ∈ h1/h2/p/table"""
    from docx import Document as DocxDocument

    doc = DocxDocument()
    for kind, text in blocks:
        if kind in ("h1", "h2"):
            doc.add_heading(text, level=1 if kind == "h1" else 2)
        elif kind == "p":
            doc.add_paragraph(text)
        elif kind == "table":
            rows = text.split(";")
            table = doc.add_table(rows=len(rows), cols=2)
            for i, row in enumerate(rows):
                a, b = row.split("|")
                table.cell(i, 0).text = a
                table.cell(i, 1).text = b
    path = tempfile.NamedTemporaryFile(suffix=".docx", delete=False).name
    doc.save(path)
    _tmp_files.append(path)
    return path


def _docx_chapters():
    blocks = [("h1", "产品手册"), ("p", "本文档说明产品规则。")]
    for ch in range(1, 4):
        blocks.append(("h1", f"第{ch}章 业务规则"))
        for i in range(5):
            blocks.append(("p", f"这是第{ch}章第{i+1}段正文，用来观察分片边界是否落在小节上。" * 2))
    return _make_docx(blocks)


# ------------------------------------------------------------------ docx
def test_docx_heading_grouping():
    """标题聚合：三章文档应聚成「开头 + 三章」几个完整小节，而不是逐段碎切"""
    docs = ParseService.parse_and_split(_docx_chapters(), source_name="手册.docx")
    assert len(docs) <= 5, f"应当按小节聚合，实际切成 {len(docs)} 片"
    paths = [d.metadata.get("heading_path") for d in docs]
    assert any(p and "第1章" in _squash(p) for p in paths), paths
    # 每片都要带上所属章节，否则检索命中后不知道这段属于哪一章
    assert all(p for p in paths), f"有分片缺少 heading_path: {paths}"
    # 正文与标题在同一片内（标题不该孤零零成片）
    assert any(
        "第1章" in _squash(d.metadata.get("heading_path")) and "段正文" in _squash(d.page_content)
        for d in docs
    ), "标题与其正文应落在同一片"


def test_docx_table_keeps_position():
    """表格必须回到正文中的真实位置，而不是被推到文档末尾"""
    path = _make_docx([
        ("h1", "第1章 入职"),
        ("p", "新员工需提交以下材料。"),
        ("table", "材料|份数;身份证|2;学历证明|1"),
        ("p", "以上材料入职前交齐。"),
        ("h1", "第2章 考勤"),
        ("p", "迟到超过30分钟按旷工处理。"),
    ])
    docs = ParseService.parse_and_split(path, source_name="手册.docx")
    table_chunks = [d for d in docs if "身份证 | 2" in d.page_content]
    assert table_chunks, "表格内容应出现在分片里"
    # v2 的老实现里表格排在最后、归不进章节；现在必须归入第1章
    assert any("第1章" in _squash(d.metadata.get("heading_path") or "") for d in table_chunks), (
        f"表格未归入所属章节：{[d.metadata.get('heading_path') for d in table_chunks]}"
    )


def test_docx_without_heading_falls_back():
    """通篇不用标题样式时退回纯递归切分，不强行编造结构"""
    path = _make_docx([("p", f"第{i}段没有任何标题样式的正文内容。" * 3) for i in range(6)])
    docs, strategy = ParseService.split_cleaned(UniversalDocParser.parse(path))
    assert strategy == STRATEGY_FLAT, strategy
    assert docs


def test_docx_strategy_is_heading():
    path = _docx_chapters()
    docs = UniversalDocParser.parse(path)
    _, strategy = StructuralSplitter.split(docs)
    assert strategy == STRATEGY_HEADING


# ------------------------------------------------------------------ PDF
def _pdf_docs():
    return [
        Document(page_content="RAG 知识平台\n使用手册 V1.0", metadata={"page_number": 1}),
        Document(page_content="目录\n第一章 概述 .... 3", metadata={"page_number": 2}),
        Document(page_content="目录续\n第三章 配置 .... 9", metadata={"page_number": 3}),
        Document(
            page_content="第一章 概述\n" + "本平台用于文档解析与检索问答，支持多种格式。" * 60,
            metadata={"page_number": 4},
        ),
    ]


def test_pdf_short_pages_merged():
    """连续超短页（封面+目录）合并成一片，页码记成区间"""
    chunks, strategy = StructuralSplitter.split(_pdf_docs())
    assert strategy == STRATEGY_PAGE
    merged = [c for c in chunks if c.metadata.get("page_start") == 1]
    assert merged, "封面页应保留"
    assert merged[0].metadata["page_end"] == 3, merged[0].metadata
    assert "目录" in merged[0].page_content


def test_pdf_long_page_title_not_orphaned():
    """超长页按行切后，页首标题不能被切成几字的孤片"""
    chunks, _ = StructuralSplitter.split(_pdf_docs())
    page4 = [c for c in chunks if c.metadata.get("page_start") == 4]
    assert len(page4) >= 2, "1200 字的一页应被切分成多片"
    for c in page4:
        assert len(c.page_content) >= MIN_PAGE_CHARS or len(page4) == 1, (
            f"出现 {len(c.page_content)} 字的碎孤片：{c.page_content[:20]!r}"
        )
    assert page4[0].page_content.startswith("第一章 概述"), page4[0].page_content[:20]


def test_pdf_short_page_not_merged_into_long():
    """一短一长不合并：把封面粘到正文上只会给检索添噪声"""
    docs = [
        Document(page_content="封面", metadata={"page_number": 1}),
        Document(page_content="正文" * 300, metadata={"page_number": 2}),
    ]
    chunks, _ = StructuralSplitter.split(docs)
    assert len(chunks) == 2
    assert chunks[0].page_content == "封面"


# ------------------------------------------------------------------ 其他
def test_slide_not_merged():
    """幻灯片一页一主题，不跨页合并"""
    docs = [
        Document(page_content="RAG 平台介绍", metadata={"slide_index": 1}),
        Document(page_content="架构总览", metadata={"slide_index": 2}),
    ]
    chunks, strategy = StructuralSplitter.split(docs)
    assert strategy == STRATEGY_SLIDE
    assert len(chunks) == 2


def test_flat_text_uses_recursive():
    docs = [Document(page_content="第一段内容。" * 80, metadata={"file_type": "text"})]
    chunks, strategy = StructuralSplitter.split(docs, chunk_size=300, chunk_overlap=50)
    assert strategy == STRATEGY_FLAT
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 320 for c in chunks)


def test_metadata_carries_version_and_strategy():
    """版本号必须落盘：换切法后 skip 幂等靠它识别旧数据"""
    chunks, strategy = StructuralSplitter.split(_pdf_docs())
    for c in chunks:
        assert c.metadata["splitter_version"] == SPLITTER_VERSION
        assert c.metadata["split_strategy"] == strategy


def test_default_chunk_size_is_800():
    assert ParseService.DEFAULT_CHUNK_SIZE == 800
    assert ParseService.DEFAULT_CHUNK_OVERLAP == 100
    assert StructuralSplitter.split.__defaults__ == (800, 100)


if __name__ == "__main__":
    cases = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    try:
        for name, fn in cases:
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    finally:
        for p in _tmp_files:
            if os.path.exists(p):
                os.unlink(p)
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
