"""行为快照测试：把解析 / 清洗 / 分片三层对同一份输入的产出**逐字**钉住。

它解决的是「版本号纪律」这个人工环节：改了 cleaner 规则、分片规则或解析顺序，
但忘了把 CLEANER_VERSION / SPLITTER_VERSION / PARSER_VERSION +1 时，
上传会把新逻辑判成 skipped，旧切片继续被检索——日志里完全看不出来。
快照一红，就必须停下来确认「这是我有意改的吗」，然后二选一：

- 有意改动 → bump 对应版本常量 + 重新生成快照（见下）
- 无意改动 → 说明改坏了，回退

重新生成快照（生成后请 `git diff tests/snapshots/` 逐行确认）：

    UPDATE_SNAPSHOTS=1 python tests/test_snapshot_behavior.py

快照文件名里带着版本号（`cleaner_c1_full.txt`），所以「版本常量变了但快照还是旧文件名」
会被下面的 _version_guard 直接拦下，逼你把新快照也提交进去。

只钉「与检索质量相关」的字段：正文、heading_path、分片策略、标题层级。
不钉 source_path / file_sha256 / 时间戳这些每跑一次都不同的东西。

运行：
    python tests/test_snapshot_behavior.py
    pytest tests/test_snapshot_behavior.py
"""

import difflib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from utils.structural_splitter import SPLITTER_VERSION, StructuralSplitter
from utils.text_cleaner import CLEANER_VERSION, TextCleaner
from utils.universal_parser import PARSER_VERSION, UniversalDocParser

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"

# 快照文件名里的版本号：改版本常量时必须一起改这里，否则下面的守卫会失败
CLEANER_SNAPSHOT_VER = "c1"
SPLITTER_SNAPSHOT_VER = "s1"
PARSER_SNAPSHOT_VER = "v4"


# ------------------------------------------------------------------ 快照读写

def _assert_snapshot(name: str, actual: str) -> None:
    """与快照逐字比对；UPDATE_SNAPSHOTS=1 时改写快照。"""
    path = os.path.join(SNAPSHOT_DIR, name)
    if UPDATE:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        # 固定 LF：Windows 上写出来的快照如果带 \r\n，Linux 部署后 git diff 会整文件变红
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(actual)
        print(f"  [已重写快照] {name}（{len(actual)} 字）")
        return

    assert os.path.exists(path), (
        f"缺少快照文件 {name}。首次引入或刚升级版本时，先跑 "
        f"UPDATE_SNAPSHOTS=1 python tests/test_snapshot_behavior.py 生成，"
        f"再 git add tests/snapshots/"
    )
    with open(path, encoding="utf-8") as f:
        expected = f.read()
    if expected == actual:
        return

    diff = "".join(
        list(difflib.unified_diff(
            expected.splitlines(True), actual.splitlines(True),
            fromfile=f"快照 {name}", tofile="当前输出", n=2,
        ))[:60]
    )
    raise AssertionError(
        f"{name} 与快照不一致——{name.split('_')[0]} 的行为变了：\n{diff}\n"
        f"确认是有意改动后：1) 把对应版本常量 +1；2) UPDATE_SNAPSHOTS=1 重新生成快照；"
        f"3) 更新本文件顶部的 CLEANER/SPLITTER/PARSER_SNAPSHOT_VER。"
    )


def _version_guard() -> None:
    """版本常量与快照文件名必须同步：防止「bump 了版本但快照还是旧的」。"""
    for label, actual, snapshot_ver in (
        ("CLEANER_VERSION", CLEANER_VERSION, CLEANER_SNAPSHOT_VER),
        ("SPLITTER_VERSION", SPLITTER_VERSION, SPLITTER_SNAPSHOT_VER),
        ("PARSER_VERSION", PARSER_VERSION, PARSER_SNAPSHOT_VER),
    ):
        assert actual == snapshot_ver, (
            f"{label} 已是 {actual}，但快照还停在 {snapshot_ver}："
            f"请把本文件的 *_SNAPSHOT_VER 改成 {actual}，然后 "
            f"UPDATE_SNAPSHOTS=1 重新生成快照并逐行 review 差异。"
        )


# ------------------------------------------------------------------ 输入样本

# 清洗样本：每一行都对应一个清洗开关，改动任何规则都会让输出变样
CLEANER_SAMPLE = """　　企业员工手册　　版本V２.０\u200b\u200b
========================================

　　第一章　入职须知
　　第一章　入职须知

１、入职当日需携带身份证原件,银行卡复印件各１份。
    缩进两级的内容，靠空格表达层级
    缩进两级的内容，靠空格表达层级

姓名: 张三    部门: 研发部
备注　　:　　仅用于测试



最后一段，前面有三个空行。
"""


def _heading_docs():
    """模拟 docx 解析结果：标题带 heading_level，正文不带。"""
    rows = [
        ("产品手册", 1),
        ("第一章 概述", 2),
        ("本手册介绍产品的基本用法，包括安装、配置与常见问题。", None),
        ("第二章 安装", 2),
        ("安装步骤如下：先下载安装包，再双击运行，最后按提示完成配置。", None),
        # 极短小节：heading 策略不做短块合并，这种情况会单独成片（现状，见快照 chunk 0/3）
        ("2.1 环境要求", 3),
        ("无", None),
        ("附录 A 术语表", 2),
        ("分片：把长文本切成检索单元。", None),
        ("向量：文本的数值表示。", None),
    ]
    docs = []
    for text, level in rows:
        meta = {"source": "手册.docx", "file_type": "docx", "parser_version": PARSER_VERSION}
        if level:
            meta["heading_level"] = level
            meta["is_heading"] = True
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def _flat_docs():
    """无结构文本：走 flat + 递归切分，验证 SEPARATORS 与合并规则。"""
    para = (
        "第一段讲的是安装前的准备工作，需要确认操作系统版本、磁盘空间与网络连通性。"
        "第二段讲的是安装过程中的注意事项，包括关闭杀毒软件、以管理员身份运行安装程序。"
        "第三段讲的是安装完成后的验证步骤，打开终端执行命令，能看到版本号即表示安装成功。"
    )
    return [
        Document(page_content=para, metadata={"source": "note.md", "file_type": "markdown"}),
        Document(page_content=para, metadata={"source": "note.md", "file_type": "markdown"}),
    ]


def _page_docs():
    """模拟 PDF 逐页解析：短页合并、超长页递归切 + 短尾并回，都在这条路径上。"""
    body = (
        "本页正文用于验证按页分组之后的长度校形规则，内容本身没有业务含义，"
        "只需要足够长以触发递归切分。安装前请确认系统版本与磁盘空间，"
        "安装过程中请关闭杀毒软件并以管理员身份运行。"
    )
    long_page = (
        "超长页的第一段，用来触发按页分组之后的递归切分与短尾合并。"
        "这一页的正文会被切分器按句号、逗号、空格由粗到细地切，切点应当落在语义边界上。"
        "超长页的第二段，继续堆叠字数，让这一页的总长度明显超过此时设置的 chunk_size。"
        "为了确保真的会触发递归切分，这里再多写几句：安装之前请确认系统版本、磁盘剩余空间，"
        "以及是否需要配置代理；安装过程中请关闭杀毒软件，并以管理员身份运行安装程序。"
        "超长页的第三段，用于观察切分点是否落在句号处而不是句子中间。"
        "超长页的第四段，再次堆叠一些字数，确保这一页切完之后还会剩下一个短尾片。"
        "超长页的第五段，结束。"
    )
    rows = [
        (1, "封面"),                                   # 单页短 → 与后续长页不合并
        (2, body),                                     # 正常页
        (3, "第二章 隔页"),                            # 连续两页短 → 合并
        (4, "目录"),
        (5, long_page),                                # 超长页 → 递归切 + 短尾并回
    ]
    return [
        Document(page_content=text, metadata={"source": "报告.pdf", "page_number": page})
        for page, text in rows
    ]


def _render_chunks(chunks, strategy: str) -> str:
    """分片渲染：只留与检索相关的字段，去掉路径/时间这类易变值。"""
    lines = [f"strategy={strategy} chunks={len(chunks)}"]
    for chunk in chunks:
        meta = chunk.metadata or {}
        head = (
            f"\n[chunk {meta.get('chunk_index')}] "
            f"chars={len(chunk.page_content)} "
            f"heading_path={meta.get('heading_path', '')!r} "
            f"page_start={meta.get('page_start')} "
            f"page_end={meta.get('page_end')} "
            f"is_heading={meta.get('is_heading')}"
        )
        lines.append(head)
        lines.append(chunk.page_content)
    return "\n".join(lines) + "\n"


def _build_docx(path: str) -> None:
    """生成一份小 docx：标题、正文、表格、空段落，用来钉解析顺序。"""
    from docx import Document as Docx

    doc = Docx()
    doc.add_heading("产品手册", level=1)
    doc.add_heading("第一章 概述", level=2)
    doc.add_paragraph("本手册介绍产品的基本用法。")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "参数"
    table.cell(0, 1).text = "取值"
    table.cell(1, 0).text = "电压"
    table.cell(1, 1).text = "220V"
    doc.add_paragraph("表格之后的一段正文，用来验证表格是否回到正文中的真实位置。")
    doc.add_heading("第二章 安装", level=2)
    doc.add_paragraph("双击安装包，按提示完成。")
    doc.add_paragraph()  # 空段落：不能被当成内容
    doc.save(path)


def _render_parse(docs) -> str:
    lines = [f"docs={len(docs)}"]
    for i, doc in enumerate(docs):
        meta = doc.metadata or {}
        lines.append(
            f"\n[{i}] heading_level={meta.get('heading_level')} "
            f"is_heading={meta.get('is_heading')} "
            f"layout_type={meta.get('layout_type')} "
            f"parse_mode={meta.get('parse_mode')} "
            f"chars={len(doc.page_content)}"
        )
        lines.append(doc.page_content)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 用例

def test_cleaner_full_snapshot():
    cleaner = TextCleaner(policy="full")
    _assert_snapshot(
        f"cleaner_{CLEANER_SNAPSHOT_VER}_full.txt",
        cleaner.clean_single_text(CLEANER_SAMPLE),
    )


def test_cleaner_safe_snapshot():
    cleaner = TextCleaner(policy="safe")
    _assert_snapshot(
        f"cleaner_{CLEANER_SNAPSHOT_VER}_safe.txt",
        cleaner.clean_single_text(CLEANER_SAMPLE),
    )


def test_cleaner_raw_snapshot():
    cleaner = TextCleaner(policy="raw")
    _assert_snapshot(
        f"cleaner_{CLEANER_SNAPSHOT_VER}_raw.txt",
        cleaner.clean_single_text(CLEANER_SAMPLE),
    )


def test_heading_split_snapshot():
    chunks, strategy = StructuralSplitter.split(_heading_docs())
    _assert_snapshot(
        f"splitter_{SPLITTER_SNAPSHOT_VER}_heading.txt",
        _render_chunks(chunks, strategy),
    )


def test_flat_split_snapshot():
    chunks, strategy = StructuralSplitter.split(_flat_docs(), chunk_size=200, chunk_overlap=0)
    _assert_snapshot(
        f"splitter_{SPLITTER_SNAPSHOT_VER}_flat.txt",
        _render_chunks(chunks, strategy),
    )


def test_page_split_snapshot():
    """按页分组：连续短页合并、一短一长不合并、超长页递归切 + 短尾并回。"""
    chunks, strategy = StructuralSplitter.split(_page_docs(), chunk_size=200, chunk_overlap=0)
    _assert_snapshot(
        f"splitter_{SPLITTER_SNAPSHOT_VER}_page.txt",
        _render_chunks(chunks, strategy),
    )


def test_docx_parse_snapshot():
    """docx 解析：标题层级、表格位置（v3 起按 body XML 顺序遍历）都要被钉住。"""
    path = os.path.join(tempfile.gettempdir(), "_snapshot_parser_v3.docx")
    _build_docx(path)
    try:
        docs = UniversalDocParser.parse(path, source_name="手册.docx")
    finally:
        os.unlink(path)
    _assert_snapshot(
        f"parser_{PARSER_SNAPSHOT_VER}_docx.txt",
        _render_parse(docs),
    )


def test_version_constants_match_snapshots():
    """版本常量与快照文件名同步，否则「bump 了却没重生成快照」会被当成通过。"""
    _version_guard()


if __name__ == "__main__":
    passed = failed = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        try:
            func()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
