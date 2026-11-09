import re
from typing import Iterator, List, Union

from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document

from docx import Document as DocxDocument

# Word 的标题样式名随语言/版本变化：英文 "Heading 1"、部分生成器 "heading 1"、
# 中文版 Word "标题 1"。只认这三种，自定义样式（如"一级标题"）不猜。
_HEADING_RE = re.compile(r"^(?:heading|标题)\s*(\d)$", re.IGNORECASE)


def _iter_blocks(doc: DocxDocument) -> Iterator[Union[Paragraph, Table]]:
    """按 body 中 XML 的真实顺序产出段落与表格。

    python-docx 的 doc.paragraphs 与 doc.tables 是两个互不相干的视图，
    先取段落再取表格会把**所有表格推到文档末尾**，并且丢掉「这张表属于第几章」
    这个信息。按标题聚合分片必须先知道表格在正文中的位置，所以这里直接遍历
    body 的子元素（w:p 段落 / w:tbl 表格）看原始顺序。

    嵌套表格（单元格内的表）不会出现在 body 顶层，走的是各自父级的 iterchildren，
    这里不处理——目前也没有按单元格递归的需求。
    """
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


class DocxParser:
    @staticmethod
    def parse(file_path: str) -> List[Document]:
        docs: List[Document] = []
        doc = DocxDocument(file_path)
        para_index = 0
        table_index = 0

        for block in _iter_blocks(doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue

                # 标题层级是结构分片的唯一依据（见 utils/structural_splitter.py），
                # 识别失败会让整篇退化成「按段落切」，所以这里多兼容几种写法。
                style_name = (block.style.name or "").strip() if block.style is not None else ""
                match = _HEADING_RE.match(style_name)
                heading_level = int(match.group(1)) if match else None

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "file_type": "docx",
                        "position": f"段落_{para_index}",
                        "is_heading": heading_level is not None,
                        "heading_level": heading_level,
                    }
                ))
                para_index += 1
                continue

            # 表格：整张作为一个原子块，不在这里拆分——拆开表头与数据行，
            # 检索出来就是一堆没有列名的数字。
            rows = []
            for row in block.rows:
                rows.append(" | ".join(cell.text.strip() for cell in row.cells))
            table_text = "\n".join(row for row in rows if row.strip()).strip()
            if not table_text:
                continue

            docs.append(Document(
                page_content=f"表格_{table_index}:\n{table_text}",
                metadata={
                    "source": file_path,
                    "file_type": "docx",
                    "position": f"表格_{table_index}",
                    "type": "table",
                }
            ))
            table_index += 1

        return docs


if __name__ == "__main__":
    file_path = "山石动态分支性能测试需求.docx"
    docus = DocxParser.parse(file_path=file_path)
    print(f"document数量：{len(docus)}")
    for i in docus:
        print(f"读取到的page_content: {i.page_content[:60]}", end="\n")
