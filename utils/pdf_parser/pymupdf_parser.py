import re
from typing import List
from langchain_core.documents import Document
import fitz

from . import pdf_probe


class PymuPDFParser:
    def __init__(self, enable_clean: bool = True):
        self.enable_clean = enable_clean

    @staticmethod
    def _clean_text(text: str) -> str:
        """基础噪声清洗，适配办公类PDF"""
        # 清理目录引导点线
        text = re.sub(r'[.．]{3,}', '', text)
        # 合并连续空行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines)

    @staticmethod
    def _effective_len(text: str) -> int:
        return len(re.sub(r'\s+', '', text or ''))

    def extract_page(self, page: "fitz.Page") -> tuple:
        """抽一页文本，返回 (text, text_area_ratio, blocks)。

        blocks 一并返回，是为了让 pdf_probe.page_metrics 直接复用同一批块：
        它需要 blocks 的 bbox 算正文区指标，若拿不到就只能再调一次
        get_text("blocks", sort=True)。带排序的 blocks 抽取是本阶段最贵的一步，
        重复抽一次等于把探测成本抬高一截。
        """
        blocks = pdf_probe.text_blocks(page)
        page_lines = [b[4].strip() for b in blocks if b[4].strip()]
        blocks_text = '\n'.join(page_lines)
        area = pdf_probe.text_area_ratio(page, blocks)

        raw_text = page.get_text("text").strip()
        blocks_len = self._effective_len(blocks_text)
        raw_len = self._effective_len(raw_text)
        if raw_len >= blocks_len + 100 and raw_len >= blocks_len * 1.5:
            # 命中「blocks 大面积漏抽」：说明 blocks 不可信，其 bbox 面积同样不可信，
            # 此时若还用面积去卡文本层，就会把正常正文页误判成图表页/空白页。
            # 因此把面积拉到阈值下限，交给字数判据兜底（见 pdf_probe.has_real_text）。
            area = max(area, pdf_probe.MIN_TEXT_AREA)
            text = raw_text
        else:
            text = blocks_text

        return (self._clean_text(text) if self.enable_clean else text), area, blocks

    def extract_page_text(self, page: "fitz.Page") -> str:
        """从已打开的 page 抽取文本。

        优先用 blocks（按坐标排序，能还原双栏/多区块的阅读顺序），但 blocks 抽取
        在部分 PDF 上会漏掉大量正文（实测某 17 页文档第 1 页：blocks 279 字，
        raw 3122 字，正文整块丢失）。因此增加一层覆盖率校验：
        当 raw 的有效字符数显著多于 blocks（多 100 字以上且达到 1.5 倍）时，
        以覆盖率为先改用 raw，避免正文被静默丢弃。
        """
        return self.extract_page(page)[0]

    def parse_by_page(self, file_path: str) -> List[Document]:
        docs = []
        with fitz.open(file_path) as pdf:
            total_pages = pdf.page_count
            for page_index in range(total_pages):
                raw_text = self.extract_page_text(pdf[page_index])

                doc = Document(
                    page_content=raw_text,
                    metadata={
                        "source": file_path,
                        "page_number": page_index + 1,
                        "total_pages": total_pages,
                        "file_type": "pdf"
                    }
                )
                docs.append(doc)
        return docs

    def extract_full_text(self, file_path: str) -> str:
        page_docs = self.parse_by_page(file_path)
        return '\n\n'.join(doc.page_content for doc in page_docs)

# 本地测试
if __name__ == "__main__":
    parser = PymuPDFParser()
    test_file = "UN38_format.pdf"
    result = parser.parse_by_page(test_file)
    print(f"解析完成，共 {len(result)} 页")
    for item in result:
        print(f"第 {item.metadata['page_number']} 页摘要：{item.page_content[:60]}...")