"""
PDF 文档解析工具类
功能：文本版PDF高精度提取、页眉页脚过滤、噪声清洗、按页输出LangChain标准Document
"""
import re
from typing import List
from langchain_core.documents import Document
import pdfplumber


class PDFPlumberParser:
    def __init__(
        self,
        header_ratio: float = 0.05,  # 顶部裁剪比例，过滤页眉
        footer_ratio: float = 0.08,  # 底部裁剪比例，过滤页脚
        enable_clean: bool = True
    ):
        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio
        self.enable_clean = enable_clean

    @staticmethod
    def _clean_text(text: str) -> str:
        """清洗文本噪声：去除异常控制字符、合并连续空行为单个空行、逐行去首尾空白

        保留段落分隔（空行），以便下游按空行分段、保留章节结构。

        注意：\n \r \t 必须保留，否则整页文本会被拼成一行，
        后续按行的去重、分段、分片全部失效。
        正则显式跳过 \x09(Tab) \x0a(LF) \x0d(CR)，
        若写成 [\x00-\x1f\x7f] 会连带删掉换行。
        """
        # 仅移除 \n \r \t 之外的异常控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # 统一换行符为 \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # 逐行去首尾空白，并合并连续空行为单个空行（保留段落分隔）
        result = []
        prev_empty = False
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if not prev_empty:
                    result.append('')
                    prev_empty = True
            else:
                result.append(line)
                prev_empty = False
        # 去掉文档首尾可能残留的空行
        return '\n'.join(result).strip()

    @staticmethod
    def _extract_page_number(page_text: str) -> int | None:
        """从页脚文本中尝试提取页码，适配常见页码格式"""
        # 匹配「第X页」格式
        match = re.search(r'第\s*(\d+)\s*页', page_text, re.MULTILINE)
        if match:
            return int(match.group(1))
        # 匹配「X/Y」格式
        match = re.search(r'(\d+)\s*/\s*\d+', page_text, re.MULTILINE)
        if match:
            return int(match.group(1))
        return None

    def extract_page_text(self, page) -> str:
        """从已打开的 pdfplumber page 抽取正文（裁剪页眉页脚 + 轻量清洗）

        降级链按页调用时复用它，避免重新打开整份 PDF。
        """
        page_height = page.height
        page_width = page.width
        top_bound = page_height * self.header_ratio
        bottom_bound = page_height * (1 - self.footer_ratio)
        cropped_page = page.crop((0, top_bound, page_width, bottom_bound))
        raw_text = cropped_page.extract_text() or ''
        return self._clean_text(raw_text) if self.enable_clean else raw_text

    def parse_by_page(self, file_path: str) -> List[Document]:
        """
        按页解析PDF，返回每页的标准Document对象
        :param file_path: PDF本地文件路径
        :return: Document列表，每个元素对应一页，携带页码等元数据
        """
        docs = []
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages, 1):
                raw_text = self.extract_page_text(page)
                full_page_text = page.extract_text() or ''

                # 提取页码，提取失败则使用物理页序号
                page_num = self._extract_page_number(full_page_text)
                if not page_num:
                    page_num = page_index

                # 构造LangChain标准Document，无缝接入后续分片与入库链路
                doc = Document(
                    page_content=raw_text,
                    metadata={
                        "source": file_path,
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "file_type": "pdf"
                    }
                )
                docs.append(doc)

        return docs

    def extract_full_text(self, file_path: str) -> str:
        """提取PDF全部文本（拼接所有页）"""
        page_docs = self.parse_by_page(file_path)
        return '\n\n'.join(doc.page_content for doc in page_docs)


# 本地测试
if __name__ == "__main__":
    parser = PDFPlumberParser()
    test_file = "UN38.3_new.pdf"
    result = parser.parse_by_page(test_file)
    print(f"解析完成，共 {len(result)} 页")
    for item in result:
        print(f"第 {item.metadata['page_number']} 页摘要：{item.page_content[:60]}...")
