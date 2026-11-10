import re
from collections import Counter
from typing import List
from langchain_core.documents import Document
import fitz

from . import pdf_probe

# 判定「同一视觉行」的纵向容差：取行高的 0.6 倍，且不小于 2pt。
# 小于这个值，上下标/轻微基线偏移会被误并；大于它，紧密排版的相邻行会被吞。
ROW_TOL_RATIO = 0.6
ROW_TOL_MIN = 2.0

# 首行缩进判据：行首 x0 比正文左边距大出这个距离，视为新段落。
# 中文首行缩进两个字，正文 10.5pt 时约 21pt，取 8pt 足够区分（左对齐的续行偏移为 0）。
INDENT_PT = 8.0

# 行距判据：某行与上一行的纵向间距超过「基准行距」的这个倍数，视为新段落。
# 实测正文行距 20.0pt、段间距 24.7~31.3pt，1.15 倍正好落在两者之间。
# 并非所有排版都做首行缩进，只靠 x0 会把整页正文并成一段。
PARA_GAP_RATIO = 1.15


class PymuPDFParser:
    def __init__(self, enable_clean: bool = True):
        self.enable_clean = enable_clean

    @staticmethod
    def _clean_text(text: str) -> str:
        """基础噪声清洗，适配办公类PDF。

        与旧实现的差别：**保留单个空行**。旧版逐行 strip 后丢掉所有空行，
        于是 `\\n\\n` 的段落边界被压成 `\\n`，下游分片再也分不出段落。
        """
        # 清理目录引导点线
        text = re.sub(r'[.．]{3,}', '', text)
        lines = [line.strip() for line in text.split('\n')]
        kept: List[str] = []
        for line in lines:
            if line:
                kept.append(line)
            elif kept and kept[-1] != '':
                kept.append('')  # 连续空行压成一个，段落边界保留
        while kept and kept[0] == '':
            kept.pop(0)
        while kept and kept[-1] == '':
            kept.pop()
        return '\n'.join(kept)

    @staticmethod
    def _effective_len(text: str) -> int:
        return len(re.sub(r'\s+', '', text or ''))

    @classmethod
    def _rebuild_text(cls, page: "fitz.Page") -> str:
        """按视觉行重建页面文本，再按缩进还原段落。

        为什么需要这一层：ReportLab 之类的排版库会把一段文字拆成多次
        drawString，同一水平线上的每个片段都成了独立的 line。PyMuPDF 的
        `get_text("text")` 逐 line 补换行，于是标题
        「一、时代之问：当AI开始写代码」会变成 12 行、分片边界被切碎。

        这里做两件事：
        1. 按 bbox 的 y 中心把同一视觉行的 line 归并，行内按 x 排序拼接；
        2. 行首 x0 比正文左边距右移 INDENT_PT 以上的，视为新段落，
           其余行直接接续（中文无需补空格），段间用 `\\n\\n` 分隔。
        """
        items = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # 0=文本块，1=图片
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not text:
                    continue
                x0, y0, _, y1 = line["bbox"]
                items.append(((y0 + y1) / 2, x0, text, y1 - y0))
        if not items:
            return ""

        items.sort(key=lambda item: item[0])
        rows = []  # [y中心, [(x0, text)]]
        for yc, x0, text, height in items:
            tol = max(ROW_TOL_MIN, height * ROW_TOL_RATIO)
            if rows and abs(yc - rows[-1][0]) <= tol:
                rows[-1][1].append((x0, text))
                rows[-1][0] = (rows[-1][0] + yc) / 2
            else:
                rows.append([yc, [(x0, text)]])

        lines = [
            ("".join(t for _, t in sorted(group, key=lambda z: z[0])), min(x for x, _ in group), yc)
            for yc, group in rows
        ]

        # 基准行距取相邻行间距的众数：正文行距占绝对多数，段间距是少数异常值。
        # 用众数而不是平均值/中位数——标题、段距会把均值拉偏，而众数正好是正文行距。
        gaps = [round(b[2] - a[2], 1) for a, b in zip(lines, lines[1:]) if b[2] - a[2] > 0.5]
        line_gap = Counter(gaps).most_common(1)[0][0] if gaps else 0.0

        left = min(x for _, x, _ in lines)
        paragraphs: List[str] = []
        buf = ""
        prev_y = None
        for text, x0, yc in lines:
            new_para = bool(buf) and (
                x0 > left + INDENT_PT
                or (line_gap and prev_y is not None and yc - prev_y > line_gap * PARA_GAP_RATIO)
            )
            if new_para:
                paragraphs.append(buf)
                buf = text
            else:
                buf += text
            prev_y = yc
        if buf:
            paragraphs.append(buf)
        return "\n\n".join(paragraphs)

    def extract_page(self, page: "fitz.Page") -> tuple:
        """抽一页文本，返回 (text, text_area_ratio, blocks)。

        blocks 一并返回，是为了让 pdf_probe.page_metrics 直接复用同一批块：
        它需要 blocks 的 bbox 算正文区指标，若拿不到就只能再调一次
        get_text("blocks", sort=True)。带排序的 blocks 抽取是本阶段最贵的一步，
        重复抽一次等于把探测成本抬高一截。

        文本来源以 `_rebuild_text` 为主（dict 是 blocks 的同一份数据，多了行内
        span 结构），仅在「重建结果显著少于 raw」时退回 raw —— 那说明这条解析
        路径确实漏了正文，此时宁可接受碎行也不能丢内容。
        """
        blocks = pdf_probe.text_blocks(page)
        area = pdf_probe.text_area_ratio(page, blocks)

        rebuilt = self._rebuild_text(page)
        raw_text = page.get_text("text").strip()

        rebuilt_len = self._effective_len(rebuilt)
        raw_len = self._effective_len(raw_text)
        if raw_len >= rebuilt_len + 100 and raw_len >= rebuilt_len * 1.5:
            # 命中「重建大面积漏抽」：其 bbox 面积同样不可信，把面积拉到阈值下限，
            # 交给字数判据兜底（见 pdf_probe.has_real_text）。
            area = max(area, pdf_probe.MIN_TEXT_AREA)
            text = raw_text
        else:
            text = rebuilt

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