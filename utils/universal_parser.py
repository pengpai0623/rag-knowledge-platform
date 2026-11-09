"""统一文档解析器（含 PDF 页级自动降级链，v1.1 两阶段实现）。

策略（详见 docs/rag/doc_parser_fallback_design_v1.1.md）：
1. 扩展名路由
2. PDF 文档级闸门（打不开 / 加密 / 0 页 → 直接失败；页数对不上 → 仅告警）
3. 探测阶段：逐页抽文本 + 算指标，得到 PagePlan；**不做 OCR**
4. 预算校验：需要 OCR 的页数超过上限 → 立即失败（此时还没花 OCR 时间）
5. 执行阶段：按 PagePlan 走 文本采用 / pdfplumber 复核 / OCR / 跳过
6. 任意页失败 → 整单异常，不写入任何向量
7. 收口：统一补 metadata（parser_version / file_sha256 / source / parse_mode / …）

与 v1 的关键差异：
- 页级判定用「文本覆盖面积」而不是字符数，页眉页脚型伪文本层不会再冒充正文；
- 新增 figure 路由（图表页 OCR），扫描页与图表页的「OCR 为空」语义不同；
- source 在解析阶段就归一化为文件名，避免路径/文件名两套身份导致删不干净。
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import fitz
from langchain_core.documents import Document

from app.core.error_codes import (
    ERR_PARSE_ALL_BLANK,
    ERR_PARSE_EMPTY_CONTENT,
    ERR_PARSE_FAILED,
    ERR_PARSE_OCR_LIMIT,
    ERR_PARSE_UNSUPPORTED,
)
from app.core.exceptions import DocumentParseError, UnsupportedFileTypeError

from .docx_parser import DocxParser
from .logger import get_logger
from .ocr import OCRPDFParser
from .pdf_parser import PDFPlumberParser, PymuPDFParser, pdf_probe
from .pptx_parser import PptxParser
from .xlsx_parser import XlsxParser

logger = get_logger("utils.universal_parser")

# 解析器版本：解析策略变更时递增，用于识别库里哪些分片是旧策略产物。
# v3：docx 改为按 body XML 顺序遍历（表格回到正文中的真实位置），
#     产出顺序与 v2 不同，必须让旧数据重灌。
PARSER_VERSION = "v3"

# 非 PDF 类型 → (解析函数, parse_mode, fallback_chain, 中文类型名)
_OFFICE_PARSERS = {
    "docx": (DocxParser.parse, "docx", "python-docx", "Word"),
    "xlsx": (XlsxParser.parse, "xlsx", "openpyxl", "Excel"),
    "pptx": (PptxParser.parse, "pptx", "python-pptx", "PowerPoint"),
}

# layout → 默认动作
_LAYOUT_ACTION = {
    pdf_probe.TEXT: "text",
    pdf_probe.IMAGE_DOMINANT: "text",
    pdf_probe.GARBLED: "plumber",   # 有整页图时改走 ocr，见 _probe_pdf
    pdf_probe.SCANNED: "ocr",
    pdf_probe.FIGURE: "ocr",
    pdf_probe.BLANK: "skip",
}


@dataclass
class PagePlan:
    """一页的判定结果与执行计划（探测阶段的产物）"""
    page_no: int
    layout: str                 # text / image_dominant / garbled / scanned / figure / blank
    text: str                   # 探测阶段抽到的 PyMuPDF 文本，缓存复用
    metrics: Dict[str, float] = field(default_factory=dict)
    action: str = "text"        # text / plumber / ocr / skip
    parse_mode: str = ""
    chain: str = ""
    ocr_score: Optional[float] = None
    ocr_dropped: Optional[float] = None

    @property
    def chars(self) -> int:
        return int(self.metrics.get("chars", 0))


class UniversalDocParser:
    """
    统一文档解析器
    支持格式：.md / .txt / .pdf / .docx / .xlsx / .pptx
    统一输出 List[Document]，元数据规范一致
    """

    SUPPORTED_EXTENSIONS = {
        ".md": "markdown",
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".pptx": "pptx",
    }

    @classmethod
    def parse(cls, file_path: str, source_name: Optional[str] = None) -> List[Document]:
        """解析文件。

        :param file_path:   磁盘上的实际路径（上传场景下是临时文件）
        :param source_name: 原始文件名。上传链路必须传，否则 source 会变成 tmpXXXX.pdf，
                            导致引用展示和「按来源删除」全部失准。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            # 外部数据（用户上传的格式）不合法 → 业务异常，不用裸 ValueError
            raise UnsupportedFileTypeError(
                f"不支持的文件格式：{ext}，支持格式：{list(cls.SUPPORTED_EXTENSIONS.keys())}",
                code=ERR_PARSE_UNSUPPORTED,
            )

        file_type = cls.SUPPORTED_EXTENSIONS[ext]
        sha256 = cls.file_sha256(file_path)
        display_name = os.path.basename(source_name) if source_name else os.path.basename(file_path)
        # 一次解析只留「开始 + 完成」两条：片段数要等跑完才知道，所以用不上 log_span
        # （那样会和手写的完成日志重复记一遍耗时）
        started = time.perf_counter()
        logger.info(
            "开始 解析文档 | file=%s file_type=%s size=%.1fKB sha=%s…",
            display_name, file_type, os.path.getsize(file_path) / 1024, sha256[:8],
        )
        if file_type == "pdf":
            docs = cls._parse_pdf(file_path)
        elif file_type in ("markdown", "text"):
            docs = cls._parse_plain(file_path, file_type)
        else:
            docs = cls._parse_office(file_path, file_type)

        cls._finalize(docs, file_path, file_type, sha256, source_name=source_name)
        logger.info(
            "完成 解析文档 | file=%s file_type=%s 片段=%d | %.1fms",
            display_name, file_type, len(docs),
            (time.perf_counter() - started) * 1000,
        )
        return docs

    # -------------------------------------------------------------- 公共收口
    @staticmethod
    def file_sha256(file_path: str, limit: int = 16) -> str:
        """文件内容指纹（取前 16 位十六进制），用于幂等与版本识别"""
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()[:limit]

    @classmethod
    def _finalize(cls, docs: List[Document], file_path: str, file_type: str, sha256: str,
                  source_name: Optional[str] = None) -> None:
        """统一写入与具体解析器无关的元数据。

        source 归一化为文件名：入库、按来源删除、前端展示统一用这一个身份；
        原始路径另存 source_path，仅排障用。历史上「路径 vs 文件名」两套身份
        导致 delete_by_source 删不干净，这里一次性收掉。
        """
        source = os.path.basename(source_name) if source_name else os.path.basename(file_path)
        for doc in docs:
            doc.metadata["source"] = source
            doc.metadata["source_path"] = file_path
            doc.metadata["file_type"] = doc.metadata.get("file_type", file_type)
            doc.metadata["file_sha256"] = sha256
            doc.metadata["parser_version"] = PARSER_VERSION
            doc.metadata["source_text_chars"] = pdf_probe.effective_chars(doc.page_content)

    # ------------------------------------------------------------------ 文本
    @staticmethod
    def _read_text(file_path: str) -> str:
        """UTF-8 优先，失败回退 GB18030（Windows 上大量 txt 是 GBK 编码）"""
        for encoding in ("utf-8", "gb18030"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise DocumentParseError(
            "文本文件编码无法识别（已尝试 UTF-8 / GB18030）",
            code=ERR_PARSE_FAILED,
        )

    @classmethod
    def _parse_plain(cls, file_path: str, file_type: str) -> List[Document]:
        content = cls._read_text(file_path)
        if not content.strip():
            raise DocumentParseError("文本文件内容为空", code=ERR_PARSE_EMPTY_CONTENT)

        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_type": file_type,
                "position": "全文",
            },
        )
        cls._fill_metadata(
            doc,
            parse_mode="text",
            doc_parse_mode="text",
            layout_type=pdf_probe.TEXT,
            chain="text",
        )
        # 这里不打文件名：上传链路下 file_path 是临时文件，打印出来只会误导排障
        logger.info("解析完成 | file_type=%s chars=%d",
                    file_type, doc.metadata["source_text_chars"])
        return [doc]

    # ------------------------------------------------------------ office 类
    @classmethod
    def _parse_office(cls, file_path: str, file_type: str) -> List[Document]:
        parser, parse_mode, chain, cn_name = _OFFICE_PARSERS[file_type]
        try:
            docs = parser(file_path)
        except DocumentParseError:
            raise
        except Exception as e:
            raise DocumentParseError(
                f"{cn_name} 文档解析失败（文件可能已损坏）：{e}",
                code=ERR_PARSE_FAILED,
            ) from e

        chars = sum(pdf_probe.effective_chars(d.page_content) for d in docs)
        # 只判「完全没有有效字符」：v1 的 <30 阈值会把一页备忘录这类合法短文档误杀
        if not docs or chars == 0:
            raise DocumentParseError(
                f"{cn_name} 文档无有效文本（可能是纯图片文档），"
                f"已解析 {len(docs)} 个片段 / {chars} 个有效字符",
                code=ERR_PARSE_EMPTY_CONTENT,
            )

        for doc in docs:
            cls._fill_metadata(
                doc,
                parse_mode=parse_mode,
                doc_parse_mode="text",
                layout_type=pdf_probe.TEXT,
                chain=chain,
            )
        logger.info("解析完成 | file_type=%s 片段=%d chars=%d",
                    file_type, len(docs), chars)
        return docs

    # ------------------------------------------------------------------ PDF
    @classmethod
    def _parse_pdf(cls, file_path: str) -> List[Document]:
        doc = pdf_probe.open_document(file_path)
        plumber = None
        try:
            plumber = pdf_probe.check_document(doc, file_path)  # G2/G3/G5
            total_pages = doc.page_count

            # ---- 阶段一：探测（不 OCR）
            probe_started = time.perf_counter()
            plans = cls._probe_pdf(doc, total_pages, PymuPDFParser())
            stats = cls._layout_stats(plans)
            logger.info(
                "探测完成｜pages=%d text=%d scanned=%d figure=%d blank=%d garbled=%d ocr_pages=%d cost=%.2fs",
                total_pages, stats["text"], stats["scanned"], stats["figure"],
                stats["blank"], stats["garbled"], stats["ocr"],
                time.perf_counter() - probe_started,
            )

            # ---- 阶段二：OCR 预算校验（在花钱之前失败）
            cls._check_ocr_budget(plans, total_pages, stats)

            # ---- 阶段三：执行
            ocr_parser = OCRPDFParser(dpi=pdf_probe.OCR_DPI)  # 整篇复用，引擎只加载一次
            return cls._execute_pdf(
                doc, plans, plumber, PDFPlumberParser(), ocr_parser,
                total_pages=total_pages,
            )
        finally:
            if plumber is not None:
                try:
                    plumber.close()
                except Exception:
                    pass
            doc.close()

    # ---------------------------------------------------------- 探测阶段
    @classmethod
    def _probe_pdf(cls, doc: "fitz.Document", total_pages: int,
                   parser: PymuPDFParser) -> List[PagePlan]:
        plans: List[PagePlan] = []
        for idx in range(total_pages):
            page = doc[idx]
            text, area, blocks = parser.extract_page(page)
            metrics = pdf_probe.page_metrics(page, text, area, blocks)
            layout = metrics["layout_type"]
            action = _LAYOUT_ACTION.get(layout, "skip")
            # 乱码页压在整页扫描图上时，pdfplumber 只会复现同一份乱码 → 直接 OCR
            if layout == pdf_probe.GARBLED and metrics["img_cov"] >= pdf_probe.SCAN_IMAGE_RATIO:
                action = "ocr"
            plans.append(PagePlan(
                page_no=idx + 1, layout=layout, text=text, metrics=metrics, action=action,
            ))
        return plans

    @staticmethod
    def _layout_stats(plans: List[PagePlan]) -> Dict[str, int]:
        stats = {"text": 0, "scanned": 0, "figure": 0, "blank": 0, "garbled": 0, "ocr": 0}
        for plan in plans:
            if plan.action == "ocr":
                stats["ocr"] += 1
            if plan.layout in (pdf_probe.TEXT, pdf_probe.IMAGE_DOMINANT):
                stats["text"] += 1
            elif plan.layout == pdf_probe.SCANNED:
                stats["scanned"] += 1
            elif plan.layout == pdf_probe.FIGURE:
                stats["figure"] += 1
            elif plan.layout == pdf_probe.BLANK:
                stats["blank"] += 1
            elif plan.layout == pdf_probe.GARBLED:
                stats["garbled"] += 1
        return stats

    @staticmethod
    def _check_ocr_budget(plans: List[PagePlan], total_pages: int, stats: Dict[str, int]) -> None:
        """OCR 页数超限 → 立即失败，不再「边识别边计数」烧掉几十页的时间"""
        limit = pdf_probe.MAX_OCR_PAGES
        if stats["ocr"] <= limit:
            return
        raise DocumentParseError(
            f"需 OCR 的页数 {stats['ocr']} 超过上限 {limit}（共 {total_pages} 页，"
            f"其中扫描页 {stats['scanned']} 页、图表页 {stats['figure']} 页、"
            f"乱码页 {stats['garbled']} 页），请拆分文档后重试",
            code=ERR_PARSE_OCR_LIMIT,
        )

    # ---------------------------------------------------------- 执行阶段
    @classmethod
    def _execute_pdf(cls, doc: "fitz.Document", plans: List[PagePlan], plumber,
                     plumber_parser: PDFPlumberParser, ocr_parser: OCRPDFParser,
                     total_pages: int) -> List[Document]:
        docs: List[Document] = []
        blank_pages: List[int] = []
        scanned_pages: List[int] = []
        figure_pages: List[int] = []
        figure_skipped: List[int] = []
        ocr_dropped_total = 0.0

        for plan in plans:
            page_no = plan.page_no

            if plan.action == "skip":
                blank_pages.append(page_no)
                logger.debug(
                    "page=%d layout=blank chars=%d area=%.1f%% img=%.1f%% 跳过（不入库）",
                    page_no, plan.chars, plan.metrics["text_area_ratio"] * 100,
                    plan.metrics["img_cov"] * 100,
                )
                continue

            if plan.action == "text":
                cls._accept_text(plan, plan.text)
                if plan.layout == pdf_probe.IMAGE_DOMINANT and plan.chars < pdf_probe.DENSE_CHARS:
                    logger.warning(
                        "第 %d 页 image_dominant 但 chars=%d（<%d），文本层可能不完整；"
                        "如检索效果差可抽查该页或调整 PARSE_MIN_TEXT_AREA",
                        page_no, plan.chars, pdf_probe.DENSE_CHARS,
                    )

            elif plan.action == "plumber":
                retry_text = cls._plumber_page_text(plumber, plumber_parser, page_no - 1)
                retry_chars = pdf_probe.effective_chars(retry_text)
                retry_garbage = pdf_probe.garbage_ratio(retry_text)
                if (retry_chars >= pdf_probe.MIN_CHARS_PER_PAGE
                        and retry_garbage <= pdf_probe.GARBAGE_RATIO_THRESHOLD):
                    cls._accept_text(plan, retry_text, parse_mode="pdfplumber",
                                     chain="pymupdf>pdfplumber")
                else:
                    logger.error(
                        "第 %d/%d 页解析失败｜layout=garbled reason=双解析器均乱码"
                        "（pymupdf 乱码率 %.0f%%，pdfplumber 复核 %d 字 / 乱码率 %.0f%%）",
                        page_no, total_pages, plan.metrics["garbage_ratio"] * 100,
                        retry_chars, retry_garbage * 100,
                    )
                    raise DocumentParseError(
                        f"第 {page_no}/{total_pages} 页解析失败：文本编码异常"
                        f"（乱码率 {plan.metrics['garbage_ratio']:.0%}，pdfplumber 复核仍无效）",
                        code=ERR_PARSE_FAILED, page=page_no,
                    )

            elif plan.action == "ocr":
                page_started = time.perf_counter()
                ocr_text, score_avg, dropped = ocr_parser.parse_page(doc[page_no - 1])
                ocr_dropped_total += dropped
                if ocr_text.strip():
                    plan.ocr_score = score_avg
                    plan.ocr_dropped = dropped
                    plan.parse_mode = "ocr"
                    plan.chain = "ocr"
                    plan.text = ocr_text
                    if plan.layout == pdf_probe.FIGURE:
                        figure_pages.append(page_no)
                    else:
                        scanned_pages.append(page_no)
                    if dropped > 0.5:
                        logger.warning(
                            "第 %d 页 ocr_dropped=%.0f%%（低置信行被丢弃），建议人工抽查",
                            page_no, dropped * 100,
                        )
                    logger.debug(
                        "page=%d layout=%s chars=%d area=%.1f%% img=%.1f%% mode=ocr "
                        "score=%.2f dropped=%.0f%% cost=%.1fs",
                        page_no, plan.layout, plan.chars,
                        plan.metrics["text_area_ratio"] * 100,
                        plan.metrics["img_cov"] * 100,
                        score_avg, dropped * 100, time.perf_counter() - page_started,
                    )
                elif plan.layout == pdf_probe.FIGURE:
                    # 图表页本就是纯图形，识别不出文字是正常结果，不能因此整单失败
                    figure_skipped.append(page_no)
                    logger.info("第 %d 页 figure OCR 无文字，跳过（不算失败）", page_no)
                    continue
                else:
                    logger.error(
                        "第 %d/%d 页解析失败｜layout=%s reason=ocr_empty cost=%.1fs",
                        page_no, total_pages, plan.layout, time.perf_counter() - page_started,
                    )
                    raise DocumentParseError(
                        f"第 {page_no}/{total_pages} 页解析失败：OCR 未识别出文字",
                        code=ERR_PARSE_FAILED, page=page_no,
                    )

            docs.append(cls._build_document(plan, total_pages))

        # ---- 文档级汇总
        if not docs:
            raise DocumentParseError(
                f"PDF 无有效文本页（共 {total_pages} 页全部为空白/空壳页）",
                code=ERR_PARSE_ALL_BLANK,
            )

        has_text = any(d.metadata["layout_type"] in (pdf_probe.TEXT, pdf_probe.IMAGE_DOMINANT)
                       for d in docs)
        has_scan = any(d.metadata["layout_type"] == pdf_probe.SCANNED for d in docs)
        has_figure = any(d.metadata["layout_type"] == pdf_probe.FIGURE for d in docs)
        if has_text and (has_scan or has_figure):
            doc_parse_mode = "mixed"
        elif has_text:
            doc_parse_mode = "text"
        elif has_scan:
            doc_parse_mode = "scan"
        else:
            doc_parse_mode = "figure"
        for d in docs:
            d.metadata["doc_parse_mode"] = doc_parse_mode

        # ---- 汇总日志
        if scanned_pages:
            logger.info(
                "扫描完成｜当前文档共 %d 页，扫描版共 %d 页，分别是第 %s 页",
                total_pages, len(scanned_pages), ",".join(str(p) for p in scanned_pages),
            )
        else:
            logger.info("扫描完成｜当前文档共 %d 页，扫描版共 0 页", total_pages)
        if figure_pages or figure_skipped:
            logger.info(
                "图表页共 %d 页，已 OCR %s%s",
                len(figure_pages) + len(figure_skipped),
                ",".join(str(p) for p in figure_pages) or "无",
                f"；无文字跳过第 {','.join(str(p) for p in figure_skipped)} 页" if figure_skipped else "",
            )
        if blank_pages:
            logger.info(
                "跳过空白页 %d 页，分别是第 %s 页（未入库）",
                len(blank_pages), ",".join(str(p) for p in blank_pages),
            )

        ocr_page_count = len(scanned_pages) + len(figure_pages)
        logger.info(
            "解析汇总｜pages=%d text=%d ocr=%d blank=%d doc_parse_mode=%s ocr_dropped=%.1f%%",
            total_pages,
            sum(1 for d in docs
                if d.metadata["layout_type"] in (pdf_probe.TEXT, pdf_probe.IMAGE_DOMINANT)),
            ocr_page_count, len(blank_pages), doc_parse_mode,
            (ocr_dropped_total / ocr_page_count * 100) if ocr_page_count else 0.0,
        )
        return docs

    @staticmethod
    def _accept_text(plan: PagePlan, text: str, parse_mode: str = "pymupdf",
                     chain: str = "pymupdf") -> None:
        plan.text = text
        plan.parse_mode = parse_mode
        plan.chain = chain
        # 文本层能抽出来就说明这一页可用；乱码经复核通过后同样归为文本页
        if plan.layout in (pdf_probe.GARBLED, pdf_probe.SCANNED, pdf_probe.FIGURE):
            plan.layout = (pdf_probe.IMAGE_DOMINANT
                           if plan.metrics.get("img_cov", 0) >= pdf_probe.SCAN_IMAGE_RATIO
                           else pdf_probe.TEXT)

    @classmethod
    def _build_document(cls, plan: PagePlan, total_pages: int) -> Document:
        doc = Document(
            page_content=plan.text,
            metadata={
                "page_number": plan.page_no,
                "total_pages": total_pages,
                "file_type": "pdf",
            },
        )
        cls._fill_metadata(
            doc,
            parse_mode=plan.parse_mode or "pymupdf",
            layout_type=plan.layout,
            chain=plan.chain or "pymupdf",
            ocr_score_avg=plan.ocr_score,
        )
        return doc

    @staticmethod
    def _plumber_page_text(plumber, plumber_parser: PDFPlumberParser, page_index: int) -> str:
        """pdfplumber 复核某个页；不可用或出错时返回空串（不让复核拖垮主链）"""
        if plumber is None:
            return ""
        try:
            return plumber_parser.extract_page_text(plumber.pages[page_index])
        except Exception as e:
            logger.debug("pdfplumber 复核第 %d 页失败：%s", page_index + 1, e)
            return ""

    # -------------------------------------------------------------- metadata
    @staticmethod
    def _fill_metadata(doc: Document, parse_mode: str, layout_type: str, chain: str,
                       doc_parse_mode: Optional[str] = None,
                       ocr_score_avg: Optional[float] = None) -> None:
        """统一补齐解析元数据。

        写入前清掉值为 None 的键：Chroma 的 metadata 只接受 str/int/float/bool，
        例如 docx 段落里的 heading_level=None 会导致入库报错。
        """
        for key in [k for k, v in doc.metadata.items() if v is None]:
            doc.metadata.pop(key)

        doc.metadata["parse_mode"] = parse_mode
        if doc_parse_mode is not None:
            doc.metadata["doc_parse_mode"] = doc_parse_mode
        doc.metadata["layout_type"] = layout_type
        doc.metadata["source_text_chars"] = pdf_probe.effective_chars(doc.page_content)
        doc.metadata["fallback_chain"] = chain
        if ocr_score_avg is not None:
            doc.metadata["ocr_score_avg"] = float(ocr_score_avg)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "rag", "UN38.3.pdf",
    )
    result = UniversalDocParser.parse(target)
    print(f"解析完成，共 {len(result)} 个片段")
    for item in result[:3]:
        md = item.metadata
        print(f"  第{md.get('page_number', '-')}页 | mode={md['parse_mode']} | "
              f"layout={md['layout_type']} | chars={md.get('source_text_chars', '-')} | "
              f"content={item.page_content[:40]!r}")
