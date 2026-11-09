"""OCR 兜底解析器（RapidOCR / ONNX Runtime 后端）。

为什么是 RapidOCR：
- PaddleOCR 依赖的 paddlepaddle 在 Windows 上把 protobuf 钉死在 <=3.20.2，
  而 chromadb / pymilvus 的 OTel 与 gRPC 链要求 protobuf>=5（chromadb 实际要 >=6.33.5），
  同一环境无法共存。RapidOCR 走 onnxruntime，不引入 protobuf 版本约束，
  识别模型仍是 PP-OCR 系列（det + rec + cls），效果与 PaddleOCR 同源。
- 引擎惰性初始化：import utils.ocr 不会加载模型，只有真正调用 parse_by_page 时才建引擎。

对外契约与旧实现保持一致：OCRPDFParser().parse_by_page(path) -> List[Document]，
metadata 含 source / page_number / total_pages / file_type / parse_mode。
"""

from typing import List

import fitz  # PyMuPDF，用于把 PDF 页渲染成位图
import numpy as np
from langchain_core.documents import Document

from .logger import get_logger

logger = get_logger(__name__)

# RapidOCR 的语言模型：ch 模型同时覆盖中英文，也是默认值
_SUPPORTED_LANGS = {"ch", "en"}


class OCRPDFParser:
    def __init__(self, lang: str = "ch", dpi: int = 200, use_angle_cls: bool = True,
                 min_score: float = 0.5):
        if lang not in _SUPPORTED_LANGS:
            raise ValueError(
                f"RapidOCR 仅支持 {sorted(_SUPPORTED_LANGS)}，收到 lang={lang!r}"
            )
        self.lang = lang
        self.dpi = dpi
        self.use_angle_cls = use_angle_cls
        self.min_score = min_score
        self._engine = None  # 惰性初始化，避免 import 阶段加载模型

    def _get_engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            kwargs = {"use_cls": self.use_angle_cls}
            if self.lang == "en":
                kwargs["lang"] = "en"
            self._engine = RapidOCR(**kwargs)
        return self._engine

    @staticmethod
    def _render_page(page: "fitz.Page", dpi: int) -> np.ndarray:
        """把 PDF 页渲染为 BGR 三通道 ndarray，供 OCR 引擎消费。"""
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:  # RGBA -> BGR
            arr = arr[:, :, :3][:, :, ::-1]
        else:  # RGB -> BGR
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    @staticmethod
    def _box_order(box) -> tuple:
        """把 4 点 bbox 归一成 (上边界 y, 左边界 x)，作为行排序键。"""
        try:
            ys = [float(p[1]) for p in box]
            xs = [float(p[0]) for p in box]
            return (round(min(ys), 1), round(min(xs), 1))
        except Exception:
            return (0.0, 0.0)

    def _ocr(self, image: np.ndarray) -> tuple:
        """识别一页，返回 (行列表, 统计)。

        行列表按 bbox 的 (y, x) 排序后再拼接：引擎返回的顺序在多栏/表格页上会串行，
        按坐标排一次成本为零，至少不会比原来更差。
        统计给出 total / kept，用于计算「低置信行丢弃率」——
        min_score 过滤是静默丢内容，只有把这个比例暴露出来才能发现识别质量塌了。
        """
        result, _elapse = self._get_engine()(image)
        stats = {"total": 0, "kept": 0}
        if not result:
            return [], stats

        items = []
        for line in result:
            if len(line) < 3:
                continue
            box, text, score = line[0], line[1], float(line[2])
            if not text:
                continue
            stats["total"] += 1
            if score < self.min_score:
                continue  # 低置信行（水印残字、印章碎片）丢弃，计入 dropped
            items.append((text.strip(), score, *self._box_order(box)))

        items.sort(key=lambda it: (it[2], it[3]))
        stats["kept"] = len(items)

        return [(text, score) for text, score, *_ in items], stats

    def _ocr_image(self, image: np.ndarray) -> List[str]:
        lines, _stats = self._ocr(image)
        return [text for text, _ in lines]

    def parse_page(self, page: "fitz.Page") -> tuple:
        """页级 OCR（降级链用）：渲染该页 → 识别，返回 (文本, 平均置信度, 丢弃率)

        引擎是实例级单例，整篇复用同一个 OCRPDFParser 对象即可，不要逐页新建，
        否则每页都会重新加载 ONNX 模型（1~2 秒 × 页数）。
        """
        lines, stats = self._ocr(self._render_page(page, self.dpi))
        if not lines:
            dropped = 1.0 if stats["total"] else 0.0
            return "", 0.0, dropped
        text = "\n".join(t for t, _ in lines)
        score_avg = round(sum(s for _, s in lines) / len(lines), 4)
        dropped = round(1.0 - stats["kept"] / stats["total"], 4) if stats["total"] else 0.0
        return text, score_avg, dropped

    def parse_by_page(self, file_path: str) -> List[Document]:
        docs: List[Document] = []
        with fitz.open(file_path) as pdf:
            total_pages = pdf.page_count
            for page_idx in range(total_pages):
                page_text, score_avg, _dropped = self.parse_page(pdf[page_idx])
                if not page_text.strip():
                    continue  # 空白页/纯图无文字页不入库，避免空 Document 污染检索
                docs.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": file_path,
                            "page_number": page_idx + 1,
                            "total_pages": total_pages,
                            "file_type": "pdf",
                            "parse_mode": "ocr",
                            "ocr_score_avg": score_avg,
                        },
                    )
                )
        return docs


if __name__ == "__main__":
    from pathlib import Path

    pdf_path = Path(__file__).resolve().parent.parent / "docs" / "rag" / "UN38.3.pdf"
    parser = OCRPDFParser()
    docs = parser.parse_by_page(str(pdf_path))
    print(f"OCR解析完成，共 {len(docs)} 页")
    if docs:
        print(f"第1页摘要：{docs[0].page_content[:120]}...")
