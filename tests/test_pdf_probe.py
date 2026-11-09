"""pdf_probe 判定逻辑单测（不依赖 pytest，也不触网、不跑 OCR）。

运行方式：
    python tests/test_pdf_probe.py          # 直接跑，全绿输出 OK
    pytest tests/test_pdf_probe.py          # 装了 pytest 也能跑

覆盖 v1.1 相对 v1 的三处关键修正：
1. 页眉页脚型「伪文本层」不能被当成正文（v1 的漏网场景）；
2. 图表页不能被当成空白页丢弃；
3. 拼贴扫描页不能因为「单张图都 < 70%」被判空白。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz

from utils.pdf_parser import pdf_probe

W, H = 595, 842


def _blank_doc():
    doc = fitz.open()
    return doc, doc.new_page(width=W, height=H)


def _solid_image_bytes():
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (180, 180, 180))
    return pix.tobytes("png")


def _make_header_only_scan():
    """整页扫描图 + 只有页眉页脚文本（真实扫描件最常见的形态）"""
    doc, page = _blank_doc()
    page.insert_image(page.rect, stream=_solid_image_bytes())
    page.insert_text((50, 40), "Internal Report Page 3 of 13", fontsize=9)
    page.insert_text((50, 54), "Confidential - Transport Test Summary", fontsize=9)
    page.insert_text((50, H - 30), "Doc No. SZHH0216638705", fontsize=9)
    return doc, page


def _make_text_page():
    doc, page = _blank_doc()
    for i in range(28):
        page.insert_text((50, 60 + i * 24),
                         "The battery test report describes transport requirements.", fontsize=11)
    return doc, page


def _make_tiled_scan():
    """两半页拼贴成整页扫描：单张都只有 50%，求和才是 100%"""
    doc, page = _blank_doc()
    img = _solid_image_bytes()
    page.insert_image(fitz.Rect(0, 0, W, H / 2), stream=img)
    page.insert_image(fitz.Rect(0, H / 2, W, H), stream=img)
    return doc, page


def _make_figure_page():
    """半页流程图：有图、没文字"""
    doc, page = _blank_doc()
    page.insert_image(fitz.Rect(60, 200, W - 60, 560), stream=_solid_image_bytes())
    return doc, page


def _make_blank_page():
    return _blank_doc()


# ---------------------------------------------------------------- 纯函数判定
def test_classify_matrix():
    # 正常文本页（全页 800 字，正文区 800 字，面积 20%）
    assert pdf_probe.classify(800, 0.0, 800, 0.20, 0.0) == pdf_probe.TEXT
    # 页眉页脚型伪文本层：全页 40 字但正文区 0 字，且整页是图 → 必须判扫描页
    assert pdf_probe.classify(40, 0.0, 0, 0.02, 0.90) == pdf_probe.SCANNED
    # 真文本层压在整页图上（别人 OCR 过）→ 用文本层
    assert pdf_probe.classify(800, 0.0, 800, 0.20, 0.90) == pdf_probe.IMAGE_DOMINANT
    # 乱码（有文本、面积够、乱码率高）
    assert pdf_probe.classify(200, 0.50, 200, 0.20, 0.0) == pdf_probe.GARBLED
    # 乱码但字数不足门槛 → 不进乱码分支，按图判定
    assert pdf_probe.classify(10, 0.60, 10, 0.0, 0.90) == pdf_probe.SCANNED
    # 图表页：有图（40%）、没文字
    assert pdf_probe.classify(3, 0.0, 0, 0.0, 0.40) == pdf_probe.FIGURE
    # 真空白
    assert pdf_probe.classify(0, 0.0, 0, 0.0, 0.0) == pdf_probe.BLANK
    # 少量零散文字、无图（只有页码）→ 空白
    assert pdf_probe.classify(4, 0.0, 4, 0.01, 0.0) == pdf_probe.BLANK
    # 稀疏文字的幻灯片（24 字、覆盖率不到 1%、有背景图）→ 用文本层，不能送去 OCR
    assert pdf_probe.classify(24, 0.0, 24, 0.008, 0.53) == pdf_probe.TEXT


def test_has_real_text():
    assert pdf_probe.has_real_text(800, 800, 0.30) is True
    # 大字标题页：字数少但面积够
    assert pdf_probe.has_real_text(12, 12, 0.08) is True
    # 极密页：面积异常但字数极多
    assert pdf_probe.has_real_text(400, 0, 0.0) is True
    # 稀疏幻灯片：字少、面积小，但字都在正文区
    assert pdf_probe.has_real_text(24, 24, 0.008) is True
    # 页眉页脚：正文区没有字
    assert pdf_probe.has_real_text(40, 0, 0.02) is False


def test_garbage_ratio():
    assert pdf_probe.garbage_ratio("正常的中文文本") == 0.0
    assert pdf_probe.garbage_ratio("") == 0.0
    assert pdf_probe.garbage_ratio("\ufffd\ufffd\ufffd\ufffd") == 1.0
    assert 0.0 < pdf_probe.garbage_ratio("正常\ufffd\ufffd\ufffd\ufffd") < 1.0
    # 控制字符/私用区同样属于乱码（编码失败产物）
    assert pdf_probe.garbage_ratio("\x00\x01\x02\x03") == 1.0
    assert pdf_probe.garbage_ratio("\ue000\ue001\ue002") == 1.0


def test_garbage_ratio_ignores_ascii_punctuation():
    """回归：半角标点不算乱码。

    修复前 `_ALLOWED_CHARS` 只有 \\w + 中文标点，ASCII 的 . , ; : ( ) / % - 等
    全部被计为乱码，参数密集页乱码率虚高到 0.31（超过 0.30 阈值），
    会被判成 garbled 并最终导致整单上传失败。
    """
    numeric_table = "Item 1: T1 = 72h, T2 = 25C, Delta = 0.5% / 1.0%"
    assert pdf_probe.garbage_ratio(numeric_table) == 0.0
    english = "Battery test report: 3.5kg (7.7lb), temp -20C to +60C, 100% SOC"
    assert pdf_probe.garbage_ratio(english) < 0.30
    # 判据仍能拦住真正的乱码：混入替换符后必须升高
    assert pdf_probe.garbage_ratio(numeric_table + "\ufffd" * 30) > 0.30


# ------------------------------------------------------------ 合成 PDF 集成
def test_header_only_scan_is_not_text():
    doc, page = _make_header_only_scan()
    text, area, _ = _extract(page)
    metrics = pdf_probe.page_metrics(page, text, area)
    doc.close()
    # 这是 v1 的漏洞：chars 足够，v1 会判 image_dominant 并用几乎空的文本层
    assert metrics["chars"] >= pdf_probe.MIN_CHARS_PER_PAGE
    assert metrics["layout_type"] == pdf_probe.SCANNED, metrics


def test_text_page_stays_text():
    doc, page = _make_text_page()
    text, area, _ = _extract(page)
    metrics = pdf_probe.page_metrics(page, text, area)
    doc.close()
    assert metrics["layout_type"] == pdf_probe.TEXT, metrics
    assert metrics["text_area_ratio"] >= pdf_probe.MIN_TEXT_AREA


def test_tiled_scan_uses_summed_coverage():
    doc, page = _make_tiled_scan()
    cov, cov_max = pdf_probe.image_coverage(page)
    metrics = pdf_probe.page_metrics(page, "", 0.0)
    doc.close()
    assert cov_max <= 0.55, f"单张图应只占一半，实际 {cov_max}"
    assert cov >= 0.95, f"求和后应接近整页，实际 {cov}"
    assert metrics["layout_type"] == pdf_probe.SCANNED, metrics


def test_figure_page_is_not_blank():
    doc, page = _make_figure_page()
    metrics = pdf_probe.page_metrics(page, "", 0.0)
    doc.close()
    assert metrics["layout_type"] == pdf_probe.FIGURE, metrics


def test_blank_page():
    doc, page = _make_blank_page()
    metrics = pdf_probe.page_metrics(page, "", 0.0)
    doc.close()
    assert metrics["layout_type"] == pdf_probe.BLANK, metrics


def _extract(page):
    from utils.pdf_parser.pymupdf_parser import PymuPDFParser

    return PymuPDFParser().extract_page(page)


def test_g4_repaired_pdf_passes_with_warning():
    """G4：xref 损坏被 fitz 自动修复的文件应放行（只告警），不判损坏"""
    import os
    import tempfile

    import fitz

    path = os.path.join(tempfile.gettempdir(), "_g4_broken_xref.pdf")
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "G4 repaired pdf test page", fontsize=12)
    doc.save(path)
    doc.close()
    data = bytearray(open(path, "rb").read())
    i = data.rfind(b"startxref")
    data[i + 9: i + 20] = b"9999999999"  # 破坏 xref 偏移，迫使 fitz 走修复路径
    open(path, "wb").write(data)

    try:
        d = fitz.open(path)
        assert d.is_repaired, "构造失败：fitz 未触发自动修复"
        handle = pdf_probe.check_document(d, path)  # 不应抛异常
        assert handle is None  # xref 已破坏，pdfplumber 打不开 → 复核链路禁用
        d.close()
    finally:
        os.remove(path)


if __name__ == "__main__":
    cases = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
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
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
