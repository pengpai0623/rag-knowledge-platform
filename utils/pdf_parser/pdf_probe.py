"""PDF 健康探针：文档级闸门 + 页级指标与分类。

职责边界：本模块只产出「数据与判定」，不做任何解析、不写 metadata，
路由与降级逻辑全部在 utils/universal_parser.py。

页级判定以「正文区文本量」为主判据，而不是全页字符数：
页眉页脚/水印常贡献 30~80 个字符，但只占页面 2%~4% 面积，
而正常正文页覆盖率在 15%~40%。用面积能区分「伪文本层」和「真文本层」，字数不能。
但幻灯片/封面的文字本身就稀疏（覆盖率可能 1%~2%），所以再加一条
「正文区（剔除页眉页脚带）字数 ≥ 20」的旁路，避免把干净的文本层误送去 OCR。

判定阈值均可通过环境变量调整：
- PARSE_MIN_CHARS_PER_PAGE：字数达到该值才进入乱码判定，默认 30
- PARSE_MIN_TEXT_AREA    ：正文区文本覆盖面积占比下限，默认 0.05
- PARSE_MIN_BODY_CHARS   ：正文区最少字符数，默认 20
- PARSE_BODY_TOP_RATIO   ：正文区上边界（占页高比例），默认 0.08
- PARSE_BODY_BOTTOM_RATIO：正文区下边界，默认 0.92
- PARSE_DENSE_CHARS      ：全页字数达到该值即视为真实文本层，默认 300
- PARSE_GARBAGE_RATIO    ：乱码率上限，默认 0.30
- PARSE_SCAN_IMAGE_RATIO ：图片覆盖率下限（求和后），达到即视为整页图，默认 0.70
- PARSE_FIGURE_IMAGE_RATIO：图片覆盖率下限，达到即视为图表页，默认 0.15
- PARSE_MAX_OCR_PAGES    ：单文档 OCR 页数上限，默认 50
- OCR_DPI                ：OCR 渲染分辨率，默认 200
"""

import os
import re
from typing import Optional, Sequence

import fitz

from app.core.error_codes import ERR_PARSE_ENCRYPTED, ERR_PARSE_NO_PAGES, ERR_PARSE_OPEN_FAILED
from app.core.exceptions import DocumentParseError
from ..logger import get_logger

logger = get_logger(__name__)

MIN_CHARS_PER_PAGE = int(os.getenv("PARSE_MIN_CHARS_PER_PAGE", "30"))
MIN_TEXT_AREA = float(os.getenv("PARSE_MIN_TEXT_AREA", "0.05"))
MIN_BODY_CHARS = int(os.getenv("PARSE_MIN_BODY_CHARS", "20"))
BODY_TOP_RATIO = float(os.getenv("PARSE_BODY_TOP_RATIO", "0.08"))
BODY_BOTTOM_RATIO = float(os.getenv("PARSE_BODY_BOTTOM_RATIO", "0.92"))
DENSE_CHARS = int(os.getenv("PARSE_DENSE_CHARS", "300"))
GARBAGE_RATIO_THRESHOLD = float(os.getenv("PARSE_GARBAGE_RATIO", "0.30"))
SCAN_IMAGE_RATIO = float(os.getenv("PARSE_SCAN_IMAGE_RATIO", "0.70"))
FIGURE_IMAGE_RATIO = float(os.getenv("PARSE_FIGURE_IMAGE_RATIO", "0.15"))
MAX_OCR_PAGES = int(os.getenv("PARSE_MAX_OCR_PAGES", "50"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))

# layout_type 的正式取值（写入 metadata）；blank 不入库，故实际只落前四个
TEXT = "text"
IMAGE_DOMINANT = "image_dominant"
SCANNED = "scanned"
FIGURE = "figure"
BLANK = "blank"
# 内部临时态：有文本但乱码率过高，需要复核后才决定用文本还是转 OCR，不写入 metadata
GARBLED = "garbled"

# 允许出现的字符：\w（中文/字母/数字/下划线）+ ASCII 可打印区 + 中英文标点与常见符号。
# 其余（U+FFFD、控制符、私用区、解码错位产生的外文残字）计入乱码。
#
# ASCII 可打印区（\x20-\x7e）必须显式列入：\w 只含字母数字下划线，不含半角标点。
# 早期版本漏了这一段，`.` `,` `;` `:` `(` `)` `/` `%` `-` `=` 等全部被算成乱码，
# 数值/参数密集的页面乱码率虚高（实测一段参数表格样本达 0.31），
# 超过 0.30 阈值后会被误判成 garbled → pdfplumber 复核仍用同一判据 → 整单失败。
_ALLOWED_CHARS = re.compile(
    r"[\w\x20-\x7e\u3000-\u303f\uff01-\uff5e\u2010-\u201f\u2026"
    r"\u2190-\u21ff\u2200-\u22ff\u2460-\u24ff"
    r"\u00a0-\u00bf\u00b7\u2100-\u214f]"
)


def effective_chars(text: str) -> int:
    """去掉所有空白后的有效字符数"""
    return len(re.sub(r"\s+", "", text or ""))


def garbage_ratio(text: str) -> float:
    """乱码率 = 不在允许字符集内的字符 / 非空白字符；文本为空时返回 0。

    判的是「编码失败产物」，不是「用词是否规范」：U+FFFD 替换符、控制字符、
    私用区字符、以及把 GBK 当 UTF-8 解出来的外文残字，都落在允许集之外。
    正常排版里的半角标点、全角标点、数学符号都在允许集内，不计入。
    """
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    allowed = len(_ALLOWED_CHARS.findall(compact))
    return round(1.0 - allowed / len(compact), 4)


def text_blocks(page: "fitz.Page") -> Sequence[tuple]:
    """取页面的文本块（按坐标排序，能还原双栏/多区块的阅读顺序）"""
    try:
        return [b for b in page.get_text("blocks", sort=True) if len(b) > 6 and b[6] == 0]
    except Exception:  # 极端畸形页兜底：调用方按「无文本块」处理
        return []


def text_area_ratio(page: "fitz.Page", blocks: Optional[Sequence[tuple]] = None) -> float:
    """文本覆盖面积占比 = Σ(文本块 bbox 面积) / 页面面积，封顶 1.0。

    这是区分「真文本层」和「页眉页脚型伪文本层」的主判据。
    """
    page_area = abs(page.rect)
    if page_area <= 0:
        return 0.0
    if blocks is None:
        # (x0, y0, x1, y1, "这一块的文字内容", block编号, 类型)
        blocks = text_blocks(page)
    total = 0.0
    for block in blocks:
        try:
            total += abs(fitz.Rect(block[:4]))
        except Exception:
            continue
    return round(min(total / page_area, 1.0), 4)


def image_coverage(page: "fitz.Page") -> tuple:
    """图片覆盖率，返回 (cov, max_ratio)。

    cov 用「求和」而不是「取单张最大」：拼贴式扫描页由多张切片拼成，
    每张可能都 < 0.7，只取最大值会把它误判成空白页。
    求和会因重叠图片超过 100%，因此封顶 1.0；max_ratio 仅留作日志排障。
    """
    page_area = abs(page.rect)
    if page_area <= 0:
        return 0.0, 0.0
    total = 0.0
    best = 0.0
    for info in page.get_image_info():
        try:
            ratio = abs(fitz.Rect(info.get("bbox", (0, 0, 0, 0)))) / page_area
        except Exception:
            continue
        total += ratio
        if ratio > best:
            best = ratio
    return round(min(total, 1.0), 4), round(min(best, 1.0), 4)


def body_text_metrics(page: "fitz.Page", blocks: Optional[Sequence[tuple]] = None) -> tuple:
    """正文区（剔除页眉/页脚带）的有效字符数与覆盖面积占比，返回 (chars, area_ratio)。

    为什么还要区分正文区：页眉页脚是最典型的「伪文本层」来源，
    但幻灯片、封面这类页面本身文字就稀疏（整页覆盖率可能只有 1%~2%），
    只按面积一刀切会把它们误判成扫描页、白跑 OCR 还引入识别噪声。
    正文区有 20 个字以上，基本可以确定「这一页的文字就是正文」。
    """
    page_area = abs(page.rect)
    height = page.rect.height
    if page_area <= 0:
        return 0, 0.0
    if blocks is None:
        blocks = text_blocks(page)
    top, bottom = height * BODY_TOP_RATIO, height * BODY_BOTTOM_RATIO
    chars = 0
    total = 0.0
    for block in blocks:
        try:
            rect = fitz.Rect(block[:4])
        except Exception:
            continue
        center_y = (rect.y0 + rect.y1) / 2
        if not (top <= center_y <= bottom):
            continue
        chars += effective_chars(block[4])
        total += abs(rect)
    return chars, round(min(total / page_area, 1.0), 4)


def has_real_text(chars: int, body_chars: int, body_area: float) -> bool:
    """是否存在真实文本层。

    - 正文区有足够的字（≥20）→ 是：普通正文页、要点稀疏的幻灯片都算
    - 正文区覆盖面积达到阈值 → 是：大字标题/封面（字少但面积大）
    - 全页字数极多 → 是：再密的页眉也凑不出 300 字
    三者都不满足 → 只有页眉页脚/水印/零星页码，不算有效文本层。
    """
    return body_chars >= MIN_BODY_CHARS or body_area >= MIN_TEXT_AREA or chars >= DENSE_CHARS


def classify(chars: int, garbage: float, body_chars: int, body_area: float,
             img_cov: float) -> str:
    """页级分类，严格按顺序先命中先返回（判定表见设计文档 v1.1 §4.2）"""
    # 1. 有文本但乱码率过高 → 待复核（复核阶段决定用 pdfplumber 还是转 OCR）
    if chars >= MIN_CHARS_PER_PAGE and garbage > GARBAGE_RATIO_THRESHOLD:
        return GARBLED
    # 2/3. 真实文本层：整页图上压着文本层时仍用文本层（多为别人 OCR 过）
    if has_real_text(chars, body_chars, body_area):
        return IMAGE_DOMINANT if img_cov >= SCAN_IMAGE_RATIO else TEXT
    # 4. 无有效文本 + 整页图 → 扫描页
    if img_cov >= SCAN_IMAGE_RATIO:
        return SCANNED
    # 5. 有图但没占满整页 → 图表/截图页（图里很可能有文字，走 OCR 而不是丢弃）
    #    FIGURE_IMAGE_RATIO 设为 0 表示关闭该路由（回退到 v1：图表页按空白跳过）
    if FIGURE_IMAGE_RATIO > 0 and img_cov >= FIGURE_IMAGE_RATIO:
        return FIGURE
    # 6. 其余：真空白页 / 空壳页
    return BLANK


def page_metrics(page: "fitz.Page", text: str, area: Optional[float] = None,
                 blocks: Optional[Sequence[tuple]] = None) -> dict:
    """一次抽文本即可拿到全部判定指标，避免「先判定再重新抽取」跑两遍。

    返回值分三组，调用方不要混用：

    - **决策指标**（进入 classify）：`chars` / `garbage_ratio` /
      `body_chars` / `body_area_ratio` / `img_cov`
    - **诊断指标**（只进日志，不参与任何判定）：`text_area_ratio` / `img_max`
    - **结论**：`layout_type`

    口径提醒：`chars` 与 `garbage_ratio` 基于传入的 `text`（调用方通常已做
    噪声清洗），`body_chars` / `body_area_ratio` 基于**原始 blocks**。
    目录页的引导点线会被清洗掉，两者可能相差一个数量级，排障时不要当成 bug。
    """
    chars = effective_chars(text)
    garbage = garbage_ratio(text)
    if blocks is None:
        blocks = text_blocks(page)
    if area is None:
        area = text_area_ratio(page, blocks)
    body_chars, body_area = body_text_metrics(page, blocks)
    cov, cov_max = image_coverage(page)
    return {
        "chars": chars,                 # 决策：乱码门槛 + ≥300 兜底
        "body_chars": body_chars,       # 决策：主判据，≥20 即认为正文有字
        "garbage_ratio": garbage,       # 决策：>30% 判 garbled
        "text_area_ratio": area,        # 诊断：全页文本覆盖率，仅日志
        "body_area_ratio": body_area,   # 决策：主判据，≥5%（字少面积大的封面）
        "img_cov": cov,                 # 决策：≥70% 扫描页，≥15% 图表页
        "img_max": cov_max,             # 诊断：单张图最大占比，仅日志
        "layout_type": classify(chars, garbage, body_chars, body_area, cov),
    }


def open_document(file_path: str) -> "fitz.Document":
    """G1：打不开即结构损坏"""
    try:
        # PyMuPDF
        return fitz.open(file_path)
    except Exception as e:  # PyMuPDF 抛的异常类型很杂，统一收敛
        raise DocumentParseError(
            f"PDF 无法打开（结构损坏或格式异常）：{e}",
            code=ERR_PARSE_OPEN_FAILED,
        ) from e


def check_document(doc: "fitz.Document", file_path: str) -> Optional[object]:
    """文档级闸门 G2/G3/G4/G5，返回 pdfplumber 句柄（可能为 None）。

    - G2 加密：需要密码时直接失败，不尝试 OCR、不猜测内容
    - G3 页数为 0：空文件
    - G4 打开时被修复（is_repaired）：**只告警，不判损坏**。xref 损坏的文件
      fitz 能自动修复并正常读取，一刀切判损坏会误杀；真正兜底的是执行阶段
      的 all_blank 判定——修复后若全篇抽不出有效文本，仍会以 400 失败。
    - G5 页数对不上：**只告警，不判损坏**。fitz 与 pdfminer 对畸形 PDF 的页数
      解读不同是常事，一刀切判损坏会把可读文件误杀成 400。
      两侧页数不一致时 pdfplumber 无法按页下标对齐，直接放弃复核链路（返回 None），
      后续只用 PyMuPDF 链路解析。
    """
    if doc.needs_pass or doc.is_encrypted:
        raise DocumentParseError("PDF 已加密，需要密码，当前无法解析", code=ERR_PARSE_ENCRYPTED)
    if doc.page_count == 0:
        raise DocumentParseError("PDF 页数为 0，无可解析内容", code=ERR_PARSE_NO_PAGES)
    if getattr(doc, "is_repaired", False):
        logger.warning(
            "PDF 打开时被自动修复（xref 损坏），文件结构不完整；继续解析，"
            "若全篇抽不出有效文本将失败"
        )

    plumber = None
    try:
        import pdfplumber

        plumber = pdfplumber.open(file_path) # 实例plumber对象
    except Exception as e:
        logger.warning("pdfplumber 打开失败，仅使用 PyMuPDF 链路：%s", e)
        return None

    try:
        plumber_pages = len(plumber.pages)
    except Exception as e:
        logger.warning("pdfplumber 读取页面树失败，仅使用 PyMuPDF 链路：%s", e)
        plumber.close()
        return None

    if plumber_pages != doc.page_count:
        logger.warning(
            "PDF 页数不一致（fitz=%d, pdfplumber=%d），以 fitz 为准，pdfplumber 复核链路已禁用",
            doc.page_count, plumber_pages,
        )
        plumber.close()
        return None
    return plumber
