import os
import shutil
import tempfile
from typing import List

from langchain_core.documents import Document
from openpyxl import load_workbook

from app.core.error_codes import ERR_PARSE_XLSX_BROKEN
from app.core.exceptions import XlsxStyleBrokenError

from .logger import get_logger

logger = get_logger("utils.xlsx_parser")


def _repair_with_excel(file_path: str) -> tuple:
    """用本机 Excel（xlwings / COM）另存一份，绕开 openpyxl 不认的样式表。

    - xlwings 惰性导入：它依赖本机 Office，且不是所有环境都装。
      顶层 import 会让 `import utils` 直接 ModuleNotFoundError，整个服务起不来。
    - 修复产物写到独立临时目录，由调用方在 finally 中清理，不再污染源目录。
    返回 (修复后路径, 需要清理的临时目录)。
    """
    try:
        import xlwings as xw
    except ImportError as e:
        raise XlsxStyleBrokenError(
            "Excel 样式表损坏，且当前环境未安装 xlwings（该修复需要本机 Excel）。"
            "请用 Excel 另存为 .xlsx 后重新上传",
            code=ERR_PARSE_XLSX_BROKEN,
        ) from e

    tmp_dir = tempfile.mkdtemp(prefix="rag_xlsx_fix_")
    fixed_path = os.path.join(tmp_dir, "fixed.xlsx")
    try:
        with xw.App(visible=False) as app:
            wb_xw = app.books.open(file_path)
            wb_xw.save(fixed_path)
            wb_xw.close()
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise XlsxStyleBrokenError(
            f"Excel 样式表损坏，自动修复失败：{e}。请用 Excel 另存为 .xlsx 后重新上传",
            code=ERR_PARSE_XLSX_BROKEN,
        ) from e

    logger.warning("Excel 样式损坏，已用本机 Excel 另存修复：%s", os.path.basename(file_path))
    return fixed_path, tmp_dir


def safe_load_workbook(file_path: str, max_retries: int = 1) -> tuple:
    """优先 openpyxl 直读；遇到样式表损坏才尝试用 Excel 另存修复。

    返回 (workbook, 需清理的临时目录或 None)。
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return load_workbook(file_path, read_only=True, data_only=True), None
        except TypeError as e:
            # 只有 openpyxl 对样式（Fill）结构的报错才走修复分支，
            # 其他 TypeError 属于代码/参数问题，直接抛
            if "Fill" not in str(e):
                raise
            last_error = e
            if attempt >= max_retries:
                break
            file_path, tmp_dir = _repair_with_excel(file_path)
            try:
                return load_workbook(file_path, read_only=True, data_only=True), tmp_dir
            except Exception as e2:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                last_error = e2
                break

    raise XlsxStyleBrokenError(
        f"Excel 文件无法读取（样式表损坏且修复无效）：{last_error}",
        code=ERR_PARSE_XLSX_BROKEN,
    )


class XlsxParser:
    @staticmethod
    def parse(file_path: str) -> List[Document]:
        docs = []
        # 只读模式加载，大文件更省内存
        wb, fix_dir = safe_load_workbook(file_path)
        try:
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows_text = []

                for row in ws.iter_rows(values_only=True):
                    # 过滤全空行
                    cells = [str(cell).strip() if cell is not None else "" for cell in row]
                    if not any(cells):
                        continue
                    rows_text.append(" | ".join(cells))

                if not rows_text:
                    continue

                sheet_text = f"工作表：{sheet_name}\n" + "\n".join(rows_text)
                docs.append(Document(
                    page_content=sheet_text,
                    metadata={
                        "source": file_path,
                        "file_type": "xlsx",
                        "position": f"工作表_{sheet_name}",
                        "row_count": len(rows_text)
                    }
                ))
        finally:
            wb.close()
            if fix_dir:
                shutil.rmtree(fix_dir, ignore_errors=True)
        return docs


if __name__ == "__main__":
    test_file = "Arashi自动化文档.xlsx"
    xlsxs = XlsxParser.parse(file_path=test_file)
    print(f"document数量：{len(xlsxs)}")
    for i in xlsxs:
        print(f"读取到的page_content: {i.page_content[:600]}", end="\n")
