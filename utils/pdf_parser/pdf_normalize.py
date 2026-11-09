"""PDF 结构重存/标准化（预留给后续「重存修复」迭代，当前未接入解析链路）。

v1.1 明确不做 repair（见 docs/rag/doc_parser_fallback_design_v1.1.md §1），
所以这个模块暂时没有调用方；保留是因为它已跑通过真实坏文件，
将来做「重存修复」时直接用，不用重写。
文件名由 fromat_pdt.py 纠正而来（原拼写有误）。
"""

import fitz


def normalize_pdf(input_path: str, output_path: str):
    """
    PDF标准化预处理：重建结构、清理冗余对象、修正文本流
    移除已弃用的linear参数，兼容新版PyMuPDF
    """
    with fitz.open(input_path) as doc:
        # garbage=4：深度清理所有未引用的冗余对象、无效资源
        # deflate：压缩优化文件体积
        # clean：清理异常绘图指令、修正无效结构
        doc.save(
            output_path,
            garbage=4,
            deflate=True,
            clean=True
        )
    print(f"标准化完成，已输出至：{output_path}")

# 使用示例
if __name__ == "__main__":
    normalize_pdf("UN38.3.pdf", "UN38.3_new.pdf")
