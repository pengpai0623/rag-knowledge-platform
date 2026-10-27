"""
Day8：LangChain 基础组件 - 文档加载与分片
对比手写版，体验标准组件的用法与差异
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_markdown(file_path: str):
    # ==================== 1. 文档加载器 ====================
    # 对应手写版：open 读取文件
    # 作用：加载 Markdown 文件，统一输出 Document 对象列表
    loader = UnstructuredMarkdownLoader(file_path)
    documents = loader.load()
    # print(f"加载文档类型：{type(documents)}")
    # print(f"\n加载documents：{documents}")
    print(f"加载完成，原始文档数：{len(documents)} 个")
    print(f"原始文档长度：{len(documents[0].page_content)} 字符")
    print(f"元数据：{documents[0].metadata}")

    # ==================== 2. 文本分割器 ====================
    # 对应手写版：split_recursive_char()
    # 作用：递归字符分片，优先在自然语义边界切分，如果分隔符不够切，默认固定长度切
    text_splitter = RecursiveCharacterTextSplitter(
        # 内置了碎片合并逻辑，过短的文本块会尽量和相邻块合并，只要总长度不超太多，就不会出现大量几十字符的碎分片。
        # 相比手写版，减少了碎片数量，提升了Embedding效率
        # 优先在段落、句子等自然语义边界切分，切分点大多在句号、段落结束处，语义完整性更好。
        chunk_size=200, # chunk_size 是目标参考值，不是绝对红线。如果一个完整语义块（句子 / 段落）只轻微超标（通常 10% 以内），会优先完整保留，不会为了卡长度强行切断
        chunk_overlap=40, # 仅在文本硬切断时生效，即断句才补、不断不补，也就是分片长度 < chunk_size 时不补
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            ""
        ],
        length_function=len  # 按字符数计算，中文友好，≈ chunk_size个汉字汉字；如果是默认 token 函数，chunk_size就填 token 数，适用场景：英文文档、或者严格对其Embedding模型的token上限 
    )

    # 执行分片
    splits = text_splitter.split_documents(documents)
    print(f"\n分片完成，共 {len(splits)} 个分片")

    # 打印前 3 个分片预览
    for i, doc in enumerate(splits[:3], 1):
        preview = doc.page_content[:300].replace("\n", "") + "..."
        print(f"  分片{i}: {preview}  | 长度={len(doc.page_content)}")

    return splits


if __name__ == "__main__":
    file_path = "../docs/rag/向量检索核心原理.md"
    splits = load_and_split_markdown(file_path)

    print("\n" + "=" * 60)
    print("对比说明：")
    print("1. Loader 对应手写的文件读取，统一输出 Document 标准结构")
    print("2. RecursiveCharacterTextSplitter 对应手写的递归字符分片")
    print("3. 所有分片都是 Document 对象，可直接接入后续 Embedding 与向量库")
