"""
Day9：通用文档处理工具类
封装 文档加载 + 文本分片 全流程
支持格式：.md / .txt
支持分片策略：recursive 递归字符 / semantic 语义分片
统一输出标准 LangChain Document 列表
"""
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        split_strategy: str = "recursive",
        embedding_model=None
    ):
        """
        初始化文档处理器
        :param chunk_size: 分片大小，单位字符
        :param chunk_overlap: 重叠长度，单位字符
        :param split_strategy: 分片策略 recursive / semantic
        :param embedding_model: 语义分片必须传入，需符合LangChain Embedding接口
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_strategy = split_strategy
        self.embedding_model = embedding_model
        self._init_splitter()

    def _init_splitter(self):
        if self.split_strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n", "\n",
                    "。", "！", "？", "；",
                    "，", "、", " ", ""
                ]
            )
        elif self.split_strategy == "semantic":
            if not self.embedding_model:
                raise ValueError("语义分片模式必须传入 embedding_model")
            # 语义分片属于实验性组件，需单独安装 langchain-experimental
            from langchain_experimental.text_splitter import SemanticChunker
            self.splitter = SemanticChunker(
                embeddings=self.embedding_model,
                breakpoint_threshold_type="percentile"
            )
        else:
            raise ValueError(f"不支持的分片策略：{self.split_strategy}")

    def load_file(self, file_path: str) -> list[Document]:
        """加载单个文件，返回原始 Document 列表（未分片）"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".md":
            # elements 模式：按元素拆分，保留结构元数据
            # Markdown 加载器会把每个标题、段落、列表项都拆成独立的 Document，并在 metadata里标注元素类型，适合做结构分片
            loader = UnstructuredMarkdownLoader(file_path, mode="elements")
        else:
            raise ValueError(f"暂不支持的文件格式：{ext}")

        return loader.load()

    def process_file(self, file_path: str) -> list[Document]:
        """完整处理：加载 + 分片，返回最终分片列表"""
        docs = self.load_file(file_path)
        splits = self.splitter.split_documents(docs)
        return splits


if __name__ == "__main__":
    # ========== 测试1：递归字符分片（默认） ==========
    print("=" * 60)
    print("测试1：递归字符分片策略")
    processor = DocumentProcessor(
        chunk_size=200,
        chunk_overlap=40,
        split_strategy="recursive"
    )

    file_path = "../docs/rag/向量检索核心原理.md"
    splits = processor.process_file(file_path)

    # print(f"文件处理完成，共 {len(splits)} 个分片")
    # for i, doc in enumerate(splits[:3], 1):
    #     preview = doc.page_content[:50].replace("\n", "") + "..."
    #     print(f"  分片{i}: {preview} | 长度={len(doc.page_content)}")

    # # 打印元数据示例
    # print(f"\n分片1元数据：{splits[0].metadata}")

    # print("\n" + "=" * 60)
    # print("说明：")
    # print("1. 语义分片需接入Embedding模型，Day10接入后可测试")
    # print("2. 所有输出均为标准Document对象，可直接接入向量库")
