"""
Day10：LangChain + Chroma 向量库完整实战
功能：文档批量分片入库 + 持久化存储 + 多模式相似度检索
"""
import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ========== 配置项 ==========
# EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
# EMBEDDING_MODEL = "qwen3.7-text-embedding"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # 改为兼容接口端点
EMBEDDING_MODEL = "text-embedding-v3" 
DOC_PATH = "../docs/rag/向量检索核心原理.md"
PERSIST_DIR = "../vector_db/chroma_rag_db"  # 向量库持久化本地目录
COLLECTION_NAME = "rag_knowledge_base"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# 初始化 Embedding 模型（替换为你实际使用的模型，保持与手写版一致便于对比）
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=EMBEDDING_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)


def init_vector_store(rebuild: bool = False) -> Chroma:
    """
    初始化向量库
    :param rebuild: True 则清空旧库重建，False 则加载本地已有数据
    """
    if rebuild and os.path.exists(PERSIST_DIR):
        # 递归删除整个目录树（目录及其所有内容）
        shutil.rmtree(PERSIST_DIR)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        # 使用余弦相似度空间，与手写版计算方式对齐
        # 对应之前的是否做归一化中 数据库的索引方式，另外两个是 "l2"（欧氏距离）和 "ip"（内积）
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 20, # 搜索时的近邻数量，数值越大召回率越高、速度越慢
        }
    )
    return vector_store


def load_and_split_docs() -> list:
    """加载文档并执行递归字符分片"""
    loader = TextLoader(DOC_PATH, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"文档分片完成，共 {len(splits)} 个分片")
    return splits


def batch_ingest(splits: list, vector_store: Chroma, batch_size: int = 10) -> list:
    """批量分片入库：自动完成向量化 + 索引构建 + 持久化"""
    all_ids = []
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        ids = vector_store.add_documents(batch)
        all_ids.extend(ids)
    print(f"批量入库完成，共写入 {len(all_ids)} 条向量")
    return all_ids

def base_search(query: str, vector_store: Chroma, top_k: int = 3):
    """基础相似度检索，仅返回文档"""
    results = vector_store.similarity_search(query, k=top_k) # # return: List[Document]，按照余弦距离升序排列
    print(f"\n=== 基础检索：{query} ===")
    for idx, doc in enumerate(results, 1):
        print(f"[{idx}] {doc.page_content[:60]}...")
        print(f"     元数据：{doc.metadata}")
    return results


def search_with_score(query: str, vector_store: Chroma, top_k: int = 3):
    """带相似度距离的检索，余弦距离越小越相似"""
    results = vector_store.similarity_search_with_score(query, k=top_k) 
    # return: List[Tuple[Document, float]]，float是余弦距离，值等于1 - cosine_similarity
    print(f"\n=== 带分检索：{query} ===")
    for idx, (doc, distance) in enumerate(results, 1):
        similarity = 1 - distance  # 余弦距离转相似度，0~1，越大越相似
        print(f"[{idx}] 相似度：{similarity:.4f} | 距离：{distance:.4f}")
        print(f"     {doc.page_content[:60]}...")
    return results


def filter_search(query: str, vector_store: Chroma, filter_condition: dict, top_k: int = 3):
    """带元数据过滤的检索，缩小检索范围"""
    results = vector_store.similarity_search(
        query,
        k=top_k,
        filter=filter_condition
    )
    print(f"\n=== 过滤检索：条件={filter_condition} ===")
    for idx, doc in enumerate(results, 1):
        print(f"[{idx}] {doc.page_content[:60]}...")
    return results


def delete_docs(vector_store: Chroma, ids: list):
    """按 ID 删除指定向量"""
    vector_store.delete(ids=ids)
    print(f"\n已删除 {len(ids)} 条向量")


if __name__ == "__main__":
    # 1. 初始化向量库（首次运行 rebuild=True 重建，后续可设为 False 直接加载）
    vector_store = init_vector_store(rebuild=True)

    # 2. 文档分片 + 批量入库
    splits = load_and_split_docs()
    batch_ingest(splits, vector_store)

    # # 3. 基础相似度检索
    # base_search("什么是近似最近邻 ANN", vector_store)

    # # 4. 带相似度得分的检索
    # search_with_score("MMR 和 MRR 有什么区别", vector_store)

    # # 5. 元数据过滤检索（按来源文件过滤）
    # filter_search(
    #     "向量检索",
    #     vector_store,
    #     filter_condition={"source": "../docs/rag/向量检索核心原理.md"} # 按文件、章节、作者、更新时间等过滤，演示用，生产标配
    # )

    # print("\n" + "=" * 60)
    # print("核心说明：")
    # print("1. 向量库已持久化到 vector_db 目录，重启程序数据不丢失")
    # print("2. 余弦距离取值 0~2，0 为完全相同；相似度 = 1 - 距离")
    # print("3. 元数据过滤是生产级 RAG 的标配，可按文件、章节、标签精细化检索")
