"""
Day11：Retriever 检索器抽象与多策略检索实战
功能：三种检索策略对比 + 元数据过滤 + 统一检索接口封装 + Top-K 结果附相似度
"""
import os
import shutil
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ========== 配置项 ==========
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 改为兼容接口端点
EMBEDDING_MODEL = "text-embedding-v3"
PERSIST_DIR = "../vector_db/chroma_rag_db"
COLLECTION_NAME = "rag_knowledge_base"
TOP_K = 3

# 初始化 Embedding 模型
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=EMBEDDING_URL,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,
)


def load_vector_store() -> Chroma:
    """加载已持久化的本地向量库"""
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"向量库加载完成，当前集合向量数：{vector_store._collection.count()}")
    return vector_store


def build_retriever(
    vector_store: Chroma,
    search_type: str = "similarity", # 检索策略类型
    k: int = 3,
    score_threshold: float = 0.7, # 相似度阈值
    lambda_mult: float = 0.5, # 相关性系数，mmr 检索时使用，越大越偏向相关性，越小越偏向多样性
    filter_condition: dict = None # 检索条件
):
    """
    统一构建 Retriever 检索器
    :param search_type: similarity / mmr / similarity_score_threshold
    """
    search_kwargs = {"k": k}
    if filter_condition:
        search_kwargs["filter"] = filter_condition

    if search_type == "mmr":
        search_kwargs["fetch_k"] = 20
        search_kwargs["lambda_mult"] = lambda_mult
    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )


def attach_similarity(query: str, docs: list) -> list:
    """
    为 Top-K 检索结果附加 query 与每条文档的余弦相似度。

    流程：
        1. docs 为空直接返回 []；
        2. query 转向量，docs 批量转向量（一次请求送 K 条）；
        3. 矩阵运算一次性算出 K 个余弦相似度；
        4. 与原 docs 按顺序 zip 配对返回。

    返回:
        [(Document, similarity), ...]，similarity 越大越相似，与 docs 顺序一一对应。
    """
    if not docs:
        return []
    query_vec = np.asarray(embeddings.embed_query(query), dtype=float)
    doc_vecs = np.asarray(
        embeddings.embed_documents([d.page_content for d in docs]), dtype=float
    )
    q_norm = np.linalg.norm(query_vec) + 1e-12
    d_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-12
    sims = (doc_vecs @ query_vec) / (d_norms * q_norm)
    return [(doc, float(s)) for doc, s in zip(docs, sims)]


def print_results(title: str, results: list):
    """统一格式化输出检索结果（带相似度）"""
    print(f"\n{'='*60}")
    print(f"【{title}】返回结果数：{len(results)}")
    for idx, (doc, score) in enumerate(results, 1):
        print(f"\n[{idx}] 相似度：{score:.4f}")
        print(f"    内容：{doc.page_content[:40]}...")
        print(f"    元数据：{doc.metadata}")


if __name__ == "__main__":
    vector_store = load_vector_store()
    test_query = "向量检索和传统关键词检索有什么区别"

    # 1. 基础相似度检索
    retriever_base = build_retriever(vector_store, search_type="similarity", k=TOP_K)
    docs_base = retriever_base.invoke(test_query)
    print_results("基础相似度检索", attach_similarity(test_query, docs_base))
    # res = vector_store.similarity_search_with_relevance_scores(test_query, k=TOP_K)
    # print_results("Chroma 原生 relevance score 检索", res)

    # 2. MMR 多样性检索
    retriever_mmr = build_retriever(vector_store, search_type="mmr", k=TOP_K, lambda_mult=0.5)
    docs_mmr = retriever_mmr.invoke(test_query)
    print_results("MMR 最大边际相关性检索", attach_similarity(test_query, docs_mmr))

    # 3. 相似度阈值检索
    retriever_threshold = build_retriever(
        vector_store,
        search_type="similarity_score_threshold",
        k=TOP_K,
        score_threshold=0.7
    )
    docs_threshold = retriever_threshold.invoke(test_query) # return: List[Document]
    print_results("相似度阈值检索（阈值0.7）", attach_similarity(test_query, docs_threshold))

    # 4. 带元数据过滤的检索
    retriever_filter = build_retriever(
        vector_store,
        search_type="similarity",
        k=TOP_K,
        filter_condition={"source": "../docs/rag/向量检索核心原理.md"}
    )
    docs_filter = retriever_filter.invoke(test_query)
    print_results("带元数据过滤检索", attach_similarity(test_query, docs_filter))

    print("\n" + "="*60)
    print("核心说明：")
    print("1. Retriever 是统一检索接口，底层存储、检索策略可自由切换")
    print("2. MMR 兼顾相关性和多样性，避免结果同质化")
    print("3. 阈值检索宁缺毋滥，是降低幻觉的有效手段")
    print("4. 所有策略都支持元数据过滤，精细化控制检索范围")
    print("5. 相似度统一由 query 与文档 embedding 的余弦相似度计算（0~1，越大越相似）")