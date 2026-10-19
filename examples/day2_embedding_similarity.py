"""
Day2：Embedding 文本转向量 + 余弦相似度计算
Qwen-Embedding(qwen3.7-text-embedding-flash)，增加query/passage前缀优化
"""
import os
import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
EMBEDDING_MODEL = "qwen3.7-text-embedding"


def get_embedding(text: str, is_query: bool = False) -> np.ndarray:
    # ✅ 增加前缀：检索场景必须区分query/passage
    if is_query:
        text = f"query: {text}"
    else:
        text = f"passage: {text}"

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": {"texts": [text]}
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(EMBEDDING_URL, headers=headers, json=payload)
        print(f"HTTP状态码：{resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["output"]["embeddings"][0]["embedding"])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # 向量点积 除以向量模长的乘积，得到余弦相似度
    return dot_product / (norm1 * norm2)


def test_pair(query_text: str, passage_text: str, label: str):
    # 第一个是用户query，第二个是知识库passage
    vec_query = get_embedding(query_text, is_query=True)
    vec_passage = get_embedding(passage_text, is_query=False)
    sim = cosine_similarity(vec_query, vec_passage)
    print(f"【{label}】")
    print(f"  Query: {query_text}")
    print(f"  Passage: {passage_text}")
    print(f"  余弦相似度: {sim:.4f}\n")
    return sim


if __name__ == "__main__":
    print("=" * 50)
    print("Qwen-Embedding 增加query/passage前缀优化测试\n")
    print(f"加载 API Key：{QWEN_API_KEY[:10]}...\n")

    test_pair(
        "什么是检索增强生成？",
        "RAG的全称和作用是什么？",
        "同义不同表述"
    )
    test_pair(
        "人工智能的发展历史",
        "今天晚上吃什么饭",
        "完全无关",
    )
    test_pair(
        "FastAPI 怎么写全局异常处理器？",
        "Python Web 框架的异常处理",
        "部分相关"
    )
    test_pair(
        "用户请求太频繁了",
        "接口触发限流策略",
        "语义接近字面不同"
    )

    test_pair(
        "所有人都说你好，只有我说你不好",
        "所有人都说你不好，只有我说你好",
        "关键词完全一样但语义相反"
    )

    test_pair(
        "所有人都说你好，只有我说你不好",
        "所有人都说你好，只有我说你不好",
        "同一句话完全重复"
    )

    test_pair(
        "我爱你",
        "我不爱你",
        "反义词"
    )
    print("=" * 50)
    print("说明：")
    print("✅ query前缀用于用户提问，passage前缀用于知识库文档")
    print("✅ 优化后：相似句子分数拉高，无关句子分数明显降低，区分度更好")
