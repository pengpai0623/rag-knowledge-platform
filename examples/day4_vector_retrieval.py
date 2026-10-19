"""
Day4：向量检索原理与实现
实现暴力TopK、相似度阈值过滤、MMR最大边际相关性
串联 分片→向量化→检索 全链路
"""
import os
import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
EMBEDDING_MODEL = "qwen3.7-text-embedding"


# ==================== 基础工具 ====================
def get_embedding(text: str) -> np.ndarray:
    """调用纯文本Embedding接口"""
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
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def split_recursive_char(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """复用Day3递归字符分片"""
    separators = ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", " ", ""]

    def _split(text, sep_idx):
        if sep_idx >= len(separators):
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end].strip())
                start = end - overlap
            return [c for c in chunks if c]

        sep = separators[sep_idx]
        if sep not in text:
            return _split(text, sep_idx + 1)

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += part + sep
            else:
                if current:
                    chunks.append(current.strip())
                # 单个部分就超过长度，往下一级分隔符切
                if len(part) > chunk_size:
                    sub_chunks = _split(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                else:
                    current = part + sep

        if current.strip():
            chunks.append(current.strip())

        # 加重叠
        if overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # 取前一个分片的末尾 overlap 字符拼到开头
                    prev_end = chunks[i-1][-overlap:]
                    chunk = prev_end + chunk
                overlapped.append(chunk)
            return overlapped

        return chunks

    return _split(text, 0)


# ==================== 构建模拟向量库 ====================
TEST_TEXT = """
检索增强生成（RAG）是一种结合了信息检索和文本生成的人工智能技术。
它的核心思想是，在回答用户问题时，先从外部知识库中检索出相关的文档片段，
再将这些片段作为上下文提供给大语言模型，辅助模型生成更准确的回答。

RAG 主要解决大模型的三个痛点：第一是幻觉问题，模型可以基于真实的文档内容回答，
减少凭空编造的情况；第二是知识时效性问题，知识库可以实时更新，
不需要重新训练模型就能获取最新信息；第三是数据安全问题，
企业内部文档不需要上传用于模型训练，只在检索时调用即可。

文本分片是 RAG 系统中非常重要的一个环节。
它的作用是将长文档切割成合适大小的文本块，再进行向量化存储。
分片的大小和策略直接影响检索的准确率和回答的质量。
分片太大容易引入无关信息，太小又会丢失上下文语义。

Embedding 向量模型负责将文本转换成高维空间中的向量表示。
语义相近的文本在向量空间中的距离更近，因此可以通过计算相似度来找到相关的文档片段。
常用的相似度计算方法是余弦相似度，取值范围在-1到1之间，数值越大表示越相似。
"""


def build_vector_store() -> list[dict]:
    """构建模拟向量库：分片 + 向量化"""
    chunks = split_recursive_char(TEST_TEXT, chunk_size=200, overlap=40)
    store = []
    for idx, text in enumerate(chunks):
        vec = get_embedding(text)
        store.append({"id": idx, "text": text, "vector": vec})
    return store


# ==================== 三种检索实现 ====================

def brute_force_topk(query_vec: np.ndarray, store: list[dict], top_k: int = 3) -> list[dict]:
    """暴力检索：TopK 相似度排序"""
    results = []
    for item in store:
        sim = cosine_similarity(query_vec, item["vector"])
        results.append({**item, "similarity": sim})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def threshold_filter(query_vec: np.ndarray, store: list[dict], top_k: int = 3, threshold: float = 0.6) -> list[dict]:
    """TopK + 相似度阈值过滤"""
    topk_results = brute_force_topk(query_vec, store, top_k=top_k)
    return [item for item in topk_results if item["similarity"] >= threshold]


def mmr_retrieval(query_vec: np.ndarray, store: list[dict],
                  top_k: int = 3, threshold: float = 0.6,
                  lambda_param: float = 0.7) -> list[dict]:
    """MMR 最大边际相关性检索"""
    # 先召回候选池
    candidates = brute_force_topk(query_vec, store, top_k=len(store))
    selected = []
    unselected = candidates.copy()

    while len(selected) < top_k and unselected:
        best_mmr = -float("inf")
        best_item = None

        for item in unselected:
            # 1. 和查询的相关性
            sim_query = item["similarity"]

            # 2. 和已选结果的最大相似度
            if selected:
                sim_selected_max = max(
                    cosine_similarity(item["vector"], s["vector"])
                    for s in selected
                )
            else:
                sim_selected_max = 0

            # 3. 计算 MMR 值
            mmr_score = lambda_param * sim_query - (1 - lambda_param) * sim_selected_max

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_item = item

        # 低于阈值就停止
        if best_item["similarity"] < threshold:
            break

        selected.append(best_item)
        unselected.remove(best_item)

    return selected


# ==================== 对比测试 ====================
QUERY = "文本分片太大和太小分别有什么问题？"


def print_results(title: str, results: list[dict]):
    print(f"\n{'='*60}")
    print(f"【{title}】")
    for i, item in enumerate(results, 1):
        preview = item["text"][:50].replace("\n", "") + "..."
        print(f"  第{i}名  相似度={item['similarity']:.4f}  |  {preview}")


if __name__ == "__main__":
    print("=" * 60)
    print("向量检索对比测试")
    print(f"查询：{QUERY}")

    # 构建向量库
    print("\n正在构建向量库（分片+向量化）...")
    store = build_vector_store()
    print(f"向量库构建完成，共 {len(store)} 个分片")

    query_vec = get_embedding(QUERY)

    # 三种方式对比
    print_results("1. 暴力 TopK 检索 (K=4)", brute_force_topk(query_vec, store, top_k=4))
    print_results("2. TopK + 阈值过滤 (K=4, 阈值=0.6)", threshold_filter(query_vec, store, top_k=4, threshold=0.6))
    print_results("3. MMR 多样性检索 (λ=0.7)", mmr_retrieval(query_vec, store, top_k=4, threshold=0.4, lambda_param=0.7))


    print("\n" + "=" * 60)
    print("观察要点：")
    print("1. 三种方式返回的排名和内容有什么差异？")
    print("2. 阈值过滤有没有过滤掉不相关的结果？")
    print("3. MMR 有没有降低结果的重复度？")
