"""
Day3：文本分片策略对比
实现3种分片方式，结合Embedding验证检索效果差异
"""
import os
import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
EMBEDDING_MODEL = "qwen3.7-text-embedding"


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
    dot_product = np.dot(vec1, vec2)
    return dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ==================== 分片策略实现 ====================

def split_by_fixed_length(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """
    策略1：固定长度分片
    按字符数硬切，首尾重叠
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # 后退重叠长度

    return [c for c in chunks if c.strip()]



def split_by_paragraph(text: str) -> list[str]:
    """
    策略2：按段落分片
    按空行切分，保留自然段落边界
    """
    # 按多个换行分割，过滤空段落
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def split_recursive_char(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    策略3：递归字符分片
    按优先级：换行 → 句子 → 字符，优先在自然边界切
    简化版实现，模拟 LangChain RecursiveCharacterTextSplitter 核心逻辑
    """
    # 分隔符优先级：从大到小
    separators = ["\n\n", "\n", "。", "！", "？", "；", "：", "，", "、", " ", ""]


    def _split(text: str, sep_idx: int) -> list[str]:
        if sep_idx >= len(separators):
            # 最后一级：按字符硬切
            return split_by_fixed_length(text, chunk_size, overlap)

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


# ==================== 测试与对比 ====================

# 测试文本：RAG相关介绍段落
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

QUERY = "文本分片太大和太小分别有什么问题？"


def rank_chunks(query: str, chunks: list[str], strategy_name: str):
    """计算query和所有分片的相似度，排序输出"""
    print(f"\n{'='*60}")
    print(f"【{strategy_name}】 共 {len(chunks)} 个分片")
    print(f"查询：{QUERY}\n")

    query_vec = get_embedding(query)
    results = []

    for i, chunk in enumerate(chunks):
        chunk_vec = get_embedding(chunk)
        sim = cosine_similarity(query_vec, chunk_vec)
        results.append((i, sim, chunk[:60] + "..." if len(chunk) > 60 else chunk))

    # 按相似度降序
    results.sort(key=lambda x: x[1], reverse=True)

    print("相似度排名：")
    for rank, (idx, sim, preview) in enumerate(results, 1):
        print(f"  第{rank}名 分片[{idx}] 相似度={sim:.4f} | {preview}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("文本分片策略对比测试")
    print("测试目标：同一段文本，不同分片策略，检索排名差异")

    # 三种策略分别切分
    chunks_fixed = split_by_fixed_length(TEST_TEXT, chunk_size=100, overlap=20)
    chunks_para = split_by_paragraph(TEST_TEXT)
    chunks_recursive = split_recursive_char(TEST_TEXT, chunk_size=200, overlap=40)

    # 分别排名对比
    rank_chunks(QUERY, chunks_fixed, "固定长度分片 (100字符)")
    rank_chunks(QUERY, chunks_para, "按段落分片")
    rank_chunks(QUERY, chunks_recursive, "递归字符分片 (200字符)")

    print("\n" + "=" * 60)
    print("观察要点：")
    print("1. 正确答案所在的分片排名第几？")
    print("2. 哪种策略把答案切得更完整？")
    print("3. 固定长度分片有没有出现语义断裂？")
