"""
Day5：最小 RAG 闭环（无框架纯手写实现）
完整链路：本地Markdown文档 → 分片 → 向量化 → JSON存储 → 检索 → LLM生成带引用回答
"""
import os
import json
import numpy as np
import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ==================== 配置 ====================
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL")
EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
EMBEDDING_MODEL = "qwen3.7-text-embedding"
CHAT_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
CHAT_MODEL = "doubao-lite-240615"

# 检索参数
TOP_K = 3
SIMILARITY_THRESHOLD = 0.6
# 分片参数
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# RAG 系统提示词
RAG_SYSTEM_PROMPT = """你是一个基于检索内容的问答助手。
请严格根据下面提供的【检索上下文】来回答用户的问题。
- 如果检索上下文里有答案，就基于原文内容准确回答，不要编造
- 如果检索上下文里没有答案，就明确回答"根据现有资料无法回答该问题"
- 每一条信息点后面，用上角标标注对应的引用编号，格式：[引用x]
- 相同来源不要重复标注，不同来源分别标注

【检索上下文】
{context}
"""


# ==================== 1. 工具函数 ====================
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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """余弦相似度计算"""
    a = np.array(vec1)
    b = np.array(vec2)
    dot = np.dot(a, b)
    return float(dot / (np.linalg.norm(a) * np.linalg.norm(b)))


def split_recursive_char(text: str, chunk_size: int, overlap: int) -> list[str]:
    """递归字符分片，复用 Day3 实现"""
    separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]

    def _split(text, sep_idx):
        if sep_idx >= len(separators):
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end].strip())
                start = end
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
                    current = ""          # 清空，避免后续复用旧值
                # 单个部分就超过长度，往下一级分隔符切
                if len(part) > chunk_size:
                    sub_chunks = _split(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                else:
                    current = part + sep

        if current.strip():
            chunks.append(current.strip())

    # 内层切分
    chunks = _split(text, 0)
    # 加重叠
    # 因为是递归函数，所以可能会重复加重叠
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



# ==================== 2. 向量存储（JSON 持久化） ====================
class VectorStore:
    """简易向量库：本地 JSON 文件持久化，暴力检索"""

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.docs = []  # 每个元素: {"id": int, "text": str, "embedding": list[float]}

    def load(self):
        """从本地 JSON 加载向量库"""
        if os.path.exists(self.store_path):
            with open(self.store_path, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            print(f"加载向量库成功，共 {len(self.docs)} 个分片")
        else:
            print("向量库文件不存在，将创建新库")

    def save(self):
        """保存到本地 JSON"""
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)
        print(f"向量库已保存，共 {len(self.docs)} 个分片")

    def add_document(self, text: str):
        """文档分片后加入向量库"""
        chunks = split_recursive_char(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"文档切分为 {len(chunks)} 个分片，正在向量化...")

        start_id = len(self.docs)
        for i, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            self.docs.append({
                "id": start_id + i,
                "text": chunk,
                "embedding": emb.tolist() # ❌ numpy 数组，不能序列化
            })

    def search(self, query: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
        """检索最相关的 TopK 分片"""
        query_emb = get_embedding(query)
        results = []

        for doc in self.docs:
            sim = cosine_similarity(query_emb, doc["embedding"])
            if sim >= threshold:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "similarity": sim,
                    "embedding": doc["embedding"],   # 新增
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def mmr_retrieval_search(self, query: str,
                  top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD,
                  lambda_param: float = 0.7) -> list[dict]:
        """MMR 最大边际相关性检索"""
        # 先召回候选池
        candidates = self.search(query)
        # 已选集合
        selected = []
        # 未选集合
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
                        cosine_similarity(item["embedding"], s["embedding"])
                        for s in selected
                    )
                else:
                    sim_selected_max = 0

                # 3. 计算 MMR 值
                mmr_score = lambda_param * sim_query - (1 - lambda_param) * sim_selected_max

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_item = item

            # # 低于阈值就停止
            # if best_item["similarity"] < threshold:
            #     break

            selected.append(best_item)
            unselected.remove(best_item)

        return selected
# ==================== 3. RAG 问答生成 ====================
def rag_answer(query: str, store: VectorStore, use_mmr:bool =True) -> dict:
    """RAG 全流程：检索 → 组装 Prompt → LLM 生成"""
    # 第一步：检索相关片段
    if use_mmr:
        refs = store.mmr_retrieval_search(query)
    else:
        refs = store.search(query)
    if not refs:
        return {
            "answer": "根据现有资料无法回答该问题",
            "references": []
        }

    # 第二步：组装上下文
    context_parts = []
    for i, ref in enumerate(refs, 1): # i 从 1 开始迭代
        context_parts.append(f"[引用{i}]（分片{ref['id']}）\n{ref['text']}")
    context_str = "\n\n".join(context_parts)

    # 第三步：组装系统提示 + 用户问题
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context_str)

    # 第四步：调用 LLM 生成
    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DOUBAO_MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(CHAT_URL, headers=headers, json=payload)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]

    return {
        "answer": answer,
        "references": refs
    }


# ==================== 4. 主程序入口 ====================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("用法：")
        print("  入库：python minimal_rag.py add <markdown文件路径> <向量库保存路径>")
        print("  问答：python minimal_rag.py ask <问题> <向量库路径>")
        print("\n示例：")
        print('  python minimal_rag.py add ../docs/rag/向量检索核心原理.md rag_store.json')
        print('  python minimal_rag.py ask "什么是MMR？" rag_store.json')
        sys.exit(1)

    cmd = sys.argv[1]
    store = VectorStore(sys.argv[3])
    store.load()

    if cmd == "add":
        # 离线索引阶段
        file_path = sys.argv[2]
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        store.add_document(text)
        store.save()

    elif cmd == "ask":
        # 在线问答阶段
        query = sys.argv[2]
        result = rag_answer(query, store)

        print("\n" + "=" * 60)
        print(f"问题：{query}")
        print("-" * 60)
        print("回答：")
        print(result["answer"])
        print("-" * 60)
        print("引用来源：")
        for ref in result["references"]:
            preview = ref["text"][:60].replace("\n", "") + "..."
            print(f"  [分片{ref['id']}] 相似度={ref['similarity']:.4f} | {preview}")
        print("=" * 60)
