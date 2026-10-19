"""
Day6：RAG 召回效果批量评估
自动计算 Recall@1 / Recall@3 / MRR 指标
"""
import os
import json
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from minimal_rag import VectorStore

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def evaluate_recall(store: VectorStore, test_set: list[dict], top_k: int = 3) -> dict:
    """
    批量评估检索召回效果
    返回指标：Recall@1, Recall@3, MRR, 详细结果
    """
    total = len(test_set)
    recall_1_count = 0
    recall_k_count = 0
    mrr_sum = 0
    details = []

    for item in test_set:
        question = item["question"]
        # 正确答案的分片 ID 集合
        answer_ids = set(item["answer_chunk_ids"])

        # 检索 TopK 结果
        results = store.search(question, threshold=0.6, top_k=top_k)
        # 检索后的TopK分片 ID 列表
        retrieved_ids = [r["id"] for r in results]

        # 找正确答案的排名
        rank = None
        for idx, rid in enumerate(retrieved_ids, 1):
            if rid in answer_ids:
                # 只要rid在answer_ids中，就说明本次检索命中，记录rank
                rank = idx
                break

        # 统计
        hit = rank is not None # hit = (rank is not None)
        if rank == 1:
            recall_1_count += 1
        if hit:
            # Hit Rate@K（命中率）而非 Recall@K（召回率）
            recall_k_count += 1
            mrr_sum += 1 / rank

        details.append({
            "question": question,
            "retrieved_ids": retrieved_ids,
            "answer_ids": list(answer_ids),
            "hit": hit,
            "rank": rank
        })

    metrics = {
        "total_questions": total,
        "top_k": top_k,
        "hit@1": round(recall_1_count / total, 4),
        f"hit@{top_k}": round(recall_k_count / total, 4),
        "mrr": round(mrr_sum / total, 4)
    }

    return {"metrics": metrics, "details": details}


def print_report(result: dict):
    """格式化输出评估报告"""
    metrics = result["metrics"]
    top_k_val = metrics["top_k"]
    details = result["details"]

    print("=" * 60)
    print("RAG 召回效果评估报告")
    print("=" * 60)
    print(f"测试问题数：{metrics['total_questions']}")
    print(f"评估 TopK：{metrics['top_k']}")
    print("-" * 60)
    print(f"Hit@1  首位命中率：{metrics['hit@1']:.2%}")
    print(f"Hit@{top_k_val}  命中率：{metrics[f'hit@{top_k_val}']:.2%}")
    print(f"MRR 平均倒数排名：{metrics['mrr']:.4f}")
    print("-" * 60)
    print("详细结果：")
    for i, d in enumerate(details, 1):
        status = "✅ 命中" if d["hit"] else "❌ 未命中"
        rank_str = f"第{d['rank']}名" if d["rank"] else "未命中"
        print(f"  {i:2d}. {status} [{rank_str}] | {d['question'][:30]}...")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法：python day6_evaluation.py <向量库路径> <测试集路径>")
        print("示例：python day6_evaluation.py rag_store.json ../docs/rag/test_set.json")
        sys.exit(1)

    store_path = sys.argv[1]
    test_set_path = sys.argv[2]

    # 加载向量库
    store = VectorStore(store_path)
    store.load()

    # 加载测试集
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # 执行评估
    result = evaluate_recall(store, test_set, top_k=3)
    print_report(result)
