"""检索层验收脚本：验证「解析 → 入库 → 召回」这条链路真的通。

为什么需要它：`chunk_count > 0` 只能证明「产出了向量」，证明不了「能被检索到」
（OCR 出来的文本行序错乱、低置信行被丢弃，都会让 chunk 涨了但召回没用）。
所以验收要看召回指标，而不是分片数。

两种跑法：

1) 自召回（不需要人工标注，最省事）：
   从库里随机抽样 N 个已入库分片，用分片原文的前若干字当问题，
   看能不能把自己召回。这条能直接暴露「OCR 文本入库了但检索不到」。
       python scripts/eval_recall.py --kb demo --auto 20

2) 人工测试集：
   一个 JSON 数组，每条是 {"question": "...", "keywords": ["...", "..."]}，
   命中判定 = top-k 结果里任一 chunk 含任一关键词。
       python scripts/eval_recall.py --kb demo --test-set docs/rag/qa_set.json

注意：脚本会真实调用 Embedding 接口（需要 .env 里的 QWEN_API_KEY）。
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.index_service import IndexService
from app.services.retrieve_service import RetrieveService

TOP_K_DEFAULT = 5


def _normalize(text: str) -> str:
    return "".join((text or "").split())


def _sample_chunks(kb: str, n: int) -> list:
    """直接从 Chroma 抽样分片（不经过检索），作为自召回的 ground truth"""
    client = IndexService._persistent_client()
    collection = client.get_collection(kb)
    result = collection.get(include=["documents", "metadatas"])
    docs, metas = result.get("documents") or [], result.get("metadatas") or []
    items = [
        {"content": d, "source": (m or {}).get("source", "?"),
         "page": (m or {}).get("page_number"), "parse_mode": (m or {}).get("parse_mode")}
        for d, m in zip(docs, metas) if d and len(_normalize(d)) >= 20
    ]
    if len(items) <= n:
        return items
    return random.sample(items, n)


def _auto_cases(kb: str, n: int) -> list:
    cases = []
    for item in _sample_chunks(kb, n):
        # 取分片开头一小段当问题：足够具体，又不需要人工写题
        question = _normalize(item["content"])[:24]
        cases.append({
            "question": question,
            "expect": _normalize(item["content"])[:40],
            "label": f"{item['source']}"
                     + (f" p{item['page']}" if item["page"] else "")
                     + f" [{item['parse_mode']}]",
        })
    return cases


def _load_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for item in raw:
        cases.append({
            "question": item["question"],
            "keywords": item.get("keywords") or [],
            "label": item["question"][:24],
        })
    return cases


def _hit(case: dict, contexts: list) -> int:
    """返回首次命中的名次（1-based），未命中返回 0"""
    for rank, doc in enumerate(contexts, 1):
        text = _normalize(doc.page_content)
        if case.get("expect"):
            if case["expect"] in text:
                return rank
        elif case.get("keywords"):
            if any(_normalize(k) in text for k in case["keywords"]):
                return rank
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 召回验收")
    parser.add_argument("--kb", default="default", help="知识库名")
    parser.add_argument("--auto", type=int, default=0, help="自召回抽样条数")
    parser.add_argument("--test-set", help="人工测试集 JSON 路径")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    parser.add_argument("--mode", default="similarity", choices=["similarity", "mmr"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not IndexService.has_collection(args.kb):
        print(f"知识库不存在：{args.kb}")
        return 2

    if args.test_set:
        cases = _load_cases(args.test_set)
    elif args.auto:
        cases = _auto_cases(args.kb, args.auto)
    else:
        print("请指定 --auto N 或 --test-set path")
        return 2

    if not cases:
        print("没有可用用例（库可能为空，或分片都太短）")
        return 2

    ranks = []
    for case in cases:
        contexts = RetrieveService.get_contexts(
            case["question"], args.kb, args.top_k, mode=args.mode,
        )
        rank = _hit(case, contexts)
        ranks.append(rank)
        if args.verbose or rank == 0:
            flag = f"hit@{rank}" if rank else "MISS"
            print(f"  [{flag}] {case['label']} <- {case['question'][:30]}")

    total = len(ranks)
    def recall(k: int) -> float:
        return sum(1 for r in ranks if 1 <= r <= k) / total
    mrr = sum(1.0 / r for r in ranks if r) / total

    print("\n=== 召回验收结果 ===")
    print(f"知识库: {args.kb}  策略: {args.mode}  用例: {total}  top_k: {args.top_k}")
    print(f"Recall@1: {recall(1):.3f}")
    print(f"Recall@3: {recall(3):.3f}")
    print(f"Recall@{args.top_k}: {recall(args.top_k):.3f}")
    print(f"MRR:      {mrr:.3f}")
    print(f"未命中:   {sum(1 for r in ranks if not r)} 条")
    return 0 if recall(args.top_k) >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
