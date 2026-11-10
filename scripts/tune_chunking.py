"""分片参数调优：扫 chunk_size × chunk_overlap，用可量化指标挑配置。

为什么单独写，不复用 eval_recall.py
----------------------------------
`eval_recall.py` 验的是「库里的东西能不能被召回」，必须连真实 Chroma + 远端
embedding（text-embedding-v3）。调参要扫几十组配置，每组都重灌一次向量库，
一次十几分钟、几十块 API 费用，迭代不起来。

这里把「分片质量」和「向量模型」解耦：用**字级 BM25** 做检索。分片边界好不好、
答案句会不会被切断，这些都是文本层面的性质，跟用哪个 embedding 无关。于是
一次完整扫描是秒级、零成本，可以放心反复试。

评估方法
--------
从清洗后的正文里抽「针」（一段 20~80 字的完整句子），把它当问题去检索自己：

- **断句率**  针在任何单片里都找不到 → 被切断了。这是最硬的缺陷指标，
             与检索器无关，纯粹是切分的锅。
- **hit@1/@3** 能召回的比例。注意：针取自原文时这个指标普遍接近 100%，
             **没有区分度**，只能用来发现明显故障，不能用来选参数。
- **上下文 / ≥90%** 命中片在答案句前后各提供了多少字（上限 250/侧）。size 太小时
             答案贴着片边、上下文被切掉，模型即使命中答案也缺依据。这是选 size 的
             主依据；「前后各 250 字」是显式假设，见 CTX_CAP。
- **冗余倍数** 命中片的长度 / 针的长度。越大说明夹带的无关内容越多，
             既费 token 又干扰生成。
- **碎片率**  <100 字的片占比。碎片检索价值低、且数量多，会稀释 top-k。
- **边界完整率** 片尾落在句末标点的比例。
- **实际重叠** 相邻片的真实公共长度。用来验证 chunk_overlap 到底有没有生效。

口径注意
--------
同一份文件的「针」集合在扫描过程中固定不变（同一 seed 抽样），否则不同配置
比的就是不同的题，没有可比性。
"""

import argparse
import math
import os
import pickle
import random
import re
import sys
import tempfile
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from app.services.parse_service import ParseService
from utils import UniversalDocParser
from utils.structural_splitter import SPLITTER_VERSION, StructuralSplitter
from utils.text_cleaner import CLEANER_VERSION
from utils.universal_parser import PARSER_VERSION

# 默认语料：覆盖 pdf / md / pptx / xlsx 四条不同策略路径，且都在 1MB 以内，
# 单文件解析 1~3 秒。要加自己的文件用 --files 覆盖。
DEFAULT_FILES = [
    "docs/rag/云海5栋4楼+三苯+SZHH0216638705(2).pdf",
    "docs/rag/向量检索核心原理.md",
    "docs/rag/doc_parser_fallback_design_v1.1.md",
    "docs/rag/LCEL与RetrievalQA对比.pptx",
    "docs/rag/Arashi自动化文档.xlsx",
]

# 针的取材范围：太短（<20）区分度不足，几乎任何片都命中；太长（>80）容易
# 天然跨片，把「切分问题」和「题目本身超长」混在一起。
NEEDLE_MIN, NEEDLE_MAX = 20, 80

# 「上下文覆盖」的假定：答案句前后各需要约 250 字，模型才够判断和引用。
# 这是一个**显式假设**，不是测量结果——换场景（比如法条、代码）应当调整。
CTX_CAP = 250

SENT_END = "。！？；.!?"
TAIL_OK = SENT_END + "）)」』”’\"'"

_SENT_RE = re.compile(r"[^。！？；.!?\n]+[。！？；.!?]?")
_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    """去空白 + 小写：中文检索里空白是噪声（换行会产生假分隔）。"""
    return _WS_RE.sub("", text or "").lower()


def _cache_path() -> str:
    return os.path.join(tempfile.gettempdir(), "_tune_chunking_cache.pkl")


def _cache_key(path: str, policy: str) -> str:
    """版本号进 key：解析/清洗改了，缓存必须失效，否则会拿旧文本调参。"""
    st = os.stat(path)
    return (
        f"{os.path.abspath(path)}|{st.st_mtime_ns}|"
        f"{PARSER_VERSION}|{CLEANER_VERSION}|{policy}"
    )


def load_docs(
    files: List[str], policy: str = "full", refresh: bool = False
) -> Dict[str, List[Document]]:
    """解析 + 清洗，带磁盘缓存。

    扫参时同一份文件要分片几十次，但解析只要一次。缓存让第二次以后的迭代
    从分钟级降到秒级——这是能「不断测试」的前提。
    """
    cache_file = _cache_path()
    cache: Dict[str, List[Document]] = {}
    if not refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    out: Dict[str, List[Document]] = {}
    dirty = False
    cleaner = ParseService.get_cleaner(policy)
    for path in files:
        if not os.path.exists(path):
            print(f"跳过（不存在）：{path}")
            continue
        key = _cache_key(path, policy)
        if key in cache:
            out[path] = cache[key]
            continue
        raw = UniversalDocParser.parse(path, source_name=os.path.basename(path))
        docs = cleaner.clean_documents(raw)
        out[path] = docs
        cache[key] = docs
        dirty = True
        print(f"解析 {os.path.basename(path)}: {len(raw)} → {len(docs)} 段")

    if dirty:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
    return out


def sample_needles(docs: List[Document], n: int, seed: int) -> List[str]:
    """从正文抽针：必须是完整句子，且足够长以保证检索有区分度。"""
    pool: List[str] = []
    for doc in docs:
        for sent in _SENT_RE.findall(doc.page_content or ""):
            sent = sent.strip()
            if NEEDLE_MIN <= len(sent) <= NEEDLE_MAX:
                pool.append(sent)
    if not pool:
        return []
    rnd = random.Random(seed)
    return rnd.sample(pool, min(n, len(pool)))


class BM25:
    """字级 bigram BM25。

    中文不分词也能用：bigram 的粒度对「这句话在哪个片里」足够，而且完全
    确定、零依赖。这里只做同一语料内的相对比较，绝对值不重要。
    """

    def __init__(self, docs: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [self._tok(d) for d in docs]
        self.lens = [len(t) for t in self.corpus]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Dict[str, int] = {}
        for toks in self.corpus:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(self.corpus)
        self.idf = {
            t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }
        self.tf: List[Dict[str, int]] = []
        for toks in self.corpus:
            d: Dict[str, int] = {}
            for t in toks:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)

    @staticmethod
    def _tok(text: str) -> List[str]:
        t = _norm(text)
        if len(t) < 2:
            return [t] if t else []
        return [t[i : i + 2] for i in range(len(t) - 1)]

    def rank(self, query: str, top_k: int = 3) -> List[int]:
        q = self._tok(query)
        scores: List[Tuple[float, int]] = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.lens[i] or 1
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:top_k]]


def evaluate(docs: List[Document], needles: List[str], size: int, overlap: int) -> Dict:
    chunks, _ = StructuralSplitter.split(docs, size, overlap)
    texts = [c.page_content for c in chunks]
    norms = [_norm(t) for t in texts]
    lens = [len(t) for t in texts]

    # 断句：针在任何一个单片里都找不到 → 被切断或跨片
    broken = 0
    hits1 = hits3 = 0
    rr = 0.0
    redundancy: List[float] = []
    cover: List[float] = []
    bm25 = BM25(texts) if texts else None
    for needle in needles:
        key = _norm(needle)
        if not any(key in n for n in norms):
            broken += 1
        ranked = bm25.rank(needle, 3) if bm25 else []
        pos = next((r + 1 for r, i in enumerate(ranked) if key in norms[i]), 0)
        if pos == 1:
            hits1 += 1
            redundancy.append(lens[ranked[0]] / max(len(needle), 1))
        if 1 <= pos <= 3:
            hits3 += 1
            rr += 1.0 / pos
            # 上下文覆盖：命中片在答案句前后各提供了多少字（各取 CAP，共 2×CAP）
            if pos == 1:
                t = norms[ranked[0]]
                idx = t.find(key)
                if idx >= 0:
                    before = min(idx, CTX_CAP)
                    after = min(len(t) - idx - len(key), CTX_CAP)
                    cover.append((before + after) / (2 * CTX_CAP))

    # 实际重叠：相邻两片的公共尾/首长度。用来验证 chunk_overlap 是否真的生效
    def _ov(a: str, b: str) -> int:
        for k in range(min(len(a), len(b)), 0, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    ovs = [_ov(texts[i - 1], texts[i]) for i in range(1, len(texts))]
    tail_ok = sum(
        1 for t in texts
        if t and (t[-1] in TAIL_OK or len(t) >= size * 0.9)
    )
    n = len(texts) or 1
    return {
        "chunks": len(texts),
        "mean": sum(lens) / n,
        "min": min(lens) if lens else 0,
        "frag": sum(1 for x in lens if x < 100) / n,
        "tail": tail_ok / n,
        "broken": broken / max(len(needles), 1),
        "hit1": hits1 / max(len(needles), 1),
        "hit3": hits3 / max(len(needles), 1),
        "mrr": rr / max(len(needles), 1),
        "redun": sum(redundancy) / len(redundancy) if redundancy else 0.0,
        "cover": sum(cover) / len(cover) if cover else 0.0,
        "full": sum(1 for c in cover if c >= 0.9) / len(cover) if cover else 0.0,
        "ov": sum(ovs) / len(ovs) if ovs else 0.0,
        "ov0": sum(1 for x in ovs if x == 0) / len(ovs) if ovs else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="分片参数调优（离线，不依赖向量库）")
    ap.add_argument("--files", nargs="*", default=None, help="语料路径，默认用内置小语料")
    ap.add_argument("--policy", default="full", choices=["full", "safe", "raw"],
                    help="清洗策略，默认与上传链路一致（full）")
    ap.add_argument("--sizes", default="200,300,400,500,600,800")
    ap.add_argument("--overlaps", default="0,50,100")
    ap.add_argument("--queries", type=int, default=40, help="每份文件抽多少条针")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--refresh", action="store_true", help="忽略解析缓存，强制重算")
    args = ap.parse_args()

    files = args.files or [os.path.join(os.getcwd(), f) for f in DEFAULT_FILES]
    docs_by_file = load_docs(files, policy=args.policy, refresh=args.refresh)
    if not docs_by_file:
        print("没有可用语料")
        return 1

    print(
        f"\n版本口径 parser={PARSER_VERSION} cleaner={CLEANER_VERSION} "
        f"splitter={SPLITTER_VERSION}  policy={args.policy}"
    )
    needles_by_file = {
        path: sample_needles(docs, args.queries, args.seed)
        for path, docs in docs_by_file.items()
    }
    for path, needles in needles_by_file.items():
        print(f"针 {os.path.basename(path)}: {len(needles)} 条")

    # 段落结构诊断：清洗策略如果把空行删了，切分器就只能退到句级，
    # overlap 也会因为「片段比 overlap 大」而失效。这个数先看一眼。
    blank = sum((d.page_content or "").count("\n\n") for d in sum(docs_by_file.values(), []))
    if blank == 0:
        print(
            "\n[诊断] 清洗后全文没有一个空行（段落边界已被压平）。切分器会退到「。」"
            "句级切分，片段固定在 40 字上下，届时 chunk_overlap 小于 40 基本不生效。"
        )
    else:
        print(f"\n[诊断] 清洗后保留段落边界：空行共 {blank} 处")

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    overlaps = [int(x) for x in args.overlaps.split(",") if x.strip()]

    rows: List[Tuple[int, int, Dict]] = []
    for size in sizes:
        for overlap in overlaps:
            agg: List[Dict] = []
            for path, docs in docs_by_file.items():
                if not needles_by_file[path]:
                    continue
                agg.append(evaluate(docs, needles_by_file[path], size, overlap))
            if not agg:
                continue
            keys = agg[0].keys()
            rows.append((size, overlap, {k: sum(r[k] for r in agg) / len(agg) for k in keys}))
            print(f"  完成 size={size} overlap={overlap}")

    print("\n" + "=" * 128)
    print(
        f"{'size':>5} {'ov':>4} | {'片数':>6} {'均值':>6} {'最短':>5} "
        f"{'碎片率':>7} {'断句率':>7} {'hit@1':>7} {'hit@3':>7} {'MRR':>6} "
        f"{'冗余×':>6} {'上下文':>7} {'≥90%':>7} {'实际ov':>7} {'零ov':>6} {'边界':>6}"
    )
    print("-" * 128)
    for size, overlap, m in rows:
        print(
            f"{size:>5} {overlap:>4} | {m['chunks']:>6.0f} {m['mean']:>6.0f} {m['min']:>5.0f} "
            f"{m['frag']*100:>6.1f}% {m['broken']*100:>6.1f}% {m['hit1']*100:>6.1f}% "
            f"{m['hit3']*100:>6.1f}% {m['mrr']:>6.3f} {m['redun']:>6.1f} "
            f"{m['cover']*100:>6.1f}% {m['full']*100:>6.1f}% "
            f"{m['ov']:>7.1f} {m['ov0']*100:>5.1f}% {m['tail']*100:>5.1f}%"
        )
    print("=" * 128)

    # 排序规则说明白：断句率是硬缺陷，先排除；再看「上下文」——答案句前后各
    # 250 字装不下，模型就缺依据；最后取冗余最小的，也就是「刚好够用」的那个。
    # 不发明加权评分——权重无法验证。
    clean = [r for r in rows if r[2]["broken"] <= 0.05]
    pool = clean or rows
    ok = [r for r in pool if r[2]["full"] >= 0.9]
    if ok:
        best = sorted(ok, key=lambda r: (r[2]["redun"], -r[2]["full"]))[:3]
    else:
        # 没有配置达标时，退而取「上下文最好」，同分取冗余小——
        # 绝不能退成「最便宜」，那会把最差的配置推荐出去。
        best = sorted(pool, key=lambda r: (-r[2]["full"], -r[2]["cover"], r[2]["redun"]))[:3]
    print(f"\n推荐（断句率 ≤5%，且要求答案句前后各 {CTX_CAP} 字上下文完整）：")
    for size, overlap, m in best:
        print(
            f"  chunk_size={size:<4} overlap={overlap:<4} "
            f"上下文={m['cover']*100:.0f}% ≥90%={m['full']*100:.0f}% "
            f"hit@1={m['hit1']*100:.1f}% 冗余={m['redun']:.1f}× "
            f"碎片={m['frag']*100:.1f}% 均值={m['mean']:.0f}"
        )
    if not ok:
        print("  （没有配置能让 90% 的命中片覆盖答案前后各 250 字）")

    # overlap 是否需要调大：看设定值与实际值的落差，以及零重叠占比
    dead = [r for r in rows if r[1] > 0 and r[2]["ov"] < r[1] * 0.5]
    if dead:
        worst = dead[0]
        print(
            f"\n注意：overlap={worst[1]} 时实际只重叠 {worst[2]['ov']:.1f} 字"
            f"（零重叠 {worst[2]['ov0']*100:.0f}%）。RecursiveCharacterTextSplitter 以"
            "「separator 片段」为单位回退，片段比 chunk_overlap 大就放弃——"
            "中文一段 40+ 字，所以 overlap 小于段落长度时基本失效，要设到与段落同量级才有实际效果。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
