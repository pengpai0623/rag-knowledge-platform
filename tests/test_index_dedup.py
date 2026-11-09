"""入库覆盖（dedup）契约测试：旧分片按「同 source 或同 file_sha256」双身份匹配。

不加载 Embedding 模型、不联网：直接用 chromadb 客户端写入假向量，只验证 where 查询语义。
运行：
    python tests/test_index_dedup.py
    pytest tests/test_index_dedup.py
"""

import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.index_service import IndexService

KB = "tmp-dedup-test"
SHA_A = "aaaaaaaaaaaaaaaa"
SHA_B = "bbbbbbbbbbbbbbbb"


@contextlib.contextmanager
def temp_collection(rows):
    """rows: [(ids, source, sha256|None)]，测试结束一定删掉临时 collection。"""
    client = IndexService._persistent_client()
    if IndexService.has_collection(KB):
        client.delete_collection(KB)
    client.create_collection(KB)
    collection = client.get_collection(KB)
    for ids, source, sha in rows:
        metadata = {"source": source, "parser_version": "v2"}
        if sha:
            metadata["file_sha256"] = sha
        collection.add(
            ids=ids,
            documents=[f"{source}#{i}" for i in ids],
            embeddings=[[float(i), 0.0, 0.0] for i in range(len(ids))],
            metadatas=[dict(metadata) for _ in ids],
        )
    try:
        yield
    finally:
        client.delete_collection(KB)


def _seed_three_docs():
    """库里三批分片：旧名同内容、新名同内容、另一个来源的另一份内容。"""
    return [
        (["old-1", "old-2"], "报告.pdf", SHA_A),
        (["new-1", "new-2"], "报告-最终版.pdf", SHA_A),
        (["other-1"], "其他.pdf", SHA_B),
    ]


def test_renamed_reupload_matches_both_identities():
    """改名重传：新名 + 同 sha → 连同旧名分片一起命中，避免重复向量"""
    with temp_collection(_seed_three_docs()):
        ids = IndexService.source_chunk_ids(KB, "报告-最终版.pdf", SHA_A)
    assert sorted(ids) == ["new-1", "new-2", "old-1", "old-2"]


def test_source_only_match_keeps_old_behaviour():
    """不传 sha（老数据没有该字段时）退化为只按 source 匹配"""
    with temp_collection(_seed_three_docs()):
        ids = IndexService.source_chunk_ids(KB, "报告-最终版.pdf")
    assert sorted(ids) == ["new-1", "new-2"]


def test_unrelated_source_not_matched():
    """别的来源、别的内容不能被误删"""
    with temp_collection(_seed_three_docs()):
        ids = IndexService.source_chunk_ids(KB, "全新.pdf", "cccccccccccccccc")
    assert ids == []


def test_legacy_chunk_without_sha_not_matched():
    """v2 之前的老分片没有 file_sha256：$eq 不匹配缺失字段，只能整库重灌"""
    rows = [(["legacy-1"], "老文档.pdf", None), (["new-1"], "新名.pdf", SHA_A)]
    with temp_collection(rows):
        ids = IndexService.source_chunk_ids(KB, "新名.pdf", SHA_A)
    assert sorted(ids) == ["new-1"]


def test_delete_by_source_counts_matched_ids():
    """删除数 = 实际命中的分片数，不再用 count() 前后差（那是全库记录数，并发会串）"""
    rows = [(["a1", "a2", "a3"], "A.pdf", SHA_A), (["b1"], "B.pdf", SHA_B)]
    with temp_collection(rows):
        deleted = IndexService.delete_by_source(KB, "A.pdf")
        again = IndexService.delete_by_source(KB, "A.pdf")
        remaining = IndexService._persistent_client().get_collection(KB).count()
    assert deleted == 3, deleted
    assert again == 0, "重复删同一来源应如实返回 0，不谎报"
    assert remaining == 1, "别的来源不能被误删"


def test_delete_ids_returns_id_count():
    """按 id 删的返回值就是这批 id 的数量"""
    with temp_collection([(["a1", "a2"], "A.pdf", SHA_A)]):
        assert IndexService.delete_ids(KB, ["a1", "a2"]) == 2
        assert IndexService.delete_ids(KB, []) == 0


def test_missing_collection_returns_empty():
    """库不存在时返回空列表，不抛异常（上传时首次建库走这条路）"""
    client = IndexService._persistent_client()
    if IndexService.has_collection(KB):
        client.delete_collection(KB)
    assert IndexService.source_chunk_ids(KB, "任意.pdf", SHA_A) == []


if __name__ == "__main__":
    passed = failed = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        try:
            func()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
