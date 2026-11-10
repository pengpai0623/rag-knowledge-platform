"""上传判定契约测试：created / skipped / replaced 三态由「库内策略 vs 本次策略」决定。

不加载 Embedding 模型、不联网：判定是纯函数，其余用例直接用 chromadb 客户端写假向量。
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
KB2 = "tmp-dedup-test-2"
SHA_A = "aaaaaaaaaaaaaaaa"
SHA_B = "bbbbbbbbbbbbbbbb"
POLICY = {"parser_version": "v3", "clean_policy": "full",
          "cleaner_version": "c1", "splitter_version": "s1"}


def _decide(profile, **override):
    """用当前策略去判定某个库内 profile，override 可单独改某一维。"""
    expected = dict(POLICY)
    expected.update(override)
    return IndexService.decide_upload_status(profile, **expected)


@contextlib.contextmanager
def temp_collections(rows_by_kb):
    """rows_by_kb: {kb: [(ids, source, sha, meta_override|None)]}，结束一定删干净。"""
    client = IndexService._persistent_client()
    names = list(rows_by_kb)
    for name in names:
        if IndexService.has_collection(name):
            client.delete_collection(name)
    try:
        for name, rows in rows_by_kb.items():
            client.create_collection(name)
            collection = client.get_collection(name)
            for ids, source, sha, override in rows:
                metadata = {"source": source, "file_sha256": sha, **POLICY}
                metadata.update(override or {})
                collection.add(
                    ids=ids,
                    documents=[f"{source}#{i}" for i in ids],
                    embeddings=[[float(i), 0.0, 0.0] for i in range(len(ids))],
                    metadatas=[dict(metadata) for _ in ids],
                )
        yield
    finally:
        for name in names:
            if IndexService.has_collection(name):
                client.delete_collection(name)


# ---------------------------------------------------------------- 三态判定（纯函数）

def test_missing_collection_is_created():
    assert _decide(None)[0] == "created"


def test_empty_collection_is_created():
    """空库是上次写入失败的残留：必须重灌，否则会永久 skipped 而库里没数据"""
    status, reason = _decide({"chunk_count": 0, **POLICY})
    assert status == "created", status
    assert "首次入库" in reason


def test_same_strategy_is_skipped():
    status, reason = _decide({"chunk_count": 5, **POLICY, "knowledge_base_id": 1001})
    assert status == "skipped", status
    assert "同内容同策略" in reason


def test_parser_version_change_is_replaced():
    status, reason = _decide({"chunk_count": 5, **POLICY, "parser_version": "v2"})
    assert status == "replaced", status
    assert "parser_version v2 → v3" in reason, reason


def test_clean_policy_change_is_replaced():
    status, reason = _decide({"chunk_count": 5, **POLICY, "clean_policy": "full"}, clean_policy="safe")
    assert status == "replaced", status
    assert "clean_policy full → safe" in reason, reason


def test_cleaner_version_change_is_replaced():
    """只改清洗实现、不改档位名：clean_policy 相同也必须重灌，否则新清洗结果进不了库"""
    status, reason = _decide({"chunk_count": 5, **POLICY, "cleaner_version": "c0"})
    assert status == "replaced", status
    assert "cleaner_version c0 → c1" in reason, reason


def test_splitter_version_change_is_replaced():
    status, reason = _decide({"chunk_count": 5, **POLICY, "splitter_version": "s0"})
    assert status == "replaced", status
    assert "splitter_version s0 → s1" in reason, reason


def test_legacy_meta_without_fields_is_replaced():
    """v3 之前的老数据没有 cleaner_version / splitter_version：宁可多算一次 embedding，也不静默沿用"""
    status, reason = _decide({"chunk_count": 2, "parser_version": "v3", "clean_policy": "full"})
    assert status == "replaced", status
    assert "<未标记> → c1" in reason, reason
    assert "<未标记> → s1" in reason, reason


def test_multiple_changes_all_reported():
    profile = {"chunk_count": 2, "parser_version": "v2", "clean_policy": "raw",
               "cleaner_version": "c0", "splitter_version": "s0"}
    _, reason = _decide(profile, clean_policy="full")
    assert "parser_version" in reason and "clean_policy" in reason, reason
    assert "cleaner_version" in reason and "splitter_version" in reason, reason


# ---------------------------------------------------------------- kb_id 生成

def test_next_kb_id_is_monotonic_and_unique():
    ids = [IndexService.next_knowledge_base_id() for _ in range(200)]
    assert ids == sorted(ids), "kb_id 必须单调递增，便于排序与排查"
    assert len(set(ids)) == len(ids), "kb_id 不允许重复"
    assert all(i > 0 and i < 2 ** 63 for i in ids), "必须落在 BIGINT 范围内"


# ---------------------------------------------------------------- 读取库内身份

def test_collection_profile_reads_identity():
    rows = [(["a1", "a2"], "报告.pdf", SHA_A, {"knowledge_base_id": 1001})]
    with temp_collections({KB: rows}):
        profile = IndexService.collection_profile(KB)
    assert profile["chunk_count"] == 2, profile
    assert profile["knowledge_base_id"] == 1001
    assert profile["source"] == "报告.pdf"
    assert profile["file_sha256"] == SHA_A
    assert profile["parser_version"] == "v3"
    assert profile["cleaner_version"] == "c1"
    assert profile["splitter_version"] == "s1"


def test_collection_profile_missing_returns_none():
    client = IndexService._persistent_client()
    if IndexService.has_collection(KB):
        client.delete_collection(KB)
    assert IndexService.collection_profile(KB) is None


def test_empty_collection_profile_has_no_identity():
    """空库里一条分片都没有：身份字段全是 None，但 chunk_count 必须是 0 而不是 None"""
    with temp_collections({KB: []}):
        profile = IndexService.collection_profile(KB)
    assert profile["chunk_count"] == 0, profile
    assert profile["knowledge_base_id"] is None
    assert profile["source"] is None


def test_collections_with_source_lists_orphans():
    """同名不同内容的旧库要被列出来（提示用），当前库自己不能出现"""
    with temp_collections({
        KB: [(["a1"], "报告.pdf", SHA_A, {"knowledge_base_id": 1001})],
        KB2: [(["b1"], "报告.pdf", SHA_B, {"knowledge_base_id": 1002})],
    }):
        others = IndexService.collections_with_source("报告.pdf", exclude=KB2)
    assert [o["name"] for o in others] == [KB], others
    assert others[0]["knowledge_base_id"] == 1001
    assert others[0]["chunk_count"] == 1


def test_collections_with_source_ignores_other_names():
    with temp_collections({
        KB: [(["a1"], "报告.pdf", SHA_A, None)],
        KB2: [(["b1"], "别的.pdf", SHA_B, None)],
    }):
        assert [o["name"] for o in IndexService.collections_with_source("报告.pdf")] == [KB]
        assert [o["name"] for o in IndexService.collections_with_source("别的.pdf")] == [KB2]


def test_update_source_refreshes_only_source():
    """改名重传走 skipped：只刷 source 一个字段，kb_id / 策略版本 / 分片数都不许变"""
    rows = [(["a1", "a2"], "旧名.pdf", SHA_A, {"knowledge_base_id": 1001})]
    with temp_collections({KB: rows}):
        count = IndexService.update_source(KB, "新名.pdf")
        profile = IndexService.collection_profile(KB)
        client = IndexService._persistent_client()
        metas = client.get_collection(KB).get(include=["metadatas"])["metadatas"]
    assert count == 2, count
    assert profile["source"] == "新名.pdf", profile
    assert profile["knowledge_base_id"] == 1001
    assert profile["chunk_count"] == 2
    assert profile["parser_version"] == "v3" and profile["splitter_version"] == "s1"
    assert {m["source"] for m in metas} == {"新名.pdf"}, "每一条分片都要改，不能只改第一条"


def test_update_source_noop_cases():
    """库不存在 / 空文件名：返回 0，不抛异常也不建库"""
    client = IndexService._persistent_client()
    if IndexService.has_collection(KB):
        client.delete_collection(KB)
    assert IndexService.update_source(KB, "新名.pdf") == 0
    assert IndexService.update_source(KB, "") == 0
    assert not IndexService.has_collection(KB)


# ---------------------------------------------------------------- 删除

def test_delete_by_source_counts_matched_ids():
    """删除数 = 实际命中的分片数，不用 count() 前后差（那是全库记录数，并发会串）"""
    rows = [
        (["a1", "a2", "a3"], "A.pdf", SHA_A, None),
        (["b1"], "B.pdf", SHA_B, None),
    ]
    with temp_collections({KB: rows}):
        deleted = IndexService.delete_by_source(KB, "A.pdf")
        again = IndexService.delete_by_source(KB, "A.pdf")
        remaining = IndexService._persistent_client().get_collection(KB).count()
    assert deleted == 3, deleted
    assert again == 0, "重复删同一来源应如实返回 0，不谎报"
    assert remaining == 1, "别的来源不能被误删"


def test_delete_collection_returns_kb_id_and_count():
    """整库删除要带回 kb_id：MySQL 接入后业务侧据此删记录，顺序是「先 Chroma 再 MySQL」"""
    rows = [(["a1", "a2"], "A.pdf", SHA_A, {"knowledge_base_id": 1001})]
    with temp_collections({KB: rows}):
        result = IndexService.delete_collection(KB)
        gone = IndexService.has_collection(KB)
    assert result == {"deleted": True, "knowledge_base_id": 1001, "chunk_count": 2}, result
    assert gone is False


def test_delete_missing_collection_is_idempotent():
    client = IndexService._persistent_client()
    if IndexService.has_collection(KB):
        client.delete_collection(KB)
    assert IndexService.delete_collection(KB) == {
        "deleted": False, "knowledge_base_id": None, "chunk_count": 0,
    }


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
