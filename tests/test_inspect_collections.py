"""巡检接口契约测试：GET /api/rag/collections。

不加载 Embedding、不联网：直接写假向量，只验证「collection → source 聚合」的语义。
运行：
    python tests/test_inspect_collections.py
    pytest tests/test_inspect_collections.py
"""

import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.index_service import IndexService

KB_A = "tmp-inspect-a"
KB_B = "tmp-inspect-b"


@contextlib.contextmanager
def temp_collections():
    """建两个临时 collection，测试结束一定删掉。"""
    client = IndexService._persistent_client()
    for name in (KB_A, KB_B):
        if IndexService.has_collection(name):
            client.delete_collection(name)

    a = client.get_collection(client.create_collection(KB_A, metadata={"hnsw:space": "cosine"}).name)
    # 4 条 a.md（v2）、2 条 b.pdf（无 parser_version）、1 条无 source（v1 老数据）
    a.add(
        ids=[f"a{i}" for i in range(4)] + [f"b{i}" for i in range(2)] + ["legacy"],
        documents=[f"doc-{i}" for i in range(7)],
        embeddings=[[float(i), 0.1, 0.2] for i in range(7)],
        metadatas=[{"source": "a.md", "parser_version": "v2"} for _ in range(4)]
        + [{"source": "b.pdf"} for _ in range(2)]
        + [{"parser_version": "v1"}],
    )
    b = client.get_collection(client.create_collection(KB_B).name)
    b.add(
        ids=["c1"],
        documents=["doc-c"],
        embeddings=[[0.5, 0.5, 0.5]],
        metadatas=[{"source": "c.txt", "parser_version": "v2"}],
    )
    try:
        yield
    finally:
        for name in (KB_A, KB_B):
            if IndexService.has_collection(name):
                client.delete_collection(name)


def _find(items, name):
    return next(item for item in items if item["name"] == name)


def test_groups_chunks_by_source():
    """同一 collection 内按 source 聚合，分片数正确、按数量降序"""
    with temp_collections():
        items = IndexService.inspect_collections(KB_A)
    only = _find(items, KB_A)
    assert only["chunk_count"] == 7, only
    assert only["doc_count"] == 3, "a.md / b.pdf / <未知来源> 三个来源"
    assert [(s["source"], s["chunk_count"]) for s in only["sources"]] == [
        ("a.md", 4), ("b.pdf", 2), ("<未知来源>", 1)
    ], only["sources"]


def test_parser_versions_visible():
    """解析版本透出：缺失标 <未标记>，用于识别 v1 残留"""
    with temp_collections():
        items = IndexService.inspect_collections(KB_A)
    sources = {s["source"]: s["parser_versions"] for s in _find(items, KB_A)["sources"]}
    assert sources["a.md"] == ["v2"], sources
    assert sources["b.pdf"] == ["<未标记>"], sources
    assert sources["<未知来源>"] == ["v1"], sources


def test_collection_metadata_exposed():
    """collection 级 metadata（距离度量）原样透出"""
    with temp_collections():
        items = IndexService.inspect_collections(KB_A)
    assert _find(items, KB_A)["metadata"].get("hnsw:space") == "cosine"


def test_filter_by_knowledge_base():
    """传库名只返回该 collection"""
    with temp_collections():
        items = IndexService.inspect_collections(KB_B)
    assert [item["name"] for item in items] == [KB_B], items
    assert items[0]["sources"] == [
        {"source": "c.txt", "chunk_count": 1, "parser_versions": ["v2"]}
    ]


def test_router_returns_payload_and_2001():
    """路由层：正常返回包装体；未知库返回 code 2001（HTTP 200）"""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    with temp_collections():
        r = client.get(f"/api/rag/collections?knowledge_base={KB_A}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["code"] == 0, body
        assert body["data"][0]["name"] == KB_A
        assert body["data"][0]["sources"][0]["source"] == "a.md"

    r = client.get("/api/rag/collections?knowledge_base=nope-kb")
    assert r.status_code == 200, r.text
    assert r.json()["code"] == 2001, r.json()


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
