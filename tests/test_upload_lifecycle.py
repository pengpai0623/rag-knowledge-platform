"""上传生命周期契约测试：created / skipped / replaced 三态走完真实 HTTP + 真实解析链路。

用假 Embedding 替掉 `IndexService._embed_model`（8 维伪向量），所以不联网、不加载
真实模型，但解析、清洗、分片、Chroma 读写全部是真实代码路径。

运行：
    python tests/test_upload_lifecycle.py
    pytest tests/test_upload_lifecycle.py
"""

import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import app
from app.services.index_service import IndexService
from utils.universal_parser import PARSER_VERSION, UniversalDocParser
from utils.structural_splitter import SPLITTER_VERSION
from utils.text_cleaner import CLEANER_VERSION

client = TestClient(app, raise_server_exceptions=False)


class _FakeEmbeddings:
    """稳定伪向量：同一段文本永远得到同一个向量，够 Chroma 建索引/做集合运算。"""

    dim = 8

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dim]]


def _sha16(data: bytes) -> str:
    """与上传链路完全一致的口径：原始字节 sha256 的十六进制前 16 位。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as f:
        f.write(data)
        path = f.name
    try:
        return UniversalDocParser.file_sha256(path)
    finally:
        os.unlink(path)


def _upload(name: str, data: bytes, policy: str = "full"):
    return client.post(
        "/api/rag/upload",
        params={"clean_policy": policy},
        files={"file": (name, data, "text/markdown")},
    ).json()


def _kb_ids_in(knowledge_base: str) -> set:
    """库里所有分片带的 kb_id（用来确认 replaced 之后没有新旧夹杂）。"""
    collection = IndexService._persistent_client().get_collection(knowledge_base)
    metas = collection.get(include=["metadatas"])["metadatas"] or []
    return {(m or {}).get("knowledge_base_id") for m in metas}


def _cleanup(*names):
    for name in names:
        if name and IndexService.has_collection(name):
            IndexService._persistent_client().delete_collection(name)


DOC_A = "# 产品说明\n\n这是一份用于验证入库生命周期的文档。\n\n## 安装\n\n执行 pip install 即可。"
DATA_A = DOC_A.encode("utf-8")
DATA_A_RENAMED = DATA_A
DATA_B = "# 另一份文档\n\n内容完全不同，因此 sha 不同、库也不同。".encode("utf-8")


def test_upload_lifecycle():
    """一条用例串起全部状态：created → skipped → replaced → 内容变 → 空库。"""
    origin = IndexService._embed_model
    IndexService._embed_model = _FakeEmbeddings()
    kb_a = "doc_" + _sha16(DATA_A)
    kb_b = "doc_" + _sha16(DATA_B)
    try:
        _cleanup(kb_a, kb_b)

        # 1) 首次入库：库名由内容决定，kb_id 由后端生成
        body = _upload("产品说明.md", DATA_A)
        assert body["code"] == 0, body
        created = body["data"]
        assert created["status"] == "created", created
        assert created["knowledge_base"] == kb_a, created
        assert created["knowledge_base_id"] and created["knowledge_base_id"] > 0
        assert created["chunk_count"] >= 1 and created["deleted_chunks"] == 0
        assert created["clean_policy"] == "full"
        assert created["cleaner_version"] == CLEANER_VERSION
        assert created["parser_version"] == PARSER_VERSION
        assert created["splitter_version"] == SPLITTER_VERSION
        assert created["same_source_kbs"] == [], created
        kb_id = created["knowledge_base_id"]
        assert _kb_ids_in(kb_a) == {kb_id}, "分片必须带同一个 kb_id"

        # 2) 原样重传：不解析、不 embed，kb_id 复用
        again = _upload("产品说明.md", DATA_A)["data"]
        assert again["status"] == "skipped", again
        assert again["knowledge_base_id"] == kb_id
        assert again["chunk_count"] == created["chunk_count"]
        assert "同内容同策略" in again["reason"], again

        # 3) 改名重传（内容相同）：库名与文件名无关，仍落回同一个库；
        #    但源文件名要顺手刷新，否则引用里永远停在第一次入库的名字
        renamed = _upload("产品说明-最终版.md", DATA_A_RENAMED)["data"]
        assert renamed["status"] == "skipped", renamed
        assert renamed["knowledge_base"] == kb_a
        assert renamed["knowledge_base_id"] == kb_id
        assert renamed["source"] == "产品说明-最终版.md", renamed
        assert "来源名 产品说明.md → 产品说明-最终版.md" in renamed["reason"], renamed
        after_rename = IndexService.collection_profile(kb_a)
        assert after_rename["source"] == "产品说明-最终版.md", "改名后库内 source 必须刷新"
        assert after_rename["knowledge_base_id"] == kb_id, "刷新 source 不能动 kb_id"
        assert after_rename["chunk_count"] == created["chunk_count"], "刷新 source 不能增删分片"
        assert after_rename["splitter_version"] == SPLITTER_VERSION, "版本号也不能被动到"

        # 4) 换清洗策略：同库同 kb_id 重灌，旧分片全部删掉
        replaced = _upload("产品说明.md", DATA_A, policy="safe")["data"]
        assert replaced["status"] == "replaced", replaced
        assert replaced["knowledge_base"] == kb_a
        assert replaced["knowledge_base_id"] == kb_id, "策略变了不该换身份"
        assert replaced["deleted_chunks"] == created["chunk_count"], replaced
        assert replaced["clean_policy"] == "safe", "回显的是库内实际生效的策略"
        assert "clean_policy" in replaced["reason"] and "safe" in replaced["reason"]
        assert _kb_ids_in(kb_a) == {kb_id}, "重灌后不能残留旧策略分片或旧 kb_id"

        # 5) 内容变了：新库新 ID，旧库作为「同名历史库」被提示出来（不自动删）
        new_doc = _upload("产品说明.md", DATA_B)["data"]
        assert new_doc["status"] == "created", new_doc
        assert new_doc["knowledge_base"] == kb_b, new_doc
        assert new_doc["knowledge_base_id"] != kb_id
        assert [o["name"] for o in new_doc["same_source_kbs"]] == [kb_a], new_doc
        assert IndexService.has_collection(kb_a), "旧库不能被自动删除（可能误伤别人的同名文件）"

        # 6) 空库（上次写入失败的残留）必须当 created 重灌，不能 skipped
        empty_data = "# 空库验证\n\n先建一个没有分片的库，再上传同一份内容。".encode("utf-8")
        empty_kb = "doc_" + _sha16(empty_data)
        _cleanup(empty_kb)
        IndexService._persistent_client().create_collection(empty_kb)
        try:
            revived = _upload("空库.md", empty_data)["data"]
            assert revived["status"] == "created", revived
            assert revived["knowledge_base"] == empty_kb
            assert revived["chunk_count"] >= 1
        finally:
            _cleanup(empty_kb)

        # 7) 列表接口能读出身份
        items = {i["name"]: i for i in client.get("/api/rag/knowledge-bases").json()["data"]}
        assert items[kb_a]["knowledge_base_id"] == kb_id
        assert items[kb_a]["source"] == "产品说明.md"
        assert items[kb_a]["doc_count"] == 1

        # 8) 整库删除带回 kb_id，便于业务侧同步删 MySQL 记录
        removed = client.delete(f"/api/rag/knowledge-bases/{kb_a}").json()["data"]
        assert removed["knowledge_base_id"] == kb_id, removed
        assert removed["deleted_chunks"] == replaced["chunk_count"]
        assert not IndexService.has_collection(kb_a)
    finally:
        _cleanup(kb_a, kb_b)
        IndexService._embed_model = origin


def test_cleaner_upgrade_triggers_replaced():
    """只升清洗实现代数、档位名不变：必须 replaced，不能被 clean_policy 相同骗成 skipped。

    模拟「改了 text_cleaner 的规则并把 CLEANER_VERSION 从 c1 提到 c9」：
    同时打补丁到写 metadata 的一侧（text_cleaner）和做判定的一侧（router）。
    """
    origin_embed = IndexService._embed_model
    IndexService._embed_model = _FakeEmbeddings()
    data = "# 清洗换代号\n\n同一份内容、同一个 full 档位，只有 cleaner 实现换代。".encode("utf-8")
    kb = "doc_" + _sha16(data)

    from app.routers import rag_router
    from utils import text_cleaner

    origin_router, origin_cleaner = rag_router.CLEANER_VERSION, text_cleaner.CLEANER_VERSION
    try:
        _cleanup(kb)
        first = _upload("换代号.md", data)["data"]
        assert first["status"] == "created", first
        assert first["cleaner_version"] == origin_cleaner, first

        rag_router.CLEANER_VERSION = "c9"
        text_cleaner.CLEANER_VERSION = "c9"
        bumped = _upload("换代号.md", data)["data"]
        assert bumped["status"] == "replaced", bumped
        assert bumped["knowledge_base_id"] == first["knowledge_base_id"], "换代不该换身份"
        assert bumped["deleted_chunks"] == first["chunk_count"], bumped
        assert "cleaner_version" in bumped["reason"] and "c9" in bumped["reason"], bumped
        assert bumped["cleaner_version"] == "c9"
    finally:
        rag_router.CLEANER_VERSION = origin_router
        text_cleaner.CLEANER_VERSION = origin_cleaner
        _cleanup(kb)
        IndexService._embed_model = origin_embed


def test_upload_rejects_unknown_extension():
    body = client.post(
        "/api/rag/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}
    ).json()
    assert body["code"] == 1008, body


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
