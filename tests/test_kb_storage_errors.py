"""向量库存储层错误码测试（2xxx）：把「故障」和「不存在」分开。

分层：
- 拿不到客户端（依赖缺失 / 目录无权 / 库文件损坏） → 2002
- collection 名不存在（含并发删除窗口）              → 2001
- 维度不匹配 / 索引异常 / 磁盘满 / 版本不兼容        → 2003 / 2004 / 2005 / 2006
- 代码传参写错（UUID 非法、参数非法、重复 id）       → 不翻译，裸抛 500
- 其它未知异常                                       → 500

运行：
    python tests/test_kb_storage_errors.py
    pytest tests/test_kb_storage_errors.py
"""

import os
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb import errors as chroma_errors
from fastapi.testclient import TestClient

from app.core.error_codes import (
    ERR_KB_DIM_MISMATCH,
    ERR_KB_INDEX_BROKEN,
    ERR_KB_NOT_FOUND,
    ERR_KB_STORAGE,
    ERR_KB_STORAGE_FULL,
    ERR_KB_VERSION_MISMATCH,
)
from app.main import app
from app.services.index_service import IndexService

client = TestClient(app, raise_server_exceptions=False)


# ------------------------------------------------------------ 翻译层（单元）

def test_dim_mismatch_is_2003():
    """换 embedding 模型 → chromadb 报维度不匹配 → 2003，提示可操作方案"""
    exc = chroma_errors.InvalidDimensionException(
        "Embedding dimension 512 does not match collection dimensionality 1024"
    )
    wrapped = IndexService._translate_chroma(exc, "demo")
    assert wrapped is not None and wrapped.code == ERR_KB_DIM_MISMATCH, wrapped
    assert "embedding" in wrapped.msg.lower()


def test_index_broken_is_2004():
    """hnsw 索引/内部状态异常 → 2004，建议重建知识库"""
    wrapped = IndexService._translate_chroma(
        chroma_errors.InternalError("Error creating hnsw index"), "demo"
    )
    assert wrapped is not None and wrapped.code == ERR_KB_INDEX_BROKEN, wrapped
    assert "重建" in wrapped.msg


def test_disk_full_is_2005():
    """磁盘满没有独立类型，靠 message / errno 识别（sqlite 与 OSError 两种都覆盖）"""
    for exc in (
        sqlite3.OperationalError("database or disk is full"),
        OSError(28, "No space left on device"),  # errno.ENOSPC
    ):
        wrapped = IndexService._translate_chroma(exc, "demo")
        assert wrapped is not None and wrapped.code == ERR_KB_STORAGE_FULL, (exc, wrapped)
        assert "磁盘空间不足" in wrapped.msg


def test_version_mismatch_is_2006():
    """chromadb 升级后打开旧库 → 2006"""
    wrapped = IndexService._translate_chroma(
        chroma_errors.VersionMismatchError("database version is newer"), "demo"
    )
    assert wrapped is not None and wrapped.code == ERR_KB_VERSION_MISMATCH, wrapped


def test_code_bugs_are_not_translated():
    """UUID 非法 / 参数非法 / 重复 id 是「我们写错了」或并发写重，不该变成业务码"""
    for exc in (
        chroma_errors.InvalidUUIDError("not a uuid"),
        chroma_errors.InvalidArgumentError("bad arg"),
        chroma_errors.DuplicateIDError("dup"),
    ):
        assert IndexService._translate_chroma(exc, "demo") is None, exc


def test_unknown_exception_is_not_translated():
    """未知异常返回 None，由调用方裸抛 → 全局兜底 500（不被误标成存储故障）"""
    assert IndexService._translate_chroma(RuntimeError("boom"), "demo") is None


# ------------------------------------------------------------ 接口层（集成）

def test_persistent_client_failure_is_2002():
    """chromadb 打不开 → 2002，且响应里不出现依赖版本、绝对路径等内部细节"""
    with patch("chromadb.PersistentClient", side_effect=RuntimeError("protobuf 6.33 required")):
        r = client.get("/api/rag/knowledge-bases")
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == ERR_KB_STORAGE, body
    assert "protobuf" not in body["msg"] and "chroma_rag_db" not in body["msg"], body["msg"]


def test_unknown_error_still_500():
    """存储层没识别出来的异常维持 500，不被包装成业务码"""
    origin = IndexService._persistent_client
    IndexService._persistent_client = classmethod(
        lambda cls: (_ for _ in ()).throw(RuntimeError("unexpected"))
    )
    try:
        r = client.get("/api/rag/knowledge-bases")
    finally:
        IndexService._persistent_client = origin
    assert r.status_code == 500, r.text
    assert r.json()["code"] == 500, r.json()
    assert "unexpected" not in r.text, "内部异常信息泄漏到响应体"


def test_collection_vanished_is_2001():
    """has_collection 说存在、取用时已被并发删掉 → NotFoundError → 2001（不是 500）"""

    class _VanishingClient:
        def get_collection(self, name):
            raise chroma_errors.NotFoundError(f"Collection {name} does not exist.")

        def list_collections(self):
            return []

    origin_has = IndexService.has_collection
    origin_client = IndexService._persistent_client
    IndexService.has_collection = classmethod(lambda cls, kb: True)
    IndexService._persistent_client = classmethod(lambda cls: _VanishingClient())
    try:
        r = client.delete("/api/rag/documents?knowledge_base=demo&source=a.md")
    finally:
        IndexService.has_collection = origin_has
        IndexService._persistent_client = origin_client
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == ERR_KB_NOT_FOUND, body


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
