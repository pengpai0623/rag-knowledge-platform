"""知识库名校验：非法名字必须是 400（参数错误），不能落成 500。

背景：knowledge_base 直接当 Chroma 的 collection 名用。名字不合规时 chromadb 抛
InvalidArgumentError，若不拦就会走到全局兜底 → 500，把「用户传参错」伪装成服务端故障。
按架构约定「外部来的数据不合法 → 业务异常」，在校验边界给 400。

规则（Chromadb 实测）：3-512 位，只允许 [a-zA-Z0-9._-]，首尾必须是字母或数字。

不联网、不加载 Embedding。
运行：
    python tests/test_kb_name_validation.py
    pytest tests/test_kb_name_validation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.core.error_codes import CODE_ERROR
from app.core.exceptions import ParamError
from app.main import app
from app.routers.rag_router import _ensure_valid_kb_name

client = TestClient(app, raise_server_exceptions=False)


def _rejected(name, *, allow_empty=False):
    """断言该名字被拒且码是 400，返回异常供检查文案"""
    try:
        _ensure_valid_kb_name(name, allow_empty=allow_empty)
    except ParamError as e:
        assert e.code == CODE_ERROR, f"{name!r} 的码是 {e.code}，应为 {CODE_ERROR}"
        assert "知识库名不合法" in e.msg
        return e
    raise AssertionError(f"应当拒绝：{name!r}")


def test_valid_names_pass():
    for name in ("default", "kb_2024", "product-docs", "a.b-c_1", "abc", "a" * 512):
        assert _ensure_valid_kb_name(name) == name


def test_too_short_rejected():
    for name in ("", "a", "ab"):
        _rejected(name)


def test_bad_first_or_last_char_rejected():
    for name in ("_abc", "abc_", "-abc", "abc.", ".abc", " abc", "abc "):
        _rejected(name)


def test_illegal_chars_rejected():
    # 中文、空格、斜杠、纯符号：这些都很容易被用户当库名输入
    for name in ("产品手册", "my kb", "a/b", "...", "kb@2024"):
        _rejected(name)


def test_oversize_rejected():
    _rejected("a" * 513)


def test_empty_allowed_only_when_explicit():
    # GET /collections 用空串表示「全部」，那条路径必须放行
    assert _ensure_valid_kb_name("", allow_empty=True) == ""
    _rejected("")


def test_upload_rejects_bad_kb_name():
    r = client.post(
        "/api/rag/upload?knowledge_base=_bad",
        files={"file": ("a.md", b"# hi", "text/markdown")},
    )
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 400, body
    assert "知识库名不合法" in body["msg"]


def test_chat_rejects_bad_kb_name():
    r = client.post("/api/rag/chat", json={"question": "你好", "knowledge_base": "产品手册"})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 400, body


def test_delete_endpoints_reject_bad_kb_name():
    r = client.delete("/api/rag/documents", params={"knowledge_base": "_bad", "source": "a.md"})
    assert r.json()["code"] == 400, r.text

    r = client.delete("/api/rag/knowledge-bases/_bad")
    assert r.json()["code"] == 400, r.text


def test_collections_accepts_empty_but_rejects_bad():
    # 空串 = 查全部，必须正常返回（当前库为空，data 是空列表）
    r = client.get("/api/rag/collections", params={"knowledge_base": ""})
    assert r.json()["code"] == 0, r.text

    r = client.get("/api/rag/collections", params={"knowledge_base": "_bad"})
    assert r.json()["code"] == 400, r.text


def test_bad_name_never_reaches_chroma():
    """回归防线：非法名若在服务层才被发现就是 500，这里确认路由层已经拦死（非 500 即已拦）"""
    body = client.get("/api/rag/collections", params={"knowledge_base": "a b"}).json()
    assert body["code"] != 500, "非法库名仍然落到了 500"
    assert body["code"] == 400, body


if __name__ == "__main__":
    cases = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
