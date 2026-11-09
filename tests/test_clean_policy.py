"""清洗策略（full / safe / raw）契约测试：验证「选哪一档，结构就保住到什么程度」。

不联网、不加载 Embedding：只测 TextCleaner 与路由参数校验。
运行：
    python tests/test_clean_policy.py
    pytest tests/test_clean_policy.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.services.parse_service import ParseService
from utils.text_cleaner import CLEAN_POLICIES, TextCleaner

client = TestClient(app, raise_server_exceptions=False)

# 覆盖四类需要保护/清理的特征：零宽字符、连续空行、缩进、全角数字、中文旁的英文逗号、重复行
SAMPLE = (
    "　　企业手册\u200b\n"
    "\n"
    "\n"
    "第一章 入职\n"
    "    缩进两级的内容\n"
    "    缩进两级的内容\n"
    "Ａ１ 规格,说明\n"
)

INDENT_LINE = "    缩进两级的内容"


def test_policies_are_declared():
    assert CLEAN_POLICIES == ("full", "safe", "raw")


def test_full_is_legacy_behaviour():
    """full = 历史默认：去缩进、去空行、翻译全角字符。"""
    out = TextCleaner("full").clean_single_text(SAMPLE)
    assert INDENT_LINE.strip() in out
    assert INDENT_LINE not in out, "full 逐行 strip，缩进应被删掉"
    assert "\n\n" not in out, "full 会删掉所有空行"
    assert "Ａ１" not in out, "full 应把全角数字转半角"
    assert "\u200b" not in out
    assert out.count("缩进两级的内容") == 1, "full 会做相邻行去重"


def test_safe_keeps_layout_but_still_cleans():
    """safe = 保留缩进与段落空行，其余清洗照做。"""
    out = TextCleaner("safe").clean_single_text(SAMPLE)
    assert INDENT_LINE in out, "safe 不得 strip 行，否则缩进丢失"
    assert "\n\n" in out, "safe 保留段落空行"
    assert "\n\n\n" not in out, "连续空行应压成一个"
    assert "Ａ１" not in out, "safe 仍做全角转半角"
    assert "，" in out, "safe 仍做中文标点规范化"
    assert "\u200b" not in out


def test_raw_only_strips_invisible_chars():
    """raw = 只删不可见字符，其余原样（代码块/公式/ASCII 图表用）。"""
    out = TextCleaner("raw").clean_single_text(SAMPLE)
    assert INDENT_LINE in out
    assert "\n\n\n" in out, "raw 不压缩空行"
    assert "Ａ１" in out, "raw 不做全角转半角"
    assert "规格,说明" in out, "raw 不改标点"
    assert "\u200b" not in out, "不可见字符在所有策略下都删"
    assert out.count("缩进两级的内容") == 2, "raw 不去重"


def test_unknown_policy_raises():
    for bad in ("", "none", "FULL"):
        try:
            TextCleaner(bad)
        except ValueError:
            continue
        raise AssertionError(f"未知策略 {bad!r} 应抛 ValueError")


def test_parse_service_rejects_unknown_policy_before_parsing():
    try:
        ParseService.parse_and_split("not-exists.md", "x.md", clean_policy="nope")
    except ValueError:
        return
    raise AssertionError("ParseService 未校验 clean_policy")


def test_metadata_carries_policy():
    """clean_policy 必须落进 metadata：它参与 mode=skip 的幂等判断。"""
    docs = [Document(page_content="第一行\n    第二行", metadata={"source": "a.md"})]
    out = TextCleaner("safe").clean_documents(docs)
    assert out and out[0].metadata["clean_policy"] == "safe"


def test_router_exposes_clean_policy_param():
    params = app.openapi()["paths"]["/api/rag/upload"]["post"]["parameters"]
    names = {p["name"] for p in params}
    assert "clean_policy" in names
    spec = next(p for p in params if p["name"] == "clean_policy")
    assert spec["schema"]["default"] == "full"


def test_router_rejects_unknown_policy():
    r = client.post(
        "/api/rag/upload?clean_policy=nope",
        files={"file": ("a.md", b"# hi", "text/markdown")},
    )
    body = r.json()
    assert r.status_code == 200, r.text          # 校验错误 → 200 + code 422
    assert body["code"] == 422, body


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
