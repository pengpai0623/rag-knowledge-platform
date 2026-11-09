"""解析预览接口 /api/rag/parse/preview 的契约测试。

它只做 解析 → 清洗 → 分片，不写向量库，所以重点验证：
1. 三个阶段的结果都能拿到，且分片参数可调；
2. **不入库**——调用前后知识库列表不变（这是它区别于 /upload 的根本点）。

不联网：全程不加载 Embedding。
运行：
    python tests/test_parse_preview.py
    pytest tests/test_parse_preview.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import app
from app.services.index_service import IndexService

client = TestClient(app, raise_server_exceptions=False)

URL = "/api/rag/parse/preview"

DOC = (
    "# 产品手册\n\n"
    "\u3000\u3000第一章 入职\n\n"
    "    缩进两级的说明，这里有   连续空格。\n\n\n\n"
    "第二章 考勤\n员工需按时打卡。\n"
)
FILES = {"file": ("手册.md", DOC.encode("utf-8"), "text/markdown")}


def _parse(**kwargs):
    return client.post(URL, files=FILES, **kwargs).json()


def test_route_registered():
    assert URL in app.openapi()["paths"]


def test_returns_three_stages():
    body = _parse()
    assert body["code"] == 0, body
    data = body["data"]
    # 只给三段，不再有 file / params / stats / cost_ms
    assert set(data) == {"raw", "cleaned", "chunks"}, data.keys()
    assert data["raw"]["docs"] >= 1
    assert data["chunks"], "应当返回分片明细"
    first = data["chunks"][0]
    assert first["content"] and first["chars"] == len(first["content"])
    assert first["metadata"]["source"] == "手册.md"


def test_chunk_content_matches_cleaned_text():
    """分片内容必须来自清洗后的文本，否则「看分片」看到的就不是真正入库的东西"""
    data = _parse()["data"]
    joined = "".join(c["content"] for c in data["chunks"])
    # 清洗后原文里的连续空格/空行已经没了，分片里也不该出现
    assert "   连续空格" not in joined
    assert "\n\n\n" not in joined


def test_policies_differ():
    raw = _parse(params={"clean_policy": "raw"})["data"]
    full = _parse(params={"clean_policy": "full"})["data"]
    # raw 保留缩进与空行，full 不会：所以 full 的清洗后字数一定更少
    assert raw["cleaned"]["chars"] > full["cleaned"]["chars"]
    assert "   连续空格" in raw["cleaned"]["preview"]


def test_chunk_size_is_adjustable():
    small = _parse(params={"chunk_size": 40, "chunk_overlap": 5})["data"]
    big = _parse(params={"chunk_size": 400, "chunk_overlap": 0})["data"]
    assert len(small["chunks"]) > len(big["chunks"]), "片切得越小，片数应越多"


def test_overlap_must_be_smaller_than_size():
    body = _parse(params={"chunk_size": 50, "chunk_overlap": 50})
    assert body["code"] == 400, body
    assert "chunk_overlap" in body["msg"]


def test_returns_all_chunks_without_truncation():
    """不再有 limit：返回的就是全部分片，len(chunks) 即真实片数"""
    tiny = _parse(params={"chunk_size": 20, "chunk_overlap": 0})["data"]
    assert len(tiny["chunks"]) >= 2, "小 chunk_size 下应切出多片且全部返回"
    assert [c["index"] for c in tiny["chunks"]] == list(range(len(tiny["chunks"])))


def test_unsupported_extension_rejected():
    r = client.post(URL, files={"file": ("a.exe", b"x", "application/octet-stream")})
    assert r.json()["code"] == 1008, r.text


def test_does_not_touch_vector_store():
    """核心约定：这个接口不写库。前后知识库清单必须完全一致"""
    before = IndexService.list_knowledge_bases()
    _parse(params={"chunk_size": 100, "chunk_overlap": 10})
    assert IndexService.list_knowledge_bases() == before


def test_preview_matches_upload_pipeline():
    """预览必须与上传链路产出完全一致，否则「看到的切法」不是「真正入库的切法」。

    这是把两条链路焊在一起的关键用例：正文逐字比、metadata 逐键比，三档策略全覆盖。
    source_path 排除在外——两边用的是不同的临时文件，路径天然不同。
    """
    import tempfile

    from app.services.parse_service import ParseService

    with tempfile.NamedTemporaryFile("wb", suffix=".md", delete=False) as f:
        f.write(DOC.encode("utf-8"))
        tmp_path = f.name
    try:
        for policy in ("full", "safe", "raw"):
            # A 路：上传实际走的
            direct = ParseService.parse_and_split(
                tmp_path, source_name="手册.md", clean_policy=policy
            )
            # B 路：预览接口
            preview = _parse(
                params={"clean_policy": policy}
            )["data"]["chunks"]

            assert len(direct) == len(preview), (
                f"{policy}: 分片数不同 {len(direct)} vs {len(preview)}"
            )
            for i, (d, p) in enumerate(zip(direct, preview)):
                assert d.page_content == p["content"], f"{policy}[{i}] 正文不同"
                a = {k: v for k, v in d.metadata.items() if k != "source_path"}
                b = {k: v for k, v in p["metadata"].items() if k != "source_path"}
                assert a == b, f"{policy}[{i}] metadata 不同：{a} != {b}"
    finally:
        os.unlink(tmp_path)


def test_cleaner_and_splitter_shared():
    """结构上的约束：清洗器与分片入口都只能有一个来源。

    行为一致性由 test_preview_matches_upload_pipeline 兜底，这条守的是「为什么会一致」——
    一旦有人给预览加个 TextCleaner(...) 或另起一套切分，两条链路就会开始各自演化。
    """
    from app.services.parse_service import ParseService
    from utils.structural_splitter import StructuralSplitter

    for policy in ("full", "safe", "raw"):
        assert ParseService.get_cleaner(policy) is ParseService.get_cleaner(policy), (
            f"{policy}: 同一策略应复用同一实例，不应每次新建"
        )

    # 分片器按 (size, overlap) 缓存，同参数必须拿到同一个对象
    assert StructuralSplitter._get_splitter(800, 100) is StructuralSplitter._get_splitter(800, 100)

    try:
        ParseService.get_cleaner("bogus")
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("未知清洗策略必须报错，不能静默回退到默认策略")


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
