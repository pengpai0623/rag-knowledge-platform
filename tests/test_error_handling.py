"""全局异常体系与统一响应体的接口测试：只跑不依赖 Embedding 的分支（不联网）。

状态码策略（架构文档 §5，方案 B）：
- 业务异常、Pydantic 校验 → HTTP 200 + body.code
- 路由 404 / 方法 405 / 上传 413 / 兜底 500 → 保留真实 HTTP 状态

运行：
    python tests/test_error_handling.py
    pytest tests/test_error_handling.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import app.routers.rag_router as rag_router
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_routes_registered():
    paths = app.openapi()["paths"]
    assert "/api/rag/upload" in paths
    assert "/api/rag/knowledge-bases/{knowledge_base}" in paths, "清空知识库端点缺失"
    assert "delete" in paths["/api/rag/knowledge-bases/{knowledge_base}"]


def test_health_wrapped():
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 200
    assert body["code"] == 0 and body["msg"] == "ok"
    assert body["data"] == {"status": "ok", "service": "rag-knowledge-platform"}
    assert body["request_id"]


def test_request_id_header_matches_body():
    r = client.get("/api/rag/__nope__")
    assert r.headers.get("X-Request-ID") == r.json()["request_id"]


# ---------------- 第 1 层：Pydantic 请求体校验 → HTTP 200 + code 422 ----------------

def test_validation_error_envelope():
    r = client.post("/api/rag/chat", json={})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 422
    assert "参数校验失败" in body["msg"]
    fields = {e["field"] for e in body["data"]["errors"]}
    assert {"question", "knowledge_base"} <= fields, body["data"]
    assert body["request_id"]


# ---------------- 第 2 层：路由级 404 / 405 / 413 → 保留 HTTP 码 ----------------

def test_route_not_found_404():
    r = client.get("/api/rag/__no_such_path__")
    body = r.json()
    assert r.status_code == 404
    assert body["code"] == 404 and "路由不存在" in body["msg"]


def test_method_not_allowed_405():
    r = client.delete("/api/rag/knowledge-bases")  # 列表端点只有 GET
    body = r.json()
    assert r.status_code == 405, r.text
    assert body["code"] == 405 and "方法不支持" in body["msg"]


def test_upload_too_large_keeps_413():
    """413 保留真实状态码：反向代理超限时返回的是裸 413，前端按状态码统一处理"""
    origin = rag_router.MAX_UPLOAD_MB
    rag_router.MAX_UPLOAD_MB = 0
    try:
        r = client.post(
            "/api/rag/upload",
            files={"file": ("big.md", b"# hello world", "text/markdown")},
        )
        body = r.json()
        assert r.status_code == 413, r.text
        assert body["code"] == 413 and "上限" in body["msg"]
    finally:
        rag_router.MAX_UPLOAD_MB = origin


# ---------------- 第 3 层：RAGBaseError 业务异常 → HTTP 200 + 业务码 ----------------

def test_unsupported_extension_business_error():
    r = client.post(
        "/api/rag/upload",
        files={"file": ("payload.exe", b"MZ...", "application/octet-stream")},
    )
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 1008, body
    assert "不支持的文件格式" in body["msg"]


def test_kb_not_found_code_2001():
    r = client.delete("/api/rag/knowledge-bases/nope-kb")
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 2001 and "知识库不存在" in body["msg"]


def test_param_error_code_400():
    r = client.delete("/api/rag/documents", params={"knowledge_base": "kb1", "source": "  "})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 400 and "source 不能为空" in body["msg"]


def test_empty_question_maps_to_param_error():
    """question 为空是外部数据问题：源头抛 ParamError，不是 500，也不是 422"""
    r = client.post("/api/rag/chat", json={"question": "   ", "knowledge_base": "nope-kb"})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 400 and "question 不能为空" in body["msg"]


def test_chat_unknown_kb_maps_to_2001():
    r = client.post("/api/rag/chat", json={"question": "q", "knowledge_base": "nope-kb"})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 2001


def test_chat_invalid_retrieval_mode_maps_to_param_error():
    r = client.post("/api/rag/chat", json={
        "question": "q", "knowledge_base": "kb1", "retrieval_mode": "bogus",
    })
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["code"] == 400 and "检索策略" in body["msg"]


# 向量库存储层的错误码（2001~2006、未知异常 500）集中在 tests/test_kb_storage_errors.py：
# 那些用例要 patch chromadb 客户端本身，和这里的「纯 handler 行为」分开更清楚。


def test_generate_error_code_3001():
    """模型端点故障 → 封装 GenerateError(3001)，上游报文不外泄"""
    from app.services.generate_service import GenerateService
    from app.services.retrieve_service import RetrieveService

    origin_ctx = RetrieveService.get_contexts
    origin_gen = GenerateService.generate
    RetrieveService.get_contexts = classmethod(lambda cls, *a, **k: [])
    GenerateService.generate = classmethod(
        lambda cls, q, c: (_ for _ in ()).throw(RuntimeError("upstream says secret"))
    )
    try:
        r = client.post("/api/rag/chat", json={"question": "q", "knowledge_base": "kb1"})
        body = r.json()
        assert r.status_code == 200, r.text
        assert body["code"] == 3001, body
        assert "upstream" not in r.text and "secret" not in r.text
    finally:
        RetrieveService.get_contexts = origin_ctx
        GenerateService.generate = origin_gen


def test_stream_error_event_uses_code_and_hides_detail():
    """SSE 流内错误：共享码表但独立结构，未预期异常用 500 且不带原始报文"""
    import json as _json

    from app.services.generate_service import GenerateService
    from app.services.retrieve_service import RetrieveService
    from langchain_core.documents import Document

    origin_ctx = RetrieveService.get_contexts
    origin_stream = GenerateService.generate_stream
    RetrieveService.get_contexts = classmethod(
        lambda cls, *a, **k: [Document(page_content="ctx", metadata={"source": "a.md"})]
    )

    def _boom(cls, q, c):
        def gen():
            raise RuntimeError("internal detail")
            yield  # pragma: no cover
        return gen()

    GenerateService.generate_stream = classmethod(_boom)
    try:
        r = client.post("/api/rag/chat",
                        json={"question": "q", "knowledge_base": "kb1", "stream": True})
        events = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
        errs = [_json.loads(e) for e in events if e != "[DONE]" and '"error"' in e]
        assert errs, r.text
        assert errs[0]["code"] == 500 and "internal detail" not in r.text
    finally:
        RetrieveService.get_contexts = origin_ctx
        GenerateService.generate_stream = origin_stream


# ---------------- 第 4 层：Exception 兜底 → 500 固定文案，不泄露内部信息 ----------------

def test_uncaught_exception_sanitized():
    from app.services.index_service import IndexService

    origin = IndexService.list_knowledge_bases
    IndexService.list_knowledge_bases = classmethod(
        lambda cls: (_ for _ in ()).throw(RuntimeError("boom secret E:\\secret\\path"))
    )
    try:
        r = client.get("/api/rag/knowledge-bases")
        body = r.json()
        assert r.status_code == 500, r.text
        assert body["code"] == 500
        assert "服务内部错误" in body["msg"]
        # 内部实现不得外泄
        assert "boom" not in r.text and "secret" not in r.text
        assert body["request_id"] and body["request_id"] != "-"
    finally:
        IndexService.list_knowledge_bases = origin


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
