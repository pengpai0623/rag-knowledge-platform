import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from app.core.handlers import register_exception_handlers
from app.core.response import ok
from app.routers.rag_router import router as rag_router
from utils.logger import get_logger, reset_request_id, set_request_id, setup_logging

setup_logging()
logger = get_logger(__name__)

# 服务标识：探针与多服务部署时用于区分来源
SERVICE_NAME = os.getenv("APP_NAME", "rag-knowledge-platform")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("RAG 知识平台 API 启动")
    yield
    logger.info("RAG 知识平台 API 停止")


app = FastAPI(title="RAG 知识平台 API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """给每个请求分配追踪标识，日志按 rid 串联；同时把 rid 回写到响应头，方便前端/排障对账"""
    request_id = request.headers.get("X-Request-ID") or uuid4().hex[:12]
    token = set_request_id(request_id)
    started = time.perf_counter()
    status = "-"
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        # 兜底 handler 执行时 contextvar 已被复位，把 rid 标注在异常对象上带到第 4 层
        setattr(exc, "request_id", request_id)
        raise
    finally:
        logger.info("%s %s -> %s | %.1fms", request.method, request.url.path, status,
                    (time.perf_counter() - started) * 1000)
        reset_request_id(token)


# 注册 RAG 业务路由
app.include_router(rag_router)

# 全局异常处理：Pydantic 校验 / 路由 404·405 / 业务异常 / 兜底
register_exception_handlers(app)


@app.get("/health", summary="健康检查")
def health_check():
    """探针用：保持轻量（不查向量库/模型），但返回体仍是统一结构，便于前端统一解析"""
    return ok({"status": "ok", "service": SERVICE_NAME})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
