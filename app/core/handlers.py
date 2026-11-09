"""全局异常 handler：四层分发，统一构造 ApiResponse 形状的错误响应。

层级（注册顺序无关，FastAPI 按异常类型的 MRO 匹配）：
1. RequestValidationError       请求体 Pydantic 校验失败 → HTTP 200 / code 422
2. StarletteHTTPException       路由 404/405 及显式 HTTPException → 保留 HTTP 码
3. RAGBaseError 及其子类        业务异常 → HTTP 200 / 自带 code
4. Exception                    兜底 → HTTP 500 / 固定文案，细节只进日志

HTTP 状态策略（架构文档 §5，方案 B）：
- 业务能解释的错误回 200，靠 body.code 区分（业务异常、参数校验）；
- 只有「请求打到了不存在的地方/方法不对」和「服务崩了」才用 404/405/500，
  上传超限保留 413（反向代理也会返回裸 413，前端需按状态码统一处理）。

注意：
- 必须注册在 StarletteHTTPException 上（fastapi.HTTPException 是它的子类），
  注册子类接不住框架在路由匹配阶段抛的 404/405；
- 第 4 层由 ServerErrorMiddleware 处理，发完响应后框架仍会 re-raise，
  uvicorn 会再打一遍 traceback——接受这个重复，不自己造中间件；
- 流式（SSE）响应不经过这里：流已开始后无法改状态码，
  流内错误由各流式端点按共享码表自行发 error 事件。
"""
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.error_codes import (
    CODE_METHOD_NOT_ALLOWED,
    CODE_NOT_FOUND,
    CODE_SERVER_ERROR,
    CODE_VALIDATE_ERROR,
)
from app.core.exceptions import RAGBaseError
from utils.logger import get_logger, get_request_id, reset_request_id, set_request_id

logger = get_logger("app.core.handlers")

# 兜底响应的固定文案：不包含任何内部实现信息
_SERVER_ERROR_MSG = "服务内部错误，请稍后重试；如持续失败请联系管理员并提供 request_id"


def _error_response(http_code: int, code: int, msg: str, data=None, request_id: str = None) -> JSONResponse:
    """统一错误响应体，字段与 ApiResponse 完全一致"""
    return JSONResponse(
        status_code=http_code,
        content={
            "code": code,
            "msg": msg,
            "data": data,
            "request_id": request_id if request_id is not None else get_request_id(),
        },
    )


async def _handle_validation_error(request, exc: RequestValidationError) -> JSONResponse:
    """第 1 层：Pydantic 校验失败。一个总码 + 逐字段明细，一次把错误改全。

    HTTP 仍返回 200：这是「请求内容不符合业务约定」，不是通信失败，
    前端只判 body.code 即可（架构文档 §5）。
    """
    errors = []
    for err in exc.errors():
        loc = list(err.get("loc", []))
        # 去掉 "body" 前缀，field 直接对齐请求体字段名（嵌套字段用点号连接）
        if loc and loc[0] == "body":
            loc = loc[1:]
        errors.append({
            "field": ".".join(str(x) for x in loc) or "(root)",
            "type": err.get("type", ""),
            "message": err.get("msg", ""),
        })
    logger.warning("参数校验失败 | %s %s | %d 处 | %s",
                   request.method, request.url.path, len(errors), errors)
    return _error_response(
        status.HTTP_200_OK,
        CODE_VALIDATE_ERROR,
        "请求参数校验失败",
        data={"errors": errors},
    )


async def _handle_http_exception(request, exc: StarletteHTTPException) -> JSONResponse:
    """第 2 层：路由级 404/405 与显式 HTTPException。code 复用 HTTP 状态码。

    这一层保留真实 HTTP 状态：404/405 是「调用方打错了地方」，413 是「请求被拒」，
    都属于通信层问题，网关与监控依赖它们（架构文档 §5）。
    """
    if exc.status_code == status.HTTP_404_NOT_FOUND:
        msg = "路由不存在，请检查请求路径"
    elif exc.status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        msg = "请求方法不支持，请检查 HTTP 方法"
    else:
        # 业务代码显式抛的 HTTPException（如 413 超限），detail 面向用户、直接透传
        msg = str(exc.detail) if exc.detail is not None else "请求错误"
    logger.info("%s %s -> %d | %s", request.method, request.url.path, exc.status_code, msg)
    return _error_response(exc.status_code, exc.status_code, msg)


async def _handle_rag_base_error(request, exc: RAGBaseError) -> JSONResponse:
    """第 3 层：业务异常。HTTP 200，靠 body.code 区分（架构文档 §5）。

    msg 允许带用户上下文（页码、文件名、库名），但不允许带内部实现细节——
    那类信息由抛异常的一侧决定，规范见 exceptions.py 顶部说明。
    """
    logger.warning("业务异常 | %s %s | code=%s | %s",
                   request.method, request.url.path, exc.code, exc.msg)
    return _error_response(status.HTTP_200_OK, exc.code, exc.msg, data=exc.data)


async def _handle_uncaught_exception(request, exc: Exception) -> JSONResponse:
    """第 4 层：兜底。内部信息（堆栈、路径、第三方报文）只进日志，响应体固定文案。"""
    # contextvar 此时已被日志中间件复位，兜底层的 rid 由中间件标注在异常对象上；
    # 这里临时恢复上下文，让这条 ERROR 日志也能带上可检索的 rid（否则是 rid=-）
    rid = getattr(exc, "request_id", None) or get_request_id()
    token = set_request_id(rid)
    try:
        logger.exception("未捕获异常 | %s %s | %s: %s",
                         request.method, request.url.path, type(exc).__name__, exc)
    finally:
        reset_request_id(token)
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, CODE_SERVER_ERROR,
                           _SERVER_ERROR_MSG, request_id=rid)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RAGBaseError, _handle_rag_base_error)
    app.add_exception_handler(Exception, _handle_uncaught_exception)
