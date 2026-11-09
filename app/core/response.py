"""全局统一响应体。

成功：{"code": 0, "msg": "ok", "data": <业务数据>, "request_id": "..."}
失败：由 app/core/handlers.py 构造，字段一致，code 为错误码（见 error_codes.py）。

用泛型而不是 data: Any，是为了让 /docs 里每个接口的 data 类型仍然可见：
    response_model=ApiResponse[UploadResponse]
"""
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

from utils.logger import get_request_id

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """全局统一返回结构体"""

    code: int = 0
    msg: str = "ok"
    data: Optional[T] = None
    request_id: str = "-"


def ok(data=None, msg: str = "ok") -> ApiResponse:
    """构造成功响应；request_id 自动取当前请求上下文"""
    return ApiResponse(code=0, msg=msg, data=data, request_id=get_request_id())
