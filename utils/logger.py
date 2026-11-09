"""全局日志配置：统一的 logger 工厂、轮转文件日志、请求上下文标记。

设计目标（方便后续埋点）：
1. 一处配置，全项目共用：所有模块 `get_logger(__name__)` 即可，不用各自 basicConfig；
2. 重复 import 不重复挂 handler（幂等），避免一条日志打三遍；
3. 控制台 + 文件双输出，文件按大小轮转，UTF-8 落盘（Windows 下不会因 cp936 崩）；
4. 可选 request_id：FastAPI 中间件后续可以 `set_request_id()`，日志自动带上，
   这样一次上传/问答的多层调用能串起来追踪；
5. `log_span` / `log_execution` 用于给「解析降级链」「检索」「生成」这类耗时环节埋点。

环境变量（都可选，不配就用默认值）：
- LOG_LEVEL   ：DEBUG / INFO / WARNING / ERROR，默认 INFO
- LOG_DIR     ：日志目录，默认 <项目根>/logs
- LOG_TO_FILE ：0 表示只输出到控制台（测试/CI 用），默认 1
- LOG_MAX_MB  ：单个日志文件上限（MB），默认 10
- LOG_BACKUP  ：保留的历史文件份数，默认 5
"""

import contextvars
import functools
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Callable, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGGER_NAME = "rag"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | rid=%(request_id)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_configured = False


class _RequestIdFilter(logging.Filter):
    """把当前请求标识注入每条日志记录，未设置时显示 '-'"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def set_request_id(request_id: Optional[str]) -> contextvars.Token:
    """绑定当前执行上下文（如一次 HTTP 请求）的追踪标识，返回可用于 reset 的 token"""
    return _request_id.set(request_id or "-")


def reset_request_id(token: contextvars.Token) -> None:
    """恢复上一层的请求标识"""
    _request_id.reset(token)


def get_request_id() -> str:
    """读取当前上下文的请求标识（异常 handler 构造响应体时用，兜底层也能取到）"""
    return _request_id.get()


def _resolve_level(level: Optional[str]) -> int:
    name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def _build_handlers(log_dir: str, to_file: bool, max_bytes: int, backup_count: int) -> list:
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers = []

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(_RequestIdFilter())
    handlers.append(console)

    if to_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "rag.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RequestIdFilter())
        handlers.append(file_handler)
    return handlers


def setup_logging(
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    to_file: Optional[bool] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> logging.Logger:
    """初始化全局 logger（幂等：重复调用不会重复挂 handler）"""
    global _configured

    logger = logging.getLogger(LOGGER_NAME)
    if _configured or logger.handlers:
        return logger

    log_dir = log_dir or os.getenv("LOG_DIR") or os.path.join(PROJECT_ROOT, "logs")
    if to_file is None:
        to_file = os.getenv("LOG_TO_FILE", "1") != "0"
    max_bytes = max_bytes or int(os.getenv("LOG_MAX_MB", "10")) * 1024 * 1024
    backup_count = backup_count if backup_count is not None else int(os.getenv("LOG_BACKUP", "5"))

    logger.setLevel(_resolve_level(level))
    for handler in _build_handlers(log_dir, to_file, max_bytes, backup_count):
        logger.addHandler(handler)
    logger.propagate = False  # 不再往 root 冒泡，避免第三方 basicConfig 造成重复输出

    _configured = True
    logger.debug("日志系统初始化完成：level=%s dir=%s to_file=%s", logger.level, log_dir, to_file)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取带全局配置的 logger；首次调用会自动完成初始化"""
    setup_logging()
    if not name:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


class log_span:
    """上下文管理器：记录一段代码的耗时（用于埋点解析/检索/生成等环节）

    用法：
        with log_span(logger, "pymupdf 抽页", page=3):
            ...
    异常时按 ERROR 记录并带上异常信息，不吞异常。
    """

    def __init__(self, logger: logging.Logger, action: str, level: int = logging.INFO, **fields):
        self.logger = logger
        self.action = action
        self.level = level
        self.fields = fields
        self._start = 0.0

    @staticmethod
    def _format_fields(fields: dict) -> str:
        return " ".join(f"{k}={v}" for k, v in fields.items())

    def __enter__(self):
        self._start = time.perf_counter()
        extra = self._format_fields(self.fields)
        self.logger.log(self.level, "开始 %s%s", self.action, f" | {extra}" if extra else "")
        return self

    def __exit__(self, exc_type, exc, tb):
        cost_ms = (time.perf_counter() - self._start) * 1000
        extra = self._format_fields(self.fields)
        suffix = f" | {extra}" if extra else ""
        if exc_type is None:
            self.logger.log(self.level, "完成 %s | %.1fms%s", self.action, cost_ms, suffix)
        else:
            self.logger.error("失败 %s | %.1fms | %s: %s%s", self.action, cost_ms,
                              exc_type.__name__, exc, suffix)
        return False


def log_execution(action: Optional[str] = None, level: int = logging.INFO):
    """装饰器版耗时埋点：@log_execution("解析 PDF")，可叠加在服务层方法上"""

    def decorator(func: Callable):
        name = action or getattr(func, "__qualname__", "call")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with log_span(logger, name, level=level):
                return func(*args, **kwargs)

        return wrapper

    return decorator
