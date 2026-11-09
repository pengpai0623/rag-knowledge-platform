"""全局业务异常体系。

层次（与全局异常 handler 一一对应）：
1. RequestValidationError          FastAPI/Pydantic 请求体校验失败      → HTTP 200 + code 422
2. StarletteHTTPException           路由层 404/405 及显式 HTTPException  → 保留对应 HTTP 码
3. RAGBaseError 及其子类            业务异常，code 由 raise 处给出        → HTTP 200 + 业务码
4. Exception                        未捕获异常兜底                       → HTTP 500

约定：
- **code 一律在构造异常实例时显式传入**（取值见 error_codes.py）：异常类只表达语义，
  不绑定码，因此同一个类可以在不同上下文给出不同的码，不存在「类名与码不一致」的二义性；
  构造点通常就是 `raise XxxError(...)`，少数是先构造、再由上层 `raise wrapped from e`
  （见 `IndexService._translate_chroma`）；
- `code` 是 keyword-only 必填参数，漏传直接 TypeError，
  不允许静默降级成 400（那种失败最难排查）；
- 异常类只声明语义，**不携带 HTTP 状态**：HTTP 状态是输出层的关注点，
  由 handlers.py 按异常类型统一决定（业务异常与校验失败一律 200，见架构文档 §5）；
- code 是对外契约，一旦发布只增不改、不复用；
- msg 面向用户：语义上可解释的细节（页码、文件名）可以带，
  内部实现（路径、堆栈、第三方报文）只进日志，绝不放进响应体；
- data 可携带结构化附加信息（如 422 的逐字段错误清单、失败的页码）。

什么时候抛什么（架构文档 §2.1）：
- 外部来的数据不合法 → 业务异常
- 第三方库报错       → raise 业务异常(...) from e
- 自己代码写错       → 裸抛，不要封装（封装会把 bug 掩盖成预期内的业务失败）

用法：
    raise KbNotFoundError(f"知识库不存在：{knowledge_base}", code=ERR_KB_NOT_FOUND)
    raise DocumentParseError(f"第 {n} 页解析失败", code=ERR_PARSE_FAILED, page=n)
"""
from typing import Any, Optional


class RAGBaseError(Exception):
    """业务异常基类：所有可预期、需要映射为用户可读响应的错误都继承它。

    :param msg:  面向用户的说明（可带页码、文件名等上下文）
    :param code: 业务错误码，必填，取值见 error_codes.py
    :param data: 结构化附加信息，原样放进响应体 data 字段
    """

    def __init__(self, msg: str = "", *, code: int, data: Any = None):
        super().__init__(msg)
        self.msg = msg or self.__class__.__name__
        self.code = code
        self.data = data


class ParamError(RAGBaseError):
    """请求参数不合法（业务校验，非 Pydantic 字段校验）。

    通用场景用 code=CODE_ERROR（400）；某一类参数问题需要前端区分时，
    在 4xxx 段分配专码后直接在 raise 处换掉即可。
    """


class KbNotFoundError(RAGBaseError):
    """知识库不存在。raise 时传 code=ERR_KB_NOT_FOUND（2001）。"""


class KbStorageError(RAGBaseError):
    """向量库（Chroma）存储层故障，一类的细分码由 raise 处选择（见 error_codes.py 2xxx）：

    - 2002 拿不到客户端（依赖缺失 / 目录无权 / 库文件损坏）
    - 2003 向量维度与库内已有数据不一致（换了 embedding 模型）
    - 2004 向量索引异常，需重建知识库
    - 2005 磁盘空间不足
    - 2006 向量库版本与当前服务不兼容

    只翻译「调用方有补救动作」的故障；InvalidUUIDError / InvalidArgumentError /
    DuplicateIDError 这类属于代码传参写错或并发写重，裸抛即可（→ 500），
    用业务码会把 bug 掩盖成预期内的失败。第三方异常必须 `raise ... from e` 保留因果链，
    msg 面向用户，原始报文只进日志。
    """


class GenerateError(RAGBaseError):
    """模型调用失败。raise 时传 code=ERR_GENERATE_FAILED（3001）。

    本期 3001 是兜底码，细分（网络/超时/限流/鉴权/上下文超长/内容拦截/解析失败）
    见 docs/global_exception_architecture.md §6.4。落地时只需在 raise 处换成
    3002~3008 中对应的码，异常类与全局 handler 都不用动。
    """


class UnsupportedFileTypeError(RAGBaseError):
    """上传了不支持的文件格式。raise 时传 code=ERR_PARSE_UNSUPPORTED（1008）。"""


class EmptyContentError(RAGBaseError):
    """文档解析后没有可入库的内容（如全空白页）。raise 时传 code=ERR_PARSE_EMPTY_CONTENT（1006）。"""


class DocumentParseError(RAGBaseError):
    """文档解析失败（文件内容/结构问题）。code 由 raise 处按具体原因选择 1xxx 码。

    :param message: 面向用户的完整说明，已包含页码等上下文
    :param code:    1xxx 段的具体码（1000~1008，见 error_codes.py）
    :param page:    失败页码（从 1 开始），无页概念的文件为 None
    """

    def __init__(self, message: str, *, code: int, page: Optional[int] = None):
        super().__init__(message, code=code, data={"page": page} if page is not None else None)
        self.page = page


class XlsxStyleBrokenError(DocumentParseError):
    """Excel 样式表损坏且自动修复无效。raise 时传 code=ERR_PARSE_XLSX_BROKEN（1007）。

    继承 DocumentParseError：既可被 `except DocumentParseError` 一网打尽，
    也能按本类精确捕获，单独给前端定制提示。
    """
