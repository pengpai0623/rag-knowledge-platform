"""全局错误码表（ApiResponse.code 的取值）。

分段规则（业务码一旦对外发布即成为契约，只增不改）：
- 0        成功
- 4xx/5xx  Web 层，复用 HTTP 状态码，语义与 HTTP 对齐
- 1xxx     文档解析类（DocumentParseError 及解析链路）
- 2xxx     知识库、检索与向量存储类
- 3xxx     生成 / 模型调用类
- 4xxx/5xxx（四位）预留：4xxx 结构化输出解析、5xxx 中间件/存储，
  与其他项目约定保持一致，启用时再扩充
"""

# ============================== Web 层 ==============================
CODE_OK = 0                    # 成功
CODE_ERROR = 400               # 通用请求错误（业务参数问题的缺省码）
CODE_NOT_FOUND = 404           # 路由不存在（与「资源不存在」的业务码区分）
CODE_METHOD_NOT_ALLOWED = 405  # 请求方法不支持
CODE_PAYLOAD_TOO_LARGE = 413   # 上传体积超限（保留 HTTP 语义，见架构文档 §5）
CODE_VALIDATE_ERROR = 422      # 请求体 Pydantic 校验失败
CODE_SERVER_ERROR = 500        # 未捕获异常兜底（对外固定文案，细节只进日志）

# ============================== 1xxx 文档解析 ==============================
ERR_PARSE_FAILED = 1000         # 解析失败（未细分的原因落这里）
ERR_PARSE_OPEN_FAILED = 1001    # PDF 无法打开（结构损坏）
ERR_PARSE_ENCRYPTED = 1002      # 文档已加密
ERR_PARSE_NO_PAGES = 1003       # 页数为 0
ERR_PARSE_OCR_LIMIT = 1004      # 需 OCR 页数超上限
ERR_PARSE_ALL_BLANK = 1005      # 全部为空白页
ERR_PARSE_EMPTY_CONTENT = 1006  # 解析后没有可入库的内容
ERR_PARSE_XLSX_BROKEN = 1007    # Excel 样式表损坏且修复无效
ERR_PARSE_UNSUPPORTED = 1008    # 不支持的文件格式

# ============================== 2xxx 知识库与检索 ==============================
# 2xxx 由 IndexService 产出：2001 是业务状态，2002~2006 是「存储层故障但用户可补救」的细分。
# 判定在异常类型上（见 IndexService._translate_chroma），识别不了的落 500。
ERR_KB_NOT_FOUND = 2001         # 知识库不存在（含并发删除的窗口期）
ERR_KB_STORAGE = 2002           # 向量库不可用：依赖缺失 / 目录无权 / 库文件损坏
ERR_KB_DIM_MISMATCH = 2003      # 向量维度与库内已有数据不一致（换了 embedding 模型）
ERR_KB_INDEX_BROKEN = 2004      # 向量索引异常，需重建知识库
ERR_KB_STORAGE_FULL = 2005      # 磁盘空间不足，写入失败
ERR_KB_VERSION_MISMATCH = 2006  # 向量库版本与当前服务不兼容（chromadb 升级后打开旧库）

# ============================== 3xxx 生成 / 模型 ==============================
# 3001 为兜底码，细分计划见 docs/global_exception_architecture.md §6.4
ERR_GENERATE_FAILED = 3001      # 模型调用失败（HTTP 200 + code 3001）

# 约定：码只在 raise 处引用（raise XxxError(msg, code=ERR_XXX)），
# 异常类不绑定码，这里也不维护任何 reason→code 映射表。
# 新增码时在此登记，并在 tests/test_error_codes.py 的类白名单中放行。
