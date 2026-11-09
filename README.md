# RAG 知识平台

一个基于 **FastAPI + LangChain + Chroma** 的轻量 RAG（检索增强生成）知识库服务：
上传文档 → 解析清洗分片 → 向量入库 → 检索 → 交给大模型作答并返回引用来源。

支持 PDF / Word / Excel / PowerPoint / Markdown / TXT，PDF 自带「逐页判定 + 扫描页 OCR」降级链路。

---

## 功能特性

- **多格式文档解析**：`.pdf` `.docx` `.xlsx` `.pptx` `.md` `.txt`，统一输出 LangChain `Document`
- **PDF 智能降级**：文本层优先，乱码页用 pdfplumber 复核，扫描页走 OCR，空白页跳过；解析失败整单不入库
- **文本清洗 + 分片**：去噪声、中文规范化、去重；按文档结构切分（docx 按标题聚合成小节、PDF 按页、pptx 按幻灯片），超长再递归切分（chunk 800 / overlap 100），无结构文档退回纯递归切分
- **多知识库**：一个 Chroma collection 即一个知识库，接口用 `knowledge_base` 参数隔离
- **两种检索策略**：`similarity`（余弦相似度 + 阈值过滤，默认）/ `mmr`（多样性优先）
- **答案带引用**：返回去重后的来源列表（文件名 + 页码），支持 SSE 流式输出
- **可运维**：统一日志（控制台 + 轮转文件）、`request_id` 全链路串联、按来源删除向量

---

## 技术栈

| 分层 | 选型 |
|---|---|
| Web 服务 | FastAPI + Uvicorn |
| RAG 编排 | LangChain (LCEL) |
| 向量库 | Chroma（本地持久化） |
| Embedding | 阿里云 DashScope `text-embedding-v3`（1024 维） |
| 对话模型 | 火山方舟 Ark（豆包），OpenAI 兼容端点 |
| 文档解析 | PyMuPDF / pdfplumber / python-docx / openpyxl / python-pptx |
| OCR | RapidOCR（ONNX Runtime，PP-OCR 模型） |

---

## 目录结构

```
rag-knowledge-platform/
├── app/                     # FastAPI 应用
│   ├── main.py              # 入口：日志、中间件、路由注册、/health
│   ├── core/                # 全局异常体系
│   │   ├── error_codes.py       # 错误码表（分段规则见文件头）
│   │   ├── exceptions.py        # RAGBaseError 及业务子类
│   │   ├── response.py          # ApiResponse 统一响应体
│   │   └── handlers.py          # 四层异常 handler
│   ├── routers/rag_router.py    # 上传 / 删除 / 知识库列表 / 问答
│   ├── schemas/rag_schema.py    # Pydantic 请求响应模型
│   └── services/
│       ├── parse_service.py     # 解析 → 清洗 → 分片
│       ├── index_service.py     # 向量入库、库列表、按来源删除
│       ├── retrieve_service.py  # similarity / mmr 召回
│       ├── generate_service.py  # Prompt 拼装、LLM 生成、来源格式化
│       └── model_factory.py     # Embedding / LLM 单例，读 .env
├── utils/                   # 通用能力
│   ├── universal_parser.py  # 统一解析入口（含 PDF 降级链）
│   ├── pdf_parser/          # pdf_probe 判定 + PyMuPDF / pdfplumber 解析
│   ├── docx_parser.py  xlsx_parser.py  pptx_parser.py
│   ├── ocr.py               # RapidOCR 兜底
│   ├── text_cleaner.py      # 清洗流水线
│   └── logger.py            # 日志 / request_id / 耗时埋点
├── examples/                # 学习用脚本（Day2–Day13，可独立运行）
├── docs/                    # 设计文档（解析策略 / 全局异常架构）与测试样本（docs/rag）
├── vector_db/chroma_rag_db/ # Chroma 持久化目录（git 忽略）
├── demo.py                  # 单文件解析调试脚本
└── requirements.txt
```

---

## 快速开始

### 1. 环境准备

Python 3.10+，建议使用虚拟环境：

```bash
python -m venv ai-env
# Windows
ai-env\Scripts\activate
# macOS / Linux
source ai-env/bin/activate

pip install -r requirements.txt
```

> 注意：Windows 上不要安装 `paddlepaddle 2.6.2`，它会把 `protobuf` 钉在 `<=3.20.2`，
> 与 `chromadb` / `pymilvus` 要求的 `>=6.33.5` 冲突。OCR 已改用 RapidOCR。

### 2. 配置 `.env`

在项目根目录创建 `.env`：

```ini
# Embedding（DashScope）
QWEN_API_KEY=sk-xxx
EMBEDDING_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3

# 对话模型（火山方舟 Ark）
DOUBAO_API_KEY=xxx
DOUBAO_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_MODEL=ep-xxxxxxxx        # Ark 上的推理接入点 ID
```

可选环境变量：

```ini
LOG_LEVEL=INFO          # DEBUG / INFO / WARNING / ERROR
LOG_TO_FILE=1           # 0 = 只输出控制台
MAX_UPLOAD_MB=50              # 上传大小上限，超出返回 413
EMBEDDING_BATCH_SIZE=10       # Embedding 单次请求条数（DashScope 上限 10，勿调大）

# PDF 页级判定（详见 docs/rag/doc_parser_fallback_design_v1.1.md）
PARSE_MIN_CHARS_PER_PAGE=30   # 进入乱码判定的字数门槛
PARSE_MIN_BODY_CHARS=20       # 正文区最少字符数（区分页眉页脚）
PARSE_MIN_TEXT_AREA=0.05      # 正文区文本覆盖面积下限
PARSE_DENSE_CHARS=300         # 全页字数达到即视为真实文本层
PARSE_GARBAGE_RATIO=0.30      # 乱码率上限
PARSE_SCAN_IMAGE_RATIO=0.70   # 图片覆盖率≥此值判扫描页
PARSE_FIGURE_IMAGE_RATIO=0.15 # 图片覆盖率≥此值判图表页，设 0 关闭
PARSE_MAX_OCR_PAGES=50        # 单文档 OCR 页数上限
OCR_DPI=200
```

### 3. 启动服务

```bash
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://127.0.0.1:8000/docs` 查看 Swagger 文档。

---

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| POST | `/api/rag/upload` | 上传文档并入库，参数 `knowledge_base`、`mode=replace\|skip`、`clean_policy=full\|safe\|raw` |
| GET | `/api/rag/knowledge-bases` | 列出知识库及其文档数、分片数 |
| GET | `/api/rag/collections` | 巡检：每个 collection 的 metadata 及内部 source 分布、解析版本（排障用，可选 `knowledge_base` 过滤） |
| DELETE | `/api/rag/documents` | 按 `knowledge_base` + `source` 删除该文档全部向量 |
| DELETE | `/api/rag/knowledge-bases/{name}` | 清空整个知识库（策略升级后整库重灌用） |
| POST | `/api/rag/chat` | 检索 + 生成，支持非流式与 SSE 流式 |

上传默认 `mode=replace`：同一来源（文件名）先写新分片再删旧分片，重复上传不会让向量累加；
`mode=skip` 则按文件 sha256 + `parser_version` + `clean_policy` 判重，库里已有同内容同策略时直接返回 `status=skipped`。

`clean_policy` 手动选择清洗强度（默认 `full`，即历史行为）：

| 策略 | 缩进 | 连续空格 / 表格对齐 | 段落空行 | 相邻行去重 | 中文规范化 | 适用 |
|---|---|---|---|---|---|---|
| `full` | 删 | 合并 | 删 | 是 | 全量 | 普通正文（docx 正文、PDF 文本层） |
| `safe` | 保留 | 保留 | 保留 1 个 | 是 | 全量 | markdown、含缩进或空格对齐的文档 |
| `raw` | 保留 | 保留 | 保留 | 否 | 仅去不可见字符 | 代码块、公式、ASCII 图表 |

策略会写进分片 metadata 的 `clean_policy`，并在响应体回显（传错位置时能立刻发现，`clean_policy` 必须放在 query string，放进 form 会被静默忽略）。换策略重传同一份文件时 `mode=skip` 不会误跳过。

### 示例

上传文档：

```bash
curl -X POST "http://127.0.0.1:8000/api/rag/upload?knowledge_base=demo" \
     -F "file=@docs/rag/UN38.3.pdf"
```

markdown 或含缩进/表格的文档，用 `safe` 保住排版：

```bash
curl -X POST "http://127.0.0.1:8000/api/rag/upload?knowledge_base=demo&clean_policy=safe" \
     -F "file=@docs/rag/doc_parser_fallback_design_v1.1.md"
```

问答（非流式）：

```bash
curl -X POST "http://127.0.0.1:8000/api/rag/chat" \
     -H "Content-Type: application/json" \
     -d '{"question":"这份报告讲了什么？","knowledge_base":"demo","top_k":4}'
```

流式请求把 `"stream": true` 打开即可，返回 SSE 事件：先 `sources`，再连续 `delta`，最后 `[DONE]`。

`ChatRequest` 主要字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `question` | — | 用户问题，必填 |
| `knowledge_base` | — | 目标知识库，必填 |
| `top_k` | 4 | 召回片段数，上限 20 |
| `stream` | false | 是否 SSE 流式返回 |
| `retrieval_mode` | `similarity` | 可选 `mmr` |
| `fetch_k` | top_k × 4 | 仅 mmr：候选池大小，上限 100 |
| `mmr_lambda` | 0.5 | 仅 mmr：0 偏多样性，1 偏相关性 |

## 统一响应体与异常体系

所有 JSON 接口（含成功响应）统一为：

```json
{"code": 0, "msg": "ok", "data": { ... }, "request_id": "e3caf8981cea"}
```

`request_id` 与响应头 `X-Request-ID`、日志里的 `rid=` 三方一致，可串联一次请求的全部日志。
SSE 流式接口是唯一例外：流式响应不包此结构体，流内错误以 `{"type":"error","code":...}` 事件下发（与 JSON 错误共用同一张码表）。

错误码分段（`app/core/error_codes.py`，一旦对外发布只增不改）：

| 段 | 含义 | 已启用 |
|---|---|---|
| `0` | 成功 | — |
| `4xx / 5xx` | Web 层，复用 HTTP 状态码（404 路由不存在、405 方法不支持、413 超限、422 参数校验失败、500 兜底） | ✓ |
| `1xxx` | 文档解析（1001 打不开 / 1002 加密 / 1004 OCR 超限 / 1005 全空白 / 1008 格式不支持 …） | ✓ |
| `2xxx` | 知识库与存储（2001 不存在 / 2002 库不可用 / 2003 维度不匹配 / 2004 索引异常 / 2005 磁盘满 / 2006 版本不兼容） | ✓ |
| `3xxx` | 生成与模型（3001 模型调用失败，细分计划见 `docs/global_exception_architecture.md` §6.4） | ✓ |
| `4xxx / 5xxx` | 预留（结构化输出解析 / 中间件与存储） | — |

异常按四层分发（`app/core/handlers.py`），**HTTP 状态只有两条线**：业务能解释的走 200 用 code 说话，
请求打错地方或服务崩了才用 4xx/5xx。

| 层 | 场景 | HTTP | body.code |
|---|---|---|---|
| 1 | 请求体校验失败，`data.errors` 带逐字段明细（field / type / message） | **200** | 422 |
| 2 | 路由不存在 / 方法不支持 / 上传超限 | 404 / 405 / 413 | 同 HTTP 码 |
| 3 | 业务异常 `RAGBaseError`（知识库不存在、解析失败、模型失败…） | **200** | 业务码 |
| 4 | 未捕获异常兜底，固定文案，堆栈只进日志 | 500 | 500 |

413 保留真实状态码，是因为反向代理超限时返回的是裸 413（没有我们的 body），前端需按状态码统一处理。

---

## 处理流程

```
上传 → 扩展名校验 → 分块落盘（>50MB 直接 413）
     → UniversalDocParser 解析
         PDF：探测（逐页算 正文区字数/覆盖率/图片覆盖率，不 OCR）
              → OCR 预算校验（超限立即失败）
              → 执行（文本页 / pdfplumber 复核 / 扫描页 OCR / 图表页 OCR / 空白跳过）
         Office：按段落、工作表、幻灯片拆分
     → TextCleaner 清洗 → RecursiveCharacterTextSplitter 分片
     → Chroma 入库（collection = knowledge_base，先写新分片再删同来源/同内容旧分片）

问答 → 校验知识库存在 → 召回 top_k 片段（similarity 阈值过滤 / mmr 重排）
     → Prompt 拼装（带 [1][2] 编号与来源标注）
     → LLM 生成答案 + 去重来源列表
```

解析后每个分片都带 `parser_version`、`file_sha256`、`parse_mode`、`layout_type`、`doc_parse_mode`，
可据此判断「这一片来自文本层还是 OCR」「是哪个版本解析器产出的」，便于定向重灌。

**错误约定**：所有错误经全局异常体系返回统一结构体。业务类错误（文档损坏 / 加密 / OCR 失败 / 知识库不存在 /
模型失败）一律 **HTTP 200 + 业务码**（1xxx / 2xxx / 3xxx）；参数校验失败 200 + code 422 + 逐字段明细；
路由 404、方法 405、上传超限 413、未捕获异常 500 保留真实 HTTP 状态。

异常抛出规范见 `docs/global_exception_architecture.md`：外部数据不合法 → 业务异常；第三方库报错 →
`raise XxxError(...) from e` 封装；自己代码写错 → 裸抛（落 500，不得封装成业务异常）。

---

## 测试与验收

```bash
# 页级判定单测（合成 PDF，不联网、不跑 OCR）
python tests/test_pdf_probe.py

# 全局异常四层分发与统一响应体（不联网；覆盖 200/404/405/413/500）
python tests/test_error_handling.py

# 召回验收：从库里抽样分片，用原文自召回，看 Recall@k / MRR
python scripts/eval_recall.py --kb demo --auto 20

# 或用手工测试集（JSON: [{"question": "...", "keywords": ["..."]}]）
python scripts/eval_recall.py --kb demo --test-set docs/rag/qa_set.json
```

> 验收看召回指标，不看 `chunk_count`——分片数只能证明「产出了向量」，
> 证明不了「OCR 文本能被检索到」。

---

## 常见问题

- **写入报维度不匹配**：现有 `vector_db/chroma_rag_db` 是 `text-embedding-v3`（1024 维）建的，
  换 Embedding 后端必须重建库。
- **`chromadb` 导入失败**：多为 `protobuf` 版本过低，需要 `>=6.33.5`。
- **扫描版 PDF 解析失败**：单文档 OCR 页数超过 `PARSE_MAX_OCR_PAGES`（默认 50），拆分后重试。
- **上传成功但搜不到**：检查召回阈值（默认相似度 0.5），必要时调低或改用 `mmr`。

---

## 参考

- `docs/rag/doc_parser_fallback_design.md` — 解析降级方案（文档级闸门 G1–G6、页级判定）
- `docs/rag/向量检索核心原理.md` — 检索原理笔记
- `examples/` — 从手写 RAG 到 LCEL、Chroma、Milvus 的逐步实现
