"""模型单例工厂：Embedding 与 LLM 统一从这里取，避免各服务重复读环境变量、重复建客户端。

配置来源：项目根目录 .env
- Embedding：QWEN_API_KEY / EMBEDDING_URL / EMBEDDING_MODEL
              现有向量库（vector_db/chroma_rag_db）是用 text-embedding-v3 建的，维度 1024，
              因此默认后端必须与之一致，否则写入/检索会因维度不匹配报错。
- LLM      ：DOUBAO_API_KEY / DOUBAO_ENDPOINT / DOUBAO_MODEL
              Ark 的 OpenAI 兼容端点；.env 里的 DOUBAO_ENDPOINT 常被写成带 /chat/completions
              的完整路径，这里会统一归一化成 base_url，避免客户端再拼一次导致 404。
"""

import os
from typing import Optional

from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger("app.services.model_factory")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def _normalize_base_url(raw: Optional[str], default: str) -> str:
    """把各式各样的端点写法归一化成 OpenAI SDK 需要的 base_url。

    - 去掉首尾空白与末尾斜杠
    - 去掉误加的 /chat/completions（SDK 自己会拼这一层）
    """
    url = (raw or default).strip().rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


class ModelFactory:
    """Embedding / LLM 的单例提供者（进程内只初始化一次）"""

    _embedding = None
    _llm = None

    @classmethod
    def get_embedding(cls):
        """Qwen text-embedding-v3（1024 维），与现有 Chroma 库保持一致"""
        if cls._embedding is None:
            from langchain_openai import OpenAIEmbeddings

            api_key = os.getenv("QWEN_API_KEY")
            if not api_key:
                raise RuntimeError("缺少 QWEN_API_KEY，无法初始化 Embedding 模型（检查项目根 .env）")
            base_url = _normalize_base_url(os.getenv("EMBEDDING_URL"), DEFAULT_EMBEDDING_BASE_URL)
            model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
            cls._embedding = OpenAIEmbeddings(
                api_key=api_key,
                base_url=base_url,
                model=model,
                # 关闭本地分词长度预校验：DashScope 侧按自己的 token 规则校验
                check_embedding_ctx_length=False,
                # DashScope 的 text-embedding-v3 单次请求最多 10 条 input，
                # 超过报 "batch size is invalid, it should not be larger than 10"。
                # 这里必须显式限制，否则任何一个 >10 分片的文档都会整单入库失败。
                chunk_size=batch_size,
            )
            # 只在首次加载时打：维度不一致、模型名没生效这类问题，靠这行才能定位
            logger.info(
                "Embedding 初始化 | model=%s base_url=%s batch=%d",
                model, base_url, batch_size,
            )
        return cls._embedding

    @classmethod
    def get_llm(cls, temperature: float = 0.0, streaming: bool = False):
        """豆包（Ark OpenAI 兼容端点）Chat 模型"""
        # streaming=True 会重建客户端，只有首次才记日志，否则每次流式问答都刷一行
        first = cls._llm is None
        if first or streaming:
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("DOUBAO_API_KEY")
            if not api_key:
                raise RuntimeError("缺少 DOUBAO_API_KEY，无法初始化生成模型（检查项目根 .env）")
            model = os.getenv("DOUBAO_MODEL")
            if not model:
                raise RuntimeError("缺少 DOUBAO_MODEL（Ark 上的推理接入点 ID）")
            base_url = _normalize_base_url(os.getenv("DOUBAO_ENDPOINT"), DEFAULT_LLM_BASE_URL)
            cls._llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                streaming=streaming,
                timeout=60,
                max_retries=2,
            )
            if first:
                logger.info("LLM 初始化 | model=%s base_url=%s", model, base_url)
        return cls._llm
