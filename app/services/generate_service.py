"""生成层：把检索到的上下文喂给 LLM，产出答案与引用来源。

设计要点：
- Prompt 强约束「只依据上下文作答」，上下文为空时直接走兜底话术，不浪费一次模型调用；
- 上下文按 [1][2] 编号并标注来源，方便答案里追溯，也方便前端高亮；
- 来源去重后按召回顺序返回（同一文件的多个片段只保留一条）。
"""

import os
import time
from typing import Iterator, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.error_codes import CODE_ERROR
from app.core.exceptions import ParamError
from app.services.model_factory import ModelFactory
from utils.logger import get_logger

logger = get_logger("app.services.generate_service")

FALLBACK_ANSWER = "抱歉，知识库中没有找到相关内容，无法为您解答。"

RAG_PROMPT_TEMPLATE = """你是专业的知识库问答助手，必须严格基于以下提供的上下文内容回答用户问题。
【回答规则】
1. 答案只能来自下方的上下文内容，绝对不可以使用自身知识编造
2. 如果上下文里没有相关信息，请直接回答："抱歉，知识库中没有找到相关内容，无法为您解答。"
3. 回答要准确、简洁、条理清晰，保留关键术语
4. 不要提及"根据上下文"、"从资料中"这类表述

【上下文】
{context}

【用户问题】
{question}

【答案】
"""


class GenerateService:
    _prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    @staticmethod
    def _page_label(metadata) -> str:
        """页码范围：合并过的跨页分片显示「第5-7页」，否则「第5页」。

        只写 page_number（起始页）会把 3 页合一的片标成单页，引用出处就不实了。
        """
        start = metadata.get("page_start") or metadata.get("page_number")
        if not start:
            return ""
        end = metadata.get("page_end") or start
        return f"第{start}页" if end == start else f"第{start}-{end}页"

    @staticmethod
    def format_contexts(contexts: Sequence[Document]) -> str:
        """把召回片段拼成带编号与出处的上下文块"""
        blocks = []
        for idx, doc in enumerate(contexts, 1):
            source = os.path.basename(str(doc.metadata.get("source", "未知来源")))
            page = GenerateService._page_label(doc.metadata)
            label = f"{source} {page}" if page else source
            blocks.append(f"[{idx}]（来源：{label}）\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def format_sources(contexts: Sequence[Document]) -> List[str]:
        """来源去重：同一文件同一页只保留一条，保持召回顺序"""
        sources: List[str] = []
        for doc in contexts:
            source = str(doc.metadata.get("source", "未知来源"))
            page = GenerateService._page_label(doc.metadata)
            label = f"{source} {page}" if page else source
            if label not in sources:
                sources.append(label)
        return sources

    @classmethod
    def generate(cls, question: str, contexts: Sequence[Document]) -> Tuple[str, List[str]]:
        """返回 (答案, 来源列表)；上下文为空时不调用模型，直接兜底。"""
        question = (question or "").strip()
        if not question:
            raise ParamError("question 不能为空", code=CODE_ERROR)

        sources = cls.format_sources(contexts)
        if not contexts:
            return FALLBACK_ANSWER, sources

        chain = cls._prompt | ModelFactory.get_llm() | StrOutputParser()
        start = time.perf_counter()
        answer = (chain.invoke({
            "context": cls.format_contexts(contexts),
            "question": question,
        }) or "").strip()
        cost_ms = (time.perf_counter() - start) * 1000

        if not answer:
            # 模型有返回但内容为空：检索层看着一切正常，只有这里能发现
            logger.warning(
                "模型返回空答案，改为兜底话术 | 上下文=%d | %.1fms", len(contexts), cost_ms
            )
            return FALLBACK_ANSWER, sources
        logger.info(
            "生成完成 | 上下文=%d 答案=%d字 来源=%d | %.1fms",
            len(contexts), len(answer), len(sources), cost_ms,
        )
        return answer, sources

    @classmethod
    def generate_stream(cls, question: str, contexts: Sequence[Document]) -> Iterator[str]:
        """流式生成：逐段吐出答案文本；无上下文时只吐兜底话术。

        注意：question 的合法性由调用方在建流之前校验（get_contexts 已覆盖），
        这里再判一次只是防御——流已开始后抛异常无法转成错误响应。
        """
        question = (question or "").strip()
        if not question:
            raise ParamError("question 不能为空", code=CODE_ERROR)
        if not contexts:
            yield FALLBACK_ANSWER
            return

        chain = cls._prompt | ModelFactory.get_llm(streaming=True) | StrOutputParser()
        start = time.perf_counter()
        chars = 0
        for chunk in chain.stream({
            "context": cls.format_contexts(contexts),
            "question": question,
        }):
            if chunk:
                chars += len(chunk)
                yield chunk
        # 只在正常吐完时记：客户端中途断开是 GeneratorExit，不该算作生成完成
        logger.info(
            "流式生成完成 | 上下文=%d 字符=%d | %.1fms",
            len(contexts), chars, (time.perf_counter() - start) * 1000,
        )
