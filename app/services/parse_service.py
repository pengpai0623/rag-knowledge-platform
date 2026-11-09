import os
import time
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from utils import CLEAN_POLICIES, StructuralSplitter, TextCleaner, UniversalDocParser
from utils.logger import get_logger
from utils.structural_splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

logger = get_logger("app.services.parse_service")


class ParseService:
    # 分片参数：上传链路与解析预览共用同一份默认值，避免两边切法不一致。
    # 800 而非 300——按小节聚合后一节通常 500~3000 字，300 的上限会让「按结构切」
    # 之后还要再切一遍，等于白做。
    DEFAULT_CHUNK_SIZE = DEFAULT_CHUNK_SIZE
    DEFAULT_CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP

    # 每个策略一个无状态 cleaner，避免每次上传都重新构造（也避免漏传策略时静默走默认）
    _cleaners: Dict[str, TextCleaner] = {p: TextCleaner(p) for p in CLEAN_POLICIES}

    @classmethod
    def get_cleaner(cls, clean_policy: str = "full") -> TextCleaner:
        """取清洗器——全项目唯一的构造入口。

        上传链路（parse_and_split）和解析预览接口都从这里拿实例，
        任何一方都不要自己 TextCleaner(...)：一旦出现第二个构造点，
        两条链路就可能各自演化成不同的清洗行为，导致「预览看到的」
        和「真正入库的」不是一回事。
        """
        cleaner = cls._cleaners.get(clean_policy)
        if cleaner is None:
            # router 已用 Literal 校验用户输入；能走到这里是代码写错，裸抛
            raise ValueError(f"未知清洗策略 {clean_policy!r}，可选：{list(CLEAN_POLICIES)}")
        return cleaner

    @classmethod
    def split_cleaned(
        cls,
        cleaned_docs: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Tuple[List[Document], str]:
        """对**清洗后**的文本做结构分片，返回 (分片列表, 结构策略)。

        同样是唯一入口：上传与预览都必须走这里。预览若自己另起一套切法，
        看到的分片就不等于真正入库的分片，调参数时会被误导。
        """
        return StructuralSplitter.split(
            cleaned_docs,
            cls.DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size,
            cls.DEFAULT_CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        )

    @classmethod
    def parse_and_split(
        cls,
        file_path: str,
        source_name: Optional[str] = None,
        clean_policy: str = "full",
    ) -> List[Document]:
        """解析文件 + 清洗 + 结构分片，返回可直接入库的分片列表。

        :param file_path:    磁盘路径（上传链路是临时文件）
        :param source_name:  原始文件名；上传必须传，否则 metadata.source 会变成 tmpXXXX.pdf
        :param clean_policy: 清洗策略 full | safe | raw，见 utils/text_cleaner.py 顶部对照表
        """
        cleaner = cls.get_cleaner(clean_policy)

        start = time.perf_counter()
        # 1. 解析原始文档
        raw_docs = UniversalDocParser.parse(file_path, source_name=source_name)
        # 2. 文本清洗（清洗后为空的片段会被丢弃并告警，不会静默消失）
        cleaned_docs = cleaner.clean_documents(raw_docs)
        # 3. 结构分片（标题聚合 / 按页 / 按幻灯片，超长再递归切）
        splits, strategy = cls.split_cleaned(cleaned_docs)

        # 这几个数量是本层唯一的可观测信号：cleaned < raw 说明有片段被洗成空；
        # strategy 说明这次走的是哪条结构路径（flat 表示没结构可用，退回纯递归）。
        # 清洗策略是否真的生效、结构识别有没有失败，只看这一行。
        logger.info(
            "清洗分片完成 | %s | policy=%s strategy=%s 解析=%d 清洗后=%d 分片=%d | %.1fms",
            source_name or os.path.basename(file_path), clean_policy, strategy,
            len(raw_docs), len(cleaned_docs), len(splits),
            (time.perf_counter() - start) * 1000,
        )
        return splits
