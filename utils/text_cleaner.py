"""文本清洗流水线：三档预设策略，由调用方（上传接口）显式选择。

三档的差别只有一个问题：**愿意为「去噪」牺牲多少「排版」**。

| 策略   | 缩进 | 连续空格 / 表格对齐 | 段落空行 | 去重 | 中文规范化 |
|--------|------|--------------------|---------|------|-----------|
| full   | 删   | 合并               | 删      | 是   | 全量      |
| safe   | 保留 | 保留               | 保留 1 个 | 是 | 全量      |
| raw    | 保留 | 保留               | 保留    | 否   | 仅去不可见字符 |

- `full`：适合普通正文（docx 正文、PDF 文本层），也是历史默认行为；
- `safe`：markdown、含缩进的文档、靠空格对齐的表格——清洗不会破坏结构；
- `raw`：代码块、含公式/ASCII 图表的文档——只移除不可见字符，一个字节都不多改。

清洗策略会写进分片 metadata 的 `clean_policy` 字段，并参与 `mode=skip` 的幂等判断：
同一份文件换了策略重传，必须重新入库，不能因为「内容没变」就跳过。
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

from langchain_core.documents import Document

from .logger import get_logger

logger = get_logger("utils.text_cleaner")

# 对外暴露的策略名（顺序即文档顺序，router 的 Literal 与此保持一致）
CLEAN_POLICIES: Tuple[str, ...] = ("full", "safe", "raw")


@dataclass(frozen=True)
class _Flags:
    """单个策略展开后的开关集合。"""

    normalize_spaces: bool = False      # 全角空格→半角 + 连续空格合并
    collapse_symbol_runs: bool = False  # ---- / ==== 压缩成 3 个
    full2half: bool = False             # 全角数字/字母转半角
    normalize_punct: bool = False       # 中文语境下的英文标点转全角
    insert_cjk_space: bool = False      # 中英文之间插空格
    dedupe_lines: bool = False          # 相邻重复行去重
    strip_lines: bool = False           # 逐行 strip（会毁掉缩进）
    collapse_blank_lines: bool = False  # 连续空行压缩


_PRESETS = {
    "full": _Flags(
        normalize_spaces=True,
        collapse_symbol_runs=True,
        full2half=True,
        normalize_punct=True,
        insert_cjk_space=True,
        dedupe_lines=True,
        strip_lines=True,
        collapse_blank_lines=True,
    ),
    "safe": _Flags(
        normalize_spaces=False,
        collapse_symbol_runs=False,
        full2half=True,
        normalize_punct=True,
        insert_cjk_space=True,
        dedupe_lines=True,
        strip_lines=False,
        collapse_blank_lines=True,
    ),
    "raw": _Flags(),
}


class TextCleaner:
    """文档清洗流水线（策略化版本）。

    :param policy:            full | safe | raw，见模块顶部对照表
    :param min_line_length:   仅 full 生效：strip 后长度小于该值的行被丢弃
    :param normalize_metadata: 是否补 standardized 元数据字段
    """

    def __init__(
        self,
        policy: str = "full",
        min_line_length: int = 1,
        normalize_metadata: bool = True,
    ):
        if policy not in _PRESETS:
            # 配置写错属开发者错误：裸抛，落 500 让堆栈说话（router 侧已用 Literal 拦住用户输入）
            raise ValueError(f"未知清洗策略 {policy!r}，可选：{list(CLEAN_POLICIES)}")
        self.policy = policy
        self.flags = _PRESETS[policy]
        self.min_line_length = min_line_length
        self.normalize_metadata = normalize_metadata

    # ------------------------------------------------------------ 原子动作

    @staticmethod
    def _strip_invisible(text: str) -> str:
        """移除不可见字符：控制符（保留 \\n\\r\\t）、零宽字符、BOM。

        这一条所有策略都执行——它不改任何可见内容，只删掉不该存在的字节。
        """
        control_chars = ''.join([chr(c) for c in range(0, 32) if chr(c) not in '\n\r\t'])
        text = re.sub(f'[{re.escape(control_chars)}\x7f]', '', text)
        return re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        """全角空格转半角，连续多个空格合并为一个。

        危险动作：会毁掉用空格对齐的表格、ASCII 图表、代码缩进。
        """
        text = text.replace('\u3000', ' ')
        return re.sub(r' +', ' ', text)

    @staticmethod
    def _collapse_symbol_runs(text: str) -> str:
        """连续符号压缩到 3 个。

        危险动作：markdown 表格分隔线 `| -------- |` 会被压短（结构还在，但列宽信息丢失）。
        """
        return re.sub(r'([-·*=])\1{3,}', r'\1'*3, text)

    @staticmethod
    def _full2half(text: str) -> str:
        """全角数字/字母转半角（中文文档高频噪声，如 `Ｖ２.０`）。"""
        result = []
        for char in text:
            code = ord(char)
            if 65281 <= code <= 65374:
                result.append(chr(code - 65248))
            elif code == 12288:
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    @staticmethod
    def _normalize_punct(text: str) -> str:
        """中文语境下的英文标点转全角（避开数字小数点：只在中文相邻时替换）。"""
        punct_map = {',': '，', '?': '？', '!': '！', ':': '：', ';': '；'}

        def replace_punct(match):
            return punct_map.get(match.group(0), match.group(0))

        text = re.sub(r'(?<=[\u4e00-\u9fa5])([,.!?;:])', replace_punct, text)
        return re.sub(r'([,.!?;:])(?=[\u4e00-\u9fa5])', replace_punct, text)

    @staticmethod
    def _insert_cjk_space(text: str) -> str:
        """中文与英文/数字之间插入空格（提升分词与检索效果）。"""
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', text)
        return re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', text)

    @staticmethod
    def _deduplicate_lines(text: str) -> str:
        """相邻重复行去重：按 strip 后内容比较，输出保留原始行（不破坏缩进）。"""
        lines = text.split('\n')
        result = []
        prev_content = ''
        for line in lines:
            current_content = line.strip()
            if current_content and current_content == prev_content:
                continue
            result.append(line)
            prev_content = current_content
        return '\n'.join(result)

    @staticmethod
    def _filter_blank_lines(text: str, min_len: int, strip_lines: bool) -> str:
        """空行处理。

        - strip_lines=True（full）：逐行 strip，并丢弃 strip 后长度 < min_len 的行。
          min_len=1 时空行会被全部删掉，段落分隔也随之消失——这是 full 的既有行为。
        - strip_lines=False（safe）：行内容原样保留，只把连续空行压成一个，
          并去掉首尾空行；缩进与行内空格完全不动。
        """
        raw_lines = text.split('\n')

        if strip_lines:
            lines = [line.strip() for line in raw_lines]
            valid = [line for line in lines if len(line) >= min_len]
        else:
            valid = raw_lines

        result = []
        prev_empty = False
        for line in valid:
            if not line.strip():
                if not prev_empty:
                    result.append('')
                    prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        joined = '\n'.join(result)
        # 保留缩进就不能用 .strip()（会削掉首行缩进），只去首尾换行
        return joined.strip() if strip_lines else joined.strip('\n')

    # ------------------------------------------------------------ 入口

    def clean_single_text(self, text: str) -> str:
        """单段文本清洗总入口。"""
        f = self.flags
        text = self._strip_invisible(text)

        if f.normalize_spaces:
            text = self._normalize_spaces(text)
        if f.collapse_symbol_runs:
            text = self._collapse_symbol_runs(text)

        if f.dedupe_lines:
            text = self._deduplicate_lines(text)

        if f.full2half:
            text = self._full2half(text)
        if f.normalize_punct:
            text = self._normalize_punct(text)
        if f.insert_cjk_space:
            text = self._insert_cjk_space(text)
            if f.normalize_spaces:  # 插入空格不会产生连续空格，但保持与 full 的历史行为一致
                text = re.sub(r' +', ' ', text)

        if f.strip_lines or f.collapse_blank_lines:
            text = self._filter_blank_lines(text, self.min_line_length, f.strip_lines)
        return text

    def clean_documents(self, docs: List[Document]) -> List[Document]:
        """批量清洗 Document 列表，保留元数据，仅清洗正文。

        清洗后正文为空的 Document 会被丢弃，但必须计数告警：
        「解析成功却少了几页」如果不记录，就又是一条静默丢内容的通道。
        """
        cleaned_docs = []
        dropped_sources = []
        for doc in docs:
            cleaned_content = self.clean_single_text(doc.page_content)
            if not cleaned_content.strip():
                meta = doc.metadata or {}
                dropped_sources.append(
                    f"{meta.get('source', '?')}"
                    + (f" 第{meta['page_number']}页" if meta.get("page_number") else "")
                )
                continue

            new_metadata = doc.metadata.copy()
            if self.normalize_metadata:
                new_metadata['source'] = new_metadata.get('source', '')
                new_metadata['file_type'] = new_metadata.get('file_type', 'unknown').lower()
                new_metadata['cleaned'] = True
                # 参与 mode=skip 的幂等判断：换策略重传必须重新入库
                new_metadata['clean_policy'] = self.policy

            cleaned_docs.append(Document(page_content=cleaned_content, metadata=new_metadata))

        if dropped_sources:
            logger.warning(
                "清洗后内容为空，已丢弃 %d 个片段：%s",
                len(dropped_sources), "; ".join(dropped_sources[:10]),
            )
        return cleaned_docs


# 本地测试：同一段文本走三档策略，观察差异
if __name__ == "__main__":
    test_text = """
    　　企业员工手册　　版本V２.０\u200b\u200b
    ========================================

    　　第一章　入职须知　
    　　第一章　入职须知　

    １、入职当日需携带身份证原件,银行卡复印件各１份。
        缩进两级的内容，靠空格表达层级
    | 字段 | 说明 |
    | ---- | ---- |
"""

    print("原始文本：")
    print(repr(test_text))
    for name in CLEAN_POLICIES:
        print(f"\n[{name}]")
        print(repr(TextCleaner(name).clean_single_text(test_text)))
