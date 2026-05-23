"""文本切分模块 —— 递归式分块，最大程度保留语义结构。

切分优先级（由粗到细）:
  段落 (\n\n) → 行 (\n) → 句子 (。！？.!) → 分句 (，,；;) → 字符

支持三种模式:
  - recursive:   通用文本递归切分（默认）
  - faq:         按一问一答对切分
  - tabular:     按表头+行数据切分
"""

import re
from typing import Any, Dict, List, Optional

# import moved inside _build_recursive_splitter to avoid scipy/numpy crash


# ── 递归切分（通用）──────────────────────────────────────

def _build_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int,
):
    """构造针对中文优化的递归文本切分器。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",       # 段落
            "\n",         # 行
            "。",         # 中文句号
            ".",          # 英文句号
            "！",         # 中文感叹号
            "!",          # 英文感叹号
            "？",         # 中文问号
            "?",          # 英文问号
            "；",         # 中文分号
            ";",          # 英文分号
            "，",         # 中文逗号
            ",",          # 英文逗号
            " ",          # 空格
            "",           # 字符
        ],
    )


# ── FAQ 切分 ────────────────────────────────────────────

# 常见问句开头标志
_FAQ_PATTERNS = re.compile(
    r"^(问[：:]|Q[：:]|问题[：:]|什么是|如何|怎样|为什么|怎么|"
    r"[A-Z][a-z]+\s+(is|are|can|do|does|has|have|will|would|could|should))",
    re.MULTILINE,
)


def _split_faq_pairs(text: str) -> List[str]:
    """将 FAQ 文本按一问一答切分为片段。"""
    # 尝试按问句标志切分
    matches = list(_FAQ_PATTERNS.finditer(text))
    if not matches:
        return [text]

    fragments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fragment = text[start:end].strip()
        if fragment:
            fragments.append(fragment)

    return fragments if fragments else [text]


# ── 表格数据切分 ─────────────────────────────────────────

def _split_tabular(text: str, chunk_size: int) -> List[str]:
    """将表格文本按"表头+数据行"为单位切分。"""
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return lines

    header = lines[0]
    data_lines = lines[1:]

    chunks = []
    current_chunk = header

    for line in data_lines:
        candidate = current_chunk + "\n" + line
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = header + "\n" + line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text]


# ── 统一入口 ─────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    mode: str = "recursive",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """将文本切分为带元数据的分块。

    Args:
        text:         待切分的文本。
        chunk_size:   每块最大字符数（默认 512）。
        chunk_overlap:相邻块重叠字符数（默认 64，约 12.5%）。
        mode:         切分模式 — "recursive"（通用）| "faq" | "tabular"。
        metadata:     附加到每个分块的元数据（如文件来源、页码等）。

    Returns:
        List[Dict[str, Any]]: 每项含 "content" 和 "metadata"。
    """
    if not text or not text.strip():
        return []

    meta = metadata or {}

    if mode == "faq":
        fragments = _split_faq_pairs(text)
        # FAQ 片段还可能很长，再用 recursive 兜底
        splitter = _build_recursive_splitter(chunk_size, chunk_overlap)
        chunks = []
        for frag in fragments:
            for doc in splitter.split_text(frag):
                chunks.append(doc)
    elif mode == "tabular":
        chunks = _split_tabular(text, chunk_size)
    else:
        splitter = _build_recursive_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_text(text)

    return [
        {"content": chunk, "metadata": {**meta, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]
