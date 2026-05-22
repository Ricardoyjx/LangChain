"""文本清洗模块 —— 生产级预处理管线。

处理流程（按顺序）:
  1. Unicode 标准化 (NFC)
  2. BOM / 零宽字符 / 控制字符 清理
  3. HTML 实体解码
  4. PDF 断字还原
  5. 空白字符归一化
  6. 行尾空白与段落规整
  7. 重复标点压缩
"""

import re
import unicodedata
from html import unescape

# ── 零宽 / 不可见字符 ───────────────────────────────────

_INVISIBLE_CHARS = re.compile(
    "["
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u2060"  # WORD JOINER
    "\u2061"  # FUNCTION APPLICATION
    "\u2062"  # INVISIBLE TIMES
    "\u2063"  # INVISIBLE SEPARATOR
    "\u2064"  # INVISIBLE PLUS
    "\ufeff"  # BOM / ZERO WIDTH NO-BREAK SPACE
    "]+"
)

# ── 控制字符保留清单 ────────────────────────────────────

# 保留: 水平制表符(\t), 换行(\n), 回车(\r)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── PDF 连字断行还原 ────────────────────────────────────

# 匹配 "word-\n" 或 "work‑\n"（含连字符/软连字符），还原为 "word"
_HYPHENATION = re.compile(r"(\w)[─\-]\s*\n\s*(\w)")


# ── 重复标点 ────────────────────────────────────────────

# 连续的重复标点压缩（保留省略号 ...）
_EXCESSIVE_PUNCT = re.compile(r"([。！？，、：；,\.!?:;]){3,}")
_EXCESSIVE_SPACES = re.compile(r"[ \t]{2,}")


# ── 行末空白 ────────────────────────────────────────────

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


# ── 连续空行 ────────────────────────────────────────────

_MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")


# ── 特殊 Unicode 字符替换 ───────────────────────────────

_UNICODE_QUOTES = {
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2012": "-",  # FIGURE DASH
    "\u2013": "–",  # EN DASH
    "\u2014": "—",  # EM DASH
    "\u00a0": " ",  # NO-BREAK SPACE
}


def _translate_unicode(text: str) -> str:
    """替换花哨引号、破折号等为 ASCII 等价物。"""
    return "".join(_UNICODE_QUOTES.get(c, c) for c in text)


# ── 公开 API ────────────────────────────────────────────


def clean_text(text: str) -> str:
    """对输入文本执行完整的生产级清洗管线。

    Args:
        text: 原始文本（通常来自 PDF / Word / Excel 解析器）。

    Returns:
        清洗后的纯文本。
    """
    if not text:
        return ""

    # 1. Unicode NFC 标准化
    text = unicodedata.normalize("NFC", text)

    # 2. 移除 BOM 与零宽字符
    text = _INVISIBLE_CHARS.sub("", text)

    # 3. 移除有害控制字符（保留 \t \n \r）
    text = _CONTROL_CHARS.sub("", text)

    # 4. Unicode 符号归一化（引号、短横等）
    text = _translate_unicode(text)

    # 5. HTML 实体解码（&amp; &lt; &gt; 等）
    text = unescape(text)

    # 6. PDF 断字还原（word-\\nword → wordword）
    text = _HYPHENATION.sub(r"\1\2", text)

    # 7. 去除行尾空白
    text = _TRAILING_WS.sub("", text)

    # 8. 规整每行首尾空白
    text = "\n".join(line.strip() for line in text.split("\n"))

    # 9. 连续空格/制表符压缩为单空格
    text = _EXCESSIVE_SPACES.sub(" ", text)

    # 10. 连续空行压缩为最多两个
    text = _MULTIPLE_BLANK_LINES.sub("\n\n", text)

    # 11. 重复标点压缩
    text = _EXCESSIVE_PUNCT.sub(r"\1\1", text)

    # 12. 首尾空白清理
    text = text.strip()

    return text
