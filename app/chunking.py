"""文本切分：先按句子切开，再贪心合并到 chunk_size，并保留 overlap。

为什么这样写（面试会问）：
- 按"定长字符"硬切会把一句话切断，检索出来的片段语义不完整；
- 按句子聚合能保证每个 chunk 语义自洽；
- overlap 是为了避免"答案正好落在两个 chunk 的接缝处"而检索不到。
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"[^。！？!?\n]+[。！？!?]?")
_WHITESPACE = re.compile(r"[ \t\u3000]+")

MIN_CHUNK_CHARS = 8


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """单句超长（比如整段没有标点）时的兜底硬切。"""
    step = max(1, size - overlap)
    pieces = []
    for i in range(0, len(text), step):
        piece = text[i : i + size]
        if piece.strip():
            pieces.append(piece)
    return pieces


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    text = _WHITESPACE.sub(" ", (text or "").replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text:
        return []

    chunk_size = max(50, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))

    sentences: list[str] = []
    for raw in _SENTENCE_SPLIT.findall(text):
        sentence = raw.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            sentences.extend(_hard_split(sentence, chunk_size, overlap))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) > chunk_size:
            chunks.append(buffer)
            # 重叠窗口：把上一块的尾部挪到下一块开头，避免答案落在接缝处被漏掉。
            # 注意要把尾部长度压到 chunk_size - len(sentence) 以内，
            # 否则"重叠"会让新块反过来超出 chunk_size（这是很容易踩的坑）。
            tail = buffer[-overlap:] if overlap else ""
            budget = chunk_size - len(sentence) - 1
            if budget <= 0:
                tail = ""
            elif len(tail) > budget:
                tail = tail[-budget:]
            buffer = f"{tail}\n{sentence}" if tail else sentence
        else:
            # 用换行拼接：中文句子之间不需要空格，但保留换行能让 Markdown
            # 标题/表格在 prompt 和引用片段里保持可读
            buffer = sentence if not buffer else f"{buffer}\n{sentence}"
    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]
def split_by_heading(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """优先按 Markdown 标题切分；某一段还是太长，再对它用按句子的方式切。"""
    parts = re.split(r"(?m)^(?=#{1,6}\s)", text)
    # ↑ 正则含义：在"行的开头且后面跟着 1~6 个 # 加一个空格"的位置切开
    #   (?m) 让 ^ 匹配每一行的开头而不是整个文本的开头
    #   (?=...) 是"零宽断言"：只作为切分点，不消耗字符（标题本身不会被切掉）
    chunks: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            chunks.append(part)                 # ← 这段不长，直接作为一段
        else:
            chunks.extend(split_text(part, chunk_size, overlap))   # ← 太长，交给原来的函数
    return chunks