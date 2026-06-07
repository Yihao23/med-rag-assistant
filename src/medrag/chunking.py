"""把长文档切成带重叠的文本块。

M1 用最朴素的"按字符滑窗 + 重叠"策略。后续 Milestone 可换成按句子 / 按 token 切分。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """一个文本块,记录来源文件名,便于后续在答案里引用出处。"""

    text: str
    source: str


def chunk_text(text: str, source: str, *, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """按字符切块。

    Args:
        text: 原始文本。
        source: 来源标识(文件名),写进每个 Chunk 方便溯源。
        size: 每块的目标字符数。
        overlap: 相邻块之间的重叠字符数,避免把一句话从中间切断。
    """
    if size <= 0:
        raise ValueError("size 必须为正数")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source))
        if start + size >= len(text):
            break
    return chunks
