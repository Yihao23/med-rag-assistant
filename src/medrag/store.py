"""内存向量库:用余弦相似度做最近邻检索。

M1 用最简单的"全量遍历 + numpy 点积"。文档量小完全够用;M2 会换成 Qdrant。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 归一化,使点积等价于余弦相似度;零向量保持为零。"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class InMemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        new_vectors = _normalize(self._embedder.embed([c.text for c in chunks]))
        self._chunks.extend(chunks)
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]:
        if self._vectors is None or not self._chunks:
            return []
        query_vec = _normalize(self._embedder.embed([query]))[0]
        scores = self._vectors @ query_vec  # 余弦相似度
        top_idx = np.argsort(scores)[::-1][:k]
        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._chunks)
