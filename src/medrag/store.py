"""内存向量库:用余弦相似度做最近邻检索。

M1 用最简单的"全量遍历 + numpy 点积"。文档量小完全够用;M2 会换成 Qdrant。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    """任何能存文本块并根据查询返回相关块的对象。"""

    def add(self, chunks: list[Chunk]) -> None:
        """把切块写入库中。"""

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]:
        """根据查询返回相关块,按相关度从高到低排序。"""

    def __len__(self) -> int: ...


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


class QdrantVectorStore:
    """M2 里基于 Qdrant 的向量库实现。这里先占位,不实现了。"""

    def __init__(
        self,
        embedder: Embedder,
        *,
        location: str = ":memory:",
        collection: str = "medrag",
    ) -> None:
        from qdrant_client import QdrantClient  # 需要安装 qdrant-client 包
        from qdrant_client.models import Distance, VectorParams

        self._embedder = embedder
        self._collection = collection
        self._client = QdrantClient(location)

        dim = int(self._embedder.embed(["dim probe"]).shape[1])
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._n = 0

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        from qdrant_client.models import PointStruct

        vectors = self._embedder.embed([c.text for c in chunks])
        points = [
            PointStruct(
                id=self._n + i,
                vector=vectors[i].tolist(),
                payload={"text": chunk.text, "source": chunk.source},
            )
            for i, chunk in enumerate(chunks)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        self._n += len(chunks)

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]:
        query_vector = self._embedder.embed([query])[0].tolist()
        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
        ).points
        return [
            SearchResult(
                chunk=Chunk(text=h.payload["text"], source=h.payload["source"]),
                score=float(h.score),
            )
            for h in hits
        ]

    def __len__(self) -> int:
        return self._n
