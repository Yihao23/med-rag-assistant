"""把文本转成向量(embedding)。

定义一个 `Embedder` 协议,提供两种实现:
- `HashingEmbedder`:零依赖、确定性、可离线——用于测试和"先把链路跑通"。
- `SentenceTransformerEmbedder`:真实语义向量(需要安装可选 extra `[embeddings]`)。

后续把它换成调用本地 embedding 服务,也只需新增一个实现类。
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """任何能把"一批文本"变成"一批向量"的对象。"""

    def embed(self, texts: list[str]) -> np.ndarray:
        """返回形状为 (len(texts), dim) 的 float32 数组。"""
        ...


class HashingEmbedder:
    """基于哈希的确定性向量。

    把每个词哈希到固定维度的桶里累加(词袋哈希)。语义能力很弱,但**零依赖、可离线、
    完全确定**,非常适合单元测试和验证整条链路是否连通。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                bucket = int(digest, 16) % self.dim
                vectors[i, bucket] += 1.0
        return vectors


class SentenceTransformerEmbedder:
    """真实语义向量,基于 sentence-transformers。需要 `pip install -e ".[embeddings]"`。"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 取决于是否装了可选依赖
            raise ImportError(
                "需要 sentence-transformers,请运行:pip install -e '.[embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, convert_to_numpy=True), dtype=np.float32)
