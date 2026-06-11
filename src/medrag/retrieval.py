"""检索器抽象与混合检索(M6)。

把"根据 query 返回相关块"抽象成 `Retriever` 协议——已有的 `VectorStore`
(稠密/向量检索)在结构上就是一种 Retriever。本模块新增两路:

- `BM25Retriever`:稀疏/关键词检索,纯 Python、零依赖、确定性。擅长"查询里的
  确切词在文档里出现"(药名、缩写如 STEMI),补稠密向量"语义近但用词不同"之外的盲区。
- `HybridRetriever`:把稠密 + 稀疏两路用 RRF(Reciprocal Rank Fusion)融合。

HybridRetriever 同样实现 add/search/__len__,所以能直接当作 RagPipeline 的检索后端,
无需改动 pipeline。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .chunking import Chunk
from .store import SearchResult


class Retriever(Protocol):
    """任何能存文本块、并根据 query 返回相关块的对象。

    `VectorStore`(稠密)已结构性满足本协议;`BM25Retriever` / `HybridRetriever` 亦然。
    """

    def add(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]: ...

    def __len__(self) -> int: ...


# ASCII 按连续字母数字成词,CJK 按单字成词。零依赖、确定性。
# (中文理想上该用 jieba 分词,但那是重依赖;按字成词已能给出可用的词项重合度。)
_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 稀疏检索。纯 Python 实现,适合关键词/术语精确匹配。"""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._doc_tf: list[Counter] = []  # 每个块的词频
        self._doc_len: list[int] = []
        self._df: Counter = Counter()  # 文档频率:多少个块含某词
        self._avgdl: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            tf = Counter(tokens)
            self._chunks.append(chunk)
            self._doc_tf.append(tf)
            self._doc_len.append(len(tokens))
            for term in tf:
                self._df[term] += 1
        if self._doc_len:
            self._avgdl = sum(self._doc_len) / len(self._doc_len)

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        # 末尾 +1.0 是 Lucene 风格,保证 idf 恒为正(df > n/2 时也不会变负)。
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]:
        if not self._chunks:
            return []
        q_terms = _tokenize(query)
        scores: list[float] = []
        for i, tf in enumerate(self._doc_tf):
            dl = self._doc_len[i]
            norm = (
                self._k1 * (1 - self._b + self._b * dl / self._avgdl) if self._avgdl else self._k1
            )
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    s += self._idf(term) * (f * (self._k1 + 1)) / (f + norm)
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [SearchResult(chunk=self._chunks[i], score=scores[i]) for i in order]

    def __len__(self) -> int:
        return len(self._chunks)


_RRF_K = 60  # RRF 常数,抑制头部名次的过度主导;60 是原论文常用默认值。


class HybridRetriever:
    """稠密 + 稀疏的混合检索,用 RRF 融合两路排名。

    RRF(Reciprocal Rank Fusion):一个块的融合得分 = Σ 1/(K + 该块在各路中的名次)。
    它只看名次、不看原始分数,因此无需归一化两路异构的分数,鲁棒且基本无参(K 固定)。
    """

    def __init__(self, *, dense: Retriever, sparse: Retriever, candidates: int = 20) -> None:
        self._dense = dense
        self._sparse = sparse
        self._candidates = candidates  # 每路先取多少候选进入融合

    def add(self, chunks: list[Chunk]) -> None:
        self._dense.add(chunks)
        self._sparse.add(chunks)

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]:
        fused: dict[Chunk, float] = {}
        for hits in (
            self._dense.search(query, k=self._candidates),
            self._sparse.search(query, k=self._candidates),
        ):
            for rank, hit in enumerate(hits, start=1):
                fused[hit.chunk] = fused.get(hit.chunk, 0.0) + 1.0 / (_RRF_K + rank)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked]

    def __len__(self) -> int:
        return len(self._dense)
