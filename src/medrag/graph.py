"""GraphRAG:基于实体图谱的检索(M6)。

与稠密(向量相似)/稀疏(词频)不同,GraphRAG 先从每个块抽取实体,建一张
"实体 ↔ 块"与"实体共现"的图;检索时从 query 的实体出发沿图遍历,把**直接提及**
以及**经共享实体相连(多跳)**的块都召回——能找到不含 query 原词、却语义相关的块,
这是 dense / BM25 都做不到的。

延续 Protocol + 可换实现:
- `EntityExtractor` 协议:`KeywordEntityExtractor`(默认、零依赖、确定性)/
  `LLMEntityExtractor`(真实,用一个 LLM 抽实体)。
- `GraphRetriever` 实现 add/search/__len__,可直接当 RagPipeline 的检索后端。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Protocol

from .chunking import Chunk
from .llm import LLM
from .store import SearchResult


class EntityExtractor(Protocol):
    """从一段文本里抽取实体(返回去重、小写的实体串)。"""

    def extract(self, text: str) -> list[str]: ...


_WORD_RE = re.compile(r"[a-z0-9]{2,}")  # ASCII 词(长度≥2)
_CJK_RE = re.compile(r"[一-鿿]+")  # 连续 CJK 串


class KeywordEntityExtractor:
    """零依赖、确定性的实体抽取(无需 LLM,适合默认/CI)。

    规则:ASCII 词整体成实体;短 CJK 串(≤4 字,如"胸痛""阿司匹林")整体成实体;
    长 CJK 串(整句)切成相邻二元组,避免把一整句当成一个"实体"。
    这是启发式近似——真实系统会用 NER 或 LLM(见 LLMEntityExtractor)。
    """

    def __init__(self, *, max_cjk_entity: int = 4) -> None:
        self._max = max_cjk_entity

    def extract(self, text: str) -> list[str]:
        text = text.lower()
        ents: set[str] = set(_WORD_RE.findall(text))
        for run in _CJK_RE.findall(text):
            if len(run) <= self._max:
                ents.add(run)
            else:
                for i in range(len(run) - 1):
                    ents.add(run[i : i + 2])
        return sorted(ents)


_EXTRACT_SYSTEM = (
    "从下面的医疗文本中抽取关键实体(疾病、药物、检查、症状等)。"
    "只输出实体,用逗号分隔,不要输出任何其它内容。"
)


class LLMEntityExtractor:
    """用一个 LLM 抽实体。质量远好于关键词法,但需要模型(在线或自托管)。"""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def extract(self, text: str) -> list[str]:
        raw = self._llm.generate(system=_EXTRACT_SYSTEM, user=text)
        parts = re.split(r"[,，、\n]", raw)
        return sorted({p.strip().lower() for p in parts if p.strip()})


class GraphRetriever:
    """基于实体图谱的检索器。

    add() 时:抽取每个块的实体,建立"实体→块"倒排,以及"实体共现"边
    (同一块里的实体两两相连)。
    search() 时:抽取 query 的实体,从它们出发做 `hops` 跳遍历;块的得分 =
    Σ 命中实体的权重(query 实体权重高,多跳相连的实体权重低),取 top-k。
    """

    def __init__(
        self,
        *,
        extractor: EntityExtractor | None = None,
        hops: int = 1,
        hop_decay: float = 0.3,
    ) -> None:
        self._extractor = extractor or KeywordEntityExtractor()
        self._hops = hops
        self._hop_decay = hop_decay  # 每多一跳,实体贡献的权重衰减
        self._chunks: list[Chunk] = []
        self._entity_chunks: dict[str, set[int]] = defaultdict(set)
        self._neighbors: dict[str, set[str]] = defaultdict(set)

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            idx = len(self._chunks)
            self._chunks.append(chunk)
            ents = self._extractor.extract(chunk.text)
            for e in ents:
                self._entity_chunks[e].add(idx)
            for e in ents:
                self._neighbors[e].update(o for o in ents if o != e)

    def search(self, query: str, *, k: int = 4) -> list[SearchResult]:
        if not self._chunks:
            return []
        q_ents = set(self._extractor.extract(query))
        # 按"距 query 实体的跳数"给每个触达实体一个权重(越远越低)。
        weight: dict[str, float] = {e: 1.0 for e in q_ents}
        frontier = set(q_ents)
        for hop in range(1, self._hops + 1):
            nxt: set[str] = set()
            for e in frontier:
                nxt |= self._neighbors.get(e, set())
            nxt -= weight.keys()
            for e in nxt:
                weight[e] = self._hop_decay**hop
            frontier = nxt

        scores: dict[int, float] = defaultdict(float)
        for entity, w in weight.items():
            for idx in self._entity_chunks.get(entity, ()):
                scores[idx] += w
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [SearchResult(chunk=self._chunks[i], score=s) for i, s in ranked]

    def __len__(self) -> int:
        return len(self._chunks)
