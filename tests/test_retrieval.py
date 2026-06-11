from medrag.chunking import Chunk
from medrag.embeddings import HashingEmbedder
from medrag.retrieval import BM25Retriever, HybridRetriever
from medrag.store import InMemoryVectorStore, SearchResult


def _chunks():
    return [
        Chunk(text="STEMI 心肌梗死 阿司匹林 替格瑞洛", source="cardio.txt"),
        Chunk(text="血糖 胰岛素 糖尿病 二甲双胍", source="diabetes.txt"),
        Chunk(text="骨折 石膏 X 光 复位", source="fracture.txt"),
    ]


# --- BM25:稀疏/关键词检索 ---


def test_bm25_ranks_exact_keyword_first():
    r = BM25Retriever()
    r.add(_chunks())
    hits = r.search("阿司匹林", k=3)
    assert hits[0].chunk.source == "cardio.txt"


def test_bm25_empty_returns_nothing():
    assert BM25Retriever().search("任意", k=3) == []


# --- RRF 融合逻辑(用假检索器,断言融合规则本身)---


class _FakeRetriever:
    """返回预设排名的假检索器,用于精确断言 RRF 融合行为。"""

    def __init__(self, ranked_sources):
        self._ranked = ranked_sources

    def add(self, chunks):
        pass

    def search(self, query, *, k=4):
        return [SearchResult(chunk=Chunk(text=s, source=s), score=1.0) for s in self._ranked[:k]]

    def __len__(self):
        return len(self._ranked)


def test_rrf_rewards_agreement_across_retrievers():
    # A 在两路都排第一 → 融合后必为第一;结果是两路的并集去重。
    dense = _FakeRetriever(["A", "B", "C"])
    sparse = _FakeRetriever(["A", "C", "D"])
    hits = HybridRetriever(dense=dense, sparse=sparse, candidates=10).search("q", k=4)

    assert hits[0].chunk.source == "A"
    assert {h.chunk.source for h in hits} == {"A", "B", "C", "D"}
    # C 两路都靠前,应排在只命中一路的 B/D 之前
    sources = [h.chunk.source for h in hits]
    assert sources.index("C") < sources.index("B")


# --- HybridRetriever 是 pipeline 的即插即用后端 ---


def test_hybrid_is_drop_in_retriever():
    h = HybridRetriever(dense=InMemoryVectorStore(HashingEmbedder()), sparse=BM25Retriever())
    h.add(_chunks())

    assert len(h) == 3  # 实现了 __len__
    hits = h.search("胰岛素", k=2)
    assert all(isinstance(x, SearchResult) for x in hits)
    assert hits[0].chunk.source == "diabetes.txt"  # 两路都指向 diabetes
