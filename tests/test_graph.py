from medrag.chunking import Chunk
from medrag.graph import GraphRetriever, KeywordEntityExtractor, LLMEntityExtractor


def test_keyword_extractor_treats_short_cjk_as_whole_entity():
    ents = KeywordEntityExtractor().extract("STEMI 阿司匹林 胸痛")
    assert "stemi" in ents
    assert "阿司匹林" in ents
    assert "胸痛" in ents


def test_graph_retrieves_indirect_chunk_via_shared_entity():
    # B 不含"阿司匹林",但经共享实体"抗血小板"与 A 相连 → GraphRAG 应召回它
    # (这正是 dense/BM25 做不到的多跳能力)。
    r = GraphRetriever()
    r.add(
        [
            Chunk(text="阿司匹林 抗血小板", source="A"),
            Chunk(text="抗血小板 出血风险", source="B"),
            Chunk(text="血糖 胰岛素", source="C"),
        ]
    )
    hits = r.search("阿司匹林", k=3)
    sources = [h.chunk.source for h in hits]

    assert sources[0] == "A"  # 直接命中,权重最高
    assert "B" in sources  # 多跳召回,虽不含 query 原词
    assert "C" not in sources  # 与 query 实体无连接,不召回


def test_graph_empty_returns_nothing():
    assert GraphRetriever().search("任意", k=3) == []


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate(self, *, system: str, user: str) -> str:
        return self._reply


def test_llm_entity_extractor_parses_separated_list():
    ex = LLMEntityExtractor(_FakeLLM("阿司匹林, 心肌梗死，STEMI"))
    assert set(ex.extract("任意文本")) == {"阿司匹林", "心肌梗死", "stemi"}
