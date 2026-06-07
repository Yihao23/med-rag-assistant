from medrag.chunking import Chunk
from medrag.embeddings import HashingEmbedder
from medrag.store import InMemoryVectorStore


def test_search_ranks_relevant_chunk_first():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add(
        [
            Chunk(text="心肌梗死 STEMI 胸痛 心电图", source="cardio.txt"),
            Chunk(text="糖尿病 血糖 胰岛素 控制", source="diabetes.txt"),
        ]
    )
    results = store.search("胸痛 心电图", k=1)
    assert len(results) == 1
    assert results[0].chunk.source == "cardio.txt"


def test_empty_store_returns_empty():
    store = InMemoryVectorStore(HashingEmbedder())
    assert store.search("任何问题") == []
    assert len(store) == 0
