from medrag.chunking import Chunk
from medrag.embeddings import HashingEmbedder
from medrag.store import QdrantVectorStore


def test_qdrant_ranks_relevant_chunk_first():
    store = QdrantVectorStore(HashingEmbedder())
    store.add(
        [
            Chunk(text="心肌梗死 STEMI 胸痛 心电图", source="cardio.txt"),
            Chunk(text="糖尿病 血糖 胰岛素 控制", source="diabetes.txt"),
        ]
    )
    results = store.search("胸痛 心电图", k=1)
    assert len(results) == 1
    assert results[0].chunk.source == "cardio.txt"


def test_qdrant_len_tracks_added_chunks():
    store = QdrantVectorStore(HashingEmbedder())
    assert len(store) == 0
    store.add([Chunk(text="心肌梗死 STEMI 胸痛", source="cardio.txt")])
    assert len(store) == 1
