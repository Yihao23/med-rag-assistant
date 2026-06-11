import pytest

from medrag.embeddings import HashingEmbedder
from medrag.llm import EchoLLM
from medrag.pipeline import RagPipeline
from medrag.store import InMemoryVectorStore


def _client():
    # 没装 api extra(fastapi/httpx)时跳过,保持 CI 行为一致。
    testclient = pytest.importorskip("fastapi.testclient")
    from medrag.api import create_app

    pipeline = RagPipeline(InMemoryVectorStore(HashingEmbedder()), EchoLLM())
    pipeline.index({"cardio.txt": "STEMI 胸痛 心电图 阿司匹林"})
    return testclient.TestClient(create_app(pipeline))


def test_health_ok():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources():
    r = _client().post("/ask", json={"question": "胸痛 心电图", "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("[EchoLLM]")
    assert body["sources"][0]["source"] == "cardio.txt"


def test_ask_validates_body():
    # 缺 question → FastAPI/pydantic 返回 422
    r = _client().post("/ask", json={"k": 1})
    assert r.status_code == 422
