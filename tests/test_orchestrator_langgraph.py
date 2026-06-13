import pytest

pytest.importorskip("langgraph")  # 没装 langgraph extra 时整文件跳过

from medrag.chunking import Chunk  # noqa: E402
from medrag.embeddings import HashingEmbedder  # noqa: E402
from medrag.llm import EchoLLM  # noqa: E402
from medrag.orchestrator import (  # noqa: E402
    JudgeAgent,
    RetrieverAgent,
    ScriptedRouter,
    SummarizerAgent,
)
from medrag.orchestrator_langgraph import LangGraphOrchestrator  # noqa: E402
from medrag.store import InMemoryVectorStore  # noqa: E402


def _agents():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add([Chunk(text="STEMI 心肌梗死 阿司匹林", source="cardio.txt")])
    return [
        RetrieverAgent(store, EchoLLM(), k=1),
        SummarizerAgent(EchoLLM()),
        JudgeAgent(EchoLLM()),
    ]


def test_langgraph_orchestrator_routes_and_delegates():
    orch = LangGraphOrchestrator(ScriptedRouter(["summarizer"]), _agents())
    res = orch.handle("概括这段")
    assert res.agent == "summarizer"
    assert res.answer.startswith("[EchoLLM]")


def test_langgraph_routes_to_retriever_yields_doc_answer():
    orch = LangGraphOrchestrator(ScriptedRouter(["retriever"]), _agents())
    res = orch.handle("主要诊断?")
    assert res.agent == "retriever"
    assert "STEMI" in res.answer  # 经检索专家,文档内容进了答案


def test_langgraph_falls_back_on_unknown_route():
    orch = LangGraphOrchestrator(ScriptedRouter(["no_such_agent"]), _agents())
    assert orch.handle("q").agent == "retriever"  # 兜底第一个


def test_langgraph_requires_at_least_one_agent():
    with pytest.raises(ValueError):
        LangGraphOrchestrator(ScriptedRouter([]), [])
