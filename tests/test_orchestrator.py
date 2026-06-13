from medrag.chunking import Chunk
from medrag.embeddings import HashingEmbedder
from medrag.llm import EchoLLM
from medrag.orchestrator import (
    JudgeAgent,
    LLMRouter,
    Orchestrator,
    RetrieverAgent,
    ScriptedRouter,
    SummarizerAgent,
)
from medrag.store import InMemoryVectorStore


def _retriever_agent():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add([Chunk(text="STEMI 心肌梗死 阿司匹林", source="cardio.txt")])
    return RetrieverAgent(store, EchoLLM(), k=1)


def _agents():
    return [_retriever_agent(), SummarizerAgent(EchoLLM()), JudgeAgent(EchoLLM())]


# --- 子 agent ---


def test_retriever_agent_answers_from_docs():
    out = _retriever_agent().handle("心肌梗死")
    assert "cardio.txt" in out  # build_prompt 带上了来源
    assert "STEMI" in out


# --- LLMRouter:从模型输出里匹配专家名 ---


class _FixedLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate(self, *, system: str, user: str) -> str:
        return self._reply


def test_llm_router_picks_named_agent():
    router = LLMRouter(_FixedLLM("应该交给 retriever 专家"))
    assert router.route("查心梗用药", _agents()) == "retriever"


def test_llm_router_falls_back_when_no_match():
    # 模型没吐出任何合法专家名 → 兜底第一个(retriever)。
    router = LLMRouter(_FixedLLM("不知道"))
    assert router.route("q", _agents()) == "retriever"


# --- Orchestrator:路由 + 委派 ---


def test_orchestrator_delegates_to_routed_agent():
    orch = Orchestrator(ScriptedRouter(["summarizer"]), _agents())
    res = orch.handle("把这段话概括一下")
    assert res.agent == "summarizer"
    assert res.answer.startswith("[EchoLLM]")


def test_orchestrator_routes_different_requests_to_different_agents():
    agents = _agents()
    # 同一组 agent,不同路由决策 → 不同专家处理(证明是"调度",不是写死)。
    r1 = Orchestrator(ScriptedRouter(["retriever"]), agents).handle("主要诊断?")
    r2 = Orchestrator(ScriptedRouter(["judge"]), agents).handle("评估这段回答")
    assert r1.agent == "retriever"
    assert r2.agent == "judge"


def test_orchestrator_falls_back_on_unknown_route():
    orch = Orchestrator(ScriptedRouter(["no_such_agent"]), _agents())
    res = orch.handle("q")
    assert res.agent == "retriever"  # 未知名字 → 兜底第一个


def test_orchestrator_requires_at_least_one_agent():
    import pytest

    with pytest.raises(ValueError):
        Orchestrator(ScriptedRouter([]), [])
