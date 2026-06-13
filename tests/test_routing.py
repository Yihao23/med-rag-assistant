from contextlib import contextmanager

from medrag.embeddings import HashingEmbedder
from medrag.orchestrator import Orchestrator, ScriptedRouter, SummarizerAgent
from medrag.pipeline import RagPipeline
from medrag.routing import LengthRoutePolicy, RoutingLLM
from medrag.store import InMemoryVectorStore


class _TagLLM:
    """假 LLM:回显时带上标签,便于断言走了哪一档。"""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def generate(self, *, system: str, user: str) -> str:
        return f"[{self._tag}] {user[:16]}"


def _routing(**kw):
    return RoutingLLM(edge=_TagLLM("EDGE"), cloud=_TagLLM("CLOUD"), **kw)


# --- 路由策略 ---


def test_length_policy_routes_by_size():
    p = LengthRoutePolicy(max_edge_chars=10)
    assert p.route(system="s", user="short") == "edge"
    assert p.route(system="s", user="x" * 50) == "cloud"


def test_routing_picks_edge_for_short():
    r = _routing(policy=LengthRoutePolicy(max_edge_chars=10))
    assert r.generate(system="s", user="hi").startswith("[EDGE]")
    assert r.last_tier == "edge"


def test_routing_picks_cloud_for_long():
    r = _routing(policy=LengthRoutePolicy(max_edge_chars=10))
    out = r.generate(system="s", user="这是一个需要云端大模型处理的很长很复杂的问题描述")
    assert out.startswith("[CLOUD]")
    assert r.last_tier == "cloud"


# --- RoutingLLM 是即插即用的 LLM ---


def test_routing_is_drop_in_llm_for_pipeline():
    pipeline = RagPipeline(InMemoryVectorStore(HashingEmbedder()), _routing())
    pipeline.index({"cardio.txt": "STEMI 胸痛 心电图"})
    text = pipeline.answer("胸痛", k=1).text
    assert text.startswith("[EDGE]") or text.startswith("[CLOUD]")  # 当作普通 LLM 用


# --- 收官:M4 + M13 + M15 串起来的完整 trace 树 ---


class _RecordingTracer:
    """把 span 按进入顺序记下来,验证跨 orchestrator/agent/路由 的可观测性。"""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name, *, input=None):
        record = {"name": name, "output": None}
        self.spans.append(record)

        class _Span:
            def update(self_, *, output=None, metadata=None):
                record["output"] = output

        yield _Span()

    def flush(self) -> None:
        pass


def test_trace_spans_orchestrator_subagent_and_routing():
    tracer = _RecordingTracer()
    routing = _routing(policy=LengthRoutePolicy(max_edge_chars=5), tracer=tracer)
    orch = Orchestrator(ScriptedRouter(["summarizer"]), [SummarizerAgent(routing)], tracer=tracer)

    orch.handle("总结")  # 2 字 → edge

    names = [s["name"] for s in tracer.spans]
    # 同一个 tracer 串起三层:编排 → 子 agent → 模型路由 → 选中的档
    assert names == ["orchestrator", "subagent:summarizer", "route", "llm:edge"]
