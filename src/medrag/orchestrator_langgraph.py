"""多 agent 编排的 LangGraph 实现(M13 第二阶段)。

与手写的 `orchestrator.Orchestrator` **对外接口完全相同**(handle(request) -> OrchestratorResult),
只是内部把"路由→委派"用 LangGraph 的 StateGraph 表达成一张图:

    START → router ──(按 state["agent"] 选择)──→ {retriever | summarizer | judge} → END

手写版让你看清机制(路由/委派/状态传递都是普通 Python);LangGraph 版则是 JD 点名的
"open-source orchestration framework" 实战——同样一套 Router/SubAgent,换个执行引擎即可,
这正是"接口隔离"的回报。复杂场景(条件边、循环、人工介入、断点续跑)正是 LangGraph
比手写循环更值的地方。

langgraph 是可选重依赖,故 lazy import;装好后 `pip install -e ".[langgraph]"`。
"""

from __future__ import annotations

from typing import TypedDict

from .orchestrator import OrchestratorResult, Router, SubAgent
from .tracing import NullTracer, Tracer


class _State(TypedDict):
    """在图节点间流动的状态。"""

    request: str
    agent: str  # router 节点写入:选中的专家名
    answer: str  # worker 节点写入:专家的回答


def _build_graph(router: Router, agents: list[SubAgent]):
    """把 Router + 子 agent 编译成一张 LangGraph 状态图。"""
    from langgraph.graph import END, START, StateGraph

    by_name = {a.name: a for a in agents}

    def route(state: _State) -> dict:
        name = router.route(state["request"], list(by_name.values()))
        return {"agent": name if name in by_name else agents[0].name}

    def make_worker(agent: SubAgent):
        def worker(state: _State) -> dict:
            return {"answer": agent.handle(state["request"])}

        return worker

    graph = StateGraph(_State)
    graph.add_node("router", route)
    for agent in agents:
        graph.add_node(agent.name, make_worker(agent))

    graph.add_edge(START, "router")
    # router 之后:按 state["agent"] 的值跳到对应的 worker 节点(只跑选中的那个)
    graph.add_conditional_edges("router", lambda s: s["agent"], {a.name: a.name for a in agents})
    for agent in agents:
        graph.add_edge(agent.name, END)
    return graph.compile()


class LangGraphOrchestrator:
    """与手写 Orchestrator 同构的 LangGraph 实现,可直接替换。"""

    def __init__(
        self, router: Router, agents: list[SubAgent], *, tracer: Tracer | None = None
    ) -> None:
        if not agents:
            raise ValueError("至少需要一个子 agent")
        self._graph = _build_graph(router, agents)
        self._first = agents[0].name
        self._tracer = tracer if tracer is not None else NullTracer()

    def handle(self, request: str) -> OrchestratorResult:
        with self._tracer.span("orchestrator.langgraph", input={"request": request}) as root:
            final = self._graph.invoke({"request": request, "agent": "", "answer": ""})
            result = OrchestratorResult(
                request=request,
                agent=final.get("agent") or self._first,
                answer=final.get("answer", ""),
            )
            root.update(output=result.answer, metadata={"routed_to": result.agent})
            return result
