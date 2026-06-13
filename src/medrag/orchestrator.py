"""多 agent 编排(M13):一个主调度 agent 协调多个专职子 agent。

这里实现最经典的多 agent 形态——**supervisor / router 模式**:主 agent(Orchestrator)
不亲自干活,而是根据请求把任务**委派**给最合适的专家子 agent,再把结果带回。
对应 JD 的 "sub-agents seamlessly connect to the main conversational agent"。

与 M11/M12 的层级关系:
- M11 工具 = 确定性函数;M13 子 agent = 自身带 LLM 推理的自治单元(委派而非调用)。
- M12 planner 把任务拆成"步骤";M13 orchestrator 把任务分给"专家"。两者可叠加
  (planner 产步骤,每步交给某个子 agent),但这里先做纯路由,保持概念清晰。

延续 Protocol + 可换实现:
- `SubAgent` 协议:有 name/description,能 handle(task) -> str。
- `RetrieverAgent` / `SummarizerAgent` / `JudgeAgent`:三个 LLM 驱动的专家。
- `Router` 协议:`ScriptedRouter`(测试)/ `LLMRouter`(让 LLM 选专家)。

下一步(可选)可把 Orchestrator 换成 LangGraph 实现、保持同一对外接口——
那时这套手写版就是"理解机制"的基准参照。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .llm import LLM
from .pipeline import SYSTEM_PROMPT, build_prompt
from .retrieval import Retriever
from .tracing import NullTracer, Tracer


class SubAgent(Protocol):
    """一个专职子 agent:有名字和能力描述(供路由),能处理一个任务。"""

    name: str
    description: str

    def handle(self, task: str) -> str: ...


class RetrieverAgent:
    """检索专家:查文档 + 让 LLM 据此作答(一个自包含的迷你 RAG)。"""

    name = "retriever"
    description = "回答需要查阅医疗文档的问题:诊断、用药、检查结果等"

    def __init__(self, retriever: Retriever, llm: LLM, *, k: int = 4) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k

    def handle(self, task: str) -> str:
        results = self._retriever.search(task, k=self._k)
        return self._llm.generate(system=SYSTEM_PROMPT, user=build_prompt(task, results))


class SummarizerAgent:
    """摘要专家:把一段文本压缩成要点。"""

    name = "summarizer"
    description = "把一段较长的文本概括成简洁要点"

    _SYSTEM = "你是摘要助手。把下面的内容提炼成简洁、忠实于原文的要点,不要添加原文没有的信息。"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def handle(self, task: str) -> str:
        return self._llm.generate(system=self._SYSTEM, user=task)


class JudgeAgent:
    """质检专家:评估一段医疗回答的质量(复用 M4 LLM-as-judge 的思路)。"""

    name = "judge"
    description = "评估一段医疗回答是否准确、有依据,指出问题"

    _SYSTEM = "你是严格的医疗质检员。评估下面的回答是否准确、有据,简要指出问题或确认无误。"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def handle(self, task: str) -> str:
        return self._llm.generate(system=self._SYSTEM, user=task)


class Router(Protocol):
    """决定把请求交给哪个子 agent(返回 agent 名字)。"""

    def route(self, request: str, agents: list[SubAgent]) -> str: ...


class ScriptedRouter:
    """假路由:依次返回预设的 agent 名(测试用,完全确定)。"""

    def __init__(self, names: list[str]) -> None:
        self._names = list(names)

    def route(self, request: str, agents: list[SubAgent]) -> str:
        return self._names.pop(0) if self._names else agents[0].name


ROUTER_SYSTEM = (
    "你是一个调度器。根据【请求】和【可选专家】,选出最合适的一个专家。"
    "只输出该专家的名字(name),不要输出任何其它内容。"
)


class LLMRouter:
    """真路由:让 LLM 在专家名册里选一个。"""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def route(self, request: str, agents: list[SubAgent]) -> str:
        roster = "\n".join(f"- {a.name}: {a.description}" for a in agents)
        raw = self._llm.generate(
            system=ROUTER_SYSTEM, user=f"【请求】\n{request}\n\n【可选专家】\n{roster}"
        )
        for agent in agents:  # 容错:从模型输出里匹配第一个出现的合法专家名
            if agent.name in raw:
                return agent.name
        return agents[0].name  # 都没匹配上 → 兜底给第一个


@dataclass
class OrchestratorResult:
    request: str
    agent: str  # 实际处理请求的子 agent
    answer: str


class Orchestrator:
    """主调度 agent:路由 → 委派给选中的子 agent → 返回结果。"""

    def __init__(
        self, router: Router, agents: list[SubAgent], *, tracer: Tracer | None = None
    ) -> None:
        if not agents:
            raise ValueError("至少需要一个子 agent")
        self._router = router
        self._agents = {a.name: a for a in agents}
        self._tracer = tracer if tracer is not None else NullTracer()

    def handle(self, request: str) -> OrchestratorResult:
        agents = list(self._agents.values())
        with self._tracer.span("orchestrator", input={"request": request}) as root:
            name = self._router.route(request, agents)
            agent = self._agents.get(name, agents[0])  # 路由给了未知名字 → 兜底第一个
            with self._tracer.span(f"subagent:{agent.name}", input={"task": request}) as span:
                answer = agent.handle(request)
                span.update(output=answer)
            root.update(output=answer, metadata={"routed_to": agent.name})
            return OrchestratorResult(request=request, agent=agent.name, answer=answer)
