"""edge-to-cloud 路由(M15):简单请求走本地小模型,复杂请求转云端大模型。

机器人/边缘场景的典型分工——"反射放本地、思考放云端":本地小模型省成本、低延迟、
数据不出域;只有复杂请求才动用云端大模型。这里在**软件层**模拟这种分工(无需真硬件)。

`RoutingLLM` 本身实现 `LLM` 协议(generate(system, user) -> str),所以它是个即插即用的
LLM——可以原样塞进 RagPipeline、planner、orchestrator 的子 agent 等任何用 LLM 的地方,
上层无感。它每次路由都发 span(route + llm:<tier>),于是 M4 的可观测性自然延伸到"模型分层"
这一维:配合 orchestrator/agent 的 span,你能看到 orchestrator → subagent → route → 哪一档
的完整 trace 树。

延续 Protocol + 可换实现:
- `RoutePolicy` 协议:`LengthRoutePolicy`(启发式、零依赖、确定性)/ 可换 LLM 路由等。
"""

from __future__ import annotations

from typing import Protocol

from .llm import LLM
from .tracing import NullTracer, Tracer


class RoutePolicy(Protocol):
    """决定一个请求走 edge 还是 cloud(返回 "edge" | "cloud")。"""

    def route(self, *, system: str, user: str) -> str: ...


class LengthRoutePolicy:
    """启发式:提示短 → edge(本地小模型够用);提示长/复杂 → cloud(云端大模型)。

    零依赖、确定性。真实系统可换成基于难度分类的 LLM 路由,但接口不变。
    """

    def __init__(self, *, max_edge_chars: int = 200) -> None:
        self._max = max_edge_chars

    def route(self, *, system: str, user: str) -> str:
        return "edge" if len(user) <= self._max else "cloud"


class RoutingLLM:
    """实现 LLM 协议:按 policy 在 edge / cloud 两个 LLM 间路由,并发 span。"""

    def __init__(
        self,
        *,
        edge: LLM,
        cloud: LLM,
        policy: RoutePolicy | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._edge = edge
        self._cloud = cloud
        self._policy = policy if policy is not None else LengthRoutePolicy()
        self._tracer = tracer if tracer is not None else NullTracer()
        self.last_tier: str | None = None  # 审计/测试:最近一次走了哪一档

    def generate(self, *, system: str, user: str) -> str:
        with self._tracer.span("route", input={"chars": len(user)}) as span:
            tier = self._policy.route(system=system, user=user)
            span.update(output=tier)
        self.last_tier = tier
        llm = self._edge if tier == "edge" else self._cloud
        with self._tracer.span(f"llm:{tier}", input={"system": system, "user": user}) as span:
            text = llm.generate(system=system, user=user)
            span.update(output=text)
        return text
