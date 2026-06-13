"""任务规划(M12):把一个目标拆成多步计划,逐步执行,失败时重规划。

与 M11 的 agent 循环互补——两种经典 agent 形态:
- agent.py(reactive):每一步都问模型"下一步干嘛",边走边想。灵活,但每步一次模型调用。
- planner.py(plan-then-execute):先一次性拆出整个计划,executor 照单执行;
  执行中某步失败,才回炉重规划(replan)。可审计、可预算、模型调用少。

延续 Protocol + 可换实现:
- `Planner` 协议:`plan(goal, tools, failed=None) -> Plan`(failed 非空即重规划)。
- `ScriptedPlanner`:按剧本出计划(CI/测试,零 API)。
- `LLMPlanner`:让 LLM 产出 JSON 计划并解析(复用 M1 的 `LLM` 协议,echo/anthropic/vllm 皆可)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .llm import LLM
from .tools import Tool
from .tracing import NullTracer, Tracer


@dataclass
class Step:
    """计划中的一步:调哪个工具、传什么参数,以及为什么(给人/模型看的说明)。"""

    description: str
    tool: str
    args: dict = field(default_factory=dict)


@dataclass
class Plan:
    goal: str
    steps: list[Step]


@dataclass
class StepResult:
    step: Step
    output: str
    ok: bool


@dataclass
class ExecutionReport:
    """一次完整执行的审计记录:最终答案 + 每步结果 + 重规划次数。"""

    goal: str
    answer: str
    results: list[StepResult]
    replans: int
    succeeded: bool


class Planner(Protocol):
    """任何能把目标拆成计划的对象。failed 非空时表示"带着失败教训重规划"。"""

    def plan(
        self, goal: str, *, tools: list[Tool], failed: list[StepResult] | None = None
    ) -> Plan: ...


class ScriptedPlanner:
    """假 planner:依次返回预设计划。第 1 个给首次 plan,之后每次 replan 取下一个。"""

    def __init__(self, plans: list[Plan]) -> None:
        self._plans = list(plans)
        self.replan_calls: list[list[StepResult]] = []  # 记录每次重规划收到的失败信息

    def plan(self, goal: str, *, tools: list[Tool], failed: list[StepResult] | None = None) -> Plan:
        if failed is not None:
            self.replan_calls.append(failed)
        if not self._plans:
            return Plan(goal=goal, steps=[])
        return self._plans.pop(0)


PLANNER_SYSTEM = (
    "你是一个任务规划器。把【目标】拆解成有序的工具调用步骤。"
    "只输出一个 JSON 数组,每个元素形如 "
    '{"description": "这一步做什么", "tool": "工具名", "args": {…}}。'
    "只能使用【可用工具】里列出的工具,不要输出 JSON 以外的任何内容。"
)


class LLMPlanner:
    """真 planner:让 LLM 产出 JSON 计划。复用 M1 的 LLM 协议(echo/anthropic/vllm 皆可)。"""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def plan(self, goal: str, *, tools: list[Tool], failed: list[StepResult] | None = None) -> Plan:
        tool_lines = "\n".join(
            f"- {t.name}: {t.description} 参数 schema:{t.parameters}" for t in tools
        )
        user = f"【目标】\n{goal}\n\n【可用工具】\n{tool_lines}"
        if failed:
            lessons = "\n".join(
                f"- 第 {i + 1} 步「{r.step.description}」失败:{r.output}"
                for i, r in enumerate(failed)
            )
            user += f"\n\n【上次执行失败,请调整计划】\n{lessons}"
        raw = self._llm.generate(system=PLANNER_SYSTEM, user=user)
        return Plan(goal=goal, steps=self._parse(raw))

    @staticmethod
    def _parse(raw: str) -> list[Step]:
        """从模型输出里抠出第一个 JSON 数组并解析;解析失败抛 ValueError。"""
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError(f"planner 输出中找不到 JSON 数组:{raw[:200]}")
        items = json.loads(match.group(0))
        return [
            Step(
                description=item.get("description", ""),
                tool=item["tool"],
                args=item.get("args", {}),
            )
            for item in items
        ]


SYNTH_SYSTEM = (
    "你是一个严谨的医疗文档助手。根据【目标】和下面各步骤的执行结果,"
    "给出最终回答;信息不足就明确说明,不要编造。标注信息来源(文件名)。"
)


class PlanExecutor:
    """照计划执行;某步失败 → 带失败信息重规划(至多 max_replans 次)。

    所有步骤完成后,用 llm 把各步结果汇总成最终答案(EchoLLM 亦可,测试友好)。
    """

    def __init__(
        self,
        planner: Planner,
        tools: list[Tool],
        llm: LLM,
        *,
        max_replans: int = 2,
        tracer: Tracer | None = None,
    ) -> None:
        self._planner = planner
        self._tools = {t.name: t for t in tools}
        self._llm = llm
        self._max_replans = max_replans
        self._tracer = tracer if tracer is not None else NullTracer()

    def run(self, goal: str) -> ExecutionReport:
        tools = list(self._tools.values())
        replans = 0
        failed: list[StepResult] | None = None
        with self._tracer.span("plan_execute", input={"goal": goal}) as root:
            while True:
                with self._tracer.span("plan", input={"replans": replans}) as span:
                    plan = self._planner.plan(goal, tools=tools, failed=failed)
                    span.update(output=[s.description for s in plan.steps])

                results = self._execute_steps(plan)
                if all(r.ok for r in results):
                    answer = self._synthesize(goal, results)
                    root.update(output=answer)
                    return ExecutionReport(
                        goal=goal, answer=answer, results=results, replans=replans, succeeded=True
                    )

                if replans >= self._max_replans:  # 重规划额度耗尽,如实报告失败
                    answer = "(多次规划后仍有步骤失败,无法完成目标)"
                    root.update(output=answer, metadata={"failed": True})
                    return ExecutionReport(
                        goal=goal, answer=answer, results=results, replans=replans, succeeded=False
                    )
                failed = [r for r in results if not r.ok]
                replans += 1

    def _execute_steps(self, plan: Plan) -> list[StepResult]:
        results = []
        for step in plan.steps:
            with self._tracer.span(f"step:{step.tool}", input=step.args) as span:
                tool = self._tools.get(step.tool)
                if tool is None:
                    result = StepResult(step, f"错误:工具 {step.tool} 不存在", ok=False)
                else:
                    try:
                        result = StepResult(step, tool.run(step.args), ok=True)
                    except Exception as exc:
                        result = StepResult(step, f"错误:工具执行失败:{exc}", ok=False)
                span.update(output=result.output, metadata={"ok": result.ok})
            results.append(result)
        return results

    def _synthesize(self, goal: str, results: list[StepResult]) -> str:
        parts = [
            f"【步骤 {i + 1}:{r.step.description}】\n{r.output}" for i, r in enumerate(results)
        ]
        user = f"【目标】\n{goal}\n\n" + "\n\n".join(parts)
        return self._llm.generate(system=SYNTH_SYSTEM, user=user)
