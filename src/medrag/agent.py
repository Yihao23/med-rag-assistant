"""Agent 循环(M11):让模型决定"调工具还是给答案"。

核心机制(tool-use loop):
    while True:
        turn = 模型看(系统提示 + 对话历史 + 工具清单)
        if turn 要调工具:本地执行 → 结果追加进历史 → 继续
        else:           turn 的文本就是最终答案 → 返回

与 RagPipeline 的本质区别:pipeline 的流程写死在代码里(必检索、检索一次);
agent 的流程由模型在运行时决定(可以不检索、检索多次、换不同 query 重试)。

延续 Protocol + 可换实现:
- `ToolLLM` 协议:任何能"看历史+工具清单,产出一步决策"的模型。
- `ScriptedToolLLM`:假实现,按预设剧本出牌——CI/测试用,零 API。
- `AnthropicToolLLM`:真实现,走 Claude 的 tool use API(lazy import)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .tools import Tool
from .tracing import NullTracer, Tracer


@dataclass
class ToolCall:
    """模型的一次工具调用请求。"""

    id: str
    name: str
    args: dict


@dataclass
class ModelTurn:
    """模型的一步输出:要么带 tool_calls(要调工具),要么 text 即最终答案。"""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ToolLLM(Protocol):
    """任何能基于(系统提示 + 历史 + 工具清单)产出一步决策的模型。"""

    def step(self, *, system: str, messages: list[dict], tools: list[Tool]) -> ModelTurn: ...


@dataclass
class AgentStep:
    """循环中执行过的一次工具调用(留作审计/调试)。"""

    tool: str
    args: dict
    result: str


@dataclass
class AgentAnswer:
    text: str
    steps: list[AgentStep]


AGENT_SYSTEM = (
    "你是一个严谨的医疗文档助手。回答前优先使用工具检索文档,"
    "只根据检索到的内容回答;查不到就明确说明,不要编造。"
    "在答案中标注信息来源(文件名)。"
)


class Agent:
    """tool-use 循环的执行器。max_steps 防止模型无限调工具。"""

    def __init__(
        self,
        llm: ToolLLM,
        tools: list[Tool],
        *,
        max_steps: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._max_steps = max_steps
        self._tracer = tracer if tracer is not None else NullTracer()

    def run(self, question: str) -> AgentAnswer:
        messages: list[dict] = [{"role": "user", "content": question}]
        steps: list[AgentStep] = []
        with self._tracer.span("agent.run", input={"question": question}) as root:
            for _ in range(self._max_steps):
                turn = self._llm.step(
                    system=AGENT_SYSTEM, messages=messages, tools=list(self._tools.values())
                )
                if not turn.tool_calls:  # 没有工具调用 → text 即最终答案
                    text = turn.text or ""
                    root.update(output=text)
                    return AgentAnswer(text=text, steps=steps)

                # 把模型这一步(含工具调用请求)记入历史,再逐个执行
                messages.append({"role": "assistant", "tool_calls": turn.tool_calls})
                for call in turn.tool_calls:
                    result = self._execute(call)
                    steps.append(AgentStep(tool=call.name, args=call.args, result=result))
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

            # 步数耗尽:防御性兜底,避免无限循环烧钱
            text = "(已达到最大工具调用步数,无法完成回答)"
            root.update(output=text, metadata={"max_steps_exceeded": True})
            return AgentAnswer(text=text, steps=steps)

    def _execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:  # 模型幻觉出不存在的工具:把错误喂回去让它自纠
            return f"错误:工具 {call.name} 不存在"
        with self._tracer.span(f"tool:{call.name}", input=call.args) as span:
            try:
                result = tool.run(call.args)
            except Exception as exc:  # 工具失败同样喂回模型,而不是让 agent 崩溃
                result = f"错误:工具执行失败:{exc}"
            span.update(output=result)
            return result


class ScriptedToolLLM:
    """假模型:按预设剧本依次出牌。测试 agent 循环用,零 API、完全确定。"""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.seen_messages: list[list[dict]] = []  # 记录每步收到的历史,供断言

    def step(self, *, system: str, messages: list[dict], tools: list[Tool]) -> ModelTurn:
        self.seen_messages.append([dict(m) for m in messages])
        if not self._turns:
            return ModelTurn(text="(剧本耗尽)")
        return self._turns.pop(0)


class AnthropicToolLLM:
    """真实现:Claude 的 tool use API。模型自行决定是否调用工具。"""

    def __init__(self, model: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要 anthropic,请运行:pip install anthropic") from exc
        import os

        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("MEDRAG_MODEL", "claude-opus-4-8")

    def step(self, *, system: str, messages: list[dict], tools: list[Tool]) -> ModelTurn:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=self._to_anthropic(messages),
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ],
        )
        calls = [
            ToolCall(id=b.id, name=b.name, args=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in response.content if b.type == "text") or None
        return ModelTurn(text=text, tool_calls=calls)

    @staticmethod
    def _to_anthropic(messages: list[dict]) -> list[dict]:
        """内部通用消息格式 → Anthropic API 格式。"""
        out: list[dict] = []
        for m in messages:
            if m["role"] == "assistant" and "tool_calls" in m:
                out.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                            for c in m["tool_calls"]
                        ],
                    }
                )
            elif m["role"] == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m["content"],
                            }
                        ],
                    }
                )
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out
