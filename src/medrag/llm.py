"""LLM 接口与实现。

定义 `LLM` 协议,提供两种实现:
- `EchoLLM`:不调用任何模型,直接回显检索到的上下文——用于测试和离线验证链路。
- `AnthropicLLM`:M1 的真实实现,调用 Claude API。

**这是整个项目里最重要的一个抽象。** M3 要换成自托管 vLLM 时,只需新增一个
`VLLMClient` 实现 `LLM` 协议(vLLM 提供 OpenAI 兼容接口),pipeline 与其余代码无需改动。
"""

from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    """任何能"根据 system + user 提示生成一段文本"的对象。"""

    def generate(self, *, system: str, user: str) -> str: ...


class EchoLLM:
    """占位实现:不调用模型,只回显收到的上下文。用于无 API key 时验证检索是否正常。"""

    def generate(self, *, system: str, user: str) -> str:
        return "[EchoLLM] 未调用真实模型。以下是组装好的提示:\n\n" + user


class AnthropicLLM:
    """M1 真实实现:调用 Claude。

    模型默认 claude-opus-4-8,可用环境变量 MEDRAG_MODEL 覆盖。
    使用 adaptive thinking——医疗问答属于对准确性敏感的任务。
    """

    def __init__(self, model: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("需要 anthropic,请运行:pip install anthropic") from exc
        # 默认从环境读取 ANTHROPIC_API_KEY
        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("MEDRAG_MODEL", "claude-opus-4-8")

    def generate(self, *, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
