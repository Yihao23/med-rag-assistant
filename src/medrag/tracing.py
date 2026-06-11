from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol


class Span(Protocol):
    """一个上下文管理器,进入时开始追踪,退出时结束追踪。"""

    def update(self, *, output=None, metadata=None) -> None: ...


class Tracer(Protocol):
    """一个追踪器,能创建 Span。"""

    def span(self, name, *, input=None) -> Span: ...

    def flush(self) -> None: ...


class _NullSpan:
    """一个不做任何事的 Span,用在没有 Tracer 的情况。"""

    def update(self, *, output=None, metadata=None) -> None:
        pass


class NullTracer:
    """返回一个不做任何事的 Tracer。"""

    @contextmanager
    def span(self, name, *, input=None) -> Iterator[Span]:
        yield _NullSpan()

    def flush(self) -> None:
        pass


class LangfuseTracer:
    """真实实现:把 span 转发给 Langfuse(Python SDK v4)。

    照 OpenAICompatLLM 的套路——支持注入一个假 client(client 参数),
    测试时无需真实 langfuse 包、也无需联网。host/key 默认从环境变量读:
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL。

    Langfuse v4 的 span 对象本身就有 update(output=, metadata=),
    恰好满足本模块的 Span 协议,所以适配器只是薄薄一层转发。
    """

    def __init__(
        self,
        *,
        public_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        client=None,
    ) -> None:
        if client is not None:
            self._client = client  # 注入假 client —— 测试用,不碰网络
        else:
            try:
                from langfuse import Langfuse
            except ImportError as exc:  # pragma: no cover
                raise ImportError("需要 langfuse,请运行:pip install langfuse") from exc
            self._client = Langfuse(
                public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
                secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
                base_url=base_url or os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000"),
            )

    @contextmanager
    def span(self, name, *, input=None) -> Iterator[Span]:
        with self._client.start_as_current_observation(
            as_type="span", name=name, input=input
        ) as span:
            yield span  # langfuse 的 span 已满足 Span 协议(有 update)

    def flush(self) -> None:
        """短生命周期进程(CLI/eval)退出前调用,确保缓冲的事件送达服务端。"""
        self._client.flush()
