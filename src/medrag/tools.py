"""工具层(M11):把系统能力包装成 LLM 可调用的工具。

agent 范式的第一块积木:一个 Tool = 名字 + 描述 + 参数 schema + run()。
描述和 schema 会发给模型,模型据此决定"调哪个工具、传什么参数";
run() 由 agent 循环在本地执行,结果再喂回模型。

注意视角转变:M1-M10 里检索是架构主角(pipeline 的固定一环);
这里它降级为"工具之一"——调不调、何时调,由模型在运行时决定。
"""

from __future__ import annotations

from typing import Protocol

from .retrieval import Retriever


class Tool(Protocol):
    """任何能被 LLM 调用的工具。name/description/parameters 发给模型,run 在本地执行。"""

    name: str
    description: str
    parameters: dict  # JSON Schema:模型按此构造参数

    def run(self, args: dict) -> str: ...


class SearchDocumentsTool:
    """把 M6 的检索器(dense/hybrid/graph 均可)包成 agent 工具。"""

    name = "search_documents"
    description = (
        "在医疗文档库中检索与查询相关的文本块。当需要病历、诊断、用药等文档内容来回答问题时使用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索查询(中文或英文)"},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever, *, k: int = 4) -> None:
        self._retriever = retriever
        self._k = k

    def run(self, args: dict) -> str:
        results = self._retriever.search(args["query"], k=self._k)
        if not results:
            return "(没有检索到相关内容)"
        parts = [f"[{r.chunk.source} | 相关度 {r.score:.3f}]\n{r.chunk.text}" for r in results]
        return "\n\n".join(parts)
