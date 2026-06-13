"""MCP server(M11.5):把 medrag 的能力暴露成标准 MCP 工具。

MCP(Model Context Protocol)是工具调用的跨厂商标准——本模块把已有能力注册成
MCP 工具后,任何 MCP 客户端(Claude Desktop / Claude Code / 其它 agent 框架)
都能即插即用本系统,无需任何定制胶水:

- search_documents : M6 的混合检索
- redact_text      : M7 的 PII 脱敏
- synthesize_record: M7 的合成病历生成

与 M11 的关系:agent.py 里的 Tool 协议是"进程内"的工具;MCP 把同样的概念
标准化成跨进程协议。设计仍是依赖注入:create_server(retriever=...) 可注入
假组件供测试,默认从 MEDRAG_* 环境变量装配(与 api.py 同一套约定)。

运行:python -m medrag.mcp_server   (stdio 传输,供 MCP 客户端拉起)
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from .pii import Redactor, RuleRedactor
from .retrieval import Retriever
from .synth import generate_record


def _default_retriever() -> Retriever:
    """按 MEDRAG_* 环境变量装配检索器并索引文档(与 api.py 同一约定)。"""
    from .chunking import chunk_text
    from .cli import _make_embedder, _make_retriever
    from .loader import load_directory

    retriever = _make_retriever(
        os.environ.get("MEDRAG_RETRIEVAL", "hybrid"),
        os.environ.get("MEDRAG_STORE", "memory"),
        _make_embedder(os.environ.get("MEDRAG_EMBEDDER", "hashing")),
    )
    data = Path(os.environ.get("MEDRAG_DATA", "data"))
    if data.is_dir():
        chunks = []
        for name, text in load_directory(data).items():
            chunks.extend(chunk_text(text, source=name))
        retriever.add(chunks)
    return retriever


def create_server(*, retriever: Retriever | None = None, redactor: Redactor | None = None):
    """创建 FastMCP server。组件可注入(测试),默认按环境变量装配。"""
    from mcp.server.fastmcp import FastMCP

    _retriever = retriever if retriever is not None else _default_retriever()
    _redactor = redactor if redactor is not None else RuleRedactor()

    server = FastMCP("medrag")

    @server.tool()
    def search_documents(query: str, k: int = 4) -> str:
        """在医疗文档库中检索与查询相关的文本块(混合检索:向量+BM25)。"""
        results = _retriever.search(query, k=k)
        if not results:
            return "(没有检索到相关内容)"
        return "\n\n".join(
            f"[{r.chunk.source} | 相关度 {r.score:.3f}]\n{r.chunk.text}" for r in results
        )

    @server.tool()
    def redact_text(text: str) -> str:
        """对文本做 PII 脱敏:遮蔽编号/姓名/日期/电话/邮箱/身份证,保留临床内容。"""
        return _redactor.redact(text)

    @server.tool()
    def synthesize_record(seed: int = 0) -> str:
        """生成一份合成病历(无真实病人信息,同 seed 同输出,可复现)。"""
        return generate_record(random.Random(seed))

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
