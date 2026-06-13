import asyncio
import sys

import pytest

pytest.importorskip("mcp")  # 没装 mcp extra 时整文件跳过,保持 CI 行为可控


def test_stdio_end_to_end(tmp_path):
    """真实端到端:SDK 客户端经 stdio 拉起 server 子进程,列出并调用工具。

    这验证的是 MCP 接线本身(注册、schema 生成、传输)——任何 MCP 客户端
    走的都是同一条路径。
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    doc = tmp_path / "cardio.txt"
    doc.write_text("STEMI 心肌梗死 阿司匹林 替格瑞洛", encoding="utf-8")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "medrag.mcp_server"],
        env={
            "MEDRAG_DATA": str(tmp_path),  # 离线:hashing + 临时文档目录
            "MEDRAG_EMBEDDER": "hashing",
            "MEDRAG_RETRIEVAL": "hybrid",
            "PATH": "/usr/bin:/bin",
        },
    )

    async def scenario():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert names == {"search_documents", "redact_text", "synthesize_record"}

                hit = await session.call_tool("search_documents", {"query": "心肌梗死", "k": 1})
                assert "cardio.txt" in hit.content[0].text

                red = await session.call_tool(
                    "redact_text", {"text": "病人编号:DEMO-001 电话 13800138000"}
                )
                out = red.content[0].text
                assert "DEMO-001" not in out and "[编号]" in out
                assert "13800138000" not in out and "[电话]" in out

                syn1 = await session.call_tool("synthesize_record", {"seed": 7})
                syn2 = await session.call_tool("synthesize_record", {"seed": 7})
                assert syn1.content[0].text == syn2.content[0].text  # 同 seed 可复现
                assert "合成示例数据" in syn1.content[0].text

    asyncio.run(scenario())
