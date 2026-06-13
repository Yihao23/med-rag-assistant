from medrag.agent import Agent, ModelTurn, ScriptedToolLLM, ToolCall
from medrag.chunking import Chunk
from medrag.embeddings import HashingEmbedder
from medrag.store import InMemoryVectorStore
from medrag.tools import SearchDocumentsTool


def _search_tool():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add([Chunk(text="STEMI 心肌梗死 阿司匹林", source="cardio.txt")])
    return SearchDocumentsTool(store, k=1)


# --- SearchDocumentsTool 本身 ---


def test_search_tool_returns_formatted_results():
    out = _search_tool().run({"query": "心肌梗死"})
    assert "cardio.txt" in out
    assert "STEMI" in out


def test_search_tool_empty_store():
    tool = SearchDocumentsTool(InMemoryVectorStore(HashingEmbedder()))
    assert tool.run({"query": "任意"}) == "(没有检索到相关内容)"


# --- agent 循环:剧本化模型,断言循环行为 ---


def test_agent_executes_tool_then_answers():
    # 剧本:第一步调 search_documents,第二步给最终答案。
    llm = ScriptedToolLLM(
        [
            ModelTurn(
                tool_calls=[ToolCall(id="c1", name="search_documents", args={"query": "心肌梗死"})]
            ),
            ModelTurn(text="主要诊断是 STEMI(来源:cardio.txt)"),
        ]
    )
    agent = Agent(llm, [_search_tool()])
    answer = agent.run("主要诊断是什么?")

    assert answer.text == "主要诊断是 STEMI(来源:cardio.txt)"
    assert len(answer.steps) == 1
    assert answer.steps[0].tool == "search_documents"
    assert "STEMI" in answer.steps[0].result

    # 第二步时,历史里应有 assistant 的调用 + tool 的结果(模型看得到检索内容)
    second = llm.seen_messages[1]
    roles = [m["role"] for m in second]
    assert roles == ["user", "assistant", "tool"]
    assert "STEMI" in second[2]["content"]


def test_agent_answers_directly_without_tools():
    # 模型可以决定不调工具(和 pipeline 必检索的本质区别)。
    llm = ScriptedToolLLM([ModelTurn(text="你好!")])
    answer = Agent(llm, [_search_tool()]).run("打个招呼")
    assert answer.text == "你好!"
    assert answer.steps == []


def test_agent_feeds_unknown_tool_error_back():
    # 模型幻觉出不存在的工具 → 错误喂回历史,模型下一步自纠。
    llm = ScriptedToolLLM(
        [
            ModelTurn(tool_calls=[ToolCall(id="c1", name="no_such_tool", args={})]),
            ModelTurn(text="好的,改用已有信息回答"),
        ]
    )
    answer = Agent(llm, [_search_tool()]).run("q")
    assert "不存在" in answer.steps[0].result
    assert answer.text == "好的,改用已有信息回答"


def test_agent_stops_at_max_steps():
    # 模型每步都要调工具、永不收口 → max_steps 兜底防无限循环。
    endless = ModelTurn(tool_calls=[ToolCall(id="c", name="search_documents", args={"query": "x"})])
    llm = ScriptedToolLLM([endless] * 10)
    answer = Agent(llm, [_search_tool()], max_steps=3).run("q")
    assert "最大工具调用步数" in answer.text
    assert len(answer.steps) == 3


def test_agent_survives_tool_exception():
    # 工具抛异常不应让 agent 崩溃,而是把错误作为结果喂回模型。
    class BoomTool:
        name = "boom"
        description = "总是失败"
        parameters = {"type": "object", "properties": {}}

        def run(self, args):
            raise RuntimeError("炸了")

    llm = ScriptedToolLLM(
        [
            ModelTurn(tool_calls=[ToolCall(id="c1", name="boom", args={})]),
            ModelTurn(text="工具失败,据此回答"),
        ]
    )
    answer = Agent(llm, [BoomTool()]).run("q")
    assert "执行失败" in answer.steps[0].result
    assert answer.text == "工具失败,据此回答"
