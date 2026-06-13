import pytest

from medrag.chunking import Chunk
from medrag.embeddings import HashingEmbedder
from medrag.llm import EchoLLM
from medrag.planner import LLMPlanner, Plan, PlanExecutor, ScriptedPlanner, Step
from medrag.store import InMemoryVectorStore
from medrag.tools import SearchDocumentsTool


def _search_tool():
    store = InMemoryVectorStore(HashingEmbedder())
    store.add([Chunk(text="STEMI 心肌梗死 阿司匹林", source="cardio.txt")])
    return SearchDocumentsTool(store, k=1)


# --- LLMPlanner 的 JSON 解析 ---


class _FixedLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate(self, *, system: str, user: str) -> str:
        return self._reply


def test_llm_planner_parses_json_plan():
    raw = (
        '好的,计划如下:[{"description": "检索病例", "tool": "search_documents",'
        ' "args": {"query": "心梗"}}]'
    )
    plan = LLMPlanner(_FixedLLM(raw)).plan("查心梗", tools=[_search_tool()])
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "search_documents"
    assert plan.steps[0].args == {"query": "心梗"}


def test_llm_planner_rejects_non_json():
    with pytest.raises(ValueError):
        LLMPlanner(_FixedLLM("我不知道怎么规划")).plan("q", tools=[])


def test_llm_planner_includes_failures_in_replan_prompt():
    # 重规划时,失败教训必须出现在发给模型的 user 提示里。
    seen = {}

    class _SpyLLM:
        def generate(self, *, system, user):
            seen["user"] = user
            return "[]"

    from medrag.planner import StepResult

    failed = [StepResult(Step("检索", "search_documents", {}), "错误:超时", ok=False)]
    LLMPlanner(_SpyLLM()).plan("q", tools=[], failed=failed)
    assert "上次执行失败" in seen["user"]
    assert "超时" in seen["user"]


# --- PlanExecutor:执行 + 重规划 ---


def test_executor_runs_plan_and_synthesizes():
    plan = Plan(goal="g", steps=[Step("检索心梗", "search_documents", {"query": "心肌梗死"})])
    ex = PlanExecutor(ScriptedPlanner([plan]), [_search_tool()], EchoLLM())
    report = ex.run("主要诊断是什么?")

    assert report.succeeded and report.replans == 0
    assert report.results[0].ok
    assert "STEMI" in report.results[0].output
    assert "STEMI" in report.answer  # 步骤结果进入了最终汇总(echo 回显)


def test_executor_replans_on_failure_with_lessons():
    # 第一版计划用错工具名 → 失败 → 重规划拿到失败信息 → 第二版成功。
    bad = Plan(goal="g", steps=[Step("检索", "wrong_tool", {})])
    good = Plan(goal="g", steps=[Step("检索", "search_documents", {"query": "心肌梗死"})])
    planner = ScriptedPlanner([bad, good])
    report = PlanExecutor(planner, [_search_tool()], EchoLLM()).run("q")

    assert report.succeeded and report.replans == 1
    assert len(planner.replan_calls) == 1  # 重规划被调用了一次
    assert "不存在" in planner.replan_calls[0][0].output  # 且带着失败原因


def test_executor_gives_up_after_max_replans():
    always_bad = Plan(goal="g", steps=[Step("x", "no_such_tool", {})])
    planner = ScriptedPlanner([always_bad] * 5)
    report = PlanExecutor(planner, [_search_tool()], EchoLLM(), max_replans=2).run("q")

    assert not report.succeeded
    assert report.replans == 2
    assert "无法完成" in report.answer


def test_executor_survives_tool_exception():
    class BoomTool:
        name = "boom"
        description = "总是失败"
        parameters = {"type": "object", "properties": {}}

        def run(self, args):
            raise RuntimeError("炸了")

    bad = Plan(goal="g", steps=[Step("会炸", "boom", {})])
    good = Plan(goal="g", steps=[Step("检索", "search_documents", {"query": "心肌梗死"})])
    report = PlanExecutor(
        ScriptedPlanner([bad, good]), [BoomTool(), _search_tool()], EchoLLM()
    ).run("q")
    assert report.succeeded and report.replans == 1
