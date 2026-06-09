from medrag.chunking import Chunk
from medrag.embeddings import HashingEmbedder
from medrag.eval import (
    EvalCase,
    hit_at_k,
    keyword_recall,
    llm_judge,
    load_eval_cases,
    run_eval,
)
from medrag.llm import EchoLLM
from medrag.pipeline import RagPipeline
from medrag.store import InMemoryVectorStore, SearchResult

# --- 单个指标:纯函数,直接喂输入断言输出 ---


def _result(source: str) -> SearchResult:
    return SearchResult(chunk=Chunk(text="x", source=source), score=1.0)


def test_hit_at_k_true_when_source_present():
    sources = [_result("cardio.txt"), _result("diabetes.txt")]
    assert hit_at_k(sources, "cardio.txt") is True


def test_hit_at_k_false_when_source_absent():
    sources = [_result("diabetes.txt")]
    assert hit_at_k(sources, "cardio.txt") is False


def test_keyword_recall_partial():
    # 2 个关键词命中 1 个 → 0.5
    assert keyword_recall("答案里有 STEMI", ["STEMI", "心肌梗死"]) == 0.5


def test_keyword_recall_empty_is_one():
    assert keyword_recall("任意答案", []) == 1.0


class FixedJudge:
    """假 judge:无论问什么都返回固定评分文本。"""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def generate(self, *, system: str, user: str) -> str:
        return self._reply


def test_llm_judge_extracts_first_score():
    assert llm_judge("q", "a", FixedJudge("我给这个答案打 4 分")) == 4


def test_llm_judge_returns_zero_when_no_digit():
    assert llm_judge("q", "a", FixedJudge("无法评分")) == 0


# --- 读取数据集 ---


def test_load_eval_cases(tmp_path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        '{"question": "q1", "expected_source": "a.txt", "expected_keywords": ["x"]}\n'
        "\n"  # 空行应被跳过
        '{"question": "q2", "expected_source": "b.txt", "expected_keywords": []}\n',
        encoding="utf-8",
    )
    cases = load_eval_cases(p)
    assert len(cases) == 2
    assert cases[0] == EvalCase("q1", "a.txt", ["x"])


# --- 端到端:证明指标会区分"命中" vs "未命中" ---


def _pipeline():
    store = InMemoryVectorStore(HashingEmbedder())
    pipeline = RagPipeline(store, EchoLLM())
    # 带空格的受控文本,让 hashing 向量器能区分
    pipeline.index(
        {
            "cardio.txt": "STEMI 胸痛 心电图 阿司匹林",
            "diabetes.txt": "血糖 胰岛素 糖尿病 二甲双胍",
        }
    )
    return pipeline


def test_run_eval_discriminates_hit_and_miss():
    cases = [
        # 必中:问心脏,期望 cardio,关键词都在被检索的块里(echo 会回显)
        EvalCase("胸痛 心电图", "cardio.txt", ["STEMI", "阿司匹林"]),
        # 必不中:问糖尿病,却把期望来源写成 cardio → 检索不到 → hit=False
        EvalCase("血糖 胰岛素", "cardio.txt", ["STEMI"]),
    ]
    report = run_eval(_pipeline(), cases, k=1)

    assert report.results[0].hit is True
    assert report.results[0].keyword_recall == 1.0
    assert report.results[1].hit is False
    assert report.results[1].keyword_recall == 0.0
    assert report.hit_rate == 0.5
    assert report.mean_judge_score is None  # 没传 judge


def test_run_eval_uses_judge_when_provided():
    cases = [EvalCase("胸痛 心电图", "cardio.txt", ["STEMI"])]
    report = run_eval(_pipeline(), cases, k=1, judge=FixedJudge("5"))
    assert report.results[0].judge_score == 5
    assert report.mean_judge_score == 5.0
