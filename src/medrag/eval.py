"""RAG 评估:衡量检索质量与答案质量。

三个指标:
- hit@k        : 期望来源是否出现在检索到的 top-k 中(检索质量)
- keyword 召回 : 答案包含了多少比例的期望关键词(答案覆盖度)
- LLM-as-judge : 用一个 LLM 给答案打分 1-5(答案质量,需传入 judge,可选)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .llm import LLM
from .pipeline import RagPipeline


@dataclass
class EvalCase:
    """一条评估样本:问题 + 期望来源 + 期望关键词。"""

    question: str
    expected_source: str
    expected_keywords: list[str]


def load_eval_cases(path: Path) -> list[EvalCase]:
    """读取 JSONL 评估集,每行一个 EvalCase。"""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cases.append(EvalCase(**obj))  # JSON 的 key 与字段名一致,直接解包
    return cases


# --- 三个指标(纯函数,各自独立可测)---


def hit_at_k(sources: list, expected_source: str) -> bool:
    """① 检索指标:期望来源是否出现在检索结果里。"""
    return any(r.chunk.source == expected_source for r in sources)


def keyword_recall(answer_text: str, expected_keywords: list[str]) -> float:
    """② 答案指标:答案命中了多少比例的期望关键词(0~1)。"""
    if not expected_keywords:
        return 1.0
    hit = sum(1 for kw in expected_keywords if kw in answer_text)
    return hit / len(expected_keywords)


JUDGE_SYSTEM = (
    "你是一个严格的医疗问答评分员。根据【问题】和【答案】,给答案的正确性"
    "与相关性打分。只输出一个 1 到 5 的整数(5 最好),不要输出任何其他内容。"
)


def llm_judge(question: str, answer_text: str, judge: LLM) -> int:
    """③ 答案质量:用一个 LLM 给答案打分(1-5);抓不到数字则给 0。"""
    user = f"【问题】\n{question}\n\n【答案】\n{answer_text}"
    raw = judge.generate(system=JUDGE_SYSTEM, user=user)
    for ch in raw:
        if ch in "12345":
            return int(ch)
    return 0


# --- 汇总结构 ---


@dataclass
class CaseResult:
    """单条样本的评估结果。"""

    case: EvalCase
    hit: bool
    keyword_recall: float
    judge_score: int | None  # 没传 judge 时为 None


@dataclass
class EvalReport:
    """整个评估集的汇总,提供几个聚合指标。"""

    results: list[CaseResult]

    @property
    def hit_rate(self) -> float:
        return _mean([1.0 if r.hit else 0.0 for r in self.results])

    @property
    def mean_keyword_recall(self) -> float:
        return _mean([r.keyword_recall for r in self.results])

    @property
    def mean_judge_score(self) -> float | None:
        scores = [r.judge_score for r in self.results if r.judge_score is not None]
        return _mean(scores) if scores else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# --- 串起来跑 ---


def run_eval(
    pipeline: RagPipeline,
    cases: list[EvalCase],
    *,
    k: int = 4,
    judge: LLM | None = None,
) -> EvalReport:
    """对每条样本跑一遍 pipeline,算三个指标,返回汇总报告。"""
    results = []
    for case in cases:
        answer = pipeline.answer(case.question, k=k)
        results.append(
            CaseResult(
                case=case,
                hit=hit_at_k(answer.sources, case.expected_source),
                keyword_recall=keyword_recall(answer.text, case.expected_keywords),
                judge_score=(llm_judge(case.question, answer.text, judge) if judge else None),
            )
        )
    return EvalReport(results=results)


# --- 命令行入口:python -m medrag.eval ---


def _print_report(report: EvalReport) -> None:
    print("\n=== RAG 评估报告 ===")
    print(f"样本数:            {len(report.results)}")
    print(f"检索 hit@k:         {report.hit_rate:.2f}")
    print(f"关键词召回(平均):  {report.mean_keyword_recall:.2f}")
    judge = report.mean_judge_score
    print(f"LLM-judge(平均):   {judge:.2f}" if judge is not None else "LLM-judge(平均):   未启用")
    print("\n逐条:")
    for r in report.results:
        mark = "✓" if r.hit else "✗"
        score = f"judge={r.judge_score}" if r.judge_score is not None else "judge=-"
        print(f"  [{mark}] kw={r.keyword_recall:.2f} {score}  {r.case.question}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="医疗 RAG 评估(M4)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="文档目录")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/eval/medical_qa.jsonl"),
        help="JSONL 评估集路径",
    )
    parser.add_argument("--k", type=int, default=4, help="检索的文本块数量")
    parser.add_argument(
        "--llm", default="echo", choices=["anthropic", "echo", "vllm"], help="回答用的 LLM"
    )
    parser.add_argument(
        "--store", default="memory", choices=["memory", "qdrant"], help="向量库实现"
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers",
        choices=["sentence-transformers", "hashing"],
        help="embedding 实现",
    )
    parser.add_argument(
        "--judge",
        default="none",
        choices=["none", "anthropic", "vllm"],
        help="LLM-as-judge 用的模型(none 则跳过该指标)",
    )
    parser.add_argument(
        "--retrieval",
        default="dense",
        choices=["dense", "hybrid", "graph"],
        help="检索方式(dense 纯向量;hybrid 向量+BM25 用 RRF 融合;graph 实体图谱多跳)",
    )
    parser.add_argument(
        "--redact",
        default="none",
        choices=["none", "rule"],
        help="摄入时 PII 脱敏(none 不脱敏;rule 正则规则,零依赖)",
    )
    parser.add_argument(
        "--tracer",
        default="none",
        choices=["none", "langfuse"],
        help="可观测性追踪(none 零依赖;langfuse 需自托管服务 + 环境变量 LANGFUSE_*)",
    )
    args = parser.parse_args(argv)

    # 工厂函数从 cli 复用,避免重复
    from .cli import _make_embedder, _make_llm, _make_redactor, _make_retriever, _make_tracer
    from .loader import load_directory

    if not args.data.is_dir():
        print(f"找不到文档目录:{args.data}", file=sys.stderr)
        return 1
    if not args.eval_set.is_file():
        print(f"找不到评估集:{args.eval_set}", file=sys.stderr)
        return 1

    documents = load_directory(args.data)
    redactor = _make_redactor(args.redact)
    documents = {name: redactor.redact(text) for name, text in documents.items()}
    store = _make_retriever(args.retrieval, args.store, _make_embedder(args.embedder))
    tracer = _make_tracer(args.tracer)
    pipeline = RagPipeline(store, _make_llm(args.llm), tracer=tracer)
    pipeline.index(documents)

    cases = load_eval_cases(args.eval_set)
    judge = _make_llm(args.judge) if args.judge != "none" else None
    try:
        report = run_eval(pipeline, cases, k=args.k, judge=judge)
        _print_report(report)
    finally:
        tracer.flush()  # 确保缓冲的 trace 在进程退出前送达
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
