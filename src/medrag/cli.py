"""命令行入口:加载文档 → 建索引 → 回答问题(单次或交互)。

用法示例:
    medrag --question "主要诊断是什么?"
    medrag                                   # 交互模式
    medrag --llm echo --embedder hashing     # 离线、无需 API key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from .llm import LLM, AnthropicLLM, EchoLLM
from .loader import load_directory
from .pipeline import RagPipeline
from .store import InMemoryVectorStore


def _make_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"未知 embedder:{name}")


def _make_llm(name: str) -> LLM:
    if name == "echo":
        return EchoLLM()
    if name == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"未知 llm:{name}")


def _print_answer(answer) -> None:
    print("\n=== 答案 ===")
    print(answer.text)
    if answer.sources:
        print("\n=== 命中来源 ===")
        for i, r in enumerate(answer.sources, 1):
            print(f"  {i}. {r.chunk.source}  (相似度 {r.score:.3f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="医疗文档 RAG 问答助手(M1)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="文档目录")
    parser.add_argument("--question", "-q", help="单次提问;省略则进入交互模式")
    parser.add_argument("--k", type=int, default=4, help="检索的文本块数量")
    parser.add_argument(
        "--llm",
        default="anthropic",
        choices=["anthropic", "echo"],
        help="LLM 实现(echo 不调用真实模型)",
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers",
        choices=["sentence-transformers", "hashing"],
        help="embedding 实现(hashing 零依赖、可离线)",
    )
    args = parser.parse_args(argv)

    if not args.data.is_dir():
        print(f"找不到文档目录:{args.data}", file=sys.stderr)
        return 1

    documents = load_directory(args.data)
    if not documents:
        print(f"目录 {args.data} 里没有可加载的文档(支持 .txt/.md/.pdf)", file=sys.stderr)
        return 1

    store = InMemoryVectorStore(_make_embedder(args.embedder))
    pipeline = RagPipeline(store, _make_llm(args.llm))
    n_chunks = pipeline.index(documents)
    print(f"已索引 {len(documents)} 个文档,共 {n_chunks} 个文本块。")

    if args.question:
        _print_answer(pipeline.answer(args.question, k=args.k))
        return 0

    # 交互模式
    print("进入交互模式,输入问题(Ctrl-D 或空行退出)。")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question:
            break
        _print_answer(pipeline.answer(question, k=args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
