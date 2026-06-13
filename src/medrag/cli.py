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
from .graph import GraphRetriever
from .llm import LLM, AnthropicLLM, EchoLLM, OpenAICompatLLM
from .loader import load_directory
from .pii import NullRedactor, Redactor, RuleRedactor
from .pipeline import RagPipeline
from .retrieval import BM25Retriever, HybridRetriever, Retriever
from .routing import RoutingLLM
from .store import InMemoryVectorStore, QdrantVectorStore
from .tracing import LangfuseTracer, NullTracer, Tracer


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
    if name == "vllm":
        return OpenAICompatLLM()
    if name == "routing":
        # edge-to-cloud:简单请求走本地 vLLM,复杂请求转云端 Claude
        return RoutingLLM(edge=OpenAICompatLLM(), cloud=AnthropicLLM())
    raise ValueError(f"未知 llm:{name}")


def _make_store(name: str, embedder: Embedder):
    if name == "memory":
        return InMemoryVectorStore(embedder)
    if name == "qdrant":
        return QdrantVectorStore(embedder)
    raise ValueError(f"未知 vector store:{name}")


def _make_tracer(name: str) -> Tracer:
    if name == "none":
        return NullTracer()
    if name == "langfuse":
        return LangfuseTracer()
    raise ValueError(f"未知 tracer:{name}")


def _make_redactor(name: str) -> Redactor:
    if name == "none":
        return NullRedactor()
    if name == "rule":
        return RuleRedactor()
    raise ValueError(f"未知 redact:{name}")


def _make_retriever(retrieval: str, store_name: str, embedder: Embedder) -> Retriever:
    if retrieval == "graph":
        return GraphRetriever()
    dense = _make_store(store_name, embedder)
    if retrieval == "dense":
        return dense
    if retrieval == "hybrid":
        return HybridRetriever(dense=dense, sparse=BM25Retriever())
    raise ValueError(f"未知 retrieval:{retrieval}")


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
        choices=["anthropic", "echo", "vllm", "routing"],
        help="LLM 实现(echo 不调用真实模型;routing 简单走本地 vLLM、复杂转云端 Claude)",
    )

    parser.add_argument(
        "--store",
        default="memory",
        choices=["memory", "qdrant"],
        help="向量库实现(memory 零依赖,qdrant 基于 Qdrant,包并运行本地或远程 Qdrant 实例)",
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers",
        choices=["sentence-transformers", "hashing"],
        help="embedding 实现(hashing 零依赖、可离线)",
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

    if not args.data.is_dir():
        print(f"找不到文档目录:{args.data}", file=sys.stderr)
        return 1

    documents = load_directory(args.data)
    if not documents:
        print(f"目录 {args.data} 里没有可加载的文档(支持 .txt/.md/.pdf)", file=sys.stderr)
        return 1

    redactor = _make_redactor(args.redact)
    documents = {name: redactor.redact(text) for name, text in documents.items()}

    store = _make_retriever(args.retrieval, args.store, _make_embedder(args.embedder))
    tracer = _make_tracer(args.tracer)
    pipeline = RagPipeline(store, _make_llm(args.llm), tracer=tracer)
    n_chunks = pipeline.index(documents)
    print(f"已索引 {len(documents)} 个文档,共 {n_chunks} 个文本块。")

    try:
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
    finally:
        tracer.flush()  # 确保缓冲的 trace 在进程退出前送达


if __name__ == "__main__":
    raise SystemExit(main())
