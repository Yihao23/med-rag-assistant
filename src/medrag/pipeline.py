"""RAG 流水线:把切块、向量库、LLM 串起来。"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_text
from .llm import LLM
from .store import SearchResult, VectorStore
from .tracing import NullTracer, Tracer

SYSTEM_PROMPT = (
    "你是一个严谨的医疗文档助手。只根据下面提供的【上下文】回答问题。"
    "如果上下文中找不到答案,就明确说「根据提供的文档无法回答」,不要编造。"
    "在答案中尽量标注信息来源(文件名)。"
)


@dataclass
class Answer:
    text: str
    sources: list[SearchResult]


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """把检索到的文本块和问题拼成发给 LLM 的 user 提示。"""
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[来源 {i} | {r.chunk.source}]\n{r.chunk.text}")
    context = "\n\n".join(context_parts) if context_parts else "(没有检索到相关内容)"
    return f"【上下文】\n{context}\n\n【问题】\n{question}"


class RagPipeline:
    def __init__(self, store: VectorStore, llm: LLM, tracer: Tracer | None = None) -> None:
        self._store = store
        self._llm = llm
        self._tracer = tracer if tracer is not None else NullTracer()

    def index(self, documents: dict[str, str], *, size: int = 800, overlap: int = 100) -> int:
        """对 {文件名: 文本} 切块并写入向量库,返回写入的块数。"""
        all_chunks = []
        for name, text in documents.items():
            all_chunks.extend(chunk_text(text, source=name, size=size, overlap=overlap))
        self._store.add(all_chunks)
        return len(all_chunks)

    def answer(self, question: str, *, k: int = 4) -> Answer:
        with self._tracer.span("rag.answer", input={"question": question, "k": k}) as root:
            with self._tracer.span("search", input={"question": question, "k": k}) as span:
                results = self._store.search(question, k=k)
                span.update(output=[{"source": r.chunk.source, "score": r.score} for r in results])
            prompt = build_prompt(question, results)

            with self._tracer.span(
                "generate", input={"system": SYSTEM_PROMPT, "user": prompt}
            ) as span:
                text = self._llm.generate(system=SYSTEM_PROMPT, user=prompt)
                span.update(output=text)

            root.update(output=text)
            return Answer(text=text, sources=results)
