"""HTTP API(M8):把 RAG 暴露成一个可部署的服务。

CLI 跑完即退,k8s 部署的是常驻服务,所以这里用 FastAPI 把 pipeline 包成
`POST /ask` + `GET /health`。

设计:
- `create_app(pipeline)` 注入一个已建好的 pipeline —— 测试时注入 echo/hashing 的假
  pipeline,无需 API key / 模型 / 真实服务。
- `build_pipeline_from_env()` 按环境变量(MEDRAG_*)装配生产 pipeline,供容器启动用。
- `get_app()` 是给 uvicorn 的工厂:`uvicorn --factory medrag.api:get_app`
  (用工厂而非模块级 app,避免 import 时就去加载文档 / 下载模型)。

注意:本文件刻意**不用** `from __future__ import annotations`——它会把注解变成字符串,
而 FastAPI 解析路由注解时找不到 create_app 里的局部名(AskRequest 等),会把 body
参数误判成 query 参数。
"""

import os
from pathlib import Path

from .pipeline import RagPipeline


def build_pipeline_from_env() -> RagPipeline:
    """按 MEDRAG_* 环境变量装配生产 pipeline,并完成索引。"""
    from .cli import _make_embedder, _make_llm, _make_redactor, _make_retriever, _make_tracer
    from .loader import load_directory

    data = Path(os.environ.get("MEDRAG_DATA", "data"))
    embedder = _make_embedder(os.environ.get("MEDRAG_EMBEDDER", "sentence-transformers"))
    store = _make_retriever(
        os.environ.get("MEDRAG_RETRIEVAL", "dense"),
        os.environ.get("MEDRAG_STORE", "memory"),
        embedder,
    )
    redactor = _make_redactor(os.environ.get("MEDRAG_REDACT", "none"))
    tracer = _make_tracer(os.environ.get("MEDRAG_TRACER", "none"))
    pipeline = RagPipeline(
        store, _make_llm(os.environ.get("MEDRAG_LLM", "anthropic")), tracer=tracer
    )
    pipeline.index({name: redactor.redact(text) for name, text in load_directory(data).items()})
    return pipeline


def create_app(pipeline: RagPipeline):
    """用一个已建好的 pipeline 创建 FastAPI app(依赖注入,便于测试)。"""
    from typing import Annotated

    from fastapi import Body, FastAPI
    from pydantic import BaseModel

    class AskRequest(BaseModel):
        question: str
        k: int = 4

    class Source(BaseModel):
        source: str
        score: float

    class AskResponse(BaseModel):
        answer: str
        sources: list[Source]

    app = FastAPI(title="med-rag-assistant", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: Annotated[AskRequest, Body()]) -> AskResponse:
        answer = pipeline.answer(req.question, k=req.k)
        return AskResponse(
            answer=answer.text,
            sources=[Source(source=r.chunk.source, score=r.score) for r in answer.sources],
        )

    return app


def get_app():
    """uvicorn 工厂入口:uvicorn --factory medrag.api:get_app"""
    return create_app(build_pipeline_from_env())
