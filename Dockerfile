# med-rag-assistant API 镜像(M8)
# 刻意不装 embeddings extra(sentence-transformers 体积 ~GB);
# 离线/演示用 MEDRAG_EMBEDDER=hashing,生产可另建带模型的镜像层。
FROM python:3.13-slim

WORKDIR /app

# 先拷依赖清单再拷代码,让依赖层能被 Docker 缓存复用
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,qdrant,vllm,langfuse]"

COPY data ./data

# 非 root 运行(k8s 安全基线)
RUN useradd --create-home medrag
USER medrag

ENV MEDRAG_DATA=/app/data
EXPOSE 8000

CMD ["uvicorn", "--factory", "medrag.api:get_app", "--host", "0.0.0.0", "--port", "8000"]
