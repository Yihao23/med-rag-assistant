from contextlib import contextmanager

from medrag.embeddings import HashingEmbedder
from medrag.llm import EchoLLM
from medrag.pipeline import RagPipeline
from medrag.store import InMemoryVectorStore
from medrag.tracing import LangfuseTracer, NullTracer


class _RecordingSpan:
    """把 update 收到的数据写回自己那条记录。"""

    def __init__(self, record: dict) -> None:
        self._record = record

    def update(self, *, output=None, metadata=None) -> None:
        self._record["output"] = output
        self._record["metadata"] = metadata


class RecordingTracer:
    """假 tracer:不发任何数据,只把 span 记在内存里供断言。"""

    def __init__(self) -> None:
        self.spans = []

    @contextmanager
    def span(self, name, *, input=None):
        record = {"name": name, "input": input, "output": None, "metadata": None}
        self.spans.append(record)
        yield _RecordingSpan(record)


def test_null_tracer_is_harmless():
    # 唯一的断言就是“不抛异常”:NullTracer 的全部职责是无害。
    with NullTracer().span("x", input={"q": 1}) as s:
        s.update(output="y", metadata={"m": 2})


def test_pipeline_records_spans():
    tracer = RecordingTracer()
    store = InMemoryVectorStore(HashingEmbedder())
    pipeline = RagPipeline(store, EchoLLM(), tracer=tracer)
    pipeline.index({"cardio.txt": "STEMI 胸痛 心电图 阿司匹林"})

    pipeline.answer("胸痛 心电图", k=1)

    # 1) 三个 span,顺序 rag.answer(根)→ search → generate
    assert [s["name"] for s in tracer.spans] == ["rag.answer", "search", "generate"]

    root, search, generate = tracer.spans

    # 2) 根 span 带了问题,output 是最终答案
    assert root["input"] == {"question": "胸痛 心电图", "k": 1}
    assert root["output"].startswith("[EchoLLM]")

    # 3) search 的 input 带了问题和 k
    assert search["input"] == {"question": "胸痛 心电图", "k": 1}

    # 4) search 的 output 是“来源+分数”摘要
    assert len(search["output"]) == 1
    assert search["output"][0]["source"] == "cardio.txt"

    # 5) generate 记录了发给 LLM 的内容和 LLM 的回答
    assert "胸痛 心电图" in generate["input"]["user"]
    assert generate["output"].startswith("[EchoLLM]")


class _FakeLangfuseSpan:
    """模拟 Langfuse v4 的 span:记下 update 收到的 output。"""

    def __init__(self, name, input) -> None:
        self.name = name
        self.input = input
        self.output = None

    def update(self, *, output=None, metadata=None) -> None:
        self.output = output


class _FakeLangfuseClient:
    """模拟 Langfuse v4 client:只把 observation 记在内存里,不联网。"""

    def __init__(self) -> None:
        self.observations = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, *, as_type, name, input=None):
        span = _FakeLangfuseSpan(name, input)
        self.observations.append(span)
        yield span

    def flush(self) -> None:
        self.flushed = True


def test_langfuse_tracer_forwards_to_client():
    # 注入假 client,验证 LangfuseTracer 把 name/input/output 正确转发,且 flush 透传。
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(client=client)

    with tracer.span("search", input={"q": 1}) as s:
        s.update(output=[{"source": "x"}])

    assert len(client.observations) == 1
    obs = client.observations[0]
    assert obs.name == "search"
    assert obs.input == {"q": 1}
    assert obs.output == [{"source": "x"}]

    tracer.flush()
    assert client.flushed is True
