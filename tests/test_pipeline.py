from medrag.embeddings import HashingEmbedder
from medrag.llm import LLM
from medrag.pipeline import RagPipeline, build_prompt
from medrag.store import InMemoryVectorStore, SearchResult


class RecordingLLM:
    """假 LLM:记录收到的提示,返回固定答案。验证 pipeline 接线是否正确。"""

    def __init__(self) -> None:
        self.last_system = ""
        self.last_user = ""

    def generate(self, *, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return "固定答案"


def test_pipeline_indexes_and_answers():
    store = InMemoryVectorStore(HashingEmbedder())
    llm = RecordingLLM()
    pipeline = RagPipeline(store, llm)

    n = pipeline.index({"cardio.txt": "急性心肌梗死 STEMI,胸痛,ST 段抬高。"})
    assert n >= 1

    answer = pipeline.answer("主要诊断是什么?")
    assert answer.text == "固定答案"
    assert answer.sources  # 检索到了内容
    # 检索到的上下文应被拼进发给 LLM 的 user 提示
    assert "STEMI" in llm.last_user


def test_recording_llm_satisfies_protocol():
    # 静态契约检查:RecordingLLM 应符合 LLM 协议
    llm: LLM = RecordingLLM()
    assert llm.generate(system="s", user="u") == "固定答案"


def test_build_prompt_marks_source_and_question():
    results = [SearchResult(chunk_text_source(), score=0.9)]
    prompt = build_prompt("诊断是什么?", results)
    assert "cardio.txt" in prompt
    assert "诊断是什么?" in prompt


def chunk_text_source():
    from medrag.chunking import Chunk

    return Chunk(text="STEMI 胸痛", source="cardio.txt")
