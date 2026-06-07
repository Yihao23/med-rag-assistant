"""OpenAICompatLLM(vLLM 等)测试。

vLLM 是独立服务器,CI 里无法启动。所以这里注入一个 *假 client*,
模仿 openai 客户端的调用结构,从而无需任何网络/服务器就能测试逻辑。
"""

from medrag.llm import LLM, OpenAICompatLLM


# --- 一组假对象,模仿 openai 客户端的嵌套结构 ---
# 真实调用链是:client.chat.completions.create(...).choices[0].message.content
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs = None  # 记录被调用时收到的参数

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse("假模型的回答")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class FakeOpenAIClient:
    """模仿 openai.OpenAI 的最小结构:只够 generate() 用。"""

    def __init__(self):
        self.chat = _FakeChat()


def test_vllm_generate_returns_model_text():
    fake = FakeOpenAIClient()
    llm = OpenAICompatLLM(client=fake, model="test-model")  # 注入假 client

    out = llm.generate(system="你是助手", user="主要诊断是什么?")

    assert out == "假模型的回答"  # 正确取回了模型回复


def test_vllm_sends_system_and_user_messages():
    fake = FakeOpenAIClient()
    llm = OpenAICompatLLM(client=fake, model="test-model")

    llm.generate(system="系统提示", user="用户问题")

    kwargs = fake.chat.completions.last_kwargs
    assert kwargs["model"] == "test-model"
    # 验证 system / user 被正确组装成两条 messages
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["system", "user"]
    assert kwargs["messages"][0]["content"] == "系统提示"
    assert kwargs["messages"][1]["content"] == "用户问题"


def test_vllm_satisfies_llm_protocol():
    llm: LLM = OpenAICompatLLM(client=FakeOpenAIClient())
    assert llm.generate(system="s", user="u") == "假模型的回答"
