import base64

import pytest

from medrag.vision import AnthropicVLM, DescribeImageTool, EchoVLM, ImageInput

# --- ImageInput ---


def test_image_input_detects_media_type(tmp_path):
    p = tmp_path / "scan.png"
    p.write_bytes(b"fake-png-bytes")
    img = ImageInput.from_path(p)
    assert img.media_type == "image/png"
    assert img.to_base64() == base64.standard_b64encode(b"fake-png-bytes").decode()


def test_image_input_rejects_unsupported_type(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        ImageInput.from_path(p)


# --- EchoVLM ---


def test_echo_vlm_reports_image_count():
    out = EchoVLM().generate(
        system="s",
        user="这是什么?",
        images=[ImageInput(b"abc", "image/png"), ImageInput(b"de", "image/jpeg")],
    )
    assert "收到 2 张图片" in out
    assert "这是什么?" in out


def test_echo_vlm_works_without_images():
    out = EchoVLM().generate(system="s", user="纯文本")
    assert "收到 0 张图片" in out


# --- AnthropicVLM:用假 client 断言消息构造 ---


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    content = [_FakeBlock("看起来是一张胸部 X 光片")]


class _FakeMessages:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.last_kwargs = kwargs
        return _FakeResponse()


class _FakeClient:
    def __init__(self):
        self.last_kwargs = None
        self.messages = _FakeMessages(self)


def test_anthropic_vlm_builds_base64_image_block():
    client = _FakeClient()
    vlm = AnthropicVLM(client=client)
    img = ImageInput(b"\x89PNG-bytes", "image/png")

    out = vlm.generate(system="你是影像助手", user="这是什么?", images=[img])

    assert out == "看起来是一张胸部 X 光片"
    content = client.last_kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == base64.standard_b64encode(b"\x89PNG-bytes").decode()
    assert content[-1] == {"type": "text", "text": "这是什么?"}


# --- DescribeImageTool:视觉作为 agent 工具 ---


def test_describe_image_tool_runs_with_echo_vlm(tmp_path):
    p = tmp_path / "xray.png"
    p.write_bytes(b"img")
    tool = DescribeImageTool(EchoVLM())
    out = tool.run({"path": str(p), "question": "有骨折吗?"})
    assert "收到 1 张图片" in out
    assert "有骨折吗?" in out
