"""VLM / 多模态(M14):给系统加"眼睛"。

M1-M13 的 LLM 只吃文字;VLM(Vision-Language Model)能同时看图 + 读写字。
本模块新增一个与 `LLM` 平行的 `VLM` 协议(输入多了一个 images 参数),
并提供一个 `DescribeImageTool` 把视觉能力包成 M11 的 agent 工具——于是 agent
可以"看一张医疗影像/图表再回答"。

延续 Protocol + 可换实现:
- `EchoVLM`:假实现,不调模型,只报告收到几张图 + 回显提示(CI/离线可测)。
- `AnthropicVLM`:真实现,走 Claude 多模态(base64 图像块);注入式 client 可测。
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class ImageInput:
    """一张待分析的图片:原始字节 + MIME 类型。"""

    data: bytes
    media_type: str

    @classmethod
    def from_path(cls, path: str | Path) -> ImageInput:
        p = Path(path)
        media_type = _MEDIA_TYPES.get(p.suffix.lower())
        if media_type is None:
            raise ValueError(f"不支持的图片类型:{p.suffix}(支持 {', '.join(_MEDIA_TYPES)})")
        return cls(data=p.read_bytes(), media_type=media_type)

    def to_base64(self) -> str:
        return base64.standard_b64encode(self.data).decode("utf-8")


class VLM(Protocol):
    """任何能"看图 + 读提示 → 生成文本"的对象。images 为空时退化成纯文本 LLM。"""

    def generate(
        self, *, system: str, user: str, images: list[ImageInput] | None = None
    ) -> str: ...


class EchoVLM:
    """占位实现:不调模型,只回报收到几张图 + 回显提示。用于离线验证链路。"""

    def generate(self, *, system: str, user: str, images: list[ImageInput] | None = None) -> str:
        images = images or []
        total = sum(len(img.data) for img in images)
        return f"[EchoVLM] 收到 {len(images)} 张图片(共 {total} 字节)。提示:\n\n{user}"


class AnthropicVLM:
    """真实现:Claude 多模态。把图片作为 base64 image 块 + 文本块一起发给模型。

    注入式 client(client 参数)便于测试——无需真实 API key 即可断言消息构造。
    """

    def __init__(self, model: str | None = None, *, client=None) -> None:
        if client is not None:
            self._client = client  # 注入假 client —— 测试用,不碰网络
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError("需要 anthropic,请运行:pip install anthropic") from exc
            self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("MEDRAG_MODEL", "claude-opus-4-8")

    def generate(self, *, system: str, user: str, images: list[ImageInput] | None = None) -> str:
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.to_base64(),
                },
            }
            for img in images or []
        ]
        content.append({"type": "text", "text": user})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=16000,
            thinking={"type": "adaptive"},  # 医疗影像判读对准确性敏感
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


_DESCRIBE_SYSTEM = "你是医疗影像助手。只描述图片中实际可见的内容,不确定就明说,不要臆测诊断。"


class DescribeImageTool:
    """把 VLM 包成 M11 的 agent 工具:让 agent 能"看一张图再回答"。"""

    name = "describe_image"
    description = "查看一张图片并回答关于它的问题(医疗影像、图表、照片等)。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "图片文件路径"},
            "question": {"type": "string", "description": "关于这张图片的问题"},
        },
        "required": ["path", "question"],
    }

    def __init__(self, vlm: VLM) -> None:
        self._vlm = vlm

    def run(self, args: dict) -> str:
        image = ImageInput.from_path(args["path"])
        return self._vlm.generate(system=_DESCRIBE_SYSTEM, user=args["question"], images=[image])
