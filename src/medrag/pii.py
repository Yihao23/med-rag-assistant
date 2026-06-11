"""PII 脱敏(M7):在文档进入向量库 / Langfuse trace 之前,把可识别信息替换成占位符。

医疗 RAG 必须防止真实病人信息外泄(DSGVO / GDPR)。即使本项目只用合成数据,
在摄入阶段脱敏也是一道纵深防御——脱敏发生在 index 之前,所以 PII 既不进向量库、
也不进 trace 的 prompt。

延续 Protocol + 可换实现:
- `NullRedactor`:不脱敏(默认),原样返回。
- `RuleRedactor`:正则规则,零依赖、确定性。真实系统可再加 NER / Presidio / LLM 版。
"""

from __future__ import annotations

import re
from typing import Protocol


class Redactor(Protocol):
    """任何能把一段文本里的 PII 替换掉的对象。"""

    def redact(self, text: str) -> str: ...


class NullRedactor:
    """不脱敏。用作默认实现,保证关闭时行为与从前一致。"""

    def redact(self, text: str) -> str:
        return text


# 每条规则:(正则, 替换串)。按顺序应用。
# 只针对“直接标识符”(编号/证件/电话/邮箱/日期/姓名字段);刻意保留年龄、生命体征等
# 临床信息——医疗去标识化的经典权衡:既要保护隐私,又不能毁掉诊疗价值。
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[邮箱]"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "[身份证]"),  # 中国大陆身份证 18 位
    (re.compile(r"\b1[3-9]\d{9}\b"), "[电话]"),  # 大陆手机号 11 位
    (re.compile(r"\d{4}-\d{1,2}-\d{1,2}"), "[日期]"),  # ISO 日期
    # 带标签的字段:保留标签,只遮值
    (re.compile(r"(病人编号|患者编号|住院号|门诊号|病案号)([:：])\s*\S+"), r"\1\2[编号]"),
    (re.compile(r"(姓名|患者姓名)([:：])\s*\S+"), r"\1\2[姓名]"),
]


class RuleRedactor:
    """基于正则的脱敏:把常见 PII 模式替换成 [类别] 占位符。"""

    def redact(self, text: str) -> str:
        for pattern, repl in _RULES:
            text = pattern.sub(repl, text)
        return text
