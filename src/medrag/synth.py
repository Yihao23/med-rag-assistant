"""合成病历生成(M7):产出结构化的假病历,提供无 PII 的安全测试 / 演示数据。

与脱敏(pii.py)互补:脱敏是"防真实数据漏出",合成数据是"一开始就只用假数据"。
完全模板化 + 随机填充,确定性可复现(同 seed 同输出)。

用法:python -m medrag.synth --n 5 --out data/synthetic
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# 每个场景:症状、诊断、用药、检查彼此自洽,避免生成医学上矛盾的病历。
_SCENARIOS = [
    {
        "主诉": "持续 {n} 天的胸痛伴气短",
        "诊断": "STEMI 急性心肌梗死",
        "用药": "阿司匹林、替格瑞洛",
        "检查": "心电图示 ST 段抬高;肌钙蛋白升高",
    },
    {
        "主诉": "多饮多尿伴体重下降 {n} 周",
        "诊断": "2 型糖尿病",
        "用药": "二甲双胍、胰岛素",
        "检查": "空腹血糖 11.2 mmol/L;糖化血红蛋白 9.1%",
    },
    {
        "主诉": "跌倒后右腕疼痛肿胀 {n} 小时",
        "诊断": "桡骨远端骨折",
        "用药": "对乙酰氨基酚",
        "检查": "X 光示桡骨远端骨折,断端移位",
    },
]


def generate_record(rng: random.Random) -> str:
    """用给定的随机源生成一份合成病历(确定性:同 rng 状态同输出)。"""
    sc = rng.choice(_SCENARIOS)
    age = rng.randint(20, 85)
    sex = rng.choice(["男", "女"])
    n = rng.randint(1, 14)
    record_id = f"SYN-{rng.randint(1000, 9999)}"  # 合成编号,非真实病案号
    return (
        "病历摘要(合成示例数据 — 非真实病人)\n\n"
        f"病人编号:{record_id}\n\n"
        f"主诉:\n{age} 岁{sex}性,因{sc['主诉'].format(n=n)}就诊。\n\n"
        f"诊断:\n{sc['诊断']}\n\n"
        f"用药:\n{sc['用药']}\n\n"
        f"辅助检查:\n{sc['检查']}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成合成病历(M7)")
    parser.add_argument("--n", type=int, default=5, help="生成多少份")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"), help="输出目录")
    parser.add_argument("--seed", type=int, default=0, help="随机种子(可复现)")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        (args.out / f"synthetic_{i:03d}.txt").write_text(generate_record(rng), encoding="utf-8")
    print(f"已生成 {args.n} 份合成病历到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
