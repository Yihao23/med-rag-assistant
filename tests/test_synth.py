import random

from medrag.pii import RuleRedactor
from medrag.synth import generate_record, main


def test_generate_record_is_deterministic_per_seed():
    a = generate_record(random.Random(0))
    b = generate_record(random.Random(0))
    assert a == b


def test_generate_record_has_expected_structure():
    rec = generate_record(random.Random(42))
    assert "病历摘要(合成示例数据 — 非真实病人)" in rec
    assert "病人编号:SYN-" in rec  # 合成编号,非真实病案号
    assert "诊断:" in rec


def test_generate_record_uses_a_known_scenario():
    # 诊断必来自模板池之一(不会生成医学上离谱的内容)。
    rec = generate_record(random.Random(7))
    assert any(dx in rec for dx in ("STEMI", "2 型糖尿病", "桡骨远端骨折"))


def test_synth_main_writes_files(tmp_path):
    code = main(["--n", "3", "--out", str(tmp_path), "--seed", "1"])
    assert code == 0
    files = sorted(tmp_path.glob("*.txt"))
    assert len(files) == 3
    # 合成数据本身不应含被 PII 规则识别为真实标识符的内容(SYN 编号会被遮成 [编号])
    text = files[0].read_text(encoding="utf-8")
    assert RuleRedactor().redact(text).count("[电话]") == 0
