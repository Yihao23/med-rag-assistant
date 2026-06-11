from medrag.pii import NullRedactor, RuleRedactor


def test_rule_redactor_masks_direct_identifiers():
    r = RuleRedactor()
    text = (
        "病人编号:DEMO-001\n"
        "姓名:张三\n"
        "就诊日期:2026-03-12\n"
        "电话 13800138000,邮箱 a.b@example.com\n"
        "身份证 11010119900307123X\n"
    )
    out = r.redact(text)

    assert "DEMO-001" not in out and "病人编号:[编号]" in out
    assert "张三" not in out and "姓名:[姓名]" in out
    assert "2026-03-12" not in out and "[日期]" in out
    assert "13800138000" not in out and "[电话]" in out
    assert "a.b@example.com" not in out and "[邮箱]" in out
    assert "11010119900307123X" not in out and "[身份证]" in out


def test_rule_redactor_keeps_clinical_content():
    # 医疗术语、年龄、生命体征不是 PII,必须保留(去标识化的关键权衡)。
    r = RuleRedactor()
    text = "55 岁男性,STEMI 急性心肌梗死,血压 158/96 mmHg,服用阿司匹林。"
    assert r.redact(text) == text


def test_null_redactor_is_passthrough():
    text = "病人编号:DEMO-001 电话 13800138000"
    assert NullRedactor().redact(text) == text
