"""端到端测试:验证 `medrag` 命令(离线模式)整条链路能跑通。

这是唯一覆盖 cli.py 入口的测试——其余测试只测底层组件。
用 echo + hashing 假实现,所以无需 API key、无需下载模型,CI 也能跑。
"""

from medrag.cli import main


def test_cli_runs_offline(tmp_path, capsys):
    # 1. 在临时目录里造一份文档(测完 pytest 自动删除)
    #    故意用空格分词,因为 HashingEmbedder 按空格切词才检索得到。
    doc = tmp_path / "report.txt"
    doc.write_text("主要诊断:STEMI 急性心肌梗死。胸痛 心电图 ST 抬高。", encoding="utf-8")

    # 2. 直接调用 main(),等价于命令行:
    #    medrag --data <临时目录> --question "..." --llm echo --embedder hashing
    exit_code = main(
        [
            "--data",
            str(tmp_path),
            "--question",
            "主要诊断是什么?",
            "--llm",
            "echo",
            "--embedder",
            "hashing",
        ]
    )

    # 3. 检查结果
    out = capsys.readouterr().out  # 取出程序 print 的全部输出
    assert exit_code == 0  # 命令正常退出
    assert "已索引" in out  # 确实建了索引
    assert "STEMI" in out  # 检索到的内容进了输出 → 整条链路端到端跑通
