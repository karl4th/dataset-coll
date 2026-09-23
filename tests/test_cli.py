from aether_dataset.cli import main


def test_benchmark_rejects_placeholder_config(capsys) -> None:
    result = main(["benchmark", "--config", "configs/loquacious-medium.yaml"])
    captured = capsys.readouterr()
    assert result == 2
    assert "output.path" in captured.err


def test_publish_requires_token(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = main(["publish", "--output", str(tmp_path)])
    captured = capsys.readouterr()
    assert result == 2
    assert "HF_TOKEN" in captured.err
