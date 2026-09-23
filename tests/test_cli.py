from aether_dataset.cli import main


def test_benchmark_rejects_placeholder_config(capsys) -> None:
    result = main(["benchmark", "--config", "configs/loquacious-medium.yaml"])
    captured = capsys.readouterr()
    assert result == 2
    assert "output.path" in captured.err
