import sys
from pathlib import Path

from cli import ask_int, build_training_command, discover_runs, print_summary


def test_ask_int_retries_invalid_input(monkeypatch) -> None:
    answers = iter(["invalid", "0", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert ask_int("Batch size", 2) == 4


def test_discover_runs_only_returns_resumable_experiments(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    (valid / "checkpoints").mkdir(parents=True)
    (valid / "config.yaml").touch()
    (valid / "checkpoints" / "epoch_0001.pt").touch()
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "config.yaml").touch()
    assert discover_runs(tmp_path) == [valid]


def test_build_continue_command() -> None:
    command = build_training_command("runtime.yaml", additional_epochs=5, resume_learning_rate=2e-5)
    assert command[0] == sys.executable
    assert command[-4:] == ["--continue-train", "5", "--resume-learning-rate", "2e-05"]


def test_continue_summary_uses_runtime_learning_rate(capsys) -> None:
    config = {
        "experiment": {"name": "test"},
        "training": {"device": "cuda", "batch_size": 2, "epochs": 100, "learning_rate": 0.00098},
        "data": {"max_items": None},
    }
    print_summary(config, additional_epochs=10, learning_rate_override=None)
    assert "Learning rate：0.00098" in capsys.readouterr().out
