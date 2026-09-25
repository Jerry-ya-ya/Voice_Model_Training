import sys
from pathlib import Path

from cli import (
    ask_int,
    build_training_command,
    next_run_number,
    numbered_run_directories,
    print_summary,
    scan_configs,
    select_menu,
)


def test_ask_int_retries_invalid_input(monkeypatch) -> None:
    answers = iter(["invalid", "0", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert ask_int("Batch size", 2) == 4


def test_scan_configs_finds_yaml_files_in_name_order(tmp_path: Path) -> None:
    second = tmp_path / "b.yml"
    first = tmp_path / "a.yaml"
    second.touch()
    first.touch()
    (tmp_path / "ignored.txt").touch()
    assert scan_configs(tmp_path) == [first, second]


def test_numbered_runs_are_sorted_and_require_checkpoints(tmp_path: Path) -> None:
    for name in ("10", "2", "draft", "0"):
        (tmp_path / name).mkdir()
    for name in ("2", "10"):
        checkpoint_dir = tmp_path / name / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "epoch_0001.pt").touch()
    assert [path.name for path in numbered_run_directories(tmp_path)] == ["2", "10"]
    assert next_run_number(tmp_path) == 11


def test_select_menu_has_numbered_non_tty_fallback(monkeypatch) -> None:
    monkeypatch.setattr("cli._is_interactive_terminal", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert select_menu("Choose", [("first", "a"), ("second", "b")]) == "b"


def test_build_continue_command() -> None:
    command = build_training_command("runtime.yaml", additional_epochs=5, resume_learning_rate=2e-5)
    assert command[0] == sys.executable
    assert command[-4:] == ["--continue-train", "5", "--resume-learning-rate", "2e-05"]


def test_continue_summary_uses_runtime_learning_rate(capsys) -> None:
    config = {
        "experiment": {"name": "2", "output_dir": "runs/test"},
        "training": {"device": "cuda", "batch_size": 2, "epochs": 100, "learning_rate": 0.00098},
        "data": {"max_items": None},
    }
    print_summary(config, category="test", additional_epochs=10, learning_rate_override=None)
    assert "Learning rate：0.00098" in capsys.readouterr().out
