import sys
from copy import deepcopy
from pathlib import Path

import cli
from cli import (
    ask_int,
    build_training_command,
    continue_training,
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


def test_build_resume_command_targets_an_explicit_checkpoint() -> None:
    command = build_training_command("runtime.yaml", resume_checkpoint="runs/test/1/checkpoints/epoch_0010.pt")
    assert command[-2:] == ["--resume", "runs/test/1/checkpoints/epoch_0010.pt"]


def test_continue_training_writes_to_next_run_directory(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "configs" / "small.yaml"
    category_dir = tmp_path / "runs" / "small"
    source_run = category_dir / "1"
    next_run = category_dir / "2"
    latest = source_run / "checkpoints" / "epoch_0010.pt"
    source_run.mkdir(parents=True)
    next_run.mkdir()
    base_config = {"experiment": {"output_dir": str(tmp_path / "runs")}}
    saved_config = {
        "experiment": {"output_dir": str(category_dir), "name": "1", "run_number": 1},
        "training": {"epochs": 10, "learning_rate": 0.001},
    }
    checkpoint = {
        "epoch": 10,
        "global_step": 100,
        "optimizer": {"param_groups": [{"lr": 0.0005}]},
    }
    captured = {}

    monkeypatch.setattr(cli, "choose_config", lambda: config_path)
    monkeypatch.setattr(cli, "load_config", lambda path: deepcopy(base_config if path == config_path else saved_config))
    monkeypatch.setattr(cli, "category_directory", lambda *_: category_dir)
    monkeypatch.setattr(cli, "choose_numbered_run", lambda _: source_run)
    monkeypatch.setattr(cli, "find_latest_checkpoint", lambda _: latest)
    monkeypatch.setattr(cli.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(cli, "ask_int", lambda *_args, **_kwargs: 4)
    monkeypatch.setattr(cli, "configure_runtime", lambda config, **_kwargs: config)
    monkeypatch.setattr(cli, "ask_yes_no", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(cli, "launch", lambda config, mode, **kwargs: captured.update(config=config, mode=mode, **kwargs))

    continue_training()

    assert captured["mode"] == "continue"
    assert captured["source_run"] == source_run
    assert captured["resume_checkpoint"] == latest
    assert captured["config"]["experiment"]["name"] == "3"
    assert captured["config"]["experiment"]["parent_run_number"] == 1
    assert captured["config"]["training"]["epochs"] == 14


def test_continue_summary_uses_runtime_learning_rate(capsys) -> None:
    config = {
        "experiment": {"name": "2", "output_dir": "runs/test"},
        "training": {"device": "cuda", "batch_size": 2, "epochs": 100, "learning_rate": 0.00098},
        "data": {"max_items": None},
    }
    print_summary(config, category="test", additional_epochs=10, learning_rate_override=None)
    assert "Learning rate：0.00098" in capsys.readouterr().out
