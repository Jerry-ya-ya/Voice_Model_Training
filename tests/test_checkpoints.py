import argparse
from pathlib import Path

import pytest
import torch

from src.training.checkpoints import find_latest_checkpoint
from train import positive_integer, prepare_continuation


def test_find_latest_numbered_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best.pt").touch()
    (checkpoint_dir / "epoch_0002.pt").touch()
    (checkpoint_dir / "epoch_0010.pt").touch()
    (checkpoint_dir / "epoch_invalid.pt").touch()
    assert find_latest_checkpoint(tmp_path).name == "epoch_0010.pt"


def test_find_latest_falls_back_to_best(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    best = checkpoint_dir / "best.pt"
    best.touch()
    assert find_latest_checkpoint(tmp_path) == best


def test_find_latest_reports_missing_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        find_latest_checkpoint(tmp_path)


def test_positive_integer() -> None:
    assert positive_integer("3") == 3
    with pytest.raises(argparse.ArgumentTypeError):
        positive_integer("0")


def test_prepare_continuation_sets_additional_epoch_target(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "example" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch_0007.pt"
    torch.save({"epoch": 7}, checkpoint)
    config = {
        "experiment": {"output_dir": str(tmp_path), "name": "example"},
        "training": {"epochs": 100},
    }
    assert prepare_continuation(config, 3) == checkpoint
    assert config["training"]["epochs"] == 10
