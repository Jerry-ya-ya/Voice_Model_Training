from __future__ import annotations

import csv

import pytest

from src.training.trainer import summarize_epoch_timings
from src.utils.io import append_metrics, format_duration


def test_format_duration() -> None:
    assert format_duration(3661.25) == "01:01:01.25"
    assert format_duration(-1) == "00:00:00.00"


def test_summarize_epoch_timings_ignores_legacy_rows() -> None:
    summary = summarize_epoch_timings(
        [
            {"epoch": 1},
            {"epoch": 2, "epoch_seconds": 12.0},
            {"epoch": 3, "epoch_seconds": 18.0},
        ]
    )

    assert summary == {
        "count": 2,
        "average_seconds": pytest.approx(15.0),
        "longest_epoch": 3,
        "longest_seconds": pytest.approx(18.0),
    }


def test_append_metrics_upgrades_legacy_csv_schema(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    legacy_row = {
        "epoch": 1,
        "global_step": 5,
        "train_loss": 1.0,
        "validation_loss": 1.1,
        "learning_rate": 0.001,
    }
    timed_row = {
        **legacy_row,
        "epoch": 2,
        "train_seconds": 10.0,
        "validation_seconds": 2.0,
        "epoch_seconds": 12.0,
    }

    append_metrics(metrics_path, legacy_row)
    append_metrics(metrics_path, timed_row)

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [*legacy_row, "train_seconds", "validation_seconds", "epoch_seconds"]
    assert rows[0]["epoch_seconds"] == ""
    assert rows[1]["epoch_seconds"] == "12.0"
