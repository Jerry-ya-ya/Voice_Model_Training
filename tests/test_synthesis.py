from datetime import datetime
from pathlib import Path

import pytest

from src.synthesis import SynthesisResult, find_checkpoint
from tts_demo import output_stem


def test_find_checkpoint_prefers_first_existing_candidate(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.touch()
    second.touch()
    assert find_checkpoint((first, second)) == first


def test_find_checkpoint_reports_missing_candidates(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        find_checkpoint((tmp_path / "missing.pt",))


def test_output_stem_is_stable() -> None:
    assert output_stem(7, datetime(2026, 9, 24, 15, 4, 5)) == "speech_20260924_150405_007"


def test_real_time_factor() -> None:
    result = SynthesisResult("test", None, None, elapsed_seconds=0.25, audio_seconds=1.0)  # type: ignore[arg-type]
    assert result.real_time_factor == pytest.approx(0.25)

