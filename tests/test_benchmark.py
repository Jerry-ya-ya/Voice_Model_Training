import pytest

from benchmark import summarize_times


def test_summarize_times() -> None:
    result = summarize_times([1.0, 2.0, 3.0, 4.0])
    assert result["mean_ms"] == pytest.approx(2.5)
    assert result["median_ms"] == pytest.approx(2.5)
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0
    assert result["p95_ms"] == 4.0


def test_summarize_times_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize_times([])

