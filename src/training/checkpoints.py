"""Checkpoint discovery helpers used by automatic continuation."""

from __future__ import annotations

import re
from pathlib import Path


EPOCH_CHECKPOINT = re.compile(r"^epoch_(\d+)\.pt$")


def find_latest_checkpoint(run_dir: str | Path) -> Path:
    """Return the highest numbered epoch checkpoint, then fall back to best.pt."""
    checkpoint_dir = Path(run_dir) / "checkpoints"
    numbered: list[tuple[int, Path]] = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("epoch_*.pt"):
            match = EPOCH_CHECKPOINT.match(path.name)
            if match:
                numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered, key=lambda item: item[0])[1]
    best = checkpoint_dir / "best.pt"
    if best.is_file():
        return best
    raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")

