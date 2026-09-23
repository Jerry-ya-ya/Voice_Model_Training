from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torchaudio

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def save_mel_plot(mel: torch.Tensor, path: str | Path, title: str = "Generated Mel spectrogram") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.imshow(mel.detach().cpu().T, origin="lower", aspect="auto", interpolation="none")
    axis.set(title=title, xlabel="Frame", ylabel="Mel bin")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_waveform(waveform: torch.Tensor, path: str | Path, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().cpu().float().reshape(1, -1)
    peak = waveform.abs().max()
    if peak > 1:
        waveform = waveform / peak
    torchaudio.save(str(path), waveform, sample_rate)


def append_metrics(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_training_plot(history: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, loss_axis = plt.subplots(figsize=(8, 4))
    loss_axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    loss_axis.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="validation")
    loss_axis.set(xlabel="Epoch", ylabel="Loss", title="Training history")
    loss_axis.legend(loc="upper left")
    learning_axis = loss_axis.twinx()
    learning_axis.plot(epochs, [row["learning_rate"] for row in history], color="gray", linestyle="--", label="learning rate")
    learning_axis.set_ylabel("Learning rate")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
