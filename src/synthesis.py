"""Reusable checkpoint-backed text-to-waveform synthesis engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.text import CharacterTokenizer
from src.models.acoustic_model import AcousticModel
from src.models.vocoder import Vocoder
from src.utils.io import resolve_device, save_mel_plot, save_waveform


DEFAULT_CHECKPOINTS = (
    Path("runs/ljspeech_mvp/checkpoints/best.pt"),
    Path("runs/smoke_test/checkpoints/best.pt"),
)


def find_checkpoint(candidates: tuple[Path, ...] = DEFAULT_CHECKPOINTS) -> Path:
    """Prefer the real LJSpeech model, with the smoke model as a fallback."""
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No checkpoint found. Looked for: {searched}")


@dataclass
class SynthesisResult:
    text: str
    mel: torch.Tensor
    waveform: torch.Tensor
    elapsed_seconds: float
    audio_seconds: float

    @property
    def real_time_factor(self) -> float:
        return self.elapsed_seconds / max(self.audio_seconds, 1e-9)


class TTSEngine:
    """Load the acoustic model and vocoder once, then synthesize many texts."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto", vocoder_backend: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.tokenizer = CharacterTokenizer()
        self.model = AcousticModel(
            self.tokenizer.vocab_size,
            int(self.config["audio"]["n_mels"]),
            self.config["model"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.vocoder_backend = vocoder_backend or self.config["vocoder"]["backend"]
        self.vocoder = Vocoder(self.vocoder_backend, self.config["audio"], self.device)
        self.sample_rate = int(self.config["audio"]["sample_rate"])

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def synthesize(self, text: str) -> SynthesisResult:
        if not text.strip():
            raise ValueError("Text must not be empty")
        tokens = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)
        token_lengths = torch.tensor([tokens.shape[1]], device=self.device)
        self._synchronize()
        started = time.perf_counter()
        outputs = self.model(tokens, token_lengths)
        frames = int(outputs["mel_lengths"][0])
        mel_for_vocoder = outputs["mel"][0, :frames].transpose(0, 1)
        waveform = self.vocoder(mel_for_vocoder)
        self._synchronize()
        elapsed = time.perf_counter() - started
        mel = mel_for_vocoder.transpose(0, 1).detach().cpu()
        return SynthesisResult(
            text=text,
            mel=mel,
            waveform=waveform,
            elapsed_seconds=elapsed,
            audio_seconds=waveform.numel() / self.sample_rate,
        )

    def save(self, result: SynthesisResult, output_dir: str | Path, stem: str = "generated") -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        wav_path = output_dir / f"{stem}.wav"
        mel_path = output_dir / f"{stem}_mel.png"
        save_waveform(result.waveform, wav_path, self.sample_rate)
        save_mel_plot(result.mel, mel_path, title=result.text)
        return wav_path, mel_path

