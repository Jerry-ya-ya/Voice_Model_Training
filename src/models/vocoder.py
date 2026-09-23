"""Explicit Mel-to-waveform boundary with pretrained HiFi-GAN and test fallback."""

from __future__ import annotations

import torch
import torchaudio


class Vocoder:
    def __init__(self, backend: str, audio_config: dict, device: torch.device) -> None:
        self.backend = backend
        self.device = device
        self.sample_rate = int(audio_config["sample_rate"])
        if backend == "hifigan":
            from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH

            if self.sample_rate != HIFIGAN_VOCODER_V3_LJSPEECH.sample_rate:
                raise ValueError("The pretrained LJSpeech HiFi-GAN requires a 22050 Hz sample rate")
            self.model = HIFIGAN_VOCODER_V3_LJSPEECH.get_vocoder().to(device).eval()
        elif backend == "griffin_lim":
            n_stft = int(audio_config["n_fft"]) // 2 + 1
            mel_filter = torchaudio.functional.melscale_fbanks(
                n_freqs=n_stft,
                f_min=float(audio_config.get("f_min", 0)),
                f_max=float(audio_config.get("f_max", self.sample_rate / 2)),
                n_mels=int(audio_config["n_mels"]),
                sample_rate=self.sample_rate,
                mel_scale="slaney",
                norm="slaney",
            )
            # TorchAudio's least-squares inverse can fail on rank-deficient
            # small smoke-test filterbanks; one pseudoinverse is deterministic.
            self.inverse_mel_filter = torch.linalg.pinv(mel_filter.transpose(0, 1).double()).float().to(device)
            self.griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=int(audio_config["n_fft"]),
                win_length=int(audio_config["win_length"]),
                hop_length=int(audio_config["hop_length"]),
                power=1.0,
                n_iter=8,
            ).to(device)
        else:
            raise ValueError(f"Unknown vocoder backend: {backend}")

    @torch.inference_mode()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        mel = log_mel.to(self.device)
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        if self.backend == "hifigan":
            return self.model(mel).squeeze().cpu()
        magnitude = torch.matmul(self.inverse_mel_filter, torch.exp(mel).clamp_max(1e3))
        return self.griffin_lim(magnitude).squeeze().cpu()
