"""A compact non-autoregressive acoustic model with learned length prediction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        values = torch.zeros(max_length, hidden_dim)
        values[:, 0::2] = torch.sin(position * scale)
        values[:, 1::2] = torch.cos(position * scale[: values[:, 1::2].shape[1]])
        self.register_buffer("values", values, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.values[: inputs.shape[1]].unsqueeze(0)


class AcousticModel(nn.Module):
    """Tokens -> Transformer encoder -> length expansion -> decoder -> log-Mel."""

    def __init__(self, vocab_size: int, n_mels: int, config: dict) -> None:
        super().__init__()
        hidden = int(config["hidden_dim"])
        heads = int(config["attention_heads"])
        dropout = float(config.get("dropout", 0.1))
        self.max_mel_frames = int(config.get("max_mel_frames", 1200))
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position = PositionalEncoding(hidden, max(self.max_mel_frames, 4096))
        encoder_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, batch_first=True, norm_first=True)
        decoder_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, int(config["encoder_layers"]), enable_nested_tensor=False)
        self.decoder = nn.TransformerEncoder(decoder_layer, int(config["decoder_layers"]), enable_nested_tensor=False)
        self.duration_predictor = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.mel_projection = nn.Linear(hidden, n_mels)

    def _expand(self, encoded: torch.Tensor, token_lengths: torch.Tensor, mel_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_frames = int(mel_lengths.max().item())
        expanded = encoded.new_zeros(encoded.shape[0], max_frames, encoded.shape[-1])
        mask = torch.ones(encoded.shape[0], max_frames, dtype=torch.bool, device=encoded.device)
        for index, (token_length, mel_length) in enumerate(zip(token_lengths.tolist(), mel_lengths.tolist())):
            source = encoded[index, :token_length].transpose(0, 1).unsqueeze(0)
            frames = F.interpolate(source, size=mel_length, mode="linear", align_corners=False).squeeze(0).transpose(0, 1)
            expanded[index, :mel_length] = frames
            mask[index, :mel_length] = False
        return expanded, mask

    def forward(self, tokens: torch.Tensor, token_lengths: torch.Tensor, mel_lengths: torch.Tensor | None = None) -> dict:
        token_mask = tokens.eq(0)
        encoded = self.encoder(self.position(self.embedding(tokens)), src_key_padding_mask=token_mask)
        log_ratios = self.duration_predictor(encoded).squeeze(-1)
        if mel_lengths is None:
            valid = (~token_mask).float()
            mean_log_ratio = (log_ratios * valid).sum(1) / valid.sum(1).clamp_min(1)
            mel_lengths = (torch.exp(mean_log_ratio).clamp(1.0, 20.0) * token_lengths).long()
            mel_lengths = mel_lengths.clamp(min=4, max=self.max_mel_frames)
        expanded, mel_mask = self._expand(encoded, token_lengths, mel_lengths)
        decoded = self.decoder(self.position(expanded), src_key_padding_mask=mel_mask)
        return {"mel": self.mel_projection(decoded), "mel_lengths": mel_lengths, "log_duration_ratios": log_ratios}

