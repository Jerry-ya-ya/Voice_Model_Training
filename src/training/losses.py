import torch
from torch.nn import functional as F


def acoustic_loss(outputs: dict, targets: torch.Tensor, token_lengths: torch.Tensor, mel_lengths: torch.Tensor, duration_weight: float) -> tuple[torch.Tensor, dict]:
    frame_ids = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
    mel_mask = frame_ids < mel_lengths.unsqueeze(1)
    mel_loss = (outputs["mel"] - targets).abs().mul(mel_mask.unsqueeze(-1)).sum() / (mel_mask.sum() * targets.shape[-1]).clamp_min(1)
    token_ids = torch.arange(outputs["log_duration_ratios"].shape[1], device=targets.device).unsqueeze(0)
    token_mask = token_ids < token_lengths.unsqueeze(1)
    target_ratio = torch.log(mel_lengths.float() / token_lengths.float()).unsqueeze(1).expand_as(outputs["log_duration_ratios"])
    duration_loss = F.mse_loss(outputs["log_duration_ratios"][token_mask], target_ratio[token_mask])
    total = mel_loss + duration_weight * duration_loss
    return total, {"mel_loss": mel_loss.item(), "duration_loss": duration_loss.item()}

