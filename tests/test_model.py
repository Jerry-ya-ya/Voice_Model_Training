import torch

from src.models.acoustic_model import AcousticModel
from src.training.losses import acoustic_loss


def test_model_shapes_and_backward() -> None:
    config = {"hidden_dim": 16, "attention_heads": 2, "encoder_layers": 1, "decoder_layers": 1, "dropout": 0.0, "max_mel_frames": 100}
    model = AcousticModel(vocab_size=50, n_mels=8, config=config)
    tokens = torch.tensor([[2, 3, 4, 0], [3, 2, 0, 0]])
    token_lengths = torch.tensor([3, 2])
    mel_lengths = torch.tensor([12, 9])
    targets = torch.randn(2, 12, 8)
    outputs = model(tokens, token_lengths, mel_lengths)
    assert outputs["mel"].shape == targets.shape
    loss, metrics = acoustic_loss(outputs, targets, token_lengths, mel_lengths, 0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["mel_loss"] > 0
    assert model.mel_projection.weight.grad is not None

