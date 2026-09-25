from pathlib import Path

import torch

from src.data.text import CharacterTokenizer
from src.models.acoustic_model import AcousticModel
from src.utils.config import load_config


PRESETS = (
    Path("configs/ljspeech_small.yaml"),
    Path("configs/ljspeech_medium.yaml"),
    Path("configs/ljspeech_large.yaml"),
)


def test_ljspeech_presets_are_ordered_and_runnable() -> None:
    tokenizer = CharacterTokenizer()
    parameter_counts = []
    batch_sizes = []
    for path in PRESETS:
        config = load_config(path)
        assert config["data"]["dataset"] == "ljspeech"
        assert config["training"]["device"] == "cuda"
        assert config["audio"]["sample_rate"] == 22050
        assert config["audio"]["n_mels"] == 80
        assert 0 < config["training"]["lr_decay"] <= 1
        assert config["model"]["hidden_dim"] % config["model"]["attention_heads"] == 0
        model = AcousticModel(tokenizer.vocab_size, config["audio"]["n_mels"], config["model"])
        tokens = torch.tensor([[2, 3, 4, 5]])
        lengths = torch.tensor([4])
        mel_lengths = torch.tensor([12])
        output = model(tokens, lengths, mel_lengths)
        assert output["mel"].shape == (1, 12, 80)
        parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
        batch_sizes.append(config["training"]["batch_size"])
    assert parameter_counts == sorted(parameter_counts)
    assert batch_sizes == sorted(batch_sizes, reverse=True)

