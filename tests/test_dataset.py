from pathlib import Path

import torch
import torchaudio

from src.data.dataset import TTSDataset, collate_tts


def test_manifest_dataset_and_cache(tmp_path: Path) -> None:
    sample_rate = 8000
    torchaudio.save(str(tmp_path / "sample.wav"), torch.zeros(1, 800), sample_rate)
    (tmp_path / "metadata.csv").write_text("sample.wav|A test.\n", encoding="utf-8")
    data = {"dataset": "manifest", "root": str(tmp_path), "manifest": str(tmp_path / "metadata.csv"), "cache_dir": str(tmp_path / "cache")}
    audio = {"sample_rate": sample_rate, "n_fft": 128, "win_length": 128, "hop_length": 32, "n_mels": 20, "f_min": 0, "f_max": 4000}
    dataset = TTSDataset(data, audio)
    item = dataset[0]
    batch = collate_tts([item])
    assert item["mel"].shape[1] == 20
    assert batch["tokens"].shape[0] == 1
    assert len(list((tmp_path / "cache").glob("*.pt"))) == 1

