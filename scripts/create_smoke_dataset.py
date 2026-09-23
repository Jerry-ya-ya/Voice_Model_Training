"""Create four deterministic tone utterances to test the full pipeline offline."""

from pathlib import Path

import torch
import torchaudio


def main() -> None:
    root = Path("data/smoke")
    root.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    texts = [
        "hello world",
        "this is a tiny speech test",
        "the model learns mel spectrograms",
        "a checkpoint can resume training",
    ]
    rows = []
    for index, text in enumerate(texts):
        duration = 0.24 + index * 0.03
        time = torch.arange(int(sample_rate * duration)) / sample_rate
        envelope = torch.sin(torch.pi * torch.arange(time.numel()) / max(time.numel() - 1, 1)).square()
        waveform = (0.2 * envelope * torch.sin(2 * torch.pi * (180 + index * 35) * time)).unsqueeze(0)
        name = f"smoke_{index}.wav"
        torchaudio.save(str(root / name), waveform, sample_rate)
        rows.append(f"{name}|{text}\n")
    (root / "metadata.csv").write_text("".join(rows), encoding="utf-8")
    print(f"Created {len(rows)} samples in {root}")


if __name__ == "__main__":
    main()

