import argparse
from pathlib import Path

import torch

from src.data.text import CharacterTokenizer
from src.models.acoustic_model import AcousticModel
from src.models.vocoder import Vocoder
from src.utils.io import resolve_device, save_mel_plot, save_waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech from a trained acoustic model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vocoder", choices=["hifigan", "griffin_lim"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = CharacterTokenizer()
    model = AcousticModel(tokenizer.vocab_size, int(config["audio"]["n_mels"]), config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = torch.tensor([tokenizer.encode(args.text)], device=device)
    token_lengths = torch.tensor([tokens.shape[1]], device=device)
    with torch.inference_mode():
        outputs = model(tokens, token_lengths)
    frames = int(outputs["mel_lengths"][0])
    mel = outputs["mel"][0, :frames].cpu()
    backend = args.vocoder or config["vocoder"]["backend"]
    waveform = Vocoder(backend, config["audio"], device)(mel.transpose(0, 1))
    output_dir = Path(args.output_dir)
    wav_path = output_dir / "generated.wav"
    mel_path = output_dir / "generated_mel.png"
    save_waveform(waveform, wav_path, int(config["audio"]["sample_rate"]))
    save_mel_plot(mel, mel_path)
    duration = waveform.numel() / int(config["audio"]["sample_rate"])
    print(f"Input text: {args.text}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocoder: {backend}")
    print(f"Generated audio duration: {duration:.2f} seconds")
    print(f"Output path: {wav_path}")
    print(f"Mel plot: {mel_path}")


if __name__ == "__main__":
    main()

