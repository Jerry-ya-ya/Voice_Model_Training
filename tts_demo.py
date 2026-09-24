"""Interactive terminal demo for trying custom sentences with a trained model."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.synthesis import TTSEngine, find_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize custom sentences with a trained TTS checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Defaults to the LJSpeech best checkpoint when available")
    parser.add_argument("--text", action="append", help="Synthesize once; repeat this option for multiple sentences")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--vocoder", choices=["hifigan", "griffin_lim"], default=None)
    parser.add_argument("--output-dir", default="demo_outputs")
    return parser.parse_args()


def output_stem(index: int, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"speech_{now.strftime('%Y%m%d_%H%M%S')}_{index:03d}"


def synthesize_and_report(engine: TTSEngine, text: str, output_dir: Path, index: int) -> None:
    result = engine.synthesize(text)
    wav_path, mel_path = engine.save(result, output_dir, output_stem(index))
    print(f"Text: {text}")
    print(f"Audio: {result.audio_seconds:.2f} s | synthesis: {result.elapsed_seconds:.3f} s | RTF: {result.real_time_factor:.4f}")
    print(f"WAV: {wav_path}")
    print(f"Mel: {mel_path}\n")


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    print(f"Loading checkpoint: {checkpoint}")
    engine = TTSEngine(checkpoint, args.device, args.vocoder)
    print(f"Ready | device={engine.device} | vocoder={engine.vocoder_backend}")
    if "smoke_test" in checkpoint.parts:
        print("Warning: the smoke checkpoint only validates the pipeline and will not produce intelligible speech.")

    output_dir = Path(args.output_dir)
    if args.text:
        for index, text in enumerate(args.text, start=1):
            synthesize_and_report(engine, text, output_dir, index)
        return

    print("Enter an English sentence. Type q, quit, or exit to stop.\n")
    index = 1
    while True:
        try:
            text = input("TTS> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break
        if text.lower() in {"q", "quit", "exit"}:
            print("Stopped.")
            break
        if not text:
            continue
        synthesize_and_report(engine, text, output_dir, index)
        index += 1


if __name__ == "__main__":
    main()

