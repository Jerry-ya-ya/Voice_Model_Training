import argparse

from src.synthesis import TTSEngine


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
    engine = TTSEngine(args.checkpoint, args.device, args.vocoder)
    result = engine.synthesize(args.text)
    wav_path, mel_path = engine.save(result, args.output_dir)
    print(f"Input text: {args.text}")
    print(f"Device: {engine.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Vocoder: {engine.vocoder_backend}")
    print(f"Generated audio duration: {result.audio_seconds:.2f} seconds")
    print(f"Synthesis time: {result.elapsed_seconds:.3f} seconds | RTF: {result.real_time_factor:.4f}")
    print(f"Output path: {wav_path}")
    print(f"Mel plot: {mel_path}")


if __name__ == "__main__":
    main()
