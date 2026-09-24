"""Repeatable speed and memory benchmark for the TTS MVP.

The synthetic model benchmark avoids dataset and disk-I/O variance. Supplying a
checkpoint additionally measures real text-to-waveform inference and reports
the real-time factor (RTF).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

from src.data.text import CharacterTokenizer
from src.models.acoustic_model import AcousticModel
from src.models.vocoder import Vocoder
from src.training.losses import acoustic_loss
from src.utils.config import load_config
from src.utils.io import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TTS model throughput, memory, and real-time factor")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Used when --checkpoint is omitted")
    parser.add_argument("--checkpoint", default=None, help="Also benchmark text-to-waveform inference")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--token-length", type=int, default=64)
    parser.add_argument("--mel-frames", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--text", default="Hello, this is a text to speech benchmark.")
    parser.add_argument("--vocoder", choices=["hifigan", "griffin_lim"], default=None)
    parser.add_argument("--skip-training", action="store_true", help="Skip forward/backward/optimizer timing")
    parser.add_argument("--output-dir", default="benchmarks")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_times(milliseconds: list[float]) -> dict[str, float]:
    if not milliseconds:
        raise ValueError("At least one timing sample is required")
    ordered = sorted(milliseconds)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "mean_ms": statistics.fmean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[p95_index],
    }


def measure(operation: Callable[[], None], device: torch.device, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    synchronize(device)
    samples = []
    for _ in range(iterations):
        synchronize(device)
        started = time.perf_counter()
        operation()
        synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    return summarize_times(samples)


def environment_report(device: torch.device) -> dict:
    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(
            {
                "gpu": properties.name,
                "gpu_total_memory_mb": properties.total_memory / (1024**2),
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    else:
        report["cpu"] = platform.processor() or "unknown"
    return report


def load_model_and_config(args: argparse.Namespace, device: torch.device) -> tuple[AcousticModel, dict, CharacterTokenizer]:
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint["config"]
    else:
        config = load_config(args.config)
    tokenizer = CharacterTokenizer()
    model = AcousticModel(tokenizer.vocab_size, int(config["audio"]["n_mels"]), config["model"]).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    return model, config, tokenizer


def benchmark_synthetic_model(model: AcousticModel, config: dict, tokenizer: CharacterTokenizer, args: argparse.Namespace, device: torch.device) -> dict:
    batch_size = args.batch_size
    token_length = args.token_length
    mel_frames = args.mel_frames
    n_mels = int(config["audio"]["n_mels"])
    tokens = torch.randint(2, tokenizer.vocab_size, (batch_size, token_length), device=device)
    token_lengths = torch.full((batch_size,), token_length, dtype=torch.long, device=device)
    mel_lengths = torch.full((batch_size,), mel_frames, dtype=torch.long, device=device)
    targets = torch.randn(batch_size, mel_frames, n_mels, device=device)

    model.eval()

    @torch.inference_mode()
    def forward_step() -> None:
        model(tokens, token_lengths, mel_lengths)

    forward = measure(forward_step, device, args.warmup, args.iterations)
    forward["samples_per_second"] = batch_size / (forward["mean_ms"] / 1000)
    forward["mel_frames_per_second"] = batch_size * mel_frames / (forward["mean_ms"] / 1000)

    result = {
        "input": {"batch_size": batch_size, "token_length": token_length, "mel_frames": mel_frames},
        "forward": forward,
    }
    if not args.skip_training:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def training_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(tokens, token_lengths, mel_lengths)
            loss, _ = acoustic_loss(outputs, targets, token_lengths, mel_lengths, 0.1)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training = measure(training_step, device, args.warmup, args.iterations)
        training["samples_per_second"] = batch_size / (training["mean_ms"] / 1000)
        training["mel_frames_per_second"] = batch_size * mel_frames / (training["mean_ms"] / 1000)
        if device.type == "cuda":
            training["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        result["training_step"] = training
    return result


def benchmark_end_to_end(model: AcousticModel, config: dict, tokenizer: CharacterTokenizer, args: argparse.Namespace, device: torch.device) -> dict:
    tokens = torch.tensor([tokenizer.encode(args.text)], dtype=torch.long, device=device)
    token_lengths = torch.tensor([tokens.shape[1]], device=device)
    backend = args.vocoder or config["vocoder"]["backend"]
    vocoder = Vocoder(backend, config["audio"], device)
    model.eval()

    with torch.inference_mode():
        initial = model(tokens, token_lengths)
        frames = int(initial["mel_lengths"][0])
        mel = initial["mel"][0, :frames].transpose(0, 1)
        waveform = vocoder(mel)
    audio_seconds = waveform.numel() / int(config["audio"]["sample_rate"])

    @torch.inference_mode()
    def acoustic_step() -> None:
        model(tokens, token_lengths)

    @torch.inference_mode()
    def vocoder_step() -> None:
        vocoder(mel)

    @torch.inference_mode()
    def end_to_end_step() -> None:
        outputs = model(tokens, token_lengths)
        output_frames = int(outputs["mel_lengths"][0])
        vocoder(outputs["mel"][0, :output_frames].transpose(0, 1))

    acoustic = measure(acoustic_step, device, args.warmup, args.iterations)
    vocoder_result = measure(vocoder_step, device, args.warmup, args.iterations)
    end_to_end = measure(end_to_end_step, device, args.warmup, args.iterations)
    end_to_end["audio_seconds"] = audio_seconds
    end_to_end["real_time_factor"] = (end_to_end["mean_ms"] / 1000) / max(audio_seconds, 1e-9)
    end_to_end["times_faster_than_realtime"] = 1 / max(end_to_end["real_time_factor"], 1e-9)
    return {
        "text": args.text,
        "vocoder": backend,
        "generated_mel_frames": frames,
        "acoustic_model": acoustic,
        "vocoder_only": vocoder_result,
        "end_to_end": end_to_end,
    }


def print_summary(report: dict, output_path: Path) -> None:
    environment = report["environment"]
    model = report["model"]
    synthetic = report["synthetic"]
    print("\nTTS MVP Benchmark")
    print(f"Device: {environment.get('gpu', environment.get('cpu'))} ({environment['device']})")
    print(f"PyTorch: {environment['torch']} | CUDA build: {environment['torch_cuda_build']}")
    print(f"Parameters: {model['parameters']:,} ({model['trainable_parameters']:,} trainable)")
    print(f"Forward: {synthetic['forward']['mean_ms']:.2f} ms | {synthetic['forward']['samples_per_second']:.2f} samples/s")
    if "training_step" in synthetic:
        training = synthetic["training_step"]
        memory = f" | peak GPU memory {training['peak_gpu_memory_mb']:.1f} MB" if "peak_gpu_memory_mb" in training else ""
        print(f"Train step: {training['mean_ms']:.2f} ms | {training['samples_per_second']:.2f} samples/s{memory}")
    if "end_to_end" in report:
        result = report["end_to_end"]
        timing = result["end_to_end"]
        print(f"End-to-end ({result['vocoder']}): {timing['mean_ms']:.2f} ms for {timing['audio_seconds']:.2f} s audio")
        print(f"RTF: {timing['real_time_factor']:.4f} ({timing['times_faster_than_realtime']:.2f}x realtime)")
    print(f"JSON: {output_path}")


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be non-negative and --iterations must be at least 1")
    for name in ("batch_size", "token_length", "mel_frames"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")

    device = resolve_device(args.device)
    set_seed(42)
    model, config, tokenizer = load_model_and_config(args, device)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": environment_report(device),
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "config": config["model"],
            "checkpoint": args.checkpoint,
        },
    }
    report["synthetic"] = benchmark_synthetic_model(model, config, tokenizer, args, device)
    if args.checkpoint:
        # The synthetic training timing updates only this in-memory model, so
        # reload checkpoint weights before measuring output quality/speed.
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        report["end_to_end"] = benchmark_end_to_end(model, config, tokenizer, args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report, output_path)


if __name__ == "__main__":
    main()

