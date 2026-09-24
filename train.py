import argparse
from pathlib import Path

import torch

from src.data.dataset import create_dataloaders
from src.models.acoustic_model import AcousticModel
from src.training.checkpoints import find_latest_checkpoint
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the educational TTS acoustic model")
    parser.add_argument("--config", default="configs/mvp.yaml")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", default=None, help="Resume from an explicit checkpoint path")
    resume_group.add_argument(
        "--continue-train",
        nargs="?",
        const=1,
        type=positive_integer,
        metavar="ADDITIONAL_EPOCHS",
        help="Auto-resume the latest checkpoint and train more epochs (default: 1)",
    )
    return parser.parse_args()


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def prepare_continuation(config: dict, additional_epochs: int) -> Path:
    experiment = config["experiment"]
    run_dir = Path(experiment["output_dir"]) / experiment["name"]
    checkpoint_path = find_latest_checkpoint(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed_epoch = int(checkpoint["epoch"])
    config["training"]["epochs"] = completed_epoch + additional_epochs
    print(
        f"Continue training: {checkpoint_path} | completed epoch {completed_epoch} "
        f"| additional epochs {additional_epochs} | target epoch {config['training']['epochs']}"
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resume_path = args.resume
    if args.continue_train is not None:
        resume_path = prepare_continuation(config, args.continue_train)
    set_seed(int(config["training"]["seed"]))
    device = resolve_device(config["training"].get("device", "auto"))
    data_config = {**config["data"], "batch_size": config["training"]["batch_size"]}
    train_loader, validation_loader, tokenizer = create_dataloaders(data_config, config["audio"], int(config["training"]["seed"]))
    model = AcousticModel(tokenizer.vocab_size, int(config["audio"]["n_mels"]), config["model"])
    trainer = Trainer(model, train_loader, validation_loader, config, tokenizer, device)
    if resume_path:
        trainer.resume(resume_path)
    print(f"Device: {device} | train batches: {len(train_loader)} | validation batches: {len(validation_loader)}")
    print(f"Best checkpoint: {trainer.fit()}")


if __name__ == "__main__":
    main()
