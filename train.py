import argparse

from src.data.dataset import create_dataloaders
from src.models.acoustic_model import AcousticModel
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the educational TTS acoustic model")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["training"]["seed"]))
    device = resolve_device(config["training"].get("device", "auto"))
    data_config = {**config["data"], "batch_size": config["training"]["batch_size"]}
    train_loader, validation_loader, tokenizer = create_dataloaders(data_config, config["audio"], int(config["training"]["seed"]))
    model = AcousticModel(tokenizer.vocab_size, int(config["audio"]["n_mels"]), config["model"])
    trainer = Trainer(model, train_loader, validation_loader, config, tokenizer, device)
    if args.resume:
        trainer.resume(args.resume)
    print(f"Device: {device} | train batches: {len(train_loader)} | validation batches: {len(validation_loader)}")
    print(f"Best checkpoint: {trainer.fit()}")


if __name__ == "__main__":
    main()

