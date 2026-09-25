from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.models.vocoder import Vocoder
from src.training.losses import acoustic_loss
from src.utils.config import save_config
from src.utils.io import append_metrics, save_mel_plot, save_training_plot, save_waveform


class Trainer:
    def __init__(self, model, train_loader, validation_loader, config: dict, tokenizer, device: torch.device) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        training = config["training"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0)))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        self.global_step = 0
        self.start_epoch = 1
        self.best_loss = float("inf")
        self.history = []
        experiment = config["experiment"]
        self.run_dir = Path(experiment["output_dir"]) / experiment["name"]
        for folder in ("checkpoints", "samples", "plots", "logs"):
            (self.run_dir / folder).mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.yaml")

    def _move(self, batch: dict) -> dict:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        count = 0
        max_steps = self.config["training"].get("max_steps_per_epoch")
        for batch in loader:
            batch = self._move(batch)
            with torch.set_grad_enabled(train):
                outputs = self.model(batch["tokens"], batch["token_lengths"], batch["mel_lengths"])
                loss, _ = acoustic_loss(outputs, batch["mels"], batch["token_lengths"], batch["mel_lengths"], float(self.config["training"].get("duration_loss_weight", 0.1)))
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["training"].get("grad_clip", 1.0)))
                    self.optimizer.step()
                    self.global_step += 1
            total += loss.item()
            count += 1
            if max_steps and count >= int(max_steps):
                break
        return total / max(count, 1)

    @torch.inference_mode()
    def _save_sample(self, epoch: int) -> None:
        batch = self._move(next(iter(self.validation_loader)))
        outputs = self.model(batch["tokens"][:1], batch["token_lengths"][:1], batch["mel_lengths"][:1])
        mel = outputs["mel"][0, : batch["mel_lengths"][0]].cpu()
        save_mel_plot(mel, self.run_dir / "samples" / f"epoch_{epoch:04d}.png")
        vocoder = Vocoder(self.config["vocoder"]["backend"], self.config["audio"], self.device)
        waveform = vocoder(mel.transpose(0, 1))
        save_waveform(waveform, self.run_dir / "samples" / f"epoch_{epoch:04d}.wav", self.config["audio"]["sample_rate"])

    def save_checkpoint(self, epoch: int, name: str) -> Path:
        path = self.run_dir / "checkpoints" / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "best_validation_loss": self.best_loss,
                "config": self.config,
                "vocab_size": self.tokenizer.vocab_size,
            },
            path,
        )
        return path

    def _load_existing_history(self) -> None:
        metrics_path = self.run_dir / "logs" / "metrics.csv"
        if not metrics_path.is_file():
            return
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            self.history = [
                {
                    "epoch": int(row["epoch"]),
                    "global_step": int(row["global_step"]),
                    "train_loss": float(row["train_loss"]),
                    "validation_loss": float(row["validation_loss"]),
                    "learning_rate": float(row["learning_rate"]),
                }
                for row in rows
            ]

    def resume(self, path: str | Path, learning_rate_override: float | None = None) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if learning_rate_override is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate_override
            self.scheduler.base_lrs = [learning_rate_override for _ in self.optimizer.param_groups]
            self.scheduler._last_lr = [learning_rate_override for _ in self.optimizer.param_groups]
            self.config["training"]["learning_rate"] = learning_rate_override
        self.global_step = int(checkpoint["global_step"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self._load_existing_history()
        historical_best = min((row["validation_loss"] for row in self.history), default=float("inf"))
        self.best_loss = float(checkpoint.get("best_validation_loss", historical_best))
        learning_rate = self.optimizer.param_groups[0]["lr"]
        print(f"Resumed {path} at epoch {self.start_epoch}, global step {self.global_step}, learning rate {learning_rate:.6g}")

    def fit(self) -> Path:
        best_path = self.run_dir / "checkpoints" / "best.pt"
        epochs = int(self.config["training"]["epochs"])
        if self.start_epoch > epochs:
            raise ValueError(
                f"Checkpoint already completed epoch {self.start_epoch - 1}, but config only requests {epochs} epochs. "
                "Increase training.epochs or use --continue-train ADDITIONAL_EPOCHS."
            )
        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            validation_loss = self._run_epoch(self.validation_loader, train=False)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "global_step": self.global_step, "train_loss": train_loss, "validation_loss": validation_loss, "learning_rate": learning_rate}
            append_metrics(self.run_dir / "logs" / "metrics.csv", row)
            self.history.append(row)
            save_training_plot(self.history, self.run_dir / "plots" / "training.png")
            print(" | ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))
            self.scheduler.step()
            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                best_path = self.save_checkpoint(epoch, "best.pt")
            if epoch % int(self.config["training"].get("checkpoint_every", 1)) == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch:04d}.pt")
            if epoch % int(self.config["training"].get("sample_every", 1)) == 0:
                self._save_sample(epoch)
        return best_path
