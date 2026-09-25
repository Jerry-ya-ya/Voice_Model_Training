"""Interactive menu for starting and continuing TTS training runs."""

from __future__ import annotations

import copy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from src.training.checkpoints import find_latest_checkpoint
from src.utils.config import load_config, save_config


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_console_encoding() -> None:
    """Use UTF-8 so Traditional Chinese prompts render in modern PowerShell."""
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8")


def ask_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def ask_int(label: str, default: int, minimum: int = 1) -> int:
    while True:
        value = ask_text(label, str(default))
        try:
            parsed = int(value)
            if parsed < minimum:
                raise ValueError
            return parsed
        except ValueError:
            print(f"請輸入大於或等於 {minimum} 的整數。")


def ask_float(label: str, default: float, minimum: float = 0.0) -> float:
    while True:
        value = ask_text(label, str(default))
        try:
            parsed = float(value)
            if parsed <= minimum:
                raise ValueError
            return parsed
        except ValueError:
            print(f"請輸入大於 {minimum} 的數字。")


def ask_optional_int(label: str, current: int | None) -> int | None:
    shown = "全部" if current is None else str(current)
    while True:
        value = input(f"{label} [{shown}]（輸入 none 代表不限制）: ").strip()
        if not value:
            return current
        if value.lower() in {"none", "all", "unlimited"}:
            return None
        try:
            parsed = int(value)
            if parsed < 1:
                raise ValueError
            return parsed
        except ValueError:
            print("請輸入正整數或 none。")


def ask_choice(label: str, choices: tuple[str, ...], default: str) -> str:
    while True:
        value = ask_text(f"{label} ({'/'.join(choices)})", default).lower()
        if value in choices:
            return value
        print(f"請選擇：{', '.join(choices)}")


def ask_yes_no(label: str, default: bool = True) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{marker}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("請輸入 y 或 n。")


def discover_runs(output_dir: str | Path = "runs") -> list[Path]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    runs = []
    for path in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        if not (path / "config.yaml").is_file():
            continue
        try:
            find_latest_checkpoint(path)
        except FileNotFoundError:
            continue
        runs.append(path)
    return runs


def choose_run(runs: list[Path]) -> Path | None:
    if not runs:
        print("找不到可以接續的實驗 checkpoint。")
        return None
    print("\n可接續的實驗：")
    for index, run_dir in enumerate(runs, start=1):
        latest = find_latest_checkpoint(run_dir)
        print(f"  {index}. {run_dir.name} — {latest.name}")
    print("  0. 返回")
    while True:
        selection = ask_int("選擇實驗", 1, minimum=0)
        if selection == 0:
            return None
        if selection <= len(runs):
            return runs[selection - 1]
        print("沒有這個選項。")


def configure_runtime(config: dict, *, new_training: bool) -> dict:
    configured = copy.deepcopy(config)
    training = configured["training"]
    data = configured["data"]
    training["device"] = ask_choice("運算裝置", ("auto", "cuda", "cpu"), str(training.get("device", "auto")))
    training["batch_size"] = ask_int("Batch size", int(training["batch_size"]))
    training["seed"] = ask_int("Random seed", int(training.get("seed", 42)), minimum=0)
    training["max_steps_per_epoch"] = ask_optional_int(
        "每個 epoch 最多 training steps", training.get("max_steps_per_epoch")
    )
    training["checkpoint_every"] = ask_int("每幾個 epoch 保存 checkpoint", int(training.get("checkpoint_every", 1)))
    training["sample_every"] = ask_int("每幾個 epoch 生成樣本", int(training.get("sample_every", 1)))
    data["max_items"] = ask_optional_int("最多使用多少筆資料", data.get("max_items"))
    data["num_workers"] = ask_int("DataLoader workers", int(data.get("num_workers", 0)), minimum=0)
    if new_training:
        training["epochs"] = ask_int("這次訓練 epochs", int(training["epochs"]))
        training["learning_rate"] = ask_float("Learning rate", float(training["learning_rate"]))
    return configured


def save_launch_config(config: dict, mode: str) -> Path:
    experiment = config["experiment"]
    run_dir = Path(experiment["output_dir"]) / experiment["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = run_dir / "launch_configs" / f"{mode}_{timestamp}.yaml"
    save_config(config, path)
    return path


def build_training_command(
    config_path: str | Path,
    *,
    additional_epochs: int | None = None,
    resume_learning_rate: float | None = None,
) -> list[str]:
    command = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(config_path)]
    if additional_epochs is not None:
        command.extend(["--continue-train", str(additional_epochs)])
    if resume_learning_rate is not None:
        command.extend(["--resume-learning-rate", str(resume_learning_rate)])
    return command


def print_summary(config: dict, *, additional_epochs: int | None, learning_rate_override: float | None) -> None:
    training = config["training"]
    data = config["data"]
    print("\n本次設定摘要")
    print(f"  實驗：{config['experiment']['name']}")
    print(f"  Device：{training['device']}")
    print(f"  Batch size：{training['batch_size']}")
    print(f"  Epochs：{additional_epochs if additional_epochs is not None else training['epochs']}")
    print(f"  Learning rate：{learning_rate_override if learning_rate_override is not None else training['learning_rate']}")
    print(f"  Max items：{data.get('max_items') or '全部'}")
    print(f"  Max steps/epoch：{training.get('max_steps_per_epoch') or '不限'}")


def launch(config: dict, mode: str, *, additional_epochs: int | None = None, resume_learning_rate: float | None = None) -> None:
    print_summary(config, additional_epochs=additional_epochs, learning_rate_override=resume_learning_rate)
    if not ask_yes_no("確認開始訓練？", default=True):
        print("已取消。")
        return
    config_path = save_launch_config(config, mode)
    command = build_training_command(
        config_path,
        additional_epochs=additional_epochs,
        resume_learning_rate=resume_learning_rate,
    )
    print(f"設定已保存：{config_path}", flush=True)
    print(f"執行：{subprocess.list2cmdline(command)}\n", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def new_training() -> None:
    config_path = Path(ask_text("基礎設定檔", "configs/mvp.yaml"))
    config = load_config(config_path)
    experiment = config["experiment"]
    experiment["name"] = ask_text("實驗名稱", str(experiment["name"]))
    config["data"]["root"] = ask_text("資料集路徑", str(config["data"]["root"]))
    run_dir = Path(experiment["output_dir"]) / experiment["name"]
    try:
        existing = find_latest_checkpoint(run_dir)
    except FileNotFoundError:
        existing = None
    if existing:
        print(f"此實驗已有 checkpoint：{existing}")
        print("為避免覆寫，請使用『再次訓練』或輸入新的實驗名稱。")
        return
    configured = configure_runtime(config, new_training=True)
    launch(configured, "new")


def continue_training() -> None:
    run_dir = choose_run(discover_runs())
    if run_dir is None:
        return
    config = load_config(run_dir / "config.yaml")
    latest = find_latest_checkpoint(run_dir)
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    print(f"\n最新 checkpoint：{latest}")
    print(f"已完成 epoch：{checkpoint['epoch']} | global step：{checkpoint['global_step']}")
    additional_epochs = ask_int("這一輪要再訓練幾個 epochs", 10)
    configured = configure_runtime(config, new_training=False)
    learning_rate_override = None
    restored_lr = float(checkpoint["optimizer"]["param_groups"][0]["lr"])
    print(f"Checkpoint learning rate：{restored_lr:.6g}")
    if ask_yes_no("這一輪要重設 learning rate？", default=False):
        learning_rate_override = ask_float("新的 learning rate", restored_lr)
    configured["training"]["learning_rate"] = learning_rate_override or restored_lr
    launch(
        configured,
        "continue",
        additional_epochs=additional_epochs,
        resume_learning_rate=learning_rate_override,
    )


def main() -> None:
    configure_console_encoding()
    while True:
        print("\nVoice Model Training CLI")
        print("  1. 全新訓練")
        print("  2. 再次訓練（接續最新 checkpoint）")
        print("  3. 離開")
        choice = ask_text("請選擇", "1")
        try:
            if choice == "1":
                new_training()
            elif choice == "2":
                continue_training()
            elif choice == "3":
                print("已離開。")
                return
            else:
                print("沒有這個選項。")
        except (FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError) as error:
            print(f"操作失敗：{error}")
        except KeyboardInterrupt:
            print("\n操作已取消。")


if __name__ == "__main__":
    main()
