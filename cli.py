"""Arrow-key menu for starting and continuing organized TTS training runs."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import torch

from src.training.checkpoints import find_latest_checkpoint
from src.utils.config import load_config, save_config


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "configs"
T = TypeVar("T")


def configure_console_encoding() -> None:
    """Use UTF-8 so Traditional Chinese prompts render in modern PowerShell."""
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8")


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_arrow_key() -> str:
    if os.name == "nt":
        import msvcrt

        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "other")
        if key == "\r":
            return "enter"
        if key == "\x1b":
            return "escape"
        return "other"

    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        key = sys.stdin.read(1)
        if key == "\x1b":
            sequence = sys.stdin.read(2)
            return {"[A": "up", "[B": "down"}.get(sequence, "escape")
        if key in {"\r", "\n"}:
            return "enter"
        return "other"
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def select_menu(title: str, options: list[tuple[str, T]], default_index: int = 0) -> T:
    """Select with arrow keys; use a numbered fallback for redirected input/tests."""
    if not options:
        raise ValueError(f"{title} 沒有可選項目")
    index = min(max(default_index, 0), len(options) - 1)
    if not _is_interactive_terminal():
        print(f"\n{title}：")
        for option_index, (label, _) in enumerate(options, start=1):
            print(f"  {option_index}. {label}")
        while True:
            value = input(f"請選擇 [1-{len(options)}]（預設 {index + 1}）: ").strip()
            if not value:
                return options[index][1]
            try:
                selected = int(value) - 1
                if 0 <= selected < len(options):
                    return options[selected][1]
            except ValueError:
                pass
            print("沒有這個選項。")

    last_width = 0
    while True:
        label = options[index][0]
        line = f"{title}（↑/↓ 選擇，Enter 確認）: ▶ {label}"
        padding = " " * max(last_width - len(line), 0)
        sys.stdout.write(f"\r{line}{padding}")
        sys.stdout.flush()
        last_width = max(last_width, len(line))
        key = _read_arrow_key()
        if key == "up":
            index = (index - 1) % len(options)
        elif key == "down":
            index = (index + 1) % len(options)
        elif key == "enter":
            print()
            return options[index][1]
        elif key == "escape":
            print()
            return options[-1][1]


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


def ask_yes_no(label: str, default: bool = True) -> bool:
    options = [("是", True), ("否", False)] if default else [("否", False), ("是", True)]
    return select_menu(label, options)


def scan_configs(config_dir: str | Path = CONFIG_DIR) -> list[Path]:
    root = Path(config_dir)
    if not root.is_dir():
        return []
    return sorted((*root.glob("*.yaml"), *root.glob("*.yml")), key=lambda path: path.name.lower())


def config_label(path: Path) -> str:
    try:
        config = load_config(path)
        model = config["model"]
        training = config["training"]
        return (
            f"{path.name} — hidden {model['hidden_dim']}, "
            f"layers {model['encoder_layers']}+{model['decoder_layers']}, batch {training['batch_size']}"
        )
    except (KeyError, TypeError):
        return path.name


def choose_config() -> Path | None:
    configs = scan_configs()
    options = [(config_label(path), path) for path in configs]
    options.append(("返回", None))
    return select_menu("選擇 Config", options)


def category_directory(config_path: Path, config: dict) -> Path:
    output_root = Path(config["experiment"].get("output_dir", "runs"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root / config_path.stem


def numbered_run_directories(category_dir: str | Path, require_checkpoint: bool = True) -> list[Path]:
    root = Path(category_dir)
    if not root.is_dir():
        return []
    runs = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.isdigit() or int(path.name) < 1:
            continue
        if require_checkpoint:
            try:
                find_latest_checkpoint(path)
            except FileNotFoundError:
                continue
        runs.append(path)
    return sorted(runs, key=lambda path: int(path.name))


def next_run_number(category_dir: str | Path) -> int:
    existing = numbered_run_directories(category_dir, require_checkpoint=False)
    return max((int(path.name) for path in existing), default=0) + 1


def choose_numbered_run(category_dir: Path) -> Path | None:
    runs = numbered_run_directories(category_dir)
    if not runs:
        print(f"{category_dir} 尚無可接續的訓練。")
        return None
    options = []
    for run_dir in runs:
        latest = find_latest_checkpoint(run_dir)
        options.append((f"第 {run_dir.name} 次訓練 — {latest.name}", run_dir))
    options.append(("返回", None))
    return select_menu(f"{category_dir.name}：選擇訓練編號", options)


def configure_runtime(config: dict, *, new_training: bool) -> dict:
    configured = copy.deepcopy(config)
    training = configured["training"]
    data = configured["data"]
    devices = [("自動選擇", "auto"), ("CUDA GPU", "cuda"), ("CPU", "cpu")]
    default_device = str(training.get("device", "auto"))
    default_index = next((i for i, (_, value) in enumerate(devices) if value == default_device), 0)
    training["device"] = select_menu("運算裝置", devices, default_index)
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
    run_dir = Path(experiment["output_dir"]) / str(experiment["name"])
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


def print_summary(config: dict, *, category: str, additional_epochs: int | None, learning_rate_override: float | None) -> None:
    training = config["training"]
    data = config["data"]
    print("\n本次設定摘要")
    print(f"  Config 分類：{category}")
    print(f"  第幾次訓練：{config['experiment']['name']}")
    print(f"  輸出：{Path(config['experiment']['output_dir']) / str(config['experiment']['name'])}")
    print(f"  Device：{training['device']}")
    print(f"  Batch size：{training['batch_size']}")
    print(f"  Epochs：{additional_epochs if additional_epochs is not None else training['epochs']}")
    print(f"  Learning rate：{learning_rate_override if learning_rate_override is not None else training['learning_rate']}")
    print(f"  Max items：{data.get('max_items') or '全部'}")
    print(f"  Max steps/epoch：{training.get('max_steps_per_epoch') or '不限'}")


def launch(
    config: dict,
    mode: str,
    *,
    category: str,
    additional_epochs: int | None = None,
    resume_learning_rate: float | None = None,
) -> None:
    print_summary(
        config,
        category=category,
        additional_epochs=additional_epochs,
        learning_rate_override=resume_learning_rate,
    )
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
    config_path = choose_config()
    if config_path is None:
        return
    config = load_config(config_path)
    category = config_path.stem
    category_dir = category_directory(config_path, config)
    run_number = next_run_number(category_dir)
    config["experiment"]["output_dir"] = str(category_dir)
    config["experiment"]["name"] = str(run_number)
    config["experiment"]["source_config"] = str(config_path)
    config["experiment"]["run_number"] = run_number
    config["data"]["root"] = ask_text("資料集路徑", str(config["data"]["root"]))
    configured = configure_runtime(config, new_training=True)
    launch(configured, "new", category=category)


def continue_training() -> None:
    config_path = choose_config()
    if config_path is None:
        return
    base_config = load_config(config_path)
    category = config_path.stem
    run_dir = choose_numbered_run(category_directory(config_path, base_config))
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
        category=category,
        additional_epochs=additional_epochs,
        resume_learning_rate=learning_rate_override,
    )


def main() -> None:
    configure_console_encoding()
    options = [
        ("全新訓練", "new"),
        ("再次訓練（接續指定 Config／訓練編號）", "continue"),
        ("離開", "exit"),
    ]
    while True:
        print("\nVoice Model Training CLI")
        choice = select_menu("主選單", options)
        try:
            if choice == "new":
                new_training()
            elif choice == "continue":
                continue_training()
            else:
                print("已離開。")
                return
        except (FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError) as error:
            print(f"操作失敗：{error}")
        except KeyboardInterrupt:
            print("\n操作已取消。")


if __name__ == "__main__":
    main()
