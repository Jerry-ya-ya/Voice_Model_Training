from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cli import (
    CONFIG_DIR,
    PROJECT_ROOT,
    build_training_command,
    category_directory,
    config_label,
    next_run_number,
    numbered_run_directories,
    save_launch_config,
    scan_configs,
)
from src.training.checkpoints import find_latest_checkpoint
from src.utils.config import load_config


STATIC_DIR = Path(__file__).resolve().parent / "static"


class RuntimeOptions(BaseModel):
    device: Literal["auto", "cuda", "cpu"] | None = None
    batch_size: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    max_steps_per_epoch: int | None = Field(default=None, ge=1)
    checkpoint_every: int | None = Field(default=None, ge=1)
    sample_every: int | None = Field(default=None, ge=1)
    max_items: int | None = Field(default=None, ge=1)
    num_workers: int | None = Field(default=None, ge=0)
    dataset_root: str | None = None


class NewTrainingRequest(RuntimeOptions):
    config_name: str
    epochs: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)


class ContinueTrainingRequest(RuntimeOptions):
    config_name: str
    source_run: int = Field(ge=1)
    additional_epochs: int = Field(ge=1)
    learning_rate: float | None = Field(default=None, gt=0)


@dataclass
class TrainingJob:
    id: str
    mode: str
    category: str
    run_number: int
    run_dir: str
    command: list[str]
    created_at: str
    status: str = "starting"
    return_code: int | None = None
    finished_at: str | None = None
    stop_requested: bool = False
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=4000))
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "mode": self.mode,
            "category": self.category,
            "run_number": self.run_number,
            "run_dir": self.run_dir,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "return_code": self.return_code,
            "log": "".join(self.logs),
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, TrainingJob] = {}
        self.lock = threading.Lock()

    def start(self, command: list[str], *, mode: str, category: str, run_number: int, run_dir: Path) -> TrainingJob:
        with self.lock:
            if any(job.status in {"starting", "running", "stopping"} for job in self.jobs.values()):
                raise RuntimeError("目前已有訓練正在執行，請等待完成或先停止該工作。")
            job = TrainingJob(
                id=uuid.uuid4().hex,
                mode=mode,
                category=category,
                run_number=run_number,
                run_dir=str(run_dir),
                command=command,
                created_at=_utc_now(),
            )
            self.jobs[job.id] = job

        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "web_training.log"
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        try:
            job.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception:
            log_handle.close()
            with self.lock:
                job.status = "failed"
                job.finished_at = _utc_now()
            raise

        job.status = "running"
        threading.Thread(target=self._collect_output, args=(job, log_handle), daemon=True).start()
        return job

    def _collect_output(self, job: TrainingJob, log_handle) -> None:
        assert job.process is not None
        assert job.process.stdout is not None
        try:
            for line in job.process.stdout:
                job.logs.append(line)
                log_handle.write(line)
            return_code = job.process.wait()
        finally:
            log_handle.close()

        with self.lock:
            job.return_code = return_code
            job.finished_at = _utc_now()
            if job.stop_requested:
                job.status = "stopped"
            else:
                job.status = "completed" if return_code == 0 else "failed"

    def list(self) -> list[dict]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)
            return [job.public() for job in jobs]

    def get(self, job_id: str) -> TrainingJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def stop(self, job_id: str) -> TrainingJob:
        job = self.get(job_id)
        with self.lock:
            if job.status not in {"starting", "running"} or job.process is None:
                raise RuntimeError("這個訓練工作目前無法停止。")
            job.stop_requested = True
            job.status = "stopping"
            job.process.terminate()
        return job


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_config(config_name: str) -> Path:
    matches = [path for path in scan_configs(CONFIG_DIR) if path.name == config_name]
    if not matches:
        raise HTTPException(status_code=404, detail="找不到指定的 config。")
    return matches[0]


def _config_payload(path: Path) -> dict:
    config = load_config(path)
    training = config["training"]
    data = config["data"]
    model = config["model"]
    category_dir = category_directory(path, config)
    return {
        "name": path.name,
        "category": path.stem,
        "label": config_label(path),
        "next_run_number": next_run_number(category_dir),
        "model": {
            "hidden_dim": model["hidden_dim"],
            "encoder_layers": model["encoder_layers"],
            "decoder_layers": model["decoder_layers"],
        },
        "defaults": {
            "dataset_root": str(data["root"]),
            "device": str(training.get("device", "auto")),
            "batch_size": int(training["batch_size"]),
            "seed": int(training.get("seed", 42)),
            "max_steps_per_epoch": training.get("max_steps_per_epoch"),
            "checkpoint_every": int(training.get("checkpoint_every", 1)),
            "sample_every": int(training.get("sample_every", 1)),
            "max_items": data.get("max_items"),
            "num_workers": int(data.get("num_workers", 0)),
            "epochs": int(training["epochs"]),
            "learning_rate": float(training["learning_rate"]),
        },
    }


def _apply_runtime_options(config: dict, request: RuntimeOptions) -> dict:
    configured = deepcopy(config)
    training = configured["training"]
    data = configured["data"]
    provided = request.model_dump(exclude_unset=True)
    for key in ("device", "batch_size", "seed", "max_steps_per_epoch", "checkpoint_every", "sample_every"):
        if key in provided:
            training[key] = provided[key]
    for key in ("max_items", "num_workers"):
        if key in provided:
            data[key] = provided[key]
    if "dataset_root" in provided:
        if not provided["dataset_root"].strip():
            raise HTTPException(status_code=422, detail="資料集路徑不可為空白。")
        data["root"] = provided["dataset_root"].strip()
    return configured


def _load_checkpoint_summary(checkpoint_path: Path) -> tuple[dict, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    learning_rate = float(checkpoint["optimizer"]["param_groups"][0]["lr"])
    return checkpoint, learning_rate


jobs = JobManager()
app = FastAPI(title="Voice Model Training", version="1.0.0")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", response_class=FileResponse)
def index() -> Path:
    return STATIC_DIR / "index.html"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "port": 4000, "cuda_available": torch.cuda.is_available()}


@app.get("/api/configs")
def list_configs() -> list[dict]:
    return [_config_payload(path) for path in scan_configs(CONFIG_DIR)]


@app.get("/api/configs/{config_name}/runs")
def list_runs(config_name: str) -> list[dict]:
    config_path = _find_config(config_name)
    category_dir = category_directory(config_path, load_config(config_path))
    result = []
    for run_dir in numbered_run_directories(category_dir):
        checkpoint_path = find_latest_checkpoint(run_dir)
        try:
            checkpoint, learning_rate = _load_checkpoint_summary(checkpoint_path)
            result.append(
                {
                    "run_number": int(run_dir.name),
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_path": str(checkpoint_path),
                    "epoch": int(checkpoint["epoch"]),
                    "global_step": int(checkpoint["global_step"]),
                    "learning_rate": learning_rate,
                }
            )
        except (KeyError, RuntimeError, ValueError) as error:
            result.append({"run_number": int(run_dir.name), "checkpoint": checkpoint_path.name, "error": str(error)})
    return result


@app.post("/api/train/new", status_code=202)
def start_new_training(request: NewTrainingRequest) -> dict:
    config_path = _find_config(request.config_name)
    base_config = load_config(config_path)
    category = config_path.stem
    category_dir = category_directory(config_path, base_config)
    run_number = next_run_number(category_dir)
    config = _apply_runtime_options(base_config, request)
    config["experiment"]["output_dir"] = str(category_dir)
    config["experiment"]["name"] = str(run_number)
    config["experiment"]["source_config"] = str(config_path)
    config["experiment"]["run_number"] = run_number
    if request.epochs is not None:
        config["training"]["epochs"] = request.epochs
    if request.learning_rate is not None:
        config["training"]["learning_rate"] = request.learning_rate
    launch_config = save_launch_config(config, "web_new")
    run_dir = category_dir / str(run_number)
    command = build_training_command(launch_config)
    try:
        job = jobs.start(command, mode="new", category=category, run_number=run_number, run_dir=run_dir)
    except (OSError, RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return job.public()


@app.post("/api/train/continue", status_code=202)
def start_continued_training(request: ContinueTrainingRequest) -> dict:
    config_path = _find_config(request.config_name)
    base_config = load_config(config_path)
    category = config_path.stem
    category_dir = category_directory(config_path, base_config)
    source_run = category_dir / str(request.source_run)
    valid_sources = numbered_run_directories(category_dir)
    if source_run not in valid_sources:
        raise HTTPException(status_code=404, detail="找不到可接續的訓練編號或 checkpoint。")
    latest = find_latest_checkpoint(source_run)
    checkpoint, restored_learning_rate = _load_checkpoint_summary(latest)
    source_config_path = source_run / "config.yaml"
    if not source_config_path.is_file():
        raise HTTPException(status_code=404, detail="來源訓練缺少 config.yaml。")

    destination_number = next_run_number(category_dir)
    config = _apply_runtime_options(load_config(source_config_path), request)
    config["experiment"]["output_dir"] = str(category_dir)
    config["experiment"]["name"] = str(destination_number)
    config["experiment"]["run_number"] = destination_number
    config["experiment"]["parent_run_number"] = request.source_run
    config["experiment"]["resume_checkpoint"] = str(latest)
    config["training"]["epochs"] = int(checkpoint["epoch"]) + request.additional_epochs
    config["training"]["learning_rate"] = request.learning_rate or restored_learning_rate
    launch_config = save_launch_config(config, "web_continue")
    run_dir = category_dir / str(destination_number)
    command = build_training_command(
        launch_config,
        resume_checkpoint=latest,
        resume_learning_rate=request.learning_rate,
    )
    try:
        job = jobs.start(
            command,
            mode="continue",
            category=category,
            run_number=destination_number,
            run_dir=run_dir,
        )
    except (OSError, RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return job.public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return jobs.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return jobs.get(job_id).public()
    except KeyError as error:
        raise HTTPException(status_code=404, detail="找不到訓練工作。") from error


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    try:
        return jobs.stop(job_id).public()
    except KeyError as error:
        raise HTTPException(status_code=404, detail="找不到訓練工作。") from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
