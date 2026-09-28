from __future__ import annotations

import sys
import time
from pathlib import Path

import cli
import torch
from fastapi.testclient import TestClient

import webapp.app as web
from src.utils.config import save_config


def sample_config() -> dict:
    return {
        "experiment": {"name": "test", "output_dir": "runs"},
        "data": {"root": "data/test", "max_items": None, "num_workers": 0},
        "audio": {"sample_rate": 22050, "n_mels": 80},
        "model": {"hidden_dim": 32, "encoder_layers": 1, "decoder_layers": 1},
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "seed": 42,
            "checkpoint_every": 1,
            "sample_every": 1,
            "epochs": 3,
            "learning_rate": 0.001,
        },
        "vocoder": {"backend": "griffin_lim"},
    }


def configure_temporary_project(tmp_path: Path, monkeypatch) -> Path:
    config_dir = tmp_path / "configs"
    config_path = config_dir / "small.yaml"
    save_config(sample_config(), config_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(web, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(web, "CONFIG_DIR", config_dir)
    return config_path


def fake_start(captured: dict):
    def start(command, *, mode, category, run_number, run_dir):
        captured.update(command=command, mode=mode, category=category, run_number=run_number, run_dir=run_dir)
        return web.TrainingJob(
            id="test-job",
            mode=mode,
            category=category,
            run_number=run_number,
            run_dir=str(run_dir),
            command=command,
            created_at="2026-09-27T00:00:00+00:00",
            status="running",
        )

    return start


def test_health_and_frontend_are_served() -> None:
    client = TestClient(web.app)
    health = client.get("/api/health")
    page = client.get("/")

    assert health.status_code == 200
    assert health.json()["port"] == 4000
    assert page.status_code == 200
    assert "Voice Model Training" in page.text


def test_new_training_uses_next_numbered_run(tmp_path: Path, monkeypatch) -> None:
    configure_temporary_project(tmp_path, monkeypatch)
    (tmp_path / "runs" / "small" / "1").mkdir(parents=True)
    captured = {}
    monkeypatch.setattr(web.jobs, "start", fake_start(captured))

    response = TestClient(web.app).post(
        "/api/train/new",
        json={
            "config_name": "small.yaml",
            "dataset_root": "data/LJSpeech-1.1",
            "device": "cpu",
            "batch_size": 1,
            "epochs": 2,
            "learning_rate": 0.0003,
        },
    )

    assert response.status_code == 202
    assert response.json()["run_number"] == 2
    assert captured["run_dir"] == tmp_path / "runs" / "small" / "2"
    assert (captured["run_dir"] / "launch_configs").is_dir()


def test_continued_training_reads_source_and_writes_next_run(tmp_path: Path, monkeypatch) -> None:
    configure_temporary_project(tmp_path, monkeypatch)
    category_dir = tmp_path / "runs" / "small"
    source_run = category_dir / "1"
    checkpoint_path = source_run / "checkpoints" / "epoch_0003.pt"
    checkpoint_path.parent.mkdir(parents=True)
    save_config(sample_config() | {"experiment": {"name": "1", "output_dir": str(category_dir)}}, source_run / "config.yaml")
    torch.save(
        {
            "epoch": 3,
            "global_step": 9,
            "optimizer": {"param_groups": [{"lr": 0.0007}]},
        },
        checkpoint_path,
    )
    captured = {}
    monkeypatch.setattr(web.jobs, "start", fake_start(captured))

    response = TestClient(web.app).post(
        "/api/train/continue",
        json={
            "config_name": "small.yaml",
            "source_run": 1,
            "additional_epochs": 4,
            "device": "cpu",
            "batch_size": 1,
        },
    )

    assert response.status_code == 202
    assert response.json()["run_number"] == 2
    assert captured["run_dir"] == category_dir / "2"
    assert "--resume" in captured["command"]
    launch_config = next((category_dir / "2" / "launch_configs").glob("*.yaml"))
    saved = web.load_config(launch_config)
    assert saved["experiment"]["parent_run_number"] == 1
    assert saved["training"]["epochs"] == 7
    assert saved["training"]["learning_rate"] == 0.0007


def test_job_manager_captures_subprocess_output(tmp_path: Path) -> None:
    manager = web.JobManager()
    job = manager.start(
        [sys.executable, "-c", "print('epoch=1')"],
        mode="new",
        category="smoke",
        run_number=1,
        run_dir=tmp_path / "run",
    )
    assert job.process is not None
    job.process.wait(timeout=10)
    attempts = 100
    while job.status == "running" and attempts:
        time.sleep(0.01)
        attempts -= 1

    assert job.status == "completed"
    assert "epoch=1" in job.public()["log"]
    assert (tmp_path / "run" / "web_training.log").is_file()
