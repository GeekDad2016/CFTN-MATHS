from __future__ import annotations

import json
import time

from tools import watch_v2_progress


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pid_probe_detects_current_process():
    assert watch_v2_progress._pid_running(watch_v2_progress.os.getpid()) is True


def test_v2_progress_snapshot_and_render(tmp_path, monkeypatch):
    artifact = tmp_path / "artifacts"
    data = tmp_path / "data"
    multi = tmp_path / "multi"
    started = time.time() - 60
    _write_json(
        artifact / "pipeline_state.json",
        {
            "state": "running",
            "pid": 999999,
            "current_stage": "train_math",
            "current_stage_index": 2,
            "stage_count": 17,
            "stages": {"train_math": {"state": "running", "started_unix": started}},
        },
    )
    _write_json(
        artifact / "math" / "status.json",
        {
            "state": "running",
            "pid": 999998,
            "epoch": 1,
            "global_step": 250,
            "elapsed_seconds": 30,
            "metrics": {
                "epoch_batch_completed": 250,
                "epoch_batches_total": 12500,
                "train_loss_so_far": 1.25,
                "steps_per_second": 50.0,
            },
        },
    )
    (artifact / "math" / "metrics.jsonl").write_text(
        json.dumps(
            {
                "epoch": 1,
                "global_step": 12500,
                "validation": {
                    "loss": 0.5,
                    "teacher_forced_sequence_accuracy": 0.75,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        artifact / "math" / "wandb_run.json",
        {"url": "https://wandb.example/run/test"},
    )
    monkeypatch.setattr(watch_v2_progress, "_process_lines", lambda: ["trainer"])
    monkeypatch.setattr(
        watch_v2_progress,
        "_gpu_status",
        lambda: [
            {
                "index": 0,
                "name": "GPU",
                "utilization_percent": 80,
                "memory_used_mib": 1000,
                "memory_total_mib": 80000,
                "temperature_c": 50,
                "power_w": 200.0,
            }
        ],
    )

    snapshot = watch_v2_progress.collect_snapshot(
        artifact,
        data_root=data,
        multi_data_root=multi,
    )
    rendered = watch_v2_progress.render_markdown(snapshot)

    assert "2/17" in rendered
    assert "250/12,500" in rendered
    assert "teacher_forced_sequence_accuracy" in rendered
    assert "https://wandb.example/run/test" in rendered
    assert "80%" in rendered


def test_v2_progress_renders_pipeline_error_tail(tmp_path, monkeypatch):
    artifact = tmp_path / "artifacts"
    _write_json(
        artifact / "pipeline_state.json",
        {
            "state": "error",
            "pid": 999999,
            "current_stage": "train_math",
            "current_stage_index": 2,
            "stage_count": 17,
            "error": "failure",
            "stages": {"train_math": {"state": "error"}},
        },
    )
    log = artifact / "pipeline_logs" / "train_math.stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("Traceback\nRuntimeError: failure\n", encoding="utf-8")
    monkeypatch.setattr(watch_v2_progress, "_process_lines", lambda: [])
    monkeypatch.setattr(watch_v2_progress, "_gpu_status", lambda: [])

    snapshot = watch_v2_progress.collect_snapshot(
        artifact,
        data_root=tmp_path / "data",
        multi_data_root=tmp_path / "multi",
    )
    rendered = watch_v2_progress.render_markdown(snapshot)

    assert "ERROR — newest stderr tail" in rendered
    assert "RuntimeError: failure" in rendered
