from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.v1_3_config import V13PrerequisiteError, audit_v1_2_pass, load_v1_3_config


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def pipeline_command(config_path: Path, *, device: str, wandb: bool) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.run_v1_3_experiment",
        "--config",
        str(config_path.resolve()),
        "--device",
        device,
        "--execute",
    ]
    if wandb:
        command.append("--wandb")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch V1.3 exactly once, only after a sealed passing V1.2 report"
    )
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_v1_3_config(config_path)
    root = Path(config["paths"]["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    continuation_path = root / "continuation_status.json"
    continuation_lock = root / "continuation.lock"
    continuation_pid = root / "continuation.pid"
    pipeline_status_path = root / "pipeline_status.json"
    existing = _read(pipeline_status_path)
    if existing and existing.get("state") in {"running", "completed"}:
        raise RuntimeError(
            f"V1.3 pipeline is already {existing['state']}; refusing duplicate continuation"
        )
    try:
        descriptor = os.open(continuation_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("V1.3 continuation lock exists") from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{os.getpid()}\n")
    continuation_pid.write_text(f"{os.getpid()}\n", encoding="ascii")
    started_at = time.time()
    poll_seconds = max(30, int(args.poll_seconds))
    v1_2_status_path = Path(config["paths"]["v1_2_pipeline_status"])
    try:
        while True:
            v1_2_status = _read(v1_2_status_path) or {}
            observed = str(v1_2_status.get("state", "missing"))
            atomic_json_dump(
                {
                    "format": "cftn_text_v1_3_continuation_status_v1",
                    "state": "waiting_for_v1_2_sealed_pass",
                    "pid": os.getpid(),
                    "v1_2_state": observed,
                    "v1_2_stage": v1_2_status.get("current_stage"),
                    "v1_2_stage_index": v1_2_status.get("stage_index"),
                    "v1_3_revision_sha256": config["_meta"]["sha256"],
                    "required_v1_2_completed_stages": config["prerequisite"][
                        "required_completed_stages"
                    ],
                    "poll_seconds": poll_seconds,
                    "elapsed_seconds": time.time() - started_at,
                },
                continuation_path,
            )
            if observed == "error":
                raise RuntimeError("V1.2 ended in error; V1.3 will not launch")
            if observed == "completed":
                report = _read(Path(config["paths"]["v1_2_report"]))
                if report is None:
                    raise RuntimeError("V1.2 completed without a sealed final report")
                if report.get("final_gates", {}).get("pass") is not True:
                    atomic_json_dump(
                        {
                            "format": "cftn_text_v1_3_continuation_status_v1",
                            "state": "blocked_v1_2_failed",
                            "pid": os.getpid(),
                            "v1_2_report": config["paths"]["v1_2_report"],
                            "failed_gates": [
                                key
                                for key, passed in report.get("final_gates", {}).items()
                                if key != "pass" and passed is not True
                            ],
                            "v1_3_revision_sha256": config["_meta"]["sha256"],
                            "elapsed_seconds": time.time() - started_at,
                        },
                        continuation_path,
                    )
                    return
                audit_v1_2_pass(config)
                break
            time.sleep(poll_seconds)
        if args.wandb and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is missing; refusing an untracked V1.3 launch")
        command = pipeline_command(config_path, device=args.device, wandb=args.wandb)
        atomic_json_dump(
            {
                "format": "cftn_text_v1_3_continuation_status_v1",
                "state": "pipeline_running",
                "pid": os.getpid(),
                "command": subprocess.list2cmdline(command),
                "pipeline_status": str(pipeline_status_path),
                "v1_3_revision_sha256": config["_meta"]["sha256"],
                "elapsed_seconds": time.time() - started_at,
            },
            continuation_path,
        )
        subprocess.run(
            command,
            check=True,
            cwd=Path(config["_meta"]["repository_root"]),
        )
        atomic_json_dump(
            {
                "format": "cftn_text_v1_3_continuation_status_v1",
                "state": "completed",
                "pid": os.getpid(),
                "pipeline_status": str(pipeline_status_path),
                "v1_3_revision_sha256": config["_meta"]["sha256"],
                "elapsed_seconds": time.time() - started_at,
            },
            continuation_path,
        )
    except BaseException as exc:
        atomic_json_dump(
            {
                "format": "cftn_text_v1_3_continuation_status_v1",
                "state": "error",
                "pid": os.getpid(),
                "error": repr(exc),
                "v1_2_status": str(v1_2_status_path),
                "v1_3_revision_sha256": config["_meta"]["sha256"],
                "elapsed_seconds": time.time() - started_at,
            },
            continuation_path,
        )
        raise
    finally:
        continuation_lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
