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
from cftn_text.config import load_config


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def pipeline_command(
    config_path: str | Path,
    synergy_protocol_path: str | Path,
    *,
    include_fixed_open: bool,
    wandb: bool,
    wandb_project: str,
    wandb_run_name: str,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.run_experiment",
        "--config",
        str(Path(config_path).resolve()),
        "--synergy-protocol",
        str(Path(synergy_protocol_path).resolve()),
        "--execute",
    ]
    if include_fixed_open:
        command.append("--include-fixed-open")
    if wandb:
        command.extend(
            [
                "--wandb",
                "--wandb-project",
                wandb_project,
                "--wandb-run-name",
                wandb_run_name,
                "--wandb-mode",
                "online",
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for one stage, then launch a fresh ordered experiment"
    )
    parser.add_argument("--wait-status", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--synergy-protocol", required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--include-fixed-open", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cftn-text")
    parser.add_argument("--wandb-run-name")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    synergy_path = Path(args.synergy_protocol).expanduser().resolve()
    wait_status_path = Path(args.wait_status).expanduser().resolve()
    config = load_config(config_path)
    artifact_root = Path(config["project"]["artifact_root"]).expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    continuation_status_path = artifact_root / "continuation_status.json"
    pipeline_status_path = artifact_root / "pipeline_status.json"
    lock_path = artifact_root / "continuation.lock"
    run_name = str(args.wandb_run_name or config["project"]["name"])
    poll_seconds = max(30, int(args.poll_seconds))

    existing_pipeline = _read_json(pipeline_status_path)
    if existing_pipeline and existing_pipeline.get("state") in {"running", "completed"}:
        raise RuntimeError(
            f"pipeline is already {existing_pipeline['state']}; refusing duplicate launch"
        )
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("continuation lock already exists") from exc
    with os.fdopen(lock_fd, "w", encoding="ascii") as handle:
        handle.write(f"{os.getpid()}\n")

    started_at = time.time()
    try:
        while True:
            observed = _read_json(wait_status_path)
            state = str((observed or {}).get("state", "missing"))
            atomic_json_dump(
                {
                    "state": "waiting",
                    "pid": os.getpid(),
                    "wait_status": str(wait_status_path),
                    "observed_state": state,
                    "observed_epoch": (observed or {}).get("epoch"),
                    "observed_global_step": (observed or {}).get("global_step"),
                    "poll_seconds": poll_seconds,
                    "elapsed_seconds": time.time() - started_at,
                },
                continuation_status_path,
            )
            if state == "completed":
                break
            if state == "error":
                raise RuntimeError("prerequisite stage ended in error")
            time.sleep(poll_seconds)

        command = pipeline_command(
            config_path,
            synergy_path,
            include_fixed_open=bool(args.include_fixed_open),
            wandb=bool(args.wandb),
            wandb_project=str(args.wandb_project),
            wandb_run_name=run_name,
        )
        atomic_json_dump(
            {
                "state": "pipeline_running",
                "pid": os.getpid(),
                "command": subprocess.list2cmdline(command),
                "wait_status": str(wait_status_path),
                "pipeline_status": str(pipeline_status_path),
                "elapsed_seconds": time.time() - started_at,
            },
            continuation_status_path,
        )
        subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        atomic_json_dump(
            {
                "state": "completed",
                "pid": os.getpid(),
                "pipeline_status": str(pipeline_status_path),
                "elapsed_seconds": time.time() - started_at,
            },
            continuation_status_path,
        )
    except BaseException as exc:
        atomic_json_dump(
            {
                "state": "error",
                "pid": os.getpid(),
                "error": repr(exc),
                "wait_status": str(wait_status_path),
                "pipeline_status": str(pipeline_status_path),
                "elapsed_seconds": time.time() - started_at,
            },
            continuation_status_path,
        )
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
