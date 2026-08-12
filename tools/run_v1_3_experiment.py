from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.v1_3_config import load_v1_3_config


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    completion_path: Path
    resumable_artifact: Path | None = None


def _wandb_arguments(
    enabled: bool, config: dict[str, Any], *, run_name: str, tags: list[str]
) -> list[str]:
    if not enabled:
        return []
    settings = config.get("wandb", {})
    arguments = [
        "--wandb",
        "--wandb-project",
        str(settings.get("project", "cftn-text")),
        "--wandb-run-name",
        run_name,
        "--wandb-group",
        str(settings.get("group", "v1-3-multi-specialist")),
        "--wandb-mode",
        str(settings.get("mode", "online")),
        "--wandb-tags",
        "v1.3",
        *tags,
    ]
    entity = settings.get("entity")
    if entity:
        arguments.extend(["--wandb-entity", str(entity)])
    return arguments


def command_plan(
    config_path: str,
    config: dict[str, Any],
    *,
    device: str,
    wandb: bool,
) -> list[Stage]:
    root = Path(config["paths"]["artifact_root"])
    data_root = Path(config["paths"]["data_root"])
    stages = [
        Stage(
            "audit_v1_2_pass",
            [sys.executable, "-u", "-m", "tools.audit_v1_3_prerequisites", "--config", config_path],
            root / "prerequisites.json",
        ),
        Stage(
            "prepare_v1_3_data",
            [sys.executable, "-u", "-m", "tools.prepare_v1_3_data", "--config", config_path],
            data_root / "manifest.json",
        ),
        Stage(
            "calibrate_frozen_gpt_language",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.calibrate_v1_3_gpt",
                "--config",
                config_path,
                "--device",
                device,
            ],
            root / "gpt_language_calibration" / "report.json",
        ),
        Stage(
            "train_exact_string_specialist",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.train_v1_3_string",
                "--config",
                config_path,
                "--device",
                device,
                *_wandb_arguments(
                    wandb,
                    config,
                    run_name="v1-3-string-specialist",
                    tags=["string-specialist", "native-training"],
                ),
            ],
            root / "string_specialist" / "summary.json",
            root / "string_specialist",
        ),
        Stage(
            "seal_native_specialists",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.evaluate_v1_3_specialists",
                "--config",
                config_path,
                "--device",
                device,
                "--specialist-generation-policy",
                "full_context_v1",
            ],
            root / "native_specialist_evaluation" / "report.json",
        ),
    ]
    for phase in config["integration_training"]["phases"]:
        name = str(phase["name"])
        stages.append(
            Stage(
                f"train_{name}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.train_v1_3_integration",
                    "--config",
                    config_path,
                    "--phase",
                    name,
                    "--device",
                    device,
                    *_wandb_arguments(
                        wandb,
                        config,
                        run_name=f"v1-3-{name.replace('_', '-')}",
                        tags=["multi-specialist", name],
                    ),
                ],
                root / name / "summary.json",
                root / name,
            )
        )
    stages.extend(
        [
            Stage(
                "evaluate_sealed_causal_suite",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.evaluate_v1_3",
                    "--config",
                    config_path,
                    "--device",
                    device,
                    "--specialist-generation-policy",
                    "full_context_v1",
                    *_wandb_arguments(
                        wandb,
                        config,
                        run_name="v1-3-sealed-causal-evaluation",
                        tags=["evaluation", "causal-suite", "sealed"],
                    ),
                ],
                root / "sealed_evaluation" / "report.json",
            ),
            Stage(
                "assemble_v1_3_evidence",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.assemble_v1_3_report",
                    "--config",
                    config_path,
                ],
                root / "v1_3_final_report.json",
            ),
        ]
    )
    return stages


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _completion_is_valid(stage: Stage, expected_revision: str | None = None) -> bool:
    value = _read_json(stage.completion_path)
    if value is None:
        return False
    if expected_revision is not None:
        revision_key = (
            "v1_3_revision_sha256"
            if stage.name == "audit_v1_2_pass"
            else "revision_sha256"
        )
        recorded_revision = value.get(revision_key)
        if recorded_revision is not None and recorded_revision != expected_revision:
            return False
        if stage.name in {"prepare_v1_3_data", "calibrate_frozen_gpt_language"} and (
            recorded_revision != expected_revision
        ):
            return False
    if stage.completion_path.name == "summary.json":
        return value.get("state") == "completed"
    if stage.name in {
        "audit_v1_2_pass",
        "calibrate_frozen_gpt_language",
        "seal_native_specialists",
    }:
        return value.get("state") in {"passed", "completed"}
    return True


def _has_checkpoint(path: Path | None) -> bool:
    return bool(path and path.is_dir() and any(path.glob("checkpoint_epoch_*.pth")))


def validate_wandb_environment(config: dict[str, Any], enabled: bool) -> None:
    settings = config.get("wandb", {})
    if not enabled or str(settings.get("mode", "online")) != "online":
        return
    if settings.get("require_api_key_environment", True) and not os.environ.get(
        "WANDB_API_KEY"
    ):
        raise RuntimeError("WANDB_API_KEY must be set for the online V1.3 pipeline")


def execute_plan(stages: list[Stage], config: dict[str, Any]) -> None:
    root = Path(config["paths"]["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "pipeline_status.json"
    lock_path = root / "pipeline.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("V1.3 pipeline lock exists; refusing duplicate launcher") from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{os.getpid()}\n")
    started_at = time.time()
    completed: list[str] = []
    try:
        for index, stage in enumerate(stages, start=1):
            if _completion_is_valid(stage, config["_meta"]["sha256"]):
                completed.append(stage.name)
                continue
            command = list(stage.command)
            if _has_checkpoint(stage.resumable_artifact) and "--resume" not in command:
                command.append("--resume")
            atomic_json_dump(
                {
                    "format": "cftn_text_v1_3_pipeline_status_v1",
                    "state": "running",
                    "pid": os.getpid(),
                    "current_stage": stage.name,
                    "stage_index": index,
                    "stages_total": len(stages),
                    "completed_stages": completed,
                    "command": subprocess.list2cmdline(command),
                    "elapsed_seconds": time.time() - started_at,
                    "revision_sha256": config["_meta"]["sha256"],
                },
                status_path,
            )
            try:
                subprocess.run(
                    command,
                    check=True,
                    cwd=Path(config["_meta"]["repository_root"]),
                )
            except BaseException as exc:
                atomic_json_dump(
                    {
                        "format": "cftn_text_v1_3_pipeline_status_v1",
                        "state": "error",
                        "pid": os.getpid(),
                        "failed_stage": stage.name,
                        "stage_index": index,
                        "stages_total": len(stages),
                        "completed_stages": completed,
                        "error": repr(exc),
                        "elapsed_seconds": time.time() - started_at,
                        "revision_sha256": config["_meta"]["sha256"],
                    },
                    status_path,
                )
                raise
            completed.append(stage.name)
        atomic_json_dump(
            {
                "format": "cftn_text_v1_3_pipeline_status_v1",
                "state": "completed",
                "pid": os.getpid(),
                "current_stage": None,
                "stage_index": len(stages),
                "stages_total": len(stages),
                "completed_stages": completed,
                "elapsed_seconds": time.time() - started_at,
                "revision_sha256": config["_meta"]["sha256"],
            },
            status_path,
        )
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the guarded V1.3 experiment")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = str(Path(args.config).expanduser().resolve())
    config = load_v1_3_config(config_path)
    validate_wandb_environment(config, args.wandb)
    stages = command_plan(config_path, config, device=args.device, wandb=args.wandb)
    preview = {
        "format": "cftn_text_v1_3_pipeline_preview_v1",
        "execute": bool(args.execute),
        "revision_sha256": config["_meta"]["sha256"],
        "hard_prerequisite": str(Path(config["paths"]["v1_2_report"])),
        "stages": [
            {
                "index": index,
                "name": stage.name,
                "command": subprocess.list2cmdline(stage.command),
                "completion_path": str(stage.completion_path.resolve()),
            }
            for index, stage in enumerate(stages, start=1)
        ],
        "epoch_limits": {
            "string_specialist": int(config["string_training"]["max_epochs"]),
            **{
                phase["name"]: int(phase["max_epochs"])
                for phase in config["integration_training"]["phases"]
            },
        },
    }
    print(json.dumps(preview, indent=2))
    if args.execute:
        execute_plan(stages, config)


if __name__ == "__main__":
    main()
