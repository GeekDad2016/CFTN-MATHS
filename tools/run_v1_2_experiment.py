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
from cftn_text.conditional_training import load_revision_config


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    completion_path: Path
    resumable_artifact: Path | None = None


def _wandb_arguments(
    enabled: bool,
    revision: dict[str, Any],
    *,
    suffix: str,
    tags: list[str],
) -> list[str]:
    if not enabled:
        return []
    settings = revision.get("wandb", {})
    arguments = [
        "--wandb",
        "--wandb-project",
        str(settings.get("project", "cftn-text")),
        "--wandb-run-name",
        f"v1-2-{suffix}",
        "--wandb-group",
        str(settings.get("group", "v1-2-conditional-bridge")),
        "--wandb-mode",
        str(settings.get("mode", "online")),
        "--wandb-tags",
        "v1.2",
        "conditional-communication",
        *tags,
    ]
    entity = settings.get("entity")
    if entity:
        arguments.extend(["--wandb-entity", str(entity)])
    return arguments


def command_plan(
    revision_path: str,
    revision: dict[str, Any],
    *,
    device: str,
    wandb: bool,
) -> list[Stage]:
    paths = revision["paths"]
    root = Path(paths["artifact_root"])
    checkpoint = root / "bridge_conditional_contextual" / "bridge_bidirectional.best.pth"
    shared_root = root / "evaluation_shared"
    complementary_root = root / "evaluation_complementary"
    validation = revision["validation"]
    return [
        Stage(
            "audit_v1_1_prerequisites",
            [
                sys.executable,
                "-m",
                "tools.audit_v1_2_prerequisites",
                "--revision-config",
                revision_path,
            ],
            root / "prerequisites.json",
        ),
        Stage(
            "train_conditional_gpt_to_math",
            [
                sys.executable,
                "-m",
                "tools.train_conditional_bridge",
                "--revision-config",
                revision_path,
                "--device",
                device,
                *_wandb_arguments(
                    wandb,
                    revision,
                    suffix="conditional-gpt-to-math",
                    tags=["bridge", "gpt-to-math", "mixed-necessity"],
                ),
            ],
            root / "bridge_conditional_contextual" / "summary.json",
            root / "bridge_conditional_contextual",
        ),
        Stage(
            "evaluate_shared_no_harm",
            [
                sys.executable,
                "-m",
                "tools.evaluate",
                "--config",
                str(paths["base_config"]),
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
                "--view-mode",
                "shared",
                "--maximum-examples",
                str(validation["shared_maximum_examples_per_split"]),
                "--output-root",
                str(shared_root),
                *_wandb_arguments(
                    wandb,
                    revision,
                    suffix="shared-no-harm-evaluation",
                    tags=["evaluation", "shared", "no-harm"],
                ),
            ],
            shared_root / "report.json",
        ),
        Stage(
            "evaluate_complementary_causality",
            [
                sys.executable,
                "-m",
                "tools.evaluate_synergy",
                "--config",
                str(paths["base_config"]),
                "--protocol",
                str(paths["synergy_protocol"]),
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
                "--maximum-pairs-per-split",
                str(validation["complementary_maximum_pairs_per_split"]),
                "--output-root",
                str(complementary_root),
            ],
            complementary_root / "report.json",
        ),
        Stage(
            "assemble_v1_2_evidence",
            [
                sys.executable,
                "-m",
                "tools.assemble_v1_2_report",
                "--revision-config",
                revision_path,
            ],
            root / "v1_2_final_report.json",
        ),
    ]


def _completion_is_valid(stage: Stage) -> bool:
    if not stage.completion_path.is_file():
        return False
    if stage.completion_path.name == "summary.json":
        try:
            with stage.completion_path.open("r", encoding="utf-8") as handle:
                return json.load(handle).get("state") == "completed"
        except (OSError, json.JSONDecodeError):
            return False
    return True


def _has_checkpoint(path: Path | None) -> bool:
    return bool(path and path.is_dir() and any(path.glob("checkpoint_epoch_*.pth")))


def _validate_wandb_environment(revision: dict[str, Any], enabled: bool) -> None:
    settings = revision.get("wandb", {})
    if not enabled or str(settings.get("mode", "online")) != "online":
        return
    if settings.get("require_api_key_environment", True) and not os.environ.get(
        "WANDB_API_KEY"
    ):
        raise RuntimeError(
            "WANDB_API_KEY must be set in the environment for the online V1.2 run"
        )


def execute_plan(stages: list[Stage], revision: dict[str, Any]) -> None:
    root = Path(revision["paths"]["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "pipeline_status.json"
    started_at = time.time()
    completed: list[str] = []
    for index, stage in enumerate(stages, start=1):
        if _completion_is_valid(stage):
            completed.append(stage.name)
            continue
        command = list(stage.command)
        if _has_checkpoint(stage.resumable_artifact) and "--resume" not in command:
            command.append("--resume")
        atomic_json_dump(
            {
                "format": "cftn_text_v1_2_pipeline_status_v1",
                "state": "running",
                "pid": os.getpid(),
                "current_stage": stage.name,
                "stage_index": index,
                "stages_total": len(stages),
                "completed_stages": completed,
                "command": subprocess.list2cmdline(command),
                "elapsed_seconds": time.time() - started_at,
                "revision_sha256": revision["_meta"]["sha256"],
            },
            status_path,
        )
        try:
            subprocess.run(command, check=True)
        except BaseException as exc:
            atomic_json_dump(
                {
                    "format": "cftn_text_v1_2_pipeline_status_v1",
                    "state": "error",
                    "pid": os.getpid(),
                    "failed_stage": stage.name,
                    "stage_index": index,
                    "stages_total": len(stages),
                    "completed_stages": completed,
                    "error": repr(exc),
                    "elapsed_seconds": time.time() - started_at,
                    "revision_sha256": revision["_meta"]["sha256"],
                },
                status_path,
            )
            raise
        completed.append(stage.name)
    atomic_json_dump(
        {
            "format": "cftn_text_v1_2_pipeline_status_v1",
            "state": "completed",
            "pid": os.getpid(),
            "current_stage": None,
            "stage_index": len(stages),
            "stages_total": len(stages),
            "completed_stages": completed,
            "elapsed_seconds": time.time() - started_at,
            "revision_sha256": revision["_meta"]["sha256"],
        },
        status_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable V1.2 conditional-communication experiment"
    )
    parser.add_argument(
        "--revision-config", default="config/v1_2_conditional_bridge.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    revision_path = str(Path(args.revision_config).expanduser().resolve())
    revision = load_revision_config(revision_path)
    _validate_wandb_environment(revision, args.wandb)
    stages = command_plan(
        revision_path,
        revision,
        device=args.device,
        wandb=args.wandb,
    )
    preview = {
        "format": "cftn_text_v1_2_pipeline_preview_v1",
        "execute": args.execute,
        "revision_sha256": revision["_meta"]["sha256"],
        "stages": [
            {
                "name": stage.name,
                "command": subprocess.list2cmdline(stage.command),
                "completion_path": str(stage.completion_path.resolve()),
            }
            for stage in stages
        ],
        "training_limits": {
            "max_epochs": int(revision["training"]["max_epochs"]),
            "minimum_epochs": int(revision["training"]["minimum_epochs"]),
            "early_stop_patience": int(
                revision["training"]["early_stop_patience"]
            ),
        },
    }
    print(json.dumps(preview, indent=2))
    if args.execute:
        execute_plan(stages, revision)


if __name__ == "__main__":
    main()
