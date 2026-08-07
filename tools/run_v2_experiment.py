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
from cftn_text.config import load_config
from cftn_text.v2_data import audit_v2_manifest


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    completion_path: Path
    resumable_artifact: Path | None = None


def _wandb_arguments(
    enabled: bool,
    config: dict[str, Any],
    *,
    suffix: str,
    tags: list[str],
) -> list[str]:
    if not enabled:
        return []
    settings = config.get("wandb", {})
    arguments = [
        "--wandb",
        "--wandb-project",
        str(settings.get("project", "cftn-text-v2")),
        "--wandb-run-name",
        f"{config['project']['name']}-{suffix}",
        "--wandb-group",
        str(settings.get("group", config["project"]["name"])),
        "--wandb-mode",
        str(settings.get("mode", "online")),
        "--wandb-tags",
        "v2",
        "end-to-end",
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
    root = Path(config["project"]["artifact_root"])
    data_root = Path(config["project"]["data_root"])
    math_checkpoint = root / "math" / "math.best.pth"
    m2g_root = root / "bridge_m2g_contextual_complementary"
    m2g_checkpoint = m2g_root / "bridge_m2g.best.pth"
    bidirectional_root = root / "bridge_bidirectional_contextual_complementary"
    bidirectional_checkpoint = bidirectional_root / "bridge_bidirectional.best.pth"
    return [
        Stage(
            "prepare_data",
            [
                sys.executable,
                "-m",
                "tools.prepare_v2_data",
                "--config",
                config_path,
            ],
            data_root / "manifest.json",
        ),
        Stage(
            "train_math",
            [
                sys.executable,
                "-m",
                "tools.train_math_tower",
                "--config",
                config_path,
                "--device",
                device,
                "--skip-calibration",
                *_wandb_arguments(
                    wandb, config, suffix="math", tags=["math-tower", "curriculum"]
                ),
            ],
            root / "math" / "summary.json",
            root / "math",
        ),
        Stage(
            "evaluate_math",
            [
                sys.executable,
                "-m",
                "tools.evaluate_v2_math",
                "--config",
                config_path,
                "--device",
                device,
                "--checkpoint",
                str(math_checkpoint),
                *_wandb_arguments(
                    wandb,
                    config,
                    suffix="math-evaluation",
                    tags=["evaluation", "generalization"],
                ),
            ],
            root / "evaluation_math_v2" / "report.json",
        ),
        Stage(
            "train_m2g",
            [
                sys.executable,
                "-m",
                "tools.train_bridges",
                "--config",
                config_path,
                "--device",
                device,
                "--stage",
                "m2g",
                "--view-mode",
                "complementary",
                "--math-checkpoint",
                str(math_checkpoint),
                *_wandb_arguments(
                    wandb,
                    config,
                    suffix="m2g",
                    tags=["bridge", "math-to-gpt", "complementary"],
                ),
            ],
            m2g_root / "summary.json",
            m2g_root,
        ),
        Stage(
            "train_bidirectional",
            [
                sys.executable,
                "-m",
                "tools.train_bridges",
                "--config",
                config_path,
                "--device",
                device,
                "--stage",
                "bidirectional",
                "--view-mode",
                "complementary",
                "--math-checkpoint",
                str(math_checkpoint),
                "--initialize-from",
                str(m2g_checkpoint),
                *_wandb_arguments(
                    wandb,
                    config,
                    suffix="bidirectional",
                    tags=["bridge", "bidirectional", "complementary"],
                ),
            ],
            bidirectional_root / "summary.json",
            bidirectional_root,
        ),
        Stage(
            "evaluate_collaboration",
            [
                sys.executable,
                "-m",
                "tools.evaluate_v2_collaboration",
                "--config",
                config_path,
                "--device",
                device,
                "--checkpoint",
                str(bidirectional_checkpoint),
                "--math-checkpoint",
                str(math_checkpoint),
                *_wandb_arguments(
                    wandb,
                    config,
                    suffix="collaboration-evaluation",
                    tags=["evaluation", "causal-ablation", "synergy"],
                ),
            ],
            root / "evaluation_collaboration_v2" / "report.json",
        ),
        Stage(
            "assess_scale",
            [
                sys.executable,
                "-m",
                "tools.assess_v2_scale",
                "--config",
                config_path,
            ],
            root / "scale_decision.json",
        ),
        Stage(
            "assemble_report",
            [
                sys.executable,
                "-m",
                "tools.assemble_v2_report",
                "--config",
                config_path,
            ],
            root / "v2_final_report.json",
        ),
    ]


def _is_complete(stage: Stage, config: dict[str, Any]) -> bool:
    if not stage.completion_path.is_file():
        return False
    if stage.name == "prepare_data":
        try:
            with stage.completion_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            audit_v2_manifest(manifest, config["project"]["data_root"])
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
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


def _validate_wandb_environment(config: dict[str, Any], enabled: bool) -> None:
    settings = config.get("wandb", {})
    if not enabled or str(settings.get("mode", "online")) != "online":
        return
    if settings.get("require_api_key_environment", False) and not os.environ.get(
        "WANDB_API_KEY"
    ):
        raise RuntimeError(
            "WANDB_API_KEY must be set in the environment for the online V2 run"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete resumable CFTN-Text V2")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    args = parser.parse_args()
    config_path = str(Path(args.config).resolve())
    config = load_config(config_path)
    stages = command_plan(
        config_path, config, device=args.device, wandb=args.wandb
    )
    names = [stage.name for stage in stages]
    if args.from_stage and args.from_stage not in names:
        raise ValueError(f"unknown --from-stage {args.from_stage}; choose from {names}")
    if args.through_stage and args.through_stage not in names:
        raise ValueError(
            f"unknown --through-stage {args.through_stage}; choose from {names}"
        )
    start = names.index(args.from_stage) if args.from_stage else 0
    end = names.index(args.through_stage) + 1 if args.through_stage else len(stages)
    stages = stages[start:end]
    preview = {
        "project": config["project"]["name"],
        "execute": args.execute,
        "resume": args.resume,
        "wandb": args.wandb,
        "wandb_api_key_source": "WANDB_API_KEY environment variable",
        "train_examples": config["data"]["train_examples"],
        "math_epochs": config["math_training"]["max_epochs"],
        "bridge_epochs_per_stage": config["bridge_training"]["max_epochs"],
        "stages": [
            {
                "name": stage.name,
                "complete": _is_complete(stage, config),
                "command": subprocess.list2cmdline(stage.command),
            }
            for stage in stages
        ],
    }
    print(json.dumps(preview, indent=2))
    if not args.execute:
        return
    _validate_wandb_environment(config, args.wandb)
    artifact_root = Path(config["project"]["artifact_root"])
    log_root = artifact_root / "pipeline_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "pipeline_state.json"
    state: dict[str, Any] = {
        "format": "cftn_text_v2_pipeline_state_v1",
        "project": config["project"]["name"],
        "state": "running",
        "started_unix": time.time(),
        "stages": {},
    }
    atomic_json_dump(state, state_path)
    try:
        for stage in stages:
            if args.resume and _is_complete(stage, config):
                state["stages"][stage.name] = {
                    "state": "skipped_completed",
                    "completion_path": str(stage.completion_path.resolve()),
                }
                atomic_json_dump(state, state_path)
                continue
            command = list(stage.command)
            if args.resume and _has_checkpoint(stage.resumable_artifact):
                command.append("--resume")
            stdout_path = log_root / f"{stage.name}.stdout.log"
            stderr_path = log_root / f"{stage.name}.stderr.log"
            state["stages"][stage.name] = {
                "state": "running",
                "started_unix": time.time(),
                "command": subprocess.list2cmdline(command),
                "stdout": str(stdout_path.resolve()),
                "stderr": str(stderr_path.resolve()),
            }
            atomic_json_dump(state, state_path)
            print(f"Starting V2 stage: {stage.name}", flush=True)
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
                "a", encoding="utf-8"
            ) as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr)
            if result.returncode:
                state["stages"][stage.name].update(
                    {"state": "error", "returncode": result.returncode}
                )
                raise subprocess.CalledProcessError(result.returncode, command)
            if not _is_complete(stage, config):
                raise RuntimeError(
                    f"stage {stage.name} exited cleanly without its completion artifact"
                )
            state["stages"][stage.name].update(
                {
                    "state": "completed",
                    "completed_unix": time.time(),
                    "completion_path": str(stage.completion_path.resolve()),
                }
            )
            atomic_json_dump(state, state_path)
        state["state"] = "completed"
        state["completed_unix"] = time.time()
        atomic_json_dump(state, state_path)
    except BaseException as exc:
        state["state"] = "error"
        state["error"] = repr(exc)
        state["failed_unix"] = time.time()
        atomic_json_dump(state, state_path)
        raise


if __name__ == "__main__":
    main()
