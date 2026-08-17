from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.conditional_training import load_revision_config
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.pipeline_lock import exclusive_pipeline_lock
from cftn_text.v2_data import audit_v2_manifest


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]
    completion_path: Path
    resumable_artifact: Path | None = None
    epoch_limit: int | None = None


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
    math_checkpoint = root / "math_selected" / "math.selected.pth"
    m2g_root = root / "bridge_m2g_contextual"
    conditional_root = root / "bridge_conditional_contextual"
    conditional_checkpoint = conditional_root / "bridge_bidirectional.best.pth"
    repository_root = Path(config_path).resolve().parent.parent
    revision_path = Path(
        config.get("conditional_bridge", {}).get(
            "revision_config", "config/v2_conditional_bridge.yaml"
        )
    )
    if not revision_path.is_absolute():
        revision_path = repository_root / revision_path
    revision = load_revision_config(revision_path)
    if Path(revision["paths"]["base_config"]) != Path(config_path).resolve():
        raise ValueError("V2 conditional revision points to a different base config")
    resolved_root = (root if root.is_absolute() else repository_root / root).resolve()
    if Path(revision["paths"]["artifact_root"]).resolve() != resolved_root:
        raise ValueError("V2 conditional revision and base config use different artifact roots")
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
                *(
                    ["--disable-early-stopping"]
                    if not bool(
                        config.get("math_training", {}).get(
                            "early_stopping_enabled", True
                        )
                    )
                    else []
                ),
                *_wandb_arguments(
                    wandb, config, suffix="math", tags=["math-tower", "curriculum"]
                ),
            ],
            root / "math" / "summary.json",
            root / "math",
            int(config["math_training"]["max_epochs"]),
        ),
        Stage(
            "select_math_checkpoint",
            [
                sys.executable,
                "-m",
                "tools.select_v2_math_checkpoint",
                "--config",
                config_path,
                "--device",
                device,
            ],
            root / "math_checkpoint_selection" / "report.json",
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
            "audit_mechanism_prerequisites",
            [
                sys.executable,
                "-m",
                "tools.audit_v2_prerequisites",
                "--config",
                config_path,
            ],
            root / "mechanism_prerequisites.json",
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
                "shared",
                "--math-checkpoint",
                str(math_checkpoint),
                *_wandb_arguments(
                    wandb,
                    config,
                    suffix="m2g",
                    tags=["bridge", "math-to-gpt", "shared"],
                ),
            ],
            m2g_root / "summary.json",
            m2g_root,
            int(config["bridge_training"]["max_epochs"]),
        ),
        Stage(
            "train_conditional_gpt_to_math",
            [
                sys.executable,
                "-m",
                "tools.train_conditional_bridge",
                "--revision-config",
                str(revision_path.resolve()),
                "--device",
                device,
                *_wandb_arguments(
                    wandb,
                    config,
                    suffix="conditional-gpt-to-math",
                    tags=["bridge", "conditional", "mixed-necessity", "no-harm"],
                ),
            ],
            conditional_root / "summary.json",
            conditional_root,
            int(revision["training"]["max_epochs"]),
        ),
        Stage(
            "evaluate_shared_no_harm",
            [
                sys.executable,
                "-m",
                "tools.evaluate_v2_collaboration",
                "--config",
                config_path,
                "--device",
                device,
                "--view-mode",
                "shared",
                "--checkpoint",
                str(conditional_checkpoint),
                "--math-checkpoint",
                str(math_checkpoint),
                *_wandb_arguments(
                    wandb,
                    config,
                    suffix="shared-no-harm-evaluation",
                    tags=["evaluation", "shared", "no-harm"],
                ),
            ],
            root / "evaluation_shared_no_harm_v2" / "report.json",
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
                str(conditional_checkpoint),
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
                summary = json.load(handle)
            if summary.get("state") != "completed":
                return False
            if stage.name == "train_conditional_gpt_to_math":
                return (
                    summary.get("best_acceptance", {}).get("gates", {}).get("pass")
                    is True
                )
            return True
        except (OSError, json.JSONDecodeError):
            return False
    try:
        with stage.completion_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if stage.name == "audit_mechanism_prerequisites":
        return report.get("state") == "passed" and report.get("pass") is True
    if stage.name == "select_math_checkpoint":
        selected = report.get("selected", {})
        path = Path(str(selected.get("path", "")))
        return (
            report.get("state") == "completed"
            and path.is_file()
            and file_sha256(path) == selected.get("sha256")
        )
    if stage.name == "evaluate_math":
        return report.get("specialist_gate", {}).get("pass") is True
    if stage.name == "evaluate_shared_no_harm":
        return report.get("shared_no_harm_gate", {}).get("pass") is True
    if stage.name == "evaluate_collaboration":
        return report.get("collaboration_gate", {}).get("pass") is True
    if stage.name == "assemble_report":
        return report.get("overall_pass") is True
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


def _project_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["project"][key]).expanduser()
    if value.is_absolute():
        return value.resolve()
    repository_root = Path(config["_meta"]["path"]).resolve().parent.parent
    return (repository_root / value).resolve()


def _git_revision(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _verify_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".cftn-write-test-{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("ok\n")
    finally:
        probe.unlink(missing_ok=True)


def _runtime_preflight(
    config: dict[str, Any], *, device: str, wandb: bool
) -> dict[str, Any]:
    if sys.version_info < (3, 11):
        raise RuntimeError("CFTN V2 requires Python 3.11 or newer")
    _validate_wandb_environment(config, wandb)

    artifact_root = _project_path(config, "artifact_root")
    data_root = _project_path(config, "data_root")
    _verify_writable(artifact_root)
    _verify_writable(data_root)
    repository_root = Path(config["_meta"]["path"]).resolve().parent.parent

    gpu: dict[str, Any] = {"required": device.casefold().startswith("cuda")}
    if gpu["required"]:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false; "
                "install a CUDA PyTorch build and attach a GPU before starting V2"
            )
        selected = torch.device(device)
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        bf16_required = any(
            str(config.get(section, {}).get("precision", "")).casefold() == "bf16"
            for section in ("math_training", "bridge_training")
        )
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        if bf16_required and not bf16_supported:
            raise RuntimeError(
                "the V2 config requires bf16 but the selected CUDA device does not support it"
            )
        gpu.update(
            {
                "available": True,
                "device": str(selected),
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
                "bf16_supported": bf16_supported,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
    else:
        gpu.update({"available": False, "device": device})

    prerequisite = config.get("prerequisites", {})
    v1_3_value = Path(str(prerequisite.get("v1_3_report", ""))).expanduser()
    v1_3_path = (
        v1_3_value.resolve()
        if v1_3_value.is_absolute()
        else (repository_root / v1_3_value).resolve()
    )
    report = {
        "format": "cftn_text_v2_startup_preflight_v1",
        "state": "passed",
        "checked_unix": time.time(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repository_root": str(repository_root),
        "repository_revision": _git_revision(repository_root),
        "config_path": config["_meta"]["path"],
        "config_sha256": config["_meta"]["sha256"],
        "artifact_root": str(artifact_root),
        "data_root": str(data_root),
        "storage": {
            "artifact_free_bytes": int(shutil.disk_usage(artifact_root).free),
            "data_free_bytes": int(shutil.disk_usage(data_root).free),
        },
        "gpu": gpu,
        "wandb": {
            "enabled": wandb,
            "mode": str(config.get("wandb", {}).get("mode", "online")),
            "project": str(config.get("wandb", {}).get("project", "cftn-text-v2")),
            "group": str(config.get("wandb", {}).get("group", "")),
            "entity": config.get("wandb", {}).get("entity") or None,
            "api_key_present": bool(os.environ.get("WANDB_API_KEY")),
        },
        "mechanism_prerequisite": {
            "timing": "after_standalone_math_gate_before_any_bridge_training",
            "v1_3_report": str(v1_3_path),
            "v1_3_report_present": v1_3_path.is_file(),
            "missing_report_blocks_math_training": False,
            "missing_or_failed_report_blocks_bridge_training": True,
        },
    }
    atomic_json_dump(report, artifact_root / "startup_preflight.json")
    return report


def _execute_stages(
    config: dict[str, Any],
    stages: list[Stage],
    *,
    resume: bool,
    control_root: str | None,
    preflight: dict[str, Any],
) -> None:
    artifact_root = _project_path(config, "artifact_root")
    log_root = artifact_root / "pipeline_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "pipeline_state.json"
    state: dict[str, Any] = {
        "format": "cftn_text_v2_pipeline_state_v1",
        "project": config["project"]["name"],
        "state": "running",
        "pid": os.getpid(),
        "repository_revision": preflight.get("repository_revision"),
        "config_sha256": config["_meta"]["sha256"],
        "started_unix": time.time(),
        "stage_count": len(stages),
        "stages": {},
    }
    atomic_json_dump(state, state_path)
    try:
        for stage_index, stage in enumerate(stages, start=1):
            if resume and _is_complete(stage, config):
                state["stages"][stage.name] = {
                    "state": "skipped_completed",
                    "completion_path": str(stage.completion_path.resolve()),
                }
                atomic_json_dump(state, state_path)
                continue
            command = list(stage.command)
            if resume and _has_checkpoint(stage.resumable_artifact):
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
            state["current_stage"] = stage.name
            state["current_stage_index"] = stage_index
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
            if control_root:
                pause_path = Path(control_root).resolve() / "pause_after_stage.json"
                if pause_path.is_file():
                    state["state"] = "paused"
                    state["paused_after_stage"] = stage.name
                    state["paused_unix"] = time.time()
                    state["current_stage"] = None
                    atomic_json_dump(state, state_path)
                    print(
                        f"V2 pipeline paused safely after stage: {stage.name}",
                        flush=True,
                    )
                    return
        state["state"] = "completed"
        state["completed_unix"] = time.time()
        state["current_stage"] = None
        atomic_json_dump(state, state_path)
    except BaseException as exc:
        state["state"] = "error"
        state["error"] = repr(exc)
        state["failed_unix"] = time.time()
        atomic_json_dump(state, state_path)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the complete resumable CFTN-Text V2")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate Python, storage, CUDA/BF16, and W&B without running a stage",
    )
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    parser.add_argument(
        "--control-root",
        default=os.environ.get("CFTN_CONTROL_ROOT"),
        help="Optional control directory used for safe pause-at-stage-boundary requests",
    )
    args = parser.parse_args(argv)
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
        "single_pipeline_lock": str(
            (_project_path(config, "artifact_root") / "pipeline.lock").resolve()
        ),
        "mechanism_prerequisite_timing": (
            "after standalone math evaluation and before any bridge training"
        ),
        "train_examples": config["data"]["train_examples"],
        "training_epoch_limits": {
            stage.name: stage.epoch_limit
            for stage in stages
            if stage.epoch_limit is not None
        },
        "stages": [
            {
                "name": stage.name,
                "complete": _is_complete(stage, config),
                "command": subprocess.list2cmdline(stage.command),
                "epoch_limit": stage.epoch_limit,
            }
            for stage in stages
        ],
    }
    print(json.dumps(preview, indent=2))
    if not args.execute and not args.preflight_only:
        return
    artifact_root = _project_path(config, "artifact_root")
    with exclusive_pipeline_lock(artifact_root / "pipeline.lock"):
        preflight = _runtime_preflight(config, device=args.device, wandb=args.wandb)
        print(json.dumps({"startup_preflight": preflight}, indent=2), flush=True)
        if args.preflight_only:
            return
        _execute_stages(
            config,
            stages,
            resume=args.resume,
            control_root=args.control_root,
            preflight=preflight,
        )


if __name__ == "__main__":
    main()
