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
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.gpt_receiver import validate_dense_causal_lm_config
from cftn_text.pipeline_lock import exclusive_pipeline_lock
from cftn_text.semantic_features import normalize_token_ids
from cftn_text.v2_data import audit_v2_manifest
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_data import audit_v1_3_manifest


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
    project_name = str(
        config.get("project", {}).get(
            "name", config.get("revision", {}).get("name", "cftn-text-v2")
        )
    )
    arguments = [
        "--wandb",
        "--wandb-project",
        str(settings.get("project", "cftn-text-v2")),
        "--wandb-run-name",
        f"{project_name}-{suffix}",
        "--wandb-group",
        str(settings.get("group", project_name)),
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
    repository_root = Path(config_path).resolve().parent.parent
    root_value = Path(config["project"]["artifact_root"])
    root = (
        root_value.resolve()
        if root_value.is_absolute()
        else (repository_root / root_value).resolve()
    )
    data_value = Path(config["project"]["data_root"])
    data_root = (
        data_value.resolve()
        if data_value.is_absolute()
        else (repository_root / data_value).resolve()
    )
    revision_value = Path(
        config.get("multi_specialist", {}).get(
            "revision_config", "config/v2_multi_specialist.yaml"
        )
    )
    revision_path = (
        revision_value.resolve()
        if revision_value.is_absolute()
        else (repository_root / revision_value).resolve()
    )
    revision = load_v1_3_config(revision_path)
    if Path(revision["paths"]["base_config"]) != Path(config_path).resolve():
        raise ValueError("V2 multi-specialist revision points to a different base config")
    if Path(revision["paths"]["artifact_root"]) != root:
        raise ValueError("V2 base and multi-specialist configs use different artifact roots")
    math_checkpoint = root / "math_selected" / "math.selected.pth"
    stages = [
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
            "assess_math_scale",
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
            "prepare_multi_specialist_data",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.prepare_v1_3_data",
                "--config",
                str(revision_path),
            ],
            Path(revision["paths"]["data_root"]) / "manifest.json",
        ),
        Stage(
            "train_learned_dispatcher",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.train_v2_dispatcher",
                "--config",
                config_path,
                "--device",
                device,
            ],
            root
            / str(revision["dispatcher"]["artifact_directory"])
            / "summary.json",
            root / str(revision["dispatcher"]["artifact_directory"]),
            int(revision["dispatcher"]["epochs"]),
        ),
        Stage(
            "calibrate_frozen_gpt_language",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.calibrate_v1_3_gpt",
                "--config",
                str(revision_path),
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
                str(revision_path),
                "--device",
                device,
                *_wandb_arguments(
                    wandb,
                    revision,
                    suffix="string-specialist",
                    tags=["string-specialist", "native-training"],
                ),
            ],
            root / "string_specialist" / "summary.json",
            root / "string_specialist",
            int(revision["string_training"]["max_epochs"]),
        ),
        Stage(
            "seal_native_specialists",
            [
                sys.executable,
                "-u",
                "-m",
                "tools.evaluate_v1_3_specialists",
                "--config",
                str(revision_path),
                "--device",
                device,
                "--specialist-generation-policy",
                "configured",
            ],
            root / "native_specialist_evaluation" / "report.json",
        ),
    ]
    for phase in revision["integration_training"]["phases"]:
        phase_name = str(phase["name"])
        if phase_name == "hardened_wake":
            stages.append(
                Stage(
                    "evaluate_zero_update_hard_baseline",
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "tools.evaluate_hard_transition_baseline",
                        "--config",
                        str(revision_path),
                        "--device",
                        device,
                        *_wandb_arguments(
                            wandb,
                            revision,
                            suffix="zero-update-hard-baseline",
                            tags=["hard-wake", "zero-update", "diagnostic"],
                        ),
                    ],
                    root / "hard_transition_baseline" / "report.json",
                )
            )
        stages.append(
            Stage(
                f"train_{phase_name}",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.train_v1_3_integration",
                    "--config",
                    str(revision_path),
                    "--phase",
                    phase_name,
                    "--device",
                    device,
                    *_wandb_arguments(
                        wandb,
                        revision,
                        suffix=phase_name.replace("_", "-"),
                        tags=["multi-specialist", phase_name],
                    ),
                ],
                root / phase_name / "summary.json",
                root / phase_name,
                int(phase["max_epochs"]),
            )
        )
    stages.extend(
        [
            Stage(
                "evaluate_native_typed_dispatch",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.evaluate_v2_native_dispatch",
                    "--config",
                    config_path,
                    "--device",
                    device,
                ],
                root
                / str(
                    revision["native_dispatch_evaluation"]["artifact_directory"]
                )
                / "report.json",
            ),
            Stage(
                "evaluate_sealed_causal_suite",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.evaluate_v1_3",
                    "--config",
                    str(revision_path),
                    "--device",
                    device,
                    "--specialist-generation-policy",
                    "configured",
                    *_wandb_arguments(
                        wandb,
                        revision,
                        suffix="sealed-causal-evaluation",
                        tags=["evaluation", "causal-suite", "conditional-compute"],
                    ),
                ],
                root / "sealed_evaluation" / "report.json",
            ),
            Stage(
                "assemble_v2_evidence",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "tools.assemble_v2_multi_report",
                    "--config",
                    str(revision_path),
                ],
                root / "v2_final_report.json",
            ),
        ]
    )
    return stages


def _multi_revision(config: dict[str, Any]) -> dict[str, Any]:
    repository_root = Path(config["_meta"]["path"]).resolve().parent.parent
    value = Path(
        config.get("multi_specialist", {}).get(
            "revision_config", "config/v2_multi_specialist.yaml"
        )
    )
    path = value if value.is_absolute() else repository_root / value
    return load_v1_3_config(path)


def _is_complete(stage: Stage, config: dict[str, Any]) -> bool:
    if not stage.completion_path.is_file():
        return False
    if stage.name == "prepare_data":
        try:
            with stage.completion_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            audit_v2_manifest(manifest, stage.completion_path.parent)
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
    if stage.name == "prepare_multi_specialist_data":
        try:
            revision = _multi_revision(config)
            audit_v1_3_manifest(revision, stage.completion_path.parent)
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
    if stage.completion_path.name == "summary.json":
        try:
            with stage.completion_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if (
                config.get("_meta")
                and stage.name.startswith("train_")
                and stage.name != "train_math"
                and summary.get("revision_sha256")
                != _multi_revision(config)["_meta"]["sha256"]
            ):
                return False
            if summary.get("state") != "completed":
                if stage.name != "train_learned_dispatcher" or summary.get(
                    "state"
                ) != "passed":
                    return False
            if stage.name == "train_learned_dispatcher":
                checkpoint = Path(str(summary.get("checkpoint", "")))
                return (
                    summary.get("acceptance", {}).get("gates", {}).get("pass")
                    is True
                    and checkpoint.is_file()
                    and file_sha256(checkpoint)
                    == summary.get("checkpoint_sha256")
                )
            if stage.name == "train_hardened_wake":
                contract = summary.get("optimizer_contract", {})
                selected_metrics = summary.get("best_metrics") or summary.get(
                    "final_metrics", {}
                )
                return (
                    contract.get("group_names") == ["gates"]
                    and contract.get("gate_only") is True
                    and contract.get("trainable_components") == ["wake_gates"]
                    and contract.get("halt_gate_frozen") is True
                    and selected_metrics
                    .get("hardening_acceptance", {})
                    .get("gates", {})
                    .get("pass")
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
    multi_revision_reports = {
        "calibrate_frozen_gpt_language",
        "seal_native_specialists",
        "evaluate_zero_update_hard_baseline",
        "evaluate_native_typed_dispatch",
        "evaluate_sealed_causal_suite",
        "assemble_v2_evidence",
    }
    if (
        config.get("_meta")
        and stage.name in multi_revision_reports
        and report.get("revision_sha256")
        != _multi_revision(config)["_meta"]["sha256"]
    ):
        return False
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
    if stage.name == "calibrate_frozen_gpt_language":
        return report.get("state") == "passed" and report.get("pass") is True
    if stage.name == "seal_native_specialists":
        return report.get("state") == "passed" and report.get("gates", {}).get("pass") is True
    if stage.name == "evaluate_zero_update_hard_baseline":
        return (
            report.get("state") == "completed"
            and report.get("optimizer_updates") == 0
            and report.get("trainable_parameters") == 0
            and report.get("full_validation") is True
        )
    if stage.name == "evaluate_native_typed_dispatch":
        return (
            report.get("format")
            == "cftn_text_v2_native_typed_dispatch_evaluation_v1"
            and report.get("state") == "passed"
            and report.get("oracle_metadata_visible_to_runtime") is False
            and report.get("deterministic_answer_composition") is True
            and report.get("acceptance", {}).get("gates", {}).get("pass") is True
        )
    if stage.name == "evaluate_sealed_causal_suite":
        return report.get("state") == "completed"
    if stage.name == "assemble_v2_evidence":
        return (
            report.get("format") == "cftn_text_v2_multi_specialist_report_v1"
            and isinstance(report.get("final_gates", {}).get("pass"), bool)
        )
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


def _coordinator_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the pinned coordinator metadata without downloading its weights."""

    from transformers import AutoConfig, AutoTokenizer, __version__ as transformers_version

    settings = config["gpt"]
    common: dict[str, Any] = {
        "revision": str(settings["revision"]),
        "local_files_only": bool(settings.get("local_files_only", False)),
        "trust_remote_code": bool(settings.get("trust_remote_code", False)),
    }
    model_config = AutoConfig.from_pretrained(settings["model_name"], **common)
    validate_dense_causal_lm_config(
        model_config,
        expected_model_type=settings.get("expected_model_type"),
        expected_hidden_size=settings.get("expected_hidden_size"),
        expected_layers=settings.get("expected_layers"),
        require_dense=bool(settings.get("require_dense", True)),
    )
    resolved_commit = getattr(model_config, "_commit_hash", None)
    requested_revision = str(settings["revision"])
    if resolved_commit and str(resolved_commit) != requested_revision:
        raise RuntimeError(
            "resolved coordinator commit differs from the sealed V2 revision "
            f"({resolved_commit} != {requested_revision})"
        )
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"], **common)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("V2 coordinator tokenizer exposes no EOS token")
    uses_chat_template = bool(settings.get("use_chat_template", False))
    sample_ids: list[int] = []
    if uses_chat_template:
        if not getattr(tokenizer, "chat_template", None) or not callable(
            getattr(tokenizer, "apply_chat_template", None)
        ):
            raise RuntimeError("V2 coordinator tokenizer exposes no chat template")
        sample_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "CFTN preflight"}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if not sample_ids:
            raise RuntimeError("V2 coordinator chat template produced no tokens")
    return {
        "model_name": str(settings["model_name"]),
        "requested_revision": requested_revision,
        "resolved_commit": str(resolved_commit or requested_revision),
        "model_type": str(model_config.model_type),
        "architectures": [
            str(value) for value in getattr(model_config, "architectures", ()) or ()
        ],
        "dense": True,
        "hidden_size": int(model_config.hidden_size),
        "layers": int(model_config.num_hidden_layers),
        "dtype": str(settings.get("dtype")),
        "chat_template": uses_chat_template,
        "chat_template_probe_tokens": len(sample_ids),
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": str(transformers_version),
        "weights_downloaded_by_preflight": False,
    }


def _runtime_preflight(
    config: dict[str, Any], *, device: str, wandb: bool
) -> dict[str, Any]:
    if sys.version_info < (3, 11):
        raise RuntimeError("CFTN V2 requires Python 3.11 or newer")
    _validate_wandb_environment(config, wandb)
    revision = _multi_revision(config)
    _validate_wandb_environment(revision, wandb)
    coordinator = _coordinator_preflight(config)

    artifact_root = _project_path(config, "artifact_root")
    data_root = _project_path(config, "data_root")
    multi_data_root = Path(revision["paths"]["data_root"])
    _verify_writable(artifact_root)
    _verify_writable(data_root)
    _verify_writable(multi_data_root)
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
        ) or any(
            str(revision.get(section, {}).get("precision", "")).casefold()
            == "bf16"
            for section in ("string_training", "integration_training")
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

    report = {
        "format": "cftn_text_v2_startup_preflight_v2",
        "state": "passed",
        "checked_unix": time.time(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repository_root": str(repository_root),
        "repository_revision": _git_revision(repository_root),
        "config_path": config["_meta"]["path"],
        "config_sha256": config["_meta"]["sha256"],
        "multi_specialist_config_path": revision["_meta"]["path"],
        "multi_specialist_revision_sha256": revision["_meta"]["sha256"],
        "artifact_root": str(artifact_root),
        "data_root": str(data_root),
        "multi_specialist_data_root": str(multi_data_root),
        "storage": {
            "artifact_free_bytes": int(shutil.disk_usage(artifact_root).free),
            "data_free_bytes": int(shutil.disk_usage(data_root).free),
        },
        "coordinator": coordinator,
        "gpu": gpu,
        "wandb": {
            "enabled": wandb,
            "mode": str(config.get("wandb", {}).get("mode", "online")),
            "project": str(config.get("wandb", {}).get("project", "cftn-text-v2")),
            "group": str(config.get("wandb", {}).get("group", "")),
            "entity": config.get("wandb", {}).get("entity") or None,
            "api_key_present": bool(os.environ.get("WANDB_API_KEY")),
        },
        "multi_specialist_initialization": {
            "mode": revision["prerequisite"]["mode"],
            "bridges": "fresh_contextual_bridges_zero_initialized_receivers",
            "prior_reports_gate_training": False,
            "active_specialists": revision["runtime"]["specialist_names"],
            "reserved_specialists": revision["specialist_registry"]["reserved"],
            "hard_transition_baseline_required": True,
            "hardening_trainable_components": revision["integration_training"][
                "phases"
            ][-1]["trainable_components"],
            "hardening_objective": "wake_required_set_only",
            "conditional_specialist_execution": True,
            "hard_halt_enabled": False,
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
    revision = _multi_revision(config)
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
        "multi_specialist_revision_sha256": revision["_meta"]["sha256"],
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
    revision = _multi_revision(config)
    preview = {
        "project": config["project"]["name"],
        "execute": args.execute,
        "resume": args.resume,
        "wandb": args.wandb,
        "wandb_api_key_source": "WANDB_API_KEY environment variable",
        "single_pipeline_lock": str(
            (_project_path(config, "artifact_root") / "pipeline.lock").resolve()
        ),
        "training_contract": "scaled_multi_specialist_with_typed_learned_dispatch_v2",
        "bridge_initialization": "fresh_contextual_bridges_zero_initialized_receivers",
        "prior_reports_gate_training": False,
        "active_specialists": revision["runtime"]["specialist_names"],
        "reserved_specialist_slots": revision["specialist_registry"]["reserved"],
        "dispatch_contract": {
            "learned_finite_graph": True,
            "value_invariant_source_spans": True,
            "lossless_typed_requests": True,
            "deterministic_result_composition": True,
            "unsupported_and_low_confidence_fail_closed": True,
            "oracle_metadata_visible_to_runtime": False,
        },
        "hardening_contract": {
            "zero_update_hard_baseline": True,
            "trainable_components": revision["integration_training"]["phases"][-1][
                "trainable_components"
            ],
            "maximum_gate_learning_rate": revision["integration_training"]["phases"][-1][
                "learning_rate"
            ],
            "objective": "wake_required_set_only",
            "halt_gate_frozen": True,
            "hard_halt_enabled": False,
            "conditional_specialist_execution": True,
        },
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
