from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .config import _expand_environment, canonical_json
from .data_generator import file_sha256


REVISION_FORMAT = "cftn_text_multi_specialist_revision_v1_3"
V2_REVISION_FORMAT = "cftn_text_multi_specialist_revision_v2"
SUPPORTED_REVISION_FORMATS = {REVISION_FORMAT, V2_REVISION_FORMAT}
V1_2_REPORT_FORMAT = "cftn_text_v1_2_revision_report_v1"


class V13PrerequisiteError(RuntimeError):
    """Raised when V1.3 is correctly blocked by its sealed V1.2 gate."""


def _revision_sha256(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if key != "_meta"}
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def _resolve(value: str, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository_root / path).resolve()


def load_v1_3_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or raw.get("format") not in SUPPORTED_REVISION_FORMATS:
        raise ValueError("unsupported multi-specialist revision configuration")
    config = _expand_environment(raw)
    for section in (
        "revision",
        "prerequisite",
        "paths",
        "data",
        "gpt_interface",
        "string_tower",
        "bridge",
        "runtime",
        "string_training",
        "integration_training",
        "evaluation",
        "acceptance",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"V1.3 configuration requires {section}")
    repository_root = config_path.parent.parent.resolve()
    prerequisite_mode = str(
        config["prerequisite"].get("mode", "sealed_v1_2_transfer")
    )
    common_paths = (
        "base_config",
        "math_checkpoint",
        "data_root",
        "artifact_root",
    )
    strict_paths = (
        "v1_2_artifact_root",
        "v1_2_report",
        "v1_2_pipeline_status",
    )
    for key in common_paths:
        if key not in config["paths"]:
            raise ValueError(f"multi-specialist configuration requires paths.{key}")
        config["paths"][key] = str(_resolve(config["paths"][key], repository_root))
    if prerequisite_mode == "sealed_v1_2_transfer":
        for key in strict_paths:
            if key not in config["paths"]:
                raise ValueError(f"sealed V1.2 transfer requires paths.{key}")
            config["paths"][key] = str(
                _resolve(config["paths"][key], repository_root)
            )
    elif prerequisite_mode == "fresh_multi_specialist":
        for key in (
            "prior_v1_2_report",
            "prior_v1_3_report",
            "math_evaluation_report",
        ):
            if key in config["paths"]:
                config["paths"][key] = str(
                    _resolve(config["paths"][key], repository_root)
                )
    else:
        raise ValueError(f"unsupported prerequisite mode: {prerequisite_mode}")
    specialists = list(config["runtime"].get("specialist_names", []))
    if specialists != ["math", "string"]:
        raise ValueError("this revision requires active specialists [math, string]")
    if raw.get("format") == V2_REVISION_FORMAT:
        registry = config.get("specialist_registry")
        if not isinstance(registry, dict):
            raise ValueError("V2 requires a specialist_registry contract")
        maximum_slots = int(registry.get("maximum_slots", 0))
        active = [str(item.get("name")) for item in registry.get("active", [])]
        reserved = registry.get("reserved", [])
        if maximum_slots != len(active) + len(reserved) or maximum_slots < 2:
            raise ValueError("V2 specialist slot count differs from its registry")
        if active != specialists:
            raise ValueError("V2 active specialist registry differs from runtime order")
        reserved_names = [str(item.get("name")) for item in reserved]
        if not reserved_names or len(set([*active, *reserved_names])) != maximum_slots:
            raise ValueError("V2 specialist registry names must be complete and unique")
        if any(item.get("state") != "reserved_inactive" for item in reserved):
            raise ValueError("every reserved specialist must remain reserved_inactive")
        if any(item.get("train") is not False for item in reserved):
            raise ValueError("reserved specialists must not be trained")
        if any(name in specialists for name in reserved_names):
            raise ValueError("active and reserved specialist slots overlap")
        for section in (
            "dispatcher",
            "answer_composer",
            "native_dispatch_evaluation",
        ):
            if not isinstance(config.get(section), dict):
                raise ValueError(f"V2 configuration requires {section}")
        dispatcher = config["dispatcher"]
        if dispatcher.get("format") != "cftn_text_v2_hierarchical_dispatcher_v2":
            raise ValueError("V2 dispatcher format is not recognized")
        if not 0.5 < float(dispatcher.get("confidence_threshold", 0.0)) < 1.0:
            raise ValueError("V2 dispatcher confidence threshold must be within (0.5, 1)")
        if int(dispatcher.get("maximum_length", 0)) < int(
            config["data"]["maximum_specialist_length"]
        ):
            raise ValueError("V2 dispatcher must cover the specialist prompt limit")
        semantic_encoder = dispatcher.get("semantic_encoder", {})
        dispatcher_model = dispatcher.get("model", {})
        dispatcher_losses = dispatcher.get("losses", {})
        if semantic_encoder.get("source") != "frozen_coordinator_prepass":
            raise ValueError("V2 dispatcher must share the frozen coordinator prepass")
        if semantic_encoder.get("pooling") != "attention_mask_mean_final_hidden_v1":
            raise ValueError("V2 dispatcher semantic pooling contract is not recognized")
        if int(semantic_encoder.get("maximum_length", 0)) < 1:
            raise ValueError("V2 dispatcher semantic length must be positive")
        if int(semantic_encoder.get("batch_size", 0)) < 1:
            raise ValueError("V2 dispatcher semantic batch size must be positive")
        if list(dispatcher_model.get("tower_names", [])) != [
            *active,
            *reserved_names,
        ]:
            raise ValueError("V2 dispatcher tower heads differ from the registry order")
        if list(dispatcher_model.get("active_tower_names", [])) != active:
            raise ValueError("V2 dispatcher active tower heads differ from runtime")
        for key in (
            "semantic_width",
            "semantic_projection_size",
            "structure_projection_size",
            "fusion_size",
            "parameter_target",
            "parameter_tolerance",
        ):
            if int(dispatcher_model.get(key, 0)) < 1:
                raise ValueError(f"V2 dispatcher model.{key} must be positive")
        if set(dispatcher_losses) != {"intent", "delegation", "towers", "rounds"}:
            raise ValueError("V2 dispatcher hierarchical losses are incomplete")
        if any(float(value) <= 0.0 for value in dispatcher_losses.values()):
            raise ValueError("V2 dispatcher loss weights must be positive")
        dispatch_acceptance = dispatcher.get("acceptance", {})
        required_dispatch_gates = {
            "minimum_registered_accuracy",
            "minimum_registered_coverage",
            "minimum_broad_accuracy",
            "minimum_broad_coverage",
            "minimum_semantic_accuracy",
            "minimum_semantic_coverage",
        }
        if not required_dispatch_gates.issubset(dispatch_acceptance):
            raise ValueError("V2 dispatcher acceptance contract is incomplete")
        if any(
            not 0.0 < float(dispatch_acceptance[key]) <= 1.0
            for key in required_dispatch_gates
        ):
            raise ValueError("V2 dispatcher acceptance thresholds must be in (0, 1]")
        native_dispatch = config["native_dispatch_evaluation"]
        if native_dispatch.get("specialist_generation_policy") != "full_context_v1":
            raise ValueError("V2 native dispatch must use full-context specialist generation")
        required_classes = {
            "explicit_math",
            "exact_string",
            "language_dependent_math",
            "multi_parallel",
            "multi_sequential",
        }
        if set(native_dispatch.get("examples_by_class", {})) != required_classes:
            raise ValueError("V2 native dispatch panel does not cover every specialist class")
        if any(
            int(value) < 1
            for value in native_dispatch["examples_by_class"].values()
        ):
            raise ValueError("V2 native dispatch class panels cannot be empty")
    rounds = int(config["runtime"].get("maximum_callosal_rounds", 0))
    if rounds < 2 or rounds > 3:
        raise ValueError("V1.3 maximum_callosal_rounds must be 2 or 3")
    threshold = float(config["runtime"].get("wake_threshold", -1))
    if not 0 < threshold < 1:
        raise ValueError("V1.3 wake threshold must be within (0, 1)")
    shares = config["data"].get("task_shares", {})
    if abs(sum(float(value) for value in shares.values()) - 1.0) > 1e-8:
        raise ValueError("V1.3 joint task shares must sum to one")
    if raw.get("format") == V2_REVISION_FORMAT:
        expected_share_names = {
            "pure_language",
            "explicit_math",
            "exact_string",
            "language_dependent_math",
            "multi_specialist",
        }
        if set(shares) != expected_share_names or any(
            float(value) <= 0.0 for value in shares.values()
        ):
            raise ValueError("V2 joint task shares must define every positive class")
    interface = config["gpt_interface"]
    if interface.get("answer_protocol") != "first_nonempty_completion_line_v1":
        raise ValueError("V1.3 requires the registered first-line GPT answer protocol")
    expected_prompt_style = (
        "open_world_generalist_v2"
        if raw.get("format") == V2_REVISION_FORMAT
        else "archival_key_value_v1"
    )
    if interface.get("pure_language_prompt_style") != expected_prompt_style:
        raise ValueError("pure-language prompt style differs from preregistration")
    if str(interface.get("generic_answer_cue")) != "Exact result:":
        raise ValueError("V1.3 generic GPT answer cue differs from preregistration")
    if str(interface.get("completion_terminator")) != "\n":
        raise ValueError("V1.3 GPT completion terminator must be one newline")
    if (
        raw.get("format") == V2_REVISION_FORMAT
        and interface.get("prompt_transport") != "tokenizer_chat_template_v1"
    ):
        raise ValueError("V2 must use the coordinator tokenizer chat template")
    if int(interface.get("calibration_max_new_tokens", 0)) < 4:
        raise ValueError("V1.3 GPT calibration token budget is too small")
    phases = config["integration_training"].get("phases", [])
    if [phase.get("name") for phase in phases] != [
        "single_specialist_capacity",
        "dense_mixed_messages",
        "dense_recurrent",
        "supervised_soft_wake",
        "hardened_wake",
    ]:
        raise ValueError("V1.3 integration phases differ from the preregistration")
    if raw.get("format") == V2_REVISION_FORMAT:
        hard = phases[-1]
        if list(hard.get("trainable_components", [])) != ["wake_gates"]:
            raise ValueError("V2 hardened_wake must train only the wake gates")
        if config["runtime"].get("conditional_execution_in_hard_mode") is not True:
            raise ValueError("V2 hard wake must physically skip closed specialists")
        if config["runtime"].get("hard_halt_enabled", False) is not False:
            raise ValueError("V2 hard halt must remain disabled until separately calibrated")
        hard_lr = float(hard.get("learning_rate", 0.0))
        if not 0.0 < hard_lr <= 5.0e-7:
            raise ValueError("V2 hardened_wake learning rate must be within (0, 5e-7]")
        if float(hard.get("warmup_fraction", -1.0)) != 0.0:
            raise ValueError("V2 hardened_wake must not restart with warmup")
        transition = config["integration_training"].get("hard_transition_baseline")
        if not isinstance(transition, dict) or transition.get("required") is not True:
            raise ValueError("V2 requires a zero-update hard-transition baseline")
    evaluation = config["evaluation"]
    if evaluation.get("primary_split") != "joint_test":
        raise ValueError("V1.3 primary acceptance split must remain joint_test")
    expected_diagnostics = {
        "joint_heldout_paraphrase",
        "joint_extrapolation",
        "joint_counterfactual",
        "joint_unseen_composition",
    }
    if set(evaluation.get("diagnostic_splits", [])) != expected_diagnostics:
        raise ValueError("V1.3 diagnostic split contract differs from preregistration")
    competence = evaluation.get("competence_conditioning")
    if not isinstance(competence, dict) or not all(
        competence.get(key) is True
        for key in ("enabled", "oracle_native_inputs", "diagnostics_are_non_gating")
    ):
        raise ValueError("V1.3 competence-conditioned evaluation must be enabled")
    config["_meta"] = {
        "path": str(config_path),
        "repository_root": str(repository_root),
        "sha256": _revision_sha256(config),
    }
    return config


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise V13PrerequisiteError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V13PrerequisiteError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise V13PrerequisiteError(f"{label} is not a JSON object: {path}")
    return value


def audit_v1_2_pass(config: dict[str, Any]) -> dict[str, Any]:
    """Verify strict transfer evidence or seal a fresh-training initialization."""

    mode = str(config["prerequisite"].get("mode", "sealed_v1_2_transfer"))
    if mode == "fresh_multi_specialist":
        evidence: dict[str, Any] = {}
        for label, key in (
            ("v1_2", "prior_v1_2_report"),
            ("v1_3", "prior_v1_3_report"),
        ):
            value = config["paths"].get(key)
            if not value:
                continue
            path = Path(value)
            evidence[label] = {
                "path": str(path.resolve()),
                "present": path.is_file(),
                "sha256": file_sha256(path) if path.is_file() else None,
                "role": "informational_provenance_only",
            }
        return {
            "format": "cftn_text_multi_specialist_initialization_audit_v1",
            "state": "passed",
            "mode": mode,
            "bridge_initialization": "fresh_contextual_bridges_zero_initialized_receivers",
            "prior_reports_required": False,
            "prior_reports_gate_training": False,
            "prior_evidence": evidence,
            "revision_sha256": config["_meta"]["sha256"],
        }
    if mode != "sealed_v1_2_transfer":
        raise V13PrerequisiteError(f"unsupported prerequisite mode: {mode}")

    paths = config["paths"]
    report_path = Path(paths["v1_2_report"])
    status_path = Path(paths["v1_2_pipeline_status"])
    report = _read_object(report_path, "V1.2 sealed report")
    status = _read_object(status_path, "V1.2 pipeline status")
    if report.get("format") != V1_2_REPORT_FORMAT:
        raise V13PrerequisiteError("V1.2 report format is not sealed/recognized")
    expected_revision = str(config["prerequisite"]["v1_2_revision_sha256"])
    if report.get("revision_sha256") != expected_revision:
        raise V13PrerequisiteError("V1.2 report revision hash does not match V1.3")
    if status.get("state") != "completed":
        raise V13PrerequisiteError(
            f"V1.2 pipeline is {status.get('state', 'unknown')}, not completed"
        )
    if status.get("revision_sha256") != expected_revision:
        raise V13PrerequisiteError("V1.2 pipeline/report revision hashes differ")
    if status.get("format") != "cftn_text_v1_2_pipeline_status_v1":
        raise V13PrerequisiteError("V1.2 pipeline status format is not recognized")
    required_stages = list(config["prerequisite"].get("required_completed_stages", []))
    if not required_stages:
        raise V13PrerequisiteError("V1.3 has no sealed V1.2 stage contract")
    if status.get("current_stage") is not None:
        raise V13PrerequisiteError("V1.2 still names an active stage")
    if int(status.get("stage_index", -1)) != len(required_stages) or int(
        status.get("stages_total", -1)
    ) != len(required_stages):
        raise V13PrerequisiteError("V1.2 did not reach its terminal stage boundary")
    if list(status.get("completed_stages", [])) != required_stages:
        raise V13PrerequisiteError("V1.2 completed-stage ledger is incomplete")
    gates = report.get("final_gates")
    if not isinstance(gates, dict) or gates.get("pass") is not True:
        raise V13PrerequisiteError("V1.2 did not pass every sealed central gate")
    if bool(config["prerequisite"].get("require_all_v1_2_gates", True)):
        failed = sorted(
            key for key, passed in gates.items() if key != "pass" and passed is not True
        )
        if failed:
            raise V13PrerequisiteError(f"V1.2 has failed central gates: {failed}")
    checkpoint = Path(str(report.get("training", {}).get("best_checkpoint", "")))
    expected_checkpoint_hash = str(
        report.get("training", {}).get("best_checkpoint_sha256", "")
    )
    if not checkpoint.is_file() or not expected_checkpoint_hash:
        raise V13PrerequisiteError("V1.2 best checkpoint provenance is incomplete")
    actual_checkpoint_hash = file_sha256(checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise V13PrerequisiteError("V1.2 best checkpoint hash no longer matches report")
    return {
        "format": "cftn_text_v1_3_prerequisite_audit_v1",
        "state": "passed",
        "v1_2_report": str(report_path.resolve()),
        "v1_2_report_sha256": file_sha256(report_path),
        "v1_2_revision_sha256": expected_revision,
        "v1_2_checkpoint": str(checkpoint.resolve()),
        "v1_2_checkpoint_sha256": actual_checkpoint_hash,
        "v1_2_final_gates": gates,
        "v1_3_revision_sha256": config["_meta"]["sha256"],
    }
