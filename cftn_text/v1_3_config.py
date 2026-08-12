from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .config import _expand_environment, canonical_json
from .data_generator import file_sha256


REVISION_FORMAT = "cftn_text_multi_specialist_revision_v1_3"
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
    if not isinstance(raw, dict) or raw.get("format") != REVISION_FORMAT:
        raise ValueError("unsupported V1.3 revision configuration")
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
    for key in (
        "base_config",
        "v1_2_artifact_root",
        "v1_2_report",
        "v1_2_pipeline_status",
        "math_checkpoint",
        "data_root",
        "artifact_root",
    ):
        config["paths"][key] = str(_resolve(config["paths"][key], repository_root))
    specialists = list(config["runtime"].get("specialist_names", []))
    if specialists != ["math", "string"]:
        raise ValueError("V1.3 preregistration requires specialists [math, string]")
    rounds = int(config["runtime"].get("maximum_callosal_rounds", 0))
    if rounds < 2 or rounds > 3:
        raise ValueError("V1.3 maximum_callosal_rounds must be 2 or 3")
    threshold = float(config["runtime"].get("wake_threshold", -1))
    if not 0 < threshold < 1:
        raise ValueError("V1.3 wake threshold must be within (0, 1)")
    shares = config["data"].get("task_shares", {})
    if abs(sum(float(value) for value in shares.values()) - 1.0) > 1e-8:
        raise ValueError("V1.3 joint task shares must sum to one")
    interface = config["gpt_interface"]
    if interface.get("answer_protocol") != "first_nonempty_completion_line_v1":
        raise ValueError("V1.3 requires the registered first-line GPT answer protocol")
    if interface.get("pure_language_prompt_style") != "archival_key_value_v1":
        raise ValueError("V1.3 pure-language prompt style differs from preregistration")
    if str(interface.get("generic_answer_cue")) != "Exact result:":
        raise ValueError("V1.3 generic GPT answer cue differs from preregistration")
    if str(interface.get("completion_terminator")) != "\n":
        raise ValueError("V1.3 GPT completion terminator must be one newline")
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
    """Verify the sealed V1.2 result before any V1.3 task is allowed to run."""

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
