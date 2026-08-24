from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v2_learned_dispatch import load_learned_dispatcher
from tools.recover_v1_3_answer_bus import evaluate_native_answer_bus


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _revision(base: dict[str, Any]) -> dict[str, Any]:
    value = Path(base["multi_specialist"]["revision_config"])
    if not value.is_absolute():
        value = Path(base["_meta"]["path"]).parent.parent / value
    return load_v1_3_config(value)


def _validated_hard_checkpoint(revision: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(revision["paths"]["artifact_root"])
    summary = _load_object(root / "hardened_wake" / "summary.json")
    if summary.get("state") != "completed":
        raise RuntimeError("V2 hard-wake phase is not complete")
    if summary.get("revision_sha256") != revision["_meta"]["sha256"]:
        raise RuntimeError("V2 hard-wake source belongs to a stale revision")
    optimizer = summary.get("optimizer_contract", {})
    if not (
        optimizer.get("group_names") == ["gates"]
        and optimizer.get("trainable_components") == ["wake_gates"]
        and optimizer.get("halt_gate_frozen") is True
    ):
        raise RuntimeError("V2 hard-wake source violates its gate-only contract")
    metrics = summary.get("best_metrics") or summary.get("final_metrics", {})
    if metrics.get("hardening_acceptance", {}).get("gates", {}).get("pass") is not True:
        raise RuntimeError("V2 hard-wake source failed its acceptance contract")
    checkpoint = Path(str(summary.get("best_checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"V2 hard-wake checkpoint is missing: {checkpoint}")
    actual_hash = file_sha256(checkpoint)
    if actual_hash != str(summary.get("best_checkpoint_sha256", "")):
        raise RuntimeError("V2 hard-wake checkpoint hash differs from its summary")
    return checkpoint, summary


def evaluate_v2_native_dispatch(
    base: dict[str, Any], *, device_name: str
) -> dict[str, Any]:
    revision = _revision(base)
    root = Path(revision["paths"]["artifact_root"])
    settings = revision["native_dispatch_evaluation"]
    dispatcher_root = root / str(revision["dispatcher"]["artifact_directory"])
    dispatcher_summary_path = dispatcher_root / "summary.json"
    dispatcher_summary = _load_object(dispatcher_summary_path)
    if (
        dispatcher_summary.get("state") != "passed"
        or dispatcher_summary.get("acceptance", {}).get("gates", {}).get("pass")
        is not True
    ):
        raise RuntimeError("V2 learned dispatcher did not pass its sealed panels")
    if dispatcher_summary.get("revision_sha256") != revision["_meta"]["sha256"]:
        raise RuntimeError("V2 dispatcher checkpoint belongs to a stale revision")
    dispatcher_checkpoint = Path(str(dispatcher_summary.get("checkpoint", ""))).resolve()
    if not dispatcher_checkpoint.is_file():
        raise FileNotFoundError(dispatcher_checkpoint)
    if file_sha256(dispatcher_checkpoint) != dispatcher_summary.get("checkpoint_sha256"):
        raise RuntimeError("V2 dispatcher checkpoint hash differs from its summary")
    sealed_dispatcher = load_learned_dispatcher(dispatcher_checkpoint, device="cpu")
    if sealed_dispatcher.metadata.get("revision_sha256") != revision["_meta"]["sha256"]:
        raise RuntimeError("V2 dispatcher checkpoint metadata names a stale revision")
    if sealed_dispatcher.metadata.get("base_config_sha256") != base["_meta"]["sha256"]:
        raise RuntimeError("V2 dispatcher checkpoint metadata names a stale base config")

    hard_checkpoint, hard_summary = _validated_hard_checkpoint(revision)
    phase = {
        "end_to_end_examples_by_class": dict(settings["examples_by_class"]),
        "end_to_end_acceptance": dict(settings["task_acceptance"]),
    }
    artifact = root / str(settings["artifact_directory"])
    artifact.mkdir(parents=True, exist_ok=True)
    status_path = artifact / "status.json"
    atomic_json_dump(
        {
            "format": "cftn_text_v2_native_dispatch_status_v1",
            "state": "running",
            "checkpoint": str(hard_checkpoint),
            "dispatcher_checkpoint": str(dispatcher_checkpoint),
        },
        status_path,
    )
    native = evaluate_native_answer_bus(
        revision,
        phase,
        hard_checkpoint,
        device_name=device_name,
        output_artifact=artifact,
        lossless_request_mode="learned_dispatcher_no_latent",
        deterministic_answer_composition=True,
        specialist_generation_policy=str(settings["specialist_generation_policy"]),
        dispatcher_checkpoint=dispatcher_checkpoint,
        dispatcher_loader=load_learned_dispatcher,
    )
    thresholds = settings["dispatch_acceptance"]
    gates = dict(native["acceptance"]["gates"])
    gates.update(
        {
            "dispatcher_training": dispatcher_summary.get("state") == "passed",
            "dispatch_plan_valid_rate": float(native["dispatch_plan_valid_rate"])
            >= float(thresholds["minimum_plan_valid_rate"]),
            "dispatch_completion_rate": float(native["dispatch_completion_rate"])
            >= float(thresholds["minimum_completion_rate"]),
            "oracle_metadata_hidden": native["oracle_metadata_visible_to_runtime"]
            is False,
            "deterministic_composition": native["deterministic_answer_composition"]
            is True,
            "lossless_no_latent_requests": native["lossless_request_mode"]
            == "learned_dispatcher_no_latent",
            "hard_source_gate_only": True,
        }
    )
    gates["pass"] = all(value for name, value in gates.items() if name != "pass")
    report = {
        **native,
        "format": "cftn_text_v2_native_typed_dispatch_evaluation_v1",
        "state": "passed" if gates["pass"] else "failed_acceptance",
        "base_config_sha256": base["_meta"]["sha256"],
        "revision_sha256": revision["_meta"]["sha256"],
        "source_hard_summary": str(
            (root / "hardened_wake" / "summary.json").resolve()
        ),
        "source_hard_summary_sha256": file_sha256(
            root / "hardened_wake" / "summary.json"
        ),
        "source_optimizer_contract": hard_summary["optimizer_contract"],
        "dispatcher_summary": str(dispatcher_summary_path.resolve()),
        "dispatcher_summary_sha256": file_sha256(dispatcher_summary_path),
        "acceptance": {
            "gates": gates,
            "task_thresholds": settings["task_acceptance"],
            "dispatch_thresholds": thresholds,
        },
    }
    atomic_json_dump(report, artifact / "report.json")
    atomic_json_dump(
        {
            "format": "cftn_text_v2_native_dispatch_status_v1",
            "state": report["state"],
            "report": str((artifact / "report.json").resolve()),
            "acceptance": report["acceptance"],
        },
        status_path,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V2 learned dispatch and deterministic answer composition"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate_v2_native_dispatch(
        load_config(args.config), device_name=args.device
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["state"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
