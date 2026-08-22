from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.config import canonical_json
from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_training import train_integration_phase
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


PHASE_NAME = "oracle_hard_fusion_recovery"
EXPECTED_SOURCE_SHA256 = (
    "236d3b4b71b595cf99f6babf2d17f090823f5a2b716e32c671487a471becdb99"
)
ALL_CLASSES = [
    "pure_language",
    "explicit_math",
    "exact_string",
    "language_dependent_math",
    "multi_parallel",
    "multi_sequential",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def resolve_source_checkpoint(
    config: dict[str, Any], requested: str
) -> tuple[Path, dict[str, Any]]:
    artifact_root = Path(config["paths"]["artifact_root"])
    summary_path = (
        artifact_root / "oracle_hard_adapter_continuation" / "summary.json"
    )
    summary = _load_json(summary_path)
    source = (
        Path(str(summary["best_checkpoint"]))
        if requested == "auto"
        else Path(requested)
    )
    if not source.is_file():
        raise FileNotFoundError(f"fusion source checkpoint is missing: {source}")
    source_sha256 = file_sha256(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "fusion recovery source is not the preserved epoch-2 checkpoint; "
            f"expected {EXPECTED_SOURCE_SHA256}, found {source_sha256}"
        )
    if str(summary.get("best_checkpoint_sha256")) != source_sha256:
        raise RuntimeError("continuation summary and source checkpoint disagree")
    return source, summary


def validate_route_evidence(
    config: dict[str, Any], source: Path
) -> tuple[Path, dict[str, Any]]:
    report_path = (
        Path(config["paths"]["artifact_root"])
        / "route_schedule_sweep_all_lengths"
        / "report.json"
    )
    report = _load_json(report_path)
    if report.get("state") != "completed":
        raise RuntimeError("route-schedule sweep is not complete")
    if report.get("route_can_materially_help") is not False:
        raise RuntimeError("route evidence does not justify a fusion recovery")
    if report.get("recommended_next_step") != "add_specialist_aware_fusion_adapter":
        raise RuntimeError("route evidence recommends a different next step")
    if str(report.get("checkpoint_sha256")) != file_sha256(source):
        raise RuntimeError("route evidence used a different source checkpoint")
    if Path(str(report.get("checkpoint"))).resolve() != source.resolve():
        raise RuntimeError("route evidence source path differs from recovery source")
    return report_path, report


def configure_fusion_recovery(
    config: dict[str, Any], source: Path
) -> dict[str, Any]:
    phase = {
        "name": PHASE_NAME,
        "objective_mode": "oracle_hard_fusion",
        "selection_mode": "adapter_recovery",
        "selection_focus_classes": ["exact_string", "multi_parallel"],
        "wake_mode": "oracle",
        "maximum_rounds": 2,
        "include_task_classes": list(ALL_CLASSES),
        "train_task_classes": [
            name for name in ALL_CLASSES if name != "pure_language"
        ],
        "train_examples_by_class": {
            "explicit_math": 1000,
            "exact_string": 5000,
            "language_dependent_math": 1000,
            "multi_parallel": 10000,
            "multi_sequential": 1000,
        },
        "task_class_weights": {
            "explicit_math": 0.5,
            "exact_string": 2.0,
            "language_dependent_math": 0.5,
            "multi_parallel": 4.0,
            "multi_sequential": 1.0,
        },
        "conditional_execution": True,
        "apply_halt": False,
        "repair_sequential_orders": True,
        "num_workers": 0,
        "batch_size": 16,
        "eval_batch_size": 16,
        "max_epochs": 6,
        "minimum_epochs": 2,
        "early_stop_patience": 2,
        "learning_rate": 1.0e-4,
        "fusion_learning_rate": 1.0e-4,
        "receiver_learning_rate": 5.0e-7,
        "minimum_learning_rate": 1.0e-5,
        "warmup_fraction": 0.05,
        "weight_decay": 0.01,
        "zero_update_candidate": True,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "adapter_acceptance": {
            "minimum_protocol_semantic_sequence_accuracy": 0.83,
            "minimum_causal_message_loss_gap": 5.0,
            "protected_task_classes": {
                "explicit_math": 0.95,
                "language_dependent_math": 0.95,
                "multi_sequential": 0.95,
            },
        },
    }
    phases = [
        value
        for value in config["integration_training"]["phases"]
        if value["name"] != PHASE_NAME
    ]
    config["integration_training"]["phases"] = [*phases, phase]
    return phase


def build_contract(
    config: dict[str, Any],
    phase: dict[str, Any],
    source: Path,
    source_summary: dict[str, Any],
    route_report_path: Path,
    route_report: dict[str, Any],
) -> dict[str, Any]:
    root = Path(__file__).parents[1]
    payload = {
        "format": "cftn_text_v1_3_fusion_recovery_contract_v1",
        "revision_sha256": config["_meta"]["sha256"],
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "source_best_epoch": source_summary.get("best_metrics", {}).get("epoch"),
        "route_evidence": str(route_report_path.resolve()),
        "route_evidence_sha256": file_sha256(route_report_path),
        "route_evidence_examples": route_report.get("full_examples"),
        "route_can_materially_help": route_report.get("route_can_materially_help"),
        "recommended_next_step": route_report.get("recommended_next_step"),
        "phase": phase,
        "invariants": {
            "legacy_checkpoint_identity_at_initialization": True,
            "specialist_towers_frozen": True,
            "request_bridges_frozen": True,
            "return_bridges_frozen": True,
            "specialist_receivers_frozen": True,
            "wake_gates_frozen": True,
            "halt_gate_frozen": True,
            "trainable_components": ["message_fusion", "gpt_receivers"],
            "oracle_hard_routes": True,
            "zero_update_source_is_checkpoint_candidate": True,
        },
        "source_files": {
            name: file_sha256(root / relative)
            for name, relative in {
                "v1_3_fusion.py": "cftn_text/v1_3_fusion.py",
                "v1_3_model.py": "cftn_text/v1_3_model.py",
                "v1_3_training.py": "cftn_text/v1_3_training.py",
                "recover_v1_3_fusion.py": "tools/recover_v1_3_fusion.py",
            }.items()
        },
    }
    payload["contract_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def smoke_fusion_recovery(
    config: dict[str, Any], phase: dict[str, Any], source: Path, *, device_name: str
) -> dict[str, Any]:
    import torch

    from cftn_text.checkpoint import gpu_status
    from cftn_text.tokenizer import ByteMathTokenizer
    from cftn_text.training import (
        autocast_context,
        precision_dtype,
        resolve_device,
    )
    from cftn_text.v1_3_dataset import (
        V13Dataset,
        V13JointCollator,
        move_v1_3_batch,
    )
    from cftn_text.v1_3_training import (
        build_v1_3_model,
        load_v1_3_data_contract,
        v1_3_objective,
    )

    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=source
    )
    model.set_trainable_phase(PHASE_NAME)
    model.train()
    all_records = V13Dataset(
        data_root / manifest["splits"]["joint_train"]["path"]
    ).records
    selected: list[dict[str, Any]] = []
    per_class = {name: 0 for name in phase["train_task_classes"]}
    for record in all_records:
        task_class = str(record["task_class"])
        if task_class in per_class and per_class[task_class] < 4:
            selected.append(record)
            per_class[task_class] += 1
        if len(selected) >= int(phase["batch_size"]):
            break
    if len(selected) != int(phase["batch_size"]):
        raise RuntimeError("could not assemble the fusion smoke batch")
    collator = V13JointCollator(
        ByteMathTokenizer(),
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    batch = move_v1_3_batch(collator(selected), device)
    dtype = precision_dtype(config["integration_training"]["precision"], device)
    with autocast_context(device, dtype):
        _output, loss, components = v1_3_objective(
            model,
            batch,
            wake_mode="oracle",
            maximum_rounds=2,
            settings=config["integration_training"],
            global_step=1,
            objective_mode="oracle_hard_fusion",
            conditional_execution=True,
            apply_halt=False,
            task_class_weights=phase["task_class_weights"],
        )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("fusion smoke loss is non-finite")
    loss.backward()
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    gradient_names = {
        name
        for name, value in model.named_parameters()
        if value.grad is not None and bool(value.grad.detach().abs().sum().gt(0))
    }
    unexpected_trainable = sorted(
        name
        for name in trainable
        if not name.startswith(("message_fusion.", "gpt_tower.receivers."))
    )
    unexpected_gradients = sorted(
        name
        for name in gradient_names
        if not name.startswith(("message_fusion.", "gpt_tower.receivers."))
    )
    projection_name = "message_fusion.output_projection.weight"
    if unexpected_trainable or unexpected_gradients or projection_name not in gradient_names:
        raise RuntimeError(
            "fusion smoke violated its gradient contract; "
            f"unexpected_trainable={unexpected_trainable}, "
            f"unexpected_gradients={unexpected_gradients}, "
            f"projection_gradient={projection_name in gradient_names}"
        )
    return {
        "format": "cftn_text_v1_3_fusion_smoke_v1",
        "state": "passed",
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "batch_examples": len(selected),
        "batch_class_counts": per_class,
        "loss": float(loss.detach()),
        "components": components,
        "trainable_parameter_count": model.trainable_parameter_count(),
        "trainable_tensor_count": len(trainable),
        "nonzero_gradient_tensor_count": len(gradient_names),
        "optimizer_groups": ["message_fusion", "gpt_receivers"],
        "provenance": provenance,
        "gpu": gpu_status(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover V1.3 parallel composition with typed message fusion"
    )
    parser.add_argument(
        "--config", default="G:/ctfn-text/config/v1_3_multi_specialist.yaml"
    )
    parser.add_argument("--source-checkpoint", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one real-checkpoint forward/backward contract check and exit",
    )
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()

    config = load_v1_3_config(args.config)
    source, source_summary = resolve_source_checkpoint(
        config, args.source_checkpoint
    )
    route_path, route_report = validate_route_evidence(config, source)
    phase = configure_fusion_recovery(config, source)
    if args.dry_run:
        print(
            json.dumps(
                smoke_fusion_recovery(
                    config, phase, source, device_name=args.device
                ),
                indent=2,
            )
        )
        return
    artifact_root = Path(config["paths"]["artifact_root"])
    artifact = artifact_root / PHASE_NAME
    if not args.resume and artifact.is_dir() and any(artifact.glob("*.pth")):
        raise RuntimeError(
            f"fusion recovery already has checkpoints; use --resume: {artifact}"
        )
    contract = build_contract(
        config, phase, source, source_summary, route_path, route_report
    )
    contract_path = artifact_root / "fusion_recovery_contract.json"
    status_path = artifact_root / "fusion_recovery_pipeline.json"
    atomic_json_dump(contract, contract_path)
    started = time.time()
    status: dict[str, Any] = {
        "format": "cftn_text_v1_3_fusion_recovery_pipeline_v1",
        "state": "running",
        "pid": os.getpid(),
        "phase": PHASE_NAME,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract["contract_sha256"],
        "started_unix": started,
    }
    atomic_json_dump(status, status_path)
    try:
        result = train_integration_phase(
            config,
            PHASE_NAME,
            device_name=args.device,
            resume=args.resume,
            max_batches=args.max_batches,
            wandb_options=wandb_options_from_args(
                args, default_run_name="v1-3-oracle-hard-fusion-recovery"
            ),
        )
        status.update(
            {
                "state": result["state"],
                "result": result,
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        atomic_json_dump(status, status_path)
    except BaseException as exc:
        status.update(
            {
                "state": "error",
                "error": repr(exc),
                "failed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        atomic_json_dump(status, status_path)
        raise
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
