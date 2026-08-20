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


ALL_CLASSES = [
    "pure_language",
    "explicit_math",
    "exact_string",
    "language_dependent_math",
    "multi_parallel",
    "multi_sequential",
]
ADAPTER_PHASE = "oracle_hard_adapter_recovery"
ROUTER_PHASE = "hard_router_recovery"


def configure_recovery(config: dict[str, Any]) -> dict[str, Any]:
    source_phases = [
        phase
        for phase in config["integration_training"]["phases"]
        if phase["name"]
        in {
            "single_specialist_capacity",
            "dense_mixed_messages",
            "dense_recurrent",
            "supervised_soft_wake",
        }
    ]
    adapter = {
        "name": ADAPTER_PHASE,
        "objective_mode": "oracle_hard_adapter",
        "wake_mode": "oracle",
        "maximum_rounds": 2,
        "include_task_classes": list(ALL_CLASSES),
        "conditional_execution": True,
        "apply_halt": False,
        "repair_sequential_orders": True,
        "max_epochs": 6,
        "minimum_epochs": 2,
        "early_stop_patience": 3,
        "learning_rate": 1.0e-6,
        "gate_learning_rate": 1.0e-6,
        "minimum_learning_rate": 1.0e-7,
        "warmup_fraction": 0.02,
        "weight_decay": 0.01,
    }
    router = {
        "name": ROUTER_PHASE,
        "objective_mode": "router_calibration",
        "wake_mode": "hard_straight_through",
        "maximum_rounds": 2,
        "include_task_classes": list(ALL_CLASSES),
        "conditional_execution": True,
        "apply_halt": False,
        "repair_sequential_orders": True,
        "max_epochs": 8,
        "minimum_epochs": 1,
        "early_stop_patience": 3,
        "learning_rate": 5.0e-7,
        "gate_learning_rate": 5.0e-7,
        "minimum_learning_rate": 5.0e-8,
        "warmup_fraction": 0.0,
        "weight_decay": 0.01,
        "routing_acceptance": {
            "minimum_exact_required_set_accuracy": 0.95,
            "minimum_wake_precision": 0.95,
            "minimum_wake_recall": 0.98,
            "maximum_pure_language_false_wake_rate": 0.01,
            "maximum_all_open_rate": 0.01,
            "maximum_all_closed_rate": 0.25,
        },
    }
    config["integration_training"]["phases"] = [*source_phases, adapter, router]
    return {"adapter": adapter, "router": router}


def _contract(config: dict[str, Any], phases: dict[str, Any]) -> dict[str, Any]:
    artifact_root = Path(config["paths"]["artifact_root"])
    source_summary = json.loads(
        (artifact_root / "supervised_soft_wake" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_checkpoint = Path(source_summary["best_checkpoint"])
    payload = {
        "format": "cftn_text_v1_3_hard_binary_recovery_contract_v1",
        "source_phase": "supervised_soft_wake",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "source_config_sha256": config["_meta"]["sha256"],
        "source_manifest": str(
            (Path(config["paths"]["data_root"]) / "manifest.json").resolve()
        ),
        "phases": phases,
        "invariants": {
            "maximum_rounds": 2,
            "binary_message_masks": True,
            "all_closed_receiver_bypass": True,
            "conditional_specialist_execution": True,
            "hard_halt_enabled": False,
            "post_halt_wake_labels_ignored": True,
            "balanced_sequential_orders": True,
            "task_gradients_into_router": False,
        },
        "source_files": {
            "v1_3_model.py": file_sha256(
                Path(__file__).parents[1] / "cftn_text" / "v1_3_model.py"
            ),
            "v1_3_training.py": file_sha256(
                Path(__file__).parents[1] / "cftn_text" / "v1_3_training.py"
            ),
            "bridges.py": file_sha256(
                Path(__file__).parents[1] / "cftn_text" / "bridges.py"
            ),
        },
    }
    payload["contract_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated V1.3 binary-routing recovery"
    )
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument(
        "--stage", choices=("adapter", "router", "pipeline"), default="pipeline"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()

    config = load_v1_3_config(args.config)
    phases = configure_recovery(config)
    artifact_root = Path(config["paths"]["artifact_root"])
    contract = _contract(config, phases)
    contract_path = artifact_root / "hard_binary_recovery_contract.json"
    atomic_json_dump(contract, contract_path)
    status_path = artifact_root / "hard_binary_recovery_pipeline.json"
    selected = (
        [ADAPTER_PHASE, ROUTER_PHASE]
        if args.stage == "pipeline"
        else [ADAPTER_PHASE if args.stage == "adapter" else ROUTER_PHASE]
    )
    started = time.time()
    status: dict[str, Any] = {
        "format": "cftn_text_v1_3_hard_binary_recovery_pipeline_v1",
        "state": "running",
        "pid": os.getpid(),
        "selected_phases": selected,
        "completed_phases": [],
        "current_phase": None,
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract["contract_sha256"],
        "started_unix": started,
    }
    atomic_json_dump(status, status_path)
    results: dict[str, Any] = {}
    try:
        for phase_name in selected:
            status["current_phase"] = phase_name
            atomic_json_dump(status, status_path)
            result = train_integration_phase(
                config,
                phase_name,
                device_name=args.device,
                resume=args.resume,
                max_batches=args.max_batches,
                wandb_options=wandb_options_from_args(
                    args, default_run_name=f"v1-3-{phase_name.replace('_', '-')}"
                ),
            )
            results[phase_name] = result
            status["completed_phases"].append(phase_name)
        status.update(
            {
                "state": "completed",
                "current_phase": None,
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "results": results,
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
