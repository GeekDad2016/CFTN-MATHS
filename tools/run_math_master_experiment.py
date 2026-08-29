from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.math_curriculum_data import audit_dataset, prepare_dataset
from cftn_text.training import train_math_tower


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_contract(manifest: dict) -> dict:
    phases = list(manifest["phases"])
    panels: list[dict] = []
    contract_phases: list[dict] = []
    prior: list[str] = []
    for index, phase in enumerate(phases):
        active = [str(value) for value in phase["criteria"]]
        active_name = f"active_{index:02d}"
        active_split = f"phase_{index:02d}_active"
        panels.append(
            {
                "name": active_name,
                "split": active_split,
                "examples": int(manifest["splits"][active_split]["records"]),
                "batch_size": 8,
                "max_new_tokens": 128,
                "failure_examples": 4,
            }
        )
        quota_groups = [
            {
                "name": "active",
                "examples": 96 if not prior else 72,
                "filters": {"families": active},
                "balance_families": True,
            }
        ]
        minimum_by_panel = {active_name: 0.85}
        minimum_valid_by_panel = {active_name: 0.95}
        panel_family: dict[str, dict[str, float]] = {}
        if prior:
            retention_name = f"retention_{index:02d}"
            retention_split = f"phase_{index:02d}_retention"
            panels.append(
                {
                    "name": retention_name,
                    "split": retention_split,
                    "examples": int(manifest["splits"][retention_split]["records"]),
                    "batch_size": 8,
                    "max_new_tokens": 128,
                    "failure_examples": 4,
                }
            )
            quota_groups.append(
                {
                    "name": "cumulative_replay",
                    "examples": 24,
                    "filters": {"families": list(prior)},
                    "balance_families": True,
                }
            )
            minimum_by_panel[retention_name] = 1.0
            minimum_valid_by_panel[retention_name] = 1.0
            panel_family[retention_name] = {criterion: 1.0 for criterion in prior}
        contract_phases.append(
            {
                "name": phase["name"],
                "minimum_epochs": 2,
                "maximum_epochs": 8,
                "advance_after_consecutive_passes": 2,
                "stop_on_pass": index == len(phases) - 1,
                "families": [*prior, *active],
                "max_difficulty": 3,
                "quota_groups": quota_groups,
                "primary_generation_panel": active_name,
                "minimum_generation_accuracy": 0.85,
                "minimum_valid_rate": 0.95,
                "minimum_generation_accuracy_by_family": {
                    criterion: 0.75 for criterion in active
                },
                "minimum_trace_exact_by_family": {
                    criterion: 0.70 for criterion in active
                },
                "minimum_generation_accuracy_by_panel": minimum_by_panel,
                "minimum_valid_rate_by_panel": minimum_valid_by_panel,
                "minimum_generation_accuracy_by_panel_family": panel_family,
            }
        )
        prior.extend(active)
    return {
        "format": "cftn_math_master_experiment_training_v1",
        "require_acceptance_for_best": True,
        "promote_final_phase_only": True,
        "production_acceptance": False,
        "remaining_pipeline_enabled": False,
        "math_training": {
            "max_epochs": sum(int(phase["maximum_epochs"]) for phase in contract_phases),
            "generation_validation": {
                "enabled": True,
                "panel_scope": "phase_required_v1",
                "require_eos": False,
                "panels": panels,
            },
        },
        "curriculum": {
            "transition_policy": "competency_gated_v3",
            "examples_per_epoch": 96,
        },
        "phases": contract_phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, start, or resume the compact local math master experiment"
    )
    parser.add_argument(
        "command", choices=("auto", "build", "audit", "smoke"), default="auto", nargs="?"
    )
    parser.add_argument("--config", default="config/math_master_experiment_local.yaml")
    parser.add_argument("--dataset-config", default="config/math_master_experiment_v1.json")
    parser.add_argument("--data", default="C:/CFTN/.datasets/math_master_experiment_v1")
    parser.add_argument("--artifact", default="C:/CFTN/artifacts/math_master_experiment_v1/run")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        if args.command == "audit":
            raise FileNotFoundError(f"dataset is not built: {data_root}")
        prepare_dataset(_load_json(Path(args.dataset_config)), data_root)
    audit = audit_dataset(data_root)
    manifest = _load_json(manifest_path)
    if args.command in {"build", "audit"}:
        print(
            json.dumps(
                {
                    "status": audit["status"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "phases": len(manifest["phases"]),
                    "records": audit["records"],
                    "phase_validation": audit["phase_validation"],
                },
                indent=2,
            )
        )
        return

    config = load_config(args.config)
    config = copy.deepcopy(config)
    config["project"]["data_root"] = str(data_root)
    config["project"]["artifact_root"] = str(Path(args.artifact).resolve().parent)
    config["data"]["dataset_manifest_sha256"] = manifest["manifest_sha256"]
    contract = build_contract(manifest)
    max_batches = None
    if args.command == "smoke":
        smoke_phase = copy.deepcopy(contract["phases"][0])
        smoke_phase.update(
            {
                "minimum_epochs": 1,
                "maximum_epochs": 1,
                "advance_after_consecutive_passes": 1,
                "stop_on_pass": True,
                "minimum_generation_accuracy": 0.0,
                "minimum_valid_rate": 0.0,
                "minimum_generation_accuracy_by_family": {},
                "minimum_trace_exact_by_family": {},
                "minimum_generation_accuracy_by_panel": {},
                "minimum_valid_rate_by_panel": {},
                "minimum_generation_accuracy_by_panel_family": {},
            }
        )
        contract["phases"] = [smoke_phase]
        contract["math_training"]["max_epochs"] = 1
        contract["math_training"]["generation_validation"]["panels"] = [
            {
                **contract["math_training"]["generation_validation"]["panels"][0],
                "examples": 2,
                "max_new_tokens": 16,
            }
        ]
        max_batches = 1
    config["math_training"]["max_epochs"] = contract["math_training"]["max_epochs"]
    artifact = Path(args.artifact + ("_smoke" if args.command == "smoke" else "")).resolve()
    resume = any(artifact.glob("math.epoch_*.pth")) or (artifact / "math.latest.pth").exists()
    result = train_math_tower(
        config,
        device_name=args.device,
        resume=resume,
        max_batches=max_batches,
        require_calibration=False,
        artifact_directory=artifact,
        working_directory=artifact,
        recovery_contract=contract,
    )
    metrics = result.get("metrics", {})
    gpu = metrics.get("gpu", result.get("gpu", {}))
    transition = metrics.get("curriculum_transition", {})
    print(
        json.dumps(
            {
                "state": result.get("state"),
                "stop_reason": result.get("stop_reason"),
                "epoch": result.get("epoch", metrics.get("epoch")),
                "global_step": result.get("global_step", metrics.get("global_step")),
                "phase": transition.get("phase"),
                "checkpoint_eligible": metrics.get("checkpoint_eligible"),
                "checkpoint_promoted": metrics.get("checkpoint_promoted"),
                "gpu": {
                    "device": gpu.get("device"),
                    "peak_allocated_bytes": gpu.get("peak_allocated_bytes"),
                },
                "artifact": str(artifact),
                "summary": str(artifact / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
