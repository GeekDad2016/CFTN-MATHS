from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.math_curriculum_data import audit_dataset, prepare_dataset
from cftn_text.training import train_math_tower


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_contract(
    manifest: dict,
    *,
    minimum_epochs_per_phase: int = 10,
    maximum_epochs_per_phase: int = 60,
    consecutive_passes: int = 2,
    examples_per_epoch: int = 512,
) -> dict:
    if minimum_epochs_per_phase < 1:
        raise ValueError("minimum_epochs_per_phase must be positive")
    if maximum_epochs_per_phase < minimum_epochs_per_phase:
        raise ValueError("maximum_epochs_per_phase must be at least the minimum")
    if consecutive_passes < 1:
        raise ValueError("consecutive_passes must be positive")
    if examples_per_epoch < 4 or examples_per_epoch % 4:
        raise ValueError("examples_per_epoch must be a positive multiple of four")
    active_examples = examples_per_epoch * 3 // 4
    replay_examples = examples_per_epoch - active_examples
    phases = list(manifest["phases"])
    criterion_operations = {
        str(criterion): [str(operation) for operation in operations]
        for criterion, operations in manifest.get("criterion_operations", {}).items()
    }
    result_balanced_criteria = {
        str(value) for value in manifest.get("result_balanced_criteria", [])
    }
    trace_acceptance_metric = str(
        manifest.get("trace_acceptance_metric", "exact_v1")
    )
    if trace_acceptance_metric not in {"exact_v1", "semantic_v1"}:
        raise ValueError("unsupported trace acceptance metric")
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
                "max_new_tokens": 224,
                "failure_examples": 4,
            }
        )
        quota_groups = [
            {
                "name": "active",
                "examples": examples_per_epoch if not prior else active_examples,
                "filters": {"families": active},
                "balance_families": True,
                "balance_operations_within_families": True,
                "balance_results_within_operations_for_families": sorted(
                    set(active) & result_balanced_criteria
                ),
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
                    "max_new_tokens": 224,
                    "failure_examples": 4,
                }
            )
            quota_groups.append(
                {
                    "name": "cumulative_replay",
                    "examples": replay_examples,
                    "filters": {"families": list(prior)},
                    "balance_families": True,
                    "balance_operations_within_families": True,
                    "balance_results_within_operations_for_families": sorted(
                        set(prior) & result_balanced_criteria
                    ),
                }
            )
            minimum_by_panel[retention_name] = 1.0
            minimum_valid_by_panel[retention_name] = 1.0
            panel_family[retention_name] = {criterion: 1.0 for criterion in prior}
        contract_phases.append(
            {
                "name": phase["name"],
                "minimum_epochs": minimum_epochs_per_phase,
                "maximum_epochs": maximum_epochs_per_phase,
                "advance_after_consecutive_passes": consecutive_passes,
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
                "minimum_generation_accuracy_by_operation": {
                    operation: 0.75
                    for criterion in active
                    for operation in criterion_operations.get(criterion, [])
                },
                "minimum_trace_exact_by_family": (
                    {criterion: 0.70 for criterion in active}
                    if trace_acceptance_metric == "exact_v1"
                    else {}
                ),
                "minimum_trace_semantic_by_family": (
                    {criterion: 0.70 for criterion in active}
                    if trace_acceptance_metric == "semantic_v1"
                    else {}
                ),
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
            "examples_per_epoch": examples_per_epoch,
            "phase_local_optimization": {
                "enabled": True,
                "reset_optimizer": True,
                "warmup_epochs": 3,
                "minimum_learning_rate": 3e-5,
            },
        },
        "phases": contract_phases,
    }


def build_v7_merged_contract(contract: dict) -> dict:
    """Merge V5 phases 1-3 and 4-5 without changing any dataset bytes."""

    merged = copy.deepcopy(contract)
    old = merged["phases"]
    if len(old) != 15:
        raise ValueError("V7 requires the exact 15-phase V5 contract")

    examples_per_epoch = int(merged["curriculum"]["examples_per_epoch"])
    active_examples = examples_per_epoch * 3 // 4
    replay_examples = examples_per_epoch - active_examples

    def active_families(index: int) -> list[str]:
        return list(old[index]["quota_groups"][0]["filters"]["families"])

    def make_phase(indices: tuple[int, ...], *, name: str, prior_indices: tuple[int, ...]) -> dict:
        active = [family for index in indices for family in active_families(index)]
        prior = [family for index in prior_indices for family in active_families(index)]
        result_balanced = sorted(
            {
                family
                for index in indices
                for family in old[index]["quota_groups"][0].get(
                    "balance_results_within_operations_for_families", []
                )
            }
        )
        phase = copy.deepcopy(old[indices[-1]])
        phase.update(
            {
                "name": name,
                "families": [*prior, *active],
                "quota_groups": [
                    {
                        "name": "active",
                        "examples": examples_per_epoch if not prior else active_examples,
                        "filters": {"families": active},
                        "balance_families": True,
                        "balance_operations_within_families": True,
                        "balance_results_within_operations_for_families": result_balanced,
                    }
                ],
                "primary_generation_panel": f"active_{indices[-1]:02d}",
                "minimum_generation_accuracy_by_family": {},
                "minimum_generation_accuracy_by_operation": {},
                "minimum_trace_exact_by_family": {},
                "minimum_trace_semantic_by_family": {},
                "minimum_generation_accuracy_by_panel": {
                    f"active_{index:02d}": 0.85 for index in indices
                },
                "minimum_valid_rate_by_panel": {
                    f"active_{index:02d}": 0.95 for index in indices
                },
                "minimum_generation_accuracy_by_panel_family": {
                    f"active_{index:02d}": old[index][
                        "minimum_generation_accuracy_by_family"
                    ]
                    for index in indices
                },
                "minimum_generation_accuracy_by_panel_operation": {
                    f"active_{index:02d}": old[index][
                        "minimum_generation_accuracy_by_operation"
                    ]
                    for index in indices
                },
                "minimum_trace_exact_by_panel_family": {
                    f"active_{index:02d}": old[index]["minimum_trace_exact_by_family"]
                    for index in indices
                },
                "minimum_trace_semantic_by_panel_family": {
                    f"active_{index:02d}": old[index][
                        "minimum_trace_semantic_by_family"
                    ]
                    for index in indices
                },
                "stop_on_pass": False,
            }
        )
        if prior:
            retention_index = indices[0]
            retention_name = f"retention_{retention_index:02d}"
            phase["quota_groups"].append(
                {
                    "name": "cumulative_replay",
                    "examples": replay_examples,
                    "filters": {"families": prior},
                    "balance_families": True,
                    "balance_operations_within_families": True,
                    "balance_results_within_operations_for_families": sorted(
                        {
                            family
                            for index in prior_indices
                            for family in old[index]["quota_groups"][0].get(
                                "balance_results_within_operations_for_families", []
                            )
                        }
                    ),
                }
            )
            phase["minimum_generation_accuracy_by_panel"][retention_name] = 1.0
            phase["minimum_valid_rate_by_panel"][retention_name] = 1.0
            phase["minimum_generation_accuracy_by_panel_family"][retention_name] = {
                family: 1.0 for family in prior
            }
        return phase

    merged["phases"] = [
        make_phase((0, 1, 2), name="stage_01_foundations_merged", prior_indices=()),
        make_phase((3, 4), name="stage_04_arithmetic_merged", prior_indices=(0, 1, 2)),
        *copy.deepcopy(old[5:]),
    ]
    merged["math_training"]["max_epochs"] = sum(
        int(phase["maximum_epochs"]) for phase in merged["phases"]
    )
    return merged


def build_v8_cumulative_contract(contract: dict, manifest: dict) -> dict:
    """Use V7's merged phases with every prior training row in later stages."""

    cumulative = build_v7_merged_contract(contract)
    criterion_counts = {
        str(key).removeprefix("train."): int(value)
        for key, value in manifest.get("audit", {}).get("criterion_counts", {}).items()
        if str(key).startswith("train.")
    }
    if not criterion_counts:
        raise ValueError("V8 cumulative allocation requires manifest train criterion counts")

    cumulative_families: list[str] = []
    for phase in cumulative["phases"]:
        current = list(phase["quota_groups"][0]["filters"]["families"])
        cumulative_families.extend(
            family for family in current if family not in cumulative_families
        )
        missing = sorted(set(cumulative_families) - set(criterion_counts))
        if missing:
            raise ValueError(
                "V8 manifest is missing train counts for: " + ", ".join(missing)
            )
        cumulative_examples = sum(
            criterion_counts[family] for family in cumulative_families
        )
        phase["families"] = list(cumulative_families)
        phase["examples_per_epoch"] = cumulative_examples
        phase["quota_groups"] = [
            {
                "name": "complete_cumulative_training_set",
                "examples": cumulative_examples,
                "filters": {"families": list(cumulative_families)},
                "balance_families": False,
            }
        ]
    return cumulative


def build_v9_cumulative_balanced_contract(contract: dict, manifest: dict) -> dict:
    """V9 uses the V5-compatible targeted dataset with full cumulative replay.

    Coverage is corrected in the sealed dataset only for the agreed original
    stage 5 multiplication and division criteria.  Do not duplicate other
    phase rows at sampling time.
    """

    return build_v8_cumulative_contract(contract, manifest)


def build_v10_multiplication_contract(contract: dict, manifest: dict) -> dict:
    """V10 uses cumulative replay with its sealed, versioned multiplication traces."""

    v10 = build_v8_cumulative_contract(contract, manifest)
    for panel in v10["math_training"]["generation_validation"]["panels"]:
        if panel["name"] == "active_04":
            # V10 multiplication emits two decompositions, four partial
            # products, and a final sum.  The default 224-token limit cuts a
            # valid trace before its answer tag.
            panel["max_new_tokens"] = 512
    return v10


def build_smoke_contract(contract: dict) -> dict:
    smoke_contract = copy.deepcopy(contract)
    smoke_phase = copy.deepcopy(smoke_contract["phases"][0])
    smoke_phase.update(
        {
            "minimum_epochs": 1,
            "maximum_epochs": 1,
            "advance_after_consecutive_passes": 1,
            "stop_on_pass": True,
            "minimum_generation_accuracy": 0.0,
            "minimum_valid_rate": 0.0,
            "minimum_generation_accuracy_by_family": {},
            "minimum_generation_accuracy_by_operation": {},
            "minimum_trace_exact_by_family": {},
            "minimum_trace_semantic_by_family": {},
            "minimum_generation_accuracy_by_panel": {},
            "minimum_valid_rate_by_panel": {},
            "minimum_generation_accuracy_by_panel_family": {},
        }
    )
    smoke_contract["phases"] = [smoke_phase]
    smoke_contract["math_training"]["max_epochs"] = 1
    smoke_contract["math_training"]["generation_validation"]["panels"] = [
        {
            **smoke_contract["math_training"]["generation_validation"]["panels"][0],
            "examples": 2,
            "max_new_tokens": 16,
        }
    ]
    return smoke_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, start, or resume the balanced local math master experiment"
    )
    parser.add_argument(
        "command", choices=("auto", "build", "audit", "smoke"), default="auto", nargs="?"
    )
    parser.add_argument("--config", default="config/math_master_experiment_local_v6.yaml")
    parser.add_argument("--dataset-config", default="config/math_master_experiment_v6.json")
    parser.add_argument("--data", default="C:/CFTN/.datasets/math_master_experiment_100k_v6")
    parser.add_argument(
        "--artifact", default="C:/CFTN/artifacts/math_master_experiment_100k_v6/run"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--contract-profile",
        choices=("v5", "v7_merged", "v8_cumulative", "v9_cumulative_balanced", "v10_multiplication"),
        default="v5",
    )
    parser.add_argument("--initial-checkpoint")
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
    curriculum_settings = config.get("data", {}).get("curriculum", {})
    contract = build_contract(
        manifest,
        minimum_epochs_per_phase=int(
            curriculum_settings.get("minimum_epochs_per_phase", 10)
        ),
        maximum_epochs_per_phase=int(
            curriculum_settings.get("maximum_epochs_per_phase", 60)
        ),
        consecutive_passes=int(
            curriculum_settings.get("advance_after_consecutive_passes", 2)
        ),
        examples_per_epoch=int(curriculum_settings.get("examples_per_epoch", 512)),
    )
    if args.contract_profile == "v7_merged":
        contract = build_v7_merged_contract(contract)
    elif args.contract_profile == "v8_cumulative":
        contract = build_v8_cumulative_contract(contract, manifest)
    elif args.contract_profile == "v9_cumulative_balanced":
        contract = build_v9_cumulative_balanced_contract(contract, manifest)
    elif args.contract_profile == "v10_multiplication":
        contract = build_v10_multiplication_contract(contract, manifest)
    if args.initial_checkpoint:
        contract["source_checkpoint_sha256"] = file_sha256(args.initial_checkpoint)
    max_batches = None
    if args.command == "smoke":
        contract = build_smoke_contract(contract)
        max_batches = 1
    config["math_training"]["max_epochs"] = contract["math_training"]["max_epochs"]
    artifact = Path(args.artifact + ("_smoke" if args.command == "smoke" else "")).resolve()
    resume = any(artifact.glob("checkpoint_epoch_*.pth")) or (artifact / "math.latest.pth").exists()
    result = train_math_tower(
        config,
        device_name=args.device,
        resume=resume,
        max_batches=max_batches,
        require_calibration=False,
        artifact_directory=artifact,
        working_directory=artifact,
        recovery_contract=contract,
        initial_checkpoint=None if resume else args.initial_checkpoint,
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
