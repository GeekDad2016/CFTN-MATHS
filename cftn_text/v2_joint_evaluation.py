from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump, gpu_status, load_checkpoint
from .complementary import apply_view_mode
from .config import config_sha256
from .data_generator import file_sha256
from .metrics import paired_bootstrap_interval
from .tokenizer import ByteMathTokenizer
from .training import build_cftn_model, load_data_contract, resolve_device, split_dataset
from .v2_metrics import score_v2_generations
from .wandb_support import initialize_wandb


CONDITIONS: dict[str, dict[str, Any]] = {
    "joint_contextual": {},
    "both_closed": {
        "gpt_to_math_enabled": False,
        "math_to_gpt_enabled": False,
    },
    "gpt_to_math_closed": {"gpt_to_math_enabled": False},
    "math_to_gpt_closed": {"math_to_gpt_enabled": False},
    "gpt_to_math_shuffled": {"shuffle_gpt_to_math": True},
    "math_to_gpt_shuffled": {"shuffle_math_to_gpt": True},
    "both_shuffled": {
        "shuffle_gpt_to_math": True,
        "shuffle_math_to_gpt": True,
    },
    "joint_fixed_open": {"gate_mode": "fixed_open"},
}


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def _condition_metrics(
    outputs: dict[str, list[dict[str, Any]]], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, list[bool]]]]:
    metrics: dict[str, Any] = {}
    correctness: dict[str, dict[str, list[bool]]] = {}
    for condition, rows in outputs.items():
        gpt, gpt_correct = score_v2_generations(
            [row["gpt_generation"] for row in rows], records
        )
        math, math_correct = score_v2_generations(
            [row["math_generation"] for row in rows], records
        )
        metrics[condition] = {"gpt": gpt, "math": math}
        correctness[condition] = {"gpt": gpt_correct, "math": math_correct}
    return metrics, correctness


def evaluate_v2_collaboration(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    math_checkpoint_path: str | Path | None = None,
    device_name: str = "cuda",
    splits: list[str] | None = None,
    maximum_examples: int | None = None,
    output_root: str | Path | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(config)
    if manifest.get("format") != "cftn_text_broad_math_v2":
        raise ValueError("V2 collaboration evaluation requires a V2 manifest")
    math_checkpoint = Path(
        math_checkpoint_path
        or Path(config["project"]["artifact_root"]) / "math" / "math.best.pth"
    )
    model, gpt_tokenizer = build_cftn_model(
        config, math_checkpoint, manifest, device
    )
    model.set_trainable_stage("bidirectional")
    bridge_checkpoint = load_checkpoint(
        checkpoint_path,
        expected_stage="bidirectional",
        expected_config_sha256=config_sha256(config),
        expected_manifest_sha256=manifest["manifest_sha256"],
        map_location=device,
    )
    model.load_trainable_state_dict(bridge_checkpoint["model_state"], strict=True)
    model.eval()
    model.set_gate_mode("contextual")
    math_tokenizer = ByteMathTokenizer()
    settings = config["evaluation"]
    maximum = int(
        maximum_examples
        if maximum_examples is not None
        else settings.get(
            "collaboration_maximum_examples",
            settings["maximum_generation_examples"],
        )
    )
    split_names = splits or list(settings["collaboration_splits"])
    artifact_root = Path(
        output_root
        or Path(config["project"]["artifact_root"]) / "evaluation_collaboration_v2"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "status.json"
    started_at = time.time()
    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact_root,
        stage="evaluation_collaboration_v2",
        config={
            "project": config["project"]["name"],
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "config_sha256": config_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    report: dict[str, Any] = {
        "format": "cftn_text_collaboration_evaluation_v2",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "math_checkpoint": str(math_checkpoint.resolve()),
        "math_checkpoint_sha256": file_sha256(math_checkpoint),
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "conditions": list(CONDITIONS),
        "splits": {},
    }
    try:
        for split_index, split in enumerate(split_names):
            raw_records = split_dataset(data_root, manifest, split).records
            raw_records = [
                record
                for record in raw_records
                if record.get("gpt_problem") and record.get("math_problem")
            ][:maximum]
            records = apply_view_mode(
                raw_records,
                view_mode="complementary",
                seed=int(config["project"]["seed"]),
            )
            if not records:
                continue
            outputs = {condition: [] for condition in CONDITIONS}
            for start, chunk in _chunks(records, int(settings["batch_size"])):
                shared = [record["problem"] for record in chunk]
                gpt_views = [record["gpt_problem"] for record in chunk]
                math_views = [record["math_problem"] for record in chunk]
                for condition, options in CONDITIONS.items():
                    gate_mode = str(options.get("gate_mode", "contextual"))
                    model.set_gate_mode(gate_mode)
                    generation_options = {
                        key: value
                        for key, value in options.items()
                        if key != "gate_mode"
                    }
                    rows = model.generate_problems(
                        shared,
                        math_tokenizer,
                        gpt_tokenizer,
                        max_math_new_tokens=int(settings["max_math_new_tokens"]),
                        max_gpt_new_tokens=int(settings["max_gpt_new_tokens"]),
                        gpt_problems=gpt_views,
                        math_problems=math_views,
                        generic_answer=True,
                        **generation_options,
                    )
                    outputs[condition].extend(rows)
                model.set_gate_mode("contextual")
                atomic_json_dump(
                    {
                        "state": "running",
                        "phase": "collaboration_ablations",
                        "split": split,
                        "split_index": split_index + 1,
                        "splits_total": len(split_names),
                        "completed": start + len(chunk),
                        "total": len(records),
                        "conditions_per_example": len(CONDITIONS),
                        "elapsed_seconds": time.time() - started_at,
                        "gpu": gpu_status(),
                    },
                    status_path,
                )
            metrics, correctness = _condition_metrics(outputs, records)
            joint = correctness["joint_contextual"]["gpt"]
            gpt_alone = correctness["both_closed"]["gpt"]
            math_alone = correctness["both_closed"]["math"]
            strongest_name, strongest = (
                ("gpt_alone", gpt_alone)
                if sum(gpt_alone) >= sum(math_alone)
                else ("math_alone", math_alone)
            )
            bootstrap_samples = int(settings["bootstrap_samples"])
            seed = int(config["project"]["seed"]) + split_index * 100
            causal = {
                "strongest_individual": strongest_name,
                "synergy_vs_strongest_individual": paired_bootstrap_interval(
                    joint, strongest, samples=bootstrap_samples, seed=seed
                ),
                "both_closed_effect": paired_bootstrap_interval(
                    joint, gpt_alone, samples=bootstrap_samples, seed=seed + 1
                ),
                "both_shuffled_effect": paired_bootstrap_interval(
                    joint,
                    correctness["both_shuffled"]["gpt"],
                    samples=bootstrap_samples,
                    seed=seed + 2,
                ),
                "math_to_gpt_effect": paired_bootstrap_interval(
                    joint,
                    correctness["math_to_gpt_closed"]["gpt"],
                    samples=bootstrap_samples,
                    seed=seed + 3,
                ),
                "gpt_to_math_effect": paired_bootstrap_interval(
                    correctness["joint_contextual"]["math"],
                    correctness["gpt_to_math_closed"]["math"],
                    samples=bootstrap_samples,
                    seed=seed + 4,
                ),
                "contextual_vs_fixed_open": paired_bootstrap_interval(
                    joint,
                    correctness["joint_fixed_open"]["gpt"],
                    samples=bootstrap_samples,
                    seed=seed + 5,
                ),
            }
            rows_path = artifact_root / f"{split}_ablations.jsonl"
            with rows_path.open("w", encoding="utf-8") as handle:
                for index, record in enumerate(records):
                    handle.write(
                        json.dumps(
                            {
                                "record": record,
                                "outputs": {
                                    condition: outputs[condition][index]
                                    for condition in CONDITIONS
                                },
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            split_report = {
                "examples": len(records),
                "metrics": metrics,
                "causal": causal,
                "generation_rows": str(rows_path.resolve()),
            }
            report["splits"][split] = split_report
            tracker.log(
                {f"collaboration/{split}": split_report},
                global_step=split_index + 1,
                event="split_completed",
            )

        synergy_intervals = [
            split["causal"]["synergy_vs_strongest_individual"]
            for split in report["splits"].values()
        ]
        shuffle_intervals = [
            split["causal"]["both_shuffled_effect"]
            for split in report["splits"].values()
        ]
        report["collaboration_gate"] = {
            "all_splits_synergy_at_least_10_points": bool(synergy_intervals)
            and all(item["mean_difference"] >= 0.10 for item in synergy_intervals),
            "all_synergy_ci95_above_zero": bool(synergy_intervals)
            and all(item["ci95_low"] > 0.0 for item in synergy_intervals),
            "shuffling_removes_at_least_5_points": bool(shuffle_intervals)
            and all(item["mean_difference"] >= 0.05 for item in shuffle_intervals),
        }
        report["collaboration_gate"]["pass"] = all(
            report["collaboration_gate"].values()
        )
        atomic_json_dump(report, artifact_root / "report.json")
        atomic_json_dump(
            {
                "state": "completed",
                "elapsed_seconds": time.time() - started_at,
                "report": str((artifact_root / "report.json").resolve()),
                "gpu": gpu_status(),
            },
            status_path,
        )
        tracker.update_summary(
            {"run/state": "completed", "collaboration_gate": report["collaboration_gate"]}
        )
        tracker.finish()
        return report
    except BaseException as exc:
        atomic_json_dump(
            {
                "state": "error",
                "error": repr(exc),
                "elapsed_seconds": time.time() - started_at,
                "gpu": gpu_status(),
            },
            status_path,
        )
        tracker.update_summary({"run/state": "error", "run/error": repr(exc)})
        tracker.finish(exit_code=1)
        raise
