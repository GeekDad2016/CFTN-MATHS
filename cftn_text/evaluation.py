from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import atomic_json_dump, gpu_status, load_checkpoint
from .complementary import apply_view_mode
from .config import config_sha256
from .data_generator import file_sha256
from .metrics import control_report, extract_answer, paired_bootstrap_interval
from .tokenizer import ByteMathTokenizer
from .training import build_cftn_model, load_data_contract, resolve_device, split_dataset
from .wandb_support import initialize_wandb


CONDITIONS = {
    "correct": {},
    "shuffled": {
        "shuffle_gpt_to_math": True,
        "shuffle_math_to_gpt": True,
    },
    "gpt_to_math_shuffled": {"shuffle_gpt_to_math": True},
    "math_to_gpt_shuffled": {"shuffle_math_to_gpt": True},
    "gpt_to_math_disabled": {"gpt_to_math_enabled": False},
    "math_to_gpt_disabled": {"math_to_gpt_enabled": False},
    "both_disabled": {
        "gpt_to_math_enabled": False,
        "math_to_gpt_enabled": False,
    },
}


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _condition_correctness(
    rows: list[dict[str, Any]], records: list[dict[str, Any]], output: str
) -> list[bool]:
    key = f"{output}_generation"
    return [
        extract_answer(row[key]) == int(record["x"])
        for row, record in zip(rows, records)
    ]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _provisional_metrics(
    correct_counts: dict[str, dict[str, int]], completed: int
) -> dict[str, dict[str, float | int]]:
    denominator = max(1, int(completed))
    return {
        condition: {
            "examples": int(completed),
            "gpt_exact_accuracy": counts["gpt"] / denominator,
            "math_exact_accuracy": counts["math"] / denominator,
        }
        for condition, counts in correct_counts.items()
    }


def _collaboration_acceptance(
    report: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    configured = config["evaluation"].get("collaboration_acceptance")
    if not configured:
        test_metrics = report["splits"].get("test", {}).get("metrics", {})
        heldout_metrics = report["splits"].get("heldout_language", {}).get(
            "metrics", {}
        )
        extrapolation_metrics = report["splits"].get("extrapolation", {}).get(
            "metrics", {}
        )
        legacy = {
            "id_accuracy_at_least_99_5": (
                test_metrics.get("correct", {})
                .get("gpt", {})
                .get("exact_accuracy", 0.0)
                >= 0.995
            ),
            "heldout_language_at_least_98": (
                heldout_metrics.get("correct", {})
                .get("gpt", {})
                .get("exact_accuracy", 0.0)
                >= 0.98
            ),
            "extrapolation_at_least_95": (
                extrapolation_metrics.get("correct", {})
                .get("gpt", {})
                .get("exact_accuracy", 0.0)
                >= 0.95
            ),
        }
        legacy["collaboration_gate_pass"] = all(legacy.values())
        return legacy

    details: dict[str, Any] = {}
    for split, criteria in configured.items():
        generation = (
            report["splits"]
            .get(split, {})
            .get("metrics", {})
            .get("correct", {})
            .get("gpt", {})
        )
        split_details: dict[str, Any] = {}
        for metric, threshold_value in criteria.items():
            if metric == "trace_exact_rate":
                raise ValueError(
                    "collaboration acceptance does not expose trace_exact_rate"
                )
            threshold = float(threshold_value)
            observed = float(generation.get(metric, 0.0))
            split_details[metric] = {
                "observed": observed,
                "threshold": threshold,
                "pass": observed >= threshold,
            }
        details[split] = split_details
    passed = all(
        criterion["pass"]
        for split_details in details.values()
        for criterion in split_details.values()
    )
    return {
        "name": "full_cftn_collaboration_acceptance",
        "criteria": details,
        "pass": passed,
        "collaboration_gate_pass": passed,
    }


def evaluate_model_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device_name: str = "cuda",
    splits: list[str] | None = None,
    maximum_examples: int | None = None,
    output_root: str | Path | None = None,
    view_mode: str | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(config)
    bridge_checkpoint = load_checkpoint(
        checkpoint_path,
        expected_config_sha256=config_sha256(config),
        expected_manifest_sha256=manifest["manifest_sha256"],
        map_location=device,
    )
    if bridge_checkpoint["stage"] not in {"m2g", "bidirectional"}:
        raise ValueError("evaluation requires a bridge checkpoint")
    math_checkpoint = bridge_checkpoint["extra"].get("math_checkpoint")
    if not math_checkpoint:
        raise ValueError("bridge checkpoint does not identify its frozen math tower")
    if file_sha256(math_checkpoint) != bridge_checkpoint["extra"].get(
        "math_checkpoint_sha256"
    ):
        raise ValueError("frozen math checkpoint hash changed")
    model, gpt_tokenizer = build_cftn_model(
        config, math_checkpoint, manifest, device
    )
    model.set_trainable_stage(bridge_checkpoint["stage"])
    model.load_trainable_state_dict(bridge_checkpoint["model_state"], strict=True)
    gate_mode = bridge_checkpoint["extra"].get("gate_mode", "contextual")
    training_view_mode = bridge_checkpoint["extra"].get("view_mode", "shared")
    evaluation_view_mode = view_mode or training_view_mode
    if evaluation_view_mode not in {"shared", "complementary"}:
        raise ValueError("evaluation view mode must be shared or complementary")
    model.set_gate_mode(gate_mode)
    model.eval()
    math_tokenizer = ByteMathTokenizer()
    settings = config["evaluation"]
    split_names = splits or list(
        settings.get(
            "splits",
            ["test", "heldout_language", "extrapolation", "compositional"],
        )
    )
    maximum = int(
        maximum_examples
        if maximum_examples is not None
        else settings["maximum_generation_examples"]
    )
    artifact_root = Path(
        output_root
        or Path(config["project"]["artifact_root"])
        / (
            f"evaluation_{bridge_checkpoint['stage']}_{gate_mode}"
            + (f"_{evaluation_view_mode}" if evaluation_view_mode != "shared" else "")
        )
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "status.json"
    progress_path = artifact_root / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    started_at = time.time()
    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact_root,
        stage="evaluation_shared_cftn",
        config={
            "project": config["project"]["name"],
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "config_sha256": config_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"],
            "evaluation_view_mode": evaluation_view_mode,
        },
    )
    progress_every_batches = max(
        1, int(settings.get("progress_every_batches", 10))
    )
    global_batch = 0
    report: dict[str, Any] = {
        "format": "cftn_text_evaluation_v1",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "math_checkpoint": str(Path(math_checkpoint).resolve()),
        "math_checkpoint_sha256": file_sha256(math_checkpoint),
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "stage": bridge_checkpoint["stage"],
        "gate_mode": gate_mode,
        "training_view_mode": training_view_mode,
        "evaluation_view_mode": evaluation_view_mode,
        "conditions": list(CONDITIONS),
        "splits": {},
    }
    atomic_json_dump(
        {
            "format": "cftn_text_evaluation_status_v1",
            "state": "running",
            "phase": "initializing",
            "pid": os.getpid(),
            "splits_total": len(split_names),
            "elapsed_seconds": time.time() - started_at,
            "gpu": gpu_status(),
        },
        status_path,
    )
    try:
        for split_index, split in enumerate(split_names):
            dataset = split_dataset(data_root, manifest, split)
            records = apply_view_mode(
                dataset.records[:maximum],
                view_mode=evaluation_view_mode,
                seed=int(config["project"]["seed"]),
            )
            outputs: dict[str, list[dict[str, Any]]] = {
                condition: [] for condition in CONDITIONS
            }
            correct_counts = {
                condition: {"gpt": 0, "math": 0} for condition in CONDITIONS
            }
            model.reset_execution_counts()
            batch_size = int(settings["batch_size"])
            batches_total = max(1, (len(records) + batch_size - 1) // batch_size)
            for batch_index, chunk in enumerate(_chunks(records, batch_size), start=1):
                problems = [record["problem"] for record in chunk]
                gpt_problems = [
                    record.get("gpt_problem", record["problem"]) for record in chunk
                ]
                math_problems = [
                    record.get("math_problem", record["problem"]) for record in chunk
                ]
                for condition, controls in CONDITIONS.items():
                    generated = model.generate_problems(
                        problems,
                        math_tokenizer,
                        gpt_tokenizer,
                        max_math_new_tokens=int(settings["max_math_new_tokens"]),
                        max_gpt_new_tokens=int(settings["max_gpt_new_tokens"]),
                        gpt_problems=gpt_problems,
                        math_problems=math_problems,
                        **controls,
                    )
                    outputs[condition].extend(generated)
                    correct_counts[condition]["gpt"] += sum(
                        _condition_correctness(generated, chunk, "gpt")
                    )
                    correct_counts[condition]["math"] += sum(
                        _condition_correctness(generated, chunk, "math")
                    )
                global_batch += 1
                completed = min(batch_index * batch_size, len(records))
                split_progress = completed / max(1, len(records))
                overall_progress = (
                    split_index + split_progress
                ) / max(1, len(split_names))
                elapsed = time.time() - started_at
                eta = (
                    elapsed * (1.0 - overall_progress) / overall_progress
                    if overall_progress > 0.0
                    else None
                )
                provisional = _provisional_metrics(correct_counts, completed)
                status = {
                    "format": "cftn_text_evaluation_status_v1",
                    "state": "running",
                    "phase": "generation",
                    "pid": os.getpid(),
                    "split": split,
                    "split_index": split_index + 1,
                    "splits_total": len(split_names),
                    "batch": batch_index,
                    "batches_total": batches_total,
                    "examples_completed": completed,
                    "examples_total": len(records),
                    "conditions_per_example": len(CONDITIONS),
                    "split_progress": split_progress,
                    "overall_progress": overall_progress,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "provisional": provisional,
                    "gpu": gpu_status(),
                }
                atomic_json_dump(status, status_path)
                if (
                    batch_index == 1
                    or batch_index % progress_every_batches == 0
                    or batch_index == batches_total
                ):
                    _append_jsonl(progress_path, status)
                    tracker.log(
                        {"evaluation": status},
                        global_step=global_batch,
                        event="evaluation_progress",
                    )
            metrics = control_report(outputs, records)
            correct = _condition_correctness(outputs["correct"], records, "gpt")
            shuffled = _condition_correctness(outputs["shuffled"], records, "gpt")
            no_g2m = _condition_correctness(
                outputs["gpt_to_math_disabled"], records, "gpt"
            )
            no_m2g = _condition_correctness(
                outputs["math_to_gpt_disabled"], records, "gpt"
            )
            correct_math = _condition_correctness(outputs["correct"], records, "math")
            no_g2m_math = _condition_correctness(
                outputs["gpt_to_math_disabled"], records, "math"
            )
            shuffled_g2m_math = _condition_correctness(
                outputs["gpt_to_math_shuffled"], records, "math"
            )
            shuffled_m2g_gpt = _condition_correctness(
                outputs["math_to_gpt_shuffled"], records, "gpt"
            )
            gpt_alone = _condition_correctness(outputs["both_disabled"], records, "gpt")
            math_alone = _condition_correctness(outputs["both_disabled"], records, "math")
            serial_gpt_to_math = _condition_correctness(outputs["correct"], records, "math")
            strongest_name, strongest_individual = (
                ("gpt_alone", gpt_alone)
                if sum(gpt_alone) >= sum(math_alone)
                else ("math_alone", math_alone)
            )
            split_report = {
            "examples": len(records),
            "metrics": metrics,
            "correct_vs_shuffled": paired_bootstrap_interval(
                correct,
                shuffled,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]),
            ),
            "gpt_to_math_contribution": paired_bootstrap_interval(
                correct_math,
                no_g2m_math,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]) + 1,
            ),
            "math_to_gpt_contribution": paired_bootstrap_interval(
                correct,
                no_m2g,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]) + 2,
            ),
            "gpt_to_math_shuffle_contribution": paired_bootstrap_interval(
                correct_math,
                shuffled_g2m_math,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]) + 3,
            ),
            "math_to_gpt_shuffle_contribution": paired_bootstrap_interval(
                correct,
                shuffled_m2g_gpt,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]) + 4,
            ),
            "gpt_to_math_end_to_end_contribution": paired_bootstrap_interval(
                correct,
                no_g2m,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]) + 5,
            ),
            "arm_accuracy": {
                "joint_cftn": sum(correct) / len(correct),
                "gpt_alone": sum(gpt_alone) / len(gpt_alone),
                "math_alone": sum(math_alone) / len(math_alone),
                "serial_gpt_to_math_readout": (
                    sum(serial_gpt_to_math) / len(serial_gpt_to_math)
                ),
            },
            "strongest_individual_arm": strongest_name,
            "synergy_vs_strongest_individual": paired_bootstrap_interval(
                correct,
                strongest_individual,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]) + 6,
            ),
            "joint_vs_serial_gpt_to_math_readout": paired_bootstrap_interval(
                correct,
                serial_gpt_to_math,
                samples=int(settings["bootstrap_samples"]),
                seed=int(config["project"]["seed"]) + 7,
            ),
            "execution_counts": model.execution_counts(),
        }
            rows_path = artifact_root / f"{split}_generations.jsonl"
            split_report["generation_rows"] = str(rows_path.resolve())
            report["splits"][split] = split_report
            with rows_path.open("w", encoding="utf-8") as handle:
                for index, record in enumerate(records):
                    row = {
                        "record": record,
                        "outputs": {
                            condition: outputs[condition][index]
                            for condition in CONDITIONS
                        },
                    }
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
            tracker.log(
                {f"evaluation/splits/{split}": split_report},
                global_step=global_batch,
                event="split_completed",
            )
        report["preregistered_gates"] = _collaboration_acceptance(report, config)
        atomic_json_dump(report, artifact_root / "report.json")
        atomic_json_dump(
            {
                "format": "cftn_text_evaluation_status_v1",
                "state": "completed",
                "phase": "completed",
                "pid": os.getpid(),
                "elapsed_seconds": time.time() - started_at,
                "report": str((artifact_root / "report.json").resolve()),
                "preregistered_gates": report["preregistered_gates"],
                "gpu": gpu_status(),
            },
            status_path,
        )
        tracker.update_summary(
            {
                "run/state": "completed",
                "evaluation/preregistered_gates": report["preregistered_gates"],
            }
        )
        tracker.finish()
        return report
    except BaseException as exc:
        atomic_json_dump(
            {
                "format": "cftn_text_evaluation_status_v1",
                "state": "error",
                "phase": "error",
                "pid": os.getpid(),
                "error": repr(exc),
                "elapsed_seconds": time.time() - started_at,
                "gpu": gpu_status(),
            },
            status_path,
        )
        tracker.update_summary({"run/state": "error", "run/error": repr(exc)})
        tracker.finish(exit_code=1)
        raise
