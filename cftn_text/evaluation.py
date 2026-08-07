from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import atomic_json_dump, load_checkpoint
from .complementary import apply_view_mode
from .config import config_sha256
from .data_generator import file_sha256
from .metrics import control_report, extract_answer, paired_bootstrap_interval
from .tokenizer import ByteMathTokenizer
from .training import build_cftn_model, load_data_contract, resolve_device, split_dataset


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


def evaluate_model_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device_name: str = "cuda",
    splits: list[str] | None = None,
    maximum_examples: int | None = None,
    output_root: str | Path | None = None,
    view_mode: str | None = None,
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
    split_names = splits or [
        "test",
        "heldout_language",
        "extrapolation",
        "compositional",
    ]
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
    for split in split_names:
        dataset = split_dataset(data_root, manifest, split)
        records = apply_view_mode(
            dataset.records[:maximum],
            view_mode=evaluation_view_mode,
            seed=int(config["project"]["seed"]),
        )
        outputs: dict[str, list[dict[str, Any]]] = {
            condition: [] for condition in CONDITIONS
        }
        model.reset_execution_counts()
        for chunk in _chunks(records, int(settings["batch_size"])):
            problems = [record["problem"] for record in chunk]
            gpt_problems = [record.get("gpt_problem", record["problem"]) for record in chunk]
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
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    test_metrics = report["splits"].get("test", {}).get("metrics", {})
    heldout_metrics = report["splits"].get("heldout_language", {}).get("metrics", {})
    extrapolation_metrics = report["splits"].get("extrapolation", {}).get("metrics", {})
    report["preregistered_gates"] = {
        "id_accuracy_at_least_99_5": (
            test_metrics.get("correct", {}).get("gpt", {}).get("exact_accuracy", 0.0)
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
    report["preregistered_gates"]["collaboration_gate_pass"] = all(
        report["preregistered_gates"].values()
    )
    atomic_json_dump(report, artifact_root / "report.json")
    return report
