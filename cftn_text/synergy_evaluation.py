from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump, gpu_status, load_checkpoint
from .config import config_sha256
from .data_generator import file_sha256
from .metrics import control_report, extract_answer, paired_bootstrap_interval
from .synergy_benchmark import (
    audit_synergy_benchmark,
    load_synergy_rows,
)
from .tokenizer import ByteMathTokenizer
from .training import build_cftn_model, load_data_contract, resolve_device


PROOF_CONDITIONS = (
    "correct",
    "shuffled",
    "gpt_to_math_shuffled",
    "math_to_gpt_shuffled",
    "gpt_to_math_disabled",
    "math_to_gpt_disabled",
    "both_disabled",
    "gpt_to_math_pair_swapped",
    "math_to_gpt_pair_swapped",
)


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def _cross_pair_permutation(size: int) -> list[int]:
    if size < 4 or size % 2:
        raise ValueError("cross-pair shuffling requires at least two complete pairs")
    return list(range(size - 2, size)) + list(range(0, size - 2))


def _pair_swap_permutation(size: int) -> list[int]:
    if size % 2:
        raise ValueError("pair swapping requires complete adjacent pairs")
    result: list[int] = []
    for index in range(0, size, 2):
        result.extend((index + 1, index))
    return result


def _controls(condition: str, size: int) -> dict[str, Any]:
    cross_pair = _cross_pair_permutation(size)
    pair_swap = _pair_swap_permutation(size)
    controls: dict[str, dict[str, Any]] = {
        "correct": {},
        "shuffled": {
            "gpt_to_math_permutation": cross_pair,
            "math_to_gpt_permutation": cross_pair,
        },
        "gpt_to_math_shuffled": {"gpt_to_math_permutation": cross_pair},
        "math_to_gpt_shuffled": {"math_to_gpt_permutation": cross_pair},
        "gpt_to_math_disabled": {"gpt_to_math_enabled": False},
        "math_to_gpt_disabled": {"math_to_gpt_enabled": False},
        "both_disabled": {
            "gpt_to_math_enabled": False,
            "math_to_gpt_enabled": False,
        },
        "gpt_to_math_pair_swapped": {"gpt_to_math_permutation": pair_swap},
        "math_to_gpt_pair_swapped": {"math_to_gpt_permutation": pair_swap},
    }
    return controls[condition]


def _correctness(
    rows: list[dict[str, Any]], *, condition: str, output: str
) -> list[bool]:
    key = f"{output}_generation"
    return [
        extract_answer(row["outputs"][condition][key]) == int(row["record"]["x"])
        for row in rows
    ]


def _predictions(
    rows: list[dict[str, Any]], *, condition: str, output: str
) -> list[int | None]:
    key = f"{output}_generation"
    return [extract_answer(row["outputs"][condition][key]) for row in rows]


def _accuracy(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _paired_report(
    candidate: list[bool],
    baseline: list[bool],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    interval = paired_bootstrap_interval(
        candidate, baseline, samples=samples, seed=seed
    )
    return {
        "candidate_accuracy": _accuracy(candidate),
        "baseline_accuracy": _accuracy(baseline),
        **interval,
    }


def _counterfactual_report(
    rows: list[dict[str, Any]], *, condition: str, output: str
) -> dict[str, Any]:
    predictions = _predictions(rows, condition=condition, output=output)
    pairs = len(rows) // 2
    both_correct = 0
    correct_delta = 0
    answer_changed = 0
    for index in range(0, len(rows), 2):
        base_row, changed_row = rows[index : index + 2]
        base_prediction, changed_prediction = predictions[index : index + 2]
        base_target = int(base_row["record"]["x"])
        changed_target = int(changed_row["record"]["x"])
        both_correct += int(
            base_prediction == base_target and changed_prediction == changed_target
        )
        if base_prediction is not None and changed_prediction is not None:
            answer_changed += int(base_prediction != changed_prediction)
            correct_delta += int(
                changed_prediction - base_prediction == changed_target - base_target
            )
    return {
        "pairs": pairs,
        "both_correct_rate": both_correct / pairs if pairs else 0.0,
        "answer_change_rate": answer_changed / pairs if pairs else 0.0,
        "correct_delta_rate": correct_delta / pairs if pairs else 0.0,
    }


def _donor_follow_rate(rows: list[dict[str, Any]]) -> float:
    predictions = _predictions(
        rows, condition="math_to_gpt_pair_swapped", output="gpt"
    )
    followed = 0
    for index in range(0, len(rows), 2):
        base_target = int(rows[index]["record"]["x"])
        changed_target = int(rows[index + 1]["record"]["x"])
        followed += int(predictions[index] == changed_target)
        followed += int(predictions[index + 1] == base_target)
    return followed / len(rows) if rows else 0.0


def _pair_swap_invariance(rows: list[dict[str, Any]]) -> float:
    matches = 0
    for row in rows:
        correct = row["outputs"]["correct"]
        swapped = row["outputs"]["gpt_to_math_pair_swapped"]
        matches += int(
            correct["math_generation"] == swapped["math_generation"]
            and correct["gpt_generation"] == swapped["gpt_generation"]
        )
    return matches / len(rows) if rows else 0.0


def _communication_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        communication = row["outputs"]["correct"].get("communication", {})
        for direction in ("gpt_to_math", "math_to_gpt"):
            gate = communication.get(f"{direction}_sender_gate", {})
            if "mean" in gate:
                values[f"{direction}_sender_gate_mean"].append(float(gate["mean"]))
            norm = communication.get(f"{direction}_message_norm")
            if norm is not None:
                values[f"{direction}_message_norm"].append(float(norm))
    return {
        key: {
            "mean": sum(items) / len(items),
            "minimum": min(items),
            "maximum": max(items),
        }
        for key, items in sorted(values.items())
        if items
    }


def _analyse_rows(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    joint = _correctness(rows, condition="correct", output="gpt")
    gpt_alone = _correctness(rows, condition="both_disabled", output="gpt")
    math_alone = _correctness(rows, condition="both_disabled", output="math")
    serial = _correctness(rows, condition="correct", output="math")
    strongest_name, strongest = (
        ("gpt_alone", gpt_alone)
        if _accuracy(gpt_alone) >= _accuracy(math_alone)
        else ("math_alone", math_alone)
    )
    shuffled = _correctness(rows, condition="shuffled", output="gpt")
    no_g2m_math = _correctness(
        rows, condition="gpt_to_math_disabled", output="math"
    )
    g2m_math = _correctness(rows, condition="correct", output="math")
    no_m2g_gpt = _correctness(
        rows, condition="math_to_gpt_disabled", output="gpt"
    )
    gpt_wrong_indices = [index for index, correct in enumerate(gpt_alone) if not correct]
    gpt_wrong_collaboration = None
    if gpt_wrong_indices:
        gpt_wrong_collaboration = _paired_report(
            [joint[index] for index in gpt_wrong_indices],
            [shuffled[index] for index in gpt_wrong_indices],
            samples=samples,
            seed=seed + 4,
        )
        gpt_wrong_collaboration["examples"] = len(gpt_wrong_indices)
    return {
        "examples": len(rows),
        "arm_accuracy": {
            "joint_cftn": _accuracy(joint),
            "gpt_alone": _accuracy(gpt_alone),
            "math_alone": _accuracy(math_alone),
            "serial_gpt_to_math_readout": _accuracy(serial),
            "both_bridges_shuffled": _accuracy(shuffled),
            "gpt_to_math_disabled_math_readout": _accuracy(no_g2m_math),
            "math_to_gpt_disabled_final_readout": _accuracy(no_m2g_gpt),
        },
        "strongest_individual_arm": strongest_name,
        "synergy_vs_strongest_individual": _paired_report(
            joint, strongest, samples=samples, seed=seed
        ),
        "correct_vs_both_shuffled": _paired_report(
            joint, shuffled, samples=samples, seed=seed + 1
        ),
        "joint_vs_serial_gpt_to_math_readout": _paired_report(
            joint, serial, samples=samples, seed=seed + 5
        ),
        "gpt_to_math_direct_contribution": _paired_report(
            g2m_math, no_g2m_math, samples=samples, seed=seed + 2
        ),
        "math_to_gpt_direct_contribution": _paired_report(
            joint, no_m2g_gpt, samples=samples, seed=seed + 3
        ),
        "collaboration_when_gpt_alone_wrong": gpt_wrong_collaboration,
        "counterfactual": {
            "joint_cftn": _counterfactual_report(
                rows, condition="correct", output="gpt"
            ),
            "math_serial_readout": _counterfactual_report(
                rows, condition="correct", output="math"
            ),
            "math_to_gpt_pair_swap_donor_follow_rate": _donor_follow_rate(rows),
            "identical_gpt_view_pair_swap_invariance": _pair_swap_invariance(rows),
        },
        "communication": _communication_summary(rows),
    }


def _pass_criteria(
    aggregate: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    thresholds = protocol["success_criteria"]
    require_ci = bool(thresholds.get("require_ci95_above_zero", True))

    def gain_pass(report: dict[str, Any], minimum: float) -> bool:
        return report["mean_difference"] >= minimum and (
            not require_ci or report["ci95_low"] > 0.0
        )

    gates = {
        "synergy_gain": gain_pass(
            aggregate["synergy_vs_strongest_individual"],
            float(thresholds["minimum_synergy_gain"]),
        ),
        "correct_vs_shuffled": gain_pass(
            aggregate["correct_vs_both_shuffled"],
            float(thresholds["minimum_correct_vs_shuffled_gap"]),
        ),
        "gpt_to_math_causal_gain": gain_pass(
            aggregate["gpt_to_math_direct_contribution"],
            float(thresholds["minimum_directional_gain"]),
        ),
        "math_to_gpt_causal_gain": gain_pass(
            aggregate["math_to_gpt_direct_contribution"],
            float(thresholds["minimum_directional_gain"]),
        ),
        "counterfactual_pair_accuracy": (
            aggregate["counterfactual"]["joint_cftn"]["both_correct_rate"]
            >= float(thresholds["minimum_counterfactual_pair_accuracy"])
        ),
        "joint_not_worse_than_serial": (
            aggregate["joint_vs_serial_gpt_to_math_readout"]["mean_difference"]
            >= -float(thresholds["maximum_joint_vs_serial_regression"])
        ),
        "message_swap_follows_donor": (
            aggregate["counterfactual"]["math_to_gpt_pair_swap_donor_follow_rate"]
            >= float(thresholds["minimum_message_swap_donor_follow"])
        ),
    }
    gates["pass"] = all(gates.values())
    return gates


def _markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    gates = report["causal_collaboration_gate"]
    lines = [
        "# CFTN-Text causal collaboration report",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "## Aggregate arm accuracy",
        "",
        "| Arm | Accuracy |",
        "|---|---:|",
    ]
    for name, value in aggregate["arm_accuracy"].items():
        lines.append(f"| {name} | {100.0 * value:.2f}% |")
    lines.extend(
        [
            "",
            "## Causal gates",
            "",
            "| Gate | Result |",
            "|---|---:|",
        ]
    )
    for name, value in gates.items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    synergy = aggregate["synergy_vs_strongest_individual"]
    lines.extend(
        [
            "",
            "## Synergy estimate",
            "",
            (
                f"CFTN minus the strongest individual tower: "
                f"{100.0 * synergy['mean_difference']:.2f} percentage points "
                f"(95% CI {100.0 * synergy['ci95_low']:.2f} to "
                f"{100.0 * synergy['ci95_high']:.2f})."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_synergy_checkpoint(
    config: dict[str, Any],
    protocol: dict[str, Any],
    checkpoint_path: str | Path,
    benchmark_manifest_path: str | Path,
    *,
    device_name: str = "cuda",
    maximum_pairs_per_split: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    data_root, source_manifest = load_data_contract(config)
    del data_root
    benchmark = audit_synergy_benchmark(benchmark_manifest_path)
    if benchmark["config_sha256"] != config_sha256(config):
        raise ValueError("synergy benchmark configuration differs")
    if benchmark["source_manifest_sha256"] != source_manifest["manifest_sha256"]:
        raise ValueError("synergy benchmark source manifest differs")
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_config_sha256=config_sha256(config),
        expected_manifest_sha256=source_manifest["manifest_sha256"],
        map_location=device,
    )
    if checkpoint["stage"] != "bidirectional":
        raise ValueError("synergy evaluation requires a bidirectional checkpoint")
    if checkpoint["extra"].get("view_mode", "shared") != "complementary":
        raise ValueError(
            "causal synergy evaluation requires complementary-view bridge training"
        )
    math_checkpoint = checkpoint["extra"].get("math_checkpoint")
    if not math_checkpoint:
        raise ValueError("bridge checkpoint does not identify its math checkpoint")
    if file_sha256(math_checkpoint) != checkpoint["extra"].get(
        "math_checkpoint_sha256"
    ):
        raise ValueError("frozen math checkpoint hash changed")
    model, gpt_tokenizer = build_cftn_model(
        config, math_checkpoint, source_manifest, device
    )
    model.set_trainable_stage("bidirectional")
    model.load_trainable_state_dict(checkpoint["model_state"], strict=True)
    gate_mode = checkpoint["extra"].get("gate_mode", "contextual")
    model.set_gate_mode(gate_mode)
    model.eval()
    math_tokenizer = ByteMathTokenizer()
    settings = protocol["evaluation"]
    maximum_pairs = int(
        maximum_pairs_per_split
        if maximum_pairs_per_split is not None
        else settings["maximum_pairs_per_split"]
    )
    batch_size = int(settings["generation_batch_size"])
    if batch_size < 4 or batch_size % 2:
        raise ValueError("synergy generation batch size must be an even value >= 4")
    artifact_root = Path(
        output_root
        or Path(config["project"]["artifact_root"])
        / f"synergy_evaluation_{gate_mode}"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "status.json"
    started_at = time.time()
    report: dict[str, Any] = {
        "format": "cftn_text_causal_synergy_evaluation_v1",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "math_checkpoint": str(Path(math_checkpoint).resolve()),
        "math_checkpoint_sha256": file_sha256(math_checkpoint),
        "config_sha256": config_sha256(config),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "benchmark_manifest": str(Path(benchmark_manifest_path).resolve()),
        "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "gate_mode": gate_mode,
        "training_view_mode": "complementary",
        "conditions": list(PROOF_CONDITIONS),
        "splits": {},
    }
    aggregate_rows: list[dict[str, Any]] = []
    for split_index, (split, metadata) in enumerate(benchmark["splits"].items()):
        records = load_synergy_rows(Path(benchmark_manifest_path).parent / metadata["path"])
        records = records[: min(len(records), maximum_pairs * 2)]
        if len(records) < 4 or len(records) % 2:
            raise ValueError(f"synergy split {split} does not contain complete pairs")
        generated_by_condition: dict[str, list[dict[str, Any]]] = {
            condition: [] for condition in PROOF_CONDITIONS
        }
        model.reset_execution_counts()
        for start, chunk in _chunks(records, batch_size):
            if len(chunk) < 4:
                # Merge a final one-pair remainder into the preceding batch by
                # requiring protocol sizes to divide into complete >=2-pair chunks.
                raise ValueError(
                    "synergy batch leaves fewer than two pairs; choose a compatible batch size"
                )
            problems = [record["problem"] for record in chunk]
            gpt_problems = [record["gpt_problem"] for record in chunk]
            math_problems = [record["math_problem"] for record in chunk]
            for condition_index, condition in enumerate(PROOF_CONDITIONS):
                generated_by_condition[condition].extend(
                    model.generate_problems(
                        problems,
                        math_tokenizer,
                        gpt_tokenizer,
                        max_math_new_tokens=int(config["evaluation"]["max_math_new_tokens"]),
                        max_gpt_new_tokens=int(config["evaluation"]["max_gpt_new_tokens"]),
                        gpt_problems=gpt_problems,
                        math_problems=math_problems,
                        **_controls(condition, len(chunk)),
                    )
                )
                atomic_json_dump(
                    {
                        "state": "running",
                        "split": split,
                        "split_index": split_index + 1,
                        "splits_total": len(benchmark["splits"]),
                        "condition": condition,
                        "condition_index": condition_index + 1,
                        "conditions_total": len(PROOF_CONDITIONS),
                        "completed_records": start + len(chunk),
                        "records_total": len(records),
                        "elapsed_seconds": time.time() - started_at,
                        "gpu": gpu_status(),
                    },
                    status_path,
                )
        rows_path = artifact_root / f"{split}_generations.jsonl"
        evaluation_rows: list[dict[str, Any]] = []
        with rows_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(records):
                row = {
                    "record": record,
                    "outputs": {
                        condition: generated_by_condition[condition][index]
                        for condition in PROOF_CONDITIONS
                    },
                }
                evaluation_rows.append(row)
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        split_analysis = _analyse_rows(
            evaluation_rows,
            samples=int(settings["bootstrap_samples"]),
            seed=int(protocol["seed"]) + split_index * 100,
        )
        split_analysis.update(
            {
                "metrics": control_report(generated_by_condition, records),
                "generation_rows": str(rows_path.resolve()),
                "execution_counts": model.execution_counts(),
            }
        )
        report["splits"][split] = split_analysis
        aggregate_rows.extend(evaluation_rows)
    aggregate = _analyse_rows(
        aggregate_rows,
        samples=int(settings["bootstrap_samples"]),
        seed=int(protocol["seed"]) + 10_000,
    )
    report["aggregate"] = aggregate
    report["causal_collaboration_gate"] = _pass_criteria(aggregate, protocol)
    report["gpu"] = gpu_status()
    report["elapsed_seconds"] = time.time() - started_at
    report_path = artifact_root / "report.json"
    atomic_json_dump(report, report_path)
    markdown_path = artifact_root / "report.md"
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    atomic_json_dump(
        {
            "state": "completed",
            "elapsed_seconds": time.time() - started_at,
            "report": str(report_path.resolve()),
            "causal_collaboration_gate": report["causal_collaboration_gate"],
            "gpu": gpu_status(),
        },
        status_path,
    )
    return report
