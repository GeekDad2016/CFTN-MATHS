from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump
from .metrics import extract_answer, paired_bootstrap_interval


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _paired_correctness(
    candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> tuple[list[bool], list[bool]]:
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("candidate and baseline row counts differ")
    candidate_correct: list[bool] = []
    baseline_correct: list[bool] = []
    for candidate, baseline in zip(candidate_rows, baseline_rows):
        candidate_record = candidate["record"]
        baseline_record = baseline["record"]
        if candidate_record["record_id"] != baseline_record["record_id"]:
            raise ValueError("candidate and baseline evaluation rows are misaligned")
        target = int(candidate_record["x"])
        candidate_correct.append(
            extract_answer(candidate["outputs"]["correct"]["gpt_generation"])
            == target
        )
        baseline_correct.append(
            extract_answer(baseline["outputs"]["correct"]["gpt_generation"])
            == target
        )
    return candidate_correct, baseline_correct


def compare_evaluation_reports(
    candidate_report_path: str | Path,
    baseline_report_path: str | Path,
    *,
    output_path: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 719,
) -> dict[str, Any]:
    with Path(candidate_report_path).open("r", encoding="utf-8") as handle:
        candidate = json.load(handle)
    with Path(baseline_report_path).open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    for key in ("config_sha256", "manifest_sha256"):
        if candidate.get(key) != baseline.get(key):
            raise ValueError(f"candidate and baseline {key} differ")
    if candidate.get("gate_mode") != "contextual":
        raise ValueError("candidate report is not the contextual-gate CFTN arm")
    if baseline.get("gate_mode") != "fixed_open":
        raise ValueError("baseline report is not the fixed-open arm")
    common_splits = sorted(set(candidate["splits"]).intersection(baseline["splits"]))
    if not common_splits:
        raise ValueError("candidate and baseline reports have no common splits")
    comparisons: dict[str, Any] = {}
    aggregate_candidate: list[bool] = []
    aggregate_baseline: list[bool] = []
    for index, split in enumerate(common_splits):
        candidate_rows = _load_rows(candidate["splits"][split]["generation_rows"])
        baseline_rows = _load_rows(baseline["splits"][split]["generation_rows"])
        paired = _paired_correctness(candidate_rows, baseline_rows)
        interval = paired_bootstrap_interval(
            *paired, samples=bootstrap_samples, seed=seed + index
        )
        interval["candidate_accuracy"] = sum(paired[0]) / len(paired[0])
        interval["baseline_accuracy"] = sum(paired[1]) / len(paired[1])
        comparisons[split] = interval
        if split in {"heldout_language", "extrapolation", "compositional"}:
            aggregate_candidate.extend(paired[0])
            aggregate_baseline.extend(paired[1])
    if not aggregate_candidate:
        aggregate_candidate, aggregate_baseline = _paired_correctness(
            _load_rows(candidate["splits"][common_splits[0]]["generation_rows"]),
            _load_rows(baseline["splits"][common_splits[0]]["generation_rows"]),
        )
    aggregate = paired_bootstrap_interval(
        aggregate_candidate,
        aggregate_baseline,
        samples=bootstrap_samples,
        seed=seed + 100,
    )
    aggregate["candidate_accuracy"] = sum(aggregate_candidate) / len(
        aggregate_candidate
    )
    aggregate["baseline_accuracy"] = sum(aggregate_baseline) / len(
        aggregate_baseline
    )
    architecture_pass = (
        aggregate["mean_difference"] >= 0.02 and aggregate["ci95_low"] > 0.0
    )
    report = {
        "format": "cftn_text_matched_comparison_v1",
        "candidate_report": str(Path(candidate_report_path).resolve()),
        "baseline_report": str(Path(baseline_report_path).resolve()),
        "comparisons": comparisons,
        "ood_aggregate": aggregate,
        "required_absolute_improvement": 0.02,
        "requires_ci95_above_zero": True,
        "architecture_claim_pass": architecture_pass,
    }
    if output_path is not None:
        atomic_json_dump(report, output_path)
    return report


def compare_synergy_evaluation_reports(
    contextual_report_path: str | Path,
    fixed_open_report_path: str | Path,
    *,
    output_path: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 719,
    minimum_improvement: float = 0.02,
) -> dict[str, Any]:
    with Path(contextual_report_path).open("r", encoding="utf-8") as handle:
        contextual = json.load(handle)
    with Path(fixed_open_report_path).open("r", encoding="utf-8") as handle:
        fixed_open = json.load(handle)
    required = {
        "format": "cftn_text_causal_synergy_evaluation_v1",
        "gate_mode": "contextual",
        "training_view_mode": "complementary",
    }
    for key, expected in required.items():
        if contextual.get(key) != expected:
            raise ValueError(f"contextual synergy report has invalid {key}")
    if fixed_open.get("format") != required["format"]:
        raise ValueError("fixed-open synergy report format differs")
    if fixed_open.get("gate_mode") != "fixed_open":
        raise ValueError("baseline synergy report is not fixed-open")
    if fixed_open.get("training_view_mode") != "complementary":
        raise ValueError("fixed-open synergy report did not use complementary views")
    for key in (
        "config_sha256",
        "source_manifest_sha256",
        "benchmark_manifest_sha256",
    ):
        if contextual.get(key) != fixed_open.get(key):
            raise ValueError(f"synergy comparison {key} differs")
    common_splits = sorted(
        set(contextual["splits"]).intersection(fixed_open["splits"])
    )
    if not common_splits:
        raise ValueError("synergy reports have no common splits")
    comparisons: dict[str, Any] = {}
    aggregate_contextual: list[bool] = []
    aggregate_fixed: list[bool] = []
    for index, split in enumerate(common_splits):
        contextual_rows = _load_rows(
            contextual["splits"][split]["generation_rows"]
        )
        fixed_rows = _load_rows(fixed_open["splits"][split]["generation_rows"])
        paired = _paired_correctness(contextual_rows, fixed_rows)
        interval = paired_bootstrap_interval(
            *paired, samples=bootstrap_samples, seed=seed + index
        )
        interval["contextual_accuracy"] = sum(paired[0]) / len(paired[0])
        interval["fixed_open_accuracy"] = sum(paired[1]) / len(paired[1])
        comparisons[split] = interval
        aggregate_contextual.extend(paired[0])
        aggregate_fixed.extend(paired[1])
    aggregate = paired_bootstrap_interval(
        aggregate_contextual,
        aggregate_fixed,
        samples=bootstrap_samples,
        seed=seed + 100,
    )
    aggregate["contextual_accuracy"] = sum(aggregate_contextual) / len(
        aggregate_contextual
    )
    aggregate["fixed_open_accuracy"] = sum(aggregate_fixed) / len(
        aggregate_fixed
    )
    report = {
        "format": "cftn_text_synergy_architecture_comparison_v1",
        "contextual_report": str(Path(contextual_report_path).resolve()),
        "fixed_open_report": str(Path(fixed_open_report_path).resolve()),
        "comparisons": comparisons,
        "aggregate": aggregate,
        "minimum_improvement": float(minimum_improvement),
        "architecture_claim_pass": (
            aggregate["mean_difference"] >= float(minimum_improvement)
            and aggregate["ci95_low"] > 0.0
        ),
    }
    if output_path is not None:
        atomic_json_dump(report, output_path)
    return report
