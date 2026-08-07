from __future__ import annotations

import json

from cftn_text.metrics import paired_bootstrap_interval
from cftn_text.research import (
    compare_evaluation_reports,
    compare_synergy_evaluation_reports,
)


def test_paired_bootstrap_detects_consistent_improvement():
    interval = paired_bootstrap_interval(
        [True] * 100,
        [False] * 100,
        samples=500,
        seed=5,
    )
    assert interval["mean_difference"] == 1.0
    assert interval["ci95_low"] == 1.0


def write_rows(path, correct_count):
    with path.open("w", encoding="utf-8") as handle:
        for index in range(10):
            answer = index if index < correct_count else index + 1
            row = {
                "record": {"record_id": f"row-{index}", "x": index},
                "outputs": {
                    "correct": {
                        "gpt_generation": f"<answer>{answer}</answer>",
                        "math_generation": "",
                    }
                },
            }
            handle.write(json.dumps(row) + "\n")


def test_matched_report_requires_contextual_advantage(tmp_path):
    candidate_rows = tmp_path / "candidate.jsonl"
    baseline_rows = tmp_path / "baseline.jsonl"
    write_rows(candidate_rows, 10)
    write_rows(baseline_rows, 0)
    candidate = {
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "gate_mode": "contextual",
        "splits": {"extrapolation": {"generation_rows": str(candidate_rows)}},
    }
    baseline = {
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "gate_mode": "fixed_open",
        "splits": {"extrapolation": {"generation_rows": str(baseline_rows)}},
    }
    candidate_path = tmp_path / "candidate_report.json"
    baseline_path = tmp_path / "baseline_report.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    report = compare_evaluation_reports(
        candidate_path, baseline_path, bootstrap_samples=500
    )
    assert report["architecture_claim_pass"]
    assert report["ood_aggregate"]["mean_difference"] == 1.0


def test_synergy_comparison_requires_contextual_advantage(tmp_path):
    candidate_rows = tmp_path / "synergy_candidate.jsonl"
    baseline_rows = tmp_path / "synergy_baseline.jsonl"
    write_rows(candidate_rows, 10)
    write_rows(baseline_rows, 0)
    common = {
        "format": "cftn_text_causal_synergy_evaluation_v1",
        "training_view_mode": "complementary",
        "config_sha256": "config",
        "source_manifest_sha256": "source",
        "benchmark_manifest_sha256": "benchmark",
    }
    candidate = {
        **common,
        "gate_mode": "contextual",
        "splits": {"test": {"generation_rows": str(candidate_rows)}},
    }
    baseline = {
        **common,
        "gate_mode": "fixed_open",
        "splits": {"test": {"generation_rows": str(baseline_rows)}},
    }
    candidate_path = tmp_path / "synergy_candidate.json"
    baseline_path = tmp_path / "synergy_baseline.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    report = compare_synergy_evaluation_reports(
        candidate_path, baseline_path, bootstrap_samples=500
    )
    assert report["architecture_claim_pass"]
    assert report["aggregate"]["mean_difference"] == 1.0
