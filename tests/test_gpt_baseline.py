from __future__ import annotations

import json
from pathlib import Path

import pytest

import cftn_text.gpt_baseline as gpt_baseline_module
from cftn_text.checkpoint import atomic_json_dump
from cftn_text.config import config_sha256
from cftn_text.data_generator import prepare_manifests
from cftn_text.gpt_baseline import (
    few_shot_prompt,
    first_generated_integer,
    plausible_candidates,
    verify_calibration_gate,
    zero_shot_prompt,
)


def test_baseline_prompts_and_lenient_integer_parser():
    problem = "Solve 7*x + (4) = 53."
    assert zero_shot_prompt(problem).endswith("Answer:")
    prompt = few_shot_prompt(problem, 3)
    assert prompt.count("Answer:<answer>") == 3
    assert prompt.endswith("Answer:")
    assert first_generated_integer(" <answer>-17</answer>") == -17
    assert first_generated_integer(" -17, therefore") == -17
    assert first_generated_integer("Solve -17*x = 34") is None
    assert first_generated_integer("no numeric answer") is None


def test_plausible_candidates_include_target_without_fixed_position(tiny_config):
    record = {
        "record_id": "abcdef0123456789",
        "x": 7,
        "a": 3,
        "b": 2,
        "c": 23,
    }
    candidates = plausible_candidates(record, 8)
    assert len(candidates) == len(set(candidates)) == 8
    assert 7 in candidates
    assert candidates[0] != 7


def test_calibration_gate_requires_matching_passing_report(tiny_config):
    manifest = prepare_manifests(tiny_config)
    report_path = (
        Path(tiny_config["project"]["artifact_root"])
        / "gpt_calibration"
        / "report.json"
    )
    with pytest.raises(FileNotFoundError):
        verify_calibration_gate(tiny_config, manifest)
    report = {
        "format": "cftn_text_frozen_gpt_calibration_v1",
        "evaluator_sha256": gpt_baseline_module.file_sha256(
            Path(gpt_baseline_module.__file__)
        ),
        "config_sha256": config_sha256(tiny_config),
        "manifest_sha256": manifest["manifest_sha256"],
        "calibration_split_sha256": manifest["splits"]["calibration"]["sha256"],
        "decision": {"proceed_to_math_training": True},
    }
    atomic_json_dump(report, report_path)
    assert verify_calibration_gate(tiny_config, manifest) == report
    report["decision"]["proceed_to_math_training"] = False
    atomic_json_dump(report, report_path)
    with pytest.raises(RuntimeError, match="already solves too much"):
        verify_calibration_gate(tiny_config, manifest)
