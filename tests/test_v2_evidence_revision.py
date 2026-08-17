from __future__ import annotations

import json
from pathlib import Path

import pytest

from cftn_text.data_generator import file_sha256
from cftn_text.v2_checkpoint_selection import candidate_score
from cftn_text.v2_prerequisites import (
    V1_3_REQUIRED_GATES,
    audit_v2_mechanism_prerequisites,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _passing_v1_3_gates() -> dict[str, bool]:
    return {**{name: True for name in V1_3_REQUIRED_GATES}, "pass": True}


def test_v2_prerequisite_audit_verifies_chained_passes(tmp_path):
    v1_2_path = tmp_path / "v1_2.json"
    _write(
        v1_2_path,
        {
            "format": "cftn_text_v1_2_revision_report_v1",
            "revision_sha256": "v12",
            "final_gates": {"pass": True},
        },
    )
    v1_3_path = tmp_path / "v1_3.json"
    _write(
        v1_3_path,
        {
            "format": "cftn_text_v1_3_revision_report_v1",
            "revision_sha256": "v13",
            "prerequisite": {"v1_2_report_sha256": file_sha256(v1_2_path)},
            "final_gates": _passing_v1_3_gates(),
        },
    )
    config = {
        "_meta": {"path": str(tmp_path / "config" / "v2.yaml")},
        "project": {"artifact_root": str(tmp_path / "artifacts")},
        "prerequisites": {
            "v1_2_report": str(v1_2_path),
            "v1_3_report": str(v1_3_path),
        },
    }
    report = audit_v2_mechanism_prerequisites(config)
    assert report["pass"] is True
    assert report["v1_2"]["sha256"] == file_sha256(v1_2_path)


def test_v2_prerequisite_audit_rejects_failed_v1_3(tmp_path):
    v1_2_path = tmp_path / "v1_2.json"
    _write(
        v1_2_path,
        {
            "format": "cftn_text_v1_2_revision_report_v1",
            "final_gates": {"pass": True},
        },
    )
    v1_3_path = tmp_path / "v1_3.json"
    _write(
        v1_3_path,
        {
            "format": "cftn_text_v1_3_revision_report_v1",
            "final_gates": {"pass": False},
        },
    )
    config = {
        "_meta": {"path": str(tmp_path / "config" / "v2.yaml")},
        "project": {"artifact_root": str(tmp_path / "artifacts")},
        "prerequisites": {
            "v1_2_report": str(v1_2_path),
            "v1_3_report": str(v1_3_path),
        },
    }
    with pytest.raises(RuntimeError, match="V1.3"):
        audit_v2_mechanism_prerequisites(config)


def test_v2_prerequisite_audit_rejects_pass_without_concrete_hard_wake_gates(
    tmp_path,
):
    v1_2_path = tmp_path / "v1_2.json"
    _write(
        v1_2_path,
        {
            "format": "cftn_text_v1_2_revision_report_v1",
            "final_gates": {"pass": True},
        },
    )
    gates = _passing_v1_3_gates()
    gates.pop("exact_required_set")
    v1_3_path = tmp_path / "v1_3.json"
    _write(
        v1_3_path,
        {
            "format": "cftn_text_v1_3_revision_report_v1",
            "prerequisite": {"v1_2_report_sha256": file_sha256(v1_2_path)},
            "final_gates": gates,
        },
    )
    config = {
        "_meta": {"path": str(tmp_path / "config" / "v2.yaml")},
        "project": {"artifact_root": str(tmp_path / "artifacts")},
        "prerequisites": {
            "v1_2_report": str(v1_2_path),
            "v1_3_report": str(v1_3_path),
        },
    }
    with pytest.raises(ValueError, match="exact_required_set"):
        audit_v2_mechanism_prerequisites(config)


def test_v2_checkpoint_ranking_prioritizes_generation_over_teacher_forcing():
    stronger_generation = {
        "generation_accuracy": 0.60,
        "valid_answer_rate": 0.90,
        "teacher_forced_sequence_accuracy": 0.50,
        "validation_loss": 1.0,
        "epoch": 3,
    }
    stronger_teacher_forcing = {
        "generation_accuracy": 0.59,
        "valid_answer_rate": 1.0,
        "teacher_forced_sequence_accuracy": 0.99,
        "validation_loss": 0.1,
        "epoch": 9,
    }
    assert candidate_score(stronger_generation) > candidate_score(
        stronger_teacher_forcing
    )
