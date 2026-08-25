from __future__ import annotations

import json
from pathlib import Path

import pytest

from cftn_text.data_generator import file_sha256
from cftn_text.v2_checkpoint_selection import (
    _candidate_provenance,
    candidate_score,
    select_v2_math_checkpoint,
)
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


def test_v2_checkpoint_selection_can_reuse_one_explicit_candidate(
    tmp_path, monkeypatch
):
    import cftn_text.v2_checkpoint_selection as selection_module

    artifact_root = tmp_path / "artifacts"
    math_root = artifact_root / "math"
    math_root.mkdir(parents=True)
    checkpoint_path = math_root / "math.best.pth"
    checkpoint_path.write_bytes(b"epoch-45-checkpoint")
    digest = file_sha256(checkpoint_path)
    candidate_root = (
        artifact_root
        / "math_checkpoint_selection"
        / f"epoch_0045_{digest[:12]}"
    )
    candidate_root.mkdir(parents=True)
    _write(
        candidate_root / "report.json",
        {
            "format": "cftn_text_math_evaluation_v2",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": digest,
            "config_sha256": "config-sha",
            "manifest_sha256": "manifest-sha",
            "splits": {
                "validation": {
                    "examples": 512,
                    "accuracy": 0.25,
                    "valid_rate": 0.75,
                }
            },
        },
    )
    monkeypatch.setattr(
        selection_module,
        "load_data_contract",
        lambda config: (tmp_path / "data", {"manifest_sha256": "manifest-sha"}),
    )
    monkeypatch.setattr(
        selection_module, "config_sha256", lambda config: "config-sha"
    )
    monkeypatch.setattr(
        selection_module,
        "load_checkpoint",
        lambda *args, **kwargs: {
            "epoch": 45,
            "extra": {
                "metrics": {
                    "validation": {
                        "teacher_forced_sequence_accuracy": 0.9,
                        "loss": 0.2,
                    }
                }
            },
        },
    )

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("completed candidate should have been reused")

    monkeypatch.setattr(
        selection_module, "evaluate_v2_math_checkpoint", unexpected_evaluation
    )
    report = select_v2_math_checkpoint(
        {
            "project": {"artifact_root": str(artifact_root)},
            "checkpoint_selection": {
                "generation_examples": 512,
                "split": "validation",
            },
        },
        device_name="cpu",
        candidate_paths=[checkpoint_path],
        working_root=tmp_path / "scratch",
    )
    assert report["candidate_scope"] == "explicit"
    assert len(report["candidates"]) == 1
    assert report["candidates"][0]["epoch"] == 45
    assert report["candidates"][0]["evaluation_reused"] is True
    assert report["selected"]["source_path"] == str(checkpoint_path.resolve())
    assert Path(report["selected"]["path"]).read_bytes() == checkpoint_path.read_bytes()


def test_external_math_recovery_candidate_requires_complete_acceptance_attestation(
    tmp_path,
):
    artifact_root = tmp_path / "artifacts"
    math_root = artifact_root / "math"
    recovery_root = artifact_root / "math_shared_trace_recovery"
    math_root.mkdir(parents=True)
    recovery_root.mkdir(parents=True)
    source = math_root / "math.best.pth"
    source.write_bytes(b"sealed-source")
    source_sha = file_sha256(source)
    candidate = recovery_root / "math.best.pth"
    candidate.write_bytes(b"accepted-recovery")
    candidate_sha = file_sha256(candidate)
    _write(
        recovery_root / "recovery_contract.json",
        {
            "format": "cftn_text_v2_math_shared_trace_recovery_v1",
            "require_acceptance_for_best": True,
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": source_sha,
            "observed_source_checkpoint_sha256": source_sha,
        },
    )
    summary = {
        "state": "completed",
        "best_checkpoint": str(candidate),
        "best_checkpoint_sha256": candidate_sha,
        "final_metrics": {
            "input_view": "shared_problem_v1",
            "target_mode": "full_trace_v1",
            "checkpoint_eligible": True,
            "checkpoint_promoted": True,
            "curriculum_acceptance": {
                "pass": True,
                "generation_accuracy": 0.953125,
                "minimum_generation_accuracy": 0.95,
                "valid_rate": 1.0,
                "minimum_valid_rate": 0.99,
            },
        },
    }
    _write(recovery_root / "summary.json", summary)

    provenance = _candidate_provenance(
        candidate,
        artifact_root=artifact_root,
        math_root=math_root,
        explicit=True,
    )

    assert provenance["kind"] == "accepted_math_recovery"
    assert provenance["generation_accuracy"] == 0.953125
    summary["final_metrics"]["curriculum_acceptance"]["pass"] = False
    _write(recovery_root / "summary.json", summary)
    with pytest.raises(ValueError, match="acceptance_pass"):
        _candidate_provenance(
            candidate,
            artifact_root=artifact_root,
            math_root=math_root,
            explicit=True,
        )
