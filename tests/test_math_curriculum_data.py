from __future__ import annotations

import json
from pathlib import Path

from cftn_text.math_curriculum_data import (
    FORMAT,
    MASTER_PHASES,
    PHASES,
    audit_dataset,
    iter_phase_training_records,
    iter_phase_validation_records,
    iter_records,
    prepare_dataset,
    solve_math_ir,
)
from cftn_text.computation_supervision import ComputationCollator
from cftn_text.tokenizer import ByteMathTokenizer
from tools.run_math_master_experiment import build_contract


def _config() -> dict:
    return {
        "format": FORMAT,
        "seed": 7,
        "objects_per_criterion": {"train": 3, "validation": 2, "test": 1},
        "replay_policy": {"active_fraction": 0.75, "prior_fraction": 0.25},
    }


def _master_config() -> dict:
    return {
        **_config(),
        "curriculum_profile": "master_experiment_v1",
        "objects_per_criterion": {"train": 3, "validation": 2, "test": 1},
    }


def test_math_tower_view_is_canonical_and_language_free() -> None:
    record = next(iter_records(_config(), "train"))
    assert json.loads(record["problem"]) == record["math_ir"]
    assert record["math_problem"] == record["problem"]
    assert record["gpt_problem"] == record["natural_language_prompt"]
    assert "What " not in record["math_problem"]
    assert record["dispatcher_target"]["route"] == "math"


def test_derivation_and_answer_are_executable() -> None:
    for record in iter_records(_config(), "validation"):
        answer, derivation = solve_math_ir(record["math_ir"])
        assert record["answer"] == answer
        assert record["derivation"] == derivation


def test_computation_objective_consumes_explicit_spans() -> None:
    record = next(iter_records(_config(), "train"))
    collator = ComputationCollator(ByteMathTokenizer(), 512)
    batch = collator([record])
    assert batch["math_roles"].eq(1).any()


def test_future_phase_is_not_declared_as_a_prerequisite() -> None:
    records = list(iter_records(_config(), "train"))
    for record in records:
        phase_index = record["curriculum_phase_index"]
        allowed = {
            criterion
            for phase in PHASES[:phase_index]
            for criterion in phase["criteria"]
        }
        assert set(record["prerequisite_ids"]) == allowed


def test_phase_views_have_no_future_data_and_use_cumulative_replay() -> None:
    config = _config()
    for phase_index, phase in enumerate(PHASES):
        rows = list(iter_phase_training_records(config, phase_index))
        allowed = {
            criterion
            for permitted in PHASES[: phase_index + 1]
            for criterion in permitted["criteria"]
        }
        assert {row["criterion_id"] for row in rows} <= allowed
        if phase_index:
            active = set(phase["criteria"])
            active_fraction = sum(row["criterion_id"] in active for row in rows) / len(rows)
            assert 0.70 <= active_fraction <= 0.80


def test_prepare_and_streaming_sqlite_audit(tmp_path: Path) -> None:
    manifest = prepare_dataset(_config(), tmp_path)
    assert manifest["audit"]["status"] == "passed"
    assert audit_dataset(tmp_path, tmp_path / "scratch")["status"] == "passed"
    assert not list((tmp_path / "scratch").glob("*.sqlite3"))


def test_sealed_output_refuses_overwrite(tmp_path: Path) -> None:
    prepare_dataset(_config(), tmp_path)
    try:
        prepare_dataset(_config(), tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("sealed dataset was overwritten")


def test_master_profile_covers_every_declared_education_level() -> None:
    assert {phase["level"] for phase in MASTER_PHASES} == {
        "KS1",
        "KS2",
        "secondary",
        "GCSE",
        "A-level",
        "undergraduate",
        "graduate",
        "research-preparation",
    }
    records = list(iter_records(_master_config(), "test"))
    assert {record["educational_level"] for record in records} == {
        phase["level"] for phase in MASTER_PHASES
    }


def test_master_retention_panel_has_one_probe_per_prior_criterion() -> None:
    config = _master_config()
    final_index = len(MASTER_PHASES) - 1
    rows = list(iter_phase_validation_records(config, final_index, "retention"))
    expected = {
        criterion
        for phase in MASTER_PHASES[:final_index]
        for criterion in phase["criteria"]
    }
    assert {row["criterion_id"] for row in rows} == expected
    assert len(rows) == len(expected)


def test_local_runner_contract_is_bounded_and_phase_gated() -> None:
    splits = {}
    for index, phase in enumerate(MASTER_PHASES):
        splits[f"phase_{index:02d}_active"] = {"records": 8 * len(phase["criteria"])}
        if index:
            splits[f"phase_{index:02d}_retention"] = {
                "records": sum(len(prior["criteria"]) for prior in MASTER_PHASES[:index])
            }
    contract = build_contract({"phases": list(MASTER_PHASES), "splits": splits})
    assert len(contract["phases"]) == 15
    assert sum(phase["maximum_epochs"] for phase in contract["phases"]) == 120
    assert contract["phases"][1]["quota_groups"][0]["examples"] == 72
    assert contract["phases"][1]["quota_groups"][1]["examples"] == 24
    assert contract["phases"][-1]["stop_on_pass"] is True
