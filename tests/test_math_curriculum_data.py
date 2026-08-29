from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cftn_text.math_curriculum_data import (
    FORMAT,
    MASTER_PHASES,
    PHASES,
    _criterion_split_counts,
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


def test_small_mastered_domains_replay_deterministically_with_replacement() -> None:
    config = _config()
    config["replay_policy"] = {"active_fraction": 0.1, "prior_fraction": 0.9}
    rows = list(iter_phase_training_records(config, 1))
    prior = [row for row in rows if row["criterion_id"] in PHASES[0]["criteria"]]
    assert len(prior) > len({row["record_id"] for row in prior})
    assert prior == list(iter_phase_training_records(config, 1))[len(rows) - len(prior) :]


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
    criterion_operations = {
        "1NPV-1": ["missing_count_sequence", "predecessor", "successor"],
        "1NPV-2": ["compare"],
        "1AS-1": ["compose", "compose_three"],
    }
    contract = build_contract(
        {
            "phases": list(MASTER_PHASES),
            "splits": splits,
            "criterion_operations": criterion_operations,
        }
    )
    assert len(contract["phases"]) == 15
    assert sum(phase["maximum_epochs"] for phase in contract["phases"]) == 900
    assert all(phase["minimum_epochs"] == 10 for phase in contract["phases"])
    assert all(phase["maximum_epochs"] == 60 for phase in contract["phases"])
    assert contract["phases"][1]["quota_groups"][0]["examples"] == 384
    assert contract["phases"][1]["quota_groups"][1]["examples"] == 128
    assert all(
        group["balance_operations_within_families"]
        for phase in contract["phases"]
        for group in phase["quota_groups"]
    )
    assert contract["phases"][0]["minimum_generation_accuracy_by_operation"]
    assert contract["curriculum"]["phase_local_optimization"] == {
        "enabled": True,
        "reset_optimizer": True,
        "warmup_epochs": 3,
        "minimum_learning_rate": 3e-5,
    }
    assert contract["math_training"]["generation_validation"]["panels"][0]["max_new_tokens"] == 224
    assert contract["phases"][-1]["stop_on_pass"] is True


def test_phase_one_generation_budget_covers_longest_procedural_target() -> None:
    config = _master_config()
    rows = list(iter_phase_validation_records(config, 0, "active"))
    assert max(len(row["target_trace"].encode("utf-8")) for row in rows) < 224


def test_phase_one_validation_is_operation_stratified() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "config" / "math_master_experiment_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rows = list(iter_phase_validation_records(config, 0, "active"))
    assert Counter(row["operation"] for row in rows) == {
        "compare": 12,
        "compose": 6,
        "compose_three": 6,
        "missing_count_sequence": 4,
        "predecessor": 4,
        "successor": 4,
    }


def test_100k_experiment_allocates_exact_distinct_training_total() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "config" / "math_master_experiment_v1.json").read_text(
            encoding="utf-8"
        )
    )
    counts = _criterion_split_counts(config)
    assert config["language_variants_per_object"] == 1
    assert sum(value["train"] for value in counts.values()) == 100_000
    assert sum(value["validation"] for value in counts.values()) == 12 * len(counts)
    assert sum(value["test"] for value in counts.values()) == 12 * len(counts)
    first_phase = MASTER_PHASES[0]["criteria"]
    assert sum(counts[criterion]["train"] for criterion in first_phase) == 3927


def test_100k_experiment_has_no_object_or_prompt_collisions() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "config" / "math_master_experiment_v1.json").read_text(
            encoding="utf-8"
        )
    )
    objects: dict[str, str] = {}
    prompts: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        count = 0
        for record in iter_records(config, split):
            object_id = record["math_object_id"]
            prompt = record["natural_language_prompt"]
            assert object_id not in objects, (object_id, objects[object_id], split)
            assert prompt not in prompts, (prompt, prompts[prompt], split)
            objects[object_id] = split
            prompts[prompt] = split
            count += 1
        counts[split] = count
    assert counts == {"train": 100_000, "validation": 516, "test": 516}


def test_local_runner_phase_budget_is_configurable_and_validated() -> None:
    splits = {
        "phase_00_active": {"records": 8 * len(MASTER_PHASES[0]["criteria"])}
    }
    manifest = {"phases": [MASTER_PHASES[0]], "splits": splits}
    contract = build_contract(
        manifest,
        minimum_epochs_per_phase=4,
        maximum_epochs_per_phase=20,
        consecutive_passes=3,
        examples_per_epoch=100,
    )
    phase = contract["phases"][0]
    assert phase["minimum_epochs"] == 4
    assert phase["maximum_epochs"] == 20
    assert phase["advance_after_consecutive_passes"] == 3
    assert phase["quota_groups"][0]["examples"] == 100
    with pytest.raises(ValueError, match="at least the minimum"):
        build_contract(
            manifest,
            minimum_epochs_per_phase=5,
            maximum_epochs_per_phase=4,
        )
