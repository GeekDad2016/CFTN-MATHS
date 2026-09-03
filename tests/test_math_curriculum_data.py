from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cftn_text.math_curriculum_data import (
    COMPACT_PROCEDURE_SCHEMA,
    FORMAT,
    MASTER_PHASES,
    PHASES,
    V6_DATASET_RECIPE,
    _candidate_irs,
    _criterion_split_counts,
    _iter_phase_training_records_from_train_file,
    _semantic_object_id,
    audit_dataset,
    iter_phase_training_records,
    iter_phase_validation_records,
    iter_records,
    prepare_dataset,
    solve_math_ir,
    trace_semantically_matches,
)
from cftn_text.computation_supervision import ComputationCollator
from cftn_text.tokenizer import ByteMathTokenizer
from tools.run_math_master_experiment import (
    build_contract,
    build_smoke_contract,
    build_v7_merged_contract,
    build_v8_cumulative_contract,
    build_v9_cumulative_balanced_contract,
)


def _config() -> dict:
    return {
        "format": FORMAT,
        "seed": 7,
        "objects_per_criterion": {"train": 3, "validation": 2, "test": 1},
        "replay_policy": {"active_fraction": 0.75, "prior_fraction": 0.25},
    }


def test_v11_stage5_remediation_appends_only_the_three_failed_criteria():
    root = Path(__file__).parents[1]
    original = json.loads(
        (root / "config" / "math_master_experiment_v11.json").read_text(
            encoding="utf-8"
        )
    )
    remediation = json.loads(
        (
            root / "config" / "math_master_experiment_v11_stage5_remediation.json"
        ).read_text(encoding="utf-8")
    )
    changed = {
        criterion: remediation["criterion_train_targets"][criterion]
        for criterion, count in original["criterion_train_targets"].items()
        if remediation["criterion_train_targets"][criterion] != count
    }
    assert changed == {
        "2MD-2": 5000,
        "KS2-LONG-MULTIPLY": 7500,
        "KS2-EXACT-DIVIDE": 7500,
    }
    assert remediation["dataset_recipe"] == original["dataset_recipe"]
    assert remediation["generator_version"] == original["generator_version"]
    assert sum(remediation["criterion_train_targets"].values()) == remediation[
        "total_train_records"
    ]


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


def test_streamed_phase_view_matches_reference_sampler(tmp_path: Path) -> None:
    config = _config()
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in iter_records(config, "train")),
        encoding="utf-8",
    )
    for phase_index in range(len(PHASES)):
        reference = list(iter_phase_training_records(config, phase_index))
        streamed = list(
            _iter_phase_training_records_from_train_file(
                config, phase_index, train_path
            )
        )
        assert [row["record_id"] for row in streamed] == [
            row["record_id"] for row in reference
        ]


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
        "1AS-1": ["add_2", "add_3"],
    }
    contract = build_contract(
        {
            "phases": list(MASTER_PHASES),
            "splits": splits,
            "criterion_operations": criterion_operations,
            "result_balanced_criteria": ["1AS-1"],
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
    assert contract["phases"][0]["quota_groups"][0][
        "balance_results_within_operations_for_families"
    ] == ["1AS-1"]
    assert contract["curriculum"]["phase_local_optimization"] == {
        "enabled": True,
        "reset_optimizer": True,
        "warmup_epochs": 3,
        "minimum_learning_rate": 3e-5,
    }
    assert contract["math_training"]["generation_validation"]["panels"][0]["max_new_tokens"] == 224
    assert contract["phases"][-1]["stop_on_pass"] is True


def test_v7_reuses_v5_panels_and_merges_only_runtime_phases() -> None:
    splits = {}
    for index, phase in enumerate(MASTER_PHASES):
        splits[f"phase_{index:02d}_active"] = {"records": 8 * len(phase["criteria"])}
        if index:
            splits[f"phase_{index:02d}_retention"] = {
                "records": sum(len(prior["criteria"]) for prior in MASTER_PHASES[:index])
            }
    manifest = {
        "phases": list(MASTER_PHASES),
        "splits": splits,
        "criterion_operations": {},
        "audit": {
            "criterion_counts": {
                f"train.{criterion}": index + 1
                for index, criterion in enumerate(
                    criterion
                    for phase in MASTER_PHASES
                    for criterion in phase["criteria"]
                )
            }
        },
    }
    base = build_contract(manifest)
    merged = build_v7_merged_contract(base)

    assert len(merged["phases"]) == 12
    assert merged["phases"][0]["quota_groups"][0]["filters"]["families"] == [
        criterion for phase in MASTER_PHASES[:3] for criterion in phase["criteria"]
    ]
    assert set(merged["phases"][0]["minimum_generation_accuracy_by_panel"]) == {
        "active_00", "active_01", "active_02"
    }
    assert merged["phases"][1]["quota_groups"][1]["filters"]["families"] == [
        criterion for phase in MASTER_PHASES[:3] for criterion in phase["criteria"]
    ]
    assert merged["phases"][2:] == base["phases"][5:]
    assert merged["math_training"]["generation_validation"]["panels"] == base[
        "math_training"
    ]["generation_validation"]["panels"]

    cumulative = build_v8_cumulative_contract(base, manifest)
    assert cumulative["phases"][0]["quota_groups"][0]["filters"]["families"] == [
        criterion for phase in MASTER_PHASES[:3] for criterion in phase["criteria"]
    ]
    assert cumulative["phases"][1]["quota_groups"][0]["filters"]["families"] == [
        criterion for phase in MASTER_PHASES[:5] for criterion in phase["criteria"]
    ]
    assert cumulative["phases"][-1]["quota_groups"][0]["filters"]["families"] == [
        criterion for phase in MASTER_PHASES for criterion in phase["criteria"]
    ]
    assert all(
        phase["quota_groups"][0]["name"] == "complete_cumulative_training_set"
        and phase["quota_groups"][0]["balance_families"] is False
        and phase["examples_per_epoch"] == phase["quota_groups"][0]["examples"]
        for phase in cumulative["phases"]
    )

    balanced = build_v9_cumulative_balanced_contract(base, manifest)
    assert balanced == cumulative


def test_smoke_contract_disables_all_scientific_acceptance_gates() -> None:
    splits = {}
    for index, phase in enumerate(MASTER_PHASES):
        splits[f"phase_{index:02d}_active"] = {"records": 8 * len(phase["criteria"])}
        if index:
            splits[f"phase_{index:02d}_retention"] = {
                "records": sum(len(prior["criteria"]) for prior in MASTER_PHASES[:index])
            }
    contract = build_contract(
        {
            "phases": list(MASTER_PHASES),
            "splits": splits,
            "criterion_operations": {"1NPV-1": ["successor"]},
        }
    )

    smoke = build_smoke_contract(contract)

    assert len(smoke["phases"]) == 1
    assert smoke["math_training"]["max_epochs"] == 1
    assert smoke["phases"][0]["minimum_generation_accuracy_by_operation"] == {}
    assert smoke["phases"][0]["minimum_generation_accuracy_by_family"] == {}
    assert smoke["phases"][0]["minimum_generation_accuracy"] == 0.0


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
        "add_2": 6,
        "add_3": 6,
        "missing_count_sequence": 4,
        "predecessor": 4,
        "successor": 4,
    }


def test_v4_phase_one_is_canonical_procedural_and_result_stratified() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "config" / "math_master_experiment_v4.json").read_text(
            encoding="utf-8"
        )
    )
    counts = _criterion_split_counts(config)
    rows = list(iter_phase_validation_records(config, 0, "active"))
    additions = [row for row in rows if row["family"] == "1AS-1"]

    assert counts["1AS-1"]["validation"] == 22
    assert counts["1AS-1"]["test"] == 22
    assert sum(value["train"] for value in counts.values()) == 100_000
    assert Counter(row["operation"] for row in additions) == {
        "add_2": 11,
        "add_3": 11,
    }
    assert all(row["math_ir"]["op"] == "add" for row in additions)
    assert all("operands" in row["math_ir"] for row in additions)
    assert all(row["derivation"][0]["op"] == "count_on" for row in additions)
    assert len({row["answer"] for row in additions if row["operation"] == "add_2"}) >= 9
    assert len({row["answer"] for row in additions if row["operation"] == "add_3"}) >= 9


def test_v5_addition_trace_is_compact_annotated_and_semantically_verified() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "config" / "math_master_experiment_v5.json").read_text(
            encoding="utf-8"
        )
    )
    additions = [
        row
        for row in iter_phase_validation_records(config, 0, "active")
        if row["family"] == "1AS-1"
    ]

    assert additions
    for row in additions:
        assert row["procedure_schema"] == COMPACT_PROCEDURE_SCHEMA
        assert len(row["derivation"]) == 1
        assert row["derivation"][0]["op"] == "count_on"
        assert all("sequence" not in step for step in row["derivation"][0]["steps"])
        compute_fragments = [
            row["target_trace"][span["start"] : span["end"]]
            for span in row["computation_spans"]
            if span["kind"] == "compute"
        ]
        assert compute_fragments == [
            str(step["end"]) for step in row["derivation"][0]["steps"]
        ]
        assert trace_semantically_matches(row["target_trace"], row)

    row = additions[0]
    noncanonical = (
        "<work>"
        + json.dumps(row["derivation"], indent=1)
        + "</work><answer>"
        + row["answer"]
        + "</answer>"
    )
    assert noncanonical != row["target_trace"]
    assert trace_semantically_matches(noncanonical, row)
    wrong = json.loads(json.dumps(row["derivation"]))
    wrong[0]["steps"][-1]["end"] += 1
    assert not trace_semantically_matches(
        "<work>"
        + json.dumps(wrong)
        + "</work><answer>"
        + row["answer"]
        + "</answer>",
        row,
    )


def test_v5_contract_uses_semantic_not_raw_trace_acceptance() -> None:
    splits = {
        f"phase_{index:02d}_active": {"records": 8 * len(phase["criteria"])}
        for index, phase in enumerate(MASTER_PHASES)
    }
    for index, phase in enumerate(MASTER_PHASES[1:], 1):
        splits[f"phase_{index:02d}_retention"] = {
            "records": sum(len(prior["criteria"]) for prior in MASTER_PHASES[:index])
        }
    contract = build_contract(
        {
            "phases": list(MASTER_PHASES),
            "splits": splits,
            "criterion_operations": {},
            "trace_acceptance_metric": "semantic_v1",
        }
    )
    assert contract["phases"][0]["minimum_trace_exact_by_family"] == {}
    assert contract["phases"][0]["minimum_trace_semantic_by_family"] == {
        criterion: 0.70 for criterion in MASTER_PHASES[0]["criteria"]
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
    assert sum(counts[criterion]["train"] for criterion in first_phase) == 3734


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


def _v6_config() -> dict:
    return json.loads(
        (
            Path(__file__).parents[1]
            / "config"
            / "math_master_experiment_v6.json"
        ).read_text(encoding="utf-8")
    )


def test_v6_has_explicit_balanced_stage_and_criterion_targets() -> None:
    config = _v6_config()
    counts = _criterion_split_counts(config)
    phase_counts = {
        phase["name"]: sum(counts[criterion]["train"] for criterion in phase["criteria"])
        for phase in MASTER_PHASES
    }

    assert sum(value["train"] for value in counts.values()) == 100_000
    assert min(phase_counts.values()) >= 4_000
    assert phase_counts == {
        "y1_number_structure": 5_000,
        "y1_add_sub_fluency": 4_000,
        "y2_place_value_and_across_10": 6_000,
        "y2_add_sub_within_100": 6_000,
        "y2_multiply_divide_2_5_10": 4_000,
        **{phase["name"]: 7_500 for phase in MASTER_PHASES[5:]},
    }
    assert min(value["train"] for value in counts.values()) >= 1_200


def test_v6_systematically_teaches_requested_multipliers_and_divisors() -> None:
    multipliers = {
        int(row["right"])
        for row in _candidate_irs("KS2-LONG-MULTIPLY", V6_DATASET_RECIPE)
        if int(row["left"]) == 101
    }
    divisors = {
        int(row["divisor"])
        for row in _candidate_irs("KS2-EXACT-DIVIDE", V6_DATASET_RECIPE)
        if int(row["dividend"]) // int(row["divisor"]) == 101
    }

    assert {7, 11, 22, 55} <= multipliers
    assert {7, 11, 22, 55} <= divisors
    assert set(range(2, 101)) <= multipliers
    assert set(range(2, 101)) <= divisors


def test_v6_split_blocks_all_views_of_a_semantic_problem() -> None:
    assert _semantic_object_id(
        {"type": "math_problem_v1", "op": "multiply", "left": 7, "right": 11}
    ) == _semantic_object_id(
        {"type": "math_problem_v1", "op": "multiply", "left": 11, "right": 7}
    )
    config = {
        **_config(),
        "dataset_recipe": V6_DATASET_RECIPE,
        "objects_per_criterion": {"train": 24, "validation": 2, "test": 2},
    }
    semantic_splits: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for record in iter_records(config, split):
            semantic_id = _semantic_object_id(record["math_ir"])
            assert record["math_semantic_id"] == semantic_id
            incumbent = semantic_splits.setdefault(semantic_id, split)
            assert incumbent == split


def test_v6_replay_budget_covers_every_accepted_criterion() -> None:
    config = _v6_config()
    counts = _criterion_split_counts(config)
    active_fraction = config["replay_policy"]["active_fraction"]
    prior_fraction = config["replay_policy"]["prior_fraction"]
    minimum = config["replay_policy"]["minimum_rows_per_prior_criterion"]
    prior: list[str] = []
    for phase in MASTER_PHASES:
        active = sum(counts[criterion]["train"] for criterion in phase["criteria"])
        if prior:
            replay = round(active * prior_fraction / active_fraction)
            assert replay // len(prior) >= minimum
        prior.extend(phase["criteria"])
