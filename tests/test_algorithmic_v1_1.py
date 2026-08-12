from __future__ import annotations

import copy
from pathlib import Path

import torch

from cftn_text.algorithmic_data_generator import (
    audit_algorithmic_manifest,
    build_algorithmic_records,
    curriculum_records,
    prepare_algorithmic_manifests,
)
from cftn_text.config import load_config, validate_config
from cftn_text.math_tower import MathTower
from cftn_text.specialist_evaluation import _acceptance_report
from cftn_text.synergy_benchmark import (
    SYNERGY_BENCHMARK_FORMAT,
    audit_synergy_benchmark,
    load_synergy_rows,
    prepare_synergy_benchmark,
)
from cftn_text.tokenizer import ByteMathTokenizer
from tools.wait_then_run_experiment import pipeline_command


def algorithmic_config(tmp_path: Path) -> dict:
    source = Path(__file__).parents[1] / "config" / "v1_1_algorithmic_linear_equations.yaml"
    config = copy.deepcopy(load_config(source))
    config.pop("_meta", None)
    config["project"]["artifact_root"] = str(tmp_path / "artifacts")
    config["project"]["data_root"] = str(tmp_path / "data")
    for key in (
        "calibration_examples",
        "validation_examples",
        "test_examples",
        "heldout_language_examples",
        "extrapolation_examples",
        "answer_extrapolation_examples",
        "compositional_examples",
    ):
        config["data"][key] = 16
    config["data"]["train_examples"] = 256
    config["data"]["curriculum"]["examples_per_epoch"] = 64
    config["data"]["curriculum"]["phases"] = [
        {"name": "foundations", "through_epoch": 1, "max_difficulty": 1},
        {"name": "two_digit", "through_epoch": 2, "max_difficulty": 2},
        {"name": "three_digit", "through_epoch": 3, "max_difficulty": 3},
        {"name": "full_support", "through_epoch": 4, "max_difficulty": 4},
    ]
    config["math_training"]["max_epochs"] = 4
    config["math_training"]["minimum_epochs"] = 4
    validate_config(config)
    return config


def test_algorithmic_ranges_are_disjoint_and_test_one_shift_at_a_time(tmp_path):
    config = algorithmic_config(tmp_path)
    records = build_algorithmic_records(config)
    train = records["train"]
    assert {record.difficulty for record in train} == {1, 2, 3, 4}
    assert all(abs(record.x) <= 200 for record in train)

    input_shift = records["extrapolation"]
    assert all(51 <= abs(record.a) <= 80 for record in input_shift)
    assert all(abs(record.x) <= 200 for record in input_shift)
    assert all(501 <= abs(record.b) <= 800 for record in input_shift)

    answer_shift = records["answer_extrapolation"]
    assert all(abs(record.a) <= 50 for record in answer_shift)
    assert all(201 <= abs(record.x) <= 400 for record in answer_shift)
    assert all(abs(record.b) <= 500 for record in answer_shift)

    equation_ids = [
        record.equation_id for split in records.values() for record in split
    ]
    assert len(equation_ids) == len(set(equation_ids))


def test_algorithmic_manifest_and_epoch_curriculum(tmp_path):
    config = algorithmic_config(tmp_path)
    manifest = prepare_algorithmic_manifests(config)
    audit = audit_algorithmic_manifest(manifest, config["project"]["data_root"])
    assert audit["pass"]
    train = build_algorithmic_records(config)["train"]
    first, first_metadata = curriculum_records(
        [record.to_dict() for record in train], config, epoch=1
    )
    final, final_metadata = curriculum_records(
        [record.to_dict() for record in train], config, epoch=4
    )
    assert first_metadata["max_difficulty"] == 1
    assert all(record["difficulty"] == 1 for record in first)
    assert final_metadata["max_difficulty"] == 4
    assert len(final) == len(train)


def test_algorithmic_records_prepare_synergy_benchmark(tmp_path):
    config = algorithmic_config(tmp_path)
    prepare_algorithmic_manifests(config)
    source_splits = [
        "test",
        "heldout_language",
        "extrapolation",
        "answer_extrapolation",
        "compositional",
    ]
    protocol = {
        "format": SYNERGY_BENCHMARK_FORMAT,
        "seed": 719,
        "benchmark": {
            "source_splits": source_splits,
            "examples_per_split": 2,
            "distractor_splits": ["compositional"],
        },
    }
    output = tmp_path / "synergy"
    manifest = prepare_synergy_benchmark(
        config, protocol, output_root=output
    )

    assert manifest["total_pairs"] == 2 * len(source_splits)
    assert manifest["total_records"] == 4 * len(source_splits)
    for split, metadata in manifest["splits"].items():
        rows = load_synergy_rows(output / metadata["path"])
        base_rows = [
            row for row in rows if row["benchmark"]["pair_variant"] == "base"
        ]
        assert len(base_rows) == 2
        assert all(
            row["schema_version"] == "cftn_linear_equation_v1_1"
            for row in base_rows
        )
        assert all("curriculum_band" in row for row in base_rows)
        assert all(
            row["benchmark"]["source_split"] == split for row in rows
        )
    audit = audit_synergy_benchmark(output / "manifest.json")
    assert audit["manifest_sha256"] == manifest["manifest_sha256"]


def test_disabled_categorical_head_cannot_bias_unseen_integer_classes(tmp_path):
    config = algorithmic_config(tmp_path)
    tower_config = copy.deepcopy(config["math_tower"])
    tower_config.update(
        {
            "layers": 1,
            "hidden_size": 32,
            "attention_heads": 4,
            "feed_forward_size": 64,
            "dropout": 0.0,
        }
    )
    model = MathTower(tower_config, ByteMathTokenizer.vocab_size)
    values = torch.tensor([-400, -51, 0, 51, 400])
    assert model.answer_classes(values).tolist() == [-100] * len(values)
    output = model(
        torch.ones((2, 4), dtype=torch.long),
        torch.ones((2, 4), dtype=torch.long),
        torch.tensor([2, 2]),
    )
    assert torch.count_nonzero(output.answer_logits) == 0


def test_specialist_acceptance_uses_generation_not_disabled_head(tmp_path):
    config = algorithmic_config(tmp_path)
    report = {
        "splits": {
            "test": {
                "generation": {"exact_accuracy": 1.0, "valid_rate": 1.0},
                "canonical_trace_exact_rate": 1.0,
            },
            "extrapolation": {
                "generation": {"exact_accuracy": 0.96, "valid_rate": 1.0},
                "canonical_trace_exact_rate": 0.0,
            },
            "answer_extrapolation": {
                "generation": {"exact_accuracy": 0.81, "valid_rate": 1.0},
                "canonical_trace_exact_rate": 0.0,
            },
        }
    }
    acceptance = _acceptance_report(report, config)
    assert acceptance["pass"]


def test_waiter_launches_the_complete_control_plan(tmp_path):
    command = pipeline_command(
        tmp_path / "config.yaml",
        tmp_path / "synergy.yaml",
        include_fixed_open=True,
        wandb=True,
        wandb_project="cftn-text",
        wandb_run_name="v1-1",
    )
    assert "--execute" in command
    assert "--include-fixed-open" in command
    assert "--wandb" in command
    assert command[command.index("--wandb-project") + 1] == "cftn-text"
