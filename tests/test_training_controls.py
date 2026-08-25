from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cftn_text.dataset import EquationDataset
from cftn_text.training import (
    _bridge_collapse_diagnostics,
    _bridge_stability_policy,
    _phase_generation_acceptance,
    _should_stop_early,
    math_epoch_dataset,
)
from tools import train_math_tower as train_math_cli


def test_early_stopping_can_be_disabled_without_changing_settings():
    settings = {"minimum_epochs": 10, "early_stop_patience": 10}
    assert _should_stop_early(
        epoch=70, patience=10, settings=settings, enabled=True
    )
    assert not _should_stop_early(
        epoch=70, patience=10, settings=settings, enabled=False
    )


def test_bridge_stability_caps_lr_and_slows_contextual_gates():
    policy = _bridge_stability_policy(
        {
            "learning_rate": 2e-4,
            "minimum_learning_rate": 1e-5,
        }
    )
    assert policy["requested_learning_rate"] == 2e-4
    assert policy["effective_learning_rate"] == 5e-5
    assert policy["minimum_learning_rate"] == 1e-5
    assert policy["gate_learning_rate_multiplier"] == 0.5


def test_bridge_collapse_guard_detects_v1_failure_pattern():
    policy = _bridge_stability_policy(
        {
            "learning_rate": 2e-4,
            "minimum_learning_rate": 1e-5,
        }
    )
    best = {
        "loss": 5.38e-5,
        "shuffled_loss_gap": 4.2523,
        "gpt_teacher_forced_sequence_accuracy": 0.9999,
        "math_teacher_forced_sequence_accuracy": 0.9504,
    }
    collapsed = {
        "loss": 9.447,
        "shuffled_loss_gap": 0.00297,
        "gpt_teacher_forced_sequence_accuracy": 0.0,
        "math_teacher_forced_sequence_accuracy": 0.9504,
    }
    diagnostics = _bridge_collapse_diagnostics(collapsed, best, policy)
    assert diagnostics["triggered"] is True
    assert diagnostics["reasons"] == [
        "sequence_accuracy_and_validation_loss",
        "bridge_message_dependence",
    ]


def test_bridge_collapse_guard_ignores_normal_validation_noise():
    policy = _bridge_stability_policy(
        {
            "learning_rate": 2e-4,
            "minimum_learning_rate": 1e-5,
        }
    )
    best = {
        "loss": 0.01,
        "shuffled_loss_gap": 3.0,
        "gpt_teacher_forced_sequence_accuracy": 0.99,
        "math_teacher_forced_sequence_accuracy": 0.95,
    }
    current = {
        "loss": 0.012,
        "shuffled_loss_gap": 2.8,
        "gpt_teacher_forced_sequence_accuracy": 0.985,
        "math_teacher_forced_sequence_accuracy": 0.948,
    }
    assert _bridge_collapse_diagnostics(current, best, policy)["triggered"] is False


def test_math_cli_passes_resume_and_early_stopping_override(monkeypatch, capsys):
    config = {"project": {"seed": 719}}
    captured = {}

    monkeypatch.setattr(train_math_cli, "load_config", lambda _: config)

    def fake_train(received_config, **kwargs):
        captured["config"] = received_config
        captured.update(kwargs)
        return {"state": "started"}

    monkeypatch.setattr(train_math_cli, "train_math_tower", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_math_tower",
            "--resume",
            "--disable-early-stopping",
            "--wandb-mode",
            "disabled",
        ],
    )
    train_math_cli.main()

    assert captured["config"] is config
    assert captured["resume"] is True
    assert captured["disable_early_stopping"] is True
    assert captured["wandb_options"]["run_name"] == "math-seed-719"
    assert '"state": "started"' in capsys.readouterr().out


def test_focused_curriculum_can_require_sampling_without_replacement():
    records = [
        {
            "record_id": str(index),
            "difficulty": 1,
            "source": "cftn_generated",
            "family": "variables_both_sides",
        }
        for index in range(3)
    ] + [
        {
            "record_id": "wrong-family",
            "difficulty": 1,
            "source": "cftn_generated",
            "family": "arithmetic",
        },
        {
            "record_id": "wrong-source",
            "difficulty": 1,
            "source": "deepmind",
            "family": "variables_both_sides",
        },
    ]
    config = {
        "data": {
            "format": "cftn_text_broad_math_v2",
            "curriculum": {
                "enabled": True,
                "examples_per_epoch": 3,
                "sampling": "without_replacement",
                "phases": [
                    {
                        "name": "focused",
                        "through_epoch": 1,
                        "max_difficulty": 1,
                        "sources": ["cftn_generated"],
                        "families": ["variables_both_sides"],
                    }
                ],
            },
        }
    }

    sampled, metadata = math_epoch_dataset(
        EquationDataset(records), config, epoch=1, seed=719
    )

    assert len(sampled) == 3
    assert len({row["record_id"] for row in sampled.records}) == 3
    assert metadata["sampling_policy"] == "without_replacement"
    assert metadata["sampling_with_replacement"] is False
    assert metadata["source_counts"] == {"cftn_generated": 3}
    assert metadata["family_counts"] == {"variables_both_sides": 3}
    assert metadata["filters"] == {
        "sources": ["cftn_generated"],
        "families": ["variables_both_sides"],
    }
    config["data"]["curriculum"]["examples_per_epoch"] = 4
    with pytest.raises(RuntimeError, match="without replacement"):
        math_epoch_dataset(EquationDataset(records), config, epoch=1, seed=719)


def test_shared_trace_recovery_contract_is_fail_closed():
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "v2_math_checkpoint45_shared_trace_recovery.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["require_acceptance_for_best"] is True
    assert contract["math_training"]["input_view"] == "shared_problem_v1"
    assert contract["math_training"]["target_mode"] == "full_trace_v1"
    assert contract["curriculum"] == {
        "examples_per_epoch": 25000,
        "sampling": "without_replacement",
    }
    assert contract["phases"][0]["sources"] == ["cftn_generated"]
    assert contract["phases"][0]["families"] == ["variables_both_sides"]
    assert contract["phases"][0]["minimum_generation_accuracy"] == 0.95
    assert contract["phases"][0]["minimum_valid_rate"] == 0.99


def test_source_quota_curriculum_is_deterministic_and_auditable():
    records = [
        {
            "record_id": f"a-{index}",
            "source": "source_a",
            "family": "family_a",
            "difficulty": 1,
        }
        for index in range(3)
    ] + [
        {
            "record_id": f"b-{index}",
            "source": "source_b",
            "family": "family_b",
            "difficulty": 2,
        }
        for index in range(5)
    ]
    config = {
        "data": {
            "format": "cftn_text_broad_math_v2",
            "curriculum": {
                "enabled": True,
                "examples_per_epoch": 8,
                "phases": [
                    {
                        "name": "balanced",
                        "through_epoch": 1,
                        "max_difficulty": 3,
                        "source_quotas": {"source_a": 4, "source_b": 4},
                    }
                ],
            },
        }
    }

    first, metadata = math_epoch_dataset(
        EquationDataset(records), config, epoch=1, seed=719
    )
    second, _ = math_epoch_dataset(
        EquationDataset(records), config, epoch=1, seed=719
    )

    assert [row["record_id"] for row in first.records] == [
        row["record_id"] for row in second.records
    ]
    assert metadata["sampling_policy"] == "source_quotas_v1"
    assert metadata["sampling_with_replacement"] is True
    assert metadata["source_sampling"]["source_a"]["replacement_examples"] == 1
    assert metadata["source_sampling"]["source_b"]["replacement_examples"] == 0
    assert sum(row["source"] == "source_a" for row in first.records) == 4
    assert sum(row["source"] == "source_b" for row in first.records) == 4


def test_broad_generation_acceptance_requires_every_panel_and_breakdown():
    phase = {
        "name": "broad",
        "through_epoch": 4,
        "primary_generation_panel": "validation_broad",
        "minimum_generation_accuracy": 0.70,
        "minimum_valid_rate": 0.95,
        "minimum_generation_accuracy_by_source": {
            "cftn_generated": 0.70,
            "deepmind_mathematics": 0.60,
        },
        "minimum_generation_accuracy_by_panel": {
            "mathqa_validation": 0.15
        },
        "minimum_valid_rate_by_panel": {"mathqa_validation": 0.90},
        "minimum_teacher_forced_token_accuracy": 0.90,
    }
    panels = {
        "validation_broad": {
            "accuracy": 0.72,
            "valid_rate": 0.97,
            "by_source": {
                "cftn_generated": {"accuracy": 0.75, "examples": 128},
                "deepmind_mathematics": {"accuracy": 0.65, "examples": 896},
            },
        },
        "mathqa_validation": {"accuracy": 0.16, "valid_rate": 0.92},
    }
    validation = {
        "teacher_forced_token_accuracy": 0.91,
        "teacher_forced_sequence_accuracy": 0.30,
        "loss": 0.5,
    }

    accepted = _phase_generation_acceptance(
        phase=phase,
        generation_panels=panels,
        validation=validation,
        epoch=2,
    )
    assert accepted is not None and accepted["pass"] is True
    panels["mathqa_validation"]["accuracy"] = 0.14
    rejected = _phase_generation_acceptance(
        phase=phase,
        generation_panels=panels,
        validation=validation,
        epoch=2,
    )
    assert rejected is not None and rejected["pass"] is False
    assert (
        rejected["checks"]["panel:mathqa_validation:generation_accuracy"]["pass"]
        is False
    )


def test_broad_shared_recovery_contract_covers_all_training_sources():
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "v2_math_checkpoint45_broad_shared_recovery.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["format"] == "cftn_text_v2_math_broad_shared_recovery_v1"
    assert contract["require_acceptance_for_best"] is True
    assert contract["math_training"]["input_view"] == "shared_problem_v1"
    assert contract["math_training"]["target_mode"] == "full_trace_v1"
    assert contract["curriculum"]["examples_per_epoch"] == 400000
    assert sum(contract["phases"][0]["source_quotas"].values()) == 400000
    assert set(contract["phases"][0]["source_quotas"]) == {
        "cftn_generated",
        "deepmind_mathematics",
        "gsm8k",
        "mathqa",
    }
    assert contract["phases"][0]["primary_generation_panel"] == "validation_broad"
    assert contract["phases"][0]["minimum_generation_accuracy_by_panel"] == {
        "validation_broad": 0.70,
        "mathqa_validation": 0.15,
    }
