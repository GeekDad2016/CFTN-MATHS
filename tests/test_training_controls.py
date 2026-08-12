from __future__ import annotations

import sys

from cftn_text.training import (
    _bridge_collapse_diagnostics,
    _bridge_stability_policy,
    _should_stop_early,
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
