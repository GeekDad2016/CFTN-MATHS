from __future__ import annotations

from pathlib import Path

import pytest

from cftn_text.config import load_config
from tools.run_v2_experiment import (
    _validate_wandb_environment,
    command_plan,
)


def test_v2_plan_is_ordered_resumable_end_to_end():
    path = Path(__file__).parents[1] / "config" / "v2_broad_math.yaml"
    config = load_config(path)
    stages = command_plan(str(path), config, device="cuda", wandb=True)
    assert [stage.name for stage in stages] == [
        "prepare_data",
        "train_math",
        "evaluate_math",
        "train_m2g",
        "train_bidirectional",
        "evaluate_collaboration",
        "assess_scale",
        "assemble_report",
    ]
    math_command = stages[1].command
    assert "--skip-calibration" in math_command
    assert "--wandb" in math_command
    assert "--view-mode" in stages[3].command
    assert "complementary" in stages[3].command
    assert stages[1].resumable_artifact is not None
    assert stages[4].resumable_artifact is not None


def test_online_v2_requires_wandb_key_only_from_environment(monkeypatch):
    path = Path(__file__).parents[1] / "config" / "v2_broad_math.yaml"
    config = load_config(path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        _validate_wandb_environment(config, True)
    monkeypatch.setenv("WANDB_API_KEY", "test-only-not-persisted")
    _validate_wandb_environment(config, True)
