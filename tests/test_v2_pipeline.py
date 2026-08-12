from __future__ import annotations

from pathlib import Path

import pytest

from cftn_text.config import load_config
from tools.run_v2_experiment import (
    Stage,
    _is_complete,
    _validate_wandb_environment,
    command_plan,
)


def test_v2_plan_is_ordered_resumable_end_to_end():
    path = Path(__file__).parents[1] / "config" / "v2_broad_math.yaml"
    config = load_config(path)
    stages = command_plan(str(path), config, device="cuda", wandb=True)
    assert [stage.name for stage in stages] == [
        "audit_mechanism_prerequisites",
        "prepare_data",
        "train_math",
        "select_math_checkpoint",
        "evaluate_math",
        "train_m2g",
        "train_conditional_gpt_to_math",
        "evaluate_shared_no_harm",
        "evaluate_collaboration",
        "assess_scale",
        "assemble_report",
    ]
    math_command = stages[2].command
    assert "--skip-calibration" in math_command
    assert "--disable-early-stopping" in math_command
    assert "--wandb" in math_command
    assert "--view-mode" in stages[5].command
    assert "shared" in stages[5].command
    assert "tools.train_conditional_bridge" in stages[6].command
    assert "--revision-config" in stages[6].command
    assert "shared" in stages[7].command
    assert stages[2].resumable_artifact is not None
    assert stages[6].resumable_artifact is not None


def test_v2_evaluation_completion_fails_closed_on_failed_gate(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"specialist_gate": {"pass": false}}', encoding="utf-8")
    stage = Stage("evaluate_math", [], path)
    assert not _is_complete(stage, {})


def test_online_v2_requires_wandb_key_only_from_environment(monkeypatch):
    path = Path(__file__).parents[1] / "config" / "v2_broad_math.yaml"
    config = load_config(path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        _validate_wandb_environment(config, True)
    monkeypatch.setenv("WANDB_API_KEY", "test-only-not-persisted")
    _validate_wandb_environment(config, True)
