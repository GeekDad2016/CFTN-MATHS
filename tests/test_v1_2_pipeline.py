from __future__ import annotations

from pathlib import Path

import pytest

from cftn_text.conditional_training import load_revision_config
from tools.run_v1_2_experiment import (
    _validate_wandb_environment,
    command_plan,
)


def test_v1_2_plan_is_ordered_and_keeps_v1_1_immutable():
    path = Path(__file__).parents[1] / "config" / "v1_2_conditional_bridge.yaml"
    revision = load_revision_config(path)
    stages = command_plan(str(path), revision, device="cuda", wandb=True)
    assert [stage.name for stage in stages] == [
        "audit_v1_1_prerequisites",
        "train_conditional_gpt_to_math",
        "evaluate_shared_no_harm",
        "evaluate_complementary_causality",
        "assemble_v1_2_evidence",
    ]
    assert stages[1].resumable_artifact is not None
    assert "--view-mode" in stages[2].command
    assert "shared" in stages[2].command
    assert "--output-root" in stages[3].command
    v1_root = Path(revision["paths"]["v1_1_artifact_root"])
    assert all(not str(stage.completion_path).startswith(str(v1_root)) for stage in stages)


def test_v1_2_online_wandb_requires_environment_key(monkeypatch):
    path = Path(__file__).parents[1] / "config" / "v1_2_conditional_bridge.yaml"
    revision = load_revision_config(path)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        _validate_wandb_environment(revision, True)
    monkeypatch.setenv("WANDB_API_KEY", "test-only-not-persisted")
    _validate_wandb_environment(revision, True)
