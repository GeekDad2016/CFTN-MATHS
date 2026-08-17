from __future__ import annotations

from pathlib import Path

import pytest

from cftn_text.config import load_config
from cftn_text.pipeline_lock import PipelineAlreadyRunning, exclusive_pipeline_lock
from run_v2 import runner_arguments
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
        "prepare_data",
        "train_math",
        "select_math_checkpoint",
        "evaluate_math",
        "audit_mechanism_prerequisites",
        "train_m2g",
        "train_conditional_gpt_to_math",
        "evaluate_shared_no_harm",
        "evaluate_collaboration",
        "assess_scale",
        "assemble_report",
    ]
    math_command = stages[1].command
    assert "--skip-calibration" in math_command
    assert "--disable-early-stopping" in math_command
    assert "--wandb" in math_command
    assert "--view-mode" in stages[5].command
    assert "shared" in stages[5].command
    assert "tools.train_conditional_bridge" in stages[6].command
    assert "--revision-config" in stages[6].command
    assert "shared" in stages[7].command
    assert stages[1].resumable_artifact is not None
    assert stages[6].resumable_artifact is not None
    assert stages[4].name == "audit_mechanism_prerequisites"
    assert stages[5].name == "train_m2g"


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


def test_one_command_launcher_is_resumable_and_wandb_enabled_by_default():
    arguments = runner_arguments([])
    assert "--execute" in arguments
    assert "--resume" in arguments
    assert "--wandb" in arguments
    assert runner_arguments(["--preview"])[-1] == "--wandb"
    assert "--execute" not in runner_arguments(["--preview"])
    assert "--wandb" not in runner_arguments(["--no-wandb"])


def test_pipeline_lock_rejects_a_duplicate_and_releases_after_exit(tmp_path):
    lock_path = tmp_path / "pipeline.lock"
    with exclusive_pipeline_lock(lock_path):
        with pytest.raises(PipelineAlreadyRunning, match="another V2 pipeline"):
            with exclusive_pipeline_lock(lock_path):
                pass
    with exclusive_pipeline_lock(lock_path):
        assert lock_path.is_file()
