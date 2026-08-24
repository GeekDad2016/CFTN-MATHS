from __future__ import annotations

import json
from pathlib import Path

import pytest

from cftn_text.config import load_config
from cftn_text.v1_3_config import load_v1_3_config
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
        "assess_math_scale",
        "prepare_multi_specialist_data",
        "train_learned_dispatcher",
        "calibrate_frozen_gpt_language",
        "train_exact_string_specialist",
        "seal_native_specialists",
        "train_single_specialist_capacity",
        "train_dense_mixed_messages",
        "train_dense_recurrent",
        "train_supervised_soft_wake",
        "evaluate_zero_update_hard_baseline",
        "train_hardened_wake",
        "evaluate_native_typed_dispatch",
        "evaluate_sealed_causal_suite",
        "assemble_v2_evidence",
    ]
    math_command = stages[1].command
    assert "--skip-calibration" in math_command
    assert "--disable-early-stopping" not in math_command
    assert "--wandb" in math_command
    assert stages[1].epoch_limit == 100
    assert config["math_training"]["minimum_epochs"] == 60
    assert config["math_training"]["early_stop_patience"] == 10
    assert config["math_training"]["early_stopping_enabled"] is True
    assert [
        phase["through_epoch"] for phase in config["data"]["curriculum"]["phases"]
    ] == [10, 30, 100]
    assert "tools.prepare_v1_3_data" in stages[5].command
    assert "tools.train_v2_dispatcher" in stages[6].command
    assert "tools.train_v1_3_string" in stages[8].command
    assert "tools.evaluate_hard_transition_baseline" in stages[14].command
    assert "hardened_wake" in stages[15].command
    assert "tools.evaluate_v2_native_dispatch" in stages[16].command
    assert stages[1].resumable_artifact is not None
    assert stages[8].resumable_artifact is not None
    assert stages[15].resumable_artifact is not None


def test_v2_evaluation_completion_fails_closed_on_failed_gate(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"specialist_gate": {"pass": false}}', encoding="utf-8")
    stage = Stage("evaluate_math", [], path)
    assert not _is_complete(stage, {})


def test_v2_resume_rejects_stale_multi_specialist_revision(tmp_path):
    root = Path(__file__).parents[1]
    base = load_config(root / "config" / "v2_broad_math.yaml")
    revision = load_v1_3_config(root / "config" / "v2_multi_specialist.yaml")
    path = tmp_path / "summary.json"
    summary = {"state": "completed", "revision_sha256": "stale"}
    path.write_text(json.dumps(summary), encoding="utf-8")
    stage = Stage("train_dense_recurrent", [], path)
    assert not _is_complete(stage, base)
    summary["revision_sha256"] = revision["_meta"]["sha256"]
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert _is_complete(stage, base)


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


def test_runpod_bootstrap_installs_preflights_and_executes_pipeline():
    repository = Path(__file__).parents[1]
    script = (repository / "start_v2_runpod.sh").read_text(
        encoding="utf-8"
    )
    assert '-e "${script_dir}"' in script
    assert "EXTERNALLY-MANAGED" in script
    assert "--break-system-packages" in script
    assert 'run_v2.py --preflight-only "$@"' in script
    assert 'exec "${python_bin}" run_v2.py "$@"' in script
    assert "read -r -s" in script
    assert "tr -d '[:space:]'" in script
    assert "WANDB_API_KEY contained only whitespace" in script
    assert "CFTN_V2_MULTI_DATA_ROOT" in script
    assert 'CFTN_STORAGE_ROOT:-/workspace/cftn-text' in script
    assert "CFTN_ALLOW_EPHEMERAL_STORAGE" in script
    assert "PIP_CACHE_DIR" in script
    assert 'argument}" == "--no-wandb"' in script
    assert 'argument}" == "--preflight-only"' in script
    assert 'git fetch origin "${expected_branch}"' in script
    assert 'git merge --ff-only "origin/${expected_branch}"' in script
    assert "git pull --ff-only" not in script
    assert "Preflight-only mode complete; training was not launched." in script
    assert "/workspace/volume" not in script
    assert "*.egg-info/" in (repository / ".gitignore").read_text(encoding="utf-8")


def test_runpod_defaults_keep_all_durable_state_on_workspace_volume():
    repository = Path(__file__).parents[1]
    durable_files = (
        ".env.example",
        "RUNPOD_V2.md",
        "scripts/bootstrap_runpod_access.sh",
        "scripts/runpod_entrypoint.sh",
        "start_v2_runpod.sh",
        "tools/check_v2_heartbeat.py",
        "tools/watch_v2_progress.py",
    )
    for relative_path in durable_files:
        text = (repository / relative_path).read_text(encoding="utf-8")
        assert "/workspace/volume" not in text, relative_path
        assert "/workspace/cftn-text" in text, relative_path

    entrypoint = (repository / "scripts/runpod_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert "CFTN_REPOSITORY_ROOT:-/workspace/CFTN-MATHS" in entrypoint
    assert "verify_durable_mount" in entrypoint


def test_persistent_bootstrap_is_safe_by_default_and_launch_is_explicit():
    repository = Path(__file__).parents[1]
    script = (repository / "scripts" / "bootstrap_runpod_access.sh").read_text(
        encoding="utf-8"
    )
    assert 'mode="prepare"' in script
    assert "--access-only" in script
    assert "--launch" in script
    assert 'start_v2_runpod.sh" --preflight-only --no-wandb' in script
    assert 'mode}" == "launch"' in script
    assert "/workspace/cftn-start.sh" in script
    assert "id_ed25519_runpod_cftn.pub" in script
    assert "ssh-ed25519 " not in script
    assert "The default mode never starts training." in script


def test_pipeline_lock_rejects_a_duplicate_and_releases_after_exit(tmp_path):
    lock_path = tmp_path / "pipeline.lock"
    with exclusive_pipeline_lock(lock_path):
        with pytest.raises(PipelineAlreadyRunning, match="another V2 pipeline"):
            with exclusive_pipeline_lock(lock_path):
                pass
    with exclusive_pipeline_lock(lock_path):
        assert lock_path.is_file()
