from __future__ import annotations

import copy

from tools.run_experiment import command_plan, execute_plan


def test_command_plan_adds_wandb_to_training_and_shared_evaluation(tmp_path):
    commands = command_plan(
        "config.yaml",
        "synergy.yaml",
        True,
        str(tmp_path),
        wandb_options={
            "enabled": True,
            "project": "test-project",
            "entity": "test-entity",
            "run_name": "study",
            "group": "study-group",
            "tags": ["test-tag"],
            "mode": "offline",
        },
    )
    training_commands = [
        command
        for command in commands
        if "tools.train_math_tower" in command or "tools.train_bridges" in command
    ]
    assert len(training_commands) == 5
    for command in training_commands:
        assert "--wandb" in command
        assert command[command.index("--wandb-project") + 1] == "test-project"
        assert command[command.index("--wandb-entity") + 1] == "test-entity"
        assert command[command.index("--wandb-group") + 1] == "study-group"
        assert command[command.index("--wandb-mode") + 1] == "offline"
        assert "test-tag" in command
        assert "orchestrated" in command

    evaluation_commands = [
        command for command in commands if "tools.evaluate" in command
    ]
    assert len(evaluation_commands) == 1
    assert "--wandb" in evaluation_commands[0]
    non_logged_commands = [
        command
        for command in commands
        if command not in training_commands and command not in evaluation_commands
    ]
    assert all("--wandb" not in command for command in non_logged_commands)
    names = {
        command[command.index("--wandb-run-name") + 1]
        for command in training_commands
    }
    assert names == {
        "study-math",
        "study-m2g-contextual",
        "study-bidirectional-contextual",
        "study-m2g-fixed-open",
        "study-bidirectional-fixed-open",
    }
    assert (
        evaluation_commands[0][evaluation_commands[0].index("--wandb-run-name") + 1]
        == "study-evaluation-shared"
    )


def test_command_plan_leaves_wandb_disabled_by_default(tmp_path):
    commands = command_plan(
        "config.yaml", "synergy.yaml", False, str(tmp_path)
    )
    assert all("--wandb" not in command for command in commands)


def test_execute_plan_can_resume_from_a_stage(tmp_path, monkeypatch):
    commands = [["python", "one"], ["python", "two"], ["python", "three"]]
    executed: list[list[str]] = []
    statuses: list[dict] = []
    monkeypatch.setattr(
        "tools.run_experiment.subprocess.run",
        lambda command, check: executed.append(command),
    )
    monkeypatch.setattr(
        "tools.run_experiment.atomic_json_dump",
        lambda payload, path: statuses.append(copy.deepcopy(payload)),
    )
    monkeypatch.setattr("tools.run_experiment.gpu_status", lambda: {})

    execute_plan(commands, tmp_path, start_at_stage=2)

    assert executed == commands[1:]
    assert statuses[0]["stage_index"] == 2
    assert statuses[0]["completed_stages"] == ["python"]
    assert statuses[-1]["state"] == "completed"
