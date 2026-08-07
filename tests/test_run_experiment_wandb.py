from __future__ import annotations

from tools.run_experiment import command_plan


def test_command_plan_adds_wandb_only_to_training_stages(tmp_path):
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

    non_training_commands = [command for command in commands if command not in training_commands]
    assert all("--wandb" not in command for command in non_training_commands)
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


def test_command_plan_leaves_wandb_disabled_by_default(tmp_path):
    commands = command_plan(
        "config.yaml", "synergy.yaml", False, str(tmp_path)
    )
    assert all("--wandb" not in command for command in commands)
