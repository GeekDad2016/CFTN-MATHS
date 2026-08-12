from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump, gpu_status
from cftn_text.config import load_config
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def _wandb_arguments(
    options: dict[str, Any] | None,
    *,
    run_suffix: str,
    stage_tags: list[str],
) -> list[str]:
    if not options or not bool(options.get("enabled", False)):
        return []
    prefix = str(options.get("run_name") or "cftn-text")
    arguments = [
        "--wandb",
        "--wandb-project",
        str(options.get("project") or "cftn-text"),
        "--wandb-run-name",
        f"{prefix}-{run_suffix}",
        "--wandb-mode",
        str(options.get("mode") or "online"),
    ]
    entity = options.get("entity")
    if entity:
        arguments.extend(["--wandb-entity", str(entity)])
    arguments.extend(
        ["--wandb-group", str(options.get("group") or prefix)]
    )
    tags = list(
        dict.fromkeys(
            [str(tag) for tag in options.get("tags", [])]
            + ["orchestrated", *stage_tags]
        )
    )
    if tags:
        arguments.extend(["--wandb-tags", *tags])
    return arguments


def command_plan(
    config_path: str,
    synergy_protocol_path: str,
    include_fixed_open: bool,
    artifact_root: str,
    wandb_options: dict[str, Any] | None = None,
) -> list[list[str]]:
    artifact_path = Path(artifact_root)
    contextual_checkpoint = str(
        artifact_path
        / "bridge_bidirectional_contextual_complementary"
        / "bridge_bidirectional.best.pth"
    )
    commands = [
        [sys.executable, "-m", "tools.prepare_data", "--config", config_path],
        [
            sys.executable,
            "-m",
            "tools.prepare_synergy_benchmark",
            "--config",
            config_path,
            "--protocol",
            synergy_protocol_path,
        ],
        [
            sys.executable,
            "-m",
            "tools.evaluate_gpt_baseline",
            "--config",
            config_path,
        ],
        [
            sys.executable,
            "-m",
            "tools.train_math_tower",
            "--config",
            config_path,
            *_wandb_arguments(
                wandb_options,
                run_suffix="math",
                stage_tags=["math-tower"],
            ),
        ],
        [sys.executable, "-m", "tools.evaluate_math_tower", "--config", config_path],
        [
            sys.executable,
            "-m",
            "tools.train_bridges",
            "--config",
            config_path,
            "--stage",
            "m2g",
            *_wandb_arguments(
                wandb_options,
                run_suffix="m2g-contextual",
                stage_tags=["bridge", "m2g", "contextual-gates"],
            ),
        ],
        [
            sys.executable,
            "-m",
            "tools.train_bridges",
            "--config",
            config_path,
            "--stage",
            "bidirectional",
            "--view-mode",
            "complementary",
            *_wandb_arguments(
                wandb_options,
                run_suffix="bidirectional-contextual",
                stage_tags=["bridge", "bidirectional", "contextual-gates"],
            ),
        ],
        [
            sys.executable,
            "-m",
            "tools.evaluate",
            "--config",
            config_path,
            "--checkpoint",
            contextual_checkpoint,
            "--view-mode",
            "shared",
            "--output-root",
            str(artifact_path / "evaluation_bidirectional_contextual_shared"),
            *_wandb_arguments(
                wandb_options,
                run_suffix="evaluation-shared",
                stage_tags=["evaluation", "shared", "contextual-gates"],
            ),
        ],
        [
            sys.executable,
            "-m",
            "tools.evaluate_synergy",
            "--config",
            config_path,
            "--protocol",
            synergy_protocol_path,
            "--checkpoint",
            contextual_checkpoint,
        ],
        [
            sys.executable,
            "-m",
            "tools.assemble_evidence",
            "--math-report",
            str(artifact_path / "evaluation_math" / "report.json"),
            "--shared-report",
            str(
                artifact_path
                / "evaluation_bidirectional_contextual_shared"
                / "report.json"
            ),
            "--synergy-report",
            str(artifact_path / "synergy_evaluation_contextual" / "report.json"),
            "--output-root",
            str(artifact_path / "evidence_candidate"),
        ],
    ]
    if include_fixed_open:
        baseline_checkpoint = str(
            artifact_path
            / "bridge_bidirectional_fixed_open_complementary"
            / "bridge_bidirectional.best.pth"
        )
        candidate_report = str(
            artifact_path
            / "synergy_evaluation_contextual"
            / "report.json"
        )
        baseline_report = str(
            artifact_path
            / "synergy_evaluation_fixed_open"
            / "report.json"
        )
        commands.extend(
            [
                [
                    sys.executable,
                    "-m",
                    "tools.train_bridges",
                    "--config",
                    config_path,
                    "--stage",
                    "m2g",
                    "--gate-mode",
                    "fixed_open",
                    *_wandb_arguments(
                        wandb_options,
                        run_suffix="m2g-fixed-open",
                        stage_tags=["bridge", "m2g", "fixed-open"],
                    ),
                ],
                [
                    sys.executable,
                    "-m",
                    "tools.train_bridges",
                    "--config",
                    config_path,
                    "--stage",
                    "bidirectional",
                    "--gate-mode",
                    "fixed_open",
                    "--view-mode",
                    "complementary",
                    *_wandb_arguments(
                        wandb_options,
                        run_suffix="bidirectional-fixed-open",
                        stage_tags=["bridge", "bidirectional", "fixed-open"],
                    ),
                ],
                [
                    sys.executable,
                    "-m",
                    "tools.evaluate_synergy",
                    "--config",
                    config_path,
                    "--protocol",
                    synergy_protocol_path,
                    "--checkpoint",
                    baseline_checkpoint,
                ],
                [
                    sys.executable,
                    "-m",
                    "tools.compare_synergy_arms",
                    "--contextual",
                    candidate_report,
                    "--fixed-open",
                    baseline_report,
                    "--output",
                    str(artifact_path / "synergy_architecture_comparison.json"),
                ],
                [
                    sys.executable,
                    "-m",
                    "tools.assemble_evidence",
                    "--math-report",
                    str(artifact_path / "evaluation_math" / "report.json"),
                    "--shared-report",
                    str(
                        artifact_path
                        / "evaluation_bidirectional_contextual_shared"
                        / "report.json"
                    ),
                    "--synergy-report",
                    candidate_report,
                    "--architecture-comparison",
                    str(artifact_path / "synergy_architecture_comparison.json"),
                    "--output-root",
                    str(artifact_path / "evidence_final"),
                ],
            ]
        )
    return commands


def command_stage_name(command: list[str]) -> str:
    try:
        module = command[command.index("-m") + 1]
    except (ValueError, IndexError):
        return Path(command[0]).stem
    direct = {
        "tools.prepare_data": "prepare_data",
        "tools.prepare_synergy_benchmark": "prepare_synergy_benchmark",
        "tools.evaluate_gpt_baseline": "evaluate_gpt_baseline",
        "tools.train_math_tower": "train_math",
        "tools.evaluate_math_tower": "evaluate_math",
        "tools.evaluate": "evaluate_shared_cftn",
        "tools.compare_synergy_arms": "compare_contextual_vs_fixed_open",
    }
    if module in direct:
        return direct[module]
    if module == "tools.train_bridges":
        stage = command[command.index("--stage") + 1]
        gate = (
            command[command.index("--gate-mode") + 1]
            if "--gate-mode" in command
            else "contextual"
        )
        view = (
            command[command.index("--view-mode") + 1]
            if "--view-mode" in command
            else "shared"
        )
        return f"train_{stage}_{gate}_{view}"
    if module == "tools.evaluate_synergy":
        checkpoint = command[command.index("--checkpoint") + 1]
        gate = "fixed_open" if "fixed_open" in checkpoint else "contextual"
        return f"evaluate_synergy_{gate}"
    if module == "tools.assemble_evidence":
        output = command[command.index("--output-root") + 1]
        return (
            "assemble_final_evidence"
            if "evidence_final" in output
            else "assemble_candidate_evidence"
        )
    return module.replace("tools.", "")


def execute_plan(
    commands: list[list[str]],
    artifact_root: str | Path,
    *,
    start_at_stage: int = 1,
) -> None:
    root = Path(artifact_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "pipeline_status.json"
    started_at = time.time()
    names = [command_stage_name(command) for command in commands]
    if not 1 <= int(start_at_stage) <= len(commands):
        raise ValueError(
            f"start_at_stage must be between 1 and {len(commands)}, got {start_at_stage}"
        )
    completed: list[str] = list(names[: int(start_at_stage) - 1])
    for zero_index in range(int(start_at_stage) - 1, len(commands)):
        index = zero_index + 1
        name = names[zero_index]
        command = commands[zero_index]
        atomic_json_dump(
            {
                "state": "running",
                "pid": os.getpid(),
                "current_stage": name,
                "stage_index": index,
                "stages_total": len(commands),
                "completed_stages": completed,
                "command": subprocess.list2cmdline(command),
                "elapsed_seconds": time.time() - started_at,
                "gpu": gpu_status(),
            },
            status_path,
        )
        try:
            subprocess.run(command, check=True)
        except BaseException as exc:
            atomic_json_dump(
                {
                    "state": "error",
                    "pid": os.getpid(),
                    "failed_stage": name,
                    "stage_index": index,
                    "stages_total": len(commands),
                    "completed_stages": completed,
                    "error": repr(exc),
                    "elapsed_seconds": time.time() - started_at,
                    "gpu": gpu_status(),
                },
                status_path,
            )
            raise
        completed.append(name)
    atomic_json_dump(
        {
            "state": "completed",
            "pid": os.getpid(),
            "current_stage": None,
            "stage_index": len(commands),
            "stages_total": len(commands),
            "completed_stages": completed,
            "elapsed_seconds": time.time() - started_at,
            "gpu": gpu_status(),
        },
        status_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ordered CFTN-Text experiment")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--synergy-protocol", default="config/synergy_v1.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-fixed-open", action="store_true")
    parser.add_argument("--start-at-stage", type=int, default=1)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    config_path = str(Path(args.config).resolve())
    synergy_protocol_path = str(Path(args.synergy_protocol).resolve())
    config = load_config(config_path)
    commands = command_plan(
        config_path,
        synergy_protocol_path,
        args.include_fixed_open,
        config["project"]["artifact_root"],
        wandb_options=wandb_options_from_args(
            args, default_run_name=config["project"]["name"]
        ),
    )
    preview = {
        "project": config["project"]["name"],
        "execute": args.execute,
        "start_at_stage": args.start_at_stage,
        "commands": [subprocess.list2cmdline(command) for command in commands],
        "training_limits": {
            "math_max_epochs": config["math_training"]["max_epochs"],
            "bridge_max_epochs_per_stage": config["bridge_training"]["max_epochs"],
            "early_stopping": True,
        },
    }
    print(json.dumps(preview, indent=2))
    if not args.execute:
        return
    execute_plan(
        commands,
        config["project"]["artifact_root"],
        start_at_stage=args.start_at_stage,
    )


if __name__ == "__main__":
    main()
