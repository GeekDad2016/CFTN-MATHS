from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ordered CFTN-Text experiment")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--synergy-protocol", default="config/synergy_v1.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-fixed-open", action="store_true")
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
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
