"""One-command, resumable launcher for the CFTN-Text V2 experiment.

RunPod usage:

    python run_v2.py

Online W&B logging is enabled by default and reads WANDB_API_KEY only from the
process environment. Use --no-wandb only for an intentional local/offline run.
"""

from __future__ import annotations

import argparse

from tools.run_v2_experiment import main as run_pipeline


def runner_arguments(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Start or safely resume the complete CFTN-Text V2 experiment"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--from-stage")
    parser.add_argument("--through-stage")
    args = parser.parse_args(argv)

    output = [
        "--config",
        args.config,
        "--device",
        args.device,
        "--resume",
    ]
    if not args.preview and not args.preflight_only:
        output.append("--execute")
    if args.preflight_only:
        output.append("--preflight-only")
    if not args.no_wandb:
        output.append("--wandb")
    if args.from_stage:
        output.extend(["--from-stage", args.from_stage])
    if args.through_stage:
        output.extend(["--through-stage", args.through_stage])
    return output


def main(argv: list[str] | None = None) -> None:
    run_pipeline(runner_arguments(argv))


if __name__ == "__main__":
    main()
