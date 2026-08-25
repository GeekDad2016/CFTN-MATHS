from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.v2_evaluation import evaluate_v2_math_checkpoint
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate broad V2 math generation")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--maximum-examples", type=int)
    parser.add_argument("--output-root")
    parser.add_argument(
        "--working-root",
        help=(
            "Optional local scratch directory for high-frequency status and "
            "generation rows; completed evidence is published to --output-root"
        ),
    )
    add_wandb_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or (
        Path(config["project"]["artifact_root"]) / "math" / "math.best.pth"
    )
    report = evaluate_v2_math_checkpoint(
        config,
        checkpoint,
        device_name=args.device,
        splits=args.splits,
        maximum_examples=args.maximum_examples,
        output_root=args.output_root,
        working_root=args.working_root,
        wandb_options=wandb_options_from_args(
            args, default_run_name="v2-math-evaluation"
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
