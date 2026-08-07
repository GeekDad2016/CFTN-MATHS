from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.training import train_math_tower
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the scratch math tower")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip the V1 integer-only GPT calibration gate (used by V2)",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Continue to max_epochs while retaining plateau metrics",
    )
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    result = train_math_tower(
        config,
        device_name=args.device,
        resume=args.resume,
        max_batches=args.max_batches,
        require_calibration=not args.skip_calibration,
        disable_early_stopping=args.disable_early_stopping,
        wandb_options=wandb_options_from_args(
            args,
            default_run_name=f"math-seed-{config['project']['seed']}",
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
