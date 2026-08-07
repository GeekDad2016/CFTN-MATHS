from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.training import train_bridges
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFTN-Text communication")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--stage", choices=("m2g", "bidirectional"), required=True)
    parser.add_argument("--gate-mode", choices=("contextual", "fixed_open"), default="contextual")
    parser.add_argument(
        "--view-mode", choices=("shared", "complementary"), default="shared"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-from")
    parser.add_argument("--math-checkpoint")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    initialize_from = args.initialize_from
    if args.stage == "bidirectional" and initialize_from is None:
        candidate = (
            Path(config["project"]["artifact_root"])
            / f"bridge_m2g_{args.gate_mode}"
            / "bridge_m2g.best.pth"
        )
        if candidate.exists():
            initialize_from = str(candidate)
    result = train_bridges(
        config,
        stage=args.stage,
        device_name=args.device,
        resume=args.resume,
        initialize_from=initialize_from,
        math_checkpoint_path=args.math_checkpoint,
        gate_mode=args.gate_mode,
        view_mode=args.view_mode,
        max_batches=args.max_batches,
        wandb_options=wandb_options_from_args(
            args,
            default_run_name=(
                f"{args.stage}-{args.gate_mode}-{args.view_mode}-"
                f"seed-{config['project']['seed']}"
            ),
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
