from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.v2_joint_evaluation import evaluate_v2_collaboration
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 causal bridge ablations")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--math-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--maximum-examples", type=int)
    parser.add_argument("--output-root")
    add_wandb_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or (
        Path(config["project"]["artifact_root"])
        / "bridge_bidirectional_contextual_complementary"
        / "bridge_bidirectional.best.pth"
    )
    report = evaluate_v2_collaboration(
        config,
        checkpoint,
        math_checkpoint_path=args.math_checkpoint,
        device_name=args.device,
        splits=args.splits,
        maximum_examples=args.maximum_examples,
        output_root=args.output_root,
        wandb_options=wandb_options_from_args(
            args, default_run_name="v2-collaboration-evaluation"
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
