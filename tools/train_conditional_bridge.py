from __future__ import annotations

import argparse
import json

from cftn_text.conditional_training import (
    load_revision_config,
    train_conditional_bridge,
)
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a V1.2/V2 mixed-necessity contextual bridge"
    )
    parser.add_argument(
        "--revision-config", default="config/v1_2_conditional_bridge.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    revision = load_revision_config(args.revision_config)
    result = train_conditional_bridge(
        revision,
        device_name=args.device,
        resume=args.resume,
        max_batches=args.max_batches,
        wandb_options=wandb_options_from_args(
            args, default_run_name="conditional-gpt-to-math"
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
