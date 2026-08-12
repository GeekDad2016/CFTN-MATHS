from __future__ import annotations

import argparse
import json

from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_training import train_string_specialist
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V1.3 exact-string tower")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    config = load_v1_3_config(args.config)
    print(
        json.dumps(
            train_string_specialist(
                config,
                device_name=args.device,
                resume=args.resume,
                max_batches=args.max_batches,
                wandb_options=wandb_options_from_args(
                    args, default_run_name="v1-3-string-specialist"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
