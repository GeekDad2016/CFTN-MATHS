from __future__ import annotations

import argparse
import json

from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_training import train_integration_phase
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ordered V1.3 integration phase")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    config = load_v1_3_config(args.config)
    print(
        json.dumps(
            train_integration_phase(
                config,
                args.phase,
                device_name=args.device,
                resume=args.resume,
                max_batches=args.max_batches,
                wandb_options=wandb_options_from_args(
                    args, default_run_name=f"v1-3-{args.phase.replace('_', '-')}"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
