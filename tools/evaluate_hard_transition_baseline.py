from __future__ import annotations

import argparse
import json

from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_training import evaluate_hard_transition_baseline
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected soft-wake checkpoint in hard mode without updates"
    )
    parser.add_argument("--config", default="config/v2_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_hard_transition_baseline(
                load_v1_3_config(args.config),
                device_name=args.device,
                max_batches=args.max_batches,
                wandb_options=wandb_options_from_args(
                    args, default_run_name="v2-zero-update-hard-baseline"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
