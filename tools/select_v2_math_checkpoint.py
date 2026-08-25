from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.v2_checkpoint_selection import select_v2_math_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select V2 math checkpoint by held-out greedy generation"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        help="Evaluate only this math checkpoint; repeat to provide multiple candidates",
    )
    parser.add_argument(
        "--working-root",
        help="Optional local scratch root for high-frequency candidate output",
    )
    parser.add_argument(
        "--reuse-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a hash-validated completed candidate evaluation",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            select_v2_math_checkpoint(
                load_config(args.config),
                device_name=args.device,
                candidate_paths=args.checkpoints,
                working_root=args.working_root,
                reuse_completed=args.reuse_completed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
