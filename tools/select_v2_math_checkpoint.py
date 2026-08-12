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
    args = parser.parse_args()
    print(
        json.dumps(
            select_v2_math_checkpoint(load_config(args.config), device_name=args.device),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
