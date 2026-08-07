from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.v2_reporting import assess_scale_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess whether V2 merits a 1M data run")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    args = parser.parse_args()
    print(json.dumps(assess_scale_gate(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
