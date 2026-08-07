from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.v2_reporting import assemble_v2_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the final V2 evidence report")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    args = parser.parse_args()
    print(json.dumps(assemble_v2_report(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
