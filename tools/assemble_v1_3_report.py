from __future__ import annotations

import argparse
import json

from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_reporting import assemble_v1_3_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble sealed V1.3 evidence")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    args = parser.parse_args()
    print(json.dumps(assemble_v1_3_report(load_v1_3_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
