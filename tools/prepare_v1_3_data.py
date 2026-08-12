from __future__ import annotations

import argparse
import json

from cftn_text.v1_3_config import audit_v1_2_pass, load_v1_3_config
from cftn_text.v1_3_data import prepare_v1_3_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic V1.3 data")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    args = parser.parse_args()
    config = load_v1_3_config(args.config)
    audit_v1_2_pass(config)
    print(json.dumps(prepare_v1_3_manifests(config), indent=2))


if __name__ == "__main__":
    main()
