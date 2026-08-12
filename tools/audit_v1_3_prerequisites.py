from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.v1_3_config import audit_v1_2_pass, load_v1_3_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the sealed V1.2 gate for V1.3")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    args = parser.parse_args()
    config = load_v1_3_config(args.config)
    report = audit_v1_2_pass(config)
    path = Path(config["paths"]["artifact_root"]) / "prerequisites.json"
    atomic_json_dump(report, path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
