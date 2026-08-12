from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.conditional_training import (
    audit_revision_prerequisites,
    load_revision_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sealed V1.1 inputs for V1.2")
    parser.add_argument(
        "--revision-config", default="config/v1_2_conditional_bridge.yaml"
    )
    args = parser.parse_args()
    revision = load_revision_config(args.revision_config)
    report = audit_revision_prerequisites(revision)
    output = Path(revision["paths"]["artifact_root"]) / "prerequisites.json"
    atomic_json_dump(report, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
