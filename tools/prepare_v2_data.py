from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.v2_data import prepare_v2_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable broad-math V2 data")
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-external-benchmarks",
        action="store_true",
        help="Testing-only mode; full experiments must include sealed benchmarks",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare_v2_manifests(
        config,
        force=args.force,
        include_external_benchmarks=not args.skip_external_benchmarks,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
