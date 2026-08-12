from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.data_generator import audit_manifest, prepare_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate immutable CFTN-Text data")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.output_root or config["project"]["data_root"]
    if config["data"].get("format") == "cftn_text_linear_equations_v1_1":
        from cftn_text.algorithmic_data_generator import (
            audit_algorithmic_manifest,
            prepare_algorithmic_manifests,
        )

        manifest = prepare_algorithmic_manifests(config, root, force=args.force)
        audit = audit_algorithmic_manifest(manifest, root)
    else:
        manifest = prepare_manifests(config, root, force=args.force)
        audit = audit_manifest(manifest, root)
    print(json.dumps({"manifest": manifest, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
