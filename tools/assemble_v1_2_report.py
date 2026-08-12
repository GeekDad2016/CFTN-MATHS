from __future__ import annotations

import argparse
import json

from cftn_text.conditional_reporting import assemble_v1_2_report_from_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble V1.2 bridge evidence")
    parser.add_argument(
        "--revision-config", default="config/v1_2_conditional_bridge.yaml"
    )
    args = parser.parse_args()
    print(json.dumps(assemble_v1_2_report_from_path(args.revision_config), indent=2))


if __name__ == "__main__":
    main()
