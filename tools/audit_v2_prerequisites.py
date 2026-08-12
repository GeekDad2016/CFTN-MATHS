from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.v2_prerequisites import audit_v2_mechanism_prerequisites


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of V1.2/V1.3 mechanism evidence before V2"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    args = parser.parse_args()
    print(json.dumps(audit_v2_mechanism_prerequisites(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
