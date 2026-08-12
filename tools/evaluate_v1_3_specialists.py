from __future__ import annotations

import argparse
import json

from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_evaluation import (
    SPECIALIST_GENERATION_POLICIES,
    evaluate_native_specialists,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal V1.3 native specialist tests")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-examples", type=int)
    parser.add_argument(
        "--specialist-generation-policy",
        choices=sorted(SPECIALIST_GENERATION_POLICIES),
        default="configured",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_native_specialists(
                load_v1_3_config(args.config),
                device_name=args.device,
                maximum_examples=args.maximum_examples,
                specialist_generation_policy=args.specialist_generation_policy,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
