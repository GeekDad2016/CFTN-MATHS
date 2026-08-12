from __future__ import annotations

import argparse
import json

from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_evaluation import evaluate_gpt_language_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate frozen GPT for V1.3")
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-examples", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_gpt_language_calibration(
                load_v1_3_config(args.config),
                device_name=args.device,
                maximum_examples=args.maximum_examples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
