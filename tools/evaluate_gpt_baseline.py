from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.gpt_baseline import evaluate_frozen_gpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the benchmark with frozen GPT")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-examples", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    report = evaluate_frozen_gpt(
        load_config(args.config),
        device_name=args.device,
        maximum_examples=args.maximum_examples,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
