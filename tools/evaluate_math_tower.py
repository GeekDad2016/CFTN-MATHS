from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.specialist_evaluation import evaluate_math_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone math-tower generation on sealed splits"
    )
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--maximum-examples", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or (
        Path(config["project"]["artifact_root"]) / "math" / "math.best.pth"
    )
    report = evaluate_math_checkpoint(
        config,
        checkpoint,
        device_name=args.device,
        splits=args.splits,
        maximum_examples=args.maximum_examples,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
