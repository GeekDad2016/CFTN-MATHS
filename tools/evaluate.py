from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.evaluation import evaluate_model_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CFTN-Text controls")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--maximum-examples", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--view-mode", choices=("shared", "complementary"))
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = (
            Path(config["project"]["artifact_root"])
            / "bridge_bidirectional_contextual"
            / "bridge_bidirectional.best.pth"
        )
    result = evaluate_model_checkpoint(
        config,
        checkpoint,
        device_name=args.device,
        splits=args.splits,
        maximum_examples=args.maximum_examples,
        output_root=args.output_root,
        view_mode=args.view_mode,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
