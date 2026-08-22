from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_route_sweep import run_v1_3_route_sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustively screen V1.3 hard specialist schedules"
    )
    parser.add_argument("--config", default="config/v1_3_multi_specialist.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="joint_validation")
    parser.add_argument("--screen-examples", type=int, default=100)
    parser.add_argument("--full-examples", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="print the complete report instead of a compact completion summary",
    )
    args = parser.parse_args()
    report = run_v1_3_route_sweep(
        load_v1_3_config(args.config),
        checkpoint=Path(args.checkpoint),
        device_name=args.device,
        split=args.split,
        screen_examples=args.screen_examples,
        full_examples=args.full_examples,
        top_k=args.top_k,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
    )
    output = (
        report
        if args.full_json
        else {
            "state": report["state"],
            "best_schedule": report["best_schedule"],
            "intended_schedule": report["intended_schedule"],
            "sequence_accuracy_gain_over_intended": report[
                "sequence_accuracy_gain_over_intended"
            ],
            "both_components_gain_over_intended": report[
                "both_components_gain_over_intended"
            ],
            "route_can_materially_help": report["route_can_materially_help"],
            "recommended_next_step": report["recommended_next_step"],
            "elapsed_seconds": report["elapsed_seconds"],
        }
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
