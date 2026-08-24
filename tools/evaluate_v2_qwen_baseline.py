from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.v2_qwen_baseline import evaluate_frozen_v2_qwen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen V2 Qwen coordinator on broad math"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--examples-per-difficulty", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-mode",
        choices=("brief_reasoning", "answer_only"),
        default="brief_reasoning",
    )
    parser.add_argument(
        "--panel-manifest",
        help="immutable gap-panel JSON; replaces difficulty-balanced selection",
    )
    parser.add_argument(
        "--panel-subset",
        choices=("full", "challenge"),
        default="full",
    )
    parser.add_argument(
        "--gpu-hourly-usd",
        type=float,
        help="optional current GPU hourly price for estimated run cost",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow Hugging Face downloads instead of requiring the pinned cache",
    )
    args = parser.parse_args()
    report = evaluate_frozen_v2_qwen(
        load_config(args.config),
        split=args.split,
        examples_per_difficulty=args.examples_per_difficulty,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=args.prompt_mode,
        panel_manifest=args.panel_manifest,
        panel_subset=args.panel_subset,
        gpu_hourly_usd=args.gpu_hourly_usd,
        device_name=args.device,
        local_files_only=not args.allow_download,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
