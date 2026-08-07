from __future__ import annotations

import argparse
import json

from cftn_text.research import compare_synergy_evaluation_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare contextual and fixed-open complementary-view arms"
    )
    parser.add_argument("--contextual", required=True)
    parser.add_argument("--fixed-open", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--minimum-improvement", type=float, default=0.02)
    args = parser.parse_args()
    report = compare_synergy_evaluation_reports(
        args.contextual,
        args.fixed_open,
        output_path=args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        minimum_improvement=args.minimum_improvement,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
