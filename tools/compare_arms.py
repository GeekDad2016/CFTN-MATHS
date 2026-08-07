from __future__ import annotations

import argparse
import json

from cftn_text.research import compare_evaluation_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gated and fixed-open arms")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=719)
    args = parser.parse_args()
    report = compare_evaluation_reports(
        args.candidate,
        args.baseline,
        output_path=args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
