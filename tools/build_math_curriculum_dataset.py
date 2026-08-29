from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.math_curriculum_data import (
    audit_dataset,
    dataset_summary,
    prepare_dataset,
    sample_records,
)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and inspect the staged canonical-math curriculum dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output", required=True)
    audit.add_argument("--scratch-dir")

    summary = subparsers.add_parser("summary")
    summary.add_argument("--output", required=True)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--output", required=True)
    sample.add_argument("--split", choices=("train", "validation", "test"), default="train")
    sample.add_argument("--limit", type=int, default=3)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_dataset(_load(args.config), Path(args.output))
    elif args.command == "audit":
        result = audit_dataset(
            Path(args.output), Path(args.scratch_dir) if args.scratch_dir else None
        )
    elif args.command == "summary":
        result = dataset_summary(Path(args.output))
    else:
        result = sample_records(Path(args.output), args.split, args.limit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
