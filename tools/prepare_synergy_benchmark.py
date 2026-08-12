from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.synergy_benchmark import (
    load_synergy_protocol,
    prepare_synergy_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the immutable complementary-view synergy benchmark"
    )
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--protocol", default="config/synergy_v1.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = load_synergy_protocol(args.protocol)
    benchmark_name = str(protocol["benchmark"].get("artifact_name", "synergy_v1"))
    output_root = args.output_root or (
        Path(config["project"]["data_root"]).parents[1]
        / "benchmarks"
        / benchmark_name
    )
    report = prepare_synergy_benchmark(
        config,
        protocol,
        output_root=output_root,
        force=args.force,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
