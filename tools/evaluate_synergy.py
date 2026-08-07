from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.synergy_benchmark import load_synergy_protocol
from cftn_text.synergy_evaluation import evaluate_synergy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether CFTN towers causally collaborate"
    )
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--protocol", default="config/synergy_v1.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--benchmark-manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-pairs-per-split", type=int)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = load_synergy_protocol(args.protocol)
    checkpoint = args.checkpoint or (
        Path(config["project"]["artifact_root"])
        / "bridge_bidirectional_contextual_complementary"
        / "bridge_bidirectional.best.pth"
    )
    benchmark_manifest = args.benchmark_manifest or (
        Path(config["project"]["data_root"]).parents[1]
        / "benchmarks"
        / "synergy_v1"
        / "manifest.json"
    )
    report = evaluate_synergy_checkpoint(
        config,
        protocol,
        checkpoint,
        benchmark_manifest,
        device_name=args.device,
        maximum_pairs_per_split=args.maximum_pairs_per_split,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
