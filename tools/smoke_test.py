from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.data_generator import prepare_manifests
from cftn_text.gpt_baseline import evaluate_frozen_gpt
from cftn_text.training import train_bridges, train_math_tower


def smoke_config(config: dict, root: Path) -> dict:
    result = copy.deepcopy(config)
    result.pop("_meta", None)
    result["project"]["data_root"] = str(root / "data")
    result["project"]["artifact_root"] = str(root / "artifacts")
    for key in (
        "calibration_examples",
        "train_examples",
        "validation_examples",
        "test_examples",
        "heldout_language_examples",
        "extrapolation_examples",
        "compositional_examples",
    ):
        result["data"][key] = 8
    result["math_tower"].update(
        {
            "layers": 2,
            "hidden_size": 64,
            "attention_heads": 4,
            "feed_forward_size": 128,
            "dropout": 0.0,
            "receiver_layers": [0, 1],
        }
    )
    result["bridge"].update(
        {
            "message_tokens": 4,
            "message_width": 64,
            "attention_heads": 4,
            "gate_hidden_size": 64,
            "dropout": 0.0,
        }
    )
    result["gpt"]["receiver_layers"] = [1, 5, 10]
    for section in ("math_training", "bridge_training"):
        result[section].update(
            {
                "batch_size": 2,
                "eval_batch_size": 2,
                "max_epochs": 1,
                "minimum_epochs": 1,
                "early_stop_patience": 1,
                "num_workers": 0,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One-batch GPU integration smoke test")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    base = load_config(args.config)
    with tempfile.TemporaryDirectory(prefix="cftn_text_smoke_") as temporary:
        config = smoke_config(base, Path(temporary))
        prepare_manifests(config)
        baseline = evaluate_frozen_gpt(
            config,
            device_name=args.device,
            maximum_examples=4,
        )
        math = train_math_tower(
            config,
            device_name=args.device,
            max_batches=1,
            require_calibration=False,
        )
        m2g = train_bridges(
            config,
            stage="m2g",
            device_name=args.device,
            max_batches=1,
        )
        bidirectional = train_bridges(
            config,
            stage="bidirectional",
            device_name=args.device,
            initialize_from=m2g["best_checkpoint"],
            max_batches=1,
        )
        print(
            json.dumps(
                {
                    "pass": True,
                    "device": args.device,
                    "frozen_gpt_calibration": {
                        "aggregate": baseline["aggregate"],
                        "decision": baseline["decision"],
                    },
                    "math": math["final_metrics"],
                    "m2g": m2g["final_metrics"],
                    "bidirectional": bidirectional["final_metrics"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
