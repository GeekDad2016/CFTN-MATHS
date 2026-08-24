from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_config import load_v1_3_config
from tools.recover_v1_3_answer_bus import (
    configure_answer_bus_recovery,
    evaluate_native_answer_bus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen V1.3 typed request path end to end"
    )
    parser.add_argument("--config", default="G:/ctfn-text/config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="C:/CFTN/typed_request_ablation")
    parser.add_argument(
        "--mode",
        choices=(
            "raw_problem_no_latent",
            "oracle_specialist_problem_no_latent",
            "typed_dispatcher_no_latent",
            "learned_dispatcher_no_latent",
        ),
        default="raw_problem_no_latent",
    )
    parser.add_argument("--deterministic-composer", action="store_true")
    parser.add_argument("--dispatcher-checkpoint")
    parser.add_argument(
        "--specialist-generation-policy",
        choices=("configured", "full_context_v1"),
        default="configured",
    )
    args = parser.parse_args()
    config = load_v1_3_config(args.config)
    root = Path(config["paths"]["artifact_root"])
    source_report = json.loads(
        (
            root
            / "oracle_hard_answer_bus_recovery"
            / "native_answer_bus_report.json"
        ).read_text(encoding="utf-8")
    )
    checkpoint = Path(source_report["checkpoint"]).resolve()
    if file_sha256(checkpoint) != source_report["checkpoint_sha256"]:
        raise RuntimeError("protected answer-composer checkpoint hash changed")
    phase = configure_answer_bus_recovery(config, checkpoint)
    report = evaluate_native_answer_bus(
        config,
        phase,
        checkpoint,
        device_name=args.device,
        output_artifact=Path(args.output),
        lossless_request_mode=args.mode,
        deterministic_answer_composition=bool(args.deterministic_composer),
        specialist_generation_policy=args.specialist_generation_policy,
        dispatcher_checkpoint=(
            Path(args.dispatcher_checkpoint) if args.dispatcher_checkpoint else None
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
