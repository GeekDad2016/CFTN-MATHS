from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.training import train_math_tower
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover V2 Stage-2 math training from a sealed model checkpoint"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument(
        "--recovery-contract",
        default="config/v2_math_checkpoint45_shared_trace_recovery.json",
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--artifact-directory", required=True)
    parser.add_argument("--working-directory", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    contract_path = Path(args.recovery_contract).expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    supported_formats = {
        "cftn_text_v2_math_answer_recovery_v1",
        "cftn_text_v2_math_shared_trace_recovery_v1",
    }
    if contract.get("format") not in supported_formats:
        raise ValueError("unsupported V2 math recovery contract")
    source = Path(args.source_checkpoint).expanduser().resolve()
    observed_source_sha256 = file_sha256(source)
    if observed_source_sha256 != contract.get("source_checkpoint_sha256"):
        raise RuntimeError("checkpoint 45 does not match the recovery contract")

    artifact_directory = Path(args.artifact_directory).expanduser().resolve()
    artifact_directory.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {
            **contract,
            "contract_path": str(contract_path),
            "source_checkpoint": str(source),
            "observed_source_checkpoint_sha256": observed_source_sha256,
            "artifact_directory": str(artifact_directory),
            "working_directory": str(Path(args.working_directory).expanduser().resolve()),
        },
        artifact_directory / "recovery_contract.json",
    )
    result = train_math_tower(
        config,
        device_name=args.device,
        max_batches=args.max_batches,
        require_calibration=False,
        disable_early_stopping=True,
        wandb_options=wandb_options_from_args(
            args,
            default_run_name="v2-math-checkpoint45-shared-trace-recovery",
        ),
        initial_checkpoint=source,
        artifact_directory=artifact_directory,
        working_directory=args.working_directory,
        recovery_contract=contract,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
