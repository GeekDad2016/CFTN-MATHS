from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.config import canonical_json
from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_training import train_integration_phase
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args
from tools.recover_v1_3_fusion import configure_fusion_recovery


SOURCE_PHASE_NAME = "oracle_hard_fusion_recovery"
PHASE_NAME = "oracle_hard_fusion_continuation"
TERMINAL_SOURCE_STATES = {"completed", "failed_acceptance", "error"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"expected an object at {path}:{line_number}")
        records.append(value)
    return records


def _acceptance_passed(metrics: dict[str, Any]) -> bool:
    return (
        metrics.get("checkpoint_eligible") is True
        and metrics.get("hardening_acceptance", {}).get("gates", {}).get("pass")
        is True
    )


def evaluate_late_improvement(
    metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    minimum_delta: float,
) -> dict[str, Any]:
    prior = [
        row
        for row in metrics
        if int(row.get("epoch", -1)) <= 4
        and _acceptance_passed(row)
        and math.isfinite(float(row.get("selection_metric", float("nan"))))
    ]
    if not prior:
        raise RuntimeError("no accepted epoch-0-to-4 selection baseline exists")
    prior_best = max(prior, key=lambda row: float(row["selection_metric"]))
    best = summary.get("best_metrics")
    if not isinstance(best, dict):
        raise RuntimeError("fusion recovery summary has no best_metrics object")
    best_epoch = int(best.get("epoch", -1))
    best_selection = float(best.get("selection_metric", float("nan")))
    prior_selection = float(prior_best["selection_metric"])
    delta = best_selection - prior_selection
    triggered = (
        best_epoch in {5, 6}
        and _acceptance_passed(best)
        and math.isfinite(best_selection)
        and delta >= float(minimum_delta)
    )
    return {
        "format": "cftn_text_v1_3_fusion_continuation_trigger_v1",
        "triggered": triggered,
        "minimum_delta": float(minimum_delta),
        "prior_best_epoch": int(prior_best["epoch"]),
        "prior_best_selection_metric": prior_selection,
        "late_best_epoch": best_epoch,
        "late_best_selection_metric": best_selection,
        "selection_delta": delta,
        "late_best_acceptance_passed": _acceptance_passed(best),
        "reason": (
            "accepted_epoch_5_or_6_meaningfully_improved"
            if triggered
            else "no_accepted_meaningful_epoch_5_or_6_improvement"
        ),
    }


def configure_fusion_continuation(
    config: dict[str, Any],
    source: Path,
    *,
    minimum_epochs: int = 50,
    maximum_epochs: int = 100,
    early_stop_patience: int = 5,
    selection_min_delta: float = 2.0e-4,
) -> dict[str, Any]:
    if minimum_epochs < 50:
        raise ValueError("fusion continuation minimum_epochs must be at least 50")
    if maximum_epochs < minimum_epochs:
        raise ValueError("maximum_epochs must be at least minimum_epochs")
    if early_stop_patience < 1:
        raise ValueError("early_stop_patience must be positive")
    if selection_min_delta < 0:
        raise ValueError("selection_min_delta must be non-negative")
    phase = configure_fusion_recovery(config, source)
    phase.update(
        {
            "name": PHASE_NAME,
            "max_epochs": int(maximum_epochs),
            "minimum_epochs": int(minimum_epochs),
            "early_stop_patience": int(early_stop_patience),
            "selection_min_delta": float(selection_min_delta),
            # Epoch 6 ends near 1e-5 / 5e-8. Re-open a conservative learning
            # window without jumping back to the recovery run's original peak.
            "learning_rate": 2.5e-5,
            "fusion_learning_rate": 2.5e-5,
            "receiver_learning_rate": 1.25e-7,
            "minimum_learning_rate": 2.5e-6,
            "warmup_fraction": 0.01,
        }
    )
    return phase


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _guard_against_duplicate(pipeline_path: Path) -> None:
    if not pipeline_path.is_file():
        return
    existing = _load_json(pipeline_path)
    if existing.get("state") not in {"waiting_for_recovery", "running"}:
        return
    pid = int(existing.get("pid", -1))
    if pid != os.getpid() and _pid_is_alive(pid):
        raise RuntimeError(f"fusion continuation watcher is already active as PID {pid}")


def _wait_for_source(
    source_pipeline_path: Path,
    continuation_pipeline_path: Path,
    status: dict[str, Any],
    *,
    poll_seconds: float,
) -> dict[str, Any]:
    while True:
        source_pipeline = _load_json(source_pipeline_path)
        source_state = str(source_pipeline.get("state", "unknown"))
        status.update(
            {
                "state": "waiting_for_recovery",
                "source_state": source_state,
                "updated_unix": time.time(),
            }
        )
        atomic_json_dump(status, continuation_pipeline_path)
        if source_state in TERMINAL_SOURCE_STATES:
            return source_pipeline
        time.sleep(max(5.0, float(poll_seconds)))


def _build_contract(
    config: dict[str, Any],
    phase: dict[str, Any],
    source: Path,
    source_summary_path: Path,
    trigger: dict[str, Any],
) -> dict[str, Any]:
    root = Path(__file__).parents[1]
    payload = {
        "format": "cftn_text_v1_3_fusion_continuation_contract_v1",
        "revision_sha256": config["_meta"]["sha256"],
        "source_phase": SOURCE_PHASE_NAME,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": file_sha256(source),
        "source_summary": str(source_summary_path.resolve()),
        "source_summary_sha256": file_sha256(source_summary_path),
        "trigger": trigger,
        "phase": phase,
        "invariants": {
            "minimum_epochs": int(phase["minimum_epochs"]),
            "plateau_stopping_after_minimum": True,
            "accepted_checkpoint_source": True,
            "specialist_towers_frozen": True,
            "request_bridges_frozen": True,
            "return_bridges_frozen": True,
            "specialist_receivers_frozen": True,
            "wake_gates_frozen": True,
            "halt_gate_frozen": True,
            "trainable_components": ["message_fusion", "gpt_receivers"],
            "oracle_hard_routes": True,
            "zero_update_source_is_checkpoint_candidate": True,
        },
        "source_files": {
            name: file_sha256(root / relative)
            for name, relative in {
                "v1_3_model.py": "cftn_text/v1_3_model.py",
                "v1_3_training.py": "cftn_text/v1_3_training.py",
                "continue_v1_3_fusion.py": "tools/continue_v1_3_fusion.py",
            }.items()
        },
    }
    payload["contract_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conditionally continue V1.3 fusion training after the recovery probe"
    )
    parser.add_argument(
        "--config", default="G:/ctfn-text/config/v1_3_multi_specialist.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait-for-recovery", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-epochs", type=int, default=50)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--trigger-min-delta", type=float, default=2.0e-4)
    parser.add_argument("--selection-min-delta", type=float, default=2.0e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int)
    add_wandb_arguments(parser)
    args = parser.parse_args()

    config = load_v1_3_config(args.config)
    artifact_root = Path(config["paths"]["artifact_root"])
    source_artifact = artifact_root / SOURCE_PHASE_NAME
    source_pipeline_path = artifact_root / "fusion_recovery_pipeline.json"
    source_summary_path = source_artifact / "summary.json"
    source_metrics_path = source_artifact / "metrics.jsonl"
    pipeline_path = artifact_root / "fusion_continuation_pipeline.json"
    contract_path = artifact_root / "fusion_continuation_contract.json"
    continuation_artifact = artifact_root / PHASE_NAME
    _guard_against_duplicate(pipeline_path)

    started = time.time()
    status: dict[str, Any] = {
        "format": "cftn_text_v1_3_fusion_continuation_pipeline_v1",
        "state": "waiting_for_recovery",
        "pid": os.getpid(),
        "source_phase": SOURCE_PHASE_NAME,
        "phase": PHASE_NAME,
        "started_unix": started,
    }
    atomic_json_dump(status, pipeline_path)
    try:
        source_pipeline = _load_json(source_pipeline_path)
        if (
            str(source_pipeline.get("state")) not in TERMINAL_SOURCE_STATES
            and args.wait_for_recovery
        ):
            source_pipeline = _wait_for_source(
                source_pipeline_path,
                pipeline_path,
                status,
                poll_seconds=args.poll_seconds,
            )
        source_state = str(source_pipeline.get("state", "unknown"))
        if source_state != "completed":
            status.update(
                {
                    "state": "blocked_source_terminal",
                    "source_state": source_state,
                    "completed_unix": time.time(),
                }
            )
            atomic_json_dump(status, pipeline_path)
            print(json.dumps(status, indent=2))
            return

        source_summary = _load_json(source_summary_path)
        trigger = evaluate_late_improvement(
            _load_metrics(source_metrics_path),
            source_summary,
            minimum_delta=args.trigger_min_delta,
        )
        status["trigger"] = trigger
        if not trigger["triggered"]:
            status.update(
                {
                    "state": "skipped_no_late_improvement",
                    "completed_unix": time.time(),
                }
            )
            atomic_json_dump(status, pipeline_path)
            print(json.dumps(status, indent=2))
            return

        source = Path(str(source_summary["best_checkpoint"]))
        if not source.is_file():
            raise FileNotFoundError(f"accepted fusion checkpoint is missing: {source}")
        source_sha256 = file_sha256(source)
        if source_sha256 != str(source_summary.get("best_checkpoint_sha256")):
            raise RuntimeError("fusion recovery summary and best checkpoint disagree")
        if continuation_artifact.is_dir() and any(
            continuation_artifact.glob("*.pth")
        ) and not args.resume:
            raise RuntimeError(
                f"fusion continuation already has checkpoints; use --resume: "
                f"{continuation_artifact}"
            )

        phase = configure_fusion_continuation(
            config,
            source,
            minimum_epochs=args.minimum_epochs,
            maximum_epochs=args.maximum_epochs,
            early_stop_patience=args.early_stop_patience,
            selection_min_delta=args.selection_min_delta,
        )
        contract = _build_contract(
            config, phase, source, source_summary_path, trigger
        )
        atomic_json_dump(contract, contract_path)
        status.update(
            {
                "state": "running",
                "source_state": source_state,
                "source_checkpoint": str(source.resolve()),
                "source_checkpoint_sha256": source_sha256,
                "contract": str(contract_path.resolve()),
                "contract_sha256": contract["contract_sha256"],
                "training_started_unix": time.time(),
            }
        )
        atomic_json_dump(status, pipeline_path)
        result = train_integration_phase(
            config,
            PHASE_NAME,
            device_name=args.device,
            resume=args.resume,
            max_batches=args.max_batches,
            wandb_options=wandb_options_from_args(
                args, default_run_name="v1-3-oracle-hard-fusion-continuation"
            ),
        )
        status.update(
            {
                "state": result["state"],
                "result": result,
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        atomic_json_dump(status, pipeline_path)
    except BaseException as exc:
        status.update(
            {
                "state": "error",
                "error": repr(exc),
                "failed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        atomic_json_dump(status, pipeline_path)
        raise
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
