from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.watch_v2_progress import (
        _format_duration,
        _format_value,
        _gpu_status,
        _pid_running,
        _read_json,
        _tail_jsonl,
        _tail_text,
    )
except ModuleNotFoundError:  # Direct `python tools/check_cftn_heartbeat.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.watch_v2_progress import (
        _format_duration,
        _format_value,
        _gpu_status,
        _pid_running,
        _read_json,
        _tail_jsonl,
        _tail_text,
    )


V13_STAGE_DIRECTORIES = {
    "train_exact_string_specialist": "string_specialist",
    "train_single_specialist_capacity": "single_specialist_capacity",
    "train_dense_mixed_messages": "dense_mixed_messages",
    "train_dense_recurrent": "dense_recurrent",
    "train_supervised_soft_wake": "supervised_soft_wake",
    "train_hardened_wake": "hardened_wake",
    "evaluate_sealed_causal_suite": "sealed_evaluation",
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def collect_v1_3_snapshot(root: str | Path) -> dict[str, Any]:
    artifact = Path(root).expanduser().resolve()
    checked = time.time()
    pipeline = _read_json(artifact / "pipeline_status.json") or {}
    stage = pipeline.get("current_stage")
    directory = V13_STAGE_DIRECTORIES.get(str(stage))
    stage_root = artifact / directory if directory else None
    status_path = stage_root / "status.json" if stage_root else None
    status = _read_json(status_path) if status_path else None
    status_mtime = status_path.stat().st_mtime if status_path and status_path.is_file() else None
    metrics_rows = _tail_jsonl(stage_root / "metrics.jsonl", 1) if stage_root else []
    latest_metrics = metrics_rows[-1] if metrics_rows else None
    checkpoints = []
    if stage_root and stage_root.is_dir():
        checkpoints = sorted(
            stage_root.glob("checkpoint_epoch_*.pth"),
            key=lambda value: value.stat().st_mtime,
            reverse=True,
        )
    log_candidates = sorted(
        artifact.glob("pipeline*.stderr.log"),
        key=lambda value: value.stat().st_mtime,
        reverse=True,
    )
    stderr = _tail_text(log_candidates[0], 50) if log_candidates else ""
    metrics = dict((status or {}).get("metrics") or {})
    return {
        "checked_unix": checked,
        "root": str(artifact),
        "pipeline": pipeline,
        "pipeline_state": pipeline.get("state", "missing"),
        "pipeline_pid": pipeline.get("pid"),
        "pipeline_pid_running": _pid_running(pipeline.get("pid")),
        "stage": stage,
        "stage_index": pipeline.get("stage_index"),
        "stage_count": pipeline.get("stages_total"),
        "status": status,
        "status_age_seconds": checked - status_mtime if status_mtime else None,
        "stage_pid": (status or {}).get("pid"),
        "stage_pid_running": _pid_running((status or {}).get("pid")),
        "epoch": (status or {}).get("epoch"),
        "global_step": (status or {}).get("global_step"),
        "batch_completed": metrics.get("epoch_batch_completed"),
        "batch_total": metrics.get("epoch_batches_total"),
        "metrics": metrics,
        "latest_metrics": latest_metrics,
        "latest_checkpoint": str(checkpoints[0]) if checkpoints else None,
        "checkpoint_count": len(checkpoints),
        "gpu": _gpu_status(),
        "stderr_tail": stderr,
    }


def _v13_identifier(*values: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:18]


def classify_v1_3(
    snapshot: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    now: float | None = None,
    prime: bool = False,
    stale_after_seconds: float = 1800.0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checked = float(now if now is not None else time.time())
    old = dict(previous or {})
    remembered = set(str(value) for value in old.get("notified_event_ids", []))
    candidates: list[dict[str, str]] = []

    def add(identifier: str, level: str, text: str) -> None:
        candidates.append({"id": identifier, "level": level, "text": text})

    state = str(snapshot.get("pipeline_state", "missing"))
    stage = str(snapshot.get("stage") or "none")
    old_stage = old.get("last_stage")
    old_state = old.get("last_pipeline_state")
    if state in {"error", "failed", "failed_acceptance"}:
        add(
            "v13-terminal-" + _v13_identifier(state, stage, snapshot.get("stderr_tail")),
            "critical",
            f"V1.3 entered `{state}` in `{stage}`.",
        )
    elif state == "completed":
        add("v13-completed", "info", "The complete V1.3 pipeline finished.")
    elif old_state is not None and state != str(old_state):
        add(
            f"v13-state-{old_state}-to-{state}-{stage}",
            "info",
            f"V1.3 state changed from `{old_state}` to `{state}` in `{stage}`.",
        )

    if state == "running" and not snapshot.get("pipeline_pid_running"):
        add(
            "v13-pipeline-dead-" + _v13_identifier(stage, snapshot.get("pipeline_pid")),
            "critical",
            "V1.3 says running, but its pipeline PID is dead.",
        )
    if (
        state == "running"
        and snapshot.get("stage_pid") is not None
        and not snapshot.get("stage_pid_running")
    ):
        add(
            "v13-stage-dead-" + _v13_identifier(stage, snapshot.get("stage_pid")),
            "critical",
            f"V1.3 `{stage}` trainer PID is dead.",
        )
    if old_stage is not None and stage != str(old_stage):
        add(
            f"v13-stage-transition-{snapshot.get('stage_index')}-{stage}",
            "info",
            f"V1.3 moved from `{old_stage}` to stage {snapshot.get('stage_index')}/{snapshot.get('stage_count')} `{stage}`.",
        )
    elif old_stage is None and not prime:
        add(
            f"v13-monitor-baseline-{stage}",
            "info",
            f"V1.3 event monitoring started in `{stage}`.",
        )

    signature = _v13_identifier(
        stage,
        snapshot.get("epoch"),
        snapshot.get("global_step"),
        snapshot.get("batch_completed"),
    )
    last_progress = float(old.get("last_progress_unix", checked))
    if signature != old.get("last_progress_signature"):
        last_progress = checked
    age = max(
        checked - last_progress,
        float(snapshot.get("status_age_seconds") or 0.0),
    )
    if state == "running" and stage.startswith("train_") and age > stale_after_seconds:
        add(
            f"v13-stalled-{stage}-{signature}",
            "critical",
            f"V1.3 `{stage}` has not advanced for about {int(age // 60)} minutes.",
        )

    metrics = snapshot.get("metrics") or {}
    train = metrics.get("train") if isinstance(metrics.get("train"), dict) else {}
    learning_rates = metrics.get("learning_rates") or {}
    if stage == "train_hardened_wake" and snapshot.get("global_step", 0):
        if float(train.get("routing_calibration_only", -1.0)) != 1.0:
            add(
                "v13-routing-objective-missing",
                "critical",
                "V1.3 hardening is not marked routing_calibration_only=1.",
            )
        if float(train.get("auxiliary_step", -1.0)) != 0.0:
            add(
                "v13-auxiliary-enabled",
                "critical",
                "V1.3 hardening auxiliary/task utility unexpectedly became active.",
            )
        total = train.get("total_loss")
        wake = train.get("wake_loss")
        if total is not None and wake is not None and not math.isclose(
            float(total), float(wake), rel_tol=1e-6, abs_tol=1e-8
        ):
            add(
                "v13-loss-contract-" + _v13_identifier(total, wake),
                "critical",
                f"V1.3 total loss {_format_value(total)} differs from wake loss {_format_value(wake)}.",
            )
        for group, value in learning_rates.items():
            if float(value) > 5.0e-7 + 1.0e-15:
                add(
                    f"v13-lr-{group}-{snapshot.get('epoch')}",
                    "critical",
                    f"V1.3 hardening `{group}` LR {float(value):.3e} exceeds 5e-7.",
                )

    latest = snapshot.get("latest_metrics") or {}
    validation = latest.get("validation") if isinstance(latest, dict) else None
    try:
        validation_epoch = int(latest.get("epoch")) if validation else None
    except (TypeError, ValueError):
        validation_epoch = None
    old_validation_epoch = int(old.get("last_validation_epoch", 0) or 0)
    if validation_epoch is not None and validation_epoch > old_validation_epoch:
        add(
            f"v13-validation-{stage}-{validation_epoch}",
            "info",
            f"V1.3 `{stage}` completed validation epoch {validation_epoch}.",
        )
        if stage == "train_hardened_wake":
            thresholds = {
                "exact_required_set_accuracy": (0.90, False),
                "wake_precision": (0.90, False),
                "wake_recall": (0.95, False),
                "pure_language_false_wake_rate": (0.05, True),
            }
            for key, (threshold, maximum) in thresholds.items():
                if key not in validation:
                    continue
                value = float(validation[key])
                failed = value > threshold if maximum else value < threshold
                if failed:
                    add(
                        f"v13-guard-{key}-{validation_epoch}",
                        "critical",
                        f"V1.3 epoch {validation_epoch} `{key}`={value:.4f} failed its {threshold:.2f} guard.",
                    )
            names = latest.get("trainable_parameter_names") or []
            unexpected = [
                str(name) for name in names if not str(name).startswith("wake_gates.")
            ]
            if unexpected:
                add(
                    f"v13-trainables-{validation_epoch}",
                    "critical",
                    "V1.3 hardening includes non-wake-gate trainables.",
                )
        old_validation_epoch = validation_epoch

    for key, value in {**train, **(validation or {})}.items():
        if "loss" in str(key).lower() and isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                add(
                    f"v13-nonfinite-{key}-{snapshot.get('epoch')}",
                    "critical",
                    f"V1.3 `{key}` became non-finite.",
                )

    new_events = [event for event in candidates if event["id"] not in remembered]
    if prime:
        new_events = [event for event in new_events if event["level"] == "critical"]
    remembered.update(event["id"] for event in candidates)
    next_state = {
        "format": "cftn_text_v1_3_event_monitor_state_v1",
        "updated_unix": checked,
        "last_stage": stage,
        "last_pipeline_state": state,
        "last_progress_signature": signature,
        "last_progress_unix": last_progress,
        "last_validation_epoch": old_validation_epoch,
        "notified_event_ids": sorted(remembered)[-150:],
    }
    return new_events, next_state


def render_v1_3(snapshot: dict[str, Any], events: list[dict[str, str]]) -> str:
    status = snapshot.get("status") or {}
    metrics = snapshot.get("metrics") or {}
    train = metrics.get("train") if isinstance(metrics.get("train"), dict) else {}
    reasons = "\n".join(
        f"- [{event['level'].upper()}] {event['text']}" for event in events
    )
    lines = [
        "## V1.3 heartbeat event",
        "",
        reasons,
        "",
        f"- Pipeline: `{snapshot.get('pipeline_state')}`",
        f"- Stage: {snapshot.get('stage_index')}/{snapshot.get('stage_count')} `{snapshot.get('stage')}`",
    ]
    if snapshot.get("epoch") is not None:
        lines.append(f"- Epoch: {snapshot.get('epoch')}")
    if snapshot.get("batch_completed") is not None and snapshot.get("batch_total"):
        percent = 100.0 * float(snapshot["batch_completed"]) / float(snapshot["batch_total"])
        lines.append(
            f"- Progress: {int(snapshot['batch_completed']):,}/{int(snapshot['batch_total']):,} ({percent:.1f}%)"
        )
    if snapshot.get("global_step") is not None:
        lines.append(f"- Global step: {int(snapshot['global_step']):,}")
    if status.get("elapsed_seconds") is not None:
        lines.append(f"- Stage elapsed: {_format_duration(status['elapsed_seconds'])}")
    if train:
        selected = (
            "total_loss",
            "wake_loss",
            "gpt_loss",
            "specialist_loss",
            "routing_calibration_only",
            "auxiliary_step",
        )
        values = ", ".join(
            f"{key}={_format_value(train[key])}" for key in selected if key in train
        )
        lines.append(f"- Rolling training averages: {values}")
    latest = snapshot.get("latest_metrics") or {}
    validation = latest.get("validation") if isinstance(latest, dict) else None
    if isinstance(validation, dict):
        names = (
            "gpt_teacher_forced_sequence_accuracy",
            "gpt_teacher_forced_token_accuracy",
            "exact_required_set_accuracy",
            "wake_precision",
            "wake_recall",
            "wake_f1",
            "pure_language_false_wake_rate",
            "causal_message_loss_gap",
        )
        values = ", ".join(
            f"{key}={_format_value(validation[key])}" for key in names if key in validation
        )
        lines.append(f"- Validation epoch {latest.get('epoch')}: {values}")
    lines.append(
        f"- Processes: pipeline={'alive' if snapshot.get('pipeline_pid_running') else 'dead'}, "
        f"stage={'alive' if snapshot.get('stage_pid_running') else 'dead'}"
    )
    for gpu in snapshot.get("gpu") or []:
        lines.append(
            f"- GPU {gpu['index']}: {gpu['utilization_percent']}%, "
            f"{gpu['memory_used_mib']:,}/{gpu['memory_total_mib']:,} MiB, "
            f"{gpu['temperature_c']}°C"
        )
    if snapshot.get("latest_checkpoint"):
        lines.append(
            f"- Checkpoints: {snapshot.get('checkpoint_count')}; latest `{Path(snapshot['latest_checkpoint']).name}`"
        )
    return "\n".join(lines)


def _run_v2_checker(args: argparse.Namespace) -> dict[str, Any]:
    checker = Path(__file__).resolve().with_name("check_v2_heartbeat.py")
    command = [
        sys.executable,
        str(checker),
        "--identity-file",
        args.identity_file,
        "--state-file",
        args.v2_state_file,
        "--stale-after-seconds",
        str(args.stale_after_seconds),
    ]
    if args.prime:
        command.append("--prime")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        return {
            "decision": "NOTIFY",
            "message": "V2 checker process failed: " + (result.stderr or result.stdout)[-3000:],
        }
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "decision": "NOTIFY",
            "message": f"V2 checker emitted invalid JSON: {exc}",
        }
    return value if isinstance(value, dict) else {
        "decision": "NOTIFY",
        "message": "V2 checker did not return an object.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Event-only monitor for active CFTN runs")
    parser.add_argument(
        "--identity-file",
        default=str(Path.home() / ".ssh" / "id_ed25519_runpod_cftn"),
    )
    parser.add_argument(
        "--v2-state-file",
        default=str(Path.home() / ".cftn" / "v2_heartbeat_state.json"),
    )
    parser.add_argument(
        "--v1-3-root",
        default=r"G:\ctfn-text\artifacts\v1_3_multi_specialist",
    )
    parser.add_argument(
        "--v1-3-state-file",
        default=str(Path.home() / ".cftn" / "v1_3_heartbeat_state.json"),
    )
    parser.add_argument("--stale-after-seconds", type=float, default=1800.0)
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    v2_result = _run_v2_checker(args)
    v13_state_path = Path(args.v1_3_state_file).expanduser().resolve()
    previous = _read_json(v13_state_path) or {}
    try:
        v13_snapshot = collect_v1_3_snapshot(args.v1_3_root)
        v13_events, next_state = classify_v1_3(
            v13_snapshot,
            previous,
            prime=args.prime,
            stale_after_seconds=args.stale_after_seconds,
        )
        _atomic_json(v13_state_path, next_state)
        v13_result = {
            "decision": "NOTIFY" if v13_events else "DONT_NOTIFY",
            "message": render_v1_3(v13_snapshot, v13_events) if v13_events else "",
        }
    except BaseException as exc:
        fingerprint = _v13_identifier(type(exc).__name__, str(exc))
        repeated = previous.get("last_monitor_error") == fingerprint
        next_state = dict(previous)
        next_state["last_monitor_error"] = fingerprint
        next_state["updated_unix"] = time.time()
        _atomic_json(v13_state_path, next_state)
        v13_result = {
            "decision": "DONT_NOTIFY" if repeated else "NOTIFY",
            "message": f"V1.3 checker failed: {type(exc).__name__}: {exc}",
        }

    notifications = [
        str(value.get("message", ""))
        for value in (v2_result, v13_result)
        if value.get("decision") == "NOTIFY"
    ]
    result = {
        "decision": "NOTIFY" if notifications else "DONT_NOTIFY",
        "message": (
            "\n\n".join(value for value in notifications if value)
            if notifications
            else "V1.3 and V2 are healthy with no new reportable event."
        ),
        "components": {
            "v1_3": v13_result.get("decision"),
            "v2": v2_result.get("decision"),
        },
    }
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
