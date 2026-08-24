from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


STAGE_DIRECTORIES = {
    "train_math": "math",
    "select_math_checkpoint": "math_checkpoint_selection",
    "evaluate_math": "evaluation_math_v2",
    "train_learned_dispatcher": "learned_dispatcher_v2",
    "calibrate_frozen_gpt_language": "gpt_language_calibration",
    "train_exact_string_specialist": "string_specialist",
    "seal_native_specialists": "native_specialist_evaluation",
    "train_single_specialist_capacity": "single_specialist_capacity",
    "train_dense_mixed_messages": "dense_mixed_messages",
    "train_dense_recurrent": "dense_recurrent",
    "train_supervised_soft_wake": "supervised_soft_wake",
    "evaluate_zero_update_hard_baseline": "hard_transition_baseline",
    "train_hardened_wake": "hardened_wake",
    "evaluate_native_typed_dispatch": "native_dispatch_evaluation",
    "evaluate_sealed_causal_suite": "sealed_evaluation",
}

EPOCH_LIMITS = {
    "train_math": 100,
    "train_learned_dispatcher": 8,
    "train_exact_string_specialist": 30,
    "train_single_specialist_capacity": 8,
    "train_dense_mixed_messages": 12,
    "train_dense_recurrent": 12,
    "train_supervised_soft_wake": 10,
    "train_hardened_wake": 10,
}

IMPORTANT_VALIDATION_KEYS = (
    "loss",
    "teacher_forced_sequence_accuracy",
    "teacher_forced_token_accuracy",
    "gpt_teacher_forced_sequence_accuracy",
    "gpt_teacher_forced_token_accuracy",
    "exact_required_set_accuracy",
    "wake_precision",
    "wake_recall",
    "wake_f1",
    "pure_language_false_wake_rate",
    "causal_message_loss_gap",
    "accuracy",
    "coverage_at_threshold",
    "correct_and_accepted_rate",
    "dispatch_plan_valid_rate",
    "dispatch_completion_rate",
    "valid_rate",
    "trace_exact_rate",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _tail_jsonl(path: Path, count: int = 1) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=max(1, int(count)))
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return list(rows)


def _tail_text(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=max(1, int(lines)))).strip()
    except OSError:
        return ""


def _pid_running(pid: Any) -> bool:
    if os.name == "nt":
        try:
            import ctypes

            process_id = int(pid)
            if process_id <= 0:
                return False
            query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                query_limited_information, False, process_id
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            # Access denied still proves that the process exists.
            return int(ctypes.windll.kernel32.GetLastError()) == 5
        except (AttributeError, OSError, SystemError, TypeError, ValueError):
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, SystemError, TypeError, ValueError):
        return False


def _process_lines() -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,stat,etime,time,%cpu,%mem,args"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    markers = (
        "run_v2.py",
        "tools.run_v2_experiment",
        "tools.train_math_tower",
        "tools.train_v1_3",
        "tools.evaluate_",
        "tools.prepare_",
        "tools.select_v2",
        "tools.assess_v2",
        "tools.assemble_v2",
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(marker in line for marker in markers)
        and "watch_v2_progress" not in line
    ]


def _gpu_status() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode:
        return []
    output = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 7:
            continue
        try:
            output.append(
                {
                    "index": int(values[0]),
                    "name": values[1],
                    "utilization_percent": int(values[2]),
                    "memory_used_mib": int(values[3]),
                    "memory_total_mib": int(values[4]),
                    "temperature_c": int(values[5]),
                    "power_w": float(values[6]),
                }
            )
        except ValueError:
            continue
    return output


def _format_duration(seconds: Any) -> str:
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "unknown"
    days, value = divmod(value, 86400)
    hours, value = divmod(value, 3600)
    minutes, secs = divmod(value, 60)
    pieces = []
    if days:
        pieces.append(f"{days}d")
    if hours or days:
        pieces.append(f"{hours}h")
    if minutes or hours or days:
        pieces.append(f"{minutes}m")
    pieces.append(f"{secs}s")
    return " ".join(pieces)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if 0 < abs(value) < 0.0001:
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not isinstance(value, dict):
        return output
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, dict):
            output.update(_flatten(item, name))
        elif isinstance(item, (bool, int, float, str)) and item is not None:
            output[name] = item
    return output


def _stage_status_path(
    stage: str | None,
    artifact_root: Path,
    data_root: Path,
    multi_data_root: Path,
) -> Path | None:
    if stage == "prepare_data":
        return data_root / "prepare_status.json"
    if stage == "prepare_multi_specialist_data":
        return multi_data_root / "prepare_status.json"
    directory = STAGE_DIRECTORIES.get(str(stage))
    return artifact_root / directory / "status.json" if directory else None


def collect_snapshot(
    artifact_root: str | Path,
    *,
    data_root: str | Path,
    multi_data_root: str | Path,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    multi_data = Path(multi_data_root).expanduser().resolve()
    checked_unix = time.time()
    pipeline = _read_json(root / "pipeline_state.json") or {}
    stage = pipeline.get("current_stage")
    stage_record = (pipeline.get("stages") or {}).get(stage, {}) if stage else {}
    status_path = _stage_status_path(stage, root, data, multi_data)
    status = _read_json(status_path) if status_path else None
    status_mtime = status_path.stat().st_mtime if status_path and status_path.is_file() else None
    stage_started = stage_record.get("started_unix")
    status_fresh = bool(
        status_mtime is not None
        and (stage_started is None or status_mtime >= float(stage_started) - 2.0)
    )
    stage_dir_name = STAGE_DIRECTORIES.get(str(stage))
    stage_dir = root / stage_dir_name if stage_dir_name else None
    metrics_rows = _tail_jsonl(stage_dir / "metrics.jsonl", 1) if stage_dir else []
    latest_metrics = metrics_rows[-1] if metrics_rows else None
    wandb = _read_json(stage_dir / "wandb_run.json") if stage_dir else None
    checkpoints = []
    if stage_dir and stage_dir.is_dir():
        checkpoints = sorted(
            [*stage_dir.glob("*.pth"), *stage_dir.glob("*.pt")],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    metrics = dict((status or {}).get("metrics") or {})
    batch_completed = metrics.get("epoch_batch_completed")
    batch_total = metrics.get("epoch_batches_total")
    if batch_completed is None and status:
        batch_completed = status.get("completed")
        batch_total = status.get("total")

    batch_rate = metrics.get("steps_per_second")
    if (
        batch_rate is None
        and previous
        and previous.get("stage") == stage
        and previous.get("epoch") == (status or {}).get("epoch")
        and previous.get("batch_completed") is not None
        and batch_completed is not None
    ):
        delta_batches = int(batch_completed) - int(previous["batch_completed"])
        delta_time = checked_unix - float(previous.get("checked_unix", checked_unix))
        if delta_batches > 0 and delta_time > 0:
            batch_rate = delta_batches / delta_time

    epoch_eta = None
    if batch_rate and batch_total is not None and batch_completed is not None:
        epoch_eta = max(0, int(batch_total) - int(batch_completed)) / float(batch_rate)

    stderr_path = root / "pipeline_logs" / f"{stage}.stderr.log" if stage else None
    stderr_tail = _tail_text(stderr_path, 50) if stderr_path else ""
    stderr_mtime = (
        stderr_path.stat().st_mtime
        if stderr_path is not None and stderr_path.is_file()
        else None
    )
    error_text = ""
    if pipeline.get("state") == "error" or (
        status_fresh and (status or {}).get("state") == "error"
    ):
        error_text = stderr_tail

    pipeline_pid = pipeline.get("pid")
    stage_pid = (status or {}).get("pid") if status_fresh else None
    return {
        "checked_unix": checked_unix,
        "artifact_root": str(root),
        "pipeline": pipeline,
        "pipeline_state": pipeline.get("state", "missing"),
        "pipeline_pid": pipeline_pid,
        "pipeline_pid_running": _pid_running(pipeline_pid),
        "stage": stage,
        "stage_index": pipeline.get("current_stage_index"),
        "stage_count": pipeline.get("stage_count"),
        "stage_record": stage_record,
        "status": status,
        "status_path": str(status_path) if status_path else None,
        "status_fresh": status_fresh,
        "status_age_seconds": checked_unix - status_mtime if status_mtime else None,
        "stage_pid": stage_pid,
        "stage_pid_running": _pid_running(stage_pid),
        "epoch": (status or {}).get("epoch"),
        "epoch_limit": EPOCH_LIMITS.get(str(stage)),
        "global_step": (status or {}).get("global_step"),
        "batch_completed": batch_completed,
        "batch_total": batch_total,
        "batch_rate": batch_rate,
        "epoch_eta_seconds": epoch_eta,
        "metrics": metrics,
        "latest_metrics": latest_metrics,
        "wandb": wandb,
        "latest_checkpoint": str(checkpoints[0]) if checkpoints else None,
        "checkpoint_count": len(checkpoints),
        "process_lines": _process_lines(),
        "gpu": _gpu_status(),
        "stderr_tail": stderr_tail,
        "stderr_mtime": stderr_mtime,
        "error_text": error_text,
    }


def _metric_rows(metrics: dict[str, Any]) -> list[tuple[str, Any]]:
    train = metrics.get("train")
    if isinstance(train, dict):
        return list(train.items())[:18]
    names = (
        "train_loss_so_far",
        "learning_rate",
        "steps_per_second",
        "examples_per_second",
        "remaining_steps_to_max_epochs",
        "eta_seconds_to_max_epochs_excluding_validation",
    )
    return [(name, metrics[name]) for name in names if name in metrics]


def render_markdown(snapshot: dict[str, Any]) -> str:
    pipeline = snapshot["pipeline"]
    status = snapshot.get("status") or {}
    metrics = snapshot.get("metrics") or {}
    state = str(snapshot["pipeline_state"]).upper()
    stage = snapshot.get("stage") or "none"
    stage_position = (
        f"{snapshot.get('stage_index')}/{snapshot.get('stage_count')}"
        if snapshot.get("stage_index") is not None
        else "unknown"
    )
    lines = [
        f"# CFTN V2 progress — {datetime.fromtimestamp(snapshot['checked_unix']).isoformat(timespec='seconds')}",
        "",
        f"**Pipeline:** {state}  ",
        f"**Stage:** {stage_position} — `{stage}`  ",
    ]

    if snapshot.get("epoch") is not None:
        epoch_text = str(snapshot["epoch"])
        if snapshot.get("epoch_limit") is not None:
            epoch_text += f"/{snapshot['epoch_limit']}"
        lines.append(f"**Epoch:** {epoch_text}  ")
    completed = snapshot.get("batch_completed")
    total = snapshot.get("batch_total")
    if completed is not None and total:
        percent = 100.0 * float(completed) / max(1.0, float(total))
        lines.append(
            f"**Progress:** {int(completed):,}/{int(total):,} batches/examples ({percent:.2f}%)  "
        )
    if snapshot.get("global_step") is not None:
        lines.append(f"**Global step:** {int(snapshot['global_step']):,}  ")
    if status.get("elapsed_seconds") is not None:
        lines.append(f"**Stage elapsed:** {_format_duration(status['elapsed_seconds'])}  ")
    if snapshot.get("epoch_eta_seconds") is not None:
        lines.append(
            f"**ETA to current epoch/split boundary:** {_format_duration(snapshot['epoch_eta_seconds'])}  "
        )
    if metrics.get("eta_seconds_to_max_epochs_excluding_validation") is not None:
        lines.append(
            "**Training ETA excluding validation:** "
            f"{_format_duration(metrics['eta_seconds_to_max_epochs_excluding_validation'])}  "
        )
    if snapshot.get("status_path"):
        freshness = "current attempt" if snapshot.get("status_fresh") else "stale/previous attempt"
        age = _format_duration(snapshot.get("status_age_seconds"))
        lines.append(f"**Status:** {freshness}; updated {age} ago  ")

    train_rows = _metric_rows(metrics)
    if train_rows:
        lines.extend(["", "## Rolling training metrics", "", "| Metric | Value |", "|---|---:|"])
        for key, value in train_rows:
            lines.append(f"| `{key}` | {_format_value(value)} |")
        lines.extend(
            ["", "These are cumulative rolling training averages, not validation accuracy."]
        )

    latest = snapshot.get("latest_metrics") or {}
    validation = latest.get("validation")
    if not isinstance(validation, dict) and isinstance(metrics.get("validation"), dict):
        validation = metrics["validation"]
    flattened_validation = _flatten(validation or {})
    selected_validation = [
        (key, value)
        for key, value in flattened_validation.items()
        if key.split("/")[-1] in IMPORTANT_VALIDATION_KEYS
    ][:20]
    if selected_validation:
        lines.extend(["", "## Latest completed validation", "", "| Metric | Value |", "|---|---:|"])
        for key, value in selected_validation:
            lines.append(f"| `{key}` | {_format_value(value)} |")
        for name in ("epoch", "global_step", "selection_metric", "best_metric", "patience"):
            if name in latest:
                lines.append(f"| `{name}` | {_format_value(latest[name])} |")
    elif str(stage).startswith("train_"):
        lines.extend(["", "**Validation:** no completed validation row yet."])

    checkpoint = snapshot.get("latest_checkpoint")
    lines.extend(["", "## Runtime health", ""])
    lines.append(
        f"- Pipeline PID `{snapshot.get('pipeline_pid')}`: "
        f"{'alive' if snapshot.get('pipeline_pid_running') else 'not running'}"
    )
    if snapshot.get("stage_pid") is not None:
        lines.append(
            f"- Stage PID `{snapshot['stage_pid']}`: "
            f"{'alive' if snapshot.get('stage_pid_running') else 'not running'}"
        )
    if checkpoint:
        lines.append(
            f"- Checkpoints in current stage: {snapshot['checkpoint_count']}; latest `{Path(checkpoint).name}`"
        )
    else:
        lines.append("- Current-stage checkpoint: none yet")
    for gpu in snapshot.get("gpu") or []:
        lines.append(
            f"- GPU {gpu['index']} `{gpu['name']}`: {gpu['utilization_percent']}%, "
            f"{gpu['memory_used_mib']:,}/{gpu['memory_total_mib']:,} MiB, "
            f"{gpu['temperature_c']}°C, {gpu['power_w']:.1f} W"
        )
    wandb = snapshot.get("wandb") or {}
    if wandb.get("url"):
        lines.append(f"- [Open W&B run]({wandb['url']})")

    if snapshot.get("process_lines"):
        lines.extend(["", "<details><summary>Exact pipeline processes</summary>", "", "```text"])
        lines.extend(snapshot["process_lines"])
        lines.extend(["```", "", "</details>"])

    if snapshot.get("error_text"):
        lines.extend(
            [
                "",
                "## ERROR — newest stderr tail",
                "",
                "```text",
                snapshot["error_text"][-8000:],
                "```",
            ]
        )
    elif state == "RUNNING" and not snapshot.get("pipeline_pid_running"):
        lines.extend(
            [
                "",
                "## WARNING",
                "",
                "Pipeline state says running, but its recorded PID is not alive.",
            ]
        )
    elif state == "RUNNING" and snapshot.get("status_fresh") and snapshot.get("status_age_seconds", 0) > 900:
        lines.extend(
            [
                "",
                "## WARNING",
                "",
                "The current stage status has not advanced for more than 15 minutes.",
            ]
        )

    if pipeline.get("error"):
        lines.extend(["", f"**Pipeline error:** `{pipeline['error']}`"])
    return "\n".join(lines)


def watch_v2_progress(
    *,
    artifact_root: str | Path | None = None,
    data_root: str | Path | None = None,
    multi_data_root: str | Path | None = None,
    interval_seconds: float = 60.0,
    once: bool = False,
) -> dict[str, Any]:
    artifact = artifact_root or os.environ.get(
        "CFTN_ARTIFACT_ROOT",
        "/workspace/cftn-text/artifacts/v2_broad_math_400k_r3",
    )
    data = data_root or os.environ.get(
        "CFTN_DATA_ROOT",
        "/workspace/cftn-text/data/v2_broad_math_400k_r3",
    )
    multi_data = multi_data_root or os.environ.get(
        "CFTN_V2_MULTI_DATA_ROOT",
        "/workspace/cftn-text/data/v2_multi_specialist_r2",
    )
    previous: dict[str, Any] | None = None
    notebook = False
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, clear_output, display

        notebook = get_ipython() is not None
    except ImportError:
        notebook = False

    while True:
        snapshot = collect_snapshot(
            artifact,
            data_root=data,
            multi_data_root=multi_data,
            previous=previous,
        )
        rendered = render_markdown(snapshot)
        if notebook:
            clear_output(wait=True)
            display(Markdown(rendered))
        else:
            print("\033[2J\033[H" + rendered, flush=True)
        state = snapshot["pipeline_state"]
        if once or state in {"completed", "error"}:
            return snapshot
        previous = snapshot
        try:
            time.sleep(max(5.0, float(interval_seconds)))
        except KeyboardInterrupt:
            return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a local V2 pipeline")
    parser.add_argument("--artifact-root")
    parser.add_argument("--data-root")
    parser.add_argument("--multi-data-root")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    watch_v2_progress(
        artifact_root=args.artifact_root,
        data_root=args.data_root,
        multi_data_root=args.multi_data_root,
        interval_seconds=args.interval_seconds,
        once=args.once,
    )


if __name__ == "__main__":
    main()
