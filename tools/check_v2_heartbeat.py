from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from tools.watch_v2_progress import render_markdown
except ModuleNotFoundError:  # Direct `python tools/check_v2_heartbeat.py` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.watch_v2_progress import render_markdown


DEFAULT_HOST = "216.243.220.243"
DEFAULT_PORT = 18480
DEFAULT_USER = "root"
DEFAULT_REMOTE_REPOSITORY = "/workspace/CFTN-MATHS"
DEFAULT_ARTIFACT_ROOT = (
    "/workspace/cftn-text/artifacts/v2_broad_math_400k_r3"
)
DEFAULT_DATA_ROOT = "/workspace/cftn-text/data/v2_broad_math_400k_r3"
DEFAULT_MULTI_DATA_ROOT = (
    "/workspace/cftn-text/data/v2_multi_specialist_r2"
)
STATE_FORMAT = "cftn_text_v2_heartbeat_state_v1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _fingerprint(*values: Any) -> str:
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


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


def _latest_validation(snapshot: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    latest = snapshot.get("latest_metrics") or {}
    validation = latest.get("validation")
    epoch = latest.get("epoch")
    if not isinstance(validation, dict):
        metrics = snapshot.get("metrics") or {}
        candidate = metrics.get("validation")
        if isinstance(candidate, dict):
            validation = candidate
            epoch = metrics.get("epoch", snapshot.get("epoch"))
    try:
        parsed_epoch = int(epoch) if epoch is not None else None
    except (TypeError, ValueError):
        parsed_epoch = None
    return parsed_epoch, validation if isinstance(validation, dict) else {}


def _progress_signature(snapshot: dict[str, Any]) -> str:
    return _fingerprint(
        snapshot.get("stage"),
        snapshot.get("epoch"),
        snapshot.get("global_step"),
        snapshot.get("batch_completed"),
        (snapshot.get("status") or {}).get("updated_unix"),
    )


def _event(identifier: str, level: str, text: str) -> dict[str, str]:
    return {"id": identifier, "level": level, "text": text}


def _hardening_events(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    if snapshot.get("stage") != "train_hardened_wake":
        return []
    events: list[dict[str, str]] = []
    metrics = snapshot.get("metrics") or {}
    latest = snapshot.get("latest_metrics") or {}
    policy = metrics.get("hardening_policy") or latest.get("hardening_policy")
    epoch, validation = _latest_validation(snapshot)
    event_epoch = epoch if epoch is not None else snapshot.get("epoch", 0)

    if snapshot.get("global_step", 0) and not isinstance(policy, dict):
        events.append(
            _event(
                "hardening-policy-missing",
                "critical",
                "Stage 10 is running without the registered wake-only hardening policy metadata.",
            )
        )
    elif isinstance(policy, dict):
        expected = {
            "objective": "wake_required_set_only",
            "trainable_components": ["wake_gates"],
            "halt_gate_trainable": False,
            "hard_halt_enabled": False,
            "conditional_specialist_execution": True,
        }
        for key, expected_value in expected.items():
            if policy.get(key) != expected_value:
                events.append(
                    _event(
                        f"hardening-policy-{key}-{_fingerprint(policy.get(key))}",
                        "critical",
                        f"Stage 10 contract violation: `{key}` is {policy.get(key)!r}, expected {expected_value!r}.",
                    )
                )

    learning_rates = metrics.get("learning_rates") or latest.get("learning_rates") or {}
    if isinstance(learning_rates, dict):
        for group, raw_value in learning_rates.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 5.0e-7 + 1.0e-15:
                events.append(
                    _event(
                        f"hardening-lr-{group}-{event_epoch}",
                        "critical",
                        f"Stage 10 `{group}` learning rate is {value:.3e}, above 5e-7.",
                    )
                )

    group_names = latest.get("optimizer_group_names")
    if isinstance(group_names, list) and group_names != ["gates"]:
        events.append(
            _event(
                f"hardening-groups-{event_epoch}-{_fingerprint(group_names)}",
                "critical",
                f"Stage 10 optimizer groups are {group_names!r}; only `gates` is allowed.",
            )
        )
    trainable_names = latest.get("trainable_parameter_names")
    if isinstance(trainable_names, list):
        unexpected = [
            str(name)
            for name in trainable_names
            if not str(name).startswith("wake_gates.")
        ]
        if unexpected:
            events.append(
                _event(
                    f"hardening-trainables-{event_epoch}-{_fingerprint(unexpected)}",
                    "critical",
                    "Stage 10 has non-wake-gate trainable parameters: "
                    + ", ".join(unexpected[:8]),
                )
            )

    thresholds = {
        "pure_language_false_wake_rate": (0.05, "maximum"),
        "exact_required_set_accuracy": (0.90, "minimum"),
        "wake_precision": (0.90, "minimum"),
        "wake_recall": (0.95, "minimum"),
    }
    if validation and epoch is not None:
        for key, (threshold, direction) in thresholds.items():
            if key not in validation:
                continue
            value = float(validation[key])
            failed = value > threshold if direction == "maximum" else value < threshold
            if failed:
                comparator = "above" if direction == "maximum" else "below"
                events.append(
                    _event(
                        f"hardening-guard-{key}-{epoch}",
                        "critical",
                        f"Stage 10 epoch {epoch} `{key}` is {value:.4f}, {comparator} its {threshold:.2f} guard.",
                    )
                )
        for key in ("all_open_rate", "all_closed_rate"):
            if float(validation.get(key, 0.0)) >= 0.95:
                events.append(
                    _event(
                        f"hardening-collapse-{key}-{epoch}",
                        "critical",
                        f"Stage 10 epoch {epoch} appears collapsed: `{key}`={float(validation[key]):.4f}.",
                    )
                )
    return events


def classify_snapshot(
    snapshot: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    now: float | None = None,
    stale_after_seconds: float = 1800.0,
    epoch_milestone: int = 10,
    prime: bool = False,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return only new noteworthy events and the next persistent state."""

    checked = float(now if now is not None else time.time())
    old = dict(previous or {})
    remembered = set(str(value) for value in old.get("notified_event_ids", []))
    candidates: list[dict[str, str]] = []
    pipeline_state = str(snapshot.get("pipeline_state", "missing"))
    stage = str(snapshot.get("stage") or "none")
    old_stage = old.get("last_stage")
    old_pipeline_state = old.get("last_pipeline_state")

    if pipeline_state in {"error", "failed", "failed_acceptance"}:
        pipeline = snapshot.get("pipeline") or {}
        candidates.append(
            _event(
                "pipeline-error-"
                + _fingerprint(stage, pipeline.get("failed_unix"), pipeline.get("error")),
                "critical",
                f"V2 entered terminal state `{pipeline_state}` in stage `{stage}`.",
            )
        )
    elif pipeline_state == "completed":
        candidates.append(
            _event(
                "pipeline-completed-" + _fingerprint((snapshot.get("pipeline") or {}).get("completed_unix")),
                "info",
                "The complete V2 pipeline finished.",
            )
        )
    elif old_pipeline_state is not None and pipeline_state != str(old_pipeline_state):
        candidates.append(
            _event(
                f"pipeline-state-{old_pipeline_state}-to-{pipeline_state}-{stage}",
                "info",
                f"V2 pipeline state changed from `{old_pipeline_state}` to `{pipeline_state}` in `{stage}`.",
            )
        )

    if pipeline_state == "running" and not snapshot.get("pipeline_pid_running"):
        candidates.append(
            _event(
                "pipeline-process-dead-"
                + _fingerprint(stage, (snapshot.get("stage_record") or {}).get("started_unix")),
                "critical",
                f"V2 says it is running in `{stage}`, but the recorded pipeline PID is dead.",
            )
        )
    if (
        pipeline_state == "running"
        and snapshot.get("stage_pid") is not None
        and snapshot.get("status_fresh")
        and not snapshot.get("stage_pid_running")
    ):
        candidates.append(
            _event(
                "stage-process-dead-"
                + _fingerprint(stage, snapshot.get("stage_pid"), snapshot.get("epoch")),
                "critical",
                f"The recorded `{stage}` trainer PID is dead while its status still says current.",
            )
        )

    if old_stage is not None and stage != str(old_stage):
        candidates.append(
            _event(
                f"stage-transition-{snapshot.get('stage_index')}-{stage}",
                "info",
                f"V2 moved from `{old_stage}` to stage {snapshot.get('stage_index')}/{snapshot.get('stage_count')} `{stage}`.",
            )
        )
    elif old_stage is None and not prime:
        candidates.append(
            _event(
                f"monitor-baseline-{snapshot.get('stage_index')}-{stage}",
                "info",
                f"V2 heartbeat monitoring started at stage {snapshot.get('stage_index')}/{snapshot.get('stage_count')} `{stage}`.",
            )
        )

    signature = _progress_signature(snapshot)
    previous_signature = old.get("last_progress_signature")
    last_progress_unix = float(old.get("last_progress_unix", checked))
    if signature != previous_signature:
        last_progress_unix = checked
    status_age = snapshot.get("status_age_seconds")
    progress_age = max(
        checked - last_progress_unix,
        float(status_age) if status_age is not None else 0.0,
    )
    if (
        pipeline_state == "running"
        and stage.startswith("train_")
        and progress_age > float(stale_after_seconds)
    ):
        candidates.append(
            _event(
                f"stalled-{stage}-{signature}",
                "critical",
                f"`{stage}` has not advanced for approximately {int(progress_age // 60)} minutes.",
            )
        )

    flattened_metrics = _flatten(
        {
            "status": snapshot.get("metrics") or {},
            "latest": snapshot.get("latest_metrics") or {},
        }
    )
    for key, value in flattened_metrics.items():
        if "loss" not in key.lower() or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            candidates.append(
                _event(
                    f"non-finite-{stage}-{key}-{snapshot.get('epoch')}",
                    "critical",
                    f"`{key}` became non-finite ({value!r}).",
                )
            )

    stderr = str(snapshot.get("stderr_tail") or snapshot.get("error_text") or "")
    lowered = stderr.lower()
    error_markers = ("traceback (most recent call last)", "cuda error", "outofmemoryerror")
    current_failure = (
        pipeline_state != "running"
        or str((snapshot.get("status") or {}).get("state", "")) == "error"
        or not snapshot.get("pipeline_pid_running")
        or (
            snapshot.get("stage_pid") is not None
            and snapshot.get("status_fresh")
            and not snapshot.get("stage_pid_running")
        )
    )
    if current_failure and any(marker in lowered for marker in error_markers):
        candidates.append(
            _event(
                "stderr-error-" + _fingerprint(stage, stderr[-4000:]),
                "critical",
                f"A new traceback/CUDA failure appeared in `{stage}` stderr.",
            )
        )

    candidates.extend(_hardening_events(snapshot))

    validation_epoch, validation = _latest_validation(snapshot)
    prior_validation = dict(old.get("last_validation_epoch_by_stage") or {})
    old_validation_epoch = int(prior_validation.get(stage, 0) or 0)
    if validation_epoch is not None and validation_epoch > old_validation_epoch:
        if validation_epoch == 1:
            candidates.append(
                _event(
                    f"first-validation-{stage}",
                    "info",
                    f"`{stage}` completed its first validation epoch.",
                )
            )
        if epoch_milestone > 0 and validation_epoch % epoch_milestone == 0:
            candidates.append(
                _event(
                    f"epoch-milestone-{stage}-{validation_epoch}",
                    "info",
                    f"`{stage}` completed validation epoch {validation_epoch}.",
                )
            )
        epoch_limit = snapshot.get("epoch_limit")
        if epoch_limit is not None and validation_epoch >= int(epoch_limit):
            candidates.append(
                _event(
                    f"epoch-limit-{stage}-{validation_epoch}",
                    "info",
                    f"`{stage}` reached its configured epoch limit ({validation_epoch}).",
                )
            )
        prior_validation[stage] = validation_epoch

    new_events = [event for event in candidates if event["id"] not in remembered]
    if prime:
        new_events = [event for event in new_events if event["level"] == "critical"]
    remembered.update(event["id"] for event in candidates)
    next_state = {
        "format": STATE_FORMAT,
        "updated_unix": checked,
        "last_stage": stage,
        "last_pipeline_state": pipeline_state,
        "last_progress_signature": signature,
        "last_progress_unix": last_progress_unix,
        "last_validation_epoch_by_stage": prior_validation,
        "notified_event_ids": sorted(remembered)[-250:],
        "last_snapshot": {
            "checked_unix": snapshot.get("checked_unix"),
            "stage": stage,
            "stage_index": snapshot.get("stage_index"),
            "pipeline_state": pipeline_state,
            "epoch": snapshot.get("epoch"),
            "global_step": snapshot.get("global_step"),
            "batch_completed": snapshot.get("batch_completed"),
            "validation_epoch": validation_epoch,
            "validation": validation,
        },
    }
    return new_events, next_state


def _remote_source(
    *,
    repository: str,
    artifact_root: str,
    data_root: str,
    multi_data_root: str,
) -> str:
    return f"""
import json
import os
import sys
os.chdir({repository!r})
sys.path.insert(0, {repository!r})
from tools.watch_v2_progress import collect_snapshot
snapshot = collect_snapshot(
    {artifact_root!r},
    data_root={data_root!r},
    multi_data_root={multi_data_root!r},
)
print('CFTN_SNAPSHOT_JSON=' + json.dumps(snapshot, separators=(',', ':')))
""".lstrip()


def collect_remote_snapshot(
    *,
    host: str,
    port: int,
    user: str,
    identity_file: str | Path,
    repository: str,
    artifact_root: str,
    data_root: str,
    multi_data_root: str,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    identity = Path(identity_file).expanduser().resolve()
    if not identity.is_file():
        raise FileNotFoundError(f"SSH identity file is missing: {identity}")
    command = [
        "ssh",
        "-i",
        str(identity),
        "-p",
        str(int(port)),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=10",
        f"{user}@{host}",
        "python3 -",
    ]
    result = subprocess.run(
        command,
        input=_remote_source(
            repository=repository,
            artifact_root=artifact_root,
            data_root=data_root,
            multi_data_root=multi_data_root,
        ),
        capture_output=True,
        text=True,
        timeout=max(10.0, float(timeout_seconds)),
        check=False,
    )
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()[-4000:]
        raise RuntimeError(
            f"SSH snapshot command failed with exit {result.returncode}: {diagnostic}"
        )
    marker = "CFTN_SNAPSHOT_JSON="
    rows = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if not rows:
        raise RuntimeError(
            "remote snapshot returned no JSON marker; output="
            + result.stdout.strip()[-2000:]
        )
    value = json.loads(rows[-1][len(marker) :])
    if not isinstance(value, dict):
        raise RuntimeError("remote snapshot was not a JSON object")
    return value


def build_result(
    snapshot: dict[str, Any], events: Iterable[dict[str, str]]
) -> dict[str, Any]:
    event_list = list(events)
    pipeline_state = str(snapshot.get("pipeline_state", "missing"))
    critical_shutdown_events = (
        "pipeline-error-",
        "pipeline-process-dead-",
        "stage-process-dead-",
        "stalled-",
        "stderr-error-",
    )
    shutdown_eligible = pipeline_state in {"error", "failed", "failed_acceptance"}
    shutdown_eligible = shutdown_eligible or any(
        str(event.get("id", "")).startswith(critical_shutdown_events)
        for event in event_list
    )
    gpus = snapshot.get("gpu") or []
    gpu_idle = bool(gpus) and all(
        int(gpu.get("utilization_percent", 100)) <= 5
        and int(gpu.get("memory_used_mib", 10**9)) <= 512
        for gpu in gpus
    ) and not snapshot.get("process_lines")
    snapshot_summary = {
        "state": pipeline_state,
        "stage": snapshot.get("stage"),
        "stage_index": snapshot.get("stage_index"),
        "stage_count": snapshot.get("stage_count"),
        "epoch": snapshot.get("epoch"),
        "global_step": snapshot.get("global_step"),
        "shutdown_eligible": shutdown_eligible,
        "gpu_idle": gpu_idle,
        "gpu": gpus,
    }
    if not event_list:
        return {
            "decision": "DONT_NOTIFY",
            "message": "V2 is healthy and no new reportable event occurred.",
            "events": [],
            "snapshot_summary": snapshot_summary,
        }
    reasons = "\n".join(
        f"- [{event['level'].upper()}] {event['text']}" for event in event_list
    )
    return {
        "decision": "NOTIFY",
        "message": "## V2 heartbeat event\n\n"
        + reasons
        + "\n\n"
        + render_markdown(snapshot),
        "events": event_list,
        "snapshot_summary": snapshot_summary,
    }


def _default_state_file() -> Path:
    configured = os.environ.get("CFTN_V2_HEARTBEAT_STATE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cftn" / "v2_heartbeat_state.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stateful, event-only heartbeat checker for the remote V2 pipeline"
    )
    parser.add_argument("--host", default=os.environ.get("CFTN_RUNPOD_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CFTN_RUNPOD_PORT", DEFAULT_PORT))
    )
    parser.add_argument("--user", default=os.environ.get("CFTN_RUNPOD_USER", DEFAULT_USER))
    parser.add_argument(
        "--identity-file",
        default=os.environ.get(
            "CFTN_RUNPOD_IDENTITY_FILE",
            str(Path.home() / ".ssh" / "id_ed25519_runpod_cftn"),
        ),
    )
    parser.add_argument("--remote-repository", default=DEFAULT_REMOTE_REPOSITORY)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--multi-data-root", default=DEFAULT_MULTI_DATA_ROOT)
    parser.add_argument("--state-file", default=str(_default_state_file()))
    parser.add_argument("--snapshot-json", help="Use a local snapshot fixture instead of SSH")
    parser.add_argument("--stale-after-seconds", type=float, default=1800.0)
    parser.add_argument("--epoch-milestone", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    state_path = Path(args.state_file).expanduser().resolve()
    previous = _read_json(state_path)
    try:
        if args.snapshot_json:
            snapshot = _read_json(Path(args.snapshot_json).expanduser().resolve())
            if not snapshot:
                raise RuntimeError("snapshot fixture is missing or invalid")
        else:
            snapshot = collect_remote_snapshot(
                host=args.host,
                port=args.port,
                user=args.user,
                identity_file=args.identity_file,
                repository=args.remote_repository,
                artifact_root=args.artifact_root,
                data_root=args.data_root,
                multi_data_root=args.multi_data_root,
                timeout_seconds=args.timeout_seconds,
            )
        events, next_state = classify_snapshot(
            snapshot,
            previous,
            stale_after_seconds=args.stale_after_seconds,
            epoch_milestone=args.epoch_milestone,
            prime=args.prime,
        )
        next_state["last_monitor_error"] = None
        _atomic_write_json(state_path, next_state)
        result = build_result(snapshot, events)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        fingerprint = _fingerprint(error)
        repeated = previous.get("last_monitor_error") == fingerprint
        next_state = dict(previous)
        next_state.update(
            {
                "format": STATE_FORMAT,
                "updated_unix": time.time(),
                "last_monitor_error": fingerprint,
            }
        )
        _atomic_write_json(state_path, next_state)
        result = {
            "decision": "DONT_NOTIFY" if repeated else "NOTIFY",
            "message": (
                "The V2 heartbeat checker remains unable to collect a snapshot."
                if repeated
                else "The V2 heartbeat checker failed and needs attention: " + error
            ),
            "events": []
            if repeated
            else [
                {
                    "id": "monitor-failure-" + fingerprint,
                    "level": "critical",
                    "text": error,
                }
            ],
        }

    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
