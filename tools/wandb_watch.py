from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.wandb_support import (
    add_wandb_arguments,
    initialize_wandb,
    wandb_options_from_args,
)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def watch_metrics(
    run_dir: str | Path,
    options: dict[str, Any],
    *,
    poll_seconds: float = 30.0,
    once: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    metrics_path = root / "metrics.jsonl"
    status_path = root / "status.json"
    state_path = root / "wandb_sync_state.json"
    state = _json(state_path) or {
        "format": "cftn_text_wandb_sync_v1",
        "metrics_rows": 0,
        "last_status_signature": None,
    }
    status = _json(status_path) or {}
    stage = str(status.get("stage") or root.name)
    tracker = initialize_wandb(
        options,
        artifact_dir=root,
        stage=stage,
        config={"source": "metrics_sidecar"},
    )
    terminal_state = None
    try:
        while True:
            rows = _jsonl_rows(metrics_path)
            start = int(state.get("metrics_rows", 0))
            if start > len(rows):
                raise RuntimeError("metrics file shrank after W&B synchronization")
            for row in rows[start:]:
                tracker.log(
                    row,
                    global_step=int(row.get("global_step", 0)),
                    epoch=int(row.get("epoch", 0)),
                    event="epoch_validation",
                )
            state["metrics_rows"] = len(rows)
            status = _json(status_path) or {}
            signature = (
                status.get("state"),
                status.get("epoch"),
                status.get("global_step"),
                status.get("updated_unix"),
            )
            if signature != tuple(state.get("last_status_signature") or ()):
                tracker.log(
                    {
                        "status": status.get("metrics", {}),
                        "runtime": {
                            "elapsed_seconds": status.get("elapsed_seconds", 0.0),
                            "state": status.get("state", "unknown"),
                        },
                    },
                    global_step=int(status.get("global_step", 0)),
                    epoch=int(status.get("epoch", 0)),
                    event="live_status",
                )
                state["last_status_signature"] = list(signature)
            atomic_json_dump(state, state_path)
            terminal_state = status.get("state")
            if once or terminal_state in {"completed", "error"}:
                break
            time.sleep(max(1.0, float(poll_seconds)))
        tracker.update_summary(
            {
                "sync/metrics_rows": int(state["metrics_rows"]),
                "sync/final_state": terminal_state or "unknown",
            }
        )
        tracker.finish(exit_code=1 if terminal_state == "error" else 0)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    return {
        "state": terminal_state or "unknown",
        "metrics_rows": int(state["metrics_rows"]),
        "run_metadata": str((root / "wandb_run.json").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill and follow a CFTN metrics directory in W&B"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    add_wandb_arguments(parser)
    args = parser.parse_args()
    options = wandb_options_from_args(
        args, default_run_name=Path(args.run_dir).resolve().name
    )
    options["enabled"] = True
    result = watch_metrics(
        args.run_dir,
        options,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
