from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump


def flatten_metrics(
    payload: dict[str, Any], *, prefix: str = ""
) -> dict[str, int | float | bool | str]:
    flattened: dict[str, int | float | bool | str] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, prefix=name))
        elif isinstance(value, (bool, int, float, str)) and value is not None:
            flattened[name] = value
    return flattened


class NullWandbTracker:
    enabled = False

    def log(
        self,
        payload: dict[str, Any],
        *,
        global_step: int,
        epoch: int | None = None,
        event: str | None = None,
    ) -> None:
        del payload, global_step, epoch, event

    def update_summary(self, payload: dict[str, Any]) -> None:
        del payload

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


class WandbTracker:
    enabled = True

    def __init__(self, run: Any) -> None:
        self.run = run
        self._warning_emitted = False

    def _warn(self, exc: BaseException) -> None:
        if not self._warning_emitted:
            print(f"W&B logging warning (training will continue): {exc}")
            self._warning_emitted = True

    def log(
        self,
        payload: dict[str, Any],
        *,
        global_step: int,
        epoch: int | None = None,
        event: str | None = None,
    ) -> None:
        values = flatten_metrics(payload)
        values["trainer/global_step"] = int(global_step)
        if epoch is not None:
            values["trainer/epoch"] = int(epoch)
        if event is not None:
            values["trainer/event"] = str(event)
        try:
            self.run.log(values)
        except BaseException as exc:
            self._warn(exc)

    def update_summary(self, payload: dict[str, Any]) -> None:
        try:
            for key, value in flatten_metrics(payload).items():
                self.run.summary[key] = value
        except BaseException as exc:
            self._warn(exc)

    def finish(self, exit_code: int = 0) -> None:
        try:
            self.run.finish(exit_code=int(exit_code))
        except BaseException as exc:
            self._warn(exc)


def initialize_wandb(
    options: dict[str, Any] | None,
    *,
    artifact_dir: str | Path,
    stage: str,
    config: dict[str, Any] | None = None,
) -> NullWandbTracker | WandbTracker:
    if not options or not bool(options.get("enabled", False)):
        return NullWandbTracker()
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was enabled but the wandb package is not installed"
        ) from exc
    root = Path(artifact_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "wandb_run.json"
    existing: dict[str, Any] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    project = str(options.get("project") or existing.get("project") or "cftn-text")
    entity = options.get("entity", existing.get("entity"))
    run_id = str(existing.get("run_id") or options.get("run_id") or wandb.util.generate_id())
    run_name = str(options.get("run_name") or existing.get("run_name") or stage)
    mode = str(options.get("mode", "online"))
    tags = [str(tag) for tag in options.get("tags", [])]
    group = options.get("group")
    run_config = {
        "stage": stage,
        "artifact_dir": str(root),
        **(config or {}),
    }
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        name=run_name,
        resume="allow",
        mode=mode,
        tags=tags or None,
        group=group,
        config=run_config,
        dir=str(root),
        settings=wandb.Settings(silent=False),
    )
    run.define_metric("trainer/global_step")
    run.define_metric("*", step_metric="trainer/global_step")
    atomic_json_dump(
        {
            "format": "cftn_text_wandb_run_v1",
            "project": project,
            "entity": entity,
            "run_id": run_id,
            "run_name": run_name,
            "stage": stage,
            "url": getattr(run, "url", None),
        },
        metadata_path,
    )
    return WandbTracker(run)


def wandb_options_from_args(args: Any, *, default_run_name: str) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(args, "wandb", False)),
        "project": getattr(args, "wandb_project", "cftn-text"),
        "entity": getattr(args, "wandb_entity", None),
        "run_name": getattr(args, "wandb_run_name", None) or default_run_name,
        "group": getattr(args, "wandb_group", None),
        "tags": getattr(args, "wandb_tags", None) or [],
        "mode": getattr(args, "wandb_mode", "online"),
    }


def add_wandb_arguments(parser: Any) -> None:
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cftn-text")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
