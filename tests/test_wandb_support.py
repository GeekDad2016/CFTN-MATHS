from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from cftn_text.wandb_support import NullWandbTracker, flatten_metrics, initialize_wandb


def test_flatten_metrics_preserves_numeric_hierarchy():
    flattened = flatten_metrics(
        {
            "epoch": 3,
            "validation": {"loss": 0.25, "pass": True},
            "ignored": None,
            "also_ignored": [1, 2],
        }
    )
    assert flattened == {
        "epoch": 3,
        "validation/loss": 0.25,
        "validation/pass": True,
    }


def test_disabled_wandb_does_not_import_client(tmp_path):
    tracker = initialize_wandb(
        {"enabled": False}, artifact_dir=tmp_path, stage="math"
    )
    assert isinstance(tracker, NullWandbTracker)
    tracker.log({"loss": 1.0}, global_step=1, epoch=1)


def test_wandb_initialization_does_not_require_util_generate_id(
    tmp_path, monkeypatch
):
    class FakeRun:
        url = "https://wandb.example/run/test"

        def __init__(self):
            self.summary = {}
            self.metrics = []

        def define_metric(self, *args, **kwargs):
            self.metrics.append((args, kwargs))

    fake_run = FakeRun()
    init_arguments = {}

    def fake_init(**kwargs):
        init_arguments.update(kwargs)
        return fake_run

    fake_wandb = SimpleNamespace(
        init=fake_init,
        Settings=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        "cftn_text.wandb_support.secrets.token_hex", lambda _: "deadbeef"
    )

    tracker = initialize_wandb(
        {
            "enabled": True,
            "project": "test-project",
            "run_name": "test-run",
            "mode": "online",
        },
        artifact_dir=tmp_path,
        stage="math",
    )

    assert tracker.enabled is True
    assert init_arguments["id"] == "deadbeef"
    metadata = json.loads((tmp_path / "wandb_run.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "deadbeef"
