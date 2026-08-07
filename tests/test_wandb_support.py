from __future__ import annotations

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
