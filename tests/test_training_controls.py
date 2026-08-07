from __future__ import annotations

import sys

from cftn_text.training import _should_stop_early
from tools import train_math_tower as train_math_cli


def test_early_stopping_can_be_disabled_without_changing_settings():
    settings = {"minimum_epochs": 10, "early_stop_patience": 10}
    assert _should_stop_early(
        epoch=70, patience=10, settings=settings, enabled=True
    )
    assert not _should_stop_early(
        epoch=70, patience=10, settings=settings, enabled=False
    )


def test_math_cli_passes_resume_and_early_stopping_override(monkeypatch, capsys):
    config = {"project": {"seed": 719}}
    captured = {}

    monkeypatch.setattr(train_math_cli, "load_config", lambda _: config)

    def fake_train(received_config, **kwargs):
        captured["config"] = received_config
        captured.update(kwargs)
        return {"state": "started"}

    monkeypatch.setattr(train_math_cli, "train_math_tower", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_math_tower",
            "--resume",
            "--disable-early-stopping",
            "--wandb-mode",
            "disabled",
        ],
    )
    train_math_cli.main()

    assert captured["config"] is config
    assert captured["resume"] is True
    assert captured["disable_early_stopping"] is True
    assert captured["wandb_options"]["run_name"] == "math-seed-719"
    assert '"state": "started"' in capsys.readouterr().out
