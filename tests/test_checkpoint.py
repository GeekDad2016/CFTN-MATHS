from __future__ import annotations

import torch

from cftn_text.checkpoint import (
    atomic_json_dump,
    atomic_torch_save,
    build_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    rotate_latest,
)


def payload(epoch: int):
    return build_checkpoint(
        stage="math",
        epoch=epoch,
        global_step=epoch * 10,
        model_state={"weight": torch.tensor([float(epoch)])},
        optimizer_state={},
        scheduler_state={},
        scaler_state=None,
        config_sha256="config",
        manifest_sha256="manifest",
        best_metric=0.5,
        patience=0,
    )


def test_checkpoint_contract_and_rotation(tmp_path):
    for epoch in range(1, 6):
        atomic_torch_save(payload(epoch), tmp_path / f"checkpoint_epoch_{epoch:04d}.pth")
    removed = rotate_latest(tmp_path, keep=3)
    assert len(removed) == 2
    remaining = sorted(tmp_path.glob("checkpoint_epoch_*.pth"))
    assert [path.stem[-4:] for path in remaining] == ["0003", "0004", "0005"]
    loaded = load_checkpoint(
        remaining[-1],
        expected_stage="math",
        expected_config_sha256="config",
        expected_manifest_sha256="manifest",
    )
    assert loaded["epoch"] == 5
    assert torch.equal(loaded["model_state"]["weight"], torch.tensor([5.0]))


def test_rng_restore_accepts_states_relocated_with_checkpoint():
    state = capture_rng_state()
    expected_cpu = state["torch_cpu"].clone()
    if torch.cuda.is_available():
        state["torch_cpu"] = state["torch_cpu"].to("cuda")
        state["torch_cuda"] = [value.to("cuda") for value in state["torch_cuda"]]
    torch.manual_seed(12345)
    restore_rng_state(state)
    assert torch.equal(torch.random.get_rng_state(), expected_cpu)


def test_atomic_json_retries_windows_reader_collision(tmp_path, monkeypatch):
    import cftn_text.checkpoint as checkpoint_module

    real_replace = checkpoint_module.os.replace
    attempts = {"count": 0}

    def flaky_replace(source, destination):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("simulated sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, "replace", flaky_replace)
    path = tmp_path / "status.json"
    atomic_json_dump({"state": "running"}, path)
    assert attempts["count"] == 3
    assert path.read_text(encoding="utf-8").strip().startswith("{")
