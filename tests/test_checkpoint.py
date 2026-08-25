from __future__ import annotations

import errno

import pytest
import torch

from cftn_text.checkpoint import (
    atomic_copy_file,
    atomic_json_dump,
    atomic_torch_save,
    build_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    rotate_latest,
)
from cftn_text.data_generator import file_sha256
from cftn_text.training import _load_math_initialization_checkpoint


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


def test_model_initialization_authenticates_source_but_allows_new_manifest(tmp_path):
    source = tmp_path / "sealed_source.pth"
    atomic_torch_save(payload(45), source)

    loaded, observed_sha256 = _load_math_initialization_checkpoint(
        source,
        expected_sha256=file_sha256(source),
        map_location="cpu",
    )

    assert loaded["epoch"] == 45
    assert loaded["manifest_sha256"] == "manifest"
    assert observed_sha256 == file_sha256(source)

    with pytest.raises(RuntimeError, match="source checkpoint hash changed"):
        _load_math_initialization_checkpoint(
            source,
            expected_sha256="0" * 64,
            map_location="cpu",
        )


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


def test_atomic_json_retries_transient_fuse_eio(tmp_path, monkeypatch):
    import cftn_text.checkpoint as checkpoint_module

    real_fsync = checkpoint_module.os.fsync
    attempts = {"count": 0}

    def flaky_fsync(descriptor):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError(errno.EIO, "simulated FUSE write failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(checkpoint_module.os, "fsync", flaky_fsync)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _: None)
    path = tmp_path / "status.json"
    atomic_json_dump({"state": "running"}, path)
    assert attempts["count"] == 3
    assert path.read_text(encoding="utf-8").strip().startswith("{")
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_copy_retries_transient_fuse_eio(tmp_path, monkeypatch):
    import cftn_text.checkpoint as checkpoint_module

    source = tmp_path / "source.bin"
    source.write_bytes(b"durable checkpoint" * 1024)
    destination = tmp_path / "published" / "checkpoint.bin"
    real_replace = checkpoint_module.os.replace
    attempts = {"count": 0}

    def flaky_replace(source_path, destination_path):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError(errno.EIO, "simulated FUSE rename failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(checkpoint_module.os, "replace", flaky_replace)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _: None)
    atomic_copy_file(source, destination)
    assert attempts["count"] == 3
    assert destination.read_bytes() == source.read_bytes()
    assert not list(destination.parent.glob("*.tmp"))


def test_atomic_torch_save_retries_transient_fuse_eio(tmp_path, monkeypatch):
    import cftn_text.checkpoint as checkpoint_module

    real_replace = checkpoint_module.os.replace
    attempts = {"count": 0}

    def flaky_replace(source_path, destination_path):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError(errno.EIO, "simulated FUSE checkpoint failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(checkpoint_module.os, "replace", flaky_replace)
    monkeypatch.setattr(checkpoint_module.time, "sleep", lambda _: None)
    destination = tmp_path / "checkpoint.pth"
    atomic_torch_save(payload(7), destination)
    assert attempts["count"] == 3
    assert load_checkpoint(destination)["epoch"] == 7
    assert not list(tmp_path.glob("*.tmp"))
