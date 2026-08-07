from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT = "cftn_text_checkpoint_v1"


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Windows temporarily denies replacement while another process has the
    # status file open for reading. Retry briefly so monitoring cannot stop a
    # training or evaluation process.
    for attempt in range(20):
        try:
            os.replace(temporary, destination)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def append_jsonl(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    # Loading a full checkpoint with map_location="cuda" also relocates RNG
    # tensors. PyTorch's RNG restoration APIs require CPU ByteTensors even
    # when restoring CUDA generator state.
    torch_cpu_state = state["torch_cpu"].detach().to(
        device="cpu", dtype=torch.uint8
    )
    torch.random.set_rng_state(torch_cpu_state)
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch_cuda_states = [
            value.detach().to(device="cpu", dtype=torch.uint8)
            for value in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(torch_cuda_states)


def build_checkpoint(
    *,
    stage: str,
    epoch: int,
    global_step: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any],
    scaler_state: dict[str, Any] | None,
    config_sha256: str,
    manifest_sha256: str,
    best_metric: float,
    patience: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "rng_state": capture_rng_state(),
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "best_metric": float(best_metric),
        "patience": int(patience),
        "extra": extra or {},
    }


def load_checkpoint(
    path: str | Path,
    *,
    expected_stage: str | None = None,
    expected_config_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if expected_stage is not None and payload.get("stage") != expected_stage:
        raise ValueError("checkpoint stage mismatch")
    if (
        expected_config_sha256 is not None
        and payload.get("config_sha256") != expected_config_sha256
    ):
        raise ValueError("checkpoint configuration hash mismatch")
    if (
        expected_manifest_sha256 is not None
        and payload.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("checkpoint data manifest hash mismatch")
    return payload


def rotate_latest(directory: str | Path, keep: int) -> list[Path]:
    if keep < 1:
        raise ValueError("keep must be positive")
    root = Path(directory)
    checkpoints = sorted(root.glob("checkpoint_epoch_*.pth"))
    removed: list[Path] = []
    for path in checkpoints[:-keep]:
        path.unlink()
        removed.append(path)
    return removed


def latest_checkpoint(directory: str | Path) -> Path | None:
    checkpoints = sorted(Path(directory).glob("checkpoint_epoch_*.pth"))
    return checkpoints[-1] if checkpoints else None


def gpu_status() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    device = torch.cuda.current_device()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
