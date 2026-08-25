from __future__ import annotations

import errno
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch


CHECKPOINT_FORMAT = "cftn_text_checkpoint_v1"
DEFAULT_IO_RETRY_ATTEMPTS = 12
DEFAULT_IO_RETRY_BASE_SECONDS = 0.1
MAX_IO_RETRY_DELAY_SECONDS = 3.0
_RETRYABLE_IO_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EIO,
    errno.ESTALE,
    errno.ETIMEDOUT,
}
_T = TypeVar("_T")


def _retryable_io_error(exc: BaseException) -> bool:
    return isinstance(exc, PermissionError) or (
        isinstance(exc, OSError) and exc.errno in _RETRYABLE_IO_ERRNOS
    )


def _with_io_retries(
    operation: Callable[[int], _T],
    *,
    attempts: int,
    base_delay_seconds: float,
) -> _T:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation(attempt)
        except BaseException as exc:
            if not _retryable_io_error(exc) or attempt + 1 >= attempts:
                raise
            delay = min(
                MAX_IO_RETRY_DELAY_SECONDS,
                max(0.0, base_delay_seconds) * (2**attempt),
            )
            if delay:
                time.sleep(delay)
    raise AssertionError("unreachable retry loop")


def _temporary_path(destination: Path, attempt: int) -> Path:
    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{attempt}.tmp"
    )


def ensure_directory(
    path: str | Path,
    *,
    retry_attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
    retry_base_seconds: float = DEFAULT_IO_RETRY_BASE_SECONDS,
) -> Path:
    directory = Path(path)

    def create(_: int) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    return _with_io_retries(
        create,
        attempts=retry_attempts,
        base_delay_seconds=retry_base_seconds,
    )


def atomic_torch_save(
    payload: Any,
    path: str | Path,
    *,
    retry_attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
    retry_base_seconds: float = DEFAULT_IO_RETRY_BASE_SECONDS,
) -> None:
    destination = Path(path)

    def write(attempt: int) -> None:
        temporary = _temporary_path(destination, attempt)
        try:
            ensure_directory(destination.parent, retry_attempts=1)
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    _with_io_retries(
        write,
        attempts=retry_attempts,
        base_delay_seconds=retry_base_seconds,
    )


def atomic_json_dump(
    payload: Any,
    path: str | Path,
    *,
    retry_attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
    retry_base_seconds: float = DEFAULT_IO_RETRY_BASE_SECONDS,
) -> None:
    destination = Path(path)
    serialized = json.dumps(
        payload, indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"

    def write(attempt: int) -> None:
        temporary = _temporary_path(destination, attempt)
        try:
            ensure_directory(
                destination.parent,
                retry_attempts=1,
                retry_base_seconds=retry_base_seconds,
            )
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    _with_io_retries(
        write,
        attempts=retry_attempts,
        base_delay_seconds=retry_base_seconds,
    )


def atomic_copy_file(
    source: str | Path,
    destination: str | Path,
    *,
    retry_attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
    retry_base_seconds: float = DEFAULT_IO_RETRY_BASE_SECONDS,
) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    def copy(attempt: int) -> None:
        temporary = _temporary_path(destination_path, attempt)
        try:
            ensure_directory(
                destination_path.parent,
                retry_attempts=1,
                retry_base_seconds=retry_base_seconds,
            )
            with source_path.open("rb") as source_handle, temporary.open(
                "wb"
            ) as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
            os.replace(temporary, destination_path)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    _with_io_retries(
        copy,
        attempts=retry_attempts,
        base_delay_seconds=retry_base_seconds,
    )


def append_jsonl(
    payload: Any,
    path: str | Path,
    *,
    retry_attempts: int = DEFAULT_IO_RETRY_ATTEMPTS,
    retry_base_seconds: float = DEFAULT_IO_RETRY_BASE_SECONDS,
) -> None:
    destination = Path(path)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"

    def append(_: int) -> None:
        ensure_directory(destination.parent, retry_attempts=1)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

    _with_io_retries(
        append,
        attempts=retry_attempts,
        base_delay_seconds=retry_base_seconds,
    )


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
