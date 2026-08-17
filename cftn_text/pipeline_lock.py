from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator


class PipelineAlreadyRunning(RuntimeError):
    """Raised when another process owns the experiment's operating-system lock."""


def _acquire(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _owner_description(handle: BinaryIO) -> str:
    try:
        handle.seek(0)
        raw = handle.read(4096).decode("utf-8", errors="replace").strip("\0\r\n ")
        if raw:
            return raw
    except OSError:
        pass
    return "owner metadata unavailable"


@contextmanager
def exclusive_pipeline_lock(path: str | Path) -> Iterator[Path]:
    """Hold a crash-safe, process-lifetime lock for one artifact root.

    The lock file is deliberately retained after release. The kernel lock, not
    file existence, is authoritative, so a killed pod releases ownership and a
    resumed process can safely acquire the same file.
    """

    lock_path = Path(path).expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            _acquire(handle)
        except OSError as exc:
            owner = _owner_description(handle)
            raise PipelineAlreadyRunning(
                f"another V2 pipeline owns {lock_path}: {owner}"
            ) from exc

        owner = {
            "format": "cftn_text_pipeline_lock_v1",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_unix": time.time(),
        }
        handle.seek(0)
        handle.truncate()
        handle.write((json.dumps(owner, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        try:
            _release(handle)
        except OSError:
            pass
        handle.close()
