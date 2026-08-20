from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.v1_3_config import load_v1_3_config

from .recover_v1_3_hard_binary import ADAPTER_PHASE, CONTINUATION_PHASE


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return int(exit_code.value) == still_active
    finally:
        kernel32.CloseHandle(handle)


def _write_state(path: Path, state: str, **details: Any) -> None:
    atomic_json_dump(
        {
            "format": "cftn_text_v1_3_adapter_continuation_handoff_v1",
            "state": state,
            "watcher_pid": os.getpid(),
            "updated_unix": time.time(),
            **details,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the V1.3 adapter continuation after a guarded clean handoff"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-pid", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_v1_3_config(config_path)
    artifact_root = Path(config["paths"]["artifact_root"])
    recovery_status_path = artifact_root / "hard_binary_recovery_pipeline.json"
    continuation_status_path = artifact_root / "hard_binary_continuation_pipeline.json"
    handoff_path = artifact_root / "hard_binary_continuation_handoff.json"
    stdout_path = artifact_root / "hard_binary_continuation.stdout.log"
    stderr_path = artifact_root / "hard_binary_continuation.stderr.log"

    if handoff_path.is_file():
        prior = _read_json(handoff_path)
        if prior.get("state") == "launched" and _process_alive(
            int(prior.get("continuation_pid", -1))
        ):
            return
    continuation_summary = artifact_root / CONTINUATION_PHASE / "summary.json"
    if continuation_summary.is_file() and _read_json(continuation_summary).get(
        "state"
    ) == "completed":
        _write_state(handoff_path, "already_completed")
        return
    if continuation_status_path.is_file():
        continuation_status = _read_json(continuation_status_path)
        if continuation_status.get("state") == "running" and _process_alive(
            int(continuation_status.get("pid", -1))
        ):
            _write_state(
                handoff_path,
                "already_running",
                continuation_pid=int(continuation_status["pid"]),
            )
            return

    if not recovery_status_path.is_file():
        raise FileNotFoundError(f"recovery status is missing: {recovery_status_path}")
    recovery = _read_json(recovery_status_path)
    if int(recovery.get("pid", -1)) != args.expected_pid:
        raise RuntimeError(
            "refusing handoff because the recovery PID differs from --expected-pid"
        )
    if _process_alive(args.expected_pid):
        if recovery.get("state") != "running" or recovery.get("current_phase") != ADAPTER_PHASE:
            raise RuntimeError("live recovery process is not the expected adapter phase")
        _write_state(
            handoff_path,
            "waiting_for_adapter",
            expected_adapter_pid=args.expected_pid,
            recovery_status=str(recovery_status_path.resolve()),
        )
        while _process_alive(args.expected_pid):
            time.sleep(max(1.0, args.poll_seconds))

    deadline = time.time() + 300.0
    while time.time() < deadline:
        recovery = _read_json(recovery_status_path)
        if recovery.get("state") in {"completed", "error"}:
            break
        time.sleep(2.0)
    completed = set(recovery.get("completed_phases", []))
    if recovery.get("state") != "completed" or ADAPTER_PHASE not in completed:
        _write_state(
            handoff_path,
            "blocked_adapter_not_clean",
            expected_adapter_pid=args.expected_pid,
            recovery_state=recovery.get("state"),
            recovery_error=recovery.get("error"),
        )
        return
    adapter_summary_path = artifact_root / ADAPTER_PHASE / "summary.json"
    if not adapter_summary_path.is_file() or _read_json(adapter_summary_path).get(
        "state"
    ) != "completed":
        _write_state(handoff_path, "blocked_adapter_summary_incomplete")
        return

    command = [
        sys.executable,
        "-u",
        "-m",
        "tools.recover_v1_3_hard_binary",
        "--config",
        str(config_path),
        "--stage",
        "continuation",
        "--source-checkpoint",
        "auto",
        "--device",
        args.device,
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
    _write_state(
        handoff_path,
        "launched",
        expected_adapter_pid=args.expected_pid,
        continuation_pid=process.pid,
        command=command,
        stdout=str(stdout_path.resolve()),
        stderr=str(stderr_path.resolve()),
    )


if __name__ == "__main__":
    main()
