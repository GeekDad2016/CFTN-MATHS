from __future__ import annotations

import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .checkpoint import atomic_json_dump


CONTROL_API_FORMAT = "cftn_text_runpod_control_v1"
_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_STAGE = re.compile(r"^[A-Za-z0-9_.-]+$")
_MAX_BODY_BYTES = 16 * 1024


class ControlError(RuntimeError):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = int(status)


def _read_json(path: Path, *, maximum_bytes: int = 2_000_000) -> Any | None:
    try:
        if not path.is_file() or path.stat().st_size > maximum_bytes:
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _tail_text(path: Path, lines: int) -> str:
    if not path.is_file():
        raise ControlError("requested log does not exist", HTTPStatus.NOT_FOUND)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=lines))


def _tail_jsonl(path: Path, lines: int = 3) -> list[Any]:
    if not path.is_file():
        return []
    output: list[Any] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in deque(handle, maxlen=lines):
            try:
                output.append(json.loads(line))
            except ValueError:
                output.append({"unparsed": line.rstrip()})
    return output


def _process_running(process: subprocess.Popen[Any] | None) -> bool:
    return process is not None and process.poll() is None


def _run_checked(
    command: list[str], *, cwd: Path, timeout: float = 300.0
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed").strip()
        raise RuntimeError(f"{command[0]} failed ({result.returncode}): {detail[-2000:]}")
    return result


class RunPodSupervisor:
    """Own one resumable V2 runner and expose a narrow control contract."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        artifact_root: str | Path,
        data_root: str | Path,
        config_path: str | Path,
        api_token: str,
        allow_updates: bool = False,
        device: str = "cuda",
        wandb: bool = True,
        allowed_remote: str | None = None,
    ) -> None:
        token = str(api_token).strip()
        if len(token) < 32:
            raise ValueError("CFTN control API token must contain at least 32 characters")
        self.repository_root = Path(repository_root).expanduser().resolve()
        artifact = Path(artifact_root).expanduser()
        data = Path(data_root).expanduser()
        self.artifact_root = (
            artifact.resolve()
            if artifact.is_absolute()
            else (self.repository_root / artifact).resolve()
        )
        self.data_root = (
            data.resolve() if data.is_absolute() else (self.repository_root / data).resolve()
        )
        config = Path(config_path)
        self.config_path = (
            config.resolve()
            if config.is_absolute()
            else (self.repository_root / config).resolve()
        )
        if not self.config_path.is_relative_to(self.repository_root):
            raise ValueError("config path must remain inside the repository")
        self.api_token = token
        self.allow_updates = bool(allow_updates)
        self.device = str(device)
        self.wandb = bool(wandb)
        self.allowed_remote = allowed_remote or os.environ.get(
            "CFTN_GIT_ALLOWED_REMOTE",
            "https://github.com/GeekDad2016/CFTN-MATHS.git",
        )
        self.control_root = self.artifact_root / "control"
        self.control_root.mkdir(parents=True, exist_ok=True)
        self.pause_request_path = self.control_root / "pause_after_stage.json"
        self.supervisor_state_path = self.control_root / "supervisor_state.json"
        self.update_state_path = self.control_root / "update_state.json"
        self.pipeline_stdout_path = self.control_root / "pipeline.stdout.log"
        self.pipeline_stderr_path = self.control_root / "pipeline.stderr.log"
        self.pipeline_state_path = self.artifact_root / "pipeline_state.json"
        self._lock = threading.RLock()
        self._child: subprocess.Popen[Any] | None = None
        self._streams: tuple[Any, Any] | None = None
        self._update_thread: threading.Thread | None = None
        self._started_unix = time.time()
        self._write_supervisor_state("initialized")

    def authenticates(self, authorization: str | None) -> bool:
        if not authorization or not authorization.startswith("Bearer "):
            return False
        supplied = authorization[len("Bearer ") :]
        return hmac.compare_digest(supplied.encode(), self.api_token.encode())

    def _pipeline_command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "tools.run_v2_experiment",
            "--config",
            str(self.config_path),
            "--device",
            self.device,
            "--execute",
            "--resume",
            "--control-root",
            str(self.control_root),
        ]
        if self.wandb:
            command.append("--wandb")
        return command

    def _write_supervisor_state(self, state: str, **extra: Any) -> None:
        child = self._child
        payload = {
            "format": CONTROL_API_FORMAT,
            "state": state,
            "updated_unix": time.time(),
            "started_unix": self._started_unix,
            "supervisor_pid": os.getpid(),
            "pipeline_pid": child.pid if _process_running(child) else None,
            "updates_enabled": self.allow_updates,
            **extra,
        }
        atomic_json_dump(payload, self.supervisor_state_path)

    def autostart(self) -> dict[str, Any]:
        pipeline = _read_json(self.pipeline_state_path) or {}
        if pipeline.get("state") == "completed":
            self._write_supervisor_state("pipeline_completed")
            return {"state": "pipeline_completed", "started": False}
        if pipeline.get("state") == "paused" or self.pause_request_path.exists():
            self._write_supervisor_state("paused")
            return {"state": "paused", "started": False}
        return self.start_pipeline(reason="autostart")

    def start_pipeline(self, *, reason: str = "resume") -> dict[str, Any]:
        with self._lock:
            if _process_running(self._child):
                raise ControlError("pipeline is already running", HTTPStatus.CONFLICT)
            if (
                self._update_thread is not None
                and self._update_thread.is_alive()
                and threading.current_thread() is not self._update_thread
            ):
                raise ControlError("a repository update is in progress", HTTPStatus.CONFLICT)
            self.pause_request_path.unlink(missing_ok=True)
            stdout = self.pipeline_stdout_path.open("a", encoding="utf-8")
            stderr = self.pipeline_stderr_path.open("a", encoding="utf-8")
            environment = dict(os.environ)
            environment["CFTN_CONTROL_ROOT"] = str(self.control_root)
            try:
                child = subprocess.Popen(
                    self._pipeline_command(),
                    cwd=self.repository_root,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                )
            except BaseException:
                stdout.close()
                stderr.close()
                raise
            self._child = child
            self._streams = (stdout, stderr)
            self._write_supervisor_state(
                "pipeline_running", reason=reason, command=self._pipeline_command()
            )
            watcher = threading.Thread(
                target=self._watch_pipeline,
                args=(child,),
                name="cftn-pipeline-watcher",
                daemon=True,
            )
            watcher.start()
            return {"state": "pipeline_running", "pid": child.pid, "reason": reason}

    def _watch_pipeline(self, child: subprocess.Popen[Any]) -> None:
        returncode = child.wait()
        with self._lock:
            if self._child is not child:
                return
            streams = self._streams
            self._streams = None
            self._child = None
            if streams:
                for stream in streams:
                    stream.close()
            pipeline = _read_json(self.pipeline_state_path) or {}
            if pipeline.get("state") == "paused":
                state = "paused"
            elif returncode == 0 and pipeline.get("state") == "completed":
                state = "pipeline_completed"
            else:
                state = "pipeline_error"
            self._write_supervisor_state(
                state,
                pipeline_returncode=returncode,
                pipeline_state=pipeline.get("state"),
            )

    def request_pause_after_stage(self) -> dict[str, Any]:
        with self._lock:
            if not _process_running(self._child):
                raise ControlError("pipeline is not running", HTTPStatus.CONFLICT)
            request = {
                "format": "cftn_text_pause_request_v1",
                "requested_unix": time.time(),
                "requested_by": "authenticated_control_api",
                "policy": "after_current_stage_completion",
            }
            atomic_json_dump(request, self.pause_request_path)
            self._write_supervisor_state("pause_requested")
            return {
                "state": "pause_requested",
                "policy": "after_current_stage_completion",
            }

    def resume_pipeline(self) -> dict[str, Any]:
        pipeline = _read_json(self.pipeline_state_path) or {}
        if pipeline.get("state") == "completed":
            raise ControlError("pipeline is already complete", HTTPStatus.CONFLICT)
        return self.start_pipeline(reason="authenticated_resume")

    def request_update_resume(
        self,
        *,
        revision: str,
        expected_current_revision: str | None,
        resume: bool,
    ) -> dict[str, Any]:
        if not self.allow_updates:
            raise ControlError(
                "remote updates are disabled; set CFTN_CONTROL_ALLOW_UPDATES=1",
                HTTPStatus.FORBIDDEN,
            )
        revision = str(revision).lower()
        if not _COMMIT.fullmatch(revision):
            raise ControlError("revision must be a full 40-character Git commit SHA")
        if expected_current_revision is not None and not _COMMIT.fullmatch(
            str(expected_current_revision)
        ):
            raise ControlError("expected_current_revision must be a full commit SHA")
        with self._lock:
            if _process_running(self._child):
                raise ControlError(
                    "pipeline is active; request pause-after-stage and wait first",
                    HTTPStatus.CONFLICT,
                )
            if self._update_thread is not None and self._update_thread.is_alive():
                raise ControlError("a repository update is already running", HTTPStatus.CONFLICT)
            job_id = uuid.uuid4().hex
            atomic_json_dump(
                {
                    "format": "cftn_text_update_job_v1",
                    "job_id": job_id,
                    "state": "queued",
                    "revision": revision,
                    "resume": bool(resume),
                    "created_unix": time.time(),
                },
                self.update_state_path,
            )
            self._update_thread = threading.Thread(
                target=self._perform_update,
                kwargs={
                    "job_id": job_id,
                    "revision": revision,
                    "expected_current_revision": (
                        str(expected_current_revision).lower()
                        if expected_current_revision
                        else None
                    ),
                    "resume": bool(resume),
                },
                name="cftn-safe-git-update",
                daemon=True,
            )
            self._update_thread.start()
            return {"state": "queued", "job_id": job_id, "revision": revision}

    def _git_output(self, *arguments: str) -> str:
        return _run_checked(
            ["git", *arguments], cwd=self.repository_root, timeout=300
        ).stdout.strip()

    def _perform_update(
        self,
        *,
        job_id: str,
        revision: str,
        expected_current_revision: str | None,
        resume: bool,
    ) -> None:
        def write(state: str, **extra: Any) -> None:
            atomic_json_dump(
                {
                    "format": "cftn_text_update_job_v1",
                    "job_id": job_id,
                    "state": state,
                    "revision": revision,
                    "resume": resume,
                    "updated_unix": time.time(),
                    **extra,
                },
                self.update_state_path,
            )

        try:
            write("validating")
            if self._git_output("status", "--porcelain"):
                raise RuntimeError("repository worktree is not clean")
            current = self._git_output("rev-parse", "HEAD").lower()
            if expected_current_revision and current != expected_current_revision:
                raise RuntimeError(
                    f"current revision changed: expected {expected_current_revision}, got {current}"
                )
            remote = self._git_output("remote", "get-url", "origin")
            if remote.rstrip("/") != self.allowed_remote.rstrip("/"):
                raise RuntimeError("origin URL does not match CFTN_GIT_ALLOWED_REMOTE")
            write("fetching", previous_revision=current)
            _run_checked(
                ["git", "fetch", "--prune", "origin", "main"],
                cwd=self.repository_root,
                timeout=600,
            )
            resolved = self._git_output("rev-parse", "--verify", f"{revision}^{{commit}}")
            if resolved.lower() != revision:
                raise RuntimeError("requested revision did not resolve exactly")
            for ancestor, descendant, label in (
                (current, revision, "requested revision is not a fast-forward"),
                (revision, "origin/main", "requested revision is not published on origin/main"),
            ):
                result = subprocess.run(
                    ["git", "merge-base", "--is-ancestor", ancestor, descendant],
                    cwd=self.repository_root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(label)
            write("installing", previous_revision=current)
            _run_checked(
                ["git", "merge", "--ff-only", revision],
                cwd=self.repository_root,
                timeout=300,
            )
            _run_checked(
                [sys.executable, "-m", "pip", "install", "-e", "."],
                cwd=self.repository_root,
                timeout=1200,
            )
            write("completed", previous_revision=current, current_revision=revision)
            if resume:
                self.start_pipeline(reason=f"updated_to_{revision[:12]}")
        except BaseException as exc:
            write("error", error=repr(exc))
            self._write_supervisor_state("update_error", update_error=repr(exc))

    def _git_status(self) -> dict[str, Any]:
        try:
            revision = self._git_output("rev-parse", "HEAD")
            branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD")
            remote = self._git_output("remote", "get-url", "origin")
            dirty = bool(self._git_output("status", "--porcelain"))
            return {
                "available": True,
                "revision": revision,
                "branch": branch,
                "origin": remote,
                "dirty": dirty,
            }
        except BaseException as exc:
            return {"available": False, "error": repr(exc)}

    @staticmethod
    def _gpu_status() -> dict[str, Any]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode:
                return {"available": False, "error": result.stderr.strip()[-500:]}
            gpus = []
            for line in result.stdout.splitlines():
                values = [item.strip() for item in line.split(",")]
                if len(values) == 6:
                    gpus.append(
                        {
                            "index": int(values[0]),
                            "name": values[1],
                            "utilization_percent": int(values[2]),
                            "memory_used_mib": int(values[3]),
                            "memory_total_mib": int(values[4]),
                            "temperature_c": int(values[5]),
                        }
                    )
            return {"available": bool(gpus), "gpus": gpus}
        except BaseException as exc:
            return {"available": False, "error": repr(exc)}

    def _recent_artifacts(self) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        for name in ("status.json", "progress.json", "wandb_run.json"):
            candidates.extend(self.artifact_root.rglob(name))
        candidates = sorted(
            (path for path in candidates if path.is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:12]
        output: list[dict[str, Any]] = []
        for path in candidates:
            output.append(
                {
                    "path": str(path.relative_to(self.artifact_root)),
                    "modified_unix": path.stat().st_mtime,
                    "content": _read_json(path),
                }
            )
        return output

    def checkpoint_inventory(self) -> dict[str, Any]:
        checkpoints = sorted(
            (
                path
                for pattern in ("*.pth", "*.pt", "*.safetensors")
                for path in self.artifact_root.rglob(pattern)
                if path.is_file()
            ),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:200]
        return {
            "count_returned": len(checkpoints),
            "checkpoints": [
                {
                    "path": str(path.relative_to(self.artifact_root)),
                    "bytes": path.stat().st_size,
                    "modified_unix": path.stat().st_mtime,
                }
                for path in checkpoints
            ],
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            child = self._child
            process = {
                "running": _process_running(child),
                "pid": child.pid if _process_running(child) else None,
                "returncode": child.poll() if child is not None else None,
            }
        metrics = sorted(
            (path for path in self.artifact_root.rglob("metrics.jsonl") if path.is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        latest_metrics = None
        if metrics:
            latest_metrics = {
                "path": str(metrics[0].relative_to(self.artifact_root)),
                "rows": _tail_jsonl(metrics[0], 3),
            }
        return {
            "format": CONTROL_API_FORMAT,
            "server_unix": time.time(),
            "pod_id": os.environ.get("RUNPOD_POD_ID"),
            "supervisor": _read_json(self.supervisor_state_path),
            "pipeline_process": process,
            "pipeline": _read_json(self.pipeline_state_path),
            "data_preparation": _read_json(self.data_root / "prepare_status.json"),
            "pause_request": _read_json(self.pause_request_path),
            "update": _read_json(self.update_state_path),
            "latest_metrics": latest_metrics,
            "recent_artifacts": self._recent_artifacts(),
            "checkpoint_summary": {
                "count": sum(
                    1
                    for pattern in ("*.pth", "*.pt", "*.safetensors")
                    for _ in self.artifact_root.rglob(pattern)
                )
            },
            "gpu": self._gpu_status(),
            "git": self._git_status(),
            "updates_enabled": self.allow_updates,
        }

    def log_tail(self, *, stage: str, stream: str, lines: int) -> dict[str, Any]:
        if stream not in {"stdout", "stderr"}:
            raise ControlError("stream must be stdout or stderr")
        if not _STAGE.fullmatch(stage):
            raise ControlError("invalid stage name")
        lines = min(500, max(1, int(lines)))
        if stage == "supervisor":
            path = self.control_root / f"pipeline.{stream}.log"
        else:
            path = self.artifact_root / "pipeline_logs" / f"{stage}.{stream}.log"
        resolved = path.resolve()
        if not resolved.is_relative_to(self.artifact_root):
            raise ControlError("log path escaped artifact root", HTTPStatus.FORBIDDEN)
        return {
            "stage": stage,
            "stream": stream,
            "lines": lines,
            "path": str(resolved.relative_to(self.artifact_root)),
            "text": _tail_text(resolved, lines),
        }


class _ControlServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        supervisor: RunPodSupervisor,
    ) -> None:
        self.supervisor = supervisor
        super().__init__(server_address, RunPodControlHandler)


class RunPodControlHandler(BaseHTTPRequestHandler):
    server: _ControlServer

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid echoing authorization headers or request bodies into training logs.
        return

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authenticate(self) -> bool:
        if self.server.supervisor.authenticates(self.headers.get("Authorization")):
            return True
        self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
        return False

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ControlError("invalid Content-Length") from exc
        if length < 0 or length > _MAX_BODY_BYTES:
            raise ControlError("request body is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ControlError("request body must be a JSON object") from exc
        if not isinstance(payload, dict):
            raise ControlError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._json(
                HTTPStatus.OK,
                {"status": "ok", "format": CONTROL_API_FORMAT},
            )
            return
        if not self._authenticate():
            return
        try:
            if parsed.path == "/v1/status":
                payload = self.server.supervisor.status()
            elif parsed.path == "/v1/checkpoints":
                payload = self.server.supervisor.checkpoint_inventory()
            elif parsed.path == "/v1/logs":
                query = parse_qs(parsed.query)
                payload = self.server.supervisor.log_tail(
                    stage=query.get("stage", ["supervisor"])[0],
                    stream=query.get("stream", ["stderr"])[0],
                    lines=int(query.get("lines", ["100"])[0]),
                )
            else:
                raise ControlError("endpoint not found", HTTPStatus.NOT_FOUND)
            self._json(HTTPStatus.OK, payload)
        except ControlError as exc:
            self._json(exc.status, {"error": str(exc)})
        except BaseException as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._authenticate():
            return
        try:
            body = self._body()
            if parsed.path == "/v1/actions/pause-after-stage":
                payload = self.server.supervisor.request_pause_after_stage()
            elif parsed.path == "/v1/actions/resume":
                payload = self.server.supervisor.resume_pipeline()
            elif parsed.path == "/v1/actions/update-resume":
                payload = self.server.supervisor.request_update_resume(
                    revision=str(body.get("revision", "")),
                    expected_current_revision=(
                        str(body["expected_current_revision"])
                        if body.get("expected_current_revision")
                        else None
                    ),
                    resume=bool(body.get("resume", True)),
                )
            else:
                raise ControlError("endpoint not found", HTTPStatus.NOT_FOUND)
            self._json(HTTPStatus.ACCEPTED, payload)
        except ControlError as exc:
            self._json(exc.status, {"error": str(exc)})
        except BaseException as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(exc)})


def create_control_server(
    supervisor: RunPodSupervisor, host: str, port: int
) -> ThreadingHTTPServer:
    return _ControlServer((str(host), int(port)), supervisor)
