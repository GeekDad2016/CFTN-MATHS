from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from cftn_text.runpod_control import ControlError, RunPodSupervisor, create_control_server


TOKEN = "test-control-token-that-is-longer-than-thirty-two-characters"


@pytest.fixture()
def control_server(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    config = repository / "config.yaml"
    config.write_text("project: test\n", encoding="utf-8")
    supervisor = RunPodSupervisor(
        repository_root=repository,
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        config_path=config,
        api_token=TOKEN,
        allow_updates=False,
        wandb=False,
    )
    server = create_control_server(supervisor, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield supervisor, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(
    url: str, *, token: str | None = None, method: str = "GET", payload=None
):
    headers = {}
    body = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, headers=headers, data=body, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.load(response)


def test_control_api_health_is_minimal_and_status_requires_bearer(control_server):
    _, base = control_server
    status, payload = _request(f"{base}/healthz")
    assert status == 200
    assert payload == {"format": "cftn_text_runpod_control_v1", "status": "ok"}
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/v1/status")
    assert error.value.code == 401
    status, payload = _request(f"{base}/v1/status", token=TOKEN)
    assert status == 200
    assert payload["updates_enabled"] is False
    assert payload["pipeline_process"]["running"] is False
    assert TOKEN not in json.dumps(payload)


def test_control_api_tails_only_allowlisted_artifact_logs(control_server):
    supervisor, base = control_server
    log = supervisor.artifact_root / "pipeline_logs" / "train_math.stdout.log"
    log.parent.mkdir(parents=True)
    log.write_text("one\ntwo\nthree\n", encoding="utf-8")
    status, payload = _request(
        f"{base}/v1/logs?stage=train_math&stream=stdout&lines=2", token=TOKEN
    )
    assert status == 200
    assert payload["text"] == "two\nthree\n"
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(f"{base}/v1/logs?stage=..%2F..&stream=stdout", token=TOKEN)
    assert error.value.code == 400


def test_remote_update_is_fail_closed_when_not_explicitly_enabled(control_server):
    _, base = control_server
    with pytest.raises(urllib.error.HTTPError) as error:
        _request(
            f"{base}/v1/actions/update-resume",
            token=TOKEN,
            method="POST",
            payload={"revision": "a" * 40, "resume": True},
        )
    assert error.value.code == 403


def test_update_rejects_non_commit_revisions_before_starting_a_job(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    config = repository / "config.yaml"
    config.write_text("project: test\n", encoding="utf-8")
    supervisor = RunPodSupervisor(
        repository_root=repository,
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        config_path=config,
        api_token=TOKEN,
        allow_updates=True,
        wandb=False,
    )
    with pytest.raises(ControlError, match="40-character"):
        supervisor.request_update_resume(
            revision="main; rm -rf /",
            expected_current_revision=None,
            resume=False,
        )
