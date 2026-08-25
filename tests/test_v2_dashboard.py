from __future__ import annotations

import json

from tools.serve_v2_remote_dashboard import PAGE, REMOTE_PROBE, RemoteProbe


def test_dashboard_exposes_live_and_completed_validation_information():
    assert "Current step" in PAGE
    assert "Math curriculum" in PAGE
    assert "Training assessment" in PAGE
    assert "Acceptable to continue — watch validation" in PAGE
    assert "validation covers all difficulty levels" in PAGE
    assert "Validation trend" in PAGE
    assert "Latest validation breakdown" in PAGE
    assert "Generation validation" in PAGE
    assert "Raw artifacts for every stage" in PAGE
    assert "teacher_forced_token_accuracy" in PAGE
    assert "generation.accuracy" in PAGE
    assert "x!==null&&x!==undefined&&x!==''" in PAGE
    assert "refreshes every 30 seconds" in PAGE
    assert "setInterval(load,30000)" in PAGE


def test_remote_probe_collects_stage_metrics_and_redacts_sensitive_fields():
    assert '"stage_artifacts"' in REMOTE_PROBE
    assert 'read_jsonl(stage_root / "metrics.jsonl")' in REMOTE_PROBE
    assert '"data_manifest"' in REMOTE_PROBE
    assert '"training_contract"' in REMOTE_PROBE
    assert '"curriculum": config.get("data", {}).get("curriculum", {})' in REMOTE_PROBE
    assert '"minimum_epochs"' in REMOTE_PROBE
    assert '"[redacted]"' in REMOTE_PROBE
    assert 'name in sensitive_names or name.endswith(sensitive_suffixes)' in REMOTE_PROBE
    assert '"math_answer_recovery": "math_answer_recovery"' in REMOTE_PROBE
    assert '"recover_v2_math"' in REMOTE_PROBE
    assert 'recovery_root / "recovery_contract.json"' in REMOTE_PROBE
    assert 'artifact_root / "math_answer_recovery.stdout.log"' in REMOTE_PROBE


def test_remote_probe_returns_structured_ssh_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        import subprocess

        raise subprocess.CalledProcessError(1, "ssh", stderr="connection failed")

    monkeypatch.setattr("tools.serve_v2_remote_dashboard.subprocess.run", fail)
    result = RemoteProbe("example", 22, "identity", 1).fetch()

    assert result["format"] == "cftn_text_remote_dashboard_v2"
    assert "connection failed" in result["error"]
    json.dumps(result)
