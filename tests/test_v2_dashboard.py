from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tools.serve_v2_remote_dashboard import PAGE, REMOTE_PROBE, RemoteProbe


def test_dashboard_exposes_live_and_completed_validation_information():
    assert "Current step" in PAGE
    assert "Math curriculum" in PAGE
    assert "Training assessment" in PAGE
    assert "Acceptable to continue — watch validation" in PAGE
    assert "validation covers all difficulty levels" in PAGE
    assert "Validation trend" in PAGE
    assert "Latest acceptance gate" in PAGE
    assert "Acceptance gates" in PAGE
    assert "checkpoint not eligible" in PAGE
    assert "Awaiting second validation" in PAGE
    assert "Latest validation breakdown" in PAGE
    assert "Generation validation" in PAGE
    assert "Acceptance panels" in PAGE
    assert "generation_panels" in PAGE
    assert "Raw artifacts for every stage" in PAGE
    assert "teacher_forced_token_accuracy" in PAGE
    assert "generation.accuracy" in PAGE
    assert "toExponential(3)" in PAGE
    assert "<circle" in PAGE
    assert "x!==null&&x!==undefined&&x!==''" in PAGE
    assert "refreshes every 30 seconds" in PAGE
    assert "setInterval(load,30000)" in PAGE
    assert "Full repaired math curriculum" in PAGE
    assert "Other skill buckets/epoch" in PAGE
    assert "competency_gated_v1" in PAGE
    assert "DeepMind numeric" in PAGE
    assert '"math_full_supervision*"' in REMOTE_PROBE
    assert '"math_competency_curriculum*"' in REMOTE_PROBE
    assert '"train_v2_full_supervision"' in REMOTE_PROBE
    assert '"run_v2_math_curriculum"' in REMOTE_PROBE


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
    assert '"math_shared_trace_recovery": "math_shared_trace_recovery"' in REMOTE_PROBE
    assert '"math_broad_shared_recovery": "math_broad_shared_recovery"' in REMOTE_PROBE
    assert '"math_capacity_recovery": "math_capacity_recovery"' in REMOTE_PROBE
    assert '"math_capacity_recovery*"' in REMOTE_PROBE
    assert "stage_directories.setdefault(candidate_root.name, candidate_root.name)" in REMOTE_PROBE
    assert 'candidate_stage = candidate_root.name' in REMOTE_PROBE
    assert 'data/manifests/v2_broad_math_400k_r4' in REMOTE_PROBE
    assert '"recover_v2_math"' in REMOTE_PROBE
    assert 'recovery_root / "recovery_contract.json"' in REMOTE_PROBE
    assert 'artifact_root.glob(f"{stage}*.stdout.log")' in REMOTE_PROBE


def test_remote_probe_returns_structured_ssh_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        import subprocess

        raise subprocess.CalledProcessError(1, "ssh", stderr="connection failed")

    monkeypatch.setattr("tools.serve_v2_remote_dashboard.subprocess.run", fail)
    result = RemoteProbe("example", 22, "identity", 1).fetch()

    assert result["format"] == "cftn_text_remote_dashboard_v2"
    assert "connection failed" in result["error"]
    json.dumps(result)


def test_school_probe_keeps_completed_trial_and_partial_live_metrics(tmp_path, monkeypatch, capsys):
    root = tmp_path / "artifacts"
    old = root / "math_capacity_recovery_r3"
    trial = root / "math_verified_school_trial_v1"
    old.mkdir(parents=True)
    trial.mkdir()
    (old / "recovery_contract.json").write_text('{"repository_revision":"old"}')
    (old / "status.json").write_text('{"state":"training"}')
    (trial / "contract.json").write_text(json.dumps({"format": "cftn_v2_verified_school_trial_v1", "revision": "tested", "settings": {"epochs": 3}, "api_key": "must-not-leak"}))
    (trial / "status.json").write_text('{"state":"completed","epochs":3}')
    (trial / "epoch_reports.json").write_text('[{"epoch":3,"band":"foundations","curriculum_gate":{"pass":false}}]')
    (trial / "metrics.json").write_text('[{"epoch":3,"global_step":3072,"loss_training_average":0.2}]')
    (trial / "summary.json").write_text('{"state":"completed","trial_only":true,"production_acceptance":false,"source_preserved":true}')
    baseline = trial / "epoch_000"
    baseline.mkdir()
    (baseline / "validation.json").write_text('{"current/addition":{"accuracy":0.1}}')
    current = trial / "epoch_003"
    current.mkdir()
    (current / "validation.json").write_text('{"current/addition":{"accuracy":0.7}}')
    (root / (trial.name + ".stdout.log")).write_text("current school log")
    source = REMOTE_PROBE.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    source = source.replace('Path("/workspace/cftn-text/artifacts/v2_broad_math_400k_r4")', f"Path({str(root)!r})")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: SimpleNamespace(stdout=""))
    exec(compile(source, "dashboard_probe", "exec"), {})
    result = json.loads(capsys.readouterr().out)
    assert result["pipeline"]["current_stage"] == trial.name
    assert result["pipeline"]["state"] == "completed"
    assert result["school_trial"]["epochs"][0]["curriculum_gate"]["pass"] is False
    assert result["school_trial"]["training_metrics"][0]["global_step"] == 3072
    assert result["school_trial"]["live_validation"]["current/addition"]["accuracy"] == .7
    assert result["school_trial"]["contract"]["api_key"] == "[redacted]"
    assert result["logs"]["stdout"]["text"] == "current school log"


def test_school_dashboard_javascript_renders_completed_and_stale_partial_states():
    node = shutil.which("node")
    if not node:
        pytest.skip("JavaScript rendering check requires Node; acceptance executes on RunPod with Node")
    families = ["addition", "subtraction", "multiplication", "division", "linear_equation"]
    panels = {"current/" + f: {"examples": 64, "accuracy": .5, "trace_exact_rate": .45, "valid_rate": 1., "budget_hits": 0} for f in families}
    trial = {"name": "math_verified_school_trial_v1", "status": {"state": "completed", "epochs": 3},
             "status_modified_unix": 1000, "contract": {"settings": {"families": families, "epochs": 3, "bands": ["foundations", "two_digit", "three_digit"],
              "examples_per_epoch": 16384, "replay_fraction": .25, "minimum_epochs_per_band": 2,
              "band_gate": {"answer_accuracy": .99, "valid_rate": 1., "trace_exact_rate": .95, "maximum_replay_drop": .03}}},
             "baseline": panels, "epochs": [{"epoch": 3, "band": "foundations", "validation": panels, "training_loss": .2,
              "curriculum_gate": {"pass": False, "gates": {f: False for f in families}}}],
             "nonexact_examples": [{"problem": "<img src=x onerror=alert(1)>", "generation": "<answer>bad</answer>"}]}
    fixture = {"school_trial": trial, "updated_unix": 2000, "pipeline": {}, "processes": [], "logs": {}}
    script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0].replace("load();setInterval(load,30000)", "")
    bootstrap = "const elements={}; const document={getElementById:x=>elements[x]||(elements[x]={innerHTML:'',textContent:''})};\n"
    checks = """
const assert=require('node:assert/strict');
draw(fixture);
assert.ok(elements.trend.innerHTML.includes('50.00%'));
assert.ok(elements.acceptance.innerHTML.includes('Trial completion is not acceptance'));
assert.ok(elements.acceptance.innerHTML.includes('MISS'));
assert.ok(elements.assessment.innerHTML.includes('All three epochs finished'));
assert.ok(!elements.generation.innerHTML.includes('<img'));
assert.ok(elements.generation.innerHTML.includes('&lt;img'));
fixture.school_trial.status={state:'evaluating',epoch:1,panel:'wording/division'};
fixture.school_trial.epochs=[];
fixture.school_trial.live_validation=fixture.school_trial.baseline;
fixture.school_trial.process_observation=[{wait_channel:'request_wait_answer',persistent_output_open:true}];
fixture.processes=[{command:'python -m tools.train_v2_verified_school run'}];
draw(fixture);
assert.ok(elements.summary.innerHTML.includes('Waiting on persistent storage'));
assert.ok(elements.trend.innerHTML.includes('partial validation'));
assert.ok(elements.trend.innerHTML.includes('Pending'));
assert.ok(elements.stamp.textContent.includes('refreshes every 30 seconds'));
console.log('school dashboard render checks passed');
"""
    result = subprocess.run([node], input=bootstrap + script + "\nconst fixture=" + json.dumps(fixture) + ";\n" + checks,
                            capture_output=True, text=True, check=True)
    assert "render checks passed" in result.stdout
