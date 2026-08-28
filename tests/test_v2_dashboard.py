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
    assert "How to read validation" in PAGE
    assert "Current-phase learning" in PAGE
    assert "Retention versus source baseline" in PAGE
    assert "Future curriculum probes" in PAGE
    assert "Global teacher-forced diagnostics" in PAGE
    assert "Generation examples" in PAGE
    assert "Current phase learned; retention below floor" in PAGE
    assert "Exact full trace" in PAGE
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
    assert "Random scratch initialization" in PAGE
    assert "Cumulative replay contract" in PAGE
    assert "Other skill buckets/epoch" in PAGE
    assert "competency_gated_v" in PAGE
    assert "Zero-update phase entrance" in PAGE
    assert "Preservation KL" in PAGE
    assert "DeepMind numeric" in PAGE
    assert "Validation in progress" in PAGE
    assert "live_generation_validation" in PAGE
    assert 'class="summary-grid"' in PAGE
    assert ".summary-grid{grid-template-columns:1fr" in PAGE
    assert "repeat(2,minmax(0,1fr))" not in PAGE
    assert '"math_full_supervision*"' in REMOTE_PROBE
    assert '"math_competency_curriculum*"' in REMOTE_PROBE
    assert '"math_scratch_curriculum*"' in REMOTE_PROBE
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
    assert 'read_json(stage_root / "retention_baseline.json")' in REMOTE_PROBE
    assert '"retention_baseline": retention_baseline' in REMOTE_PROBE
    assert '"retention_baseline": recovery_contract.get("retention_baseline")' in REMOTE_PROBE
    assert 'read_json(stage_root / "entrance_evaluations.json")' in REMOTE_PROBE
    assert '"live_generation_validation": live_generation_validation' in REMOTE_PROBE
    assert 'generation_validation_{name}_epoch_{current_epoch:04d}.jsonl' in REMOTE_PROBE
    assert '"entrance_evaluations": entrance_evaluations' in REMOTE_PROBE
    assert '"preservation_distillation": recovery_contract.get("preservation_distillation")' in REMOTE_PROBE
    assert '"initialization": recovery_contract.get("initialization")' in REMOTE_PROBE


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
                            capture_output=True, text=True, encoding="utf-8", check=True)
    assert "render checks passed" in result.stdout


def test_competency_dashboard_separates_phase_retention_future_and_global_metrics():
    node = shutil.which("node")
    if not node:
        pytest.skip("JavaScript rendering check requires Node; acceptance executes on RunPod with Node")
    phase = {
        "name": "verified_integer_foundations",
        "minimum_epochs": 2,
        "maximum_epochs": 8,
        "advance_after_consecutive_passes": 2,
        "primary_generation_panel": "school",
        "minimum_generation_accuracy": 0.99,
        "minimum_valid_rate": 1.0,
        "minimum_generation_accuracy_by_panel": {"generated_foundations": 0.9, "wording": 0.9},
        "minimum_valid_rate_by_panel": {"generated_foundations": 0.99, "wording": 0.99},
        "quota_groups": [],
    }
    panels = {
        "school": {"examples": 320, "accuracy": 1.0, "valid_rate": 1.0},
        "generated_foundations": {"examples": 384, "accuracy": 1.0, "valid_rate": 1.0},
        "wording": {"examples": 320, "accuracy": 0.92, "valid_rate": 1.0},
        "generated_rational": {"examples": 384, "accuracy": 0.08, "valid_rate": 1.0,
                               "failure_examples": [{"problem": "<img src=x>", "expected_answer": "1", "generation": "bad"}]},
        "broad": {"examples": 512, "accuracy": 0.18, "valid_rate": 1.0,
                  "by_source": {"cftn_generated": {"examples": 60, "accuracy": 0.8},
                                "deepmind_mathematics": {"examples": 452, "accuracy": 0.1}},
                  "by_difficulty": {"1": {"examples": 10, "accuracy": 1.0},
                                    "2": {"examples": 472, "accuracy": 0.14},
                                    "3": {"examples": 30, "accuracy": 0.1}},
                  "by_family": {"comparison__pair": {"examples": 10, "accuracy": 0.4},
                                "algebra__polynomial_roots": {"examples": 10, "accuracy": 0.1}}},
    }
    checks = {
        "broad_retention": {"observed": 0.18, "minimum": 0.263, "pass": False},
        "primary_generation_accuracy": {"observed": 1.0, "minimum": 0.99, "pass": True},
        "primary_valid_rate": {"observed": 1.0, "minimum": 1.0, "pass": True},
        "panel:generated_foundations:generation_accuracy": {"observed": 1.0, "minimum": 0.9, "pass": True},
        "panel:generated_foundations:valid_rate": {"observed": 1.0, "minimum": 0.99, "pass": True},
        "panel:wording:generation_accuracy": {"observed": 0.92, "minimum": 0.9, "pass": True},
        "panel:wording:valid_rate": {"observed": 1.0, "minimum": 0.99, "pass": True},
    }
    breakdowns = {
        "by_source": {"cftn_generated": {"examples": 5000, "language_loss": 2.4,
                                          "teacher_forced_token_accuracy": 0.77,
                                          "teacher_forced_sequence_accuracy": 0.0},
                      "deepmind_mathematics": {"examples": 7000, "language_loss": 0.4,
                                               "teacher_forced_token_accuracy": 0.88,
                                               "teacher_forced_sequence_accuracy": 0.14}},
        "by_difficulty": {"1": {"examples": 834, "language_loss": 2.2,
                                  "teacher_forced_token_accuracy": 0.82,
                                  "teacher_forced_sequence_accuracy": 0.0}},
        "by_family": {"algebra__polynomial_roots": {"examples": 146, "language_loss": 0.7,
                                                      "teacher_forced_token_accuracy": 0.8,
                                                      "teacher_forced_sequence_accuracy": 0.0}},
    }
    row = {"epoch": 3, "global_step": 30000, "train_loss": 0.2, "learning_rate": 1e-5,
           "validation": {"loss": 1.1, "teacher_forced_token_accuracy": 0.84,
                          "teacher_forced_sequence_accuracy": 0.08, "generation_panels": panels,
                          "breakdowns": breakdowns},
           "competency_curriculum_state": {"phase_index": 0, "phase_epoch": 3, "consecutive_passes": 0},
           "curriculum_acceptance": {"pass": False, "phase": phase["name"], "phase_epoch": 3,
                                     "primary_panel": "school", "checks": checks},
           "curriculum_transition": {"advance": False, "complete": False, "failed": False,
                                     "phase": phase["name"], "phase_epoch": 3, "maximum_epochs": 8,
                                     "consecutive_passes": 0, "required_consecutive_passes": 2}}
    fixture = {
        "updated_unix": 2000,
        "pipeline": {"state": "running", "current_stage": "math_competency_curriculum_v3", "stages": {}},
        "processes": [{"pid": 1, "command": "python trainer"}], "logs": {}, "gpu": {"gpus": []},
        "training_contract": {"competency_curriculum": True, "full_supervision": True,
                              "validation_examples": 12000,
                              "curriculum": {"transition_policy": "competency_gated_v2",
                                             "examples_per_epoch": 100000, "phases": [phase]},
                              "preservation_distillation": {"enabled": True, "weight": 0.1,
                                                              "sources": ["deepmind_mathematics", "gsm8k"]},
                              "math_training": {}},
        "current_stage_artifact": {"status": {"state": "training", "epoch": 4, "metrics": {}},
                                   "metrics": [row],
                                   "entrance_evaluations": {
                                       "evaluations": [{"phase": phase["name"],
                                                        "zero_optimizer_updates": True,
                                                        "skipped": False,
                                                        "acceptance": {"pass": False}}],
                                       "resulting_curriculum_state": {"phase_index": 0}},
                                   "retention_baseline": {"accuracy": 0.293,
                                                          "by_source": {"cftn_generated": {"examples": 60, "accuracy": 0.9},
                                                                        "deepmind_mathematics": {"examples": 452, "accuracy": 0.21}},
                                                          "by_difficulty": {"1": {"examples": 10, "accuracy": 1.0},
                                                                            "2": {"examples": 472, "accuracy": 0.23},
                                                                            "3": {"examples": 30, "accuracy": 1.0}},
                                                          "by_family": {"comparison__pair": {"examples": 10, "accuracy": 0.78},
                                                                        "algebra__polynomial_roots": {"examples": 10, "accuracy": 0.0}}}},
        "data_manifest": {"splits": {"validation": {"count": 12000}}}, "stage_artifacts": {},
        "checkpoints": [], "wandb": {}, "disk": [],
    }
    script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0].replace("load();setInterval(load,30000)", "")
    bootstrap = "const elements={}; const document={getElementById:x=>elements[x]||(elements[x]={innerHTML:'',textContent:''})};\n"
    checks_js = """
const assert=require('node:assert/strict');
draw(fixture);
assert.ok(elements.summary.innerHTML.includes('Latest phase answers'));
assert.ok(elements.assessment.innerHTML.includes('Current phase learned; retention below floor'));
assert.ok(elements.phaseValidation.innerHTML.includes('generated foundations'));
assert.ok(!elements.phaseValidation.innerHTML.includes('generated rational'));
assert.ok(elements.retentionValidation.innerHTML.includes('Retention below floor'));
assert.ok(elements.retentionValidation.innerHTML.includes('Source baseline'));
assert.ok(elements.futureValidation.innerHTML.includes('generated rational'));
assert.ok(elements.futureValidation.innerHTML.includes('Diagnostic only'));
assert.ok(elements.breakdowns.innerHTML.includes('12,000'));
assert.ok(elements.breakdowns.innerHTML.includes('Exact full trace'));
assert.ok(elements.curriculum.innerHTML.includes('Zero-update phase entrance'));
assert.ok(elements.curriculum.innerHTML.includes('Frozen-source preservation'));
assert.ok(elements.curriculum.innerHTML.includes('First unmet phase'));
assert.ok(!elements.generation.innerHTML.includes('<img'));
assert.ok(elements.generation.innerHTML.includes('&lt;img'));
console.log('competency dashboard render checks passed');
"""
    result = subprocess.run([node], input=bootstrap + script + "\nconst fixture=" + json.dumps(fixture) + ";\n" + checks_js,
                            capture_output=True, text=True, encoding="utf-8", check=True)
    assert "competency dashboard render checks passed" in result.stdout
