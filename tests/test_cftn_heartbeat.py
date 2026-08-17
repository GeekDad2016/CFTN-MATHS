from __future__ import annotations

from tools.check_cftn_heartbeat import classify_v1_3


def _snapshot(**overrides):
    value = {
        "pipeline_state": "running",
        "pipeline_pid": 1,
        "pipeline_pid_running": True,
        "stage": "train_hardened_wake",
        "stage_index": 10,
        "stage_count": 12,
        "status": {"state": "running"},
        "status_age_seconds": 1.0,
        "stage_pid": 2,
        "stage_pid_running": True,
        "epoch": 1,
        "global_step": 400,
        "batch_completed": 400,
        "batch_total": 8334,
        "metrics": {
            "learning_rates": {"gates": 5e-7},
            "train": {
                "total_loss": 0.2,
                "wake_loss": 0.2,
                "routing_calibration_only": 1.0,
                "auxiliary_step": 0.0,
            },
        },
        "latest_metrics": None,
        "gpu": [],
        "stderr_tail": "",
    }
    value.update(overrides)
    return value


def test_v1_3_event_monitor_is_quiet_when_primed_and_healthy():
    snapshot = _snapshot()
    events, state = classify_v1_3(snapshot, {}, now=1000.0, prime=True)
    assert events == []
    events, _ = classify_v1_3(snapshot, state, now=1010.0)
    assert events == []


def test_v1_3_event_monitor_detects_loss_contract_failure():
    snapshot = _snapshot(
        metrics={
            "learning_rates": {"gates": 1e-6},
            "train": {
                "total_loss": 1.2,
                "wake_loss": 0.2,
                "routing_calibration_only": 0.0,
                "auxiliary_step": 1.0,
            },
        }
    )
    events, _ = classify_v1_3(snapshot, {}, now=1000.0, prime=True)
    messages = "\n".join(event["text"] for event in events)
    assert "routing_calibration_only" in messages
    assert "differs from wake loss" in messages
    assert "exceeds 5e-7" in messages


def test_v1_3_event_monitor_reports_failed_validation_guard():
    snapshot = _snapshot(
        latest_metrics={
            "epoch": 1,
            "trainable_parameter_names": ["wake_gates.network.1.weight"],
            "validation": {
                "exact_required_set_accuracy": 0.8,
                "wake_precision": 0.95,
                "wake_recall": 0.99,
                "pure_language_false_wake_rate": 0.0,
            },
        }
    )
    events, _ = classify_v1_3(snapshot, {}, now=1000.0, prime=True)
    assert any("exact_required_set_accuracy" in event["text"] for event in events)
