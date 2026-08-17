from __future__ import annotations

from tools.check_v2_heartbeat import build_result, classify_snapshot


def _snapshot(**overrides):
    value = {
        "checked_unix": 1000.0,
        "pipeline_state": "running",
        "pipeline_pid_running": True,
        "pipeline": {"state": "running", "pid": 10},
        "stage": "train_math",
        "stage_index": 2,
        "stage_count": 17,
        "stage_record": {"started_unix": 900.0},
        "status": {"state": "running", "updated_unix": 999.0},
        "status_fresh": True,
        "status_age_seconds": 1.0,
        "stage_pid": 11,
        "stage_pid_running": True,
        "epoch": 4,
        "epoch_limit": 100,
        "global_step": 400,
        "batch_completed": 100,
        "batch_total": 1000,
        "metrics": {"train_loss_so_far": 0.5},
        "latest_metrics": {
            "epoch": 3,
            "validation": {"loss": 0.8},
        },
        "gpu": [],
        "stderr_tail": "",
    }
    value.update(overrides)
    return value


def test_heartbeat_prime_and_unchanged_snapshot_are_quiet():
    snapshot = _snapshot()
    events, state = classify_snapshot(snapshot, {}, now=1000.0, prime=True)
    assert events == []
    events, _ = classify_snapshot(snapshot, state, now=1010.0)
    assert events == []
    result = build_result(snapshot, events)
    assert result["decision"] == "DONT_NOTIFY"


def test_heartbeat_reports_stage_transition_and_epoch_milestone_once():
    snapshot = _snapshot()
    _, state = classify_snapshot(snapshot, {}, now=1000.0, prime=True)
    next_snapshot = _snapshot(
        stage="train_exact_string_specialist",
        stage_index=7,
        epoch=11,
        latest_metrics={"epoch": 10, "validation": {"loss": 0.4}},
    )
    events, next_state = classify_snapshot(next_snapshot, state, now=1100.0)
    identifiers = {event["id"] for event in events}
    assert any(value.startswith("stage-transition-") for value in identifiers)
    assert "epoch-milestone-train_exact_string_specialist-10" in identifiers
    repeated, _ = classify_snapshot(next_snapshot, next_state, now=1110.0)
    assert repeated == []


def test_heartbeat_reports_hardening_contract_and_routing_collapse():
    snapshot = _snapshot(
        stage="train_hardened_wake",
        stage_index=14,
        epoch=1,
        global_step=100,
        metrics={
            "learning_rates": {"gates": 1e-5},
            "hardening_policy": {
                "objective": "joint_task",
                "trainable_components": ["wake_gates", "halt_gate"],
                "halt_gate_trainable": True,
                "hard_halt_enabled": True,
                "conditional_specialist_execution": False,
            },
        },
        latest_metrics={
            "epoch": 1,
            "optimizer_group_names": ["bridges_and_receivers", "gates"],
            "trainable_parameter_names": ["wake_gates.network.1.weight", "halt_gate.network.1.weight"],
            "validation": {
                "loss": 2.0,
                "pure_language_false_wake_rate": 1.0,
                "exact_required_set_accuracy": 0.2,
                "wake_precision": 0.2,
                "wake_recall": 1.0,
                "all_open_rate": 1.0,
                "all_closed_rate": 0.0,
            },
        },
    )
    events, _ = classify_snapshot(snapshot, {}, now=1000.0, prime=True)
    messages = "\n".join(event["text"] for event in events)
    assert "contract violation" in messages
    assert "above 5e-7" in messages
    assert "appears collapsed" in messages
    assert "non-wake-gate" in messages


def test_heartbeat_reports_dead_pipeline_and_stalled_training():
    snapshot = _snapshot(pipeline_pid_running=False, status_age_seconds=1900.0)
    events, _ = classify_snapshot(
        snapshot, {}, now=3000.0, stale_after_seconds=1800.0, prime=True
    )
    messages = "\n".join(event["text"] for event in events)
    assert "pipeline PID is dead" in messages
    assert "has not advanced" in messages
