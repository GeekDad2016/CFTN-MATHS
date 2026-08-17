from __future__ import annotations

import copy
import json
from pathlib import Path

from cftn_text.v1_3_config import audit_v1_2_pass, load_v1_3_config
from cftn_text.config import load_config
from cftn_text.v1_3_training import gpt_interface_config, hardening_acceptance
from tools.run_v2_experiment import Stage, _is_complete


ROOT = Path(__file__).parents[1]


def _validation(**overrides):
    value = {
        "pure_language_false_wake_rate": 0.0,
        "exact_required_set_accuracy": 0.96,
        "wake_precision": 0.97,
        "wake_recall": 0.98,
        "gpt_teacher_forced_sequence_accuracy": 0.80,
        "gpt_teacher_forced_token_accuracy": 0.85,
        "all_open_rate": 0.0,
        "all_closed_rate": 0.0,
        "causal_message_loss_gap": 1.0,
        "loss": 0.5,
    }
    value.update(overrides)
    return value


def test_v2_uses_fresh_training_and_reserves_one_inactive_specialist(tmp_path):
    config = copy.deepcopy(
        load_v1_3_config(ROOT / "config" / "v2_multi_specialist.yaml")
    )
    config["paths"]["prior_v1_2_report"] = str(tmp_path / "missing-v1-2.json")
    config["paths"]["prior_v1_3_report"] = str(tmp_path / "missing-v1-3.json")
    audit = audit_v1_2_pass(config)
    assert audit["state"] == "passed"
    assert audit["prior_reports_gate_training"] is False
    assert audit["bridge_initialization"] == (
        "fresh_contextual_bridges_zero_initialized_receivers"
    )
    registry = config["specialist_registry"]
    assert [item["name"] for item in registry["active"]] == ["math", "string"]
    assert registry["reserved"] == [
        {
            "name": "extension_1",
            "state": "reserved_inactive",
            "train": False,
            "capability": "pending_user_selection",
            "dataset": "pending_user_selection",
            "note": "excluded_from_wake_targets_optimizers_checkpoints_and_compute",
        }
    ]
    assert config["runtime"]["conditional_execution_in_hard_mode"] is True


def test_v2_gpt_receivers_use_the_multi_specialist_bridge_dimensions():
    integration = load_v1_3_config(ROOT / "config" / "v2_multi_specialist.yaml")
    base = load_config(ROOT / "config" / "v2_broad_math.yaml")
    assert base["bridge"]["message_width"] != integration["bridge"]["message_width"]
    interface = gpt_interface_config(base, integration)
    assert interface["bridge"] == integration["bridge"]
    assert interface["bridge"]["message_width"] == 384
    assert base["bridge"]["message_width"] == 256


def test_v2_hardening_guard_rejects_always_open_collapse():
    config = load_v1_3_config(ROOT / "config" / "v2_multi_specialist.yaml")
    baseline = {"hard_metrics": _validation(exact_required_set_accuracy=0.92)}
    healthy = hardening_acceptance(
        _validation(), baseline, config["integration_training"]
    )
    assert healthy["gates"]["pass"] is True
    collapsed = hardening_acceptance(
        _validation(
            pure_language_false_wake_rate=1.0,
            exact_required_set_accuracy=0.20,
            wake_precision=0.20,
            all_open_rate=1.0,
        ),
        baseline,
        config["integration_training"],
    )
    assert collapsed["gates"]["pass"] is False
    assert collapsed["collapse_guard"]["triggered"] is True
    assert "not_always_open" in collapsed["collapse_guard"]["failed"]


def test_v2_hard_checkpoint_completion_requires_gate_only_contract(tmp_path):
    path = tmp_path / "summary.json"
    summary = {
        "state": "completed",
        "optimizer_contract": {"group_names": ["gates"], "gate_only": True},
        "final_metrics": {"hardening_acceptance": {"gates": {"pass": True}}},
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert _is_complete(Stage("train_hardened_wake", [], path), {})
    summary["optimizer_contract"]["group_names"] = [
        "bridges_and_receivers",
        "gates",
    ]
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert not _is_complete(Stage("train_hardened_wake", [], path), {})
