from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cftn_text.gpt_receiver import (
    FrozenCausalLMTower,
    pretrained_dtype_kwargs,
    validate_dense_causal_lm_config,
)
from cftn_text.v1_3_config import audit_v1_2_pass, load_v1_3_config
from cftn_text.v1_3_data import generate_joint_record
from cftn_text.config import load_config
from cftn_text.v1_3_training import gpt_interface_config, hardening_acceptance
from tools.run_v2_experiment import Stage, _coordinator_preflight, _is_complete


ROOT = Path(__file__).parents[1]


class _FakeQwenBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden + 0.01,)


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
            hidden_size=8,
            num_hidden_layers=2,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        self.embed = nn.Embedding(32, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeQwenBlock(), _FakeQwenBlock()])
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids: torch.Tensor, **_: object):
        hidden = self.embed(input_ids)
        states = [hidden]
        for block in self.model.layers:
            hidden = block(hidden)[0]
            states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(states), logits=self.lm_head(hidden))


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


def test_v2_uses_fresh_training_and_reserves_twelve_slot_target(tmp_path):
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
    assert registry["maximum_slots"] == 12
    assert len(registry["reserved"]) == 10
    assert [item["name"] for item in registry["reserved"]][:3] == [
        "code",
        "formal_logic",
        "science",
    ]
    assert all(item["state"] == "reserved_inactive" for item in registry["reserved"])
    assert all(item["train"] is False for item in registry["reserved"])
    assert config["runtime"]["conditional_execution_in_hard_mode"] is True
    assert config["runtime"]["hard_halt_enabled"] is False
    assert config["integration_training"]["phases"][-1][
        "trainable_components"
    ] == ["wake_gates"]
    assert config["dispatcher"]["confidence_threshold"] == 0.90
    assert config["dispatcher"]["format"] == "cftn_text_v2_hierarchical_dispatcher_v2"
    assert config["dispatcher"]["model"]["parameter_target"] == 5_000_000
    assert config["dispatcher"]["model"]["active_tower_names"] == ["math", "string"]
    assert config["dispatcher"]["acceptance"][
        "minimum_registered_accuracy"
    ] == 1.0
    assert config["native_dispatch_evaluation"][
        "specialist_generation_policy"
    ] == "full_context_v1"
    assert config["native_dispatch_evaluation"]["examples_by_class"].keys() == {
        "explicit_math",
        "exact_string",
        "language_dependent_math",
        "multi_parallel",
        "multi_sequential",
    }
    assert config["gpt_interface"]["pure_language_prompt_style"] == (
        "open_world_generalist_v2"
    )


def test_v2_joint_generator_uses_registered_task_shares_exactly():
    config = load_v1_3_config(ROOT / "config" / "v2_multi_specialist.yaml")
    records = [
        generate_joint_record(
            seed=int(config["revision"]["seed"]),
            split="joint_train",
            index=index,
            config=config,
        )
        for index in range(40)
    ]
    counts = Counter(record["task_class"] for record in records)
    assert counts == {
        "pure_language": 6,
        "explicit_math": 6,
        "exact_string": 6,
        "language_dependent_math": 8,
        "multi_parallel": 7,
        "multi_sequential": 7,
    }
    pure = [record for record in records if record["task_class"] == "pure_language"]
    assert all(record["metadata"]["dispatch_intent"] == "pure_language" for record in pure)
    assert all("archival label" not in record["problem"].casefold() for record in pure)
    assert all(record["gpt_prompt"].startswith("Problem: ") for record in pure)


def test_v2_gpt_receivers_use_the_multi_specialist_bridge_dimensions():
    integration = load_v1_3_config(ROOT / "config" / "v2_multi_specialist.yaml")
    base = load_config(ROOT / "config" / "v2_broad_math.yaml")
    assert base["bridge"]["message_width"] != integration["bridge"]["message_width"]
    interface = gpt_interface_config(base, integration)
    assert interface["bridge"] == integration["bridge"]
    assert interface["bridge"]["message_width"] == 384
    assert base["bridge"]["message_width"] == 256


def test_v2_dense_qwen_block_layout_is_supported_and_base_is_frozen():
    tower = FrozenCausalLMTower(
        _FakeQwen(),
        [1],
        {
            "message_width": 8,
            "attention_heads": 2,
            "gate_hidden_size": 8,
            "dropout": 0.0,
            "gate_init": -2.0,
            "zero_init_output": True,
        },
    )
    assert tower.hidden_size == 8
    assert tower.receiver_layers == (1,)
    assert not any(parameter.requires_grad for parameter in tower.model.parameters())
    assert all(parameter.requires_grad for parameter in tower.receivers.parameters())
    tower.close()


def test_v2_dense_guard_rejects_moe_configuration():
    with pytest.raises(ValueError, match="must be dense"):
        validate_dense_causal_lm_config(
            SimpleNamespace(
                model_type="qwen3_moe",
                architectures=["Qwen3MoeForCausalLM"],
                hidden_size=2560,
                num_hidden_layers=36,
                num_experts=64,
            )
        )


def test_transformers_dtype_keyword_tracks_supported_major_versions():
    assert pretrained_dtype_kwargs("bfloat16", "4.51.3") == {
        "torch_dtype": torch.bfloat16
    }
    assert pretrained_dtype_kwargs("bfloat16", "5.15.1") == {
        "dtype": torch.bfloat16
    }


def test_v2_coordinator_preflight_checks_revision_and_chat_template(monkeypatch):
    config = load_config(ROOT / "config" / "v2_broad_math.yaml")
    revision = config["gpt"]["revision"]

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            assert model_name == "Qwen/Qwen3-4B-Instruct-2507"
            assert kwargs["revision"] == revision
            return SimpleNamespace(
                _commit_hash=revision,
                model_type="qwen3",
                architectures=["Qwen3ForCausalLM"],
                hidden_size=2560,
                num_hidden_layers=36,
            )

    class FakeTokenizer:
        eos_token_id = 151645
        chat_template = "registered"

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "CFTN preflight"}]
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            assert model_name == "Qwen/Qwen3-4B-Instruct-2507"
            assert kwargs["revision"] == revision
            return FakeTokenizer()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = FakeAutoConfig
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.__version__ = "test"
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    report = _coordinator_preflight(config)
    assert report["resolved_commit"] == revision
    assert report["dense"] is True
    assert report["hidden_size"] == 2560
    assert report["layers"] == 36
    assert report["chat_template_probe_tokens"] == 3
    assert report["weights_downloaded_by_preflight"] is False


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
        "optimizer_contract": {
            "group_names": ["gates"],
            "gate_only": True,
            "trainable_components": ["wake_gates"],
            "halt_gate_frozen": True,
        },
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
