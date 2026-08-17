from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from cftn_text.data_generator import file_sha256
from cftn_text.gpt_receiver import FrozenGPT2Tower
from cftn_text.math_tower import MathTower
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.v1_3_config import (
    V13PrerequisiteError,
    audit_v1_2_pass,
    load_v1_3_config,
)
from cftn_text.v1_3_data import generate_joint_record, generate_string_record
from cftn_text.v1_3_dataset import V13JointCollator
from cftn_text.v1_3_evaluation import (
    _arm_metrics,
    _oracle_specialist_items,
    extract_completion_answer,
    resolve_specialist_generation_budget,
)
from cftn_text.v1_3_model import V13MultiTowerModel
from cftn_text.v1_3_reporting import _weighted_accuracy
from tools.run_v1_3_experiment import Stage, _completion_is_valid, command_plan
from tools.wait_then_run_v1_3 import pipeline_command


ROOT = Path(__file__).parents[1]


class ByteExternalTokenizer:
    pad_token_id = ByteMathTokenizer.pad_token_id
    eos_token_id = ByteMathTokenizer.eos_token_id

    def encode(self, text: str, **_kwargs) -> list[int]:
        return [ByteMathTokenizer.byte_offset + value for value in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True) -> str:
        del skip_special_tokens
        return bytes(
            int(value) - ByteMathTokenizer.byte_offset
            for value in ids
            if int(value) >= ByteMathTokenizer.byte_offset
        ).decode("utf-8", errors="replace")


def _config() -> dict:
    return load_v1_3_config(ROOT / "config" / "v1_3_multi_specialist.yaml")


def test_v1_3_prerequisite_requires_sealed_pass_and_checkpoint_hash(tmp_path: Path):
    config = copy.deepcopy(_config())
    report_path = tmp_path / "v1_2_report.json"
    status_path = tmp_path / "v1_2_status.json"
    checkpoint = tmp_path / "bridge.pth"
    checkpoint.write_bytes(b"sealed checkpoint")
    config["paths"]["v1_2_report"] = str(report_path)
    config["paths"]["v1_2_pipeline_status"] = str(status_path)
    with pytest.raises(V13PrerequisiteError, match="missing"):
        audit_v1_2_pass(config)
    revision = config["prerequisite"]["v1_2_revision_sha256"]
    required_stages = config["prerequisite"]["required_completed_stages"]
    status_path.write_text(
        json.dumps(
            {
                "format": "cftn_text_v1_2_pipeline_status_v1",
                "state": "completed",
                "revision_sha256": revision,
                "current_stage": None,
                "stage_index": len(required_stages),
                "stages_total": len(required_stages),
                "completed_stages": required_stages,
            }
        ),
        encoding="utf-8",
    )
    gates = {"communication": True, "no_harm": True, "pass": True}
    report = {
        "format": "cftn_text_v1_2_revision_report_v1",
        "revision_sha256": revision,
        "final_gates": gates,
        "training": {
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_sha256": file_sha256(checkpoint),
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert audit_v1_2_pass(config)["state"] == "passed"
    incomplete = json.loads(status_path.read_text(encoding="utf-8"))
    incomplete["current_stage"] = "assemble_v1_2_evidence"
    status_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(V13PrerequisiteError, match="active stage"):
        audit_v1_2_pass(config)
    incomplete["current_stage"] = None
    status_path.write_text(json.dumps(incomplete), encoding="utf-8")
    report["final_gates"] = {"communication": False, "pass": False}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(V13PrerequisiteError, match="did not pass"):
        audit_v1_2_pass(config)


def test_v1_3_data_is_deterministic_balanced_and_sequential():
    config = _config()
    first = generate_string_record(
        seed=719, split="string_test", index=17, config=config
    )
    second = generate_string_record(
        seed=719, split="string_test", index=17, config=config
    )
    assert first == second
    records = [
        generate_joint_record(seed=719, split="joint_train", index=index, config=config)
        for index in range(10)
    ]
    assert [record["task_class"] for record in records] == [
        "pure_language",
        "pure_language",
        "explicit_math",
        "explicit_math",
        "exact_string",
        "exact_string",
        "language_dependent_math",
        "language_dependent_math",
        "multi_parallel",
        "multi_sequential",
    ]
    pure = records[0]
    assert pure["gpt_prompt"].endswith("Requested archival label:")
    assert pure["gpt_target"] == "amber"
    assert pure["gpt_answer_protocol"] == "first_nonempty_completion_line_v1"
    sequential = records[-1]
    assert sequential["required_specialists_by_round"][:2] == [
        ["string"],
        ["math"],
    ]
    assert sequential["specialist_targets_by_round"]["string"][0] is not None
    assert sequential["specialist_targets_by_round"]["math"][1] is not None
    assert sequential["halt_round"] == 2
    for specialist in ("math", "string"):
        prompts = sequential["specialist_oracle_problems_by_round"][specialist]
        targets = sequential["specialist_targets_by_round"][specialist]
        assert [prompt is not None for prompt in prompts] == [
            target is not None for target in targets
        ]
    math_items, math_locations = _oracle_specialist_items([sequential], "math")
    string_items, string_locations = _oracle_specialist_items([sequential], "string")
    assert len(math_items) == len(math_locations) == 1
    assert len(string_items) == len(string_locations) == 1
    assert "For an integer x" in math_items[0]["problem"]
    assert "How many times" in string_items[0]["problem"]


def test_joint_collator_keeps_raw_prompt_out_of_specialist_workspace():
    config = copy.deepcopy(_config())
    config["runtime"]["maximum_callosal_rounds"] = 2
    record = generate_joint_record(
        seed=719, split="joint_train", index=9, config=config
    )
    collator = V13JointCollator(
        ByteMathTokenizer(),
        ByteExternalTokenizer(),
        maximum_gpt_length=256,
        maximum_specialist_length=256,
        maximum_rounds=2,
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    batch = collator([record])
    tokenizer = ByteMathTokenizer()
    for name in ("math", "string"):
        item = batch["specialists"][name][0]
        prefix_length = int(item["prefix_lengths"][0])
        prefix = tokenizer.decode(item["input_ids"][0, :prefix_length].tolist())
        assert config["runtime"]["neutral_workspaces"][name] in prefix
        assert record["problem"] not in prefix
    assert batch["wake_targets"][0].tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert batch["halt_targets"][0].tolist() == [0.0, 1.0]
    gpt_prefix = collator.gpt_tokenizer.decode(
        batch["gpt_prepass_input_ids"][0].tolist()
    )
    assert gpt_prefix.endswith("Exact result:")
    labelled = batch["gpt_labels"][0].ne(-100)
    completion = collator.gpt_tokenizer.decode(
        batch["gpt_input_ids"][0][labelled].tolist()
    )
    assert completion.startswith(record["gpt_target"] + "\n")


def test_v1_3_completion_parser_separates_semantics_from_tag_format():
    assert extract_completion_answer("  amber\nColour: blue") == "amber"
    assert extract_completion_answer("<answer>amber</answer> trailing") == "amber"
    assert extract_completion_answer("\n\n") is None


def test_v1_3_stale_manifest_does_not_skip_data_preparation(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"revision_sha256": "old-revision"}), encoding="utf-8"
    )
    stage = Stage("prepare_v1_3_data", ["python"], manifest)
    assert not _completion_is_valid(stage, "new-revision")


def _tiny_model_and_batch():
    from transformers import GPT2Config, GPT2LMHeadModel

    config = copy.deepcopy(_config())
    config["bridge"].update(
        {
            "message_tokens": 2,
            "message_width": 16,
            "attention_heads": 4,
            "gate_hidden_size": 16,
            "dropout": 0.0,
        }
    )
    config["runtime"]["maximum_callosal_rounds"] = 2
    tower_config = {
        "layers": 1,
        "hidden_size": 16,
        "attention_heads": 4,
        "feed_forward_size": 32,
        "dropout": 0.0,
        "max_sequence_length": 256,
        "receiver_layers": [0],
        "answer_min": -5,
        "answer_max": 5,
        "answer_head_mode": "disabled",
    }
    gpt_model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=ByteMathTokenizer.vocab_size,
            n_positions=256,
            n_ctx=256,
            n_embd=16,
            n_layer=1,
            n_head=4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=ByteMathTokenizer.bos_token_id,
            eos_token_id=ByteMathTokenizer.eos_token_id,
            pad_token_id=ByteMathTokenizer.pad_token_id,
        )
    )
    model = V13MultiTowerModel(
        gpt_tower=FrozenGPT2Tower(gpt_model, [0], config["bridge"]),
        specialists={
            "math": MathTower(tower_config, ByteMathTokenizer.vocab_size),
            "string": MathTower(tower_config, ByteMathTokenizer.vocab_size),
        },
        config=config,
    )
    records = [
        generate_joint_record(seed=719, split="joint_train", index=index, config=config)
        for index in (0, 9)
    ]
    batch = V13JointCollator(
        ByteMathTokenizer(),
        ByteExternalTokenizer(),
        maximum_gpt_length=256,
        maximum_specialist_length=256,
        maximum_rounds=2,
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )(records)
    return model, batch


def test_v1_3_multi_tower_forward_backward_and_independent_wakes():
    torch.manual_seed(13)
    model, batch = _tiny_model_and_batch()
    model.set_trainable_phase("dense_recurrent")
    output = model(batch, wake_mode="oracle", maximum_rounds=2)
    assert output.gpt_logits.shape[:2] == batch["gpt_input_ids"].shape
    assert output.rounds[0].wake_activations.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    assert output.rounds[1].wake_activations.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    output.loss.backward()
    assert all(not parameter.requires_grad for parameter in model.specialists.parameters())
    assert any(
        parameter.grad is not None for parameter in model.request_bridges.parameters()
    )


def test_hardened_wake_freezes_bridges_receivers_and_specialists():
    model, _ = _tiny_model_and_batch()
    model.set_trainable_phase("hardened_wake")
    model.train()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith("wake_gates.") or name.startswith("halt_gate.")
        for name in trainable
    )
    assert not any(
        parameter.requires_grad for parameter in model.request_bridges.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.return_bridges.parameters()
    )
    assert model.wake_gates.training is True
    assert model.halt_gate.training is True
    assert model.request_bridges.training is False
    assert model.return_bridges.training is False
    assert model.specialist_receivers.training is False
    assert model.gpt_tower.receivers.training is False


def test_configured_hard_mode_physically_skips_closed_specialists():
    model, batch = _tiny_model_and_batch()
    model.config["runtime"]["conditional_execution_in_hard_mode"] = True
    with torch.no_grad():
        for parameter in model.wake_gates.parameters():
            parameter.zero_()
        model.wake_gates.network[-1].bias.fill_(-10.0)
    for tower in model.specialists.values():
        tower.reset_execution_count()
    model(batch, wake_mode="hard", maximum_rounds=1)
    assert all(tower.execution_count == 0 for tower in model.specialists.values())


def test_v1_3_pipeline_is_ordered_and_guarded():
    path = ROOT / "config" / "v1_3_multi_specialist.yaml"
    config = load_v1_3_config(path)
    stages = command_plan(str(path), config, device="cuda", wandb=True)
    assert [stage.name for stage in stages] == [
        "audit_v1_2_pass",
        "prepare_v1_3_data",
        "calibrate_frozen_gpt_language",
        "train_exact_string_specialist",
        "seal_native_specialists",
        "train_single_specialist_capacity",
        "train_dense_mixed_messages",
        "train_dense_recurrent",
        "train_supervised_soft_wake",
        "train_hardened_wake",
        "evaluate_sealed_causal_suite",
        "assemble_v1_3_evidence",
    ]
    assert stages[0].completion_path.name == "prerequisites.json"
    assert stages[4].command[-2:] == [
        "--specialist-generation-policy",
        "full_context_v1",
    ]
    assert "full_context_v1" in stages[10].command
    assert all(
        not str(stage.completion_path).startswith(config["paths"]["v1_2_artifact_root"])
        for stage in stages
    )
    command = pipeline_command(path, device="cuda", wandb=True)
    assert "--execute" in command
    assert "--wandb" in command


def test_v1_3_full_context_generation_policy_uses_tower_limit():
    config = _config()
    tower = type("Tower", (), {"max_sequence_length": 256})()
    assert resolve_specialist_generation_budget(config, tower, "configured") == 96
    assert resolve_specialist_generation_budget(config, tower, "full_context_v1") == 256
    with pytest.raises(ValueError, match="unsupported specialist generation policy"):
        resolve_specialist_generation_budget(config, tower, "unknown")


def test_v1_3_primary_metrics_are_competence_conditioned():
    rows = [
        {
            "generation": "7\nAdditional continuation text",
            "correct": True,
            "oracle_specialist_capable": True,
        },
        {
            "generation": "<answer>0</answer>",
            "correct": False,
            "oracle_specialist_capable": False,
        },
    ]
    metrics = _arm_metrics(rows)
    assert metrics["exact_accuracy"] == 0.5
    assert metrics["competence_supported"] == {
        "examples": 1,
        "coverage": 0.5,
        "exact_accuracy": 1.0,
        "valid_rate": 1.0,
    }
    split = {
        "arms": {
            "learned": {
                "by_task_class": {
                    "language_dependent_math": metrics,
                }
            }
        }
    }
    assert _weighted_accuracy(
        split, "learned", ("language_dependent_math",)
    ) == 0.5
    assert _weighted_accuracy(
        split,
        "learned",
        ("language_dependent_math",),
        competence_supported=True,
    ) == 1.0
