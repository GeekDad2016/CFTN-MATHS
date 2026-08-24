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
from cftn_text.v1_3_answer_bus import (
    TypedAnswerComposer,
    compose_registered_answer,
    extract_answer_payload,
)
from cftn_text.v1_3_config import (
    V13PrerequisiteError,
    audit_v1_2_pass,
    load_v1_3_config,
)
from cftn_text.v1_3_data import (
    generate_joint_record,
    generate_string_record,
    joint_string_operation,
)
from cftn_text.v1_3_dataset import V13JointCollator, V13JointInferenceCollator
from cftn_text.v1_3_dispatch import (
    DISPATCH_INTENTS,
    DispatchError,
    compile_specialist_request,
    compile_v1_3_intent,
    compose_dispatch_results,
    dispatch_intent_from_plan,
    dispatch_v1_3_prompt,
)
from cftn_text.v1_3_learned_dispatch import (
    ByteIntentClassifier,
    LearnedV13Dispatcher,
    encode_dispatch_prompts,
    load_learned_dispatcher,
    save_learned_dispatcher_checkpoint,
)
from cftn_text.v1_3_evaluation import (
    _arm_metrics,
    _oracle_specialist_items,
    extract_completion_answer,
    generate_joint_batch,
    resolve_specialist_generation_budget,
)
from cftn_text.v1_3_model import V13MultiTowerModel
from cftn_text.v1_3_training import (
    _limit_records_by_class,
    _repair_sequential_orders,
    adapter_recovery_acceptance,
    apply_protocol_aware_adapter_metrics,
    integration_selection_score,
    v1_3_objective,
)
from cftn_text.v1_3_reporting import _weighted_accuracy
from tools.run_v1_3_experiment import Stage, _completion_is_valid, command_plan
from tools.wait_then_run_v1_3 import pipeline_command
from tools.recover_v1_3_hard_binary import select_adapter_continuation_source
from tools.recover_v1_3_fusion import PHASE_NAME, configure_fusion_recovery
from tools.recover_v1_3_answer_bus import (
    PHASE_NAME as ANSWER_BUS_PHASE_NAME,
    configure_answer_bus_recovery,
)
from tools.continue_v1_3_fusion import (
    PHASE_NAME as FUSION_CONTINUATION_PHASE_NAME,
    configure_fusion_continuation,
    evaluate_late_improvement,
)
from tools.continue_v1_3_native_answer_bus import audit_validation_class_coverage


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
    parallel = records[8]
    expected_reversal = parallel["gpt_target"].split("|", 1)[1]
    assert parallel["metadata"]["string"]["operation"] == "reverse"
    assert parallel["metadata"]["string"]["value"] == expected_reversal
    assert joint_string_operation(parallel) == "reverse"
    pure = records[0]
    assert pure["gpt_prompt"].endswith("Requested archival label:")
    assert pure["gpt_target"] == "amber"
    assert pure["gpt_answer_protocol"] == "first_nonempty_completion_line_v1"
    sequential = records[-1]
    assert sequential["required_specialists_by_round"][:2] == [
        ["math"],
        ["string"],
    ]
    reverse_order = generate_joint_record(
        seed=719, split="joint_train", index=19, config=config
    )
    assert reverse_order["required_specialists_by_round"][:2] == [
        ["string"],
        ["math"],
    ]
    assert sequential["specialist_targets_by_round"]["math"][0] is not None
    assert sequential["specialist_targets_by_round"]["string"][1] is not None
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
    assert "zero-based indexing" in string_items[0]["problem"]


def test_joint_string_operation_repairs_legacy_metadata_from_target():
    record = generate_joint_record(
        seed=719, split="joint_validation", index=8, config=_config()
    )
    record["metadata"]["string"].pop("operation", None)
    record["metadata"]["string"]["value"] = "stale-count-or-index-value"
    assert joint_string_operation(record) == "reverse"

    exact = generate_joint_record(
        seed=719, split="joint_validation", index=4, config=_config()
    )
    exact["metadata"]["string"].pop("operation", None)
    assert joint_string_operation(exact) in {"count", "index", "reverse"}


def test_answer_payload_extraction_is_lossless_and_fail_closed():
    assert extract_answer_payload("trace<answer>a|b c</answer>") == "a|b c"
    with pytest.raises(ValueError, match="exactly one"):
        extract_answer_payload("<answer>a</answer><answer>b</answer>")
    assert extract_answer_payload("malformed", strict=False) is None


def test_registered_answer_bus_has_a_perfect_deterministic_composer_ceiling():
    config = _config()
    for index in range(12):
        record = generate_joint_record(
            seed=719, split="joint_validation", index=index, config=config
        )
        composed = compose_registered_answer(record)
        if record["task_class"] == "pure_language":
            assert composed is None
        else:
            assert composed == record["gpt_target"]


def test_joint_collator_can_supply_a_typed_native_answer_bus_override():
    config = _config()
    record = generate_joint_record(
        seed=719, split="joint_validation", index=2, config=config
    )
    specialist = record["required_specialists_by_round"][0][0]
    record["answer_bus_override"] = [
        {"round": 0, "specialist": specialist, "payload": "native-near-miss"}
    ]
    collator = V13JointCollator(
        ByteMathTokenizer(),
        ByteExternalTokenizer(),
        maximum_gpt_length=config["data"]["maximum_gpt_length"],
        maximum_specialist_length=config["data"]["maximum_specialist_length"],
        maximum_rounds=config["runtime"]["maximum_callosal_rounds"],
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    override = collator([record])["answer_bus_override"]
    active = override["attention_mask"][0]
    assert ByteMathTokenizer().decode(override["token_ids"][0][active].tolist()) == "native-near-miss"
    assert override["round_ids"][0][active].eq(0).all()


def test_native_answer_bus_holdout_requires_protocol_validation_classes():
    records = [{"task_class": name} for name in (
        "pure_language",
        "explicit_math",
        "exact_string",
        "language_dependent_math",
        "multi_parallel",
        "multi_sequential",
    )]
    counts = audit_validation_class_coverage(records)
    assert counts["pure_language"] == 1
    with pytest.raises(RuntimeError, match="pure_language"):
        audit_validation_class_coverage(records[1:])
    with pytest.raises(RuntimeError, match="differs from calibration"):
        audit_validation_class_coverage(
            records, expected_pure_language_examples=1000
        )


def test_typed_answer_composer_has_finite_pointer_copy_gradients():
    torch.manual_seed(29)
    tokenizer = ByteMathTokenizer()
    composer = TypedAnswerComposer(
        prompt_width=16,
        hidden_size=16,
        specialist_count=2,
        maximum_rounds=2,
        attention_heads=4,
        decoder_layers=1,
        dropout=0.0,
        maximum_source_positions=32,
        maximum_target_positions=16,
    )
    source = tokenizer.encode("abc|cba")
    target = tokenizer.encode("cba")
    source_ids = torch.tensor([source], dtype=torch.long)
    source_mask = torch.ones_like(source_ids)
    result = composer(
        prompt_context=torch.randn(1, 16),
        source_token_ids=source_ids,
        source_attention_mask=source_mask,
        source_specialist_ids=torch.zeros_like(source_ids),
        source_round_ids=torch.zeros_like(source_ids),
        source_position_ids=torch.arange(len(source)).unsqueeze(0),
        decoder_input_ids=torch.tensor(
            [[tokenizer.bos_token_id, *target]], dtype=torch.long
        ),
        decoder_attention_mask=torch.ones(1, len(target) + 1, dtype=torch.long),
    )
    labels = torch.tensor([[*target, tokenizer.eos_token_id]], dtype=torch.long)
    loss = torch.nn.functional.nll_loss(
        result.log_probabilities.flatten(0, 1), labels.flatten()
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert composer.pointer_query.weight.grad is not None
    assert bool(composer.pointer_query.weight.grad.abs().sum().gt(0))


def test_recovery_derivation_balances_preserved_sequential_records():
    config = _config()
    records = [
        generate_joint_record(seed=719, split="joint_train", index=index, config=config)
        for index in range(100)
    ]
    # Simulate the preserved parity-bug manifest.
    for index, record in enumerate(records):
        if record["task_class"] == "multi_sequential":
            records[index] = generate_joint_record(
                seed=719,
                split="joint_train",
                index=(index // 10) * 20 + 19,
                config=config,
            )
    repaired, report = _repair_sequential_orders(
        records, config=config, split="joint_train"
    )
    assert len(repaired) == len(records)
    assert report["sequential_orders"] == {
        "math_then_string": 5,
        "string_then_math": 5,
    }


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
    assert batch["wake_targets"][0].tolist() == [[1.0, 0.0], [0.0, 1.0]]
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
    assert output.rounds[0].wake_activations.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    assert output.rounds[1].wake_activations.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    output.loss.backward()
    assert all(not parameter.requires_grad for parameter in model.specialists.parameters())
    assert any(
        parameter.grad is not None for parameter in model.request_bridges.parameters()
    )


def test_hardened_wake_trains_only_wake_gates():
    model, batch = _tiny_model_and_batch()
    model.set_trainable_phase("hardened_wake")
    model.train()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("wake_gates.") for name in trainable)
    assert not any(name.startswith("halt_gate.") for name in trainable)
    assert not any(
        parameter.requires_grad for parameter in model.request_bridges.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.return_bridges.parameters()
    )
    assert model.wake_gates.training is True
    assert model.request_bridges.training is False
    assert model.return_bridges.training is False
    assert model.specialist_receivers.training is False
    assert model.gpt_tower.receivers.training is False

    output, loss, components = v1_3_objective(
        model,
        batch,
        wake_mode="hard_straight_through",
        maximum_rounds=2,
        settings=_config()["integration_training"],
        global_step=0,
    )
    assert torch.allclose(loss, output.wake_loss)
    assert components["routing_calibration_only"] == 1.0
    assert components["auxiliary_step"] == 0.0


def test_recovery_phases_separate_adapters_and_router():
    model, _ = _tiny_model_and_batch()
    model.set_trainable_phase("oracle_hard_adapter_recovery")
    adapter_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert adapter_names
    assert all(
        name.startswith(
            (
                "request_bridges.",
                "return_bridges.",
                "specialist_receivers.",
                "gpt_tower.receivers.",
            )
        )
        for name in adapter_names
    )
    model.set_trainable_phase("hard_router_recovery")
    router_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert router_names
    assert all(
        name.startswith(("wake_gates.", "wake_round_embeddings."))
        for name in router_names
    )
    assert not any(name.startswith("halt_gate.") for name in router_names)


def test_adapter_continuation_uses_weighted_task_loss_and_adapter_parameters():
    model, batch = _tiny_model_and_batch()
    model.set_trainable_phase("oracle_hard_adapter_continuation")
    continuation_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert continuation_names
    assert all(
        name.startswith(
            (
                "request_bridges.",
                "return_bridges.",
                "specialist_receivers.",
                "gpt_tower.receivers.",
            )
        )
        for name in continuation_names
    )
    output, loss, components = v1_3_objective(
        model,
        batch,
        wake_mode="oracle",
        maximum_rounds=2,
        settings=_config()["integration_training"],
        global_step=1,
        objective_mode="oracle_hard_adapter",
        conditional_execution=True,
        apply_halt=False,
        task_class_weights={"pure_language": 0.5, "multi_parallel": 3.0},
    )
    assert torch.isfinite(loss)
    assert components["weighted_gpt_loss"] != pytest.approx(
        float(output.gpt_loss.detach())
    )
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.request_bridges.parameters()
        if parameter.requires_grad
    )


def test_collaboration_checkpoint_excludes_all_frozen_tower_weights():
    model, _batch = _tiny_model_and_batch()
    state = model.collaboration_state_dict()
    assert state
    assert all(name.startswith(model._collaboration_prefixes()) for name in state)
    assert not any(name.startswith("gpt_tower.model.") for name in state)
    assert not any(name.startswith("specialists.") for name in state)


def test_fusion_recovery_trains_only_fusion_and_gpt_receivers():
    model, batch = _tiny_model_and_batch()
    legacy_state = {
        name: value
        for name, value in model.collaboration_state_dict().items()
        if not name.startswith("message_fusion.")
    }
    model.load_collaboration_state_dict(legacy_state, strict=True)
    model.set_trainable_phase("oracle_hard_fusion_recovery")
    model.train()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("message_fusion.", "gpt_tower.receivers."))
        for name in trainable
    )
    assert any(name.startswith("message_fusion.") for name in trainable)
    assert any(name.startswith("gpt_tower.receivers.") for name in trainable)
    assert model.message_fusion.training is True
    assert model.gpt_tower.receivers.training is True
    assert model.request_bridges.training is False
    assert model.return_bridges.training is False
    assert model.specialist_receivers.training is False

    _output, loss, components = v1_3_objective(
        model,
        batch,
        wake_mode="oracle",
        maximum_rounds=2,
        settings=_config()["integration_training"],
        global_step=1,
        objective_mode="oracle_hard_fusion",
        conditional_execution=True,
        apply_halt=False,
        task_class_weights={"pure_language": 0.5, "multi_parallel": 3.0},
    )
    assert torch.isfinite(loss)
    assert components["oracle_hard_fusion"] == 1.0
    loss.backward()
    assert model.message_fusion.output_projection.weight.grad is not None
    assert not any(
        parameter.grad is not None for parameter in model.return_bridges.parameters()
    )


def test_answer_bus_recovery_trains_only_composer():
    model, batch = _tiny_model_and_batch()
    model.set_trainable_phase("oracle_hard_answer_bus_recovery")
    model.train()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("answer_composer.") for name in trainable)
    assert model.answer_composer.training is True
    assert model.message_fusion.training is False
    assert model.gpt_tower.receivers.training is False

    _output, loss, components = v1_3_objective(
        model,
        batch,
        wake_mode="oracle",
        maximum_rounds=2,
        settings=_config()["integration_training"],
        global_step=1,
        objective_mode="oracle_hard_answer_bus",
        conditional_execution=True,
        apply_halt=False,
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.abs().sum().gt(0))
    }
    assert gradients
    assert all(name.startswith("answer_composer.") for name in gradients)
    assert components["oracle_hard_answer_bus"] == pytest.approx(1.0)


def test_fusion_continuation_trains_only_fusion_and_gpt_receivers():
    model, _batch = _tiny_model_and_batch()
    model.set_trainable_phase(FUSION_CONTINUATION_PHASE_NAME)
    model.train()
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("message_fusion.", "gpt_tower.receivers."))
        for name in trainable
    )
    assert model.message_fusion.training is True
    assert model.gpt_tower.receivers.training is True
    assert model.request_bridges.training is False
    assert model.return_bridges.training is False
    assert model.specialist_receivers.training is False


def test_recovery_subset_is_deterministic_and_class_stratified():
    records = [
        {"record_id": f"math-{index}", "task_class": "explicit_math"}
        for index in range(7)
    ] + [
        {"record_id": f"parallel-{index}", "task_class": "multi_parallel"}
        for index in range(9)
    ]
    first, report = _limit_records_by_class(
        records,
        {"explicit_math": 3, "multi_parallel": 5},
        seed=719,
    )
    second, second_report = _limit_records_by_class(
        list(reversed(records)),
        {"explicit_math": 3, "multi_parallel": 5},
        seed=719,
    )
    assert [record["record_id"] for record in first] == [
        record["record_id"] for record in second
    ]
    assert report["selected_counts"] == {
        "explicit_math": 3,
        "multi_parallel": 5,
    }
    assert report["selected_record_ids_sha256"] == second_report[
        "selected_record_ids_sha256"
    ]


def test_fusion_recovery_configuration_pins_optimizer_and_source(tmp_path: Path):
    config = copy.deepcopy(_config())
    source = tmp_path / "source.pth"
    source.write_bytes(b"preserved fusion source")
    phase = configure_fusion_recovery(config, source)
    assert phase["name"] == PHASE_NAME
    assert phase["objective_mode"] == "oracle_hard_fusion"
    assert phase["wake_mode"] == "oracle"
    assert phase["conditional_execution"] is True
    assert phase["apply_halt"] is False
    assert phase["zero_update_candidate"] is True
    assert phase["fusion_learning_rate"] == pytest.approx(1.0e-4)
    assert phase["receiver_learning_rate"] == pytest.approx(5.0e-7)
    assert phase["source_checkpoint_sha256"] == file_sha256(source)
    assert phase["train_examples_by_class"]["multi_parallel"] == 10_000
    assert config["integration_training"]["phases"][-1] is phase


def test_answer_bus_recovery_configuration_is_isolated_and_fail_closed(tmp_path: Path):
    config = copy.deepcopy(_config())
    source = tmp_path / "source.pth"
    source.write_bytes(b"protected fusion epoch four")
    phase = configure_answer_bus_recovery(config, source)
    assert phase["name"] == ANSWER_BUS_PHASE_NAME
    assert phase["objective_mode"] == "oracle_hard_answer_bus"
    assert phase["selection_mode"] == "answer_bus_recovery"
    assert phase["minimum_epochs"] == 10
    assert phase["max_epochs"] == 50
    assert phase["source_checkpoint_sha256"] == file_sha256(source)
    assert phase["zero_update_candidate"] is False
    assert set(phase["train_task_classes"]) == {
        "explicit_math",
        "exact_string",
        "language_dependent_math",
        "multi_parallel",
        "multi_sequential",
    }
    assert config["answer_composer"]["maximum_target_positions"] == 48
    assert config["integration_training"]["phases"][-1] is phase


def test_fusion_continuation_configuration_guarantees_fifty_epochs(tmp_path: Path):
    config = copy.deepcopy(_config())
    source = tmp_path / "source.pth"
    source.write_bytes(b"accepted fusion source")
    phase = configure_fusion_continuation(config, source)
    assert phase["name"] == FUSION_CONTINUATION_PHASE_NAME
    assert phase["minimum_epochs"] == 50
    assert phase["max_epochs"] == 100
    assert phase["early_stop_patience"] == 5
    assert phase["selection_min_delta"] == pytest.approx(2.0e-4)
    assert phase["fusion_learning_rate"] == pytest.approx(2.5e-5)
    assert phase["receiver_learning_rate"] == pytest.approx(1.25e-7)
    assert phase["source_checkpoint_sha256"] == file_sha256(source)


def test_fusion_continuation_requires_meaningful_late_improvement():
    def row(epoch: int, selection: float) -> dict:
        return {
            "epoch": epoch,
            "selection_metric": selection,
            "checkpoint_eligible": True,
            "hardening_acceptance": {"gates": {"pass": True}},
        }

    metrics = [row(0, 1.0), row(4, 1.0025), row(5, 1.0026)]
    too_small = evaluate_late_improvement(
        metrics,
        {"best_metrics": row(5, 1.0026)},
        minimum_delta=2.0e-4,
    )
    assert too_small["triggered"] is False

    improved = evaluate_late_improvement(
        [*metrics, row(6, 1.0030)],
        {"best_metrics": row(6, 1.0030)},
        minimum_delta=2.0e-4,
    )
    assert improved["triggered"] is True
    assert improved["selection_delta"] == pytest.approx(5.0e-4)


def test_protocol_aware_adapter_selection_rewards_real_focus_improvement():
    calibration = {
        "answer_protocol": "first_nonempty_completion_line_v1",
        "examples": 20,
        "semantic_accuracy": 1.0,
        "revision_sha256": "revision",
    }

    def validation(*, focus_token: float, causal_gap: float) -> dict:
        value = {
            "examples": 100,
            "gpt_teacher_forced_sequence_accuracy": 0.63,
            "gpt_teacher_forced_token_accuracy": 0.70 + focus_token / 100,
            "gpt_teacher_forced_loss": 2.0 - focus_token / 10,
            "causal_message_loss_gap": causal_gap,
            "task_class_metrics": {
                "pure_language": {
                    "examples": 20,
                    "sequence_accuracy": 0.0,
                    "token_accuracy": 0.5,
                    "gpt_loss": 3.0,
                },
                "explicit_math": {"examples": 16, "sequence_accuracy": 1.0},
                "language_dependent_math": {
                    "examples": 16,
                    "sequence_accuracy": 1.0,
                },
                "multi_sequential": {"examples": 16, "sequence_accuracy": 1.0},
                "exact_string": {
                    "examples": 16,
                    "sequence_accuracy": 0.6,
                    "token_accuracy": 0.45 + focus_token,
                },
                "multi_parallel": {
                    "examples": 16,
                    "sequence_accuracy": 0.1,
                    "token_accuracy": 0.60 + focus_token,
                },
            },
        }
        return apply_protocol_aware_adapter_metrics(value, calibration)

    earlier = validation(focus_token=0.0, causal_gap=9.0)
    later = validation(focus_token=0.03, causal_gap=5.5)
    assert earlier["protocol_semantic_sequence_accuracy_lower_bound"] == pytest.approx(
        0.83
    )
    assert integration_selection_score(
        later, selection_mode="adapter_recovery"
    ) > integration_selection_score(earlier, selection_mode="adapter_recovery")
    acceptance = adapter_recovery_acceptance(
        later,
        {
            "minimum_protocol_semantic_sequence_accuracy": 0.83,
            "minimum_causal_message_loss_gap": 5.0,
        },
    )
    assert acceptance["gates"]["pass"] is True


def test_continuation_source_selector_prefers_late_valid_improvement(tmp_path: Path):
    config = copy.deepcopy(_config())
    config["paths"]["artifact_root"] = str(tmp_path)
    adapter = tmp_path / "oracle_hard_adapter_recovery"
    adapter.mkdir()
    rows = []
    for epoch, sequence, token, loss, gap in (
        (1, 0.6360, 0.7000, 3.8, 7.2),
        (2, 0.6360, 0.7110, 3.6, 7.0),
        (3, 0.6366, 0.7132, 3.5, 4.0),
    ):
        (adapter / f"checkpoint_epoch_{epoch:04d}.pth").write_bytes(
            f"checkpoint-{epoch}".encode()
        )
        rows.append(
            json.dumps(
                {
                    "epoch": epoch,
                    "checkpoint_eligible": True,
                    "validation": {
                        "gpt_teacher_forced_sequence_accuracy": sequence,
                        "gpt_teacher_forced_token_accuracy": token,
                        "loss": loss,
                        "causal_message_loss_gap": gap,
                    },
                }
            )
        )
    (adapter / "metrics.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = select_adapter_continuation_source(config)
    assert report["selected"]["epoch"] == 2
    assert Path(report["selected"]["checkpoint"]).is_file()


def test_hard_all_closed_matches_receiver_bypassed_path_exactly():
    model, batch = _tiny_model_and_batch()
    model.eval()
    with torch.no_grad():
        for parameter in model.wake_gates.parameters():
            parameter.zero_()
        model.wake_gates.network[-1].bias.fill_(-10.0)
        hard = model(
            batch,
            wake_mode="hard",
            maximum_rounds=2,
            conditional_execution=True,
            apply_halt=False,
        )
        bypassed = model(
            batch,
            wake_mode="hard",
            maximum_rounds=2,
            conditional_execution=True,
            apply_halt=False,
            disable_all_communication=True,
        )
    assert torch.equal(hard.gpt_logits, bypassed.gpt_logits)


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


def test_hard_halt_is_separate_and_disabled_by_default():
    model, batch = _tiny_model_and_batch()
    model.eval()
    with torch.no_grad():
        model.wake_gates.network[-1].weight.zero_()
        model.wake_gates.network[-1].bias.fill_(10.0)
        model.halt_gate.network[-1].weight.zero_()
        model.halt_gate.network[-1].bias.fill_(10.0)
        default_output = model(batch, wake_mode="hard", maximum_rounds=2)
        halted_output = model(
            batch, wake_mode="hard", maximum_rounds=2, apply_halt=True
        )
    assert default_output.rounds[1].wake_activations.eq(1).all()
    assert halted_output.rounds[0].wake_activations.eq(1).all()
    assert halted_output.rounds[1].wake_activations.eq(0).all()


def test_conditional_execution_preserves_autocast_output_dtype():
    model, batch = _tiny_model_and_batch()
    specialist_batch = batch["specialists"]["math"][0]
    activation = torch.tensor([1.0, 0.0])
    request = torch.zeros(2, 2, model.request_bridges["math"].message_width)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        direct = model.specialists["math"](
            specialist_batch["input_ids"][:1],
            specialist_batch["attention_mask"][:1],
            specialist_batch["prefix_lengths"][:1],
            message=request[:1],
            receivers=model.specialist_receivers["math"],
            receive_enabled=True,
            gate_mode=model.gate_mode,
        )
        output = model._run_specialist(
            "math",
            specialist_batch,
            request,
            activation,
            conditional_execution=True,
        )
    assert output.logits.dtype == direct.logits.dtype
    assert output.hidden_states.dtype == direct.hidden_states.dtype
    assert output.answer_logits.dtype == direct.answer_logits.dtype


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


def test_v1_3_generation_rejects_unknown_lossless_request_mode():
    model, batch = _tiny_model_and_batch()
    with pytest.raises(ValueError, match="unsupported lossless request mode"):
        generate_joint_batch(
            model,
            batch,
            ByteMathTokenizer(),
            ByteExternalTokenizer(),
            wake_mode="oracle",
            maximum_rounds=1,
            max_specialist_new_tokens=1,
            max_gpt_new_tokens=1,
            lossless_request_mode="unknown",
        )


def test_v1_3_typed_dispatcher_fails_closed_in_generation():
    model, batch = _tiny_model_and_batch()
    batch["records"] = [
        {"problem": "an unregistered request"} for _ in batch["records"]
    ]
    results = generate_joint_batch(
        model,
        batch,
        ByteMathTokenizer(),
        ByteExternalTokenizer(),
        wake_mode="hard",
        maximum_rounds=1,
        max_specialist_new_tokens=1,
        max_gpt_new_tokens=1,
        lossless_request_mode="typed_dispatcher_no_latent",
    )
    assert all(result["generation"] == "" for result in results)
    assert all(result["dispatch_plan"] is None for result in results)
    assert all(result["dispatch_error"] for result in results)


@pytest.mark.parametrize("index", range(10))
def test_v1_3_dispatcher_compiles_registered_prompt_without_oracle_metadata(index: int):
    config = _config()
    record = generate_joint_record(
        seed=int(config["revision"]["seed"]),
        split="joint_validation",
        index=index,
        config=config,
    )
    plan = dispatch_v1_3_prompt(record["problem"])
    actual_routes = [
        [
            specialist
            for specialist in ("math", "string")
            if plan.call_for(round_index, specialist) is not None
        ]
        for round_index in range(int(config["runtime"]["maximum_callosal_rounds"]))
    ]
    assert actual_routes == record["required_specialists_by_round"]
    results: dict[str, str] = {}
    for call in sorted(plan.calls, key=lambda value: value.round_index):
        request = compile_specialist_request(plan, call, results)
        expected = record["specialist_oracle_problems_by_round"][call.specialist][
            call.round_index
        ]
        assert request == expected
        target = record["specialist_targets_by_round"][call.specialist][
            call.round_index
        ]
        payload = extract_answer_payload(target, strict=True)
        assert payload is not None
        results[call.result_id] = payload
    assert compose_dispatch_results(plan, results) == (
        record["gpt_target"] if plan.calls else None
    )
    serialized = json.dumps(plan.to_dict(), sort_keys=True)
    assert "specialist_oracle" not in serialized
    assert "gpt_target" not in serialized


def test_v1_3_dispatcher_supports_both_sequential_dependency_orders():
    config = _config()
    plans = []
    for index in (9, 19):
        record = generate_joint_record(
                seed=int(config["revision"]["seed"]),
            split="joint_validation",
            index=index,
            config=config,
        )
        plans.append(dispatch_v1_3_prompt(record["problem"]))
    assert {plan.plan_kind for plan in plans} == {
        "math_then_string",
        "string_then_math",
    }
    for plan in plans:
        second = next(call for call in plan.calls if call.round_index == 1)
        with pytest.raises(DispatchError, match="unavailable"):
            compile_specialist_request(plan, second, {})


@pytest.mark.parametrize(
    "split",
    (
        "joint_train",
        "joint_validation",
        "joint_heldout_paraphrase",
        "joint_extrapolation",
        "joint_counterfactual",
        "joint_unseen_composition",
    ),
)
def test_v1_3_dispatcher_covers_registered_split_grammars(split: str):
    config = _config()
    for index in range(20):
        record = generate_joint_record(
            seed=int(config["revision"]["seed"]),
            split=split,
            index=index,
            config=config,
        )
        plan = dispatch_v1_3_prompt(record["problem"])
        results: dict[str, str] = {}
        for call in sorted(plan.calls, key=lambda value: value.round_index):
            assert compile_specialist_request(plan, call, results) == (
                record["specialist_oracle_problems_by_round"][call.specialist][
                    call.round_index
                ]
            )
            target = record["specialist_targets_by_round"][call.specialist][
                call.round_index
            ]
            payload = extract_answer_payload(target, strict=True)
            assert payload is not None
            results[call.result_id] = payload
        assert compose_dispatch_results(plan, results) == (
            record["gpt_target"] if plan.calls else None
        )


def test_v1_3_dispatcher_fails_closed_on_unknown_prompt():
    with pytest.raises(DispatchError, match="outside"):
        dispatch_v1_3_prompt("Please maybe do some mathematics with an unknown format.")


def test_v1_3_inference_collator_requires_no_oracle_or_answer_fields():
    config = _config()
    record = generate_joint_record(
        seed=int(config["revision"]["seed"]),
        split="joint_validation",
        index=8,
        config=config,
    )
    runtime_record = {
        key: record[key]
        for key in ("schema_version", "record_id", "problem", "gpt_prompt")
    }
    collator = V13JointInferenceCollator(
        ByteMathTokenizer(),
        ByteExternalTokenizer(),
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    batch = collator([runtime_record])
    assert not batch["wake_targets"].bool().any()
    assert not batch["answer_composer_eligible"].any()
    assert batch["records"] == [runtime_record]
    assert batch["task_classes"] == ["unknown"]
    for rounds in batch["specialists"].values():
        for specialist_batch in rounds:
            assert (specialist_batch["labels"] == -100).all()
            assert specialist_batch["answer_ids"].shape == (1, 0)


@pytest.mark.parametrize("index", range(20))
def test_v1_3_constrained_learned_intent_compiler_preserves_exact_operands(index: int):
    config = _config()
    record = generate_joint_record(
        seed=int(config["revision"]["seed"]),
        split="joint_heldout_paraphrase",
        index=index,
        config=config,
    )
    oracle_plan = dispatch_v1_3_prompt(record["problem"])
    intent = dispatch_intent_from_plan(oracle_plan)
    assert intent in DISPATCH_INTENTS
    plan = compile_v1_3_intent(record["problem"], intent)
    actual_routes = [
        [
            specialist
            for specialist in ("math", "string")
            if plan.call_for(round_index, specialist) is not None
        ]
        for round_index in range(int(config["runtime"]["maximum_callosal_rounds"]))
    ]
    assert actual_routes == record["required_specialists_by_round"]
    results: dict[str, str] = {}
    for call in sorted(plan.calls, key=lambda value: value.round_index):
        request = compile_specialist_request(plan, call, results)
        assert request
        target = record["specialist_targets_by_round"][call.specialist][
            call.round_index
        ]
        payload = extract_answer_payload(target, strict=True)
        assert payload is not None
        results[call.result_id] = payload
    assert compose_dispatch_results(plan, results) == (
        record["gpt_target"] if plan.calls else None
    )


def test_v1_3_learned_intent_compiler_handles_unregistered_math_wording():
    prompt = "Could you determine the integer when -7 times it plus 13 reaches -29?"
    plan = compile_v1_3_intent(prompt, "single_math")
    call = plan.call_for(0, "math")
    assert call is not None
    assert compile_specialist_request(plan, call, {}) == (
        "For an integer x, -7 times x together with 13 gives -29. Determine x."
    )


def test_v1_3_learned_intent_compiler_rejects_unsupported_intent():
    with pytest.raises(DispatchError, match="rejected an unsupported prompt"):
        compile_v1_3_intent("Sort 'cab' alphabetically.", "unsupported")


def test_v1_3_learned_dispatcher_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(719)
    prompts = [
        "Solve 2*x + (3) = 7. Return x.",
        "Reverse 'abcdef'.",
    ]
    input_ids, attention_mask = encode_dispatch_prompts(prompts, maximum_length=128)
    model = ByteIntentClassifier(embedding_size=8, channels=8, kernels=(3,), dropout=0.0)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, len(DISPATCH_INTENTS))
    checkpoint = tmp_path / "dispatcher.pth"
    save_learned_dispatcher_checkpoint(
        checkpoint,
        model,
        maximum_length=128,
        confidence_threshold=0.0,
        metadata={"test": True},
    )
    loaded = load_learned_dispatcher(checkpoint)
    assert loaded.predict_intents(prompts) == LearnedV13Dispatcher(
        model,
        maximum_length=128,
        confidence_threshold=0.0,
    ).predict_intents(prompts)


def test_v1_3_learned_dispatcher_encoding_hides_operand_values():
    prompts = [
        "Resolve 4*x+(16)=56 and flip 'abcdef'.",
        "Resolve -14*x+(11)=151 and flip 'zyx'.",
    ]
    input_ids, attention_mask = encode_dispatch_prompts(
        prompts, maximum_length=128
    )
    assert torch.equal(input_ids[0], input_ids[1])
    assert torch.equal(attention_mask[0], attention_mask[1])

    label_ids, label_mask = encode_dispatch_prompts(
        [
            "The registry name amber is requested.",
            "The registry name cobalt is requested.",
        ],
        maximum_length=128,
    )
    assert torch.equal(label_ids[0], label_ids[1])
    assert torch.equal(label_mask[0], label_mask[1])


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
