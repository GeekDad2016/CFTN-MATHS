from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cftn_text.checkpoint import append_jsonl, atomic_json_dump
from cftn_text.config import canonical_json
from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_answer_bus import (
    compose_registered_answer,
    extract_answer_payload,
    registered_answer_bus,
)
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_data import joint_string_operation
from cftn_text.v1_3_training import train_integration_phase
from cftn_text.wandb_support import add_wandb_arguments, wandb_options_from_args


PHASE_NAME = "oracle_hard_answer_bus_recovery"
EXPECTED_SOURCE_SHA256 = (
    "231566e1d17dca7a35bd7028c7af38bb9f450c38ff9f727fa7d278dcbb8cd790"
)
ALL_CLASSES = [
    "pure_language",
    "explicit_math",
    "exact_string",
    "language_dependent_math",
    "multi_parallel",
    "multi_sequential",
]
TRAIN_CLASSES = [name for name in ALL_CLASSES if name != "pure_language"]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def resolve_source_checkpoint(
    config: dict[str, Any], requested: str
) -> tuple[Path, dict[str, Any]]:
    artifact_root = Path(config["paths"]["artifact_root"])
    summary_path = artifact_root / "oracle_hard_fusion_recovery" / "summary.json"
    summary = _load_json(summary_path)
    source = (
        Path(str(summary["best_checkpoint"]))
        if requested == "auto"
        else Path(requested)
    ).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"answer-bus source checkpoint is missing: {source}")
    source_sha256 = file_sha256(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "answer-bus recovery source is not the protected fusion epoch-4 checkpoint; "
            f"expected {EXPECTED_SOURCE_SHA256}, found {source_sha256}"
        )
    if str(summary.get("best_checkpoint_sha256")) != source_sha256:
        raise RuntimeError("fusion summary and protected source checkpoint disagree")
    best = summary.get("best_metrics", {})
    if int(best.get("epoch", -1)) != 4 or best.get("checkpoint_eligible") is not True:
        raise RuntimeError("fusion summary does not protect an eligible epoch-4 source")
    acceptance = best.get("hardening_acceptance", {}).get("gates", {})
    if acceptance.get("pass") is not True:
        raise RuntimeError("protected fusion source did not pass its acceptance contract")
    return source, summary


def configure_answer_bus_recovery(
    config: dict[str, Any], source: Path
) -> dict[str, Any]:
    config["answer_composer"] = {
        "hidden_size": 256,
        "attention_heads": 8,
        "decoder_layers": 2,
        "dropout": 0.10,
        "maximum_source_positions": 64,
        "maximum_target_positions": 48,
    }
    phase = {
        "name": PHASE_NAME,
        "objective_mode": "oracle_hard_answer_bus",
        "selection_mode": "answer_bus_recovery",
        "selection_focus_classes": ["exact_string", "multi_parallel"],
        "wake_mode": "oracle",
        "maximum_rounds": 2,
        "include_task_classes": list(ALL_CLASSES),
        "train_task_classes": list(TRAIN_CLASSES),
        "train_examples_by_class": {
            "explicit_math": 1000,
            "exact_string": 6000,
            "language_dependent_math": 1000,
            "multi_parallel": 10000,
            "multi_sequential": 2000,
        },
        "task_class_weights": {
            "explicit_math": 0.5,
            "exact_string": 2.0,
            "language_dependent_math": 0.5,
            "multi_parallel": 4.0,
            "multi_sequential": 1.0,
        },
        "conditional_execution": True,
        "apply_halt": False,
        "repair_sequential_orders": True,
        "num_workers": 0,
        # The production mix is 50% multi-parallel. Although a balanced
        # batch-96 smoke passed, its first production batch reserved 11.9/12.3
        # GiB. Batch 64 passed the three-forward causal smoke at 4.73 GiB and
        # leaves a robust margin for composition-dependent sequence variance.
        "batch_size": 64,
        "eval_batch_size": 128,
        "max_epochs": 50,
        "minimum_epochs": 10,
        "early_stop_patience": 6,
        "selection_min_delta": 2.0e-4,
        "learning_rate": 3.0e-4,
        "answer_composer_learning_rate": 3.0e-4,
        "minimum_learning_rate": 1.0e-5,
        "warmup_fraction": 0.03,
        "weight_decay": 0.01,
        "zero_update_candidate": False,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "answer_bus_acceptance": {
            "minimum_protocol_semantic_sequence_accuracy": 0.83,
            "minimum_answer_bus_causal_loss_gap": 0.10,
            "minimum_reverse_generated_accuracy": 0.90,
            "minimum_multi_parallel_generated_accuracy": 0.90,
            "minimum_short_string_generated_accuracy": 0.98,
            "protected_task_classes": {
                "explicit_math": 0.95,
                "language_dependent_math": 0.95,
                "multi_sequential": 0.95,
            },
        },
        "end_to_end_examples_by_class": {
            "explicit_math": 64,
            "exact_string": 300,
            "language_dependent_math": 64,
            "multi_parallel": 192,
            "multi_sequential": 128,
        },
        "end_to_end_acceptance": {
            "minimum_answer_bus_valid_rate": 0.95,
            "minimum_explicit_math_accuracy": 0.95,
            "minimum_language_dependent_math_accuracy": 0.95,
            "minimum_exact_string_reverse_accuracy": 0.85,
            "minimum_multi_parallel_accuracy": 0.85,
            "minimum_multi_sequential_accuracy": 0.90,
        },
    }
    phases = [
        value
        for value in config["integration_training"]["phases"]
        if value["name"] != PHASE_NAME
    ]
    config["integration_training"]["phases"] = [*phases, phase]
    return phase


def audit_registered_answer_bus(
    config: dict[str, Any], phase: dict[str, Any]
) -> dict[str, Any]:
    from cftn_text.v1_3_dataset import V13Dataset
    from cftn_text.v1_3_training import load_v1_3_data_contract

    data_root, manifest = load_v1_3_data_contract(config)
    records = V13Dataset(
        data_root / manifest["splits"]["joint_validation"]["path"]
    ).records
    class_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    exact = 0
    eligible = 0
    maximum_payload_bytes = 0
    maximum_target_bytes = 0
    legacy_stale_parallel_metadata = 0
    for record in records:
        task_class = str(record["task_class"])
        class_counts[task_class] += 1
        composed = compose_registered_answer(record)
        if task_class == "pure_language":
            if composed is not None:
                raise RuntimeError("pure-language record has a composed specialist answer")
            continue
        eligible += 1
        exact += int(composed == record["gpt_target"])
        maximum_target_bytes = max(
            maximum_target_bytes, len(str(record["gpt_target"]).encode("utf-8")) + 1
        )
        entries = registered_answer_bus(record)
        maximum_payload_bytes = max(
            maximum_payload_bytes,
            *(len(payload.encode("utf-8")) for _, _, payload in entries),
        )
        operation = joint_string_operation(record)
        if operation is not None:
            operation_counts[f"{task_class}/{operation}"] += 1
        if task_class == "multi_parallel":
            string_metadata = record.get("metadata", {}).get("string", {})
            if string_metadata.get("operation") != "reverse":
                legacy_stale_parallel_metadata += 1
    if exact != eligible:
        raise RuntimeError(
            f"registered answer bus is not lossless: {exact}/{eligible} exact"
        )
    composer = config["answer_composer"]
    if maximum_payload_bytes > int(composer["maximum_source_positions"]):
        raise RuntimeError("registered answer payload exceeds composer source positions")
    if maximum_target_bytes > int(composer["maximum_target_positions"]):
        raise RuntimeError("registered target exceeds composer target positions")
    return {
        "format": "cftn_text_v1_3_registered_answer_bus_audit_v1",
        "state": "passed",
        "examples": len(records),
        "eligible_examples": eligible,
        "deterministic_composer_exact": exact,
        "deterministic_composer_accuracy": exact / max(1, eligible),
        "class_counts": dict(sorted(class_counts.items())),
        "string_operation_counts": dict(sorted(operation_counts.items())),
        "maximum_payload_bytes": maximum_payload_bytes,
        "maximum_target_bytes_including_eos": maximum_target_bytes,
        "legacy_stale_parallel_metadata_repaired_at_load": legacy_stale_parallel_metadata,
        "configured_train_examples": sum(phase["train_examples_by_class"].values()),
    }


def build_contract(
    config: dict[str, Any],
    phase: dict[str, Any],
    source: Path,
    source_summary: dict[str, Any],
    bus_audit: dict[str, Any],
) -> dict[str, Any]:
    root = Path(__file__).parents[1]
    payload = {
        "format": "cftn_text_v1_3_answer_bus_recovery_contract_v1",
        "revision_sha256": config["_meta"]["sha256"],
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "source_best_epoch": source_summary.get("best_metrics", {}).get("epoch"),
        "registered_answer_bus_audit": bus_audit,
        "answer_composer": config["answer_composer"],
        "phase": phase,
        "invariants": {
            "protected_source_immutable": True,
            "legacy_checkpoint_identity_at_initialization": True,
            "specialist_towers_frozen": True,
            "request_bridges_frozen": True,
            "return_bridges_frozen": True,
            "message_fusion_frozen": True,
            "gpt_receivers_frozen": True,
            "specialist_receivers_frozen": True,
            "wake_gates_frozen": True,
            "halt_gate_frozen": True,
            "trainable_components": ["answer_composer"],
            "oracle_hard_routes": True,
            "ground_truth_bus_for_composer_training": True,
            "native_specialist_bus_for_terminal_evaluation": True,
            "zero_update_source_is_not_a_candidate": True,
        },
        "source_files": {
            name: file_sha256(root / relative)
            for name, relative in {
                "gpt_receiver.py": "cftn_text/gpt_receiver.py",
                "v1_3_answer_bus.py": "cftn_text/v1_3_answer_bus.py",
                "v1_3_data.py": "cftn_text/v1_3_data.py",
                "v1_3_dataset.py": "cftn_text/v1_3_dataset.py",
                "v1_3_evaluation.py": "cftn_text/v1_3_evaluation.py",
                "v1_3_model.py": "cftn_text/v1_3_model.py",
                "v1_3_training.py": "cftn_text/v1_3_training.py",
                "recover_v1_3_answer_bus.py": "tools/recover_v1_3_answer_bus.py",
            }.items()
        },
    }
    payload["contract_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def smoke_answer_bus_recovery(
    config: dict[str, Any], phase: dict[str, Any], source: Path, *, device_name: str
) -> dict[str, Any]:
    import torch

    from cftn_text.checkpoint import gpu_status
    from cftn_text.tokenizer import ByteMathTokenizer
    from cftn_text.training import autocast_context, precision_dtype, resolve_device
    from cftn_text.v1_3_dataset import (
        V13Dataset,
        V13JointCollator,
        move_v1_3_batch,
    )
    from cftn_text.v1_3_training import (
        build_v1_3_model,
        load_v1_3_data_contract,
        v1_3_objective,
    )

    torch.manual_seed(int(config["revision"]["seed"]) + 101)
    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=source
    )
    model.set_trainable_phase(PHASE_NAME)
    model.train()
    all_records = V13Dataset(
        data_root / manifest["splits"]["joint_train"]["path"]
    ).records
    selected: list[dict[str, Any]] = []
    per_class = {name: 0 for name in TRAIN_CLASSES}
    smoke_batch_size = int(phase["batch_size"])
    target_per_class = {
        name: smoke_batch_size // len(TRAIN_CLASSES) for name in TRAIN_CLASSES
    }
    target_per_class["multi_parallel"] += smoke_batch_size % len(TRAIN_CLASSES)
    for record in all_records:
        task_class = str(record["task_class"])
        if task_class in per_class and per_class[task_class] < target_per_class[task_class]:
            selected.append(record)
            per_class[task_class] += 1
        if len(selected) >= smoke_batch_size:
            break
    if len(selected) != smoke_batch_size:
        raise RuntimeError("could not assemble the answer-bus smoke batch")
    collator = V13JointCollator(
        ByteMathTokenizer(),
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    batch = move_v1_3_batch(collator(selected), device)
    dtype = precision_dtype(config["integration_training"]["precision"], device)
    with autocast_context(device, dtype):
        output, loss, components = v1_3_objective(
            model,
            batch,
            wake_mode="oracle",
            maximum_rounds=2,
            settings=config["integration_training"],
            # Step zero exercises both causal corruptions, which is the
            # recovery's peak-memory and strongest gradient-contract path.
            global_step=0,
            objective_mode="oracle_hard_answer_bus",
            conditional_execution=True,
            apply_halt=False,
            task_class_weights=phase["task_class_weights"],
        )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("answer-bus smoke loss is non-finite")
    if output.answer_bus_attention_mask.shape != output.answer_bus_token_ids.shape:
        raise RuntimeError("answer-bus smoke produced mismatched source tensors")
    loss.backward()
    trainable = {
        name for name, value in model.named_parameters() if value.requires_grad
    }
    gradient_names = {
        name
        for name, value in model.named_parameters()
        if value.grad is not None and bool(value.grad.detach().abs().sum().gt(0))
    }
    unexpected_trainable = sorted(
        name for name in trainable if not name.startswith("answer_composer.")
    )
    unexpected_gradients = sorted(
        name for name in gradient_names if not name.startswith("answer_composer.")
    )
    required_gradients = {
        "answer_composer.pointer_query.weight",
        "answer_composer.vocabulary_projection.weight",
    }
    missing_required = sorted(required_gradients.difference(gradient_names))
    if unexpected_trainable or unexpected_gradients or missing_required:
        raise RuntimeError(
            "answer-bus smoke violated its gradient contract; "
            f"unexpected_trainable={unexpected_trainable}, "
            f"unexpected_gradients={unexpected_gradients}, "
            f"missing_required_gradients={missing_required}"
        )
    return {
        "format": "cftn_text_v1_3_answer_bus_smoke_v1",
        "state": "passed",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "batch_examples": len(selected),
        "batch_class_counts": per_class,
        "loss": float(loss.detach()),
        "components": components,
        "answer_bus_tokens": int(output.answer_bus_attention_mask.sum()),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "trainable_tensor_count": len(trainable),
        "nonzero_gradient_tensor_count": len(gradient_names),
        "optimizer_groups": ["answer_composer"],
        "provenance": provenance,
        "gpu": gpu_status(),
    }


def _select_end_to_end_records(
    records: list[dict[str, Any]], limits: dict[str, int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    exact_operations: Counter[str] = Counter()
    exact_limit = int(limits.get("exact_string", 0))
    per_operation_limit = exact_limit // 3
    for record in records:
        task_class = str(record["task_class"])
        if task_class not in limits or counts[task_class] >= int(limits[task_class]):
            continue
        if task_class == "exact_string":
            operation = joint_string_operation(record)
            if operation is None or exact_operations[operation] >= per_operation_limit:
                continue
            exact_operations[operation] += 1
        selected.append(record)
        counts[task_class] += 1
    missing = {
        name: int(limit) - counts[name]
        for name, limit in limits.items()
        if counts[name] != int(limit)
    }
    if missing:
        raise RuntimeError(f"could not assemble end-to-end validation panel: {missing}")
    return selected


def evaluate_native_answer_bus(
    config: dict[str, Any],
    phase: dict[str, Any],
    checkpoint: Path,
    *,
    device_name: str,
    artifact_phase_name: str = PHASE_NAME,
    output_artifact: Path | None = None,
    lossless_request_mode: str = "disabled",
    deterministic_answer_composition: bool = False,
    specialist_generation_policy: str = "configured",
    dispatcher_checkpoint: Path | None = None,
    dispatcher_loader: Any | None = None,
) -> dict[str, Any]:
    import torch

    from cftn_text.checkpoint import gpu_status
    from cftn_text.tokenizer import ByteMathTokenizer
    from cftn_text.training import resolve_device
    from cftn_text.v1_3_dataset import (
        V13Dataset,
        V13JointInferenceCollator,
        V13JointCollator,
        move_v1_3_batch,
    )
    from cftn_text.v1_3_evaluation import (
        extract_completion_answer,
        generate_joint_batch,
        resolve_specialist_generation_budget,
    )
    from cftn_text.v1_3_learned_dispatch import load_learned_dispatcher
    from cftn_text.v1_3_training import build_v1_3_model, load_v1_3_data_contract

    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    all_records = V13Dataset(
        data_root / manifest["splits"]["joint_validation"]["path"]
    ).records
    records = _select_end_to_end_records(
        all_records, dict(phase["end_to_end_examples_by_class"])
    )
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=checkpoint
    )
    model.eval()
    math_tokenizer = ByteMathTokenizer()
    dispatcher_enabled = lossless_request_mode in {
        "typed_dispatcher_no_latent",
        "learned_dispatcher_no_latent",
    }
    learned_dispatcher = None
    dispatcher_checkpoint_hash = None
    if lossless_request_mode == "learned_dispatcher_no_latent":
        if dispatcher_checkpoint is None:
            raise ValueError("learned dispatcher evaluation requires a checkpoint")
        dispatcher_checkpoint = Path(dispatcher_checkpoint).resolve()
        if not dispatcher_checkpoint.is_file():
            raise FileNotFoundError(dispatcher_checkpoint)
        dispatcher_checkpoint_hash = file_sha256(dispatcher_checkpoint)
        loader = dispatcher_loader or load_learned_dispatcher
        learned_dispatcher = loader(
            dispatcher_checkpoint, device="cpu"
        )
    collator_class = V13JointInferenceCollator if dispatcher_enabled else V13JointCollator
    collator = collator_class(
        math_tokenizer,
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    specialist_budget = max(
        resolve_specialist_generation_budget(
            config, tower, specialist_generation_policy
        )
        for tower in model.specialists.values()
    )
    artifact = (
        Path(output_artifact)
        if output_artifact is not None
        else Path(config["paths"]["artifact_root"]) / artifact_phase_name
    )
    artifact.mkdir(parents=True, exist_ok=True)
    rows_path = artifact / "native_answer_bus_generations.jsonl"
    rows_path.unlink(missing_ok=True)
    class_totals: Counter[str] = Counter()
    class_correct: Counter[str] = Counter()
    operation_totals: Counter[str] = Counter()
    operation_correct: Counter[str] = Counter()
    valid_bus = 0
    valid_specialist_payloads = 0
    required_specialist_payloads = 0
    valid_dispatch_plans = 0
    completed_dispatches = 0
    batch_size = 8
    started = time.time()
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        runtime_records = (
            [
                {
                    "schema_version": record["schema_version"],
                    "record_id": record["record_id"],
                    "problem": record["problem"],
                    "gpt_prompt": record["gpt_prompt"],
                }
                for record in batch_records
            ]
            if dispatcher_enabled
            else batch_records
        )
        batch = move_v1_3_batch(collator(runtime_records), device)
        generated = generate_joint_batch(
            model,
            batch,
            math_tokenizer,
            gpt_tokenizer,
            wake_mode="oracle",
            maximum_rounds=2,
            max_specialist_new_tokens=specialist_budget,
            max_gpt_new_tokens=1,
            use_answer_composer=not deterministic_answer_composition,
            max_answer_new_tokens=int(
                config["answer_composer"]["maximum_target_positions"]
            ),
            lossless_request_mode=lossless_request_mode,
            deterministic_answer_composition=deterministic_answer_composition,
            typed_dispatcher=learned_dispatcher,
        )
        for record, result in zip(batch_records, generated):
            task_class = str(record["task_class"])
            parsed = extract_completion_answer(str(result["generation"]))
            correct = parsed == str(record["gpt_target"])
            class_totals[task_class] += 1
            class_correct[task_class] += int(correct)
            valid_bus += int(result["answer_bus_valid"])
            if dispatcher_enabled:
                valid_dispatch_plans += int(result["dispatch_plan"] is not None)
                completed_dispatches += int(
                    result["dispatch_plan"] is not None
                    and result["dispatch_error"] is None
                    and result["answer_bus_valid"]
                )
            operation = joint_string_operation(record)
            operation_key = (
                f"{task_class}/{operation}" if operation is not None else None
            )
            if operation_key is not None:
                operation_totals[operation_key] += 1
                operation_correct[operation_key] += int(correct)
            for round_index, required in enumerate(
                record["required_specialists_by_round"][:2]
            ):
                for specialist in required:
                    required_specialist_payloads += 1
                    text = result["specialist_generations"][specialist][round_index]
                    valid_specialist_payloads += int(
                        extract_answer_payload(text, strict=False) is not None
                    )
            append_jsonl(
                {
                    "record_id": record["record_id"],
                    "task_class": task_class,
                    "operation": operation,
                    "target": record["gpt_target"],
                    "generation": result["generation"],
                    "gpt_generation": result["gpt_generation"],
                    "answer_composer_generation": result[
                        "answer_composer_generation"
                    ],
                    "answer_bus_valid": result["answer_bus_valid"],
                    "specialist_generations": result["specialist_generations"],
                    "dispatch_plan": result["dispatch_plan"],
                    "dispatch_requests": result["dispatch_requests"],
                    "dispatch_error": result["dispatch_error"],
                    "correct": correct,
                },
                rows_path,
            )
    class_metrics = {
        name: {
            "examples": class_totals[name],
            "sequence_accuracy": class_correct[name] / class_totals[name],
        }
        for name in sorted(class_totals)
    }
    operation_metrics = {
        name: {
            "examples": operation_totals[name],
            "sequence_accuracy": operation_correct[name] / operation_totals[name],
        }
        for name in sorted(operation_totals)
    }
    acceptance = phase["end_to_end_acceptance"]
    gates = {
        "answer_bus_valid_rate": valid_bus / len(records)
        >= float(acceptance["minimum_answer_bus_valid_rate"]),
        "explicit_math": class_metrics["explicit_math"]["sequence_accuracy"]
        >= float(acceptance["minimum_explicit_math_accuracy"]),
        "language_dependent_math": class_metrics["language_dependent_math"][
            "sequence_accuracy"
        ]
        >= float(acceptance["minimum_language_dependent_math_accuracy"]),
        "exact_string_reverse": operation_metrics["exact_string/reverse"][
            "sequence_accuracy"
        ]
        >= float(acceptance["minimum_exact_string_reverse_accuracy"]),
        "multi_parallel": class_metrics["multi_parallel"]["sequence_accuracy"]
        >= float(acceptance["minimum_multi_parallel_accuracy"]),
        "multi_sequential": class_metrics["multi_sequential"]["sequence_accuracy"]
        >= float(acceptance["minimum_multi_sequential_accuracy"]),
    }
    gates["pass"] = all(gates.values())
    report = {
        "format": "cftn_text_v1_3_native_answer_bus_evaluation_v1",
        "state": "passed" if gates["pass"] else "failed_acceptance",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "examples": len(records),
        "answer_bus_valid_rate": valid_bus / len(records),
        "specialist_payload_valid_rate": valid_specialist_payloads
        / max(1, required_specialist_payloads),
        "task_class_metrics": class_metrics,
        "lossless_request_mode": lossless_request_mode,
        "deterministic_answer_composition": deterministic_answer_composition,
        "specialist_generation_policy": specialist_generation_policy,
        "max_specialist_new_tokens": specialist_budget,
        "oracle_metadata_visible_to_runtime": not dispatcher_enabled,
        "dispatch_plan_valid_rate": (
            valid_dispatch_plans / len(records) if dispatcher_enabled else None
        ),
        "dispatch_completion_rate": (
            completed_dispatches / len(records) if dispatcher_enabled else None
        ),
        "dispatcher_checkpoint": (
            str(dispatcher_checkpoint) if dispatcher_checkpoint is not None else None
        ),
        "dispatcher_checkpoint_sha256": dispatcher_checkpoint_hash,
        "string_operation_metrics": operation_metrics,
        "acceptance": {"gates": gates, "thresholds": acceptance},
        "generation_rows": str(rows_path.resolve()),
        "elapsed_seconds": time.time() - started,
        "provenance": provenance,
        "gpu": gpu_status(),
    }
    atomic_json_dump(report, artifact / "native_answer_bus_report.json")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover V1.3 exact composition with a typed pointer-copy answer bus"
    )
    parser.add_argument(
        "--config", default="G:/ctfn-text/config/v1_3_multi_specialist.yaml"
    )
    parser.add_argument("--source-checkpoint", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run a real-checkpoint forward/backward contract check and exit",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--skip-native-evaluation", action="store_true")
    add_wandb_arguments(parser)
    args = parser.parse_args()

    config = load_v1_3_config(args.config)
    source, source_summary = resolve_source_checkpoint(
        config, args.source_checkpoint
    )
    phase = configure_answer_bus_recovery(config, source)
    bus_audit = audit_registered_answer_bus(config, phase)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "registered_answer_bus": bus_audit,
                    "smoke": smoke_answer_bus_recovery(
                        config, phase, source, device_name=args.device
                    ),
                },
                indent=2,
            )
        )
        return
    artifact_root = Path(config["paths"]["artifact_root"])
    artifact = artifact_root / PHASE_NAME
    if not args.resume and artifact.is_dir() and any(artifact.glob("*.pth")):
        raise RuntimeError(
            f"answer-bus recovery already has checkpoints; use --resume: {artifact}"
        )
    contract = build_contract(
        config, phase, source, source_summary, bus_audit
    )
    contract_path = artifact_root / "answer_bus_recovery_contract.json"
    status_path = artifact_root / "answer_bus_recovery_pipeline.json"
    atomic_json_dump(contract, contract_path)
    smoke = smoke_answer_bus_recovery(
        config, phase, source, device_name=args.device
    )
    smoke_path = artifact_root / "answer_bus_recovery_smoke.json"
    atomic_json_dump(smoke, smoke_path)
    started = time.time()
    status: dict[str, Any] = {
        "format": "cftn_text_v1_3_answer_bus_recovery_pipeline_v1",
        "state": "running",
        "pid": os.getpid(),
        "phase": PHASE_NAME,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract["contract_sha256"],
        "smoke": str(smoke_path.resolve()),
        "started_unix": started,
    }
    atomic_json_dump(status, status_path)
    try:
        result = train_integration_phase(
            config,
            PHASE_NAME,
            device_name=args.device,
            resume=args.resume,
            max_batches=args.max_batches,
            wandb_options=wandb_options_from_args(
                args, default_run_name="v1-3-oracle-hard-answer-bus-recovery"
            ),
        )
        native_report = None
        if not args.skip_native_evaluation:
            status["state"] = "native_evaluation"
            status["training_result"] = result
            atomic_json_dump(status, status_path)
            native_report = evaluate_native_answer_bus(
                config,
                phase,
                Path(str(result["best_checkpoint"])),
                device_name=args.device,
            )
        terminal_state = (
            "failed_native_acceptance"
            if native_report is not None and native_report["state"] != "passed"
            else result["state"]
        )
        status.update(
            {
                "state": terminal_state,
                "result": result,
                "native_answer_bus_evaluation": native_report,
                "completed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        atomic_json_dump(status, status_path)
    except BaseException as exc:
        status.update(
            {
                "state": "error",
                "error": repr(exc),
                "failed_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        atomic_json_dump(status, status_path)
        raise
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
