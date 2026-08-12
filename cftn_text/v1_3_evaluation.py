from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from .checkpoint import append_jsonl, atomic_json_dump, gpu_status, load_checkpoint
from .config import config_sha256, load_config
from .data_generator import file_sha256
from .gpt_baseline import greedy_generate
from .math_tower import MathTower
from .metrics import paired_bootstrap_interval
from .tokenizer import ByteMathTokenizer, pad_1d
from .training import load_data_contract, load_gpt_components, precision_dtype, resolve_device
from .v1_3_config import audit_v1_2_pass
from .v1_3_data import SPECIALISTS, audit_v1_3_manifest
from .v1_3_dataset import V13Dataset, V13JointCollator, move_v1_3_batch
from .v1_3_model import V13MultiTowerModel, _masked_mean
from .v1_3_training import build_string_tower, build_v1_3_model, load_v1_3_data_contract
from .wandb_support import initialize_wandb


_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
SPECIALIST_GENERATION_POLICIES = frozenset({"configured", "full_context_v1"})


def extract_exact_answer(text: str) -> str | None:
    match = _ANSWER.search(text)
    return match.group(1).strip() if match else None


def extract_completion_answer(text: str) -> str | None:
    """Parse tagged legacy output or the registered first non-empty field."""

    tagged = extract_exact_answer(text)
    if tagged is not None:
        return tagged
    for line in text.splitlines():
        value = line.strip()
        if value:
            return value
    value = text.strip()
    return value or None


def _target(record: dict[str, Any]) -> str:
    parsed = extract_exact_answer(str(record["target_answer"]))
    if parsed is None:
        raise ValueError("record target has no <answer> tag")
    return parsed


def _exact(generation: str, record: dict[str, Any]) -> bool:
    return extract_completion_answer(generation) == _target(record)


def resolve_specialist_generation_budget(
    config: dict[str, Any], tower: MathTower, policy: str
) -> int:
    """Resolve an explicit, reportable native-generation token policy."""

    if policy not in SPECIALIST_GENERATION_POLICIES:
        raise ValueError(f"unsupported specialist generation policy: {policy}")
    configured = int(config["evaluation"]["max_specialist_new_tokens"])
    context = int(tower.max_sequence_length)
    if configured < 1 or context < 1:
        raise ValueError("specialist generation budgets must be positive")
    return min(configured, context) if policy == "configured" else context


_ORACLE_TASK_CELLS = {
    "math": (
        "explicit_math",
        "language_dependent_math",
        "multi_parallel",
        "multi_sequential",
    ),
    "string": ("exact_string", "multi_parallel", "multi_sequential"),
}


def _oracle_specialist_items(
    records: list[dict[str, Any]], specialist: str
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    items: list[dict[str, Any]] = []
    locations: list[tuple[int, int]] = []
    for record_index, record in enumerate(records):
        prompts = record.get("specialist_oracle_problems_by_round", {}).get(
            specialist, []
        )
        targets = record.get("specialist_targets_by_round", {}).get(specialist, [])
        for round_index, (problem, target) in enumerate(zip(prompts, targets)):
            if problem is None and target is None:
                continue
            if not problem or not target:
                raise ValueError(
                    f"incomplete oracle-native contract for {specialist} on "
                    f"{record.get('record_id')} round {round_index + 1}"
                )
            items.append(
                {
                    "record_id": (
                        f"{record['record_id']}:{specialist}:round-{round_index + 1}"
                    ),
                    "task_class": record["task_class"],
                    "problem": str(problem),
                    "target_answer": str(target),
                }
            )
            locations.append((record_index, round_index))
    return items, locations


def _oracle_capability_batch(
    specialists: dict[str, MathTower] | torch.nn.ModuleDict,
    records: list[dict[str, Any]],
    tokenizer: ByteMathTokenizer,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows = [
        {
            "record_id": record["record_id"],
            "task_class": record["task_class"],
            "checks": [],
        }
        for record in records
    ]
    for specialist in SPECIALISTS:
        items, locations = _oracle_specialist_items(records, specialist)
        generations = generate_native_specialist(
            specialists[specialist],
            items,
            tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        for item, (record_index, round_index), generation in zip(
            items, locations, generations
        ):
            rows[record_index]["checks"].append(
                {
                    "specialist": specialist,
                    "round": round_index + 1,
                    "problem": item["problem"],
                    "target": item["target_answer"],
                    "generation": generation,
                    "correct": _exact(generation, item),
                }
            )
    for row in rows:
        row["required"] = bool(row["checks"])
        row["supported"] = all(check["correct"] for check in row["checks"])
    return rows


def _oracle_capability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [row for row in rows if row["required"]]
    summary: dict[str, Any] = {
        "examples": len(rows),
        "required_examples": len(required),
        "supported_examples": sum(bool(row["supported"]) for row in rows),
        "supported_required_examples": sum(
            bool(row["supported"]) for row in required
        ),
        "coverage": sum(bool(row["supported"]) for row in rows) / max(1, len(rows)),
        "required_coverage": sum(bool(row["supported"]) for row in required)
        / max(1, len(required)),
        "by_task_class": {},
        "by_specialist_and_task_class": {},
    }
    for task_class in sorted({str(row["task_class"]) for row in rows}):
        selected = [row for row in rows if row["task_class"] == task_class]
        summary["by_task_class"][task_class] = {
            "examples": len(selected),
            "supported_examples": sum(bool(row["supported"]) for row in selected),
            "coverage": sum(bool(row["supported"]) for row in selected)
            / max(1, len(selected)),
        }
    for specialist, task_classes in _ORACLE_TASK_CELLS.items():
        specialist_report: dict[str, Any] = {}
        for task_class in task_classes:
            checks = [
                check
                for row in rows
                if row["task_class"] == task_class
                for check in row["checks"]
                if check["specialist"] == specialist
            ]
            specialist_report[task_class] = {
                "examples": len(checks),
                "correct": sum(bool(check["correct"]) for check in checks),
                "exact_accuracy": sum(bool(check["correct"]) for check in checks)
                / max(1, len(checks)),
            }
        summary["by_specialist_and_task_class"][specialist] = specialist_report
    return summary


@torch.no_grad()
def generate_native_specialist(
    model: MathTower,
    records: list[dict[str, Any]],
    tokenizer: ByteMathTokenizer,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    if not records:
        return []
    sequences = [
        tokenizer.encode_generation_prefix(
            str(record["problem"]), model.max_sequence_length
        )
        for record in records
    ]
    prefix_lengths = torch.tensor([len(value) for value in sequences], device=device)
    finished = [False] * len(sequences)
    model.eval()
    for _ in range(int(max_new_tokens)):
        ids, mask = pad_1d(sequences, tokenizer.pad_token_id)
        ids, mask = ids.to(device), mask.to(device)
        output = model(ids, mask, prefix_lengths)
        lengths = mask.sum(dim=1) - 1
        next_tokens = output.logits[
            torch.arange(len(sequences), device=device), lengths
        ].argmax(dim=-1)
        for row, token in enumerate(next_tokens.tolist()):
            if finished[row]:
                continue
            if len(sequences[row]) >= model.max_sequence_length:
                finished[row] = True
                continue
            sequences[row].append(int(token))
            if int(token) == tokenizer.eos_token_id:
                finished[row] = True
        if all(finished):
            break
    return [
        tokenizer.decode(sequence[int(prefix_lengths[row].item()) :])
        for row, sequence in enumerate(sequences)
    ]


@torch.no_grad()
def _latent_specialist_generation(
    model: V13MultiTowerModel,
    name: str,
    specialist_batch: dict[str, torch.Tensor],
    request_message: torch.Tensor,
    activation: torch.Tensor,
    tokenizer: ByteMathTokenizer,
    *,
    max_new_tokens: int,
    return_enabled: bool,
) -> tuple[torch.Tensor, list[str], int]:
    tower = model.specialists[name]
    active_indices = activation.ge(model.wake_threshold).nonzero(as_tuple=False).flatten()
    batch_size = int(activation.shape[0])
    message_tokens = int(model.return_bridges[name].message_tokens)
    message_width = int(model.return_bridges[name].message_width)
    returned_full = request_message.new_zeros((batch_size, message_tokens, message_width))
    generations = [""] * batch_size
    if active_indices.numel() == 0:
        return returned_full, generations, 0
    prefixes: list[list[int]] = []
    for row in active_indices.tolist():
        length = int(specialist_batch["prefix_lengths"][row])
        prefixes.append(specialist_batch["input_ids"][row, :length].tolist())
    sequences = [list(prefix) for prefix in prefixes]
    prefix_lengths = torch.tensor(
        [len(value) for value in sequences], device=request_message.device
    )
    selected_request = request_message.index_select(0, active_indices)
    finished = [False] * len(sequences)
    for _ in range(int(max_new_tokens)):
        ids, mask = pad_1d(sequences, tokenizer.pad_token_id)
        ids, mask = ids.to(request_message.device), mask.to(request_message.device)
        output = tower(
            ids,
            mask,
            prefix_lengths,
            message=selected_request,
            receivers=model.specialist_receivers[name],
            receive_enabled=True,
            gate_mode=model.gate_mode,
        )
        lengths = mask.sum(dim=1) - 1
        next_tokens = output.logits[
            torch.arange(len(sequences), device=ids.device), lengths
        ].argmax(dim=-1)
        for local_row, token in enumerate(next_tokens.tolist()):
            if finished[local_row]:
                continue
            if len(sequences[local_row]) >= tower.max_sequence_length:
                finished[local_row] = True
                continue
            sequences[local_row].append(int(token))
            if int(token) == tokenizer.eos_token_id:
                finished[local_row] = True
        if all(finished):
            break
    ids, mask = pad_1d(sequences, tokenizer.pad_token_id)
    ids, mask = ids.to(request_message.device), mask.to(request_message.device)
    final_output = tower(
        ids,
        mask,
        prefix_lengths,
        message=selected_request,
        receivers=model.specialist_receivers[name],
        receive_enabled=True,
        gate_mode=model.gate_mode,
    )
    if return_enabled:
        returned = model.return_bridges[name](
            final_output.hidden_states,
            mask,
            enabled=True,
            gate_mode=model.gate_mode,
        )
        returned_full.index_copy_(0, active_indices, returned.message)
    for local_row, global_row in enumerate(active_indices.tolist()):
        generations[global_row] = tokenizer.decode(
            sequences[local_row][int(prefix_lengths[local_row].item()) :]
        )
    return returned_full, generations, int(active_indices.numel())


@torch.no_grad()
def generate_joint_batch(
    model: V13MultiTowerModel,
    batch: dict[str, Any],
    math_tokenizer: ByteMathTokenizer,
    gpt_tokenizer: Any,
    *,
    wake_mode: str,
    maximum_rounds: int,
    max_specialist_new_tokens: int,
    max_gpt_new_tokens: int,
    disabled_specialists: set[str] | None = None,
    disabled_requests: set[str] | None = None,
    disabled_returns: set[str] | None = None,
    shuffled_messages: bool = False,
    swap_first_round_returns: bool = False,
    wrong_specialist: bool = False,
    all_closed: bool = False,
    fixed_open: bool = False,
) -> list[dict[str, Any]]:
    old_gate_mode = model.gate_mode
    model.set_gate_mode("fixed_open" if fixed_open else "contextual")
    disabled = set(disabled_specialists or ())
    request_disabled = set(disabled_requests or ())
    return_disabled = set(disabled_returns or ())
    unknown = (disabled | request_disabled | return_disabled).difference(SPECIALISTS)
    if unknown:
        raise ValueError(f"unknown specialists in generation arm: {sorted(unknown)}")
    gpt_hidden = model.gpt_tower.prepass(
        batch["gpt_prepass_input_ids"], batch["gpt_prepass_attention_mask"]
    )
    accumulated: list[torch.Tensor] = []
    specialist_generations = {
        name: [[] for _ in range(int(batch["gpt_prepass_input_ids"].shape[0]))]
        for name in SPECIALISTS
    }
    wake_rows: list[dict[str, Any]] = [
        {"probabilities": [], "activations": [], "halt_probabilities": []}
        for _ in range(int(batch["gpt_prepass_input_ids"].shape[0]))
    ]
    halted = torch.zeros(
        batch["gpt_prepass_input_ids"].shape[0], dtype=torch.bool, device=gpt_hidden.device
    )
    active_executions = torch.zeros_like(halted, dtype=torch.long)
    active_by_specialist = {
        name: torch.zeros_like(halted, dtype=torch.long) for name in SPECIALISTS
    }
    try:
        for round_index in range(int(maximum_rounds)):
            pooled = _masked_mean(gpt_hidden, batch["gpt_prepass_attention_mask"])
            wake_logits = model.wake_gates(pooled)
            targets = batch["wake_targets"][:, round_index]
            if wrong_specialist:
                targets = targets.flip(dims=(-1,))
                effective_mode = "oracle"
            else:
                effective_mode = wake_mode
            probabilities, activations = model._wake_activation(
                wake_logits, targets, effective_mode
            )
            activations = activations * (~halted).to(activations.dtype).unsqueeze(1)
            if all_closed:
                activations = torch.zeros_like(activations)
            round_returns: list[torch.Tensor] = []
            for specialist_index, name in enumerate(SPECIALISTS):
                activation = activations[:, specialist_index]
                if name in disabled:
                    activation = torch.zeros_like(activation)
                request = model.request_bridges[name](
                    gpt_hidden,
                    batch["gpt_prepass_attention_mask"],
                    enabled=True,
                    gate_mode=model.gate_mode,
                ).message
                request = request * activation[:, None, None]
                if name in request_disabled:
                    request = torch.zeros_like(request)
                if shuffled_messages and request.shape[0] > 1:
                    request = request.roll(1, dims=0)
                returned, generated, executions = _latent_specialist_generation(
                    model,
                    name,
                    batch["specialists"][name][round_index],
                    request,
                    activation,
                    math_tokenizer,
                    max_new_tokens=max_specialist_new_tokens,
                    return_enabled=name not in return_disabled and not all_closed,
                )
                if shuffled_messages and returned.shape[0] > 1:
                    returned = returned.roll(1, dims=0)
                if swap_first_round_returns and round_index == 0 and returned.shape[0] > 1:
                    returned = returned.roll(1, dims=0)
                round_returns.append(returned)
                active_executions += activation.ge(model.wake_threshold).long()
                active_by_specialist[name] += activation.ge(model.wake_threshold).long()
                for row, text in enumerate(generated):
                    specialist_generations[name][row].append(text)
            accumulated.extend(round_returns)
            combined = torch.cat(accumulated, dim=1)
            mask = torch.ones(combined.shape[:2], dtype=torch.long, device=combined.device)
            update = model.gpt_tower(
                batch["gpt_prepass_input_ids"],
                batch["gpt_prepass_attention_mask"],
                message=combined,
                message_mask=mask,
                receive_enabled=not all_closed,
                gate_mode=model.gate_mode,
            )
            gpt_hidden = update.hidden_states[-1]
            halt_probabilities = torch.sigmoid(
                model.halt_gate(_masked_mean(gpt_hidden, batch["gpt_prepass_attention_mask"]))
            )
            if wake_mode == "hard":
                halted |= halt_probabilities.ge(0.5)
            for row in range(len(wake_rows)):
                wake_rows[row]["probabilities"].append(probabilities[row].tolist())
                wake_rows[row]["activations"].append(activations[row].tolist())
                wake_rows[row]["halt_probabilities"].append(float(halt_probabilities[row]))
        combined = torch.cat(accumulated, dim=1)
        prefixes = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(
                batch["gpt_prepass_input_ids"], batch["gpt_prepass_attention_mask"]
            )
        ]
        generated_ids = model.gpt_tower.generate_greedy(
            prefixes,
            combined,
            int(gpt_tokenizer.eos_token_id),
            int(max_gpt_new_tokens),
            receive_enabled=not all_closed,
            gate_mode=model.gate_mode,
        )
        results: list[dict[str, Any]] = []
        for row, token_ids in enumerate(generated_ids):
            results.append(
                {
                    "generation": gpt_tokenizer.decode(token_ids, skip_special_tokens=True),
                    "specialist_generations": {
                        name: specialist_generations[name][row] for name in SPECIALISTS
                    },
                    "wake": wake_rows[row],
                    "active_specialist_executions": int(active_executions[row]),
                    "active_specialist_executions_by_name": {
                        name: int(active_by_specialist[name][row]) for name in SPECIALISTS
                    },
                }
            )
        return results
    finally:
        model.set_gate_mode(old_gate_mode)


def _batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


@torch.no_grad()
def _serial_pipeline_gpt(
    model: V13MultiTowerModel,
    records: list[dict[str, Any]],
    math_generations: list[str],
    string_generations: list[str],
    gpt_tokenizer: Any,
    *,
    max_new_tokens: int,
) -> list[str]:
    prefixes = []
    for record, math_text, string_text in zip(
        records, math_generations, string_generations
    ):
        prompt = (
            f"Problem: {record['problem']}\n"
            f"Math specialist output: {math_text}\n"
            f"String specialist output: {string_text}\n"
            "Use the specialist outputs.\nExact result:"
        )
        prefixes.append(list(gpt_tokenizer.encode(prompt, add_special_tokens=False)))
    maximum_context = int(
        getattr(
            model.gpt_tower.model.config,
            "n_positions",
            getattr(model.gpt_tower.model.config, "max_position_embeddings", 1024),
        )
    )
    if any(len(prefix) + int(max_new_tokens) > maximum_context for prefix in prefixes):
        # Keep the control deterministic and explicit: trim only the beginning
        # of the original problem, preserving both actual specialist outputs.
        prefixes = [prefix[-(maximum_context - int(max_new_tokens)) :] for prefix in prefixes]
    messages = torch.zeros(
        len(prefixes),
        1,
        int(model.config["bridge"]["message_width"]),
        device=next(model.parameters()).device,
    )
    ids = model.gpt_tower.generate_greedy(
        prefixes,
        messages,
        int(gpt_tokenizer.eos_token_id),
        int(max_new_tokens),
        receive_enabled=False,
    )
    return [gpt_tokenizer.decode(value, skip_special_tokens=True) for value in ids]


def evaluate_gpt_language_calibration(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    maximum_examples: int | None = None,
) -> dict[str, Any]:
    prerequisite = audit_v1_2_pass(config)
    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    records = [
        record
        for record in V13Dataset(
            data_root / manifest["splits"]["joint_validation"]["path"]
        ).records
        if record["task_class"] == "pure_language"
    ][:maximum_examples]
    base = load_config(config["paths"]["base_config"])
    tokenizer, tower = load_gpt_components(base)
    tower = tower.to(device).eval()
    semantic_correct: list[bool] = []
    strict_tag_correct: list[bool] = []
    valid_fields: list[bool] = []
    prompt_copies: list[bool] = []
    rows: list[dict[str, Any]] = []
    batch_size = int(config["evaluation"]["generation_batch_size"])
    prompts = [V13JointCollator.gpt_prompt(record) for record in records]
    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        generated_texts = greedy_generate(
            tower.model,
            tokenizer,
            prompts,
            batch_size=batch_size,
            max_new_tokens=int(config["gpt_interface"]["calibration_max_new_tokens"]),
            device=device,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    for record, prompt, text in zip(records, prompts, generated_texts):
        parsed = extract_completion_answer(text)
        target = _target(record)
        is_correct = parsed == target
        tag_correct = extract_exact_answer(text) == target
        valid_field = parsed is not None
        copied_prompt = text.strip().startswith(str(record["problem"]).strip()) or (
            text.strip().startswith(prompt.strip())
        )
        semantic_correct.append(is_correct)
        strict_tag_correct.append(tag_correct)
        valid_fields.append(valid_field)
        prompt_copies.append(copied_prompt)
        rows.append(
            {
                "record_id": record["record_id"],
                "problem": record["problem"],
                "prompt": prompt,
                "target": record["target_answer"],
                "generation": text,
                "parsed_completion_field": parsed,
                "semantic_correct": is_correct,
                "strict_tag_correct": tag_correct,
                "valid_completion_field": valid_field,
                "prompt_copy": copied_prompt,
                "correct": is_correct,
            }
        )
    accuracy = sum(semantic_correct) / max(1, len(semantic_correct))
    strict_tag_accuracy = sum(strict_tag_correct) / max(1, len(strict_tag_correct))
    valid_field_rate = sum(valid_fields) / max(1, len(valid_fields))
    prompt_copy_rate = sum(prompt_copies) / max(1, len(prompt_copies))
    threshold = float(config["acceptance"]["minimum_gpt_language_calibration_accuracy"])
    artifact = Path(config["paths"]["artifact_root"]) / "gpt_language_calibration"
    artifact.mkdir(parents=True, exist_ok=True)
    row_path = artifact / "generations.jsonl"
    row_path.unlink(missing_ok=True)
    for row in rows:
        append_jsonl(row, row_path)
    report = {
        "format": "cftn_text_v1_3_gpt_language_calibration_v2",
        "state": "passed" if accuracy >= threshold else "failed",
        "examples": len(semantic_correct),
        "answer_protocol": config["gpt_interface"]["answer_protocol"],
        "gate_metric": "semantic_accuracy",
        "semantic_accuracy": accuracy,
        "exact_accuracy": accuracy,
        "strict_tag_accuracy_diagnostic": strict_tag_accuracy,
        "valid_completion_field_rate": valid_field_rate,
        "prompt_copy_rate": prompt_copy_rate,
        "threshold": threshold,
        "pass": accuracy >= threshold,
        "generations": str(row_path.resolve()),
        "revision_sha256": config["_meta"]["sha256"],
        "prerequisite": prerequisite,
    }
    atomic_json_dump(report, artifact / "report.json")
    if not report["pass"]:
        raise RuntimeError(
            "frozen GPT semantic language calibration failed: "
            f"{accuracy:.4f} < {threshold:.4f}"
        )
    return report


def evaluate_native_specialists(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    maximum_examples: int | None = None,
    specialist_generation_policy: str = "configured",
) -> dict[str, Any]:
    audit_v1_2_pass(config)
    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    checkpoint_path = (
        Path(config["paths"]["artifact_root"]) / "string_specialist" / "string.best.pth"
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_stage="v1_3_string",
        expected_config_sha256=config["_meta"]["sha256"],
        expected_manifest_sha256=manifest["manifest_sha256"],
        map_location=device,
    )
    model = build_string_tower(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    string_generation_budget = resolve_specialist_generation_budget(
        config, model, specialist_generation_policy
    )
    tokenizer = ByteMathTokenizer()
    artifact = Path(config["paths"]["artifact_root"]) / "native_specialist_evaluation"
    artifact.mkdir(parents=True, exist_ok=True)
    status_path = artifact / "status.json"
    progress_path = artifact / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    started_at = time.time()
    selected_splits = (
        "string_test",
        "string_heldout_paraphrase",
        "string_extrapolation",
        "string_compositional",
    )
    total_examples = sum(
        min(
            int(manifest["splits"][split]["examples"]),
            int(maximum_examples)
            if maximum_examples is not None
            else int(manifest["splits"][split]["examples"]),
        )
        for split in selected_splits
    )
    joint_test_examples = min(
        int(manifest["splits"]["joint_test"]["examples"]),
        int(maximum_examples)
        if maximum_examples is not None
        else int(manifest["splits"]["joint_test"]["examples"]),
    )
    total_examples += joint_test_examples
    completed_examples = 0
    next_boundary = 0.10
    splits: dict[str, Any] = {}
    for split_index, split in enumerate(selected_splits, start=1):
        records = V13Dataset(data_root / manifest["splits"][split]["path"]).records
        if maximum_examples is not None:
            records = records[: int(maximum_examples)]
        generations: list[str] = []
        rows_path = artifact / f"{split}_generations.jsonl"
        rows_path.unlink(missing_ok=True)
        batches_total = math.ceil(
            len(records) / int(config["evaluation"]["generation_batch_size"])
        )
        split_completed = 0
        for batch_index, batch_records in enumerate(_batched(
            records, int(config["evaluation"]["generation_batch_size"])
        ), start=1):
            batch_generations = generate_native_specialist(
                model,
                batch_records,
                tokenizer,
                device=device,
                max_new_tokens=string_generation_budget,
            )
            generations.extend(batch_generations)
            for record, text in zip(batch_records, batch_generations):
                append_jsonl(
                    {
                        "record_id": record["record_id"],
                        "operation": record["operation"],
                        "problem": record["problem"],
                        "target": record["target_answer"],
                        "generation": text,
                        "correct": _exact(text, record),
                    },
                    rows_path,
                )
            split_completed += len(batch_records)
            completed_examples += len(batch_records)
            elapsed = time.time() - started_at
            rate = completed_examples / max(1e-9, elapsed)
            progress = {
                "format": "cftn_text_v1_3_native_evaluation_status_v1",
                "state": "running",
                "pid": os.getpid(),
                "split_index": split_index,
                "splits_total": len(selected_splits),
                "split": split,
                "batch_completed": batch_index,
                "batches_total": batches_total,
                "split_examples_completed": split_completed,
                "split_examples_total": len(records),
                "overall_examples_completed": completed_examples,
                "overall_examples_total": total_examples,
                "overall_progress": completed_examples / max(1, total_examples),
                "elapsed_seconds": elapsed,
                "eta_seconds": (
                    (total_examples - completed_examples) / rate if rate > 0 else None
                ),
                "gpu": gpu_status(),
            }
            atomic_json_dump(progress, status_path)
            if progress["overall_progress"] >= next_boundary or split_completed == len(records):
                append_jsonl(progress, progress_path)
                while progress["overall_progress"] >= next_boundary:
                    next_boundary += 0.10
        correct = [_exact(text, record) for text, record in zip(generations, records)]
        valid = [extract_exact_answer(text) is not None for text in generations]
        splits[split] = {
            "examples": len(records),
            "exact_accuracy": sum(correct) / max(1, len(correct)),
            "valid_rate": sum(valid) / max(1, len(valid)),
            "generations": str(rows_path.resolve()),
        }
    base = load_config(config["paths"]["base_config"])
    math_report_path = Path(base["project"]["artifact_root"]) / "evaluation_math" / "report.json"
    if not math_report_path.is_file():
        raise FileNotFoundError(f"sealed V1.1 math evaluation is missing: {math_report_path}")
    math_report = json.loads(math_report_path.read_text(encoding="utf-8"))
    _, math_manifest = load_data_contract(base)
    math_checkpoint = load_checkpoint(
        config["paths"]["math_checkpoint"],
        expected_stage="math",
        expected_config_sha256=config_sha256(base),
        expected_manifest_sha256=math_manifest["manifest_sha256"],
        map_location=device,
    )
    math_model = MathTower(base["math_tower"], ByteMathTokenizer.vocab_size).to(device)
    math_model.load_state_dict(math_checkpoint["model_state"], strict=True)
    math_generation_budget = resolve_specialist_generation_budget(
        config, math_model, specialist_generation_policy
    )
    oracle_generation_budget = max(
        math_generation_budget, string_generation_budget
    )
    joint_records = V13Dataset(
        data_root / manifest["splits"]["joint_test"]["path"]
    ).records[:joint_test_examples]
    oracle_rows_path = artifact / "joint_test_oracle_native_capability.jsonl"
    oracle_rows_path.unlink(missing_ok=True)
    oracle_rows: list[dict[str, Any]] = []
    oracle_batches = math.ceil(
        len(joint_records) / int(config["evaluation"]["generation_batch_size"])
    )
    for batch_index, batch_records in enumerate(
        _batched(joint_records, int(config["evaluation"]["generation_batch_size"])),
        start=1,
    ):
        batch_rows = _oracle_capability_batch(
            {"math": math_model, "string": model},
            batch_records,
            tokenizer,
            device=device,
            max_new_tokens=oracle_generation_budget,
        )
        oracle_rows.extend(batch_rows)
        for row in batch_rows:
            append_jsonl(row, oracle_rows_path)
        completed_examples += len(batch_records)
        elapsed = time.time() - started_at
        rate = completed_examples / max(1e-9, elapsed)
        progress = {
            "format": "cftn_text_v1_3_native_evaluation_status_v1",
            "state": "running",
            "pid": os.getpid(),
            "split_index": len(selected_splits) + 1,
            "splits_total": len(selected_splits) + 1,
            "split": "joint_test_oracle_native_capability",
            "batch_completed": batch_index,
            "batches_total": oracle_batches,
            "split_examples_completed": min(
                len(joint_records),
                batch_index * int(config["evaluation"]["generation_batch_size"]),
            ),
            "split_examples_total": len(joint_records),
            "overall_examples_completed": completed_examples,
            "overall_examples_total": total_examples,
            "overall_progress": completed_examples / max(1, total_examples),
            "elapsed_seconds": elapsed,
            "eta_seconds": (
                (total_examples - completed_examples) / rate if rate > 0 else None
            ),
            "gpu": gpu_status(),
        }
        atomic_json_dump(progress, status_path)
        if progress["overall_progress"] >= next_boundary or batch_index == oracle_batches:
            append_jsonl(progress, progress_path)
            while progress["overall_progress"] >= next_boundary:
                next_boundary += 0.10
    oracle_summary = _oracle_capability_summary(oracle_rows)
    math_familiar = float(math_report["splits"]["test"]["generation"]["exact_accuracy"])
    threshold = float(config["acceptance"]["minimum_native_familiar_exact_accuracy"])
    oracle_threshold = float(
        config["acceptance"]["minimum_task_matched_oracle_exact_accuracy"]
    )
    minimum_coverage = float(
        config["acceptance"]["minimum_primary_competence_coverage"]
    )
    oracle_cells = oracle_summary["by_specialist_and_task_class"]
    gates = {
        "math_familiar": math_familiar >= threshold,
        "string_familiar": splits["string_test"]["exact_accuracy"] >= threshold,
        "task_matched_math": all(
            int(item["examples"]) > 0
            and float(item["exact_accuracy"]) >= oracle_threshold
            for item in oracle_cells["math"].values()
        ),
        "task_matched_string": all(
            int(item["examples"]) > 0
            and float(item["exact_accuracy"]) >= oracle_threshold
            for item in oracle_cells["string"].values()
        ),
        "primary_competence_coverage": float(oracle_summary["required_coverage"])
        >= minimum_coverage,
    }
    gates["pass"] = all(gates.values())
    report = {
        "format": "cftn_text_v1_3_native_specialist_report_v1",
        "state": "passed" if gates["pass"] else "failed",
        "string_checkpoint": str(checkpoint_path.resolve()),
        "string_checkpoint_sha256": file_sha256(checkpoint_path),
        "string_splits": splits,
        "math_report": str(math_report_path.resolve()),
        "math_report_sha256": file_sha256(math_report_path),
        "math_splits": math_report["splits"],
        "threshold": threshold,
        "task_matched_oracle_threshold": oracle_threshold,
        "minimum_primary_competence_coverage": minimum_coverage,
        "generation_contract": {
            "policy": specialist_generation_policy,
            "configured_max_new_tokens": int(
                config["evaluation"]["max_specialist_new_tokens"]
            ),
            "effective_max_new_tokens": {
                "math": math_generation_budget,
                "string": string_generation_budget,
                "oracle_batch": oracle_generation_budget,
            },
            "per_sequence_hard_stop": "EOS or tower max_sequence_length",
            "repair_reason": (
                "full_context_v1 removes the observed 96-token truncation of "
                "otherwise-correct long string traces"
                if specialist_generation_policy == "full_context_v1"
                else None
            ),
            "evaluator_sha256": file_sha256(Path(__file__)),
        },
        "joint_test_oracle_capability": {
            **oracle_summary,
            "generations": str(oracle_rows_path.resolve()),
        },
        "gates": gates,
        "revision_sha256": config["_meta"]["sha256"],
    }
    atomic_json_dump(report, artifact / "report.json")
    atomic_json_dump(
        {
            "format": "cftn_text_v1_3_native_evaluation_status_v1",
            "state": "completed",
            "pid": os.getpid(),
            "split_index": len(selected_splits) + 1,
            "splits_total": len(selected_splits) + 1,
            "overall_examples_completed": completed_examples,
            "overall_examples_total": total_examples,
            "overall_progress": 1.0,
            "elapsed_seconds": time.time() - started_at,
            "eta_seconds": 0.0,
            "report": str((artifact / "report.json").resolve()),
            "gpu": gpu_status(),
        },
        status_path,
    )
    if not gates["pass"]:
        raise RuntimeError(f"V1.3 native specialist precondition failed: {gates}")
    return report


_JOINT_ARMS: dict[str, dict[str, Any]] = {
    "gpt_alone": {"all_closed": True, "wake_mode": "oracle"},
    "dense_cftn": {"wake_mode": "dense"},
    "learned_wake_cftn": {"wake_mode": "hard"},
    "oracle_wake_cftn": {"wake_mode": "oracle"},
    "all_closed": {"all_closed": True, "wake_mode": "oracle"},
    "math_disabled": {"wake_mode": "hard", "disabled_specialists": {"math"}},
    "string_disabled": {"wake_mode": "hard", "disabled_specialists": {"string"}},
    "math_request_disabled": {"wake_mode": "hard", "disabled_requests": {"math"}},
    "string_request_disabled": {"wake_mode": "hard", "disabled_requests": {"string"}},
    "math_return_disabled": {"wake_mode": "hard", "disabled_returns": {"math"}},
    "string_return_disabled": {"wake_mode": "hard", "disabled_returns": {"string"}},
    "messages_shuffled": {"wake_mode": "hard", "shuffled_messages": True},
    "first_round_return_swapped": {
        "wake_mode": "oracle",
        "swap_first_round_returns": True,
    },
    "wrong_specialist_forced": {"wake_mode": "oracle", "wrong_specialist": True},
    "fixed_open": {"wake_mode": "dense", "fixed_open": True},
    "one_round": {"wake_mode": "hard", "maximum_rounds": 1},
}


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [bool(value["correct"]) for value in rows]
    valid = [
        extract_completion_answer(str(value["generation"])) is not None
        for value in rows
    ]
    supported = [
        value for value in rows if value.get("oracle_specialist_capable") is not False
    ]
    supported_correct = [bool(value["correct"]) for value in supported]
    supported_valid = [
        extract_completion_answer(str(value["generation"])) is not None
        for value in supported
    ]
    report: dict[str, Any] = {
        "examples": len(rows),
        "exact_accuracy": sum(correct) / max(1, len(correct)),
        "valid_rate": sum(valid) / max(1, len(valid)),
        "correct": correct,
        "competence_supported": {
            "examples": len(supported),
            "coverage": len(supported) / max(1, len(rows)),
            "exact_accuracy": sum(supported_correct) / max(1, len(supported_correct)),
            "valid_rate": sum(supported_valid) / max(1, len(supported_valid)),
        },
    }
    wake_rows = [value for value in rows if value.get("wake") is not None]
    if wake_rows:
        true_positive = false_positive = false_negative = exact_sets = 0
        false_wakes = pure_examples = 0
        active = 0
        active_by_name = {name: 0 for name in SPECIALISTS}
        halt_rounds: list[int] = []
        for row in wake_rows:
            activations = row["wake"]["activations"]
            targets = row["required_specialists_by_round"][: len(activations)]
            predicted_flat: list[bool] = []
            target_flat: list[bool] = []
            for round_activations, round_required in zip(activations, targets):
                for specialist_index, name in enumerate(SPECIALISTS):
                    predicted_flat.append(float(round_activations[specialist_index]) >= 0.5)
                    target_flat.append(name in round_required)
            true_positive += sum(a and b for a, b in zip(predicted_flat, target_flat))
            false_positive += sum(a and not b for a, b in zip(predicted_flat, target_flat))
            false_negative += sum(not a and b for a, b in zip(predicted_flat, target_flat))
            exact_sets += int(predicted_flat == target_flat)
            if row["task_class"] == "pure_language":
                pure_examples += 1
                false_wakes += int(any(predicted_flat))
            active += int(row["active_specialist_executions"])
            for name in SPECIALISTS:
                active_by_name[name] += int(
                    row.get("active_specialist_executions_by_name", {}).get(name, 0)
                )
            halts = row["wake"].get("halt_probabilities", [])
            halt_rounds.append(
                next(
                    (index + 1 for index, probability in enumerate(halts) if probability >= 0.5),
                    len(halts),
                )
            )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        report["wake"] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "exact_required_set_accuracy": exact_sets / max(1, len(wake_rows)),
            "pure_language_false_wake_rate": false_wakes / max(1, pure_examples),
            "mean_active_specialist_executions": active / max(1, len(wake_rows)),
            "mean_active_executions_by_specialist": {
                name: active_by_name[name] / max(1, len(wake_rows))
                for name in SPECIALISTS
            },
            "mean_halt_round": sum(halt_rounds) / max(1, len(halt_rounds)),
        }
    return report


def _paired_subset_interval(
    left: list[bool],
    right: list[bool],
    indices: list[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not indices:
        return {
            "examples": 0,
            "available": False,
            "mean_difference": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
        }
    return {
        "examples": len(indices),
        "available": True,
        **paired_bootstrap_interval(
            [left[index] for index in indices],
            [right[index] for index in indices],
            samples=samples,
            seed=seed,
        ),
    }


def evaluate_v1_3_causal_suite(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    maximum_examples: int | None = None,
    specialist_generation_policy: str = "configured",
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit_v1_2_pass(config)
    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    final_phase = config["integration_training"]["phases"][-1]["name"]
    summary_path = Path(config["paths"]["artifact_root"]) / final_phase / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("V1.3 hardened-wake phase is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=summary["best_checkpoint"]
    )
    model.eval()
    specialist_generation_budgets = {
        name: resolve_specialist_generation_budget(
            config, model.specialists[name], specialist_generation_policy
        )
        for name in SPECIALISTS
    }
    oracle_generation_budget = max(specialist_generation_budgets.values())
    math_tokenizer = ByteMathTokenizer()
    collator = V13JointCollator(
        math_tokenizer,
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    artifact = Path(config["paths"]["artifact_root"]) / "sealed_evaluation"
    artifact.mkdir(parents=True, exist_ok=True)
    status_path = artifact / "status.json"
    progress_path = artifact / "progress.jsonl"
    progress_path.unlink(missing_ok=True)
    started_at = time.time()
    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact,
        stage="v1_3_sealed_causal_evaluation",
        config={
            "revision_sha256": config["_meta"]["sha256"],
            "checkpoint_sha256": summary["best_checkpoint_sha256"],
        },
    )
    split_reports: dict[str, Any] = {}
    primary_split = str(config["evaluation"]["primary_split"])
    diagnostic_splits = tuple(config["evaluation"]["diagnostic_splits"])
    split_names = (primary_split, *diagnostic_splits)
    limit = maximum_examples or int(config["evaluation"]["maximum_examples_per_split"])
    arms_total = len(_JOINT_ARMS) + 3
    total_arm_examples = sum(
        min(int(manifest["splits"][split]["examples"]), int(limit)) * arms_total
        for split in split_names
    )
    completed_arm_examples = 0
    next_boundary = 0.10
    atomic_json_dump(
        {
            "format": "cftn_text_v1_3_evaluation_status_v1",
            "state": "running",
            "pid": os.getpid(),
            "split_index": 0,
            "splits_total": len(split_names),
            "overall_progress": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
        },
        status_path,
    )
    for split_index, split in enumerate(split_names, start=1):
        records = V13Dataset(data_root / manifest["splits"][split]["path"]).records
        records = records[: int(limit)]
        arm_rows: dict[str, list[dict[str, Any]]] = {
            name: []
            for name in (*_JOINT_ARMS, "math_alone", "string_alone", "serial_pipeline")
        }
        arm_seconds = {name: 0.0 for name in arm_rows}
        arm_peak_memory = {name: 0 for name in arm_rows}
        oracle_rows: list[dict[str, Any]] = []
        oracle_rows_path = artifact / f"{split}_oracle_native_capability.jsonl"
        oracle_rows_path.unlink(missing_ok=True)
        batches_total = math.ceil(
            len(records) / int(config["evaluation"]["generation_batch_size"])
        )
        split_examples_completed = 0
        for batch_index, batch_records in enumerate(_batched(
            records, int(config["evaluation"]["generation_batch_size"])
        ), start=1):
            raw_batch = collator(batch_records)
            batch = move_v1_3_batch(raw_batch, device)
            batch_oracle_rows = _oracle_capability_batch(
                model.specialists,
                batch_records,
                math_tokenizer,
                device=device,
                max_new_tokens=oracle_generation_budget,
            )
            oracle_rows.extend(batch_oracle_rows)
            for oracle_row in batch_oracle_rows:
                append_jsonl(oracle_row, oracle_rows_path)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            math_native = generate_native_specialist(
                model.specialists["math"],
                batch_records,
                math_tokenizer,
                device=device,
                max_new_tokens=specialist_generation_budgets["math"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            arm_seconds["math_alone"] += time.perf_counter() - started
            arm_peak_memory["math_alone"] = max(
                arm_peak_memory["math_alone"],
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            string_native = generate_native_specialist(
                model.specialists["string"],
                batch_records,
                math_tokenizer,
                device=device,
                max_new_tokens=specialist_generation_budgets["string"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            arm_seconds["string_alone"] += time.perf_counter() - started
            arm_peak_memory["string_alone"] = max(
                arm_peak_memory["string_alone"],
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            serial = _serial_pipeline_gpt(
                model,
                batch_records,
                math_native,
                string_native,
                gpt_tokenizer,
                max_new_tokens=int(config["evaluation"]["max_gpt_new_tokens"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            arm_seconds["serial_pipeline"] += time.perf_counter() - started
            arm_peak_memory["serial_pipeline"] = max(
                arm_peak_memory["serial_pipeline"],
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            )
            for arm, generations in (
                ("math_alone", math_native),
                ("string_alone", string_native),
                ("serial_pipeline", serial),
            ):
                for row_index, (record, generation) in enumerate(
                    zip(batch_records, generations)
                ):
                    arm_rows[arm].append(
                        {
                            "record_id": record["record_id"],
                            "task_class": record["task_class"],
                            "required_specialists": record["required_specialists"],
                            "required_specialists_by_round": record[
                                "required_specialists_by_round"
                            ],
                            "target": record["target_answer"],
                            "generation": generation,
                            "specialist_generations": {
                                "math": math_native[len(arm_rows[arm]) % len(batch_records)]
                                if batch_records
                                else "",
                                "string": string_native[len(arm_rows[arm]) % len(batch_records)]
                                if batch_records
                                else "",
                            },
                            "wake": None,
                            "oracle_specialist_capable": bool(
                                batch_oracle_rows[row_index]["supported"]
                            ),
                            "active_specialist_executions": (
                                1 if arm in {"math_alone", "string_alone"} else 2
                            ),
                            "correct": _exact(generation, record),
                        }
                    )
            for arm, options in _JOINT_ARMS.items():
                options = dict(options)
                rounds = int(
                    options.pop(
                        "maximum_rounds", config["runtime"]["maximum_callosal_rounds"]
                    )
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                generated = generate_joint_batch(
                    model,
                    batch,
                    math_tokenizer,
                    gpt_tokenizer,
                    maximum_rounds=rounds,
                    max_specialist_new_tokens=oracle_generation_budget,
                    max_gpt_new_tokens=int(config["evaluation"]["max_gpt_new_tokens"]),
                    **options,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                arm_seconds[arm] += time.perf_counter() - started
                arm_peak_memory[arm] = max(
                    arm_peak_memory[arm],
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0,
                )
                for row_index, (record, result) in enumerate(
                    zip(batch_records, generated)
                ):
                    arm_rows[arm].append(
                        {
                            "record_id": record["record_id"],
                            "task_class": record["task_class"],
                            "required_specialists": record["required_specialists"],
                            "required_specialists_by_round": record[
                                "required_specialists_by_round"
                            ],
                            "target": record["target_answer"],
                            "oracle_specialist_capable": bool(
                                batch_oracle_rows[row_index]["supported"]
                            ),
                            **result,
                            "correct": _exact(result["generation"], record),
                        }
                    )
            split_examples_completed += len(batch_records)
            completed_arm_examples += len(batch_records) * arms_total
            elapsed = time.time() - started_at
            overall = completed_arm_examples / max(1, total_arm_examples)
            rate = completed_arm_examples / max(1e-9, elapsed)
            progress = {
                "format": "cftn_text_v1_3_evaluation_status_v1",
                "state": "running",
                "pid": os.getpid(),
                "split_index": split_index,
                "splits_total": len(split_names),
                "split": split,
                "batch_completed": batch_index,
                "batches_total": batches_total,
                "split_examples_completed": split_examples_completed,
                "split_examples_total": len(records),
                "split_progress": split_examples_completed / max(1, len(records)),
                "arm_count": arms_total,
                "completed_arm_examples": completed_arm_examples,
                "total_arm_examples": total_arm_examples,
                "overall_progress": overall,
                "elapsed_seconds": elapsed,
                "eta_seconds": (
                    (total_arm_examples - completed_arm_examples) / rate
                    if rate > 0
                    else None
                ),
                "current_arm": list(_JOINT_ARMS)[-1],
                "gpu": gpu_status(),
            }
            atomic_json_dump(progress, status_path)
            if overall >= next_boundary or split_examples_completed == len(records):
                append_jsonl(progress, progress_path)
                tracker.log(
                    {"evaluation": progress},
                    global_step=completed_arm_examples,
                    event="evaluation_progress",
                )
                while overall >= next_boundary:
                    next_boundary += 0.10
        arm_seconds["serial_pipeline"] += (
            arm_seconds["math_alone"] + arm_seconds["string_alone"]
        )
        arm_peak_memory["serial_pipeline"] = max(
            arm_peak_memory["serial_pipeline"],
            arm_peak_memory["math_alone"],
            arm_peak_memory["string_alone"],
        )
        arm_metrics: dict[str, Any] = {}
        for arm, rows in arm_rows.items():
            path = artifact / f"{split}_{arm}.jsonl"
            path.unlink(missing_ok=True)
            for row in rows:
                append_jsonl(row, path)
            metrics = _arm_metrics(rows)
            metrics.pop("correct")
            metrics["generations"] = str(path.resolve())
            metrics["elapsed_seconds"] = arm_seconds[arm]
            metrics["examples_per_second"] = len(rows) / max(1e-12, arm_seconds[arm])
            metrics["peak_memory_bytes"] = arm_peak_memory[arm]
            by_class: dict[str, Any] = {}
            for task_class in sorted({row["task_class"] for row in rows}):
                selected = [row for row in rows if row["task_class"] == task_class]
                class_metrics = _arm_metrics(selected)
                class_metrics.pop("correct")
                by_class[task_class] = class_metrics
            metrics["by_task_class"] = by_class
            arm_metrics[arm] = metrics
        learned_correct = [row["correct"] for row in arm_rows["learned_wake_cftn"]]
        baselines = {
            name: [row["correct"] for row in arm_rows[name]]
            for name in ("gpt_alone", "math_alone", "string_alone")
        }
        strongest_name = max(
            baselines,
            key=lambda name: sum(baselines[name]),
        )
        joint_classes = {
            "language_dependent_math",
            "multi_parallel",
            "multi_sequential",
        }
        joint_indices = [
            index
            for index, row in enumerate(arm_rows["learned_wake_cftn"])
            if row["task_class"] in joint_classes
        ]
        supported_joint_indices = [
            index
            for index in joint_indices
            if arm_rows["learned_wake_cftn"][index]["oracle_specialist_capable"]
        ]
        joint_baseline_name = max(
            baselines,
            key=lambda name: sum(baselines[name][index] for index in joint_indices),
        )
        supported_joint_baseline_name = max(
            baselines,
            key=lambda name: sum(
                baselines[name][index] for index in supported_joint_indices
            ),
        )
        oracle_summary = _oracle_capability_summary(oracle_rows)
        split_reports[split] = {
            "examples": len(records),
            "role": "primary" if split == primary_split else "diagnostic_non_gating",
            "competence_contract": {
                **oracle_summary,
                "rows": str(oracle_rows_path.resolve()),
            },
            "arms": arm_metrics,
            "learned_vs_strongest_available_baseline": {
                "baseline": strongest_name,
                **paired_bootstrap_interval(
                    learned_correct,
                    baselines[strongest_name],
                    samples=int(config["evaluation"]["bootstrap_samples"]),
                    seed=int(config["revision"]["seed"]),
                ),
            },
            "learned_vs_serial_pipeline": paired_bootstrap_interval(
                learned_correct,
                [row["correct"] for row in arm_rows["serial_pipeline"]],
                samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["revision"]["seed"]),
            ),
            "central_joint_learned_vs_strongest_individual": {
                "baseline": joint_baseline_name,
                **paired_bootstrap_interval(
                    [learned_correct[index] for index in joint_indices],
                    [baselines[joint_baseline_name][index] for index in joint_indices],
                    samples=int(config["evaluation"]["bootstrap_samples"]),
                    seed=int(config["revision"]["seed"]),
                ),
            },
            "central_supported_joint_learned_vs_strongest_individual": {
                "baseline": supported_joint_baseline_name,
                **_paired_subset_interval(
                    learned_correct,
                    baselines[supported_joint_baseline_name],
                    supported_joint_indices,
                    samples=int(config["evaluation"]["bootstrap_samples"]),
                    seed=int(config["revision"]["seed"]),
                ),
            },
        }
    report = {
        "format": "cftn_text_v1_3_causal_evaluation_v1",
        "state": "completed",
        "checkpoint": summary["best_checkpoint"],
        "checkpoint_sha256": summary["best_checkpoint_sha256"],
        "revision_sha256": config["_meta"]["sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "provenance": provenance,
        "evaluation_contract": {
            "primary_split": primary_split,
            "diagnostic_splits": list(diagnostic_splits),
            "competence_conditioned_primary_claims": True,
            "diagnostic_splits_are_non_gating": True,
            "specialist_generation": {
                "policy": specialist_generation_policy,
                "configured_max_new_tokens": int(
                    config["evaluation"]["max_specialist_new_tokens"]
                ),
                "effective_max_new_tokens": specialist_generation_budgets,
                "joint_effective_max_new_tokens": oracle_generation_budget,
                "per_sequence_hard_stop": "EOS or tower max_sequence_length",
                "evaluator_sha256": file_sha256(Path(__file__)),
            },
        },
        "compute_contract": {
            "specialist_parameter_counts": {
                name: sum(parameter.numel() for parameter in model.specialists[name].parameters())
                for name in SPECIALISTS
            },
            "measurement": (
                "Actual conditional specialist executions, wall-clock arm latency, and "
                "CUDA peak allocated memory are reported. Hardware energy and exact "
                "kernel FLOPs are unavailable on this local runner."
            ),
        },
        "splits": split_reports,
        "gpu": gpu_status(),
    }
    atomic_json_dump(report, artifact / "report.json")
    completed_status = {
        "format": "cftn_text_v1_3_evaluation_status_v1",
        "state": "completed",
        "pid": os.getpid(),
        "split_index": len(split_names),
        "splits_total": len(split_names),
        "completed_arm_examples": completed_arm_examples,
        "total_arm_examples": total_arm_examples,
        "overall_progress": 1.0,
        "elapsed_seconds": time.time() - started_at,
        "eta_seconds": 0.0,
        "report": str((artifact / "report.json").resolve()),
        "gpu": gpu_status(),
    }
    atomic_json_dump(completed_status, status_path)
    tracker.log(
        {"evaluation": completed_status},
        global_step=completed_arm_examples,
        event="evaluation_completed",
    )
    tracker.finish()
    return report
