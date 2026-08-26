from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .dataset import SHARED_MATH_INPUT_VIEW, math_problem_for_view
from .tokenizer import ByteMathTokenizer, SequenceTooLongError
from .v2_metrics import extract_v2_answer, score_v2_generations


_BREAKDOWN_DIMENSIONS = ("source", "family", "difficulty")
DEFAULT_V2_GENERATION_VALIDATION: dict[str, int | bool] = {
    "enabled": True,
    "every_epochs": 1,
    "examples": 96,
    "batch_size": 16,
    "max_new_tokens": 512,
    "failure_examples": 8,
}


def _new_group() -> dict[str, float | int]:
    return {
        "examples": 0,
        "supervised_tokens": 0,
        "correct_tokens": 0,
        "correct_sequences": 0,
        "language_loss_sum": 0.0,
    }


def update_teacher_forced_breakdowns(
    groups: dict[str, dict[str, dict[str, float | int]]],
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    records: list[dict[str, Any]],
) -> None:
    """Accumulate token-weighted teacher-forced metrics for record cohorts."""

    predictions = logits[:, :-1].argmax(dim=-1)
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    correct = predictions.eq(targets) & valid
    sequence_correct = (correct | ~valid).all(dim=1) & valid.any(dim=1)
    token_losses = F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        targets.contiguous().view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    for row, record in enumerate(records):
        supervised_tokens = int(valid[row].sum().item())
        correct_tokens = int(correct[row].sum().item())
        loss_sum = float(token_losses[row][valid[row]].sum().item())
        for dimension in _BREAKDOWN_DIMENSIONS:
            name = str(record.get(dimension, "unknown"))
            bucket = groups.setdefault(dimension, {}).setdefault(name, _new_group())
            bucket["examples"] = int(bucket["examples"]) + 1
            bucket["supervised_tokens"] = (
                int(bucket["supervised_tokens"]) + supervised_tokens
            )
            bucket["correct_tokens"] = int(bucket["correct_tokens"]) + correct_tokens
            bucket["correct_sequences"] = int(bucket["correct_sequences"]) + int(
                sequence_correct[row].item()
            )
            bucket["language_loss_sum"] = (
                float(bucket["language_loss_sum"]) + loss_sum
            )


def summarize_teacher_forced_breakdowns(
    groups: dict[str, dict[str, dict[str, float | int]]]
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for dimension, buckets in groups.items():
        result[f"by_{dimension}"] = {}
        for name, bucket in sorted(buckets.items()):
            examples = int(bucket["examples"])
            tokens = int(bucket["supervised_tokens"])
            result[f"by_{dimension}"][name] = {
                "examples": examples,
                "supervised_tokens": tokens,
                "language_loss": float(bucket["language_loss_sum"])
                / max(1, tokens),
                "teacher_forced_token_accuracy": int(bucket["correct_tokens"])
                / max(1, tokens),
                "teacher_forced_sequence_accuracy": int(
                    bucket["correct_sequences"]
                )
                / max(1, examples),
            }
    return result


def stratified_validation_panel(
    records: Iterable[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Select deterministic round-robin coverage across source/family/difficulty."""

    maximum = max(0, int(maximum))
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("source", "unknown")),
            str(record.get("family", "unknown")),
            int(record.get("difficulty", 0)),
        )
        buckets[key].append(record)
    ordered = [buckets[key] for key in sorted(buckets)]
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < maximum:
        added = False
        for bucket in ordered:
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) >= maximum:
                    break
        if not added:
            break
        offset += 1
    return selected


@torch.inference_mode()
def evaluate_generation_panel(
    model: Any,
    tokenizer: ByteMathTokenizer,
    records: Iterable[dict[str, Any]],
    *,
    maximum_examples: int,
    batch_size: int,
    max_new_tokens: int,
    failure_examples: int,
    rows_path: str | Path | None = None,
    input_view: str = SHARED_MATH_INPUT_VIEW,
    require_eos: bool = False,
) -> dict[str, Any]:
    """Run a compact native greedy-generation panel during math training."""

    from .specialist_evaluation import generate_math_tower

    selected = stratified_validation_panel(records, maximum_examples)
    eligible: list[dict[str, Any]] = []
    excluded_over_context = 0
    for record in selected:
        try:
            tokenizer.encode_generation_prefix(
                math_problem_for_view(record, input_view), model.max_sequence_length
            )
        except SequenceTooLongError:
            excluded_over_context += 1
            continue
        eligible.append(record)
    started_at = time.time()
    generations: list[str] = []
    terminations = []
    for start in range(0, len(eligible), max(1, int(batch_size))):
        chunk = eligible[start : start + max(1, int(batch_size))]
        if require_eos:
            from tools.pilot_math_primitives import generate_with_termination
            decoded = generate_with_termination(model, tokenizer,
                [math_problem_for_view(record, input_view) for record in chunk], int(max_new_tokens))
            generated = [d["generation"] for d in decoded]
            terminations.extend(decoded)
        else:
            generated, _ = generate_math_tower(
                model, tokenizer, [math_problem_for_view(record, input_view) for record in chunk],
                max_new_tokens=int(max_new_tokens))
        generations.extend(generated)
    clean = ([d["eos_terminated"] and not d["unexpected_control_token"] and not d["context_limit_hit"] and not d["budget_hit"]
              for d in terminations] if require_eos else [True] * len(generations))
    metrics, correctness = score_v2_generations([g if ok else "" for g, ok in zip(generations, clean)], eligible)
    trace_groups = defaultdict(lambda: {"examples": 0, "correct": 0})
    for record, generation, ok in zip(eligible, generations, clean):
        group = trace_groups[str(record.get("family", "unknown"))]
        group["examples"] += 1
        group["correct"] += int(ok and generation.strip() == str(record.get("target_trace", "")))
    trace_exact = {name: {**group, "rate": group["correct"] / max(1, group["examples"])}
                   for name, group in trace_groups.items()}
    rows = []
    for record, generation, correct in zip(eligible, generations, correctness):
        rows.append(
            {
                "record_id": record.get("record_id", record.get("content_id")),
                "source": record.get("source"),
                "family": record.get("family"),
                "difficulty": record.get("difficulty"),
                "problem": record.get("problem"),
                "expected_answer": record.get("normalized_answer"),
                "generation": generation,
                "parsed_answer": extract_v2_answer(generation),
                "correct": bool(correct),
            }
        )
    if rows_path is not None:
        output = Path(rows_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    failures = [row for row in rows if not row["correct"]][: max(0, int(failure_examples))]
    elapsed = time.time() - started_at
    return {
        **metrics,
        "trace_exact_by_family": trace_exact,
        "require_eos": require_eos,
        "unclean_terminations": len(clean) - sum(clean),
        "budget_hits": sum(d["budget_hit"] for d in terminations),
        "panel_policy": "deterministic_round_robin_source_family_difficulty_v1",
        "input_view": str(input_view),
        "requested_examples": int(maximum_examples),
        "excluded_over_context": excluded_over_context,
        "excluded_over_context_rate": excluded_over_context
        / max(1, excluded_over_context + len(eligible)),
        "max_new_tokens": int(max_new_tokens),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(eligible) / max(1e-9, elapsed),
        "failure_examples": failures,
        "rows_path": str(Path(rows_path).resolve()) if rows_path is not None else None,
    }
