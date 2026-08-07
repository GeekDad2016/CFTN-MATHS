from __future__ import annotations

import math
import re
from typing import Any, Iterable

import torch


ANSWER_PATTERN = re.compile(r"<answer>\s*([+-]?\d+)\s*</answer>", re.IGNORECASE)


def extract_answer(text: str) -> int | None:
    match = ANSWER_PATTERN.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def answer_generation_metrics(
    generations: Iterable[str], targets: Iterable[int]
) -> dict[str, float | int]:
    generated = list(generations)
    expected = [int(value) for value in targets]
    if len(generated) != len(expected):
        raise ValueError("generation and target lengths differ")
    parsed = [extract_answer(text) for text in generated]
    valid = [value is not None for value in parsed]
    correct = [value == target for value, target in zip(parsed, expected)]
    count = len(expected)
    return {
        "examples": count,
        "valid_answers": sum(valid),
        "correct_answers": sum(correct),
        "valid_rate": sum(valid) / count if count else 0.0,
        "exact_accuracy": sum(correct) / count if count else 0.0,
    }


def masked_token_statistics(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int, int]:
    predictions = logits[:, :-1].argmax(dim=-1)
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    correct = predictions.eq(targets) & valid
    sequence_correct = (correct | ~valid).all(dim=1) & valid.any(dim=1)
    return int(correct.sum()), int(valid.sum()), int(sequence_correct.sum())


def summarize_gate(gates: list[torch.Tensor]) -> dict[str, float]:
    if not gates:
        return {"mean": 0.0, "std": 0.0, "minimum": 0.0, "maximum": 0.0}
    values = torch.cat([gate.detach().float().reshape(-1).cpu() for gate in gates])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def paired_bootstrap_interval(
    candidate_correct: list[bool],
    baseline_correct: list[bool],
    samples: int = 10_000,
    seed: int = 719,
) -> dict[str, float]:
    if len(candidate_correct) != len(baseline_correct) or not candidate_correct:
        raise ValueError("paired bootstrap requires equal nonempty samples")
    differences = [int(a) - int(b) for a, b in zip(candidate_correct, baseline_correct)]
    size = len(differences)
    # A paired correctness difference can only be -1, 0, or +1. A bootstrap
    # resample is therefore exactly a multinomial draw over those three values;
    # this avoids constructing a samples-by-examples index matrix.
    counts = torch.tensor(
        [
            differences.count(-1),
            differences.count(0),
            differences.count(1),
        ],
        dtype=torch.float64,
    )
    probabilities = counts / size
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        draws = torch.distributions.Multinomial(
            total_count=size, probs=probabilities
        ).sample((samples,))
    estimates = ((draws[:, 2] - draws[:, 0]) / size).sort().values
    low_index = max(0, math.floor(0.025 * (samples - 1)))
    high_index = min(samples - 1, math.ceil(0.975 * (samples - 1)))
    return {
        "mean_difference": sum(differences) / size,
        "ci95_low": float(estimates[low_index]),
        "ci95_high": float(estimates[high_index]),
    }


def control_report(
    outputs: dict[str, list[dict[str, str]]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    targets = [int(record["x"]) for record in records]
    report: dict[str, Any] = {}
    for condition, rows in outputs.items():
        if len(rows) != len(records):
            raise ValueError(f"condition {condition} has the wrong row count")
        report[condition] = {
            "gpt": answer_generation_metrics(
                [row["gpt_generation"] for row in rows], targets
            ),
            "math": answer_generation_metrics(
                [row["math_generation"] for row in rows], targets
            ),
        }
    return report
