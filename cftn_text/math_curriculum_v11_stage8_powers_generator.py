"""Versioned Stage-8 structured procedures for V11 KS3 powers.

This remediation keeps the existing integer-power scope while replacing the
one-line ``power -> result`` target with four executable views of the same
semantic problem. Holdout splits remain semantic-problem disjoint.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


V11_STAGE8_POWERS_DATASET_RECIPE = "canonical_v11_stage8_powers_scaffold_v1"
V11_STAGE8_POWERS_GENERATOR_VERSION = "v11_stage8_powers_scaffold_v1"

_STRATEGIES = (
    "repeated_multiplication",
    "successive_powers",
    "paired_squares",
    "sign_then_magnitude",
)


def powers_candidate_irs() -> Iterator[dict[str, Any]]:
    """Emit structured views over the established Stage-7 powers domain."""

    for base in range(-200, 201):
        for exponent in range(2, 21):
            for strategy in _STRATEGIES:
                yield {
                    "type": "math_problem_v1",
                    "op": "power",
                    "base": base,
                    "exponent": exponent,
                    "representation": "integer_power",
                    "strategy": strategy,
                }


def _multiply_chain(base: int, exponent: int, *, op: str) -> tuple[int, list[dict[str, Any]]]:
    running = base
    derivation: list[dict[str, Any]] = [
        {"base": base, "exponent": 1, "op": "power_seed", "result": str(base)}
    ]
    for power in range(2, exponent + 1):
        previous = running
        running *= base
        derivation.append(
            {
                "base": base,
                "exponent": power,
                "left": previous,
                "op": op,
                "result": str(running),
                "right": base,
            }
        )
    return running, derivation


def solve_stage8_powers_procedure(
    math_ir: dict[str, Any], *, criterion: str | None = None
) -> tuple[str, list[dict[str, Any]]] | None:
    """Render compact, executable subskill traces for KS3 integer powers."""

    if criterion != "KS3-POWERS" or math_ir.get("op") != "power":
        return None
    base, exponent = int(math_ir["base"]), int(math_ir["exponent"])
    if exponent < 2:
        raise ValueError("Stage-8 powers requires an exponent of at least two")
    strategy = str(math_ir.get("strategy", "repeated_multiplication"))
    if strategy == "repeated_multiplication":
        result, derivation = _multiply_chain(base, exponent, op="multiply_step")
    elif strategy == "successive_powers":
        result, derivation = _multiply_chain(base, exponent, op="power_step")
    elif strategy == "paired_squares":
        square = base * base
        derivation = [{"base": base, "exponent": 2, "op": "square", "result": str(square)}]
        pairs, remainder = divmod(exponent, 2)
        running = 1
        for index in range(pairs):
            previous = running
            running *= square
            derivation.append({"left": previous, "op": "multiply_square", "pair": index + 1, "result": str(running), "right": square})
        if remainder:
            previous = running
            running *= base
            derivation.append({"left": previous, "op": "multiply_remainder", "result": str(running), "right": base})
        result = running
    elif strategy == "sign_then_magnitude":
        magnitude, derivation = _multiply_chain(abs(base), exponent, op="magnitude_step")
        sign = -1 if base < 0 and exponent % 2 else 1
        result = sign * magnitude
        derivation.append({"base_sign": "negative" if base < 0 else "positive", "exponent": exponent, "op": "apply_sign", "result": str(result)})
    else:
        raise ValueError(f"unknown Stage-8 powers strategy: {strategy}")
    if result != base**exponent:
        raise ValueError("Stage-8 powers derivation disagrees with exponentiation")
    return str(result), derivation
