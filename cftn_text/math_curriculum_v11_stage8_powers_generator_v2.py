"""Trace-safe revision of the structured V11 Stage-8 powers generator."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .math_curriculum_v11_stage8_powers_generator import (
    solve_stage8_powers_procedure,
)


V11_STAGE8_POWERS_V2_DATASET_RECIPE = "canonical_v11_stage8_powers_scaffold_v2"
V11_STAGE8_POWERS_V2_GENERATOR_VERSION = "v11_stage8_powers_scaffold_v2"

_STRATEGIES = (
    "repeated_multiplication",
    "successive_powers",
    "paired_squares",
    "sign_then_magnitude",
)


def powers_candidate_irs() -> Iterator[dict[str, Any]]:
    """Emit only procedure targets proven to fit the 2,048-token contract."""

    for base in range(-100, 101):
        for exponent in range(2, 13):
            for strategy in _STRATEGIES:
                yield {
                    "type": "math_problem_v1",
                    "op": "power",
                    "base": base,
                    "exponent": exponent,
                    "representation": "integer_power",
                    "strategy": strategy,
                }


__all__ = [
    "V11_STAGE8_POWERS_V2_DATASET_RECIPE",
    "V11_STAGE8_POWERS_V2_GENERATOR_VERSION",
    "powers_candidate_irs",
    "solve_stage8_powers_procedure",
]
