"""Versioned V11 procedural targets for KS2 multiplication and division.

This module is deliberately separate from earlier dataset recipes.  It adds
only in-scope structural supervision: progressive partial-product aggregation
for long multiplication and executable long-division steps for exact division.
"""

from __future__ import annotations

from typing import Any


V11_DATASET_RECIPE = "canonical_v11_ks2_procedures_v1"
V11_GENERATOR_VERSION = "v11_ks2_procedures_v1"


def solve_v11_procedure(
    math_ir: dict[str, Any], *, criterion: str | None = None
) -> tuple[str, list[dict[str, Any]]] | None:
    """Return the V11 derivation for operations whose procedures are upgraded."""

    operation = str(math_ir.get("op", ""))
    if operation == "multiply" and criterion == "KS2-LONG-MULTIPLY":
        left, right = int(math_ir["left"]), int(math_ir["right"])
        result = left * right
        left_tens, left_ones = divmod(left, 10)
        right_tens, right_ones = divmod(right, 10)
        left_tens_value, right_tens_value = left_tens * 10, right_tens * 10
        partials = (
            left_tens_value * right_tens_value,
            left_tens_value * right_ones,
            left_ones * right_tens_value,
            left_ones * right_ones,
        )
        derivation: list[dict[str, Any]] = [
            {"ones": left_ones, "op": "decompose", "tens": left_tens, "value": left},
            {"ones": right_ones, "op": "decompose", "tens": right_tens, "value": right},
            {"left": left_tens_value, "op": "multiply", "result": str(partials[0]), "right": right_tens_value},
            {"left": left_tens_value, "op": "multiply", "result": str(partials[1]), "right": right_ones},
            {"left": left_ones, "op": "multiply", "result": str(partials[2]), "right": right_tens_value},
            {"left": left_ones, "op": "multiply", "result": str(partials[3]), "right": right_ones},
        ]
        running = partials[0]
        for partial in partials[1:]:
            next_total = running + partial
            derivation.append(
                {"left": running, "op": "add", "result": str(next_total), "right": partial}
            )
            running = next_total
        if running != result:
            raise ValueError("long-multiplication aggregation disagrees with product")
        return str(result), derivation

    if operation == "divide" and criterion == "KS2-EXACT-DIVIDE":
        dividend, divisor = int(math_ir["dividend"]), int(math_ir["divisor"])
        if divisor <= 0 or dividend % divisor:
            raise ValueError("V11 exact division requires a non-zero exact divisor")
        quotient = dividend // divisor
        remainder = 0
        digits: list[int] = []
        derivation = []
        for digit in str(dividend):
            partial = remainder * 10 + int(digit)
            quotient_digit, remainder = divmod(partial, divisor)
            digits.append(quotient_digit)
            derivation.append(
                {
                    "dividend": partial,
                    "divisor": divisor,
                    "op": "long_divide_step",
                    "quotient_digit": quotient_digit,
                    "remainder": remainder,
                }
            )
        derivation.extend(
            [
                {"digits": digits, "op": "compose", "result": str(quotient)},
                {"left": divisor, "op": "multiply", "result": str(dividend), "right": quotient},
            ]
        )
        if remainder or int("".join(str(value) for value in digits)) != quotient:
            raise ValueError("long-division derivation disagrees with quotient")
        return str(quotient), derivation

    return None
