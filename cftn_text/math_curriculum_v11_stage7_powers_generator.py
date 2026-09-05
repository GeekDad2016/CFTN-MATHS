"""Versioned Stage-7 candidate domain for the V11 KS3 powers remediation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


V11_STAGE7_POWERS_DATASET_RECIPE = "canonical_v11_stage7_powers_v1"
V11_STAGE7_POWERS_GENERATOR_VERSION = "v11_stage7_powers_v1"


def powers_candidate_irs() -> Iterator[dict[str, Any]]:
    """Provide enough distinct, in-scope KS3 integer-power objects.

    V11 supplied 1,087 distinct objects (-50..50, powers 2..12).  This
    remediation only broadens that same powers domain to -200..200 and
    exponents 2..20; no later curriculum operation is exposed.
    """

    for base in range(-200, 201):
        for exponent in range(2, 21):
            yield {"type": "math_problem_v1", "op": "power", "base": base, "exponent": exponent}
