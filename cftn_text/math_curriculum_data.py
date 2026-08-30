from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from heapq import heappush, heapreplace, nsmallest
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import canonical_json
from .data_generator import file_sha256
from .v2_data import make_v2_record, validate_v2_record


FORMAT = "cftn_canonical_math_curriculum_v1"
SCHEMA = "cftn_canonical_math_record_v1"
SPLITS = ("train", "validation", "test")
EXPANDED_PROCEDURE_SCHEMA = "expanded_count_on_v1"
COMPACT_PROCEDURE_SCHEMA = "compact_executable_v2"
PROCEDURE_SCHEMAS = {EXPANDED_PROCEDURE_SCHEMA, COMPACT_PROCEDURE_SCHEMA}
V6_DATASET_RECIPE = "canonical_balanced_progression_v6"
V9_DATASET_RECIPE = "canonical_targeted_v5_stage235_v9"
V9_VARIANT_CRITERIA = frozenset(
    {"1NF-1", "1AS-2", "2NPV-1", "2NPV-2", "2AS-1", "2AS-2", "2MD-1", "2MD-2"}
)
PEDAGOGICAL_VARIANT_FIELDS = frozenset({"representation", "strategy"})


PHASES: tuple[dict[str, Any], ...] = (
    {
        "name": "y1_number_structure",
        "criteria": ("1NPV-1", "1NPV-2", "1AS-1"),
    },
    {
        "name": "y1_add_sub_fluency",
        "criteria": ("1NF-1", "1AS-2"),
    },
    {
        "name": "y2_place_value_and_across_10",
        "criteria": ("2NPV-1", "2NPV-2", "2AS-1", "2AS-2"),
    },
    {
        "name": "y2_add_sub_within_100",
        "criteria": ("2AS-3", "2AS-4"),
    },
    {
        "name": "y2_multiply_divide_2_5_10",
        "criteria": ("2MD-1", "2MD-2"),
    },
)

MASTER_EXTENSION_PHASES: tuple[dict[str, Any], ...] = (
    {
        "name": "ks2_four_operations",
        "level": "KS2",
        "criteria": ("KS2-MULTI-DIGIT", "KS2-LONG-MULTIPLY", "KS2-EXACT-DIVIDE"),
    },
    {
        "name": "ks2_fractions_ratio_geometry",
        "level": "KS2",
        "criteria": ("KS2-FRACTION-ADD", "KS2-PERCENT", "KS2-RECTANGLE"),
    },
    {
        "name": "secondary_algebra_number",
        "level": "secondary",
        "criteria": ("KS3-LINEAR", "KS3-POWERS", "KS3-PYTHAGORAS"),
    },
    {
        "name": "gcse_higher",
        "level": "GCSE",
        "criteria": ("GCSE-QUADRATIC", "GCSE-SIMULTANEOUS", "GCSE-SEQUENCE"),
    },
    {
        "name": "alevel_pure",
        "level": "A-level",
        "criteria": ("AL-DIFFERENTIATE", "AL-INTEGRATE", "AL-BINOMIAL-COEFFICIENT"),
    },
    {
        "name": "alevel_statistics_mechanics",
        "level": "A-level",
        "criteria": ("AL-COMBINATION", "AL-BINOMIAL-PROB", "AL-SUVAT"),
    },
    {
        "name": "undergraduate_calculus_linear_algebra",
        "level": "undergraduate",
        "criteria": ("UG-MATRIX-DET", "UG-MATRIX-SOLVE", "UG-POLYNOMIAL-LIMIT"),
    },
    {
        "name": "undergraduate_discrete_probability_algebra",
        "level": "undergraduate",
        "criteria": ("UG-MOD-INVERSE", "UG-PERMUTATIONS", "UG-EXPECTATION"),
    },
    {
        "name": "graduate_analysis_algebra",
        "level": "graduate",
        "criteria": ("GRAD-SERIES-RADIUS", "GRAD-CYCLIC-ORDER", "GRAD-EIGEN-SPECTRUM"),
    },
    {
        "name": "formal_research_preparation",
        "level": "research-preparation",
        "criteria": ("FORMAL-POLY-IDENTITY", "FORMAL-COUNTEREXAMPLE", "FORMAL-EUCLID-INVARIANT"),
    },
)

MASTER_PHASES: tuple[dict[str, Any], ...] = tuple(
    {**phase, "level": "KS1"} for phase in PHASES
) + MASTER_EXTENSION_PHASES

def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def phases_for_config(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    profile = str(config.get("curriculum_profile", "ks1_v1"))
    if profile == "ks1_v1":
        return PHASES
    if profile == "master_experiment_v1":
        return MASTER_PHASES
    raise ValueError(f"unsupported curriculum profile: {profile}")


def _phase_prerequisites(
    phase_index: int, phases: tuple[dict[str, Any], ...] = PHASES
) -> list[str]:
    return [
        criterion
        for prior in phases[:phase_index]
        for criterion in prior["criteria"]
    ]


def _base_candidate_irs(criterion: str) -> Iterator[dict[str, Any]]:
    if criterion == "1NPV-1":
        for value in range(1, 100):
            yield {"type": "math_problem_v1", "op": "predecessor", "value": value}
            yield {"type": "math_problem_v1", "op": "successor", "value": value}
        for start in range(97):
            for missing_index in (1, 2, 3):
                sequence: list[int | None] = [start + offset for offset in range(5)]
                sequence[missing_index] = None
                yield {
                    "type": "math_problem_v1",
                    "op": "missing_count_sequence",
                    "sequence": sequence,
                    "missing_index": missing_index,
                }
    elif criterion == "1NPV-2":
        for left in range(101):
            for right in range(101):
                yield {"type": "math_problem_v1", "op": "compare", "left": left, "right": right}
    elif criterion == "1AS-1":
        for left in range(11):
            for right in range(11 - left):
                yield {
                    "type": "math_problem_v1",
                    "op": "add",
                    "operands": [left, right],
                }
        for first in range(11):
            for second in range(11 - first):
                for third in range(11 - first - second):
                    yield {
                        "type": "math_problem_v1",
                        "op": "add",
                        "operands": [first, second, third],
                    }
    elif criterion == "1NF-1":
        # Stage 2 adds non-overlapping two-digit-plus-small-number facts;
        # facts owned by the across-ten and later within-100 stages remain
        # in those stages.
        for left in range(11):
            for right in range(11):
                if left + right <= 10:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
        for left in range(11, 21):
            for right in range(10):
                if left + right <= 20:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if left >= right and left <= 10:
                    yield {"type": "math_problem_v1", "op": "subtract", "left": left, "right": right}
    elif criterion == "1AS-2":
        for total in range(2, 21):
            for known in range(total + 1):
                yield {"type": "math_problem_v1", "op": "missing_addend", "known": known, "total": total}
    elif criterion == "2NPV-1":
        for value in range(10, 100):
            yield {"type": "math_problem_v1", "op": "place_value", "value": value}
    elif criterion == "2NPV-2":
        for value in range(11, 100):
            if value % 10:
                yield {"type": "math_problem_v1", "op": "neighbouring_tens", "value": value}
    elif criterion == "2AS-1":
        # Stage 3 teaches both directions of crossing ten, not only the
        # subset with a single-digit first operand.
        for left in range(21):
            for right in range(21):
                if 10 < left + right <= 20 and left <= 10 < left + right:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if left >= right and right < 10 and 0 <= left - right < 10 and left >= 10:
                    yield {"type": "math_problem_v1", "op": "subtract", "left": left, "right": right}
    elif criterion == "2AS-2":
        for left in range(21):
            for right in range(left + 1):
                yield {"type": "math_problem_v1", "op": "difference", "left": left, "right": right}
    elif criterion == "2AS-3":
        for value in range(10, 100):
            for delta in (-10, -1, 1, 10):
                result = value + delta
                if 0 <= result <= 100:
                    yield {"type": "math_problem_v1", "op": "add_signed", "value": value, "delta": delta}
    elif criterion == "2AS-4":
        for left in range(10, 100):
            for right in range(10, 100):
                if left + right <= 100:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if left >= right and (left, right) != (10, 10):
                    yield {"type": "math_problem_v1", "op": "subtract", "left": left, "right": right}
    elif criterion == "2MD-1":
        # Stage 5 must teach multiplication as a general operation.  Keep
        # the original table facts, but include the missing factors such as
        # 7, 11, 22 and 55 instead of hard-coding 2/5/10.
        for factor in range(2, 100):
            for groups in range(0, 101):
                yield {"type": "math_problem_v1", "op": "multiply", "left": factor, "right": groups}
    elif criterion == "2MD-2":
        for divisor in range(2, 101):
            for quotient in range(1, 101):
                if divisor <= 51 and quotient >= 2 and not (divisor in {2, 5, 10} and quotient <= 30):
                    continue
                yield {"type": "math_problem_v1", "op": "divide", "dividend": divisor * quotient, "divisor": divisor}
    elif criterion == "KS2-MULTI-DIGIT":
        for left in range(100, 300):
            for right in range(100, 200):
                yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
    elif criterion == "KS2-LONG-MULTIPLY":
        for left in range(100, 500):
            for right in range(10, 60):
                yield {"type": "math_problem_v1", "op": "multiply", "left": left, "right": right}
    elif criterion == "KS2-EXACT-DIVIDE":
        for divisor in range(2, 52):
            for quotient in range(2, 402):
                # Earlier KS1/KS2 multiplication-table division is replayed,
                # not relabelled as a new KS2 long-division object.
                if divisor in {2, 5, 10} and quotient <= 30:
                    continue
                yield {"type": "math_problem_v1", "op": "divide", "dividend": divisor * quotient, "divisor": divisor}
    elif criterion == "KS2-FRACTION-ADD":
        for denominator in range(2, 51):
            for left in range(1, denominator):
                for right in range(1, denominator):
                    yield {"type": "math_problem_v1", "op": "fraction_add", "left": [left, denominator], "right": [right, denominator]}
    elif criterion == "KS2-PERCENT":
        for percent in range(1, 101):
            for base in range(10, 1001, 10):
                yield {"type": "math_problem_v1", "op": "percent_of", "percent": percent, "base": base}
    elif criterion == "KS2-RECTANGLE":
        for width in range(2, 102):
            for height in range(2, 102):
                yield {"type": "math_problem_v1", "op": "rectangle_area", "width": width, "height": height}
    elif criterion == "KS3-LINEAR":
        for coefficient in range(2, 30):
            for solution in range(-100, 101):
                constant = 3 * solution - 5
                yield {"type": "math_problem_v1", "op": "linear_solve", "a": coefficient, "b": constant - coefficient * solution, "c": constant}
    elif criterion == "KS3-POWERS":
        for base in range(-50, 51):
            for exponent in range(2, 13):
                yield {"type": "math_problem_v1", "op": "power", "base": base, "exponent": exponent}
    elif criterion == "KS3-PYTHAGORAS":
        for scale in range(1, 1001):
            for triple in ((3, 4, 5), (5, 12, 13), (8, 15, 17)):
                yield {"type": "math_problem_v1", "op": "pythagoras", "a": triple[0] * scale, "b": triple[1] * scale}
    elif criterion == "GCSE-QUADRATIC":
        for root1 in range(-50, 51):
            for root2 in range(root1, 51):
                yield {"type": "math_problem_v1", "op": "quadratic_roots", "b": -(root1 + root2), "c": root1 * root2}
    elif criterion == "GCSE-SIMULTANEOUS":
        for x in range(-50, 51):
            for y in range(-50, 51):
                yield {"type": "math_problem_v1", "op": "simultaneous_solve", "equations": [[1, 1, x + y], [2, -1, 2 * x - y]]}
    elif criterion == "GCSE-SEQUENCE":
        for first in range(-50, 50):
            for difference in range(1, 21):
                for index in range(5, 15):
                    yield {"type": "math_problem_v1", "op": "arithmetic_sequence", "first": first, "difference": difference, "index": index}
    elif criterion == "AL-DIFFERENTIATE":
        for coefficient in range(1, 301):
            for power in range(2, 13):
                yield {"type": "math_problem_v1", "op": "differentiate_monomial", "coefficient": coefficient, "power": power}
    elif criterion == "AL-INTEGRATE":
        for coefficient in range(1, 301):
            for power in range(1, 12):
                yield {"type": "math_problem_v1", "op": "integrate_monomial", "coefficient": coefficient * (power + 1), "power": power}
    elif criterion in {"AL-BINOMIAL-COEFFICIENT", "AL-COMBINATION"}:
        for n in range(5, 301):
            for k in range(1, min(n, 20)):
                yield {"type": "math_problem_v1", "op": "binomial_coefficient" if criterion == "AL-BINOMIAL-COEFFICIENT" else "combination", "n": n, "k": k}
    elif criterion == "AL-BINOMIAL-PROB":
        for trials in range(3, 101):
            for successes in range(trials + 1):
                yield {"type": "math_problem_v1", "op": "binomial_probability_half", "trials": trials, "successes": successes}
    elif criterion == "AL-SUVAT":
        for initial in range(-50, 50):
            for acceleration in range(1, 11):
                for time in range(1, 21):
                    yield {"type": "math_problem_v1", "op": "constant_acceleration", "u": initial, "a": acceleration, "t": time}
    elif criterion == "UG-MATRIX-DET":
        for a in range(-10, 10):
            for b in range(-20, 20):
                for c in range(-12, 13):
                    yield {"type": "math_problem_v1", "op": "matrix_det_2x2", "matrix": [[a, b], [c, a + 1]]}
    elif criterion == "UG-MATRIX-SOLVE":
        for x in range(-50, 51):
            for y in range(-50, 51):
                yield {"type": "math_problem_v1", "op": "simultaneous_solve", "equations": [[2, 1, 2 * x + y], [1, -1, x - y]]}
    elif criterion == "UG-POLYNOMIAL-LIMIT":
        for point in range(-100, 101):
            for slope in range(1, 51):
                yield {"type": "math_problem_v1", "op": "cancelled_linear_limit", "point": point, "slope": slope}
    elif criterion == "UG-MOD-INVERSE":
        for modulus in range(3, 301):
            for value in range(2, modulus):
                if math.gcd(value, modulus) == 1:
                    yield {"type": "math_problem_v1", "op": "mod_inverse", "value": value, "modulus": modulus}
    elif criterion == "UG-PERMUTATIONS":
        for n in range(5, 301):
            for k in range(2, min(n, 20)):
                yield {"type": "math_problem_v1", "op": "permutation", "n": n, "k": k}
    elif criterion == "UG-EXPECTATION":
        for maximum in range(3, 5001):
            yield {"type": "math_problem_v1", "op": "uniform_expectation", "minimum": 1, "maximum": maximum}
        for maximum in range(3, 5001):
            yield {"type": "math_problem_v1", "op": "uniform_expectation", "minimum": -maximum, "maximum": maximum}
    elif criterion == "GRAD-SERIES-RADIUS":
        for base in range(2, 5002):
            yield {"type": "math_problem_v1", "op": "geometric_series_radius", "coefficient_base": base}
    elif criterion == "GRAD-CYCLIC-ORDER":
        for modulus in range(8, 209):
            for element in range(1, modulus):
                yield {"type": "math_problem_v1", "op": "cyclic_element_order", "element": element, "modulus": modulus}
    elif criterion == "GRAD-EIGEN-SPECTRUM":
        for left in range(-100, 101):
            for right in range(left, 101):
                yield {"type": "math_problem_v1", "op": "triangular_eigenvalues", "matrix": [[left, 1], [0, right]]}
    elif criterion == "FORMAL-POLY-IDENTITY":
        for a in range(-70, 70):
            for b in range(-70, 70):
                yield {"type": "math_problem_v1", "op": "square_identity", "a": a, "b": b}
    elif criterion == "FORMAL-COUNTEREXAMPLE":
        for value in range(2, 5002):
            yield {"type": "math_problem_v1", "op": "odd_square_counterexample", "value": value}
    elif criterion == "FORMAL-EUCLID-INVARIANT":
        for a in range(10, 500):
            for b in range(2, a):
                yield {"type": "math_problem_v1", "op": "euclid_gcd_invariant", "a": a, "b": b}
    else:
        raise ValueError(f"unknown criterion: {criterion}")


def _variant_specs(criterion: str) -> tuple[dict[str, str], ...]:
    """Typed, language-free views used only by the balanced v6 recipe.

    The representation and strategy tokens are part of the mathematical IR.
    They therefore teach invariance across useful mathematical views without
    putting natural-language wording into the tower input.
    """

    domains: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "1NPV-1": (
            ("count_sequence", "number_line"),
            ("forward", "backward"),
        ),
        "1AS-1": (
            (
                "equation",
                "part_whole",
                "number_line",
                "ten_frame",
                "base_ten",
                "fact_family",
                "balance",
                "bar_model",
                "recomposition",
            ),
            ("direct", "count_on", "make_ten", "inverse_check"),
        ),
        "1NF-1": (
            (
                "equation",
                "part_whole",
                "number_line",
                "ten_frame",
                "fact_family",
                "balance",
                "bar_model",
            ),
            ("direct_recall", "count_on", "make_ten", "inverse_check"),
        ),
        "1AS-2": (
            (
                "equation", "part_whole", "number_line", "fact_family", "balance",
                "bar_model", "ten_frame",
            ),
            ("complete_whole", "inverse_check"),
        ),
        "2NPV-1": (
            (
                "base_ten", "expanded_form", "place_value_chart", "partition", "recombine",
                "digit_cards", "abacus", "place_value_table", "bundles", "arrow_cards",
            ),
            ("read", "compose", "decompose", "verify"),
        ),
        "2NPV-2": (
            (
                "number_line", "place_value_chart", "interval", "rounding_frame", "base_ten", "ordering",
                "open_number_line", "tens_frame", "compare_bounds", "partition", "benchmark_tens",
            ),
            ("locate", "bound", "decompose", "verify"),
        ),
        "2AS-1": (
            (
                "equation", "part_whole", "number_line", "base_ten", "ten_frame",
                "bar_model", "fact_family", "balance", "number_bond", "open_number_line",
            ),
            ("bridge_ten", "make_ten"),
        ),
        "2AS-2": (
            (
                "equation", "part_whole", "number_line", "comparison", "fact_family",
                "ten_frame", "bar_model", "number_bond", "open_number_line",
            ),
            ("count_up", "subtract"),
        ),
        "2AS-3": (
            ("equation", "number_line", "place_value", "base_ten"),
            ("step", "inverse_check"),
        ),
        "2MD-1": (
            ("equation", "array", "equal_groups", "number_line"),
            ("skip_count", "repeated_addition"),
        ),
        "2MD-2": (
            ("equation", "array", "equal_groups", "fact_family"),
            ("sharing", "inverse_multiply"),
        ),
    }
    domain = domains.get(criterion)
    if domain is None:
        return ({},)
    representations, strategies = domain
    return tuple(
        {"representation": representation, "strategy": strategy}
        for representation in representations
        for strategy in strategies
    )


def _v6_base_candidate_irs(criterion: str) -> Iterator[dict[str, Any]]:
    """Broaden taught domains without leaking examples from a future phase."""

    if criterion == "1AS-1":
        for left in range(11):
            for right in range(left, 11 - left):
                yield {
                    "type": "math_problem_v1",
                    "op": "add",
                    "operands": [left, right],
                }
        for first in range(11):
            for second in range(first, 11 - first):
                for third in range(second, 11 - first - second):
                    yield {
                        "type": "math_problem_v1",
                        "op": "add",
                        "operands": [first, second, third],
                    }
        return
    if criterion == "1NF-1":
        for left in range(11):
            for right in range(left, 11):
                if left + right <= 10:
                    yield {
                        "type": "math_problem_v1",
                        "op": "add",
                        "left": left,
                        "right": right,
                    }
        for left in range(11):
            for right in range(left + 1):
                yield {
                    "type": "math_problem_v1",
                    "op": "subtract",
                    "left": left,
                    "right": right,
                }
        return
    if criterion == "2AS-1":
        # This phase teaches the new 11..20 result band. Facts wholly inside
        # 0..10 remain replay from the accepted Year-1 phase.
        for left in range(21):
            for right in range(left, 21):
                if 11 <= left + right <= 20:
                    yield {
                        "type": "math_problem_v1",
                        "op": "add",
                        "left": left,
                        "right": right,
                    }
        for left in range(11, 21):
            for right in range(left + 1):
                yield {
                    "type": "math_problem_v1",
                    "op": "subtract",
                    "left": left,
                    "right": right,
                }
        return
    if criterion == "2AS-4":
        for left in range(10, 100):
            for right in range(10, 100):
                if left <= right and 21 <= left + right <= 100:
                    yield {
                        "type": "math_problem_v1",
                        "op": "add",
                        "left": left,
                        "right": right,
                    }
                if left >= right and left > 20:
                    yield {
                        "type": "math_problem_v1",
                        "op": "subtract",
                        "left": left,
                        "right": right,
                    }
        return
    if criterion == "2MD-1":
        seen: set[tuple[int, int]] = set()
        for factor in (2, 5, 10):
            for groups in range(101):
                left, right = sorted((factor, groups))
                if (left, right) in seen:
                    continue
                seen.add((left, right))
                yield {
                    "type": "math_problem_v1",
                    "op": "multiply",
                    "left": left,
                    "right": right,
                }
        return
    if criterion == "2MD-2":
        for divisor in (2, 5, 10):
            for quotient in range(1, 101):
                yield {
                    "type": "math_problem_v1",
                    "op": "divide",
                    "dividend": divisor * quotient,
                    "divisor": divisor,
                }
        return
    if criterion == "KS2-MULTI-DIGIT":
        for left in range(100, 1000):
            for right in range(10, 500):
                if left <= right and left + right <= 1498:
                    yield {
                        "type": "math_problem_v1",
                        "op": "add",
                        "left": left,
                        "right": right,
                    }
                if left >= right:
                    yield {
                        "type": "math_problem_v1",
                        "op": "subtract",
                        "left": left,
                        "right": right,
                    }
        return
    if criterion == "KS2-LONG-MULTIPLY":
        # Includes one-digit and two-digit multipliers systematically, so
        # factors such as 7, 11, 22 and 55 are explicitly taught.
        for left in range(10, 1000):
            for right in range(2, min(100, left) + 1):
                if right in {2, 5, 10} and left <= 100:
                    continue
                yield {
                    "type": "math_problem_v1",
                    "op": "multiply",
                    "left": left,
                    "right": right,
                }
        return
    if criterion == "KS2-FRACTION-ADD":
        for denominator in range(2, 51):
            for left in range(1, denominator):
                for right in range(left, denominator):
                    yield {
                        "type": "math_problem_v1",
                        "op": "fraction_add",
                        "left": [left, denominator],
                        "right": [right, denominator],
                    }
        return
    if criterion == "KS2-RECTANGLE":
        for width in range(2, 102):
            for height in range(width, 102):
                yield {
                    "type": "math_problem_v1",
                    "op": "rectangle_area",
                    "width": width,
                    "height": height,
                }
        return
    if criterion == "KS2-EXACT-DIVIDE":
        for divisor in range(2, 101):
            for quotient in range(2, 501):
                if divisor in {2, 5, 10} and quotient <= 100:
                    continue
                yield {
                    "type": "math_problem_v1",
                    "op": "divide",
                    "dividend": divisor * quotient,
                    "divisor": divisor,
                }
        return
    if criterion == "KS3-POWERS":
        for base in range(-200, 201):
            for exponent in range(2, 13):
                yield {
                    "type": "math_problem_v1",
                    "op": "power",
                    "base": base,
                    "exponent": exponent,
                }
        return
    yield from _base_candidate_irs(criterion)


def _candidate_irs(
    criterion: str, dataset_recipe: str | None = None
) -> Iterator[dict[str, Any]]:
    if dataset_recipe == V6_DATASET_RECIPE:
        for base in _v6_base_candidate_irs(criterion):
            for variant in _variant_specs(criterion):
                yield {**base, **variant}
        return
    if dataset_recipe == V9_DATASET_RECIPE and criterion in V9_VARIANT_CRITERIA:
        for base in _base_candidate_irs(criterion):
            for variant in _variant_specs(criterion):
                yield {**base, **variant}
        return
    if dataset_recipe != V6_DATASET_RECIPE:
        yield from _base_candidate_irs(criterion)
        return


def _uses_pedagogical_variants(criterion: str, dataset_recipe: str | None) -> bool:
    return dataset_recipe == V6_DATASET_RECIPE or (
        dataset_recipe == V9_DATASET_RECIPE and criterion in V9_VARIANT_CRITERIA
    )


def _semantic_math_ir(math_ir: dict[str, Any]) -> dict[str, Any]:
    semantic = {
        key: value
        for key, value in math_ir.items()
        if key not in PEDAGOGICAL_VARIANT_FIELDS
    }
    operation = str(semantic.get("op", ""))
    if operation in {"add", "multiply"} and {
        "left",
        "right",
    } <= semantic.keys():
        semantic["left"], semantic["right"] = sorted(
            (semantic["left"], semantic["right"])
        )
    if operation == "add" and isinstance(semantic.get("operands"), list):
        semantic["operands"] = sorted(semantic["operands"])
    if operation == "fraction_add":
        semantic["left"], semantic["right"] = sorted(
            (semantic["left"], semantic["right"])
        )
    if operation == "rectangle_area":
        semantic["width"], semantic["height"] = sorted(
            (semantic["width"], semantic["height"])
        )
    return semantic


def _semantic_object_id(math_ir: dict[str, Any]) -> str:
    return _sha(_semantic_math_ir(math_ir))


def _numeric_values(value: Any) -> Iterator[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _numeric_values(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _numeric_values(item)


def _value_band(math_ir: dict[str, Any]) -> str:
    values = [abs(value) for value in _numeric_values(_semantic_math_ir(math_ir))]
    maximum = max(values, default=0)
    if maximum <= 10:
        return "0_10"
    if maximum <= 20:
        return "11_20"
    if maximum <= 100:
        return "21_100"
    if maximum <= 1000:
        return "101_1000"
    return "over_1000"


def solve_math_ir(
    math_ir: dict[str, Any],
    *,
    procedure_schema: str = EXPANDED_PROCEDURE_SCHEMA,
) -> tuple[str, list[dict[str, Any]]]:
    if procedure_schema not in PROCEDURE_SCHEMAS:
        raise ValueError(f"unsupported procedure schema: {procedure_schema}")
    op = math_ir["op"]
    if op == "successor":
        result = int(math_ir["value"]) + 1
        derivation = [
            {"direction": "forward", "from": int(math_ir["value"]), "op": "count_one", "to": result},
            {"op": op, "result": str(result)},
        ]
    elif op == "predecessor":
        result = int(math_ir["value"]) - 1
        derivation = [
            {"direction": "backward", "from": int(math_ir["value"]), "op": "count_one", "to": result},
            {"op": op, "result": str(result)},
        ]
    elif op == "missing_count_sequence":
        sequence = list(math_ir["sequence"])
        missing_index = int(math_ir["missing_index"])
        result = int(sequence[missing_index - 1]) + 1
        derivation = [
            {"index": missing_index, "op": "continue_count", "previous": sequence[missing_index - 1], "result": result},
            {"op": op, "result": str(result)},
        ]
    elif op == "compare":
        left, right = int(math_ir["left"]), int(math_ir["right"])
        result = "<" if left < right else ">" if left > right else "="
        derivation = [
            {"ones": left % 10, "op": "decompose", "tens": left // 10, "value": left},
            {"ones": right % 10, "op": "decompose", "tens": right // 10, "value": right},
            {"left": left, "op": op, "result": result, "right": right},
        ]
    elif op == "add":
        if "operands" in math_ir:
            operands = [int(value) for value in math_ir["operands"]]
            if len(operands) not in (2, 3) or any(value < 0 for value in operands):
                raise ValueError("canonical addition requires two or three non-negative operands")
            running = operands[0]
            steps: list[dict[str, Any]] = []
            for operand in operands[1:]:
                start = running
                running += operand
                if procedure_schema == COMPACT_PROCEDURE_SCHEMA:
                    steps.append({"add": operand, "end": running})
                else:
                    sequence = list(range(start + 1, running + 1))
                    steps.append(
                        {"add": operand, "result": running, "sequence": sequence}
                    )
            result = running
            derivation = [{"op": "count_on", "start": operands[0], "steps": steps}]
            if procedure_schema == EXPANDED_PROCEDURE_SCHEMA:
                derivation.append(
                    {"op": "add", "operands": operands, "result": str(result)}
                )
        else:
            result = int(math_ir["left"]) + int(math_ir["right"])
    elif op == "subtract":
        result = int(math_ir["left"]) - int(math_ir["right"])
    elif op == "difference":
        result = abs(int(math_ir["left"]) - int(math_ir["right"]))
    elif op == "missing_addend":
        result = int(math_ir["total"]) - int(math_ir["known"])
    elif op == "place_value":
        value = int(math_ir["value"])
        result = f"{value // 10},{value % 10}"
    elif op == "neighbouring_tens":
        value = int(math_ir["value"])
        result = f"{value // 10 * 10},{value // 10 * 10 + 10}"
    elif op == "add_signed":
        result = int(math_ir["value"]) + int(math_ir["delta"])
    elif op == "multiply":
        result = int(math_ir["left"]) * int(math_ir["right"])
    elif op == "divide":
        dividend, divisor = int(math_ir["dividend"]), int(math_ir["divisor"])
        if divisor == 0 or dividend % divisor:
            raise ValueError("division example must have an exact non-zero divisor")
        result = dividend // divisor
    elif op == "fraction_add":
        left, right = math_ir["left"], math_ir["right"]
        result = Fraction(int(left[0]), int(left[1])) + Fraction(int(right[0]), int(right[1]))
    elif op == "percent_of":
        result = Fraction(int(math_ir["percent"]) * int(math_ir["base"]), 100)
    elif op == "rectangle_area":
        result = int(math_ir["width"]) * int(math_ir["height"])
    elif op == "linear_solve":
        a, b, c = int(math_ir["a"]), int(math_ir["b"]), int(math_ir["c"])
        if not a:
            raise ValueError("linear coefficient cannot be zero")
        result = Fraction(c - b, a)
    elif op == "power":
        result = int(math_ir["base"]) ** int(math_ir["exponent"])
    elif op == "pythagoras":
        square = int(math_ir["a"]) ** 2 + int(math_ir["b"]) ** 2
        result = math.isqrt(square)
        if result * result != square:
            raise ValueError("Pythagoras pilot requires an integral hypotenuse")
    elif op == "quadratic_roots":
        b, c = int(math_ir["b"]), int(math_ir["c"])
        discriminant = b * b - 4 * c
        root = math.isqrt(discriminant)
        if root * root != discriminant:
            raise ValueError("quadratic pilot requires integral roots")
        result = f"{min((-b - root) // 2, (-b + root) // 2)},{max((-b - root) // 2, (-b + root) // 2)}"
    elif op == "simultaneous_solve":
        (a, b, e), (c, d, f) = math_ir["equations"]
        determinant = int(a) * int(d) - int(b) * int(c)
        if not determinant:
            raise ValueError("simultaneous system must be nonsingular")
        x = Fraction(int(e) * int(d) - int(b) * int(f), determinant)
        y = Fraction(int(a) * int(f) - int(e) * int(c), determinant)
        result = f"{_fraction_answer(x)},{_fraction_answer(y)}"
    elif op == "arithmetic_sequence":
        result = int(math_ir["first"]) + (int(math_ir["index"]) - 1) * int(math_ir["difference"])
    elif op == "differentiate_monomial":
        coefficient, power = int(math_ir["coefficient"]), int(math_ir["power"])
        result = f"{coefficient * power}*x^{power - 1}"
    elif op == "integrate_monomial":
        coefficient, power = int(math_ir["coefficient"]), int(math_ir["power"])
        result = f"{_fraction_answer(Fraction(coefficient, power + 1))}*x^{power + 1}+C"
    elif op in {"combination", "binomial_coefficient"}:
        result = math.comb(int(math_ir["n"]), int(math_ir["k"]))
    elif op == "binomial_probability_half":
        n, k = int(math_ir["trials"]), int(math_ir["successes"])
        result = Fraction(math.comb(n, k), 2**n)
    elif op == "constant_acceleration":
        result = int(math_ir["u"]) * int(math_ir["t"]) + Fraction(
            int(math_ir["a"]) * int(math_ir["t"]) ** 2, 2
        )
    elif op == "matrix_det_2x2":
        matrix = math_ir["matrix"]
        result = int(matrix[0][0]) * int(matrix[1][1]) - int(matrix[0][1]) * int(matrix[1][0])
    elif op == "cancelled_linear_limit":
        result = int(math_ir["slope"])
    elif op == "mod_inverse":
        result = pow(int(math_ir["value"]), -1, int(math_ir["modulus"]))
    elif op == "permutation":
        n, k = int(math_ir["n"]), int(math_ir["k"])
        result = math.factorial(n) // math.factorial(n - k)
    elif op == "uniform_expectation":
        result = Fraction(int(math_ir["minimum"]) + int(math_ir["maximum"]), 2)
    elif op == "geometric_series_radius":
        result = Fraction(1, int(math_ir["coefficient_base"]))
    elif op == "cyclic_element_order":
        element, modulus = int(math_ir["element"]), int(math_ir["modulus"])
        result = modulus // math.gcd(element, modulus)
    elif op == "triangular_eigenvalues":
        matrix = math_ir["matrix"]
        values = sorted((int(matrix[0][0]), int(matrix[1][1])))
        result = f"{values[0]},{values[1]}"
    elif op == "square_identity":
        a, b = int(math_ir["a"]), int(math_ir["b"])
        result = str((a + b) ** 2 == a * a + 2 * a * b + b * b).lower()
    elif op == "odd_square_counterexample":
        value = int(math_ir["value"])
        result = f"{value},{value * value}"
    elif op == "euclid_gcd_invariant":
        a, b = int(math_ir["a"]), int(math_ir["b"])
        result = str(math.gcd(a, b) == math.gcd(b, a % b)).lower()
    else:
        raise ValueError(f"unsupported operation: {op}")
    answer = _fraction_answer(result) if isinstance(result, Fraction) else str(result)
    if "derivation" not in locals():
        derivation = [{"op": op, "result": answer}]
    return answer, derivation


def _fraction_answer(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _base_language_prompts(
    math_ir: dict[str, Any], *, criterion: str | None = None
) -> tuple[str, ...]:
    op = math_ir["op"]
    if op == "successor":
        value = math_ir["value"]
        return (f"What number comes after {value}?", f"Give the successor of {value}.")
    if op == "predecessor":
        value = math_ir["value"]
        return (f"What number comes before {value}?", f"Give the predecessor of {value}.")
    if op == "missing_count_sequence":
        rendered = ", ".join("?" if value is None else str(value) for value in math_ir["sequence"])
        return (f"Complete the counting sequence: {rendered}.", f"Which number is missing from {rendered}?")
    if op == "compare":
        left, right = math_ir["left"], math_ir["right"]
        return (f"Compare {left} and {right}.", f"Which symbol, <, >, or =, belongs between {left} and {right}?")
    if op == "add" and "operands" in math_ir:
        operands = [int(value) for value in math_ir["operands"]]
        rendered = ", ".join(str(value) for value in operands)
        return (
            f"Compose {rendered} into a whole.",
            f"What whole is made from parts {rendered}?",
        )
    if op in {"add", "subtract", "difference", "multiply"}:
        left, right = math_ir["left"], math_ir["right"]
        templates = {
            "add": (f"Calculate {left} + {right}.", f"What is the sum of {left} and {right}?"),
            "subtract": (f"Calculate {left} - {right}.", f"Subtract {right} from {left}."),
            "difference": (f"Find the difference between {left} and {right}.", f"How far apart are {left} and {right}?"),
            "multiply": (f"Calculate {left} x {right}.", f"What is the product of {left} and {right}?"),
        }
        if criterion == "2AS-1":
            return tuple(f"{prompt} Use a make-ten step." for prompt in templates[op])
        if criterion in {"2MD-1", "2MD-2"}:
            return tuple(f"{prompt} Use a fact-family check." for prompt in templates[op])
        return templates[op]
    if op == "missing_addend":
        known, total = math_ir["known"], math_ir["total"]
        return (f"Complete {known} + ? = {total}.", f"What must be added to {known} to make {total}?")
    if op == "place_value":
        value = math_ir["value"]
        return (f"How many tens and ones are in {value}?", f"Partition {value} into tens and ones.")
    if op == "neighbouring_tens":
        value = math_ir["value"]
        return (f"Give the multiples of ten immediately below and above {value}.", f"Which two tens does {value} lie between?")
    if op == "add_signed":
        value, delta = math_ir["value"], math_ir["delta"]
        return (f"Calculate {value} + ({delta}).", f"Change {value} by {delta}.")
    if op == "divide":
        dividend, divisor = math_ir["dividend"], math_ir["divisor"]
        prompts = (f"Calculate {dividend} divided by {divisor}.", f"How many groups of {divisor} are in {dividend}?")
        if criterion == "2MD-2":
            return tuple(f"{prompt} Use a fact-family check." for prompt in prompts)
        return prompts
    if op == "fraction_add":
        left, right = math_ir["left"], math_ir["right"]
        return (f"Add {left[0]}/{left[1]} and {right[0]}/{right[1]}.", f"Find {left[0]}/{left[1]} + {right[0]}/{right[1]} in simplest form.")
    if op == "percent_of":
        return (f"Find {math_ir['percent']}% of {math_ir['base']}.", f"Calculate {math_ir['percent']} percent of {math_ir['base']}.")
    if op == "rectangle_area":
        return (f"Find the area of a rectangle of width {math_ir['width']} and height {math_ir['height']}.", f"A rectangle is {math_ir['width']} by {math_ir['height']}. What is its area?")
    if op == "linear_solve":
        return (f"Solve {math_ir['a']}x + {math_ir['b']} = {math_ir['c']}.", f"Find x when {math_ir['a']}x + {math_ir['b']} equals {math_ir['c']}.")
    if op == "power":
        return (f"Calculate {math_ir['base']} to the power {math_ir['exponent']}.", f"Evaluate {math_ir['base']}^{math_ir['exponent']}.")
    if op == "pythagoras":
        return (f"A right triangle has legs {math_ir['a']} and {math_ir['b']}. Find its hypotenuse.", f"Use Pythagoras for perpendicular sides {math_ir['a']} and {math_ir['b']}.")
    if op == "quadratic_roots":
        return (f"Find both roots of x^2 + {math_ir['b']}x + {math_ir['c']} = 0.", f"Solve the quadratic with coefficients 1, {math_ir['b']}, {math_ir['c']}.")
    if op == "simultaneous_solve":
        first, second = math_ir["equations"]
        return (f"Solve {first[0]}x + {first[1]}y = {first[2]} and {second[0]}x + {second[1]}y = {second[2]}.", f"Find x,y satisfying the two linear equations {first} and {second}.")
    if op == "arithmetic_sequence":
        return (f"An arithmetic sequence starts at {math_ir['first']} with difference {math_ir['difference']}. Find term {math_ir['index']}.", f"Find the {math_ir['index']}th term when a1={math_ir['first']} and d={math_ir['difference']}.")
    if op == "differentiate_monomial":
        return (f"Differentiate {math_ir['coefficient']}x^{math_ir['power']} with respect to x.", f"Find d/dx of {math_ir['coefficient']}x^{math_ir['power']}.")
    if op == "integrate_monomial":
        return (f"Find the indefinite integral of {math_ir['coefficient']}x^{math_ir['power']} dx.", f"Integrate {math_ir['coefficient']}x^{math_ir['power']} with respect to x.")
    if op == "binomial_coefficient":
        return (f"Find the coefficient of x^{math_ir['k']} in (1+x)^{math_ir['n']}.", f"In the binomial expansion of (1+x)^{math_ir['n']}, give the x^{math_ir['k']} coefficient.")
    if op == "combination":
        return (f"Calculate {math_ir['n']} choose {math_ir['k']}.", f"How many ways can {math_ir['k']} items be selected from {math_ir['n']}?")
    if op == "binomial_probability_half":
        return (f"For X~Bin({math_ir['trials']}, 1/2), find P(X={math_ir['successes']}).", f"Find the probability of exactly {math_ir['successes']} successes in {math_ir['trials']} fair trials.")
    if op == "constant_acceleration":
        return (f"Find displacement when u={math_ir['u']}, a={math_ir['a']}, and t={math_ir['t']} under constant acceleration.", f"Use s=ut+at^2/2 for u={math_ir['u']}, a={math_ir['a']}, t={math_ir['t']}.")
    if op == "matrix_det_2x2":
        return (f"Find the determinant of the matrix {math_ir['matrix']}.", f"Calculate det({math_ir['matrix']}).")
    if op == "cancelled_linear_limit":
        point, slope = math_ir["point"], math_ir["slope"]
        return (f"Find lim x->{point} of ({slope}x-{slope * point})/(x-{point}).", f"Evaluate the removable limit at {point} with linear factor {slope}.")
    if op == "mod_inverse":
        return (f"Find the inverse of {math_ir['value']} modulo {math_ir['modulus']}.", f"Solve {math_ir['value']}x = 1 mod {math_ir['modulus']}.")
    if op == "permutation":
        return (f"Calculate P({math_ir['n']},{math_ir['k']}).", f"Count ordered selections of {math_ir['k']} objects from {math_ir['n']}.")
    if op == "uniform_expectation":
        return (f"Find the expectation of the discrete uniform distribution on integers {math_ir['minimum']} through {math_ir['maximum']}.", f"Calculate the mean of a uniform integer variable from {math_ir['minimum']} to {math_ir['maximum']}.")
    if op == "geometric_series_radius":
        return (f"Find the radius of convergence of sum ({math_ir['coefficient_base']}x)^n.", f"For the geometric power series with ratio {math_ir['coefficient_base']}x, give the convergence radius.")
    if op == "cyclic_element_order":
        return (f"Find the additive order of {math_ir['element']} in Z/{math_ir['modulus']}Z.", f"In the cyclic group modulo {math_ir['modulus']}, determine the order of element {math_ir['element']}.")
    if op == "triangular_eigenvalues":
        return (f"Find the eigenvalues of the triangular matrix {math_ir['matrix']}.", f"Give the spectrum of {math_ir['matrix']}.")
    if op == "square_identity":
        return (f"Verify (a+b)^2=a^2+2ab+b^2 for a={math_ir['a']}, b={math_ir['b']}.", f"Check the square identity at ({math_ir['a']},{math_ir['b']}).")
    if op == "odd_square_counterexample":
        return (f"Use n={math_ir['value']} as a witness when testing the claim that every square is odd.", f"Return n and n^2 for n={math_ir['value']} to test the universal odd-square claim.")
    if op == "euclid_gcd_invariant":
        return (f"Verify gcd({math_ir['a']},{math_ir['b']}) = gcd({math_ir['b']},{math_ir['a']} mod {math_ir['b']}).", f"Check one Euclidean-algorithm invariant step for {math_ir['a']} and {math_ir['b']}.")
    raise ValueError(f"no language templates for operation: {op}")


def _language_prompts(
    math_ir: dict[str, Any], *, criterion: str | None = None
) -> tuple[str, ...]:
    prompts = _base_language_prompts(math_ir, criterion=criterion)
    representation = str(math_ir.get("representation", "")).strip()
    strategy = str(math_ir.get("strategy", "")).strip()
    if not representation and not strategy:
        return prompts
    instruction = (
        f" Encode the mathematical request with representation={representation}"
        f" and strategy={strategy}."
    )
    return tuple(prompt + instruction for prompt in prompts)


def _canonical_json_with_spans(
    value: Any,
    roles: dict[tuple[Any, ...], str],
    path: tuple[Any, ...] = (),
) -> tuple[str, list[dict[str, Any]]]:
    """Serialize canonical JSON while retaining exact semantic-role offsets."""

    if isinstance(value, dict):
        text = "{"
        spans: list[dict[str, Any]] = []
        for index, key in enumerate(sorted(value)):
            if index:
                text += ","
            text += canonical_json(str(key)) + ":"
            child, child_spans = _canonical_json_with_spans(
                value[key], roles, path + (key,)
            )
            offset = len(text)
            text += child
            spans.extend(
                {**span, "start": span["start"] + offset, "end": span["end"] + offset}
                for span in child_spans
            )
        return text + "}", spans
    if isinstance(value, list):
        text = "["
        spans = []
        for index, child_value in enumerate(value):
            if index:
                text += ","
            child, child_spans = _canonical_json_with_spans(
                child_value, roles, path + (index,)
            )
            offset = len(text)
            text += child
            spans.extend(
                {**span, "start": span["start"] + offset, "end": span["end"] + offset}
                for span in child_spans
            )
        return text + "]", spans
    text = canonical_json(value)
    role = roles.get(path)
    spans = [] if role is None else [{"kind": role, "start": 0, "end": len(text)}]
    return text, spans


def _trace_roles(
    math_ir: dict[str, Any], derivation: list[dict[str, Any]]
) -> dict[tuple[Any, ...], str]:
    """Identify values produced by computation instead of copied from the IR."""

    op = str(math_ir["op"])
    computed: set[tuple[Any, ...]] = set()
    if op in {"successor", "predecessor"}:
        computed.update({(0, "to"), (1, "result")})
    elif op == "missing_count_sequence":
        computed.update({(0, "result"), (1, "result")})
    elif op == "compare":
        computed.update(
            {
                (0, "ones"),
                (0, "tens"),
                (1, "ones"),
                (1, "tens"),
                (2, "result"),
            }
        )
    elif op == "add" and "operands" in math_ir:
        for index, step in enumerate(derivation[0]["steps"]):
            if "end" in step:
                computed.add((0, "steps", index, "end"))
            if "result" in step:
                computed.add((0, "steps", index, "result"))
            for sequence_index, _ in enumerate(step.get("sequence", [])):
                computed.add((0, "steps", index, "sequence", sequence_index))
        if len(derivation) > 1:
            computed.add((1, "result"))
    else:
        computed.add((0, "result"))
    return {path: "compute" for path in computed}


def _trace(
    answer: str,
    derivation: list[dict[str, Any]],
    math_ir: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    work, work_spans = _canonical_json_with_spans(
        derivation, _trace_roles(math_ir, derivation)
    )
    trace = f"<work>{work}</work><answer>{answer}</answer>"
    answer_start = trace.index(answer, trace.index("<answer>"))
    work_offset = len("<work>")
    spans = [
        {
            **span,
            "start": int(span["start"]) + work_offset,
            "end": int(span["end"]) + work_offset,
        }
        for span in work_spans
    ]
    spans.append(
        {"kind": "copy", "start": answer_start, "end": answer_start + len(answer)}
    )
    spans.sort(key=lambda span: int(span["start"]))
    return trace, spans


def trace_semantically_matches(generation: str, record: dict[str, Any]) -> bool:
    """Parse a generated trace and verify it against the executable math IR."""

    if not isinstance(generation, str):
        return False
    text = generation.strip()
    work_open, work_close = "<work>", "</work>"
    answer_open, answer_close = "<answer>", "</answer>"
    if not text.startswith(work_open) or not text.endswith(answer_close):
        return False
    work_end = text.find(work_close, len(work_open))
    if work_end < 0 or text[work_end + len(work_close) :].count(answer_open) != 1:
        return False
    answer_start = work_end + len(work_close)
    if not text.startswith(answer_open, answer_start):
        return False
    answer_value_start = answer_start + len(answer_open)
    answer_end = text.find(answer_close, answer_value_start)
    if answer_end < 0 or answer_end + len(answer_close) != len(text):
        return False
    try:
        work = json.loads(text[len(work_open) : work_end])
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    procedure_schema = str(
        record.get("procedure_schema", EXPANDED_PROCEDURE_SCHEMA)
    )
    expected_answer, expected_work = solve_math_ir(
        record["math_ir"], procedure_schema=procedure_schema
    )
    return (
        text[answer_value_start:answer_end] == expected_answer
        and work == expected_work
    )


def _records_for_object(
    *,
    split: str,
    criterion: str,
    math_ir: dict[str, Any],
    phase_index: int,
    phases: tuple[dict[str, Any], ...] = PHASES,
    language_variants_per_object: int = 2,
    procedure_schema: str = EXPANDED_PROCEDURE_SCHEMA,
    dataset_recipe: str | None = None,
) -> Iterator[dict[str, Any]]:
    answer, derivation = solve_math_ir(math_ir, procedure_schema=procedure_schema)
    math_ir_text = canonical_json(math_ir)
    trace, spans = _trace(answer, derivation, math_ir)
    object_id = _sha(math_ir)
    semantic_object_id = (
        _semantic_object_id(math_ir)
        if _uses_pedagogical_variants(criterion, dataset_recipe)
        else object_id
    )
    phase = phases[phase_index]
    prompts = _language_prompts(math_ir, criterion=criterion)
    if not 1 <= language_variants_per_object <= len(prompts):
        raise ValueError("language_variants_per_object exceeds available prompts")
    for variant, prompt in enumerate(prompts[:language_variants_per_object]):
        dispatcher_target = {
            "route": "math",
            "criterion_id": criterion,
            "operation": _operation_key(math_ir),
            "math_ir": math_ir,
        }
        extras = {
            "curriculum_schema": SCHEMA,
            "natural_language_prompt": prompt,
            "dispatcher_target": dispatcher_target,
            "math_ir": math_ir,
            "derivation": derivation,
            "answer": answer,
            "verifier_spec": {"kind": "exact_math_ir_v1", "math_ir": math_ir},
            "criterion_id": criterion,
            "operation": _operation_key(math_ir),
            "curriculum_phase": phase["name"],
            "curriculum_phase_index": phase_index,
            "prerequisite_ids": _phase_prerequisites(phase_index, phases),
            "educational_level": phase.get("level", "KS1"),
            "numeric_domain": "criterion_defined_taught_domain_v1",
            "representation": "canonical_json_math_ir_v1",
            "evaluation_mode": "held_out_objects_within_taught_domain",
            "math_object_id": object_id,
            "math_semantic_id": semantic_object_id,
            "language_variant": variant,
            "procedure_schema": procedure_schema,
            "computation_spans": spans,
            "pedagogical_representation": str(
                math_ir.get("representation", "canonical")
            ),
            "pedagogical_strategy": str(math_ir.get("strategy", "canonical")),
            "value_band": _value_band(math_ir),
        }
        yield make_v2_record(
            split=split,
            source="canonical_primary_math",
            family=criterion,
            difficulty=min(phase_index + 1, 3),
            problem=math_ir_text,
            raw_problem=prompt,
            answer=answer,
            target_trace=trace,
            native_program=math_ir_text,
            execution_trace=f"<work>{canonical_json(derivation)}</work>",
            gpt_problem=prompt,
            math_problem=math_ir_text,
            metadata={"criterion_id": criterion, "phase": phase["name"]},
            extra_fields=extras,
        )


@lru_cache(maxsize=None)
def _candidate_capacity(criterion: str, dataset_recipe: str | None = None) -> int:
    return sum(1 for _ in _candidate_irs(criterion, dataset_recipe))


@lru_cache(maxsize=None)
def _semantic_candidate_capacity(
    criterion: str, dataset_recipe: str | None = None
) -> int:
    if not _uses_pedagogical_variants(criterion, dataset_recipe):
        return _candidate_capacity(criterion, dataset_recipe)
    return sum(1 for _ in _v6_base_candidate_irs(criterion))


def _operation_key(math_ir: dict[str, Any]) -> str:
    operation = str(math_ir["op"])
    operands = math_ir.get("operands")
    if operation == "add" and isinstance(operands, list):
        return f"add_{len(operands)}"
    return operation


@lru_cache(maxsize=None)
def _criterion_operations(
    criterion: str, dataset_recipe: str | None = None
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _operation_key(item)
                for item in _candidate_irs(criterion, dataset_recipe)
            }
        )
    )


def _criterion_split_count(config: dict[str, Any], criterion: str, split: str) -> int:
    overrides = config.get("split_overrides", {}).get(criterion, {})
    return int(overrides.get(split, config["objects_per_criterion"][split]))


def _criterion_split_counts(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    phases = phases_for_config(config)
    criteria = [criterion for phase in phases for criterion in phase["criteria"]]
    dataset_recipe = str(config.get("dataset_recipe", "")) or None
    total_records = config.get("total_train_records")
    if total_records is None:
        return {
            criterion: {
                split: _criterion_split_count(config, criterion, split)
                for split in SPLITS
            }
            for criterion in criteria
        }
    variants = int(config.get("language_variants_per_object", 1))
    if int(total_records) % variants:
        raise ValueError("total_train_records must be divisible by language variants")
    target = int(total_records) // variants
    validation = {
        criterion: _criterion_split_count(config, criterion, "validation")
        for criterion in criteria
    }
    test = {
        criterion: _criterion_split_count(config, criterion, "test")
        for criterion in criteria
    }
    available = {
        criterion: _candidate_capacity(criterion, dataset_recipe)
        - (validation[criterion] + test[criterion])
        * (
            len(_variant_specs(criterion))
            if _uses_pedagogical_variants(criterion, dataset_recipe)
            else 1
        )
        for criterion in criteria
    }
    semantic_capacity = {
        criterion: _semantic_candidate_capacity(criterion, dataset_recipe)
        for criterion in criteria
    }
    insufficient_holdout = [
        criterion
        for criterion in criteria
        if validation[criterion] + test[criterion] >= semantic_capacity[criterion]
    ]
    if insufficient_holdout:
        raise ValueError(
            "criteria lack semantically disjoint holdout capacity: "
            + ", ".join(insufficient_holdout)
        )
    if any(value < 1 for value in available.values()):
        short = [criterion for criterion, value in available.items() if value < 1]
        raise ValueError(f"criteria lack split capacity: {short}")
    if sum(available.values()) < target:
        raise ValueError(
            f"candidate capacity {sum(available.values())} is below requested {target}"
        )
    explicit_targets = config.get("criterion_train_targets")
    if explicit_targets is not None:
        normalized_targets = {
            str(criterion): int(count)
            for criterion, count in dict(explicit_targets).items()
        }
        missing = sorted(set(criteria) - set(normalized_targets))
        unexpected = sorted(set(normalized_targets) - set(criteria))
        if missing or unexpected:
            raise ValueError(
                "criterion_train_targets must exactly cover the curriculum; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if any(count < 1 for count in normalized_targets.values()):
            raise ValueError("criterion_train_targets must all be positive")
        if sum(normalized_targets.values()) != target:
            raise ValueError(
                "criterion_train_targets must sum to total_train_records"
            )
        shortfalls = {
            criterion: {
                "requested": normalized_targets[criterion],
                "available": available[criterion],
            }
            for criterion in criteria
            if normalized_targets[criterion] > available[criterion]
        }
        if shortfalls:
            raise ValueError(f"criterion train target capacity shortfall: {shortfalls}")
        train = normalized_targets
    else:
        train = {criterion: 0 for criterion in criteria}
        remaining = target
        while remaining:
            eligible = [
                criterion
                for criterion in criteria
                if train[criterion] < available[criterion]
            ]
            if not eligible:
                raise RuntimeError("training allocation exhausted unexpectedly")
            share = max(1, remaining // len(eligible))
            progressed = 0
            for criterion in eligible:
                count = min(
                    share, available[criterion] - train[criterion], remaining
                )
                train[criterion] += count
                remaining -= count
                progressed += count
                if not remaining:
                    break
            if not progressed:
                raise RuntimeError("training allocation made no progress")
    return {
        criterion: {
            "train": train[criterion],
            "validation": validation[criterion],
            "test": test[criterion],
        }
        for criterion in criteria
    }


def _balanced_result_rows(
    *,
    criterion: str,
    operation: str,
    count: int,
    split: str,
    seed: int,
    held_out_semantic_ids: set[str],
    dataset_recipe: str | None,
) -> list[dict[str, Any]]:
    identity = (
        _semantic_object_id
        if _uses_pedagogical_variants(criterion, dataset_recipe)
        else _sha
    )
    bucket_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for item in _candidate_irs(criterion, dataset_recipe):
        semantic_id = identity(item)
        if (
            _operation_key(item) != operation
            or semantic_id in held_out_semantic_ids
        ):
            continue
        answer, _ = solve_math_ir(item)
        answer_bucket = bucket_maps.setdefault(answer, {})
        incumbent = answer_bucket.get(semantic_id)
        if incumbent is None or _sha(
            [seed, criterion, split, operation, item]
        ) < _sha([seed, criterion, split, operation, incumbent]):
            answer_bucket[semantic_id] = item
    buckets = {
        answer: list(rows.values()) for answer, rows in bucket_maps.items()
    }
    selected: list[dict[str, Any]] = []
    selected_semantic_ids: set[str] = set()
    round_index = 0
    while len(selected) < count:
        progressed = False
        ordered_answers = sorted(
            buckets,
            key=lambda answer: _sha(
                [seed, criterion, split, operation, round_index, answer]
            ),
        )
        for answer in ordered_answers:
            remaining = [
                item
                for item in buckets[answer]
                if identity(item) not in selected_semantic_ids
            ]
            # Always retain at least one example of every result in training.
            if len(remaining) <= 1:
                continue
            chosen = min(
                remaining,
                key=lambda item: _sha(
                    [seed, criterion, split, operation, answer, item]
                ),
            )
            selected.append(chosen)
            selected_semantic_ids.add(identity(chosen))
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(
                f"criterion {criterion} operation {operation} lacks "
                f"result-balanced {split} capacity"
            )
        round_index += 1
    return selected


def _smallest_distinct_semantic_rows(
    count: int,
    rows: Iterable[dict[str, Any]],
    *,
    key,
) -> list[dict[str, Any]]:
    best_by_semantic: dict[str, dict[str, Any]] = {}
    for item in rows:
        semantic_id = _semantic_object_id(item)
        incumbent = best_by_semantic.get(semantic_id)
        if incumbent is None or key(item) < key(incumbent):
            best_by_semantic[semantic_id] = item
    return nsmallest(count, best_by_semantic.values(), key=key)


@lru_cache(maxsize=None)
def _split_objects_cached(
    criterion: str,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
    result_stratified: bool,
    dataset_recipe: str | None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    identity = (
        _semantic_object_id
        if _uses_pedagogical_variants(criterion, dataset_recipe)
        else _sha
    )
    operations = _criterion_operations(criterion, dataset_recipe)
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    held_out_semantic_ids: set[str] = set()
    has_variants = (
        _uses_pedagogical_variants(criterion, dataset_recipe)
        and len(_variant_specs(criterion)) > 1
    )
    for split, total in (("validation", validation_count), ("test", test_count)):
        selected: list[dict[str, Any]] = []
        for index, operation in enumerate(operations):
            count = total // len(operations) + int(index < total % len(operations))
            if result_stratified:
                operation_rows = _balanced_result_rows(
                    criterion=criterion,
                    operation=operation,
                    count=count,
                    split=split,
                    seed=seed,
                    held_out_semantic_ids=held_out_semantic_ids,
                    dataset_recipe=dataset_recipe,
                )
            else:
                candidates = (
                    item
                    for item in _candidate_irs(criterion, dataset_recipe)
                    if _operation_key(item) == operation
                    and identity(item) not in held_out_semantic_ids
                )
                selection_key = lambda item: _sha(
                    [seed, criterion, split, operation, item]
                )
                operation_rows = (
                    _smallest_distinct_semantic_rows(
                        count, candidates, key=selection_key
                    )
                    if has_variants
                    else nsmallest(count, candidates, key=selection_key)
                )
            if len(operation_rows) < count:
                raise ValueError(
                    f"criterion {criterion} operation {operation} lacks {split} capacity"
                )
            selected.extend(operation_rows)
            held_out_semantic_ids.update(
                identity(item) for item in operation_rows
            )
        output[split] = tuple(selected)
    train = nsmallest(
        train_count,
        (
            item
            for item in _candidate_irs(criterion, dataset_recipe)
            if identity(item) not in held_out_semantic_ids
        ),
        key=lambda item: _sha([seed, criterion, "train", item]),
    )
    if len(train) < train_count:
        raise ValueError(f"criterion {criterion} lacks train capacity")
    output["train"] = tuple(train)
    return output


def _split_objects(
    criterion: str,
    split_object_counts: dict[str, int],
    seed: int,
    *,
    result_stratified: bool = False,
    dataset_recipe: str | None = None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    return _split_objects_cached(
        criterion,
        int(split_object_counts["train"]),
        int(split_object_counts["validation"]),
        int(split_object_counts["test"]),
        int(seed),
        bool(result_stratified),
        dataset_recipe,
    )


def iter_records(config: dict[str, Any], split: str) -> Iterator[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    seed = int(config["seed"])
    result_stratified = {
        str(value) for value in config.get("result_balanced_criteria", [])
    }
    phases = phases_for_config(config)
    dataset_recipe = str(config.get("dataset_recipe", "")) or None
    counts_by_criterion = _criterion_split_counts(config)
    variants = int(config.get("language_variants_per_object", 2))
    procedure_schema = str(
        config.get("procedure_schema", EXPANDED_PROCEDURE_SCHEMA)
    )
    if procedure_schema not in PROCEDURE_SCHEMAS:
        raise ValueError(f"unsupported procedure schema: {procedure_schema}")
    for phase_index, phase in enumerate(phases):
        for criterion in phase["criteria"]:
            split_object_counts = counts_by_criterion[criterion]
            objects = _split_objects(
                criterion,
                split_object_counts,
                seed,
                result_stratified=criterion in result_stratified,
                dataset_recipe=dataset_recipe,
            )[split]
            for math_ir in objects:
                yield from _records_for_object(
                    split=split,
                    criterion=criterion,
                    math_ir=math_ir,
                    phase_index=phase_index,
                    phases=phases,
                    language_variants_per_object=variants,
                    procedure_schema=procedure_schema,
                    dataset_recipe=dataset_recipe,
                )


def iter_phase_training_records(
    config: dict[str, Any], phase_index: int
) -> Iterator[dict[str, Any]]:
    """Yield one deterministic phase view without exposing future criteria."""
    phases = phases_for_config(config)
    if phase_index < 0 or phase_index >= len(phases):
        raise ValueError(f"invalid phase index: {phase_index}")
    active_criteria = set(phases[phase_index]["criteria"])
    prior_criteria = _phase_prerequisites(phase_index, phases)
    active_rows = [
        record
        for record in iter_records(config, "train")
        if record["criterion_id"] in active_criteria
    ]
    active_fraction = float(config["replay_policy"]["active_fraction"])
    prior_fraction = float(config["replay_policy"]["prior_fraction"])
    if abs(active_fraction + prior_fraction - 1.0) > 1e-9:
        raise ValueError("active and prior replay fractions must sum to one")
    yield from active_rows
    if not prior_criteria:
        return
    replay_total = round(len(active_rows) * prior_fraction / active_fraction)
    minimum_per_prior = int(
        config["replay_policy"].get(
            "minimum_rows_per_prior_criterion", 0
        )
    )
    if minimum_per_prior < 0:
        raise ValueError("minimum replay rows per prior criterion cannot be negative")
    required_replay = minimum_per_prior * len(prior_criteria)
    if minimum_per_prior and replay_total < required_replay:
        raise ValueError(
            f"phase {phase_index} replay budget {replay_total} cannot provide "
            f"{minimum_per_prior} rows for each of {len(prior_criteria)} "
            "previously accepted criteria"
        )
    base, remainder = divmod(replay_total, len(prior_criteria))
    rows_by_criterion: dict[str, list[dict[str, Any]]] = {
        criterion: [] for criterion in prior_criteria
    }
    for record in iter_records(config, "train"):
        criterion = record["criterion_id"]
        if criterion in rows_by_criterion:
            rows_by_criterion[criterion].append(record)
    for criterion_index, criterion in enumerate(prior_criteria):
        count = base + (1 if criterion_index < remainder else 0)
        rows = rows_by_criterion[criterion]
        rows.sort(key=lambda record: _sha([config["seed"], phase_index, record["record_id"]]))
        if not rows:
            raise ValueError(f"no replay rows for criterion {criterion}")
        # Replay is intentionally sampled with replacement. Small, already
        # mastered domains (for example bounded KS1 number bonds) may contain
        # fewer distinct objects than their fair cumulative replay quota.
        for index in range(count):
            yield rows[index % len(rows)]


def _iter_phase_training_records_from_train_file(
    config: dict[str, Any], phase_index: int, train_path: Path
) -> Iterator[dict[str, Any]]:
    """Build a phase view in one bounded-memory pass over sealed train rows."""

    phases = phases_for_config(config)
    if phase_index < 0 or phase_index >= len(phases):
        raise ValueError(f"invalid phase index: {phase_index}")
    counts = _criterion_split_counts(config)
    variants = int(config.get("language_variants_per_object", 2))
    active_criteria = set(phases[phase_index]["criteria"])
    prior_criteria = _phase_prerequisites(phase_index, phases)
    active_total = variants * sum(
        counts[criterion]["train"] for criterion in active_criteria
    )
    active_fraction = float(config["replay_policy"]["active_fraction"])
    prior_fraction = float(config["replay_policy"]["prior_fraction"])
    if abs(active_fraction + prior_fraction - 1.0) > 1e-9:
        raise ValueError("active and prior replay fractions must sum to one")
    replay_total = (
        round(active_total * prior_fraction / active_fraction)
        if prior_criteria
        else 0
    )
    base, remainder = (
        divmod(replay_total, len(prior_criteria))
        if prior_criteria
        else (0, 0)
    )
    replay_quotas = {
        criterion: base + int(index < remainder)
        for index, criterion in enumerate(prior_criteria)
    }
    minimum_per_prior = int(
        config["replay_policy"].get(
            "minimum_rows_per_prior_criterion", 0
        )
    )
    if minimum_per_prior < 0:
        raise ValueError("minimum replay rows per prior criterion cannot be negative")
    if minimum_per_prior and any(
        quota < minimum_per_prior for quota in replay_quotas.values()
    ):
        raise ValueError(
            f"phase {phase_index} cannot provide the configured replay floor"
        )

    # Each heap retains only the deterministically smallest rows required for
    # that criterion. Negative integer hashes turn heapq into a bounded max-heap.
    replay_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        criterion: [] for criterion in prior_criteria
    }
    active_seen = 0
    for _line_number, record in _iter_jsonl(train_path):
        criterion = str(record["criterion_id"])
        if criterion in active_criteria:
            active_seen += 1
            yield record
            continue
        quota = replay_quotas.get(criterion, 0)
        if not quota:
            continue
        row_hash = _sha([config["seed"], phase_index, record["record_id"]])
        item = (-int(row_hash, 16), str(record["record_id"]), record)
        heap = replay_heaps[criterion]
        if len(heap) < quota:
            heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapreplace(heap, item)
    if active_seen != active_total:
        raise ValueError(
            f"phase {phase_index} active row count {active_seen} != {active_total}"
        )
    for criterion in prior_criteria:
        quota = replay_quotas[criterion]
        if not quota:
            continue
        rows = [item[2] for item in replay_heaps[criterion]]
        rows.sort(
            key=lambda record: _sha(
                [config["seed"], phase_index, record["record_id"]]
            )
        )
        if not rows:
            raise ValueError(f"no replay rows for criterion {criterion}")
        for index in range(quota):
            yield rows[index % len(rows)]


def iter_phase_validation_records(
    config: dict[str, Any], phase_index: int, mode: str
) -> Iterator[dict[str, Any]]:
    """Build compact active or criterion-balanced retention validation panels."""
    phases = phases_for_config(config)
    if phase_index < 0 or phase_index >= len(phases):
        raise ValueError(f"invalid phase index: {phase_index}")
    if mode == "active":
        criteria = set(phases[phase_index]["criteria"])
        for record in iter_records(config, "validation"):
            if record["criterion_id"] in criteria:
                yield record
        return
    if mode != "retention":
        raise ValueError(f"unsupported phase validation mode: {mode}")
    prior = _phase_prerequisites(phase_index, phases)
    selected: dict[str, dict[str, Any]] = {}
    for record in iter_records(config, "validation"):
        criterion = record["criterion_id"]
        if criterion in prior:
            incumbent = selected.get(criterion)
            if incumbent is None or _sha(record["record_id"]) < _sha(incumbent["record_id"]):
                selected[criterion] = record
    for criterion in prior:
        if criterion not in selected:
            raise ValueError(f"retention panel is missing criterion {criterion}")
        yield selected[criterion]


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def prepare_dataset(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if config.get("format") != FORMAT:
        raise ValueError(f"config format must be {FORMAT!r}")
    output_root = Path(output_root)
    if (output_root / "manifest.json").exists():
        raise FileExistsError(
            f"sealed dataset already exists at {output_root}; choose a new output path"
        )
    files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = output_root / f"{split}.jsonl"
        count = _write_jsonl(path, iter_records(config, split))
        files[split] = {
            "path": path.name,
            "records": count,
            "sha256": file_sha256(path),
        }
    phases = phases_for_config(config)
    dataset_recipe = str(config.get("dataset_recipe", "")) or None
    split_counts = _criterion_split_counts(config)
    phase_files: dict[str, dict[str, Any]] = {}
    phase_validation_files: dict[str, dict[str, Any]] = {}
    split_files = dict(files)
    for phase_index, phase in enumerate(phases):
        path = output_root / "phase_views" / f"{phase_index:02d}_{phase['name']}.train.jsonl"
        count = _write_jsonl(
            path,
            _iter_phase_training_records_from_train_file(
                config, phase_index, output_root / files["train"]["path"]
            ),
        )
        phase_files[phase["name"]] = {
            "phase_index": phase_index,
            "path": path.relative_to(output_root).as_posix(),
            "records": count,
            "sha256": file_sha256(path),
        }
        active_path = output_root / "phase_validation" / f"{phase_index:02d}_{phase['name']}.active.jsonl"
        active_count = _write_jsonl(
            active_path, iter_phase_validation_records(config, phase_index, "active")
        )
        active_info = {
            "path": active_path.relative_to(output_root).as_posix(),
            "records": active_count,
            "sha256": file_sha256(active_path),
        }
        split_files[f"phase_{phase_index:02d}_active"] = active_info
        validation_info: dict[str, Any] = {"active": active_info}
        if phase_index:
            retention_path = output_root / "phase_validation" / f"{phase_index:02d}_{phase['name']}.retention.jsonl"
            retention_count = _write_jsonl(
                retention_path,
                iter_phase_validation_records(config, phase_index, "retention"),
            )
            retention_info = {
                "path": retention_path.relative_to(output_root).as_posix(),
                "records": retention_count,
                "sha256": file_sha256(retention_path),
            }
            split_files[f"phase_{phase_index:02d}_retention"] = retention_info
            validation_info["retention"] = retention_info
        phase_validation_files[phase["name"]] = validation_info
    manifest = {
        "format": FORMAT,
        "schema": SCHEMA,
        "procedure_schema": str(
            config.get("procedure_schema", EXPANDED_PROCEDURE_SCHEMA)
        ),
        "trace_acceptance_metric": str(
            config.get("trace_acceptance_metric", "exact_v1")
        ),
        "config": config,
        "config_sha256": _sha(config),
        "generator_sha256": file_sha256(Path(__file__)),
        "seed": int(config["seed"]),
        "objects_per_criterion": config["objects_per_criterion"],
        "criterion_split_object_counts": split_counts,
        "criterion_capacity": {
            criterion: {
                "candidate_objects": _candidate_capacity(
                    criterion, dataset_recipe
                ),
                "semantic_objects": _semantic_candidate_capacity(
                    criterion, dataset_recipe
                ),
                "selected_train_objects": split_counts[criterion]["train"],
            }
            for phase in phases
            for criterion in phase["criteria"]
        },
        "phase_train_targets": {
            phase["name"]: sum(
                split_counts[criterion]["train"]
                for criterion in phase["criteria"]
            )
            for phase in phases
        },
        "criterion_operations": {
            criterion: list(
                _criterion_operations(criterion, dataset_recipe)
            )
            for phase in phases
            for criterion in phase["criteria"]
        },
        "result_balanced_criteria": sorted(
            str(value) for value in config.get("result_balanced_criteria", [])
        ),
        "language_variants_per_object": int(
            config.get("language_variants_per_object", 2)
        ),
        "phases": list(phases),
        "replay_policy": {
            "active_fraction": float(config["replay_policy"]["active_fraction"]),
            "prior_fraction": float(config["replay_policy"]["prior_fraction"]),
            "prior_sampling": "criterion_balanced_all_accepted_phases",
            "prior_replacement": "deterministic_cycle_when_quota_exceeds_domain",
            "minimum_rows_per_prior_criterion": int(
                config["replay_policy"].get(
                    "minimum_rows_per_prior_criterion", 0
                )
            ),
            "future_phase_exposure": "forbidden",
        },
        "files": files,
        "splits": split_files,
        "phase_files": phase_files,
        "phase_validation_files": phase_validation_files,
    }
    _atomic_write_json(output_root / "manifest.json", manifest)
    audit = audit_dataset(output_root)
    manifest["audit"] = audit
    manifest["manifest_sha256"] = _sha(
        {key: value for key, value in manifest.items() if key not in {"audit", "manifest_sha256"}}
    )
    _atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def audit_dataset(output_root: Path, scratch_dir: Path | None = None) -> dict[str, Any]:
    output_root = Path(output_root)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("not a canonical math curriculum manifest")
    procedure_schema = str(
        manifest.get("procedure_schema", EXPANDED_PROCEDURE_SCHEMA)
    )
    if procedure_schema not in PROCEDURE_SCHEMAS:
        raise ValueError("unsupported procedure schema in manifest")
    if procedure_schema != str(
        manifest.get("config", {}).get(
            "procedure_schema", EXPANDED_PROCEDURE_SCHEMA
        )
    ):
        raise ValueError("manifest procedure schema disagrees with config")
    trace_acceptance_metric = str(
        manifest.get("trace_acceptance_metric", "exact_v1")
    )
    if trace_acceptance_metric not in {"exact_v1", "semantic_v1"}:
        raise ValueError("unsupported trace acceptance metric in manifest")
    if trace_acceptance_metric != str(
        manifest.get("config", {}).get("trace_acceptance_metric", "exact_v1")
    ):
        raise ValueError("manifest trace acceptance metric disagrees with config")
    if _sha(manifest.get("config")) != manifest.get("config_sha256"):
        raise ValueError("manifest config hash mismatch")
    if file_sha256(Path(__file__)) != manifest.get("generator_sha256"):
        raise ValueError("dataset was built by a different generator revision")
    recorded_manifest_sha = manifest.get("manifest_sha256")
    if recorded_manifest_sha is not None:
        expected_manifest_sha = _sha(
            {key: value for key, value in manifest.items() if key not in {"audit", "manifest_sha256"}}
        )
        if recorded_manifest_sha != expected_manifest_sha:
            raise ValueError("dataset manifest identity mismatch")
    phases = tuple(manifest["phases"])
    criterion_phase = {
        criterion: phase_index
        for phase_index, phase in enumerate(phases)
        for criterion in phase["criteria"]
    }
    scratch = Path(scratch_dir) if scratch_dir else None
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)
    fd, db_name = tempfile.mkstemp(prefix="math-curriculum-audit-", suffix=".sqlite3", dir=scratch)
    os.close(fd)
    db_path = Path(db_name)
    counters: Counter[str] = Counter()
    criterion_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    value_band_counts: Counter[str] = Counter()
    try:
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE records (record_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        connection.execute("CREATE TABLE objects (object_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE semantic_objects "
            "(semantic_id TEXT PRIMARY KEY, split TEXT NOT NULL)"
        )
        connection.execute("CREATE TABLE prompts (prompt_hash TEXT PRIMARY KEY, split TEXT NOT NULL)")
        for split in SPLITS:
            file_info = manifest["files"][split]
            path = output_root / file_info["path"]
            if file_sha256(path) != file_info["sha256"]:
                raise ValueError(f"hash mismatch for {path}")
            for line_number, record in _iter_jsonl(path):
                try:
                    validate_v2_record(record)
                    if record.get("curriculum_schema") != SCHEMA:
                        raise ValueError("wrong curriculum schema")
                    if record["split"] != split:
                        raise ValueError("record split mismatch")
                    criterion = record["criterion_id"]
                    phase_index = int(record["curriculum_phase_index"])
                    if criterion_phase.get(criterion) != phase_index:
                        raise ValueError("criterion appears in the wrong phase")
                    if record["prerequisite_ids"] != _phase_prerequisites(phase_index, phases):
                        raise ValueError("prerequisite list is not cumulative and exact")
                    expected_procedure_schema = str(
                        manifest.get(
                            "procedure_schema", EXPANDED_PROCEDURE_SCHEMA
                        )
                    )
                    if record.get("procedure_schema") != expected_procedure_schema:
                        raise ValueError("record procedure schema disagrees with manifest")
                    expected_answer, expected_derivation = solve_math_ir(
                        record["math_ir"],
                        procedure_schema=expected_procedure_schema,
                    )
                    if record["answer"] != expected_answer:
                        raise ValueError("answer does not match executable math IR")
                    if record["derivation"] != expected_derivation:
                        raise ValueError("derivation does not match executable math IR")
                    if record["problem"] != canonical_json(record["math_ir"]):
                        raise ValueError("math tower input is not canonical math IR")
                    if record["gpt_problem"] != record["natural_language_prompt"]:
                        raise ValueError("dispatcher prompt columns disagree")
                    if record["math_problem"] != record["problem"]:
                        raise ValueError("math private view contains language")
                    if record["dispatcher_target"]["math_ir"] != record["math_ir"]:
                        raise ValueError("dispatcher target math IR disagrees")
                    validate_spans(record)
                    connection.execute(
                        "INSERT INTO records(record_id, split) VALUES (?, ?)",
                        (record["record_id"], split),
                    )
                    object_id = record["math_object_id"]
                    existing = connection.execute(
                        "SELECT split FROM objects WHERE object_id = ?", (object_id,)
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            "INSERT INTO objects(object_id, split) VALUES (?, ?)",
                            (object_id, split),
                        )
                    elif existing[0] != split:
                        raise ValueError("math object occurs in multiple splits")
                    expected_semantic_id = (
                        _semantic_object_id(record["math_ir"])
                        if _uses_pedagogical_variants(
                            str(record["criterion_id"]),
                            str(manifest.get("config", {}).get("dataset_recipe", "")),
                        )
                        else _sha(record["math_ir"])
                    )
                    if record.get("math_semantic_id") != expected_semantic_id:
                        raise ValueError("math semantic identity mismatch")
                    existing_semantic = connection.execute(
                        "SELECT split FROM semantic_objects WHERE semantic_id = ?",
                        (expected_semantic_id,),
                    ).fetchone()
                    if existing_semantic is None:
                        connection.execute(
                            "INSERT INTO semantic_objects(semantic_id, split) "
                            "VALUES (?, ?)",
                            (expected_semantic_id, split),
                        )
                    elif existing_semantic[0] != split:
                        raise ValueError(
                            "semantic math problem occurs in multiple splits"
                        )
                    prompt_hash = _sha(record["natural_language_prompt"].casefold())
                    connection.execute(
                        "INSERT INTO prompts(prompt_hash, split) VALUES (?, ?)",
                        (prompt_hash, split),
                    )
                except Exception as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                counters[f"records.{split}"] += 1
                criterion_counts[f"{split}.{criterion}"] += 1
                if split == "train":
                    operation_counts[
                        f"{criterion}.{record['operation']}"
                    ] += 1
                    variant_counts[
                        f"{criterion}."
                        f"{record.get('pedagogical_representation', 'canonical')}."
                        f"{record.get('pedagogical_strategy', 'canonical')}"
                    ] += 1
                    value_band_counts[
                        f"{criterion}.{record.get('value_band', 'unknown')}"
                    ] += 1
            connection.commit()
            if counters[f"records.{split}"] != int(file_info["records"]):
                raise ValueError(f"record count mismatch for {split}")
        phase_audits: dict[str, Any] = {}
        for phase_name, file_info in manifest["phase_files"].items():
            phase_index = int(file_info["phase_index"])
            path = output_root / file_info["path"]
            if file_sha256(path) != file_info["sha256"]:
                raise ValueError(f"hash mismatch for {path}")
            allowed = set(_phase_prerequisites(phase_index, phases)) | set(
                phases[phase_index]["criteria"]
            )
            active = set(phases[phase_index]["criteria"])
            phase_counts: Counter[str] = Counter()
            for line_number, record in _iter_jsonl(path):
                criterion = record.get("criterion_id")
                if criterion not in allowed:
                    raise ValueError(f"{path}:{line_number}: future-phase exposure")
                phase_counts["total"] += 1
                phase_counts["active" if criterion in active else "replay"] += 1
                phase_counts[f"criterion.{criterion}"] += 1
            if phase_counts["total"] != int(file_info["records"]):
                raise ValueError(f"phase view count mismatch for {phase_name}")
            phase_audits[phase_name] = {
                "total": phase_counts["total"],
                "active": phase_counts["active"],
                "replay": phase_counts["replay"],
                "active_fraction": phase_counts["active"] / phase_counts["total"],
                "criterion_counts": {
                    key.removeprefix("criterion."): value
                    for key, value in sorted(phase_counts.items())
                    if key.startswith("criterion.")
                },
            }
        phase_validation_audits: dict[str, Any] = {}
        for phase_name, modes in manifest.get("phase_validation_files", {}).items():
            phase_index = next(
                index for index, phase in enumerate(phases) if phase["name"] == phase_name
            )
            active_criteria = set(phases[phase_index]["criteria"])
            prior_criteria = set(_phase_prerequisites(phase_index, phases))
            mode_counts: dict[str, int] = {}
            for mode, file_info in modes.items():
                path = output_root / file_info["path"]
                if file_sha256(path) != file_info["sha256"]:
                    raise ValueError(f"hash mismatch for {path}")
                criteria = Counter(
                    str(record["criterion_id"])
                    for _line_number, record in _iter_jsonl(path)
                )
                count = sum(criteria.values())
                if count != int(file_info["records"]):
                    raise ValueError(
                        f"phase validation count mismatch for {phase_name}:{mode}"
                    )
                if mode == "active" and set(criteria) != active_criteria:
                    raise ValueError(f"active validation criteria mismatch for {phase_name}")
                if mode == "retention" and (
                    set(criteria) != prior_criteria
                    or any(value != 1 for value in criteria.values())
                ):
                    raise ValueError(
                        f"retention validation must contain one row per prior criterion: {phase_name}"
                    )
                mode_counts[mode] = count
            phase_validation_audits[phase_name] = mode_counts
        requirements = dict(
            manifest.get("config", {}).get(
                "dataset_quality_requirements", {}
            )
        )
        minimum_criterion = int(
            requirements.get("minimum_train_records_per_criterion", 0)
        )
        criterion_shortfalls = {
            criterion: criterion_counts[f"train.{criterion}"]
            for criterion in criterion_phase
            if criterion_counts[f"train.{criterion}"] < minimum_criterion
        }
        if criterion_shortfalls:
            raise ValueError(
                f"criterion training count shortfall: {criterion_shortfalls}"
            )
        minimum_operation = int(
            requirements.get("minimum_train_records_per_operation", 0)
        )
        operation_shortfalls = {}
        for criterion, operations in manifest.get(
            "criterion_operations", {}
        ).items():
            for operation in operations:
                count = operation_counts[f"{criterion}.{operation}"]
                if count < minimum_operation:
                    operation_shortfalls[f"{criterion}.{operation}"] = count
        if operation_shortfalls:
            raise ValueError(
                f"operation training count shortfall: {operation_shortfalls}"
            )
        minimum_variant = int(
            requirements.get("minimum_train_records_per_variant", 0)
        )
        variant_shortfalls = {}
        recipe = str(manifest.get("config", {}).get("dataset_recipe", ""))
        if recipe in {V6_DATASET_RECIPE, V9_DATASET_RECIPE}:
            for criterion in criterion_phase:
                if not _uses_pedagogical_variants(criterion, recipe):
                    continue
                for variant in _variant_specs(criterion):
                    if not variant:
                        continue
                    key = (
                        f"{criterion}.{variant['representation']}."
                        f"{variant['strategy']}"
                    )
                    count = variant_counts[key]
                    if count < minimum_variant:
                        variant_shortfalls[key] = count
        if variant_shortfalls:
            raise ValueError(
                f"pedagogical variant training count shortfall: {variant_shortfalls}"
            )
        return {
            "status": "passed",
            "records": {split: counters[f"records.{split}"] for split in SPLITS},
            "criterion_counts": dict(sorted(criterion_counts.items())),
            "training_diversity": {
                "operation_counts": dict(sorted(operation_counts.items())),
                "variant_counts": dict(sorted(variant_counts.items())),
                "value_band_counts": dict(sorted(value_band_counts.items())),
            },
            "phase_views": phase_audits,
            "phase_validation": phase_validation_audits,
            "checks": [
                "streaming_json_validation",
                "sqlite_bounded_memory_uniqueness",
                "no_math_object_split_overlap",
                "no_semantic_math_problem_split_overlap",
                "no_prompt_split_overlap",
                "executable_answer_and_derivation",
                "canonical_language_free_math_view",
                "strict_phase_and_prerequisite_metadata",
                "future_phase_training_exposure_forbidden",
                "criterion_balanced_cumulative_replay",
                "explicit_criterion_operation_and_variant_capacity",
                "valid_computation_spans",
            ],
        }
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
        db_path.unlink(missing_ok=True)


def validate_spans(record: dict[str, Any]) -> None:
    trace = record["target_trace"]
    spans = record["computation_spans"]
    kinds = {span["kind"] for span in spans}
    if "compute" not in kinds or "copy" not in kinds:
        raise ValueError("trace needs compute and copy spans")
    previous_end = -1
    for span in sorted(spans, key=lambda value: int(value["start"])):
        start, end = int(span["start"]), int(span["end"])
        if start < 0 or end <= start or end > len(trace) or start < previous_end:
            raise ValueError("invalid or overlapping computation span")
        previous_end = end


def dataset_summary(output_root: Path) -> dict[str, Any]:
    manifest = json.loads((Path(output_root) / "manifest.json").read_text(encoding="utf-8"))
    return {
        "format": manifest["format"],
        "schema": manifest["schema"],
        "files": manifest["files"],
        "phase_files": manifest["phase_files"],
        "phase_validation_files": manifest.get("phase_validation_files", {}),
        "phases": manifest["phases"],
        "replay_policy": manifest["replay_policy"],
        "audit": manifest.get("audit"),
    }


def sample_records(output_root: Path, split: str, limit: int) -> list[dict[str, Any]]:
    manifest = json.loads((Path(output_root) / "manifest.json").read_text(encoding="utf-8"))
    path = Path(output_root) / manifest["files"][split]["path"]
    rows: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(path):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows
