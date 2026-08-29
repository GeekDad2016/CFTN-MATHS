from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import canonical_json
from .data_generator import file_sha256
from .v2_data import make_v2_record, validate_v2_record


FORMAT = "cftn_canonical_math_curriculum_v1"
SCHEMA = "cftn_canonical_math_record_v1"
SPLITS = ("train", "validation", "test")


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


def _candidate_irs(criterion: str) -> Iterator[dict[str, Any]]:
    if criterion == "1NPV-1":
        for value in range(1, 100):
            yield {"type": "math_problem_v1", "op": "predecessor", "value": value}
            yield {"type": "math_problem_v1", "op": "successor", "value": value}
    elif criterion == "1NPV-2":
        for left in range(21):
            for right in range(21):
                if left != right:
                    yield {"type": "math_problem_v1", "op": "compare", "left": left, "right": right}
    elif criterion == "1AS-1":
        for left in range(11):
            for right in range(11 - left):
                yield {"type": "math_problem_v1", "op": "compose", "left": left, "right": right}
    elif criterion == "1NF-1":
        for left in range(11):
            for right in range(11):
                if left + right <= 10:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if left >= right:
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
        for left in range(1, 20):
            for right in range(1, 10):
                if left < 10 < left + right <= 20:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if 11 <= left <= 20 and left - right < 10:
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
                if left >= right:
                    yield {"type": "math_problem_v1", "op": "subtract", "left": left, "right": right}
    elif criterion == "2MD-1":
        for factor in (2, 5, 10):
            for groups in range(22):
                yield {"type": "math_problem_v1", "op": "multiply", "left": factor, "right": groups}
    elif criterion == "2MD-2":
        for divisor in (2, 5, 10):
            for quotient in range(1, 31):
                yield {"type": "math_problem_v1", "op": "divide", "dividend": divisor * quotient, "divisor": divisor}
    elif criterion == "KS2-MULTI-DIGIT":
        for index in range(80):
            yield {"type": "math_problem_v1", "op": "add", "left": 120 + 7 * index, "right": 205 + 11 * index}
    elif criterion == "KS2-LONG-MULTIPLY":
        for left in range(12, 32):
            for right in range(3, 8):
                yield {"type": "math_problem_v1", "op": "multiply", "left": left, "right": right}
    elif criterion == "KS2-EXACT-DIVIDE":
        for divisor in (3, 4, 6, 7, 8, 9, 11, 12):
            for quotient in range(20, 41):
                yield {"type": "math_problem_v1", "op": "divide", "dividend": divisor * quotient, "divisor": divisor}
    elif criterion == "KS2-FRACTION-ADD":
        for denominator in range(2, 13):
            for left in range(1, denominator):
                for right in range(1, denominator):
                    yield {"type": "math_problem_v1", "op": "fraction_add", "left": [left, denominator], "right": [right, denominator]}
    elif criterion == "KS2-PERCENT":
        for percent in (5, 10, 20, 25, 50, 75):
            for base in range(20, 201, 20):
                yield {"type": "math_problem_v1", "op": "percent_of", "percent": percent, "base": base}
    elif criterion == "KS2-RECTANGLE":
        for width in range(2, 13):
            for height in range(2, 13):
                yield {"type": "math_problem_v1", "op": "rectangle_area", "width": width, "height": height}
    elif criterion == "KS3-LINEAR":
        for coefficient in range(2, 10):
            for solution in range(-8, 9):
                constant = 3 * solution - 5
                yield {"type": "math_problem_v1", "op": "linear_solve", "a": coefficient, "b": constant - coefficient * solution, "c": constant}
    elif criterion == "KS3-POWERS":
        for base in range(-5, 6):
            for exponent in range(2, 6):
                yield {"type": "math_problem_v1", "op": "power", "base": base, "exponent": exponent}
    elif criterion == "KS3-PYTHAGORAS":
        for scale in range(1, 21):
            for triple in ((3, 4, 5), (5, 12, 13), (8, 15, 17)):
                yield {"type": "math_problem_v1", "op": "pythagoras", "a": triple[0] * scale, "b": triple[1] * scale}
    elif criterion == "GCSE-QUADRATIC":
        for root1 in range(-8, 9):
            for root2 in range(root1, 9):
                yield {"type": "math_problem_v1", "op": "quadratic_roots", "b": -(root1 + root2), "c": root1 * root2}
    elif criterion == "GCSE-SIMULTANEOUS":
        for x in range(-6, 7):
            for y in range(-6, 7):
                yield {"type": "math_problem_v1", "op": "simultaneous_solve", "equations": [[1, 1, x + y], [2, -1, 2 * x - y]]}
    elif criterion == "GCSE-SEQUENCE":
        for first in range(-10, 11):
            for difference in range(1, 8):
                for index in range(5, 10):
                    yield {"type": "math_problem_v1", "op": "arithmetic_sequence", "first": first, "difference": difference, "index": index}
    elif criterion == "AL-DIFFERENTIATE":
        for coefficient in range(1, 10):
            for power in range(2, 8):
                yield {"type": "math_problem_v1", "op": "differentiate_monomial", "coefficient": coefficient, "power": power}
    elif criterion == "AL-INTEGRATE":
        for coefficient in range(1, 10):
            for power in range(1, 7):
                yield {"type": "math_problem_v1", "op": "integrate_monomial", "coefficient": coefficient * (power + 1), "power": power}
    elif criterion in {"AL-BINOMIAL-COEFFICIENT", "AL-COMBINATION"}:
        for n in range(5, 21):
            for k in range(1, min(n, 8)):
                yield {"type": "math_problem_v1", "op": "binomial_coefficient" if criterion == "AL-BINOMIAL-COEFFICIENT" else "combination", "n": n, "k": k}
    elif criterion == "AL-BINOMIAL-PROB":
        for trials in range(3, 11):
            for successes in range(trials + 1):
                yield {"type": "math_problem_v1", "op": "binomial_probability_half", "trials": trials, "successes": successes}
    elif criterion == "AL-SUVAT":
        for initial in range(-5, 11):
            for acceleration in range(1, 6):
                for time in range(1, 8):
                    yield {"type": "math_problem_v1", "op": "constant_acceleration", "u": initial, "a": acceleration, "t": time}
    elif criterion == "UG-MATRIX-DET":
        for a in range(-4, 5):
            for b in range(-3, 4):
                for c in range(-2, 3):
                    yield {"type": "math_problem_v1", "op": "matrix_det_2x2", "matrix": [[a, b], [c, a + 1]]}
    elif criterion == "UG-MATRIX-SOLVE":
        for x in range(-5, 6):
            for y in range(-5, 6):
                yield {"type": "math_problem_v1", "op": "simultaneous_solve", "equations": [[2, 1, 2 * x + y], [1, -1, x - y]]}
    elif criterion == "UG-POLYNOMIAL-LIMIT":
        for point in range(-5, 6):
            for slope in range(1, 9):
                yield {"type": "math_problem_v1", "op": "cancelled_linear_limit", "point": point, "slope": slope}
    elif criterion == "UG-MOD-INVERSE":
        for modulus in (7, 11, 13, 17, 19, 23, 29):
            for value in range(2, modulus):
                yield {"type": "math_problem_v1", "op": "mod_inverse", "value": value, "modulus": modulus}
    elif criterion == "UG-PERMUTATIONS":
        for n in range(5, 15):
            for k in range(2, min(n, 7)):
                yield {"type": "math_problem_v1", "op": "permutation", "n": n, "k": k}
    elif criterion == "UG-EXPECTATION":
        for maximum in range(3, 21):
            yield {"type": "math_problem_v1", "op": "uniform_expectation", "minimum": 1, "maximum": maximum}
        for maximum in range(3, 21):
            yield {"type": "math_problem_v1", "op": "uniform_expectation", "minimum": -maximum, "maximum": maximum}
    elif criterion == "GRAD-SERIES-RADIUS":
        for base in range(2, 31):
            yield {"type": "math_problem_v1", "op": "geometric_series_radius", "coefficient_base": base}
    elif criterion == "GRAD-CYCLIC-ORDER":
        for modulus in range(8, 41):
            for element in range(1, modulus):
                yield {"type": "math_problem_v1", "op": "cyclic_element_order", "element": element, "modulus": modulus}
    elif criterion == "GRAD-EIGEN-SPECTRUM":
        for left in range(-8, 9):
            for right in range(left, 9):
                yield {"type": "math_problem_v1", "op": "triangular_eigenvalues", "matrix": [[left, 1], [0, right]]}
    elif criterion == "FORMAL-POLY-IDENTITY":
        for a in range(-8, 9):
            for b in range(-8, 9):
                yield {"type": "math_problem_v1", "op": "square_identity", "a": a, "b": b}
    elif criterion == "FORMAL-COUNTEREXAMPLE":
        for value in range(2, 80):
            yield {"type": "math_problem_v1", "op": "odd_square_counterexample", "value": value}
    elif criterion == "FORMAL-EUCLID-INVARIANT":
        for a in range(10, 60):
            for b in range(2, a):
                yield {"type": "math_problem_v1", "op": "euclid_gcd_invariant", "a": a, "b": b}
    else:
        raise ValueError(f"unknown criterion: {criterion}")


def solve_math_ir(math_ir: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    op = math_ir["op"]
    if op == "successor":
        result = int(math_ir["value"]) + 1
    elif op == "predecessor":
        result = int(math_ir["value"]) - 1
    elif op == "compare":
        left, right = int(math_ir["left"]), int(math_ir["right"])
        result = "<" if left < right else ">" if left > right else "="
    elif op in {"add", "compose"}:
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
    return answer, [{"op": op, "result": answer}]


def _fraction_answer(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _language_prompts(math_ir: dict[str, Any]) -> tuple[str, ...]:
    op = math_ir["op"]
    if op == "successor":
        value = math_ir["value"]
        return (f"What number comes after {value}?", f"Give the successor of {value}.")
    if op == "predecessor":
        value = math_ir["value"]
        return (f"What number comes before {value}?", f"Give the predecessor of {value}.")
    if op == "compare":
        left, right = math_ir["left"], math_ir["right"]
        return (f"Compare {left} and {right}.", f"Which symbol, <, >, or =, belongs between {left} and {right}?")
    if op in {"add", "compose", "subtract", "difference", "multiply"}:
        left, right = math_ir["left"], math_ir["right"]
        templates = {
            "add": (f"Calculate {left} + {right}.", f"What is the sum of {left} and {right}?"),
            "compose": (f"Compose {left} and {right} into a whole.", f"What whole is made from parts {left} and {right}?"),
            "subtract": (f"Calculate {left} - {right}.", f"Subtract {right} from {left}."),
            "difference": (f"Find the difference between {left} and {right}.", f"How far apart are {left} and {right}?"),
            "multiply": (f"Calculate {left} x {right}.", f"What is the product of {left} and {right}?"),
        }
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
        return (f"Calculate {dividend} divided by {divisor}.", f"How many groups of {divisor} are in {dividend}?")
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


def _trace(answer: str, derivation: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    work = canonical_json(derivation)
    trace = f"<work>{work}</work><answer>{answer}</answer>"
    work_result = json.dumps(answer, ensure_ascii=False)
    work_start = trace.index(work_result)
    answer_start = trace.index(answer, trace.index("<answer>"))
    spans = [
        {"kind": "compute", "start": work_start, "end": work_start + len(work_result)},
        {"kind": "copy", "start": answer_start, "end": answer_start + len(answer)},
    ]
    return trace, spans


def _records_for_object(
    *,
    split: str,
    criterion: str,
    math_ir: dict[str, Any],
    phase_index: int,
    phases: tuple[dict[str, Any], ...] = PHASES,
) -> Iterator[dict[str, Any]]:
    answer, derivation = solve_math_ir(math_ir)
    math_ir_text = canonical_json(math_ir)
    trace, spans = _trace(answer, derivation)
    object_id = _sha(math_ir)
    phase = phases[phase_index]
    prompts = _language_prompts(math_ir)
    for variant, prompt in enumerate(prompts):
        dispatcher_target = {
            "route": "math",
            "criterion_id": criterion,
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
            "curriculum_phase": phase["name"],
            "curriculum_phase_index": phase_index,
            "prerequisite_ids": _phase_prerequisites(phase_index, phases),
            "educational_level": phase.get("level", "KS1"),
            "numeric_domain": "criterion_defined_taught_domain_v1",
            "representation": "canonical_json_math_ir_v1",
            "evaluation_mode": "held_out_objects_within_taught_domain",
            "math_object_id": object_id,
            "language_variant": variant,
            "computation_spans": spans,
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


def _split_objects(
    criterion: str, split_object_counts: dict[str, int], seed: int
) -> dict[str, list[dict[str, Any]]]:
    candidates = list(_candidate_irs(criterion))
    required = sum(split_object_counts.values())
    if len(candidates) < required:
        raise ValueError(
            f"criterion {criterion} has {len(candidates)} objects, needs {required}"
        )
    candidates.sort(key=lambda item: _sha([seed, criterion, item]))
    output: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split in SPLITS:
        count = int(split_object_counts[split])
        output[split] = candidates[offset : offset + count]
        offset += count
    return output


def iter_records(config: dict[str, Any], split: str) -> Iterator[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    seed = int(config["seed"])
    split_object_counts = {
        name: int(config["objects_per_criterion"][name]) for name in SPLITS
    }
    phases = phases_for_config(config)
    for phase_index, phase in enumerate(phases):
        for criterion in phase["criteria"]:
            objects = _split_objects(criterion, split_object_counts, seed)[split]
            for math_ir in objects:
                yield from _records_for_object(
                    split=split,
                    criterion=criterion,
                    math_ir=math_ir,
                    phase_index=phase_index,
                    phases=phases,
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
        if count > len(rows):
            raise ValueError(f"not enough replay rows for criterion {criterion}")
        yield from rows[:count]


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
    phase_files: dict[str, dict[str, Any]] = {}
    phase_validation_files: dict[str, dict[str, Any]] = {}
    split_files = dict(files)
    for phase_index, phase in enumerate(phases):
        path = output_root / "phase_views" / f"{phase_index:02d}_{phase['name']}.train.jsonl"
        count = _write_jsonl(path, iter_phase_training_records(config, phase_index))
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
        "config": config,
        "config_sha256": _sha(config),
        "generator_sha256": file_sha256(Path(__file__)),
        "seed": int(config["seed"]),
        "objects_per_criterion": config["objects_per_criterion"],
        "language_variants_per_object": 2,
        "phases": list(phases),
        "replay_policy": {
            "active_fraction": float(config["replay_policy"]["active_fraction"]),
            "prior_fraction": float(config["replay_policy"]["prior_fraction"]),
            "prior_sampling": "criterion_balanced_all_accepted_phases",
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
    try:
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE records (record_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        connection.execute("CREATE TABLE objects (object_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
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
                    expected_answer, expected_derivation = solve_math_ir(record["math_ir"])
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
                    prompt_hash = _sha(record["natural_language_prompt"].casefold())
                    connection.execute(
                        "INSERT INTO prompts(prompt_hash, split) VALUES (?, ?)",
                        (prompt_hash, split),
                    )
                except Exception as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                counters[f"records.{split}"] += 1
                criterion_counts[f"{split}.{criterion}"] += 1
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
        return {
            "status": "passed",
            "records": {split: counters[f"records.{split}"] for split in SPLITS},
            "criterion_counts": dict(sorted(criterion_counts.items())),
            "phase_views": phase_audits,
            "phase_validation": phase_validation_audits,
            "checks": [
                "streaming_json_validation",
                "sqlite_bounded_memory_uniqueness",
                "no_math_object_split_overlap",
                "no_prompt_split_overlap",
                "executable_answer_and_derivation",
                "canonical_language_free_math_view",
                "strict_phase_and_prerequisite_metadata",
                "future_phase_training_exposure_forbidden",
                "criterion_balanced_cumulative_replay",
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
