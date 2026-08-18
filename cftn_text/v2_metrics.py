from __future__ import annotations

import re
from collections import defaultdict
from fractions import Fraction
from typing import Any, Iterable

from .v2_data import normalize_answer


_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_SAFE_EXPRESSION = re.compile(r"^[A-Za-z0-9_+\-*/^().,;= \[\]]+$")
_ASSIGNMENT = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^;,]+)")


def extract_v2_answer(text: str) -> str | None:
    matches = _ANSWER.findall(str(text))
    if not matches:
        return None
    value = normalize_answer(matches[-1])
    return value or None


def _fraction(value: str) -> Fraction | None:
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _assignments(value: str) -> dict[str, str] | None:
    matches = _ASSIGNMENT.findall(value)
    if not matches:
        return None
    result = {name.casefold(): normalize_answer(expression) for name, expression in matches}
    return result if len(result) == len(matches) else None


def _symbolically_equal(candidate: str, target: str) -> bool:
    if len(candidate) > 256 or len(target) > 256:
        return False
    if not _SAFE_EXPRESSION.fullmatch(candidate) or not _SAFE_EXPRESSION.fullmatch(target):
        return False
    try:
        import sympy
    except ImportError:
        return False
    try:
        candidate_expr = sympy.sympify(candidate.replace("^", "**"), evaluate=True)
        target_expr = sympy.sympify(target.replace("^", "**"), evaluate=True)
        return bool(sympy.simplify(candidate_expr - target_expr) == 0)
    # Candidate strings are untrusted model output. SymPy can raise parser- and
    # object-specific exceptions (including AttributeError for inputs such as
    # ``e19/8``), so any ordinary exception means "not symbolically equal"
    # rather than an evaluator failure.
    except Exception:
        return False


def answers_equivalent(candidate: str | None, target: str) -> bool:
    if candidate is None:
        return False
    left = normalize_answer(candidate)
    right = normalize_answer(target)
    if left.casefold() == right.casefold():
        return True
    left_fraction = _fraction(left)
    right_fraction = _fraction(right)
    if left_fraction is not None and right_fraction is not None:
        return left_fraction == right_fraction
    left_assignments = _assignments(left)
    right_assignments = _assignments(right)
    if left_assignments is not None and right_assignments is not None:
        if left_assignments.keys() != right_assignments.keys():
            return False
        return all(
            answers_equivalent(left_assignments[key], right_assignments[key])
            for key in left_assignments
        )
    return _symbolically_equal(left, right)


def score_v2_generations(
    generations: Iterable[str], records: Iterable[dict[str, Any]]
) -> tuple[dict[str, Any], list[bool]]:
    generated = list(generations)
    expected = list(records)
    if len(generated) != len(expected):
        raise ValueError("generation and V2 record lengths differ")
    parsed = [extract_v2_answer(text) for text in generated]
    correct = [
        answers_equivalent(value, record["normalized_answer"])
        for value, record in zip(parsed, expected)
    ]
    valid = [value is not None for value in parsed]
    exact_strings = [
        value is not None
        and normalize_answer(value).casefold()
        == normalize_answer(record["normalized_answer"]).casefold()
        for value, record in zip(parsed, expected)
    ]
    count = len(expected)
    by_source: dict[str, list[bool]] = defaultdict(list)
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for record, value in zip(expected, correct):
        by_source[str(record["source"])].append(value)
        by_family[str(record["family"])].append(value)
        by_difficulty[str(record["difficulty"])].append(value)

    def summarize(groups: dict[str, list[bool]]) -> dict[str, Any]:
        return {
            name: {"examples": len(values), "accuracy": sum(values) / len(values)}
            for name, values in sorted(groups.items())
        }

    report = {
        "examples": count,
        "valid_answers": sum(valid),
        "correct_answers": sum(correct),
        "valid_rate": sum(valid) / count if count else 0.0,
        "accuracy": sum(correct) / count if count else 0.0,
        "canonical_string_accuracy": sum(exact_strings) / count if count else 0.0,
        "by_source": summarize(by_source),
        "by_family": summarize(by_family),
        "by_difficulty": summarize(by_difficulty),
    }
    return report, correct
