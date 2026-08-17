from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .config import canonical_json, config_sha256
from .checkpoint import atomic_json_dump
from .data_generator import file_sha256, sha256_bytes


V2_FORMAT = "cftn_text_broad_math_v2"
V2_RECORD_SCHEMA = "cftn_math_record_v2_1"
LOCAL_FAMILIES = (
    "variables_both_sides",
    "nested_parentheses",
    "signed_fractions",
    "two_variable_systems",
    "multi_step_word_problem",
    "distractor_word_problem",
)
DEEPMIND_LICENSE = "Apache-2.0"
GSM8K_LICENSE = "MIT"
MATHQA_LICENSE = "Apache-2.0"
MATHQA_DATA_REVISION = "19d7ec749e673c6bf764ae968f78fd082ac8ad3e"
MATHQA_PARQUET_ROOT = (
    "https://huggingface.co/datasets/allenai/math_qa/resolve/"
    f"{MATHQA_DATA_REVISION}/data"
)
GSM_SYMBOLIC_LICENSE = "Apple-Sample-Code-License"
GSM_SYMBOLIC_FILES = {
    "gsm_symbolic": "GSM_symbolic.jsonl",
    "gsm_symbolic_p1": "GSM_p1.jsonl",
    "gsm_symbolic_p2": "GSM_p2.jsonl",
}
GSM_SYMBOLIC_RAW_ROOT = (
    "https://raw.githubusercontent.com/apple/ml-gsm-symbolic/main/generated_data"
)

_INTEGER = re.compile(r"^[+-]?\d+$")
_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_WORK_TAG = re.compile(r"<work>\s*(.*?)\s*</work>", re.IGNORECASE | re.DOTALL)
_GSM8K_CALCULATION = re.compile(r"<<(.*?)>>", re.DOTALL)
_MATHQA_OPTION = re.compile(r"(?:^|,\s*)([a-e])\s*\)\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_SLOTS = tuple("PQRSTUVWXYZABCDEFGHIJKLMNO")


class _IntegralFloatRandintProxy:
    """Preserve old random.randint support for mathematically integral floats."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    @staticmethod
    def _bound(value: Any) -> Any:
        if isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"non-integral randint bound: {value!r}")
            return int(value)
        return value

    def randint(self, lower: Any, upper: Any) -> int:
        return self._delegate.randint(self._bound(lower), self._bound(upper))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def normalize_problem(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text)).strip()


def normalize_answer(text: str) -> str:
    value = str(text).strip()
    tagged = _ANSWER_TAG.search(value)
    if tagged:
        value = tagged.group(1)
    if "####" in value:
        value = value.rsplit("####", 1)[-1]
    value = value.replace("−", "-").replace("–", "-")
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = _WHITESPACE.sub(" ", value).strip()
    return value


def _integer_value(answer: str) -> int | None:
    normalized = normalize_answer(answer)
    return int(normalized) if _INTEGER.fullmatch(normalized) else None


def _fraction_text(value: Fraction | int) -> str:
    fraction = Fraction(value)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        canonical_json([int(seed), *[str(part) for part in parts]]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _signature(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def make_v2_record(
    *,
    split: str,
    source: str,
    family: str,
    difficulty: int,
    problem: str,
    answer: str,
    target_trace: str | None = None,
    raw_problem: str | None = None,
    native_program: str | None = None,
    execution_trace: str | None = None,
    answer_value: int | None = None,
    gpt_problem: str | None = None,
    math_problem: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_problem = str(problem if raw_problem is None else raw_problem).strip()
    problem = normalize_problem(problem)
    normalized_answer = normalize_answer(answer)
    if not problem:
        raise ValueError("V2 problem cannot be empty")
    if not normalized_answer:
        raise ValueError("V2 answer cannot be empty")
    target_answer = f"<answer>{normalized_answer}</answer>"
    normalized_execution_trace = (
        str(execution_trace).strip() if execution_trace is not None else None
    )
    trace = str(
        target_trace
        or (
            f"{normalized_execution_trace}{target_answer}"
            if normalized_execution_trace
            else target_answer
        )
    ).strip()
    if not trace.endswith(target_answer):
        trace = f"{trace}{target_answer}"
    if (gpt_problem is None) != (math_problem is None):
        raise ValueError("private GPT and math views must be supplied together")
    content_id = _signature({"problem": problem.casefold()})
    payload: dict[str, Any] = {
        "schema_version": V2_RECORD_SCHEMA,
        "content_id": content_id,
        "split": str(split),
        "source": str(source),
        "family": str(family),
        "difficulty": int(difficulty),
        "raw_problem": original_problem,
        "problem": problem,
        "native_program": (
            str(native_program).strip() if native_program is not None else None
        ),
        "execution_trace": normalized_execution_trace,
        "target_trace": trace,
        "target_answer": target_answer,
        "normalized_answer": normalized_answer,
        "answer_value": (
            int(answer_value)
            if answer_value is not None
            else _integer_value(normalized_answer)
        ),
        "gpt_problem": normalize_problem(gpt_problem) if gpt_problem else None,
        "math_problem": normalize_problem(math_problem) if math_problem else None,
        "metadata": dict(metadata or {}),
    }
    payload["record_id"] = _signature(payload)
    validate_v2_record(payload)
    return payload


def validate_v2_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "record_id",
        "content_id",
        "split",
        "source",
        "family",
        "difficulty",
        "raw_problem",
        "problem",
        "native_program",
        "execution_trace",
        "target_trace",
        "target_answer",
        "normalized_answer",
        "answer_value",
        "gpt_problem",
        "math_problem",
        "metadata",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"V2 record is missing fields: {sorted(missing)}")
    if record["schema_version"] != V2_RECORD_SCHEMA:
        raise ValueError("unsupported V2 record schema")
    if int(record["difficulty"]) not in {1, 2, 3}:
        raise ValueError("V2 difficulty must be 1, 2, or 3")
    if not isinstance(record["raw_problem"], str) or not record["raw_problem"].strip():
        raise ValueError("V2 raw_problem must be a non-empty string")
    problem = normalize_problem(record["problem"])
    if record["content_id"] != _signature({"problem": problem.casefold()}):
        raise ValueError("V2 content_id does not match the normalized problem")
    normalized_answer = normalize_answer(record["normalized_answer"])
    if normalized_answer != record["normalized_answer"]:
        raise ValueError("V2 normalized_answer is not canonical")
    expected_answer = f"<answer>{normalized_answer}</answer>"
    if record["target_answer"] != expected_answer:
        raise ValueError("V2 target_answer does not match normalized_answer")
    if not str(record["target_trace"]).endswith(expected_answer):
        raise ValueError("V2 target_trace must end with target_answer")
    for field in ("native_program", "execution_trace"):
        value = record[field]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"V2 {field} must be a non-empty string or null")
    execution_trace = record["execution_trace"]
    if execution_trace is not None and not str(record["target_trace"]).startswith(
        execution_trace
    ):
        raise ValueError("V2 target_trace must begin with execution_trace")
    answer_value = record["answer_value"]
    if answer_value is not None and not isinstance(answer_value, int):
        raise ValueError("V2 answer_value must be an integer or null")
    if (record["gpt_problem"] is None) != (record["math_problem"] is None):
        raise ValueError("V2 private views are incomplete")
    if not isinstance(record["metadata"], dict):
        raise ValueError("V2 metadata must be a mapping")
    unsigned = dict(record)
    recorded_id = unsigned.pop("record_id")
    if recorded_id != _signature(unsigned):
        raise ValueError("V2 record_id does not match record contents")


def load_v2_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                validate_v2_record(item)
            except Exception as exc:
                raise ValueError(
                    f"invalid V2 record at {path}:{line_number}: {exc}"
                ) from exc
            records.append(item)
    return records


def _private_views(
    role_values: dict[str, str],
    semantic_template: str,
    *,
    seed: int,
    signature: str,
) -> tuple[str, str, dict[str, str]]:
    if len(role_values) > len(_SLOTS):
        raise ValueError("too many roles for private-view slots")
    rng = random.Random(_stable_seed(seed, signature, "private_views"))
    slots = list(_SLOTS[: len(role_values)])
    rng.shuffle(slots)
    assignment = dict(zip(role_values, slots))
    language_view = (
        "The math side holds the numeric slot values. "
        + semantic_template.format(**assignment)
    )
    slot_values = sorted(
        ((assignment[role], value) for role, value in role_values.items()),
        key=lambda item: item[0],
    )
    numeric_view = (
        "Opaque numeric slots: "
        + "; ".join(f"{slot}={value}" for slot, value in slot_values)
        + ". Use the language-side role mapping and operations to solve."
    )
    return language_view, numeric_view, assignment


def _nonzero(rng: random.Random, low: int, high: int) -> int:
    while True:
        value = rng.randint(low, high)
        if value:
            return value


def _magnitude(split: str) -> int:
    return 250 if split == "extrapolation" else 30


def _language_template(
    split: str, normal: tuple[str, ...], heldout: tuple[str, ...]
) -> tuple[str, ...]:
    pool = heldout if split == "heldout_language" else normal
    return pool


def _local_payload(
    family: str, split: str, rng: random.Random
) -> tuple[str, str, str, dict[str, str], str, str, int, dict[str, Any]]:
    bound = _magnitude(split)
    if family == "variables_both_sides":
        x = rng.randint(-bound, bound)
        a = _nonzero(rng, -bound, bound)
        c = _nonzero(rng, -bound, bound)
        while c == a:
            c = _nonzero(rng, -bound, bound)
        b = rng.randint(-2 * bound, 2 * bound)
        d = (a - c) * x + b
        normal = (
            f"Solve {a}*x + ({b}) = {c}*x + ({d}).",
            f"Find x when {a} times x plus {b} equals {c} times x plus {d}.",
        )
        heldout = (
            f"A balance has {a} copies of an unknown and a shift of {b} on one side; "
            f"the other has {c} copies and a shift of {d}. Recover the unknown.",
            f"Which value makes ({a}x+{b}) and ({c}x+{d}) identical?",
        )
        problem = rng.choice(_language_template(split, normal, heldout))
        answer = str(x)
        trace = f"<work>({a}-{c})*x={d-b};x={x}</work>"
        roles = {
            "left_coefficient": str(a),
            "left_offset": str(b),
            "right_coefficient": str(c),
            "right_offset": str(d),
        }
        semantic = (
            "{left_coefficient} is the coefficient on the left and {left_offset} "
            "its offset; {right_coefficient} is the coefficient on the right and "
            "{right_offset} its offset. Solve left_coefficient*x+left_offset="
            "right_coefficient*x+right_offset."
        )
        signature = _signature([family, a, b, c, d, x])
        return problem, answer, trace, roles, semantic, signature, 1, {"steps": 3}

    if family == "nested_parentheses":
        x = rng.randint(-bound, bound)
        outer = _nonzero(rng, -12, 12)
        inner = _nonzero(rng, -12, 12)
        shift = rng.randint(-bound, bound)
        offset = rng.randint(-bound, bound)
        result = outer * (inner * x + shift) + offset
        normal = (
            f"Solve {outer}*({inner}*x + ({shift})) + ({offset}) = {result}.",
            f"Find x: {result} equals {outer} times ({inner} times x plus {shift}), "
            f"then plus {offset}.",
        )
        heldout = (
            f"Undo the outer shift {offset} and scale {outer}, then the inner shift "
            f"{shift} and scale {inner}, to recover x from {result}.",
            f"The nested transformation x -> {inner}x+({shift}) -> "
            f"{outer}*value+({offset}) produces {result}. What entered it?",
        )
        problem = rng.choice(_language_template(split, normal, heldout))
        answer = str(x)
        trace = (
            f"<work>{outer}*({inner}*x+({shift}))={result-offset};"
            f"{inner}*x+({shift})={(result-offset)//outer};x={x}</work>"
        )
        roles = {
            "outer_scale": str(outer),
            "inner_scale": str(inner),
            "inner_shift": str(shift),
            "outer_shift": str(offset),
            "result": str(result),
        }
        semantic = (
            "The equation is {outer_scale}*({inner_scale}*x+{inner_shift})+"
            "{outer_shift}={result}. Each brace name is an opaque slot. Solve x."
        )
        signature = _signature([family, outer, inner, shift, offset, result, x])
        return problem, answer, trace, roles, semantic, signature, 2, {"steps": 4}

    if family == "signed_fractions":
        def fraction(nonzero: bool = False) -> Fraction:
            while True:
                value = Fraction(rng.randint(-25, 25), rng.randint(2, 12))
                if value or not nonzero:
                    return value

        x = fraction()
        coefficient = fraction(nonzero=True)
        offset = fraction()
        result = coefficient * x + offset
        coefficient_text = _fraction_text(coefficient)
        offset_text = _fraction_text(offset)
        result_text = _fraction_text(result)
        answer = _fraction_text(x)
        normal = (
            f"Solve ({coefficient_text})*x + ({offset_text}) = {result_text}.",
            f"A signed fraction {coefficient_text} multiplies x; after adding "
            f"{offset_text}, the result is {result_text}. Find x.",
        )
        heldout = (
            f"Which rational value is carried to {result_text} by scaling with "
            f"{coefficient_text} and translating by {offset_text}?",
            f"Reverse y=({coefficient_text})x+({offset_text}) at y={result_text}.",
        )
        problem = rng.choice(_language_template(split, normal, heldout))
        trace = (
            f"<work>x=({result_text}-({offset_text}))/({coefficient_text})={answer}</work>"
        )
        roles = {
            "coefficient": coefficient_text,
            "offset": offset_text,
            "result": result_text,
        }
        semantic = (
            "Solve {coefficient}*x+{offset}={result}; the three brace names are "
            "opaque numeric slots supplied by the math side."
        )
        signature = _signature([family, coefficient_text, offset_text, result_text, answer])
        return problem, answer, trace, roles, semantic, signature, 2, {"steps": 3}

    if family == "two_variable_systems":
        x = rng.randint(-bound, bound)
        y = rng.randint(-bound, bound)
        while True:
            a, b = (_nonzero(rng, -12, 12), _nonzero(rng, -12, 12))
            c, d = (_nonzero(rng, -12, 12), _nonzero(rng, -12, 12))
            if a * d - b * c:
                break
        first = a * x + b * y
        second = c * x + d * y
        normal = (
            f"Solve the system {a}*x + ({b})*y = {first}; "
            f"{c}*x + ({d})*y = {second}. Give x and y.",
            f"Find both unknowns: ({a})x+({b})y={first}, and "
            f"({c})x+({d})y={second}.",
        )
        heldout = (
            f"Two balances share unknowns x and y. The first combines them with "
            f"weights {a} and {b} to make {first}; the second uses {c} and {d} "
            f"to make {second}. Recover the ordered pair.",
            f"Which pair simultaneously satisfies [{a}, {b}] dot [x, y] = {first} "
            f"and [{c}, {d}] dot [x, y] = {second}?",
        )
        problem = rng.choice(_language_template(split, normal, heldout))
        answer = f"x={x};y={y}"
        determinant = a * d - b * c
        trace = (
            f"<work>det={determinant};x={x};y={y};verify=({first},{second})</work>"
        )
        roles = {
            "a": str(a),
            "b": str(b),
            "first_result": str(first),
            "c": str(c),
            "d": str(d),
            "second_result": str(second),
        }
        semantic = (
            "Solve {a}*x+{b}*y={first_result} and "
            "{c}*x+{d}*y={second_result}. Return x and y. Brace names are slots."
        )
        signature = _signature([family, a, b, first, c, d, second, x, y])
        return problem, answer, trace, roles, semantic, signature, 3, {"steps": 4}

    if family in {"multi_step_word_problem", "distractor_word_problem"}:
        steps = 4 if family == "distractor_word_problem" else rng.randint(2, 4)
        groups = rng.randint(2, 15)
        per_group = rng.randint(2, 25)
        removed = rng.randint(1, 30)
        recipients = rng.randint(2, 12)
        if steps == 4:
            each = rng.randint(2, max(3, bound))
            start = recipients * each - groups * per_group + removed
            while start < 1:
                each += 1
                start = recipients * each - groups * per_group + removed
            answer_value = each
        else:
            start = rng.randint(10, max(20, 4 * bound))
            subtotal = start + groups * per_group
            answer_value = subtotal if steps == 2 else subtotal - removed
        if steps == 2:
            core = (
                f"A store starts with {start} notebooks and receives {groups} boxes "
                f"with {per_group} notebooks in each box. How many notebooks are there?"
            )
            semantic = (
                "Start with {start}, then add {groups} groups of {per_group}. "
                "Return the final quantity."
            )
            roles = {
                "start": str(start),
                "groups": str(groups),
                "per_group": str(per_group),
            }
            trace = f"<work>{groups}*{per_group}={groups*per_group};total={answer_value}</work>"
        elif steps == 3:
            core = (
                f"A store starts with {start} notebooks, receives {groups} boxes of "
                f"{per_group}, then donates {removed}. How many remain?"
            )
            semantic = (
                "Start with {start}; add {groups} groups of {per_group}; subtract "
                "{removed}. Return what remains."
            )
            roles = {
                "start": str(start),
                "groups": str(groups),
                "per_group": str(per_group),
                "removed": str(removed),
            }
            trace = (
                f"<work>{start}+{groups}*{per_group}={start+groups*per_group};"
                f"remain={answer_value}</work>"
            )
        else:
            core = (
                f"A store starts with {start} notebooks, receives {groups} boxes of "
                f"{per_group}, donates {removed}, and shares the rest equally among "
                f"{recipients} classrooms. How many notebooks does each classroom get?"
            )
            semantic = (
                "Start with {start}; add {groups} groups of {per_group}; subtract "
                "{removed}; divide equally among {recipients}. Return each share."
            )
            roles = {
                "start": str(start),
                "groups": str(groups),
                "per_group": str(per_group),
                "removed": str(removed),
                "recipients": str(recipients),
            }
            trace = (
                f"<work>{start}+{groups}*{per_group}-{removed}="
                f"{recipients*answer_value};share={answer_value}</work>"
            )
        if split == "heldout_language":
            core = (
                "Work only with quantities that affect the requested result. "
                + core.replace("starts with", "initially records")
                .replace("receives", "later adds")
                .replace("donates", "removes")
            )
        if split == "compositional":
            core = "Evaluate the events in chronological order. " + core
        if family == "distractor_word_problem":
            distractors = (
                "The notebooks have blue covers, which does not affect their count. ",
                "A delivery van travelled 18 kilometres; that fact is irrelevant. ",
                "The manager drank two cups of tea before counting. ",
            )
            core = rng.choice(distractors) + core + " Ignore any decorative details."
        answer = str(answer_value)
        signature = _signature([family, steps, sorted(roles.items()), answer])
        return core, answer, trace, roles, semantic, signature, 3, {"steps": steps}

    raise ValueError(f"unknown local V2 family: {family}")


def iter_local_records(
    *,
    count: int,
    split: str,
    seed: int,
    seen_content: set[str] | None = None,
    seen_signatures: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    seen_content = seen_content if seen_content is not None else set()
    seen_signatures = seen_signatures if seen_signatures is not None else set()
    rng = random.Random(_stable_seed(seed, split, "local_v2"))
    produced = 0
    attempts = 0
    while produced < int(count):
        attempts += 1
        if attempts > max(20_000, int(count) * 100):
            raise RuntimeError(f"could not generate {count} unique local V2 records")
        family = LOCAL_FAMILIES[produced % len(LOCAL_FAMILIES)]
        (
            problem,
            answer,
            trace,
            roles,
            semantic,
            problem_signature,
            difficulty,
            extra_metadata,
        ) = _local_payload(family, split, rng)
        if problem_signature in seen_signatures:
            continue
        gpt_problem, math_problem, role_assignment = _private_views(
            roles,
            semantic,
            seed=seed,
            signature=problem_signature,
        )
        record = make_v2_record(
            split=split,
            source="cftn_generated",
            family=family,
            difficulty=difficulty,
            problem=problem,
            answer=answer,
            target_trace=trace,
            raw_problem=problem,
            native_program=(
                _WORK_TAG.search(trace).group(1) if _WORK_TAG.search(trace) else None
            ),
            execution_trace=trace,
            gpt_problem=gpt_problem,
            math_problem=math_problem,
            metadata={
                "problem_signature": problem_signature,
                "role_assignment": role_assignment,
                "curriculum": family,
                "license": "project-generated",
                **extra_metadata,
            },
        )
        if record["content_id"] in seen_content:
            continue
        seen_content.add(record["content_id"])
        seen_signatures.add(problem_signature)
        produced += 1
        yield record


def _install_deepmind_sympy_compatibility() -> None:
    """Bridge the old generator import to modern SymPy without downgrading torch."""

    try:
        import numpy as np

        if int(np.__version__.split(".", 1)[0]) >= 2:
            raise RuntimeError(
                "mathematics_dataset 1.0.1 requires NumPy < 2; install the "
                "V2 project dependencies before generating data"
            )
        if "object" not in np.__dict__:
            # mathematics_dataset 1.0.1 predates removal of this alias.
            np.object = object  # type: ignore[attr-defined]
    except ImportError as exc:
        raise RuntimeError("NumPy is required by mathematics_dataset") from exc
    public = importlib.import_module("sympy.solvers.diophantine")
    if not hasattr(public, "base_solution_linear"):
        implementation = importlib.import_module(
            "sympy.solvers.diophantine.diophantine"
        )
        public.base_solution_linear = implementation.base_solution_linear
    from mathematics_dataset.sample import number, ops

    if not getattr(number.is_integer, "_cftn_numpy_integer_compatible", False):
        import sympy

        def is_integer(value: Any) -> bool:
            return isinstance(value, (int, np.integer, sympy.Integer))

        is_integer._cftn_numpy_integer_compatible = True  # type: ignore[attr-defined]
        number.is_integer = is_integer
    if not getattr(ops.Constant._is_simple, "_cftn_numpy_integer_compatible", False):
        original_is_simple = ops.Constant._is_simple

        def constant_is_simple(self: Any) -> bool:
            if isinstance(self._value, np.integer):
                return bool(self._value >= 0)
            return original_is_simple(self)

        constant_is_simple._cftn_numpy_integer_compatible = True  # type: ignore[attr-defined]
        ops.Constant._is_simple = constant_is_simple
    # Python 3.12 removed random.randrange's deprecated conversion of integral
    # floats. mathematics_dataset.modules.numbers.round_number computes its
    # inclusive bounds with true division, so both bounds are floats even
    # though they are always whole numbers. Limit the compatibility shim to
    # that upstream module and reject genuinely fractional bounds.
    deepmind_numbers = importlib.import_module("mathematics_dataset.modules.numbers")
    if not isinstance(deepmind_numbers.random, _IntegralFloatRandintProxy):
        deepmind_numbers.random = _IntegralFloatRandintProxy(deepmind_numbers.random)


def _flatten_modules(values: dict[str, Any], prefix: str = "") -> dict[str, Callable]:
    flattened: dict[str, Callable] = {}
    for key, value in values.items():
        name = f"{prefix}__{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_modules(value, name))
        else:
            flattened[name] = value
    return flattened


def _entropy_modifier(level: int, levels: int = 3) -> Callable:
    lower_fraction = int(level) / int(levels)
    upper_fraction = (int(level) + 1) / int(levels)

    def modify(bounds: tuple[float, float]) -> tuple[float, float]:
        lower, upper = bounds
        width = upper - lower
        return (
            lower + width * lower_fraction,
            lower + width * upper_fraction,
        )

    return modify


def _deepmind_module_pool(
    mode: str, selected_names: list[str]
) -> list[tuple[str, int, Callable]]:
    _install_deepmind_sympy_compatibility()
    from mathematics_dataset.modules import modules

    pool: list[tuple[str, int, Callable]] = []
    if mode == "train":
        for level in range(3):
            available = _flatten_modules(modules.train(_entropy_modifier(level)))
            for name in selected_names:
                if name not in available:
                    raise ValueError(f"unknown DeepMind training module: {name}")
                pool.append((name, level + 1, available[name]))
        return pool
    if mode == "interpolate":
        available = _flatten_modules(modules.test())
        difficulty = 2
    elif mode == "extrapolate":
        available = _flatten_modules(modules.test_extra())
        difficulty = 3
    else:
        raise ValueError("DeepMind mode must be train, interpolate, or extrapolate")
    for name in selected_names:
        if name not in available:
            raise ValueError(f"unknown DeepMind {mode} module: {name}")
        pool.append((name, difficulty, available[name]))
    return pool


def iter_deepmind_records(
    *,
    count: int,
    split: str,
    seed: int,
    mode: str,
    selected_modules: list[str],
    seen_content: set[str] | None = None,
    seen_signatures: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    seen_content = seen_content if seen_content is not None else set()
    seen_signatures = seen_signatures if seen_signatures is not None else set()
    pool = _deepmind_module_pool(mode, selected_modules)
    if not pool:
        raise ValueError("DeepMind module pool is empty")
    random.seed(_stable_seed(seed, split, mode, "deepmind"))
    try:
        import numpy as np

        np.random.seed(_stable_seed(seed, split, mode, "numpy") % (2**32 - 1))
    except ImportError:
        pass
    total = int(count)
    base_quota, remainder = divmod(total, len(pool))
    produced = 0
    for pool_index, (name, difficulty, function) in enumerate(pool):
        quota = base_quota + int(pool_index < remainder)
        module_produced = 0
        attempts = 0
        while module_produced < quota:
            attempts += 1
            if attempts > max(1_000, quota * 25):
                raise RuntimeError(
                    f"DeepMind module {name} difficulty {difficulty} could not "
                    f"produce its balanced quota of {quota} unique records"
                )
            try:
                example = function()
            except (ArithmeticError, ValueError, OverflowError, AssertionError):
                # mathematics_dataset uses assertions to reject occasional
                # degenerate stochastic samples (for example, a zero-term
                # polynomial expansion with residual entropy).  These are
                # invalid draws, not broken module contracts, and the bounded
                # attempt budget above prevents an unhealthy module from
                # retrying forever.
                continue
            raw_problem = str(example.question)
            problem = normalize_problem(raw_problem)
            answer = normalize_answer(str(example.answer))
            signature = _signature(["deepmind", mode, name, problem.casefold()])
            if signature in seen_signatures:
                continue
            record = make_v2_record(
                split=split,
                source="deepmind_mathematics",
                family=name,
                difficulty=difficulty,
                problem=problem,
                answer=answer,
                raw_problem=raw_problem,
                metadata={
                    "problem_signature": signature,
                    "module": name,
                    "generator_partition": mode,
                    "entropy_level": difficulty if mode == "train" else None,
                    "license": DEEPMIND_LICENSE,
                    "source_url": "https://github.com/google-deepmind/mathematics_dataset",
                },
            )
            if record["content_id"] in seen_content:
                continue
            seen_content.add(record["content_id"])
            seen_signatures.add(signature)
            module_produced += 1
            produced += 1
            yield record
    if produced != total:
        raise AssertionError(f"DeepMind generator produced {produced}, expected {total}")


def _gsm8k_answer(raw_answer: str) -> str:
    if "####" not in raw_answer:
        raise ValueError("GSM8K answer has no final #### delimiter")
    return normalize_answer(raw_answer.rsplit("####", 1)[-1])


def _gsm8k_program_and_trace(raw_answer: str) -> tuple[str | None, str | None]:
    calculations = [
        normalize_problem(item)
        for item in _GSM8K_CALCULATION.findall(str(raw_answer))
        if normalize_problem(item)
    ]
    if not calculations:
        return None, None
    expressions = [
        item.rsplit("=", 1)[0].strip() if "=" in item else item
        for item in calculations
    ]
    return "; ".join(expressions), f"<work>{'; '.join(calculations)}</work>"


def iter_gsm8k_records(
    *,
    hf_split: str,
    output_split: str,
    count: int | None,
    seen_content: set[str] | None = None,
    seen_signatures: set[str] | None = None,
    dataset_rows: Iterable[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    seen_content = seen_content if seen_content is not None else set()
    seen_signatures = seen_signatures if seen_signatures is not None else set()
    if dataset_rows is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets is required to download GSM8K") from exc
        dataset_rows = load_dataset("openai/gsm8k", "main", split=hf_split)
    produced = 0
    for index, row in enumerate(dataset_rows):
        if count is not None and produced >= int(count):
            break
        raw_problem = str(row["question"])
        raw_answer = str(row["answer"])
        problem = normalize_problem(raw_problem)
        answer = _gsm8k_answer(raw_answer)
        native_program, execution_trace = _gsm8k_program_and_trace(raw_answer)
        signature = _signature(["gsm8k", hf_split, problem.casefold()])
        if signature in seen_signatures:
            continue
        step_count = max(2, str(row["answer"]).count("<<"))
        record = make_v2_record(
            split=output_split,
            source="gsm8k",
            family="multi_step_word_problem",
            difficulty=3 if step_count >= 4 else 2,
            problem=problem,
            answer=answer,
            raw_problem=raw_problem,
            native_program=native_program,
            execution_trace=execution_trace,
            metadata={
                "problem_signature": signature,
                "official_split": hf_split,
                "official_index": index,
                "estimated_steps": step_count,
                "trace_provenance": (
                    "gsm8k_calculator_annotations"
                    if execution_trace is not None
                    else "answer_only"
                ),
                "license": GSM8K_LICENSE,
                "source_url": "https://huggingface.co/datasets/openai/gsm8k",
            },
        )
        if record["content_id"] in seen_content:
            continue
        seen_content.add(record["content_id"])
        seen_signatures.add(signature)
        produced += 1
        yield record
    if count is not None and produced != int(count):
        raise RuntimeError(
            f"requested {count} unique GSM8K {hf_split} rows, found {produced}"
        )


def _mathqa_options(options: str) -> dict[str, str]:
    text = str(options).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            serialized = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            serialized = None
        if isinstance(serialized, (list, tuple)) and all(
            isinstance(item, str) for item in serialized
        ):
            text = ", ".join(serialized)
    matches = list(_MATHQA_OPTION.finditer(text))
    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip().strip(",").strip()
        if value:
            parsed[match.group(1).casefold()] = value
    return parsed


def iter_mathqa_records(
    *,
    hf_split: str,
    output_split: str,
    count: int | None,
    seen_content: set[str] | None = None,
    seen_signatures: set[str] | None = None,
    dataset_rows: Iterable[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Adapt MathQA while retaining its validated native operation program."""

    seen_content = seen_content if seen_content is not None else set()
    seen_signatures = seen_signatures if seen_signatures is not None else set()
    if dataset_rows is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets is required to download MathQA") from exc
        # MathQA's main branch still contains a legacy dataset script, which
        # modern `datasets` versions intentionally refuse to execute. Use the
        # immutable official Parquet conversion directly instead.
        dataset_rows = load_dataset(
            "parquet",
            data_files={
                hf_split: (
                    f"{MATHQA_PARQUET_ROOT}/{hf_split}-00000-of-00001.parquet"
                )
            },
            split=hf_split,
        )
    produced = 0
    for index, row in enumerate(dataset_rows):
        if count is not None and produced >= int(count):
            break
        raw_problem = str(row["Problem"])
        raw_options = str(row["options"])
        problem = normalize_problem(f"{raw_problem} Answer choices: {raw_options}")
        correct = str(row["correct"]).strip().casefold()
        options = _mathqa_options(raw_options)
        if correct not in options:
            raise ValueError(
                f"MathQA {hf_split} row {index} has no option {correct!r}"
            )
        answer = normalize_answer(options[correct])
        annotated_program = normalize_problem(str(row.get("annotated_formula") or ""))
        linear_program = normalize_problem(str(row.get("linear_formula") or "")).rstrip(
            "|"
        )
        # The linear form is the source-native execution trace: it names each
        # operation once and references prior results instead of recursively
        # duplicating the whole expression tree (some nested forms exceed 6K
        # bytes despite representing only a few dozen operations).
        native_program = linear_program or annotated_program
        if not native_program:
            raise ValueError(f"MathQA {hf_split} row {index} has no operation program")
        execution_trace = f"<program>{native_program}</program>"
        signature = _signature(["mathqa", hf_split, problem.casefold()])
        if signature in seen_signatures:
            continue
        operation_count = max(
            1,
            (linear_program.count("|") + 1) if linear_program else annotated_program.count("("),
        )
        record = make_v2_record(
            split=output_split,
            source="mathqa",
            family=str(row.get("category") or "mathqa_word_problem"),
            difficulty=1 if operation_count == 1 else (2 if operation_count <= 3 else 3),
            problem=problem,
            answer=answer,
            raw_problem=raw_problem,
            native_program=native_program,
            execution_trace=execution_trace,
            metadata={
                "problem_signature": signature,
                "official_split": hf_split,
                "official_index": index,
                "correct_option": correct,
                "options": raw_options,
                "annotated_formula": annotated_program,
                "program_representation": (
                    "mathqa_linear_formula" if linear_program else "mathqa_annotated_formula"
                ),
                "operation_count": operation_count,
                "trace_provenance": "mathqa_official_operation_program",
                "dataset_revision": MATHQA_DATA_REVISION,
                "license": MATHQA_LICENSE,
                "source_url": "https://huggingface.co/datasets/allenai/math_qa",
            },
        )
        if record["content_id"] in seen_content:
            continue
        seen_content.add(record["content_id"])
        seen_signatures.add(signature)
        produced += 1
        yield record
    if count is not None and produced != int(count):
        raise RuntimeError(
            f"requested {count} unique MathQA {hf_split} rows, found {produced}"
        )


def _download_if_missing(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        urllib.request.urlretrieve(url, temporary)
        with temporary.open("rb") as handle:
            header = handle.read(128).decode("utf-8", errors="ignore")
        if header.startswith("version https://git-lfs.github.com"):
            raise RuntimeError(f"downloaded a Git LFS pointer instead of data: {url}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def iter_gsm_symbolic_records(
    *,
    variant: str,
    cache_root: str | Path,
    count: int | None,
    seen_content: set[str] | None = None,
    seen_signatures: set[str] | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    if variant not in GSM_SYMBOLIC_FILES:
        raise ValueError(f"unknown GSM-Symbolic variant: {variant}")
    seen_content = seen_content if seen_content is not None else set()
    seen_signatures = seen_signatures if seen_signatures is not None else set()
    if rows is None:
        filename = GSM_SYMBOLIC_FILES[variant]
        cache_path = Path(cache_root) / "gsm_symbolic" / filename
        _download_if_missing(f"{GSM_SYMBOLIC_RAW_ROOT}/{filename}", cache_path)

        def loaded_rows() -> Iterator[dict[str, Any]]:
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

        rows = loaded_rows()
    produced = 0
    for index, row in enumerate(rows):
        if count is not None and produced >= int(count):
            break
        raw_problem = str(row["question"])
        raw_answer = str(row["answer"])
        problem = normalize_problem(raw_problem)
        answer = _gsm8k_answer(raw_answer)
        native_program, execution_trace = _gsm8k_program_and_trace(raw_answer)
        signature = _signature(["gsm_symbolic", variant, problem.casefold()])
        if signature in seen_signatures:
            continue
        record = make_v2_record(
            split=variant,
            source="gsm_symbolic",
            family="symbolic_word_problem",
            difficulty=3,
            problem=problem,
            answer=answer,
            raw_problem=raw_problem,
            native_program=native_program,
            execution_trace=execution_trace,
            metadata={
                "problem_signature": signature,
                "variant": variant,
                "official_index": index,
                "id": row.get("id"),
                "original_id": row.get("original_id"),
                "canary_sha256": (
                    _signature(row.get("canary")) if row.get("canary") else None
                ),
                "license": GSM_SYMBOLIC_LICENSE,
                "evaluation_only": True,
                "source_url": "https://github.com/apple/ml-gsm-symbolic",
            },
        )
        if record["content_id"] in seen_content:
            continue
        seen_content.add(record["content_id"])
        seen_signatures.add(signature)
        produced += 1
        yield record
    if count is not None and produced != int(count):
        raise RuntimeError(
            f"requested {count} unique {variant} rows, found {produced}"
        )


def _atomic_write_records(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    expected_count: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
    max_math_length: int | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    count = 0
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    maximum_math_tokens = 0
    try:
        with temporary.open("wb") as handle:
            for record in records:
                validate_v2_record(record)
                math_problem = str(record.get("math_problem") or record["problem"])
                math_tokens = (
                    len(f"Problem: {math_problem}\nSolution:".encode("utf-8"))
                    + len(str(record["target_trace"]).encode("utf-8"))
                    + 3  # BOS, separator, and EOS control tokens.
                )
                maximum_math_tokens = max(maximum_math_tokens, math_tokens)
                if max_math_length is not None and math_tokens > int(max_math_length):
                    raise RuntimeError(
                        f"record {record['record_id']} requires {math_tokens} math tokens; "
                        f"configured maximum is {max_math_length}"
                    )
                payload = (canonical_json(record) + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
                count += 1
                source_counts[str(record["source"])] += 1
                family_counts[str(record["family"])] += 1
                difficulty_counts[str(record["difficulty"])] += 1
                if progress_callback is not None and (
                    count == 1 or count % 1000 == 0
                ):
                    progress_callback(count)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if expected_count is not None and count != int(expected_count):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"split wrote {count} records; expected {expected_count}")
    os.replace(temporary, path)
    return {
        "path": path.name,
        "count": count,
        "sha256": digest.hexdigest(),
        "source_counts": dict(sorted(source_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "maximum_math_sequence_tokens": maximum_math_tokens,
    }


def _chain_sources(
    config: dict[str, Any],
    *,
    split: str,
    allocations: dict[str, int],
    seen_content: set[str],
    seen_signatures: set[str],
) -> Iterator[dict[str, Any]]:
    data = config["data"]
    seed = int(config["project"]["seed"])
    modules_config = data["deepmind_modules"]
    for source, count in allocations.items():
        count = int(count)
        if source == "local":
            yield from iter_local_records(
                count=count,
                split=split,
                seed=seed,
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            )
        elif source in {"deepmind", "deepmind_train"}:
            yield from iter_deepmind_records(
                count=count,
                split=split,
                seed=seed,
                mode="train",
                selected_modules=list(modules_config["train"]),
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            )
        elif source == "deepmind_interpolate":
            yield from iter_deepmind_records(
                count=count,
                split=split,
                seed=seed,
                mode="interpolate",
                selected_modules=list(modules_config["interpolate"]),
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            )
        elif source == "deepmind_extrapolate":
            yield from iter_deepmind_records(
                count=count,
                split=split,
                seed=seed,
                mode="extrapolate",
                selected_modules=list(modules_config["extrapolate"]),
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            )
        elif source == "gsm8k":
            yield from iter_gsm8k_records(
                hf_split="train",
                output_split=split,
                count=count,
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            )
        elif source == "mathqa":
            yield from iter_mathqa_records(
                hf_split="train",
                output_split=split,
                count=count,
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            )
        else:
            raise ValueError(f"unknown V2 source allocation: {source}")


def prepare_v2_manifests(
    config: dict[str, Any],
    output_root: str | Path | None = None,
    *,
    force: bool = False,
    include_external_benchmarks: bool = True,
) -> dict[str, Any]:
    if config["data"].get("format") != V2_FORMAT:
        raise ValueError("prepare_v2_manifests requires a V2 configuration")
    root = Path(output_root or config["project"]["data_root"]).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("config_sha256") != config_sha256(config):
            raise FileExistsError(
                f"{manifest_path} belongs to another configuration; use --force"
            )
        audit_v2_manifest(existing, root)
        return existing

    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "prepare_status.json"
    started_at = time.time()
    atomic_json_dump(
        {
            "state": "running",
            "phase": "initializing",
            "elapsed_seconds": 0.0,
        },
        status_path,
    )
    seen_content: set[str] = set()
    seen_signatures: set[str] = set()
    split_metadata: dict[str, Any] = {}
    data = config["data"]
    requested = {
        "train": dict(data["training_sources"]),
        **{
            split: dict(allocations)
            for split, allocations in data["split_sources"].items()
        },
    }
    expected_counts = {
        "train": int(data["train_examples"]),
        "calibration": int(data["calibration_examples"]),
        "validation": int(data["validation_examples"]),
        "test": int(data["test_examples"]),
        "heldout_language": int(data["heldout_language_examples"]),
        "extrapolation": int(data["extrapolation_examples"]),
        "compositional": int(data["compositional_examples"]),
    }
    for split, allocations in requested.items():
        expected = expected_counts[split]
        if sum(int(value) for value in allocations.values()) != expected:
            raise ValueError(f"V2 {split} allocations do not sum to {expected}")
        def progress(completed: int, current_split: str = split) -> None:
            atomic_json_dump(
                {
                    "state": "running",
                    "phase": "writing_split",
                    "split": current_split,
                    "completed": int(completed),
                    "total": int(expected),
                    "elapsed_seconds": time.time() - started_at,
                },
                status_path,
            )
            if completed == 1 or completed % 5000 == 0:
                print(
                    f"V2 data {current_split}: {completed}/{expected}",
                    flush=True,
                )

        split_metadata[split] = _atomic_write_records(
            root / f"{split}.jsonl",
            _chain_sources(
                config,
                split=split,
                allocations=allocations,
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            ),
            expected_count=expected,
            progress_callback=progress,
            max_math_length=int(data["max_math_length"]),
        )

    if include_external_benchmarks:
        gsm8k_test_count = int(data.get("gsm8k_test_examples", 1319))

        def benchmark_progress(
            split_name: str, total_count: int
        ) -> Callable[[int], None]:
            def progress(completed: int) -> None:
                atomic_json_dump(
                    {
                        "state": "running",
                        "phase": "writing_sealed_benchmark",
                        "split": split_name,
                        "completed": int(completed),
                        "total": int(total_count),
                        "elapsed_seconds": time.time() - started_at,
                    },
                    status_path,
                )
                if completed == 1 or completed % 1000 == 0:
                    print(
                        f"V2 data {split_name}: {completed}/{total_count}",
                        flush=True,
                    )

            return progress

        split_metadata["gsm8k_test"] = _atomic_write_records(
            root / "gsm8k_test.jsonl",
            iter_gsm8k_records(
                hf_split="test",
                output_split="gsm8k_test",
                count=gsm8k_test_count,
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            ),
            expected_count=gsm8k_test_count,
            progress_callback=benchmark_progress(
                "gsm8k_test", gsm8k_test_count
            ),
            max_math_length=int(data["max_math_length"]),
        )
        for mathqa_split, configured_key in (
            ("validation", "mathqa_validation_examples"),
            ("test", "mathqa_test_examples"),
        ):
            benchmark_name = f"mathqa_{mathqa_split}"
            benchmark_count = int(
                data.get(
                    configured_key,
                    4475 if mathqa_split == "validation" else 2985,
                )
            )
            split_metadata[benchmark_name] = _atomic_write_records(
                root / f"{benchmark_name}.jsonl",
                iter_mathqa_records(
                    hf_split=mathqa_split,
                    output_split=benchmark_name,
                    count=benchmark_count,
                    seen_content=seen_content,
                    seen_signatures=seen_signatures,
                ),
                expected_count=benchmark_count,
                progress_callback=benchmark_progress(benchmark_name, benchmark_count),
                max_math_length=int(data["max_math_length"]),
            )
        maximum_symbolic = data.get("gsm_symbolic_examples_per_variant")
        for variant in GSM_SYMBOLIC_FILES:
            iterator = iter_gsm_symbolic_records(
                variant=variant,
                cache_root=root / "raw_cache",
                count=(int(maximum_symbolic) if maximum_symbolic is not None else None),
                seen_content=seen_content,
                seen_signatures=seen_signatures,
            )
            split_metadata[variant] = _atomic_write_records(
                root / f"{variant}.jsonl",
                iterator,
                expected_count=(
                    int(maximum_symbolic) if maximum_symbolic is not None else None
                ),
                progress_callback=(
                    benchmark_progress(variant, int(maximum_symbolic))
                    if maximum_symbolic is not None
                    else None
                ),
                max_math_length=int(data["max_math_length"]),
            )

    source_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "format": V2_FORMAT,
        "record_schema": V2_RECORD_SCHEMA,
        "seed": int(config["project"]["seed"]),
        "config_sha256": config_sha256(config),
        "generator_sha256": file_sha256(source_path),
        "splits": split_metadata,
        "total_records": sum(item["count"] for item in split_metadata.values()),
        "train_records": split_metadata["train"]["count"],
        "content_overlap": 0,
        "training_uses_gsm8k_test": False,
        "training_uses_mathqa_validation_or_test": False,
        "training_uses_gsm_symbolic": False,
        "external_data_committed_to_git": False,
        "licenses": {
            "deepmind_mathematics": DEEPMIND_LICENSE,
            "gsm8k": GSM8K_LICENSE,
            "mathqa": MATHQA_LICENSE,
            "gsm_symbolic": GSM_SYMBOLIC_LICENSE,
        },
    }
    unsigned = canonical_json(manifest).encode("utf-8")
    manifest["manifest_sha256"] = sha256_bytes(unsigned)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
    audit_v2_manifest(manifest, root)
    atomic_json_dump(
        {
            "state": "completed",
            "phase": "complete",
            "manifest": str(manifest_path.resolve()),
            "total_records": manifest["total_records"],
            "elapsed_seconds": time.time() - started_at,
        },
        status_path,
    )
    return manifest


def audit_v2_manifest(manifest: dict[str, Any], root: str | Path) -> dict[str, Any]:
    if manifest.get("format") != V2_FORMAT:
        raise ValueError("unsupported V2 data manifest format")
    unsigned = dict(manifest)
    recorded_hash = unsigned.pop("manifest_sha256", None)
    if recorded_hash != sha256_bytes(canonical_json(unsigned).encode("utf-8")):
        raise ValueError("V2 manifest hash mismatch")
    if manifest.get("generator_sha256") != file_sha256(Path(__file__).resolve()):
        raise ValueError("V2 generator source hash mismatch")
    root_path = Path(root)
    seen_content: set[str] = set()
    signatures_by_split: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for split, metadata in manifest["splits"].items():
        path = root_path / metadata["path"]
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"V2 split hash mismatch: {split}")
        records = load_v2_records(path)
        if len(records) != int(metadata["count"]):
            raise ValueError(f"V2 split count mismatch: {split}")
        content_ids = {record["content_id"] for record in records}
        if len(content_ids) != len(records):
            raise ValueError(f"duplicate V2 content within split: {split}")
        overlap = seen_content.intersection(content_ids)
        if overlap:
            raise ValueError(f"V2 content overlap across splits: {split}")
        seen_content.update(content_ids)
        signatures = {
            str(record["metadata"].get("problem_signature", record["content_id"]))
            for record in records
        }
        signatures_by_split[split] = signatures
        counts[split] = len(records)
    train_signatures = signatures_by_split.get("train", set())
    for split, signatures in signatures_by_split.items():
        if split != "train" and train_signatures.intersection(signatures):
            raise ValueError(f"V2 semantic overlap with training data: {split}")
    if int(manifest["train_records"]) != counts.get("train", 0):
        raise ValueError("V2 train record total is inconsistent")
    return {
        "pass": True,
        "counts": counts,
        "total_unique_content": len(seen_content),
        "training_overlap": 0,
    }


def curriculum_records(
    records: list[dict[str, Any]], config: dict[str, Any], epoch: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curriculum = config["data"].get("curriculum", {})
    if not curriculum.get("enabled", False):
        return records, {"enabled": False, "phase": "all", "max_difficulty": 3}
    phase = next(
        (
            item
            for item in curriculum["phases"]
            if int(epoch) <= int(item["through_epoch"])
        ),
        curriculum["phases"][-1],
    )
    maximum = int(phase["max_difficulty"])
    selected = [record for record in records if int(record.get("difficulty", 3)) <= maximum]
    if not selected:
        raise RuntimeError(f"curriculum phase {phase['name']} selected no records")
    return selected, {
        "enabled": True,
        "phase": str(phase["name"]),
        "max_difficulty": maximum,
        "available_examples": len(selected),
        "total_examples": len(records),
    }
