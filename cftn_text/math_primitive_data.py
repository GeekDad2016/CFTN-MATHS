"""Small, independently versioned lessons; no changes to the sealed V2 corpus.

This is an offline curriculum and diagnostic, not an inference calculator.
Inputs contain only public questions. Splits are assigned to mathematical objects
before rendering; alternate target representations share the same questions.
"""
from __future__ import annotations

import random
import re
from fractions import Fraction

from .computation_supervision import ComputationCollator
from .verified_math_data import DECIMAL, decimal_text, fingerprint, operate, question_operands

VERSION = "cftn_math_primitive_v1"
FOUNDATIONS = ("bind_product", "bind_system", "decimal_scale", "decimal_restore",
               "integer_multiply", "integer_subtract", "integer_divide")
COMPOSITIONS = ("decimal_multiply", "multiply_add")
ARMS = ("answer_only", "compact_worked")


def fixed_decimal(n: int, places: int) -> str:
    if not 0 <= places <= 3:
        raise ValueError("primitive decimal scale outside support")
    digits = str(abs(n)).zfill(places + 1)
    return ("-" if n < 0 else "") + (digits if not places else digits[:-places] + "." + digits[-places:])


def parse_question(question: str) -> tuple[str, tuple]:
    """Strict public-only binder. Unsupported wording is an explicit failure."""
    if len(question) > 1000:
        raise ValueError("primitive question too long")
    if question.startswith("Read operands only: ") and question.endswith(" Return a and b."):
        inner = question[len("Read operands only: "):-len(" Return a and b.")]
        return "bind_product", question_operands({"family": "arithmetic__mul", "problem": inner})
    if question.startswith("Read coefficients only: ") and question.endswith(" Return a,b,r,c,d,s."):
        inner = question[len("Read coefficients only: "):-len(" Return a,b,r,c,d,s.")]
        return "bind_system", question_operands({"family": "two_variable_systems", "problem": inner})
    patterns = (
        ("decimal_scale", rf"Write ({DECIMAL}) as n/10\^k with k=([0-3]). Return n and s=10\^k\."),
        ("decimal_restore", r"Write (-?\d+)/(1|10|100|1000) as a decimal\."),
        ("integer_multiply", r"Calculate (-?\d+)\*(-?\d+)\."),
        ("integer_subtract", r"Calculate (-?\d+)-\((-?\d+)\)\."),
        ("integer_divide", r"Calculate (-?\d+)/\(([1-9]\d*)\)\. Return an exact integer\."),
        ("decimal_multiply", rf"Calculate ({DECIMAL})\*({DECIMAL})\."),
        ("multiply_add", r"Calculate \((-?\d+)\*(-?\d+)\)\+\((-?\d+)\)\."),
    )
    for family, pattern in patterns:
        match = re.fullmatch(pattern, question)
        if match:
            return family, tuple(Fraction(v) for v in match.groups())
    raise ValueError("unsupported primitive public question")


def object_key(question: str) -> str:
    family, values = parse_question(question)
    if family == "bind_product":
        # Order matters for extraction; reserve BOTH permutations together.
        values = tuple(sorted(values))
    elif family == "integer_multiply" or family == "decimal_multiply":
        family, values = "product", tuple(sorted(values))
    elif family in ("decimal_scale", "decimal_restore"):
        value = values[0] if family == "decimal_scale" else values[0] / values[1]
        scale = 10 ** int(values[1]) if family == "decimal_scale" else values[1]
        family, values = "decimal_conversion", (value, Fraction(scale))
    return fingerprint([family, [str(v) for v in values]])


def object_split(key: str) -> str:
    return "validation" if int(key[:12], 16) % 5 == 0 else "train"


def lesson(question: str, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown primitive target arm")
    family, values = parse_question(question)
    if any(max(v.numerator.bit_length(), v.denominator.bit_length()) > 64 for v in values):
        raise ValueError("primitive operand outside size bound")
    work, work_spans, steps = "", [], []

    def step(op: str, a: Fraction | int, b: Fraction | int, text: str | None = None):
        nonlocal work
        a, b = Fraction(a), Fraction(b)
        result = operate(op, a, b)
        result_text = decimal_text(result) if text is None else text
        if Fraction(result_text) != result:
            raise ValueError("incorrect rendered primitive step")
        symbol = {"multiply": "*", "subtract": "-", "divide": "/", "power": "^", "add": "+"}[op]
        name = {"multiply": "MUL", "subtract": "SUB", "divide": "DIV", "power": "POW", "add": "ADD"}[op]
        work += (";" if work else "") + f"{name} {decimal_text(a)}{symbol}({decimal_text(b)})="
        work_spans.append({"start": len(work), "end": len(work) + len(result_text), "kind": "compute", "name": f"step{len(steps)}"})
        work += result_text
        steps.append({"op": op, "left": str(a), "right": str(b), "result": str(result)})
        return result

    if family == "bind_product":
        visible = re.findall(DECIMAL, question)
        if len(visible) != 2 or tuple(Fraction(v) for v in visible) != values:
            raise ValueError("ambiguous lexical operand extraction")
        answer = f"a={visible[0]};b={visible[1]}"
    elif family == "bind_system":
        answer = ";".join(f"{name}={value}" for name, value in zip(("a", "b", "r", "c", "d", "s"), values))
    elif family == "decimal_scale":
        value, k = values
        scale = step("power", 10, k)
        integer = step("multiply", value, scale)
        if integer.denominator != 1:
            raise ValueError("requested decimal scale cannot represent value exactly")
        answer = f"n={integer};s={scale}"
    elif family == "decimal_restore":
        answer = decimal_text(step("divide", *values))
    elif family in ("integer_multiply", "integer_subtract", "integer_divide"):
        op = {"integer_multiply": "multiply", "integer_subtract": "subtract", "integer_divide": "divide"}[family]
        answer_value = step(op, *values)
        if answer_value.denominator != 1:
            raise ValueError("integer lesson has fractional answer")
        answer = str(answer_value)
    elif family == "multiply_add":
        answer = str(step("add", step("multiply", values[0], values[1]), values[2]))
    else:
        # Explicit, short decimal conversion before the product: no implicit
        # digit stripping or silent sign/scale adjustment as in the first pilot.
        integers, scales = [], []
        for value in values:
            text = decimal_text(value)
            places = len(text.split(".")[1]) if "." in text else 0
            scale = step("power", 10, places)
            integers.append(step("multiply", value, scale))
            scales.append(scale)
        numerator = step("multiply", *integers)
        denominator = step("multiply", *scales)
        answer = decimal_text(step("divide", numerator, denominator))
        if Fraction(answer) != values[0] * values[1]:
            raise ValueError("independent product check failed")

    spans = []
    target = ""
    if arm == "compact_worked" and work:
        target = "<work>" + work + "</work>"
        spans = [dict(s, start=s["start"] + 6, end=s["end"] + 6) for s in work_spans]
    target += "<answer>"
    spans.append({"start": len(target), "end": len(target) + len(answer),
                  "kind": "copy" if family.startswith("bind_") or (arm == "compact_worked" and work) else "compute", "name": "answer"})
    target += answer + "</answer>"
    key = object_key(question)
    row = {"schema_version": VERSION, "source": "verified_primitive_synthetic",
           "family": family, "problem": question, "target_trace": target,
           "normalized_answer": answer, "supervision_spans": spans, "steps": steps,
           "arm": arm, "computation_key": key, "split": object_split(key),
           "difficulty": 2 if family in COMPOSITIONS else 1}
    row["record_id"] = fingerprint(row)
    return row


def validate_lesson(row: dict) -> None:
    if row != lesson(row["problem"], row["arm"]):
        raise ValueError("primitive lesson differs from public-question verification")


class PrimitiveCollator(ComputationCollator):
    def supervision_spans(self, row: dict) -> list[dict]:
        if row.get("schema_version") == VERSION:
            validate_lesson(row)
            return row["supervision_spans"]
        return super().supervision_spans(row)


def candidate(family: str, rng: random.Random) -> str:
    a, b = rng.randint(-99, 99), rng.randint(-9, 9)
    if family == "bind_product":
        a, b = fixed_decimal(rng.randint(-999, 999), rng.randrange(3)), fixed_decimal(rng.randint(-999, 999), rng.randrange(3))
        return f"Read operands only: Product of {a} and {b}. Return a and b."
    if family == "bind_system":
        a, b, r, c, d, s = [rng.randint(-20, 20) for _ in range(6)]
        return f"Read coefficients only: Solve the system {a}*x + ({b})*y = {r}; {c}*x + ({d})*y = {s}. Give x and y. Return a,b,r,c,d,s."
    if family == "decimal_scale":
        n, k = rng.randint(-999, 999), rng.randint(1, 2)
        return f"Write {fixed_decimal(n, k)} as n/10^k with k={k}. Return n and s=10^k."
    if family == "decimal_restore":
        return f"Write {rng.randint(-999, 999)}/{10 ** rng.randint(1, 2)} as a decimal."
    if family == "integer_multiply":
        return f"Calculate {a}*{b}."
    if family == "integer_subtract":
        return f"Calculate {a}-({rng.randint(-99, 99)})."
    if family == "integer_divide":
        divisor, result = rng.randint(1, 20), rng.randint(-20, 20)
        return f"Calculate {divisor * result}/({divisor}). Return an exact integer."
    if family == "multiply_add":
        return f"Calculate ({rng.randint(-12, 12)}*{b})+({rng.randint(-30, 30)})."
    if family == "decimal_multiply":
        return f"Calculate {fixed_decimal(a, 1)}*{fixed_decimal(rng.randint(-99, 99), 1)}."
    raise ValueError("unknown lesson family")


def make_corpus(train_count: int = 512, validation_count: int = 32) -> dict:
    if not 16 <= train_count <= 512 or not 8 <= validation_count <= 64:
        raise ValueError("primitive corpus exceeds bounded diagnostic support")
    rng, corpus = random.Random(20260826), {}
    for family in FOUNDATIONS + COMPOSITIONS:
        pools, seen = {"train": [], "validation": []}, set()
        for _ in range(100000):
            question = candidate(family, rng)
            row = lesson(question, "compact_worked")
            if row["family"] != family:
                continue
            key, split = row["computation_key"], row["split"]
            if key in seen:
                continue
            seen.add(key)
            cap = train_count if split == "train" else validation_count
            if len(pools[split]) < cap:
                pools[split].append(question)
            if len(pools["train"]) == train_count and len(pools["validation"]) == validation_count:
                break
        else:
            raise ValueError(f"insufficient primitive support: {family}")
        corpus[family] = pools
    train_keys = {object_key(q) for p in corpus.values() for q in p["train"]}
    val_keys = {object_key(q) for p in corpus.values() for q in p["validation"]}
    if train_keys & val_keys:
        raise ValueError("mathematical objects overlap between splits")
    return corpus
