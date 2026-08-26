"""Verified V1-style worked mathematics with explicit numerical curriculum.

New versioned training corpus: never rewrites or re-signs the sealed V2 data.
The public question alone determines every intermediate and final target.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
import random
import re

from .computation_supervision import ComputationCollator
from .data_generator import canonical_trace, render_problem
from .verified_math_data import fingerprint, operate

VERSION = "cftn_v2_verified_school_v1"
FAMILIES = ("addition", "subtraction", "multiplication", "division", "linear_equation")
BANDS = ("foundations", "two_digit", "three_digit")
V1_TEMPLATES = ("symbolic_standard", "symbolic_reversed", "verbal_add", "heldout_balance")


def question(family: str, values: tuple[int, ...], style: int) -> str:
    if style not in range(4):
        raise ValueError("unknown school wording")
    if family == "linear_equation":
        return render_problem(*values, V1_TEMPLATES[style])
    a, b = values
    options = {
        "addition": (f"Calculate {a}+({b}).", f"What is {a} plus {b}?", f"Add {a} and {b}.", f"Find the sum of {a} and {b}."),
        "subtraction": (f"Calculate {a}-({b}).", f"What is {a} minus {b}?", f"Subtract {b} from {a}.", f"Find the difference {a} minus {b}."),
        "multiplication": (f"Calculate {a}*{b}.", f"What is {a} times {b}?", f"Product of {a} and {b}.", f"Multiply {a} and {b}."),
        "division": (f"Calculate {a}/({b}).", f"What is {a} divided by {b}?", f"Divide {a} by {b}.", f"Find the quotient of {a} and {b}."),
    }
    return options[family][style]


def parse_public(text: str) -> tuple[str, tuple[int, ...]]:
    n = r"(-?\d+)"
    patterns = {
        "addition": (rf"Calculate {n}\+\({n}\)\.", rf"What is {n} plus {n}\?", rf"Add {n} and {n}\.", rf"Find the sum of {n} and {n}\."),
        "subtraction": (rf"Calculate {n}-\({n}\)\.", rf"What is {n} minus {n}\?", rf"Subtract {n} from {n}\.", rf"Find the difference {n} minus {n}\."),
        "multiplication": (rf"Calculate {n}\*{n}\.", rf"What is {n} times {n}\?", rf"Product of {n} and {n}\.", rf"Multiply {n} and {n}\."),
        "division": (rf"Calculate {n}/\({n}\)\.", rf"What is {n} divided by {n}\?", rf"Divide {n} by {n}\.", rf"Find the quotient of {n} and {n}\."),
        "linear_equation": (rf"Solve {n}\*x \+ \({n}\) = {n}\.", rf"Find x if {n} = \({n}\) \+ {n}\*x\.",
                            rf"Add {n} to {n} times an integer\. The result is {n}\. What is the integer\?",
                            rf"A balance states that {n} multiplied by an unknown, plus {n}, has the same value as {n}\. What is the unknown\?"),
    }
    if len(text) > 1000:
        raise ValueError("question too long")
    for family, variants in patterns.items():
        for style, pattern in enumerate(variants):
            match = re.fullmatch(pattern, text)
            if match:
                values = tuple(int(v) for v in match.groups())
                if max(map(abs, values)) > 10**8:
                    raise ValueError("school operand outside support")
                if family == "subtraction" and style == 2:
                    values = values[::-1]
                if family == "linear_equation" and style == 1:
                    c, b, a = values
                    values = (a, b, c)
                elif family == "linear_equation" and style == 2:
                    b, a, c = values
                    values = (a, b, c)
                return family, values
    raise ValueError("unsupported public school question")


def solve(family, values):
    if family not in FAMILIES or len(values) != (3 if family == "linear_equation" else 2):
        raise ValueError("unknown school operation or operand count")
    a, b = values[:2]
    if family == "linear_equation":
        if not a:
            raise ValueError("zero linear coefficient")
        result = Fraction(values[2] - b, a)
    elif family == "division":
        if not b:
            raise ValueError("zero divisor")
        result = Fraction(a, b)
    else:
        result = Fraction(a + b if family == "addition" else a - b if family == "subtraction" else a * b)
    if result.denominator != 1:
        raise ValueError("integer school phase requires exact integer solutions")
    return int(result)


def numerical_band(family, values):
    a, b = values[:2]
    if family == "linear_equation":
        x = solve(family, values)
        for index, (ca, cx, cb) in enumerate(((8, 20, 50), (16, 50, 125), (32, 100, 250))):
            if abs(a) <= ca and abs(x) <= cx and abs(b) <= cb:
                return BANDS[index]
    else:
        size = max(abs(b), abs(solve(family, values)) if family == "division" else abs(a))
        for index, limit in enumerate((15, 99, 999)):
            if size <= limit:
                return BANDS[index]
    raise ValueError("question outside declared numerical bands")


def math_key(family, values):
    if family in ("addition", "multiplication"):
        values = tuple(sorted(values))
    elif family == "linear_equation":
        # Scalar multiples of the same equation must not cross splits.
        divisor = gcd(gcd(abs(values[0]), abs(values[1])), abs(values[2]))
        if not divisor or not values[0]:
            raise ValueError("invalid linear equation")
        divisor *= -1 if values[0] < 0 else 1
        values = tuple(v // divisor for v in values)
    return fingerprint([family, values])


def split_for(key):
    return "validation" if int(key[:12], 16) % 5 == 0 else "train"


def school_record(text: str) -> dict:
    family, values = parse_public(text)
    answer = solve(family, values)
    a, b = values[:2]
    work, spans, steps = "<work>", [], []

    def computed(prefix, result, op, left, right):
        nonlocal work
        if operate(op, Fraction(left), Fraction(right)) != result:
            raise ValueError("incorrect school operation")
        work += prefix
        spans.append({"start": len(work), "end": len(work) + len(str(result)), "kind": "compute", "name": str(len(steps))})
        work += str(result)
        steps.append({"operation": op, "left": left, "right": right, "result": result})

    if family == "linear_equation":
        c = values[2]
        work += f"{a}*x+({b})={c};SUB({b});"
        computed(f"{a}*x=", c - b, "subtract", c, b)
        computed(f";DIV({a});x=", answer, "divide", c - b, a)
    elif family == "multiplication" and abs(a) >= 10:
        # Signed decimal-place decomposition: teach each contribution before
        # adding it. No jump from two operands straight to a copied final value.
        tens = (abs(a) // 10 * 10) * (-1 if a < 0 else 1)
        units = a - tens
        work += f"{a}*({b});SPLIT({a});{a}=("
        for name, value, suffix in (("tens", tens, ")+("), ("units", units, f");MUL({b});")):
            spans.append({"start": len(work), "end": len(work) + len(str(value)), "kind": "compute", "name": name})
            work += str(value) + suffix
        if tens + units != a or abs(units) > 9 or tens % 10:
            raise ValueError("invalid decimal-place decomposition")
        steps.append({"operation": "split", "input": a, "tens": tens, "units": units})
        computed(f"({tens})*({b})=", tens * b, "multiply", tens, b)
        computed(f";({units})*({b})=", units * b, "multiply", units, b)
        computed(f";ADD({units * b});{tens * b}+({units * b})=", answer, "add", tens * b, units * b)
    else:
        op = {"addition": "add", "subtraction": "subtract", "multiplication": "multiply", "division": "divide"}[family]
        symbol = {"addition": "+", "subtraction": "-", "multiplication": "*", "division": "/"}[family]
        name = {"addition": "ADD", "subtraction": "SUB", "multiplication": "MUL", "division": "DIV"}[family]
        work += f"{a}{symbol}({b});{name}({b});"
        computed("value=", answer, op, a, b)
    work += "</work><answer>"
    spans.append({"start": len(work), "end": len(work) + len(str(answer)), "kind": "copy", "name": "answer"})
    target = work + str(answer) + "</answer>"
    if family == "linear_equation" and target != canonical_trace(a, b, values[2], answer):
        raise ValueError("linear trace diverges from the proven V1 recipe")
    key = math_key(family, values)
    row = {"schema_version": VERSION, "source": "verified_school", "family": family,
           "problem": text, "normalized_answer": str(answer), "target_trace": target,
           "supervision_spans": spans, "steps": steps, "computation_key": key,
           "split": split_for(key), "band": numerical_band(family, values),
           "difficulty": BANDS.index(numerical_band(family, values)) + 1}
    row["record_id"] = fingerprint(row)
    return row


class SchoolCollator(ComputationCollator):
    def supervision_spans(self, row):
        if row.get("schema_version") == VERSION:
            if row != school_record(row["problem"]):
                raise ValueError("school record fails public-question verification")
            return row["supervision_spans"]
        return super().supervision_spans(row)


def build_school_corpus():
    rng, result = random.Random(719), {}
    for band_index, band in enumerate(BANDS):
        result[band] = {}
        for family in FAMILIES:
            seen, pools = set(), {"train": [], "validation": []}
            limit = (15, 99, 999)[band_index]
            # Elementary arithmetic support is finite. Enumerate it completely
            # before choosing a held-out subset; sample larger supports.
            if band_index == 0 and family != "linear_equation":
                candidates = [(a * b, b) if family == "division" else (a, b)
                              for a in range(-15, 16) for b in (range(1, 16) if family == "division" else range(-15, 16))]
                rng.shuffle(candidates)
            else:
                candidates = []
                for _ in range(15000):
                    if family == "linear_equation":
                        ca, cx, cb = ((8, 20, 50), (16, 50, 125), (32, 100, 250))[band_index]
                        a = rng.choice([v for v in range(-ca, ca + 1) if v])
                        x, b = rng.randint(-cx, cx), rng.randint(-cb, cb)
                        candidates.append((a, b, a * x + b))
                    elif family == "division":
                        b, answer = rng.randint(1, limit), rng.randint(-limit, limit)
                        candidates.append((b * answer, b))
                    else:
                        candidates.append((rng.randint(-limit, limit), rng.randint(-limit, limit)))
            for values in candidates:
                if numerical_band(family, values) != band:
                    continue
                key = math_key(family, values)
                if key in seen:
                    continue
                seen.add(key)
                split = split_for(key)
                if len(pools[split]) < (2048 if split == "train" else 64):
                    style = rng.randrange(3)
                    pools[split].append(school_record(question(family, values, style)))
            if len(pools["validation"]) != 64 or len(pools["train"]) < 250:
                raise ValueError(f"insufficient school support: {band}/{family}")
            result[band][family] = pools
    keys = {split: {r["computation_key"] for bands in result.values() for pools in bands.values() for r in pools[split]}
            for split in ("train", "validation")}
    if keys["train"] & keys["validation"]:
        raise ValueError("school mathematical-object split leakage")
    return result
