"""Opt-in, versioned procedural supervision; never rewrites the sealed V2 corpus.

The solver/verifier is an OFFLINE data-building tool, not an inference fallback.
Only two explicit public-question grammars are supported initially. Unsupported
or contradictory rows fail closed, rather than being silently called verified.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


VERSION = "cftn_verified_procedure_v1"
TARGET_FAMILIES = ("two_variable_systems", "arithmetic__mul")
REPLAY_FAMILIES = ("variables_both_sides", "nested_parentheses")
NUMBER = r"-?\d+(?:\.\d+)?(?:/\d+)?"
DECIMAL = r"-?\d+(?:\.\d+)?"
GROUPS = {"format": 0, "compute": 1, "copy": 2}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def bounded(value: Fraction | int) -> Fraction:
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
        raise ValueError("arithmetic exceeds verifier size bound")
    return value


def operate(op: str, left: Fraction, right: Fraction) -> Fraction:
    left, right = bounded(left), bounded(right)
    if op == "add":
        value = left + right
    elif op == "subtract":
        value = left - right
    elif op == "multiply":
        value = left * right
    elif op == "divide":
        if right == 0:
            raise ValueError("division by zero")
        value = left / right
    elif op == "power" and right.denominator == 1 and abs(right) <= 12:
        if left == 0 and right < 0:
            raise ValueError("negative power of zero")
        value = left ** int(right)
    else:
        raise ValueError("unsupported operation or exponent")
    return bounded(value)


def decimal_text(value: Fraction) -> str:
    """Exact finite-decimal rendering, without a float/rounding conversion."""
    value = bounded(value)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return str(value)
    places = max(twos, fives)
    scaled = abs(value.numerator) * (10 ** places // value.denominator)
    digits = str(scaled).zfill(places + 1)
    text = digits if not places else (digits[:-places] + "." + digits[-places:]).rstrip("0").rstrip(".")
    return ("-" if value < 0 else "") + text


def question_operands(row: dict) -> tuple[Fraction, ...]:
    """Bind ONLY visible question text, not hidden x/y or metadata fields."""
    question, family = row["problem"], row["family"]
    if len(question) > 2000:
        raise ValueError("question exceeds parser bound")
    if family == "two_variable_systems":
        patterns = (
            r"Solve the system (-?\d+)\*x \+ \((-?\d+)\)\*y = (-?\d+); (-?\d+)\*x \+ \((-?\d+)\)\*y = (-?\d+)\. Give x and y\.",
            r"Find both unknowns: \((-?\d+)\)x\+\((-?\d+)\)y=(-?\d+), and \((-?\d+)\)x\+\((-?\d+)\)y=(-?\d+)\.",
        )
    elif family == "arithmetic__mul":
        a = f"({DECIMAL})"
        patterns = (
            rf"(?:Calculate |Work out |What is )?{a}\s*\*\s*{a}[.?]?",
            rf"(?:What is )?{a} times {a}[.?]?",
            rf"(?:What is the product|Product) of {a} and {a}[.?]?",
            rf"Multiply {a} and {a}\.",
        )
    else:
        raise ValueError("family has no verified procedure")
    for pattern in patterns:
        match = re.fullmatch(pattern, question)
        if match:
            return tuple(bounded(Fraction(v)) for v in match.groups())
    raise ValueError("unsupported public-question grammar")


def independent_answer(row: dict) -> str:
    """Direct product / Gaussian elimination, separate from trace construction."""
    values = question_operands(row)
    if row["family"] == "arithmetic__mul":
        return decimal_text(values[0] * values[1])
    a, b, r, c, d, s = values
    if not a:
        a, b, r, c, d, s = c, d, s, a, b, r
    if not a:
        raise ValueError("singular system")
    denominator = d - c * b / a
    if not denominator:
        raise ValueError("singular system")
    y = (s - c * r / a) / denominator
    x = (r - b * y) / a
    if a * x + b * y != r or c * x + d * y != s:
        raise ValueError("independent residual check failed")
    return f"x={x};y={y}"


@dataclass(frozen=True)
class Step:
    name: str
    op: str
    left: str
    right: str
    result: str

    def check(self) -> None:
        if operate(self.op, Fraction(self.left), Fraction(self.right)) != Fraction(self.result):
            raise ValueError(f"incorrect arithmetic at {self.name}")


def procedure(row: dict) -> tuple[list[Step], str]:
    values = question_operands(row)
    steps: list[Step] = []

    def add(name: str, op: str, a: Fraction | int, b: Fraction | int) -> Fraction:
        result = operate(op, Fraction(a), Fraction(b))
        steps.append(Step(name, op, str(a), str(b), str(result)))
        return result

    if row["family"] == "two_variable_systems":
        a, b, r, c, d, s = values
        ad, bc = add("ad", "multiply", a, d), add("bc", "multiply", b, c)
        det = add("det", "subtract", ad, bc)
        rd, bs = add("rd", "multiply", r, d), add("bs", "multiply", b, s)
        nx = add("nx", "subtract", rd, bs)
        ass, rc = add("as", "multiply", a, s), add("rc", "multiply", r, c)
        ny = add("ny", "subtract", ass, rc)
        x, y = add("x", "divide", nx, det), add("y", "divide", ny, det)
        ax, by = add("ax", "multiply", a, x), add("by", "multiply", b, y)
        lhs1 = add("lhs1", "add", ax, by)
        res1 = add("r1", "subtract", lhs1, r)
        cx, dy = add("cx", "multiply", c, x), add("dy", "multiply", d, y)
        lhs2 = add("lhs2", "add", cx, dy)
        res2 = add("r2", "subtract", lhs2, s)
        if res1 != 0 or res2 != 0:
            raise ValueError("trace residual check failed")
        answer = f"x={x};y={y}"
    else:
        a, b = values
        # Finite decimal -> integer operands, distributive partial products,
        # sign, decimal rescaling. No rounded arithmetic is used.
        sa, sb = decimal_text(abs(a)), decimal_text(abs(b))
        places = sum(len(v.split(".")[1]) if "." in v else 0 for v in (sa, sb))
        ia, ib = int(sa.replace(".", "")), int(sb.replace(".", ""))
        if max(len(str(ia)), len(str(ib))) > 12:
            raise ValueError("multiplication exceeds twelve-digit pilot support")
        total = Fraction(0)
        for index, digit in enumerate(reversed(str(ib))):
            partial = add(f"p{index}", "multiply", ia, int(digit))
            shifted = add(f"q{index}", "multiply", partial, 10 ** index)
            total = add(f"s{index}", "add", total, shifted)
        signed = add("signed", "multiply", total, -1 if (a < 0) != (b < 0) else 1)
        result = add("value", "divide", signed, 10 ** places)
        answer = decimal_text(result)
    if answer != independent_answer(row) or answer != str(row["normalized_answer"]):
        raise ValueError("procedure/public question/final label disagree")
    return steps, answer


def _span(start: int, end: int, kind: str, name: str) -> dict:
    return {"start": start, "end": end, "kind": kind, "name": name}


def render(steps: list[Step], answer: str) -> tuple[str, list[dict]]:
    target = "<work>"
    spans = []
    for index, step in enumerate(steps):
        step.check()
        if index:
            target += ";"
        target += f"{step.name}={step.op}({step.left},{step.right})="
        spans.append(_span(len(target), len(target) + len(step.result), "compute", step.name))
        target += step.result
    target += "</work><answer>"
    spans.append(_span(len(target), len(target) + len(answer), "copy", "answer"))
    return target + answer + "</answer>", spans


def verified_record(row: dict) -> dict:
    steps, answer = procedure(row)
    target, spans = render(steps, answer)
    payload = {
        "schema_version": VERSION, "parent_record_id": row["record_id"],
        "parent_content_id": row["content_id"], "source": row["source"],
        "family": row["family"], "split": row["split"],
        "difficulty": row["difficulty"], "problem": row["problem"],
        "normalized_answer": answer, "target_answer": f"<answer>{answer}</answer>",
        "target_trace": target, "supervision_spans": spans,
        "steps": [step.__dict__ for step in steps],
        "verification": "exact_steps_and_independent_public_question_solution",
    }
    payload["record_id"] = fingerprint(payload)
    return payload


def validate_verified_record(row: dict) -> None:
    if row.get("schema_version") != VERSION:
        raise ValueError("not a versioned verified record")
    payload = {k: v for k, v in row.items() if k != "record_id"}
    if fingerprint(payload) != row.get("record_id"):
        raise ValueError("verified record digest mismatch")
    steps, answer = procedure(row)
    target, spans = render(steps, answer)
    if (target != row["target_trace"] or spans != row["supervision_spans"]
            or row["steps"] != [s.__dict__ for s in steps]
            or row["target_answer"] != f"<answer>{answer}</answer>"):
        raise ValueError("verified procedure content mismatch")


def legacy_spans(row: dict) -> list[dict]:
    """Result-only focus for known legacy traces; never guesses MathQA steps."""
    target = row["target_trace"]
    if "<program>" in target:
        raise ValueError("unverified program is not eligible for numeric supervision")
    match = re.fullmatch(r"(?:<work>(.*?)</work>)?<answer>(.*?)</answer>", target)
    if not match:
        raise ValueError("unsupported legacy target")
    spans = []
    if match[1] is not None:
        work = match[1]
        # Only the final numeric RHS of each equality/assignment is focused.
        # Operands and repeated final answers remain low-weight copy tokens.
        for value in re.finditer(rf"=({NUMBER})(?=;|$)", work):
            spans.append(_span(6 + value.start(1), 6 + value.end(1), "compute", "legacy_result"))
    kind = "compute" if match[1] is None else "copy"
    spans.append(_span(match.start(2), match.end(2), kind, "answer"))
    if not any(s["kind"] == "compute" for s in spans):
        raise ValueError("no verified computation span in legacy target")
    return spans


def computation_key(row: dict) -> str:
    if row["family"] in TARGET_FAMILIES:
        values = question_operands(row)
        if row["family"] == "arithmetic__mul":
            values = tuple(sorted(values))  # a*b and b*a cannot cross splits
        return fingerprint([row["family"], *map(str, values)])
    return row["content_id"]


def curriculum_band(row: dict) -> str:
    values = question_operands(row)
    if row["family"] == "arithmetic__mul":
        lengths = [len(decimal_text(abs(v)).replace(".", "").lstrip("0")) for v in values]
        return "foundation" if max(lengths) <= 3 else "expanded"
    a, b, _, c, d, _ = values
    return "foundation" if max(abs(v) for v in (a, b, c, d)) <= 6 else "expanded"


def audit_mathqa_program(row: dict) -> dict:
    """Conservative consistency triage, NOT certification of question semantics.

    Unsupported operations, ambiguous number binding, approximation and units
    stay out of the verified-program pool. A mismatch is a review flag, not an
    automatically corrected answer. Even exact matches still require review.
    """
    result = {"record_id": row["record_id"], "training_eligible": False,
              "semantic_verification": False}
    try:
        target = row["target_trace"]
        match = re.fullmatch(r"<program>(.*?)</program><answer>(.*?)</answer>", target)
        if not match or match[1] != row["native_program"] or match[2] != row["normalized_answer"]:
            raise ValueError("program/target serialization mismatch")
        raw = row["raw_problem"]
        # These forms need a dataset-specific number binder, not a guessed one.
        if re.search(r"\d\s*/\s*\d|\d\s*:\s*\d|\d\s+[0-9]+/|\d[eE][+-]?\d", raw):
            raise ValueError("ambiguous question number binding")
        numeric = re.findall(r"-?\d[\d,]*(?:\.\d+)?", raw)
        numbers = [bounded(Fraction(n.replace(",", ""))) for n in numeric]
        values: list[Fraction] = []
        operations = match[1].split("|")
        if not 1 <= len(operations) <= 128:
            raise ValueError("program length bound")

        def argument(text: str) -> Fraction:
            if re.fullmatch(r"n\d+", text):
                return numbers[int(text[1:])]
            if re.fullmatch(r"#\d+", text):
                return values[int(text[1:])]
            if re.fullmatch(r"const_\d+(?:_\d+)?", text):
                return bounded(Fraction(text[6:].replace("_", ".")))
            raise ValueError("unsupported program argument")

        for step in operations:
            op = re.fullmatch(r"([a-z]+)\(([^(),]+),([^(),]+)\)", step.strip())
            if not op:
                raise ValueError("unsupported program step")
            values.append(operate(op[1], argument(op[2]), argument(op[3])))
        if not re.fullmatch(NUMBER, row["normalized_answer"]):
            raise ValueError("answer needs explicit units/rounding policy")
        expected = Fraction(row["normalized_answer"])
        result.update(program_value=str(values[-1]), expected=str(expected))
        if values[-1] != expected:
            return dict(result, status="quarantine_program_answer_mismatch")
        return dict(result, status="internally_consistent_needs_semantic_review")
    except (ValueError, IndexError, ZeroDivisionError) as exc:
        return dict(result, status="quarantine_unsupported_or_ambiguous", reason=str(exc))
