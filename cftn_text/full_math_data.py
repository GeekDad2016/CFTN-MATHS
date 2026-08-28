"""Versioned full-math supervision repair; the sealed parent stays immutable.

Only public-question-verifiable procedures are labelled verified. Published
answers remain source-supervised, not magically certified by a file hash.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import random
import re

from .checkpoint import atomic_copy_file, atomic_json_dump
from .computation_supervision import ComputationCollator
from .data_generator import file_sha256
from .tokenizer import ByteMathTokenizer
from .v2_data import audit_v2_manifest, load_v2_records, normalize_problem
from .v2_school_data import (BANDS, FAMILIES, math_key, numerical_band,
                             parse_public, question, school_record, split_for)
from .verified_math_data import (audit_mathqa_program, fingerprint, operate,
                                 question_operands, independent_answer)

FORMAT = "cftn_full_math_supervision_v1"
PARENT_SHA = "a0a1cec180d5400faafe3e6794793b949dc08743ca9b2c7899a273a181ae21f0"
TRAIN_STYLES = (0, 1, 2, 4, 5, 6, 7)  # old held-out wording 3 remains held out


def full_question(family, values, style):
    if style < 4:
        return question(family, values, style)
    a, b = values[:2]
    if family == "linear_equation":
        c = values[2]
        return (f"Determine x: {a}*x + ({b}) = {c}.",
                f"Solve for x in {a}x + ({b}) = {c}.",
                f"The equation is {a}*x + ({b}) = {c}. Find x.",
                f"Recover x from {a}*x + ({b}) = {c}.",
                f"Which x satisfies {a}*x + ({b}) = {c}?")[style - 4]
    noun = {"addition": "sum", "subtraction": "difference",
            "multiplication": "product", "division": "quotient"}[family]
    symbol = {"addition": "+", "subtraction": "-", "multiplication": "*", "division": "/"}[family]
    return (f"Compute the {noun} of {a} and {b}.",
            f"Evaluate ({a}){symbol}({b}).", f"Return ({a}){symbol}({b}).",
            f"What is the result of ({a}){symbol}({b})?",
            f"Give the value of ({a}){symbol}({b}).")[style - 4]


def full_parse(text):
    try:
        return parse_public(text)
    except ValueError:
        pass
    n = r"(-?\d+)"
    for family in FAMILIES:
        if family == "linear_equation":
            patterns = (rf"Determine x: {n}\*x \+ \({n}\) = {n}\.",
                        rf"Solve for x in {n}x \+ \({n}\) = {n}\.",
                        rf"The equation is {n}\*x \+ \({n}\) = {n}\. Find x\.",
                        rf"Recover x from {n}\*x \+ \({n}\) = {n}\.",
                        rf"Which x satisfies {n}\*x \+ \({n}\) = {n}\?")
        else:
            noun = {"addition": "sum", "subtraction": "difference", "multiplication": "product", "division": "quotient"}[family]
            op = re.escape({"addition": "+", "subtraction": "-", "multiplication": "*", "division": "/"}[family])
            expression = rf"\({n}\){op}\({n}\)"
            patterns = (rf"Compute the {noun} of {n} and {n}\.", rf"Evaluate {expression}\.",
                        rf"Return {expression}\.", rf"What is the result of {expression}\?",
                        rf"Give the value of {expression}\.")
        for pattern in patterns:
            m = re.fullmatch(pattern, text)
            if m:
                return family, tuple(map(int, m.groups()))
    raise ValueError("unsupported full-school question")


def full_school_record(text):
    family, values = full_parse(text)
    base = school_record(question(family, values, 0))
    base.update(schema_version=FORMAT, source="verified_school_full", problem=text,
                verification="public_question_exact_procedure", supervision_kind="school")
    base.pop("record_id")
    base["record_id"] = fingerprint(base)
    return base


def generated_procedure(row):
    """Short, explicitly computed procedures for all six generated families.

    Bind from the PUBLIC text, never from hidden roles, then independently
    check the final equation/residual against the parent label.
    """
    text, family = row["problem"], row["family"]
    n, q = r"(-?\d+)", r"(-?\d+(?:/\d+)?)"
    target, spans = "<work>", []

    def step(name, op, a, b):
        nonlocal target
        a, b = Fraction(a), Fraction(b)
        result = operate(op, a, b)
        if len(spans):
            target += ";"
        symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op]
        target += f"{name}=({a}){symbol}({b})="
        spans.append({"start": len(target), "end": len(target) + len(str(result)), "kind": "compute", "name": name})
        target += str(result)
        return result

    def bind(patterns):
        for pattern in patterns:
            match = re.fullmatch(pattern, text)
            if match:
                return tuple(Fraction(v) for v in match.groups())
        raise ValueError("unrecognized generated public wording")

    if family == "variables_both_sides":
        a, b, c, d = bind((rf"Solve {n}\*x \+ \({n}\) = {n}\*x \+ \({n}\)\.",
                          rf"Find x when {n} times x plus {n} equals {n} times x plus {n}\."))
        coefficient = step("a", "subtract", a, c)
        rhs = step("b", "subtract", d, b)
        x = step("x", "divide", rhs, coefficient)
        if a * x + b != c * x + d:
            raise ValueError("linear residual")
        answer = str(x)
    elif family == "signed_fractions":
        a, b, c = bind((rf"Solve \({q}\)\*x \+ \({q}\) = {q}\.",
                       rf"A signed fraction {q} multiplies x; after adding {q}, the result is {q}\. Find x\."))
        rhs = step("b", "subtract", c, b)
        x = step("x", "divide", rhs, a)
        if a * x + b != c:
            raise ValueError("fraction residual")
        answer = str(x)
    elif family == "nested_parentheses":
        if text.startswith("Solve "):
            outer, inner, shift, offset, result = bind((rf"Solve {n}\*\({n}\*x \+ \({n}\)\) \+ \({n}\) = {n}\.",))
        else:
            result, outer, inner, shift, offset = bind((rf"Find x: {n} equals {n} times \({n} times x plus {n}\), then plus {n}\.",))
        value = step("u", "subtract", result, offset)
        value = step("v", "divide", value, outer)
        value = step("w", "subtract", value, shift)
        x = step("x", "divide", value, inner)
        if outer * (inner * x + shift) + offset != result:
            raise ValueError("nested residual")
        answer = str(x)
    elif family == "two_variable_systems":
        a, b, r, c, d, s = question_operands(row)
        ad, bc = step("ad", "multiply", a, d), step("bc", "multiply", b, c)
        det = step("det", "subtract", ad, bc)
        rd, bs = step("rd", "multiply", r, d), step("bs", "multiply", b, s)
        nx = step("nx", "subtract", rd, bs)
        ass, rc = step("as", "multiply", a, s), step("rc", "multiply", r, c)
        ny = step("ny", "subtract", ass, rc)
        x, y = step("x", "divide", nx, det), step("y", "divide", ny, det)
        answer = f"x={x};y={y}"
        if answer != independent_answer(row) or a*x+b*y != r or c*x+d*y != s:
            raise ValueError("system residual")
    elif family in {"multi_step_word_problem", "distractor_word_problem"}:
        # Strip only the known, irrelevant wrapper; no generic number guessing.
        for prefix in ("The notebooks have blue covers, which does not affect their count. ",
                       "A delivery van travelled 18 kilometres; that fact is irrelevant. ",
                       "The manager drank two cups of tea before counting. "):
            if text.startswith(prefix):
                text = text[len(prefix):]
        text = text.removesuffix(" Ignore any decorative details.")
        if "notebooks and receives" in text:
            start, groups, per = bind((rf"A store starts with {n} notebooks and receives {n} boxes with {n} notebooks in each box\. How many notebooks are there\?",))
            removed, recipients = None, None
        elif "then donates" in text:
            start, groups, per, removed = bind((rf"A store starts with {n} notebooks, receives {n} boxes of {n}, then donates {n}\. How many remain\?",))
            recipients = None
        else:
            start, groups, per, removed, recipients = bind((rf"A store starts with {n} notebooks, receives {n} boxes of {n}, donates {n}, and shares the rest equally among {n} classrooms\. How many notebooks does each classroom get\?",))
        value = step("received", "multiply", groups, per)
        value = step("total", "add", start, value)
        if removed is not None:
            value = step("remain", "subtract", value, removed)
        if recipients is not None:
            value = step("share", "divide", value, recipients)
        expected = (start + groups * per - (removed or 0)) / (recipients or 1)
        if value != expected:
            raise ValueError("word problem residual")
        answer = str(value)
    else:
        raise ValueError("unknown generated family")
    if answer != row["normalized_answer"]:
        raise ValueError("public question disagrees with stored answer")
    target += "</work><answer>"
    spans.append({"start": len(target), "end": len(target) + len(answer), "kind": "copy", "name": "answer"})
    return target + answer + "</answer>", spans


def repair_parent(row):
    if row["source"] == "mathqa":
        return None, audit_mathqa_program(row)
    out = dict(row)
    out.update(schema_version=FORMAT, parent_record_id=row["record_id"], supervision_kind="published")
    if row["source"] == "cftn_generated":
        out["target_trace"], out["supervision_spans"] = generated_procedure(row)
        out.update(verification="public_question_exact_procedure", supervision_kind="generated")
    else:
        # No synthetic reasoning is fabricated for unsupported imported tasks.
        target = out["target_trace"]
        begin = target.rfind("<answer>") + len("<answer>")
        if begin < len("<answer>") or not target.endswith("</answer>"):
            raise ValueError("missing published answer")
        out["supervision_spans"] = [{"start": begin, "end": len(target) - len("</answer>"),
                                     "kind": "compute", "name": "source_supervised_answer"}]
        out["verification"] = "published_target_not_independently_certified"
    out.pop("record_id")
    out["record_id"] = fingerprint(out)
    return out, None


def check_row(row):
    if fingerprint({k: v for k, v in row.items() if k != "record_id"}) != row["record_id"]:
        raise ValueError("full-supervision record hash mismatch")
    if row.get("supervision_kind") == "school":
        if row != full_school_record(row["problem"]):
            raise ValueError("school public-question verification failed")
    elif row.get("supervision_kind") == "generated":
        target, spans = generated_procedure(row)
        if target != row["target_trace"] or spans != row["supervision_spans"]:
            raise ValueError("generated procedure mismatch")
    elif row.get("supervision_kind") != "published":
        raise ValueError("unknown full-supervision record")


class FullMathCollator(ComputationCollator):
    def supervision_spans(self, row):
        # Manifest audit checks every record before training; cheap digest here
        # also protects accidental runtime mutation without re-solving batches.
        if row.get("schema_version") == FORMAT:
            if fingerprint({k: v for k, v in row.items() if k != "record_id"}) != row["record_id"]:
                raise ValueError("full-supervision record changed after audit")
            return row["supervision_spans"]
        return super().supervision_spans(row)


def school_rows(blocked_keys, blocked_questions, objects_per_family=12000):
    rng = random.Random(826410)
    train, validation, wording = [], [], []
    for band_index, band in enumerate(BANDS):
        for family in FAMILIES:
            limit = (15, 99, 999)[band_index]
            seen, counts = set(), Counter()
            if band_index == 0 and family != "linear_equation":
                candidates = [(a*b, b) if family == "division" else (a, b)
                              for a in range(-15, 16) for b in range(-15, 16)
                              if family != "division" or b]
                rng.shuffle(candidates)
            else:
                candidates = []
                for _ in range(objects_per_family * 12):
                    if family == "linear_equation":
                        ca, cx, cb = ((8, 20, 50), (16, 50, 125), (32, 100, 250))[band_index]
                        a = rng.choice([v for v in range(-ca, ca + 1) if v])
                        x, b = rng.randint(-cx, cx), rng.randint(-cb, cb)
                        values = (a, b, a*x+b)
                    elif family == "division":
                        b = rng.choice([v for v in range(-limit, limit + 1) if v])
                        values = (b * rng.randint(-limit, limit), b)
                    else:
                        values = (rng.randint(-limit, limit), rng.randint(-limit, limit))
                    candidates.append(values)
            for values in candidates:
                if numerical_band(family, values) != band:
                    continue
                key = math_key(family, values)
                if key in seen or key in blocked_keys:
                    continue
                seen.add(key)
                split = split_for(key)
                if counts[split] >= (objects_per_family if split == "train" else 64):
                    continue
                text = full_question(family, values, rng.choice(TRAIN_STYLES))
                if normalize_problem(text).casefold() in blocked_questions:
                    continue
                row = full_school_record(text)
                (train if split == "train" else validation).append(row)
                if split != "train":
                    wording.append(full_school_record(full_question(family, values, 3 if counts[split] % 2 else 8)))
                counts[split] += 1
                if counts["train"] >= objects_per_family and counts["validation"] >= 64:
                    break
            if counts["validation"] != 64 or counts["train"] < min(objects_per_family, 200):
                raise ValueError(f"insufficient full school coverage: {band}/{family}: {counts}")
    return train, validation, wording


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": Path(path).name, "sha256": file_sha256(path), "count": len(rows)}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_full_data(
    parent_root,
    output,
    *,
    objects_per_family=12000,
    expected_parent_manifest_sha256=PARENT_SHA,
):
    parent_root, output = Path(parent_root), Path(output)
    parent = json.loads((parent_root / "manifest.json").read_text())
    if parent["manifest_sha256"] != expected_parent_manifest_sha256:
        raise ValueError("unexpected full-data parent")
    audit = audit_v2_manifest(parent, parent_root)
    output.mkdir(parents=True, exist_ok=False)
    splits, blocked, questions = {}, set(), set()
    for name, metadata in parent["splits"].items():
        if name == "train":
            continue
        rows = load_v2_records(parent_root / metadata["path"])
        for row in rows:
            questions.add(normalize_problem(row["problem"]).casefold())
            try:
                blocked.add(math_key(*full_parse(row["problem"])))
            except ValueError:
                pass
        destination = output / (name + ".jsonl")
        atomic_copy_file(parent_root / metadata["path"], destination)
        splits[name] = dict(metadata, path=destination.name)
    repaired, ledger, rejected = [], [], []
    for row in load_v2_records(parent_root / parent["splits"]["train"]["path"]):
        fixed, reason = repair_parent(row)
        if fixed is None:
            ledger.append(reason)
            rejected.append(row)
        else:
            # Prevent supported old math objects entering a synthetic holdout.
            try:
                blocked.add(math_key(*full_parse(row["problem"])))
            except ValueError:
                pass
            repaired.append(fixed)
    added, validation, wording = school_rows(blocked, questions, objects_per_family)
    repaired.extend(added)
    tokenizer = ByteMathTokenizer()
    max_length = 0
    for row in repaired + validation + wording:
        check_row(row)
        max_length = max(max_length, len(tokenizer.encode_training_example(row["problem"], row["target_trace"], 4096).input_ids))
    splits["train"] = write_rows(output / "train.jsonl", repaired)
    splits["school_validation"] = write_rows(output / "school_validation.jsonl", validation)
    splits["school_wording"] = write_rows(output / "school_wording.jsonl", wording)
    quarantine = write_rows(output / "quarantine_mathqa.jsonl", rejected)
    decisions = write_rows(output / "quarantine_decisions.jsonl", ledger)
    manifest = {"format": "cftn_text_broad_math_v2", "derivative_format": FORMAT,
                "parent_root": str(parent_root.resolve()), "parent_manifest_sha256": parent["manifest_sha256"],
                "parent_manifest_file_sha256": file_sha256(parent_root / "manifest.json"),
                "parent_audit": audit, "splits": splits, "train_records": len(repaired),
                "generator_sha256": file_sha256(__file__),
                "dependency_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in
                                      ("v2_school_data.py", "verified_math_data.py", "computation_supervision.py")},
                "generation_settings": {"seed": 826410, "objects_per_family": objects_per_family,
                                        "training_wordings": list(TRAIN_STYLES), "heldout_wordings": [3, 8]},
                "training_sources": dict(Counter(r["source"] for r in repaired)),
                "training_families": dict(Counter(r["family"] for r in repaired)),
                "supervision_counts": dict(Counter(r["verification"] for r in repaired)),
                "school_unique_objects": len({r["computation_key"] for r in added}),
                "quarantine": quarantine, "quarantine_decisions": decisions,
                "quarantine_reasons": dict(Counter(r["status"] for r in ledger)),
                "maximum_encoded_length": max_length,
                "original_evaluation_unchanged": True, "all_imported_answers_verified": False}
    manifest["manifest_sha256"] = fingerprint(manifest)
    atomic_json_dump(manifest, output / "manifest.json")
    return audit_full_data(output)


def audit_full_data(root, *, expected_parent_manifest_sha256=PARENT_SHA):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("derivative_format") != FORMAT or fingerprint({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("full-data manifest signature mismatch")
    if manifest["generator_sha256"] != file_sha256(__file__):
        raise ValueError("full-data generator source changed")
    for name, digest in manifest["dependency_sha256"].items():
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("full-data generator dependency changed")
    parent_root = Path(manifest["parent_root"])
    parent = json.loads((parent_root / "manifest.json").read_text())
    recorded_parent_sha256 = manifest.get("parent_manifest_sha256")
    if (
        recorded_parent_sha256 != expected_parent_manifest_sha256
        or parent["manifest_sha256"] != recorded_parent_sha256
        or file_sha256(parent_root / "manifest.json") != manifest["parent_manifest_file_sha256"]
    ):
        raise ValueError("sealed parent changed")
    audit_v2_manifest(parent, parent_root)
    train_keys, val_keys, train_questions, eval_questions = set(), set(), set(), set()
    for name, meta in manifest["splits"].items():
        path = (root / meta["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path) != meta["sha256"]:
            raise ValueError("full-data split hash/path mismatch")
        rows = read_rows(path)
        if len(rows) != meta["count"]:
            raise ValueError("full-data split count mismatch")
        if name in parent["splits"] and name != "train" and meta["sha256"] != parent["splits"][name]["sha256"]:
            raise ValueError("original evaluation was altered")
        for row in rows:
            (train_questions if name == "train" else eval_questions).add(normalize_problem(row["problem"]).casefold())
            if row.get("schema_version") == FORMAT:
                check_row(row)
                ByteMathTokenizer().encode_training_example(row["problem"], row["target_trace"], 4096)
                if row.get("computation_key"):
                    (train_keys if name == "train" else val_keys).add(row["computation_key"])
    if train_keys & val_keys or train_questions & eval_questions:
        raise ValueError("full-data train/evaluation overlap")
    for key in ("quarantine", "quarantine_decisions"):
        meta = manifest[key]
        if file_sha256(root / meta["path"]) != meta["sha256"] or len(read_rows(root / meta["path"])) != meta["count"]:
            raise ValueError("quarantine evidence changed")
    return manifest
