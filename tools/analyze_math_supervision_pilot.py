"""Locate the first procedural error in saved greedy outputs; no model calls."""
from __future__ import annotations

import argparse
import collections
from fractions import Fraction
import json
from pathlib import Path
import re

from cftn_text.verified_math_data import TARGET_FAMILIES, Step, procedure


def first_procedure_error(row: dict, text: str) -> dict:
    expected, _ = procedure(row)
    result = {"correct_prefix_steps": 0, "expected_steps": len(expected)}
    if not text.startswith("<work>"):
        return dict(result, error="missing_work")
    work = text[6:].split("</work>", 1)[0]
    generated_steps = work.split(";")
    for index, want in enumerate(expected):
        if index >= len(generated_steps):
            return dict(result, error="missing_step", first_step=want.name)
        raw = generated_steps[index]
        # Without a closing delimiter, "...=4" could be a truncated "...=48".
        # Do not count that unfinished final field as a demonstrated arithmetic error.
        if index == len(generated_steps) - 1 and "</work>" not in text and not work.endswith(";"):
            return dict(result, error="malformed_or_incomplete_step", first_step=want.name, observed=raw[:160])
        match = re.fullmatch(r"([a-z][a-z0-9]*)=([a-z]+)\(([^,()]+),([^,()]+)\)=([^;<>]+)", raw)
        if not match:
            return dict(result, error="malformed_or_incomplete_step", first_step=want.name, observed=raw[:160])
        got = Step(*match.groups())
        if (got.name, got.op) != (want.name, want.op):
            return dict(result, error="wrong_step_or_operation", first_step=want.name, observed=raw[:160])
        try:
            left, right = Fraction(got.left), Fraction(got.right)
            wanted_left, wanted_right = Fraction(want.left), Fraction(want.right)
            operands_match = ((left, right) == (wanted_left, wanted_right) or
                              (got.op in ("add", "multiply") and
                               (left, right) == (wanted_right, wanted_left)))
            if not operands_match:
                return dict(result, error="wrong_operand_binding", first_step=want.name, observed=raw[:160])
            got.check()
        except (ValueError, ZeroDivisionError):
            return dict(result, error="wrong_computed_value", first_step=want.name,
                        expected=want.result, observed=raw[:160])
        result["correct_prefix_steps"] += 1
    if len(generated_steps) != len(expected) or "</work>" not in text:
        return dict(result, error="extra_steps_or_missing_close")
    return dict(result, error=None)


def analyze(root: Path) -> dict:
    summary = json.loads((root / "summary.json").read_text())
    if summary["state"] != "completed":
        raise ValueError("pilot is not completed")
    result = {"source_sha256": summary["contract"]["source_sha256"], "arms": {}}
    for arm in ("baseline", "control", "loss_only", "verified"):
        families = {}
        for family in TARGET_FAMILIES:
            rows = [json.loads(line) for line in (root / arm / f"{family}.generations.jsonl").read_text().splitlines()]
            errors, stages = collections.Counter(), collections.Counter()
            details = []
            for row in rows:
                diagnosis = first_procedure_error({"problem": row["problem"], "family": family,
                                                   "normalized_answer": row["expected"]}, row["generation"])
                errors[diagnosis["error"] or "complete_procedure"] += 1
                stages[diagnosis["correct_prefix_steps"]] += 1
                details.append({"record_id": row["record_id"], "answer_correct": row["correct"], **diagnosis})
            families[family] = {"examples": len(rows), "first_error_counts": dict(errors),
                                "correct_prefix_step_histogram": dict(stages), "rows": details}
        result["arms"][arm] = families
    result["caveat"] = ("This checks the prescribed procedure, accepting commutative operand swaps, not every possible correct derivation. "
                        "Legacy arms were not trained for this grammar; their grammar failures are not comparable arithmetic scores.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({arm: {family: value["first_error_counts"] for family, value in families.items()}
                      for arm, families in result["arms"].items()}))


if __name__ == "__main__":
    main()
