from __future__ import annotations

import json

import torch

from cftn_text.dataset import PRIVATE_MATH_INPUT_VIEW
from cftn_text.math_validation import (
    DEFAULT_V2_GENERATION_VALIDATION,
    evaluate_generation_panel,
    stratified_validation_panel,
    summarize_teacher_forced_breakdowns,
    update_teacher_forced_breakdowns,
)
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.v2_data import make_v2_record


def _record(source: str, family: str, difficulty: int, answer: str = "2"):
    return make_v2_record(
        split="validation",
        source=source,
        family=family,
        difficulty=difficulty,
        problem=f"Compute the {family} result.",
        answer=answer,
    )


def test_v2_generation_validation_defaults_are_bounded_and_epoch_level():
    assert DEFAULT_V2_GENERATION_VALIDATION == {
        "enabled": True,
        "every_epochs": 1,
        "examples": 96,
        "batch_size": 16,
        "max_new_tokens": 512,
        "failure_examples": 8,
    }


def test_teacher_forced_breakdowns_report_source_family_and_difficulty():
    records = [_record("a", "addition", 1), _record("b", "fraction", 2)]
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 1, 2, 3]])
    logits = torch.full((2, 4, 8), -10.0)
    logits[0, 0, 1] = 10
    logits[0, 1, 2] = 10
    logits[0, 2, 3] = 10
    logits[1, 0, 1] = 10
    logits[1, 1, 7] = 10
    logits[1, 2, 3] = 10
    groups = {}

    update_teacher_forced_breakdowns(
        groups, logits=logits, labels=labels, records=records
    )
    result = summarize_teacher_forced_breakdowns(groups)

    assert result["by_source"]["a"]["teacher_forced_sequence_accuracy"] == 1.0
    assert result["by_family"]["fraction"]["teacher_forced_token_accuracy"] == 2 / 3
    assert result["by_difficulty"]["2"]["examples"] == 1


def test_stratified_validation_panel_covers_cohorts_before_repeating():
    records = [
        _record("a", "addition", 1),
        _record("a", "addition", 1),
        _record("b", "fraction", 2),
        _record("c", "algebra", 3),
    ]

    panel = stratified_validation_panel(records, 3)

    assert {(row["source"], row["family"], row["difficulty"]) for row in panel} == {
        ("a", "addition", 1),
        ("b", "fraction", 2),
        ("c", "algebra", 3),
    }


def test_seeded_stratified_panel_is_reproducible_but_varies_its_draw():
    records = [
        _record("a", "addition", 1, str(index))
        for index in range(6)
    ] + [
        _record("b", "fraction", 2, str(index))
        for index in range(6)
    ]
    first = stratified_validation_panel(records, 8, selection_seed=713)
    repeat = stratified_validation_panel(records, 8, selection_seed=713)
    assert [row["record_id"] for row in first] == [row["record_id"] for row in repeat]
    assert len({row["family"] for row in first}) == 2


def test_generation_panel_reports_failures_and_writes_audit_rows(
    tmp_path, monkeypatch
):
    records = [_record("a", "addition", 1, "2"), _record("b", "fraction", 2, "1/2")]

    def fake_generate(
        _model, _tokenizer, problems, *, max_new_tokens, diagnostics=None
    ):
        assert max_new_tokens == 32
        values = ["<answer>2</answer>", "<answer>3/4</answer>"]
        if diagnostics is not None:
            diagnostics.extend(
                {
                    "generated_tokens": 4,
                    "eos_terminated": True,
                    "context_limit_hit": False,
                    "budget_hit": False,
                    "cached_incremental": True,
                }
                for _ in problems
            )
        return values[: len(problems)], [None] * len(problems)

    monkeypatch.setattr(
        "cftn_text.specialist_evaluation.generate_math_tower", fake_generate
    )

    class Model(torch.nn.Module):
        max_sequence_length = 256

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

    rows_path = tmp_path / "generation.jsonl"
    progress = []
    report = evaluate_generation_panel(
        Model(),
        ByteMathTokenizer(),
        records,
        maximum_examples=2,
        batch_size=2,
        max_new_tokens=32,
        failure_examples=1,
        rows_path=rows_path,
        progress_callback=progress.append,
    )

    assert report["valid_rate"] == 1.0
    assert report["accuracy"] == 0.5
    assert report["by_family"]["fraction"]["accuracy"] == 0.0
    assert report["failure_examples"][0]["expected_answer"] == "1/2"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["termination"]["eos_terminated"] is True
    assert progress[-1]["completed_examples"] == 2
    assert progress[-1]["eos_terminated"] == 2


def test_generation_panel_uses_the_declared_input_view(monkeypatch):
    record = _record("cftn_generated", "variables_both_sides", 1, "2")
    record["math_problem"] = "PRIVATE ROLE-MAPPED REQUEST"

    def fake_generate(
        _model, _tokenizer, problems, *, max_new_tokens, diagnostics=None
    ):
        assert problems == ["PRIVATE ROLE-MAPPED REQUEST"]
        if diagnostics is not None:
            diagnostics.append(
                {
                    "generated_tokens": 4,
                    "eos_terminated": True,
                    "context_limit_hit": False,
                    "budget_hit": False,
                    "cached_incremental": True,
                }
            )
        return ["<answer>2</answer>"], [None]

    monkeypatch.setattr(
        "cftn_text.specialist_evaluation.generate_math_tower", fake_generate
    )

    class Model(torch.nn.Module):
        max_sequence_length = 256

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

    report = evaluate_generation_panel(
        Model(),
        ByteMathTokenizer(),
        [record],
        maximum_examples=1,
        batch_size=1,
        max_new_tokens=32,
        failure_examples=1,
        input_view=PRIVATE_MATH_INPUT_VIEW,
    )

    assert report["input_view"] == PRIVATE_MATH_INPUT_VIEW
    assert report["accuracy"] == 1.0
