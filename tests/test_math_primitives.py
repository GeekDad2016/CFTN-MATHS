import copy
from fractions import Fraction
import random

import pytest
import torch

from cftn_text.computation_supervision import computation_loss
from cftn_text.math_primitive_data import (
    ARMS, COMPOSITIONS, FOUNDATIONS, PrimitiveCollator, candidate, lesson, make_corpus,
    object_key, object_split, parse_question, validate_lesson,
)
from cftn_text.tokenizer import ByteMathTokenizer, SequenceTooLongError
from cftn_text.verified_math_data import fingerprint
from tools.pilot_math_primitives import native_gate, prerequisite_gate, primitive_score, schedule


@pytest.mark.parametrize("question,expected", [
    ("Read operands only: Product of -1.20 and 0.05. Return a and b.", "a=-1.20;b=0.05"),
    ("Read coefficients only: Solve the system 1*x + (-2)*y = 3; -4*x + (5)*y = -6. Give x and y. Return a,b,r,c,d,s.", "a=1;b=-2;r=3;c=-4;d=5;s=-6"),
    ("Write -3.04 as n/10^k with k=2. Return n and s=10^k.", "n=-304;s=100"),
    ("Write 0.00 as n/10^k with k=2. Return n and s=10^k.", "n=0;s=100"),
    ("Write -304/100 as a decimal.", "-3.04"),
    ("Write 30/100 as a decimal.", "0.3"),
    ("Calculate -99*-9.", "891"),
    ("Calculate 0*-9.", "0"),
    ("Calculate -33-(-17).", "-16"),
    ("Calculate -120/(10). Return an exact integer.", "-12"),
    ("Calculate -1.3*2.4.", "-3.12"),
    ("Calculate (-7*-8)+(-20).", "36"),
])
def test_exact_public_lessons_and_arms(question, expected):
    for arm in ARMS:
        row = lesson(question, arm)
        assert row["normalized_answer"] == expected
        validate_lesson(row)
        assert primitive_score(row, row["target_trace"], 256)["correct"]
        assert len(ByteMathTokenizer().encode(row["target_trace"])) < 256
        if arm == "answer_only":
            assert "<work>" not in row["target_trace"]


@pytest.mark.parametrize("question", [
    "Write 1.234 as n/10^k with k=2. Return n and s=10^k.",
    "Calculate 7/(3). Return an exact integer.",
    "Write 12/0 as a decimal.", "Calculate __import__('os').",
])
def test_ambiguous_or_invalid_lessons_fail_closed(question):
    with pytest.raises(ValueError):
        lesson(question, "compact_worked")


def test_re_signed_wrong_label_or_work_rejected():
    row = lesson("Calculate -33-(-17).", "compact_worked")
    for key in ("normalized_answer", "target_trace", "problem", "computation_key"):
        bad = copy.deepcopy(row)
        bad[key] = "wrong"
        bad["record_id"] = fingerprint({k: v for k, v in bad.items() if k != "record_id"})
        with pytest.raises(ValueError):
            validate_lesson(bad)


def test_numeric_split_before_rendering_and_representation():
    a = "Calculate -3*4."
    b = "Calculate 4*-3."
    c = "Calculate -3.0*4.00."
    assert object_key(a) == object_key(b) == object_key(c)
    assert object_key("Write -3.04 as n/10^k with k=2. Return n and s=10^k.") == object_key("Write -304/100 as a decimal.")
    for arm in ARMS:
        assert lesson(a, arm)["split"] == object_split(object_key(a))


def test_deterministic_corpus_and_leakage_checks():
    corpus = make_corpus(16, 8)
    assert corpus == make_corpus(16, 8)
    keys = {s: {object_key(q) for pools in corpus.values() for q in pools[s]} for s in ("train", "validation")}
    assert not keys["train"] & keys["validation"]
    for family, pools in corpus.items():
        assert len(pools["train"]) == 16 and len(pools["validation"]) == 8
        for split, questions in pools.items():
            for q in questions:
                assert parse_question(q)[0] == family
                assert object_split(object_key(q)) == split


def test_loss_allows_explicit_copy_only_lesson_without_changing_default():
    row = lesson("Read operands only: Product of -1.20 and 0.05. Return a and b.", "compact_worked")
    batch = PrimitiveCollator(ByteMathTokenizer(), 4096)([row])
    logits = torch.randn(1, batch["math_labels"].shape[1], 260, requires_grad=True)
    with pytest.raises(ValueError, match="computation"):
        computation_loss(logits, batch["math_labels"], batch["math_roles"])
    loss = computation_loss(logits, batch["math_labels"], batch["math_roles"], weights=(.25, .5, .25), require_computation=False)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert batch["math_roles"].ne(-100).equal(batch["math_labels"].ne(-100))
    assert batch["math_roles"].eq(1).sum() == 0
    with pytest.raises(ValueError, match="weights"):
        computation_loss(logits, batch["math_labels"], batch["math_roles"], weights=(0, 1, 0))
    with pytest.raises(SequenceTooLongError):
        PrimitiveCollator(ByteMathTokenizer(), 20)([row])


def test_schedule_identical_across_arms_and_does_not_train_on_validation():
    corpus = make_corpus(16, 8)
    replay = {f: [{"family": f, "split": "train"}] for f in ("variables_both_sides", "nested_parentheses")}
    for stage in ("memorization", "foundations", "composition"):
        batches = schedule(corpus, stage, 14, replay, 826)
        assert batches == schedule(corpus, stage, 14, replay, 826)
        for batch in batches:
            assert len(batch) == 16
            assert sum(isinstance(r, dict) for r in batch) == (0 if stage == "memorization" else 4)
            for row in batch:
                if isinstance(row, str):
                    assert object_split(object_key(row)) == "train"


def test_strict_scoring_and_fail_closed_gates():
    row = lesson("Calculate 2*3.", "answer_only")
    assert primitive_score(row, "<answer>6</answer>", 256)["correct"]
    assert not primitive_score(row, "<answer>5</answer><answer>6</answer>", 256)["format_valid"]
    assert not primitive_score(row, "<answer>6", 256)["correct"]
    families = FOUNDATIONS + ("variables_both_sides", "nested_parentheses")
    report = {f: {"accuracy": 1., "valid_rate": 1., "budget_hits": 0} for f in families}
    baseline = copy.deepcopy(report)
    assert prerequisite_gate(report, baseline)["pass"]
    report["integer_multiply"]["accuracy"] = .89
    assert not prerequisite_gate(report, baseline)["pass"]
    report["integer_multiply"]["accuracy"] = 1
    report["decimal_scale"]["budget_hits"] = 1
    assert not prerequisite_gate(report, baseline)["pass"]
    assert not prerequisite_gate({}, baseline)["pass"]
    target = {f: {"accuracy": .5, "budget_hits": 0} for f in ("two_variable_systems", "arithmetic__mul", "broad_diagnostic")}
    gate = native_gate(target, target, baseline, baseline)
    assert not gate["pass"] and not gate["production_acceptance"]
