from __future__ import annotations

import torch

from cftn_text.model import optional_answer_loss
from cftn_text.v2_data import make_v2_record
from cftn_text.v2_metrics import answers_equivalent, extract_v2_answer, score_v2_generations


def test_v2_answer_scoring_handles_fractions_systems_and_symbolic_expressions():
    assert answers_equivalent("2/4", "1/2")
    assert answers_equivalent("y=3; x=-2", "x=-2;y=3")
    assert answers_equivalent("2*x + 2", "2*(x+1)")
    assert extract_v2_answer("work <answer> -3/4 </answer>") == "-3/4"


def test_v2_generation_metrics_require_answer_tags():
    records = [
        make_v2_record(
            split="test",
            source="unit",
            family="fraction",
            difficulty=2,
            problem="What is one half?",
            answer="1/2",
        )
    ]
    report, correctness = score_v2_generations(
        ["<answer>2/4</answer>"], records
    )
    assert report["accuracy"] == 1.0
    assert correctness == [True]
    untagged, _ = score_v2_generations(["1/2"], records)
    assert untagged["valid_rate"] == 0.0


def test_v2_generation_metrics_reject_malformed_symbolic_output_without_crashing():
    records = [
        make_v2_record(
            split="test",
            source="unit",
            family="fraction",
            difficulty=2,
            problem="What is nineteen eighths?",
            answer="19/8",
        )
    ]
    report, correctness = score_v2_generations(
        ["<answer>e19/8</answer>"], records
    )
    assert report["accuracy"] == 0.0
    assert correctness == [False]
    assert not answers_equivalent("e19/8", "19/8")


def test_optional_answer_loss_is_finite_when_batch_has_no_integer_targets():
    logits = torch.randn(3, 7, requires_grad=True)
    classes = torch.full((3,), -100, dtype=torch.long)
    loss = optional_answer_loss(logits, classes)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    loss.backward()
    assert logits.grad is not None
