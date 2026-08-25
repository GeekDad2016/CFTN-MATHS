from __future__ import annotations

import torch

from cftn_text.dataset import MathCollator
from cftn_text.tokenizer import ByteMathTokenizer


def test_math_collator_maps_answers_outside_int64_to_sentinel():
    records = [
        {
            "problem": "Return a very large integer.",
            "target_trace": f"<answer>{10**100}</answer>",
            "answer_value": 10**100,
        },
        {
            "problem": "Return a very small integer.",
            "target_trace": f"<answer>{-(10**100)}</answer>",
            "answer_value": -(10**100),
        },
        {
            "problem": "Return seven.",
            "target_trace": "<answer>7</answer>",
            "answer_value": 7,
        },
    ]

    batch = MathCollator(ByteMathTokenizer(), max_length=512)(records)

    assert batch["answer_values"].tolist() == [-2_000_000_000, -2_000_000_000, 7]


def test_answer_only_collator_canonicalizes_mixed_trace_styles():
    tokenizer = ByteMathTokenizer()
    records = [
        {
            "problem": "Compute two plus three.",
            "target_trace": "<program>add(2,3)</program><answer>5</answer>",
            "normalized_answer": "5",
        },
        {
            "problem": "Return x.",
            "target_trace": "<work>solve</work><answer>x=2;y=3</answer>",
            "normalized_answer": "x=2;y=3",
        },
    ]
    batch = MathCollator(
        tokenizer, max_length=512, target_mode="answer_only_v1"
    )(records)
    for row, prefix_length, expected in zip(
        batch["math_input_ids"],
        batch["math_prefix_lengths"],
        ("<answer>5</answer>", "<answer>x=2;y=3</answer>"),
    ):
        decoded = tokenizer.decode(row[int(prefix_length) :])
        assert decoded == expected
    assert torch.equal(batch["math_labels"], batch["math_answer_labels"])
