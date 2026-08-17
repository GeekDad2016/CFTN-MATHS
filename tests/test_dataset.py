from __future__ import annotations

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
