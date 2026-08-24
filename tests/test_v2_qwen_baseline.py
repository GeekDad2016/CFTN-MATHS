from __future__ import annotations

from cftn_text.v2_qwen_baseline import (
    difficulty_balanced_panel,
    qwen_math_messages,
)


def _record(index: int, difficulty: int, family: str) -> dict:
    return {
        "record_id": f"r-{difficulty}-{index}",
        "source": "source-a" if index % 2 else "source-b",
        "family": family,
        "difficulty": difficulty,
        "problem": f"problem {difficulty}-{index}",
        "gpt_problem": "masked coordinator-only problem",
        "normalized_answer": str(index),
    }


def test_qwen_math_messages_preserve_problem_and_exact_answer_contract():
    messages = qwen_math_messages(_record(1, 2, "fractions"))

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "problem 2-1" in messages[1]["content"]
    assert "masked coordinator-only problem" not in messages[1]["content"]
    assert "<answer>...</answer>" in messages[1]["content"]
    assert "concise reasoning" in messages[1]["content"]
    assert "external tools" in messages[0]["content"]


def test_qwen_math_messages_support_answer_only_control():
    messages = qwen_math_messages(
        _record(1, 2, "fractions"), prompt_mode="answer_only"
    )

    assert "Do not include an explanation" in messages[1]["content"]


def test_difficulty_balanced_panel_is_deterministic_and_balanced():
    records = [
        _record(index, difficulty, f"family-{index % 3}")
        for difficulty in (1, 2, 3)
        for index in range(8)
    ]

    first = difficulty_balanced_panel(records, examples_per_difficulty=4)
    second = difficulty_balanced_panel(records, examples_per_difficulty=4)

    assert [record["record_id"] for record in first] == [
        record["record_id"] for record in second
    ]
    assert len(first) == 12
    assert {difficulty: sum(row["difficulty"] == difficulty for row in first) for difficulty in (1, 2, 3)} == {1: 4, 2: 4, 3: 4}


def test_difficulty_balanced_panel_rejects_undersized_cohort():
    records = [_record(1, difficulty, "family") for difficulty in (1, 2, 3)]

    try:
        difficulty_balanced_panel(records, examples_per_difficulty=2)
    except RuntimeError as exc:
        assert "difficulty 1" in str(exc)
    else:
        raise AssertionError("undersized difficulty cohort was accepted")
