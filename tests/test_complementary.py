from __future__ import annotations

from cftn_text.complementary import complementary_record
from cftn_text.data_generator import build_records
from cftn_text.dataset import CFTNCollator
from cftn_text.tokenizer import ByteMathTokenizer


def test_complementary_views_split_roles_from_values(tiny_config):
    records = [record.to_dict() for record in build_records(tiny_config)["train"]]
    transformed = [
        complementary_record(record, seed=719) for record in records
    ]
    assert len({tuple(sorted(row["role_assignment"].items())) for row in transformed}) > 1
    for source, row in zip(records, transformed):
        assert not any(character.isdigit() for character in row["gpt_problem"])
        assignment = row["role_assignment"]
        slots = row["slot_values"]
        assert slots[assignment["coefficient"]] == source["a"]
        assert slots[assignment["offset"]] == source["b"]
        assert slots[assignment["result"]] == source["c"]


def test_pair_assignment_keeps_language_view_identical(tiny_config):
    record = build_records(tiny_config)["train"][0].to_dict()
    changed = dict(record)
    changed["x"] = int(record["x"]) + 1
    changed["c"] = int(record["a"]) * int(changed["x"]) + int(record["b"])
    first = complementary_record(record, seed=719, assignment_key="pair")
    second = complementary_record(changed, seed=719, assignment_key="pair")
    assert first["gpt_problem"] == second["gpt_problem"]
    assert first["math_problem"] != second["math_problem"]


def test_collator_uses_private_views(tiny_config):
    record = build_records(tiny_config)["train"][0].to_dict()
    record["math_problem"] = "MATH PRIVATE VIEW"
    record["gpt_problem"] = "GPT PRIVATE VIEW"
    tokenizer = ByteMathTokenizer()
    collator = CFTNCollator(
        tokenizer,
        tokenizer,
        tiny_config["data"]["max_math_length"],
        tiny_config["data"]["max_gpt_length"],
    )
    batch = collator([record])
    math_text = tokenizer.decode(batch["math_input_ids"][0].tolist())
    gpt_text = tokenizer.decode(batch["gpt_prepass_input_ids"][0].tolist())
    assert "MATH PRIVATE VIEW" in math_text
    assert "GPT PRIVATE VIEW" in gpt_text
    assert record["problem"] not in math_text
    assert record["problem"] not in gpt_text
