from __future__ import annotations

from pathlib import Path

import torch

from cftn_text.conditional_training import (
    REDUNDANT_REQUIREMENT,
    REQUIRED_REQUIREMENT,
    _set_conditional_trainable,
    build_mixed_necessity_records,
    conditional_acceptance,
    conditional_objective,
    load_revision_config,
    per_example_causal_loss,
    specialist_preservation_kl,
)
from cftn_text.data_generator import build_records
from cftn_text.dataset import CFTNCollator
from cftn_text.tokenizer import ByteMathTokenizer


def test_revision_config_is_explicit_and_portable():
    path = Path(__file__).parents[1] / "config" / "v1_2_conditional_bridge.yaml"
    revision = load_revision_config(path)
    assert revision["format"] == "cftn_text_conditional_bridge_revision_v1_2"
    assert Path(revision["paths"]["base_config"]).is_absolute()
    assert revision["training"]["learning_rate"] == 2e-5
    assert revision["training"]["gate_learning_rate_multiplier"] == 0.25
    assert revision["acceptance"]["maximum_redundant_math_regression"] == 0.02


def test_mixed_necessity_records_pair_required_and_redundant_views(tiny_config):
    records = [record.to_dict() for record in build_records(tiny_config)["train"]]
    mixed = build_mixed_necessity_records(records, seed=719, limit=3)
    assert len(mixed) == 6
    for pair_index in range(3):
        shared, required = mixed[pair_index * 2 : pair_index * 2 + 2]
        assert shared["communication_requirement"] == REDUNDANT_REQUIREMENT
        assert required["communication_requirement"] == REQUIRED_REQUIREMENT
        assert shared["x"] == required["x"]
        assert shared["view_mode"] == "shared"
        assert required["view_mode"] == "complementary"
        assert "gpt_problem" not in shared
        assert required["gpt_problem"] != required["math_problem"]


def test_per_example_causal_loss_respects_masked_prefix():
    logits = torch.zeros(2, 4, 5)
    labels = torch.tensor(
        [[-100, -100, 2, 3], [-100, 1, 2, -100]], dtype=torch.long
    )
    logits[0, 1, 2] = 8
    logits[0, 2, 3] = 8
    logits[1, 0, 1] = 8
    logits[1, 1, 2] = 8
    losses = per_example_causal_loss(logits, labels)
    assert losses.shape == (2,)
    assert torch.all(losses < 0.01)


def test_preservation_kl_uses_only_baseline_correct_rows():
    baseline = torch.zeros(2, 4, 4)
    current = torch.zeros_like(baseline, requires_grad=True)
    labels = torch.tensor(
        [[-100, -100, 1, 2], [-100, -100, 1, 2]], dtype=torch.long
    )
    baseline[0, 1, 1] = 8
    baseline[0, 2, 2] = 8
    baseline[1, 1, 3] = 8
    baseline[1, 2, 3] = 8
    current.data[0, 1, 3] = 4
    loss, rows = specialist_preservation_kl(
        current, baseline, labels, torch.tensor([True, True])
    )
    assert rows == 1
    assert float(loss) > 0
    loss.backward()
    assert current.grad is not None


def test_conditional_objective_updates_only_gpt_to_math_path(
    tiny_model, tiny_config
):
    records = [record.to_dict() for record in build_records(tiny_config)["train"]]
    mixed = build_mixed_necessity_records(records, seed=719, limit=2)
    tokenizer = ByteMathTokenizer()
    collator = CFTNCollator(
        tokenizer,
        tokenizer,
        tiny_config["data"]["max_math_length"],
        tiny_config["data"]["max_gpt_length"],
    )
    batch = collator(mixed)
    _set_conditional_trainable(tiny_model)
    _, components = conditional_objective(
        tiny_model,
        batch,
        {
            "math_loss_weight": 1.0,
            "gpt_loss_weight": 1.0,
            "answer_head_weight": 0.0,
            "preservation_weight": 2.0,
            "contrastive_weight": 0.5,
            "contrastive_margin": 0.25,
            "redundant_gate_weight": 0.05,
        },
    )
    loss = components["loss"]
    assert torch.is_tensor(loss) and torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in tiny_model.gpt_to_math.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in tiny_model.math_to_gpt.parameters()
    )


def test_conditional_acceptance_requires_utility_and_no_harm():
    generation = {
        REQUIRED_REQUIREMENT: {
            "gpt_to_math_sender_gate_mean": 0.6,
            "accuracy": {
                "correct": {"gpt": 0.9, "math": 0.9},
                "gpt_to_math_disabled": {"gpt": 0.1, "math": 0.2},
                "gpt_to_math_shuffled": {"gpt": 0.2, "math": 0.3},
                "math_to_gpt_disabled": {"gpt": 0.0, "math": 0.9},
                "both_disabled": {"gpt": 0.0, "math": 0.1},
            },
        },
        REDUNDANT_REQUIREMENT: {
            "gpt_to_math_sender_gate_mean": 0.2,
            "accuracy": {
                "correct": {"gpt": 0.9, "math": 0.98},
                "gpt_to_math_disabled": {"gpt": 0.9, "math": 0.99},
                "gpt_to_math_shuffled": {"gpt": 0.8, "math": 0.7},
                "math_to_gpt_disabled": {"gpt": 0.0, "math": 0.98},
                "both_disabled": {"gpt": 0.0, "math": 0.99},
            },
        },
    }
    thresholds = {
        "minimum_required_synergy_gain": 0.10,
        "minimum_required_gpt_to_math_gain": 0.10,
        "minimum_required_correct_vs_shuffled_gap": 0.02,
        "minimum_required_math_to_gpt_gain": 0.10,
        "maximum_redundant_math_regression": 0.02,
        "minimum_gate_separation": 0.05,
    }
    report = conditional_acceptance(generation, thresholds)
    assert report["gates"]["pass"] is True
    generation[REDUNDANT_REQUIREMENT]["accuracy"]["correct"]["math"] = 0.80
    report = conditional_acceptance(generation, thresholds)
    assert report["gates"]["redundant_no_harm"] is False
    assert report["gates"]["pass"] is False
