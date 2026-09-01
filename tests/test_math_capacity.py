from __future__ import annotations

import json
from pathlib import Path

import torch

from cftn_text.config import load_config
from cftn_text.math_tower import MathTower
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.training import (
    _initialize_math_capacity_expansion,
    build_math_tower_for_checkpoint,
)


def _tiny_tower_config(layers: int) -> dict:
    return {
        "layers": layers,
        "hidden_size": 32,
        "attention_heads": 4,
        "feed_forward_size": 64,
        "dropout": 0.0,
        "max_sequence_length": 64,
        "answer_head_mode": "disabled",
        "answer_min": -8,
        "answer_max": 8,
        "receiver_layers": [0, 1],
    }


def test_identity_depth_expansion_preserves_source_function():
    torch.manual_seed(719)
    source = MathTower(_tiny_tower_config(2), ByteMathTokenizer.vocab_size)
    target = MathTower(_tiny_tower_config(5), ByteMathTokenizer.vocab_size)

    attestation = _initialize_math_capacity_expansion(
        target,
        source.state_dict(),
        {
            "method": "append_identity_transformer_blocks_v1",
            "source_layers": 2,
            "target_layers": 5,
            "hidden_size": 32,
            "maximum_function_error": 1.0e-6,
        },
        device=torch.device("cpu"),
    )

    assert attestation["identity_preserved"] is True
    assert attestation["added_layers"] == 3
    assert attestation["target_parameters"] > attestation["source_parameters"]
    assert attestation["observed_logit_error"] <= 1.0e-6
    assert attestation["observed_hidden_state_error"] <= 1.0e-6
    for block in target.blocks[2:]:
        assert torch.count_nonzero(block.self_attn.out_proj.weight) == 0
        assert torch.count_nonzero(block.linear2.weight) == 0

    rebuilt = build_math_tower_for_checkpoint(
        {"math_tower": _tiny_tower_config(2)},
        {
            "extra": {
                "effective_math_tower": _tiny_tower_config(5),
                "metrics": {
                    "source_checkpoint": {"capacity_expansion": attestation}
                },
            }
        },
    )
    assert len(rebuilt.blocks) == 5


def test_cached_incremental_math_decoding_matches_full_forward():
    torch.manual_seed(901)
    model = MathTower(_tiny_tower_config(2), ByteMathTokenizer.vocab_size).eval()
    prompt = torch.tensor([[2, 10, 20, 30]], dtype=torch.long)
    prompt_mask = torch.ones_like(prompt)
    prefix_lengths = torch.tensor([prompt.shape[1]], dtype=torch.long)

    with torch.inference_mode():
        full_prompt = model(prompt, prompt_mask, prefix_lengths)
        cache, cached_prompt = model.begin_cached_generation(prompt)
        next_token = 7
        extended = torch.cat(
            [prompt, torch.tensor([[next_token]], dtype=torch.long)], dim=1
        )
        full_extended = model(
            extended, torch.ones_like(extended), prefix_lengths
        )
        cached_extended = model.cached_generation_step(cache, next_token)

    assert torch.allclose(
        full_prompt.logits[:, -1:], cached_prompt.logits, atol=1.0e-6
    )
    assert torch.allclose(
        full_extended.logits[:, -1:], cached_extended.logits, atol=1.0e-6
    )


def test_v2_capacity_contract_is_a_fail_closed_single_variable_ablation():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "v2_broad_math.yaml")
    contract = json.loads(
        (root / "config" / "v2_math_checkpoint45_capacity_recovery.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["math_tower"]["layers"] == 8
    assert config["math_tower"]["hidden_size"] == 384
    assert contract["capacity_expansion"] == {
        "method": "append_identity_transformer_blocks_v1",
        "source_layers": 8,
        "target_layers": 24,
        "hidden_size": 384,
        "expected_target_parameters": 47414913,
        "maximum_function_error": 0.000001,
    }
    assert contract["phases"][0]["through_epoch"] == 8
    assert contract["phases"][0]["minimum_generation_accuracy"] == 0.30
    assert contract["phases"][1]["through_epoch"] == 20
    assert contract["phases"][1]["minimum_generation_accuracy"] == 0.70
    assert contract["require_acceptance_for_best"] is True
