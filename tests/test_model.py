from __future__ import annotations

import torch

from cftn_text.data_generator import build_records
from cftn_text.dataset import CFTNCollator
from cftn_text.model import answer_weighted_causal_language_loss
from cftn_text.tokenizer import ByteMathTokenizer


def test_answer_weighted_loss_prioritizes_focused_payload_tokens():
    logits = torch.zeros(1, 4, 8, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 1] = 5.0
        logits[0, 1, 2] = 5.0
    labels = torch.tensor([[-100, 1, 2, 3]])
    focused = torch.tensor([[-100, -100, -100, 3]])
    base = answer_weighted_causal_language_loss(
        logits, labels, focused, answer_weight=1.0
    )
    weighted = answer_weighted_causal_language_loss(
        logits, labels, focused, answer_weight=4.0
    )
    assert torch.isfinite(base)
    assert torch.isfinite(weighted)
    assert weighted > base
    weighted.backward()
    assert logits.grad is not None


def make_batch(config):
    records = [record.to_dict() for record in build_records(config)["train"][:2]]
    tokenizer = ByteMathTokenizer()
    collator = CFTNCollator(
        tokenizer,
        tokenizer,
        config["data"]["max_math_length"],
        config["data"]["max_gpt_length"],
    )
    return collator(records)


def test_frozen_gpt_disabled_receiver_is_bit_exact(tiny_model):
    tiny_model.eval()
    ids = torch.randint(4, 100, (2, 12))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        base = tiny_model.gpt_tower.model(
            input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
        ).logits
        disabled = tiny_model.gpt_tower(
            ids,
            mask,
            message=torch.randn(2, 4, 32),
            receive_enabled=False,
        ).logits
    assert torch.equal(base, disabled)


def test_full_cftn_always_executes_both_towers_without_modules_that_route(
    tiny_model, tiny_config
):
    batch = make_batch(tiny_config)
    tiny_model.set_trainable_stage("bidirectional")
    tiny_model.reset_execution_counts()
    output = tiny_model(batch)
    counts = tiny_model.execution_counts()
    assert counts["math_tower"] == 1
    assert counts["gpt_prepass"] == 1
    assert counts["gpt_receiver"] == 1
    assert counts["gpt_to_math_bridge"] == 1
    assert counts["math_to_gpt_bridge"] == 1
    module_names = [name.lower() for name, _ in tiny_model.named_modules()]
    assert not any("router" in name or "expert_selector" in name for name in module_names)
    output.loss.backward()
    assert tiny_model.gpt_tower.receivers["0"].output_projection.weight.grad is not None
    assert tiny_model.math_receivers["0"].output_projection.weight.grad is not None


def test_closing_gates_does_not_skip_towers(tiny_model, tiny_config):
    batch = make_batch(tiny_config)
    tiny_model.reset_execution_counts()
    output = tiny_model(
        batch, gpt_to_math_enabled=False, math_to_gpt_enabled=False
    )
    counts = tiny_model.execution_counts()
    assert counts["math_tower"] == 1
    assert counts["gpt_prepass"] == 1
    assert counts["gpt_receiver"] == 1
    assert torch.count_nonzero(output.gpt_to_math.message) == 0
    assert torch.count_nonzero(output.math_to_gpt.message) == 0


def test_stage_freezing_preserves_roles(tiny_model):
    tiny_model.set_trainable_stage("m2g")
    assert not any(parameter.requires_grad for parameter in tiny_model.math_tower.parameters())
    assert not any(parameter.requires_grad for parameter in tiny_model.gpt_tower.model.parameters())
    assert any(parameter.requires_grad for parameter in tiny_model.math_to_gpt.parameters())
    assert not any(parameter.requires_grad for parameter in tiny_model.gpt_to_math.parameters())
    tiny_model.set_trainable_stage("bidirectional")
    assert any(parameter.requires_grad for parameter in tiny_model.gpt_to_math.parameters())
    assert any(parameter.requires_grad for parameter in tiny_model.math_receivers.parameters())


def test_trainable_checkpoint_excludes_frozen_gpt_base(tiny_model):
    tiny_model.set_trainable_stage("bidirectional")
    state = tiny_model.trainable_state_dict()
    expected = {
        name for name, parameter in tiny_model.named_parameters() if parameter.requires_grad
    }
    assert state
    assert set(state) == expected
    assert not any(name.startswith("gpt_tower.model.") for name in state)
    assert not any(name.startswith("math_tower.") for name in state)


def test_batched_gpt_generation_matches_individual_generation(tiny_model):
    tiny_model.eval()
    tokenizer = ByteMathTokenizer()
    prefixes = [
        tokenizer.encode("short", add_bos=True),
        tokenizer.encode("a somewhat longer prefix", add_bos=True),
    ]
    torch.manual_seed(11)
    messages = torch.randn(2, 4, 32)
    batched = tiny_model.gpt_tower.generate_greedy(
        prefixes,
        messages,
        tokenizer.eos_token_id,
        4,
        receive_enabled=True,
    )
    individual = [
        tiny_model.gpt_tower.generate_greedy(
            [prefixes[index]],
            messages[index : index + 1],
            tokenizer.eos_token_id,
            4,
            receive_enabled=True,
        )[0]
        for index in range(2)
    ]
    assert batched == individual
