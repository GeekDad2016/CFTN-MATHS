from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from cftn_text.config import load_config
from cftn_text.math_tower import MathTower
from cftn_text.specialist_evaluation import generate_math_tower
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


def test_checkpoint_builder_accepts_attested_unexpanded_architecture():
    config = {"math_tower": _tiny_tower_config(2)}
    rebuilt = build_math_tower_for_checkpoint(
        config,
        {"extra": {"effective_math_tower": _tiny_tower_config(2)}},
    )
    assert len(rebuilt.blocks) == 2


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


def test_batched_cached_incremental_math_decoding_matches_full_forward():
    torch.manual_seed(902)
    model = MathTower(_tiny_tower_config(2), ByteMathTokenizer.vocab_size).eval()
    prompt = torch.tensor([[2, 10, 20, 30], [2, 11, 21, 31]], dtype=torch.long)
    prompt_mask = torch.ones_like(prompt)
    prefix_lengths = torch.tensor([prompt.shape[1], prompt.shape[1]], dtype=torch.long)

    with torch.inference_mode():
        full_prompt = model(prompt, prompt_mask, prefix_lengths)
        cache, cached_prompt = model.begin_cached_generation(prompt)
        next_tokens = torch.tensor([7, 8], dtype=torch.long)
        extended = torch.cat([prompt, next_tokens.unsqueeze(1)], dim=1)
        full_extended = model(
            extended, torch.ones_like(extended), prefix_lengths
        )
        cached_extended = model.cached_generation_step(cache, next_tokens)

    assert torch.allclose(
        full_prompt.logits[:, -1:], cached_prompt.logits, atol=1.0e-6
    )
    assert torch.allclose(
        full_extended.logits[:, -1:], cached_extended.logits, atol=1.0e-6
    )


def test_batched_cached_generation_compacts_finished_rows():
    class Tokenizer:
        eos_token_id = 2

        def encode_generation_prefix(self, problem, _maximum):
            return {"a": [11, 9], "b": [12, 9], "c": [13, 9]}[problem]

        def decode(self, values):
            return ",".join(str(value) for value in values)

    class Output:
        def __init__(self, tokens):
            self.logits = torch.full((len(tokens), 1, 16), -100.0)
            self.logits[torch.arange(len(tokens)), 0, torch.tensor(tokens)] = 100.0
            self.answer_logits = torch.zeros((len(tokens), 1))

    class Cache:
        def __init__(self, targets, length):
            self.targets = targets
            self.steps = [0] * len(targets)
            self.length = length

    class Model(torch.nn.Module):
        max_sequence_length = 8
        answer_head_enabled = False
        answer_min = 0

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.compactions = []

        @staticmethod
        def _output(cache):
            tokens = [
                2 if step >= target else 7
                for step, target in zip(cache.steps, cache.targets)
            ]
            return Output(tokens)

        def begin_cached_generation(self, input_ids):
            targets = [int(value) - 11 for value in input_ids[:, 0].tolist()]
            cache = Cache(targets, int(input_ids.shape[1]))
            return cache, self._output(cache)

        def cached_generation_step(self, cache, _token_ids):
            cache.steps = [value + 1 for value in cache.steps]
            cache.length += 1
            return self._output(cache)

        def compact_cached_generation(self, cache, rows):
            positions = rows.tolist()
            self.compactions.append(positions)
            cache.targets = [cache.targets[row] for row in positions]
            cache.steps = [cache.steps[row] for row in positions]
            return cache

    model = Model()
    diagnostics = []
    generations, answers = generate_math_tower(
        model, Tokenizer(), ["a", "b", "c"], max_new_tokens=6, diagnostics=diagnostics
    )

    assert generations == ["2", "7,2", "7,7,2"]
    assert answers == [None, None, None]
    assert [row["generated_tokens"] for row in diagnostics] == [1, 2, 3]
    assert all(row["eos_terminated"] for row in diagnostics)
    assert all(row["cached_batch_size"] == 3 for row in diagnostics)
    assert model.compactions == [[1, 2], [1]]


@torch.inference_mode()
def _serial_cached_generate(model, tokenizer, problems, max_new_tokens):
    """Reference implementation retained only for batched-decoder testing."""

    device = next(model.parameters()).device
    generations = []
    diagnostics = []
    for problem in problems:
        sequence = tokenizer.encode_generation_prefix(problem, model.max_sequence_length)
        prefix_length = len(sequence)
        cache, output = model.begin_cached_generation(
            torch.tensor([sequence], dtype=torch.long, device=device)
        )
        reason = "budget"
        for _ in range(max_new_tokens):
            token = int(output.logits[0, -1].argmax(dim=-1).item())
            if len(sequence) >= model.max_sequence_length:
                reason = "context_limit"
                break
            sequence.append(token)
            if token == tokenizer.eos_token_id:
                reason = "eos"
                break
            output = model.cached_generation_step(cache, token)
        generations.append(tokenizer.decode(sequence[prefix_length:]))
        diagnostics.append(
            {
                "eos_terminated": reason == "eos",
                "context_limit_hit": reason == "context_limit",
                "budget_hit": reason == "budget",
            }
        )
    return generations, diagnostics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_cached_generation_cuda_matches_serial_and_is_faster():
    torch.manual_seed(903)
    model = MathTower(_tiny_tower_config(2), ByteMathTokenizer.vocab_size).cuda().eval()
    tokenizer = ByteMathTokenizer()
    problems = [f'{{"n":{value:02d}}}' for value in range(8)]
    max_new_tokens = 96

    # Warm the CUDA kernels before timing either implementation.
    generate_math_tower(model, tokenizer, problems, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()

    started = time.perf_counter()
    serial_generations, serial_diagnostics = _serial_cached_generate(
        model, tokenizer, problems, max_new_tokens
    )
    torch.cuda.synchronize()
    serial_seconds = time.perf_counter() - started

    diagnostics = []
    started = time.perf_counter()
    batched_generations, _ = generate_math_tower(
        model,
        tokenizer,
        problems,
        max_new_tokens=max_new_tokens,
        diagnostics=diagnostics,
    )
    torch.cuda.synchronize()
    batched_seconds = time.perf_counter() - started

    assert batched_generations == serial_generations
    assert [row["eos_terminated"] for row in diagnostics] == [
        row["eos_terminated"] for row in serial_diagnostics
    ]
    assert [row["context_limit_hit"] for row in diagnostics] == [
        row["context_limit_hit"] for row in serial_diagnostics
    ]
    assert batched_seconds < serial_seconds


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
