from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_dispatch import compile_specialist_request
from cftn_text.v2_dispatch import (
    DISPATCH_INTENTS,
    DispatchError,
    compile_v2_intent,
    compose_dispatch_results,
)
from cftn_text.v2_learned_dispatch import (
    ByteIntentClassifier,
    LearnedV2Dispatcher,
    encode_dispatch_prompts,
    load_learned_dispatcher,
    save_learned_dispatcher_checkpoint,
)
from tools.run_v2_experiment import Stage, _is_complete
from tools.train_v2_dispatcher import _semantic_rows


def test_v2_broad_math_dispatch_is_an_exact_lossless_prompt_copy():
    prompt = "Differentiate 7*x^2 - 3*x + 11 with respect to x."
    plan = compile_v2_intent(prompt, "broad_math")
    plan.validate()
    call = plan.call_for(0, "math")
    assert call is not None
    assert call.operation == "solve_native"
    assert compile_specialist_request(plan, call, {}) == prompt
    assert compose_dispatch_results(plan, {"math_0": "14*x-3"}) == "14*x-3"
    payload = plan.to_dict()["calls"][0]["arguments"]["problem"]
    assert payload == {
        "kind": "source_span",
        "start": 0,
        "end": len(prompt),
        "text": prompt,
    }


@pytest.mark.parametrize(
    ("intent", "prompt", "expected_specialists"),
    [
        ("single_math", "Find x when 4 times x plus -3 equals 17.", ["math"]),
        ("string_reverse", "Output the symbols of 'forest' in opposite order.", ["string"]),
        (
            "multi_parallel",
            "Independently solve 4*x+(-3)=17 and reverse 'forest'; emit x|reversal.",
            ["math", "string"],
        ),
        (
            "string_then_math",
            "First tally 'o' in 'forest' and call the count n; then solve 4*x+n=18.",
            ["string", "math"],
        ),
        (
            "math_then_string",
            "Solve 4*x+(-3)=17; use x as a zero-based offset into 'forest'.",
            ["math", "string"],
        ),
    ],
)
def test_v2_registered_graphs_preserve_finite_specialist_dependencies(
    intent: str, prompt: str, expected_specialists: list[str]
):
    plan = compile_v2_intent(prompt, intent)
    plan.validate()
    assert [call.specialist for call in plan.calls] == expected_specialists


def test_v2_unsupported_intent_fails_closed():
    with pytest.raises(DispatchError, match="rejected an unsupported"):
        compile_v2_intent("Translate this greeting into French.", "unsupported")


def test_v2_semantic_training_controls_cover_every_intent_without_oracle_fields():
    rows = _semantic_rows(count=100, seed=41, heldout=False)
    assert {intent for _, intent in rows} == set(DISPATCH_INTENTS)
    assert all(isinstance(prompt, str) and prompt for prompt, _ in rows)


def test_v2_dispatch_encoding_is_value_invariant_for_source_operands():
    first = "Solve 4*x+(-3)=17; use x as a zero-based offset into 'forest'."
    second = "Solve 9*x+(12)=48; use x as a zero-based offset into 'planet'."
    ids, masks = encode_dispatch_prompts([first, second], maximum_length=256)
    assert torch.equal(masks[0], masks[1])
    assert torch.equal(ids[0], ids[1])


def test_v2_dispatcher_checkpoint_is_versioned_and_round_trips(tmp_path: Path):
    torch.manual_seed(7)
    model = ByteIntentClassifier(embedding_size=8, channels=8, kernels=(3,))
    checkpoint = tmp_path / "dispatcher.pth"
    save_learned_dispatcher_checkpoint(
        checkpoint,
        model,
        maximum_length=256,
        confidence_threshold=0.9,
        metadata={"purpose": "test"},
    )
    loaded = load_learned_dispatcher(checkpoint)
    assert loaded.metadata == {"purpose": "test"}
    direct = LearnedV2Dispatcher(
        model, maximum_length=256, confidence_threshold=0.9
    )
    prompts = [
        "Find x when 4 times x plus -3 equals 17.",
        "Differentiate 7*x^2 with respect to x.",
    ]
    assert loaded.predict_intents(prompts) == direct.predict_intents(prompts)


def test_v2_dispatcher_structural_mask_keeps_rejection_and_broad_math_eligible():
    model = ByteIntentClassifier(embedding_size=8, channels=8, kernels=(3,))
    ids, mask = encode_dispatch_prompts(
        ["Reverse 'forest'."], maximum_length=128
    )
    logits = model(ids, mask)
    assert logits[0, DISPATCH_INTENTS.index("string_reverse")] > -1e3
    assert logits[0, DISPATCH_INTENTS.index("broad_math")] > -1e3
    assert logits[0, DISPATCH_INTENTS.index("unsupported")] > -1e3
    assert logits[0, DISPATCH_INTENTS.index("single_math")] < -1e3


def test_v2_dispatch_stage_completion_checks_hash_and_every_gate(tmp_path: Path):
    checkpoint = tmp_path / "dispatcher.pth"
    checkpoint.write_bytes(b"sealed")
    summary = {
        "state": "passed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "acceptance": {"gates": {"registered": True, "pass": True}},
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    stage = Stage("train_learned_dispatcher", [], summary_path)
    assert _is_complete(stage, {})
    summary["acceptance"]["gates"]["pass"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert not _is_complete(stage, {})


def test_v2_native_dispatch_completion_requires_no_oracle_and_determinism(
    tmp_path: Path,
):
    report = {
        "format": "cftn_text_v2_native_typed_dispatch_evaluation_v1",
        "state": "passed",
        "oracle_metadata_visible_to_runtime": False,
        "deterministic_answer_composition": True,
        "acceptance": {"gates": {"pass": True}},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    stage = Stage("evaluate_native_typed_dispatch", [], path)
    assert _is_complete(stage, {})
    report["oracle_metadata_visible_to_runtime"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not _is_complete(stage, {})
