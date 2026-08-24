from __future__ import annotations

from pathlib import Path

import pytest

from cftn_text.v1_3_inference import (
    V13InferenceEngine,
    compile_gpt_inference_prompt,
    compile_registered_gpt_prompt,
    normalize_tower_selection,
)
from tools.serve_v1_3_inference import InferenceApplication


class FakeDispatcher:
    confidence_threshold = 0.8

    def __init__(self, intent: str, confidence: float = 0.99) -> None:
        self.intent = intent
        self.confidence = confidence

    def predict_intents(self, prompts: list[str]) -> list[tuple[str, float]]:
        return [(self.intent, self.confidence) for _ in prompts]


def build_engine(intent: str):
    calls: list[tuple[str, str, int]] = []
    gpt_calls: list[tuple[str, int]] = []

    def specialist(name: str, request: str, max_new_tokens: int):
        calls.append((name, request, max_new_tokens))
        if name == "string" and (
            "count" in request.lower() or "how many times" in request.lower()
        ):
            payload = "2"
        elif name == "string" and "reverse" in request.lower():
            payload = "egdirb"
        elif name == "string":
            payload = "l"
        else:
            payload = "5" if "together with 2" in request else "6"
        return {
            "generation": f"<work>test</work><answer>{payload}</answer>",
            "payload": payload,
        }

    def generalist(prompt: str, max_new_tokens: int):
        gpt_calls.append((prompt, max_new_tokens))
        return {
            "generation": "cedar\n",
            "response": "cedar",
            "interface": "generalist_fallback_v1",
        }

    engine = V13InferenceEngine(
        FakeDispatcher(intent),
        specialist_runner=specialist,
        generalist_runner=generalist,
        device="cpu",
        artifacts={"test": True},
    )
    return engine, calls, gpt_calls


def test_parallel_plan_runs_both_specialists_and_composes_losslessly():
    engine, calls, gpt_calls = build_engine("multi_parallel")
    result = engine.infer(
        "Solve 6*x + (-5) = 31 and independently reverse 'bridge'. "
        "Return the result as x|reversed."
    )

    assert result["ok"] is True
    assert result["response"] == "6|egdirb"
    assert [call[0] for call in calls] == ["math", "string"]
    assert gpt_calls == []
    assert result["trace"]["towers"]["gpt"]["executed"] is False
    assert result["trace"]["generalist"]["status"] == "not_required"
    assert result["trace"]["composition"]["status"] == "completed"


def test_disabled_parallel_tower_is_not_called_and_trace_is_incomplete():
    engine, calls, _ = build_engine("multi_parallel")
    result = engine.infer(
        "Solve 6*x + (-5) = 31 and independently reverse 'bridge'. "
        "Return the result as x|reversed.",
        enabled_towers={"gpt": True, "math": True, "string": False},
    )

    assert result["ok"] is False
    assert result["response"] is None
    assert [call[0] for call in calls] == ["math"]
    string_call = result["trace"]["rounds"][0]["calls"][1]
    assert string_call["status"] == "skipped"
    assert string_call["reason"] == "disabled_by_user"
    assert result["trace"]["composition"]["missing_results"] == ["string_0"]


def test_disabled_dependency_prevents_later_sequential_tower_execution():
    engine, calls, _ = build_engine("string_then_math")
    result = engine.infer(
        "First count 'a' in 'callosal'. Let that count be n. "
        "Then solve 4*x+n=22. Return x.",
        enabled_towers={"gpt": True, "math": True, "string": False},
    )

    assert calls == []
    assert result["trace"]["rounds"][0]["calls"][0]["reason"] == "disabled_by_user"
    assert result["trace"]["rounds"][1]["calls"][0]["reason"] == "dependency_unavailable"


def test_sequential_result_is_inserted_into_the_next_typed_request():
    engine, calls, _ = build_engine("string_then_math")
    result = engine.infer(
        "First count 'a' in 'callosal'. Let that count be n. "
        "Then solve 4*x+n=22. Return x."
    )

    assert result["response"] == "5"
    assert calls[0][0] == "string"
    assert calls[1][0] == "math"
    assert "together with 2 gives 22" in calls[1][1]


def test_pure_language_uses_gpt_only_and_respects_disable_switch():
    prompt = (
        "The archival label is cedar. Ignore the colour red. "
        "Return the archival label."
    )
    engine, specialist_calls, gpt_calls = build_engine("pure_language")
    enabled_result = engine.infer(prompt)
    disabled_result = engine.infer(
        prompt,
        enabled_towers={"gpt": False, "math": True, "string": True},
    )

    assert enabled_result["response"] == "cedar"
    assert specialist_calls == []
    assert gpt_calls == [(prompt, 32)]
    assert disabled_result["response"] is None
    assert disabled_result["trace"]["generalist"]["reason"] == "disabled_by_user"


def test_raw_archival_prompt_compiles_to_the_proven_gpt_interface_without_metadata():
    assert compile_registered_gpt_prompt(
        "The archival label is cedar. Ignore the colour red. "
        "Return the archival label."
    ) == (
        "Archival label: cedar\n"
        "Colour: red\n"
        "Requested archival label:"
    )
    with pytest.raises(ValueError, match="outside the registered archival interface"):
        compile_registered_gpt_prompt("Summarize this text")


def test_general_prompt_uses_a_generalist_cue_instead_of_the_archival_adapter():
    prompt, interface = compile_gpt_inference_prompt("Hello")
    assert prompt == "User: Hello\nAssistant:"
    assert interface == "generalist_fallback_v1"


def test_dispatcher_rejection_falls_back_to_gpt_without_becoming_an_error():
    engine, specialist_calls, gpt_calls = build_engine("unsupported")
    result = engine.infer("Hello")

    assert result["ok"] is True
    assert result["response"] == "cedar"
    assert specialist_calls == []
    assert gpt_calls == [("Hello", 32)]
    assert result["trace"]["dispatcher"]["accepted"] is False
    assert result["trace"]["generalist"]["route_reason"] == (
        "dispatcher_rejected_generalist_fallback"
    )
    assert result["trace"]["errors"] == []
    assert "outside the validated V1.3" in result["trace"]["warnings"][-1]


def test_tower_selection_defaults_and_rejects_unknown_or_non_boolean_values():
    assert normalize_tower_selection(None) == {
        "gpt": True,
        "math": True,
        "string": True,
    }
    assert normalize_tower_selection({"math": False})["math"] is False
    with pytest.raises(ValueError, match="unknown tower"):
        normalize_tower_selection({"vision": False})
    with pytest.raises(ValueError, match="must be boolean"):
        normalize_tower_selection({"math": 0})


class FakeRuntime:
    def __init__(self) -> None:
        self.kwargs = None

    def health(self):
        return {"ok": True, "device": "cpu"}

    def infer(self, prompt, **kwargs):
        self.kwargs = (prompt, kwargs)
        return {"ok": True, "response": "ok", "trace": {}}


def test_http_application_validates_and_normalizes_request_contract():
    runtime = FakeRuntime()
    app = InferenceApplication(runtime)
    result = app.infer_payload(
        {
            "prompt": "Reverse 'abc'.",
            "towers": {"string": False},
            "generation": {
                "gpt_max_new_tokens": 12,
                "specialist_max_new_tokens": 77,
            },
        }
    )

    assert result["response"] == "ok"
    assert runtime.kwargs == (
        "Reverse 'abc'.",
        {
            "enabled_towers": {"string": False},
            "max_gpt_new_tokens": 12,
            "max_specialist_new_tokens": 77,
        },
    )
    with pytest.raises(ValueError, match="non-empty"):
        app.infer_payload({"prompt": ""})
    with pytest.raises(ValueError, match="must be an integer"):
        app.infer_payload(
            {"prompt": "x", "generation": {"gpt_max_new_tokens": True}}
        )


def test_web_console_assets_include_prompt_toggles_and_trace_surface():
    root = Path(__file__).resolve().parents[1] / "cftn_text" / "web_static"
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    assert 'id="prompt"' in html
    assert 'id="tower-gpt"' in html
    assert 'id="tower-math"' in html
    assert 'id="tower-string"' in html
    assert 'id="raw-json"' in html
    assert 'fetch("/api/infer"' in script
