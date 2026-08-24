from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .data_generator import file_sha256
from .tokenizer import ByteMathTokenizer
from .v1_3_answer_bus import extract_answer_payload
from .v1_3_config import load_v1_3_config
from .v1_3_dispatch import (
    DispatchError,
    DispatchPlan,
    ResultReference,
    compile_specialist_request,
    compile_v1_3_intent,
    compose_dispatch_results,
)
from .v1_3_evaluation import generate_native_specialist
from .v1_3_learned_dispatch import LearnedV13Dispatcher, load_learned_dispatcher
from .v1_3_training import build_v1_3_model


TOWER_NAMES = ("gpt", "math", "string")
SPECIALIST_NAMES = ("math", "string")
DEFAULT_MAX_GPT_NEW_TOKENS = 32
DEFAULT_MAX_SPECIALIST_NEW_TOKENS = 96


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_nonempty_line(text: str) -> str | None:
    for line in str(text).splitlines():
        value = line.strip()
        if value:
            return value
    return None


def compile_registered_gpt_prompt(prompt: str) -> str:
    """Reconstruct the proven GPT interface from the raw archival prompt.

    This copies the label from an immutable source span. It does not read a
    record, target, task class, or other oracle metadata.
    """

    match = re.fullmatch(
        r"The archival label is (?P<label>[a-z]+)\. "
        r"Ignore the colour red\. Return the archival label\.",
        str(prompt),
    )
    if match is None:
        raise DispatchError(
            "pure-language prompt is outside the registered archival interface"
        )
    label = match.group("label")
    return (
        f"Archival label: {label}\n"
        "Colour: red\n"
        "Requested archival label:"
    )


def compile_gpt_inference_prompt(prompt: str) -> tuple[str, str]:
    """Select the proven archival adapter or a general GPT continuation cue."""

    try:
        return compile_registered_gpt_prompt(prompt), "registered_archival_v1"
    except DispatchError:
        return f"User: {str(prompt).strip()}\nAssistant:", "generalist_fallback_v1"


def normalize_tower_selection(value: Mapping[str, Any] | None) -> dict[str, bool]:
    if value is None:
        return {name: True for name in TOWER_NAMES}
    unknown = sorted(set(value).difference(TOWER_NAMES))
    if unknown:
        raise ValueError(f"unknown tower selection: {', '.join(unknown)}")
    result: dict[str, bool] = {}
    for name in TOWER_NAMES:
        selected = value.get(name, True)
        if not isinstance(selected, bool):
            raise ValueError(f"tower selection {name!r} must be boolean")
        result[name] = selected
    return result


class V13InferenceEngine:
    """Execute the accepted typed V1.3 route and retain a complete trace.

    The dispatcher chooses a finite call graph. Exact operands are copied from
    the prompt by the constrained compiler. Specialist outputs travel on the
    typed answer bus and are composed deterministically. The earlier latent
    wake/halt path is deliberately not invoked by this accepted runtime.
    """

    def __init__(
        self,
        dispatcher: LearnedV13Dispatcher,
        *,
        specialist_runner: Callable[[str, str, int], Mapping[str, Any]],
        generalist_runner: Callable[[str, int], Mapping[str, Any]],
        device: str,
        artifacts: Mapping[str, Any] | None = None,
        maximum_rounds: int = 3,
    ) -> None:
        self.dispatcher = dispatcher
        self.specialist_runner = specialist_runner
        self.generalist_runner = generalist_runner
        self.device = str(device)
        self.artifacts = dict(artifacts or {})
        self.maximum_rounds = int(maximum_rounds)
        if self.maximum_rounds < 1:
            raise ValueError("maximum_rounds must be positive")

    def infer(
        self,
        prompt: str,
        *,
        enabled_towers: Mapping[str, Any] | None = None,
        max_gpt_new_tokens: int = DEFAULT_MAX_GPT_NEW_TOKENS,
        max_specialist_new_tokens: int = DEFAULT_MAX_SPECIALIST_NEW_TOKENS,
    ) -> dict[str, Any]:
        prompt = str(prompt)
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        if max_gpt_new_tokens < 1 or max_gpt_new_tokens > 128:
            raise ValueError("max_gpt_new_tokens must be within 1..128")
        if max_specialist_new_tokens < 1 or max_specialist_new_tokens > 256:
            raise ValueError("max_specialist_new_tokens must be within 1..256")
        enabled = normalize_tower_selection(enabled_towers)
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "request_id": uuid.uuid4().hex,
            "started_at": utc_now(),
            "runtime": {
                "device": self.device,
                "routing_mode": "learned_typed_dispatch_v1",
                "answer_path": "lossless_typed_bus_deterministic_composition",
                "legacy_latent_wake_gates": "bypassed_by_accepted_runtime",
                "maximum_rounds": self.maximum_rounds,
            },
            "prompt": prompt,
            "towers": {
                name: {
                    "enabled": enabled[name],
                    "executed": False,
                    "role": (
                        "generalist_fallback"
                        if name == "gpt"
                        else "typed_specialist"
                    ),
                }
                for name in TOWER_NAMES
            },
            "dispatcher": {},
            "rounds": [],
            "composition": {},
            "generalist": {"executed": False},
            "artifacts": self.artifacts,
            "warnings": [],
            "errors": [],
        }

        response: str | None = None

        def execute_generalist(route_reason: str) -> None:
            nonlocal response
            if not enabled["gpt"]:
                trace["generalist"] = {
                    "executed": False,
                    "status": "skipped",
                    "reason": "disabled_by_user",
                    "route_reason": route_reason,
                }
                trace["warnings"].append(
                    "The GPT fallback is required for this prompt but is disabled."
                )
                return
            generalist_started = time.perf_counter()
            try:
                generalist = dict(
                    self.generalist_runner(prompt, int(max_gpt_new_tokens))
                )
                response = generalist.get("response")
                if response is None or not str(response):
                    raise RuntimeError("GPT produced no non-empty completion line")
                response = str(response)
                trace["towers"]["gpt"]["executed"] = True
                trace["generalist"] = {
                    "executed": True,
                    "status": "completed",
                    "route_reason": route_reason,
                    "response": response,
                    **{
                        key: value
                        for key, value in generalist.items()
                        if key != "response"
                    },
                }
                if generalist.get("interface") == "generalist_fallback_v1":
                    trace["warnings"].append(
                        "This response used the open-world GPT fallback and is "
                        "outside the validated V1.3 specialist task panels."
                    )
            except Exception as exc:  # Preserve a trace for model failures.
                trace["generalist"] = {
                    "executed": True,
                    "status": "error",
                    "route_reason": route_reason,
                    "error": str(exc),
                }
                trace["errors"].append(
                    {"stage": "generalist", "message": str(exc)}
                )
            trace["generalist"]["elapsed_ms"] = round(
                (time.perf_counter() - generalist_started) * 1000.0, 3
            )

        plan: DispatchPlan | None = None
        try:
            intent, confidence = self.dispatcher.predict_intents([prompt])[0]
            threshold = float(self.dispatcher.confidence_threshold)
            trace["dispatcher"] = {
                "executed": True,
                "intent": intent,
                "confidence": confidence,
                "confidence_threshold": threshold,
                "accepted": confidence >= threshold,
            }
            if confidence < threshold:
                raise DispatchError(
                    f"learned dispatcher confidence {confidence:.6f} is below "
                    f"{threshold:.6f}"
                )
            plan = compile_v1_3_intent(prompt, intent)
            if plan.calls and max(call.round_index for call in plan.calls) >= self.maximum_rounds:
                raise DispatchError("dispatch plan exceeds the configured round limit")
            trace["dispatcher"]["plan"] = plan.to_dict()
        except (DispatchError, ValueError) as exc:
            trace["dispatcher"].setdefault("executed", True)
            trace["dispatcher"]["accepted"] = False
            trace["dispatcher"]["error"] = str(exc)
            trace["warnings"].append(
                "The typed specialist dispatcher did not accept this prompt; "
                "using the GPT generalist fallback."
            )

        results: dict[str, str] = {}
        if plan is None:
            trace["composition"] = {
                "kind": "none",
                "result_ids": [],
                "separator": "",
                "status": "not_applicable",
                "reason": "dispatcher_rejected_generalist_fallback",
            }
            execute_generalist("dispatcher_rejected_generalist_fallback")
        else:
            for round_index in range(self.maximum_rounds):
                calls = [call for call in plan.calls if call.round_index == round_index]
                if not calls:
                    continue
                round_started = time.perf_counter()
                round_trace: dict[str, Any] = {
                    "round": round_index,
                    "calls": [],
                }
                for call in calls:
                    call_started = time.perf_counter()
                    call_trace: dict[str, Any] = {
                        "specialist": call.specialist,
                        "operation": call.operation,
                        "result_id": call.result_id,
                        "status": "pending",
                    }
                    dependencies = sorted(
                        {
                            value.result_id
                            for _, value in call.arguments
                            if isinstance(value, ResultReference)
                        }
                    )
                    call_trace["dependencies"] = dependencies
                    if not enabled[call.specialist]:
                        call_trace.update(
                            status="skipped",
                            reason="disabled_by_user",
                        )
                    elif any(result_id not in results for result_id in dependencies):
                        call_trace.update(
                            status="skipped",
                            reason="dependency_unavailable",
                            missing_dependencies=[
                                result_id
                                for result_id in dependencies
                                if result_id not in results
                            ],
                        )
                    else:
                        try:
                            request = compile_specialist_request(plan, call, results)
                            call_trace["compiled_request"] = request
                            generated = dict(
                                self.specialist_runner(
                                    call.specialist,
                                    request,
                                    int(max_specialist_new_tokens),
                                )
                            )
                            payload = generated.get("payload")
                            if payload is None or not str(payload):
                                raise RuntimeError(
                                    f"{call.specialist} produced no complete <answer> payload"
                                )
                            payload = str(payload)
                            results[call.result_id] = payload
                            trace["towers"][call.specialist]["executed"] = True
                            call_trace.update(
                                status="completed",
                                payload=payload,
                                generation=str(generated.get("generation", "")),
                            )
                            for key, value in generated.items():
                                if key not in {"payload", "generation"}:
                                    call_trace[key] = value
                        except Exception as exc:  # Preserve a trace for model failures.
                            call_trace.update(status="error", error=str(exc))
                            trace["errors"].append(
                                {
                                    "stage": f"round_{round_index}.{call.specialist}",
                                    "message": str(exc),
                                }
                            )
                    call_trace["elapsed_ms"] = round(
                        (time.perf_counter() - call_started) * 1000.0, 3
                    )
                    round_trace["calls"].append(call_trace)
                round_trace["elapsed_ms"] = round(
                    (time.perf_counter() - round_started) * 1000.0, 3
                )
                trace["rounds"].append(round_trace)

            if plan.composition.kind == "none":
                trace["composition"] = {
                    **plan.composition.to_dict(),
                    "status": "not_applicable",
                }
                execute_generalist("dispatcher_selected_pure_language")
            else:
                response = compose_dispatch_results(plan, results)
                trace["generalist"] = {
                    "executed": False,
                    "status": "not_required",
                    "reason": "typed_specialist_plan_uses_deterministic_composition",
                }
                missing = [
                    result_id
                    for result_id in plan.composition.result_ids
                    if result_id not in results
                ]
                trace["composition"] = {
                    **plan.composition.to_dict(),
                    "status": "completed" if response is not None else "incomplete",
                    "available_results": dict(results),
                    "missing_results": missing,
                    "output": response,
                }
                if missing:
                    trace["warnings"].append(
                        "The response is incomplete because one or more required "
                        "tower results are unavailable."
                    )

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        trace["elapsed_ms"] = elapsed_ms
        return {
            "ok": response is not None and not trace["errors"],
            "response": response,
            "trace": trace,
        }


class LoadedV13Runtime:
    """Persistent artifact-backed runtime used by the LAN HTTP service."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        collaboration_checkpoint: str | Path,
        dispatcher_checkpoint: str | Path,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.config_path = Path(config_path).expanduser().resolve()
        self.collaboration_checkpoint = Path(collaboration_checkpoint).expanduser().resolve()
        self.dispatcher_checkpoint = Path(dispatcher_checkpoint).expanduser().resolve()
        for path in (
            self.config_path,
            self.collaboration_checkpoint,
            self.dispatcher_checkpoint,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.config = load_v1_3_config(self.config_path)
        answer_bus_contract = (
            Path(self.config["paths"]["artifact_root"])
            / "answer_bus_recovery_contract.json"
        )
        if not answer_bus_contract.is_file():
            raise FileNotFoundError(answer_bus_contract)
        contract = json.loads(answer_bus_contract.read_text(encoding="utf-8"))
        if contract.get("format") != "cftn_text_v1_3_answer_bus_recovery_contract_v1":
            raise RuntimeError("answer-bus recovery contract format is invalid")
        if contract.get("revision_sha256") != self.config["_meta"]["sha256"]:
            raise RuntimeError("answer-bus contract and sealed configuration disagree")
        composer = contract.get("answer_composer")
        if not isinstance(composer, dict):
            raise RuntimeError("answer-bus contract has no composer dimensions")
        self.config["answer_composer"] = dict(composer)
        self.model, self.gpt_tokenizer, provenance = build_v1_3_model(
            self.config,
            device=self.device,
            collaboration_checkpoint=self.collaboration_checkpoint,
        )
        self.model.eval()
        self.dispatcher = load_learned_dispatcher(
            self.dispatcher_checkpoint, device=self.device
        )
        self.byte_tokenizer = ByteMathTokenizer()
        self.lock = threading.Lock()
        self.loaded_at = utc_now()
        self.artifacts = {
            **provenance,
            "config": {
                "path": str(self.config_path),
                "sha256": self.config["_meta"]["sha256"],
            },
            "collaboration_checkpoint": {
                "path": str(self.collaboration_checkpoint),
                "sha256": file_sha256(self.collaboration_checkpoint),
            },
            "dispatcher_checkpoint": {
                "path": str(self.dispatcher_checkpoint),
                "sha256": file_sha256(self.dispatcher_checkpoint),
            },
            "answer_bus_contract": {
                "path": str(answer_bus_contract),
                "sha256": file_sha256(answer_bus_contract),
            },
        }
        self.engine = V13InferenceEngine(
            self.dispatcher,
            specialist_runner=self._run_specialist,
            generalist_runner=self._run_generalist,
            device=str(self.device),
            artifacts=self.artifacts,
            maximum_rounds=int(self.config["runtime"]["maximum_callosal_rounds"]),
        )

    def _run_specialist(
        self, specialist: str, request: str, max_new_tokens: int
    ) -> Mapping[str, Any]:
        if specialist not in self.model.specialists:
            raise ValueError(f"unknown specialist: {specialist}")
        started = time.perf_counter()
        generation = generate_native_specialist(
            self.model.specialists[specialist],
            [{"problem": request}],
            self.byte_tokenizer,
            device=self.device,
            max_new_tokens=int(max_new_tokens),
        )[0]
        payload = extract_answer_payload(generation, strict=False)
        return {
            "generation": generation,
            "payload": payload,
            "model_elapsed_ms": round(
                (time.perf_counter() - started) * 1000.0, 3
            ),
        }

    def _run_generalist(self, prompt: str, max_new_tokens: int) -> Mapping[str, Any]:
        registered_prompt, interface = compile_gpt_inference_prompt(prompt)
        prefix = self.gpt_tokenizer.encode(
            registered_prompt, add_special_tokens=False
        )
        message_tokens = int(self.config["bridge"]["message_tokens"])
        message_width = int(self.config["bridge"]["message_width"])
        empty_message = torch.zeros(
            (1, message_tokens, message_width),
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )
        started = time.perf_counter()
        generated_ids = self.model.gpt_tower.generate_greedy(
            [prefix],
            empty_message,
            int(self.gpt_tokenizer.eos_token_id),
            int(max_new_tokens),
            receive_enabled=False,
        )[0]
        raw = self.gpt_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {
            "registered_prompt": registered_prompt,
            "interface": interface,
            "generation": raw,
            "response": first_nonempty_line(raw),
            "model_elapsed_ms": round(
                (time.perf_counter() - started) * 1000.0, 3
            ),
        }

    def infer(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            return self.engine.infer(*args, **kwargs)

    def health(self) -> dict[str, Any]:
        gpu: dict[str, Any] | None = None
        if self.device.type == "cuda":
            index = self.device.index or torch.cuda.current_device()
            gpu = {
                "name": torch.cuda.get_device_name(index),
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
            }
        return {
            "ok": True,
            "loaded_at": self.loaded_at,
            "device": str(self.device),
            "gpu": gpu,
            "towers": {
                "gpt": {"available": True, "role": "generalist_fallback"},
                "math": {"available": True, "role": "typed_specialist"},
                "string": {"available": True, "role": "typed_specialist"},
            },
            "routing_mode": "learned_typed_dispatch_v1",
            "legacy_latent_wake_gates": "bypassed_by_accepted_runtime",
            "artifacts": self.artifacts,
        }
