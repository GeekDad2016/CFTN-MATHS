from __future__ import annotations

from .v1_3_dispatch import (
    Composition,
    DispatchError,
    DispatchPlan,
    ResultReference,
    SourceSpan,
    SpecialistCall,
    compile_specialist_request,
    compose_dispatch_results,
    compile_v1_3_intent,
    dispatch_intent_from_plan,
    dispatch_v1_3_prompt,
)


DISPATCH_INTENTS = (
    "pure_language",
    "broad_math",
    "single_math",
    "string_count",
    "string_reverse",
    "string_index",
    "multi_parallel",
    "string_then_math",
    "math_then_string",
    "unsupported",
)

CHECKPOINT_CONTRACT = "cftn_text_v2_typed_dispatch_contract_v1"


def _broad_math_plan(prompt: str) -> DispatchPlan:
    if not prompt:
        raise DispatchError("broad-math dispatch cannot receive an empty prompt")
    plan = DispatchPlan(
        prompt=prompt,
        calls=(
            SpecialistCall(
                round_index=0,
                specialist="math",
                operation="solve_native",
                arguments=(("problem", SourceSpan(0, len(prompt), prompt)),),
                result_id="math_0",
            ),
        ),
        composition=Composition("return", ("math_0",)),
        plan_kind="v2_broad_math",
    )
    plan.validate()
    return plan


def compile_v2_intent(prompt: str, intent: str) -> DispatchPlan:
    """Compile a learned V2 intent into a finite, lossless call graph.

    The model chooses only the graph.  Every operand is recovered from an
    immutable source span, and broad-math requests preserve the complete user
    prompt byte-for-byte for the already-validated V2 math interface.
    """

    prompt = str(prompt)
    intent = str(intent)
    if intent not in DISPATCH_INTENTS:
        raise DispatchError(f"unknown V2 dispatch intent: {intent}")
    if intent == "unsupported":
        raise DispatchError("V2 dispatcher rejected an unsupported prompt")
    if intent == "broad_math":
        return _broad_math_plan(prompt)
    return compile_v1_3_intent(prompt, intent)


def dispatch_v2_registered_prompt(prompt: str) -> DispatchPlan:
    """Compile the finite joint-task grammar without record metadata.

    Broad mathematical language intentionally has no permissive regex
    fallback; it must be selected by the learned dispatcher at sufficient
    confidence.  That keeps unknown natural-language requests fail-closed.
    """

    return dispatch_v1_3_prompt(str(prompt))


def dispatch_v2_intent_from_registered_prompt(prompt: str) -> str:
    return dispatch_intent_from_plan(dispatch_v2_registered_prompt(prompt))


__all__ = [
    "CHECKPOINT_CONTRACT",
    "DISPATCH_INTENTS",
    "Composition",
    "DispatchError",
    "DispatchPlan",
    "ResultReference",
    "SourceSpan",
    "SpecialistCall",
    "compile_specialist_request",
    "compile_v2_intent",
    "compose_dispatch_results",
    "dispatch_v2_intent_from_registered_prompt",
    "dispatch_v2_registered_prompt",
]
