from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Union


SPECIALIST_OPERATIONS = {
    # ``solve_native`` is the version-neutral passthrough used by V2's broad
    # math dispatcher.  V1.3 never emits it, so adding it is checkpoint- and
    # behavior-compatible with the sealed V1.3 grammar.
    "math": frozenset({"solve_linear", "solve_native"}),
    "string": frozenset(
        {"length", "count", "index", "reverse", "contains", "substitute"}
    ),
}

DISPATCH_INTENTS = (
    "pure_language",
    "single_math",
    "string_count",
    "string_reverse",
    "string_index",
    "multi_parallel",
    "string_then_math",
    "math_then_string",
    "unsupported",
)


class DispatchError(ValueError):
    """Raised when a raw prompt cannot be compiled into a safe typed call."""


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    expected: str

    def resolve(self, prompt: str) -> str:
        if self.start < 0 or self.end <= self.start or self.end > len(prompt):
            raise DispatchError("dispatcher source span is outside the prompt")
        value = prompt[self.start : self.end]
        if value != self.expected:
            raise DispatchError("dispatcher source span no longer matches the prompt")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "source_span",
            "start": self.start,
            "end": self.end,
            "text": self.expected,
        }


@dataclass(frozen=True)
class ResultReference:
    result_id: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": "result_reference", "result_id": self.result_id}


DispatchValue = Union[SourceSpan, ResultReference]


@dataclass(frozen=True)
class SpecialistCall:
    round_index: int
    specialist: str
    operation: str
    arguments: tuple[tuple[str, DispatchValue], ...]
    result_id: str

    def argument(self, name: str) -> DispatchValue:
        for key, value in self.arguments:
            if key == name:
                return value
        raise DispatchError(f"typed call has no {name!r} argument")

    def to_dict(self) -> dict[str, object]:
        return {
            "round": self.round_index,
            "specialist": self.specialist,
            "operation": self.operation,
            "arguments": {
                name: value.to_dict() for name, value in self.arguments
            },
            "result_id": self.result_id,
        }


@dataclass(frozen=True)
class Composition:
    kind: str
    result_ids: tuple[str, ...] = ()
    separator: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "result_ids": list(self.result_ids),
            "separator": self.separator,
        }


@dataclass(frozen=True)
class DispatchPlan:
    prompt: str
    calls: tuple[SpecialistCall, ...]
    composition: Composition
    plan_kind: str

    def call_for(self, round_index: int, specialist: str) -> SpecialistCall | None:
        matches = [
            call
            for call in self.calls
            if call.round_index == round_index and call.specialist == specialist
        ]
        if len(matches) > 1:
            raise DispatchError("dispatch plan has duplicate specialist calls in one round")
        return matches[0] if matches else None

    def validate(self) -> None:
        result_rounds: dict[str, int] = {}
        occupied: set[tuple[int, str]] = set()
        for call in self.calls:
            if call.round_index < 0:
                raise DispatchError("dispatch round cannot be negative")
            if call.specialist not in SPECIALIST_OPERATIONS:
                raise DispatchError(f"unknown specialist: {call.specialist}")
            if call.operation not in SPECIALIST_OPERATIONS[call.specialist]:
                raise DispatchError(
                    f"operation {call.operation!r} is invalid for {call.specialist}"
                )
            if not call.result_id or call.result_id in result_rounds:
                raise DispatchError("dispatch result IDs must be non-empty and unique")
            key = (call.round_index, call.specialist)
            if key in occupied:
                raise DispatchError("dispatch plan has duplicate specialist calls in one round")
            occupied.add(key)
            for _, value in call.arguments:
                if isinstance(value, SourceSpan):
                    value.resolve(self.prompt)
                elif value.result_id not in result_rounds:
                    raise DispatchError("dispatch dependency must refer to an earlier result")
                elif result_rounds[value.result_id] >= call.round_index:
                    raise DispatchError("dispatch dependency must come from an earlier round")
            result_rounds[call.result_id] = call.round_index
        if self.composition.kind == "none":
            if self.composition.result_ids:
                raise DispatchError("empty composition cannot reference results")
        elif self.composition.kind in {"return", "join"}:
            if not self.composition.result_ids:
                raise DispatchError("composition has no result references")
            if self.composition.kind == "return" and len(self.composition.result_ids) != 1:
                raise DispatchError("return composition must reference exactly one result")
            for result_id in self.composition.result_ids:
                if result_id not in result_rounds:
                    raise DispatchError("composition references an unknown result")
        else:
            raise DispatchError(f"unknown composition kind: {self.composition.kind}")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "cftn_text_v1_3_dispatch_plan_v1",
            "plan_kind": self.plan_kind,
            "calls": [call.to_dict() for call in self.calls],
            "composition": self.composition.to_dict(),
        }


_INTEGER = r"[+-]?\d+"
_TEXT = r"[^']+"


def _span(match: re.Match[str], name: str) -> SourceSpan:
    start, end = match.span(name)
    return SourceSpan(start=start, end=end, expected=match.group(name))


def _call(
    *,
    round_index: int,
    specialist: str,
    operation: str,
    result_id: str,
    **arguments: DispatchValue,
) -> SpecialistCall:
    return SpecialistCall(
        round_index=round_index,
        specialist=specialist,
        operation=operation,
        arguments=tuple(arguments.items()),
        result_id=result_id,
    )


def _plan(
    prompt: str,
    plan_kind: str,
    calls: tuple[SpecialistCall, ...],
    composition: Composition,
) -> DispatchPlan:
    plan = DispatchPlan(
        prompt=prompt,
        calls=calls,
        composition=composition,
        plan_kind=plan_kind,
    )
    plan.validate()
    return plan


def dispatch_v1_3_prompt(prompt: str) -> DispatchPlan:
    """Compile the registered V1.3 prompt grammar without record metadata.

    Exact operands are represented as validated spans into ``prompt``. The
    parser never reads task labels, oracle prompts, wake targets, or answers.
    Unknown or ambiguous prompts fail closed.
    """

    prompt = str(prompt)
    patterns: tuple[tuple[str, str], ...] = (
        (
            "multi_parallel",
            rf"Solve\s+(?P<a>{_INTEGER})\*x\s*\+\s*\((?P<b>{_INTEGER})\)\s*=\s*(?P<c>{_INTEGER})\s+"
            rf"and independently reverse\s+'(?P<text>{_TEXT})'\.\s*"
            r"Return the result as x\|reversed\.",
        ),
        (
            "string_then_math",
            rf"First count\s+'(?P<char>{_TEXT})'\s+in\s+'(?P<text>{_TEXT})'\.\s*"
            rf"Let that count be n\.\s*Then solve\s+(?P<a>{_INTEGER})\*x\+n=(?P<c>{_INTEGER})\.\s*Return x\.",
        ),
        (
            "math_then_string",
            rf"Solve\s+(?P<a>{_INTEGER})\*x\+\((?P<b>{_INTEGER})\)=(?P<c>{_INTEGER})\.\s*"
            rf"Use x as a zero-based index into\s+'(?P<text>{_TEXT})',\s*"
            r"then return the selected character\.",
        ),
        (
            "explicit_math",
            rf"Solve\s+(?P<a>{_INTEGER})\*x\s*\+\s*\((?P<b>{_INTEGER})\)\s*=\s*(?P<c>{_INTEGER})\.\s*Return x\.",
        ),
        (
            "language_math_train",
            rf"Mira thinks of an integer\.\s*Multiplying it by\s+(?P<a>{_INTEGER})\s+"
            rf"and then adding\s+(?P<b>{_INTEGER})\s+gives\s+(?P<c>{_INTEGER})\.\s*"
            r"What integer did Mira choose\?",
        ),
        (
            "language_math_heldout",
            rf"A latent quantity is scaled by\s+(?P<a>{_INTEGER}),\s*translated by\s+(?P<b>{_INTEGER}),\s*"
            rf"and arrives at\s+(?P<c>{_INTEGER})\.\s*Recover the latent quantity;\s*"
            r"the note about seven lanterns is irrelevant\.",
        ),
    )
    for plan_kind, pattern in patterns:
        match = re.fullmatch(pattern, prompt)
        if match is None:
            continue
        if plan_kind == "multi_parallel":
            calls = (
                _call(
                    round_index=0,
                    specialist="math",
                    operation="solve_linear",
                    result_id="math_0",
                    a=_span(match, "a"),
                    b=_span(match, "b"),
                    c=_span(match, "c"),
                ),
                _call(
                    round_index=0,
                    specialist="string",
                    operation="reverse",
                    result_id="string_0",
                    text=_span(match, "text"),
                ),
            )
            return _plan(
                prompt,
                plan_kind,
                calls,
                Composition("join", ("math_0", "string_0"), "|"),
            )
        if plan_kind == "string_then_math":
            calls = (
                _call(
                    round_index=0,
                    specialist="string",
                    operation="count",
                    result_id="string_0",
                    character=_span(match, "char"),
                    text=_span(match, "text"),
                ),
                _call(
                    round_index=1,
                    specialist="math",
                    operation="solve_linear",
                    result_id="math_1",
                    a=_span(match, "a"),
                    b=ResultReference("string_0"),
                    c=_span(match, "c"),
                ),
            )
            return _plan(
                prompt, plan_kind, calls, Composition("return", ("math_1",))
            )
        if plan_kind == "math_then_string":
            calls = (
                _call(
                    round_index=0,
                    specialist="math",
                    operation="solve_linear",
                    result_id="math_0",
                    a=_span(match, "a"),
                    b=_span(match, "b"),
                    c=_span(match, "c"),
                ),
                _call(
                    round_index=1,
                    specialist="string",
                    operation="index",
                    result_id="string_1",
                    text=_span(match, "text"),
                    index=ResultReference("math_0"),
                ),
            )
            return _plan(
                prompt, plan_kind, calls, Composition("return", ("string_1",))
            )
        call = _call(
            round_index=0,
            specialist="math",
            operation="solve_linear",
            result_id="math_0",
            a=_span(match, "a"),
            b=_span(match, "b"),
            c=_span(match, "c"),
        )
        return _plan(
            prompt, plan_kind, (call,), Composition("return", ("math_0",))
        )

    string_patterns: tuple[tuple[str, str], ...] = (
        ("length", rf"How many characters are in\s+'(?P<text>{_TEXT})'\?"),
        ("length", rf"Return the character length of\s+'(?P<text>{_TEXT})'\."),
        ("length", rf"Determine the cardinality of the character sequence\s+'(?P<text>{_TEXT})'\."),
        ("count", rf"How many times does\s+'(?P<char>{_TEXT})'\s+occur in\s+'(?P<text>{_TEXT})'\?"),
        ("count", rf"Count the character\s+'(?P<char>{_TEXT})'\s+in\s+'(?P<text>{_TEXT})'\."),
        ("count", rf"What is the frequency of symbol\s+'(?P<char>{_TEXT})'\s+within\s+'(?P<text>{_TEXT})'\?"),
        ("index", rf"Using zero-based indexing, which character is at position\s+(?P<index>{_INTEGER})\s+in\s+'(?P<text>{_TEXT})'\?"),
        ("index", rf"Return character\s+(?P<index>{_INTEGER})\s+of\s+'(?P<text>{_TEXT})'\s+when the first position is zero\."),
        ("index", rf"Indexing from zero, extract offset\s+(?P<index>{_INTEGER})\s+from\s+'(?P<text>{_TEXT})'\."),
        ("reverse", rf"Reverse\s+'(?P<text>{_TEXT})'\."),
        ("reverse", rf"Write\s+'(?P<text>{_TEXT})'\s+backwards\."),
        ("reverse", rf"Emit the mirror ordering of the symbols in\s+'(?P<text>{_TEXT})'\."),
        ("contains", rf"Does\s+'(?P<text>{_TEXT})'\s+contain the exact substring\s+'(?P<substring>{_TEXT})'\?\s*Answer yes or no\."),
        ("contains", rf"Return yes if\s+'(?P<substring>{_TEXT})'\s+occurs contiguously in\s+'(?P<text>{_TEXT})',\s*otherwise no\."),
        ("contains", rf"Is\s+'(?P<substring>{_TEXT})'\s+a contiguous infix of\s+'(?P<text>{_TEXT})'\?\s*Reply yes or no\."),
        ("substitute", rf"In\s+'(?P<text>{_TEXT})',\s*replace every\s+'(?P<char>{_TEXT})'\s+with\s+'(?P<replacement>{_TEXT})'\."),
        ("substitute", rf"Substitute\s+'(?P<replacement>{_TEXT})'\s+for all\s+'(?P<char>{_TEXT})'\s+characters in\s+'(?P<text>{_TEXT})'\."),
        ("substitute", rf"Map each\s+'(?P<char>{_TEXT})'\s+in\s+'(?P<text>{_TEXT})'\s+to\s+'(?P<replacement>{_TEXT})'\."),
    )
    for operation, pattern in string_patterns:
        match = re.fullmatch(pattern, prompt)
        if match is None:
            continue
        group_names = {
            name for name, value in match.groupdict().items() if value is not None
        }
        arguments = {name: _span(match, name) for name in group_names}
        call = _call(
            round_index=0,
            specialist="string",
            operation=operation,
            result_id="string_0",
            **arguments,
        )
        return _plan(
            prompt,
            f"exact_string_{operation}",
            (call,),
            Composition("return", ("string_0",)),
        )

    if re.fullmatch(
        r"The archival label is [a-z]+\. Ignore the colour red\. Return the archival label\.",
        prompt,
    ):
        return _plan(prompt, "pure_language", (), Composition("none"))

    raise DispatchError("prompt is outside the registered V1.3 dispatch grammar")


def dispatch_intent_from_plan(plan: DispatchPlan) -> str:
    if plan.plan_kind in {
        "explicit_math",
        "language_math_train",
        "language_math_heldout",
    }:
        return "single_math"
    if plan.plan_kind.startswith("exact_string_"):
        intent = plan.plan_kind.removeprefix("exact_string_")
        resolved = f"string_{intent}"
        if resolved in DISPATCH_INTENTS:
            return resolved
    if plan.plan_kind in DISPATCH_INTENTS:
        return plan.plan_kind
    raise DispatchError(f"dispatch plan has no learned intent: {plan.plan_kind}")


def _lexical_spans(pattern: str, prompt: str, group: str) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    for match in re.finditer(pattern, prompt):
        start, end = match.span(group)
        spans.append(SourceSpan(start, end, match.group(group)))
    return spans


def compile_v1_3_intent(prompt: str, intent: str) -> DispatchPlan:
    """Compile a learned intent using only exact lexical spans from ``prompt``.

    The learned component selects a finite call graph. Operands are never
    generated: quoted values and signed integers are copied from the raw prompt
    and validated by ``DispatchPlan``.
    """

    prompt = str(prompt)
    if intent not in DISPATCH_INTENTS:
        raise DispatchError(f"unknown learned dispatch intent: {intent}")
    if intent == "unsupported":
        raise DispatchError("learned dispatcher rejected an unsupported prompt")
    quoted = _lexical_spans(r"'(?P<value>[^']+)'", prompt, "value")
    integers = _lexical_spans(
        r"(?<![A-Za-z0-9])(?P<value>[+-]?\d+)(?![A-Za-z0-9])",
        prompt,
        "value",
    )

    def require(values: list[SourceSpan], count: int, label: str) -> None:
        if len(values) != count:
            raise DispatchError(
                f"{intent} requires {count} {label} span(s), found {len(values)}"
            )

    if intent == "pure_language":
        require(quoted, 0, "quoted")
        require(integers, 0, "integer")
        return _plan(prompt, intent, (), Composition("none"))
    if intent == "single_math":
        require(quoted, 0, "quoted")
        require(integers, 3, "integer")
        call = _call(
            round_index=0,
            specialist="math",
            operation="solve_linear",
            result_id="math_0",
            a=integers[0],
            b=integers[1],
            c=integers[2],
        )
        return _plan(prompt, intent, (call,), Composition("return", ("math_0",)))
    if intent == "string_count":
        require(quoted, 2, "quoted")
        require(integers, 0, "integer")
        call = _call(
            round_index=0,
            specialist="string",
            operation="count",
            result_id="string_0",
            character=quoted[0],
            text=quoted[1],
        )
        return _plan(prompt, intent, (call,), Composition("return", ("string_0",)))
    if intent == "string_reverse":
        require(quoted, 1, "quoted")
        require(integers, 0, "integer")
        call = _call(
            round_index=0,
            specialist="string",
            operation="reverse",
            result_id="string_0",
            text=quoted[0],
        )
        return _plan(prompt, intent, (call,), Composition("return", ("string_0",)))
    if intent == "string_index":
        require(quoted, 1, "quoted")
        require(integers, 1, "integer")
        call = _call(
            round_index=0,
            specialist="string",
            operation="index",
            result_id="string_0",
            text=quoted[0],
            index=integers[0],
        )
        return _plan(prompt, intent, (call,), Composition("return", ("string_0",)))
    if intent == "multi_parallel":
        require(quoted, 1, "quoted")
        require(integers, 3, "integer")
        calls = (
            _call(
                round_index=0,
                specialist="math",
                operation="solve_linear",
                result_id="math_0",
                a=integers[0],
                b=integers[1],
                c=integers[2],
            ),
            _call(
                round_index=0,
                specialist="string",
                operation="reverse",
                result_id="string_0",
                text=quoted[0],
            ),
        )
        return _plan(
            prompt,
            intent,
            calls,
            Composition("join", ("math_0", "string_0"), "|"),
        )
    if intent == "string_then_math":
        require(quoted, 2, "quoted")
        require(integers, 2, "integer")
        calls = (
            _call(
                round_index=0,
                specialist="string",
                operation="count",
                result_id="string_0",
                character=quoted[0],
                text=quoted[1],
            ),
            _call(
                round_index=1,
                specialist="math",
                operation="solve_linear",
                result_id="math_1",
                a=integers[0],
                b=ResultReference("string_0"),
                c=integers[1],
            ),
        )
        return _plan(prompt, intent, calls, Composition("return", ("math_1",)))
    require(quoted, 1, "quoted")
    require(integers, 3, "integer")
    calls = (
        _call(
            round_index=0,
            specialist="math",
            operation="solve_linear",
            result_id="math_0",
            a=integers[0],
            b=integers[1],
            c=integers[2],
        ),
        _call(
            round_index=1,
            specialist="string",
            operation="index",
            result_id="string_1",
            text=quoted[0],
            index=ResultReference("math_0"),
        ),
    )
    return _plan(prompt, intent, calls, Composition("return", ("string_1",)))


def _resolve_value(
    value: DispatchValue, prompt: str, results: Mapping[str, str]
) -> str:
    if isinstance(value, SourceSpan):
        return value.resolve(prompt)
    resolved = results.get(value.result_id)
    if resolved is None or not str(resolved):
        raise DispatchError(f"dependency {value.result_id!r} is unavailable")
    return str(resolved)


def compile_specialist_request(
    plan: DispatchPlan,
    call: SpecialistCall,
    results: Mapping[str, str],
) -> str:
    """Render a typed call into a specialist's proven familiar interface."""

    plan.validate()
    arguments = {
        name: _resolve_value(value, plan.prompt, results)
        for name, value in call.arguments
    }
    if call.operation == "solve_native":
        problem = arguments.get("problem")
        if problem is None or problem != plan.prompt:
            raise DispatchError(
                "native math request must be an exact span covering the prompt"
            )
        return problem
    if plan.plan_kind.startswith("exact_string_"):
        # Standalone string prompts are already on the tower's proven native
        # interface. Preserve the original bytes and held-out paraphrase.
        return plan.prompt
    if call.operation == "solve_linear":
        a, b, c = (arguments[name] for name in ("a", "b", "c"))
        if not all(re.fullmatch(_INTEGER, value) for value in (a, b, c)):
            raise DispatchError("linear-solver arguments must be signed integers")
        return (
            f"For an integer x, {a} times x together with {b} gives {c}. "
            "Determine x."
        )
    if call.operation == "length":
        return f"How many characters are in '{arguments['text']}'?"
    if call.operation == "count":
        character = arguments.get("character", arguments.get("char"))
        if character is None:
            raise DispatchError("count call has no character argument")
        return (
            f"How many times does '{character}' occur in "
            f"'{arguments['text']}'?"
        )
    if call.operation == "index":
        index = arguments["index"]
        if not re.fullmatch(_INTEGER, index):
            raise DispatchError("string index must be a signed integer")
        return (
            "Using zero-based indexing, which character is at position "
            f"{index} in '{arguments['text']}'?"
        )
    if call.operation == "reverse":
        return f"Reverse '{arguments['text']}'."
    if call.operation == "contains":
        return (
            f"Does '{arguments['text']}' contain the exact substring "
            f"'{arguments['substring']}'? Answer yes or no."
        )
    if call.operation == "substitute":
        return (
            f"In '{arguments['text']}', replace every '{arguments['char']}' "
            f"with '{arguments['replacement']}'."
        )
    raise DispatchError(f"cannot compile operation: {call.operation}")


def compose_dispatch_results(
    plan: DispatchPlan, results: Mapping[str, str]
) -> str | None:
    plan.validate()
    if plan.composition.kind == "none":
        return None
    values: list[str] = []
    for result_id in plan.composition.result_ids:
        value = results.get(result_id)
        if value is None or not str(value):
            return None
        values.append(str(value))
    if plan.composition.kind == "return":
        return values[0]
    return plan.composition.separator.join(values)
