from __future__ import annotations

import hashlib
import itertools
from typing import Any


SLOTS = ("P", "Q", "R")
ROLES = ("coefficient", "offset", "result")
SLOT_PERMUTATIONS = tuple(itertools.permutations(SLOTS))

TRAIN_ROLE_TEMPLATES = (
    (
        "{coefficient} is the multiplier applied to x; {offset} is added "
        "afterward; the total is {result}. Determine x."
    ),
    (
        "Multiply x by the value named {coefficient}, add the value named "
        "{offset}, and equate the outcome to {result}. Solve for x."
    ),
    (
        "The slot {result} equals {coefficient} copies of x together with "
        "the signed shift {offset}. Find x."
    ),
)

HELDOUT_ROLE_TEMPLATES = (
    (
        "Treat {coefficient} as a scale, {offset} as a subsequent translation, "
        "and {result} as the final value. Which integer was scaled?"
    ),
    (
        "Starting from the quantity identified by {result}, remove the slot "
        "identified by {offset}; what remains is {coefficient} times x."
    ),
    (
        "A balance uses {coefficient} for the factor on x, {offset} for the "
        "signed adjustment, and {result} for the opposite side. Recover x."
    ),
)

DISTRACTOR_TEMPLATES = (
    "Ignore the unrelated note that a blue notebook has twelve pages. ",
    "The colour of the labels is irrelevant to the calculation. ",
    "A separate example about five chairs is not part of this problem. ",
)


def _digest_index(key: str, size: int) -> int:
    if size < 1:
        raise ValueError("digest selection requires a nonempty collection")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def role_assignment(record_key: str, seed: int) -> dict[str, str]:
    permutation = SLOT_PERMUTATIONS[
        _digest_index(f"{seed}:{record_key}:roles", len(SLOT_PERMUTATIONS))
    ]
    return dict(zip(ROLES, permutation))


def complementary_record(
    record: dict[str, Any],
    *,
    seed: int,
    assignment_key: str | None = None,
    add_distractor: bool = False,
) -> dict[str, Any]:
    """Create complementary private views without changing the target.

    GPT receives the algebraic roles of three opaque slots but no values. The
    math tower receives the slot values but no role assignment. The six role
    permutations vary deterministically by record, preventing either tower
    from learning a static slot convention.
    """

    if record.get("schema_version") == "cftn_math_record_v2":
        if not record.get("gpt_problem") or not record.get("math_problem"):
            raise ValueError("V2 record has no prepared complementary private views")
        result = dict(record)
        result["shared_problem"] = str(record["problem"])
        result["view_mode"] = "complementary"
        return result

    required = {"record_id", "split", "a", "b", "c", "x", "problem"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"record is missing complementary-view fields: {sorted(missing)}")
    key = assignment_key or str(record["record_id"])
    assignment = role_assignment(key, int(seed))
    role_values = {
        "coefficient": int(record["a"]),
        "offset": int(record["b"]),
        "result": int(record["c"]),
    }
    slot_values = {
        assignment[role]: value for role, value in role_values.items()
    }
    template_pool = (
        HELDOUT_ROLE_TEMPLATES
        if str(record["split"]) == "heldout_language"
        else TRAIN_ROLE_TEMPLATES
    )
    template = template_pool[
        _digest_index(f"{seed}:{key}:wording", len(template_pool))
    ]
    gpt_problem = template.format(**assignment)
    if add_distractor:
        distractor = DISTRACTOR_TEMPLATES[
            _digest_index(f"{seed}:{key}:distractor", len(DISTRACTOR_TEMPLATES))
        ]
        gpt_problem = distractor + gpt_problem
    math_problem = (
        f"Opaque values: P={slot_values['P']}; Q={slot_values['Q']}; "
        f"R={slot_values['R']}. Use the language-side role assignment and solve x."
    )
    result = dict(record)
    result.update(
        {
            "shared_problem": str(record["problem"]),
            "gpt_problem": gpt_problem,
            "math_problem": math_problem,
            "view_mode": "complementary",
            "role_assignment": assignment,
            "slot_values": slot_values,
            "has_language_distractor": bool(add_distractor),
        }
    )
    return result


def apply_view_mode(
    records: list[dict[str, Any]], *, view_mode: str, seed: int
) -> list[dict[str, Any]]:
    if view_mode == "shared":
        return [dict(record) for record in records]
    if view_mode != "complementary":
        raise ValueError("view mode must be shared or complementary")
    return [
        complementary_record(
            record,
            seed=seed,
            add_distractor=str(record.get("split")) == "compositional",
        )
        for record in records
    ]
