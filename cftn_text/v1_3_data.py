from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .config import canonical_json
from .data_generator import file_sha256


STRING_SCHEMA = "cftn_exact_string_record_v1_3"
JOINT_SCHEMA = "cftn_multi_specialist_record_v1_3_1"
MANIFEST_FORMAT = "cftn_text_v1_3_manifest_v1"
SPECIALISTS = ("math", "string")
TASK_CLASSES = (
    "pure_language",
    "explicit_math",
    "exact_string",
    "language_dependent_math",
    "multi_parallel",
    "multi_sequential",
)

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_LABELS = ("amber", "birch", "cobalt", "delta", "ember", "fable", "garnet")

_STRING_TEMPLATES = {
    "train": {
        "length": (
            "How many characters are in '{text}'?",
            "Return the character length of '{text}'.",
        ),
        "count": (
            "How many times does '{char}' occur in '{text}'?",
            "Count the character '{char}' in '{text}'.",
        ),
        "index": (
            "Using zero-based indexing, which character is at position {index} in '{text}'?",
            "Return character {index} of '{text}' when the first position is zero.",
        ),
        "reverse": ("Reverse '{text}'.", "Write '{text}' backwards."),
        "contains": (
            "Does '{text}' contain the exact substring '{substring}'? Answer yes or no.",
            "Return yes if '{substring}' occurs contiguously in '{text}', otherwise no.",
        ),
        "substitute": (
            "In '{text}', replace every '{char}' with '{replacement}'.",
            "Substitute '{replacement}' for all '{char}' characters in '{text}'.",
        ),
    },
    "heldout": {
        "length": ("Determine the cardinality of the character sequence '{text}'.",),
        "count": ("What is the frequency of symbol '{char}' within '{text}'?",),
        "index": ("Indexing from zero, extract offset {index} from '{text}'.",),
        "reverse": ("Emit the mirror ordering of the symbols in '{text}'.",),
        "contains": ("Is '{substring}' a contiguous infix of '{text}'? Reply yes or no.",),
        "substitute": ("Map each '{char}' in '{text}' to '{replacement}'.",),
    },
}


def _rng(seed: int, *parts: object) -> random.Random:
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _record_id(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "record_id"}
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def _random_string(rng: random.Random, minimum: int, maximum: int) -> str:
    length = rng.randint(minimum, maximum)
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def _answer(value: object) -> str:
    return f"<answer>{value}</answer>"


def _string_problem(
    *,
    operation: str,
    text: str,
    rng: random.Random,
    template_family: str,
) -> tuple[str, str, str, dict[str, Any]]:
    char = rng.choice(text)
    replacement = rng.choice([value for value in _ALPHABET if value != char])
    index = rng.randrange(len(text))
    if rng.random() < 0.5:
        start = rng.randrange(len(text))
        width = rng.randint(1, min(4, len(text) - start))
        substring = text[start : start + width]
    else:
        substring = "".join(rng.choice(_ALPHABET) for _ in range(min(3, len(text))))
        if substring in text:
            substring = substring[::-1] + "z"
    if operation == "length":
        value: object = len(text)
        work = f"LEN({text})={value}"
    elif operation == "count":
        value = text.count(char)
        work = f"COUNT({char},{text})={value}"
    elif operation == "index":
        value = text[index]
        work = f"INDEX0({text},{index})={value}"
    elif operation == "reverse":
        value = text[::-1]
        work = f"REVERSE({text})={value}"
    elif operation == "contains":
        value = "yes" if substring in text else "no"
        work = f"CONTAINS({text},{substring})={value}"
    elif operation == "substitute":
        value = text.replace(char, replacement)
        work = f"SUBSTITUTE({text},{char},{replacement})={value}"
    else:
        raise ValueError(f"unsupported string operation: {operation}")
    templates = _STRING_TEMPLATES[template_family][operation]
    problem = rng.choice(templates).format(
        text=text,
        char=char,
        replacement=replacement,
        index=index,
        substring=substring,
    )
    metadata = {
        "operation": operation,
        "source_string": text,
        "character": char,
        "replacement": replacement,
        "index": index,
        "substring": substring,
        "value": str(value),
    }
    return problem, str(value), f"<work>{work}</work>{_answer(value)}", metadata


def generate_string_record(
    *, seed: int, split: str, index: int, config: dict[str, Any]
) -> dict[str, Any]:
    rng = _rng(seed, "string", split, index)
    maximum = int(config["data"]["maximum_string_length_train"])
    minimum = 3
    if split == "string_extrapolation":
        minimum = maximum + 1
        maximum = int(config["data"]["maximum_string_length_extrapolation"])
    text = _random_string(rng, minimum, maximum)
    template_family = "heldout" if split == "string_heldout_paraphrase" else "train"
    operations = ("length", "count", "index", "reverse", "contains", "substitute")
    operation = operations[index % len(operations)]
    problem, value, trace, metadata = _string_problem(
        operation=operation,
        text=text,
        rng=rng,
        template_family=template_family,
    )
    if split == "string_compositional":
        operation = "reverse_then_count"
        char = rng.choice(text)
        reversed_text = text[::-1]
        value = str(reversed_text.count(char))
        problem = (
            f"Reverse '{text}', then count how many '{char}' characters occur "
            "in the reversed result."
        )
        trace = (
            f"<work>REVERSE({text})={reversed_text};"
            f"COUNT({char},{reversed_text})={value}</work>{_answer(value)}"
        )
        metadata.update({"character": char, "value": value})
    record: dict[str, Any] = {
        "schema_version": STRING_SCHEMA,
        "split": split,
        "operation": operation,
        "problem": problem,
        "target_answer": _answer(value),
        "target_trace": trace,
        "answer_value": None,
        **metadata,
    }
    record["record_id"] = _record_id(record)
    return record


def _linear_equation(rng: random.Random, *, extrapolation: bool = False) -> dict[str, int]:
    if extrapolation:
        x = rng.choice((-1, 1)) * rng.randint(201, 500)
        a = rng.choice((-1, 1)) * rng.randint(51, 100)
        b = rng.choice((-1, 1)) * rng.randint(501, 1200)
    else:
        x = rng.randint(-50, 50)
        a = rng.choice([value for value in range(-20, 21) if value])
        b = rng.randint(-150, 150)
    c = a * x + b
    return {"a": a, "b": b, "c": c, "x": x}


def _math_trace(values: dict[str, int], *, offset_symbol: str | None = None) -> str:
    a, b, c, x = (values[key] for key in ("a", "b", "c", "x"))
    offset = offset_symbol if offset_symbol is not None else str(b)
    return (
        f"<work>{a}*x+({offset})={c};SUB({offset});"
        f"{a}*x={a*x};DIV({a});x={x}</work>{_answer(x)}"
    )


def _round_targets(
    math: list[str | None], string: list[str | None], rounds: int
) -> dict[str, list[str | None]]:
    return {
        "math": (math + [None] * rounds)[:rounds],
        "string": (string + [None] * rounds)[:rounds],
    }


def _round_prompts(
    math: list[str | None], string: list[str | None], rounds: int
) -> dict[str, list[str | None]]:
    return {
        "math": (math + [None] * rounds)[:rounds],
        "string": (string + [None] * rounds)[:rounds],
    }


def _math_oracle_problem(values: dict[str, int]) -> str:
    """Use the specialist's proven familiar interface for capability controls."""

    a, b, c = (values[key] for key in ("a", "b", "c"))
    return (
        f"For an integer x, {a} times x together with {b} gives {c}. "
        "Determine x."
    )


def _joint_task_class(index: int) -> str:
    # A ten-example cycle keeps the registered 20% top-level shares while
    # dividing multi-specialist examples equally into parallel and sequential.
    cycle = index % 10
    if cycle < 2:
        return "pure_language"
    if cycle < 4:
        return "explicit_math"
    if cycle < 6:
        return "exact_string"
    if cycle < 8:
        return "language_dependent_math"
    return "multi_parallel" if cycle == 8 else "multi_sequential"


def generate_joint_record(
    *, seed: int, split: str, index: int, config: dict[str, Any]
) -> dict[str, Any]:
    rng = _rng(seed, "joint", split, index)
    rounds = int(config["runtime"]["maximum_callosal_rounds"])
    task_class = _joint_task_class(index)
    heldout = split == "joint_heldout_paraphrase"
    extrapolation = split == "joint_extrapolation"
    equation = _linear_equation(rng, extrapolation=extrapolation)
    text_max = int(config["data"]["maximum_string_length_train"])
    text_min = 4
    if extrapolation:
        text_min = text_max + 1
        text_max = int(config["data"]["maximum_string_length_extrapolation"])
    text = _random_string(rng, text_min, text_max)
    char = rng.choice(text)
    string_problem, string_value, string_trace, string_meta = _string_problem(
        operation=("count", "reverse", "index")[index % 3],
        text=text,
        rng=rng,
        template_family="heldout" if heldout else "train",
    )
    required: list[str]
    required_by_round: list[list[str]]
    specialist_targets: dict[str, list[str | None]]
    specialist_oracle_problems: dict[str, list[str | None]]
    halt_round = 1
    metadata: dict[str, Any] = {"equation": equation, "string": string_meta}

    if task_class == "pure_language":
        label = _LABELS[index % len(_LABELS)]
        problem = (
            f"The archival label is {label}. Ignore the colour red. Return the archival label."
        )
        gpt_prompt = (
            f"Archival label: {label}\n"
            "Colour: red\n"
            "Requested archival label:"
        )
        value = label
        required = []
        required_by_round = [[] for _ in range(rounds)]
        specialist_targets = _round_targets([], [], rounds)
        specialist_oracle_problems = _round_prompts([], [], rounds)
    elif task_class == "explicit_math":
        a, b, c, x = (equation[key] for key in ("a", "b", "c", "x"))
        problem = f"Solve {a}*x + ({b}) = {c}. Return x."
        value = str(x)
        required = ["math"]
        required_by_round = [["math"]] + [[] for _ in range(rounds - 1)]
        specialist_targets = _round_targets([_math_trace(equation)], [], rounds)
        specialist_oracle_problems = _round_prompts(
            [_math_oracle_problem(equation)], [], rounds
        )
    elif task_class == "exact_string":
        problem = string_problem
        value = string_value
        required = ["string"]
        required_by_round = [["string"]] + [[] for _ in range(rounds - 1)]
        specialist_targets = _round_targets([], [string_trace], rounds)
        specialist_oracle_problems = _round_prompts([], [string_problem], rounds)
    elif task_class == "language_dependent_math":
        a, b, c, x = (equation[key] for key in ("a", "b", "c", "x"))
        if heldout:
            problem = (
                f"A latent quantity is scaled by {a}, translated by {b}, and arrives at {c}. "
                "Recover the latent quantity; the note about seven lanterns is irrelevant."
            )
        else:
            problem = (
                f"Mira thinks of an integer. Multiplying it by {a} and then adding {b} gives {c}. "
                "What integer did Mira choose?"
            )
        value = str(x)
        required = ["math"]
        required_by_round = [["math"]] + [[] for _ in range(rounds - 1)]
        specialist_targets = _round_targets([_math_trace(equation)], [], rounds)
        specialist_oracle_problems = _round_prompts(
            [_math_oracle_problem(equation)], [], rounds
        )
    elif task_class == "multi_parallel":
        a, b, c, x = (equation[key] for key in ("a", "b", "c", "x"))
        reversed_text = text[::-1]
        problem = (
            f"Solve {a}*x + ({b}) = {c} and independently reverse '{text}'. "
            "Return the result as x|reversed."
        )
        value = f"{x}|{reversed_text}"
        required = ["math", "string"]
        required_by_round = [["math", "string"]] + [[] for _ in range(rounds - 1)]
        reverse_trace = f"<work>REVERSE({text})={reversed_text}</work>{_answer(reversed_text)}"
        # ``string_meta`` is created before the joint task class is selected.
        # Multi-parallel always overrides that provisional operation with a
        # reversal, so keep the registered metadata aligned with the actual
        # specialist target and GPT completion target.
        metadata["string"].update(
            {"operation": "reverse", "value": reversed_text}
        )
        specialist_targets = _round_targets([_math_trace(equation)], [reverse_trace], rounds)
        specialist_oracle_problems = _round_prompts(
            [_math_oracle_problem(equation)], [f"Reverse '{text}'."], rounds
        )
    else:
        # Alternate both dependency orders so another callosal round is
        # necessary for each specialist, rather than privileging one tower.
        # ``multi_sequential`` occupies index 9 of every ten-record cycle, so
        # testing ``index % 2`` made every generated example string->math.
        # Alternate by the cycle number instead.
        if (index // 10) % 2:
            count = text.count(char)
            a = rng.choice([value for value in range(-12, 13) if value])
            x = rng.randint(-30, 30)
            c = a * x + count
            sequential_equation = {"a": a, "b": count, "c": c, "x": x}
            problem = (
                f"First count '{char}' in '{text}'. Let that count be n. "
                f"Then solve {a}*x+n={c}. Return x."
            )
            value = str(x)
            count_trace = f"<work>COUNT({char},{text})={count}</work>{_answer(count)}"
            required_by_round = [["string"], ["math"]] + [
                [] for _ in range(rounds - 2)
            ]
            specialist_targets = _round_targets(
                [None, _math_trace(sequential_equation, offset_symbol="n")],
                [count_trace, None],
                rounds,
            )
            specialist_oracle_problems = _round_prompts(
                [None, _math_oracle_problem(sequential_equation)],
                [f"How many times does '{char}' occur in '{text}'?", None],
                rounds,
            )
            metadata["sequential_order"] = "string_then_math"
            metadata["equation"] = sequential_equation
        else:
            index_value = rng.randrange(min(len(text), 12))
            a = rng.choice([value for value in range(1, 8)])
            b = rng.randint(-20, 20)
            c = a * index_value + b
            sequential_equation = {"a": a, "b": b, "c": c, "x": index_value}
            selected = text[index_value]
            problem = (
                f"Solve {a}*x+({b})={c}. Use x as a zero-based index into '{text}', "
                "then return the selected character."
            )
            value = selected
            index_trace = (
                f"<work>INDEX0({text},x={index_value})={selected}</work>{_answer(selected)}"
            )
            required_by_round = [["math"], ["string"]] + [
                [] for _ in range(rounds - 2)
            ]
            specialist_targets = _round_targets(
                [_math_trace(sequential_equation), None],
                [None, index_trace],
                rounds,
            )
            specialist_oracle_problems = _round_prompts(
                [_math_oracle_problem(sequential_equation), None],
                [
                    None,
                    (
                        "Using zero-based indexing, which character is at position "
                        f"{index_value} in '{text}'?"
                    ),
                ],
                rounds,
            )
            metadata["sequential_order"] = "math_then_string"
            metadata["equation"] = sequential_equation
        required = ["math", "string"]
        halt_round = 2

    if split == "joint_counterfactual":
        # The pair key deliberately groups adjacent examples while the index
        # parity changes either operation or operand.
        metadata["counterfactual_pair_id"] = f"cf-{index // 2:08d}"
        metadata["counterfactual_member"] = index % 2
    if split == "joint_unseen_composition" and task_class != "multi_sequential":
        task_class = "multi_sequential"
        return generate_joint_record(
            seed=seed + 99_991,
            split=split,
            index=index * 10 + 9,
            config=config,
        )
    if task_class != "pure_language":
        gpt_prompt = (
            f"Problem: {problem}\n"
            f"{config['gpt_interface']['generic_answer_cue']}"
        )
    record: dict[str, Any] = {
        "schema_version": JOINT_SCHEMA,
        "split": split,
        "task_class": task_class,
        "problem": problem,
        "gpt_prompt": gpt_prompt,
        "gpt_target": str(value),
        "gpt_answer_protocol": config["gpt_interface"]["answer_protocol"],
        "target_answer": _answer(value),
        "required_specialists": required,
        "required_specialists_by_round": required_by_round,
        "specialist_targets_by_round": specialist_targets,
        "specialist_oracle_problems_by_round": specialist_oracle_problems,
        "halt_round": halt_round,
        "metadata": metadata,
    }
    record["record_id"] = _record_id(record)
    return record


def joint_string_operation(record: dict[str, Any]) -> str | None:
    """Return the effective string operation for old and repaired joint rows.

    Historical V1.3 manifests predate the explicit ``operation`` metadata and
    some multi-parallel rows retained provisional count/index metadata.  The
    specialist target is the authoritative fallback because it is the tensor
    contract actually consumed by integration training.
    """

    task_class = str(record.get("task_class", ""))
    if task_class == "multi_parallel":
        return "reverse"
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        string_metadata = metadata.get("string", {})
        if isinstance(string_metadata, dict):
            operation = string_metadata.get("operation")
            if isinstance(operation, str) and operation:
                return operation.lower()
    targets = record.get("specialist_targets_by_round", {})
    if not isinstance(targets, dict):
        return None
    string_targets = targets.get("string", [])
    if not isinstance(string_targets, list):
        return None
    for target in string_targets:
        if not isinstance(target, str):
            continue
        work = target.upper()
        for marker, operation in (
            ("<WORK>REVERSE(", "reverse"),
            ("<WORK>COUNT(", "count"),
            ("<WORK>INDEX0(", "index"),
            ("<WORK>LEN(", "length"),
            ("<WORK>CONTAINS(", "contains"),
            ("<WORK>SUBSTITUTE(", "substitute"),
        ):
            if marker in work:
                return operation
    return None


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
            count += 1
        handle.flush()
    temporary.replace(path)
    return count


def load_v1_3_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("schema_version") not in {STRING_SCHEMA, JOINT_SCHEMA}:
                    raise ValueError(f"unsupported V1.3 record schema in {path}")
                records.append(record)
    return records


def prepare_v1_3_manifests(config: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    destination = Path(root or config["paths"]["data_root"]).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seed = int(config["revision"]["seed"])
    sizes = {
        "string_train": int(config["data"]["string_train_examples"]),
        "string_validation": int(config["data"]["string_validation_examples"]),
        "string_test": int(config["data"]["string_test_examples"]),
        "string_heldout_paraphrase": int(
            config["data"]["string_heldout_paraphrase_examples"]
        ),
        "string_extrapolation": int(config["data"]["string_extrapolation_examples"]),
        "string_compositional": int(config["data"]["string_compositional_examples"]),
        "joint_train": int(config["data"]["joint_train_examples"]),
        "joint_validation": int(config["data"]["joint_validation_examples"]),
        "joint_test": int(config["data"]["joint_test_examples"]),
        "joint_heldout_paraphrase": int(
            config["data"]["joint_heldout_paraphrase_examples"]
        ),
        "joint_extrapolation": int(config["data"]["joint_extrapolation_examples"]),
        "joint_counterfactual": int(config["data"]["joint_counterfactual_examples"]),
        "joint_unseen_composition": int(
            config["data"]["joint_unseen_composition_examples"]
        ),
    }
    splits: dict[str, Any] = {}
    for split, size in sizes.items():
        path = destination / f"{split}.jsonl"
        generator = generate_string_record if split.startswith("string_") else generate_joint_record
        count = _write_jsonl(
            (
                generator(seed=seed, split=split, index=index, config=config)
                for index in range(size)
            ),
            path,
        )
        splits[split] = {
            "path": path.name,
            "examples": count,
            "sha256": file_sha256(path),
        }
    manifest = {
        "format": MANIFEST_FORMAT,
        "revision_sha256": config["_meta"]["sha256"],
        "seed": seed,
        "ascii_only": bool(config["data"]["ascii_only"]),
        "splits": splits,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    manifest_path = destination / "manifest.json"
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def audit_v1_3_manifest(config: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    destination = Path(root or config["paths"]["data_root"]).expanduser().resolve()
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V1.3 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("unsupported V1.3 manifest format")
    if manifest.get("revision_sha256") != config["_meta"]["sha256"]:
        raise ValueError("V1.3 manifest revision hash mismatch")
    clean = dict(manifest)
    expected_manifest_hash = clean.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("V1.3 manifest content hash mismatch")
    for split, metadata in manifest["splits"].items():
        path = destination / metadata["path"]
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"V1.3 split hash mismatch: {split}")
    return manifest
