from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import canonical_json, config_sha256


GENERATOR_FORMAT = "cftn_text_linear_equations_v1"

TRAIN_TEMPLATE_IDS = (
    "symbolic_standard",
    "symbolic_reversed",
    "verbal_add",
    "verbal_increased",
    "verbal_result",
    "verbal_unknown",
)
HELDOUT_TEMPLATE_IDS = (
    "heldout_sum",
    "heldout_starting",
    "heldout_balance",
    "heldout_quantity",
)
COMPOSITIONAL_TEMPLATE_IDS = (
    "composed_rhs",
    "composed_isolated_product",
    "composed_difference_first",
    "composed_parenthesized",
)


@dataclass(frozen=True)
class EquationRecord:
    record_id: str
    equation_id: str
    split: str
    a: int
    b: int
    c: int
    x: int
    template_id: str
    problem: str
    target_trace: str
    target_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equation_key(a: int, b: int, c: int, x: int) -> str:
    return canonical_json({"a": a, "b": b, "c": c, "x": x})


def canonical_trace(a: int, b: int, c: int, x: int) -> str:
    reduced = c - b
    return (
        f"<work>{a}*x+({b})={c};SUB({b});{a}*x={reduced};"
        f"DIV({a});x={x}</work><answer>{x}</answer>"
    )


def canonical_answer(x: int) -> str:
    return f"<answer>{x}</answer>"


def render_problem(a: int, b: int, c: int, template_id: str) -> str:
    templates = {
        "symbolic_standard": f"Solve {a}*x + ({b}) = {c}.",
        "symbolic_reversed": f"Find x if {c} = ({b}) + {a}*x.",
        "verbal_add": (
            f"Add {b} to {a} times an integer. The result is {c}. "
            "What is the integer?"
        ),
        "verbal_increased": (
            f"An unknown number multiplied by {a}, then increased by {b}, "
            f"equals {c}. Find the unknown number."
        ),
        "verbal_result": (
            f"The result of multiplying a number by {a} and adding {b} is {c}. "
            "Solve for the number."
        ),
        "verbal_unknown": (
            f"For an integer x, {a} times x together with {b} gives {c}. "
            "Determine x."
        ),
        "heldout_sum": (
            f"The sum of {a} copies of a quantity and {b} is {c}. "
            "Identify the quantity."
        ),
        "heldout_starting": (
            f"Starting from {c}, remove {b}; what remains is {a} times an "
            "unknown integer. Name that integer."
        ),
        "heldout_balance": (
            f"A balance states that {a} multiplied by an unknown, plus {b}, "
            f"has the same value as {c}. What is the unknown?"
        ),
        "heldout_quantity": (
            f"Which integer becomes {c} after it is scaled by {a} and then "
            f"shifted by {b}?"
        ),
        "composed_rhs": f"Solve the rearranged equality {c} = {b} + ({a}*x).",
        "composed_isolated_product": f"Find x from {a}*x = {c - b}.",
        "composed_difference_first": f"If {c} - ({b}) = {a}*x, determine x.",
        "composed_parenthesized": f"Solve (({a})*x) + ({b}) - ({c}) = 0.",
    }
    try:
        return templates[template_id]
    except KeyError as exc:
        raise ValueError(f"unknown equation template: {template_id}") from exc


def validate_record(record: EquationRecord | dict[str, Any]) -> None:
    item = record if isinstance(record, EquationRecord) else EquationRecord(**record)
    if item.a == 0:
        raise ValueError("coefficient a must be nonzero")
    if item.a * item.x + item.b != item.c:
        raise ValueError("equation target is mathematically invalid")
    if item.target_trace != canonical_trace(item.a, item.b, item.c, item.x):
        raise ValueError("canonical trace does not match the equation")
    if item.target_answer != canonical_answer(item.x):
        raise ValueError("canonical answer does not match x")
    expected_equation_id = sha256_bytes(
        equation_key(item.a, item.b, item.c, item.x).encode("utf-8")
    )
    if item.equation_id != expected_equation_id:
        raise ValueError("equation_id does not match normalized coefficients")
    expected_record_id = sha256_bytes(
        canonical_json(
            {
                "equation_id": item.equation_id,
                "problem": item.problem,
                "split": item.split,
                "template_id": item.template_id,
            }
        ).encode("utf-8")
    )
    if item.record_id != expected_record_id:
        raise ValueError("record_id does not match record contents")


def _sample_nonzero(rng: random.Random, minimum: int, maximum: int) -> int:
    while True:
        value = rng.randint(minimum, maximum)
        if value:
            return value


def _sample_signed_magnitude(rng: random.Random, minimum: int, maximum: int) -> int:
    magnitude = rng.randint(minimum, maximum)
    return magnitude if rng.random() < 0.5 else -magnitude


def _ranges(config: dict[str, Any], split: str, rng: random.Random) -> tuple[int, int, int]:
    data = config["data"]
    if split == "extrapolation":
        a = _sample_signed_magnitude(
            rng,
            int(data["extrapolation_a_abs_min"]),
            int(data["extrapolation_a_abs_max"]),
        )
        x = _sample_signed_magnitude(
            rng,
            int(data["extrapolation_x_abs_min"]),
            int(data["extrapolation_x_abs_max"]),
        )
        b = _sample_signed_magnitude(
            rng,
            int(data["extrapolation_b_abs_min"]),
            int(data["extrapolation_b_abs_max"]),
        )
        return a, x, b
    return (
        _sample_nonzero(rng, int(data["train_a_min"]), int(data["train_a_max"])),
        rng.randint(int(data["train_x_min"]), int(data["train_x_max"])),
        rng.randint(int(data["train_b_min"]), int(data["train_b_max"])),
    )


def _template_pool(split: str) -> tuple[str, ...]:
    if split == "heldout_language":
        return HELDOUT_TEMPLATE_IDS
    if split == "compositional":
        return COMPOSITIONAL_TEMPLATE_IDS
    return TRAIN_TEMPLATE_IDS


def generate_split(
    config: dict[str, Any],
    split: str,
    count: int,
    seen_equations: set[str],
) -> list[EquationRecord]:
    if count < 1:
        raise ValueError("split count must be positive")
    seed = int(config["project"]["seed"])
    split_seed = int.from_bytes(hashlib.sha256(split.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed ^ split_seed)
    templates = _template_pool(split)
    maximum_abs = int(config["data"]["maximum_abs_intermediate"])
    records: list[EquationRecord] = []
    attempts = 0
    maximum_attempts = max(10_000, count * 50)
    while len(records) < count:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError(
                f"could not create {count} unique records for {split}; "
                "increase coefficient ranges"
            )
        a, x, b = _ranges(config, split, rng)
        c = a * x + b
        if max(abs(c), abs(c - b), abs(a * x)) > maximum_abs:
            continue
        key = equation_key(a, b, c, x)
        equation_id = sha256_bytes(key.encode("utf-8"))
        if equation_id in seen_equations:
            continue
        template_id = templates[rng.randrange(len(templates))]
        problem = render_problem(a, b, c, template_id)
        record_payload = {
            "equation_id": equation_id,
            "problem": problem,
            "split": split,
            "template_id": template_id,
        }
        item = EquationRecord(
            record_id=sha256_bytes(canonical_json(record_payload).encode("utf-8")),
            equation_id=equation_id,
            split=split,
            a=a,
            b=b,
            c=c,
            x=x,
            template_id=template_id,
            problem=problem,
            target_trace=canonical_trace(a, b, c, x),
            target_answer=canonical_answer(x),
        )
        validate_record(item)
        seen_equations.add(equation_id)
        records.append(item)
    return records


def build_records(config: dict[str, Any]) -> dict[str, list[EquationRecord]]:
    data = config["data"]
    requests = (
        ("calibration", int(data["calibration_examples"])),
        ("train", int(data["train_examples"])),
        ("validation", int(data["validation_examples"])),
        ("test", int(data["test_examples"])),
        ("heldout_language", int(data["heldout_language_examples"])),
        ("extrapolation", int(data["extrapolation_examples"])),
        ("compositional", int(data["compositional_examples"])),
    )
    seen: set[str] = set()
    return {
        split: generate_split(config, split, count, seen)
        for split, count in requests
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonl_bytes(records: Iterable[EquationRecord]) -> bytes:
    return b"".join(
        (canonical_json(record.to_dict()) + "\n").encode("utf-8")
        for record in records
    )


def prepare_manifests(
    config: dict[str, Any],
    output_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root or config["project"]["data_root"]).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("config_sha256") != config_sha256(config):
            raise FileExistsError(
                f"{manifest_path} belongs to a different configuration; use --force"
            )
        audit_manifest(existing, root)
        return existing

    records_by_split = build_records(config)
    split_metadata: dict[str, Any] = {}
    all_equations: set[str] = set()
    for split, records in records_by_split.items():
        payload = _jsonl_bytes(records)
        path = root / f"{split}.jsonl"
        _atomic_write(path, payload)
        equations = {record.equation_id for record in records}
        if all_equations.intersection(equations):
            raise AssertionError("equation overlap was introduced across splits")
        all_equations.update(equations)
        split_metadata[split] = {
            "path": path.name,
            "count": len(records),
            "sha256": sha256_bytes(payload),
            "template_ids": sorted({record.template_id for record in records}),
        }

    source_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "format": GENERATOR_FORMAT,
        "seed": int(config["project"]["seed"]),
        "config_sha256": config_sha256(config),
        "generator_path": str(source_path),
        "generator_sha256": file_sha256(source_path),
        "splits": split_metadata,
        "total_records": sum(item["count"] for item in split_metadata.values()),
        "normalized_equation_overlap": 0,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    audit_manifest(manifest, root)
    return manifest


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                validate_record(item)
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
            records.append(item)
    return records


def audit_manifest(manifest: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    if manifest.get("format") != GENERATOR_FORMAT:
        raise ValueError("unsupported data manifest format")
    unsigned = dict(manifest)
    recorded_hash = unsigned.pop("manifest_sha256", None)
    actual_hash = sha256_bytes(canonical_json(unsigned).encode("utf-8"))
    if recorded_hash != actual_hash:
        raise ValueError("manifest hash mismatch")
    generator_path = Path(manifest["generator_path"])
    if not generator_path.is_file():
        raise FileNotFoundError(f"recorded generator source is missing: {generator_path}")
    if file_sha256(generator_path) != manifest["generator_sha256"]:
        raise ValueError("data generator source hash mismatch")
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for split, metadata in manifest["splits"].items():
        path = root_path / metadata["path"]
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"split hash mismatch: {split}")
        records = load_records(path)
        if len(records) != int(metadata["count"]):
            raise ValueError(f"split count mismatch: {split}")
        equation_ids = {record["equation_id"] for record in records}
        if len(equation_ids) != len(records):
            raise ValueError(f"duplicate normalized equation within split: {split}")
        overlap = seen.intersection(equation_ids)
        if overlap:
            raise ValueError(f"normalized equation overlap across splits: {split}")
        seen.update(equation_ids)
        counts[split] = len(records)
    return {
        "pass": True,
        "counts": counts,
        "total_unique_equations": len(seen),
        "overlap": 0,
    }
