from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import canonical_json, config_sha256
from .data_generator import (
    COMPOSITIONAL_TEMPLATE_IDS,
    HELDOUT_TEMPLATE_IDS,
    TRAIN_TEMPLATE_IDS,
    canonical_answer,
    canonical_trace,
    equation_key,
    file_sha256,
    render_problem,
    sha256_bytes,
)


ALGORITHMIC_GENERATOR_FORMAT = "cftn_text_linear_equations_v1_1"
ALGORITHMIC_RECORD_SCHEMA = "cftn_linear_equation_v1_1"


@dataclass(frozen=True)
class AlgorithmicEquationRecord:
    schema_version: str
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
    difficulty: int
    curriculum_band: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_identity(item: AlgorithmicEquationRecord) -> dict[str, Any]:
    return {
        "schema_version": item.schema_version,
        "equation_id": item.equation_id,
        "problem": item.problem,
        "split": item.split,
        "template_id": item.template_id,
        "difficulty": item.difficulty,
        "curriculum_band": item.curriculum_band,
    }


def validate_algorithmic_record(
    record: AlgorithmicEquationRecord | dict[str, Any],
) -> None:
    item = (
        record
        if isinstance(record, AlgorithmicEquationRecord)
        else AlgorithmicEquationRecord(**record)
    )
    if item.schema_version != ALGORITHMIC_RECORD_SCHEMA:
        raise ValueError("unsupported algorithmic equation schema")
    if item.a == 0:
        raise ValueError("coefficient a must be nonzero")
    if item.a * item.x + item.b != item.c:
        raise ValueError("equation target is mathematically invalid")
    if item.target_trace != canonical_trace(item.a, item.b, item.c, item.x):
        raise ValueError("canonical trace does not match the equation")
    if item.target_answer != canonical_answer(item.x):
        raise ValueError("canonical answer does not match x")
    if item.difficulty < 1:
        raise ValueError("difficulty must be positive")
    expected_equation_id = sha256_bytes(
        equation_key(item.a, item.b, item.c, item.x).encode("utf-8")
    )
    if item.equation_id != expected_equation_id:
        raise ValueError("equation_id does not match normalized coefficients")
    expected_record_id = sha256_bytes(
        canonical_json(_record_identity(item)).encode("utf-8")
    )
    if item.record_id != expected_record_id:
        raise ValueError("record_id does not match record contents")


def _sample_nonzero(rng: random.Random, minimum: int, maximum: int) -> int:
    while True:
        value = rng.randint(minimum, maximum)
        if value:
            return value


def _sample_signed_magnitude(
    rng: random.Random, minimum: int, maximum: int
) -> int:
    magnitude = rng.randint(minimum, maximum)
    return magnitude if rng.random() < 0.5 else -magnitude


def _bands(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in config["data"]["numeric_curriculum_bands"]]


def _difficulty_for_values(
    a: int, x: int, b: int, bands: list[dict[str, Any]]
) -> int:
    for index, band in enumerate(bands, start=1):
        if (
            abs(a) <= int(band["max_abs_a"])
            and abs(x) <= int(band["max_abs_x"])
            and abs(b) <= int(band["max_abs_b"])
        ):
            return index
    return len(bands) + 1


def _choose_band(
    rng: random.Random, bands: list[dict[str, Any]], weight_key: str
) -> int:
    weights = [float(band.get(weight_key, band.get("weight", 1.0))) for band in bands]
    point = rng.random() * sum(weights)
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if point <= cumulative:
            return index
    return len(bands) - 1


def _sample_supported_values(
    rng: random.Random,
    bands: list[dict[str, Any]],
    *,
    weight_key: str,
) -> tuple[int, int, int, int, str]:
    desired_index = _choose_band(rng, bands, weight_key)
    band = bands[desired_index]
    desired_difficulty = desired_index + 1
    for _ in range(10_000):
        a = _sample_nonzero(
            rng, -int(band["max_abs_a"]), int(band["max_abs_a"])
        )
        x = rng.randint(-int(band["max_abs_x"]), int(band["max_abs_x"]))
        b = rng.randint(-int(band["max_abs_b"]), int(band["max_abs_b"]))
        difficulty = _difficulty_for_values(a, x, b, bands)
        if difficulty == desired_difficulty:
            return a, x, b, difficulty, str(band["name"])
    raise RuntimeError(f"could not sample curriculum band {band['name']}")


def _ranges(
    config: dict[str, Any], split: str, rng: random.Random
) -> tuple[int, int, int, int, str]:
    data = config["data"]
    bands = _bands(config)
    final_band = bands[-1]
    if split == "extrapolation":
        a = _sample_signed_magnitude(
            rng,
            int(data["extrapolation_a_abs_min"]),
            int(data["extrapolation_a_abs_max"]),
        )
        x = rng.randint(
            -int(final_band["max_abs_x"]), int(final_band["max_abs_x"])
        )
        b = _sample_signed_magnitude(
            rng,
            int(data["extrapolation_b_abs_min"]),
            int(data["extrapolation_b_abs_max"]),
        )
        return a, x, b, len(bands) + 1, "input_extrapolation"
    if split == "answer_extrapolation":
        a = _sample_nonzero(
            rng,
            -int(final_band["max_abs_a"]),
            int(final_band["max_abs_a"]),
        )
        x = _sample_signed_magnitude(
            rng,
            int(data["answer_extrapolation_x_abs_min"]),
            int(data["answer_extrapolation_x_abs_max"]),
        )
        b = rng.randint(
            -int(final_band["max_abs_b"]), int(final_band["max_abs_b"])
        )
        return a, x, b, len(bands) + 1, "answer_extrapolation"
    return _sample_supported_values(
        rng,
        bands,
        weight_key="weight" if split == "train" else "evaluation_weight",
    )


def _template_pool(split: str) -> tuple[str, ...]:
    if split == "heldout_language":
        return HELDOUT_TEMPLATE_IDS
    if split == "compositional":
        return COMPOSITIONAL_TEMPLATE_IDS
    return TRAIN_TEMPLATE_IDS


def generate_algorithmic_split(
    config: dict[str, Any],
    split: str,
    count: int,
    seen_equations: set[str],
) -> list[AlgorithmicEquationRecord]:
    if count < 1:
        raise ValueError("split count must be positive")
    seed = int(config["project"]["seed"])
    split_seed = int.from_bytes(
        hashlib.sha256(split.encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed ^ split_seed)
    templates = _template_pool(split)
    maximum_abs = int(config["data"]["maximum_abs_intermediate"])
    records: list[AlgorithmicEquationRecord] = []
    attempts = 0
    maximum_attempts = max(20_000, count * 100)
    while len(records) < count:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError(
                f"could not create {count} unique records for {split}; "
                "increase numeric ranges"
            )
        a, x, b, difficulty, band_name = _ranges(config, split, rng)
        c = a * x + b
        if max(abs(c), abs(c - b), abs(a * x)) > maximum_abs:
            continue
        equation_id = sha256_bytes(
            equation_key(a, b, c, x).encode("utf-8")
        )
        if equation_id in seen_equations:
            continue
        template_id = templates[rng.randrange(len(templates))]
        problem = render_problem(a, b, c, template_id)
        provisional = AlgorithmicEquationRecord(
            schema_version=ALGORITHMIC_RECORD_SCHEMA,
            record_id="",
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
            difficulty=difficulty,
            curriculum_band=band_name,
        )
        item = AlgorithmicEquationRecord(
            **{
                **provisional.to_dict(),
                "record_id": sha256_bytes(
                    canonical_json(_record_identity(provisional)).encode("utf-8")
                ),
            }
        )
        validate_algorithmic_record(item)
        seen_equations.add(equation_id)
        records.append(item)
    return records


def build_algorithmic_records(
    config: dict[str, Any],
) -> dict[str, list[AlgorithmicEquationRecord]]:
    data = config["data"]
    requests = [
        ("calibration", int(data["calibration_examples"])),
        ("train", int(data["train_examples"])),
        ("validation", int(data["validation_examples"])),
        ("test", int(data["test_examples"])),
        ("heldout_language", int(data["heldout_language_examples"])),
        ("extrapolation", int(data["extrapolation_examples"])),
        ("answer_extrapolation", int(data["answer_extrapolation_examples"])),
        ("compositional", int(data["compositional_examples"])),
    ]
    seen: set[str] = set()
    return {
        split: generate_algorithmic_split(config, split, count, seen)
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


def _jsonl_bytes(records: Iterable[AlgorithmicEquationRecord]) -> bytes:
    return b"".join(
        (canonical_json(record.to_dict()) + "\n").encode("utf-8")
        for record in records
    )


def prepare_algorithmic_manifests(
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
        audit_algorithmic_manifest(existing, root)
        return existing

    records_by_split = build_algorithmic_records(config)
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
            "difficulty_counts": dict(
                sorted(Counter(str(record.difficulty) for record in records).items())
            ),
            "curriculum_band_counts": dict(
                sorted(Counter(record.curriculum_band for record in records).items())
            ),
        }

    source_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "format": ALGORITHMIC_GENERATOR_FORMAT,
        "record_schema": ALGORITHMIC_RECORD_SCHEMA,
        "seed": int(config["project"]["seed"]),
        "config_sha256": config_sha256(config),
        "generator_path": str(source_path),
        "generator_sha256": file_sha256(source_path),
        "splits": split_metadata,
        "total_records": sum(item["count"] for item in split_metadata.values()),
        "normalized_equation_overlap": 0,
    }
    manifest["manifest_sha256"] = sha256_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    _atomic_write(
        manifest_path,
        (
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    audit_algorithmic_manifest(manifest, root)
    return manifest


def load_algorithmic_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                validate_algorithmic_record(item)
            except Exception as exc:
                raise ValueError(
                    f"invalid algorithmic record at {path}:{line_number}: {exc}"
                ) from exc
            records.append(item)
    return records


def audit_algorithmic_manifest(
    manifest: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    root_path = Path(root)
    if manifest.get("format") != ALGORITHMIC_GENERATOR_FORMAT:
        raise ValueError("unsupported algorithmic data manifest format")
    unsigned = dict(manifest)
    recorded_hash = unsigned.pop("manifest_sha256", None)
    if recorded_hash != sha256_bytes(canonical_json(unsigned).encode("utf-8")):
        raise ValueError("manifest hash mismatch")
    generator_path = Path(manifest["generator_path"])
    if not generator_path.is_file():
        raise FileNotFoundError(f"recorded generator source is missing: {generator_path}")
    if file_sha256(generator_path) != manifest["generator_sha256"]:
        raise ValueError("algorithmic data generator source hash mismatch")
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for split, metadata in manifest["splits"].items():
        path = root_path / metadata["path"]
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"split hash mismatch: {split}")
        records = load_algorithmic_records(path)
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


def curriculum_records(
    records: list[dict[str, Any]], config: dict[str, Any], epoch: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curriculum = config["data"].get("curriculum", {})
    if not curriculum.get("enabled", False):
        return records, {"enabled": False, "phase": "all"}
    phase = next(
        (
            item
            for item in curriculum["phases"]
            if int(epoch) <= int(item["through_epoch"])
        ),
        curriculum["phases"][-1],
    )
    maximum = int(phase["max_difficulty"])
    selected = [
        record for record in records if int(record.get("difficulty", 1)) <= maximum
    ]
    if not selected:
        raise RuntimeError(f"curriculum phase {phase['name']} selected no records")
    return selected, {
        "enabled": True,
        "phase": str(phase["name"]),
        "max_difficulty": maximum,
        "available_examples": len(selected),
        "total_examples": len(records),
    }
