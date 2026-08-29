from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import canonical_json
from .data_generator import file_sha256
from .v2_data import make_v2_record, validate_v2_record


FORMAT = "cftn_canonical_math_curriculum_v1"
SCHEMA = "cftn_canonical_math_record_v1"
SPLITS = ("train", "validation", "test")


PHASES: tuple[dict[str, Any], ...] = (
    {
        "name": "y1_number_structure",
        "criteria": ("1NPV-1", "1NPV-2", "1AS-1"),
    },
    {
        "name": "y1_add_sub_fluency",
        "criteria": ("1NF-1", "1AS-2"),
    },
    {
        "name": "y2_place_value_and_across_10",
        "criteria": ("2NPV-1", "2NPV-2", "2AS-1", "2AS-2"),
    },
    {
        "name": "y2_add_sub_within_100",
        "criteria": ("2AS-3", "2AS-4"),
    },
    {
        "name": "y2_multiply_divide_2_5_10",
        "criteria": ("2MD-1", "2MD-2"),
    },
)

CRITERION_PHASE = {
    criterion: phase_index
    for phase_index, phase in enumerate(PHASES)
    for criterion in phase["criteria"]
}


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _phase_prerequisites(phase_index: int) -> list[str]:
    return [
        criterion
        for prior in PHASES[:phase_index]
        for criterion in prior["criteria"]
    ]


def _candidate_irs(criterion: str) -> Iterator[dict[str, Any]]:
    if criterion == "1NPV-1":
        for value in range(1, 100):
            yield {"type": "math_problem_v1", "op": "predecessor", "value": value}
            yield {"type": "math_problem_v1", "op": "successor", "value": value}
    elif criterion == "1NPV-2":
        for left in range(21):
            for right in range(21):
                if left != right:
                    yield {"type": "math_problem_v1", "op": "compare", "left": left, "right": right}
    elif criterion == "1AS-1":
        for left in range(11):
            for right in range(11 - left):
                yield {"type": "math_problem_v1", "op": "compose", "left": left, "right": right}
    elif criterion == "1NF-1":
        for left in range(11):
            for right in range(11):
                if left + right <= 10:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if left >= right:
                    yield {"type": "math_problem_v1", "op": "subtract", "left": left, "right": right}
    elif criterion == "1AS-2":
        for total in range(2, 21):
            for known in range(total + 1):
                yield {"type": "math_problem_v1", "op": "missing_addend", "known": known, "total": total}
    elif criterion == "2NPV-1":
        for value in range(10, 100):
            yield {"type": "math_problem_v1", "op": "place_value", "value": value}
    elif criterion == "2NPV-2":
        for value in range(11, 100):
            if value % 10:
                yield {"type": "math_problem_v1", "op": "neighbouring_tens", "value": value}
    elif criterion == "2AS-1":
        for left in range(1, 20):
            for right in range(1, 10):
                if left < 10 < left + right <= 20:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if 11 <= left <= 20 and left - right < 10:
                    yield {"type": "math_problem_v1", "op": "subtract", "left": left, "right": right}
    elif criterion == "2AS-2":
        for left in range(21):
            for right in range(left + 1):
                yield {"type": "math_problem_v1", "op": "difference", "left": left, "right": right}
    elif criterion == "2AS-3":
        for value in range(10, 100):
            for delta in (-10, -1, 1, 10):
                result = value + delta
                if 0 <= result <= 100:
                    yield {"type": "math_problem_v1", "op": "add_signed", "value": value, "delta": delta}
    elif criterion == "2AS-4":
        for left in range(10, 100):
            for right in range(10, 100):
                if left + right <= 100:
                    yield {"type": "math_problem_v1", "op": "add", "left": left, "right": right}
                if left >= right:
                    yield {"type": "math_problem_v1", "op": "subtract", "left": left, "right": right}
    elif criterion == "2MD-1":
        for factor in (2, 5, 10):
            for groups in range(22):
                yield {"type": "math_problem_v1", "op": "multiply", "left": factor, "right": groups}
    elif criterion == "2MD-2":
        for divisor in (2, 5, 10):
            for quotient in range(1, 31):
                yield {"type": "math_problem_v1", "op": "divide", "dividend": divisor * quotient, "divisor": divisor}
    else:
        raise ValueError(f"unknown criterion: {criterion}")


def solve_math_ir(math_ir: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    op = math_ir["op"]
    if op == "successor":
        result = int(math_ir["value"]) + 1
    elif op == "predecessor":
        result = int(math_ir["value"]) - 1
    elif op == "compare":
        left, right = int(math_ir["left"]), int(math_ir["right"])
        result = "<" if left < right else ">" if left > right else "="
    elif op in {"add", "compose"}:
        result = int(math_ir["left"]) + int(math_ir["right"])
    elif op == "subtract":
        result = int(math_ir["left"]) - int(math_ir["right"])
    elif op == "difference":
        result = abs(int(math_ir["left"]) - int(math_ir["right"]))
    elif op == "missing_addend":
        result = int(math_ir["total"]) - int(math_ir["known"])
    elif op == "place_value":
        value = int(math_ir["value"])
        result = f"{value // 10},{value % 10}"
    elif op == "neighbouring_tens":
        value = int(math_ir["value"])
        result = f"{value // 10 * 10},{value // 10 * 10 + 10}"
    elif op == "add_signed":
        result = int(math_ir["value"]) + int(math_ir["delta"])
    elif op == "multiply":
        result = int(math_ir["left"]) * int(math_ir["right"])
    elif op == "divide":
        dividend, divisor = int(math_ir["dividend"]), int(math_ir["divisor"])
        if divisor == 0 or dividend % divisor:
            raise ValueError("division example must have an exact non-zero divisor")
        result = dividend // divisor
    else:
        raise ValueError(f"unsupported operation: {op}")
    answer = str(result)
    return answer, [{"op": op, "result": answer}]


def _language_prompts(math_ir: dict[str, Any]) -> tuple[str, ...]:
    op = math_ir["op"]
    if op == "successor":
        value = math_ir["value"]
        return (f"What number comes after {value}?", f"Give the successor of {value}.")
    if op == "predecessor":
        value = math_ir["value"]
        return (f"What number comes before {value}?", f"Give the predecessor of {value}.")
    if op == "compare":
        left, right = math_ir["left"], math_ir["right"]
        return (f"Compare {left} and {right}.", f"Which symbol, <, >, or =, belongs between {left} and {right}?")
    if op in {"add", "compose", "subtract", "difference", "multiply"}:
        left, right = math_ir["left"], math_ir["right"]
        templates = {
            "add": (f"Calculate {left} + {right}.", f"What is the sum of {left} and {right}?"),
            "compose": (f"Compose {left} and {right} into a whole.", f"What whole is made from parts {left} and {right}?"),
            "subtract": (f"Calculate {left} - {right}.", f"Subtract {right} from {left}."),
            "difference": (f"Find the difference between {left} and {right}.", f"How far apart are {left} and {right}?"),
            "multiply": (f"Calculate {left} x {right}.", f"What is the product of {left} and {right}?"),
        }
        return templates[op]
    if op == "missing_addend":
        known, total = math_ir["known"], math_ir["total"]
        return (f"Complete {known} + ? = {total}.", f"What must be added to {known} to make {total}?")
    if op == "place_value":
        value = math_ir["value"]
        return (f"How many tens and ones are in {value}?", f"Partition {value} into tens and ones.")
    if op == "neighbouring_tens":
        value = math_ir["value"]
        return (f"Give the multiples of ten immediately below and above {value}.", f"Which two tens does {value} lie between?")
    if op == "add_signed":
        value, delta = math_ir["value"], math_ir["delta"]
        return (f"Calculate {value} + ({delta}).", f"Change {value} by {delta}.")
    if op == "divide":
        dividend, divisor = math_ir["dividend"], math_ir["divisor"]
        return (f"Calculate {dividend} divided by {divisor}.", f"How many groups of {divisor} are in {dividend}?")
    raise ValueError(f"no language templates for operation: {op}")


def _trace(answer: str, derivation: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    work = canonical_json(derivation)
    trace = f"<work>{work}</work><answer>{answer}</answer>"
    work_result = json.dumps(answer, ensure_ascii=False)
    work_start = trace.index(work_result)
    answer_start = trace.index(answer, trace.index("<answer>"))
    spans = [
        {"kind": "compute", "start": work_start, "end": work_start + len(work_result)},
        {"kind": "copy", "start": answer_start, "end": answer_start + len(answer)},
    ]
    return trace, spans


def _records_for_object(
    *, split: str, criterion: str, math_ir: dict[str, Any], phase_index: int
) -> Iterator[dict[str, Any]]:
    answer, derivation = solve_math_ir(math_ir)
    math_ir_text = canonical_json(math_ir)
    trace, spans = _trace(answer, derivation)
    object_id = _sha(math_ir)
    phase = PHASES[phase_index]
    prompts = _language_prompts(math_ir)
    for variant, prompt in enumerate(prompts):
        dispatcher_target = {
            "route": "math",
            "criterion_id": criterion,
            "math_ir": math_ir,
        }
        extras = {
            "curriculum_schema": SCHEMA,
            "natural_language_prompt": prompt,
            "dispatcher_target": dispatcher_target,
            "math_ir": math_ir,
            "derivation": derivation,
            "answer": answer,
            "verifier_spec": {"kind": "exact_math_ir_v1", "math_ir": math_ir},
            "criterion_id": criterion,
            "curriculum_phase": phase["name"],
            "curriculum_phase_index": phase_index,
            "prerequisite_ids": _phase_prerequisites(phase_index),
            "numeric_domain": "non_negative_integers_0_100",
            "representation": "canonical_json_math_ir_v1",
            "evaluation_mode": "held_out_objects_within_taught_domain",
            "math_object_id": object_id,
            "language_variant": variant,
            "computation_spans": spans,
        }
        yield make_v2_record(
            split=split,
            source="canonical_primary_math",
            family=criterion,
            difficulty=min(phase_index + 1, 3),
            problem=math_ir_text,
            raw_problem=prompt,
            answer=answer,
            target_trace=trace,
            native_program=math_ir_text,
            execution_trace=f"<work>{canonical_json(derivation)}</work>",
            gpt_problem=prompt,
            math_problem=math_ir_text,
            metadata={"criterion_id": criterion, "phase": phase["name"]},
            extra_fields=extras,
        )


def _split_objects(
    criterion: str, split_object_counts: dict[str, int], seed: int
) -> dict[str, list[dict[str, Any]]]:
    candidates = list(_candidate_irs(criterion))
    required = sum(split_object_counts.values())
    if len(candidates) < required:
        raise ValueError(
            f"criterion {criterion} has {len(candidates)} objects, needs {required}"
        )
    candidates.sort(key=lambda item: _sha([seed, criterion, item]))
    output: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split in SPLITS:
        count = int(split_object_counts[split])
        output[split] = candidates[offset : offset + count]
        offset += count
    return output


def iter_records(config: dict[str, Any], split: str) -> Iterator[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    seed = int(config["seed"])
    split_object_counts = {
        name: int(config["objects_per_criterion"][name]) for name in SPLITS
    }
    for phase_index, phase in enumerate(PHASES):
        for criterion in phase["criteria"]:
            objects = _split_objects(criterion, split_object_counts, seed)[split]
            for math_ir in objects:
                yield from _records_for_object(
                    split=split,
                    criterion=criterion,
                    math_ir=math_ir,
                    phase_index=phase_index,
                )


def iter_phase_training_records(
    config: dict[str, Any], phase_index: int
) -> Iterator[dict[str, Any]]:
    """Yield one deterministic phase view without exposing future criteria."""
    if phase_index < 0 or phase_index >= len(PHASES):
        raise ValueError(f"invalid phase index: {phase_index}")
    active_criteria = set(PHASES[phase_index]["criteria"])
    prior_criteria = _phase_prerequisites(phase_index)
    active_rows = [
        record
        for record in iter_records(config, "train")
        if record["criterion_id"] in active_criteria
    ]
    active_fraction = float(config["replay_policy"]["active_fraction"])
    prior_fraction = float(config["replay_policy"]["prior_fraction"])
    if abs(active_fraction + prior_fraction - 1.0) > 1e-9:
        raise ValueError("active and prior replay fractions must sum to one")
    yield from active_rows
    if not prior_criteria:
        return
    replay_total = round(len(active_rows) * prior_fraction / active_fraction)
    base, remainder = divmod(replay_total, len(prior_criteria))
    rows_by_criterion: dict[str, list[dict[str, Any]]] = {
        criterion: [] for criterion in prior_criteria
    }
    for record in iter_records(config, "train"):
        criterion = record["criterion_id"]
        if criterion in rows_by_criterion:
            rows_by_criterion[criterion].append(record)
    for criterion_index, criterion in enumerate(prior_criteria):
        count = base + (1 if criterion_index < remainder else 0)
        rows = rows_by_criterion[criterion]
        rows.sort(key=lambda record: _sha([config["seed"], phase_index, record["record_id"]]))
        if count > len(rows):
            raise ValueError(f"not enough replay rows for criterion {criterion}")
        yield from rows[:count]


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def prepare_dataset(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    if config.get("format") != FORMAT:
        raise ValueError(f"config format must be {FORMAT!r}")
    output_root = Path(output_root)
    if (output_root / "manifest.json").exists():
        raise FileExistsError(
            f"sealed dataset already exists at {output_root}; choose a new output path"
        )
    files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = output_root / f"{split}.jsonl"
        count = _write_jsonl(path, iter_records(config, split))
        files[split] = {
            "path": path.name,
            "records": count,
            "sha256": file_sha256(path),
        }
    phase_files: dict[str, dict[str, Any]] = {}
    for phase_index, phase in enumerate(PHASES):
        path = output_root / "phase_views" / f"{phase_index:02d}_{phase['name']}.train.jsonl"
        count = _write_jsonl(path, iter_phase_training_records(config, phase_index))
        phase_files[phase["name"]] = {
            "phase_index": phase_index,
            "path": path.relative_to(output_root).as_posix(),
            "records": count,
            "sha256": file_sha256(path),
        }
    manifest = {
        "format": FORMAT,
        "schema": SCHEMA,
        "config": config,
        "config_sha256": _sha(config),
        "generator_sha256": file_sha256(Path(__file__)),
        "seed": int(config["seed"]),
        "objects_per_criterion": config["objects_per_criterion"],
        "language_variants_per_object": 2,
        "phases": list(PHASES),
        "replay_policy": {
            "active_fraction": float(config["replay_policy"]["active_fraction"]),
            "prior_fraction": float(config["replay_policy"]["prior_fraction"]),
            "prior_sampling": "criterion_balanced_all_accepted_phases",
            "future_phase_exposure": "forbidden",
        },
        "files": files,
        "phase_files": phase_files,
    }
    _atomic_write_json(output_root / "manifest.json", manifest)
    audit = audit_dataset(output_root)
    manifest["audit"] = audit
    _atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def audit_dataset(output_root: Path, scratch_dir: Path | None = None) -> dict[str, Any]:
    output_root = Path(output_root)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("not a canonical math curriculum manifest")
    if _sha(manifest.get("config")) != manifest.get("config_sha256"):
        raise ValueError("manifest config hash mismatch")
    if file_sha256(Path(__file__)) != manifest.get("generator_sha256"):
        raise ValueError("dataset was built by a different generator revision")
    scratch = Path(scratch_dir) if scratch_dir else None
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)
    fd, db_name = tempfile.mkstemp(prefix="math-curriculum-audit-", suffix=".sqlite3", dir=scratch)
    os.close(fd)
    db_path = Path(db_name)
    counters: Counter[str] = Counter()
    criterion_counts: Counter[str] = Counter()
    try:
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE records (record_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        connection.execute("CREATE TABLE objects (object_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        connection.execute("CREATE TABLE prompts (prompt_hash TEXT PRIMARY KEY, split TEXT NOT NULL)")
        for split in SPLITS:
            file_info = manifest["files"][split]
            path = output_root / file_info["path"]
            if file_sha256(path) != file_info["sha256"]:
                raise ValueError(f"hash mismatch for {path}")
            for line_number, record in _iter_jsonl(path):
                try:
                    validate_v2_record(record)
                    if record.get("curriculum_schema") != SCHEMA:
                        raise ValueError("wrong curriculum schema")
                    if record["split"] != split:
                        raise ValueError("record split mismatch")
                    criterion = record["criterion_id"]
                    phase_index = int(record["curriculum_phase_index"])
                    if CRITERION_PHASE.get(criterion) != phase_index:
                        raise ValueError("criterion appears in the wrong phase")
                    if record["prerequisite_ids"] != _phase_prerequisites(phase_index):
                        raise ValueError("prerequisite list is not cumulative and exact")
                    expected_answer, expected_derivation = solve_math_ir(record["math_ir"])
                    if record["answer"] != expected_answer:
                        raise ValueError("answer does not match executable math IR")
                    if record["derivation"] != expected_derivation:
                        raise ValueError("derivation does not match executable math IR")
                    if record["problem"] != canonical_json(record["math_ir"]):
                        raise ValueError("math tower input is not canonical math IR")
                    if record["gpt_problem"] != record["natural_language_prompt"]:
                        raise ValueError("dispatcher prompt columns disagree")
                    if record["math_problem"] != record["problem"]:
                        raise ValueError("math private view contains language")
                    if record["dispatcher_target"]["math_ir"] != record["math_ir"]:
                        raise ValueError("dispatcher target math IR disagrees")
                    validate_spans(record)
                    connection.execute(
                        "INSERT INTO records(record_id, split) VALUES (?, ?)",
                        (record["record_id"], split),
                    )
                    object_id = record["math_object_id"]
                    existing = connection.execute(
                        "SELECT split FROM objects WHERE object_id = ?", (object_id,)
                    ).fetchone()
                    if existing is None:
                        connection.execute(
                            "INSERT INTO objects(object_id, split) VALUES (?, ?)",
                            (object_id, split),
                        )
                    elif existing[0] != split:
                        raise ValueError("math object occurs in multiple splits")
                    prompt_hash = _sha(record["natural_language_prompt"].casefold())
                    connection.execute(
                        "INSERT INTO prompts(prompt_hash, split) VALUES (?, ?)",
                        (prompt_hash, split),
                    )
                except Exception as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                counters[f"records.{split}"] += 1
                criterion_counts[f"{split}.{criterion}"] += 1
            connection.commit()
            if counters[f"records.{split}"] != int(file_info["records"]):
                raise ValueError(f"record count mismatch for {split}")
        phase_audits: dict[str, Any] = {}
        for phase_name, file_info in manifest["phase_files"].items():
            phase_index = int(file_info["phase_index"])
            path = output_root / file_info["path"]
            if file_sha256(path) != file_info["sha256"]:
                raise ValueError(f"hash mismatch for {path}")
            allowed = set(_phase_prerequisites(phase_index)) | set(
                PHASES[phase_index]["criteria"]
            )
            active = set(PHASES[phase_index]["criteria"])
            phase_counts: Counter[str] = Counter()
            for line_number, record in _iter_jsonl(path):
                criterion = record.get("criterion_id")
                if criterion not in allowed:
                    raise ValueError(f"{path}:{line_number}: future-phase exposure")
                phase_counts["total"] += 1
                phase_counts["active" if criterion in active else "replay"] += 1
                phase_counts[f"criterion.{criterion}"] += 1
            if phase_counts["total"] != int(file_info["records"]):
                raise ValueError(f"phase view count mismatch for {phase_name}")
            phase_audits[phase_name] = {
                "total": phase_counts["total"],
                "active": phase_counts["active"],
                "replay": phase_counts["replay"],
                "active_fraction": phase_counts["active"] / phase_counts["total"],
                "criterion_counts": {
                    key.removeprefix("criterion."): value
                    for key, value in sorted(phase_counts.items())
                    if key.startswith("criterion.")
                },
            }
        return {
            "status": "passed",
            "records": {split: counters[f"records.{split}"] for split in SPLITS},
            "criterion_counts": dict(sorted(criterion_counts.items())),
            "phase_views": phase_audits,
            "checks": [
                "streaming_json_validation",
                "sqlite_bounded_memory_uniqueness",
                "no_math_object_split_overlap",
                "no_prompt_split_overlap",
                "executable_answer_and_derivation",
                "canonical_language_free_math_view",
                "strict_phase_and_prerequisite_metadata",
                "future_phase_training_exposure_forbidden",
                "criterion_balanced_cumulative_replay",
                "valid_computation_spans",
            ],
        }
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
        db_path.unlink(missing_ok=True)


def validate_spans(record: dict[str, Any]) -> None:
    trace = record["target_trace"]
    spans = record["computation_spans"]
    kinds = {span["kind"] for span in spans}
    if "compute" not in kinds or "copy" not in kinds:
        raise ValueError("trace needs compute and copy spans")
    previous_end = -1
    for span in sorted(spans, key=lambda value: int(value["start"])):
        start, end = int(span["start"]), int(span["end"])
        if start < 0 or end <= start or end > len(trace) or start < previous_end:
            raise ValueError("invalid or overlapping computation span")
        previous_end = end


def dataset_summary(output_root: Path) -> dict[str, Any]:
    manifest = json.loads((Path(output_root) / "manifest.json").read_text(encoding="utf-8"))
    return {
        "format": manifest["format"],
        "schema": manifest["schema"],
        "files": manifest["files"],
        "phase_files": manifest["phase_files"],
        "phases": manifest["phases"],
        "replay_policy": manifest["replay_policy"],
        "audit": manifest.get("audit"),
    }


def sample_records(output_root: Path, split: str, limit: int) -> list[dict[str, Any]]:
    manifest = json.loads((Path(output_root) / "manifest.json").read_text(encoding="utf-8"))
    path = Path(output_root) / manifest["files"][split]["path"]
    rows: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(path):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows
