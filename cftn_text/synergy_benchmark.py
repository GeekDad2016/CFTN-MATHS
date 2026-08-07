from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .complementary import complementary_record
from .config import canonical_json, config_sha256
from .data_generator import (
    EquationRecord,
    canonical_answer,
    canonical_trace,
    equation_key,
    file_sha256,
    load_records,
    render_problem,
    sha256_bytes,
    validate_record,
)
from .training import load_data_contract


SYNERGY_BENCHMARK_FORMAT = "cftn_text_synergy_benchmark_v1"


def load_synergy_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).expanduser().resolve()
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("synergy protocol root must be a mapping")
    if protocol.get("format") != SYNERGY_BENCHMARK_FORMAT:
        raise ValueError("unsupported synergy protocol format")
    benchmark = protocol.get("benchmark", {})
    splits = benchmark.get("source_splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("synergy protocol requires source splits")
    if int(benchmark.get("examples_per_split", 0)) < 1:
        raise ValueError("synergy examples_per_split must be positive")
    protocol["_meta"] = {
        "path": str(protocol_path),
        "sha256": sha256_bytes(canonical_json(protocol).encode("utf-8")),
    }
    return protocol


def _selection_score(seed: int, split: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{record_id}".encode("utf-8")).hexdigest()


def _selected_records(
    records: list[dict[str, Any]], *, split: str, count: int, seed: int
) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(
            f"synergy benchmark requests {count} {split} records but only "
            f"{len(records)} exist"
        )
    return sorted(
        records,
        key=lambda record: _selection_score(seed, split, str(record["record_id"])),
    )[:count]


def _counterfactual_record(
    record: dict[str, Any],
    *,
    seed: int,
    seen_equations: set[str],
    answer_min: int,
    answer_max: int,
    maximum_abs_intermediate: int,
) -> tuple[dict[str, Any], int]:
    deltas = tuple(range(-16, 0)) + tuple(range(1, 17))
    offset = int.from_bytes(
        hashlib.sha256(f"{seed}:{record['record_id']}:delta".encode("utf-8")).digest()[:2],
        "big",
    ) % len(deltas)
    for index in range(len(deltas)):
        delta = deltas[(offset + index) % len(deltas)]
        x = int(record["x"]) + delta
        if not answer_min <= x <= answer_max:
            continue
        a = int(record["a"])
        b = int(record["b"])
        c = a * x + b
        if max(abs(c), abs(c - b), abs(a * x)) > maximum_abs_intermediate:
            continue
        equation_id = sha256_bytes(equation_key(a, b, c, x).encode("utf-8"))
        if equation_id in seen_equations:
            continue
        template_id = str(record["template_id"])
        problem = render_problem(a, b, c, template_id)
        payload = {
            "equation_id": equation_id,
            "problem": problem,
            "split": str(record["split"]),
            "template_id": template_id,
        }
        item = EquationRecord(
            record_id=sha256_bytes(canonical_json(payload).encode("utf-8")),
            equation_id=equation_id,
            split=str(record["split"]),
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
        return item.to_dict(), delta
    raise RuntimeError(f"could not construct counterfactual for {record['record_id']}")


def _benchmark_row(
    record: dict[str, Any],
    *,
    pair_id: str,
    variant: str,
    source_split: str,
    target_delta: int,
    seed: int,
    add_distractor: bool,
) -> dict[str, Any]:
    row = complementary_record(
        record,
        seed=seed,
        assignment_key=pair_id,
        add_distractor=add_distractor,
    )
    row["benchmark"] = {
        "pair_id": pair_id,
        "pair_variant": variant,
        "source_split": source_split,
        "target_delta_from_base": int(target_delta),
        "language_novelty": source_split == "heldout_language",
        "numerical_extrapolation": source_split == "extrapolation",
        "structural_rearrangement": source_split == "compositional",
        "counterfactual": variant == "counterfactual",
        "distractor": bool(add_distractor),
        "gpt_problem_utf8_bytes": len(row["gpt_problem"].encode("utf-8")),
        "math_problem_utf8_bytes": len(row["math_problem"].encode("utf-8")),
    }
    row["benchmark_record_id"] = sha256_bytes(
        canonical_json(
            {
                "pair_id": pair_id,
                "variant": variant,
                "record_id": row["record_id"],
                "gpt_problem": row["gpt_problem"],
                "math_problem": row["math_problem"],
            }
        ).encode("utf-8")
    )
    return row


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_synergy_benchmark(
    config: dict[str, Any],
    protocol: dict[str, Any],
    *,
    output_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not force:
        existing = audit_synergy_benchmark(manifest_path)
        clean_protocol = dict(protocol)
        clean_protocol.pop("_meta", None)
        if existing["config_sha256"] != config_sha256(config):
            raise FileExistsError(
                "existing synergy benchmark belongs to a different configuration"
            )
        if existing["protocol_sha256"] != sha256_bytes(
            canonical_json(clean_protocol).encode("utf-8")
        ):
            raise FileExistsError(
                "existing synergy benchmark belongs to a different protocol"
            )
        return existing
    data_root, source_manifest = load_data_contract(config)
    settings = protocol["benchmark"]
    seed = int(protocol.get("seed", config["project"]["seed"]))
    count = int(settings["examples_per_split"])
    source_splits = [str(split) for split in settings["source_splits"]]
    distractor_splits = set(settings.get("distractor_splits", []))
    all_source_equations: set[str] = set()
    for metadata in source_manifest["splits"].values():
        all_source_equations.update(
            record["equation_id"]
            for record in load_records(data_root / metadata["path"])
        )
    seen_equations = set(all_source_equations)
    split_metadata: dict[str, Any] = {}
    total_pairs = 0
    for split in source_splits:
        if split not in source_manifest["splits"]:
            raise ValueError(f"synergy source split does not exist: {split}")
        records = load_records(data_root / source_manifest["splits"][split]["path"])
        selected = _selected_records(records, split=split, count=count, seed=seed)
        rows: list[dict[str, Any]] = []
        for base in selected:
            pair_id = sha256_bytes(
                f"{SYNERGY_BENCHMARK_FORMAT}:{seed}:{base['record_id']}".encode("utf-8")
            )
            counterfactual, delta = _counterfactual_record(
                base,
                seed=seed,
                seen_equations=seen_equations,
                answer_min=int(config["math_tower"]["answer_min"]),
                answer_max=int(config["math_tower"]["answer_max"]),
                maximum_abs_intermediate=int(config["data"]["maximum_abs_intermediate"]),
            )
            add_distractor = split in distractor_splits
            rows.append(
                _benchmark_row(
                    base,
                    pair_id=pair_id,
                    variant="base",
                    source_split=split,
                    target_delta=0,
                    seed=seed,
                    add_distractor=add_distractor,
                )
            )
            rows.append(
                _benchmark_row(
                    counterfactual,
                    pair_id=pair_id,
                    variant="counterfactual",
                    source_split=split,
                    target_delta=delta,
                    seed=seed,
                    add_distractor=add_distractor,
                )
            )
        payload = b"".join(
            (canonical_json(row) + "\n").encode("utf-8") for row in rows
        )
        path = root / f"{split}.jsonl"
        _atomic_write(path, payload)
        split_metadata[split] = {
            "path": path.name,
            "pairs": len(selected),
            "records": len(rows),
            "sha256": sha256_bytes(payload),
            "source_split_sha256": source_manifest["splits"][split]["sha256"],
            "selected_source_record_ids_sha256": sha256_bytes(
                "\n".join(record["record_id"] for record in selected).encode("utf-8")
            ),
        }
        total_pairs += len(selected)
    clean_protocol = dict(protocol)
    clean_protocol.pop("_meta", None)
    manifest: dict[str, Any] = {
        "format": SYNERGY_BENCHMARK_FORMAT,
        "seed": seed,
        "config_sha256": config_sha256(config),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "protocol": clean_protocol,
        "protocol_sha256": sha256_bytes(canonical_json(clean_protocol).encode("utf-8")),
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": file_sha256(__file__),
        "splits": split_metadata,
        "total_pairs": total_pairs,
        "total_records": total_pairs * 2,
        "counterfactual_source_overlap": 0,
    }
    unsigned = dict(manifest)
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(unsigned).encode("utf-8"))
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return audit_synergy_benchmark(manifest_path)


def load_synergy_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("view_mode") != "complementary":
                raise ValueError(f"invalid synergy view at {path}:{line_number}")
            assignment = row["role_assignment"]
            slots = row["slot_values"]
            expected = {
                "coefficient": int(row["a"]),
                "offset": int(row["b"]),
                "result": int(row["c"]),
            }
            for role, value in expected.items():
                if int(slots[assignment[role]]) != value:
                    raise ValueError(f"slot/role mismatch at {path}:{line_number}")
            rows.append(row)
    return rows


def audit_synergy_benchmark(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != SYNERGY_BENCHMARK_FORMAT:
        raise ValueError("unsupported synergy benchmark manifest")
    unsigned = dict(manifest)
    recorded_hash = unsigned.pop("manifest_sha256", None)
    if recorded_hash != sha256_bytes(canonical_json(unsigned).encode("utf-8")):
        raise ValueError("synergy benchmark manifest hash mismatch")
    builder_path = Path(manifest["builder_path"])
    if file_sha256(builder_path) != manifest["builder_sha256"]:
        raise ValueError("synergy benchmark builder source changed")
    total_pairs = 0
    total_records = 0
    for split, metadata in manifest["splits"].items():
        rows_path = path.parent / metadata["path"]
        if file_sha256(rows_path) != metadata["sha256"]:
            raise ValueError(f"synergy split hash mismatch: {split}")
        rows = load_synergy_rows(rows_path)
        if len(rows) != int(metadata["records"]):
            raise ValueError(f"synergy split count mismatch: {split}")
        if len(rows) % 2:
            raise ValueError(f"synergy split has an unpaired row: {split}")
        for index in range(0, len(rows), 2):
            base, counterfactual = rows[index : index + 2]
            if base["benchmark"]["pair_variant"] != "base":
                raise ValueError("synergy pair does not start with its base row")
            if counterfactual["benchmark"]["pair_variant"] != "counterfactual":
                raise ValueError("synergy pair does not end with its counterfactual row")
            if base["benchmark"]["pair_id"] != counterfactual["benchmark"]["pair_id"]:
                raise ValueError("synergy pair IDs differ")
            if base["gpt_problem"] != counterfactual["gpt_problem"]:
                raise ValueError("counterfactual changed the language-side role view")
            delta = int(counterfactual["benchmark"]["target_delta_from_base"])
            if int(counterfactual["x"]) - int(base["x"]) != delta:
                raise ValueError("counterfactual target delta is invalid")
        total_pairs += len(rows) // 2
        total_records += len(rows)
    if total_pairs != int(manifest["total_pairs"]):
        raise ValueError("synergy total pair count mismatch")
    if total_records != int(manifest["total_records"]):
        raise ValueError("synergy total record count mismatch")
    return {
        **manifest,
        "audit": {
            "pass": True,
            "total_pairs": total_pairs,
            "total_records": total_records,
        },
    }
