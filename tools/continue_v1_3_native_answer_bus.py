from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter
from pathlib import Path

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_answer_bus import extract_answer_payload
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_dataset import V13Dataset
from cftn_text.v1_3_training import load_v1_3_data_contract, train_integration_phase
from tools.recover_v1_3_answer_bus import (
    ALL_CLASSES,
    TRAIN_CLASSES,
    configure_answer_bus_recovery,
    evaluate_native_answer_bus,
)


PHASE_NAME = "oracle_hard_answer_bus_native_continuation"
REQUIRED_VALIDATION_CLASSES = set(ALL_CLASSES)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in records), encoding="utf-8")


def _native_entries(record: dict, row: dict) -> list[dict]:
    entries: list[dict] = []
    for round_index, required in enumerate(record["required_specialists_by_round"][:2]):
        for specialist in required:
            generated = row["specialist_generations"][specialist][round_index]
            payload = extract_answer_payload(generated, strict=False)
            if payload is not None:
                entries.append({"round": round_index, "specialist": specialist, "payload": payload})
    return entries


def audit_validation_class_coverage(
    records: list[dict], *, expected_pure_language_examples: int | None = None
) -> dict[str, int]:
    counts = Counter(str(record["task_class"]) for record in records)
    missing = REQUIRED_VALIDATION_CLASSES.difference(counts)
    if missing:
        raise RuntimeError(
            "native continuation validation is missing required classes: "
            + ", ".join(sorted(missing))
        )
    if (
        expected_pure_language_examples is not None
        and int(counts["pure_language"]) != int(expected_pure_language_examples)
    ):
        raise RuntimeError(
            "native continuation pure-language panel differs from calibration: "
            f"expected {expected_pure_language_examples}, found {counts['pure_language']}"
        )
    return {name: int(counts[name]) for name in sorted(counts)}


def build_mixed_records(config: dict, failed_rows: Path, artifact: Path) -> tuple[Path, Path, dict]:
    data_root, manifest = load_v1_3_data_contract(config)
    records = V13Dataset(data_root / manifest["splits"]["joint_validation"]["path"]).records
    by_id = {record["record_id"]: record for record in records}
    usable: list[tuple[dict, list[dict]]] = []
    for row in _read_jsonl(failed_rows):
        record = by_id.get(row["record_id"])
        if record is None or record["task_class"] not in TRAIN_CLASSES:
            continue
        entries = _native_entries(record, row)
        if entries:
            usable.append((record, entries))
    usable.sort(key=lambda value: value[0]["record_id"])
    validation_ids = {record["record_id"] for index, (record, _) in enumerate(usable) if index % 5 == 0}
    train: list[dict] = []
    validation: list[dict] = []
    for record, entries in usable:
        noisy = copy.deepcopy(record)
        noisy["answer_bus_override"] = entries
        if record["record_id"] in validation_ids:
            validation.append(noisy)
        else:
            train.append(copy.deepcopy(record))
            train.append(noisy)
    # Protocol-aware validation always includes a clean language-preservation
    # panel. These rows do not train the composer, but they are required to
    # prove that specialist adaptation does not alter GPT-only behavior.
    calibration_path = (
        Path(config["paths"]["artifact_root"])
        / "gpt_language_calibration"
        / "report.json"
    )
    calibration = _read_json(calibration_path)
    if calibration.get("state") != "passed" or calibration.get("pass") is not True:
        raise RuntimeError("GPT language calibration is not a protected pass")
    expected_pure_language_examples = int(calibration["examples"])
    pure_language = [
        copy.deepcopy(record)
        for record in records
        if record["task_class"] == "pure_language"
    ][:expected_pure_language_examples]
    validation.extend(pure_language)
    validation_class_counts = audit_validation_class_coverage(
        validation,
        expected_pure_language_examples=expected_pure_language_examples,
    )
    train_path = artifact / "mixed_train.jsonl"
    validation_path = artifact / "native_holdout.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)
    audit = {
        "format": "cftn_text_v1_3_native_answer_bus_mix_v1",
        "clean_train_examples": len(train) // 2,
        "native_train_examples": len(train) // 2,
        "native_holdout_examples": len(validation),
        "native_specialist_holdout_examples": len(validation) - len(pure_language),
        "clean_pure_language_holdout_examples": len(pure_language),
        "validation_class_counts": validation_class_counts,
        "gpt_language_calibration": str(calibration_path.resolve()),
        "gpt_language_calibration_examples": expected_pure_language_examples,
        "holdout_rule": "sorted_record_id_modulo_5_equals_0",
        "failed_rows": str(failed_rows.resolve()),
        "failed_rows_sha256": file_sha256(failed_rows),
        "train_path": str(train_path.resolve()),
        "validation_path": str(validation_path.resolve()),
    }
    atomic_json_dump(audit, artifact / "mixture_audit.json")
    return train_path, validation_path, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="G:/ctfn-text/config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_v1_3_config(args.config)
    root = Path(config["paths"]["artifact_root"])
    recovery = root / "oracle_hard_answer_bus_recovery"
    failed_report = _read_json(recovery / "native_answer_bus_report.json")
    if failed_report.get("state") != "failed_acceptance":
        raise RuntimeError("native continuation requires the preserved failed native report")
    source = Path(failed_report["checkpoint"]).resolve()
    if file_sha256(source) != failed_report["checkpoint_sha256"]:
        raise RuntimeError("protected answer-composer checkpoint hash changed")
    artifact = root / PHASE_NAME
    if artifact.exists() and any(artifact.glob("*.pth")):
        raise RuntimeError("native continuation artifact already contains checkpoints")
    artifact.mkdir(parents=True, exist_ok=True)
    train_path, validation_path, audit = build_mixed_records(
        config, Path(failed_report["generation_rows"]), artifact
    )
    phase = configure_answer_bus_recovery(config, source)
    phase.update({
        "name": PHASE_NAME,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "train_records_override_jsonl": str(train_path),
        "validation_records_override_jsonl": str(validation_path),
        "repair_sequential_orders": False,
        "train_examples_by_class": {},
        "batch_size": 32,
        "eval_batch_size": 64,
        "minimum_epochs": 10,
        "max_epochs": 40,
        "early_stop_patience": 6,
        "answer_composer_learning_rate": 5.0e-5,
        "learning_rate": 5.0e-5,
        "minimum_learning_rate": 2.0e-6,
    })
    config["integration_training"]["phases"][-1] = phase
    status_path = root / "answer_bus_native_continuation_pipeline.json"
    status = {"state": "running", "pid": os.getpid(), "phase": PHASE_NAME, "source_checkpoint": str(source), "source_checkpoint_sha256": file_sha256(source), "mixture": audit, "started_unix": time.time()}
    atomic_json_dump(status, status_path)
    try:
        result = train_integration_phase(config, PHASE_NAME, device_name=args.device)
        report = evaluate_native_answer_bus(config, phase, Path(result["best_checkpoint"]), device_name=args.device, artifact_phase_name=PHASE_NAME)
        status.update({"state": "completed" if report["state"] == "passed" else "failed_acceptance", "result": result, "native_evaluation": report, "completed_unix": time.time()})
        atomic_json_dump(status, status_path)
    except BaseException as exc:
        status.update({"state": "error", "error": repr(exc), "failed_unix": time.time()})
        atomic_json_dump(status, status_path)
        raise


if __name__ == "__main__":
    main()
