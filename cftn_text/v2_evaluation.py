from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump, gpu_status, load_checkpoint
from .config import config_sha256
from .data_generator import file_sha256
from .specialist_evaluation import generate_math_tower
from .tokenizer import ByteMathTokenizer
from .tokenizer import SequenceTooLongError
from .training import build_math_tower, load_data_contract, resolve_device, split_dataset
from .v2_metrics import extract_v2_answer, score_v2_generations
from .wandb_support import initialize_wandb


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def evaluate_v2_math_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device_name: str = "cuda",
    splits: list[str] | None = None,
    maximum_examples: int | None = None,
    output_root: str | Path | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(config)
    if manifest.get("format") != "cftn_text_broad_math_v2":
        raise ValueError("V2 evaluation requires a V2 manifest")
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_stage="math",
        expected_config_sha256=config_sha256(config),
        expected_manifest_sha256=manifest["manifest_sha256"],
        map_location=device,
    )
    model = build_math_tower(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    tokenizer = ByteMathTokenizer()
    settings = config["evaluation"]
    maximum = int(
        maximum_examples
        if maximum_examples is not None
        else settings["maximum_generation_examples"]
    )
    split_names = splits or list(settings["specialist_splits"])
    artifact_root = Path(
        output_root
        or Path(config["project"]["artifact_root"]) / "evaluation_math_v2"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "status.json"
    started_at = time.time()
    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact_root,
        stage="evaluation_math_v2",
        config={
            "project": config["project"]["name"],
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "config_sha256": config_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    report: dict[str, Any] = {
        "format": "cftn_text_math_evaluation_v2",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": {},
    }
    try:
        for split_index, split in enumerate(split_names):
            if split not in manifest["splits"]:
                continue
            available_records = split_dataset(data_root, manifest, split).records
            records: list[dict[str, Any]] = []
            excluded_over_context = 0
            for record in available_records:
                try:
                    tokenizer.encode_generation_prefix(
                        record["problem"], model.max_sequence_length
                    )
                except SequenceTooLongError:
                    excluded_over_context += 1
                    continue
                records.append(record)
                if len(records) >= maximum:
                    break
            generations: list[str] = []
            answer_head_predictions: list[int | None] = []
            for start, chunk in _chunks(records, int(settings["batch_size"])):
                generated, predicted = generate_math_tower(
                    model,
                    tokenizer,
                    [record["problem"] for record in chunk],
                    max_new_tokens=int(settings["max_math_new_tokens"]),
                )
                generations.extend(generated)
                answer_head_predictions.extend(predicted)
                atomic_json_dump(
                    {
                        "state": "running",
                        "phase": "v2_math_generation",
                        "split": split,
                        "split_index": split_index + 1,
                        "splits_total": len(split_names),
                        "completed": start + len(chunk),
                        "total": len(records),
                        "elapsed_seconds": time.time() - started_at,
                        "gpu": gpu_status(),
                    },
                    status_path,
                )
            metrics, correctness = score_v2_generations(generations, records)
            integer_rows = [
                index
                for index, record in enumerate(records)
                if record.get("answer_value") is not None
            ]
            answer_head_accuracy = (
                sum(
                    answer_head_predictions[index] == records[index]["answer_value"]
                    for index in integer_rows
                )
                / len(integer_rows)
                if integer_rows
                else None
            )
            rows_path = artifact_root / f"{split}_generations.jsonl"
            with rows_path.open("w", encoding="utf-8") as handle:
                for record, generation, prediction, correct in zip(
                    records, generations, answer_head_predictions, correctness
                ):
                    handle.write(
                        json.dumps(
                            {
                                "record": record,
                                "math_generation": generation,
                                "parsed_answer": extract_v2_answer(generation),
                                "correct": correct,
                                "answer_head_prediction": prediction,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            split_report = {
                **metrics,
                "excluded_over_context": excluded_over_context,
                "integer_answer_head_examples": len(integer_rows),
                "integer_answer_head_accuracy": answer_head_accuracy,
                "generation_rows": str(rows_path.resolve()),
            }
            report["splits"][split] = split_report
            tracker.log(
                {f"evaluation/{split}": split_report},
                global_step=split_index + 1,
                event="split_completed",
            )
        gates = settings.get("specialist_gates", {})
        gate_results: dict[str, bool] = {}
        for split, threshold in gates.items():
            if split in report["splits"]:
                gate_results[f"{split}_at_least_{threshold}"] = (
                    float(report["splits"][split]["accuracy"]) >= float(threshold)
                )
        report["specialist_gate"] = {
            **gate_results,
            "pass": bool(gate_results) and all(gate_results.values()),
        }
        atomic_json_dump(report, artifact_root / "report.json")
        atomic_json_dump(
            {
                "state": "completed",
                "elapsed_seconds": time.time() - started_at,
                "report": str((artifact_root / "report.json").resolve()),
                "gpu": gpu_status(),
            },
            status_path,
        )
        tracker.update_summary(
            {"run/state": "completed", "specialist_gate": report["specialist_gate"]}
        )
        tracker.finish()
        return report
    except BaseException as exc:
        atomic_json_dump(
            {
                "state": "error",
                "error": repr(exc),
                "elapsed_seconds": time.time() - started_at,
                "gpu": gpu_status(),
            },
            status_path,
        )
        tracker.update_summary({"run/state": "error", "run/error": repr(exc)})
        tracker.finish(exit_code=1)
        raise
