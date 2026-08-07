from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import atomic_json_dump, gpu_status, load_checkpoint
from .config import config_sha256
from .data_generator import file_sha256
from .metrics import answer_generation_metrics
from .tokenizer import ByteMathTokenizer, pad_1d
from .training import (
    build_math_tower,
    load_data_contract,
    resolve_device,
    split_dataset,
)


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


@torch.inference_mode()
def generate_math_tower(
    model,
    tokenizer: ByteMathTokenizer,
    problems: list[str],
    *,
    max_new_tokens: int,
) -> tuple[list[str], list[int | None]]:
    if not problems:
        return [], []
    device = next(model.parameters()).device
    sequences = [
        tokenizer.encode_generation_prefix(problem, model.max_sequence_length)
        for problem in problems
    ]
    prefix_lengths = torch.tensor(
        [len(sequence) for sequence in sequences], dtype=torch.long, device=device
    )
    answer_predictions: list[int | None] | None = None
    finished = [False] * len(sequences)
    for _ in range(int(max_new_tokens)):
        input_ids, attention_mask = pad_1d(sequences, tokenizer.pad_token_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask, prefix_lengths)
        if answer_predictions is None:
            classes = output.answer_logits.argmax(dim=-1).tolist()
            answer_predictions = [
                int(index) + int(model.answer_min) for index in classes
            ]
        lengths = attention_mask.sum(dim=1) - 1
        next_tokens = output.logits[
            torch.arange(len(sequences), device=device), lengths
        ].argmax(dim=-1)
        for row, token in enumerate(next_tokens.tolist()):
            if finished[row]:
                continue
            if len(sequences[row]) >= model.max_sequence_length:
                finished[row] = True
                continue
            sequences[row].append(int(token))
            if int(token) == tokenizer.eos_token_id:
                finished[row] = True
        if all(finished):
            break
    generations = [
        tokenizer.decode(sequence[int(prefix_lengths[row].item()) :])
        for row, sequence in enumerate(sequences)
    ]
    return generations, answer_predictions or [None] * len(problems)


def evaluate_math_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device_name: str = "cuda",
    splits: list[str] | None = None,
    maximum_examples: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(config)
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
    split_names = splits or [
        "test",
        "heldout_language",
        "extrapolation",
        "compositional",
    ]
    artifact_root = Path(
        output_root
        or Path(config["project"]["artifact_root"]) / "evaluation_math"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "status.json"
    started_at = time.time()
    report: dict[str, Any] = {
        "format": "cftn_text_math_evaluation_v1",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": {},
    }
    for split_index, split in enumerate(split_names):
        records = split_dataset(data_root, manifest, split).records[:maximum]
        generations: list[str] = []
        answer_head_predictions: list[int | None] = []
        batch_size = int(settings["batch_size"])
        for start, chunk in _chunks(records, batch_size):
            generated, predicted = generate_math_tower(
                model,
                tokenizer,
                [record["problem"] for record in chunk],
                max_new_tokens=int(settings["max_math_new_tokens"]),
            )
            generations.extend(generated)
            answer_head_predictions.extend(predicted)
            completed = start + len(chunk)
            atomic_json_dump(
                {
                    "state": "running",
                    "phase": "math_generation",
                    "split": split,
                    "split_index": split_index + 1,
                    "splits_total": len(split_names),
                    "completed": completed,
                    "total": len(records),
                    "elapsed_seconds": time.time() - started_at,
                    "gpu": gpu_status(),
                },
                status_path,
            )
        targets = [int(record["x"]) for record in records]
        generation_metrics = answer_generation_metrics(generations, targets)
        answer_head_valid = [prediction is not None for prediction in answer_head_predictions]
        answer_head_correct = [
            prediction == target
            for prediction, target in zip(answer_head_predictions, targets)
        ]
        trace_exact = [
            generation.strip() == str(record["target_trace"])
            for generation, record in zip(generations, records)
        ]
        rows_path = artifact_root / f"{split}_generations.jsonl"
        with rows_path.open("w", encoding="utf-8") as handle:
            for record, generation, answer_prediction in zip(
                records, generations, answer_head_predictions
            ):
                handle.write(
                    json.dumps(
                        {
                            "record": record,
                            "math_generation": generation,
                            "answer_head_prediction": answer_prediction,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        report["splits"][split] = {
            "examples": len(records),
            "generation": generation_metrics,
            "answer_head_valid_rate": sum(answer_head_valid) / len(records),
            "answer_head_accuracy": sum(answer_head_correct) / len(records),
            "canonical_trace_exact_rate": sum(trace_exact) / len(records),
            "generation_rows": str(rows_path.resolve()),
        }
    test = report["splits"].get("test", {})
    extrapolation = report["splits"].get("extrapolation", {})
    report["specialist_gate"] = {
        "test_exact_accuracy_at_least_99_9": (
            test.get("generation", {}).get("exact_accuracy", 0.0) >= 0.999
        ),
        "test_valid_rate_100": (
            test.get("generation", {}).get("valid_rate", 0.0) >= 1.0
        ),
        "test_trace_exact_at_least_99": (
            test.get("canonical_trace_exact_rate", 0.0) >= 0.99
        ),
        "extrapolation_accuracy_at_least_95": (
            extrapolation.get("generation", {}).get("exact_accuracy", 0.0) >= 0.95
        ),
    }
    report["specialist_gate"]["pass"] = all(report["specialist_gate"].values())
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
    return report
