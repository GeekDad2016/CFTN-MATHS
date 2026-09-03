from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .checkpoint import atomic_json_dump, gpu_status, load_checkpoint
from .config import config_sha256
from .data_generator import file_sha256
from .metrics import answer_generation_metrics
from .tokenizer import ByteMathTokenizer, pad_1d
from .training import (
    build_math_tower_for_checkpoint,
    load_data_contract,
    resolve_device,
    split_dataset,
)


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def _acceptance_report(
    report: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    configured = config["evaluation"].get("specialist_acceptance")
    if not configured:
        test = report["splits"].get("test", {})
        extrapolation = report["splits"].get("extrapolation", {})
        legacy = {
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
                extrapolation.get("generation", {}).get("exact_accuracy", 0.0)
                >= 0.95
            ),
        }
        legacy["pass"] = all(legacy.values())
        return legacy

    details: dict[str, Any] = {}
    for split, criteria in configured.items():
        split_report = report["splits"].get(split, {})
        generation = split_report.get("generation", {})
        split_details: dict[str, Any] = {}
        for metric, threshold_value in criteria.items():
            threshold = float(threshold_value)
            if metric == "trace_exact_rate":
                observed = float(split_report.get("canonical_trace_exact_rate", 0.0))
            else:
                observed = float(generation.get(metric, 0.0))
            split_details[metric] = {
                "observed": observed,
                "threshold": threshold,
                "pass": observed >= threshold,
            }
        details[split] = split_details
    return {
        "name": "standalone_specialist_acceptance",
        "criteria": details,
        "pass": all(
            criterion["pass"]
            for split_details in details.values()
            for criterion in split_details.values()
        ),
    }


@torch.inference_mode()
def generate_math_tower(
    model,
    tokenizer: ByteMathTokenizer,
    problems: list[str],
    *,
    max_new_tokens: int,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[int | None]]:
    if not problems:
        return [], []
    device = next(model.parameters()).device
    if hasattr(model, "begin_cached_generation") and hasattr(
        model, "cached_generation_step"
    ) and hasattr(model, "compact_cached_generation"):
        # Group identical-length prefixes so that cached decoding can use a
        # real batch without padding.  Finished rows are removed from the
        # active KV cache before the next token step, so a runaway trace does
        # not keep already-complete answers on the GPU.
        prefixes_by_length: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
        for row, problem in enumerate(problems):
            sequence = tokenizer.encode_generation_prefix(
                problem, model.max_sequence_length
            )
            prefixes_by_length[len(sequence)].append((row, sequence))

        generated_ids: list[list[int]] = [[] for _ in problems]
        answer_predictions: list[int | None] = [None] * len(problems)
        finish_reasons: list[str | None] = [None] * len(problems)
        generated_token_counts = [0] * len(problems)
        cached_batch_sizes = [0] * len(problems)

        for grouped in prefixes_by_length.values():
            original_rows = [row for row, _ in grouped]
            prefixes = [sequence for _, sequence in grouped]
            cache, output = model.begin_cached_generation(
                torch.tensor(prefixes, dtype=torch.long, device=device)
            )
            for local_row, original_row in enumerate(original_rows):
                cached_batch_sizes[original_row] = len(original_rows)
                if model.answer_head_enabled:
                    answer_predictions[original_row] = (
                        int(output.answer_logits[local_row].argmax(dim=-1).item())
                        + int(model.answer_min)
                    )

            active_rows = list(range(len(original_rows)))
            for step in range(int(max_new_tokens)):
                if cache.length >= model.max_sequence_length:
                    for local_row in active_rows:
                        finish_reasons[original_rows[local_row]] = "context_limit"
                    active_rows = []
                    break

                next_tokens = output.logits[:, -1].argmax(dim=-1)
                survivor_positions: list[int] = []
                for position, (local_row, token) in enumerate(
                    zip(active_rows, next_tokens.tolist())
                ):
                    original_row = original_rows[local_row]
                    generated_ids[original_row].append(int(token))
                    generated_token_counts[original_row] += 1
                    if int(token) == tokenizer.eos_token_id:
                        finish_reasons[original_row] = "eos"
                    else:
                        survivor_positions.append(position)

                if not survivor_positions:
                    active_rows = []
                    break
                if step + 1 >= int(max_new_tokens):
                    break

                survivor_tensor = torch.tensor(
                    survivor_positions, dtype=torch.long, device=device
                )
                if len(survivor_positions) != len(active_rows):
                    model.compact_cached_generation(cache, survivor_tensor)
                active_rows = [active_rows[position] for position in survivor_positions]
                output = model.cached_generation_step(
                    cache, next_tokens.index_select(0, survivor_tensor)
                )

            for local_row in active_rows:
                original_row = original_rows[local_row]
                if finish_reasons[original_row] is None:
                    finish_reasons[original_row] = "budget"

        generations = [tokenizer.decode(values) for values in generated_ids]
        if diagnostics is not None:
            diagnostics.extend(
                {
                    "generated_tokens": generated_token_counts[row],
                    "eos_terminated": finish_reasons[row] == "eos",
                    "context_limit_hit": finish_reasons[row] == "context_limit",
                    "budget_hit": finish_reasons[row] == "budget",
                    "cached_incremental": True,
                    "cached_batch_size": cached_batch_sizes[row],
                    "cached_active_compaction": True,
                }
                for row in range(len(problems))
            )
        return generations, answer_predictions
    sequences = [
        tokenizer.encode_generation_prefix(problem, model.max_sequence_length)
        for problem in problems
    ]
    prefix_lengths = torch.tensor(
        [len(sequence) for sequence in sequences], dtype=torch.long, device=device
    )
    answer_predictions: list[int | None] | None = None
    finished = [False] * len(sequences)
    finish_reasons: list[str | None] = [None] * len(sequences)
    generated_token_counts = [0] * len(sequences)
    for _ in range(int(max_new_tokens)):
        input_ids, attention_mask = pad_1d(sequences, tokenizer.pad_token_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask, prefix_lengths)
        if answer_predictions is None and model.answer_head_enabled:
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
                finish_reasons[row] = "context_limit"
                continue
            sequences[row].append(int(token))
            generated_token_counts[row] += 1
            if int(token) == tokenizer.eos_token_id:
                finished[row] = True
                finish_reasons[row] = "eos"
        if all(finished):
            break
    for row, reason in enumerate(finish_reasons):
        if reason is None:
            finish_reasons[row] = "budget"
    generations = [
        tokenizer.decode(sequence[int(prefix_lengths[row].item()) :])
        for row, sequence in enumerate(sequences)
    ]
    if diagnostics is not None:
        diagnostics.extend(
            {
                "generated_tokens": generated_token_counts[row],
                "eos_terminated": finish_reasons[row] == "eos",
                "context_limit_hit": finish_reasons[row] == "context_limit",
                "budget_hit": finish_reasons[row] == "budget",
                "cached_incremental": False,
            }
            for row in range(len(sequences))
        )
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
    model = build_math_tower_for_checkpoint(config, checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    tokenizer = ByteMathTokenizer()
    settings = config["evaluation"]
    maximum = int(
        maximum_examples
        if maximum_examples is not None
        else settings["maximum_generation_examples"]
    )
    split_names = splits or list(
        settings.get(
            "splits",
            ["test", "heldout_language", "extrapolation", "compositional"],
        )
    )
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
            "answer_head_enabled": bool(model.answer_head_enabled),
            "answer_head_valid_rate": sum(answer_head_valid) / len(records),
            "answer_head_accuracy": sum(answer_head_correct) / len(records),
            "canonical_trace_exact_rate": sum(trace_exact) / len(records),
            "generation_rows": str(rows_path.resolve()),
        }
    acceptance = _acceptance_report(report, config)
    report["specialist_acceptance"] = acceptance
    # Backward-compatible alias for evidence bundles created before the
    # acceptance-check terminology was clarified.
    report["specialist_gate"] = acceptance
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
