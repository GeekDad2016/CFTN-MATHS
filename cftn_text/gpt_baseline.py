from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .checkpoint import atomic_json_dump, gpu_status
from .config import config_sha256
from .dataset import CFTNCollator
from .metrics import extract_answer
from .data_generator import file_sha256
from .training import load_data_contract, precision_dtype, resolve_device, split_dataset


INTEGER_PATTERN = re.compile(
    r"^\s*(?:<answer>\s*)?([+-]?\d+)", re.IGNORECASE
)

FEW_SHOT_DEMONSTRATIONS = (
    ("Solve 2*x + (3) = 11.", 4),
    (
        "An unknown number multiplied by -3, then increased by 5, equals 14. "
        "Find the unknown number.",
        -3,
    ),
    ("Find x if 18 = (-7) + 5*x.", 5),
)


def first_generated_integer(text: str) -> int | None:
    match = INTEGER_PATTERN.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def zero_shot_prompt(problem: str) -> str:
    return CFTNCollator.gpt_prompt(problem)


def few_shot_prompt(problem: str, demonstrations: int) -> str:
    if not 0 <= demonstrations <= len(FEW_SHOT_DEMONSTRATIONS):
        raise ValueError("few-shot demonstration count is unsupported")
    pieces = [
        "Solve each problem and return only the integer in <answer> tags.\n\n"
    ]
    for example, answer in FEW_SHOT_DEMONSTRATIONS[:demonstrations]:
        pieces.append(f"Problem: {example}\nAnswer:<answer>{answer}</answer>\n\n")
    pieces.append(f"Problem: {problem}\nAnswer:")
    return "".join(pieces)


def plausible_candidates(record: dict[str, Any], count: int) -> list[int]:
    if count < 2:
        raise ValueError("candidate count must be at least two")
    target = int(record["x"])
    proposals = [
        target,
        target - 1,
        target + 1,
        target - 2,
        target + 2,
        -target,
        0,
        int(record["a"]),
        int(record["b"]),
        int(record["c"]),
    ]
    delta = 3
    while len(set(proposals)) < count:
        proposals.extend((target - delta, target + delta))
        delta += 1
    unique: list[int] = []
    for value in proposals:
        if value not in unique:
            unique.append(value)
        if len(unique) == count:
            break
    # Deterministically rotate the list so the true answer is not always first.
    rotation = int(record["record_id"][:8], 16) % len(unique)
    return unique[rotation:] + unique[:rotation]


def _model_and_tokenizer(config: dict[str, Any], device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = config["gpt"]
    local_only = bool(settings.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_name"], local_files_only=local_only
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("GPT tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        settings["model_name"], local_files_only=local_only
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dtype = precision_dtype(config["gpt_calibration"]["precision"], device)
    if dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer, dtype


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    outputs: list[str] = []
    maximum_context = int(
        getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024))
    )
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        if encoded.input_ids.shape[1] + max_new_tokens > maximum_context:
            raise ValueError("calibration prompt exceeds the frozen GPT context window")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            use_cache=True,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        if progress is not None:
            progress(min(len(prompts), start + len(batch_prompts)), len(prompts))
    return outputs


def _right_pad(
    sequences: list[list[int]], labels: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    target_labels = torch.full((len(sequences), width), -100, dtype=torch.long)
    for row, (sequence, label) in enumerate(zip(sequences, labels)):
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention_mask[row, : len(sequence)] = 1
        target_labels[row, : len(label)] = torch.tensor(label)
    return input_ids, attention_mask, target_labels


@torch.inference_mode()
def rank_candidate_answers(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    *,
    candidates_per_problem: int,
    batch_size: int,
    device: torch.device,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        prompt_ids = tokenizer.encode(
            zero_shot_prompt(record["problem"]), add_special_tokens=False
        )
        for candidate in plausible_candidates(record, candidates_per_problem):
            completion = f"<answer>{candidate}</answer>"
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            completion_ids.append(int(tokenizer.eos_token_id))
            sequence = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            entries.append(
                {
                    "record_index": record_index,
                    "candidate": candidate,
                    "input_ids": sequence,
                    "labels": labels,
                }
            )
    scores: list[list[tuple[int, float]]] = [[] for _ in records]
    for start in range(0, len(entries), batch_size):
        batch_entries = entries[start : start + batch_size]
        input_ids, attention_mask, labels = _right_pad(
            [entry["input_ids"] for entry in batch_entries],
            [entry["labels"] for entry in batch_entries],
            int(tokenizer.pad_token_id),
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits
        shifted_logits = logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        valid = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~valid, 0)
        token_log_probabilities = F.log_softmax(shifted_logits, dim=-1).gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        mean_scores = (token_log_probabilities * valid).sum(dim=1) / valid.sum(
            dim=1
        ).clamp_min(1)
        for entry, score in zip(batch_entries, mean_scores.tolist()):
            scores[entry["record_index"]].append((entry["candidate"], float(score)))
        if progress is not None:
            completed = min(len(entries), start + len(batch_entries))
            progress(completed, len(entries))
    results: list[dict[str, Any]] = []
    for record, row_scores in zip(records, scores):
        ordered = sorted(row_scores, key=lambda item: item[1], reverse=True)
        target = int(record["x"])
        correct_score = next(score for candidate, score in row_scores if candidate == target)
        best_wrong = max(score for candidate, score in row_scores if candidate != target)
        results.append(
            {
                "prediction": ordered[0][0],
                "correct": ordered[0][0] == target,
                "correct_score": correct_score,
                "best_wrong_score": best_wrong,
                "correct_margin": correct_score - best_wrong,
                "scores": [
                    {"candidate": candidate, "mean_log_probability": score}
                    for candidate, score in row_scores
                ],
            }
        )
    return results


def _family(record: dict[str, Any]) -> str:
    return "symbolic" if record["template_id"].startswith("symbolic") else "verbal"


def _sign_pattern(record: dict[str, Any]) -> str:
    def sign(value: int) -> str:
        return "negative" if value < 0 else "zero" if value == 0 else "positive"

    return f"a_{sign(int(record['a']))}__b_{sign(int(record['b']))}__x_{sign(int(record['x']))}"


def _accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def _record_ids_sha256(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(record["record_id"] for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_phase_cache(
    path: Path,
    *,
    phase: str,
    config: dict[str, Any],
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> Any | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "format": "cftn_text_gpt_calibration_phase_v1",
        "phase": phase,
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "record_ids_sha256": _record_ids_sha256(records),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"stale or misaligned GPT calibration cache: {path}")
    values = payload.get("values")
    if not isinstance(values, list) or len(values) != len(records):
        raise ValueError(f"GPT calibration cache has the wrong row count: {path}")
    return values


def _save_phase_cache(
    path: Path,
    *,
    phase: str,
    values: list[Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    atomic_json_dump(
        {
            "format": "cftn_text_gpt_calibration_phase_v1",
            "phase": phase,
            "config_sha256": config_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"],
            "record_ids_sha256": _record_ids_sha256(records),
            "values": values,
        },
        path,
    )


def _group_metrics(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {
        group: {
            "examples": len(items),
            "zero_shot_strict_accuracy": _accuracy(items, "zero_shot_strict_correct"),
            "zero_shot_lenient_accuracy": _accuracy(items, "zero_shot_lenient_correct"),
            "few_shot_strict_accuracy": _accuracy(items, "few_shot_strict_correct"),
            "few_shot_lenient_accuracy": _accuracy(items, "few_shot_lenient_correct"),
            "candidate_ranking_accuracy": _accuracy(items, "candidate_correct"),
        }
        for group, items in sorted(groups.items())
    }


def evaluate_frozen_gpt(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    maximum_examples: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(config)
    records = split_dataset(data_root, manifest, "calibration").records
    settings = config["gpt_calibration"]
    maximum = int(
        maximum_examples
        if maximum_examples is not None
        else settings["maximum_examples"]
    )
    records = records[:maximum]
    if not records:
        raise RuntimeError("calibration split is empty")
    artifact_root = Path(
        output_root
        or Path(config["project"]["artifact_root"]) / "gpt_calibration"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    status_path = artifact_root / "status.json"
    started = time.time()

    def status(phase: str, completed: int, total: int) -> None:
        atomic_json_dump(
            {
                "state": "running",
                "phase": phase,
                "completed": completed,
                "total": total,
                "elapsed_seconds": time.time() - started,
                "gpu": gpu_status(),
            },
            status_path,
        )

    model, tokenizer, dtype = _model_and_tokenizer(config, device)
    zero_cache = artifact_root / "zero_shot_cache.json"
    zero_outputs = _load_phase_cache(
        zero_cache,
        phase="zero_shot_generation",
        config=config,
        manifest=manifest,
        records=records,
    )
    if zero_outputs is None:
        zero_outputs = greedy_generate(
            model,
            tokenizer,
            [zero_shot_prompt(record["problem"]) for record in records],
            batch_size=int(settings["batch_size"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            device=device,
            progress=lambda done, total: status("zero_shot_generation", done, total),
        )
        _save_phase_cache(
            zero_cache,
            phase="zero_shot_generation",
            values=zero_outputs,
            config=config,
            manifest=manifest,
            records=records,
        )
    few_cache = artifact_root / "few_shot_cache.json"
    few_outputs = _load_phase_cache(
        few_cache,
        phase="few_shot_generation",
        config=config,
        manifest=manifest,
        records=records,
    )
    if few_outputs is None:
        few_outputs = greedy_generate(
            model,
            tokenizer,
            [
                few_shot_prompt(record["problem"], int(settings["few_shot_examples"]))
                for record in records
            ],
            batch_size=int(settings["batch_size"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            device=device,
            progress=lambda done, total: status("few_shot_generation", done, total),
        )
        _save_phase_cache(
            few_cache,
            phase="few_shot_generation",
            values=few_outputs,
            config=config,
            manifest=manifest,
            records=records,
        )
    ranking_cache = artifact_root / "candidate_ranking_cache.json"
    rankings = _load_phase_cache(
        ranking_cache,
        phase="candidate_ranking",
        config=config,
        manifest=manifest,
        records=records,
    )
    if rankings is None:
        rankings = rank_candidate_answers(
            model,
            tokenizer,
            records,
            candidates_per_problem=int(settings["plausible_candidates"]),
            batch_size=int(settings["candidate_batch_size"]),
            device=device,
            progress=lambda done, total: status("candidate_ranking", done, total),
        )
        _save_phase_cache(
            ranking_cache,
            phase="candidate_ranking",
            values=rankings,
            config=config,
            manifest=manifest,
            records=records,
        )
    rows: list[dict[str, Any]] = []
    for record, zero_output, few_output, ranking in zip(
        records, zero_outputs, few_outputs, rankings
    ):
        target = int(record["x"])
        zero_strict = extract_answer(zero_output)
        zero_lenient = first_generated_integer(zero_output)
        few_strict = extract_answer(few_output)
        few_lenient = first_generated_integer(few_output)
        solved_directly = zero_lenient == target
        solved_with_few_shot = few_lenient == target
        solved_by_ranking = bool(ranking["correct"])
        difficulty = (
            "easy"
            if solved_directly
            else "medium"
            if solved_with_few_shot or solved_by_ranking
            else "hard"
        )
        rows.append(
            {
                "record_id": record["record_id"],
                "problem": record["problem"],
                "target": target,
                "template_id": record["template_id"],
                "family": _family(record),
                "sign_pattern": _sign_pattern(record),
                "zero_shot_output": zero_output,
                "zero_shot_strict_prediction": zero_strict,
                "zero_shot_lenient_prediction": zero_lenient,
                "zero_shot_strict_correct": zero_strict == target,
                "zero_shot_lenient_correct": zero_lenient == target,
                "zero_shot_prompt_copy": record["problem"] in zero_output,
                "few_shot_output": few_output,
                "few_shot_strict_prediction": few_strict,
                "few_shot_lenient_prediction": few_lenient,
                "few_shot_strict_correct": few_strict == target,
                "few_shot_lenient_correct": few_lenient == target,
                "candidate_prediction": ranking["prediction"],
                "candidate_correct": ranking["correct"],
                "candidate_correct_margin": ranking["correct_margin"],
                "candidate_scores": ranking["scores"],
                "difficulty": difficulty,
            }
        )
    rows_path = artifact_root / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    aggregate = {
        "examples": len(rows),
        "zero_shot_strict_accuracy": _accuracy(rows, "zero_shot_strict_correct"),
        "zero_shot_lenient_accuracy": _accuracy(rows, "zero_shot_lenient_correct"),
        "few_shot_strict_accuracy": _accuracy(rows, "few_shot_strict_correct"),
        "few_shot_lenient_accuracy": _accuracy(rows, "few_shot_lenient_correct"),
        "candidate_ranking_accuracy": _accuracy(rows, "candidate_correct"),
        "candidate_random_chance_accuracy": 1.0
        / int(settings["plausible_candidates"]),
        "zero_shot_prompt_copy_rate": sum(
            row["zero_shot_prompt_copy"] for row in rows
        )
        / len(rows),
    }
    aggregate["candidate_accuracy_minus_random_chance"] = (
        aggregate["candidate_ranking_accuracy"]
        - aggregate["candidate_random_chance_accuracy"]
    )
    best_capability = max(
        aggregate["zero_shot_lenient_accuracy"],
        aggregate["few_shot_lenient_accuracy"],
        aggregate["candidate_ranking_accuracy"],
    )
    headroom_points = 100.0 * (1.0 - best_capability)
    decision = {
        "best_frozen_gpt_capability_accuracy": best_capability,
        "estimated_headroom_percentage_points": headroom_points,
        "maximum_acceptable_gpt_accuracy": float(
            settings["maximum_acceptable_gpt_accuracy"]
        ),
        "minimum_headroom_percentage_points": float(
            settings["minimum_headroom_percentage_points"]
        ),
    }
    decision["proceed_to_math_training"] = (
        best_capability <= decision["maximum_acceptable_gpt_accuracy"]
        and headroom_points >= decision["minimum_headroom_percentage_points"]
    )
    report = {
        "format": "cftn_text_frozen_gpt_calibration_v1",
        "model_name": config["gpt"]["model_name"],
        "model_commit": getattr(model.config, "_commit_hash", None),
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "all_gpt_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "precision": str(dtype or torch.float32),
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest["manifest_sha256"],
        "calibration_split_sha256": manifest["splits"]["calibration"]["sha256"],
        "rows": str(rows_path.resolve()),
        "aggregate": aggregate,
        "by_family": _group_metrics(rows, "family"),
        "by_template": _group_metrics(rows, "template_id"),
        "by_sign_pattern": _group_metrics(rows, "sign_pattern"),
        "difficulty": {
            difficulty: sum(row["difficulty"] == difficulty for row in rows)
            for difficulty in ("easy", "medium", "hard")
        },
        "decision": decision,
        "elapsed_seconds": time.time() - started,
        "gpu": gpu_status(),
    }
    atomic_json_dump(report, artifact_root / "report.json")
    atomic_json_dump(
        {
            "state": "completed",
            "phase": "complete",
            "completed": len(rows),
            "total": len(rows),
            "elapsed_seconds": time.time() - started,
            "decision": decision,
            "gpu": gpu_status(),
        },
        status_path,
    )
    return report


def verify_calibration_gate(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    report_path = (
        Path(config["project"]["artifact_root"])
        / "gpt_calibration"
        / "report.json"
    )
    if not report_path.is_file():
        raise FileNotFoundError(
            "frozen-GPT calibration is required before math training: "
            f"{report_path}"
        )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("format") != "cftn_text_frozen_gpt_calibration_v1":
        raise ValueError("unsupported frozen-GPT calibration report")
    if report.get("evaluator_sha256") != file_sha256(Path(__file__).resolve()):
        raise ValueError("frozen-GPT calibration evaluator source changed")
    if report.get("config_sha256") != config_sha256(config):
        raise ValueError("frozen-GPT calibration used a different configuration")
    if report.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("frozen-GPT calibration used a different data manifest")
    if report.get("calibration_split_sha256") != manifest["splits"]["calibration"][
        "sha256"
    ]:
        raise ValueError("frozen-GPT calibration split hash differs")
    if not report.get("decision", {}).get("proceed_to_math_training", False):
        raise RuntimeError(
            "frozen GPT already solves too much of this benchmark; revise task difficulty"
        )
    return report
