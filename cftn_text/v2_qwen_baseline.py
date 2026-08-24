from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

from .checkpoint import atomic_json_dump, gpu_status
from .config import canonical_json, config_sha256
from .data_generator import file_sha256
from .gpt_receiver import pretrained_dtype_kwargs, validate_dense_causal_lm_config
from .math_validation import stratified_validation_panel
from .training import load_data_contract, resolve_device, split_dataset
from .v2_metrics import extract_v2_answer, score_v2_generations


QWEN_BASELINE_FORMAT = "cftn_text_v2_frozen_qwen_baseline_v1"
QWEN_BASELINE_PANEL = "difficulty_balanced_source_family_round_robin_v1"
QWEN_GAP_PANEL_FORMAT = "cftn_text_v2_qwen_math_gap_panel_v1"


QwenPromptMode = Literal["brief_reasoning", "answer_only"]


def qwen_math_messages(
    record: dict[str, Any], *, prompt_mode: QwenPromptMode = "brief_reasoning"
) -> list[dict[str, str]]:
    problem = str(record["problem"])
    if prompt_mode == "brief_reasoning":
        instruction = (
            "Use concise reasoning, then finish with exactly one "
            "<answer>...</answer> tag containing only the final exact result."
        )
    elif prompt_mode == "answer_only":
        instruction = (
            "Return exactly one <answer>...</answer> tag containing only the "
            "final exact result. Do not include an explanation or any other text."
        )
    else:
        raise ValueError(f"unsupported Qwen baseline prompt mode: {prompt_mode}")
    return [
        {
            "role": "system",
            "content": (
                "You are a precise mathematical problem solver. Solve the problem "
                "internally and do not use external tools."
            ),
        },
        {
            "role": "user",
            "content": f"Problem: {problem}\n{instruction}",
        },
    ]


def difficulty_balanced_panel(
    records: Iterable[dict[str, Any]],
    *,
    examples_per_difficulty: int,
    difficulties: Iterable[int] = (1, 2, 3),
) -> list[dict[str, Any]]:
    maximum = int(examples_per_difficulty)
    if maximum < 1:
        raise ValueError("examples_per_difficulty must be positive")
    values = list(records)
    selected: list[dict[str, Any]] = []
    for difficulty in [int(value) for value in difficulties]:
        cohort = [
            record for record in values if int(record.get("difficulty", 0)) == difficulty
        ]
        rows = stratified_validation_panel(cohort, maximum)
        if len(rows) != maximum:
            raise RuntimeError(
                f"difficulty {difficulty} exposes {len(rows)} rows; {maximum} required"
            )
        selected.extend(rows)
    return selected


def load_qwen_gap_panel(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != QWEN_GAP_PANEL_FORMAT:
        raise ValueError("unsupported Qwen math gap panel format")
    record_ids = [str(value) for value in payload.get("record_ids", [])]
    if not record_ids or len(record_ids) != len(set(record_ids)):
        raise ValueError("Qwen math gap panel record IDs must be unique and nonempty")
    observed_hash = hashlib.sha256(
        canonical_json(record_ids).encode("utf-8")
    ).hexdigest()
    if observed_hash != payload.get("record_ids_sha256"):
        raise ValueError("Qwen math gap panel record ID hash mismatch")
    challenge_ids = [
        str(item["record_id"]) for item in payload.get("challenge_cases", [])
    ]
    if not challenge_ids or len(challenge_ids) != len(set(challenge_ids)):
        raise ValueError("Qwen math gap challenge IDs must be unique and nonempty")
    if not set(challenge_ids).issubset(record_ids):
        raise ValueError("Qwen math gap challenge IDs must belong to the full panel")
    payload["path"] = str(source)
    payload["sha256"] = file_sha256(source)
    return payload


def fixed_record_panel(
    records: Iterable[dict[str, Any]], record_ids: Iterable[str]
) -> list[dict[str, Any]]:
    requested = [str(value) for value in record_ids]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("fixed panel record IDs must be unique and nonempty")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("record_id", record.get("content_id")))
        if record_id in indexed:
            raise ValueError(f"dataset contains duplicate record ID {record_id}")
        indexed[record_id] = record
    missing = [record_id for record_id in requested if record_id not in indexed]
    if missing:
        raise RuntimeError(f"fixed panel is missing dataset record IDs: {missing}")
    return [indexed[record_id] for record_id in requested]


def _load_qwen(
    config: dict[str, Any],
    *,
    device: torch.device,
    local_files_only: bool,
) -> tuple[Any, Any, str | None]:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        __version__ as transformers_version,
    )

    settings = config["gpt"]
    common: dict[str, Any] = {
        "revision": str(settings["revision"]),
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(settings.get("trust_remote_code", False)),
    }
    model_config = AutoConfig.from_pretrained(settings["model_name"], **common)
    validate_dense_causal_lm_config(
        model_config,
        expected_model_type=settings.get("expected_model_type"),
        expected_hidden_size=settings.get("expected_hidden_size"),
        expected_layers=settings.get("expected_layers"),
        require_dense=bool(settings.get("require_dense", True)),
    )
    requested_revision = str(settings["revision"])
    resolved_revision = str(getattr(model_config, "_commit_hash", None) or "")
    if resolved_revision and resolved_revision != requested_revision:
        raise RuntimeError(
            "resolved Qwen revision differs from the configured immutable revision"
        )
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"], **common)
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RuntimeError("configured Qwen tokenizer exposes no chat template")
    if tokenizer.eos_token_id is None:
        raise RuntimeError("configured Qwen tokenizer exposes no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs = dict(common)
    model_kwargs.update(
        pretrained_dtype_kwargs(settings.get("dtype"), transformers_version)
    )
    if settings.get("attn_implementation"):
        model_kwargs["attn_implementation"] = str(settings["attn_implementation"])
    model = AutoModelForCausalLM.from_pretrained(settings["model_name"], **model_kwargs)
    validate_dense_causal_lm_config(
        model.config,
        expected_model_type=settings.get("expected_model_type"),
        expected_hidden_size=settings.get("expected_hidden_size"),
        expected_layers=settings.get("expected_layers"),
        require_dense=bool(settings.get("require_dense", True)),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()
    return model, tokenizer, resolved_revision or None


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[str], dict[str, float | int]]:
    outputs: list[str] = []
    prompt_tokens = 0
    generated_tokens = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    maximum_context = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    for start in range(0, len(prompts), max(1, int(batch_size))):
        chunk = prompts[start : start + max(1, int(batch_size))]
        encoded = tokenizer(
            chunk,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        prompt_width = int(encoded.input_ids.shape[1])
        prompt_tokens += int(encoded.attention_mask.sum().item())
        if maximum_context and prompt_width + int(max_new_tokens) > maximum_context:
            raise ValueError("Qwen baseline prompt exceeds the model context window")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            use_cache=True,
        )
        decoded = tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        outputs.extend(decoded)
        generated_tokens += sum(
            len(tokenizer.encode(text, add_special_tokens=False)) for text in decoded
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    return outputs, {
        "generation_seconds": generation_seconds,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens
        / max(generation_seconds, 1e-9),
        "milliseconds_per_example": generation_seconds
        * 1000.0
        / max(len(prompts), 1),
    }


def evaluate_frozen_v2_qwen(
    config: dict[str, Any],
    *,
    split: str = "validation",
    examples_per_difficulty: int = 12,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    prompt_mode: QwenPromptMode = "brief_reasoning",
    panel_manifest: str | Path | None = None,
    panel_subset: Literal["full", "challenge"] = "full",
    gpu_hourly_usd: float | None = None,
    device_name: str = "cuda",
    local_files_only: bool = True,
    output_root: str | Path,
) -> dict[str, Any]:
    if gpu_hourly_usd is not None and float(gpu_hourly_usd) <= 0:
        raise ValueError("gpu_hourly_usd must be positive when provided")
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(config)
    records = split_dataset(data_root, manifest, split).records
    gap_panel: dict[str, Any] | None = None
    if panel_manifest is not None:
        gap_panel = load_qwen_gap_panel(panel_manifest)
        dataset_contract = gap_panel["dataset"]
        if str(dataset_contract["split"]) != str(split):
            raise ValueError("Qwen math gap panel split differs from requested split")
        if dataset_contract["manifest_sha256"] != manifest.get("manifest_sha256"):
            raise ValueError("Qwen math gap panel manifest hash mismatch")
        if dataset_contract["split_sha256"] != manifest["splits"][split]["sha256"]:
            raise ValueError("Qwen math gap panel split hash mismatch")
        if panel_subset == "challenge":
            requested_ids = [
                str(item["record_id"]) for item in gap_panel["challenge_cases"]
            ]
        elif panel_subset == "full":
            requested_ids = [str(value) for value in gap_panel["record_ids"]]
        else:
            raise ValueError(f"unsupported Qwen math gap panel subset: {panel_subset}")
        selected = fixed_record_panel(records, requested_ids)
        panel_policy = f"fixed_qwen_math_gap_{panel_subset}_v1"
    else:
        selected = difficulty_balanced_panel(
            records,
            examples_per_difficulty=examples_per_difficulty,
        )
        panel_policy = QWEN_BASELINE_PANEL
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    before = gpu_status()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model, tokenizer, resolved_revision = _load_qwen(
        config,
        device=device,
        local_files_only=local_files_only,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    prompts = [
        str(
            tokenizer.apply_chat_template(
                qwen_math_messages(record, prompt_mode=prompt_mode),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        for record in selected
    ]
    generations, generation_telemetry = _generate(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    metrics, correctness = score_v2_generations(generations, selected)
    rows_path = output / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for record, prompt, generation, correct in zip(
            selected, prompts, generations, correctness
        ):
            handle.write(
                json.dumps(
                    {
                        "record_id": record.get("record_id", record.get("content_id")),
                        "source": record.get("source"),
                        "family": record.get("family"),
                        "difficulty": record.get("difficulty"),
                        "problem": record.get("problem"),
                        "expected_answer": record.get("normalized_answer"),
                        "prompt": prompt,
                        "generation": generation,
                        "parsed_answer": extract_v2_answer(generation),
                        "correct": bool(correct),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    elapsed = time.time() - started
    record_ids = [
        str(record.get("record_id", record.get("content_id"))) for record in selected
    ]
    report = {
        "format": QWEN_BASELINE_FORMAT,
        "model_name": str(config["gpt"]["model_name"]),
        "requested_revision": str(config["gpt"]["revision"]),
        "resolved_revision": resolved_revision,
        "architecture": str(config["gpt"].get("architecture")),
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "precision": str(next(model.parameters()).dtype),
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "split": str(split),
        "split_sha256": manifest["splits"][split]["sha256"],
        "panel_policy": panel_policy,
        "panel_manifest": (
            {
                "path": gap_panel["path"],
                "sha256": gap_panel["sha256"],
                "subset": panel_subset,
            }
            if gap_panel is not None
            else None
        ),
        "record_ids_sha256": hashlib.sha256(
            canonical_json(record_ids).encode("utf-8")
        ).hexdigest(),
        "examples_per_difficulty": int(examples_per_difficulty),
        "sample_counts": dict(
            sorted(Counter(str(record["difficulty"]) for record in selected).items())
        ),
        "generation": {
            "decoding": "greedy",
            "batch_size": int(batch_size),
            "max_new_tokens": int(max_new_tokens),
            "chat_template": True,
            "prompt_contract": f"raw_problem_{prompt_mode}_v1",
            "problem_field": "problem",
        },
        "metrics": metrics,
        "telemetry": {
            "model_load_seconds": model_load_seconds,
            **generation_telemetry,
            "gpu_hourly_usd": gpu_hourly_usd,
            "estimated_generation_cost_usd": (
                float(generation_telemetry["generation_seconds"])
                * float(gpu_hourly_usd)
                / 3600.0
                if gpu_hourly_usd is not None
                else None
            ),
            "estimated_cold_run_cost_usd": (
                (model_load_seconds + float(generation_telemetry["generation_seconds"]))
                * float(gpu_hourly_usd)
                / 3600.0
                if gpu_hourly_usd is not None
                else None
            ),
        },
        "rows": str(rows_path),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(selected) / max(elapsed, 1e-9),
        "gpu_before": before,
        "gpu_after": gpu_status(),
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
    }
    atomic_json_dump(report, output / "report.json")
    return report


__all__ = [
    "QWEN_BASELINE_FORMAT",
    "QWEN_BASELINE_PANEL",
    "QWEN_GAP_PANEL_FORMAT",
    "difficulty_balanced_panel",
    "evaluate_frozen_v2_qwen",
    "fixed_record_panel",
    "load_qwen_gap_panel",
    "qwen_math_messages",
]
