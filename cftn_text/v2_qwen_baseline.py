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
from .gpt_receiver import pretrained_dtype_kwargs, validate_dense_causal_lm_config
from .math_validation import stratified_validation_panel
from .training import load_data_contract, resolve_device, split_dataset
from .v2_metrics import extract_v2_answer, score_v2_generations


QWEN_BASELINE_FORMAT = "cftn_text_v2_frozen_qwen_baseline_v1"
QWEN_BASELINE_PANEL = "difficulty_balanced_source_family_round_robin_v1"


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
) -> list[str]:
    outputs: list[str] = []
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
        outputs.extend(
            tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
        )
    return outputs


def evaluate_frozen_v2_qwen(
    config: dict[str, Any],
    *,
    split: str = "validation",
    examples_per_difficulty: int = 12,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    prompt_mode: QwenPromptMode = "brief_reasoning",
    device_name: str = "cuda",
    local_files_only: bool = True,
    output_root: str | Path,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(config)
    records = split_dataset(data_root, manifest, split).records
    selected = difficulty_balanced_panel(
        records,
        examples_per_difficulty=examples_per_difficulty,
    )
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    before = gpu_status()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, resolved_revision = _load_qwen(
        config,
        device=device,
        local_files_only=local_files_only,
    )
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
    generations = _generate(
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
        "panel_policy": QWEN_BASELINE_PANEL,
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
    "difficulty_balanced_panel",
    "evaluate_frozen_v2_qwen",
    "qwen_math_messages",
]
