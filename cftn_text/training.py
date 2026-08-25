from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .checkpoint import (
    atomic_copy_file,
    append_jsonl,
    atomic_json_dump,
    atomic_torch_save,
    build_checkpoint,
    gpu_status,
    latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    rotate_latest,
)
from .config import config_sha256
from .complementary import apply_view_mode
from .data_generator import audit_manifest, file_sha256, prepare_manifests
from .dataset import (
    CFTNCollator,
    EquationDataset,
    MathCollator,
    SHARED_MATH_INPUT_VIEW,
)
from .gpt_receiver import FrozenCausalLMTower
from .math_tower import MathTower
from .math_validation import (
    DEFAULT_V2_GENERATION_VALIDATION,
    evaluate_generation_panel,
    summarize_teacher_forced_breakdowns,
    update_teacher_forced_breakdowns,
)
from .metrics import masked_token_statistics, summarize_gate
from .model import (
    CFTNTextModel,
    answer_weighted_causal_language_loss,
    causal_language_loss,
    optional_answer_loss,
)
from .tokenizer import ByteMathTokenizer
from .wandb_support import initialize_wandb


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_math_initialization_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None,
    map_location: str | torch.device,
) -> tuple[dict[str, Any], str]:
    """Load sealed model weights for a new run, not resumable training state.

    A recovery run intentionally has a new data manifest, optimizer, scheduler,
    and RNG stream.  The source file is authenticated by SHA-256 and its stage
    is checked here; architecture compatibility is checked by the caller's
    strict ``load_state_dict``.  Manifest/config equality remains mandatory for
    true resume paths.
    """

    source_path = Path(path).expanduser().resolve()
    observed_sha256 = file_sha256(source_path)
    if expected_sha256 and observed_sha256 != expected_sha256:
        raise RuntimeError("math recovery source checkpoint hash changed")
    checkpoint = load_checkpoint(
        source_path,
        expected_stage="math",
        map_location=map_location,
    )
    return checkpoint, observed_sha256


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def precision_dtype(name: str, device: torch.device) -> torch.dtype | None:
    normalized = str(name).lower()
    if device.type != "cuda" or normalized in {"float32", "fp32", "none"}:
        return None
    if normalized in {"bfloat16", "bf16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this GPU does not support it")
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unsupported precision: {name}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scaler(device: torch.device, dtype: torch.dtype | None):
    enabled = device.type == "cuda" and dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_data_contract(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(config["project"]["data_root"]).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        data_format = config["data"].get("format")
        if data_format == "cftn_text_broad_math_v2":
            from .v2_data import prepare_v2_manifests

            prepare_v2_manifests(config, root)
        elif data_format == "cftn_text_linear_equations_v1_1":
            from .algorithmic_data_generator import prepare_algorithmic_manifests

            prepare_algorithmic_manifests(config, root)
        else:
            prepare_manifests(config, root)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("config_sha256") != config_sha256(config):
        raise ValueError("data manifest was generated from a different configuration")
    if manifest.get("format") == "cftn_text_broad_math_v2":
        from .v2_data import audit_v2_manifest

        audit_v2_manifest(manifest, root)
    elif manifest.get("format") == "cftn_text_linear_equations_v1_1":
        from .algorithmic_data_generator import audit_algorithmic_manifest

        audit_algorithmic_manifest(manifest, root)
    else:
        audit_manifest(manifest, root)
    return root, manifest


def split_dataset(root: Path, manifest: dict[str, Any], split: str) -> EquationDataset:
    metadata = manifest["splits"][split]
    return EquationDataset(root / metadata["path"])


def make_loader(
    dataset: EquationDataset,
    collator,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch) * 1_000_003)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
        generator=generator,
    )


def math_epoch_dataset(
    dataset: EquationDataset,
    config: dict[str, Any],
    *,
    epoch: int,
    seed: int,
    phase_override: dict[str, Any] | None = None,
) -> tuple[EquationDataset, dict[str, Any]]:
    data_format = config["data"].get("format")
    phase: dict[str, Any] | None = None
    if data_format == "cftn_text_broad_math_v2":
        curriculum = config["data"].get("curriculum", {})
        if not curriculum.get("enabled", False):
            selected = dataset.records
            metadata = {"enabled": False, "phase": "all", "max_difficulty": 3}
        else:
            phases = list(curriculum.get("phases", []))
            if not phases and phase_override is None:
                raise ValueError("V2 curriculum requires at least one phase")
            phase = phase_override or next(
                (
                    item
                    for item in phases
                    if epoch <= int(item["through_epoch"])
                ),
                phases[-1],
            )
            selected = _filter_v2_records_for_phase(dataset.records, phase)
            if not selected:
                raise RuntimeError(
                    f"curriculum phase {phase.get('name', 'unknown')} selected no records"
                )
            metadata = _v2_curriculum_metadata(dataset.records, selected, phase)
    elif data_format == "cftn_text_linear_equations_v1_1":
        from .algorithmic_data_generator import curriculum_records
        selected, metadata = curriculum_records(dataset.records, config, epoch)
    else:
        return dataset, {"enabled": False, "phase": "all"}
    target = int(
        config["data"].get("curriculum", {}).get(
            "examples_per_epoch", len(dataset)
        )
    )
    source_quotas = (
        phase.get("source_quotas")
        if data_format == "cftn_text_broad_math_v2" and phase is not None
        else None
    )
    if source_quotas is not None:
        if not isinstance(source_quotas, dict) or not source_quotas:
            raise ValueError("curriculum phase source_quotas must be a non-empty object")
        normalized_quotas = {
            str(source): int(examples) for source, examples in source_quotas.items()
        }
        if any(examples <= 0 for examples in normalized_quotas.values()):
            raise ValueError("curriculum phase source quotas must be positive")
        if sum(normalized_quotas.values()) != target:
            raise ValueError(
                "curriculum source quotas must sum to examples_per_epoch"
            )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in selected:
            grouped.setdefault(str(record.get("source", "unknown")), []).append(record)
        missing_sources = sorted(set(normalized_quotas) - set(grouped))
        if missing_sources:
            raise RuntimeError(
                "curriculum source quotas selected unavailable sources: "
                + ", ".join(missing_sources)
            )
        rng = random.Random(int(seed) + int(epoch) * 1_000_003)
        epoch_records: list[dict[str, Any]] = []
        source_sampling: dict[str, dict[str, Any]] = {}
        for source in sorted(normalized_quotas):
            available = grouped[source]
            requested = normalized_quotas[source]
            if requested <= len(available):
                sampled = rng.sample(available, requested)
                replacement_examples = 0
            else:
                # Cover every available row before drawing deterministic replay.
                sampled = rng.sample(available, len(available))
                replacement_examples = requested - len(available)
                sampled.extend(
                    available[rng.randrange(len(available))]
                    for _ in range(replacement_examples)
                )
            epoch_records.extend(sampled)
            source_sampling[source] = {
                "available_examples": len(available),
                "requested_examples": requested,
                "unique_examples": len({str(row.get("record_id")) for row in sampled}),
                "replacement_examples": replacement_examples,
                "sampling_with_replacement": replacement_examples > 0,
            }
        rng.shuffle(epoch_records)
        metadata = {
            **metadata,
            "examples_this_epoch": len(epoch_records),
            "unique_examples_this_epoch": len(
                {str(record.get("record_id")) for record in epoch_records}
            ),
            "sampling_policy": "source_quotas_v1",
            "sampling_with_replacement": any(
                details["sampling_with_replacement"]
                for details in source_sampling.values()
            ),
            "source_quotas": normalized_quotas,
            "source_sampling": source_sampling,
        }
        return EquationDataset(epoch_records), metadata
    sampling_policy = str(
        config["data"].get("curriculum", {}).get("sampling", "auto")
    )
    if sampling_policy not in {"auto", "without_replacement", "with_replacement"}:
        raise ValueError(f"unsupported curriculum sampling policy: {sampling_policy}")
    if sampling_policy == "without_replacement" and target > len(selected):
        raise RuntimeError(
            f"curriculum requested {target} examples without replacement, but "
            f"phase {metadata.get('phase', 'unknown')} exposes only {len(selected)}"
        )
    rng = random.Random(int(seed) + int(epoch) * 1_000_003)
    if sampling_policy == "with_replacement":
        epoch_records = [selected[rng.randrange(len(selected))] for _ in range(target)]
        sampling_with_replacement = True
    elif len(selected) >= target:
        epoch_records = rng.sample(selected, target)
        sampling_with_replacement = False
    else:
        epoch_records = [selected[rng.randrange(len(selected))] for _ in range(target)]
        sampling_with_replacement = True
    metadata = {
        **metadata,
        "examples_this_epoch": len(epoch_records),
        "sampling_policy": sampling_policy,
        "sampling_with_replacement": sampling_with_replacement,
    }
    return EquationDataset(epoch_records), metadata


def _filter_v2_records_for_phase(
    records: list[dict[str, Any]], phase: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply runtime-only semantic filters without changing the sealed generator."""

    configured_sources = phase.get("sources")
    configured_families = phase.get("families")
    sources = (
        {str(value) for value in configured_sources}
        if configured_sources is not None
        else None
    )
    families = (
        {str(value) for value in configured_families}
        if configured_families is not None
        else None
    )
    if sources is not None and not sources:
        raise ValueError("curriculum phase sources cannot be empty")
    if families is not None and not families:
        raise ValueError("curriculum phase families cannot be empty")
    maximum_difficulty = int(phase.get("max_difficulty", 3))
    return [
        record
        for record in records
        if int(record.get("difficulty", 3)) <= maximum_difficulty
        and (sources is None or str(record.get("source", "unknown")) in sources)
        and (families is None or str(record.get("family", "unknown")) in families)
    ]


def _v2_curriculum_metadata(
    all_records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    phase: dict[str, Any],
) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        return dict(
            sorted(Counter(str(record.get(key, "unknown")) for record in selected).items())
        )

    return {
        "enabled": True,
        "phase": str(phase["name"]),
        "max_difficulty": int(phase.get("max_difficulty", 3)),
        "available_examples": len(selected),
        "total_examples": len(all_records),
        "difficulty_counts": counts("difficulty"),
        "source_counts": counts("source"),
        "family_counts": counts("family"),
        "filters": {
            "sources": (
                sorted(str(value) for value in phase["sources"])
                if phase.get("sources") is not None
                else None
            ),
            "families": (
                sorted(str(value) for value in phase["families"])
                if phase.get("families") is not None
                else None
            ),
        },
    }


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * float(warmup_fraction)))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(minimum_ratio, (step + 1) / warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / denominator)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def build_math_tower(config: dict[str, Any]) -> MathTower:
    return MathTower(config["math_tower"], ByteMathTokenizer.vocab_size)


def load_gpt_components(config: dict[str, Any]) -> tuple[Any, FrozenCausalLMTower]:
    from transformers import AutoTokenizer

    gpt = config["gpt"]
    local_only = bool(gpt.get("local_files_only", True))
    tokenizer_kwargs: dict[str, Any] = {
        "local_files_only": local_only,
        "trust_remote_code": bool(gpt.get("trust_remote_code", False)),
    }
    if gpt.get("revision"):
        tokenizer_kwargs["revision"] = str(gpt["revision"])
    tokenizer = AutoTokenizer.from_pretrained(gpt["model_name"], **tokenizer_kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("GPT tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    setattr(
        tokenizer,
        "_cftn_use_chat_template",
        bool(gpt.get("use_chat_template", False)),
    )
    tower = FrozenCausalLMTower.from_pretrained(
        gpt["model_name"],
        [int(layer) for layer in gpt["receiver_layers"]],
        config["bridge"],
        local_files_only=local_only,
        revision=gpt.get("revision"),
        dtype=gpt.get("dtype"),
        trust_remote_code=bool(gpt.get("trust_remote_code", False)),
        attn_implementation=gpt.get("attn_implementation"),
        expected_model_type=gpt.get("expected_model_type"),
        expected_hidden_size=gpt.get("expected_hidden_size"),
        expected_layers=gpt.get("expected_layers"),
        require_dense=bool(gpt.get("require_dense", False)),
    )
    return tokenizer, tower


def build_cftn_model(
    config: dict[str, Any],
    math_checkpoint_path: str | Path,
    manifest: dict[str, Any],
    device: torch.device,
) -> tuple[CFTNTextModel, Any]:
    checkpoint = load_checkpoint(
        math_checkpoint_path,
        expected_stage="math",
        expected_config_sha256=config_sha256(config),
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    math_tower = build_math_tower(config)
    math_tower.load_state_dict(checkpoint["model_state"], strict=True)
    tokenizer, gpt_tower = load_gpt_components(config)
    model = CFTNTextModel(math_tower, gpt_tower, config).to(device)
    return model, tokenizer


def _status_payload(
    *,
    stage: str,
    state: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any] | None = None,
    started_at: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "state": state,
        "pid": os.getpid(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "elapsed_seconds": time.time() - started_at,
        "metrics": metrics or {},
        "gpu": gpu_status(),
        "updated_unix": time.time(),
    }


def _should_stop_early(
    *,
    epoch: int,
    patience: int,
    settings: dict[str, Any],
    enabled: bool,
) -> bool:
    return bool(
        enabled
        and epoch >= int(settings["minimum_epochs"])
        and patience >= int(settings["early_stop_patience"])
    )


def _bridge_stability_policy(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve conservative bridge defaults without changing the data contract.

    Bridge settings are part of the immutable experiment configuration hash.  The
    effective cap therefore lives in the trainer and is recorded in every metric
    row/checkpoint, allowing an already-running prerequisite stage to finish while
    later bridge subprocesses pick up the stability fix.
    """

    requested_learning_rate = float(settings["learning_rate"])
    maximum_learning_rate = float(
        settings.get("stability_maximum_learning_rate", 5e-5)
    )
    effective_learning_rate = min(requested_learning_rate, maximum_learning_rate)
    minimum_learning_rate = min(
        float(settings["minimum_learning_rate"]), effective_learning_rate
    )
    policy = {
        "enabled": bool(settings.get("stability_guard_enabled", True)),
        "requested_learning_rate": requested_learning_rate,
        "maximum_learning_rate": maximum_learning_rate,
        "effective_learning_rate": effective_learning_rate,
        "minimum_learning_rate": minimum_learning_rate,
        "gate_learning_rate_multiplier": float(
            settings.get("gate_learning_rate_multiplier", 0.5)
        ),
        "sequence_accuracy_drop": float(
            settings.get("collapse_sequence_accuracy_drop", 0.25)
        ),
        "loss_multiplier": float(settings.get("collapse_loss_multiplier", 10.0)),
        "absolute_loss_increase": float(
            settings.get("collapse_absolute_loss_increase", 1.0)
        ),
        "minimum_reference_shuffled_gap": float(
            settings.get("collapse_minimum_reference_shuffled_gap", 0.1)
        ),
        "maximum_shuffled_gap_retention": float(
            settings.get("collapse_maximum_shuffled_gap_retention", 0.1)
        ),
    }
    if policy["maximum_learning_rate"] <= 0 or policy["effective_learning_rate"] <= 0:
        raise ValueError("bridge stability learning rates must be positive")
    if not 0 < policy["gate_learning_rate_multiplier"] <= 1:
        raise ValueError("gate learning-rate multiplier must be within (0, 1]")
    if policy["minimum_learning_rate"] <= 0:
        raise ValueError("bridge minimum learning rate must be positive")
    if policy["loss_multiplier"] <= 1:
        raise ValueError("bridge collapse loss multiplier must exceed one")
    if not 0 <= policy["maximum_shuffled_gap_retention"] <= 1:
        raise ValueError("bridge shuffled-gap retention must be within [0, 1]")
    return policy


def _bridge_collapse_diagnostics(
    validation: dict[str, Any],
    best_validation: dict[str, Any] | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Detect the abrupt accuracy/loss/dependence failure seen in V1."""

    diagnostics: dict[str, Any] = {
        "triggered": False,
        "reasons": [],
    }
    if not policy["enabled"] or best_validation is None:
        return diagnostics

    sequence_keys = (
        "gpt_teacher_forced_sequence_accuracy",
        "math_teacher_forced_sequence_accuracy",
    )
    sequence_drops = {
        key: float(best_validation[key]) - float(validation[key])
        for key in sequence_keys
    }
    largest_sequence_drop = max(sequence_drops.values())
    best_loss = float(best_validation["loss"])
    current_loss = float(validation["loss"])
    required_loss = max(
        best_loss * float(policy["loss_multiplier"]),
        best_loss + float(policy["absolute_loss_increase"]),
    )
    best_gap = max(0.0, float(best_validation["shuffled_loss_gap"]))
    current_gap = max(0.0, float(validation["shuffled_loss_gap"]))
    gap_retention = current_gap / best_gap if best_gap > 0 else 1.0

    loss_collapse = (
        largest_sequence_drop >= float(policy["sequence_accuracy_drop"])
        and current_loss >= required_loss
    )
    dependence_collapse = (
        best_gap >= float(policy["minimum_reference_shuffled_gap"])
        and gap_retention <= float(policy["maximum_shuffled_gap_retention"])
        and largest_sequence_drop
        >= 0.5 * float(policy["sequence_accuracy_drop"])
    )
    reasons: list[str] = []
    if loss_collapse:
        reasons.append("sequence_accuracy_and_validation_loss")
    if dependence_collapse:
        reasons.append("bridge_message_dependence")
    diagnostics.update(
        {
            "triggered": bool(reasons),
            "reasons": reasons,
            "sequence_accuracy_drops": sequence_drops,
            "largest_sequence_accuracy_drop": largest_sequence_drop,
            "best_loss": best_loss,
            "current_loss": current_loss,
            "required_loss": required_loss,
            "best_shuffled_loss_gap": best_gap,
            "current_shuffled_loss_gap": current_gap,
            "shuffled_gap_retention": gap_retention,
        }
    )
    return diagnostics


def _training_progress_metrics(
    *,
    epoch: int,
    batch_completed: int,
    batches_total: int,
    global_step: int,
    total_steps: int,
    loss_sum: float,
    trained_examples: int,
    learning_rate: float,
    interval_started_at: float,
    interval_start_step: int,
) -> dict[str, Any]:
    elapsed = max(1e-9, time.time() - interval_started_at)
    completed_steps = max(0, global_step - interval_start_step)
    steps_per_second = completed_steps / elapsed
    remaining_steps = max(0, total_steps - global_step)
    return {
        "phase": "training",
        "epoch": int(epoch),
        "epoch_batch_completed": int(batch_completed),
        "epoch_batches_total": int(batches_total),
        "epoch_progress": batch_completed / max(1, batches_total),
        "train_loss_so_far": loss_sum / max(1, trained_examples),
        "learning_rate": float(learning_rate),
        "steps_per_second": steps_per_second,
        "examples_per_second": trained_examples / elapsed,
        "remaining_steps_to_max_epochs": remaining_steps,
        "eta_seconds_to_max_epochs_excluding_validation": (
            remaining_steps / steps_per_second if steps_per_second > 0 else None
        ),
    }


@torch.no_grad()
def evaluate_math_tower(
    model: MathTower,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    answer_head_weight: float,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    lm_loss_sum = 0.0
    examples = 0
    answer_correct = 0
    answer_valid = 0
    token_correct = 0
    token_total = 0
    sequence_correct = 0
    breakdown_sums: dict[str, dict[str, dict[str, float | int]]] = {}
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        with autocast_context(device, dtype):
            output = model(
                batch["math_input_ids"],
                batch["math_attention_mask"],
                batch["math_prefix_lengths"],
            )
            lm_loss = causal_language_loss(output.logits, batch["math_labels"])
            classes = model.answer_classes(batch["answer_values"])
            answer_loss = optional_answer_loss(output.answer_logits, classes)
            loss = lm_loss + float(answer_head_weight) * answer_loss
        batch_size = batch["math_input_ids"].shape[0]
        loss_sum += float(loss) * batch_size
        lm_loss_sum += float(lm_loss) * batch_size
        examples += batch_size
        valid = classes.ne(-100)
        answer_valid += int(valid.sum())
        answer_correct += int((output.answer_logits.argmax(-1).eq(classes) & valid).sum())
        correct, total, sequence = masked_token_statistics(
            output.logits, batch["math_labels"]
        )
        token_correct += correct
        token_total += total
        sequence_correct += sequence
        update_teacher_forced_breakdowns(
            breakdown_sums,
            logits=output.logits,
            labels=batch["math_labels"],
            records=list(batch["records"]),
        )
    if not examples:
        raise RuntimeError("math evaluation loader produced no examples")
    return {
        "examples": examples,
        "loss": loss_sum / examples,
        "language_loss": lm_loss_sum / examples,
        "answer_head_enabled": bool(model.answer_head_enabled),
        "answer_head_accuracy": answer_correct / max(1, answer_valid),
        "answer_head_examples": answer_valid,
        "teacher_forced_token_accuracy": token_correct / max(1, token_total),
        "teacher_forced_sequence_accuracy": sequence_correct / examples,
        "breakdowns": summarize_teacher_forced_breakdowns(breakdown_sums),
    }


def _phase_generation_acceptance(
    *,
    phase: dict[str, Any] | None,
    generation_panels: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    epoch: int,
) -> dict[str, Any] | None:
    """Build an auditable, fail-closed generation acceptance decision."""

    if (
        phase is None
        or "minimum_generation_accuracy" not in phase
        or "minimum_valid_rate" not in phase
    ):
        return None
    primary_name = str(phase.get("primary_generation_panel", "validation"))
    primary = generation_panels.get(primary_name)
    if primary is None:
        raise RuntimeError(
            f"generation acceptance primary panel is unavailable: {primary_name}"
        )

    checks: dict[str, dict[str, Any]] = {}

    def add_check(name: str, observed: float, minimum: float) -> None:
        checks[name] = {
            "observed": float(observed),
            "minimum": float(minimum),
            "pass": float(observed) >= float(minimum),
        }

    minimum_accuracy = float(phase["minimum_generation_accuracy"])
    minimum_valid_rate = float(phase["minimum_valid_rate"])
    add_check("primary_generation_accuracy", primary.get("accuracy", 0.0), minimum_accuracy)
    add_check("primary_valid_rate", primary.get("valid_rate", 0.0), minimum_valid_rate)

    for dimension in ("source", "family", "difficulty"):
        threshold_key = f"minimum_generation_accuracy_by_{dimension}"
        configured = phase.get(threshold_key, {})
        if not isinstance(configured, dict):
            raise ValueError(f"{threshold_key} must be an object")
        observed_groups = {
            str(name): values
            for name, values in primary.get(f"by_{dimension}", {}).items()
        }
        for name, minimum in configured.items():
            group = observed_groups.get(str(name))
            if group is None:
                checks[f"primary_{dimension}:{name}"] = {
                    "observed": None,
                    "minimum": float(minimum),
                    "examples": 0,
                    "pass": False,
                }
                continue
            add_check(
                f"primary_{dimension}:{name}",
                group.get("accuracy", 0.0),
                float(minimum),
            )
            checks[f"primary_{dimension}:{name}"]["examples"] = int(
                group.get("examples", 0)
            )

    panel_accuracy_thresholds = phase.get(
        "minimum_generation_accuracy_by_panel", {}
    )
    panel_validity_thresholds = phase.get("minimum_valid_rate_by_panel", {})
    if not isinstance(panel_accuracy_thresholds, dict) or not isinstance(
        panel_validity_thresholds, dict
    ):
        raise ValueError("generation panel acceptance thresholds must be objects")
    panel_names = sorted(
        set(panel_accuracy_thresholds) | set(panel_validity_thresholds)
    )
    for name in panel_names:
        panel = generation_panels.get(str(name))
        if panel is None:
            checks[f"panel:{name}:available"] = {
                "observed": None,
                "minimum": 1.0,
                "pass": False,
            }
            continue
        if name in panel_accuracy_thresholds:
            add_check(
                f"panel:{name}:generation_accuracy",
                panel.get("accuracy", 0.0),
                float(panel_accuracy_thresholds[name]),
            )
        if name in panel_validity_thresholds:
            add_check(
                f"panel:{name}:valid_rate",
                panel.get("valid_rate", 0.0),
                float(panel_validity_thresholds[name]),
            )

    if "minimum_teacher_forced_token_accuracy" in phase:
        add_check(
            "teacher_forced_token_accuracy",
            validation.get("teacher_forced_token_accuracy", 0.0),
            float(phase["minimum_teacher_forced_token_accuracy"]),
        )
    if "minimum_teacher_forced_sequence_accuracy" in phase:
        add_check(
            "teacher_forced_sequence_accuracy",
            validation.get("teacher_forced_sequence_accuracy", 0.0),
            float(phase["minimum_teacher_forced_sequence_accuracy"]),
        )
    if "maximum_validation_loss" in phase:
        observed_loss = float(validation.get("loss", float("inf")))
        maximum_loss = float(phase["maximum_validation_loss"])
        checks["validation_loss"] = {
            "observed": observed_loss,
            "maximum": maximum_loss,
            "pass": observed_loss <= maximum_loss,
        }

    return {
        "phase": str(phase["name"]),
        "primary_panel": primary_name,
        "minimum_generation_accuracy": minimum_accuracy,
        "minimum_valid_rate": minimum_valid_rate,
        "generation_accuracy": float(primary.get("accuracy", 0.0)),
        "valid_rate": float(primary.get("valid_rate", 0.0)),
        "terminal_epoch": epoch == int(phase["through_epoch"]),
        "checks": checks,
        "pass": bool(checks) and all(bool(check["pass"]) for check in checks.values()),
    }


def train_math_tower(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    resume: bool = False,
    max_batches: int | None = None,
    require_calibration: bool = True,
    disable_early_stopping: bool = False,
    wandb_options: dict[str, Any] | None = None,
    initial_checkpoint: str | Path | None = None,
    artifact_directory: str | Path | None = None,
    working_directory: str | Path | None = None,
    recovery_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    root, manifest = load_data_contract(config)
    if require_calibration:
        from .gpt_baseline import verify_calibration_gate

        verify_calibration_gate(config, manifest)
    train_dataset = split_dataset(root, manifest, "train")
    validation_dataset = split_dataset(root, manifest, "validation")
    contract = copy.deepcopy(recovery_contract or {})
    require_acceptance_for_best = bool(
        contract.get("require_acceptance_for_best", False)
    )
    contract_sha256 = (
        hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if contract
        else None
    )
    settings = copy.deepcopy(config["math_training"])
    settings.update(contract.get("math_training", {}))
    generation_validation_settings = copy.deepcopy(DEFAULT_V2_GENERATION_VALIDATION)
    generation_validation_settings.update(
        settings.get("generation_validation", {})
    )
    configured_generation_panels = list(
        generation_validation_settings.get("panels", [])
    )
    generation_panel_datasets: dict[str, EquationDataset] = {}
    if configured_generation_panels:
        seen_panel_names: set[str] = set()
        for panel in configured_generation_panels:
            if not isinstance(panel, dict):
                raise ValueError("generation validation panels must be objects")
            name = str(panel.get("name", "")).strip()
            split = str(panel.get("split", "")).strip()
            if not name or name in seen_panel_names:
                raise ValueError("generation validation panel names must be unique")
            if split not in manifest.get("splits", {}):
                raise ValueError(
                    f"generation validation panel {name} uses unavailable split {split}"
                )
            seen_panel_names.add(name)
            generation_panel_datasets[name] = split_dataset(root, manifest, split)
    curriculum_config = copy.deepcopy(config)
    curriculum_config["data"]["curriculum"].update(contract.get("curriculum", {}))
    phases = list(contract.get("phases", []))
    tokenizer = ByteMathTokenizer()
    collator = MathCollator(
        tokenizer,
        int(config["data"]["max_math_length"]),
        target_mode=str(settings.get("target_mode", "full_trace_v1")),
        input_view=str(settings.get("input_view", SHARED_MATH_INPUT_VIEW)),
    )
    early_stopping_enabled = not bool(disable_early_stopping)
    artifact_dir = Path(
        artifact_directory
        or (Path(config["project"]["artifact_root"]) / "math")
    ).expanduser().resolve()
    work_dir = Path(working_directory or artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = work_dir / "metrics.jsonl"
    durable_metrics_path = artifact_dir / "metrics.jsonl"
    status_path = work_dir / "status.json"
    durable_status_path = artifact_dir / "status.json"
    mirrored_status_write_failures = 0

    def write_status(payload: dict[str, Any]) -> None:
        nonlocal mirrored_status_write_failures
        atomic_json_dump(payload, status_path)
        if status_path != durable_status_path:
            try:
                atomic_json_dump(payload, durable_status_path)
            except OSError:
                mirrored_status_write_failures += 1
    started_at = time.time()
    model = build_math_tower(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduled_examples = int(
        curriculum_config["data"].get("curriculum", {}).get(
            "examples_per_epoch", len(train_dataset)
        )
    )
    steps_per_epoch = max(
        1, math.ceil(scheduled_examples / int(settings["batch_size"]))
    )
    total_steps = int(settings["max_epochs"]) * steps_per_epoch
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=float(settings["warmup_fraction"]),
        minimum_ratio=float(settings["minimum_learning_rate"])
        / float(settings["learning_rate"]),
    )
    dtype = precision_dtype(settings["precision"], device)
    scaler = make_scaler(device, dtype)
    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    best_checkpoint_metric: float | None = None
    patience = 0
    source_provenance: dict[str, Any] | None = None
    if resume and initial_checkpoint is not None:
        raise ValueError("resume and initial_checkpoint are mutually exclusive")
    if initial_checkpoint is not None:
        source_path = Path(initial_checkpoint).expanduser().resolve()
        expected_source_sha256 = contract.get("source_checkpoint_sha256")
        checkpoint, source_sha256 = _load_math_initialization_checkpoint(
            source_path,
            expected_sha256=expected_source_sha256,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        source_provenance = {
            "path": str(source_path),
            "sha256": source_sha256,
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "source_manifest_sha256": checkpoint.get("manifest_sha256"),
            "recovery_manifest_sha256": manifest["manifest_sha256"],
            "manifest_reset": True,
            "optimizer_reset": True,
            "scheduler_reset": True,
        }
    if resume:
        checkpoint_path = latest_checkpoint(artifact_dir)
        if checkpoint_path is None:
            raise FileNotFoundError("no math checkpoint is available to resume")
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_stage="math",
            expected_config_sha256=config_sha256(config),
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        best_checkpoint_metric = best_metric
        patience = int(checkpoint["patience"])
    write_status(
        _status_payload(
            stage="math",
            state="running",
            epoch=start_epoch - 1,
            global_step=global_step,
            started_at=started_at,
        ),
    )
    best_path = artifact_dir / "math.best.pth"
    working_best_path = work_dir / "math.best.pth"
    final_metrics: dict[str, Any] = {}
    state = "completed"
    active_phase_name: str | None = None
    report_every_steps = max(1, int(settings.get("report_every_steps", 100)))
    stop_reason = "max_epochs"
    wandb_tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact_dir,
        stage="math",
        config={
            "project": config["project"]["name"],
            "seed": seed,
            "model": config["math_tower"],
            "training": settings,
            "runtime_overrides": {
                "early_stopping_enabled": early_stopping_enabled,
                "recovery_contract_sha256": contract_sha256,
                "source_checkpoint": source_provenance,
            },
            "config_sha256": config_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    try:
        for epoch in range(start_epoch, int(settings["max_epochs"]) + 1):
            model.train()
            phase = next(
                (
                    item
                    for item in phases
                    if epoch <= int(item["through_epoch"])
                ),
                phases[-1] if phases else None,
            )
            phase_name = str(phase["name"]) if phase is not None else None
            if phase_name != active_phase_name:
                best_metric = float("-inf")
                patience = 0
                active_phase_name = phase_name
            epoch_dataset, curriculum = math_epoch_dataset(
                train_dataset,
                curriculum_config,
                epoch=epoch,
                seed=seed,
                phase_override=phase,
            )
            train_loader = make_loader(
                epoch_dataset,
                collator,
                batch_size=int(settings["batch_size"]),
                shuffle=True,
                seed=seed,
                epoch=epoch,
                num_workers=int(settings["num_workers"]),
            )
            train_loss = 0.0
            trained_examples = 0
            epoch_started_at = time.time()
            epoch_start_step = global_step
            epoch_batches_total = len(train_loader)
            if max_batches is not None:
                epoch_batches_total = min(epoch_batches_total, int(max_batches))
            for batch_index, raw_batch in enumerate(train_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, dtype):
                    output = model(
                        batch["math_input_ids"],
                        batch["math_attention_mask"],
                        batch["math_prefix_lengths"],
                    )
                    lm_loss = answer_weighted_causal_language_loss(
                        output.logits,
                        batch["math_labels"],
                        batch["math_answer_labels"],
                        answer_weight=float(settings.get("answer_token_weight", 1.0)),
                    )
                    classes = model.answer_classes(batch["answer_values"])
                    answer_loss = optional_answer_loss(output.answer_logits, classes)
                    loss = lm_loss + float(settings["answer_head_weight"]) * answer_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                batch_size = batch["math_input_ids"].shape[0]
                train_loss += float(loss.detach()) * batch_size
                trained_examples += batch_size
                global_step += 1
                batch_completed = batch_index + 1
                if (
                    batch_completed == 1
                    or batch_completed % report_every_steps == 0
                    or batch_completed == epoch_batches_total
                ):
                    progress_metrics = _training_progress_metrics(
                        epoch=epoch,
                        batch_completed=batch_completed,
                        batches_total=epoch_batches_total,
                        global_step=global_step,
                        total_steps=total_steps,
                        loss_sum=train_loss,
                        trained_examples=trained_examples,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        interval_started_at=epoch_started_at,
                        interval_start_step=epoch_start_step,
                    )
                    write_status(
                        _status_payload(
                            stage="math",
                            state="running",
                            epoch=epoch,
                            global_step=global_step,
                            metrics=progress_metrics,
                            started_at=started_at,
                        ),
                    )
                    wandb_tracker.log(
                        {"train": progress_metrics},
                        global_step=global_step,
                        epoch=epoch,
                        event="training_progress",
                    )
            training_finished_at = time.time()
            validation_loader = make_loader(
                validation_dataset,
                collator,
                batch_size=int(settings["eval_batch_size"]),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=int(settings["num_workers"]),
            )
            validation = evaluate_math_tower(
                model,
                validation_loader,
                device,
                dtype,
                float(settings["answer_head_weight"]),
                max_batches=max_batches,
            )
            generation_settings = generation_validation_settings
            generation_validation: dict[str, Any] | None = None
            generation_panels: dict[str, dict[str, Any]] = {}
            if (
                manifest.get("format") == "cftn_text_broad_math_v2"
                and bool(generation_settings.get("enabled", False))
                and epoch % max(1, int(generation_settings.get("every_epochs", 1)))
                == 0
            ):
                panel_specs = configured_generation_panels or [
                    {
                        "name": "validation",
                        "split": "validation",
                        "phase_filtered": True,
                    }
                ]
                for panel_spec in panel_specs:
                    panel_name = str(panel_spec["name"])
                    panel_dataset = generation_panel_datasets.get(
                        panel_name, validation_dataset
                    )
                    panel_records = panel_dataset.records
                    if bool(panel_spec.get("phase_filtered", False)) and phase is not None:
                        panel_records = _filter_v2_records_for_phase(
                            panel_records, phase
                        )
                    if not panel_records:
                        raise RuntimeError(
                            f"generation validation panel {panel_name} selected no records"
                        )
                    generation_rows_name = (
                        f"generation_validation_{panel_name}_epoch_{epoch:04d}.jsonl"
                        if configured_generation_panels
                        else f"generation_validation_epoch_{epoch:04d}.jsonl"
                    )
                    generation_rows_path = work_dir / generation_rows_name
                    generation_panels[panel_name] = evaluate_generation_panel(
                        model,
                        tokenizer,
                        panel_records,
                        maximum_examples=int(
                            panel_spec.get(
                                "examples", generation_settings.get("examples", 96)
                            )
                        ),
                        batch_size=int(
                            panel_spec.get(
                                "batch_size",
                                generation_settings.get("batch_size", 16),
                            )
                        ),
                        max_new_tokens=int(
                            panel_spec.get(
                                "max_new_tokens",
                                generation_settings.get("max_new_tokens", 512),
                            )
                        ),
                        failure_examples=int(
                            panel_spec.get(
                                "failure_examples",
                                generation_settings.get("failure_examples", 8),
                            )
                        ),
                        rows_path=generation_rows_path,
                        input_view=collator.input_view,
                    )
                    generation_panels[panel_name]["split"] = str(
                        panel_spec.get("split", "validation")
                    )
                    if work_dir != artifact_dir:
                        atomic_copy_file(
                            generation_rows_path,
                            artifact_dir / generation_rows_path.name,
                        )
                primary_panel_name = str(
                    (phase or {}).get(
                        "primary_generation_panel", panel_specs[0]["name"]
                    )
                )
                generation_validation = generation_panels.get(primary_panel_name)
                if generation_validation is None:
                    raise RuntimeError(
                        "primary generation validation panel is unavailable: "
                        + primary_panel_name
                    )
                validation["generation"] = generation_validation
                validation["generation_panels"] = generation_panels
            if generation_validation is not None:
                panel_accuracies = [
                    float(panel["accuracy"]) for panel in generation_panels.values()
                ]
                panel_valid_rates = [
                    float(panel["valid_rate"]) for panel in generation_panels.values()
                ]
                selection_metric = (
                    sum(panel_accuracies) / max(1, len(panel_accuracies))
                    + 0.01 * min(panel_valid_rates)
                    + 0.0001 * float(validation["teacher_forced_sequence_accuracy"])
                    - 1e-8 * float(validation["loss"])
                )
                selection_basis = (
                    "mean_greedy_generation_accuracy_across_panels"
                    if len(generation_panels) > 1
                    else "greedy_generation_accuracy"
                )
            elif (
                manifest.get("format") == "cftn_text_broad_math_v2"
                or not model.answer_head_enabled
            ):
                selection_metric = float(
                    validation["teacher_forced_sequence_accuracy"]
                ) - 1e-6 * float(validation["loss"])
                selection_basis = "teacher_forced_sequence_accuracy"
            else:
                selection_metric = float(
                    validation["answer_head_accuracy"]
                ) - 1e-6 * float(validation["loss"])
                selection_basis = "answer_head_accuracy"
            phase_acceptance = _phase_generation_acceptance(
                phase=phase,
                generation_panels=generation_panels,
                validation=validation,
                epoch=epoch,
            )
            improved = selection_metric > best_metric
            if improved:
                best_metric = selection_metric
                patience = 0
            else:
                patience += 1
            checkpoint_eligible = (
                not require_acceptance_for_best
                or phase_acceptance is None
                or bool(phase_acceptance["pass"])
            )
            promote_best = checkpoint_eligible and (
                improved
                or (
                    phase_acceptance is not None
                    and bool(phase_acceptance["pass"])
                    and not best_path.exists()
                )
            )
            phase_gate = (
                phase_acceptance
                if phase_acceptance is not None
                and bool(phase_acceptance["terminal_epoch"])
                else None
            )
            final_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss / max(1, trained_examples),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": validation,
                "selection_metric": selection_metric,
                "selection_basis": selection_basis,
                "best_metric": best_metric,
                "patience": patience,
                "early_stopping_enabled": early_stopping_enabled,
                "curriculum": curriculum,
                "recovery_contract_sha256": contract_sha256,
                "source_checkpoint": source_provenance,
                "input_view": collator.input_view,
                "target_mode": collator.target_mode,
                "answer_token_weight": float(
                    settings.get("answer_token_weight", 1.0)
                ),
                "require_acceptance_for_best": require_acceptance_for_best,
                "checkpoint_eligible": checkpoint_eligible,
                "checkpoint_promoted": promote_best,
                "curriculum_acceptance": phase_acceptance,
                "curriculum_gate": phase_gate,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "timing": {
                    "training_seconds": training_finished_at - epoch_started_at,
                    "validation_seconds": time.time() - training_finished_at,
                    "epoch_seconds": time.time() - epoch_started_at,
                    "training_steps_per_second": (
                        (global_step - epoch_start_step)
                        / max(1e-9, training_finished_at - epoch_started_at)
                    ),
                    "training_examples_per_second": (
                        trained_examples
                        / max(1e-9, training_finished_at - epoch_started_at)
                    ),
                    "eta_seconds_to_max_epochs": (
                        (int(settings["max_epochs"]) - epoch)
                        * (time.time() - epoch_started_at)
                    ),
                },
                "gpu": gpu_status(),
            }
            append_jsonl(final_metrics, metrics_path)
            if metrics_path != durable_metrics_path:
                atomic_copy_file(metrics_path, durable_metrics_path)
            wandb_tracker.log(
                final_metrics,
                global_step=global_step,
                epoch=epoch,
                event="epoch_validation",
            )
            payload = build_checkpoint(
                stage="math",
                epoch=epoch,
                global_step=global_step,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                config_sha256=config_sha256(config),
                manifest_sha256=manifest["manifest_sha256"],
                best_metric=best_metric,
                patience=patience,
                extra={"metrics": final_metrics},
            )
            checkpoint_path = work_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            atomic_torch_save(payload, checkpoint_path)
            durable_checkpoint_path = artifact_dir / checkpoint_path.name
            if checkpoint_path != durable_checkpoint_path:
                atomic_copy_file(checkpoint_path, durable_checkpoint_path)
            rotate_latest(
                work_dir,
                int(
                    settings.get(
                        "keep_latest_checkpoints",
                        config["monitoring"]["keep_latest_checkpoints"],
                    )
                ),
            )
            if work_dir != artifact_dir:
                rotate_latest(
                    artifact_dir,
                    int(settings.get("keep_latest_checkpoints", 3)),
                )
            if promote_best:
                best_checkpoint_metric = selection_metric
                atomic_torch_save(payload, working_best_path)
                if working_best_path != best_path:
                    atomic_copy_file(working_best_path, best_path)
            write_status(
                _status_payload(
                    stage="math",
                    state="running",
                    epoch=epoch,
                    global_step=global_step,
                    metrics=final_metrics,
                    started_at=started_at,
                ),
                )
            if (
                phase_acceptance is not None
                and bool(phase_acceptance["pass"])
                and bool(phase.get("stop_on_pass", False))
            ):
                stop_reason = f"curriculum_gate_passed_{phase['name']}"
                state = "completed"
                break
            if phase_gate is not None and not phase_gate["pass"]:
                stop_reason = f"curriculum_gate_failed_{phase['name']}"
                state = "failed_acceptance"
                break
            if _should_stop_early(
                epoch=epoch,
                patience=patience,
                settings=settings,
                enabled=early_stopping_enabled,
            ):
                stop_reason = "early_stopping_validation_plateau"
                break
    except BaseException as exc:
        write_status(
            _status_payload(
                stage="math",
                state="error",
                epoch=locals().get("epoch", start_epoch - 1),
                global_step=global_step,
                metrics={"error": repr(exc)},
                started_at=started_at,
            ),
        )
        wandb_tracker.update_summary(
            {"run/state": "error", "run/error": repr(exc)}
        )
        wandb_tracker.finish(exit_code=1)
        raise
    best_checkpoint_exists = best_path.is_file()
    result = {
        "stage": "math",
        "state": state,
        "stop_reason": stop_reason,
        "early_stopping_enabled": early_stopping_enabled,
        "best_checkpoint": str(best_path.resolve()) if best_checkpoint_exists else None,
        "best_checkpoint_sha256": (
            file_sha256(best_path) if best_checkpoint_exists else None
        ),
        "best_checkpoint_metric": best_checkpoint_metric,
        "final_metrics": final_metrics,
        "recovery_contract_sha256": contract_sha256,
        "source_checkpoint": source_provenance,
        "storage": {
            "working_directory": str(work_dir),
            "artifact_directory": str(artifact_dir),
            "mirrored_status_write_failures": mirrored_status_write_failures,
        },
    }
    atomic_json_dump(result, artifact_dir / "summary.json")
    write_status(
        _status_payload(
            stage="math",
            state=state,
            epoch=int(final_metrics.get("epoch", 0)),
            global_step=global_step,
            metrics=final_metrics,
            started_at=started_at,
        ),
    )
    wandb_tracker.update_summary(
        {
            "run/state": state,
            "run/stop_reason": stop_reason,
            "run/best_metric": best_metric,
            "run/final_epoch": int(final_metrics.get("epoch", 0)),
            "run/early_stopping_enabled": early_stopping_enabled,
        }
    )
    wandb_tracker.finish()
    return result


@torch.no_grad()
def evaluate_bridge_model(
    model: CFTNTextModel,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    settings: dict[str, Any],
    stage: str,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    examples = 0
    loss_sum = 0.0
    shuffled_loss_sum = 0.0
    math_correct = math_total = math_sequences = 0
    gpt_correct = gpt_total = gpt_sequences = 0
    answer_correct = answer_valid = 0
    g2m_gates: list[torch.Tensor] = []
    m2g_gates: list[torch.Tensor] = []
    math_receiver_gates: list[torch.Tensor] = []
    gpt_receiver_gates: list[torch.Tensor] = []
    g2m_enabled = stage == "bidirectional"
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        kwargs = {
            "gpt_to_math_enabled": g2m_enabled,
            "math_to_gpt_enabled": True,
            "math_loss_weight": float(settings["math_loss_weight"]) if g2m_enabled else 0.0,
            "gpt_loss_weight": float(settings["gpt_loss_weight"]),
            "answer_head_weight": float(settings["answer_head_weight"]) if g2m_enabled else 0.0,
        }
        with autocast_context(device, dtype):
            output = model(batch, **kwargs)
            shuffled = model(
                batch,
                shuffle_gpt_to_math=g2m_enabled,
                shuffle_math_to_gpt=True,
                **kwargs,
            )
        batch_size = batch["math_input_ids"].shape[0]
        examples += batch_size
        loss_sum += float(output.loss) * batch_size
        shuffled_loss_sum += float(shuffled.loss) * batch_size
        correct, total, sequence = masked_token_statistics(
            output.math_output.logits, batch["math_labels"]
        )
        math_correct += correct
        math_total += total
        math_sequences += sequence
        correct, total, sequence = masked_token_statistics(
            output.gpt_logits, batch["gpt_labels"]
        )
        gpt_correct += correct
        gpt_total += total
        gpt_sequences += sequence
        classes = model.math_tower.answer_classes(batch["answer_values"])
        valid = classes.ne(-100)
        answer_valid += int(valid.sum())
        answer_correct += int(
            (output.math_output.answer_logits.argmax(-1).eq(classes) & valid).sum()
        )
        g2m_gates.append(output.gpt_to_math.gate)
        m2g_gates.append(output.math_to_gpt.gate)
        math_receiver_gates.extend(output.math_receiver_gates.values())
        gpt_receiver_gates.extend(output.gpt_receiver_gates.values())
    if not examples:
        raise RuntimeError("bridge evaluation loader produced no examples")
    correct_loss = loss_sum / examples
    shuffled_loss = shuffled_loss_sum / examples
    return {
        "examples": examples,
        "loss": correct_loss,
        "shuffled_loss": shuffled_loss,
        "shuffled_loss_gap": shuffled_loss - correct_loss,
        "math_teacher_forced_token_accuracy": math_correct / max(1, math_total),
        "math_teacher_forced_sequence_accuracy": math_sequences / examples,
        "gpt_teacher_forced_token_accuracy": gpt_correct / max(1, gpt_total),
        "gpt_teacher_forced_sequence_accuracy": gpt_sequences / examples,
        "answer_head_accuracy": answer_correct / max(1, answer_valid),
        "gpt_to_math_gate": summarize_gate(g2m_gates),
        "math_to_gpt_gate": summarize_gate(m2g_gates),
        "math_receiver_gate": summarize_gate(math_receiver_gates),
        "gpt_receiver_gate": summarize_gate(gpt_receiver_gates),
    }


def train_bridges(
    config: dict[str, Any],
    *,
    stage: str,
    device_name: str = "cuda",
    resume: bool = False,
    initialize_from: str | Path | None = None,
    math_checkpoint_path: str | Path | None = None,
    gate_mode: str = "contextual",
    view_mode: str = "shared",
    max_batches: int | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in {"m2g", "bidirectional"}:
        raise ValueError("bridge stage must be m2g or bidirectional")
    if view_mode not in {"shared", "complementary"}:
        raise ValueError("bridge view mode must be shared or complementary")
    if stage == "m2g" and view_mode != "shared":
        raise ValueError(
            "math-to-GPT pretraining requires shared views so the frozen math "
            "tower has the complete equation"
        )
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    root, manifest = load_data_contract(config)
    settings = config["bridge_training"]
    math_checkpoint_path = Path(
        math_checkpoint_path
        or Path(config["project"]["artifact_root"]) / "math" / "math.best.pth"
    )
    model, gpt_tokenizer = build_cftn_model(
        config, math_checkpoint_path, manifest, device
    )
    model.set_gate_mode(gate_mode)
    model.set_trainable_stage(stage)
    if initialize_from is not None:
        initial = load_checkpoint(
            initialize_from,
            expected_config_sha256=config_sha256(config),
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location=device,
        )
        model.load_trainable_state_dict(initial["model_state"], strict=False)
    math_tokenizer = ByteMathTokenizer()
    collator = CFTNCollator(
        math_tokenizer,
        gpt_tokenizer,
        int(config["data"]["max_math_length"]),
        int(config["data"]["max_gpt_length"]),
    )
    bridge_train_split = str(
        config["data"].get("bridge_train_split", "train")
    )
    bridge_validation_split = str(
        config["data"].get("bridge_validation_split", "validation")
    )
    train_dataset = split_dataset(root, manifest, bridge_train_split)
    validation_dataset = split_dataset(root, manifest, bridge_validation_split)
    if config["data"].get("format") == "cftn_text_broad_math_v2":
        eligible_train = [
            record
            for record in train_dataset.records
            if record.get("gpt_problem") and record.get("math_problem")
        ]
        eligible_validation = [
            record
            for record in validation_dataset.records
            if record.get("gpt_problem") and record.get("math_problem")
        ]
        train_limit = int(
            config["data"].get("bridge_train_examples", len(eligible_train))
        )
        validation_limit = int(
            config["data"].get(
                "bridge_validation_examples", len(eligible_validation)
            )
        )
        train_dataset = EquationDataset(eligible_train[:train_limit])
        validation_dataset = EquationDataset(
            eligible_validation[:validation_limit]
        )
        if not train_dataset.records or not validation_dataset.records:
            raise RuntimeError("V2 bridge training requires private-view records")
    if view_mode != "shared":
        train_dataset = EquationDataset(
            apply_view_mode(train_dataset.records, view_mode=view_mode, seed=seed)
        )
        validation_dataset = EquationDataset(
            apply_view_mode(validation_dataset.records, view_mode=view_mode, seed=seed)
        )
    artifact_name = f"bridge_{stage}_{gate_mode}"
    if view_mode != "shared":
        artifact_name += f"_{view_mode}"
    artifact_dir = Path(config["project"]["artifact_root"]) / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.jsonl"
    status_path = artifact_dir / "status.json"
    started_at = time.time()
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise RuntimeError("bridge stage has no trainable parameters")
    stability_policy = _bridge_stability_policy(settings)
    gate_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".gate_network." in name
    ]
    non_gate_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".gate_network." not in name
    ]
    optimizer_groups: list[dict[str, Any]] = []
    if non_gate_parameters:
        optimizer_groups.append(
            {
                "params": non_gate_parameters,
                "lr": float(stability_policy["effective_learning_rate"]),
                "weight_decay": float(settings["weight_decay"]),
                "group_name": "bridge",
            }
        )
    if gate_parameters:
        optimizer_groups.append(
            {
                "params": gate_parameters,
                "lr": float(stability_policy["effective_learning_rate"])
                * float(stability_policy["gate_learning_rate_multiplier"]),
                # Decoupled weight decay moves the negative gate bias toward
                # zero and therefore toward an always-open gate.  Do not apply
                # it to contextual gate parameters.
                "weight_decay": 0.0,
                "group_name": "contextual_gates",
            }
        )
    optimizer = AdamW(
        optimizer_groups,
        lr=float(stability_policy["effective_learning_rate"]),
    )
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / int(settings["batch_size"])))
    total_steps = int(settings["max_epochs"]) * steps_per_epoch
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=float(settings["warmup_fraction"]),
        minimum_ratio=float(stability_policy["minimum_learning_rate"])
        / float(stability_policy["effective_learning_rate"]),
    )
    dtype = precision_dtype(settings["precision"], device)
    scaler = make_scaler(device, dtype)
    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    best_validation: dict[str, Any] | None = None
    patience = 0
    if resume:
        checkpoint_path = latest_checkpoint(artifact_dir)
        if checkpoint_path is None:
            raise FileNotFoundError("no bridge checkpoint is available to resume")
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_stage=stage,
            expected_config_sha256=config_sha256(config),
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location=device,
        )
        if checkpoint["extra"].get("gate_mode") != gate_mode:
            raise ValueError("checkpoint gate mode differs")
        if checkpoint["extra"].get("view_mode", "shared") != view_mode:
            raise ValueError("checkpoint view mode differs")
        model.load_trainable_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        best_validation = checkpoint["extra"].get("best_validation")
        patience = int(checkpoint["patience"])
    atomic_json_dump(
        _status_payload(
            stage=stage,
            state="running",
            epoch=start_epoch - 1,
            global_step=global_step,
            started_at=started_at,
        ),
        status_path,
    )
    best_path = artifact_dir / f"bridge_{stage}.best.pth"
    final_metrics: dict[str, Any] = {}
    g2m_enabled = stage == "bidirectional"
    report_every_steps = max(1, int(settings.get("report_every_steps", 100)))
    stop_reason = "max_epochs"
    wandb_tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact_dir,
        stage=stage,
        config={
            "project": config["project"]["name"],
            "seed": seed,
            "gate_mode": gate_mode,
            "view_mode": view_mode,
            "bridge": config["bridge"],
            "training": settings,
            "stability_policy": stability_policy,
            "config_sha256": config_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"],
            "math_checkpoint": str(math_checkpoint_path.resolve()),
        },
    )
    try:
        for epoch in range(start_epoch, int(settings["max_epochs"]) + 1):
            model.train()
            model.math_tower.eval()
            train_loader = make_loader(
                train_dataset,
                collator,
                batch_size=int(settings["batch_size"]),
                shuffle=True,
                seed=seed,
                epoch=epoch,
                num_workers=int(settings["num_workers"]),
            )
            train_loss = 0.0
            trained_examples = 0
            epoch_started_at = time.time()
            epoch_start_step = global_step
            epoch_batches_total = len(train_loader)
            if max_batches is not None:
                epoch_batches_total = min(epoch_batches_total, int(max_batches))
            for batch_index, raw_batch in enumerate(train_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, dtype):
                    output = model(
                        batch,
                        gpt_to_math_enabled=g2m_enabled,
                        math_to_gpt_enabled=True,
                        math_loss_weight=(
                            float(settings["math_loss_weight"]) if g2m_enabled else 0.0
                        ),
                        gpt_loss_weight=float(settings["gpt_loss_weight"]),
                        answer_head_weight=(
                            float(settings["answer_head_weight"]) if g2m_enabled else 0.0
                        ),
                    )
                scaler.scale(output.loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(settings["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                batch_size = batch["math_input_ids"].shape[0]
                train_loss += float(output.loss.detach()) * batch_size
                trained_examples += batch_size
                global_step += 1
                batch_completed = batch_index + 1
                if (
                    batch_completed == 1
                    or batch_completed % report_every_steps == 0
                    or batch_completed == epoch_batches_total
                ):
                    progress_metrics = _training_progress_metrics(
                        epoch=epoch,
                        batch_completed=batch_completed,
                        batches_total=epoch_batches_total,
                        global_step=global_step,
                        total_steps=total_steps,
                        loss_sum=train_loss,
                        trained_examples=trained_examples,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        interval_started_at=epoch_started_at,
                        interval_start_step=epoch_start_step,
                    )
                    atomic_json_dump(
                        _status_payload(
                            stage=stage,
                            state="running",
                            epoch=epoch,
                            global_step=global_step,
                            metrics=progress_metrics,
                            started_at=started_at,
                        ),
                        status_path,
                    )
                    wandb_tracker.log(
                        {"train": progress_metrics},
                        global_step=global_step,
                        epoch=epoch,
                        event="training_progress",
                    )
            training_finished_at = time.time()
            validation_loader = make_loader(
                validation_dataset,
                collator,
                batch_size=int(settings["eval_batch_size"]),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=int(settings["num_workers"]),
            )
            validation = evaluate_bridge_model(
                model,
                validation_loader,
                device,
                dtype,
                settings,
                stage,
                max_batches=max_batches,
            )
            selection_metric = (
                float(validation["gpt_teacher_forced_sequence_accuracy"])
                + (0.1 * float(validation["math_teacher_forced_sequence_accuracy"]) if g2m_enabled else 0.0)
                + 0.01 * max(0.0, float(validation["shuffled_loss_gap"]))
                - 1e-6 * float(validation["loss"])
            )
            collapse_guard = _bridge_collapse_diagnostics(
                validation, best_validation, stability_policy
            )
            improved = not collapse_guard["triggered"] and selection_metric > best_metric
            if improved:
                best_metric = selection_metric
                best_validation = validation
                patience = 0
            else:
                patience += 1
            final_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss / max(1, trained_examples),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "validation": validation,
                "selection_metric": selection_metric,
                "best_metric": best_metric,
                "patience": patience,
                "stability_policy": stability_policy,
                "collapse_guard": collapse_guard,
                "gate_mode": gate_mode,
                "view_mode": view_mode,
                "trainable_parameters": model.trainable_parameter_count(),
                "execution_counts": model.execution_counts(),
                "timing": {
                    "training_seconds": training_finished_at - epoch_started_at,
                    "validation_seconds": time.time() - training_finished_at,
                    "epoch_seconds": time.time() - epoch_started_at,
                    "training_steps_per_second": (
                        (global_step - epoch_start_step)
                        / max(1e-9, training_finished_at - epoch_started_at)
                    ),
                    "training_examples_per_second": (
                        trained_examples
                        / max(1e-9, training_finished_at - epoch_started_at)
                    ),
                    "eta_seconds_to_max_epochs": (
                        (int(settings["max_epochs"]) - epoch)
                        * (time.time() - epoch_started_at)
                    ),
                },
                "gpu": gpu_status(),
            }
            append_jsonl(final_metrics, metrics_path)
            wandb_tracker.log(
                final_metrics,
                global_step=global_step,
                epoch=epoch,
                event="epoch_validation",
            )
            payload = build_checkpoint(
                stage=stage,
                epoch=epoch,
                global_step=global_step,
                model_state=model.trainable_state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                config_sha256=config_sha256(config),
                manifest_sha256=manifest["manifest_sha256"],
                best_metric=best_metric,
                patience=patience,
                extra={
                    "metrics": final_metrics,
                    "best_validation": best_validation,
                    "stability_policy": stability_policy,
                    "collapse_guard": collapse_guard,
                    "gate_mode": gate_mode,
                    "view_mode": view_mode,
                    "math_checkpoint": str(math_checkpoint_path.resolve()),
                    "math_checkpoint_sha256": file_sha256(math_checkpoint_path),
                },
            )
            checkpoint_path = artifact_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            atomic_torch_save(payload, checkpoint_path)
            rotate_latest(
                artifact_dir,
                int(
                    settings.get(
                        "keep_latest_checkpoints",
                        config["monitoring"]["keep_latest_checkpoints"],
                    )
                ),
            )
            if improved:
                atomic_torch_save(payload, best_path)
            atomic_json_dump(
                _status_payload(
                    stage=stage,
                    state="running",
                    epoch=epoch,
                    global_step=global_step,
                    metrics=final_metrics,
                    started_at=started_at,
                ),
                status_path,
            )
            if collapse_guard["triggered"]:
                stop_reason = "validation_collapse_guard_best_checkpoint_preserved"
                break
            if (
                epoch >= int(settings["minimum_epochs"])
                and patience >= int(settings["early_stop_patience"])
            ):
                stop_reason = "early_stopping_validation_plateau"
                break
        state = "completed"
    except BaseException as exc:
        atomic_json_dump(
            _status_payload(
                stage=stage,
                state="error",
                epoch=locals().get("epoch", start_epoch - 1),
                global_step=global_step,
                metrics={"error": repr(exc)},
                started_at=started_at,
            ),
            status_path,
        )
        wandb_tracker.update_summary(
            {"run/state": "error", "run/error": repr(exc)}
        )
        wandb_tracker.finish(exit_code=1)
        raise
    result = {
        "stage": stage,
        "state": state,
        "stop_reason": stop_reason,
        "gate_mode": gate_mode,
        "view_mode": view_mode,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": file_sha256(best_path),
        "math_checkpoint": str(math_checkpoint_path.resolve()),
        "stability_policy": stability_policy,
        "collapse_guard": final_metrics.get("collapse_guard", {}),
        "final_metrics": final_metrics,
    }
    atomic_json_dump(result, artifact_dir / "summary.json")
    atomic_json_dump(
        _status_payload(
            stage=stage,
            state=state,
            epoch=int(final_metrics.get("epoch", 0)),
            global_step=global_step,
            metrics=final_metrics,
            started_at=started_at,
        ),
        status_path,
    )
    wandb_tracker.update_summary(
        {
            "run/state": state,
            "run/stop_reason": stop_reason,
            "run/best_metric": best_metric,
            "run/final_epoch": int(final_metrics.get("epoch", 0)),
        }
    )
    wandb_tracker.finish()
    return result
