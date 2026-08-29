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
import torch.nn.functional as F
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


def _initialize_math_capacity_expansion(
    model: MathTower,
    source_state: dict[str, torch.Tensor],
    expansion: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Load a narrower-depth checkpoint into an identity-expanded tower.

    The existing embedding, trained Transformer blocks, norms, and heads are
    loaded byte-for-byte. Appended pre-norm blocks are made exact residual
    identities by zeroing their attention and feed-forward output projections.
    A deterministic forward comparison then proves that the expansion has not
    changed the source function before optimization starts.
    """

    method = str(expansion.get("method", ""))
    if method != "append_identity_transformer_blocks_v1":
        raise ValueError(f"unsupported math capacity expansion method: {method}")
    source_layers = int(expansion["source_layers"])
    target_layers = int(expansion["target_layers"])
    hidden_size = int(expansion["hidden_size"])
    tolerance = float(expansion.get("maximum_function_error", 1.0e-6))
    if source_layers < 1 or target_layers <= source_layers:
        raise ValueError("math capacity expansion must append at least one layer")
    if len(model.blocks) != target_layers:
        raise ValueError("configured math layer count differs from capacity contract")
    if model.hidden_size != hidden_size:
        raise ValueError("configured math width differs from capacity contract")

    observed_source_layers = 1 + max(
        (
            int(name.split(".")[1])
            for name in source_state
            if name.startswith("blocks.") and name.split(".")[1].isdigit()
        ),
        default=-1,
    )
    if observed_source_layers != source_layers:
        raise ValueError("source checkpoint layer count differs from capacity contract")
    source_embedding = source_state.get("token_embedding.weight")
    if source_embedding is None or int(source_embedding.shape[1]) != hidden_size:
        raise ValueError("source checkpoint width differs from capacity contract")

    source_config = dict(model.config)
    source_config["layers"] = source_layers
    source_model = MathTower(source_config, model.vocabulary_size).to(device)
    source_model.load_state_dict(source_state, strict=True)

    for block in model.blocks[source_layers:]:
        torch.nn.init.zeros_(block.self_attn.out_proj.weight)
        if block.self_attn.out_proj.bias is not None:
            torch.nn.init.zeros_(block.self_attn.out_proj.bias)
        torch.nn.init.zeros_(block.linear2.weight)
        if block.linear2.bias is not None:
            torch.nn.init.zeros_(block.linear2.bias)

    incompatible = model.load_state_dict(source_state, strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith("blocks.")
        and int(name.split(".")[1]) >= source_layers
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError("capacity expansion has unexpected missing checkpoint keys")
    if incompatible.unexpected_keys:
        raise RuntimeError("capacity expansion has unexpected source checkpoint keys")

    source_was_training = source_model.training
    target_was_training = model.training
    source_model.eval()
    model.eval()
    probe_ids = torch.tensor(
        [
            [1, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 2],
            [1, 9, 27, 45, 63, 81, 99, 117, 135, 153, 171, 2],
        ],
        dtype=torch.long,
        device=device,
    ) % model.vocabulary_size
    probe_mask = torch.ones_like(probe_ids, dtype=torch.bool)
    probe_prefix = torch.tensor([4, 5], dtype=torch.long, device=device)
    with torch.no_grad():
        source_output = source_model(probe_ids, probe_mask, probe_prefix)
        target_output = model(probe_ids, probe_mask, probe_prefix)
    logit_error = float(
        (source_output.logits - target_output.logits).abs().max().item()
    )
    hidden_error = float(
        (source_output.hidden_states - target_output.hidden_states).abs().max().item()
    )
    if source_was_training:
        source_model.train()
    if target_was_training:
        model.train()
    if max(logit_error, hidden_error) > tolerance:
        raise RuntimeError("identity capacity expansion changed the source function")

    source_parameters = sum(parameter.numel() for parameter in source_model.parameters())
    target_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_target_parameters = expansion.get("expected_target_parameters")
    if (
        expected_target_parameters is not None
        and target_parameters != int(expected_target_parameters)
    ):
        raise RuntimeError("expanded math parameter count differs from contract")
    return {
        "method": method,
        "source_layers": source_layers,
        "target_layers": target_layers,
        "added_layers": target_layers - source_layers,
        "hidden_size": hidden_size,
        "source_parameters": source_parameters,
        "target_parameters": target_parameters,
        "parameter_ratio": target_parameters / max(1, source_parameters),
        "maximum_function_error": tolerance,
        "observed_logit_error": logit_error,
        "observed_hidden_state_error": hidden_error,
        "identity_preserved": True,
    }


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
    if config.get("data", {}).get("full_supervision_root"):
        from .full_math_data import PARENT_SHA, audit_full_data
        root = Path(config["data"]["full_supervision_root"]).resolve()
        manifest = audit_full_data(
            root,
            expected_parent_manifest_sha256=config["data"].get(
                "full_supervision_parent_manifest_sha256", PARENT_SHA
            ),
        )
        if manifest["manifest_sha256"] != config["data"].get("full_supervision_sha256"):
            raise ValueError("full-supervision data identity differs from run config")
        return root, manifest
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
    if manifest.get("format") == "cftn_canonical_math_curriculum_v1":
        from .math_curriculum_data import audit_dataset

        audit_dataset(root)
        expected = config["data"].get("dataset_manifest_sha256")
        if expected and manifest.get("manifest_sha256") != expected:
            raise ValueError("canonical curriculum data identity differs from run config")
        return root, manifest
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
    if manifest.get("derivative_format") == "cftn_full_math_supervision_v1":
        from .full_math_data import read_rows
        return EquationDataset(read_rows(root / metadata["path"]))
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
    broad_curriculum_format = data_format in {
        "cftn_text_broad_math_v2",
        "cftn_canonical_math_curriculum_v1",
    }
    phase: dict[str, Any] | None = None
    if broad_curriculum_format:
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
    quota_groups = (
        phase.get("quota_groups")
        if broad_curriculum_format and phase is not None
        else None
    )
    if quota_groups is not None:
        if not isinstance(quota_groups, list) or not quota_groups:
            raise ValueError("curriculum quota_groups must be a non-empty list")
        names = [str(group.get("name", "")).strip() for group in quota_groups]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("curriculum quota group names must be unique and non-empty")
        requested_total = sum(int(group.get("examples", 0)) for group in quota_groups)
        if requested_total != target:
            raise ValueError("curriculum quota groups must sum to examples_per_epoch")

        rng = random.Random(int(seed) + int(epoch) * 1_000_003)
        epoch_records: list[dict[str, Any]] = []
        group_sampling: dict[str, dict[str, Any]] = {}
        eligible_record_ids: set[str] = set()
        for group, name in zip(quota_groups, names):
            requested = int(group["examples"])
            if requested <= 0:
                raise ValueError("curriculum quota group examples must be positive")
            filters = dict(group.get("filters", {}))
            filters.setdefault("name", name)
            pool = _filter_v2_records_for_phase(selected, filters)
            if not pool:
                raise RuntimeError(f"curriculum quota group {name} selected no records")
            pool_ids = {str(row.get("record_id", "")) for row in pool}
            if "" in pool_ids:
                raise ValueError("curriculum quota group rows require record_id")
            overlap = eligible_record_ids & pool_ids
            if overlap:
                raise ValueError(
                    f"curriculum quota groups overlap on {len(overlap)} records; "
                    "skill buckets must be disjoint"
                )
            eligible_record_ids.update(pool_ids)

            sampled: list[dict[str, Any]] = []
            replacement_examples = 0
            if bool(group.get("balance_families", True)):
                families: dict[str, list[dict[str, Any]]] = {}
                for record in pool:
                    families.setdefault(str(record.get("family", "unknown")), []).append(record)
                for index, family in enumerate(sorted(families)):
                    family_pool = families[family]
                    count = requested // len(families) + int(index < requested % len(families))
                    if bool(group.get("balance_operations_within_families", False)):
                        operations: dict[str, list[dict[str, Any]]] = {}
                        for record in family_pool:
                            operation = str(record.get("operation", "")).strip()
                            if not operation:
                                raise ValueError(
                                    "operation-balanced curriculum rows require operation"
                                )
                            operations.setdefault(operation, []).append(record)
                        for operation_index, operation in enumerate(sorted(operations)):
                            operation_pool = operations[operation]
                            operation_count = count // len(operations) + int(
                                operation_index < count % len(operations)
                            )
                            sampled.extend(
                                rng.sample(
                                    operation_pool,
                                    min(operation_count, len(operation_pool)),
                                )
                            )
                            extra = max(0, operation_count - len(operation_pool))
                            sampled.extend(
                                rng.choice(operation_pool) for _ in range(extra)
                            )
                            replacement_examples += extra
                    else:
                        sampled.extend(
                            rng.sample(family_pool, min(count, len(family_pool)))
                        )
                        extra = max(0, count - len(family_pool))
                        sampled.extend(rng.choice(family_pool) for _ in range(extra))
                        replacement_examples += extra
            elif requested <= len(pool):
                sampled = rng.sample(pool, requested)
            else:
                sampled = rng.sample(pool, len(pool))
                replacement_examples = requested - len(pool)
                sampled.extend(rng.choice(pool) for _ in range(replacement_examples))
            epoch_records.extend(sampled)
            group_sampling[name] = {
                "filters": filters,
                "available_examples": len(pool),
                "requested_examples": requested,
                "unique_examples": len({str(row["record_id"]) for row in sampled}),
                "replacement_examples": replacement_examples,
                "sampling_with_replacement": replacement_examples > 0,
                "family_counts": dict(sorted(Counter(str(row.get("family", "unknown")) for row in sampled).items())),
                "operation_counts": dict(
                    sorted(
                        Counter(
                            str(row.get("operation", "unknown")) for row in sampled
                        ).items()
                    )
                ),
            }
        rng.shuffle(epoch_records)
        metadata = {
            **metadata,
            "examples_this_epoch": len(epoch_records),
            "unique_examples_this_epoch": len({str(record["record_id"]) for record in epoch_records}),
            "sampling_policy": "quota_groups_v1",
            "sampling_with_replacement": any(
                details["sampling_with_replacement"] for details in group_sampling.values()
            ),
            "quota_groups": group_sampling,
            "sampled_source_counts": dict(sorted(Counter(str(row.get("source", "unknown")) for row in epoch_records).items())),
            "sampled_family_counts": dict(sorted(Counter(str(row.get("family", "unknown")) for row in epoch_records).items())),
            "sampled_operation_counts": dict(
                sorted(
                    Counter(
                        str(row.get("operation", "unknown"))
                        for row in epoch_records
                    ).items()
                )
            ),
        }
        return EquationDataset(epoch_records), metadata
    source_quotas = (
        phase.get("source_quotas")
        if broad_curriculum_format and phase is not None
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
            if phase.get("balance_families_within_source", False):
                families = {}
                for record in available:
                    families.setdefault(record["family"], []).append(record)
                sampled, replacement_examples = [], 0
                for index, family in enumerate(sorted(families)):
                    pool = families[family]
                    count = requested // len(families) + int(index < requested % len(families))
                    sampled.extend(rng.sample(pool, min(count, len(pool))))
                    extra = max(0, count - len(pool))
                    sampled.extend(rng.choice(pool) for _ in range(extra))
                    replacement_examples += extra
            elif requested <= len(available):
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
    configured_supervision = phase.get("supervision_kinds")
    configured_verification = phase.get("verifications")
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
    supervision_kinds = (
        {str(value) for value in configured_supervision}
        if configured_supervision is not None
        else None
    )
    verifications = (
        {str(value) for value in configured_verification}
        if configured_verification is not None
        else None
    )
    if sources is not None and not sources:
        raise ValueError("curriculum phase sources cannot be empty")
    if families is not None and not families:
        raise ValueError("curriculum phase families cannot be empty")
    if supervision_kinds is not None and not supervision_kinds:
        raise ValueError("curriculum supervision_kinds cannot be empty")
    if verifications is not None and not verifications:
        raise ValueError("curriculum verifications cannot be empty")
    maximum_difficulty = int(phase.get("max_difficulty", 3))
    return [
        record
        for record in records
        if int(record.get("difficulty", 3)) <= maximum_difficulty
        and (record.get("source") != "verified_school_full"
             or int(record["difficulty"]) <= int(phase.get("max_school_difficulty", 3)))
        and (sources is None or str(record.get("source", "unknown")) in sources)
        and (families is None or str(record.get("family", "unknown")) in families)
        and (
            supervision_kinds is None
            or str(record.get("supervision_kind", "unknown")) in supervision_kinds
        )
        and (
            verifications is None
            or str(record.get("verification", "unknown")) in verifications
        )
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

    filters = {
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
    }
    if phase.get("supervision_kinds") is not None:
        filters["supervision_kinds"] = sorted(
            str(value) for value in phase["supervision_kinds"]
        )
    if phase.get("verifications") is not None:
        filters["verifications"] = sorted(
            str(value) for value in phase["verifications"]
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
        "filters": filters,
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


def build_math_tower_for_checkpoint(
    config: dict[str, Any], checkpoint: dict[str, Any]
) -> MathTower:
    """Build the checkpoint-attested effective tower architecture."""

    effective = checkpoint.get("extra", {}).get("effective_math_tower")
    if effective is None:
        return build_math_tower(config)
    if not isinstance(effective, dict):
        raise ValueError("checkpoint effective math architecture is invalid")
    base = config["math_tower"]
    for key in set(base).union(effective) - {"layers"}:
        if effective.get(key) != base.get(key):
            raise ValueError(
                "checkpoint changes unsupported math architecture field " f"{key}"
            )
    expansion = (
        checkpoint.get("extra", {})
        .get("metrics", {})
        .get("source_checkpoint", {})
        .get("capacity_expansion")
    )
    if not isinstance(expansion, dict):
        raise ValueError("expanded math checkpoint lacks capacity attestation")
    if expansion.get("method") != "append_identity_transformer_blocks_v1":
        raise ValueError("expanded math checkpoint method is not recognized")
    if int(expansion.get("source_layers", -1)) != int(base["layers"]):
        raise ValueError("expanded math checkpoint source depth differs from config")
    if int(expansion.get("target_layers", -1)) != int(effective.get("layers", -2)):
        raise ValueError("expanded math checkpoint target depth is inconsistent")
    effective_config = copy.deepcopy(config)
    effective_config["math_tower"] = copy.deepcopy(effective)
    return build_math_tower(effective_config)


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
    math_tower = build_math_tower_for_checkpoint(config, checkpoint)
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


def generation_panel_specs_for_phase(
    panel_specs: list[dict[str, Any]],
    phase: dict[str, Any] | None,
    *,
    scope: str,
) -> list[dict[str, Any]]:
    """Return the configured generation panels required by this phase.

    ``phase_required_v1`` is deliberately an execution-time optimization, not
    an acceptance change: it runs the primary panel and every explicitly
    thresholded panel for the active phase.  Future diagnostic panels remain
    compulsory once their own phase becomes active and in final evaluation.
    """

    if scope == "all_configured_v1":
        return list(panel_specs)
    if scope != "phase_required_v1":
        raise ValueError(f"unknown generation panel scope: {scope}")
    if phase is None:
        raise ValueError("phase-required generation scope needs an active phase")

    required = {str(phase.get("primary_generation_panel", "validation"))}
    for key in (
        "minimum_generation_accuracy_by_panel",
        "minimum_valid_rate_by_panel",
    ):
        thresholds = phase.get(key, {})
        if not isinstance(thresholds, dict):
            raise ValueError(f"{key} must be an object")
        required.update(str(name) for name in thresholds)

    available = {str(spec.get("name")) for spec in panel_specs}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            "phase-required generation panels are unavailable: " + ", ".join(missing)
        )
    return [spec for spec in panel_specs if str(spec.get("name")) in required]


def _phase_generation_acceptance(
    *,
    phase: dict[str, Any] | None,
    generation_panels: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    epoch: int,
    phase_epoch: int | None = None,
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

    for dimension in ("source", "family", "difficulty", "operation"):
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

    for name, minimum in phase.get("minimum_trace_exact_by_family", {}).items():
        observed = primary.get("trace_exact_by_family", {}).get(name, {})
        add_check(f"primary_trace:{name}", observed.get("rate", 0.0), minimum)

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

    panel_family_thresholds = phase.get(
        "minimum_generation_accuracy_by_panel_family", {}
    )
    if not isinstance(panel_family_thresholds, dict):
        raise ValueError("minimum_generation_accuracy_by_panel_family must be an object")
    for panel_name, thresholds in panel_family_thresholds.items():
        if not isinstance(thresholds, dict):
            raise ValueError("panel-family acceptance thresholds must be objects")
        panel = generation_panels.get(str(panel_name))
        if panel is None:
            checks[f"panel_family:{panel_name}:available"] = {
                "observed": None,
                "minimum": 1.0,
                "pass": False,
            }
            continue
        observed_families = panel.get("by_family", {})
        for family, minimum in thresholds.items():
            observed = observed_families.get(str(family))
            key = f"panel_family:{panel_name}:{family}"
            if observed is None:
                checks[key] = {
                    "observed": None,
                    "minimum": float(minimum),
                    "examples": 0,
                    "pass": False,
                }
            else:
                add_check(key, observed.get("accuracy", 0.0), float(minimum))
                checks[key]["examples"] = int(observed.get("examples", 0))

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

    terminal_epoch = (
        int(phase_epoch) >= int(phase["maximum_epochs"])
        if phase_epoch is not None and "maximum_epochs" in phase
        else epoch == int(phase["through_epoch"])
    )
    return {
        "phase": str(phase["name"]),
        "primary_panel": primary_name,
        "minimum_generation_accuracy": minimum_accuracy,
        "minimum_valid_rate": minimum_valid_rate,
        "generation_accuracy": float(primary.get("accuracy", 0.0)),
        "valid_rate": float(primary.get("valid_rate", 0.0)),
        "phase_epoch": phase_epoch,
        "terminal_epoch": terminal_epoch,
        "checks": checks,
        "pass": bool(checks) and all(bool(check["pass"]) for check in checks.values()),
    }


def _checked_competency_curriculum(
    contract: dict[str, Any], phases: list[dict[str, Any]]
) -> bool:
    enabled = contract.get("curriculum", {}).get("transition_policy") in {
        "competency_gated_v1",
        "competency_gated_v2",
        "competency_gated_v3",
    }
    if not enabled:
        return False
    if not phases:
        raise ValueError("competency curriculum requires phases")
    total_maximum = 0
    for phase in phases:
        minimum = int(phase.get("minimum_epochs", 0))
        maximum = int(phase.get("maximum_epochs", 0))
        consecutive = int(phase.get("advance_after_consecutive_passes", 0))
        if minimum < 1 or maximum < minimum or consecutive < 1:
            raise ValueError("invalid competency curriculum phase bounds")
        total_maximum += maximum
    if total_maximum > int(contract["math_training"]["max_epochs"]):
        raise ValueError("competency phase maxima exceed global max_epochs")
    return True


def _initial_competency_curriculum_state() -> dict[str, Any]:
    return {"phase_index": 0, "phase_epoch": 0, "consecutive_passes": 0}


def _zero_update_phase_skip_state(
    *,
    phases: list[dict[str, Any]],
    state: dict[str, Any],
    accepted: bool,
) -> tuple[dict[str, Any], bool]:
    """Skip an already-mastered non-final phase without consuming an update."""

    index = int(state["phase_index"])
    if not 0 <= index < len(phases):
        raise ValueError("competency curriculum phase index is invalid")
    skipped = bool(accepted) and index < len(phases) - 1
    if not skipped:
        return dict(state), False
    return {
        "phase_index": index + 1,
        "phase_epoch": 0,
        "consecutive_passes": 0,
    }, True


def _update_competency_curriculum_state(
    *,
    phases: list[dict[str, Any]],
    state: dict[str, Any],
    accepted: bool,
    policy: str = "competency_gated_v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance only after repeated generation passes; fail closed at a phase cap."""

    index = int(state["phase_index"])
    if not 0 <= index < len(phases):
        raise ValueError("competency curriculum phase index is invalid")
    phase = phases[index]
    phase_epoch = int(state["phase_epoch"]) + 1
    minimum = int(phase["minimum_epochs"])
    maximum = int(phase["maximum_epochs"])
    required = int(phase["advance_after_consecutive_passes"])
    if phase_epoch > maximum:
        raise RuntimeError("competency curriculum advanced beyond its phase cap")
    consecutive = (
        int(state["consecutive_passes"]) + 1
        if accepted and phase_epoch >= minimum
        else 0
    )
    passed = consecutive >= required
    final_phase = index == len(phases) - 1
    advance = passed and not final_phase
    complete = passed and final_phase and bool(phase.get("stop_on_pass", False))
    failed = phase_epoch >= maximum and not (advance or complete)
    next_state = {
        "phase_index": index + 1 if advance else index,
        "phase_epoch": 0 if advance else phase_epoch,
        "consecutive_passes": 0 if advance else consecutive,
    }
    transition = {
        "policy": str(policy),
        "phase": str(phase["name"]),
        "phase_index": index,
        "phase_epoch": phase_epoch,
        "minimum_epochs": minimum,
        "maximum_epochs": maximum,
        "consecutive_passes": consecutive,
        "required_consecutive_passes": required,
        "advance": advance,
        "complete": complete,
        "failed": failed,
        "next_phase": str(phases[index + 1]["name"]) if advance else None,
    }
    return next_state, transition


def _teacher_preservation_kl(
    current_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    selected_rows: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Preserve only rows whose complete target the frozen source already knew.

    This prevents distillation from canonising an incorrect source answer.  The
    teacher is only a behavioural retention constraint on exact source-correct
    sequences; ordinary supervised loss remains responsible for improvement.
    """

    targets = labels[:, 1:]
    valid = targets.ne(-100)
    teacher_predictions = teacher_logits[:, :-1].argmax(dim=-1)
    teacher_correct = (teacher_predictions.eq(targets) | ~valid).all(dim=1)
    rows = selected_rows.bool() & teacher_correct & valid.any(dim=1)
    token_mask = valid & rows.unsqueeze(1)
    if not bool(token_mask.any()):
        return current_logits.sum() * 0.0, 0
    current = current_logits[:, :-1][token_mask].float()
    teacher = teacher_logits[:, :-1][token_mask].detach().float()
    return (
        F.kl_div(
            F.log_softmax(current, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="batchmean",
        ),
        int(rows.sum()),
    )


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
    competency_curriculum_enabled = _checked_competency_curriculum(
        contract, phases
    )
    competency_curriculum_state = _initial_competency_curriculum_state()
    tokenizer = ByteMathTokenizer()
    collator_class = MathCollator
    if settings.get("objective") == "computation_roles_v1":
        from .full_math_data import FullMathCollator
        collator_class = FullMathCollator
    collator = collator_class(
        tokenizer,
        int(config["data"]["max_math_length"]),
        target_mode=str(settings.get("target_mode", "full_trace_v1")),
        input_view=str(settings.get("input_view", SHARED_MATH_INPUT_VIEW)),
    )
    early_stopping_enabled = not bool(disable_early_stopping)
    validation_collator = MathCollator(tokenizer, int(config["data"]["max_math_length"]),
                                      target_mode=collator.target_mode, input_view=collator.input_view)
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
    effective_config = copy.deepcopy(config)
    capacity_expansion = contract.get("capacity_expansion")
    if capacity_expansion is not None:
        if int(capacity_expansion["source_layers"]) != int(
            config["math_tower"]["layers"]
        ):
            raise ValueError("capacity source depth differs from base math config")
        if int(capacity_expansion["hidden_size"]) != int(
            config["math_tower"]["hidden_size"]
        ):
            raise ValueError("capacity width differs from base math config")
        effective_config["math_tower"]["layers"] = int(
            capacity_expansion["target_layers"]
        )
    model = build_math_tower(effective_config).to(device)
    phase_local_optimization = dict(
        contract.get("curriculum", {}).get("phase_local_optimization", {})
    )
    phase_local_enabled = bool(phase_local_optimization.get("enabled", False))
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
    phase_maximum_epochs = max(
        (int(phase.get("maximum_epochs", 1)) for phase in phases),
        default=int(settings["max_epochs"]),
    )
    total_steps = (
        phase_maximum_epochs * steps_per_epoch
        if phase_local_enabled
        else int(settings["max_epochs"]) * steps_per_epoch
    )
    scheduler_warmup_fraction = (
        float(phase_local_optimization.get("warmup_epochs", 0))
        / max(1, phase_maximum_epochs)
        if phase_local_enabled
        else float(settings["warmup_fraction"])
    )
    scheduler_minimum_learning_rate = (
        float(phase_local_optimization["minimum_learning_rate"])
        if phase_local_enabled
        else float(settings["minimum_learning_rate"])
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=scheduler_warmup_fraction,
        minimum_ratio=scheduler_minimum_learning_rate
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
    optimizer_phase_name: str | None = None
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
        expansion_attestation = (
            _initialize_math_capacity_expansion(
                model,
                checkpoint["model_state"],
                dict(capacity_expansion),
                device=device,
            )
            if capacity_expansion is not None
            else None
        )
        if expansion_attestation is None:
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
        if expansion_attestation is not None:
            source_provenance["capacity_expansion"] = expansion_attestation
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
        if competency_curriculum_enabled:
            saved_curriculum_state = checkpoint.get("extra", {}).get(
                "competency_curriculum_state"
            )
            if not isinstance(saved_curriculum_state, dict):
                raise ValueError(
                    "competency resume checkpoint lacks curriculum state"
                )
            competency_curriculum_state = {
                "phase_index": int(saved_curriculum_state["phase_index"]),
                "phase_epoch": int(saved_curriculum_state["phase_epoch"]),
                "consecutive_passes": int(
                    saved_curriculum_state["consecutive_passes"]
                ),
            }
            optimizer_phase_name = checkpoint.get("extra", {}).get(
                "optimizer_phase_name"
            )
    preservation_settings = dict(contract.get("preservation_distillation") or {})
    preservation_teacher: MathTower | None = None
    preservation_sources = {
        str(source) for source in preservation_settings.get("sources", [])
    }
    preservation_weight = float(preservation_settings.get("weight", 0.0))
    if bool(preservation_settings.get("enabled", False)):
        if not preservation_sources or not 0.0 < preservation_weight <= 1.0:
            raise ValueError("invalid preservation distillation settings")
        teacher_path = Path(contract["source_checkpoint"]).expanduser().resolve()
        teacher_checkpoint, _ = _load_math_initialization_checkpoint(
            teacher_path,
            expected_sha256=contract.get("source_checkpoint_sha256"),
            map_location=device,
        )
        preservation_teacher = build_math_tower(effective_config).to(device)
        preservation_teacher.load_state_dict(
            teacher_checkpoint["model_state"], strict=True
        )
        preservation_teacher.requires_grad_(False)
        preservation_teacher.eval()
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
            "model": effective_config["math_tower"],
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
    generation_settings = generation_validation_settings

    def evaluate_generation_panels_for_phase(
        phase: dict[str, Any] | None,
        *,
        suffix: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
        generation_panels: dict[str, dict[str, Any]] = {}
        if not (
            manifest.get("format") in {
                "cftn_text_broad_math_v2",
                "cftn_canonical_math_curriculum_v1",
            }
            and bool(generation_settings.get("enabled", False))
        ):
            return generation_panels, None
        panel_specs = configured_generation_panels or [
            {
                "name": "validation",
                "split": "validation",
                "phase_filtered": True,
            }
        ]
        panel_specs = generation_panel_specs_for_phase(
            panel_specs,
            phase,
            scope=str(generation_settings.get("panel_scope", "all_configured_v1")),
        )
        for panel_spec in panel_specs:
            panel_name = str(panel_spec["name"])
            panel_dataset = generation_panel_datasets.get(
                panel_name, validation_dataset
            )
            panel_records = panel_dataset.records
            if bool(panel_spec.get("phase_filtered", False)) and phase is not None:
                panel_records = _filter_v2_records_for_phase(panel_records, phase)
            panel_filters = panel_spec.get("filters")
            if panel_filters is not None:
                if not isinstance(panel_filters, dict):
                    raise ValueError(
                        f"generation validation panel {panel_name} filters must be an object"
                    )
                panel_records = _filter_v2_records_for_phase(
                    panel_records,
                    {"name": panel_name, **panel_filters},
                )
            if not panel_records:
                raise RuntimeError(
                    f"generation validation panel {panel_name} selected no records"
                )
            generation_rows_name = (
                f"generation_validation_{panel_name}_{suffix}.jsonl"
                if configured_generation_panels
                else f"generation_validation_{suffix}.jsonl"
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
                        "batch_size", generation_settings.get("batch_size", 16)
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
                require_eos=bool(generation_settings.get("require_eos", False)),
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
            (phase or {}).get("primary_generation_panel", panel_specs[0]["name"])
        )
        primary = generation_panels.get(primary_panel_name)
        if primary is None:
            raise RuntimeError(
                "primary generation validation panel is unavailable: "
                + primary_panel_name
            )
        return generation_panels, primary

    try:
        retention_baseline = None
        if contract.get("retention_baseline"):
            baseline_path = artifact_dir / "retention_baseline.json"
            if resume:
                retention_baseline = json.loads(baseline_path.read_text())
            else:
                model.eval()
                write_status(_status_payload(stage="math", state="evaluating_baseline", epoch=0,
                             global_step=global_step, started_at=started_at))
                retention_baseline = evaluate_generation_panel(
                    model, tokenizer, validation_dataset.records,
                    maximum_examples=int(contract["retention_baseline"]["examples"]),
                    batch_size=int(generation_validation_settings["batch_size"]),
                    max_new_tokens=int(generation_validation_settings["max_new_tokens"]),
                    failure_examples=8, rows_path=work_dir / "retention_baseline_rows.jsonl",
                    input_view=collator.input_view,
                    require_eos=bool(generation_validation_settings.get("require_eos", False)))
                atomic_json_dump(retention_baseline, baseline_path)
                if work_dir != artifact_dir:
                    atomic_copy_file(work_dir / "retention_baseline_rows.jsonl", artifact_dir / "retention_baseline_rows.jsonl")
        entrance_evaluations: list[dict[str, Any]] = []
        entrance_settings = dict(contract.get("zero_update_entrance") or {})
        if (
            competency_curriculum_enabled
            and not resume
            and bool(entrance_settings.get("enabled", False))
        ):
            model.eval()
            validation_loader = make_loader(
                validation_dataset,
                validation_collator,
                batch_size=int(settings["eval_batch_size"]),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=int(settings["num_workers"]),
            )
            entrance_teacher_forced = evaluate_math_tower(
                model,
                validation_loader,
                device,
                dtype,
                float(settings["answer_head_weight"]),
                max_batches=max_batches,
            )
            maximum_skips = min(
                int(entrance_settings.get("maximum_skipped_phases", 0)),
                max(0, len(phases) - 1),
            )
            while (
                len(entrance_evaluations) < maximum_skips
                and int(competency_curriculum_state["phase_index"])
                < len(phases) - 1
            ):
                phase_index = int(competency_curriculum_state["phase_index"])
                phase = phases[phase_index]
                phase_name = str(phase["name"])
                write_status(
                    _status_payload(
                        stage="math",
                        state="evaluating_entrance",
                        epoch=0,
                        global_step=global_step,
                        metrics={"phase": phase_name, "phase_index": phase_index},
                        started_at=started_at,
                    )
                )
                entrance_panels, entrance_primary = (
                    evaluate_generation_panels_for_phase(
                        phase,
                        suffix=f"entrance_{phase_index:02d}_{phase_name}",
                    )
                )
                entrance_validation = copy.deepcopy(entrance_teacher_forced)
                if entrance_primary is not None:
                    entrance_validation["generation"] = entrance_primary
                    entrance_validation["generation_panels"] = entrance_panels
                entrance_acceptance = _phase_generation_acceptance(
                    phase=phase,
                    generation_panels=entrance_panels,
                    validation=entrance_validation,
                    epoch=0,
                    phase_epoch=0,
                )
                if retention_baseline is not None:
                    observed = entrance_panels["broad"]
                    minimum = max(
                        0.0,
                        float(retention_baseline["accuracy"])
                        - float(contract["retention_baseline"]["maximum_drop"]),
                    )
                    entrance_acceptance["checks"]["broad_retention"] = {
                        "observed": observed["accuracy"],
                        "minimum": minimum,
                        "pass": observed["accuracy"] >= minimum,
                    }
                    entrance_acceptance["pass"] = all(
                        bool(check["pass"])
                        for check in entrance_acceptance["checks"].values()
                    )
                next_entrance_state, skipped = _zero_update_phase_skip_state(
                    phases=phases,
                    state=competency_curriculum_state,
                    accepted=bool(entrance_acceptance["pass"]),
                )
                entrance_report = {
                    "phase": phase_name,
                    "phase_index": phase_index,
                    "zero_optimizer_updates": True,
                    "skipped": skipped,
                    "acceptance": entrance_acceptance,
                    "validation": entrance_validation,
                }
                entrance_evaluations.append(entrance_report)
                wandb_tracker.log(
                    {
                        "entrance": {
                            "phase": phase_name,
                            "phase_index": phase_index,
                            "pass": entrance_acceptance["pass"],
                            "skipped": skipped,
                            "generation_accuracy": entrance_acceptance[
                                "generation_accuracy"
                            ],
                            "valid_rate": entrance_acceptance["valid_rate"],
                        }
                    },
                    global_step=global_step,
                    epoch=0,
                    event="zero_update_entrance",
                )
                if not skipped:
                    break
                competency_curriculum_state = next_entrance_state
            entrance_path = work_dir / "entrance_evaluations.json"
            atomic_json_dump(
                {
                    "format": "cftn_math_zero_update_entrance_v1",
                    "evaluations": entrance_evaluations,
                    "resulting_curriculum_state": competency_curriculum_state,
                },
                entrance_path,
            )
            if entrance_path != artifact_dir / entrance_path.name:
                atomic_copy_file(entrance_path, artifact_dir / entrance_path.name)
        for epoch in range(start_epoch, int(settings["max_epochs"]) + 1):
            model.train()
            if competency_curriculum_enabled:
                phase = phases[int(competency_curriculum_state["phase_index"])]
                phase_epoch = int(competency_curriculum_state["phase_epoch"]) + 1
            else:
                phase = next(
                    (
                        item
                        for item in phases
                        if epoch <= int(item["through_epoch"])
                    ),
                    phases[-1] if phases else None,
                )
                phase_epoch = None
            phase_name = str(phase["name"]) if phase is not None else None
            phase_optimizer_reset = False
            if phase_local_enabled and optimizer_phase_name != phase_name:
                if optimizer_phase_name is not None:
                    optimizer = AdamW(
                        model.parameters(),
                        lr=float(settings["learning_rate"]),
                        weight_decay=float(settings["weight_decay"]),
                    )
                    scheduler = make_scheduler(
                        optimizer,
                        total_steps=total_steps,
                        warmup_fraction=scheduler_warmup_fraction,
                        minimum_ratio=scheduler_minimum_learning_rate
                        / float(settings["learning_rate"]),
                    )
                    scaler = make_scaler(device, dtype)
                    phase_optimizer_reset = True
                optimizer_phase_name = phase_name
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
            train_task_loss = 0.0
            train_preservation_loss = 0.0
            preserved_rows = 0
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
                    if settings.get("objective") == "computation_roles_v1":
                        from .computation_supervision import computation_loss
                        lm_loss = computation_loss(output.logits, batch["math_labels"], batch["math_roles"],
                                                   weights=tuple(settings["role_weights"]))
                    else:
                        lm_loss = answer_weighted_causal_language_loss(
                            output.logits, batch["math_labels"], batch["math_answer_labels"],
                            answer_weight=float(settings.get("answer_token_weight", 1.0)))
                    classes = model.answer_classes(batch["answer_values"])
                    answer_loss = optional_answer_loss(output.answer_logits, classes)
                    task_loss = lm_loss + float(settings["answer_head_weight"]) * answer_loss
                    preservation_loss = output.logits.sum() * 0.0
                    batch_preserved_rows = 0
                    if preservation_teacher is not None:
                        with torch.no_grad():
                            teacher_output = preservation_teacher(
                                batch["math_input_ids"],
                                batch["math_attention_mask"],
                                batch["math_prefix_lengths"],
                            )
                        source_mask = torch.tensor(
                            [
                                str(record.get("source")) in preservation_sources
                                for record in batch["records"]
                            ],
                            dtype=torch.bool,
                            device=device,
                        )
                        preservation_loss, batch_preserved_rows = (
                            _teacher_preservation_kl(
                                output.logits,
                                teacher_output.logits,
                                batch["math_labels"],
                                source_mask,
                            )
                        )
                    loss = task_loss + preservation_weight * preservation_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite math training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(settings["gradient_clip"]), error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                batch_size = batch["math_input_ids"].shape[0]
                train_loss += float(loss.detach()) * batch_size
                train_task_loss += float(task_loss.detach()) * batch_size
                train_preservation_loss += (
                    float(preservation_loss.detach()) * batch_size
                )
                preserved_rows += batch_preserved_rows
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
                    progress_metrics.update(
                        task_loss_so_far=train_task_loss
                        / max(1, trained_examples),
                        preservation_loss_so_far=train_preservation_loss
                        / max(1, trained_examples),
                        preservation_rows=preserved_rows,
                        preservation_weight=preservation_weight,
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
                validation_collator,
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
            generation_validation: dict[str, Any] | None = None
            generation_panels: dict[str, dict[str, Any]] = {}
            if (
                manifest.get("format") in {
                    "cftn_text_broad_math_v2",
                    "cftn_canonical_math_curriculum_v1",
                }
                and bool(generation_settings.get("enabled", False))
                and epoch % max(1, int(generation_settings.get("every_epochs", 1)))
                == 0
            ):
                generation_panels, generation_validation = (
                    evaluate_generation_panels_for_phase(
                        phase,
                        suffix=f"epoch_{epoch:04d}",
                    )
                )
                validation["generation"] = generation_validation
                validation["generation_panels"] = generation_panels
                validation["generation_panel_scope"] = str(
                    generation_settings.get("panel_scope", "all_configured_v1")
                )
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
                manifest.get("format") in {
                    "cftn_text_broad_math_v2",
                    "cftn_canonical_math_curriculum_v1",
                }
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
                phase_epoch=phase_epoch,
            )
            if retention_baseline is not None:
                observed = generation_panels["broad"]
                minimum = max(0.0, float(retention_baseline["accuracy"]) - float(contract["retention_baseline"]["maximum_drop"]))
                phase_acceptance["checks"]["broad_retention"] = {
                    "observed": observed["accuracy"], "minimum": minimum,
                    "pass": observed["accuracy"] >= minimum}
                phase_acceptance["pass"] = all(c["pass"] for c in phase_acceptance["checks"].values())
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
            if contract.get("promote_final_phase_only"):
                checkpoint_eligible = checkpoint_eligible and phase is phases[-1]
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
            checkpoint_curriculum_state = competency_curriculum_state
            curriculum_transition = None
            if competency_curriculum_enabled:
                checkpoint_curriculum_state, curriculum_transition = (
                    _update_competency_curriculum_state(
                        phases=phases,
                        state=competency_curriculum_state,
                        accepted=bool(
                            phase_acceptance is not None
                            and phase_acceptance["pass"]
                        ),
                        policy=str(
                            contract.get("curriculum", {}).get(
                                "transition_policy", "competency_gated_v1"
                            )
                        ),
                    )
                )
                phase_gate = (
                    phase_acceptance
                    if any(
                        bool(curriculum_transition[key])
                        for key in ("advance", "complete", "failed")
                    )
                    else None
                )
            if contract.get("promote_final_phase_only"):
                # Compare accepted candidates, not scores from failed epochs.
                promote_best = checkpoint_eligible and (
                    best_checkpoint_metric is None or selection_metric > best_checkpoint_metric)
            final_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss / max(1, trained_examples),
                "train_task_loss": train_task_loss / max(1, trained_examples),
                "train_preservation_loss": train_preservation_loss
                / max(1, trained_examples),
                "preservation_rows": preserved_rows,
                "preservation_weight": preservation_weight,
                "preservation_sources": sorted(preservation_sources),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "phase_local_optimization": {
                    "enabled": phase_local_enabled,
                    "optimizer_reset_this_epoch": phase_optimizer_reset,
                    "optimizer_phase": optimizer_phase_name,
                    "warmup_epochs": phase_local_optimization.get("warmup_epochs"),
                    "minimum_learning_rate": scheduler_minimum_learning_rate,
                },
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
                "objective": settings.get("objective", "answer_weighted_causal_language_loss"),
                "role_weights": settings.get("role_weights"),
                "answer_token_weight": float(
                    settings.get("answer_token_weight", 1.0)
                ),
                "require_acceptance_for_best": require_acceptance_for_best,
                "checkpoint_eligible": checkpoint_eligible,
                "checkpoint_promoted": promote_best,
                "curriculum_acceptance": phase_acceptance,
                "curriculum_gate": phase_gate,
                "curriculum_transition": curriculum_transition,
                "competency_curriculum_state": (
                    checkpoint_curriculum_state
                    if competency_curriculum_enabled
                    else None
                ),
                "zero_update_entrance": [
                    {
                        "phase": item["phase"],
                        "phase_index": item["phase_index"],
                        "skipped": item["skipped"],
                        "pass": item["acceptance"]["pass"],
                    }
                    for item in entrance_evaluations
                ],
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
                extra={
                    "metrics": final_metrics,
                    "effective_math_tower": effective_config["math_tower"],
                    "competency_curriculum_state": (
                        checkpoint_curriculum_state
                        if competency_curriculum_enabled
                        else None
                    ),
                    "optimizer_phase_name": optimizer_phase_name,
                },
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
                        "keep_working_checkpoints",
                        settings.get("keep_latest_checkpoints", config["monitoring"]["keep_latest_checkpoints"]),
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
            if competency_curriculum_enabled:
                competency_curriculum_state = checkpoint_curriculum_state
                if bool(curriculum_transition["complete"]):
                    stop_reason = f"curriculum_gate_passed_{phase['name']}"
                    state = "completed"
                    break
                if bool(curriculum_transition["failed"]):
                    stop_reason = f"curriculum_gate_failed_{phase['name']}"
                    state = "failed_acceptance"
                    break
            elif (
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
        "effective_math_tower": effective_config["math_tower"],
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
