from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW

from .checkpoint import (
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
from .complementary import complementary_record
from .config import _expand_environment, canonical_json, config_sha256, load_config
from .data_generator import file_sha256
from .dataset import CFTNCollator, EquationDataset
from .metrics import extract_answer, masked_token_statistics, summarize_gate
from .tokenizer import ByteMathTokenizer
from .training import (
    _bridge_collapse_diagnostics,
    _bridge_stability_policy,
    _status_payload,
    autocast_context,
    build_cftn_model,
    load_data_contract,
    make_loader,
    make_scaler,
    make_scheduler,
    move_batch,
    precision_dtype,
    resolve_device,
    seed_everything,
    split_dataset,
)
from .wandb_support import initialize_wandb


REVISION_FORMAT = "cftn_text_conditional_bridge_revision_v1_2"
V2_REVISION_FORMAT = "cftn_text_conditional_bridge_revision_v2"
REQUIRED_REQUIREMENT = "required"
REDUNDANT_REQUIREMENT = "redundant"
BRIDGE_PREFIXES = (
    "gpt_to_math.",
    "math_receivers.",
    "math_to_gpt.",
    "gpt_tower.receivers.",
)


def _revision_sha256(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if key != "_meta"}
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def _resolve_revision_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository_root / path).resolve()


def load_revision_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("conditional-bridge revision root must be a mapping")
    config = _expand_environment(raw)
    revision_format = config.get("format")
    if revision_format not in {REVISION_FORMAT, V2_REVISION_FORMAT}:
        raise ValueError(f"unsupported conditional-bridge revision format: {revision_format}")
    for section in ("revision", "paths", "training", "validation", "acceptance"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"conditional-bridge revision requires {section}")

    repository_root = config_path.parent.parent
    required_paths = (
        (
            "base_config",
            "v1_1_artifact_root",
            "artifact_root",
            "math_checkpoint",
            "source_bridge_checkpoint",
            "synergy_protocol",
        )
        if revision_format == REVISION_FORMAT
        else (
            "base_config",
            "artifact_root",
            "math_checkpoint",
            "source_bridge_checkpoint",
            "mechanism_prerequisite_report",
            "specialist_report",
        )
    )
    for key in required_paths:
        if key not in config["paths"]:
            raise ValueError(f"conditional-bridge paths.{key} is required")
        config["paths"][key] = str(
            _resolve_revision_path(config["paths"][key], repository_root)
        )

    training = config["training"]
    positive = (
        "base_train_examples",
        "base_validation_examples",
        "batch_size",
        "eval_batch_size",
        "max_epochs",
        "minimum_epochs",
        "early_stop_patience",
        "learning_rate",
        "minimum_learning_rate",
        "preservation_weight",
        "contrastive_weight",
        "contrastive_margin",
        "redundant_gate_weight",
        "gradient_clip",
    )
    for key in positive:
        if float(training.get(key, 0)) <= 0:
            raise ValueError(f"conditional-bridge training.{key} must be positive")
    if int(training["minimum_epochs"]) > int(training["max_epochs"]):
        raise ValueError("conditional-bridge minimum_epochs exceeds max_epochs")
    gate_multiplier = float(training.get("gate_learning_rate_multiplier", 0.25))
    if not 0 < gate_multiplier <= 1:
        raise ValueError("gate_learning_rate_multiplier must be within (0, 1]")
    if str(training.get("precision", "bf16")).lower() not in {
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
        "fp32",
        "float32",
        "none",
    }:
        raise ValueError("unsupported conditional-bridge precision")

    validation = config["validation"]
    if int(validation.get("generation_examples_per_view", 0)) < 2:
        raise ValueError("generation panel needs at least two examples per view")
    if int(validation.get("generation_batch_size", 0)) < 2:
        raise ValueError("generation batch size must be at least two")

    for key, value in config["acceptance"].items():
        if not 0 <= float(value) <= 1:
            raise ValueError(f"conditional-bridge acceptance.{key} must be within [0, 1]")

    config["_meta"] = {
        "path": str(config_path),
        "repository_root": str(repository_root.resolve()),
        "sha256": _revision_sha256(config),
    }
    return config


def build_mixed_necessity_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Pair each selected target with required and redundant communication views."""

    if limit < 1:
        raise ValueError("mixed-necessity record limit must be positive")
    if not records:
        raise ValueError("cannot build mixed-necessity records from an empty split")
    indices = list(range(len(records)))
    random.Random(int(seed)).shuffle(indices)
    selected = [records[index] for index in indices[: min(limit, len(indices))]]
    result: list[dict[str, Any]] = []
    for index, source in enumerate(selected):
        shared = dict(source)
        shared.update(
            {
                "shared_problem": str(source["problem"]),
                "view_mode": "shared",
                "communication_requirement": REDUNDANT_REQUIREMENT,
                "necessity_pair_index": index,
            }
        )
        shared.pop("gpt_problem", None)
        shared.pop("math_problem", None)
        required = complementary_record(
            source,
            seed=int(seed),
            assignment_key=str(source.get("record_id", index)),
            add_distractor=False,
        )
        required.update(
            {
                "communication_requirement": REQUIRED_REQUIREMENT,
                "necessity_pair_index": index,
            }
        )
        result.extend((shared, required))
    return result


def per_example_causal_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels have incompatible shapes")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shifted_labels.shape)
    return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def sequence_correct_mask(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    predictions = logits[:, :-1].argmax(dim=-1)
    correct = predictions.eq(shifted_labels) | ~valid
    return correct.all(dim=1) & valid.any(dim=1)


def specialist_preservation_kl(
    current_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    labels: torch.Tensor,
    row_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Distil only frozen-baseline-correct redundant rows."""

    baseline_correct = sequence_correct_mask(baseline_logits, labels)
    selected_rows = row_mask.bool() & baseline_correct
    shifted_labels = labels[:, 1:]
    token_mask = shifted_labels.ne(-100) & selected_rows.unsqueeze(1)
    if not bool(token_mask.any()):
        return current_logits.sum() * 0.0, 0
    current = current_logits[:, :-1][token_mask].float()
    baseline = baseline_logits[:, :-1][token_mask].detach().float()
    loss = F.kl_div(
        F.log_softmax(current, dim=-1),
        F.softmax(baseline, dim=-1),
        reduction="batchmean",
    )
    return loss, int(selected_rows.sum())


def _selected_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return value.sum() * 0.0
    return value[mask].float().mean()


def redundant_gate_loss(output: Any, redundant_mask: torch.Tensor) -> torch.Tensor:
    terms = [_selected_mean(output.gpt_to_math.gate, redundant_mask)]
    terms.extend(
        _selected_mean(gate, redundant_mask)
        for gate in output.math_receiver_gates.values()
    )
    return torch.stack(terms).mean()


def _requirement_masks(
    records: list[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    requirements = [record.get("communication_requirement") for record in records]
    unknown = sorted(
        {value for value in requirements if value not in {REQUIRED_REQUIREMENT, REDUNDANT_REQUIREMENT}}
    )
    if unknown:
        raise ValueError(f"unknown communication requirements: {unknown}")
    required = torch.tensor(
        [value == REQUIRED_REQUIREMENT for value in requirements],
        dtype=torch.bool,
        device=device,
    )
    return required, ~required


def conditional_objective(
    model: Any,
    batch: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[Any, dict[str, torch.Tensor | float | int]]:
    required, redundant = _requirement_masks(
        batch["records"], batch["math_input_ids"].device
    )
    output = model(
        batch,
        gpt_to_math_enabled=True,
        math_to_gpt_enabled=True,
        math_loss_weight=float(settings.get("math_loss_weight", 1.0)),
        gpt_loss_weight=float(settings.get("gpt_loss_weight", 1.0)),
        answer_head_weight=float(settings.get("answer_head_weight", 0.0)),
    )
    with torch.no_grad():
        disabled = model(
            batch,
            gpt_to_math_enabled=False,
            math_to_gpt_enabled=False,
            math_loss_weight=1.0,
            gpt_loss_weight=0.0,
            answer_head_weight=0.0,
        )
        shuffled = model(
            batch,
            gpt_to_math_enabled=True,
            math_to_gpt_enabled=False,
            shuffle_gpt_to_math=True,
            math_loss_weight=1.0,
            gpt_loss_weight=0.0,
            answer_head_weight=0.0,
        )

    current_per_row = per_example_causal_loss(
        output.math_output.logits, batch["math_labels"]
    )
    disabled_per_row = per_example_causal_loss(
        disabled.math_output.logits, batch["math_labels"]
    )
    shuffled_per_row = per_example_causal_loss(
        shuffled.math_output.logits, batch["math_labels"]
    )
    if bool(required.any()):
        strongest_control = torch.minimum(disabled_per_row, shuffled_per_row).detach()
        contrastive = F.relu(
            float(settings["contrastive_margin"])
            + current_per_row[required]
            - strongest_control[required]
        ).mean()
    else:
        contrastive = output.loss * 0.0
    preservation, preserved_rows = specialist_preservation_kl(
        output.math_output.logits,
        disabled.math_output.logits,
        batch["math_labels"],
        redundant,
    )
    neutral = redundant_gate_loss(output, redundant)
    total = (
        output.loss
        + float(settings["preservation_weight"]) * preservation
        + float(settings["contrastive_weight"]) * contrastive
        + float(settings["redundant_gate_weight"]) * neutral
    )
    return output, {
        "loss": total,
        "task_loss": output.loss.detach(),
        "preservation_loss": preservation.detach(),
        "contrastive_loss": contrastive.detach(),
        "redundant_gate_loss": neutral.detach(),
        "preserved_rows": preserved_rows,
        "required_rows": int(required.sum()),
        "redundant_rows": int(redundant.sum()),
        "required_correct_math_loss": (
            float(current_per_row[required].mean()) if bool(required.any()) else 0.0
        ),
        "required_disabled_math_loss": (
            float(disabled_per_row[required].mean()) if bool(required.any()) else 0.0
        ),
        "required_shuffled_math_loss": (
            float(shuffled_per_row[required].mean()) if bool(required.any()) else 0.0
        ),
    }


def _bridge_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(BRIDGE_PREFIXES)
    }


def _set_conditional_trainable(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.gpt_to_math, model.math_receivers):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _set_conditional_train_mode(model: Any) -> None:
    model.eval()
    model.gpt_to_math.train()
    model.math_receivers.train()


def _group_statistics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    examples = int(mask.sum())
    if examples == 0:
        return {"examples": 0, "token_accuracy": 0.0, "sequence_accuracy": 0.0}
    correct, total, sequence = masked_token_statistics(logits[mask], labels[mask])
    return {
        "examples": examples,
        "token_accuracy": correct / max(1, total),
        "sequence_accuracy": sequence / examples,
    }


@torch.no_grad()
def evaluate_conditional_teacher_forcing(
    model: Any,
    loader: Any,
    device: torch.device,
    dtype: torch.dtype | None,
    settings: dict[str, Any],
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {
        "examples": 0,
        "loss": 0.0,
        "required_examples": 0,
        "required_correct_math_loss": 0.0,
        "required_disabled_math_loss": 0.0,
        "required_shuffled_math_loss": 0.0,
    }
    group_counts = {
        name: {"math_correct": 0, "math_total": 0, "math_sequences": 0,
               "gpt_correct": 0, "gpt_total": 0, "gpt_sequences": 0,
               "examples": 0, "baseline_math_sequences": 0}
        for name in (REQUIRED_REQUIREMENT, REDUNDANT_REQUIREMENT)
    }
    gates: dict[str, list[torch.Tensor]] = {
        "required_sender": [],
        "redundant_sender": [],
        "required_receiver": [],
        "redundant_receiver": [],
    }
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        required, redundant = _requirement_masks(batch["records"], device)
        with autocast_context(device, dtype):
            output = model(
                batch,
                gpt_to_math_enabled=True,
                math_to_gpt_enabled=True,
                math_loss_weight=float(settings.get("math_loss_weight", 1.0)),
                gpt_loss_weight=float(settings.get("gpt_loss_weight", 1.0)),
                answer_head_weight=float(settings.get("answer_head_weight", 0.0)),
            )
            disabled = model(
                batch,
                gpt_to_math_enabled=False,
                math_to_gpt_enabled=False,
                math_loss_weight=1.0,
                gpt_loss_weight=0.0,
                answer_head_weight=0.0,
            )
            shuffled = model(
                batch,
                gpt_to_math_enabled=True,
                math_to_gpt_enabled=False,
                shuffle_gpt_to_math=True,
                math_loss_weight=1.0,
                gpt_loss_weight=0.0,
                answer_head_weight=0.0,
            )
        batch_size = int(batch["math_input_ids"].shape[0])
        totals["examples"] += batch_size
        totals["loss"] += float(output.loss) * batch_size
        correct_per_row = per_example_causal_loss(
            output.math_output.logits, batch["math_labels"]
        )
        disabled_per_row = per_example_causal_loss(
            disabled.math_output.logits, batch["math_labels"]
        )
        shuffled_per_row = per_example_causal_loss(
            shuffled.math_output.logits, batch["math_labels"]
        )
        required_examples = int(required.sum())
        totals["required_examples"] += required_examples
        if required_examples:
            totals["required_correct_math_loss"] += float(
                correct_per_row[required].sum()
            )
            totals["required_disabled_math_loss"] += float(
                disabled_per_row[required].sum()
            )
            totals["required_shuffled_math_loss"] += float(
                shuffled_per_row[required].sum()
            )

        for name, mask in (
            (REQUIRED_REQUIREMENT, required),
            (REDUNDANT_REQUIREMENT, redundant),
        ):
            count = int(mask.sum())
            if not count:
                continue
            stats = group_counts[name]
            stats["examples"] += count
            correct, total, sequence = masked_token_statistics(
                output.math_output.logits[mask], batch["math_labels"][mask]
            )
            stats["math_correct"] += correct
            stats["math_total"] += total
            stats["math_sequences"] += sequence
            correct, total, sequence = masked_token_statistics(
                output.gpt_logits[mask], batch["gpt_labels"][mask]
            )
            stats["gpt_correct"] += correct
            stats["gpt_total"] += total
            stats["gpt_sequences"] += sequence
            _, _, baseline_sequence = masked_token_statistics(
                disabled.math_output.logits[mask], batch["math_labels"][mask]
            )
            stats["baseline_math_sequences"] += baseline_sequence
            gate_prefix = "required" if name == REQUIRED_REQUIREMENT else "redundant"
            gates[f"{gate_prefix}_sender"].append(output.gpt_to_math.gate[mask])
            gates[f"{gate_prefix}_receiver"].extend(
                gate[mask] for gate in output.math_receiver_gates.values()
            )

    examples = int(totals["examples"])
    if not examples:
        raise RuntimeError("conditional validation loader produced no examples")
    required_examples = max(1, int(totals["required_examples"]))
    groups: dict[str, Any] = {}
    for name, raw in group_counts.items():
        count = max(1, int(raw["examples"]))
        groups[name] = {
            "examples": int(raw["examples"]),
            "math_teacher_forced_token_accuracy": raw["math_correct"]
            / max(1, raw["math_total"]),
            "math_teacher_forced_sequence_accuracy": raw["math_sequences"] / count,
            "gpt_teacher_forced_token_accuracy": raw["gpt_correct"]
            / max(1, raw["gpt_total"]),
            "gpt_teacher_forced_sequence_accuracy": raw["gpt_sequences"] / count,
            "gpt_to_math_disabled_math_sequence_accuracy": raw[
                "baseline_math_sequences"
            ]
            / count,
        }
    required_correct = totals["required_correct_math_loss"] / required_examples
    required_shuffled = totals["required_shuffled_math_loss"] / required_examples
    return {
        "examples": examples,
        "loss": totals["loss"] / examples,
        "shuffled_loss": required_shuffled,
        "shuffled_loss_gap": required_shuffled - required_correct,
        "math_teacher_forced_sequence_accuracy": groups[REQUIRED_REQUIREMENT][
            "math_teacher_forced_sequence_accuracy"
        ],
        "gpt_teacher_forced_sequence_accuracy": groups[REQUIRED_REQUIREMENT][
            "gpt_teacher_forced_sequence_accuracy"
        ],
        "required_correct_math_loss": required_correct,
        "required_disabled_math_loss": totals["required_disabled_math_loss"]
        / required_examples,
        "required_shuffled_math_loss": required_shuffled,
        "groups": groups,
        "gates": {
            key: summarize_gate(value)
            for key, value in gates.items()
        },
    }


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        chunk = values[start : start + size]
        if len(chunk) == 1 and start:
            # The configured panels are normally divisible by batch size.  A
            # final singleton cannot support a shuffled-message control.
            continue
        yield chunk


def _generation_is_correct(text: str, record: dict[str, Any]) -> bool:
    if str(record.get("schema_version", "")).startswith("cftn_math_record_v2"):
        from .v2_metrics import answers_equivalent, extract_v2_answer

        return answers_equivalent(
            extract_v2_answer(text), str(record["normalized_answer"])
        )
    return extract_answer(text) == int(record["x"])


@torch.no_grad()
def evaluate_conditional_generation(
    model: Any,
    records: list[dict[str, Any]],
    math_tokenizer: ByteMathTokenizer,
    gpt_tokenizer: Any,
    *,
    batch_size: int,
    max_math_new_tokens: int,
    max_gpt_new_tokens: int,
) -> dict[str, Any]:
    model.eval()
    conditions = {
        "correct": {},
        "gpt_to_math_disabled": {"gpt_to_math_enabled": False},
        "gpt_to_math_shuffled": {"shuffle_gpt_to_math": True},
        "math_to_gpt_disabled": {"math_to_gpt_enabled": False},
        "both_disabled": {
            "gpt_to_math_enabled": False,
            "math_to_gpt_enabled": False,
        },
    }
    by_requirement = {
        requirement: [
            record
            for record in records
            if record.get("communication_requirement") == requirement
        ]
        for requirement in (REQUIRED_REQUIREMENT, REDUNDANT_REQUIREMENT)
    }
    report: dict[str, Any] = {}
    for requirement, selected in by_requirement.items():
        if len(selected) < 2:
            raise RuntimeError(f"generation panel lacks {requirement} examples")
        arm_correct: dict[str, dict[str, int]] = {
            name: {"gpt": 0, "math": 0, "examples": 0}
            for name in conditions
        }
        sender_gates: list[float] = []
        for chunk in _chunks(selected, int(batch_size)):
            problems = [str(record.get("shared_problem", record["problem"])) for record in chunk]
            gpt_problems = [str(record.get("gpt_problem", record["problem"])) for record in chunk]
            math_problems = [str(record.get("math_problem", record["problem"])) for record in chunk]
            for name, controls in conditions.items():
                outputs = model.generate_problems(
                    problems,
                    math_tokenizer,
                    gpt_tokenizer,
                    max_math_new_tokens=int(max_math_new_tokens),
                    max_gpt_new_tokens=int(max_gpt_new_tokens),
                    gpt_problems=gpt_problems,
                    math_problems=math_problems,
                    **controls,
                )
                arm_correct[name]["examples"] += len(outputs)
                for output, record in zip(outputs, chunk):
                    arm_correct[name]["gpt"] += int(
                        _generation_is_correct(output["gpt_generation"], record)
                    )
                    arm_correct[name]["math"] += int(
                        _generation_is_correct(output["math_generation"], record)
                    )
                    if name == "correct":
                        sender_gates.append(
                            float(
                                output["communication"][
                                    "gpt_to_math_sender_gate"
                                ]["mean"]
                            )
                        )
                        # Generation output currently exposes sender gates.  The
                        # teacher-forced panel records receiving-layer gates.
        accuracy = {
            name: {
                "gpt": values["gpt"] / max(1, values["examples"]),
                "math": values["math"] / max(1, values["examples"]),
                "examples": values["examples"],
            }
            for name, values in arm_correct.items()
        }
        report[requirement] = {
            "accuracy": accuracy,
            "gpt_to_math_sender_gate_mean": (
                sum(sender_gates) / len(sender_gates) if sender_gates else 0.0
            ),
        }
    return report


def conditional_acceptance(
    generation: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    required = generation[REQUIRED_REQUIREMENT]
    redundant = generation[REDUNDANT_REQUIREMENT]
    req = required["accuracy"]
    red = redundant["accuracy"]
    strongest_individual = max(
        req["both_disabled"]["gpt"], req["both_disabled"]["math"]
    )
    metrics = {
        "required_synergy_gain": req["correct"]["gpt"] - strongest_individual,
        "required_gpt_to_math_gain": req["correct"]["math"]
        - req["gpt_to_math_disabled"]["math"],
        "required_correct_vs_shuffled_gap": req["correct"]["math"]
        - req["gpt_to_math_shuffled"]["math"],
        "required_math_to_gpt_gain": req["correct"]["gpt"]
        - req["math_to_gpt_disabled"]["gpt"],
        "redundant_math_regression": red["gpt_to_math_disabled"]["math"]
        - red["correct"]["math"],
        "gate_separation": required["gpt_to_math_sender_gate_mean"]
        - redundant["gpt_to_math_sender_gate_mean"],
    }
    gates = {
        "required_synergy": metrics["required_synergy_gain"]
        >= float(thresholds["minimum_required_synergy_gain"]),
        "required_content_specific_gpt_to_math": metrics[
            "required_correct_vs_shuffled_gap"
        ]
        >= float(thresholds["minimum_required_correct_vs_shuffled_gap"]),
        "required_gpt_to_math": metrics["required_gpt_to_math_gain"]
        >= float(thresholds["minimum_required_gpt_to_math_gain"]),
        "required_math_to_gpt": metrics["required_math_to_gpt_gain"]
        >= float(thresholds["minimum_required_math_to_gpt_gain"]),
        "redundant_no_harm": metrics["redundant_math_regression"]
        <= float(thresholds["maximum_redundant_math_regression"]),
        "contextual_gate_separation": metrics["gate_separation"]
        >= float(thresholds["minimum_gate_separation"]),
    }
    gates["pass"] = all(gates.values())
    return {"metrics": metrics, "gates": gates}


def conditional_selection_score(acceptance: dict[str, Any]) -> float:
    metrics = acceptance["metrics"]
    return float(
        metrics["required_synergy_gain"]
        + 0.25 * metrics["required_gpt_to_math_gain"]
        + 0.25 * metrics["required_correct_vs_shuffled_gap"]
        + 0.10 * metrics["required_math_to_gpt_gain"]
        + 0.05 * metrics["gate_separation"]
        - 2.0 * max(0.0, metrics["redundant_math_regression"])
    )


def _audit_v1_2_revision_prerequisites(
    revision: dict[str, Any]
) -> dict[str, Any]:
    paths = revision["paths"]
    v1_root = Path(paths["v1_1_artifact_root"])
    pipeline_status_path = v1_root / "pipeline_status.json"
    if not pipeline_status_path.is_file():
        raise FileNotFoundError("V1.1 pipeline status is missing")
    with pipeline_status_path.open("r", encoding="utf-8") as handle:
        pipeline_status = json.load(handle)
    if pipeline_status.get("state") != "completed":
        raise RuntimeError(
            "V1.2 refuses to start before the sealed V1.1 pipeline completes"
        )
    base_config = load_config(paths["base_config"])
    _, manifest = load_data_contract(base_config)
    math_checkpoint = load_checkpoint(
        paths["math_checkpoint"],
        expected_stage="math",
        expected_config_sha256=config_sha256(base_config),
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    bridge_checkpoint = load_checkpoint(
        paths["source_bridge_checkpoint"],
        expected_stage="bidirectional",
        expected_config_sha256=config_sha256(base_config),
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    if bridge_checkpoint["extra"].get("gate_mode") != "contextual":
        raise ValueError("V1.2 source bridge must use contextual gates")
    if bridge_checkpoint["extra"].get("view_mode") != "complementary":
        raise ValueError("V1.2 source bridge must be complementary-view trained")
    required_reports = (
        v1_root / "evaluation_bidirectional_contextual_shared" / "report.json",
        v1_root / "synergy_evaluation_contextual" / "report.json",
    )
    missing = [str(path) for path in required_reports if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V1.1 evidence is incomplete: {missing}")
    return {
        "format": "cftn_text_v1_2_prerequisite_audit_v1",
        "state": "passed",
        "revision_sha256": revision["_meta"]["sha256"],
        "base_config_sha256": config_sha256(base_config),
        "manifest_sha256": manifest["manifest_sha256"],
        "math_checkpoint": str(Path(paths["math_checkpoint"]).resolve()),
        "math_checkpoint_sha256": file_sha256(paths["math_checkpoint"]),
        "source_bridge_checkpoint": str(
            Path(paths["source_bridge_checkpoint"]).resolve()
        ),
        "source_bridge_checkpoint_sha256": file_sha256(
            paths["source_bridge_checkpoint"]
        ),
        "source_bridge_epoch": int(bridge_checkpoint["epoch"]),
        "math_checkpoint_epoch": int(math_checkpoint["epoch"]),
        "v1_1_pipeline_status": str(pipeline_status_path.resolve()),
    }


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _audit_v2_revision_prerequisites(
    revision: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed unless prior mechanisms and the new specialist have passed."""

    paths = revision["paths"]
    mechanism_path = Path(paths["mechanism_prerequisite_report"])
    mechanism = _load_json_object(
        mechanism_path, "V2 mechanism-prerequisite report"
    )
    if mechanism.get("state") != "passed" or mechanism.get("pass") is not True:
        raise RuntimeError("V2 conditional training requires passed V1.2 and V1.3 evidence")

    specialist_path = Path(paths["specialist_report"])
    specialist = _load_json_object(specialist_path, "V2 specialist report")
    if specialist.get("specialist_gate", {}).get("pass") is not True:
        raise RuntimeError("V2 conditional training requires a passed generative specialist gate")

    base_config = load_config(paths["base_config"])
    _, manifest = load_data_contract(base_config)
    math_checkpoint = load_checkpoint(
        paths["math_checkpoint"],
        expected_stage="math",
        expected_config_sha256=config_sha256(base_config),
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    source_stage = str(revision.get("source_bridge_stage", "m2g"))
    if source_stage not in {"m2g", "bidirectional"}:
        raise ValueError("V2 source_bridge_stage must be m2g or bidirectional")
    source_checkpoint = load_checkpoint(
        paths["source_bridge_checkpoint"],
        expected_stage=source_stage,
        expected_config_sha256=config_sha256(base_config),
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    if source_checkpoint["extra"].get("gate_mode") != "contextual":
        raise ValueError("V2 source bridge must use contextual gates")
    source_view_mode = str(revision.get("source_bridge_view_mode", "shared"))
    if source_checkpoint["extra"].get("view_mode") != source_view_mode:
        raise ValueError(
            f"V2 source bridge must be {source_view_mode}-view trained"
        )
    return {
        "format": "cftn_text_v2_conditional_prerequisite_audit_v1",
        "state": "passed",
        "revision_sha256": revision["_meta"]["sha256"],
        "base_config_sha256": config_sha256(base_config),
        "manifest_sha256": manifest["manifest_sha256"],
        "mechanism_prerequisite_report": str(mechanism_path.resolve()),
        "mechanism_prerequisite_report_sha256": file_sha256(mechanism_path),
        "specialist_report": str(specialist_path.resolve()),
        "specialist_report_sha256": file_sha256(specialist_path),
        "math_checkpoint": str(Path(paths["math_checkpoint"]).resolve()),
        "math_checkpoint_sha256": file_sha256(paths["math_checkpoint"]),
        "math_checkpoint_epoch": int(math_checkpoint["epoch"]),
        "source_bridge_stage": source_stage,
        "source_bridge_view_mode": source_view_mode,
        "source_bridge_checkpoint": str(
            Path(paths["source_bridge_checkpoint"]).resolve()
        ),
        "source_bridge_checkpoint_sha256": file_sha256(
            paths["source_bridge_checkpoint"]
        ),
        "source_bridge_epoch": int(source_checkpoint["epoch"]),
    }


def audit_revision_prerequisites(revision: dict[str, Any]) -> dict[str, Any]:
    if revision.get("format") == V2_REVISION_FORMAT:
        return _audit_v2_revision_prerequisites(revision)
    return _audit_v1_2_revision_prerequisites(revision)


def train_conditional_bridge(
    revision: dict[str, Any],
    *,
    device_name: str = "cuda",
    resume: bool = False,
    max_batches: int | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Keep direct entry points as safe as ordered runners. Each revision must
    # independently revalidate the evidence and checkpoints it consumes.
    prerequisite_audit = audit_revision_prerequisites(revision)
    paths = revision["paths"]
    settings = revision["training"]
    validation_settings = revision["validation"]
    acceptance_settings = revision["acceptance"]
    base_config = load_config(paths["base_config"])
    seed = int(base_config["project"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    data_root, manifest = load_data_contract(base_config)
    model, gpt_tokenizer = build_cftn_model(
        base_config, paths["math_checkpoint"], manifest, device
    )
    model.set_gate_mode("contextual")
    source_stage = str(revision.get("source_bridge_stage", "bidirectional"))
    model.set_trainable_stage(source_stage)
    source = load_checkpoint(
        paths["source_bridge_checkpoint"],
        expected_stage=source_stage,
        expected_config_sha256=config_sha256(base_config),
        expected_manifest_sha256=manifest["manifest_sha256"],
        map_location=device,
    )
    model.load_trainable_state_dict(source["model_state"], strict=True)
    model.set_trainable_stage("bidirectional")
    _set_conditional_trainable(model)

    train_source = split_dataset(data_root, manifest, "train")
    validation_source = split_dataset(data_root, manifest, "validation")
    train_source_records = train_source.records
    validation_source_records = validation_source.records
    if manifest.get("format") == "cftn_text_broad_math_v2":
        train_source_records = [
            record
            for record in train_source_records
            if record.get("gpt_problem") and record.get("math_problem")
        ]
        validation_source_records = [
            record
            for record in validation_source_records
            if record.get("gpt_problem") and record.get("math_problem")
        ]
        if (
            len(train_source_records) < int(settings["base_train_examples"])
            or len(validation_source_records)
            < int(settings["base_validation_examples"])
        ):
            raise RuntimeError(
                "V2 conditional training lacks enough task-matched private-view records"
            )
    train_records = build_mixed_necessity_records(
        train_source_records,
        seed=seed,
        limit=int(settings["base_train_examples"]),
    )
    validation_records = build_mixed_necessity_records(
        validation_source_records,
        seed=seed + 1,
        limit=int(settings["base_validation_examples"]),
    )
    train_dataset = EquationDataset(train_records)
    validation_dataset = EquationDataset(validation_records)
    generation_per_view = int(validation_settings["generation_examples_per_view"])
    # Preserve equal deterministic panels instead of slicing the interleaved
    # list, which would bias one requirement when limits change.
    generation_records: list[dict[str, Any]] = []
    for requirement in (REQUIRED_REQUIREMENT, REDUNDANT_REQUIREMENT):
        generation_records.extend(
            [
                record
                for record in validation_records
                if record["communication_requirement"] == requirement
            ][:generation_per_view]
        )

    math_tokenizer = ByteMathTokenizer()
    collator = CFTNCollator(
        math_tokenizer,
        gpt_tokenizer,
        int(base_config["data"]["max_math_length"]),
        int(base_config["data"]["max_gpt_length"]),
    )
    artifact_root = Path(paths["artifact_root"])
    artifact_dir = artifact_root / str(
        revision.get("output_subdirectory", "bridge_conditional_contextual")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "metrics.jsonl"
    status_path = artifact_dir / "status.json"
    best_path = artifact_dir / "bridge_bidirectional.best.pth"

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("conditional bridge has no trainable GPT-to-math parameters")
    gate_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".gate_network." in name
    ]
    bridge_parameters = [
        parameter
        for name, parameter in named_parameters
        if ".gate_network." not in name
    ]
    optimizer = AdamW(
        [
            {
                "params": bridge_parameters,
                "lr": float(settings["learning_rate"]),
                "weight_decay": float(settings.get("weight_decay", 0.01)),
                "group_name": "gpt_to_math_bridge",
            },
            {
                "params": gate_parameters,
                "lr": float(settings["learning_rate"])
                * float(settings.get("gate_learning_rate_multiplier", 0.25)),
                "weight_decay": 0.0,
                "group_name": "contextual_gates",
            },
        ]
    )
    steps_per_epoch = max(
        1, math.ceil(len(train_dataset) / int(settings["batch_size"]))
    )
    total_steps = int(settings["max_epochs"]) * steps_per_epoch
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=float(settings.get("warmup_fraction", 0.05)),
        minimum_ratio=float(settings["minimum_learning_rate"])
        / float(settings["learning_rate"]),
    )
    dtype = precision_dtype(str(settings.get("precision", "bf16")), device)
    scaler = make_scaler(device, dtype)
    stability_settings = {
        "learning_rate": float(settings["learning_rate"]),
        "minimum_learning_rate": float(settings["minimum_learning_rate"]),
        "stability_maximum_learning_rate": float(settings["learning_rate"]),
        "gate_learning_rate_multiplier": float(
            settings.get("gate_learning_rate_multiplier", 0.25)
        ),
    }
    stability_policy = _bridge_stability_policy(stability_settings)
    started_at = time.time()
    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    best_validation: dict[str, Any] | None = None
    best_acceptance: dict[str, Any] | None = None
    patience = 0
    if resume:
        checkpoint_path = latest_checkpoint(artifact_dir)
        if checkpoint_path is None:
            raise FileNotFoundError("no conditional-bridge checkpoint is available to resume")
        checkpoint = load_checkpoint(
            checkpoint_path,
            expected_stage="bidirectional",
            expected_config_sha256=config_sha256(base_config),
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location=device,
        )
        expected_policy = str(
            revision.get("training_view_policy", "mixed_necessity_v1_2")
        )
        if checkpoint["extra"].get("training_view_policy") != expected_policy:
            raise ValueError("checkpoint belongs to another conditional-bridge policy")
        if checkpoint["extra"].get("revision_sha256") != revision["_meta"]["sha256"]:
            raise ValueError("conditional-bridge revision configuration changed")
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
        best_acceptance = checkpoint["extra"].get("best_acceptance")
        patience = int(checkpoint["patience"])

    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact_dir,
        stage="conditional_gpt_to_math",
        config={
            "base_config_sha256": config_sha256(base_config),
            "manifest_sha256": manifest["manifest_sha256"],
            "revision": revision,
            "frozen_math_to_gpt": True,
            "source_bridge_stage": source_stage,
            "prerequisite_audit": prerequisite_audit,
            "trainable_parameters": sum(p.numel() for _, p in named_parameters),
        },
    )
    atomic_json_dump(
        _status_payload(
            stage="conditional_gpt_to_math",
            state="running",
            epoch=start_epoch - 1,
            global_step=global_step,
            started_at=started_at,
        ),
        status_path,
    )
    stop_reason = "max_epochs"
    final_metrics: dict[str, Any] = {}
    try:
        for epoch in range(start_epoch, int(settings["max_epochs"]) + 1):
            _set_conditional_train_mode(model)
            train_loader = make_loader(
                train_dataset,
                collator,
                batch_size=int(settings["batch_size"]),
                shuffle=True,
                seed=seed,
                epoch=epoch,
                num_workers=int(settings.get("num_workers", 2)),
            )
            component_sums = {
                "loss": 0.0,
                "task_loss": 0.0,
                "preservation_loss": 0.0,
                "contrastive_loss": 0.0,
                "redundant_gate_loss": 0.0,
            }
            trained_examples = 0
            epoch_started_at = time.time()
            for batch_index, raw_batch in enumerate(train_loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, dtype):
                    _, components = conditional_objective(model, batch, settings)
                loss = components["loss"]
                assert torch.is_tensor(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in named_parameters],
                    float(settings["gradient_clip"]),
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                batch_examples = int(batch["math_input_ids"].shape[0])
                trained_examples += batch_examples
                global_step += 1
                for key in component_sums:
                    value = components[key]
                    component_sums[key] += float(value) * batch_examples
                report_every = max(1, int(settings.get("report_every_steps", 100)))
                if batch_index == 0 or (batch_index + 1) % report_every == 0:
                    progress = {
                        "phase": "training",
                        "epoch": epoch,
                        "epoch_batch_completed": batch_index + 1,
                        "epoch_batches_total": (
                            min(len(train_loader), int(max_batches))
                            if max_batches is not None
                            else len(train_loader)
                        ),
                        "global_step": global_step,
                        "train_loss_so_far": component_sums["loss"]
                        / max(1, trained_examples),
                        "learning_rates": {
                            str(group.get("group_name", index)): float(group["lr"])
                            for index, group in enumerate(optimizer.param_groups)
                        },
                    }
                    atomic_json_dump(
                        _status_payload(
                            stage="conditional_gpt_to_math",
                            state="running",
                            epoch=epoch,
                            global_step=global_step,
                            metrics=progress,
                            started_at=started_at,
                        ),
                        status_path,
                    )
                    tracker.log(
                        {"train": progress},
                        global_step=global_step,
                        epoch=epoch,
                        event="training_progress",
                    )

            validation_loader = make_loader(
                validation_dataset,
                collator,
                batch_size=int(settings["eval_batch_size"]),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=int(settings.get("num_workers", 2)),
            )
            teacher_validation = evaluate_conditional_teacher_forcing(
                model,
                validation_loader,
                device,
                dtype,
                settings,
                max_batches=max_batches,
            )
            generation = evaluate_conditional_generation(
                model,
                generation_records,
                math_tokenizer,
                gpt_tokenizer,
                batch_size=int(validation_settings["generation_batch_size"]),
                max_math_new_tokens=int(validation_settings["max_math_new_tokens"]),
                max_gpt_new_tokens=int(validation_settings["max_gpt_new_tokens"]),
            )
            acceptance = conditional_acceptance(generation, acceptance_settings)
            selection_metric = conditional_selection_score(acceptance)
            collapse_guard = _bridge_collapse_diagnostics(
                teacher_validation, best_validation, stability_policy
            )
            improved = not collapse_guard["triggered"] and selection_metric > best_metric
            if improved:
                best_metric = selection_metric
                best_validation = teacher_validation
                best_acceptance = acceptance
                patience = 0
            else:
                patience += 1
            final_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train": {
                    key: value / max(1, trained_examples)
                    for key, value in component_sums.items()
                },
                "teacher_forced_validation": teacher_validation,
                "generation_validation": generation,
                "acceptance": acceptance,
                "selection_metric": selection_metric,
                "best_metric": best_metric,
                "patience": patience,
                "collapse_guard": collapse_guard,
                "learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "frozen_math_to_gpt": True,
                "trainable_parameters": sum(
                    parameter.numel() for _, parameter in named_parameters
                ),
                "timing": {
                    "epoch_seconds": time.time() - epoch_started_at,
                    "eta_seconds_to_max_epochs": (
                        int(settings["max_epochs"]) - epoch
                    )
                    * (time.time() - epoch_started_at),
                },
                "gpu": gpu_status(),
            }
            append_jsonl(final_metrics, metrics_path)
            tracker.log(
                final_metrics,
                global_step=global_step,
                epoch=epoch,
                event="epoch_validation",
            )
            payload = build_checkpoint(
                stage="bidirectional",
                epoch=epoch,
                global_step=global_step,
                model_state=_bridge_state_dict(model),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                config_sha256=config_sha256(base_config),
                manifest_sha256=manifest["manifest_sha256"],
                best_metric=best_metric,
                patience=patience,
                extra={
                    "metrics": final_metrics,
                    "best_validation": best_validation,
                    "best_acceptance": best_acceptance,
                    "gate_mode": "contextual",
                    "view_mode": "complementary",
                    "training_view_policy": str(
                        revision.get("training_view_policy", "mixed_necessity_v1_2")
                    ),
                    "revision_sha256": revision["_meta"]["sha256"],
                    "math_checkpoint": str(Path(paths["math_checkpoint"]).resolve()),
                    "math_checkpoint_sha256": file_sha256(paths["math_checkpoint"]),
                    "source_bridge_checkpoint": str(
                        Path(paths["source_bridge_checkpoint"]).resolve()
                    ),
                    "source_bridge_checkpoint_sha256": file_sha256(
                        paths["source_bridge_checkpoint"]
                    ),
                    "frozen_math_to_gpt": True,
                },
            )
            checkpoint_path = artifact_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            atomic_torch_save(payload, checkpoint_path)
            rotate_latest(
                artifact_dir,
                int(revision.get("monitoring", {}).get("keep_latest_checkpoints", 3)),
            )
            if improved:
                atomic_torch_save(payload, best_path)
            atomic_json_dump(
                _status_payload(
                    stage="conditional_gpt_to_math",
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
                stop_reason = "early_stopping_generation_plateau"
                break
        state = "completed"
    except BaseException as exc:
        atomic_json_dump(
            _status_payload(
                stage="conditional_gpt_to_math",
                state="error",
                epoch=locals().get("epoch", start_epoch - 1),
                global_step=global_step,
                metrics={"error": repr(exc)},
                started_at=started_at,
            ),
            status_path,
        )
        tracker.update_summary({"run/state": "error", "run/error": repr(exc)})
        tracker.finish(exit_code=1)
        raise

    result = {
        "format": (
            "cftn_text_v2_conditional_training_result_v1"
            if revision.get("format") == V2_REVISION_FORMAT
            else "cftn_text_v1_2_conditional_training_result_v1"
        ),
        "state": state,
        "stop_reason": stop_reason,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": file_sha256(best_path),
        "best_metric": best_metric,
        "best_acceptance": best_acceptance,
        "final_metrics": final_metrics,
        "revision_sha256": revision["_meta"]["sha256"],
    }
    atomic_json_dump(result, artifact_dir / "summary.json")
    atomic_json_dump(
        _status_payload(
            stage="conditional_gpt_to_math",
            state="completed",
            epoch=int(final_metrics.get("epoch", 0)),
            global_step=global_step,
            metrics=final_metrics,
            started_at=started_at,
        ),
        status_path,
    )
    tracker.update_summary(
        {
            "run/state": "completed",
            "run/stop_reason": stop_reason,
            "run/best_metric": best_metric,
            "run/final_epoch": int(final_metrics.get("epoch", 0)),
        }
    )
    tracker.finish()
    return result
