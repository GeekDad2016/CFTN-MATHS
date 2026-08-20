from __future__ import annotations

import json
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

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
from .config import config_sha256, load_config
from .data_generator import file_sha256
from .math_tower import MathTower
from .metrics import masked_token_statistics
from .model import causal_language_loss
from .tokenizer import ByteMathTokenizer
from .training import (
    autocast_context,
    load_data_contract,
    load_gpt_components,
    make_scaler,
    make_scheduler,
    precision_dtype,
    resolve_device,
    seed_everything,
)
from .v1_3_config import audit_v1_2_pass
from .v1_3_data import (
    SPECIALISTS,
    audit_v1_3_manifest,
    generate_joint_record,
    prepare_v1_3_manifests,
)
from .v1_3_dataset import (
    V13Dataset,
    V13JointCollator,
    V13StringCollator,
    move_v1_3_batch,
)
from .v1_3_model import V13ModelOutput, V13MultiTowerModel
from .wandb_support import initialize_wandb


def _status(
    *, stage: str, state: str, epoch: int, global_step: int, started_at: float, metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "format": "cftn_text_v1_3_stage_status_v1",
        "stage": stage,
        "state": state,
        "pid": os.getpid(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "elapsed_seconds": time.time() - started_at,
        "metrics": metrics,
        "gpu": gpu_status(),
        "updated_unix": time.time(),
    }


def load_v1_3_data_contract(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(config["paths"]["data_root"])
    if not (root / "manifest.json").is_file():
        prepare_v1_3_manifests(config, root)
    return root, audit_v1_3_manifest(config, root)


def _loader(
    dataset: V13Dataset,
    collator: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed) + int(epoch) * 1_000_003)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=collator,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
        generator=generator,
    )


def build_string_tower(config: dict[str, Any]) -> MathTower:
    return MathTower(config["string_tower"], ByteMathTokenizer.vocab_size)


def gpt_interface_config(
    base_config: dict[str, Any], integration_config: dict[str, Any]
) -> dict[str, Any]:
    """Build GPT receivers against the integration bridge, not the math-only bridge."""

    return {**base_config, "bridge": dict(integration_config["bridge"])}


@torch.no_grad()
def evaluate_string_teacher_forcing(
    model: MathTower,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    *,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    examples = token_correct = token_total = sequence_correct = 0
    for batch_index, raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_v1_3_batch(raw, device)
        with autocast_context(device, dtype):
            output = model(
                batch["input_ids"], batch["attention_mask"], batch["prefix_lengths"]
            )
            loss = causal_language_loss(output.logits, batch["labels"])
        count = int(batch["input_ids"].shape[0])
        loss_sum += float(loss) * count
        examples += count
        correct, total, sequences = masked_token_statistics(output.logits, batch["labels"])
        token_correct += correct
        token_total += total
        sequence_correct += sequences
    if not examples:
        raise RuntimeError("string validation produced no examples")
    return {
        "examples": examples,
        "loss": loss_sum / examples,
        "teacher_forced_token_accuracy": token_correct / max(1, token_total),
        "teacher_forced_sequence_accuracy": sequence_correct / examples,
    }


def train_string_specialist(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    resume: bool = False,
    max_batches: int | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prerequisite = audit_v1_2_pass(config)
    seed = int(config["revision"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    root, manifest = load_v1_3_data_contract(config)
    train_data = V13Dataset(root / manifest["splits"]["string_train"]["path"])
    validation_data = V13Dataset(
        root / manifest["splits"]["string_validation"]["path"]
    )
    collator = V13StringCollator(
        ByteMathTokenizer(), int(config["data"]["maximum_specialist_length"])
    )
    settings = config["string_training"]
    artifact = Path(config["paths"]["artifact_root"]) / "string_specialist"
    artifact.mkdir(parents=True, exist_ok=True)
    status_path = artifact / "status.json"
    metrics_path = artifact / "metrics.jsonl"
    best_path = artifact / "string.best.pth"
    started_at = time.time()
    model = build_string_tower(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    steps_per_epoch = max(1, math.ceil(len(train_data) / int(settings["batch_size"])))
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
    patience = 0
    if resume and latest_checkpoint(artifact):
        checkpoint = load_checkpoint(
            latest_checkpoint(artifact),
            expected_stage="v1_3_string",
            expected_config_sha256=config["_meta"]["sha256"],
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler.is_enabled() and checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        patience = int(checkpoint["patience"])
    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact,
        stage="v1_3_string_specialist",
        config={"revision_sha256": config["_meta"]["sha256"]},
    )
    final_metrics: dict[str, Any] = {}
    stop_reason = "maximum_epochs"
    try:
        for epoch in range(start_epoch, int(settings["max_epochs"]) + 1):
            epoch_started = time.time()
            model.train()
            train_loss = 0.0
            examples = 0
            loader = _loader(
                train_data,
                collator,
                batch_size=int(settings["batch_size"]),
                shuffle=True,
                seed=seed,
                epoch=epoch,
                num_workers=int(settings["num_workers"]),
            )
            for batch_index, raw in enumerate(loader, start=1):
                if max_batches is not None and batch_index > max_batches:
                    break
                batch = move_v1_3_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, dtype):
                    output = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["prefix_lengths"],
                    )
                    loss = causal_language_loss(output.logits, batch["labels"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                count = int(batch["input_ids"].shape[0])
                examples += count
                train_loss += float(loss.detach()) * count
                if global_step % int(settings["report_every_steps"]) == 0:
                    progress = {
                        "epoch_batch_completed": batch_index,
                        "epoch_batches_total": min(len(loader), max_batches or len(loader)),
                        "train_loss_so_far": train_loss / max(1, examples),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                    atomic_json_dump(
                        _status(
                            stage="string_specialist",
                            state="running",
                            epoch=epoch,
                            global_step=global_step,
                            started_at=started_at,
                            metrics=progress,
                        ),
                        status_path,
                    )
            validation_loader = _loader(
                validation_data,
                collator,
                batch_size=int(settings["eval_batch_size"]),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=int(settings["num_workers"]),
            )
            validation = evaluate_string_teacher_forcing(
                model, validation_loader, device, dtype, max_batches=max_batches
            )
            selection = float(validation["teacher_forced_sequence_accuracy"]) - 1e-6 * float(
                validation["loss"]
            )
            improved = selection > best_metric
            if improved:
                best_metric = selection
                patience = 0
            else:
                patience += 1
            final_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss / max(1, examples),
                "validation": validation,
                "selection_metric": selection,
                "best_metric": best_metric,
                "patience": patience,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "timing": {
                    "epoch_seconds": time.time() - epoch_started,
                    "eta_seconds_to_max_epochs": (
                        int(settings["max_epochs"]) - epoch
                    )
                    * (time.time() - epoch_started),
                },
                "gpu": gpu_status(),
            }
            append_jsonl(final_metrics, metrics_path)
            tracker.log(final_metrics, global_step=global_step, epoch=epoch, event="epoch_validation")
            payload = build_checkpoint(
                stage="v1_3_string",
                epoch=epoch,
                global_step=global_step,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                config_sha256=config["_meta"]["sha256"],
                manifest_sha256=manifest["manifest_sha256"],
                best_metric=best_metric,
                patience=patience,
                extra={"metrics": final_metrics, "prerequisite": prerequisite},
            )
            path = artifact / f"checkpoint_epoch_{epoch:04d}.pth"
            atomic_torch_save(payload, path)
            rotate_latest(artifact, int(config["integration_training"]["keep_latest_checkpoints"]))
            if improved:
                atomic_torch_save(payload, best_path)
            atomic_json_dump(
                _status(
                    stage="string_specialist",
                    state="running",
                    epoch=epoch,
                    global_step=global_step,
                    started_at=started_at,
                    metrics=final_metrics,
                ),
                status_path,
            )
            if (
                epoch >= int(settings["minimum_epochs"])
                and patience >= int(settings["early_stop_patience"])
            ):
                stop_reason = "early_stopping_validation_plateau"
                break
    except BaseException as exc:
        atomic_json_dump(
            _status(
                stage="string_specialist",
                state="error",
                epoch=locals().get("epoch", start_epoch - 1),
                global_step=global_step,
                started_at=started_at,
                metrics={"error": repr(exc)},
            ),
            status_path,
        )
        tracker.finish(exit_code=1)
        raise
    result = {
        "format": "cftn_text_v1_3_string_training_result_v1",
        "state": "completed",
        "stop_reason": stop_reason,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": file_sha256(best_path),
        "final_metrics": final_metrics,
        "revision_sha256": config["_meta"]["sha256"],
    }
    atomic_json_dump(result, artifact / "summary.json")
    atomic_json_dump(
        _status(
            stage="string_specialist",
            state="completed",
            epoch=int(final_metrics["epoch"]),
            global_step=global_step,
            started_at=started_at,
            metrics=final_metrics,
        ),
        status_path,
    )
    tracker.finish()
    return result


def build_v1_3_model(
    config: dict[str, Any],
    *,
    device: torch.device,
    collaboration_checkpoint: str | Path | None = None,
) -> tuple[V13MultiTowerModel, Any, dict[str, Any]]:
    prerequisite = audit_v1_2_pass(config)
    root, manifest = load_v1_3_data_contract(config)
    del root
    base = load_config(config["paths"]["base_config"])
    _, math_manifest = load_data_contract(base)
    math_checkpoint = load_checkpoint(
        config["paths"]["math_checkpoint"],
        expected_stage="math",
        expected_config_sha256=config_sha256(base),
        expected_manifest_sha256=math_manifest["manifest_sha256"],
        map_location=device,
    )
    math_tower = MathTower(base["math_tower"], ByteMathTokenizer.vocab_size)
    math_tower.load_state_dict(math_checkpoint["model_state"], strict=True)
    string_checkpoint_path = (
        Path(config["paths"]["artifact_root"]) / "string_specialist" / "string.best.pth"
    )
    string_checkpoint = load_checkpoint(
        string_checkpoint_path,
        expected_stage="v1_3_string",
        expected_config_sha256=config["_meta"]["sha256"],
        expected_manifest_sha256=manifest["manifest_sha256"],
        map_location=device,
    )
    string_tower = build_string_tower(config)
    string_tower.load_state_dict(string_checkpoint["model_state"], strict=True)
    gpt_tokenizer, gpt_tower = load_gpt_components(
        gpt_interface_config(base, config)
    )
    model = V13MultiTowerModel(
        gpt_tower=gpt_tower,
        specialists={"math": math_tower, "string": string_tower},
        config=config,
    ).to(device)
    if prerequisite.get("bridge_initialization") != (
        "fresh_contextual_bridges_zero_initialized_receivers"
    ):
        v1_2_checkpoint = load_checkpoint(
            prerequisite["v1_2_checkpoint"],
            expected_stage="bidirectional",
            expected_config_sha256=config_sha256(base),
            expected_manifest_sha256=math_manifest["manifest_sha256"],
            map_location=device,
        )
        model.load_v1_2_bridge_state(v1_2_checkpoint["model_state"])
    if collaboration_checkpoint is not None:
        state = load_checkpoint(
            collaboration_checkpoint,
            expected_config_sha256=config["_meta"]["sha256"],
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location=device,
        )
        model.load_collaboration_state_dict(state["model_state"], strict=True)
    return model, gpt_tokenizer, {
        "initialization": prerequisite,
        "math_checkpoint_sha256": file_sha256(config["paths"]["math_checkpoint"]),
        "string_checkpoint_sha256": file_sha256(string_checkpoint_path),
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _per_example_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(shifted_labels.shape)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _preservation_kl(
    current: torch.Tensor, baseline: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    token_mask = labels[:, 1:].ne(-100) & rows.unsqueeze(1)
    if not bool(token_mask.any()):
        return current.sum() * 0.0
    current_tokens = current[:, :-1][token_mask].float()
    baseline_tokens = baseline[:, :-1][token_mask].detach().float()
    return F.kl_div(
        F.log_softmax(current_tokens, dim=-1),
        F.softmax(baseline_tokens, dim=-1),
        reduction="batchmean",
    )


def v1_3_objective(
    model: V13MultiTowerModel,
    batch: dict[str, Any],
    *,
    wake_mode: str,
    maximum_rounds: int,
    settings: dict[str, Any],
    global_step: int,
    objective_mode: str = "auto",
    conditional_execution: bool | None = None,
    apply_halt: bool | None = None,
    task_class_weights: Mapping[str, float] | None = None,
) -> tuple[V13ModelOutput, torch.Tensor, dict[str, float]]:
    configured = settings["losses"]
    routing_calibration_only = (
        objective_mode == "router_calibration"
        or (objective_mode == "auto" and wake_mode == "hard_straight_through")
    )
    oracle_hard_adapter = objective_mode == "oracle_hard_adapter"
    weighted_task_loss = task_class_weights is not None and not routing_calibration_only
    compute_weight = (
        0.0
        if routing_calibration_only
        else float(configured["active_compute_initial"])
    )
    output = model(
        batch,
        wake_mode=wake_mode,
        maximum_rounds=maximum_rounds,
        conditional_execution=conditional_execution,
        apply_halt=apply_halt,
        loss_weights={
            "task": (
                0.0
                if routing_calibration_only or weighted_task_loss
                else float(configured["task"])
            ),
            "specialist": (
                0.0 if routing_calibration_only else float(configured["specialist"])
            ),
            "wake_required_set": (
                float(configured["wake_required_set"])
                if routing_calibration_only or not oracle_hard_adapter
                else 0.0
            ),
            "halt": (
                0.0
                if routing_calibration_only or oracle_hard_adapter
                else float(configured["halt"])
            ),
            "active_compute": compute_weight,
        },
    )
    total = output.loss
    weighted_gpt = output.gpt_loss
    if weighted_task_loss:
        row_weights = torch.tensor(
            [float(task_class_weights.get(value, 1.0)) for value in batch["task_classes"]],
            dtype=torch.float32,
            device=output.gpt_logits.device,
        )
        if bool(row_weights.le(0).any()):
            raise ValueError("task-class loss weights must be positive")
        per_example_gpt = _per_example_loss(output.gpt_logits, batch["gpt_labels"])
        weighted_gpt = (per_example_gpt * row_weights).sum() / row_weights.sum()
        total = total + float(configured["task"]) * weighted_gpt
    preservation = output.loss.detach() * 0.0
    causal_message = output.loss.detach() * 0.0
    causal_wake = output.loss.detach() * 0.0
    auxiliary = (
        not routing_calibration_only
        and global_step % int(settings.get("auxiliary_every_steps", 4)) == 0
    )
    classes = batch["task_classes"]
    pure_rows = torch.tensor(
        [value == "pure_language" for value in classes],
        dtype=torch.bool,
        device=output.gpt_logits.device,
    )
    required_rows = batch["wake_targets"][:, :maximum_rounds].sum(dim=(1, 2)).gt(0)
    if auxiliary and bool(pure_rows.any()):
        with torch.no_grad():
            baseline = model(
                batch,
                wake_mode="oracle",
                maximum_rounds=maximum_rounds,
                conditional_execution=conditional_execution,
                apply_halt=apply_halt,
                disable_all_communication=True,
                loss_weights={"wake_required_set": 0.0, "halt": 0.0},
            )
        preservation = _preservation_kl(
            output.gpt_logits, baseline.gpt_logits, batch["gpt_labels"], pure_rows
        )
        total = total + float(configured["no_harm"]) * preservation
    if auxiliary and bool(required_rows.any()):
        shuffled = model(
            batch,
            wake_mode=wake_mode,
            maximum_rounds=maximum_rounds,
            conditional_execution=conditional_execution,
            apply_halt=apply_halt,
            shuffled_requests=set(SPECIALISTS),
            shuffled_returns=set(SPECIALISTS),
            loss_weights={"wake_required_set": 0.0, "halt": 0.0},
        )
        correct_loss = _per_example_loss(output.gpt_logits, batch["gpt_labels"])
        shuffled_loss = _per_example_loss(shuffled.gpt_logits, batch["gpt_labels"])
        margin = float(configured["causal_message_margin"])
        causal_message = F.relu(
            margin - (shuffled_loss[required_rows] - correct_loss[required_rows])
        ).mean()
        total = total + float(configured["causal_message"]) * causal_message
    if (
        auxiliary
        and (wake_mode in {"soft", "hard_straight_through"} or oracle_hard_adapter)
        and bool(required_rows.any())
    ):
        specialist_index = (global_step // int(settings.get("auxiliary_every_steps", 4))) % len(
            SPECIALISTS
        )
        specialist = SPECIALISTS[specialist_index]
        specialist_rows = batch["wake_targets"][:, :maximum_rounds, specialist_index].sum(dim=1).gt(0)
        if bool(specialist_rows.any()):
            disabled = model(
                batch,
                wake_mode=wake_mode,
                maximum_rounds=maximum_rounds,
                conditional_execution=conditional_execution,
                apply_halt=apply_halt,
                disabled_specialists={specialist},
                loss_weights={"wake_required_set": 0.0, "halt": 0.0},
            )
            correct_loss = _per_example_loss(output.gpt_logits, batch["gpt_labels"])
            disabled_loss = _per_example_loss(disabled.gpt_logits, batch["gpt_labels"])
            margin = float(configured["causal_wake_margin"])
            causal_wake = F.relu(
                margin - (disabled_loss[specialist_rows] - correct_loss[specialist_rows])
            ).mean()
            total = total + float(configured["causal_wake_utility"]) * causal_wake
    return output, total, {
        "model_loss": float(output.loss.detach()),
        "gpt_loss": float(output.gpt_loss.detach()),
        "weighted_gpt_loss": float(weighted_gpt.detach()),
        "specialist_loss": float(output.specialist_loss.detach()),
        "wake_loss": float(output.wake_loss.detach()),
        "halt_loss": float(output.halt_loss.detach()),
        "compute_loss": float(output.compute_loss.detach()),
        "preservation_loss": float(preservation.detach()),
        "causal_message_loss": float(causal_message.detach()),
        "causal_wake_loss": float(causal_wake.detach()),
        "auxiliary_step": float(auxiliary),
        "routing_calibration_only": float(routing_calibration_only),
        "oracle_hard_adapter": float(oracle_hard_adapter),
    }


@torch.no_grad()
def evaluate_joint_teacher_forcing(
    model: V13MultiTowerModel,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    *,
    wake_mode: str,
    maximum_rounds: int,
    causal_batches: int,
    max_batches: int | None = None,
    conditional_execution: bool | None = None,
    apply_halt: bool | None = None,
) -> dict[str, Any]:
    model.eval()
    examples = token_correct = token_total = sequence_correct = 0
    loss_sum = 0.0
    wake_tp = wake_fp = wake_fn = exact_sets = wake_labels = 0
    pre_halt_tp = pre_halt_fp = pre_halt_fn = pre_halt_exact_sets = 0
    wake_predictions = wake_target_positives = all_open_sets = all_closed_sets = 0
    pure_examples = pure_false_wakes = 0
    causal_correct = causal_shuffled = causal_examples = 0.0
    gpt_loss_sum = 0.0
    task_class_sums: dict[str, dict[str, float | int]] = {}
    for batch_index, raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_v1_3_batch(raw, device)
        with autocast_context(device, dtype):
            output = model(
                batch,
                wake_mode=wake_mode,
                maximum_rounds=maximum_rounds,
                conditional_execution=conditional_execution,
                apply_halt=apply_halt,
            )
        count = int(batch["gpt_input_ids"].shape[0])
        examples += count
        loss_sum += float(output.loss) * count
        correct, total, sequences = masked_token_statistics(
            output.gpt_logits, batch["gpt_labels"]
        )
        token_correct += correct
        token_total += total
        sequence_correct += sequences
        per_example_gpt = _per_example_loss(output.gpt_logits, batch["gpt_labels"])
        shifted_labels = batch["gpt_labels"][:, 1:]
        valid_tokens = shifted_labels.ne(-100)
        predicted_tokens = output.gpt_logits[:, :-1].argmax(dim=-1)
        row_token_correct = ((predicted_tokens == shifted_labels) & valid_tokens).sum(dim=1)
        row_token_total = valid_tokens.sum(dim=1)
        row_sequence_correct = (
            ((predicted_tokens == shifted_labels) | ~valid_tokens).all(dim=1)
            & row_token_total.gt(0)
        )
        gpt_loss_sum += float(per_example_gpt.sum())
        for row_index, task_class in enumerate(batch["task_classes"]):
            sums = task_class_sums.setdefault(
                str(task_class),
                {
                    "examples": 0,
                    "token_correct": 0,
                    "token_total": 0,
                    "sequence_correct": 0,
                    "gpt_loss_sum": 0.0,
                },
            )
            sums["examples"] = int(sums["examples"]) + 1
            sums["token_correct"] = int(sums["token_correct"]) + int(
                row_token_correct[row_index]
            )
            sums["token_total"] = int(sums["token_total"]) + int(
                row_token_total[row_index]
            )
            sums["sequence_correct"] = int(sums["sequence_correct"]) + int(
                row_sequence_correct[row_index]
            )
            sums["gpt_loss_sum"] = float(sums["gpt_loss_sum"]) + float(
                per_example_gpt[row_index]
            )
        logits = torch.stack([item.wake_logits for item in output.rounds], dim=1)
        pre_halt_predicted = torch.sigmoid(logits).ge(model.wake_threshold)
        predicted = (
            torch.stack(
                [item.wake_activations for item in output.rounds], dim=1
            ).ge(model.wake_threshold)
            if wake_mode in {"hard", "hard_straight_through"}
            else pre_halt_predicted
        )
        targets = batch["wake_targets"][:, :maximum_rounds].bool()
        reachable = (
            batch["halt_targets"][:, :maximum_rounds]
            .ge(0)
            .unsqueeze(-1)
            .expand_as(targets)
        )
        pre_halt_tp += int((pre_halt_predicted & targets & reachable).sum())
        pre_halt_fp += int((pre_halt_predicted & ~targets & reachable).sum())
        pre_halt_fn += int((~pre_halt_predicted & targets & reachable).sum())
        pre_halt_exact_sets += int(
            (pre_halt_predicted.eq(targets) | ~reachable).all(dim=(1, 2)).sum()
        )
        wake_tp += int((predicted & targets & reachable).sum())
        wake_fp += int((predicted & ~targets & reachable).sum())
        wake_fn += int((~predicted & targets & reachable).sum())
        wake_predictions += int((predicted & reachable).sum())
        wake_target_positives += int((targets & reachable).sum())
        exact_sets += int(
            (predicted.eq(targets) | ~reachable).all(dim=(1, 2)).sum()
        )
        all_open_sets += int((predicted | ~reachable).all(dim=(1, 2)).sum())
        all_closed_sets += int(((~predicted) | ~reachable).all(dim=(1, 2)).sum())
        wake_labels += int(reachable.sum())
        pure_mask = torch.tensor(
            [value == "pure_language" for value in batch["task_classes"]],
            device=device,
            dtype=torch.bool,
        )
        pure_examples += int(pure_mask.sum())
        if bool(pure_mask.any()):
            pure_false_wakes += int(
                (predicted[pure_mask] & reachable[pure_mask]).any(dim=(1, 2)).sum()
            )
        if batch_index < causal_batches:
            required = targets.any(dim=(1, 2))
            if bool(required.any()):
                with autocast_context(device, dtype):
                    shuffled = model(
                        batch,
                        wake_mode=wake_mode,
                        maximum_rounds=maximum_rounds,
                        conditional_execution=conditional_execution,
                        apply_halt=apply_halt,
                        shuffled_requests=set(SPECIALISTS),
                        shuffled_returns=set(SPECIALISTS),
                    )
                correct_losses = _per_example_loss(output.gpt_logits, batch["gpt_labels"])
                shuffled_losses = _per_example_loss(shuffled.gpt_logits, batch["gpt_labels"])
                causal_correct += float(correct_losses[required].sum())
                causal_shuffled += float(shuffled_losses[required].sum())
                causal_examples += int(required.sum())
    if not examples:
        raise RuntimeError("V1.3 validation produced no examples")
    precision = wake_tp / max(1, wake_tp + wake_fp)
    recall = wake_tp / max(1, wake_tp + wake_fn)
    pre_halt_precision = pre_halt_tp / max(1, pre_halt_tp + pre_halt_fp)
    pre_halt_recall = pre_halt_tp / max(1, pre_halt_tp + pre_halt_fn)
    task_class_metrics = {
        name: {
            "examples": int(sums["examples"]),
            "gpt_loss": float(sums["gpt_loss_sum"]) / max(1, int(sums["examples"])),
            "sequence_accuracy": int(sums["sequence_correct"])
            / max(1, int(sums["examples"])),
            "token_accuracy": int(sums["token_correct"])
            / max(1, int(sums["token_total"])),
        }
        for name, sums in sorted(task_class_sums.items())
    }
    return {
        "examples": examples,
        "loss": loss_sum / examples,
        "gpt_teacher_forced_token_accuracy": token_correct / max(1, token_total),
        "gpt_teacher_forced_sequence_accuracy": sequence_correct / examples,
        "gpt_teacher_forced_loss": gpt_loss_sum / examples,
        "task_class_metrics": task_class_metrics,
        "wake_precision": precision,
        "wake_recall": recall,
        "wake_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "exact_required_set_accuracy": exact_sets / examples,
        "pre_halt_wake_precision": pre_halt_precision,
        "pre_halt_wake_recall": pre_halt_recall,
        "pre_halt_wake_f1": (
            2
            * pre_halt_precision
            * pre_halt_recall
            / max(1e-12, pre_halt_precision + pre_halt_recall)
        ),
        "pre_halt_exact_required_set_accuracy": pre_halt_exact_sets / examples,
        "pure_language_false_wake_rate": pure_false_wakes / max(1, pure_examples),
        "wake_positive_rate": wake_predictions / max(1, wake_labels),
        "wake_target_positive_rate": wake_target_positives / max(1, wake_labels),
        "all_open_rate": all_open_sets / examples,
        "all_closed_rate": all_closed_sets / examples,
        "causal_message_loss_gap": (
            (causal_shuffled - causal_correct) / max(1, causal_examples)
        ),
        "causal_panel_examples": int(causal_examples),
        "wake_labels": wake_labels,
    }


def _phase(config: dict[str, Any], name: str) -> tuple[int, dict[str, Any]]:
    for index, value in enumerate(config["integration_training"]["phases"]):
        if value["name"] == name:
            return index, value
    raise ValueError(f"unknown integration phase: {name}")


def _repair_sequential_orders(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive balanced sequential orders without mutating the sealed manifest."""

    repaired: list[dict[str, Any]] = []
    source_sequential = 0
    for index, record in enumerate(records):
        if record["task_class"] != "multi_sequential":
            repaired.append(record)
            continue
        source_sequential += 1
        replacement = generate_joint_record(
            seed=int(config["revision"]["seed"]),
            split=split,
            index=index,
            config=config,
        )
        if replacement["task_class"] != "multi_sequential":
            raise RuntimeError("sequential recovery regeneration changed task class")
        repaired.append(replacement)
    orders: dict[str, int] = {}
    for record in repaired:
        if record["task_class"] == "multi_sequential":
            order = str(record["metadata"].get("sequential_order"))
            orders[order] = orders.get(order, 0) + 1
    if source_sequential and (
        set(orders) != {"string_then_math", "math_then_string"}
        or abs(orders["string_then_math"] - orders["math_then_string"]) > 1
    ):
        raise RuntimeError(f"sequential recovery data is not balanced: {orders}")
    digest = hashlib.sha256(
        "\n".join(str(record["record_id"]) for record in repaired).encode("utf-8")
    ).hexdigest()
    return repaired, {
        "format": "cftn_text_v1_3_recovery_data_derivation_v1",
        "split": split,
        "source_examples": len(records),
        "source_sequential_examples": source_sequential,
        "sequential_orders": orders,
        "derived_record_ids_sha256": digest,
    }


def routing_recovery_acceptance(
    validation: dict[str, Any], guard: Mapping[str, Any]
) -> dict[str, Any]:
    gates = {
        "pure_language_false_wake": float(
            validation["pure_language_false_wake_rate"]
        )
        <= float(guard["maximum_pure_language_false_wake_rate"]),
        "exact_required_set": float(validation["exact_required_set_accuracy"])
        >= float(guard["minimum_exact_required_set_accuracy"]),
        "wake_precision": float(validation["wake_precision"])
        >= float(guard["minimum_wake_precision"]),
        "wake_recall": float(validation["wake_recall"])
        >= float(guard["minimum_wake_recall"]),
        "not_always_open": float(validation["all_open_rate"])
        <= float(guard["maximum_all_open_rate"]),
        "not_always_closed": float(validation["all_closed_rate"])
        <= float(guard["maximum_all_closed_rate"]),
    }
    gates["pass"] = all(gates.values())
    return {
        "gates": gates,
        "collapse_guard": {
            "triggered": not gates["pass"],
            "failed": sorted(name for name, passed in gates.items() if not passed),
        },
    }


def _previous_phase_checkpoint(config: dict[str, Any], phase_index: int) -> Path | None:
    if phase_index == 0:
        return None
    previous = config["integration_training"]["phases"][phase_index - 1]["name"]
    summary = Path(config["paths"]["artifact_root"]) / previous / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(f"previous V1.3 phase is incomplete: {summary}")
    value = json.loads(summary.read_text(encoding="utf-8"))
    if value.get("state") != "completed":
        raise RuntimeError(f"previous V1.3 phase did not complete: {previous}")
    return Path(value["best_checkpoint"])


def _hard_transition_baseline_path(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["artifact_root"]) / "hard_transition_baseline" / "report.json"


def _load_hard_transition_baseline(config: dict[str, Any]) -> dict[str, Any]:
    path = _hard_transition_baseline_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"zero-update hard-transition baseline is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("state") != "completed" or int(value.get("optimizer_updates", -1)) != 0:
        raise RuntimeError("hard-transition baseline is incomplete or contains updates")
    if value.get("full_validation") is not True:
        raise RuntimeError("hard-transition baseline used only a partial validation panel")
    if value.get("revision_sha256") != config["_meta"]["sha256"]:
        raise RuntimeError("hard-transition baseline revision hash mismatch")
    source = Path(str(value.get("source_checkpoint", "")))
    if not source.is_file() or file_sha256(source) != value.get(
        "source_checkpoint_sha256"
    ):
        raise RuntimeError("hard-transition baseline source checkpoint changed")
    return value


def hardening_acceptance(
    validation: dict[str, Any],
    baseline: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on the routing collapse observed in the first V1.3 hard run."""

    guard = settings["hardening_guard"]
    baseline_metrics = baseline["hard_metrics"]
    gates = {
        "pure_language_false_wake": float(
            validation["pure_language_false_wake_rate"]
        )
        <= float(guard["maximum_pure_language_false_wake_rate"]),
        "exact_required_set": float(validation["exact_required_set_accuracy"])
        >= float(guard["minimum_exact_required_set_accuracy"]),
        "wake_precision": float(validation["wake_precision"])
        >= float(guard["minimum_wake_precision"]),
        "wake_recall": float(validation["wake_recall"])
        >= float(guard["minimum_wake_recall"]),
        "sequence_no_regression": float(
            validation["gpt_teacher_forced_sequence_accuracy"]
        )
        >= float(baseline_metrics["gpt_teacher_forced_sequence_accuracy"])
        - float(guard["maximum_sequence_accuracy_drop_from_zero_update_baseline"]),
        "exact_set_no_regression": float(validation["exact_required_set_accuracy"])
        >= float(baseline_metrics["exact_required_set_accuracy"])
        - float(guard["maximum_exact_set_drop_from_zero_update_baseline"]),
        "not_always_open": float(validation["all_open_rate"])
        <= float(guard["maximum_all_open_rate"]),
        "not_always_closed": float(validation["all_closed_rate"])
        <= float(guard["maximum_all_closed_rate"]),
    }
    gates["pass"] = all(gates.values())
    return {
        "gates": gates,
        "collapse_guard": {
            "triggered": not gates["pass"],
            "failed": sorted(name for name, passed in gates.items() if not passed),
            "zero_update_baseline": baseline_metrics,
        },
    }


def _load_gpt_language_calibration(config: dict[str, Any]) -> dict[str, Any]:
    path = (
        Path(config["paths"]["artifact_root"])
        / "gpt_language_calibration"
        / "report.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"GPT language calibration is missing: {path}")
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("state") != "passed" or calibration.get("pass") is not True:
        raise RuntimeError("GPT language calibration did not pass")
    if calibration.get("revision_sha256") != config["_meta"]["sha256"]:
        raise RuntimeError("GPT language calibration revision hash mismatch")
    if calibration.get("answer_protocol") != "first_nonempty_completion_line_v1":
        raise RuntimeError("unsupported GPT language answer protocol")
    return calibration


def apply_protocol_aware_adapter_metrics(
    validation: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    """Replace the known strict-format pure-language diagnostic with its semantic gate.

    Non-pure classes remain strict teacher-forced measurements, so the combined value is
    deliberately labelled as a lower bound rather than a generation accuracy estimate.
    """

    class_metrics = validation.get("task_class_metrics", {})
    pure = class_metrics.get("pure_language")
    if not isinstance(pure, dict):
        raise RuntimeError("protocol-aware validation requires pure_language class metrics")
    pure_examples = int(pure["examples"])
    if int(calibration.get("examples", -1)) != pure_examples:
        raise RuntimeError(
            "GPT language calibration panel does not match validation pure-language rows"
        )
    examples = int(validation["examples"])
    strict_correct = (
        float(validation["gpt_teacher_forced_sequence_accuracy"]) * examples
    )
    pure_strict_correct = float(pure["sequence_accuracy"]) * pure_examples
    pure_semantic_accuracy = float(calibration["semantic_accuracy"])
    protocol_correct = (
        strict_correct - pure_strict_correct + pure_semantic_accuracy * pure_examples
    )
    enriched = dict(validation)
    enriched["protocol_semantic_sequence_accuracy_lower_bound"] = (
        protocol_correct / max(1, examples)
    )
    enriched["protocol_aware"] = {
        "answer_protocol": calibration["answer_protocol"],
        "pure_language_examples": pure_examples,
        "pure_language_semantic_accuracy": pure_semantic_accuracy,
        "pure_language_strict_sequence_accuracy_diagnostic": float(
            pure["sequence_accuracy"]
        ),
        "calibration_pass": True,
        "calibration_revision_sha256": calibration["revision_sha256"],
    }
    return enriched


def adapter_recovery_acceptance(
    validation: dict[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    class_metrics = validation["task_class_metrics"]
    protected = settings.get(
        "protected_task_classes",
        {
            "explicit_math": 0.95,
            "language_dependent_math": 0.95,
            "multi_sequential": 0.95,
        },
    )
    gates = {
        "protocol_semantic_floor": float(
            validation["protocol_semantic_sequence_accuracy_lower_bound"]
        )
        >= float(settings.get("minimum_protocol_semantic_sequence_accuracy", 0.83)),
        "causal_message_floor": float(validation["causal_message_loss_gap"])
        >= float(settings.get("minimum_causal_message_loss_gap", 5.0)),
        "pure_language_calibration": validation.get("protocol_aware", {}).get(
            "calibration_pass"
        )
        is True,
    }
    for task_class, minimum in protected.items():
        metrics = class_metrics.get(task_class)
        gates[f"protected_{task_class}"] = bool(metrics) and float(
            metrics["sequence_accuracy"]
        ) >= float(minimum)
    gates["pass"] = all(gates.values())
    return {
        "gates": gates,
        "collapse_guard": {
            "triggered": not gates["pass"],
            "failed": sorted(name for name, passed in gates.items() if not passed),
        },
    }


def integration_selection_score(
    validation: dict[str, Any],
    *,
    hardening: dict[str, Any] | None = None,
    selection_mode: str = "legacy",
    focus_classes: list[str] | tuple[str, ...] | None = None,
) -> float:
    if selection_mode == "adapter_recovery":
        focus = tuple(focus_classes or ("exact_string", "multi_parallel"))
        class_metrics = validation.get("task_class_metrics", {})
        missing = sorted(name for name in focus if name not in class_metrics)
        if missing:
            raise RuntimeError(f"adapter selection is missing focus classes: {missing}")
        focus_sequence = sum(
            float(class_metrics[name]["sequence_accuracy"]) for name in focus
        ) / len(focus)
        focus_token = sum(
            float(class_metrics[name]["token_accuracy"]) for name in focus
        ) / len(focus)
        return (
            float(validation["protocol_semantic_sequence_accuracy_lower_bound"])
            + 0.20 * focus_sequence
            + 0.10 * focus_token
            + 0.05 * float(validation["gpt_teacher_forced_token_accuracy"])
            - 0.001 * float(validation["gpt_teacher_forced_loss"])
        )
    if selection_mode != "legacy":
        raise ValueError(f"unknown integration selection mode: {selection_mode}")
    if hardening is None:
        return (
            float(validation["gpt_teacher_forced_sequence_accuracy"])
            + 0.10 * float(validation["wake_f1"])
            + 0.05 * max(0.0, float(validation["causal_message_loss_gap"]))
            - 1.0e-5 * float(validation["loss"])
        )
    return (
        float(validation["gpt_teacher_forced_sequence_accuracy"])
        + 0.25 * float(validation["gpt_teacher_forced_token_accuracy"])
        + 0.25 * float(validation["exact_required_set_accuracy"])
        + 0.10 * float(validation["wake_precision"])
        + 0.10 * float(validation["wake_recall"])
        + 0.05 * max(0.0, float(validation["causal_message_loss_gap"]))
        - 0.25 * float(validation["pure_language_false_wake_rate"])
        - 1.0e-5 * float(validation["loss"])
    )


@torch.no_grad()
def evaluate_hard_transition_baseline(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    max_batches: int | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure thresholding alone from the selected soft checkpoint."""

    audit_v1_2_pass(config)
    transition = config["integration_training"]["hard_transition_baseline"]
    source_phase = str(transition["source_phase"])
    _, source_definition = _phase(config, source_phase)
    _, hard_definition = _phase(config, "hardened_wake")
    source_summary_path = (
        Path(config["paths"]["artifact_root"]) / source_phase / "summary.json"
    )
    if not source_summary_path.is_file():
        raise FileNotFoundError(
            f"soft-wake source summary is missing: {source_summary_path}"
        )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("state") != "completed":
        raise RuntimeError("soft-wake source phase is not complete")
    source_checkpoint = Path(source_summary["best_checkpoint"])
    source_sha256 = file_sha256(source_checkpoint)
    if source_sha256 != source_summary.get("best_checkpoint_sha256"):
        raise RuntimeError("soft-wake source checkpoint hash mismatch")

    seed = int(config["revision"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=source_checkpoint
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    allowed_classes = set(hard_definition["include_task_classes"])
    validation_records = [
        record
        for record in V13Dataset(
            data_root / manifest["splits"]["joint_validation"]["path"]
        ).records
        if record["task_class"] in allowed_classes
    ]
    collator = V13JointCollator(
        ByteMathTokenizer(),
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    settings = config["integration_training"]
    loader = _loader(
        V13Dataset(validation_records),
        collator,
        batch_size=int(settings["eval_batch_size"]),
        shuffle=False,
        seed=seed,
        epoch=0,
        num_workers=int(settings["num_workers"]),
    )
    causal_batches = math.ceil(
        int(settings["validation_causal_examples"])
        / int(settings["eval_batch_size"])
    )
    dtype = precision_dtype(settings["precision"], device)
    started_at = time.time()
    soft_metrics = evaluate_joint_teacher_forcing(
        model,
        loader,
        device,
        dtype,
        wake_mode=str(source_definition["wake_mode"]),
        maximum_rounds=int(source_definition["maximum_rounds"]),
        causal_batches=causal_batches,
        max_batches=max_batches,
        conditional_execution=False,
        apply_halt=False,
    )
    hard_metrics = evaluate_joint_teacher_forcing(
        model,
        loader,
        device,
        dtype,
        wake_mode=str(transition["wake_mode"]),
        maximum_rounds=int(hard_definition["maximum_rounds"]),
        causal_batches=causal_batches,
        max_batches=max_batches,
        conditional_execution=True,
        apply_halt=False,
    )
    if file_sha256(source_checkpoint) != source_sha256:
        raise RuntimeError("zero-update baseline modified its source checkpoint")
    artifact = _hard_transition_baseline_path(config).parent
    artifact.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "cftn_text_hard_transition_baseline_v1",
        "state": "completed",
        "optimizer_updates": 0,
        "trainable_parameters": 0,
        "full_validation": max_batches is None
        and int(soft_metrics["examples"]) == len(validation_records)
        and int(hard_metrics["examples"]) == len(validation_records),
        "validation_examples_expected": len(validation_records),
        "maximum_batches": max_batches,
        "source_phase": source_phase,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "soft_metrics": soft_metrics,
        "hard_metrics": hard_metrics,
        "hard_runtime_policy": {
            "conditional_specialist_execution": True,
            "apply_halt": False,
            "halt_gate_role": "diagnostic_only_until_separately_calibrated",
        },
        "thresholding_delta": {
            "sequence_accuracy": float(
                hard_metrics["gpt_teacher_forced_sequence_accuracy"]
            )
            - float(soft_metrics["gpt_teacher_forced_sequence_accuracy"]),
            "token_accuracy": float(hard_metrics["gpt_teacher_forced_token_accuracy"])
            - float(soft_metrics["gpt_teacher_forced_token_accuracy"]),
            "exact_required_set_accuracy": float(
                hard_metrics["exact_required_set_accuracy"]
            )
            - float(soft_metrics["exact_required_set_accuracy"]),
        },
        "elapsed_seconds": time.time() - started_at,
        "revision_sha256": config["_meta"]["sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "provenance": provenance,
        "gpu": gpu_status(),
    }
    atomic_json_dump(report, _hard_transition_baseline_path(config))
    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact,
        stage="hard_transition_baseline",
        config={"revision_sha256": config["_meta"]["sha256"]},
    )
    tracker.log(report, global_step=0, epoch=0, event="zero_update_hard_baseline")
    tracker.update_summary({"run/state": "completed", "baseline": report})
    tracker.finish()
    return report


def train_integration_phase(
    config: dict[str, Any],
    phase_name: str,
    *,
    device_name: str = "cuda",
    resume: bool = False,
    max_batches: int | None = None,
    wandb_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit_v1_2_pass(config)
    phase_index, phase = _phase(config, phase_name)
    objective_mode = str(phase.get("objective_mode", "auto"))
    router_recovery = objective_mode == "router_calibration"
    selection_mode = str(phase.get("selection_mode", "legacy"))
    adapter_selection = selection_mode == "adapter_recovery"
    conditional_execution = phase.get("conditional_execution")
    apply_halt = phase.get("apply_halt")
    baseline_required = bool(
        config["integration_training"]
        .get("hard_transition_baseline", {})
        .get("required", False)
    )
    hardening_baseline = (
        _load_hard_transition_baseline(config)
        if phase_name == "hardened_wake" and baseline_required
        else None
    )
    seed = int(config["revision"]["seed"])
    seed_everything(seed + phase_index)
    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    configured_source = phase.get("source_checkpoint")
    previous = (
        Path(str(configured_source))
        if configured_source is not None
        else _previous_phase_checkpoint(config, phase_index)
    )
    if previous is not None and not previous.is_file():
        raise FileNotFoundError(f"V1.3 phase source checkpoint is missing: {previous}")
    expected_source_sha256 = phase.get("source_checkpoint_sha256")
    if (
        previous is not None
        and expected_source_sha256 is not None
        and file_sha256(previous) != str(expected_source_sha256)
    ):
        raise RuntimeError("V1.3 phase source checkpoint hash changed")
    if hardening_baseline is not None:
        if previous is None or previous.resolve() != Path(
            hardening_baseline["source_checkpoint"]
        ).resolve():
            raise RuntimeError(
                "hardened_wake source differs from the zero-update baseline source"
            )
        if file_sha256(previous) != hardening_baseline["source_checkpoint_sha256"]:
            raise RuntimeError("hardened_wake source checkpoint hash changed")
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=previous
    )
    if previous is not None:
        provenance["phase_source_checkpoint"] = str(previous.resolve())
        provenance["phase_source_checkpoint_sha256"] = file_sha256(previous)
    model.set_trainable_phase(phase_name)
    settings = config["integration_training"]
    phase_num_workers = int(phase.get("num_workers", settings["num_workers"]))
    if phase_num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    wake_mode = str(phase["wake_mode"]).replace("oracle_dense", "oracle")
    maximum_rounds = int(phase["maximum_rounds"])
    allowed_classes = set(phase["include_task_classes"])
    train_classes = set(phase.get("train_task_classes", allowed_classes))
    if not train_classes.issubset(allowed_classes):
        raise ValueError("train_task_classes must be a subset of include_task_classes")
    task_class_weights = phase.get("task_class_weights")
    calibration = _load_gpt_language_calibration(config) if adapter_selection else None
    all_train_records = V13Dataset(
        data_root / manifest["splits"]["joint_train"]["path"]
    ).records
    all_validation_records = V13Dataset(
        data_root / manifest["splits"]["joint_validation"]["path"]
    ).records
    recovery_data: dict[str, Any] | None = None
    if bool(phase.get("repair_sequential_orders", False)):
        all_train_records, train_repair = _repair_sequential_orders(
            all_train_records, config=config, split="joint_train"
        )
        all_validation_records, validation_repair = _repair_sequential_orders(
            all_validation_records, config=config, split="joint_validation"
        )
        recovery_data = {"train": train_repair, "validation": validation_repair}
        provenance["recovery_data"] = recovery_data
    train_records = [
        record for record in all_train_records if record["task_class"] in train_classes
    ]
    validation_records = [
        record
        for record in all_validation_records
        if record["task_class"] in allowed_classes
    ]
    if not train_records or not validation_records:
        raise RuntimeError("V1.3 phase class filtering produced an empty split")
    collator = V13JointCollator(
        ByteMathTokenizer(),
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    artifact = Path(config["paths"]["artifact_root"]) / phase_name
    artifact.mkdir(parents=True, exist_ok=True)
    status_path = artifact / "status.json"
    metrics_path = artifact / "metrics.jsonl"
    best_path = artifact / f"{phase_name}.best.pth"
    started_at = time.time()
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not named:
        raise RuntimeError(f"{phase_name} has no trainable parameters")
    if phase_name == "hardened_wake" or router_recovery:
        unexpected = sorted(
            name
            for name, _ in named
            if not (
                name.startswith("wake_gates.")
                or (router_recovery and name.startswith("wake_round_embeddings."))
            )
        )
        if unexpected:
            raise RuntimeError(
                "router calibration must be gate-only; unexpected trainable parameters: "
                f"{unexpected}"
            )
        if any(name.startswith("halt_gate.") for name, _ in named):
            raise RuntimeError("router calibration must keep the halt gate frozen")
    gate_parameters = [
        value
        for name, value in named
        if "gate" in name
        or name.startswith("wake_gates")
        or name.startswith("wake_round_embeddings")
        or name.startswith("halt_gate")
    ]
    gate_ids = {id(value) for value in gate_parameters}
    other_parameters = [value for _, value in named if id(value) not in gate_ids]
    phase_learning_rate = float(phase.get("learning_rate", settings["learning_rate"]))
    gate_learning_rate = float(
        phase.get(
            "gate_learning_rate",
            phase_learning_rate
            if (phase_name == "hardened_wake" or router_recovery)
            and "learning_rate" in phase
            else phase_learning_rate * float(settings["gate_learning_rate_multiplier"]),
        )
    )
    optimizer_groups: list[dict[str, Any]] = []
    if other_parameters:
        optimizer_groups.append(
            {
                "params": other_parameters,
                "lr": phase_learning_rate,
                "group_name": "bridges_and_receivers",
            }
        )
    if gate_parameters:
        optimizer_groups.append(
            {
                "params": gate_parameters,
                "lr": gate_learning_rate,
                "group_name": "gates",
            }
        )
    if (phase_name == "hardened_wake" or router_recovery) and [
        group["group_name"] for group in optimizer_groups
    ] != ["gates"]:
        raise RuntimeError("router optimizer must contain only the gates group")
    hardening_policy = (
        {
            "objective": "wake_required_set_only",
            "trainable_components": (
                ["wake_gates", "wake_round_embeddings"]
                if router_recovery
                else ["wake_gates"]
            ),
            "halt_gate_trainable": False,
            "hard_halt_enabled": False,
            "conditional_specialist_execution": True,
        }
        if phase_name == "hardened_wake" or router_recovery
        else None
    )
    optimizer = AdamW(
        optimizer_groups,
        weight_decay=float(phase.get("weight_decay", settings["weight_decay"])),
    )
    steps_per_epoch = max(1, math.ceil(len(train_records) / int(settings["batch_size"])))
    total_steps = int(phase["max_epochs"]) * steps_per_epoch
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=float(phase.get("warmup_fraction", settings["warmup_fraction"])),
        minimum_ratio=float(
            phase.get("minimum_learning_rate", settings["minimum_learning_rate"])
        )
        / max(phase_learning_rate, gate_learning_rate),
    )
    dtype = precision_dtype(settings["precision"], device)
    scaler = make_scaler(device, dtype)
    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    patience = 0
    eligible_best_exists = False
    best_metrics: dict[str, Any] = {}
    if hardening_baseline is not None and not resume:
        baseline_acceptance = hardening_acceptance(
            hardening_baseline["hard_metrics"], hardening_baseline, settings
        )
        if baseline_acceptance["gates"]["pass"] is True:
            best_metric = integration_selection_score(
                hardening_baseline["hard_metrics"],
                hardening=baseline_acceptance,
            )
            eligible_best_exists = True
            best_metrics = {
                "epoch": 0,
                "global_step": 0,
                "phase": phase_name,
                "wake_mode": wake_mode,
                "maximum_rounds": maximum_rounds,
                "train": None,
                "validation": hardening_baseline["hard_metrics"],
                "selection_metric": best_metric,
                "checkpoint_eligible": True,
                "hardening_acceptance": baseline_acceptance,
                "best_metric": best_metric,
                "patience": 0,
                "learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "trainable_parameters": model.trainable_parameter_count(),
                "trainable_parameter_names": [name for name, _ in named],
                "optimizer_group_names": [
                    str(group.get("group_name")) for group in optimizer.param_groups
                ],
                "candidate_source": "zero_update_hard_baseline",
                "hardening_policy": hardening_policy,
            }
            baseline_payload = build_checkpoint(
                stage=f"v1_3_{phase_name}",
                epoch=0,
                global_step=0,
                model_state=model.collaboration_state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                config_sha256=config["_meta"]["sha256"],
                manifest_sha256=manifest["manifest_sha256"],
                best_metric=best_metric,
                patience=0,
                extra={
                    "metrics": best_metrics,
                    "best_metrics": best_metrics,
                    "provenance": provenance,
                    "zero_update_candidate": True,
                },
            )
            atomic_torch_save(baseline_payload, artifact / "checkpoint_epoch_0000.pth")
            atomic_torch_save(baseline_payload, best_path)
    if resume and latest_checkpoint(artifact):
        checkpoint = load_checkpoint(
            latest_checkpoint(artifact),
            expected_stage=f"v1_3_{phase_name}",
            expected_config_sha256=config["_meta"]["sha256"],
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location=device,
        )
        model.load_collaboration_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler.is_enabled() and checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        patience = int(checkpoint["patience"])
        eligible_best_exists = math.isfinite(best_metric) and best_path.is_file()
        best_metrics = dict(checkpoint.get("extra", {}).get("best_metrics", {}))
    tracker = initialize_wandb(
        wandb_options,
        artifact_dir=artifact,
        stage=f"v1_3_{phase_name}",
        config={
            "revision_sha256": config["_meta"]["sha256"],
            "wake_mode": wake_mode,
            "maximum_rounds": maximum_rounds,
        },
    )
    final_metrics: dict[str, Any] = dict(best_metrics)
    try:
        for epoch in range(start_epoch, int(phase["max_epochs"]) + 1):
            epoch_started = time.time()
            model.train()
            component_sums: dict[str, float] = {}
            examples = 0
            train_loader = _loader(
                V13Dataset(train_records),
                collator,
                batch_size=int(settings["batch_size"]),
                shuffle=True,
                seed=seed + phase_index,
                epoch=epoch,
                num_workers=phase_num_workers,
            )
            for batch_index, raw in enumerate(train_loader, start=1):
                if max_batches is not None and batch_index > max_batches:
                    break
                batch = move_v1_3_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, dtype):
                    output, loss, components = v1_3_objective(
                        model,
                        batch,
                        wake_mode=wake_mode,
                        maximum_rounds=maximum_rounds,
                        settings=settings,
                        global_step=global_step,
                        objective_mode=objective_mode,
                        conditional_execution=conditional_execution,
                        apply_halt=apply_halt,
                        task_class_weights=task_class_weights,
                    )
                del output
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [value for _, value in named], float(settings["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                count = int(batch["gpt_input_ids"].shape[0])
                examples += count
                components["total_loss"] = float(loss.detach())
                for key, value in components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + value * count
                if global_step % int(settings["report_every_steps"]) == 0:
                    progress = {
                        "epoch_batch_completed": batch_index,
                        "epoch_batches_total": min(len(train_loader), max_batches or len(train_loader)),
                        "train": {
                            key: value / max(1, examples)
                            for key, value in component_sums.items()
                        },
                        "learning_rates": {
                            str(group.get("group_name", index)): float(group["lr"])
                            for index, group in enumerate(optimizer.param_groups)
                        },
                        "hardening_policy": hardening_policy,
                    }
                    atomic_json_dump(
                        _status(
                            stage=phase_name,
                            state="running",
                            epoch=epoch,
                            global_step=global_step,
                            started_at=started_at,
                            metrics=progress,
                        ),
                        status_path,
                    )
                    tracker.log(progress, global_step=global_step, epoch=epoch, event="training_progress")
            validation_loader = _loader(
                V13Dataset(validation_records),
                collator,
                batch_size=int(settings["eval_batch_size"]),
                shuffle=False,
                seed=seed,
                epoch=0,
                num_workers=phase_num_workers,
            )
            causal_batches = math.ceil(
                int(settings["validation_causal_examples"])
                / int(settings["eval_batch_size"])
            )
            validation = evaluate_joint_teacher_forcing(
                model,
                validation_loader,
                device,
                dtype,
                wake_mode=wake_mode,
                maximum_rounds=maximum_rounds,
                causal_batches=causal_batches,
                max_batches=max_batches,
                conditional_execution=(
                    conditional_execution
                    if conditional_execution is not None
                    else (True if phase_name == "hardened_wake" else None)
                ),
                apply_halt=(
                    apply_halt
                    if apply_halt is not None
                    else (False if phase_name == "hardened_wake" else None)
                ),
            )
            if calibration is not None:
                validation = apply_protocol_aware_adapter_metrics(
                    validation, calibration
                )
            hardening = (
                hardening_acceptance(validation, hardening_baseline, settings)
                if hardening_baseline is not None
                else (
                    routing_recovery_acceptance(
                        validation, phase["routing_acceptance"]
                    )
                    if router_recovery
                    else (
                        adapter_recovery_acceptance(
                            validation, phase.get("adapter_acceptance", {})
                        )
                        if adapter_selection
                        else None
                    )
                )
            )
            selection = integration_selection_score(
                validation,
                hardening=hardening,
                selection_mode=selection_mode,
                focus_classes=phase.get("selection_focus_classes"),
            )
            checkpoint_eligible = hardening is None or hardening["gates"]["pass"] is True
            improved = checkpoint_eligible and selection > best_metric
            patience = 0 if improved else patience + 1
            if improved:
                best_metric = selection
                eligible_best_exists = True
            final_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "phase": phase_name,
                "wake_mode": wake_mode,
                "maximum_rounds": maximum_rounds,
                "train": {
                    key: value / max(1, examples) for key, value in component_sums.items()
                },
                "validation": validation,
                "selection_metric": selection,
                "checkpoint_eligible": checkpoint_eligible,
                "hardening_acceptance": hardening,
                "best_metric": best_metric,
                "patience": patience,
                "learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "trainable_parameters": model.trainable_parameter_count(),
                "trainable_parameter_names": [name for name, _ in named],
                "optimizer_group_names": [
                    str(group.get("group_name")) for group in optimizer.param_groups
                ],
                "hardening_policy": hardening_policy,
                "objective_mode": objective_mode,
                "selection_mode": selection_mode,
                "train_task_classes": sorted(train_classes),
                "validation_task_classes": sorted(allowed_classes),
                "task_class_weights": task_class_weights,
                "num_workers": phase_num_workers,
                "recovery_data": recovery_data,
                "timing": {
                    "epoch_seconds": time.time() - epoch_started,
                    "eta_seconds_to_phase_end": (
                        int(phase["max_epochs"]) - epoch
                    )
                    * (time.time() - epoch_started),
                },
                "gpu": gpu_status(),
            }
            if improved:
                best_metrics = dict(final_metrics)
            append_jsonl(final_metrics, metrics_path)
            tracker.log(final_metrics, global_step=global_step, epoch=epoch, event="epoch_validation")
            payload = build_checkpoint(
                stage=f"v1_3_{phase_name}",
                epoch=epoch,
                global_step=global_step,
                model_state=model.collaboration_state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                config_sha256=config["_meta"]["sha256"],
                manifest_sha256=manifest["manifest_sha256"],
                best_metric=best_metric,
                patience=patience,
                extra={
                    "metrics": final_metrics,
                    "best_metrics": best_metrics,
                    "provenance": provenance,
                },
            )
            checkpoint_path = artifact / f"checkpoint_epoch_{epoch:04d}.pth"
            atomic_torch_save(payload, checkpoint_path)
            rotate_latest(artifact, int(settings["keep_latest_checkpoints"]))
            if improved:
                atomic_torch_save(payload, best_path)
            atomic_json_dump(
                _status(
                    stage=phase_name,
                    state="running",
                    epoch=epoch,
                    global_step=global_step,
                    started_at=started_at,
                    metrics=final_metrics,
                ),
                status_path,
            )
            early_stop_patience = phase.get("early_stop_patience")
            if (
                early_stop_patience is not None
                and epoch >= int(phase.get("minimum_epochs", 1))
                and patience >= int(early_stop_patience)
            ):
                break
    except BaseException as exc:
        atomic_json_dump(
            _status(
                stage=phase_name,
                state="error",
                epoch=locals().get("epoch", start_epoch - 1),
                global_step=global_step,
                started_at=started_at,
                metrics={"error": repr(exc)},
            ),
            status_path,
        )
        tracker.finish(exit_code=1)
        raise
    hardening_passed = eligible_best_exists
    result = {
        "format": "cftn_text_v1_3_integration_training_result_v1",
        "state": "completed" if hardening_passed else "failed_acceptance",
        "phase": phase_name,
        "best_checkpoint": str(best_path.resolve()) if hardening_passed else None,
        "best_checkpoint_sha256": file_sha256(best_path) if hardening_passed else None,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "optimizer_contract": {
            "group_names": [
                str(group.get("group_name")) for group in optimizer.param_groups
            ],
            "peak_learning_rates": {
                str(group.get("group_name")): float(group.get("initial_lr", group["lr"]))
                for group in optimizer.param_groups
            },
            "gate_only": phase_name == "hardened_wake" or router_recovery,
            "trainable_components": (
                ["wake_gates", "wake_round_embeddings"]
                if router_recovery
                else (["wake_gates"] if phase_name == "hardened_wake" else None)
            ),
            "halt_gate_frozen": (
                True if phase_name == "hardened_wake" or router_recovery else None
            ),
        },
        "hardening_policy": hardening_policy,
        "hard_transition_baseline": (
            str(_hard_transition_baseline_path(config).resolve())
            if hardening_baseline is not None
            else None
        ),
        "revision_sha256": config["_meta"]["sha256"],
    }
    atomic_json_dump(result, artifact / "summary.json")
    atomic_json_dump(
        _status(
            stage=phase_name,
            state=str(result["state"]),
            epoch=int(final_metrics["epoch"]),
            global_step=global_step,
            started_at=started_at,
            metrics=final_metrics,
        ),
        status_path,
    )
    if not hardening_passed:
        tracker.update_summary(
            {
                "run/state": "failed_acceptance",
                "collapse_guard": final_metrics.get("hardening_acceptance"),
            }
        )
        tracker.finish(exit_code=1)
        raise RuntimeError(
            f"{phase_name} produced no checkpoint eligible under the collapse guard"
        )
    tracker.finish()
    return result
