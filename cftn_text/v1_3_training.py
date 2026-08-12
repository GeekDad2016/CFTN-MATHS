from __future__ import annotations

import json
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
from .v1_3_data import SPECIALISTS, audit_v1_3_manifest, prepare_v1_3_manifests
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
    gpt_tokenizer, gpt_tower = load_gpt_components(base)
    model = V13MultiTowerModel(
        gpt_tower=gpt_tower,
        specialists={"math": math_tower, "string": string_tower},
        config=config,
    ).to(device)
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
        "v1_2": prerequisite,
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
) -> tuple[V13ModelOutput, torch.Tensor, dict[str, float]]:
    configured = settings["losses"]
    compute_weight = (
        float(configured["active_compute_hardening"])
        if wake_mode == "hard_straight_through"
        else float(configured["active_compute_initial"])
    )
    output = model(
        batch,
        wake_mode=wake_mode,
        maximum_rounds=maximum_rounds,
        loss_weights={
            "task": float(configured["task"]),
            "specialist": float(configured["specialist"]),
            "wake_required_set": float(configured["wake_required_set"]),
            "halt": float(configured["halt"]),
            "active_compute": compute_weight,
        },
    )
    total = output.loss
    preservation = output.loss.detach() * 0.0
    causal_message = output.loss.detach() * 0.0
    causal_wake = output.loss.detach() * 0.0
    auxiliary = global_step % int(settings.get("auxiliary_every_steps", 4)) == 0
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
    if auxiliary and wake_mode in {"soft", "hard_straight_through"} and bool(required_rows.any()):
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
        "specialist_loss": float(output.specialist_loss.detach()),
        "wake_loss": float(output.wake_loss.detach()),
        "halt_loss": float(output.halt_loss.detach()),
        "compute_loss": float(output.compute_loss.detach()),
        "preservation_loss": float(preservation.detach()),
        "causal_message_loss": float(causal_message.detach()),
        "causal_wake_loss": float(causal_wake.detach()),
        "auxiliary_step": float(auxiliary),
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
) -> dict[str, Any]:
    model.eval()
    examples = token_correct = token_total = sequence_correct = 0
    loss_sum = 0.0
    wake_tp = wake_fp = wake_fn = exact_sets = wake_labels = 0
    pure_examples = pure_false_wakes = 0
    causal_correct = causal_shuffled = causal_examples = 0.0
    for batch_index, raw in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_v1_3_batch(raw, device)
        with autocast_context(device, dtype):
            output = model(
                batch, wake_mode=wake_mode, maximum_rounds=maximum_rounds
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
        logits = torch.stack([item.wake_logits for item in output.rounds], dim=1)
        predicted = torch.sigmoid(logits).ge(model.wake_threshold)
        targets = batch["wake_targets"][:, :maximum_rounds].bool()
        wake_tp += int((predicted & targets).sum())
        wake_fp += int((predicted & ~targets).sum())
        wake_fn += int((~predicted & targets).sum())
        exact_sets += int(predicted.eq(targets).all(dim=(1, 2)).sum())
        wake_labels += int(targets.numel())
        pure_mask = torch.tensor(
            [value == "pure_language" for value in batch["task_classes"]],
            device=device,
            dtype=torch.bool,
        )
        pure_examples += int(pure_mask.sum())
        if bool(pure_mask.any()):
            pure_false_wakes += int(predicted[pure_mask].any(dim=(1, 2)).sum())
        if batch_index < causal_batches:
            required = targets.any(dim=(1, 2))
            if bool(required.any()):
                with autocast_context(device, dtype):
                    shuffled = model(
                        batch,
                        wake_mode=wake_mode,
                        maximum_rounds=maximum_rounds,
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
    return {
        "examples": examples,
        "loss": loss_sum / examples,
        "gpt_teacher_forced_token_accuracy": token_correct / max(1, token_total),
        "gpt_teacher_forced_sequence_accuracy": sequence_correct / examples,
        "wake_precision": precision,
        "wake_recall": recall,
        "wake_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "exact_required_set_accuracy": exact_sets / examples,
        "pure_language_false_wake_rate": pure_false_wakes / max(1, pure_examples),
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
    seed = int(config["revision"]["seed"])
    seed_everything(seed + phase_index)
    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    previous = _previous_phase_checkpoint(config, phase_index)
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=previous
    )
    model.set_trainable_phase(phase_name)
    settings = config["integration_training"]
    wake_mode = str(phase["wake_mode"]).replace("oracle_dense", "oracle")
    maximum_rounds = int(phase["maximum_rounds"])
    allowed_classes = set(phase["include_task_classes"])
    train_records = [
        record
        for record in V13Dataset(
            data_root / manifest["splits"]["joint_train"]["path"]
        ).records
        if record["task_class"] in allowed_classes
    ]
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
    artifact = Path(config["paths"]["artifact_root"]) / phase_name
    artifact.mkdir(parents=True, exist_ok=True)
    status_path = artifact / "status.json"
    metrics_path = artifact / "metrics.jsonl"
    best_path = artifact / f"{phase_name}.best.pth"
    started_at = time.time()
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    gate_parameters = [
        value
        for name, value in named
        if "gate" in name or name.startswith("wake_gates") or name.startswith("halt_gate")
    ]
    gate_ids = {id(value) for value in gate_parameters}
    other_parameters = [value for _, value in named if id(value) not in gate_ids]
    optimizer = AdamW(
        [
            {
                "params": other_parameters,
                "lr": float(settings["learning_rate"]),
                "group_name": "bridges_and_receivers",
            },
            {
                "params": gate_parameters,
                "lr": float(settings["learning_rate"])
                * float(settings["gate_learning_rate_multiplier"]),
                "group_name": "gates",
            },
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    steps_per_epoch = max(1, math.ceil(len(train_records) / int(settings["batch_size"])))
    total_steps = int(phase["max_epochs"]) * steps_per_epoch
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
    final_metrics: dict[str, Any] = {}
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
                num_workers=int(settings["num_workers"]),
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
                num_workers=int(settings["num_workers"]),
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
            )
            selection = (
                float(validation["gpt_teacher_forced_sequence_accuracy"])
                + 0.10 * float(validation["wake_f1"])
                + 0.05 * max(0.0, float(validation["causal_message_loss_gap"]))
                - 1e-5 * float(validation["loss"])
            )
            improved = selection > best_metric
            patience = 0 if improved else patience + 1
            if improved:
                best_metric = selection
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
                "best_metric": best_metric,
                "patience": patience,
                "learning_rates": {
                    str(group.get("group_name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "trainable_parameters": model.trainable_parameter_count(),
                "timing": {
                    "epoch_seconds": time.time() - epoch_started,
                    "eta_seconds_to_phase_end": (
                        int(phase["max_epochs"]) - epoch
                    )
                    * (time.time() - epoch_started),
                },
                "gpu": gpu_status(),
            }
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
                extra={"metrics": final_metrics, "provenance": provenance},
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
    result = {
        "format": "cftn_text_v1_3_integration_training_result_v1",
        "state": "completed",
        "phase": phase_name,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": file_sha256(best_path),
        "final_metrics": final_metrics,
        "revision_sha256": config["_meta"]["sha256"],
    }
    atomic_json_dump(result, artifact / "summary.json")
    atomic_json_dump(
        _status(
            stage=phase_name,
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
