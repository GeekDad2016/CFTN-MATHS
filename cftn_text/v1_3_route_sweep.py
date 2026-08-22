from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from .checkpoint import atomic_json_dump, gpu_status
from .data_generator import file_sha256
from .tokenizer import ByteMathTokenizer
from .training import autocast_context, precision_dtype, resolve_device
from .v1_3_data import SPECIALISTS
from .v1_3_dataset import V13JointCollator, move_v1_3_batch
from .v1_3_training import build_v1_3_model, load_v1_3_data_contract


RouteAction = tuple[int, ...]
RouteSchedule = tuple[RouteAction, ...]

_ACTION_NAMES = {
    (0, 0): "closed",
    (1, 0): "math",
    (0, 1): "string",
    (1, 1): "both",
}


def enumerate_route_schedules(
    maximum_rounds: int, specialist_count: int = len(SPECIALISTS)
) -> tuple[RouteSchedule, ...]:
    """Enumerate every independent hard wake-set over the configured rounds."""

    if maximum_rounds < 1:
        raise ValueError("maximum_rounds must be positive")
    if specialist_count < 1:
        raise ValueError("specialist_count must be positive")
    actions = tuple(itertools.product((0, 1), repeat=specialist_count))
    return tuple(itertools.product(actions, repeat=maximum_rounds))


def enumerate_route_schedules_up_to(
    maximum_rounds: int, specialist_count: int = len(SPECIALISTS)
) -> tuple[RouteSchedule, ...]:
    """Enumerate every hard schedule that halts within ``maximum_rounds``."""

    if maximum_rounds < 1:
        raise ValueError("maximum_rounds must be positive")
    return tuple(
        schedule
        for rounds in range(1, maximum_rounds + 1)
        for schedule in enumerate_route_schedules(rounds, specialist_count)
    )


def route_schedule_name(schedule: RouteSchedule) -> str:
    if not schedule:
        raise ValueError("route schedule must contain at least one round")
    names: list[str] = []
    for action in schedule:
        key = tuple(int(value) for value in action)
        names.append(_ACTION_NAMES.get(key, "wake_" + "".join(map(str, key))))
    return ">".join(names)


def route_schedule_tensor(
    schedule: RouteSchedule,
    *,
    batch_size: int,
    specialist_count: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not schedule or any(len(action) != specialist_count for action in schedule):
        raise ValueError("route schedule shape does not match the specialists")
    values = torch.tensor(schedule, dtype=dtype, device=device)
    if not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("route schedules must be binary")
    return values.unsqueeze(0).expand(batch_size, -1, -1).clone()


def _completion_field(text: str) -> str:
    for line in text.splitlines():
        value = line.strip()
        if value:
            return value
    return text.strip()


def parallel_component_statistics(prediction: str, target: str) -> dict[str, int]:
    """Score the two fields of a teacher-forced ``x|reversed`` prediction."""

    if target.count("|") != 1:
        raise ValueError("multi_parallel target must contain exactly one delimiter")
    expected_math, expected_string = target.split("|", 1)
    completion = _completion_field(prediction)
    valid_format = completion.count("|") == 1
    if valid_format:
        predicted_math, predicted_string = completion.split("|", 1)
    else:
        predicted_math = predicted_string = ""
    math_correct = valid_format and predicted_math == expected_math
    string_correct = valid_format and predicted_string == expected_string
    contains_both = expected_math in completion and expected_string in completion
    exact = completion == target
    return {
        "exact": int(exact),
        "valid_format": int(valid_format),
        "math_component_correct": int(math_correct),
        "string_component_correct": int(string_correct),
        "both_components_correct": int(math_correct and string_correct),
        "contains_both_components": int(contains_both),
        "format_only_error": int(contains_both and not exact),
    }


@dataclass
class _ScheduleTotals:
    examples: int = 0
    token_correct: int = 0
    token_total: int = 0
    sequence_correct: int = 0
    per_example_loss_sum: float = 0.0
    exact: int = 0
    valid_format: int = 0
    math_component_correct: int = 0
    string_component_correct: int = 0
    both_components_correct: int = 0
    contains_both_components: int = 0
    format_only_error: int = 0

    def add_components(self, values: dict[str, int]) -> None:
        for name, value in values.items():
            setattr(self, name, int(getattr(self, name)) + int(value))

    def report(self, schedule: RouteSchedule) -> dict[str, Any]:
        examples = max(1, self.examples)
        calls = sum(sum(action) for action in schedule)
        return {
            "schedule": route_schedule_name(schedule),
            "actions": [list(action) for action in schedule],
            "rounds": len(schedule),
            "specialist_calls_per_example": int(calls),
            "compute_fraction_of_dense": calls / max(1, len(schedule) * len(SPECIALISTS)),
            "examples": self.examples,
            "teacher_forced_sequence_accuracy": self.sequence_correct / examples,
            "teacher_forced_token_accuracy": self.token_correct
            / max(1, self.token_total),
            "teacher_forced_gpt_loss": self.per_example_loss_sum / examples,
            "decoded_exact_accuracy": self.exact / examples,
            "valid_format_rate": self.valid_format / examples,
            "math_component_accuracy": self.math_component_correct / examples,
            "string_component_accuracy": self.string_component_correct / examples,
            "both_components_accuracy": self.both_components_correct / examples,
            "contains_both_components_rate": self.contains_both_components / examples,
            "format_only_error_rate": self.format_only_error / examples,
        }


def _rank_key(result: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(result["teacher_forced_sequence_accuracy"]),
        float(result["both_components_accuracy"]),
        float(result["teacher_forced_token_accuracy"]),
        -float(result["teacher_forced_gpt_loss"]),
        -float(result["specialist_calls_per_example"]),
    )


def _batched(values: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


@torch.no_grad()
def evaluate_route_schedules_teacher_forced(
    model: Any,
    records: Sequence[dict[str, Any]],
    collator: V13JointCollator,
    gpt_tokenizer: Any,
    schedules: Sequence[RouteSchedule],
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    batch_size: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("route sweep requires at least one record")
    if not schedules:
        raise ValueError("route sweep requires at least one schedule")
    if any(record.get("task_class") != "multi_parallel" for record in records):
        raise ValueError("route sweep currently accepts only multi_parallel records")
    maximum_rounds = len(schedules[0])
    if any(len(schedule) != maximum_rounds for schedule in schedules):
        raise ValueError("all route schedules must have the same number of rounds")
    if maximum_rounds > int(model.maximum_rounds):
        raise ValueError("route schedule exceeds the model runtime")

    model.eval()
    results: list[dict[str, Any]] = []
    batches_per_schedule = math.ceil(len(records) / batch_size)
    evaluations_total = len(schedules) * batches_per_schedule
    evaluations_completed = 0
    started_at = time.time()
    for schedule_index, schedule in enumerate(schedules, start=1):
        totals = _ScheduleTotals()
        for raw_records in _batched(records, batch_size):
            batch = move_v1_3_batch(collator(raw_records), device)
            batch["wake_targets"] = route_schedule_tensor(
                schedule,
                batch_size=len(raw_records),
                specialist_count=len(SPECIALISTS),
                device=device,
            )
            with autocast_context(device, dtype):
                output = model(
                    batch,
                    wake_mode="oracle",
                    maximum_rounds=maximum_rounds,
                    conditional_execution=True,
                    apply_halt=False,
                )

            shifted_labels = batch["gpt_labels"][:, 1:]
            valid = shifted_labels.ne(-100)
            predictions = output.gpt_logits[:, :-1].argmax(dim=-1)
            token_correct = ((predictions == shifted_labels) & valid).sum(dim=1)
            token_total = valid.sum(dim=1)
            sequence_correct = (
                ((predictions == shifted_labels) | ~valid).all(dim=1)
                & token_total.gt(0)
            )
            token_losses = F.cross_entropy(
                output.gpt_logits[:, :-1].float().reshape(-1, output.gpt_logits.shape[-1]),
                shifted_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape(shifted_labels.shape)
            row_losses = (token_losses * valid).sum(dim=1) / token_total.clamp_min(1)

            totals.examples += len(raw_records)
            totals.token_correct += int(token_correct.sum())
            totals.token_total += int(token_total.sum())
            totals.sequence_correct += int(sequence_correct.sum())
            totals.per_example_loss_sum += float(row_losses.sum())
            for row_index, record in enumerate(raw_records):
                predicted_ids = predictions[row_index][valid[row_index]].tolist()
                predicted_text = gpt_tokenizer.decode(predicted_ids)
                totals.add_components(
                    parallel_component_statistics(
                        predicted_text, str(record["gpt_target"])
                    )
                )

            evaluations_completed += 1
            if progress is not None:
                elapsed = time.time() - started_at
                fraction = evaluations_completed / max(1, evaluations_total)
                progress(
                    {
                        "state": "running",
                        "phase": "teacher_forced_route_sweep",
                        "schedule": route_schedule_name(schedule),
                        "schedule_index": schedule_index,
                        "schedules_total": len(schedules),
                        "evaluations_completed": evaluations_completed,
                        "evaluations_total": evaluations_total,
                        "progress": fraction,
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed * (1.0 - fraction) / max(fraction, 1e-12),
                        "gpu": gpu_status(),
                    }
                )
        results.append(totals.report(schedule))
    return sorted(results, key=_rank_key, reverse=True)


def _intended_schedule(maximum_rounds: int) -> RouteSchedule:
    return ((1, 1),) + ((0, 0),) * (maximum_rounds - 1)


def run_v1_3_route_sweep(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    device_name: str,
    split: str,
    screen_examples: int,
    full_examples: int,
    top_k: int,
    batch_size: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    if min(screen_examples, full_examples, top_k, batch_size) < 1:
        raise ValueError("route sweep counts must be positive")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"route sweep checkpoint is missing: {checkpoint}")
    artifact = Path(output_dir).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    report_path = artifact / "report.json"
    status_path = artifact / "status.json"
    if report_path.exists():
        raise FileExistsError(f"route sweep report already exists: {report_path}")

    device = resolve_device(device_name)
    data_root, manifest = load_v1_3_data_contract(config)
    split_entry = manifest["splits"].get(split)
    if not isinstance(split_entry, dict):
        raise ValueError(f"unknown V1.3 split: {split}")
    from .v1_3_data import load_v1_3_records

    all_records = load_v1_3_records(data_root / str(split_entry["path"]))
    records = [
        record for record in all_records if record.get("task_class") == "multi_parallel"
    ]
    if not records:
        raise RuntimeError(f"split {split} contains no multi_parallel rows")
    full_count = min(int(full_examples), len(records))
    screen_count = min(int(screen_examples), full_count)
    screen_records = records[:screen_count]
    full_records = records[:full_count]

    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=checkpoint
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    collator = V13JointCollator(
        ByteMathTokenizer(),
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    dtype = precision_dtype(config["integration_training"]["precision"], device)
    maximum_rounds = int(config["runtime"]["maximum_callosal_rounds"])
    schedule_groups = {
        rounds: enumerate_route_schedules(rounds, len(SPECIALISTS))
        for rounds in range(1, maximum_rounds + 1)
    }
    schedules_total = sum(len(values) for values in schedule_groups.values())
    started_at = time.time()
    checkpoint_sha256 = file_sha256(checkpoint)

    def write_status(values: dict[str, Any]) -> None:
        atomic_json_dump(
            {
                "format": "cftn_text_v1_3_route_sweep_status_v1",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "split": split,
                "task_class": "multi_parallel",
                **values,
                "updated_unix": time.time(),
            },
            status_path,
        )

    write_status(
        {
            "state": "running",
            "phase": "loading",
            "screen_examples": screen_count,
            "full_examples": full_count,
            "schedules_total": schedules_total,
            "round_counts": list(schedule_groups),
            "gpu": gpu_status(),
        }
    )
    try:
        screen_results: list[dict[str, Any]] = []
        for rounds, schedules in schedule_groups.items():
            screen_results.extend(
                evaluate_route_schedules_teacher_forced(
                    model,
                    screen_records,
                    collator,
                    gpt_tokenizer,
                    schedules,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                    progress=lambda values, route_rounds=rounds: write_status(
                        {**values, "route_rounds": route_rounds}
                    ),
                )
            )
        screen_results.sort(key=_rank_key, reverse=True)
        # Multi-parallel declares both specialists in round one and halts there.
        intended = _intended_schedule(1)
        selected = [
            tuple(tuple(int(value) for value in action) for action in result["actions"])
            for result in screen_results[: min(top_k, len(screen_results))]
        ]
        if intended not in selected:
            selected.append(intended)
        write_status(
            {
                "state": "running",
                "phase": "full_confirmation",
                "selected_schedules": [route_schedule_name(value) for value in selected],
                "screen_examples": screen_count,
                "full_examples": full_count,
                "gpu": gpu_status(),
            }
        )
        full_results: list[dict[str, Any]] = []
        for rounds in sorted({len(schedule) for schedule in selected}):
            same_length = [schedule for schedule in selected if len(schedule) == rounds]
            full_results.extend(
                evaluate_route_schedules_teacher_forced(
                    model,
                    full_records,
                    collator,
                    gpt_tokenizer,
                    same_length,
                    device=device,
                    dtype=dtype,
                    batch_size=batch_size,
                    progress=lambda values, route_rounds=rounds: write_status(
                        {
                            **values,
                            "phase": "full_confirmation",
                            "route_rounds": route_rounds,
                        }
                    ),
                )
            )
        full_results.sort(key=_rank_key, reverse=True)
        intended_name = route_schedule_name(intended)
        intended_result = next(
            result for result in full_results if result["schedule"] == intended_name
        )
        best_result = full_results[0]
        sequence_gain = float(best_result["teacher_forced_sequence_accuracy"]) - float(
            intended_result["teacher_forced_sequence_accuracy"]
        )
        component_gain = float(best_result["both_components_accuracy"]) - float(
            intended_result["both_components_accuracy"]
        )
        route_can_materially_help = sequence_gain >= 0.02 or component_gain >= 0.05
        report = {
            "format": "cftn_text_v1_3_route_schedule_sweep_v2",
            "state": "completed",
            "revision_sha256": config["_meta"]["sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "split": split,
            "task_class": "multi_parallel",
            "specialists": list(SPECIALISTS),
            "maximum_rounds": maximum_rounds,
            "round_counts_screened": list(schedule_groups),
            "schedules_screened": schedules_total,
            "screen_examples": screen_count,
            "full_examples": full_count,
            "screen_results": screen_results,
            "full_results": full_results,
            "intended_schedule": intended_name,
            "best_schedule": best_result["schedule"],
            "sequence_accuracy_gain_over_intended": sequence_gain,
            "both_components_gain_over_intended": component_gain,
            "route_can_materially_help": route_can_materially_help,
            "recommended_next_step": (
                "train_or_distill_route_policy"
                if route_can_materially_help
                else "add_specialist_aware_fusion_adapter"
            ),
            "decision_thresholds": {
                "minimum_sequence_accuracy_gain": 0.02,
                "minimum_both_components_gain": 0.05,
            },
            "provenance": provenance,
            "elapsed_seconds": time.time() - started_at,
            "gpu": gpu_status(),
        }
        atomic_json_dump(report, report_path)
        write_status(
            {
                "state": "completed",
                "phase": "completed",
                "best_schedule": best_result["schedule"],
                "route_can_materially_help": route_can_materially_help,
                "recommended_next_step": report["recommended_next_step"],
                "elapsed_seconds": report["elapsed_seconds"],
                "report": str(report_path),
                "gpu": gpu_status(),
            }
        )
        return report
    except Exception as error:
        write_status(
            {
                "state": "error",
                "phase": "error",
                "error": repr(error),
                "elapsed_seconds": time.time() - started_at,
                "gpu": gpu_status(),
            }
        )
        raise
