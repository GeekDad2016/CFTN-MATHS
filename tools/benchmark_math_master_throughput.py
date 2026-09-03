"""Bounded, non-mutating CUDA throughput probe for a maths-tower checkpoint.

This intentionally loads a checkpoint into an in-memory model and runs a few
real teacher-forced optimisation steps.  It never writes a checkpoint, changes
the source run, or opens a training loop.  The probe is useful when choosing a
safe batch size for a resumed curriculum run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from cftn_text.checkpoint import load_checkpoint
from cftn_text.computation_supervision import hybrid_computation_loss
from cftn_text.config import load_config
from cftn_text.full_math_data import FullMathCollator
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.training import (
    autocast_context,
    build_math_tower_for_checkpoint,
    move_batch,
    precision_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--batch-sizes", default="8,16,32,48,64")
    parser.add_argument("--families-prefix", default="KS2-")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=6)
    return parser.parse_args()


def representative_rows(data_root: Path, prefix: str, required: int) -> list[dict]:
    """Round-robin the requested families without loading the full dataset."""

    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    train_path = data_root / manifest["splits"]["train"]["path"]
    by_family: dict[str, list[dict]] = {}
    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            family = str(row.get("family", ""))
            if family.startswith(prefix):
                by_family.setdefault(family, []).append(row)
    if not by_family:
        raise ValueError(f"no training rows match family prefix {prefix!r}")
    families = sorted(by_family)
    # Taking varied positions avoids a run of near-identical generator records.
    rows: list[dict] = []
    index = 0
    while len(rows) < required:
        family = families[index % len(families)]
        candidates = by_family[family]
        rows.append(candidates[(index * 97) % len(candidates)])
        index += 1
    return rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a throughput probe")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, expected_stage="math", map_location="cpu")
    required_rows = max(batch_sizes)
    rows = representative_rows(args.data, args.families_prefix, required_rows)
    settings = config["math_training"]
    collator = FullMathCollator(
        ByteMathTokenizer(),
        int(config["data"]["max_math_length"]),
        target_mode=str(settings["target_mode"]),
        input_view=str(settings["input_view"]),
    )
    device = torch.device("cuda")
    dtype = precision_dtype(str(settings["precision"]), device)
    results: list[dict] = []
    for requested_batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = build_math_tower_for_checkpoint(config, checkpoint).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.train()
        optimizer = AdamW(model.parameters(), lr=float(settings["learning_rate"]))
        batch = move_batch(collator(rows[:requested_batch_size]), device)

        def step() -> float:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                output = model(
                    batch["math_input_ids"],
                    batch["math_attention_mask"],
                    batch["math_prefix_lengths"],
                )
                loss = hybrid_computation_loss(
                    output.logits,
                    batch["math_labels"],
                    batch["math_roles"],
                    weights=tuple(settings["role_weights"]),
                    role_fraction=float(settings["hybrid_role_fraction"]),
                )
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        try:
            for _ in range(args.warmup_steps):
                step()
            torch.cuda.synchronize(device)
            elapsed: list[float] = []
            losses: list[float] = []
            for _ in range(args.timed_steps):
                started = time.perf_counter()
                losses.append(step())
                torch.cuda.synchronize(device)
                elapsed.append(time.perf_counter() - started)
            result = {
                "batch_size": requested_batch_size,
                "sequence_width": int(batch["math_input_ids"].shape[1]),
                "median_step_seconds": statistics.median(elapsed),
                "examples_per_second": requested_batch_size / statistics.median(elapsed),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                "finite_loss": all(torch.isfinite(torch.tensor(losses)).tolist()),
            }
        except torch.cuda.OutOfMemoryError:
            result = {"batch_size": requested_batch_size, "oom": True}
            torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del optimizer, model, batch
        torch.cuda.empty_cache()
    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
