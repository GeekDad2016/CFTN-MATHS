"""Bounded three-arm supervision pilot. Never resumes/releases the V2 pipeline.

prepare: build a NEW derivative manifest, auditing immutable source split hashes.
run: same source weights, questions, ordering, curriculum and update budget for
     legacy control, result-focused legacy targets, and verified worked targets.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from cftn_text.checkpoint import append_jsonl, atomic_json_dump, atomic_torch_save, load_checkpoint
from cftn_text.computation_supervision import ComputationCollator, computation_loss
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.dataset import MathCollator
from cftn_text.math_validation import stratified_validation_panel
from cftn_text.model import answer_weighted_causal_language_loss
from cftn_text.specialist_evaluation import generate_math_tower
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.training import build_math_tower_for_checkpoint
from cftn_text.v2_data import validate_v2_record
from cftn_text.v2_metrics import extract_v2_answer, score_v2_generations
from cftn_text.verified_math_data import (
    TARGET_FAMILIES, REPLAY_FAMILIES, VERSION, audit_mathqa_program,
    computation_key, curriculum_band, fingerprint, legacy_spans,
    validate_verified_record, verified_record,
)


PARENT_SHA = "a0a1cec180d5400faafe3e6794793b949dc08743ca9b2c7899a273a181ae21f0"
SOURCE_SHA = "2ddb776715b0ee0accfd03e2d98ea4f29cb47c7b4954c02a6beb759150357b08"
PROTECTED_SHA = "fe2c056a1ee1d4a3514537681d82124b0312f45c27f72f8e73d2afc747d53973"
ARMS = ("control", "loss_only", "verified")


def write_rows(path: Path, rows: list[dict]) -> dict:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return {"path": path.name, "count": len(rows), "sha256": file_sha256(path)}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checked_parent_rows(root: Path, manifest: dict, split: str):
    meta = manifest["splits"][split]
    path = (root / meta["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_sha256(path) != meta["sha256"]:
        raise ValueError(f"parent split hash/path mismatch: {split}")
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                validate_v2_record(row)
                if row["split"] != split:
                    raise ValueError("parent split label mismatch")
                count += 1
                yield row
    if count != meta["count"]:
        raise ValueError("parent split count mismatch")


def prepare(root: Path, destination: Path, per_family: int = 2048) -> dict:
    if destination.exists():
        raise FileExistsError("derivative destination already exists; choose a new version")
    if not 128 <= per_family <= 4096:
        raise ValueError("bounded pilot requires 128..4096 training rows per family")
    parent_path = root / "manifest.json"
    parent = json.loads(parent_path.read_text())
    unsigned = dict(parent)
    if unsigned.pop("manifest_sha256") != PARENT_SHA or fingerprint(unsigned) != PARENT_SHA:
        raise ValueError("sealed parent manifest identity mismatch")
    generator = Path(__file__).resolve().parents[1] / "cftn_text/v2_data.py"
    if file_sha256(generator) != parent["generator_sha256"]:
        raise ValueError("sealed V2 generator changed")
    validation = list(checked_parent_rows(root, parent, "validation"))
    families = TARGET_FAMILIES + REPLAY_FAMILIES
    heldout_keys = set()
    for row in validation:
        if row["family"] in families:
            heldout_keys.add(computation_key(row))
    pools = collections.defaultdict(list)
    flags, skipped = [], collections.Counter()
    seen = set()
    for row in checked_parent_rows(root, parent, "train"):
        if row["source"] == "mathqa":
            flags.append(audit_mathqa_program(row))
        if row["family"] not in families:
            continue
        try:
            key = computation_key(row)
            if key in heldout_keys or key in seen:
                skipped["duplicate_or_validation_computation"] += 1
                continue
            traced = verified_record(row) if row["family"] in TARGET_FAMILIES else None
            band = curriculum_band(row) if traced else "replay"
            if traced:
                validate_verified_record(traced)
                ByteMathTokenizer().encode_training_example(row["problem"], traced["target_trace"], 4096)
            legacy_spans(row)
        except ValueError as exc:
            # Explicitly counted training exclusions; validation is NEVER filtered.
            skipped[str(exc)] += 1
            continue
        seen.add(key)
        pools[row["family"]].append({"original": row, "verified": traced,
                                      "band": band, "computation_key": key})
    selected = []
    for family in families:
        pool = sorted(pools[family], key=lambda b: fingerprint([719, b["original"]["record_id"]]))
        if len(pool) < per_family:
            raise ValueError(f"insufficient verified support: {family}: {len(pool)}")
        selected.extend(pool[:per_family])
    panels = {}
    for family in families:
        panel = sorted([r for r in validation if r["family"] == family],
                       key=lambda r: fingerprint([881, r["record_id"]]))[:64 if family in TARGET_FAMILIES else 32]
        if family in TARGET_FAMILIES:
            for row in panel:
                verified_record(row)  # fail rather than drop an unsupported evaluation row
        panels[family] = panel
    panels["broad_diagnostic"] = stratified_validation_panel(validation, 128)
    destination.mkdir(parents=True, exist_ok=False)
    files = {"train": write_rows(destination / "train.jsonl", selected),
             "mathqa_triage": write_rows(destination / "mathqa_triage.jsonl", flags)}
    for name, rows in panels.items():
        files[name] = write_rows(destination / f"{name}.jsonl", rows)
    manifest = {
        "format": "cftn_math_supervision_pilot_v1", "procedure_version": VERSION,
        "parent_manifest_sha256": PARENT_SHA, "parent_manifest_file_sha256": file_sha256(parent_path),
        "audited_parent_splits": {k: parent["splits"][k] for k in ("train", "validation")},
        "parent_generator_sha256": parent["generator_sha256"],
        "builder_sha256": file_sha256(Path(__file__)),
        "procedure_code_sha256": file_sha256(generator.with_name("verified_math_data.py")),
        "files": files, "skipped_training_rows": dict(skipped),
        "mathqa_triage_counts": dict(collections.Counter(f["status"] for f in flags)),
        "mathqa_training_eligible": 0,
        "training_family_band_counts": dict(collections.Counter(b["original"]["family"] + ":" + b["band"] for b in selected)),
        "train_validation_computation_overlap": 0,
        "production_acceptance_unchanged": True,
        "claim": "Diagnostic subset, not a repaired certification of all 60 V2 families.",
    }
    manifest["manifest_sha256"] = fingerprint(manifest)
    atomic_json_dump(manifest, destination / "manifest.json")
    return manifest


def validate_bundle(bundle: dict) -> None:
    original, traced = bundle["original"], bundle["verified"]
    validate_v2_record(original)
    if original["split"] != "train":
        raise ValueError("evaluation data in training split")
    if bundle["computation_key"] != computation_key(original):
        raise ValueError("incorrect mathematical split key")
    if original["family"] in TARGET_FAMILIES:
        if traced is None:
            raise ValueError("targeted family is missing verified supervision")
        validate_verified_record(traced)
        if (traced["parent_record_id"] != original["record_id"]
                or traced["parent_content_id"] != original["content_id"]
                or any(traced[key] != original[key] for key in
                       ("problem", "normalized_answer", "source", "family", "split", "difficulty"))):
            raise ValueError("derivative parent record mismatch")
        expected_band = curriculum_band(original)
    else:
        if original["family"] not in REPLAY_FAMILIES or traced is not None:
            raise ValueError("unexpected replay family or changed replay target")
        expected_band = "replay"
    if bundle["band"] != expected_band:
        raise ValueError("incorrect numerical curriculum band")


def checked_derivative(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    unsigned = dict(manifest)
    if unsigned.pop("manifest_sha256") != fingerprint(unsigned):
        raise ValueError("derivative manifest hash mismatch")
    if manifest["parent_manifest_sha256"] != PARENT_SHA:
        raise ValueError("unexpected parent")
    code = Path(__file__).resolve().parents[1] / "cftn_text/verified_math_data.py"
    if file_sha256(code) != manifest["procedure_code_sha256"]:
        raise ValueError("procedure version/code changed since building data")
    data = {}
    for name, meta in manifest["files"].items():
        path = (root / meta["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path) != meta["sha256"]:
            raise ValueError(f"derivative file hash/path mismatch: {name}")
        rows = read_rows(path)
        if len(rows) != meta["count"]:
            raise ValueError("derivative count mismatch")
        data[name] = rows
    for bundle in data["train"]:
        validate_bundle(bundle)
    train_keys = [b["computation_key"] for b in data["train"]]
    eval_keys = {computation_key(r) for name, rows in data.items()
                 if name not in ("train", "mathqa_triage") for r in rows}
    if len(set(train_keys)) != len(train_keys) or set(train_keys) & eval_keys:
        raise ValueError("duplicate training computations or train/evaluation overlap")
    return manifest, data


def assert_idle(*, ignore_pids: set[int] | None = None) -> None:
    if os.name != "posix":
        raise RuntimeError("RunPod/Linux execution only")
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid() or int(path.name) in (ignore_pids or set()):
            continue
        try:
            argv = (path / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        # Ignore shells carrying a quoted command; inspect actual Python argv.
        if not argv or "python" not in Path(os.fsdecode(argv[0])).name:
            continue
        text = " ".join(os.fsdecode(arg) for arg in argv)
        if any(marker in text for marker in ("tools.recover_v2_math", "tools.run_v2_experiment",
                  "tools.train_", "tools.evaluate_", "tools.pilot_")):
            raise RuntimeError("another CFTN trainer/evaluator/pilot exists")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    for line in gpu.strip().splitlines():
        utilization, memory = map(int, line.split(","))
        if utilization > 5 or memory > 512:
            raise RuntimeError("GPU is not idle")


def training_schedule(bundles: list[dict], steps: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pools = collections.defaultdict(list)
    for index, bundle in enumerate(bundles):
        pools[bundle["original"]["family"], bundle["band"]].append(index)
    result = []
    # 75% targeted families, 25% unchanged replay, identical in all arms.
    family_cycle = (TARGET_FAMILIES[0], TARGET_FAMILIES[1], TARGET_FAMILIES[0], TARGET_FAMILIES[1],
                    TARGET_FAMILIES[0], TARGET_FAMILIES[1], REPLAY_FAMILIES[0], REPLAY_FAMILIES[1])
    for step in range(steps):
        batch = []
        for slot in range(batch_size):
            family = family_cycle[(step * batch_size + slot) % len(family_cycle)]
            foundation = pools[family, "foundation"]
            expanded = pools[family, "expanded"]
            pool = pools[family, "replay"] if family in REPLAY_FAMILIES else (
                foundation if step < steps // 3 else foundation + expanded)
            if not pool:
                raise ValueError(f"empty curriculum pool: {family}")
            batch.append(rng.choice(pool))
        rng.shuffle(batch)
        result.append(batch)
    return result


def selected_rows(bundles: list[dict], arm: str) -> list[dict]:
    return [b["verified"] if arm == "verified" and b["verified"] else b["original"] for b in bundles]


def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.inference_mode()
def evaluate(model, panels: dict, arm: str, output: Path, max_new_tokens: int) -> dict:
    model.eval()
    tokenizer = ByteMathTokenizer()
    report = {}
    for name, records in panels.items():
        started = time.monotonic()
        generations = []
        for offset in range(0, len(records), 16):
            chunk = records[offset:offset + 16]
            generated, _ = generate_math_tower(model, tokenizer, [r["problem"] for r in chunk],
                                               max_new_tokens=max_new_tokens)
            generations.extend(generated)
        metrics, correct = score_v2_generations(generations, records)
        rows = []
        for row, text, success in zip(records, generations, correct):
            trace_valid = None
            if row["family"] in TARGET_FAMILIES:
                # Strict canonical trace verifies program structure and every
                # operation, not merely that a generated residual says "0".
                trace_valid = text == verified_record(row)["target_trace"]
            rows.append({"record_id": row["record_id"], "problem": row["problem"],
                         "family": row["family"], "expected": row["normalized_answer"],
                         "generation": text, "correct": success,
                         "verified_full_trace": trace_valid,
                         "at_generation_budget": len(tokenizer.encode(text)) >= max_new_tokens})
        metrics["elapsed_seconds"] = time.monotonic() - started
        metrics["generated_bytes"] = sum(len(t.encode()) for t in generations)
        metrics["generation_budget_hits"] = sum(r["at_generation_budget"] for r in rows)
        metrics["verified_full_traces"] = sum(r["verified_full_trace"] is True for r in rows)
        # This diagnostic is teacher forced, explicitly separate from generation.
        if name in TARGET_FAMILIES:
            targets = [verified_record(r) if arm == "verified" else r for r in records]
            collator = ComputationCollator(tokenizer, model.max_sequence_length)
            first_correct = compute_correct = compute_tokens = 0
            for offset in range(0, len(targets), 16):
                chunk = targets[offset:offset + 16]
                batch = to_device(collator(chunk), "cuda")
                logits = model(batch["math_input_ids"], batch["math_attention_mask"], batch["math_prefix_lengths"]).logits
                predictions = logits[:, :-1].argmax(-1)
                roles, labels = batch["math_roles"][:, 1:], batch["math_labels"][:, 1:]
                mask = roles.eq(1)
                compute_tokens += int(mask.sum())
                compute_correct += int((predictions.eq(labels) & mask).sum())
                for i, row in enumerate(chunk):
                    spans = row.get("supervision_spans") or legacy_spans(row)
                    span = next(s for s in spans if s["kind"] == "compute")
                    prefix = int(batch["math_prefix_lengths"][i]) - 1
                    start = prefix + len(tokenizer.encode(row["target_trace"][:span["start"]]))
                    end = prefix + len(tokenizer.encode(row["target_trace"][:span["end"]]))
                    first_correct += int(predictions[i, start:end].eq(labels[i, start:end]).all())
            metrics["teacher_forced_computed_token_accuracy"] = compute_correct / compute_tokens
            metrics["teacher_forced_first_result_accuracy"] = first_correct / len(records)
            metrics["first_result_caveat"] = "Arm-specific procedure/first operation; not a common free-generation metric."
        write_rows(output / f"{name}.generations.jsonl", rows)
        report[name] = metrics
        atomic_json_dump(report, output / "validation.json")
        print(json.dumps({"event": "validation_panel", "arm": arm, "panel": name,
                          "correct": metrics["correct_answers"], "n": len(records)}), flush=True)
    return report


def screening_decision(reports: dict) -> dict:
    def accuracy(arm, family):
        return reports[arm][family]["accuracy"]
    def target(arm):
        return sum(accuracy(arm, f) for f in TARGET_FAMILIES) / len(TARGET_FAMILIES)
    gates = {
        "target_gain_at_least_5pp_vs_both_controls_and_baseline": target("verified") >= max(target(a) for a in ("baseline", "control", "loss_only")) + 0.05,
        "neither_target_family_regresses": all(accuracy("verified", f) >= max(accuracy(a, f) for a in ("baseline", "control")) for f in TARGET_FAMILIES),
        "replay_regression_at_most_3pp_each": all(accuracy("verified", f) >= accuracy("baseline", f) - 0.03 for f in REPLAY_FAMILIES),
        "broad_diagnostic_regression_at_most_3pp": accuracy("verified", "broad_diagnostic") >= accuracy("baseline", "broad_diagnostic") - 0.03,
        "no_verified_target_generation_cap_hits": all(reports["verified"][f]["generation_budget_hits"] == 0 for f in TARGET_FAMILIES),
    }
    return {"gates": gates, "screen_pass": all(gates.values()),
            "production_acceptance": False, "checkpoint_promotion": False,
            "long_training_authorized_by_this_tool": False,
            "meaning": "A pass warrants a larger confirmation, not production acceptance; a fail preserves all evidence."}


def run(args) -> None:
    assert_idle()
    source, protected = Path(args.source), Path(args.protected)
    if source.resolve() == protected.resolve():
        raise ValueError("pilot source must be the separate capacity epoch-5 checkpoint")
    for path, expected in ((source, SOURCE_SHA), (protected, PROTECTED_SHA)):
        if file_sha256(path) != expected:
            raise ValueError("protected/source checkpoint hash mismatch")
    if not 8 <= args.steps <= 600 or args.batch_size != 16:
        raise ValueError("pilot budget requires 8..600 steps per arm, batch size 16")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest, data = checked_derivative(Path(args.data))
    schedule = training_schedule(data["train"], args.steps, args.batch_size, 719)
    panels = {k: v for k, v in data.items() if k not in ("train", "mathqa_triage")}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("pilot requires a clean test checkout")
    contract = {"format": "cftn_bounded_math_supervision_pilot_v1", "revision": revision,
                "source_sha256": SOURCE_SHA, "protected_sha256": PROTECTED_SHA,
                "data_manifest_sha256": manifest["manifest_sha256"], "arms": ARMS,
                "steps_per_arm": args.steps, "batch_size": args.batch_size, "seed": 719,
                "learning_rate": 5e-5, "optimizer": "fresh AdamW per arm; no scheduler",
                "schedule_sha256": fingerprint(schedule), "max_new_tokens": 1024,
                "max_training_seconds_per_arm": 600, "max_total_seconds": 5400,
                "production_checkpoint_eligible": False,
                "curriculum": "first third foundations, then all supported magnitudes; 25% replay throughout",
                "group_weights": {"compute": 0.7, "copy": 0.2, "format": 0.1},
                "warning": "Equal updates/examples, not equal target tokens or GPU time. Report both.",
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0)}
    atomic_json_dump(contract, output / "contract.json")
    checkpoint = load_checkpoint(source, expected_manifest_sha256=PARENT_SHA)
    if checkpoint["epoch"] != 5 or checkpoint["global_step"] != 62500:
        raise ValueError("unexpected source epoch/step")
    config = load_config(args.config)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = ByteMathTokenizer()
    started = time.monotonic()
    reports, costs = {}, {}
    for arm in ("baseline",) + ARMS:
        if time.monotonic() - started > contract["max_total_seconds"]:
            raise RuntimeError("pilot wall-time budget exceeded")
        torch.manual_seed(719)
        torch.cuda.manual_seed_all(719)
        model = build_math_tower_for_checkpoint(config, checkpoint).cuda()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        arm_dir = output / arm
        arm_dir.mkdir()
        if arm != "baseline":
            records = selected_rows(data["train"], arm)
            collator = (MathCollator if arm == "control" else ComputationCollator)(tokenizer, model.max_sequence_length)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            train_started = time.monotonic()
            supervised_tokens = 0
            model.train()
            torch.cuda.reset_peak_memory_stats()
            for step, indices in enumerate(schedule, 1):
                if time.monotonic() - train_started > contract["max_training_seconds_per_arm"]:
                    raise RuntimeError("per-arm training time budget exceeded; no unequal-arm success claim")
                batch = to_device(collator([records[i] for i in indices]), "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch["math_input_ids"], batch["math_attention_mask"], batch["math_prefix_lengths"]).logits
                    loss = (answer_weighted_causal_language_loss(logits.float(), batch["math_labels"], batch["math_answer_labels"], answer_weight=4.0)
                            if arm == "control" else computation_loss(logits, batch["math_labels"], batch["math_roles"]))
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite pilot loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                supervised_tokens += int(batch["math_labels"].ne(-100).sum())
                if step == 1 or step % 25 == 0 or step == args.steps:
                    status = {"state": "training", "arm": arm, "step": step, "steps": args.steps,
                              "loss_training_batch": float(loss.detach()), "grad_norm": float(grad),
                              "elapsed_seconds": time.monotonic() - train_started}
                    append_jsonl(status, arm_dir / "metrics.jsonl")
                    atomic_json_dump(status, output / "status.json")
                    print(json.dumps(status), flush=True)
            torch.cuda.synchronize()
            costs[arm] = {"training_seconds": time.monotonic() - train_started,
                          "supervised_tokens": supervised_tokens, "examples": args.steps * args.batch_size,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            atomic_torch_save({"format": "cftn_math_pilot_not_promotable_v1", "model_state": model.cpu().state_dict(),
                               "effective_math_tower": model.config, "contract": contract, "arm": arm},
                              arm_dir / "pilot.final.pth")
            model.cuda()
            del optimizer
        atomic_json_dump({"state": "evaluating", "arm": arm}, output / "status.json")
        reports[arm] = evaluate(model, panels, arm, arm_dir, contract["max_new_tokens"])
        del model
        torch.cuda.empty_cache()
    # Validate source preservation after all arms and evaluation work.
    for path, expected in ((source, SOURCE_SHA), (protected, PROTECTED_SHA)):
        if file_sha256(path) != expected:
            raise RuntimeError("source checkpoint changed during pilot")
    result = {"state": "completed", "screening": screening_decision(reports),
              "reports": reports, "costs": costs, "source_preserved": True,
              "elapsed_seconds": time.monotonic() - started, "contract": contract}
    atomic_json_dump(result, output / "summary.json")
    atomic_json_dump({"state": "completed", "screen_pass": result["screening"]["screen_pass"]}, output / "status.json")
    print(json.dumps({"event": "pilot_completed", "screening": result["screening"], "costs": costs}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_args = sub.add_parser("prepare")
    prepare_args.add_argument("--parent", required=True)
    prepare_args.add_argument("--output", required=True)
    prepare_args.add_argument("--per-family", type=int, default=2048)
    run_args = sub.add_parser("run")
    for name in ("source", "protected", "data", "output"):
        run_args.add_argument("--" + name, required=True)
    run_args.add_argument("--config", default="config/v2_broad_math.yaml")
    run_args.add_argument("--steps", type=int, default=300)
    run_args.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(Path(args.parent), Path(args.output), args.per_family)))
    else:
        try:
            run(args)
        except Exception as exc:
            output = Path(args.output)
            # Never overwrite an existing completed run's status on a bad retry.
            if (output / "contract.json").exists() and not (output / "summary.json").exists():
                atomic_json_dump({"state": "error", "error": str(exc), "checkpoint_promotion": False}, output / "status.json")
            raise


if __name__ == "__main__":
    main()
