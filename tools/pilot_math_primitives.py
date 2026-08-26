"""Detached, bounded prerequisite diagnostic. Never launches the V2 pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import torch

from cftn_text.checkpoint import atomic_json_dump, atomic_torch_save, load_checkpoint
from cftn_text.computation_supervision import computation_loss
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.math_primitive_data import (
    ARMS, COMPOSITIONS, FOUNDATIONS, PrimitiveCollator, lesson, make_corpus,
)
from cftn_text.tokenizer import ByteMathTokenizer, pad_1d
from cftn_text.training import build_math_tower_for_checkpoint
from cftn_text.v2_metrics import score_v2_generations
from cftn_text.verified_math_data import REPLAY_FAMILIES, TARGET_FAMILIES, fingerprint
from tools.pilot_verified_math import (
    PARENT_SHA, PROTECTED_SHA, SOURCE_SHA, assert_idle, checked_derivative, to_device, write_rows,
)

_PAYLOAD = re.compile(r"(?:<work>[^<>]+</work>)?<answer>([^<>]+)</answer>")


def verified_snapshot(payload: list | dict, path: Path) -> None:
    """Canonical diagnostic evidence uses atomic replacement plus readback.

    A migrated volume produced leading NUL holes in three append-only JSONL
    files despite fsync. Preserve those originals; new runs avoid appends for
    their canonical generation/metric evidence and fail on readback mismatch.
    """
    atomic_json_dump(payload, path)
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise OSError("diagnostic snapshot readback mismatch")


def primitive_score(row: dict, text: str, cap: int) -> dict:
    match = _PAYLOAD.fullmatch(text)
    return {"correct": bool(match and match[1] == row["normalized_answer"]),
            "format_valid": match is not None, "exact_target": text == row["target_trace"],
            "budget_hit": len(ByteMathTokenizer().encode(text)) >= cap}


def prerequisite_gate(reports: dict, baseline_replay: dict, *, require_exact_trace: bool = False) -> dict:
    gates = {family: (reports.get(family, {}).get("accuracy", 0) >= .90
                      and reports[family]["valid_rate"] >= .99
                      and reports[family]["budget_hits"] == 0
                      and (not require_exact_trace or reports[family].get("exact_target_rate", 0) >= .90)) for family in FOUNDATIONS}
    gates["replay"] = all(reports.get(f, {}).get("accuracy", 0) >= baseline_replay[f]["accuracy"] - .03
                          for f in REPLAY_FAMILIES)
    return {"gates": gates, "pass": all(gates.values()), "production_acceptance": False}


@torch.inference_mode()
def generate_with_termination(model, tokenizer, problems, cap):
    """Diagnostic greedy decoder retaining EOS/control-token evidence.

    Decoded byte length alone can miss capped runs containing special tokens or
    count UTF-8 replacement characters incorrectly. Production decoder is unchanged.
    """
    device = next(model.parameters()).device
    sequences = [tokenizer.encode_generation_prefix(p, model.max_sequence_length) for p in problems]
    prefixes = [len(s) for s in sequences]
    prefix_tensor = torch.tensor(prefixes, device=device, dtype=torch.long)
    ended = [False] * len(problems)
    for _ in range(cap):
        ids, mask = pad_1d(sequences, tokenizer.pad_token_id)
        ids, mask = ids.to(device), mask.to(device)
        logits = model(ids, mask, prefix_tensor).logits
        last = mask.sum(dim=1) - 1
        tokens = logits[torch.arange(len(problems), device=device), last].argmax(-1).tolist()
        for i, token in enumerate(tokens):
            if ended[i] or len(sequences[i]) >= model.max_sequence_length:
                continue
            sequences[i].append(token)
            ended[i] = token == tokenizer.eos_token_id
        if all(ended) or all(done or len(s) >= model.max_sequence_length for done, s in zip(ended, sequences)):
            break
    result = []
    for i, sequence in enumerate(sequences):
        tokens = sequence[prefixes[i]:]
        result.append({"generation": tokenizer.decode(tokens), "generated_tokens": len(tokens),
                       "eos_terminated": ended[i], "budget_hit": not ended[i] and len(tokens) >= cap,
                       "unexpected_control_token": any(t in (0, 1, 3) for t in tokens),
                       "context_limit_hit": not ended[i] and len(sequence) >= model.max_sequence_length})
    return result


def native_gate(report: dict, baseline: dict, replay: dict, baseline_replay: dict) -> dict:
    gates = {
        "target_mean_gain_5pp": sum(report[f]["accuracy"] for f in TARGET_FAMILIES) / 2
            >= sum(baseline[f]["accuracy"] for f in TARGET_FAMILIES) / 2 + .05,
        "neither_target_regresses": all(report[f]["accuracy"] >= baseline[f]["accuracy"] for f in TARGET_FAMILIES),
        "replay_drop_at_most_3pp": all(replay[f]["accuracy"] >= baseline_replay[f]["accuracy"] - .03 for f in REPLAY_FAMILIES),
        "broad_drop_at_most_3pp": report["broad_diagnostic"]["accuracy"] >= baseline["broad_diagnostic"]["accuracy"] - .03,
        "zero_target_caps": all(report[f]["budget_hits"] == 0 for f in TARGET_FAMILIES),
    }
    return {"pass": all(gates.values()), "gates": gates, "production_acceptance": False}


def schedule(corpus: dict, stage: str, steps: int, replay: dict, seed: int) -> list[list[dict | str]]:
    rng = random.Random(seed)
    result = []
    for step in range(steps):
        batch = []
        for slot in range(16):
            if stage != "memorization" and slot >= 12:
                batch.append(rng.choice(replay[REPLAY_FAMILIES[slot % 2]]))
                continue
            families = FOUNDATIONS
            if stage == "composition" and slot < 8:
                families = COMPOSITIONS
            family = families[(step * 16 + slot) % len(families)]
            pool = corpus[family]["train"]
            if stage == "memorization":
                pool = sorted(pool, key=lambda q: (len(q), q))[:4]
            batch.append(rng.choice(pool))
        rng.shuffle(batch)
        result.append(batch)
    return result


def run(args) -> None:
    if os.name != "posix":
        raise RuntimeError("all tests and pilots must run on RunPod")
    import fcntl
    # A separate lock prevents two concurrent launches of this diagnostic.
    lock_path = Path(args.output).parent / ".math_primitive_pilot.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)


def run_locked(args) -> None:
    assert_idle()
    source, protected, output = Path(args.source), Path(args.protected), Path(args.output)
    if source.resolve() == protected.resolve():
        raise ValueError("separate capacity source required")
    for path, expected in ((source, SOURCE_SHA), (protected, PROTECTED_SHA)):
        if file_sha256(path) != expected:
            raise ValueError("source/protected hash mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("clean tested worktree required")
    if not (100 <= args.memory_steps <= 400 and 200 <= args.foundation_steps <= 1000
            and 100 <= args.composition_steps <= 500):
        raise ValueError("bounded diagnostic update limits exceeded")
    parent_manifest, parent_data = checked_derivative(Path(args.data))
    corpus = make_corpus()
    output.mkdir(parents=True, exist_ok=False)
    args.claimed_output = True
    corpus_rows = [{"family": f, "split": s, "questions": questions}
                   for f, pools in corpus.items() for s, questions in pools.items()]
    data_file = write_rows(output / "corpus.jsonl", corpus_rows)
    replay = {f: [b["original"] for b in parent_data["train"] if b["original"]["family"] == f]
              for f in REPLAY_FAMILIES}
    schedules = {stage: schedule(corpus, stage, steps, replay, 826)
                 for stage, steps in (("memorization", args.memory_steps),
                                      ("foundations", args.foundation_steps),
                                      ("composition", args.composition_steps))}
    contract = {"format": "cftn_prerequisite_pilot_v1", "revision": revision,
                "source_sha256": SOURCE_SHA, "protected_sha256": PROTECTED_SHA,
                "parent_derivative_manifest_sha256": parent_manifest["manifest_sha256"],
                "corpus_file": data_file, "corpus_sha256": fingerprint(corpus),
                "corpus_code_sha256": file_sha256(Path(__file__).parents[1] / "cftn_text/math_primitive_data.py"),
                "schedule_hashes": {s: fingerprint(v) for s, v in schedules.items()},
                "seed": 826, "arms": ARMS, "batch_size": 16, "learning_rate": 1e-4,
                "role_weights": {"format": .25, "compute": .5, "copy": .25},
                "optimizer": "fresh AdamW per arm; same optimizer across prerequisite stages",
                "steps": {s: len(v) for s, v in schedules.items()},
                "replay_fraction": "0 in tiny memorization sanity; 0.25 thereafter",
                "max_total_seconds": 2100, "max_seconds_per_training_stage": 600,
                "primitive_generation_cap": 256, "native_generation_cap": 1024,
                "memorization_gate": "training-set exact target >=.95 and valid >=.99, no budget hits",
                "foundation_gate": "every family >=.90 native accuracy (and exact targets for compact work), >=.99 valid/EOS, zero caps; replay drop <=.03",
                "composition_gate": "each family >=.85 native accuracy (and exact targets for compact work), >=.99 valid/EOS, zero caps; replay drop <=.03",
                "native_screen": "targeted mean +.05 vs saved zero-update; neither target worse; replay/broad drop <=.03; zero target caps",
                "production_acceptance": False, "checkpoint_promotion": False,
                "long_training_authorized": False, "inference_solver": False,
                "warning": "Tiny single-seed diagnostic; target lengths/cost differ between arms. New lesson grammar is not open-world math.",
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    atomic_json_dump(contract, output / "contract.json")
    checkpoint = load_checkpoint(source, expected_manifest_sha256=PARENT_SHA)
    if checkpoint["epoch"] != 5 or checkpoint["global_step"] != 62500:
        raise ValueError("unexpected source epoch/step")
    config = load_config(args.config)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = ByteMathTokenizer()
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > contract["max_total_seconds"]:
            raise RuntimeError("bounded pilot wall-time exceeded")

    def status(**fields):
        record = {"elapsed_seconds": time.monotonic() - started, **fields}
        atomic_json_dump(record, output / "status.json")
        print(json.dumps(record), flush=True)

    def build_model():
        torch.manual_seed(826)
        torch.cuda.manual_seed_all(826)
        model = build_math_tower_for_checkpoint(config, checkpoint).cuda()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        if model.answer_head_enabled:
            raise ValueError("expected byte-generation-only math tower")
        return model

    @torch.inference_mode()
    def evaluate(model, panels, directory, arm, stage, cap=256):
        model.eval()
        directory.mkdir(parents=True, exist_ok=False)
        reports = {}
        for family, rows in panels.items():
            tick, generated = time.monotonic(), []
            for offset in range(0, len(rows), 16):
                deadline()
                status(state="evaluating", arm=arm, stage=stage, family=family, batch=offset // 16 + 1)
                chunk = rows[offset:offset + 16]
                decoded = generate_with_termination(model, tokenizer, [r["problem"] for r in chunk], cap)
                for row, decoding in zip(chunk, decoded):
                    text = decoding["generation"]
                    item = {"record_id": row["record_id"], "problem": row["problem"],
                            "expected": row["normalized_answer"], "expected_trace": row["target_trace"],
                            **primitive_score(row, text, cap), **decoding}
                    clean_stop = decoding["eos_terminated"] and not decoding["unexpected_control_token"]
                    item["format_valid"] &= clean_stop
                    item["correct"] &= clean_stop
                    item["exact_target"] &= clean_stop
                    generated.append(item)
                verified_snapshot(generated, directory / f"{family}.generations.json")
            n = len(rows)
            report = {"examples": n, "correct": sum(r["correct"] for r in generated),
                      "accuracy": sum(r["correct"] for r in generated) / n,
                      "valid_rate": sum(r["format_valid"] for r in generated) / n,
                      "exact_target_rate": sum(r["exact_target"] for r in generated) / n,
                      "budget_hits": sum(r["budget_hit"] for r in generated),
                      "elapsed_seconds": time.monotonic() - tick,
                      "generated_bytes": sum(len(r["generation"].encode()) for r in generated)}
            if family in REPLAY_FAMILIES or stage == "native":
                metrics, successes = score_v2_generations([g["generation"] for g in generated], rows)
                # Keep standard V2 scoring alongside stricter termination checks.
                clean = [g["eos_terminated"] and not g["unexpected_control_token"] for g in generated]
                report["accuracy"] = sum(ok and stop for ok, stop in zip(successes, clean)) / n
                report["valid_rate"] = sum(g["format_valid"] for g in generated) / n
                report["v2_metrics"] = metrics
            reports[family] = report
            atomic_json_dump(reports, directory / "validation.json")
            print(json.dumps({"event": "panel", "arm": arm, "stage": stage, "family": family, **report}), flush=True)
        return reports

    def panels(arm, families, split="validation", micro=False):
        result = {}
        for family in families:
            questions = corpus[family][split]
            if micro:
                questions = sorted(questions, key=lambda q: (len(q), q))[:4]
            result[family] = [lesson(q, arm) for q in questions]
        if not micro:
            result.update({f: parent_data[f] for f in REPLAY_FAMILIES})
        return result

    reports, costs, decisions = {}, {}, {}
    baseline = build_model()
    reports["baseline"] = evaluate(baseline, panels("compact_worked", FOUNDATIONS), output / "baseline", "baseline", "foundations")
    native_panels = {f: parent_data[f] for f in TARGET_FAMILIES + ("broad_diagnostic",)}
    reports["baseline_native"] = evaluate(baseline, native_panels, output / "baseline_native", "baseline", "native", 1024)
    del baseline
    torch.cuda.empty_cache()
    for arm in ARMS:
        model = build_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        collator = PrimitiveCollator(tokenizer, model.max_sequence_length)
        cache = {}
        arm_dir = output / arm
        arm_dir.mkdir()
        reports[arm], costs[arm] = {}, {}
        training_metrics = []
        for stage in ("memorization", "foundations", "composition"):
            tick, tokens = time.monotonic(), 0
            torch.cuda.reset_peak_memory_stats()
            model.train()
            for step, examples in enumerate(schedules[stage], 1):
                deadline()
                if time.monotonic() - tick > 600:
                    raise RuntimeError("bounded training-stage time exceeded")
                rows = []
                for example in examples:
                    if isinstance(example, str):
                        if example not in cache:
                            cache[example] = lesson(example, arm)
                        rows.append(cache[example])
                    else:
                        rows.append(example)
                batch = to_device(collator(rows), "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(batch["math_input_ids"], batch["math_attention_mask"], batch["math_prefix_lengths"]).logits
                    loss = computation_loss(logits, batch["math_labels"], batch["math_roles"],
                                            weights=(.25, .5, .25), require_computation=False)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite primitive loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                tokens += int(batch["math_labels"].ne(-100).sum())
                if step == 1 or step % 25 == 0 or step == len(schedules[stage]):
                    record = {"state": "training", "arm": arm, "stage": stage, "step": step,
                              "steps": len(schedules[stage]), "loss_training_batch": float(loss.detach()),
                              "gradient_norm": float(grad)}
                    training_metrics.append(record)
                    verified_snapshot(training_metrics, arm_dir / "metrics.json")
                    status(**record)
            torch.cuda.synchronize()
            costs[arm][stage] = {"training_seconds": time.monotonic() - tick, "supervised_tokens": tokens,
                                 "examples": len(schedules[stage]) * 16,
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            atomic_torch_save({"format": "cftn_primitive_pilot_not_promotable_v1",
                               "model_state": model.cpu().state_dict(), "effective_math_tower": model.config,
                               "contract": contract, "arm": arm, "stage": stage}, arm_dir / f"{stage}.pth")
            model.cuda()
            if stage == "memorization":
                panel = panels(arm, FOUNDATIONS, "train", micro=True)
            else:
                panel = panels(arm, FOUNDATIONS if stage == "foundations" else FOUNDATIONS + COMPOSITIONS)
            report = evaluate(model, panel, arm_dir / stage, arm, stage)
            reports[arm][stage] = report
            if stage == "memorization":
                passed = (sum(v["exact_target_rate"] * v["examples"] for v in report.values()) / 28 >= .95
                          and all(v["valid_rate"] >= .99 and v["budget_hits"] == 0 for v in report.values()))
                decision = {"pass": passed, "training_recall_only": True}
            else:
                decision = prerequisite_gate(report, reports["baseline"], require_exact_trace=arm == "compact_worked")
                if stage == "composition":
                    decision["composition_gates"] = {f: (report[f]["accuracy"] >= .85
                        and report[f]["valid_rate"] >= .99 and report[f]["budget_hits"] == 0
                        and (arm != "compact_worked" or report[f]["exact_target_rate"] >= .85)) for f in COMPOSITIONS}
                    decision["pass"] &= all(decision["composition_gates"].values())
            decisions[f"{arm}/{stage}"] = decision
            atomic_json_dump({"reports": reports, "costs": costs, "decisions": decisions}, output / "interim.json")
            print(json.dumps({"event": "stage_gate", "arm": arm, "stage": stage, **decision}), flush=True)
            if not decision["pass"]:
                break
        # Always quantify replay at the terminal candidate, even a recall failure.
        reports[arm]["terminal_replay"] = evaluate(model, {f: parent_data[f] for f in REPLAY_FAMILIES},
                                                  arm_dir / "terminal_replay", arm, "terminal_replay")
        if decisions.get(f"{arm}/composition", {}).get("pass"):
            # This is a separate native transfer check, never a pipeline release.
            reports[arm]["native"] = evaluate(model, native_panels, arm_dir / "native", arm, "native", 1024)
            decisions[f"{arm}/native"] = native_gate(reports[arm]["native"], reports["baseline_native"],
                                                      reports[arm]["terminal_replay"], reports["baseline"])
        else:
            decisions[f"{arm}/native"] = {"state": "not_run_prerequisite_failed", "pass": False}
        del optimizer, model
        torch.cuda.empty_cache()
    for path, expected in ((source, SOURCE_SHA), (protected, PROTECTED_SHA)):
        if file_sha256(path) != expected:
            raise RuntimeError("protected/source checkpoint changed during pilot")
    result = {"state": "completed", "reports": reports, "costs": costs, "decisions": decisions,
              "source_preserved": True, "production_acceptance": False, "checkpoint_promotion": False,
              "elapsed_seconds": time.monotonic() - started, "contract": contract}
    atomic_json_dump(result, output / "summary.json")
    status(state="completed", decisions=decisions, production_acceptance=False)


def launch(args) -> None:
    assert_idle()
    output = Path(args.output).resolve()
    targets = [output, Path(str(output) + ".stdout.log"), Path(str(output) + ".stderr.log"), Path(str(output) + ".launcher.json")]
    if any(p.exists() for p in targets):
        raise FileExistsError("fresh artifact, logs and launcher targets required")
    command = [sys.executable, "-u", "-m", "tools.pilot_math_primitives", "run"]
    for key in ("source", "protected", "data", "output", "config", "memory_steps", "foundation_steps", "composition_steps"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    # setsid-equivalent session isolation plus an outer hard time limit. The
    # timeout process is the process-group leader; the Python child is recorded
    # by status and visible in the exact process tree. SSH is not its parent.
    command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30", "2400"] + command
    with targets[1].open("x") as stdout, targets[2].open("x") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=True, cwd=Path.cwd())
    record = {"pid": process.pid, "pgid": process.pid, "command": command,
              "worktree": str(Path.cwd()), "output": str(output), "launched_at": time.time()}
    atomic_json_dump(record, targets[3])
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "run"))
    for field in ("source", "protected", "data", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--memory-steps", type=int, default=300)
    parser.add_argument("--foundation-steps", type=int, default=800)
    parser.add_argument("--composition-steps", type=int, default=400)
    args = parser.parse_args()
    try:
        (launch if args.command == "launch" else run)(args)
    except Exception as exc:
        if getattr(args, "claimed_output", False):
            atomic_json_dump({"state": "error", "error": str(exc), "production_acceptance": False}, Path(args.output) / "status.json")
        raise


if __name__ == "__main__":
    main()
