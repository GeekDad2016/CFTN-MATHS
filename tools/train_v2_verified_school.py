"""Three-epoch V1-style V2 math curriculum repair trial, not pipeline release."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import torch

from cftn_text.checkpoint import atomic_json_dump, atomic_torch_save, capture_rng_state, load_checkpoint
from cftn_text.computation_supervision import computation_loss
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.training import build_math_tower_for_checkpoint
from cftn_text.v2_metrics import score_v2_generations
from cftn_text.v2_school_data import BANDS, FAMILIES, SchoolCollator, build_school_corpus, parse_public, question, school_record
from cftn_text.verified_math_data import REPLAY_FAMILIES, TARGET_FAMILIES, fingerprint
from tools.pilot_math_primitives import generate_with_termination, verified_snapshot
from tools.pilot_verified_math import PARENT_SHA, PROTECTED_SHA, SOURCE_SHA, assert_idle, checked_derivative, to_device


def curriculum_gate(report, baseline_replay, settings):
    gates = {}
    thresholds = settings["band_gate"]
    for family in FAMILIES:
        p = report.get("current/" + family, {})
        gates[family] = (p.get("accuracy", 0) >= thresholds["answer_accuracy"]
                         and p.get("valid_rate", 0) >= thresholds["valid_rate"]
                         and p.get("trace_exact_rate", 0) >= thresholds["trace_exact_rate"]
                         and p.get("budget_hits", 1) == 0)
    gates["replay"] = all(report.get("replay/" + f, {}).get("accuracy", 0)
                           >= baseline_replay["replay/" + f]["accuracy"] - thresholds["maximum_replay_drop"]
                           for f in REPLAY_FAMILIES)
    return {"pass": all(gates.values()), "gates": gates, "production_acceptance": False}


def next_band(index, completed_band_epochs, gate, settings):
    advance = (gate["pass"] and completed_band_epochs >= settings["minimum_epochs_per_band"]
               and index + 1 < len(BANDS))
    return (index + 1, 0) if advance else (index, completed_band_epochs)


def intact_output(decoded):
    return (decoded["eos_terminated"] and not decoded["unexpected_control_token"]
            and not decoded["budget_hit"] and not decoded["context_limit_hit"]
            and re.fullmatch(r"<work>[^<>]+</work><answer>[^<>]+</answer>", decoded["generation"]) is not None)


def epoch_schedule(corpus, replay, band_index, epoch, settings):
    rng = random.Random(settings["seed"] + epoch * 1009)
    for step in range(settings["examples_per_epoch"] // settings["batch_size"]):
        rows = []
        for slot in range(16):
            if slot >= 12:
                rows.append(rng.choice(replay[REPLAY_FAMILIES[slot % 2]]))
            else:
                family = FAMILIES[(step * 12 + slot) % len(FAMILIES)]
                index = band_index if band_index == 0 or rng.random() < .7 else rng.randrange(band_index)
                prototype = rng.choice(corpus[BANDS[index]][family]["train"])
                parsed_family, values = parse_public(prototype["problem"])
                rows.append(school_record(question(parsed_family, values, rng.randrange(3))))
        rng.shuffle(rows)
        yield rows


def settings_checked(path):
    settings = json.loads(Path(path).read_text())
    if (settings["epochs"] != 3 or settings["batch_size"] != 16
            or not 1024 <= settings["examples_per_epoch"] <= 16384
            or settings["examples_per_epoch"] % 16 or settings["replay_fraction"] != .25
            or settings["families"] != list(FAMILIES) or settings["bands"] != list(BANDS)
            or settings["production_acceptance"] or settings["checkpoint_promotion"]
            or settings["minimum_epochs_per_band"] < 2
            or settings["role_weights"] != {"format": .25, "compute": .5, "copy": .25}
            or settings["validation_per_family"] != 64
            or settings["wording_diagnostic_per_family"] != 16
            or settings["next_band_diagnostic_per_family"] != 16
            or settings["max_new_tokens"] != 256
            or settings["legacy_diagnostic_max_new_tokens"] != 512
            or not 0 < settings["max_wall_seconds"] <= 1800
            or not 0 < settings["warmup_updates"] < settings["epochs"] * settings["examples_per_epoch"] // 16):
        raise ValueError("not the authorized bounded school trial")
    gate = settings["band_gate"]
    if (gate["answer_accuracy"] < .99 or gate["valid_rate"] < 1.
            or gate["trace_exact_rate"] < .95 or gate["maximum_replay_drop"] > .03):
        raise ValueError("school gates may not be weakened")
    if not (0 < settings["minimum_learning_rate"] <= settings["learning_rate"] <= 1e-4):
        raise ValueError("learning rate outside bounded repair contract")
    if not (0 < settings["gradient_clip"] <= 1 and 0 <= settings["weight_decay"] <= .01):
        raise ValueError("optimizer outside bounded repair contract")
    return settings


def run(args):
    assert_idle()
    settings = settings_checked(args.trial_config)
    output = Path(args.output)
    for path, digest in ((Path(args.source), SOURCE_SHA), (Path(args.protected), PROTECTED_SHA)):
        if file_sha256(path) != digest:
            raise ValueError("protected/source checkpoint mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("clean tested checkout required")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    parent_manifest, parent_data = checked_derivative(Path(args.replay_data))
    corpus = build_school_corpus()
    output.mkdir(parents=True, exist_ok=False)
    args.claimed = True
    verified_snapshot(corpus, output / "school_corpus.json")
    counts = {b: {f: {s: len(v) for s, v in p.items()} for f, p in families.items()} for b, families in corpus.items()}
    max_length = max(len(ByteMathTokenizer().encode_training_example(r["problem"], r["target_trace"], 4096).input_ids)
                     for families in corpus.values() for pools in families.values() for rows in pools.values() for r in rows)
    manifest = {"format": "cftn_verified_school_manifest_v1", "corpus_sha256": fingerprint(corpus),
                "corpus_file_sha256": file_sha256(output / "school_corpus.json"), "counts": counts,
                "generator_sha256": file_sha256(Path(__file__).parents[1] / "cftn_text/v2_school_data.py"),
                "revision": revision, "maximum_encoded_length": max_length,
                "split_policy": "mathematical object before wording; commutative swaps and scaled linear equations grouped",
                "heldout_wording_style": 3, "training_wording_styles": [0, 1, 2],
                "unverified_imported_rows": 0, "train_validation_object_overlap": 0,
                "replay_manifest_sha256": parent_manifest["manifest_sha256"]}
    manifest["manifest_sha256"] = fingerprint(manifest)
    verified_snapshot(manifest, output / "manifest.json")
    contract = {"format": "cftn_v2_verified_school_trial_v1", "revision": revision, "settings": settings,
                "source_sha256": SOURCE_SHA, "protected_sha256": PROTECTED_SHA,
                "manifest_sha256": manifest["manifest_sha256"], "production_acceptance": False,
                "checkpoint_promotion": False, "architecture_change": False, "tokenizer_change": False,
                "seed": settings["seed"], "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    verified_snapshot(contract, output / "contract.json")
    checkpoint = load_checkpoint(args.source, expected_manifest_sha256=PARENT_SHA)
    config = load_config(args.config)
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_math_tower_for_checkpoint(config, checkpoint).cuda()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.answer_head_enabled:
        raise ValueError("byte generation, not categorical answer classification, required")
    if len(model.blocks) != 24 or model.max_sequence_length != 4096:
        raise ValueError("source architecture unexpectedly changed")
    tokenizer = ByteMathTokenizer()
    collator = SchoolCollator(tokenizer, model.max_sequence_length)
    replay = {f: [b["original"] for b in parent_data["train"] if b["original"]["family"] == f] for f in REPLAY_FAMILIES}
    started = time.monotonic()
    training_metrics, epoch_reports = [], []

    def status(**fields):
        if time.monotonic() - started > settings["max_wall_seconds"]:
            raise RuntimeError("three-epoch trial time budget exceeded")
        record = {"elapsed_seconds": time.monotonic() - started, **fields}
        atomic_json_dump(record, output / "status.json")
        print(json.dumps(record), flush=True)

    def panels(index):
        panel = {"current/" + f: corpus[BANDS[index]][f]["validation"] for f in FAMILIES}
        for family in FAMILIES:
            panel["wording/" + family] = [school_record(question(*parse_public(r["problem"]), 3))
                for r in corpus[BANDS[index]][family]["validation"][:settings["wording_diagnostic_per_family"]]]
            if index + 1 < len(BANDS):
                panel["next_band/" + family] = corpus[BANDS[index + 1]][family]["validation"][:settings["next_band_diagnostic_per_family"]]
        panel.update({"replay/" + f: parent_data[f] for f in REPLAY_FAMILIES})
        return panel

    @torch.inference_mode()
    def evaluate(panel, epoch, tag="validation"):
        model.eval()
        reports = {}
        for name, records in panel.items():
            rows, tick = [], time.monotonic()
            cap = settings["legacy_diagnostic_max_new_tokens"] if tag == "native" else settings["max_new_tokens"]
            for offset in range(0, len(records), 16):
                status(state="evaluating", epoch=epoch, panel=name, batch=offset // 16 + 1)
                chunk = records[offset:offset + 16]
                for record, decoded in zip(chunk, generate_with_termination(model, tokenizer, [r["problem"] for r in chunk], cap)):
                    rows.append({"record_id": record["record_id"], "problem": record["problem"],
                                 "expected": record["normalized_answer"], "expected_trace": record["target_trace"], **decoded})
                verified_snapshot(rows, output / f"epoch_{epoch:03d}" / tag / f"{name}.json")
            metrics, correctness = score_v2_generations([r["generation"] for r in rows], records)
            clean = [r["eos_terminated"] and not r["unexpected_control_token"]
                     and not r["budget_hit"] and not r["context_limit_hit"] for r in rows]
            report = {"examples": len(rows), "accuracy": sum(c and e for c, e in zip(correctness, clean)) / len(rows),
                      "valid_rate": sum(intact_output(r) for r in rows) / len(rows),
                      "trace_exact_rate": sum(e and r["generation"] == r["expected_trace"] for r, e in zip(rows, clean)) / len(rows),
                      "budget_hits": sum(r["budget_hit"] or r["context_limit_hit"] for r in rows),
                      "elapsed_seconds": time.monotonic() - tick, "generated_tokens": sum(r["generated_tokens"] for r in rows),
                      "v2_metrics": metrics}
            reports[name] = report
            verified_snapshot(reports, output / f"epoch_{epoch:03d}" / f"{tag}.json")
            print(json.dumps({"event": "validation", "epoch": epoch, "panel": name,
                              "accuracy": report["accuracy"], "trace_exact": report["trace_exact_rate"]}), flush=True)
        return reports

    baseline = evaluate(panels(0), 0)
    legacy_panel = {f: parent_data[f] for f in TARGET_FAMILIES + ("broad_diagnostic",)}
    baseline_native = evaluate(legacy_panel, 0, "native")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    updates = settings["examples_per_epoch"] // 16
    total_updates = updates * settings["epochs"]
    band_index = band_epochs = global_step = 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        tick, losses, supervised_tokens = time.monotonic(), [], 0
        sampled, objects = collections.Counter(), set()
        torch.cuda.reset_peak_memory_stats()
        for batch_index, records in enumerate(epoch_schedule(corpus, replay, band_index, epoch, settings), 1):
            global_step += 1
            if global_step <= settings["warmup_updates"]:
                lr = settings["learning_rate"] * global_step / settings["warmup_updates"]
            else:
                fraction = (global_step - settings["warmup_updates"]) / (total_updates - settings["warmup_updates"])
                lr = settings["minimum_learning_rate"] + (settings["learning_rate"] - settings["minimum_learning_rate"]) * .5 * (1 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = to_device(collator(records), "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["math_input_ids"], batch["math_attention_mask"], batch["math_prefix_lengths"]).logits
                loss = computation_loss(logits, batch["math_labels"], batch["math_roles"], weights=(.25, .5, .25))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite school training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            supervised_tokens += int(batch["math_labels"].ne(-100).sum())
            sampled.update(r["family"] for r in records)
            objects.update(r.get("computation_key", r["record_id"]) for r in records)
            if batch_index == 1 or batch_index % 25 == 0 or batch_index == updates:
                row = {"epoch": epoch, "band": BANDS[band_index], "batch": batch_index, "batches": updates,
                       "global_step": global_step, "loss_training_average": sum(losses) / len(losses),
                       "learning_rate": lr, "gradient_norm": float(grad)}
                training_metrics.append(row)
                verified_snapshot(training_metrics, output / "metrics.json")
                status(state="training", **row)
        torch.cuda.synchronize()
        cost = {"training_seconds": time.monotonic() - tick, "supervised_tokens": supervised_tokens,
                "sampled_by_family": dict(sampled), "distinct_objects_sampled": len(objects),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        validation = evaluate(panels(band_index), epoch)
        gate = curriculum_gate(validation, baseline, settings)
        band_epochs += 1
        epoch_report = {"epoch": epoch, "band": BANDS[band_index], "training_loss": sum(losses) / len(losses),
                        "validation": validation, "curriculum_gate": gate, "cost": cost}
        epoch_reports.append(epoch_report)
        verified_snapshot(epoch_reports, output / "epoch_reports.json")
        checkpoint_path = output / f"checkpoint_epoch_{epoch:04d}.pth"
        atomic_torch_save({"format": "cftn_v2_school_trial_not_promotable_v1", "model_state": model.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "rng_state": capture_rng_state(),
                           "effective_math_tower": model.config, "epoch": epoch, "global_step": global_step,
                           "band": BANDS[band_index], "band_epochs": band_epochs,
                           "contract": contract, "report": epoch_report}, checkpoint_path)
        verified_snapshot({"path": checkpoint_path.name, "sha256": file_sha256(checkpoint_path),
                           "production_eligible": False}, output / f"checkpoint_epoch_{epoch:04d}.json")
        print(json.dumps({"event": "epoch_completed", "epoch": epoch, "band": BANDS[band_index], "gate": gate}), flush=True)
        band_index, band_epochs = next_band(band_index, band_epochs, gate, settings)
    final_native = evaluate(legacy_panel, settings["epochs"], "native")
    for path, digest in ((Path(args.source), SOURCE_SHA), (Path(args.protected), PROTECTED_SHA)):
        if file_sha256(path) != digest:
            raise RuntimeError("protected/source checkpoint changed during trial")
    result = {"state": "completed", "trial_only": True, "epochs": epoch_reports,
              "baseline": baseline, "baseline_native": baseline_native, "final_native": final_native,
              "source_preserved": True, "production_acceptance": False, "checkpoint_promotion": False,
              "elapsed_seconds": time.monotonic() - started, "contract": contract, "manifest": manifest}
    verified_snapshot(result, output / "summary.json")
    status(state="completed", epochs=3, final_band=epoch_reports[-1]["band"], production_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "run"))
    for name in ("source", "protected", "replay-data", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--trial-config", default="config/v2_verified_school_trial.json")
    parser.add_argument("--launcher-pid", type=int)
    args = parser.parse_args()
    if args.command == "launch":
        assert_idle()
        settings_checked(args.trial_config)
        output = Path(args.output)
        paths = [output, Path(str(output) + ".stdout.log"), Path(str(output) + ".stderr.log"), Path(str(output) + ".launcher.json")]
        if any(p.exists() for p in paths):
            raise FileExistsError("fresh trial output/log/launcher targets required")
        command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30", "1920", sys.executable, "-u", "-m", "tools.train_v2_verified_school", "run"]
        for key in ("source", "protected", "replay_data", "output", "config", "trial_config"):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        command += ["--launcher-pid", str(os.getpid())]
        with paths[1].open("x") as stdout, paths[2].open("x") as stderr:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
        launch = {"pid": proc.pid, "pgid": proc.pid, "command": command, "worktree": str(Path.cwd())}
        verified_snapshot(launch, paths[3])
        print(json.dumps(launch), flush=True)
    else:
        # The launch parent has no model; wait for it to exit before the strict
        # duplicate-process check rather than exempting another Python process.
        if args.launcher_pid:
            for _ in range(100):
                if not Path(f"/proc/{args.launcher_pid}").exists():
                    break
                time.sleep(.1)
        # Lock shared with the earlier diagnostic; no concurrent experiment.
        import fcntl
        with (Path(args.output).parent / ".math_primitive_pilot.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(args)
            except Exception as exc:
                if getattr(args, "claimed", False):
                    atomic_json_dump({"state": "error", "error": str(exc)}, Path(args.output) / "status.json")
                raise


if __name__ == "__main__":
    main()
