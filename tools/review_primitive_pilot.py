"""Validate a completed diagnostic and preserve explicitly recovered evidence.

Only leading NUL padding in JSONL is recoverable. Every original remains intact.
All records, order, gold fields and scores must agree with the sealed corpus and
completed in-memory summary; a corrupt/truncated record is not guessed/rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from cftn_text.data_generator import file_sha256
from cftn_text.math_primitive_data import COMPOSITIONS, FOUNDATIONS, lesson
from cftn_text.v2_metrics import score_v2_generations
from cftn_text.verified_math_data import fingerprint
from tools.pilot_math_primitives import primitive_score, verified_snapshot
from tools.pilot_verified_math import PROTECTED_SHA, SOURCE_SHA, checked_derivative


def parse_padded_jsonl(raw: bytes) -> tuple[list[dict], int]:
    remainder = raw.lstrip(b"\0")
    padding = len(raw) - len(remainder)
    if padding > 8192 or b"\0" in remainder:
        raise ValueError("not solely bounded leading NUL padding")
    rows = [json.loads(line) for line in remainder.decode("utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("expected complete JSONL object records")
    return rows, padding


def check_checkpoint_contract(checkpoint: dict, contract: dict) -> None:
    # Torch preserves tuples while the JSON summary round-trip turns them into
    # arrays. Compare canonical serialized content, not Python container types.
    if (checkpoint["format"] != "cftn_primitive_pilot_not_promotable_v1"
            or fingerprint(checkpoint["contract"]) != fingerprint(contract)):
        raise ValueError("unexpected pilot checkpoint contract")


def review(root: Path, output: Path, data_root: Path, source: Path, protected: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("fresh evidence directory outside original run required")
    summary = json.loads((root / "summary.json").read_text())
    if summary["state"] != "completed" or not summary["source_preserved"]:
        raise ValueError("only terminal, source-preserved diagnostic can be reviewed")
    for path, expected in ((source, SOURCE_SHA), (protected, PROTECTED_SHA)):
        if file_sha256(path) != expected:
            raise ValueError("protected/source checkpoint changed")
    manifest, data = checked_derivative(data_root)
    if manifest["manifest_sha256"] != summary["contract"]["parent_derivative_manifest_sha256"]:
        raise ValueError("different parent derivative")
    corpus_rows, corpus_padding = parse_padded_jsonl((root / "corpus.jsonl").read_bytes())
    if corpus_padding:
        raise ValueError("corpus padding is not silently repaired")
    corpus = {}
    for row in corpus_rows:
        corpus.setdefault(row["family"], {})[row["split"]] = row["questions"]
    if fingerprint(corpus) != summary["contract"]["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    plans, source_hashes, recovered = [], {}, []
    for path in sorted(root.rglob("*.json*")):
        raw = path.read_bytes()
        relative = str(path.relative_to(root))
        source_hashes[relative] = hashlib.sha256(raw).hexdigest()
        if path.suffix == ".json":
            json.loads(raw.decode("utf-8"))
            continue
        rows, padding = parse_padded_jsonl(raw)
        if ".generations." not in path.name:
            if padding:
                raise ValueError("non-generation padding requires separate investigation")
            continue
        family = path.name.removesuffix(".generations.jsonl")
        parts = path.relative_to(root).parts
        arm = parts[0]
        stage = parts[1] if len(parts) == 3 else ("native" if arm == "baseline_native" else "foundations")
        panel = summary["reports"][arm][family] if arm.startswith("baseline") else summary["reports"][arm][stage][family]
        if family in FOUNDATIONS + COMPOSITIONS:
            questions = corpus[family]["train" if stage == "memorization" else "validation"]
            if stage == "memorization":
                questions = sorted(questions, key=lambda q: (len(q), q))[:4]
            expected = [lesson(q, "compact_worked" if arm.startswith("baseline") else arm) for q in questions]
        else:
            expected = data[family]
        if len(rows) != len(expected) or len(rows) != panel["examples"]:
            raise ValueError(f"missing/extra generation records: {relative}")
        cap = 1024 if stage == "native" else 256
        for saved, gold in zip(rows, expected):
            for field, original in (("record_id", "record_id"), ("problem", "problem"),
                                    ("expected", "normalized_answer"), ("expected_trace", "target_trace")):
                if saved[field] != gold[original]:
                    raise ValueError(f"record identity/gold mismatch: {relative}")
            score = primitive_score(gold, saved["generation"], cap)
            if any(saved[k] != v for k, v in score.items()):
                raise ValueError(f"saved generation score mismatch: {relative}")
        if sum(row["correct"] for row in rows) != panel["correct"]:
            raise ValueError("summary count differs from complete saved records")
        accuracy = sum(row["correct"] for row in rows) / len(rows)
        if family not in FOUNDATIONS + COMPOSITIONS:
            metrics, _ = score_v2_generations([v["generation"] for v in rows], expected)
            accuracy = metrics["accuracy"]
        if not math.isclose(accuracy, panel["accuracy"], abs_tol=1e-12):
            raise ValueError("summary accuracy differs from reconstructed evidence")
        plans.append((path.relative_to(root).with_suffix(".json"), rows))
        if padding:
            recovered.append({"path": relative, "leading_nul_bytes": padding, "records": len(rows),
                              "original_sha256": source_hashes[relative],
                              "action": "explicit separate canonical copy; no record reconstructed or original overwritten"})
    checkpoints = {}
    torch.set_num_threads(4)
    for path in sorted(root.rglob("*.pth")):
        digest = file_sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        check_checkpoint_contract(checkpoint, summary["contract"])
        if not all(bool(torch.isfinite(v).all()) for v in checkpoint["model_state"].values() if torch.is_tensor(v)):
            raise ValueError("non-finite saved model weights")
        if file_sha256(path) != digest:
            raise ValueError("checkpoint changed during integrity review")
        checkpoints[str(path.relative_to(root))] = {"sha256": digest, "finite": True, "production_eligible": False}
        del checkpoint
    output.mkdir(parents=True, exist_ok=False)
    for relative, rows in plans:
        verified_snapshot(rows, output / relative)
    for relative, digest in source_hashes.items():
        if file_sha256(root / relative) != digest:
            raise ValueError("original evidence changed during review")
    report = {"state": "completed", "source_run": str(root), "originals_preserved": True,
              "all_records_and_counts_verified": True, "panels": len(plans), "recovered": recovered,
              "original_file_hashes": source_hashes, "checkpoints": checkpoints,
              "protected_sha256": PROTECTED_SHA, "source_sha256": SOURCE_SHA,
              "production_acceptance": False,
              "cause": "Leading NUL holes observed after append+fsync on migrated volume; precise filesystem cause unproven. Not attributed to SSH."}
    verified_snapshot(report, output / "integrity_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "output", "data", "source", "protected"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.run, args.output, args.data, args.source, args.protected)))


if __name__ == "__main__":
    main()
