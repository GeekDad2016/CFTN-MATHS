from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump, load_checkpoint
from .config import config_sha256
from .data_generator import file_sha256
from .training import load_data_contract
from .v2_evaluation import evaluate_v2_math_checkpoint


def candidate_score(candidate: dict[str, Any]) -> tuple[float, float, float, float, int]:
    """Generation is primary; teacher forcing only resolves generation ties."""

    return (
        float(candidate["generation_accuracy"]),
        float(candidate["valid_answer_rate"]),
        float(candidate["teacher_forced_sequence_accuracy"]),
        -float(candidate["validation_loss"]),
        int(candidate["epoch"]),
    )


def select_v2_math_checkpoint(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
) -> dict[str, Any]:
    root = Path(config["project"]["artifact_root"])
    math_root = root / "math"
    selection_root = root / "math_checkpoint_selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    settings = config.get("checkpoint_selection", {})
    maximum = int(settings.get("generation_examples", 512))
    split = str(settings.get("split", "validation"))

    paths = [math_root / "math.best.pth", *sorted(math_root.glob("checkpoint_epoch_*.pth"))]
    unique: list[tuple[Path, str]] = []
    seen_hashes: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        digest = file_sha256(path)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique.append((path, digest))
    if not unique:
        raise FileNotFoundError("V2 math checkpoint selection found no candidates")

    _, manifest = load_data_contract(config)
    candidates: list[dict[str, Any]] = []
    for path, digest in unique:
        checkpoint = load_checkpoint(
            path,
            expected_stage="math",
            expected_config_sha256=config_sha256(config),
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location="cpu",
        )
        metrics = checkpoint.get("extra", {}).get("metrics", {})
        validation = metrics.get("validation", {})
        evaluation = evaluate_v2_math_checkpoint(
            config,
            path,
            device_name=device_name,
            splits=[split],
            maximum_examples=maximum,
            output_root=selection_root / f"epoch_{int(checkpoint['epoch']):04d}_{digest[:12]}",
        )
        generated = evaluation["splits"][split]
        candidate = {
            "path": str(path.resolve()),
            "sha256": digest,
            "epoch": int(checkpoint["epoch"]),
            "generation_examples": int(generated["examples"]),
            "generation_accuracy": float(generated["accuracy"]),
            "valid_answer_rate": float(generated["valid_rate"]),
            "teacher_forced_sequence_accuracy": float(
                validation.get("teacher_forced_sequence_accuracy", 0.0)
            ),
            "validation_loss": float(validation.get("loss", 1.0e9)),
        }
        candidate["score"] = list(candidate_score(candidate))
        candidates.append(candidate)

    selected = max(candidates, key=candidate_score)
    selected_path = root / "math_selected" / "math.selected.pth"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected_path.with_name(f".{selected_path.name}.tmp")
    shutil.copy2(selected["path"], temporary)
    temporary.replace(selected_path)
    selected_sha = file_sha256(selected_path)
    if selected_sha != selected["sha256"]:
        raise RuntimeError("selected V2 math checkpoint copy failed hash verification")

    report = {
        "format": "cftn_text_v2_math_checkpoint_selection_v1",
        "state": "completed",
        "selection_policy": (
            "validation greedy-generation accuracy, then valid-answer rate, "
            "teacher-forced sequence accuracy, lower validation loss, later epoch"
        ),
        "selection_split": split,
        "sealed_test_splits_used": False,
        "candidates": candidates,
        "selected": {**selected, "path": str(selected_path.resolve())},
    }
    atomic_json_dump(report, selection_root / "report.json")
    return report
