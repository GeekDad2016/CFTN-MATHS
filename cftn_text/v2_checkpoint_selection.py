from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import (
    atomic_copy_file,
    atomic_json_dump,
    ensure_directory,
    load_checkpoint,
)
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


def _candidate_provenance(
    path: Path,
    *,
    artifact_root: Path,
    math_root: Path,
    explicit: bool,
) -> dict[str, Any]:
    """Validate the trust boundary for a math checkpoint selection candidate."""

    resolved = path.resolve()
    if resolved.is_relative_to(math_root.resolve()):
        return {"kind": "native_math_artifact"}
    if not explicit or not resolved.is_relative_to(artifact_root.resolve()):
        raise ValueError(
            f"V2 math selection candidate is outside the math artifact: {resolved}"
        )

    recovery_root = resolved.parent
    summary_path = recovery_root / "summary.json"
    contract_path = recovery_root / "recovery_contract.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"external V2 math candidate lacks recovery attestation: {resolved}"
        ) from exc

    metrics = summary.get("final_metrics") or {}
    acceptance = metrics.get("curriculum_acceptance") or {}
    minimum_accuracy = float(acceptance.get("minimum_generation_accuracy", 1.0))
    minimum_valid_rate = float(acceptance.get("minimum_valid_rate", 1.0))
    observed_accuracy = float(acceptance.get("generation_accuracy", 0.0))
    observed_valid_rate = float(acceptance.get("valid_rate", 0.0))
    candidate_sha256 = file_sha256(resolved)
    source_path = Path(str(contract.get("source_checkpoint", ""))).expanduser()
    source_sha256 = str(contract.get("source_checkpoint_sha256", ""))
    checks = {
        "contract_format": contract.get("format")
        == "cftn_text_v2_math_shared_trace_recovery_v1",
        "require_acceptance_for_best": contract.get("require_acceptance_for_best")
        is True,
        "summary_completed": summary.get("state") == "completed",
        "summary_best_path": Path(str(summary.get("best_checkpoint", ""))).resolve()
        == resolved,
        "summary_best_sha256": summary.get("best_checkpoint_sha256")
        == candidate_sha256,
        "acceptance_pass": acceptance.get("pass") is True,
        "generation_accuracy": observed_accuracy >= minimum_accuracy,
        "valid_rate": observed_valid_rate >= minimum_valid_rate,
        "checkpoint_eligible": metrics.get("checkpoint_eligible") is True,
        "checkpoint_promoted": metrics.get("checkpoint_promoted") is True,
        "input_view": metrics.get("input_view") == "shared_problem_v1",
        "target_mode": metrics.get("target_mode") == "full_trace_v1",
        "source_exists": source_path.is_file(),
        "source_sha256": bool(source_sha256)
        and source_path.is_file()
        and file_sha256(source_path) == source_sha256,
        "observed_source_sha256": contract.get("observed_source_checkpoint_sha256")
        == source_sha256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "external V2 math recovery candidate failed attestation: "
            + ", ".join(failed)
        )
    return {
        "kind": "accepted_math_recovery",
        "recovery_root": str(recovery_root),
        "summary": str(summary_path),
        "contract": str(contract_path),
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "generation_accuracy": observed_accuracy,
        "minimum_generation_accuracy": minimum_accuracy,
        "valid_rate": observed_valid_rate,
        "minimum_valid_rate": minimum_valid_rate,
    }


def select_v2_math_checkpoint(
    config: dict[str, Any],
    *,
    device_name: str = "cuda",
    candidate_paths: list[str | Path] | None = None,
    working_root: str | Path | None = None,
    reuse_completed: bool = True,
) -> dict[str, Any]:
    root = Path(config["project"]["artifact_root"])
    math_root = root / "math"
    selection_root = root / "math_checkpoint_selection"
    ensure_directory(selection_root)
    settings = config.get("checkpoint_selection", {})
    maximum = int(settings.get("generation_examples", 512))
    split = str(settings.get("split", "validation"))

    paths = (
        [Path(path) for path in candidate_paths]
        if candidate_paths
        else [
            math_root / "math.best.pth",
            *sorted(math_root.glob("checkpoint_epoch_*.pth")),
        ]
    )
    unique: list[tuple[Path, str, dict[str, Any]]] = []
    seen_hashes: set[str] = set()
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            if candidate_paths:
                raise FileNotFoundError(path)
            continue
        provenance = _candidate_provenance(
            path,
            artifact_root=root,
            math_root=math_root,
            explicit=bool(candidate_paths),
        )
        digest = file_sha256(path)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique.append((path, digest, provenance))
    if not unique:
        raise FileNotFoundError("V2 math checkpoint selection found no candidates")

    _, manifest = load_data_contract(config)
    expected_config_sha256 = config_sha256(config)
    candidates: list[dict[str, Any]] = []
    for path, digest, provenance in unique:
        checkpoint = load_checkpoint(
            path,
            expected_stage="math",
            expected_config_sha256=expected_config_sha256,
            expected_manifest_sha256=manifest["manifest_sha256"],
            map_location="cpu",
        )
        metrics = checkpoint.get("extra", {}).get("metrics", {})
        validation = metrics.get("validation", {})
        candidate_name = f"epoch_{int(checkpoint['epoch']):04d}_{digest[:12]}"
        candidate_root = selection_root / candidate_name
        evaluation_path = candidate_root / "report.json"
        evaluation: dict[str, Any] | None = None
        evaluation_reused = False
        if reuse_completed and evaluation_path.is_file():
            try:
                loaded = json.loads(evaluation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            generated = loaded.get("splits", {}).get(split, {})
            if (
                loaded.get("format") == "cftn_text_math_evaluation_v2"
                and loaded.get("checkpoint_sha256") == digest
                and loaded.get("config_sha256") == expected_config_sha256
                and loaded.get("manifest_sha256") == manifest["manifest_sha256"]
                and int(generated.get("examples", -1)) == maximum
            ):
                evaluation = loaded
                evaluation_reused = True
        if evaluation is None:
            candidate_working_root = (
                Path(working_root).expanduser().resolve() / candidate_name
                if working_root
                else None
            )
            evaluation = evaluate_v2_math_checkpoint(
                config,
                path,
                device_name=device_name,
                splits=[split],
                maximum_examples=maximum,
                output_root=candidate_root,
                working_root=candidate_working_root,
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
            "evaluation_report": str(evaluation_path.resolve()),
            "evaluation_reused": evaluation_reused,
            "provenance": provenance,
        }
        candidate["score"] = list(candidate_score(candidate))
        candidates.append(candidate)

    selected = max(candidates, key=candidate_score)
    selected_path = root / "math_selected" / "math.selected.pth"
    atomic_copy_file(selected["path"], selected_path)
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
        "candidate_scope": "explicit" if candidate_paths else "all_retained",
        "requested_candidates": (
            [str(Path(path).expanduser().resolve()) for path in candidate_paths]
            if candidate_paths
            else None
        ),
        "working_root": str(Path(working_root).expanduser().resolve())
        if working_root
        else None,
        "reuse_completed": bool(reuse_completed),
        "candidates": candidates,
        "selected": {
            **selected,
            "source_path": selected["path"],
            "path": str(selected_path.resolve()),
        },
    }
    atomic_json_dump(report, selection_root / "report.json")
    return report
