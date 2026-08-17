from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump
from .data_generator import file_sha256


V1_2_REPORT_FORMAT = "cftn_text_v1_2_revision_report_v1"
V1_3_REPORT_FORMAT = "cftn_text_v1_3_revision_report_v1"

# V1.3's top-level pass flag is only meaningful when the concrete transition
# gates that motivated V2 are present. In particular, the Stage-10 recovery
# showed that recall can look perfect during an always-open collapse, so V2
# explicitly requires precision, exact routing, false-wake/no-harm, and the
# hard-vs-dense comparison before it is allowed to train communication.
V1_3_REQUIRED_GATES = frozenset(
    {
        "v1_2_prerequisite",
        "gpt_language_precondition",
        "native_specialists_familiar",
        "native_specialists_task_matched",
        "primary_competence_coverage",
        "pure_language_no_harm",
        "pure_language_false_wake",
        "wake_recall",
        "wake_precision",
        "exact_required_set",
        "joint_positive_synergy",
        "math_bridge_causal",
        "string_bridge_causal",
        "messages_content_specific",
        "irrelevant_bridge_no_harm",
        "hard_matches_dense",
        "compute_reduction",
        "sequential_accuracy",
        "multiround_causality",
        "beats_fixed_open",
        "beats_serial_pipeline",
    }
)


def _resolve(path: str | Path, repository_root: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else repository_root / value).resolve()


def _load_report(
    path: Path,
    expected_format: str,
    label: str,
    *,
    required_gates: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} is missing: {path}. V2 fails closed until this sealed report is supplied."
        )
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or report.get("format") != expected_format:
        raise ValueError(f"{label} has an unsupported format: {path}")
    gates = report.get("final_gates", {})
    if not isinstance(gates, dict) or gates.get("pass") is not True:
        raise RuntimeError(f"{label} did not pass its sealed acceptance gates")
    missing = sorted(required_gates.difference(gates))
    failed = sorted(name for name in required_gates if gates.get(name) is not True)
    if missing:
        raise ValueError(f"{label} is missing required sealed gates: {missing}")
    if failed:
        raise RuntimeError(f"{label} failed required sealed gates: {failed}")
    return report


def audit_v2_mechanism_prerequisites(
    config: dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the mechanism evidence V2 is allowed to scale.

    V1.2 proves conditional, non-destructive message utility. V1.3 proves
    natural-prompt wake decisions and multi-specialist cooperation. The
    independent V2 specialist may be trained first, but no V2 communication
    stage is allowed to start while either mechanism is unproven.
    """

    settings = config.get("prerequisites")
    if not isinstance(settings, dict):
        raise ValueError("V2 config requires a prerequisites section")
    repository_root = Path(config["_meta"]["path"]).parent.parent
    v1_2_path = _resolve(settings["v1_2_report"], repository_root)
    v1_3_path = _resolve(settings["v1_3_report"], repository_root)
    v1_2 = _load_report(v1_2_path, V1_2_REPORT_FORMAT, "sealed V1.2 report")
    v1_3 = _load_report(
        v1_3_path,
        V1_3_REPORT_FORMAT,
        "sealed V1.3 report",
        required_gates=V1_3_REQUIRED_GATES,
    )

    v1_2_sha = file_sha256(v1_2_path)
    chained_sha = v1_3.get("prerequisite", {}).get("v1_2_report_sha256")
    if chained_sha != v1_2_sha:
        raise ValueError(
            "V1.3 was not sealed against the supplied V1.2 report; evidence chain differs"
        )

    report = {
        "format": "cftn_text_v2_mechanism_prerequisites_v1",
        "state": "passed",
        "pass": True,
        "v1_2": {
            "path": str(v1_2_path),
            "sha256": v1_2_sha,
            "revision_sha256": v1_2.get("revision_sha256"),
            "final_gates": v1_2["final_gates"],
        },
        "v1_3": {
            "path": str(v1_3_path),
            "sha256": file_sha256(v1_3_path),
            "revision_sha256": v1_3.get("revision_sha256"),
            "final_gates": v1_3["final_gates"],
        },
        "claims_allowed": {
            "conditional_non_destructive_messages": True,
            "natural_prompt_specialist_wake": True,
            "multi_specialist_cooperation": True,
            "broad_math_generalization": False,
        },
    }
    target = Path(
        output_path
        or Path(config["project"]["artifact_root"])
        / "mechanism_prerequisites.json"
    )
    atomic_json_dump(report, target)
    return report
