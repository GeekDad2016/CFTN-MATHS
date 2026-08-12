from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump
from .data_generator import file_sha256


V1_2_REPORT_FORMAT = "cftn_text_v1_2_revision_report_v1"
V1_3_REPORT_FORMAT = "cftn_text_v1_3_revision_report_v1"


def _resolve(path: str | Path, repository_root: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else repository_root / value).resolve()


def _load_report(path: Path, expected_format: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} is missing: {path}. V2 fails closed until this sealed report is supplied."
        )
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict) or report.get("format") != expected_format:
        raise ValueError(f"{label} has an unsupported format: {path}")
    if report.get("final_gates", {}).get("pass") is not True:
        raise RuntimeError(f"{label} did not pass its sealed acceptance gates")
    return report


def audit_v2_mechanism_prerequisites(
    config: dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the mechanism evidence V2 is allowed to scale.

    V1.2 proves conditional, non-destructive message utility. V1.3 proves
    natural-prompt wake decisions and multi-specialist cooperation. V2 must
    not spend the broad-data run while either mechanism is unproven.
    """

    settings = config.get("prerequisites")
    if not isinstance(settings, dict):
        raise ValueError("V2 config requires a prerequisites section")
    repository_root = Path(config["_meta"]["path"]).parent.parent
    v1_2_path = _resolve(settings["v1_2_report"], repository_root)
    v1_3_path = _resolve(settings["v1_3_report"], repository_root)
    v1_2 = _load_report(v1_2_path, V1_2_REPORT_FORMAT, "sealed V1.2 report")
    v1_3 = _load_report(v1_3_path, V1_3_REPORT_FORMAT, "sealed V1.3 report")

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
