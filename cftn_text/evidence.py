from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assemble_evidence_report(
    math_report_path: str | Path,
    shared_report_path: str | Path,
    synergy_report_path: str | Path,
    *,
    architecture_comparison_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    math_report = _load(math_report_path)
    shared_report = _load(shared_report_path)
    synergy_report = _load(synergy_report_path)
    if math_report.get("format") != "cftn_text_math_evaluation_v1":
        raise ValueError("invalid standalone math evaluation report")
    if shared_report.get("format") != "cftn_text_evaluation_v1":
        raise ValueError("invalid shared-view CFTN evaluation report")
    if synergy_report.get("format") != "cftn_text_causal_synergy_evaluation_v1":
        raise ValueError("invalid causal synergy evaluation report")
    config_hashes = {
        math_report.get("config_sha256"),
        shared_report.get("config_sha256"),
        synergy_report.get("config_sha256"),
    }
    if len(config_hashes) != 1:
        raise ValueError("evidence reports use different configurations")
    source_hashes = {
        math_report.get("manifest_sha256"),
        shared_report.get("manifest_sha256"),
        synergy_report.get("source_manifest_sha256"),
    }
    if len(source_hashes) != 1:
        raise ValueError("evidence reports use different source manifests")
    architecture = (
        _load(architecture_comparison_path)
        if architecture_comparison_path is not None
        else None
    )
    if architecture is not None and architecture.get("format") != (
        "cftn_text_synergy_architecture_comparison_v1"
    ):
        raise ValueError("invalid architecture comparison report")
    specialist_pass = bool(math_report.get("specialist_gate", {}).get("pass"))
    shared_utility_pass = bool(
        shared_report.get("preregistered_gates", {}).get("collaboration_gate_pass")
    )
    causal_pass = bool(
        synergy_report.get("causal_collaboration_gate", {}).get("pass")
    )
    architecture_pass = (
        bool(architecture.get("architecture_claim_pass"))
        if architecture is not None
        else None
    )
    report = {
        "format": "cftn_text_evidence_bundle_v1",
        "config_sha256": next(iter(config_hashes)),
        "manifest_sha256": next(iter(source_hashes)),
        "inputs": {
            "math_report": str(Path(math_report_path).resolve()),
            "shared_report": str(Path(shared_report_path).resolve()),
            "synergy_report": str(Path(synergy_report_path).resolve()),
            "architecture_comparison": (
                str(Path(architecture_comparison_path).resolve())
                if architecture_comparison_path is not None
                else None
            ),
        },
        "claims": {
            "standalone_specialist_pass": specialist_pass,
            "shared_view_utility_pass": shared_utility_pass,
            "causal_collaboration_pass": causal_pass,
            "contextual_gate_architecture_pass": architecture_pass,
            "two_towers_work_together": specialist_pass and causal_pass,
            "full_cftn_architecture_claim": (
                specialist_pass
                and shared_utility_pass
                and causal_pass
                and architecture_pass is True
            ),
        },
        "important_interpretation": {
            "shared_view": (
                "Tests practical performance when both towers see the full prompt."
            ),
            "complementary_view": (
                "Tests causal information exchange when neither private view is sufficient."
            ),
            "architecture_comparison": (
                "Required before claiming contextual gates beat conventional fixed-open communication."
            ),
        },
    }
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(report, root / "evidence.json")
        claims = report["claims"]
        lines = [
            "# CFTN-Text evidence bundle",
            "",
            "| Claim | Result |",
            "|---|---:|",
        ]
        for name, value in claims.items():
            rendered = "NOT RUN" if value is None else "PASS" if value else "FAIL"
            lines.append(f"| {name} | {rendered} |")
        (root / "evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
