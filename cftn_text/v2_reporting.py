from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assess_scale_gate(config: dict[str, Any]) -> dict[str, Any]:
    artifact_root = Path(config["project"]["artifact_root"])
    metrics_path = artifact_root / "math" / "metrics.jsonl"
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    scaling = config["scaling"]
    recent_count = int(scaling["recent_epochs"])
    recent = rows[-recent_count:]
    if len(recent) < recent_count:
        raise RuntimeError("not enough completed epochs to assess the V2 scale gate")
    values = [
        float(row["validation"]["teacher_forced_sequence_accuracy"])
        for row in recent
    ]
    gain = values[-1] - values[0]
    threshold = float(scaling["minimum_recent_validation_gain"])
    latest_is_recent_best = values[-1] >= max(values) - 1e-12
    eligible = gain >= threshold and latest_is_recent_best
    decision = {
        "format": "cftn_text_v2_scale_decision_v1",
        "initial_train_examples": int(scaling["initial_train_examples"]),
        "maximum_train_examples": int(scaling["maximum_train_examples"]),
        "recent_epochs": [int(row["epoch"]) for row in recent],
        "recent_validation_sequence_accuracy": values,
        "recent_gain": gain,
        "minimum_required_gain": threshold,
        "latest_is_recent_best": latest_is_recent_best,
        "eligible_to_scale": eligible,
        "automatic_scaling_started": False,
        "reason": (
            "held-out teacher-forced sequence accuracy is still improving"
            if eligible
            else "the recent held-out curve does not justify one million examples yet"
        ),
    }
    atomic_json_dump(decision, artifact_root / "scale_decision.json")
    return decision


def assemble_v2_report(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project"]["artifact_root"])
    specialist = _load_json(root / "evaluation_math_v2" / "report.json")
    collaboration = _load_json(
        root / "evaluation_collaboration_v2" / "report.json"
    )
    scale = _load_json(root / "scale_decision.json")
    report = {
        "format": "cftn_text_v2_end_to_end_report_v1",
        "project": config["project"]["name"],
        "train_examples": int(config["data"]["train_examples"]),
        "frozen_gpt": config["gpt"]["model_name"],
        "gpt_weights_frozen": True,
        "bridge_architecture": "contextual_message_bridge_and_gated_cross_receivers",
        "specialist_gate": specialist.get("specialist_gate", {}),
        "collaboration_gate": collaboration.get("collaboration_gate", {}),
        "scale_decision": scale,
        "overall_pass": bool(specialist.get("specialist_gate", {}).get("pass"))
        and bool(collaboration.get("collaboration_gate", {}).get("pass")),
        "reports": {
            "specialist": str((root / "evaluation_math_v2" / "report.json").resolve()),
            "collaboration": str(
                (root / "evaluation_collaboration_v2" / "report.json").resolve()
            ),
            "scale": str((root / "scale_decision.json").resolve()),
        },
    }
    atomic_json_dump(report, root / "v2_final_report.json")
    return report
