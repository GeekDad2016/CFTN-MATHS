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
    selection = _load_json(artifact_root / "math_checkpoint_selection" / "report.json")
    specialist = _load_json(artifact_root / "evaluation_math_v2" / "report.json")
    scaling = config["scaling"]
    recent_count = int(scaling["recent_epochs"])
    ordered = sorted(selection["candidates"], key=lambda item: int(item["epoch"]))
    recent = ordered[-recent_count:]
    if len(recent) < 2:
        raise RuntimeError("not enough generated checkpoint panels to assess V2 scaling")
    values = [float(row["generation_accuracy"]) for row in recent]
    gain = values[-1] - values[0]
    threshold = float(
        scaling.get(
            "minimum_recent_generation_gain",
            scaling.get("minimum_recent_validation_gain", 0.002),
        )
    )
    latest_is_recent_best = values[-1] >= max(values) - 1e-12
    specialist_pass = specialist.get("specialist_gate", {}).get("pass") is True
    eligible = specialist_pass and gain >= threshold and latest_is_recent_best
    decision = {
        "format": "cftn_text_v2_scale_decision_v2",
        "initial_train_examples": int(scaling["initial_train_examples"]),
        "maximum_train_examples": int(scaling["maximum_train_examples"]),
        "recent_epochs": [int(row["epoch"]) for row in recent],
        "recent_validation_generation_accuracy": values,
        "recent_gain": gain,
        "minimum_required_gain": threshold,
        "latest_is_recent_best": latest_is_recent_best,
        "specialist_gate_pass": specialist_pass,
        "eligible_to_scale": eligible,
        "automatic_scaling_started": False,
        "reason": (
            "held-out greedy-generation accuracy is still improving and the specialist gate passed"
            if eligible
            else "generated held-out evidence does not justify one million examples yet"
        ),
    }
    atomic_json_dump(decision, artifact_root / "scale_decision.json")
    return decision


def assemble_v2_report(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["project"]["artifact_root"])
    mechanism = _load_json(root / "mechanism_prerequisites.json")
    selection = _load_json(root / "math_checkpoint_selection" / "report.json")
    specialist = _load_json(root / "evaluation_math_v2" / "report.json")
    conditional = _load_json(root / "bridge_conditional_contextual" / "summary.json")
    shared = _load_json(root / "evaluation_shared_no_harm_v2" / "report.json")
    collaboration = _load_json(
        root / "evaluation_collaboration_v2" / "report.json"
    )
    scale = _load_json(root / "scale_decision.json")
    conditional_gate = (
        conditional.get("final_metrics", {}).get("acceptance", {}).get("gates", {})
    )
    report = {
        "format": "cftn_text_v2_end_to_end_report_v2",
        "project": config["project"]["name"],
        "train_examples": int(config["data"]["train_examples"]),
        "frozen_gpt": config["gpt"]["model_name"],
        "gpt_weights_frozen": True,
        "bridge_architecture": "contextual_message_bridge_and_gated_cross_receivers",
        "mechanism_prerequisites": mechanism,
        "math_checkpoint_selection": selection["selected"],
        "specialist_gate": specialist.get("specialist_gate", {}),
        "conditional_training_gate": conditional_gate,
        "shared_no_harm_gate": shared.get("shared_no_harm_gate", {}),
        "collaboration_gate": collaboration.get("collaboration_gate", {}),
        "scale_decision": scale,
        "overall_pass": bool(mechanism.get("pass"))
        and bool(specialist.get("specialist_gate", {}).get("pass"))
        and bool(conditional_gate.get("pass"))
        and bool(shared.get("shared_no_harm_gate", {}).get("pass"))
        and bool(collaboration.get("collaboration_gate", {}).get("pass")),
        "claim_scope": {
            "broad_math_specialist_generalization": "gated by specialist_gate",
            "safe_conditional_communication": "gated by conditional_training_gate and shared_no_harm_gate",
            "controlled_private_view_collaboration": "gated by collaboration_gate",
            "autonomous_natural_prompt_wake": (
                "inherited only as prior V1.3 evidence; this V2 run does not re-prove it "
                "with the broad math specialist"
            ),
        },
        "reports": {
            "mechanism_prerequisites": str((root / "mechanism_prerequisites.json").resolve()),
            "math_checkpoint_selection": str(
                (root / "math_checkpoint_selection" / "report.json").resolve()
            ),
            "specialist": str((root / "evaluation_math_v2" / "report.json").resolve()),
            "conditional_training": str(
                (root / "bridge_conditional_contextual" / "summary.json").resolve()
            ),
            "shared_no_harm": str(
                (root / "evaluation_shared_no_harm_v2" / "report.json").resolve()
            ),
            "collaboration": str(
                (root / "evaluation_collaboration_v2" / "report.json").resolve()
            ),
            "scale": str((root / "scale_decision.json").resolve()),
        },
    }
    atomic_json_dump(report, root / "v2_final_report.json")
    return report
