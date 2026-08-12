from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint import atomic_json_dump, load_checkpoint
from .conditional_training import load_revision_config


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _arm_accuracy(report: dict[str, Any], split: str, arm: str) -> float:
    return float(report["splits"][split]["arm_accuracy"][arm])


def _condition_accuracy(
    report: dict[str, Any], split: str, condition: str, output: str
) -> float:
    return float(
        report["splits"][split]["metrics"][condition][output]["exact_accuracy"]
    )


GATE_EXPLANATIONS = {
    "training_mixed_necessity_gate": (
        "The selected training checkpoint passed the fixed mixed-necessity "
        "generation panel."
    ),
    "shared_specialist_no_harm": (
        "GPT-to-math stayed within the preregistered shared-view specialist "
        "regression limit."
    ),
    "complementary_synergy_gain": (
        "Full CFTN exceeded the strongest isolated arm on complementary tasks."
    ),
    "complementary_correct_vs_shuffled": (
        "Correct bridge messages outperformed shuffled-message controls."
    ),
    "complementary_gpt_to_math_gain": (
        "GPT-to-math made a positive causal contribution."
    ),
    "complementary_math_to_gpt_gain": (
        "Math-to-GPT made a positive causal contribution."
    ),
    "preserves_v1_1_familiar_and_compositional": (
        "The revision retained V1.1 familiar and compositional capability."
    ),
}


GATE_FAILURE_HYPOTHESES = {
    "training_mixed_necessity_gate": (
        "The checkpoint-selection objective or optimization balance may not "
        "have satisfied required-message utility and redundant-message safety "
        "at the same epoch."
    ),
    "shared_specialist_no_harm": (
        "The GPT-to-math residual may still perturb baseline-correct specialist "
        "states more than preservation distillation can constrain."
    ),
    "complementary_synergy_gain": (
        "The revised request message may not preserve enough of the original "
        "complementary cooperation to beat the strongest isolated arm."
    ),
    "complementary_correct_vs_shuffled": (
        "The learned message may not be sufficiently prompt-specific; the "
        "receiver could be relying on a generic residual pattern."
    ),
    "complementary_gpt_to_math_gain": (
        "The outgoing GPT request may not encode the operation and role mapping "
        "in a form the frozen specialist can use."
    ),
    "complementary_math_to_gpt_gain": (
        "The frozen return path may no longer receive a sufficiently accurate "
        "specialist state after the GPT-to-math revision."
    ),
    "preserves_v1_1_familiar_and_compositional": (
        "GPT-to-math updates may have overwritten a useful region of the "
        "original complementary bridge solution."
    ),
}


GATE_RECOMMENDATIONS = {
    "training_mixed_necessity_gate": (
        "Inspect the per-epoch generation arms and rebalance selection or loss "
        "weights before changing tower capacity."
    ),
    "shared_specialist_no_harm": (
        "Increase baseline-preservation/bridge-off training, lower GPT-to-math "
        "and gate learning rates, and retain the frozen specialist baseline."
    ),
    "complementary_synergy_gain": (
        "Restore from the V1.1 source bridge and run a smaller bridge-capacity "
        "test before another full revision."
    ),
    "complementary_correct_vs_shuffled": (
        "Add harder cross-example donors, paired counterfactuals, and a larger "
        "correct-versus-shuffled margin."
    ),
    "complementary_gpt_to_math_gain": (
        "Train explicit operation/role/span request probes and verify receiver "
        "alignment before enabling broader tasks."
    ),
    "complementary_math_to_gpt_gain": (
        "Audit specialist generation before the return bridge and restore the "
        "proven V1.1 math-to-GPT state if its input distribution drifted."
    ),
    "preserves_v1_1_familiar_and_compositional": (
        "Strengthen source-checkpoint distillation and reduce update scope or "
        "learning rate on the GPT-to-math receivers."
    ),
}


def _atomic_text_dump(value: str, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(destination)


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _build_findings(final_gates: dict[str, bool]) -> dict[str, list[str]]:
    passed = [
        GATE_EXPLANATIONS[name]
        for name, result in final_gates.items()
        if name != "pass" and result
    ]
    failed = [
        f"{GATE_EXPLANATIONS[name]} (failed)"
        for name, result in final_gates.items()
        if name != "pass" and not result
    ]
    hypotheses = [
        f"{name}: {GATE_FAILURE_HYPOTHESES[name]}"
        for name, result in final_gates.items()
        if name != "pass" and not result
    ]
    recommendations = [
        GATE_RECOMMENDATIONS[name]
        for name, result in final_gates.items()
        if name != "pass" and not result
    ]
    if final_gates["pass"]:
        recommendations.append(
            "Proceed to the preregistered V1.3 wake-gated recurrent "
            "multi-specialist experiment."
        )
    else:
        recommendations.append(
            "Run a targeted V1.2.x bridge repair before V1.3; do not hide the "
            "failed gate by scaling the towers or dataset first."
        )
    return {
        "passed": passed,
        "failed": failed,
        "diagnostic_hypotheses": hypotheses,
        "recommendations": recommendations,
    }


def render_v1_2_markdown(report: dict[str, Any]) -> str:
    final_gates = report["final_gates"]
    findings = report["findings"]
    result = "PASS" if final_gates["pass"] else "FAIL"
    lines = [
        "# CFTN-Text V1.2 experiment results",
        "",
        f"Status: sealed **{result}**.",
        "",
        "Preregistered design: `V1_2_BRIDGE_REVISION.md`",
        "",
        "## Executive conclusion",
        "",
        (
            "V1.2 passed every central conditional-communication gate. The "
            "result supports advancing to V1.3."
            if final_gates["pass"]
            else "V1.2 did not pass every central conditional-communication "
            "gate. The failed criteria remain binding and require a targeted "
            "V1.2.x repair before V1.3."
        ),
        "",
        "## Immutable provenance",
        "",
        f"- Generated UTC: `{report['generated_utc']}`",
        f"- Revision hash: `{report['revision_sha256']}`",
        f"- Best epoch: {report['training']['best_epoch']}",
        f"- Best checkpoint: `{report['training']['best_checkpoint']}`",
        f"- Best checkpoint SHA-256: `{report['training']['best_checkpoint_sha256']}`",
    ]
    wandb_url = report["training"].get("wandb_url")
    if wandb_url:
        lines.append(f"- W&B run: {wandb_url}")
    provenance = report.get("provenance", {})
    for label, key in (
        ("Base configuration hash", "base_config_sha256"),
        ("Dataset manifest hash", "manifest_sha256"),
        ("Math checkpoint SHA-256", "math_checkpoint_sha256"),
        ("Source bridge checkpoint SHA-256", "source_bridge_checkpoint_sha256"),
    ):
        if provenance.get(key):
            lines.append(f"- {label}: `{provenance[key]}`")

    lines.extend(["", "## What went well", ""])
    if findings["passed"]:
        lines.extend(f"- {item}" for item in findings["passed"])
    else:
        lines.append("- The sealed pipeline completed and produced all matched controls.")

    lines.extend(["", "## What did not pass", ""])
    if findings["failed"]:
        lines.extend(f"- {item}" for item in findings["failed"])
    else:
        lines.append("- No preregistered central gate failed.")

    lines.extend(
        [
            "",
            "## Shared-view no-harm results",
            "",
            "| Split | V1.2 full math | GPT-to-math disabled | Regression | V1.1 regression |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, item in report["shared_view"]["splits"].items():
        lines.append(
            f"| {split} | {_percent(item['v1_2_full_math_accuracy'])} | "
            f"{_percent(item['v1_2_gpt_to_math_disabled_math_accuracy'])} | "
            f"{_percent(item['v1_2_redundant_math_regression'])} | "
            f"{_percent(item['v1_1_redundant_math_regression'])} |"
        )

    lines.extend(
        [
            "",
            "## Complementary causal results",
            "",
            "| Split | V1.2 joint | V1.1 joint | Change | GPT-to-math gain | Correct-vs-shuffled |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, item in report["complementary_view"]["splits"].items():
        lines.append(
            f"| {split} | {_percent(item['v1_2_joint_accuracy'])} | "
            f"{_percent(item['v1_1_joint_accuracy'])} | "
            f"{_percent(item['joint_accuracy_change'])} | "
            f"{_percent(item['v1_2_gpt_to_math_gain'])} | "
            f"{_percent(item['v1_2_correct_vs_shuffled'])} |"
        )

    lines.extend(["", "## Acceptance gates", ""])
    for name, passed in final_gates.items():
        if name == "pass":
            continue
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")

    lines.extend(
        [
            "",
            "## Why: diagnostic hypotheses",
            "",
            "The statements below are hypotheses suggested by the ablations, not measured facts.",
            "",
        ]
    )
    if findings["diagnostic_hypotheses"]:
        lines.extend(
            f"- {item}" for item in findings["diagnostic_hypotheses"]
        )
    else:
        lines.append(
            "- No failure diagnosis is required; remaining limitations concern "
            "broader tower knowledge and conditional compute, which V1.2 did not test."
        )

    lines.extend(["", "## Recommended fixes or next experiment", ""])
    lines.extend(f"- {item}" for item in findings["recommendations"])
    lines.extend(
        [
            "",
            "## Scope limitation",
            "",
            report["interpretation"]["communication_revision"],
            "",
            report["interpretation"]["generalization"],
            "",
        ]
    )
    return "\n".join(lines)


def assemble_v1_2_report(revision: dict[str, Any]) -> dict[str, Any]:
    paths = revision["paths"]
    root = Path(paths["artifact_root"])
    v1_root = Path(paths["v1_1_artifact_root"])
    training_summary_path = root / "bridge_conditional_contextual" / "summary.json"
    shared_path = root / "evaluation_shared" / "report.json"
    synergy_path = root / "evaluation_complementary" / "report.json"
    baseline_shared_path = (
        v1_root / "evaluation_bidirectional_contextual_shared" / "report.json"
    )
    baseline_synergy_path = v1_root / "synergy_evaluation_contextual" / "report.json"
    prerequisites_path = root / "prerequisites.json"
    wandb_path = root / "bridge_conditional_contextual" / "wandb_run.json"
    for path in (
        training_summary_path,
        shared_path,
        synergy_path,
        baseline_shared_path,
        baseline_synergy_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"V1.2 report input is missing: {path}")

    training = _load_json(training_summary_path)
    shared = _load_json(shared_path)
    synergy = _load_json(synergy_path)
    baseline_shared = _load_json(baseline_shared_path)
    baseline_synergy = _load_json(baseline_synergy_path)
    provenance = _load_json(prerequisites_path) if prerequisites_path.is_file() else {}
    wandb_run = _load_json(wandb_path) if wandb_path.is_file() else {}
    checkpoint = load_checkpoint(training["best_checkpoint"])
    best_training_metrics = checkpoint["extra"]["metrics"]
    threshold = revision["acceptance"]

    shared_splits: dict[str, Any] = {}
    for split in sorted(shared["splits"]):
        full_math = _condition_accuracy(shared, split, "correct", "math")
        disabled_math = _condition_accuracy(
            shared, split, "gpt_to_math_disabled", "math"
        )
        baseline_full_math = _condition_accuracy(
            baseline_shared, split, "correct", "math"
        )
        baseline_disabled_math = _condition_accuracy(
            baseline_shared, split, "gpt_to_math_disabled", "math"
        )
        shared_splits[split] = {
            "v1_2_full_math_accuracy": full_math,
            "v1_2_gpt_to_math_disabled_math_accuracy": disabled_math,
            "v1_2_redundant_math_regression": disabled_math - full_math,
            "v1_1_redundant_math_regression": baseline_disabled_math
            - baseline_full_math,
            "regression_improvement": (baseline_disabled_math - baseline_full_math)
            - (disabled_math - full_math),
        }

    complementary_splits: dict[str, Any] = {}
    for split in sorted(synergy["splits"]):
        complementary_splits[split] = {
            "v1_2_joint_accuracy": _arm_accuracy(
                synergy, split, "joint_cftn"
            ),
            "v1_1_joint_accuracy": _arm_accuracy(
                baseline_synergy, split, "joint_cftn"
            ),
            "joint_accuracy_change": _arm_accuracy(
                synergy, split, "joint_cftn"
            )
            - _arm_accuracy(baseline_synergy, split, "joint_cftn"),
            "v1_2_gpt_to_math_gain": float(
                synergy["splits"][split]["gpt_to_math_direct_contribution"][
                    "mean_difference"
                ]
            ),
            "v1_2_correct_vs_shuffled": float(
                synergy["splits"][split]["correct_vs_both_shuffled"][
                    "mean_difference"
                ]
            ),
        }

    maximum_shared_regression = max(
        item["v1_2_redundant_math_regression"]
        for item in shared_splits.values()
    )
    retention_splits = [
        split
        for split in ("test", "compositional")
        if split in complementary_splits
    ]
    minimum_retention_change = min(
        complementary_splits[split]["joint_accuracy_change"]
        for split in retention_splits
    )
    causal = synergy["causal_collaboration_gate"]
    final_gates = {
        "training_mixed_necessity_gate": bool(
            best_training_metrics["acceptance"]["gates"]["pass"]
        ),
        "shared_specialist_no_harm": maximum_shared_regression
        <= float(threshold["maximum_redundant_math_regression"]),
        "complementary_synergy_gain": bool(causal["synergy_gain"]),
        "complementary_correct_vs_shuffled": bool(causal["correct_vs_shuffled"]),
        "complementary_gpt_to_math_gain": bool(causal["gpt_to_math_causal_gain"]),
        "complementary_math_to_gpt_gain": bool(causal["math_to_gpt_causal_gain"]),
        "preserves_v1_1_familiar_and_compositional": minimum_retention_change
        >= -0.02,
    }
    final_gates["pass"] = all(final_gates.values())
    findings = _build_findings(final_gates)
    artifact_document = root / "V1_2_EXPERIMENT_RESULTS.md"
    repository_document = (
        Path(revision["_meta"]["repository_root"]) / "V1_2_EXPERIMENT_RESULTS.md"
    )
    report = {
        "format": "cftn_text_v1_2_revision_report_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "revision_sha256": revision["_meta"]["sha256"],
        "provenance": provenance,
        "training": {
            "summary": str(training_summary_path.resolve()),
            "best_checkpoint": training["best_checkpoint"],
            "best_checkpoint_sha256": training["best_checkpoint_sha256"],
            "best_epoch": int(checkpoint["epoch"]),
            "best_generation_acceptance": best_training_metrics["acceptance"],
            "wandb_url": wandb_run.get("url"),
        },
        "shared_view": {
            "report": str(shared_path.resolve()),
            "maximum_redundant_math_regression": maximum_shared_regression,
            "splits": shared_splits,
        },
        "complementary_view": {
            "report": str(synergy_path.resolve()),
            "causal_collaboration_gate": causal,
            "splits": complementary_splits,
        },
        "final_gates": final_gates,
        "findings": findings,
        "interpretation": {
            "communication_revision": (
                "Tests when GPT-to-math information should cross; it does not "
                "test wake-gate conditional compute or multi-specialist routing."
            ),
            "generalization": (
                "V1.2 reuses frozen V1.1 towers, so broad specialist capability "
                "remains a separate V2 question."
            ),
        },
        "result_documents": {
            "artifact": str(artifact_document.resolve()),
            "repository": str(repository_document.resolve()),
        },
    }
    atomic_json_dump(report, root / "v1_2_final_report.json")
    markdown = render_v1_2_markdown(report)
    _atomic_text_dump(markdown, artifact_document)
    _atomic_text_dump(markdown, repository_document)
    return report


def assemble_v1_2_report_from_path(path: str | Path) -> dict[str, Any]:
    return assemble_v1_2_report(load_revision_config(path))
