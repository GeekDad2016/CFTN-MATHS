from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import atomic_json_dump
from .data_generator import file_sha256
from .v1_3_config import audit_v1_2_pass
from .v1_3_data import SPECIALISTS


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V1.3 report input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V1.3 report input is not an object: {path}")
    return value


def _weighted_accuracy(
    split: dict[str, Any],
    arm: str,
    classes: Iterable[str],
    *,
    competence_supported: bool = False,
) -> float:
    selected = split["arms"][arm]["by_task_class"]
    numerator = denominator = 0.0
    for name in classes:
        item = selected.get(name)
        if item:
            metrics = item["competence_supported"] if competence_supported else item
            numerator += float(metrics["exact_accuracy"]) * int(metrics["examples"])
            denominator += int(metrics["examples"])
    return numerator / max(1.0, denominator)


def _competence_coverage(split: dict[str, Any], classes: Iterable[str]) -> float:
    by_class = split["competence_contract"]["by_task_class"]
    supported = examples = 0
    for name in classes:
        item = by_class.get(name)
        if item:
            supported += int(item["supported_examples"])
            examples += int(item["examples"])
    return supported / max(1, examples)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def render_v1_3_markdown(report: dict[str, Any]) -> str:
    result = "PASS" if report["final_gates"]["pass"] else "FAIL"
    lines = [
        "# CFTN-Text V1.3 experiment results",
        "",
        f"Status: sealed **{result}**.",
        "",
        "Preregistered design: `V1_3_EXPERIMENT_PLAN.md`",
        "",
        "## Executive conclusion",
        "",
        report["interpretation"]["conclusion"],
        "",
        "## Immutable provenance",
        "",
        f"- Generated UTC: `{report['generated_utc']}`",
        f"- V1.3 revision SHA-256: `{report['revision_sha256']}`",
        f"- Dataset manifest SHA-256: `{report['manifest_sha256']}`",
        f"- Final checkpoint: `{report['checkpoint']}`",
        f"- Final checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        f"- V1.2 sealed report SHA-256: `{report['prerequisite']['v1_2_report_sha256']}`",
        "",
        "## Central measurements",
        "",
        f"- GPT pure-language accuracy: {_percent(report['measurements']['gpt_pure_language_accuracy'])}",
        f"- CFTN pure-language accuracy: {_percent(report['measurements']['cftn_pure_language_accuracy'])}",
        f"- Wake precision / recall / exact set: {_percent(report['measurements']['wake_precision'])} / "
        f"{_percent(report['measurements']['wake_recall'])} / "
        f"{_percent(report['measurements']['exact_required_set_accuracy'])}",
        f"- Joint synergy over strongest individual: {_percent(report['measurements']['joint_synergy_gain'])} "
        f"(95% CI {_percent(report['measurements']['joint_synergy_ci95_low'])} to "
        f"{_percent(report['measurements']['joint_synergy_ci95_high'])})",
        f"- Primary specialist-competence coverage: "
        f"{_percent(report['measurements']['primary_competence_coverage'])}",
        f"- Hard-wake compute reduction versus dense: {_percent(report['measurements']['compute_reduction'])}",
        f"- Sequential accuracy: {_percent(report['measurements']['sequential_accuracy'])}",
        f"- Multi-round gain on sequential tasks: {_percent(report['measurements']['multiround_gain'])}",
        f"- First-round donor-swap loss on sequential tasks: "
        f"{_percent(report['measurements']['first_round_return_swap_loss'])}",
        f"- Learned-wake throughput / peak CUDA memory: "
        f"{report['measurements']['learned_wake_examples_per_second']:.2f} examples/s / "
        f"{report['measurements']['learned_wake_peak_memory_bytes'] / (1024 ** 2):.1f} MiB",
        "",
        "## Acceptance gates",
        "",
    ]
    for name, passed in report["final_gates"].items():
        if name != "pass":
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(["", "## What went well", ""])
    lines.extend(f"- {value}" for value in report["findings"]["passed"])
    if not report["findings"]["passed"]:
        lines.append("- The ordered pipeline completed and retained all matched controls.")
    lines.extend(["", "## What did not pass", ""])
    lines.extend(f"- {value}" for value in report["findings"]["failed"])
    if not report["findings"]["failed"]:
        lines.append("- No preregistered central gate failed.")
    lines.extend(
        [
            "",
            "## Diagnostic hypotheses",
            "",
            "These are hypotheses suggested by the ablations, not measured facts.",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in report["findings"]["hypotheses"])
    if not report["findings"]["hypotheses"]:
        lines.append("- No failed central mechanism requires a repair hypothesis.")
    lines.extend(["", "## Recommended next action", ""])
    lines.extend(f"- {value}" for value in report["findings"]["recommendations"])
    lines.extend(
        [
            "",
            "## Scope",
            "",
            report["interpretation"]["scope"],
            "",
        ]
    )
    return "\n".join(lines)


def assemble_v1_3_report(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["artifact_root"])
    prerequisite = audit_v1_2_pass(config)
    calibration_path = root / "gpt_language_calibration" / "report.json"
    native_path = root / "native_specialist_evaluation" / "report.json"
    evaluation_path = root / "sealed_evaluation" / "report.json"
    calibration = _load(calibration_path)
    native = _load(native_path)
    evaluation = _load(evaluation_path)
    if calibration.get("pass") is not True:
        raise RuntimeError("V1.3 report refuses a failed GPT calibration precondition")
    if native.get("gates", {}).get("pass") is not True:
        raise RuntimeError("V1.3 report refuses a failed native-specialist precondition")
    phases: dict[str, Any] = {}
    for phase in config["integration_training"]["phases"]:
        name = phase["name"]
        summary_path = root / name / "summary.json"
        summary = _load(summary_path)
        if summary.get("state") != "completed":
            raise RuntimeError(f"V1.3 phase is not complete: {name}")
        phases[name] = {
            "summary": str(summary_path.resolve()),
            "summary_sha256": file_sha256(summary_path),
            "best_checkpoint": summary["best_checkpoint"],
            "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
            "final_metrics": summary["final_metrics"],
        }
    contract = evaluation.get("evaluation_contract", {})
    primary_split = str(config["evaluation"]["primary_split"])
    if contract.get("primary_split") != primary_split:
        raise RuntimeError("V1.3 evaluation primary split differs from preregistration")
    if contract.get("diagnostic_splits_are_non_gating") is not True:
        raise RuntimeError("V1.3 diagnostic splits were not sealed as non-gating")
    split = evaluation["splits"][primary_split]
    thresholds = config["acceptance"]
    learned = split["arms"]["learned_wake_cftn"]
    dense = split["arms"]["dense_cftn"]
    wake = learned["wake"]
    pure_gpt = _weighted_accuracy(split, "gpt_alone", ("pure_language",))
    pure_cftn = _weighted_accuracy(split, "learned_wake_cftn", ("pure_language",))
    pure_regression = pure_gpt - pure_cftn
    joint_classes = ("language_dependent_math", "multi_parallel", "multi_sequential")
    math_classes = (
        "explicit_math",
        "language_dependent_math",
        "multi_parallel",
        "multi_sequential",
    )
    string_classes = ("exact_string", "multi_parallel", "multi_sequential")
    required_classes = (
        "explicit_math",
        "exact_string",
        "language_dependent_math",
        "multi_parallel",
        "multi_sequential",
    )
    all_classes = ("pure_language", *required_classes)
    joint_learned_unconditioned = _weighted_accuracy(
        split, "learned_wake_cftn", joint_classes
    )
    joint_learned = _weighted_accuracy(
        split,
        "learned_wake_cftn",
        joint_classes,
        competence_supported=True,
    )
    joint_fixed = _weighted_accuracy(
        split, "fixed_open", joint_classes, competence_supported=True
    )
    joint_serial = _weighted_accuracy(
        split, "serial_pipeline", joint_classes, competence_supported=True
    )
    math_required_accuracy = _weighted_accuracy(
        split, "learned_wake_cftn", math_classes, competence_supported=True
    )
    math_ablation = math_required_accuracy - _weighted_accuracy(
        split, "math_disabled", math_classes, competence_supported=True
    )
    string_required_accuracy = _weighted_accuracy(
        split, "learned_wake_cftn", string_classes, competence_supported=True
    )
    string_ablation = string_required_accuracy - _weighted_accuracy(
        split, "string_disabled", string_classes, competence_supported=True
    )
    required_accuracy = _weighted_accuracy(
        split, "learned_wake_cftn", required_classes, competence_supported=True
    )
    shuffled_gap = required_accuracy - _weighted_accuracy(
        split, "messages_shuffled", required_classes, competence_supported=True
    )
    irrelevant_effects = (
        abs(
            _weighted_accuracy(
                split,
                "learned_wake_cftn",
                ("explicit_math",),
                competence_supported=True,
            )
            - _weighted_accuracy(
                split,
                "string_disabled",
                ("explicit_math",),
                competence_supported=True,
            )
        ),
        abs(
            _weighted_accuracy(
                split,
                "learned_wake_cftn",
                ("exact_string",),
                competence_supported=True,
            )
            - _weighted_accuracy(
                split,
                "math_disabled",
                ("exact_string",),
                competence_supported=True,
            )
        ),
    )
    maximum_irrelevant_effect = max(irrelevant_effects)
    dense_active = float(dense["wake"]["mean_active_specialist_executions"])
    learned_active = float(wake["mean_active_specialist_executions"])
    compute_reduction = 1.0 - learned_active / max(1e-12, dense_active)
    hard_dense_regression = _weighted_accuracy(
        split, "dense_cftn", all_classes, competence_supported=True
    ) - _weighted_accuracy(
        split, "learned_wake_cftn", all_classes, competence_supported=True
    )
    sequential = _weighted_accuracy(
        split,
        "learned_wake_cftn",
        ("multi_sequential",),
        competence_supported=True,
    )
    one_round = _weighted_accuracy(
        split, "one_round", ("multi_sequential",), competence_supported=True
    )
    multiround_gain = sequential - one_round
    first_round_swap_loss = sequential - _weighted_accuracy(
        split,
        "first_round_return_swapped",
        ("multi_sequential",),
        competence_supported=True,
    )
    specialist_parameters = evaluation["compute_contract"]["specialist_parameter_counts"]
    mean_active_by_name = wake["mean_active_executions_by_specialist"]
    active_parameter_executions = sum(
        float(mean_active_by_name[name]) * int(specialist_parameters[name])
        for name in SPECIALISTS
    )
    synergy = split["central_supported_joint_learned_vs_strongest_individual"]
    primary_competence_coverage = _competence_coverage(split, required_classes)
    task_matched_native = bool(native["gates"].get("task_matched_math")) and bool(
        native["gates"].get("task_matched_string")
    )
    gates = {
        "v1_2_prerequisite": True,
        "gpt_language_precondition": bool(calibration["pass"]),
        "native_specialists_familiar": bool(native["gates"].get("math_familiar"))
        and bool(native["gates"].get("string_familiar")),
        "native_specialists_task_matched": task_matched_native,
        "primary_competence_coverage": primary_competence_coverage
        >= float(thresholds["minimum_primary_competence_coverage"]),
        "pure_language_no_harm": pure_regression
        <= float(thresholds["maximum_pure_language_regression"]),
        "pure_language_false_wake": float(wake["pure_language_false_wake_rate"])
        <= float(thresholds["maximum_pure_language_false_wake_rate"]),
        "wake_recall": float(wake["recall"]) >= float(thresholds["minimum_wake_recall"]),
        "wake_precision": float(wake["precision"])
        >= float(thresholds["minimum_wake_precision"]),
        "exact_required_set": float(wake["exact_required_set_accuracy"])
        >= float(thresholds["minimum_exact_required_set_accuracy"]),
        "joint_positive_synergy": float(synergy["mean_difference"])
        >= float(thresholds["minimum_joint_synergy_gain"])
        and float(synergy["ci95_low"]) > 0.0,
        "math_bridge_causal": math_ablation
        >= float(thresholds["minimum_required_bridge_ablation_loss"]),
        "string_bridge_causal": string_ablation
        >= float(thresholds["minimum_required_bridge_ablation_loss"]),
        "messages_content_specific": shuffled_gap
        >= float(thresholds["minimum_required_bridge_ablation_loss"]),
        "irrelevant_bridge_no_harm": maximum_irrelevant_effect
        <= float(thresholds["maximum_irrelevant_bridge_effect"]),
        "hard_matches_dense": hard_dense_regression
        <= float(thresholds["maximum_hard_vs_dense_regression"]),
        "compute_reduction": compute_reduction
        >= float(thresholds["minimum_compute_reduction"]),
        "sequential_accuracy": sequential
        >= float(thresholds["minimum_sequential_accuracy"]),
        "multiround_causality": multiround_gain
        >= float(thresholds["minimum_multiround_gain"]),
        "beats_fixed_open": joint_learned > joint_fixed,
        "beats_serial_pipeline": joint_learned > joint_serial,
    }
    gates["pass"] = all(gates.values())
    explanations = {
        "v1_2_prerequisite": "The sealed V1.2 conditional bridge passed.",
        "gpt_language_precondition": "Frozen GPT passed the no-specialist calibration.",
        "native_specialists_familiar": "Both native specialists passed familiar exact generation.",
        "native_specialists_task_matched": (
            "Each specialist passed every oracle-native operation used by the primary joint benchmark."
        ),
        "primary_competence_coverage": (
            "Primary bridge claims retained at least the preregistered fraction of specialist-capable examples."
        ),
        "pure_language_no_harm": "Specialist integration preserved GPT-only language tasks.",
        "pure_language_false_wake": "Specialists usually slept on pure-language prompts.",
        "wake_recall": "Required specialists were woken reliably.",
        "wake_precision": "Wake decisions rarely activated irrelevant specialists.",
        "exact_required_set": "The complete per-round required set was predicted reliably.",
        "joint_positive_synergy": "CFTN beat the strongest individual tower with a positive paired interval.",
        "math_bridge_causal": "Removing math caused the preregistered loss.",
        "string_bridge_causal": "Removing string caused the preregistered loss.",
        "messages_content_specific": "Shuffled cross-example messages removed capability.",
        "irrelevant_bridge_no_harm": "Irrelevant specialist ablation had little effect.",
        "hard_matches_dense": "Hard conditional execution retained dense-model accuracy.",
        "compute_reduction": "Learned wakes reduced specialist execution sufficiently.",
        "sequential_accuracy": "Sequential two-specialist tasks reached the target.",
        "multiround_causality": "Restricting the system to one round caused the required loss.",
        "beats_fixed_open": "Contextual wake/message gates beat fixed-open communication.",
        "beats_serial_pipeline": "The state-connected model beat the simple serial baseline.",
    }
    passed = [text for name, text in explanations.items() if gates[name]]
    failed = [f"{text} (failed)" for name, text in explanations.items() if not gates[name]]
    hypotheses = [
        f"{name}: inspect the matched disabled/shuffled rows before changing tower capacity."
        for name in explanations
        if not gates[name]
    ]
    recommendations = (
        [
            "Advance to V1.4 only as an incremental third-specialist extension; retain all V1.3 regressions and controls."
        ]
        if gates["pass"]
        else [
            "Run a targeted V1.3.x repair for the failed gate(s); do not add more specialists until conditional communication and compute pass."
        ]
    )
    final_phase = config["integration_training"]["phases"][-1]["name"]
    report = {
        "format": "cftn_text_v1_3_revision_report_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "revision_sha256": config["_meta"]["sha256"],
        "manifest_sha256": evaluation["manifest_sha256"],
        "checkpoint": phases[final_phase]["best_checkpoint"],
        "checkpoint_sha256": phases[final_phase]["best_checkpoint_sha256"],
        "prerequisite": prerequisite,
        "calibration": calibration,
        "native_specialists": native,
        "phases": phases,
        "evaluation": str(evaluation_path.resolve()),
        "measurements": {
            "gpt_pure_language_accuracy": pure_gpt,
            "cftn_pure_language_accuracy": pure_cftn,
            "pure_language_regression": pure_regression,
            "pure_language_false_wake_rate": wake["pure_language_false_wake_rate"],
            "wake_precision": wake["precision"],
            "wake_recall": wake["recall"],
            "exact_required_set_accuracy": wake["exact_required_set_accuracy"],
            "joint_accuracy": joint_learned,
            "joint_accuracy_unconditioned": joint_learned_unconditioned,
            "primary_competence_coverage": primary_competence_coverage,
            "joint_synergy_gain": synergy["mean_difference"],
            "joint_synergy_ci95_low": synergy["ci95_low"],
            "joint_synergy_ci95_high": synergy["ci95_high"],
            "math_ablation_loss": math_ablation,
            "string_ablation_loss": string_ablation,
            "shuffled_message_loss": shuffled_gap,
            "maximum_irrelevant_bridge_effect": maximum_irrelevant_effect,
            "hard_vs_dense_regression": hard_dense_regression,
            "compute_reduction": compute_reduction,
            "sequential_accuracy": sequential,
            "multiround_gain": multiround_gain,
            "first_round_return_swap_loss": first_round_swap_loss,
            "fixed_open_joint_accuracy": joint_fixed,
            "serial_pipeline_joint_accuracy": joint_serial,
            "mean_active_parameter_executions": active_parameter_executions,
            "learned_wake_elapsed_seconds": learned["elapsed_seconds"],
            "learned_wake_examples_per_second": learned["examples_per_second"],
            "learned_wake_peak_memory_bytes": learned["peak_memory_bytes"],
        },
        "final_gates": gates,
        "findings": {
            "passed": passed,
            "failed": failed,
            "hypotheses": hypotheses,
            "recommendations": recommendations,
        },
        "interpretation": {
            "conclusion": (
                "V1.3 passed the preregistered multi-specialist communication, recurrence, and conditional-compute gates."
                if gates["pass"]
                else "V1.3 completed but did not pass every preregistered multi-specialist gate."
            ),
            "scope": (
                "This result concerns two controlled specialists and a frozen GPT-2 workspace. "
                "Central bridge claims are conditioned on independently verified task-matched "
                "specialist competence with at least 95% primary coverage. Held-out, extrapolation, "
                "counterfactual, and unseen-composition splits remain non-gating diagnostics and "
                "cannot be used to blame a bridge for knowledge absent from a specialist. This does "
                "not establish broad mathematical, linguistic, scientific, or visual intelligence."
            ),
        },
        "diagnostic_splits": {
            name: {
                "role": evaluation["splits"][name]["role"],
                "competence_coverage": evaluation["splits"][name][
                    "competence_contract"
                ]["required_coverage"],
                "learned_accuracy": evaluation["splits"][name]["arms"][
                    "learned_wake_cftn"
                ]["exact_accuracy"],
            }
            for name in config["evaluation"]["diagnostic_splits"]
        },
    }
    artifact_document = root / "V1_3_EXPERIMENT_RESULTS.md"
    repository_document = Path(config["_meta"]["repository_root"]) / "V1_3_EXPERIMENT_RESULTS.md"
    report["result_documents"] = {
        "artifact": str(artifact_document.resolve()),
        "repository": str(repository_document.resolve()),
    }
    atomic_json_dump(report, root / "v1_3_final_report.json")
    markdown = render_v1_3_markdown(report)
    _atomic_text(markdown, artifact_document)
    _atomic_text(markdown, repository_document)
    return report
