from __future__ import annotations

from cftn_text.conditional_reporting import (
    _build_findings,
    render_v1_2_markdown,
)


def _report(*, passed: bool) -> dict:
    gates = {
        "training_mixed_necessity_gate": passed,
        "shared_specialist_no_harm": passed,
        "complementary_synergy_gain": passed,
        "complementary_correct_vs_shuffled": passed,
        "complementary_gpt_to_math_gain": passed,
        "complementary_math_to_gpt_gain": passed,
        "preserves_v1_1_familiar_and_compositional": passed,
        "pass": passed,
    }
    split_shared = {
        "v1_2_full_math_accuracy": 0.98,
        "v1_2_gpt_to_math_disabled_math_accuracy": 0.99,
        "v1_2_redundant_math_regression": 0.01,
        "v1_1_redundant_math_regression": 0.20,
    }
    split_complementary = {
        "v1_2_joint_accuracy": 0.90,
        "v1_1_joint_accuracy": 0.80,
        "joint_accuracy_change": 0.10,
        "v1_2_gpt_to_math_gain": 0.50,
        "v1_2_correct_vs_shuffled": 0.45,
    }
    return {
        "generated_utc": "2026-08-10T00:00:00+00:00",
        "revision_sha256": "revision",
        "provenance": {
            "base_config_sha256": "config",
            "manifest_sha256": "manifest",
        },
        "training": {
            "best_epoch": 8,
            "best_checkpoint": "checkpoint.pth",
            "best_checkpoint_sha256": "checkpoint-hash",
            "wandb_url": "https://wandb.example/run",
        },
        "shared_view": {"splits": {"test": split_shared}},
        "complementary_view": {"splits": {"test": split_complementary}},
        "final_gates": gates,
        "findings": _build_findings(gates),
        "interpretation": {
            "communication_revision": "Communication scope.",
            "generalization": "Generalization scope.",
        },
    }


def test_v1_2_markdown_records_pass_and_next_experiment():
    markdown = render_v1_2_markdown(_report(passed=True))
    assert "sealed **PASS**" in markdown
    assert "95.00%" not in markdown
    assert "98.00%" in markdown
    assert "Proceed to the preregistered V1.3" in markdown
    assert "W&B run" in markdown


def test_v1_2_markdown_records_failure_hypotheses_and_repair():
    markdown = render_v1_2_markdown(_report(passed=False))
    assert "sealed **FAIL**" in markdown
    assert "hypotheses suggested by the ablations" in markdown
    assert "targeted V1.2.x bridge repair" in markdown
    assert "FAIL — `shared_specialist_no_harm`" in markdown
