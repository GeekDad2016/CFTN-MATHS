from __future__ import annotations

import json
from pathlib import Path

from tools.serve_math_master_dashboard import PAGE, collect_snapshot


def test_dashboard_is_single_column_and_refreshes_every_30_seconds() -> None:
    assert "grid-template-columns:1fr" in PAGE
    assert "setInterval(refresh,30000)" in PAGE
    assert "Phase epoch" in PAGE
    assert "Validation trend" in PAGE


def test_dashboard_collects_compact_phase_budget(tmp_path: Path) -> None:
    artifact = tmp_path / "run"
    artifact.mkdir()
    (artifact / "status.json").write_text(
        json.dumps({"state": "running", "epoch": 1, "global_step": 12}),
        encoding="utf-8",
    )
    metric = {
        "epoch": 1,
        "global_step": 12,
        "train_loss": 1.0,
        "validation": {"loss": 2.0, "teacher_forced_token_accuracy": 0.5},
        "curriculum_gate": {"generation_accuracy": 0.1, "valid_rate": 0.9},
        "curriculum_transition": {
            "phase": "one",
            "phase_epoch": 1,
            "consecutive_passes": 0,
        },
    }
    (artifact / "metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "phases": [{"name": "one", "criteria": ["a"]}],
                "splits": {"phase_00_active": {"records": 4}},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        """
project: {name: test, seed: 1, artifact_root: x, data_root: y}
data:
  format: cftn_canonical_math_curriculum_v1
  calibration_examples: 1
  train_examples: 1
  validation_examples: 1
  test_examples: 1
  heldout_language_examples: 1
  extrapolation_examples: 1
  compositional_examples: 1
  max_math_length: 32
  max_gpt_length: 32
  curriculum: {enabled: true, examples_per_epoch: 96, minimum_epochs_per_phase: 10, maximum_epochs_per_phase: 60, advance_after_consecutive_passes: 2, phases: [{name: x, through_epoch: 60, max_difficulty: 1}]}
gpt: {model_name: gpt2, local_files_only: true, receiver_layers: [1]}
math_tower: {layers: 1, hidden_size: 32, attention_heads: 1, feed_forward_size: 64, dropout: 0.0, max_sequence_length: 32, receiver_layers: [1], answer_min: -10, answer_max: 10}
bridge: {message_tokens: 1, message_width: 32, attention_heads: 1, dropout: 0.0, gate_hidden_size: 32, gate_init: -2.0, zero_init_output: true}
math_training: {batch_size: 1, eval_batch_size: 1, max_epochs: 60, minimum_epochs: 10, early_stop_patience: 60, learning_rate: 0.001, minimum_learning_rate: 0.00001, warmup_fraction: 0.0, weight_decay: 0.0, gradient_clip: 1.0, answer_head_weight: 0.0, precision: fp32, num_workers: 0, report_every_steps: 1}
bridge_training: {batch_size: 1, eval_batch_size: 1, max_epochs: 1, minimum_epochs: 1, early_stop_patience: 1, learning_rate: 0.001, minimum_learning_rate: 0.00001, warmup_fraction: 0.0, weight_decay: 0.0, gradient_clip: 1.0, math_loss_weight: 1.0, gpt_loss_weight: 1.0, answer_head_weight: 0.0, precision: fp32, num_workers: 0, report_every_steps: 1}
evaluation: {batch_size: 1, maximum_generation_examples: 1, max_math_new_tokens: 1, max_gpt_new_tokens: 1, bootstrap_samples: 1}
monitoring: {status_interval_minutes: 1, detailed_report_every_epochs: 1, keep_latest_checkpoints: 1}
gpt_calibration: {batch_size: 1, candidate_batch_size: 1, maximum_examples: 1, max_new_tokens: 1, few_shot_examples: 0, precision: fp32, plausible_candidates: 1, minimum_headroom_percentage_points: 0.0, maximum_acceptable_gpt_accuracy: 1.0}
""",
        encoding="utf-8",
    )
    snapshot = collect_snapshot(artifact, manifest, config)
    assert snapshot["latest"]["phase"] == "one"
    assert snapshot["contract"]["maximum_epochs_per_phase"] == 60
    assert snapshot["phases"][0]["state"] == "active"


def test_dashboard_accepts_live_interim_metrics_with_null_gate(tmp_path: Path) -> None:
    from tools.serve_math_master_dashboard import _compact_metric

    compact = _compact_metric(
        {
            "epoch": 2,
            "global_step": 13,
            "validation": None,
            "curriculum_gate": None,
            "curriculum_transition": None,
        }
    )
    assert compact["epoch"] == 2
    assert compact["generation_accuracy"] is None
    assert compact["phase"] is None
