import copy
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from cftn_text.computation_supervision import computation_loss
from cftn_text.dataset import EquationDataset
from cftn_text.full_math_data import (FORMAT, FullMathCollator, audit_full_data,
    check_row, full_parse, full_question, full_school_record, generated_procedure,
    repair_parent, school_rows)
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.training import _filter_v2_records_for_phase, _phase_generation_acceptance, math_epoch_dataset
from cftn_text.v2_data import iter_local_records, make_v2_record
from cftn_text.v2_school_data import FAMILIES
from cftn_text.verified_math_data import fingerprint
from tools.train_v2_full_supervision import checked_settings
from tools.run_v2_math_curriculum import checked_settings as checked_competency_settings


@pytest.mark.parametrize("family,values", [(f, (3, -2, 10) if f == "linear_equation" else (12, -3)) for f in FAMILIES])
def test_full_wording_is_bound_from_public_question(family, values):
    for style in range(9):
        text = full_question(family, values, style)
        assert full_parse(text) == (family, values)
        check_row(full_school_record(text))
    if family != "linear_equation":
        texts = [full_question(f, (12, -3), 4) for f in FAMILIES[:-1]]
        assert len(set(texts)) == 4


def test_all_six_full_generated_families_are_exact_and_public():
    records = list(iter_local_records(count=600, split="train", seed=419))
    assert len({r["family"] for r in records}) == 6
    for original in records:
        frozen = copy.deepcopy(original)
        row, flag = repair_parent(original)
        assert flag is None and original == frozen
        check_row(row)
        assert row["normalized_answer"] == original["normalized_answer"]
        assert row["target_trace"].endswith(original["target_answer"])
        # Hidden metadata is never a solver input.
        spoofed = dict(original, metadata={}, math_problem="incorrect private input")
        assert generated_procedure(spoofed) == generated_procedure(original)
        bad = dict(original, normalized_answer="9999999999")
        with pytest.raises(ValueError, match="answer"):
            repair_parent(bad)


def test_mathqa_quarantined_and_published_answers_not_claimed_verified():
    qa = make_v2_record(split="train", source="mathqa", family="mathqa", difficulty=3,
        problem="There are 2 and 3 items.", answer="7", native_program="add(n0,n1)",
        target_trace="<program>add(n0,n1)</program>")
    fixed, flag = repair_parent(qa)
    assert fixed is None and not flag["training_eligible"]
    assert flag["status"] == "quarantine_program_answer_mismatch"
    dm = make_v2_record(split="train", source="deepmind_mathematics", family="polynomial", difficulty=2,
                        problem="Expand x*(x+1).", answer="x^2+x")
    fixed, flag = repair_parent(dm)
    assert flag is None and fixed["target_trace"] == dm["target_trace"]
    assert fixed["verification"] == "published_target_not_independently_certified"


def test_curriculum_gates_school_range_but_keeps_broad_replay():
    rows = [{"source": "verified_school_full", "difficulty": d} for d in (1, 2, 3)]
    rows += [{"source": "deepmind_mathematics", "difficulty": 3}]
    assert _filter_v2_records_for_phase(rows, {"max_difficulty": 3, "max_school_difficulty": 1}) == [rows[0], rows[3]]


def test_family_balance_does_not_let_large_linear_pool_dominate():
    rows = [{"record_id": str(i), "source": "a", "family": "large" if i else "small", "difficulty": 1} for i in range(100)]
    phase = {"name": "test", "through_epoch": 1, "max_difficulty": 3,
             "source_quotas": {"a": 100}, "balance_families_within_source": True}
    config = {"data": {"format": "cftn_text_broad_math_v2", "curriculum": {
        "enabled": True, "phases": [phase], "examples_per_epoch": 100}}}
    sampled, meta = math_epoch_dataset(EquationDataset(rows), config, epoch=1, seed=42)
    assert Counter(r["family"] for r in sampled.records) == {"large": 50, "small": 50}
    assert meta["source_sampling"]["a"]["replacement_examples"] == 49


def test_full_loss_roles_and_mutation_protection():
    rows = [full_school_record("What is 12 times -3?")]
    rows += [repair_parent(r)[0] for r in iter_local_records(count=6, split="train", seed=22)]
    collator = FullMathCollator(ByteMathTokenizer(), 4096)
    batch = collator(rows)
    assert torch.equal(batch["math_roles"].ne(-100), batch["math_labels"].ne(-100))
    logits = torch.randn(*batch["math_labels"].shape, 260, requires_grad=True)
    loss = computation_loss(logits, batch["math_labels"], batch["math_roles"], weights=(.25, .5, .25))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    rows[0]["target_trace"] += "corruption"
    with pytest.raises(ValueError, match="changed"):
        collator(rows)


def test_full_settings_fail_closed(tmp_path):
    path = Path(__file__).parents[1] / "config/v2_full_supervision.json"
    value = checked_settings(path)
    assert value["math_training"]["max_epochs"] == 100
    assert not value["remaining_pipeline_enabled"]
    value["phases"][0]["minimum_generation_accuracy"] = .9
    bad = tmp_path / "weakened.json"
    bad.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="weakened"):
        checked_settings(bad)


def test_competency_settings_stage_verified_signal_and_fail_closed(tmp_path):
    path = Path(__file__).parents[1] / "config/v2_full_supervision_v2.json"
    value = checked_competency_settings(path)
    assert value["curriculum"]["transition_policy"] == "competency_gated_v1"
    assert [phase["maximum_epochs"] for phase in value["phases"]] == [8, 10, 12, 18, 22, 30]
    assert all(phase["advance_after_consecutive_passes"] >= 2 for phase in value["phases"])
    assert all(sum(group["examples"] for group in phase["quota_groups"]) == 400000 for phase in value["phases"])
    value["phases"][0]["quota_groups"][0]["examples"] -= 1
    bad = tmp_path / "bad_competency.json"
    bad.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="quota groups"):
        checked_competency_settings(bad)


def test_competency_v3_uses_entrance_replay_and_lower_learning_rate():
    path = Path(__file__).parents[1] / "config/v2_full_supervision_v3.json"
    value = checked_competency_settings(path)
    assert value["format"] == "cftn_full_math_training_v3"
    assert value["curriculum"] == {
        "enabled": True,
        "examples_per_epoch": 100000,
        "sampling": "auto",
        "transition_policy": "competency_gated_v2",
    }
    assert value["math_training"]["max_epochs"] == 42
    assert value["math_training"]["learning_rate"] == 2e-5
    assert value["zero_update_entrance"] == {
        "enabled": True,
        "maximum_skipped_phases": 5,
    }
    assert value["preservation_distillation"]["baseline_correct_only"]
    assert [phase["maximum_epochs"] for phase in value["phases"]] == [
        3,
        4,
        5,
        8,
        10,
        12,
    ]
    assert all(
        sum(group["examples"] for group in phase["quota_groups"]) == 100000
        for phase in value["phases"]
    )


def test_competency_v4_is_random_scratch_and_stages_future_sources():
    path = Path(__file__).parents[1] / "config/v2_math_scratch_v4.json"
    value = checked_competency_settings(path)
    assert value["format"] == "cftn_full_math_training_v4"
    assert value["initialization"] == {
        "mode": "random_scratch_v1",
        "layers": 24,
        "source_checkpoint": None,
    }
    assert value["retention_baseline"] is None
    assert value["parent_dataset"] == {
        "manifest_sha256": "2a4efc0b96c6d404327e219da9c078c69718367368ae156e134d4776535daef3",
        "config_sha256": "e247a6f8275297264552b41aec1ec3481ebfd05dd20785041d295bf06ee869c6",
        "generator_sha256": "5d7048ccbee653599e6037c222591be10eff6d7c25a2eaaf90bc4d1379094e40",
    }
    assert not value["zero_update_entrance"]["enabled"]
    assert not value["preservation_distillation"]["enabled"]
    assert value["curriculum"]["transition_policy"] == "competency_gated_v3"
    assert value["math_training"]["learning_rate"] == 1e-4
    assert [phase["maximum_epochs"] for phase in value["phases"]] == [
        8,
        10,
        12,
        18,
        22,
        30,
    ]
    assert all(
        sum(group["examples"] for group in phase["quota_groups"]) == 100000
        for phase in value["phases"]
    )
    verified = {"verified_school_full", "cftn_generated"}
    assert all(
        set(group["filters"]["sources"]) <= verified
        for phase in value["phases"][:2]
        for group in phase["quota_groups"]
    )
    assert any(
        group["filters"].get("sources") == ["deepmind_mathematics"]
        for phase in value["phases"][3:]
        for group in phase["quota_groups"]
    )


def test_competency_v4_rejects_checkpoint_or_future_foundation_data(tmp_path):
    path = Path(__file__).parents[1] / "config/v2_math_scratch_v4.json"
    overlay = json.loads(path.read_text())
    overlay["base_settings"] = str(path.parent / "v2_full_supervision_v3.json")
    overlay["initialization"]["source_checkpoint"] = "old.pth"
    bad = tmp_path / "bad_source.json"
    bad.write_text(json.dumps(overlay))
    with pytest.raises(ValueError, match="random-scratch"):
        checked_competency_settings(bad)

    overlay = json.loads(path.read_text())
    overlay["base_settings"] = str(path.parent / "v2_full_supervision_v3.json")
    overlay["phases"][0]["quota_groups"][0]["filters"] = {
        "sources": ["deepmind_mathematics"]
    }
    bad = tmp_path / "bad_foundation.json"
    bad.write_text(json.dumps(overlay))
    with pytest.raises(ValueError, match="verified procedural"):
        checked_competency_settings(bad)

    overlay = json.loads(path.read_text())
    overlay["base_settings"] = str(path.parent / "v2_full_supervision_v3.json")
    overlay.pop("parent_dataset")
    bad = tmp_path / "unsealed_parent.json"
    bad.write_text(json.dumps(overlay))
    with pytest.raises(ValueError, match="sealed parent"):
        checked_competency_settings(bad)


def test_competency_v4_parent_is_only_required_for_scratch_contracts():
    path = Path(__file__).parents[1] / "config/v2_full_supervision_v3.json"
    value = checked_competency_settings(path)
    assert value["format"] == "cftn_full_math_training_v3"
    assert "parent_dataset" not in value


def test_full_data_audit_forwards_an_explicit_parent_pin(monkeypatch, tmp_path):
    import cftn_text.full_math_data as full_math_data

    observed = {}
    monkeypatch.setattr(full_math_data, "audit_full_data", lambda root, **kwargs: observed.update(kwargs) or {"pass": True})
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_text(json.dumps({
        "manifest_sha256": "x" * 64,
        "splits": {"train": {"path": "train.jsonl"}},
    }))
    monkeypatch.setattr(full_math_data, "audit_v2_manifest", lambda *args: {"pass": True})
    monkeypatch.setattr(full_math_data, "load_v2_records", lambda *args: [])
    monkeypatch.setattr(full_math_data, "school_rows", lambda *args: ([], [], []))
    monkeypatch.setattr(full_math_data, "write_rows", lambda path, rows: {"path": Path(path).name, "sha256": "a", "count": 0})
    monkeypatch.setattr(full_math_data, "file_sha256", lambda *args: "a")
    result = full_math_data.prepare_full_data(
        parent,
        tmp_path / "full",
        expected_parent_manifest_sha256="x" * 64,
    )
    assert result == {"pass": True}
    assert observed == {"expected_parent_manifest_sha256": "x" * 64}


def test_exact_trace_gate_cannot_be_satisfied_by_correct_answer_only():
    phase = {"name": "foundation", "through_epoch": 20, "minimum_generation_accuracy": .99,
             "minimum_valid_rate": 1.0, "minimum_trace_exact_by_family": {"addition": .95}}
    panel = {"accuracy": 1.0, "valid_rate": 1.0, "trace_exact_by_family": {"addition": {"rate": .8}}}
    result = _phase_generation_acceptance(phase=phase, generation_panels={"validation": panel}, validation={}, epoch=20)
    assert not result["pass"] and result["terminal_epoch"]


def test_strict_validation_rejects_capped_answer(monkeypatch, tmp_path):
    from cftn_text.math_validation import evaluate_generation_panel
    import tools.pilot_math_primitives as primitives
    row = full_school_record("What is 12 times -3?")
    monkeypatch.setattr(primitives, "generate_with_termination", lambda *a: [{
        "generation": row["target_trace"], "eos_terminated": False, "budget_hit": True,
        "unexpected_control_token": False, "context_limit_hit": False}])
    result = evaluate_generation_panel(type("Model", (), {"max_sequence_length": 4096})(), ByteMathTokenizer(), [row],
        maximum_examples=1, batch_size=1, max_new_tokens=256, failure_examples=1, require_eos=True,
        rows_path=tmp_path / "rows.jsonl")
    assert result["accuracy"] == 0 and result["valid_rate"] == 0
    assert result["budget_hits"] == 1 and result["trace_exact_by_family"]["multiplication"]["rate"] == 0


def test_full_training_uses_repaired_batches_without_promoting_failed_gate(monkeypatch, tiny_config, tmp_path):
    import cftn_text.training as training
    from cftn_text.full_math_data import write_rows
    rows = [full_school_record("What is 12 times -3?")]
    root = tmp_path / "fixture"
    root.mkdir()
    meta = write_rows(root / "train.jsonl", rows)
    manifest = {"format": "cftn_text_broad_math_v2", "derivative_format": FORMAT,
                "manifest_sha256": fingerprint(rows), "splits": {"train": meta, "validation": meta}}
    monkeypatch.setattr(training, "load_data_contract", lambda c: (root, manifest))
    tiny_config["data"]["format"] = "cftn_text_broad_math_v2"
    tiny_config["data"]["curriculum"] = {"enabled": True, "examples_per_epoch": 1}
    contract = {"require_acceptance_for_best": True, "promote_final_phase_only": True,
                "math_training": {"objective": "computation_roles_v1", "role_weights": [.25, .5, .25],
                    "max_epochs": 1, "generation_validation": {"enabled": True, "examples": 1, "max_new_tokens": 2}},
                "phases": [{"name": "test", "through_epoch": 1, "max_difficulty": 3,
                            "minimum_generation_accuracy": .99, "minimum_valid_rate": 1.0}]}
    result = training.train_math_tower(tiny_config, device_name="cpu", max_batches=1, require_calibration=False,
                                      recovery_contract=contract, disable_early_stopping=True)
    assert result["state"] == "failed_acceptance"
    assert result["best_checkpoint"] is None
    assert Path(tiny_config["project"]["artifact_root"], "math", "checkpoint_epoch_0001.pth").is_file()
