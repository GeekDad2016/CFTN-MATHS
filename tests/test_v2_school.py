import copy
import json
from pathlib import Path

import pytest
import torch

from cftn_text.computation_supervision import computation_loss
from cftn_text.data_generator import canonical_trace
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.v2_school_data import (
    BANDS, FAMILIES, SchoolCollator, build_school_corpus, math_key,
    numerical_band, parse_public, question, school_record,
)
from cftn_text.verified_math_data import REPLAY_FAMILIES
from tools.train_v2_verified_school import curriculum_gate, epoch_schedule, intact_output, next_band, settings_checked


@pytest.fixture(scope="module")
def corpus():
    return build_school_corpus()


@pytest.fixture
def settings():
    return settings_checked(Path(__file__).parents[1] / "config/v2_verified_school_trial.json")


@pytest.mark.parametrize("family,values,answer", [
    ("addition", (-15, 7), -8), ("subtraction", (54, 66), -12),
    ("multiplication", (84, 2), 168), ("multiplication", (59, -5), -295),
    ("multiplication", (-99, -99), 9801), ("multiplication", (0, 3), 0),
    ("multiplication", (-10, 3), -30), ("division", (-225, 15), -15),
    ("division", (12, -3), -4), ("linear_equation", (2, -22, -62), -20),
    ("linear_equation", (-8, -50, 110), -20),
])
def test_all_public_wordings_and_verified_arithmetic(family, values, answer):
    for style in range(4):
        text = question(family, values, style)
        assert parse_public(text) == (family, values)
        row = school_record(text)
        assert row["normalized_answer"] == str(answer)
        if family == "linear_equation":
            assert row["target_trace"] == canonical_trace(*values, answer)
        else:
            assert row["steps"][-1]["result"] == answer
        for span in row["supervision_spans"]:
            assert row["target_trace"][span["start"]:span["end"]]
        assert row["supervision_spans"][-1]["kind"] == "copy"


@pytest.mark.parametrize("text", ["Calculate 7/(3).", "Calculate 3/(0).", "Solve 0*x + (2) = 4.",
                                  "Solve 3*x + (2) = 4.", "Calculate 1000*1000.", "Calculate 2+2;answer=4."])
def test_invalid_or_unsupported_school_questions_rejected(text):
    with pytest.raises(ValueError):
        school_record(text)


def test_object_split_groups_paraphrases_swaps_and_scaled_equations():
    assert math_key("multiplication", (-15, 8)) == math_key("multiplication", (8, -15))
    assert math_key("linear_equation", (2, -22, -62)) == math_key("linear_equation", (-4, 44, 124))
    assert len({school_record(question("linear_equation", (2, -22, -62), s))["computation_key"] for s in range(4)}) == 1
    assert math_key("subtraction", (7, 2)) != math_key("subtraction", (2, 7))


def test_v1_numerical_boundaries():
    assert numerical_band("linear_equation", (8, 50, 210)) == "foundations"
    assert numerical_band("linear_equation", (9, 50, 230)) == "two_digit"
    assert numerical_band("linear_equation", (16, 125, 925)) == "two_digit"
    assert numerical_band("linear_equation", (17, 125, 975)) == "three_digit"
    assert numerical_band("multiplication", (15, -15)) == "foundations"
    assert numerical_band("multiplication", (16, -15)) == "two_digit"
    assert numerical_band("division", (225, 15)) == "foundations"


def test_corpus_support_and_no_mathematical_object_leakage(corpus):
    keys = {s: set() for s in ("train", "validation")}
    tokenizer = ByteMathTokenizer()
    for band, families in corpus.items():
        for family, pools in families.items():
            assert len(pools["validation"]) == 64
            assert len(pools["train"]) >= 250
            for split, rows in pools.items():
                for row in rows:
                    assert row["band"] == band and row["family"] == family and row["split"] == split
                    assert row == school_record(row["problem"])
                    values = parse_public(row["problem"])[1]
                    assert row["problem"] in [question(family, values, s) for s in range(3)]
                    keys[split].add(row["computation_key"])
                    tokenizer.encode_training_example(row["problem"], row["target_trace"], 4096)
                    assert len(tokenizer.encode(row["target_trace"])) + 1 <= 256
    assert not keys["train"] & keys["validation"]


def test_compute_roles_cover_only_target_and_reject_hidden_overrides():
    rows = [school_record("Calculate -15*7."), school_record("Solve 2*x + (-22) = -62.")]
    collator = SchoolCollator(ByteMathTokenizer(), 4096)
    batch = collator(rows)
    assert torch.equal(batch["math_roles"].ne(-100), batch["math_labels"].ne(-100))
    for index, row in enumerate(rows):
        prefix = int(batch["math_prefix_lengths"][index])
        assert batch["math_roles"][index, :prefix].eq(-100).all()
        for span in row["supervision_spans"]:
            roles = batch["math_roles"][index, prefix + span["start"]:prefix + span["end"]]
            assert roles.eq(1 if span["kind"] == "compute" else 2).all()
    logits = torch.zeros(*batch["math_labels"].shape, 260, requires_grad=True)
    loss = computation_loss(logits, batch["math_labels"], batch["math_roles"], weights=(.25, .5, .25))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    for key, value in (("normalized_answer", "999"), ("target_trace", "<answer>999</answer>"),
                       ("math_problem", "hidden solution"), ("supervision_spans", [])):
        bad = copy.deepcopy(rows[0])
        bad[key] = value
        with pytest.raises(ValueError):
            collator([bad])


def test_epoch_is_balanced_math_with_exact_quarter_replay(corpus, settings):
    replay = {f: [{"family": f, "record_id": f}] for f in REPLAY_FAMILIES}
    settings["examples_per_epoch"] = 1024
    for index in range(3):
        batches = list(epoch_schedule(corpus, replay, index, 1, settings))
        assert batches == list(epoch_schedule(corpus, replay, index, 1, settings))
        assert len(batches) == 64
        school_rows = []
        for batch in batches:
            assert len(batch) == 16
            assert sum(r["family"] in REPLAY_FAMILIES for r in batch) == 4
            assert all(sum(r["family"] == f for r in batch) == 2 for f in REPLAY_FAMILIES)
            school_rows.extend(r for r in batch if r["family"] in FAMILIES)
        assert {r["family"] for r in school_rows} == set(FAMILIES)
        assert all(r["split"] == "train" and BANDS.index(r["band"]) <= index for r in school_rows)
        if index:
            assert any(BANDS.index(r["band"]) < index for r in school_rows)


def test_gates_are_not_loss_based_and_require_retention(settings):
    report = {"current/" + f: {"accuracy": 1., "valid_rate": 1., "trace_exact_rate": 1., "budget_hits": 0} for f in FAMILIES}
    baseline = {"replay/" + f: {"accuracy": .9} for f in REPLAY_FAMILIES}
    report.update(copy.deepcopy(baseline))
    gate = curriculum_gate(report, baseline, settings)
    assert gate["pass"] and not gate["production_acceptance"]
    assert next_band(0, 1, gate, settings) == (0, 1)
    assert next_band(0, 2, gate, settings) == (1, 0)
    for field, value in (("accuracy", .98), ("valid_rate", .99), ("trace_exact_rate", .94), ("budget_hits", 1)):
        bad = copy.deepcopy(report)
        bad["current/addition"][field] = value
        failed = curriculum_gate(bad, baseline, settings)
        assert not failed["pass"] and next_band(0, 3, failed, settings) == (0, 3)
    report["replay/" + REPLAY_FAMILIES[0]]["accuracy"] = .86
    assert not curriculum_gate(report, baseline, settings)["pass"]
    assert not curriculum_gate({}, baseline, settings)["pass"]


def test_output_requires_a_single_complete_work_answer_and_eos():
    row = {"generation": "<work>1+1=2</work><answer>2</answer>", "eos_terminated": True,
           "unexpected_control_token": False, "budget_hit": False, "context_limit_hit": False}
    assert intact_output(row)
    for text in ("</answer>", "<work>1+1=2</work><answer></answer>", row["generation"] * 2):
        assert not intact_output({**row, "generation": text})
    assert not intact_output({**row, "eos_terminated": False})
    assert not intact_output({**row, "budget_hit": True})


@pytest.mark.parametrize("key,value", [("epochs", 4), ("production_acceptance", True),
    ("checkpoint_promotion", True), ("minimum_epochs_per_band", 1), ("replay_fraction", 0),
    ("max_wall_seconds", 9999), ("warmup_updates", 99999), ("learning_rate", .001),
    ("role_weights", {"format": .8, "compute": .1, "copy": .1})])
def test_trial_cannot_silently_expand_or_change_objective(settings, tmp_path, key, value):
    settings[key] = value
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        settings_checked(path)
