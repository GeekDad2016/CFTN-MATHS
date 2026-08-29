from __future__ import annotations

import copy
from fractions import Fraction
import json

import pytest
import torch

from cftn_text.computation_supervision import (
    ComputationCollator,
    computation_loss,
    hybrid_computation_loss,
)
from cftn_text.dataset import MathCollator
from cftn_text.tokenizer import ByteMathTokenizer, SequenceTooLongError
from cftn_text.verified_math_data import (
    Step, audit_mathqa_program, computation_key, curriculum_band, decimal_text,
    fingerprint, independent_answer, legacy_spans, operate, procedure,
    validate_verified_record, verified_record,
)
from tools.pilot_verified_math import screening_decision, training_schedule, validate_bundle, write_rows
from tools.analyze_math_supervision_pilot import first_procedure_error


def record(family="two_variable_systems", problem=None, answer=None):
    if family == "two_variable_systems":
        problem = problem or "Find both unknowns: (4)x+(1)y=-117, and (5)x+(12)y=-157."
        answer = answer or "x=-29;y=-1"
        trace = f"<work>det=43;x=-29;y=-1;verify=(-117,-157)</work><answer>{answer}</answer>"
    else:
        problem = problem or "Calculate -77.3*0.02."
        answer = answer or "-1.546"
        trace = f"<answer>{answer}</answer>"
    return {"record_id": "parent", "content_id": "content", "source": "fixture",
            "family": family, "difficulty": 1, "split": "train", "problem": problem,
            "normalized_answer": answer, "target_trace": trace}


def test_system_procedure_verified_by_independent_elimination_and_residuals():
    original = record()
    before = copy.deepcopy(original)
    traced = verified_record(original)
    validate_verified_record(traced)
    assert original == before
    assert traced["normalized_answer"] == "x=-29;y=-1"
    assert "ad=multiply(4,12)=48" in traced["target_trace"]
    assert "nx=subtract(-1404,-157)=-1247" in traced["target_trace"]
    assert [s["result"] for s in traced["steps"] if s["name"] in ("r1", "r2")] == ["0", "0"]
    assert traced["parent_record_id"] == original["record_id"]


def test_first_error_analysis_distinguishes_arithmetic_from_wrong_binding():
    row = record()
    trace = verified_record(row)["target_trace"]
    assert first_procedure_error(row, trace)["error"] is None
    swapped = trace.replace("ad=multiply(4,12)=48", "ad=multiply(12,4)=48")
    assert first_procedure_error(row, swapped)["error"] is None
    wrong_value = trace.replace("ad=multiply(4,12)=48", "ad=multiply(4,12)=49")
    assert first_procedure_error(row, wrong_value)["error"] == "wrong_computed_value"
    wrong_operand = trace.replace("ad=multiply(4,12)=48", "ad=multiply(4,13)=52")
    assert first_procedure_error(row, wrong_operand)["error"] == "wrong_operand_binding"
    assert first_procedure_error(row, trace[:25])["error"] == "malformed_or_incomplete_step"


@pytest.mark.parametrize("a,b,answer", [(0, 15, "0"), (-13, -21, "273"),
                                      (999, 999, "998001"), ("0.001", "-0.02", "-0.00002"),
                                      ("-1.25", "0.8", "-1")])
def test_signed_decimal_multiplication(a, b, answer):
    row = record("arithmetic__mul", f"What is the product of {a} and {b}?", answer)
    traced = verified_record(row)
    validate_verified_record(traced)
    assert traced["normalized_answer"] == answer
    assert traced["steps"][-1]["name"] == "value"


@pytest.mark.parametrize("problem", ["Work out -2 * 3.", "-2 times 3", "Product of -2 and 3.", "What is -2*3?", "Multiply -2 and 3."])
def test_supported_visible_multiplication_templates(problem):
    assert independent_answer(record("arithmetic__mul", problem, "-6")) == "-6"


def test_no_hidden_answer_metadata_used_and_wrong_final_answer_rejected():
    row = record()
    row.update(x=123, y=456, math_problem="Opaque slots: 999", metadata={"solution": "bad"})
    assert independent_answer(row) == "x=-29;y=-1"
    assert verified_record(row)["normalized_answer"] == "x=-29;y=-1"
    row["normalized_answer"] = "x=123;y=456"
    with pytest.raises(ValueError, match="disagree"):
        verified_record(row)


def test_singular_and_unsupported_grammar_fail_closed():
    row = record(problem="Solve the system 1*x + (2)*y = 3; 2*x + (4)*y = 6. Give x and y.")
    with pytest.raises(ValueError):
        verified_record(row)
    with pytest.raises(ValueError, match="grammar"):
        verified_record(record("arithmetic__mul", "Prove Riemann; 3 4", "12"))


def test_zero_pivot_and_fractional_system():
    row = record(problem="Solve the system 0*x + (2)*y = 1; 3*x + (4)*y = 3. Give x and y.", answer="x=1/3;y=1/2")
    assert independent_answer(row) == "x=1/3;y=1/2"
    validate_verified_record(verified_record(row))


def test_incorrect_step_and_resigned_corruption_rejected():
    with pytest.raises(ValueError, match="incorrect arithmetic"):
        Step("value", "multiply", "3", "4", "13").check()
    row = verified_record(record())
    row["steps"][0]["result"] = "49"
    row["record_id"] = fingerprint({k: v for k, v in row.items() if k != "record_id"})
    with pytest.raises(ValueError, match="content mismatch"):
        validate_verified_record(row)


@pytest.mark.parametrize("op,a,b", [("divide", 3, 0), ("power", 2, 300), ("eval", 2, 3), ("multiply", 2**256, 2)])
def test_arithmetic_safety_bounds(op, a, b):
    with pytest.raises(ValueError):
        operate(op, Fraction(a), Fraction(b))


def test_finite_decimal_rendering_does_not_round():
    assert decimal_text(Fraction(1, 8)) == "0.125"
    assert decimal_text(Fraction(-1, 10000000000)) == "-0.0000000001"
    assert decimal_text(Fraction(1, 3)) == "1/3"


def test_duplicate_arithmetic_keys_ignore_prompt_and_operand_order():
    a = record("arithmetic__mul", "Calculate -1.5*2.", "-3")
    b = record("arithmetic__mul", "Product of 2 and -1.50.", "-3")
    assert computation_key(a) == computation_key(b)
    assert curriculum_band(a) == "foundation"


def mathqa(program="multiply(n0,n1)|subtract(n3,#0)", answer="52"):
    return {"record_id": "frisbee", "raw_problem": "64 frisbees priced at 3 or 4 for 204 total",
            "native_program": program, "normalized_answer": answer,
            "target_trace": f"<program>{program}</program><answer>{answer}</answer>"}


def test_incomplete_mathqa_program_quarantined_not_relabeled():
    row = mathqa()
    before = copy.deepcopy(row)
    result = audit_mathqa_program(row)
    assert result["status"] == "quarantine_program_answer_mismatch"
    assert result["program_value"] == "12"
    assert result["expected"] == "52"
    assert result["training_eligible"] is False
    assert row == before


def test_consistent_mathqa_program_is_not_automatic_semantic_certification():
    result = audit_mathqa_program(mathqa("multiply(n0,n1)|subtract(n3,#0)|subtract(n0,#1)"))
    assert result["status"] == "internally_consistent_needs_semantic_review"
    assert result["program_value"] == "52"
    assert not result["training_eligible"] and not result["semantic_verification"]


@pytest.mark.parametrize("program", ["divide(n0,const_0)", "power(n0,const_100)",
                                    "multiply(n99,n0)", "add(#0,n0)", "os.system(n0,n1)"])
def test_unsupported_or_invalid_mathqa_program_quarantined(program):
    assert audit_mathqa_program(mathqa(program))["status"] == "quarantine_unsupported_or_ambiguous"


def test_mathqa_rounding_and_ambiguous_bindings_do_not_silently_pass():
    row = mathqa("divide(n0,n1)", "21.33")
    assert audit_mathqa_program(row)["status"] == "quarantine_program_answer_mismatch"
    row["raw_problem"] = "A ratio of 64:3 or 4 for 204"
    assert audit_mathqa_program(row)["status"] == "quarantine_unsupported_or_ambiguous"


def test_masks_cover_exact_targets_and_focus_calculation_not_copied_answer():
    tokenizer = ByteMathTokenizer()
    row = verified_record(record())
    batch = ComputationCollator(tokenizer, 4096)([row, verified_record(record("arithmetic__mul"))])
    assert torch.equal(batch["math_roles"].ne(-100), batch["math_labels"].ne(-100))
    for index in range(2):
        prefix = int(batch["math_prefix_lengths"][index])
        assert batch["math_labels"][index, :prefix].eq(-100).all()
    span = next(s for s in row["supervision_spans"] if s["name"] == "ad")
    prefix = int(batch["math_prefix_lengths"][0])
    assert tokenizer.decode(batch["math_input_ids"][0, prefix + span["start"]:prefix + span["end"]]) == "48"
    assert batch["math_roles"][0, prefix + span["start"]:prefix + span["end"]].eq(1).all()
    span = row["supervision_spans"][-1]
    assert batch["math_roles"][0, prefix + span["start"]:prefix + span["end"]].eq(2).all()
    with pytest.raises(SequenceTooLongError):
        ComputationCollator(tokenizer, 20)([row])


def test_legacy_payload_focus_excludes_answer_tags_and_default_collator_unchanged():
    row = record("arithmetic__mul")
    batch = ComputationCollator(ByteMathTokenizer(), 4096)([row])
    assert int(batch["math_roles"].eq(1).sum()) == len("-1.546")
    legacy = MathCollator(ByteMathTokenizer(), 4096)([row])
    for key in legacy:
        if torch.is_tensor(legacy[key]):
            assert torch.equal(legacy[key], batch[key])
    with pytest.raises(ValueError, match="unverified program"):
        legacy_spans(mathqa())


def test_fraction_to_decimal_output_is_computation_not_copying():
    row = verified_record(record("arithmetic__mul"))
    assert row["steps"][-1]["result"] == "-773/500"
    assert row["normalized_answer"] == "-1.546"
    assert row["supervision_spans"][-1]["kind"] == "compute"
    integer = verified_record(record("arithmetic__mul", "Calculate 3*4.", "12"))
    assert integer["supervision_spans"][-1]["kind"] == "copy"
    batch = ComputationCollator(ByteMathTokenizer(), 4096)([row])
    start = int(batch["math_prefix_lengths"][0]) + row["supervision_spans"][-1]["start"]
    assert batch["math_roles"][0, start:start + len("-1.546")].eq(1).all()


def test_role_balanced_loss_has_gradients_and_padding_invariance():
    torch.manual_seed(3)
    labels = torch.tensor([[-100, 1, 2, 3, 1, -100]])
    roles = torch.tensor([[-100, 0, 1, 1, 2, -100]])
    logits = torch.randn(1, 6, 4, requires_grad=True)
    loss = computation_loss(logits, labels, roles)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 4:].eq(0).all()
    assert torch.allclose(loss, computation_loss(logits[:, :5], labels[:, :5], roles[:, :5]))
    with pytest.raises(ValueError):
        computation_loss(logits, labels, roles.masked_fill(roles.eq(1), 0))


def test_role_mean_prevents_long_computation_spans_from_dominating():
    # Same prediction distribution per role; duplicate only compute positions.
    short_logits = torch.tensor([[[3., 1.], [1., 3.], [2., 1.], [0., 0.]]])
    short_labels = torch.tensor([[-100, 0, 0, 0]])
    short_roles = torch.tensor([[-100, 0, 1, 2]])
    long_logits = torch.cat([short_logits[:, :2], short_logits[:, 1:2].repeat(1, 5, 1), short_logits[:, 2:]], dim=1)
    long_labels = torch.tensor([[-100] + [0] * 8])
    long_roles = torch.tensor([[-100, 0] + [1] * 6 + [2]])
    assert torch.allclose(computation_loss(short_logits, short_labels, short_roles),
                          computation_loss(long_logits, long_labels, long_roles))


def test_hybrid_computation_loss_preserves_full_sequence_and_role_gradients():
    torch.manual_seed(17)
    labels = torch.tensor([[-100, 1, 2, 3, 1, -100]])
    roles = torch.tensor([[-100, 0, 1, 1, 2, -100]])
    logits = torch.randn(1, 6, 4, requires_grad=True)
    loss = hybrid_computation_loss(
        logits,
        labels,
        roles,
        weights=(0.15, 0.65, 0.20),
        role_fraction=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 4:].eq(0).all()
    with pytest.raises(ValueError, match="role fraction"):
        hybrid_computation_loss(logits, labels, roles, role_fraction=1.0)


def test_bundle_provenance_and_curriculum_are_checked_not_just_digests():
    from cftn_text.v2_data import iter_local_records

    row = next(r for r in iter_local_records(count=12, split="train", seed=719)
               if r["family"] == "two_variable_systems")
    bundle = {"original": row, "verified": verified_record(row),
              "band": curriculum_band(row), "computation_key": computation_key(row)}
    validate_bundle(bundle)
    bad = copy.deepcopy(bundle)
    bad["band"] = "made_up"
    with pytest.raises(ValueError, match="curriculum"):
        validate_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad["verified"]["parent_content_id"] = "unrelated"
    bad["verified"]["record_id"] = fingerprint({k: v for k, v in bad["verified"].items() if k != "record_id"})
    with pytest.raises(ValueError, match="parent"):
        validate_bundle(bad)
    bad = copy.deepcopy(bundle)
    bad["computation_key"] = "unrelated"
    with pytest.raises(ValueError, match="split key"):
        validate_bundle(bad)


def test_curriculum_schedule_is_identical_and_replays_every_step():
    rows = []
    for family in ("two_variable_systems", "arithmetic__mul", "variables_both_sides", "nested_parentheses"):
        for band in (("replay",) if family in ("variables_both_sides", "nested_parentheses") else ("foundation", "expanded")):
            rows.append({"original": {"family": family}, "band": band})
    schedule = training_schedule(rows, 30, 16, 719)
    assert schedule == training_schedule(rows, 30, 16, 719)
    assert all(rows[i]["band"] != "expanded" for batch in schedule[:10] for i in batch)
    assert any(rows[i]["band"] == "expanded" for batch in schedule[10:] for i in batch)
    assert all(sum(rows[i]["band"] == "replay" for i in batch) == 4 for batch in schedule)


def test_writer_refuses_existing_target_and_screen_never_releases_pipeline(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_rows(path, [{"preserve": True}])
    with pytest.raises(FileExistsError):
        write_rows(path, [])
    assert json.loads(path.read_text())["preserve"]
    families = ("two_variable_systems", "arithmetic__mul", "variables_both_sides", "nested_parentheses", "broad_diagnostic")
    reports = {a: {f: {"accuracy": 0.5, "generation_budget_hits": 0} for f in families}
               for a in ("baseline", "control", "loss_only", "verified")}
    assert not screening_decision(reports)["screen_pass"]
    reports["verified"]["two_variable_systems"]["accuracy"] = 0.8
    reports["verified"]["arithmetic__mul"]["accuracy"] = 0.8
    result = screening_decision(reports)
    assert result["screen_pass"]
    assert not result["production_acceptance"] and not result["checkpoint_promotion"]
