from __future__ import annotations

from cftn_text.synergy_evaluation import _analyse_rows, _pass_criteria


CONDITIONS = (
    "correct",
    "shuffled",
    "gpt_to_math_shuffled",
    "math_to_gpt_shuffled",
    "gpt_to_math_disabled",
    "math_to_gpt_disabled",
    "both_disabled",
    "gpt_to_math_pair_swapped",
    "math_to_gpt_pair_swapped",
)


def _generation(answer, communication=False):
    result = {
        "gpt_generation": f"<answer>{answer}</answer>",
        "math_generation": f"<answer>{answer}</answer>",
    }
    if communication:
        result["communication"] = {
            "gpt_to_math_sender_gate": {"mean": 0.5},
            "gpt_to_math_message_norm": 2.0,
            "math_to_gpt_sender_gate": {"mean": 0.6},
            "math_to_gpt_message_norm": 3.0,
        }
    return result


def test_causal_analysis_requires_both_directions_and_beats_individuals():
    rows = []
    for pair in range(10):
        targets = (pair * 2, pair * 2 + 1)
        for variant, target in enumerate(targets):
            wrong = target + 100
            donor = targets[1 - variant]
            outputs = {condition: _generation(wrong) for condition in CONDITIONS}
            outputs["correct"] = _generation(target, communication=True)
            outputs["gpt_to_math_pair_swapped"] = _generation(
                target, communication=True
            )
            outputs["math_to_gpt_pair_swapped"] = _generation(donor)
            # Disabling only M->G leaves the serial math result intact.
            outputs["math_to_gpt_disabled"]["math_generation"] = (
                f"<answer>{target}</answer>"
            )
            rows.append(
                {
                    "record": {
                        "record_id": f"{pair}-{variant}",
                        "x": target,
                    },
                    "outputs": outputs,
                }
            )
    analysis = _analyse_rows(rows, samples=200, seed=7)
    protocol = {
        "success_criteria": {
            "minimum_synergy_gain": 0.10,
            "minimum_correct_vs_shuffled_gap": 0.10,
            "minimum_directional_gain": 0.02,
            "maximum_joint_vs_serial_regression": 0.02,
            "minimum_counterfactual_pair_accuracy": 0.90,
            "minimum_message_swap_donor_follow": 0.80,
            "require_ci95_above_zero": True,
        }
    }
    gates = _pass_criteria(analysis, protocol)
    assert analysis["arm_accuracy"]["joint_cftn"] == 1.0
    assert analysis["counterfactual"]["math_to_gpt_pair_swap_donor_follow_rate"] == 1.0
    assert gates["pass"]
