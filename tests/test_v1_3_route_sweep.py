from __future__ import annotations

import torch

from cftn_text.v1_3_route_sweep import (
    enumerate_route_schedules,
    enumerate_route_schedules_up_to,
    parallel_component_statistics,
    route_schedule_name,
    route_schedule_tensor,
)


def test_enumerates_every_two_specialist_three_round_schedule() -> None:
    schedules = enumerate_route_schedules(3, 2)
    assert len(schedules) == 64
    assert len(set(schedules)) == 64
    assert ((1, 1), (0, 0), (0, 0)) in schedules
    assert route_schedule_name(((1, 1), (1, 0), (0, 1))) == "both>math>string"
    assert len(enumerate_route_schedules_up_to(3, 2)) == 84


def test_route_schedule_tensor_repeats_schedule_per_example() -> None:
    schedule = ((1, 1), (0, 1), (0, 0))
    values = route_schedule_tensor(
        schedule,
        batch_size=3,
        specialist_count=2,
        device=torch.device("cpu"),
    )
    assert values.shape == (3, 3, 2)
    assert values[0].tolist() == [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    assert torch.equal(values[0], values[2])


def test_parallel_component_statistics_separates_content_and_format() -> None:
    exact = parallel_component_statistics("-3|cba\n", "-3|cba")
    assert exact["exact"] == 1
    assert exact["both_components_correct"] == 1

    wrong_string = parallel_component_statistics("-3|cab", "-3|cba")
    assert wrong_string["math_component_correct"] == 1
    assert wrong_string["string_component_correct"] == 0

    malformed = parallel_component_statistics("answer=-3 and cba", "-3|cba")
    assert malformed["valid_format"] == 0
    assert malformed["contains_both_components"] == 1
    assert malformed["format_only_error"] == 1
