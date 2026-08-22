from __future__ import annotations

import torch

from cftn_text.v1_3_fusion import SpecialistAwareMessageFusion


def _fusion() -> SpecialistAwareMessageFusion:
    return SpecialistAwareMessageFusion(
        message_width=16,
        message_tokens=2,
        specialist_count=2,
        maximum_rounds=3,
        heads=4,
        dropout=0.0,
    )


def test_fusion_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(17)
    fusion = _fusion()
    messages = torch.randn(3, 8, 16)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    output = fusion(messages, mask, rounds=2)
    assert torch.equal(output, messages)
    assert torch.isfinite(output).all()


def test_fusion_projection_receives_gradient_from_active_rows() -> None:
    fusion = _fusion()
    messages = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
    output = fusion(messages, mask, rounds=1)
    output[0].square().mean().backward()
    assert fusion.output_projection.weight.grad is not None
    assert bool(fusion.output_projection.weight.grad.abs().sum().gt(0))


def test_fusion_rejects_message_layout_mismatch() -> None:
    fusion = _fusion()
    messages = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5)
    try:
        fusion(messages, mask, rounds=1)
    except ValueError as error:
        assert "expected 4 tokens" in str(error)
    else:
        raise AssertionError("fusion accepted an invalid return-message layout")

