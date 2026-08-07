from __future__ import annotations

import torch

from cftn_text.bridges import ContextualMessageBridge, GatedCrossReceiver


def make_bridge():
    torch.manual_seed(11)
    return ContextualMessageBridge(
        sender_width=16,
        message_width=16,
        message_tokens=4,
        heads=4,
        gate_hidden_size=16,
        dropout=0.0,
        gate_init=-1.0,
        zero_init_output=False,
    )


def test_message_gates_are_contextual_independent_and_not_routes():
    bridge = make_bridge().eval()
    # Use genuinely different token patterns. Layer normalization intentionally
    # treats constant offset/scale copies as equivalent contexts.
    hidden = torch.randn(2, 5, 16)
    output = bridge(hidden, torch.ones(2, 5, dtype=torch.long))
    assert output.message.shape == (2, 4, 16)
    assert output.gate.shape == (2, 4, 1)
    assert torch.all((output.gate > 0) & (output.gate < 1))
    assert not torch.equal(output.gate[0], output.gate[1])
    # Independent sigmoid gates do not form a probability distribution over towers.
    assert not torch.allclose(output.gate.sum(dim=1), torch.ones(2, 1))


def test_disabled_bridge_still_executes_but_sends_exact_zero():
    bridge = make_bridge().eval()
    output = bridge(torch.randn(2, 5, 16), enabled=False)
    assert bridge.execution_count == 1
    assert torch.count_nonzero(output.gate) == 0
    assert torch.count_nonzero(output.message) == 0


def test_receiver_disabled_path_is_bit_exact():
    receiver = GatedCrossReceiver(
        receiver_width=16,
        message_width=16,
        heads=4,
        gate_hidden_size=16,
        dropout=0.0,
        gate_init=-1.0,
    ).eval()
    hidden = torch.randn(2, 6, 16)
    message = torch.randn(2, 4, 16)
    disabled = receiver(hidden, message, enabled=False)
    assert torch.equal(disabled, hidden)
    assert receiver.execution_count == 1


def test_zero_initialized_receiver_has_a_live_first_step_gradient():
    receiver = GatedCrossReceiver(
        receiver_width=16,
        message_width=16,
        heads=4,
        gate_hidden_size=16,
        dropout=0.0,
        gate_init=-1.0,
        zero_init_output=True,
    )
    hidden = torch.randn(2, 6, 16)
    message = make_bridge()(torch.randn(2, 5, 16)).message.detach()
    receiver(hidden, message).square().mean().backward()
    assert receiver.output_projection.weight.grad is not None
    assert receiver.output_projection.weight.grad.abs().sum() > 0
