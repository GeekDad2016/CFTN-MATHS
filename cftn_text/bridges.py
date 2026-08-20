from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BridgeOutput:
    message: torch.Tensor
    gate: torch.Tensor
    attention_entropy: torch.Tensor


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    if mask.shape != hidden.shape[:2]:
        raise ValueError("attention mask must match hidden-state sequence dimensions")
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    probabilities = weights.clamp_min(1e-9)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    denominator = math.log(max(2, weights.shape[-1]))
    return entropy.mean(dim=-1) / denominator


class ContextualMessageBridge(nn.Module):
    """Translate sender states into fixed-width messages with contextual gates.

    Every sample and message token receives its own independent sigmoid gate.
    Gates are not normalized against one another and never select a tower.
    """

    def __init__(
        self,
        sender_width: int,
        message_width: int,
        message_tokens: int,
        heads: int,
        gate_hidden_size: int,
        dropout: float,
        gate_init: float,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if message_width % heads:
            raise ValueError("message width must be divisible by attention heads")
        if min(sender_width, message_width, message_tokens, gate_hidden_size) < 1:
            raise ValueError("bridge dimensions must be positive")
        self.sender_width = int(sender_width)
        self.message_width = int(message_width)
        self.message_tokens = int(message_tokens)
        self.sender_norm = nn.LayerNorm(sender_width)
        self.sender_projection = nn.Linear(sender_width, message_width)
        self.message_queries = nn.Parameter(
            torch.randn(1, message_tokens, message_width) * 0.02
        )
        self.attention = nn.MultiheadAttention(
            message_width, heads, dropout=dropout, batch_first=True
        )
        self.message_norm = nn.LayerNorm(message_width)
        self.output_projection = nn.Linear(message_width, message_width)
        self.gate_network = nn.Sequential(
            nn.Linear(message_width * 2, gate_hidden_size),
            nn.GELU(),
            nn.Linear(gate_hidden_size, 1),
        )
        nn.init.constant_(self.gate_network[-1].bias, float(gate_init))
        nn.init.normal_(self.gate_network[-1].weight, std=0.02)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
        self.execution_count = 0
        self.last_gate: torch.Tensor | None = None

    def reset_execution_count(self) -> None:
        self.execution_count = 0

    def forward(
        self,
        sender_hidden: torch.Tensor,
        sender_attention_mask: torch.Tensor | None = None,
        *,
        enabled: bool = True,
        gate_mode: str = "contextual",
    ) -> BridgeOutput:
        self.execution_count += 1
        if sender_hidden.ndim != 3 or sender_hidden.shape[-1] != self.sender_width:
            raise ValueError(
                f"sender hidden states must have shape [B, L, {self.sender_width}]"
            )
        if gate_mode not in {"contextual", "fixed_open"}:
            raise ValueError("gate_mode must be contextual or fixed_open")
        batch = sender_hidden.shape[0]
        memory = self.sender_projection(self.sender_norm(sender_hidden))
        queries = self.message_queries.expand(batch, -1, -1)
        padding_mask = (
            None
            if sender_attention_mask is None
            else ~sender_attention_mask.to(dtype=torch.bool)
        )
        attended, weights = self.attention(
            queries,
            memory,
            memory,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = self.message_norm(attended)
        pooled = _masked_mean(memory, sender_attention_mask)
        pooled = pooled.unsqueeze(1).expand(-1, self.message_tokens, -1)
        contextual_gate = torch.sigmoid(
            self.gate_network(torch.cat((attended, pooled), dim=-1))
        )
        gate = torch.ones_like(contextual_gate) if gate_mode == "fixed_open" else contextual_gate
        if not enabled:
            gate = torch.zeros_like(gate)
        message = gate * self.output_projection(attended)
        entropy = _normalized_entropy(weights)
        self.last_gate = gate.detach()
        return BridgeOutput(message=message, gate=gate, attention_entropy=entropy)


class GatedCrossReceiver(nn.Module):
    """Inject a translated message as a context-dependent residual."""

    def __init__(
        self,
        receiver_width: int,
        message_width: int,
        heads: int,
        gate_hidden_size: int,
        dropout: float,
        gate_init: float,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if message_width % heads:
            raise ValueError("message width must be divisible by attention heads")
        self.receiver_width = int(receiver_width)
        self.message_width = int(message_width)
        self.receiver_norm = nn.LayerNorm(receiver_width)
        self.query_projection = nn.Linear(receiver_width, message_width)
        self.message_norm = nn.LayerNorm(message_width)
        self.attention = nn.MultiheadAttention(
            message_width, heads, dropout=dropout, batch_first=True
        )
        self.output_projection = nn.Linear(message_width, receiver_width)
        self.gate_network = nn.Sequential(
            nn.Linear(message_width * 2, gate_hidden_size),
            nn.GELU(),
            nn.Linear(gate_hidden_size, 1),
        )
        nn.init.constant_(self.gate_network[-1].bias, float(gate_init))
        nn.init.normal_(self.gate_network[-1].weight, std=0.02)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
        self.execution_count = 0
        self.last_gate: torch.Tensor | None = None

    def reset_execution_count(self) -> None:
        self.execution_count = 0

    def forward(
        self,
        receiver_hidden: torch.Tensor,
        message: torch.Tensor | None,
        message_attention_mask: torch.Tensor | None = None,
        *,
        enabled: bool = True,
        gate_mode: str = "contextual",
    ) -> torch.Tensor:
        self.execution_count += 1
        if not enabled or message is None:
            self.last_gate = None
            return receiver_hidden
        if receiver_hidden.ndim != 3 or receiver_hidden.shape[-1] != self.receiver_width:
            raise ValueError(
                f"receiver states must have shape [B, L, {self.receiver_width}]"
            )
        if message.ndim != 3 or message.shape[-1] != self.message_width:
            raise ValueError(
                f"message must have shape [B, M, {self.message_width}]"
            )
        if message.shape[0] != receiver_hidden.shape[0]:
            raise ValueError("message and receiver batch sizes differ")
        if gate_mode not in {"contextual", "fixed_open"}:
            raise ValueError("gate_mode must be contextual or fixed_open")
        active_rows = (
            torch.ones(
                receiver_hidden.shape[0],
                dtype=torch.bool,
                device=receiver_hidden.device,
            )
            if message_attention_mask is None
            else message_attention_mask.to(dtype=torch.bool).any(dim=1)
        )
        if not bool(active_rows.any()):
            self.last_gate = torch.zeros(
                (*receiver_hidden.shape[:2], 1),
                dtype=receiver_hidden.dtype,
                device=receiver_hidden.device,
            )
            return receiver_hidden

        active_indices = active_rows.nonzero(as_tuple=False).flatten()
        selected_hidden = receiver_hidden.index_select(0, active_indices)
        selected_message = message.index_select(0, active_indices)
        selected_mask = (
            None
            if message_attention_mask is None
            else message_attention_mask.index_select(0, active_indices)
        )
        query = self.query_projection(self.receiver_norm(selected_hidden))
        memory = self.message_norm(selected_message)
        padding_mask = (
            None if selected_mask is None else ~selected_mask.to(dtype=torch.bool)
        )
        context, _ = self.attention(
            query,
            memory,
            memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        pooled_message = _masked_mean(memory, selected_mask)
        pooled_message = pooled_message.unsqueeze(1).expand(-1, query.shape[1], -1)
        contextual_gate = torch.sigmoid(
            self.gate_network(torch.cat((query, pooled_message), dim=-1))
        )
        gate = torch.ones_like(contextual_gate) if gate_mode == "fixed_open" else contextual_gate
        delta = self.output_projection(context)
        selected_output = selected_hidden + gate * delta
        if bool(active_rows.all()):
            self.last_gate = gate.detach()
            return selected_output

        output = receiver_hidden.clone()
        output.index_copy_(0, active_indices, selected_output)
        full_gate = torch.zeros(
            (*receiver_hidden.shape[:2], 1),
            dtype=gate.dtype,
            device=gate.device,
        )
        full_gate.index_copy_(0, active_indices, gate.detach())
        self.last_gate = full_gate
        return output
