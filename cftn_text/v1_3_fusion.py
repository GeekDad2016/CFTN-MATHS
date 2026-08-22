from __future__ import annotations

import torch
import torch.nn as nn


class SpecialistAwareMessageFusion(nn.Module):
    """Typed residual fusion over specialist return-message tokens.

    The final projection is zero initialized, so adding this module to a legacy
    checkpoint is an exact identity operation before recovery training.
    """

    def __init__(
        self,
        *,
        message_width: int,
        message_tokens: int,
        specialist_count: int,
        maximum_rounds: int,
        heads: int,
        dropout: float,
        feed_forward_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if message_width < 1 or message_tokens < 1:
            raise ValueError("fusion message dimensions must be positive")
        if specialist_count < 1 or maximum_rounds < 1:
            raise ValueError("fusion routing dimensions must be positive")
        if message_width % heads:
            raise ValueError("fusion message width must be divisible by its heads")
        if feed_forward_multiplier < 1:
            raise ValueError("fusion feed-forward multiplier must be positive")
        self.message_width = int(message_width)
        self.message_tokens = int(message_tokens)
        self.specialist_count = int(specialist_count)
        self.maximum_rounds = int(maximum_rounds)

        self.specialist_embeddings = nn.Embedding(specialist_count, message_width)
        self.round_embeddings = nn.Embedding(maximum_rounds, message_width)
        self.slot_embeddings = nn.Embedding(message_tokens, message_width)
        self.input_norm = nn.LayerNorm(message_width)
        self.attention = nn.MultiheadAttention(
            message_width, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(message_width)
        hidden_width = message_width * int(feed_forward_multiplier)
        self.feed_forward = nn.Sequential(
            nn.Linear(message_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, message_width),
        )
        self.output_norm = nn.LayerNorm(message_width)
        self.output_projection = nn.Linear(message_width, message_width)
        nn.init.normal_(self.specialist_embeddings.weight, std=0.02)
        nn.init.normal_(self.round_embeddings.weight, std=0.02)
        nn.init.normal_(self.slot_embeddings.weight, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        specialist_ids: list[int] = []
        round_ids: list[int] = []
        slot_ids: list[int] = []
        for round_index in range(maximum_rounds):
            for specialist_index in range(specialist_count):
                for slot_index in range(message_tokens):
                    specialist_ids.append(specialist_index)
                    round_ids.append(round_index)
                    slot_ids.append(slot_index)
        self.register_buffer(
            "_specialist_ids", torch.tensor(specialist_ids, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_round_ids", torch.tensor(round_ids, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_slot_ids", torch.tensor(slot_ids, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        messages: torch.Tensor,
        message_mask: torch.Tensor,
        *,
        rounds: int,
    ) -> torch.Tensor:
        if messages.ndim != 3 or messages.shape[-1] != self.message_width:
            raise ValueError(
                f"fusion messages must have shape [B, M, {self.message_width}]"
            )
        if message_mask.shape != messages.shape[:2]:
            raise ValueError("fusion message mask shape differs from messages")
        if rounds < 1 or rounds > self.maximum_rounds:
            raise ValueError("fusion rounds are outside the configured runtime")
        expected_tokens = (
            int(rounds) * self.specialist_count * self.message_tokens
        )
        if messages.shape[1] != expected_tokens:
            raise ValueError(
                f"fusion expected {expected_tokens} tokens for {rounds} rounds, "
                f"received {messages.shape[1]}"
            )
        active_mask = message_mask.to(dtype=torch.bool)
        active_rows = active_mask.any(dim=1)
        if not bool(active_rows.any()):
            return messages

        indices = active_rows.nonzero(as_tuple=False).flatten()
        selected_messages = messages.index_select(0, indices)
        selected_mask = active_mask.index_select(0, indices)
        typed = self.input_norm(selected_messages)
        typed = (
            typed
            + self.specialist_embeddings(self._specialist_ids[:expected_tokens])
            + self.round_embeddings(self._round_ids[:expected_tokens])
            + self.slot_embeddings(self._slot_ids[:expected_tokens])
        )
        context, _ = self.attention(
            typed,
            typed,
            typed,
            key_padding_mask=~selected_mask,
            need_weights=False,
        )
        hidden = typed + context
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        delta = self.output_projection(self.output_norm(hidden))
        selected_output = selected_messages + delta * selected_mask.unsqueeze(-1)
        if bool(active_rows.all()):
            return selected_output
        output = messages.clone()
        output.index_copy_(0, indices, selected_output)
        return output

