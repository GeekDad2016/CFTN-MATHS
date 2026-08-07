from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from .bridges import GatedCrossReceiver


@dataclass
class MathTowerOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    answer_logits: torch.Tensor


class MathTower(nn.Module):
    """Small causal Transformer specialized for exact symbolic procedures."""

    def __init__(self, config: dict, vocabulary_size: int) -> None:
        super().__init__()
        self.config = dict(config)
        self.vocabulary_size = int(vocabulary_size)
        self.hidden_size = int(config["hidden_size"])
        self.max_sequence_length = int(config["max_sequence_length"])
        self.answer_min = int(config["answer_min"])
        self.answer_max = int(config["answer_max"])
        layers = int(config["layers"])
        heads = int(config["attention_heads"])
        feed_forward = int(config["feed_forward_size"])
        dropout = float(config["dropout"])
        if self.hidden_size % heads:
            raise ValueError("math hidden size must be divisible by attention heads")
        self.token_embedding = nn.Embedding(vocabulary_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_sequence_length, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=heads,
                    dim_feedforward=feed_forward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.answer_head = nn.Linear(
            self.hidden_size, self.answer_max - self.answer_min + 1
        )
        self.apply(self._initialize_weights)
        self.execution_count = 0

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_execution_count(self) -> None:
        self.execution_count = 0

    def answer_classes(self, values: torch.Tensor) -> torch.Tensor:
        classes = values.to(dtype=torch.long) - self.answer_min
        valid = (values >= self.answer_min) & (values <= self.answer_max)
        return torch.where(valid, classes, torch.full_like(classes, -100))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
        *,
        message: torch.Tensor | None = None,
        receivers: Mapping[str, GatedCrossReceiver] | None = None,
        receive_enabled: bool = True,
        gate_mode: str = "contextual",
    ) -> MathTowerOutput:
        self.execution_count += 1
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("math input IDs and attention mask must have shape [B, L]")
        batch, sequence_length = input_ids.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"math sequence length {sequence_length} exceeds "
                f"{self.max_sequence_length}"
            )
        if prefix_lengths.shape != (batch,):
            raise ValueError("prefix lengths must have shape [B]")
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        hidden = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        padding_mask = ~attention_mask.to(dtype=torch.bool)
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
            if receivers is not None and str(index) in receivers:
                hidden = receivers[str(index)](
                    hidden,
                    message,
                    enabled=receive_enabled,
                    gate_mode=gate_mode,
                )
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        prefix_indices = (prefix_lengths.to(input_ids.device) - 1).clamp(
            min=0, max=sequence_length - 1
        )
        pooled = hidden[torch.arange(batch, device=input_ids.device), prefix_indices]
        return MathTowerOutput(
            logits=logits,
            hidden_states=hidden,
            answer_logits=self.answer_head(pooled),
        )
