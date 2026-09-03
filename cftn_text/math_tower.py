from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridges import GatedCrossReceiver


@dataclass
class MathTowerOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    answer_logits: torch.Tensor


@dataclass
class MathTowerKVCache:
    """Preallocated per-layer key/value state for eval-only greedy decoding."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    length: int


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
        self.answer_head_mode = str(config.get("answer_head_mode", "categorical"))
        self.answer_head_enabled = self.answer_head_mode == "categorical"
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
        if not self.answer_head_enabled:
            return torch.full_like(values.to(dtype=torch.long), -100)
        classes = values.to(dtype=torch.long) - self.answer_min
        valid = (values >= self.answer_min) & (values <= self.answer_max)
        return torch.where(valid, classes, torch.full_like(classes, -100))

    def _cached_block_forward(
        self,
        block: nn.TransformerEncoderLayer,
        hidden: torch.Tensor,
        *,
        cached_keys: torch.Tensor,
        cached_values: torch.Tensor,
        cache_length: int,
        causal_prefill: bool,
    ) -> torch.Tensor:
        """Run one frozen encoder block while appending its attention KV state.

        ``TransformerEncoderLayer`` does not expose past key/value inputs.
        This is the equivalent eval-only, norm-first calculation using its
        existing parameters, so it remains compatible with every checkpoint.
        """

        batch, tokens, _ = hidden.shape
        heads = int(block.self_attn.num_heads)
        head_size = self.hidden_size // heads
        normalized = block.norm1(hidden)
        query, key, value = F.linear(
            normalized,
            block.self_attn.in_proj_weight,
            block.self_attn.in_proj_bias,
        ).chunk(3, dim=-1)

        def to_heads(value_tensor: torch.Tensor) -> torch.Tensor:
            return value_tensor.view(batch, tokens, heads, head_size).transpose(1, 2)

        query = to_heads(query)
        key = to_heads(key)
        value = to_heads(value)
        end = cache_length + tokens
        if end > self.max_sequence_length:
            raise ValueError("cached math decoding exceeded max_sequence_length")
        cached_keys[:, :, cache_length:end].copy_(key)
        cached_values[:, :, cache_length:end].copy_(value)
        keys = cached_keys[:, :, :end]
        values = cached_values[:, :, :end]
        attended = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal_prefill,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, self.hidden_size)
        hidden = hidden + block.dropout1(block.self_attn.out_proj(attended))
        feed_forward = block.linear2(block.dropout(block.activation(block.linear1(block.norm2(hidden)))))
        return hidden + block.dropout2(feed_forward)

    def begin_cached_generation(
        self, input_ids: torch.Tensor
    ) -> tuple[MathTowerKVCache, MathTowerOutput]:
        """Prefill equally sized prompts and return reusable greedy state.

        Callers group prompts by prefix length before using this method.  That
        keeps every row unpadded, which is essential because the cache has one
        shared position counter for the active batch.
        """

        if self.training:
            raise RuntimeError("cached generation is eval-only")
        if input_ids.ndim != 2 or input_ids.shape[0] < 1:
            raise ValueError("cached math decoding expects input IDs shaped [B, L]")
        batch, tokens = input_ids.shape
        if not 0 < tokens <= self.max_sequence_length:
            raise ValueError("cached math prompt length is outside the supported range")
        positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch, -1)
        hidden = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        heads = int(self.blocks[0].self_attn.num_heads) if self.blocks else 1
        head_size = self.hidden_size // heads
        cache_shape = (batch, heads, self.max_sequence_length, head_size)
        keys = [hidden.new_empty(cache_shape) for _ in self.blocks]
        values = [hidden.new_empty(cache_shape) for _ in self.blocks]
        for index, block in enumerate(self.blocks):
            hidden = self._cached_block_forward(
                block,
                hidden,
                cached_keys=keys[index],
                cached_values=values[index],
                cache_length=0,
                causal_prefill=True,
            )
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden[:, -1:])
        answer_logits = (
            self.answer_head(hidden[:, -1])
            if self.answer_head_enabled
            else hidden.new_zeros((batch, self.answer_max - self.answer_min + 1))
        )
        return MathTowerKVCache(keys=keys, values=values, length=tokens), MathTowerOutput(
            logits=logits,
            hidden_states=hidden[:, -1:],
            answer_logits=answer_logits,
        )

    def cached_generation_step(
        self, cache: MathTowerKVCache, token_id: int | torch.Tensor
    ) -> MathTowerOutput:
        """Append one token to an eval-only cache and predict the following token."""

        if self.training:
            raise RuntimeError("cached generation is eval-only")
        if cache.length >= self.max_sequence_length:
            raise ValueError("cached math decoding exceeded max_sequence_length")
        device = cache.keys[0].device if cache.keys else self.token_embedding.weight.device
        batch = int(cache.keys[0].shape[0]) if cache.keys else 1
        if torch.is_tensor(token_id):
            input_ids = token_id.to(device=device, dtype=torch.long).reshape(-1, 1)
        else:
            input_ids = torch.full(
                (batch, 1), int(token_id), dtype=torch.long, device=device
            )
        if input_ids.shape[0] != batch:
            raise ValueError("cached generation token batch does not match cache batch")
        positions = torch.full_like(input_ids, cache.length)
        hidden = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        for index, block in enumerate(self.blocks):
            hidden = self._cached_block_forward(
                block,
                hidden,
                cached_keys=cache.keys[index],
                cached_values=cache.values[index],
                cache_length=cache.length,
                causal_prefill=False,
            )
        cache.length += 1
        hidden = self.final_norm(hidden)
        return MathTowerOutput(
            logits=self.lm_head(hidden),
            hidden_states=hidden,
            answer_logits=(
                self.answer_head(hidden[:, 0])
                if self.answer_head_enabled
                else hidden.new_zeros((1, self.answer_max - self.answer_min + 1))
            ),
        )

    def compact_cached_generation(
        self, cache: MathTowerKVCache, active_rows: torch.Tensor | list[int]
    ) -> MathTowerKVCache:
        """Discard finished rows from an eval-only cached-generation batch."""

        if cache.keys:
            device = cache.keys[0].device
            batch = int(cache.keys[0].shape[0])
        else:
            device = self.token_embedding.weight.device
            batch = 1
        rows = torch.as_tensor(active_rows, dtype=torch.long, device=device).reshape(-1)
        if rows.numel() == 0:
            raise ValueError("cached generation compaction requires at least one row")
        if int(rows.min().item()) < 0 or int(rows.max().item()) >= batch:
            raise ValueError("cached generation compaction row is outside the cache batch")
        cache.keys = [value.index_select(0, rows).contiguous() for value in cache.keys]
        cache.values = [value.index_select(0, rows).contiguous() for value in cache.values]
        return cache

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
        answer_logits = (
            self.answer_head(pooled)
            if self.answer_head_enabled
            else pooled.new_zeros(
                (batch, self.answer_max - self.answer_min + 1)
            )
        )
        return MathTowerOutput(
            logits=logits,
            hidden_states=hidden,
            answer_logits=answer_logits,
        )
