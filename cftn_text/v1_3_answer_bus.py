from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from .tokenizer import ByteMathTokenizer


ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"


def extract_answer_payload(text: str, *, strict: bool = True) -> str | None:
    """Extract the final registered answer span from a specialist trace.

    Registered training targets use exactly one answer span. Native generation
    can occasionally contain a malformed prefix, so non-strict extraction uses
    the last complete span and lets the caller fall back when none is present.
    """

    value = str(text)
    starts: list[int] = []
    offset = 0
    while True:
        index = value.find(ANSWER_OPEN, offset)
        if index < 0:
            break
        starts.append(index)
        offset = index + len(ANSWER_OPEN)
    if strict and len(starts) != 1:
        raise ValueError("specialist target must contain exactly one <answer> span")
    if not starts:
        return None
    start = starts[-1] + len(ANSWER_OPEN)
    end = value.find(ANSWER_CLOSE, start)
    if end < 0:
        if strict:
            raise ValueError("specialist target has no closing </answer> tag")
        return None
    payload = value[start:end]
    if strict and (not payload or ANSWER_OPEN in payload or ANSWER_CLOSE in payload):
        raise ValueError("specialist target contains an invalid answer payload")
    return payload if payload else None


def registered_answer_bus(
    record: Mapping[str, Any],
) -> list[tuple[int, str, str]]:
    """Return lossless `(round, specialist, payload)` entries from a joint record."""

    required_by_round = record.get("required_specialists_by_round")
    targets_by_round = record.get("specialist_targets_by_round")
    if not isinstance(required_by_round, list) or not isinstance(targets_by_round, Mapping):
        raise ValueError("joint record has no registered specialist answer bus")
    entries: list[tuple[int, str, str]] = []
    for round_index, required in enumerate(required_by_round):
        if not isinstance(required, list):
            raise ValueError("joint record has an invalid required-specialist round")
        for specialist in required:
            targets = targets_by_round.get(specialist)
            if not isinstance(targets, list) or round_index >= len(targets):
                raise ValueError("joint record has an invalid specialist target layout")
            target = targets[round_index]
            if target is None:
                raise ValueError("required specialist has no registered target")
            payload = extract_answer_payload(str(target), strict=True)
            if payload is None:
                raise ValueError("required specialist has no registered answer payload")
            entries.append((round_index, str(specialist), payload))
    return entries


def compose_registered_answer(record: Mapping[str, Any]) -> str | None:
    """Deterministic upper-bound composer for the registered V1.3 task grammar."""

    entries = registered_answer_bus(record)
    task_class = str(record.get("task_class", ""))
    if task_class == "pure_language":
        if entries:
            raise ValueError("pure-language record unexpectedly requires a specialist")
        return None
    if task_class == "multi_parallel":
        first_round = {
            specialist: payload
            for round_index, specialist, payload in entries
            if round_index == 0
        }
        if set(first_round) != {"math", "string"}:
            raise ValueError("parallel record does not expose math and string answers")
        return f"{first_round['math']}|{first_round['string']}"
    if not entries:
        raise ValueError("specialist task has an empty registered answer bus")
    if task_class == "multi_sequential":
        final_round = max(round_index for round_index, _, _ in entries)
        final_payloads = [
            payload
            for round_index, _, payload in entries
            if round_index == final_round
        ]
        if len(final_payloads) != 1:
            raise ValueError("sequential record has an ambiguous final answer")
        return final_payloads[0]
    if len(entries) != 1:
        raise ValueError("single-specialist record has an ambiguous answer bus")
    return entries[0][2]


@dataclass
class AnswerComposerResult:
    log_probabilities: torch.Tensor
    copy_probabilities: torch.Tensor


class TypedAnswerComposer(nn.Module):
    """Autoregressive byte decoder with a typed pointer-copy answer bus.

    The semantic callosal bridge remains responsible for latent collaboration.
    This module supplies the missing lossless path for exact specialist results:
    byte tokens retain specialist, round, and within-answer position identities,
    while the decoder can either copy a bus byte or generate protocol bytes.
    """

    def __init__(
        self,
        *,
        prompt_width: int,
        hidden_size: int,
        specialist_count: int,
        maximum_rounds: int,
        attention_heads: int,
        decoder_layers: int,
        dropout: float,
        maximum_source_positions: int,
        maximum_target_positions: int,
    ) -> None:
        super().__init__()
        if min(
            prompt_width,
            hidden_size,
            specialist_count,
            maximum_rounds,
            attention_heads,
            decoder_layers,
            maximum_source_positions,
            maximum_target_positions,
        ) < 1:
            raise ValueError("answer-composer dimensions must be positive")
        if hidden_size % attention_heads:
            raise ValueError("answer-composer width must divide its attention heads")
        self.hidden_size = int(hidden_size)
        self.specialist_count = int(specialist_count)
        self.maximum_rounds = int(maximum_rounds)
        self.maximum_source_positions = int(maximum_source_positions)
        self.maximum_target_positions = int(maximum_target_positions)
        self.vocabulary_size = int(ByteMathTokenizer.vocab_size)

        self.token_embedding = nn.Embedding(self.vocabulary_size, hidden_size)
        self.specialist_embedding = nn.Embedding(specialist_count, hidden_size)
        self.round_embedding = nn.Embedding(maximum_rounds, hidden_size)
        self.source_position_embedding = nn.Embedding(
            maximum_source_positions, hidden_size
        )
        self.target_position_embedding = nn.Embedding(
            maximum_target_positions, hidden_size
        )
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(prompt_width),
            nn.Linear(prompt_width, hidden_size),
        )
        self.prompt_type = nn.Parameter(torch.zeros(hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        self.vocabulary_projection = nn.Linear(hidden_size, self.vocabulary_size)
        self.pointer_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pointer_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.copy_gate = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.copy_gate.bias, 1.0)
        nn.init.normal_(self.prompt_type, std=0.02)

    def _validate_source(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        specialist_ids: torch.Tensor,
        round_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        tensors = (
            attention_mask,
            specialist_ids,
            round_ids,
            position_ids,
        )
        if token_ids.ndim != 2 or any(value.shape != token_ids.shape for value in tensors):
            raise ValueError("answer-bus tensors must share shape [B, S]")
        if token_ids.numel() and (
            int(token_ids.min()) < 0
            or int(token_ids.max()) >= self.vocabulary_size
        ):
            raise ValueError("answer-bus token ID is outside the byte vocabulary")
        active = attention_mask.to(dtype=torch.bool)
        if bool(active.any()):
            if int(specialist_ids[active].min()) < 0 or int(
                specialist_ids[active].max()
            ) >= self.specialist_count:
                raise ValueError("answer-bus specialist ID is outside the runtime")
            if int(round_ids[active].min()) < 0 or int(
                round_ids[active].max()
            ) >= self.maximum_rounds:
                raise ValueError("answer-bus round ID is outside the runtime")
            if int(position_ids[active].min()) < 0 or int(
                position_ids[active].max()
            ) >= self.maximum_source_positions:
                raise ValueError("answer-bus source position exceeds its embedding")

    def _source_memory(
        self,
        prompt_context: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        specialist_ids: torch.Tensor,
        round_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._validate_source(
            token_ids,
            attention_mask,
            specialist_ids,
            round_ids,
            position_ids,
        )
        if prompt_context.ndim != 2 or prompt_context.shape[0] != token_ids.shape[0]:
            raise ValueError("answer-composer prompt context must have shape [B, H]")
        source_mask = attention_mask.to(dtype=torch.bool)
        if token_ids.shape[1] == 0:
            # Transformer and pointer attention both require at least one source
            # slot. It stays masked and is used only by ineligible fallback rows.
            shape = (token_ids.shape[0], 1)
            token_ids = token_ids.new_zeros(shape)
            specialist_ids = specialist_ids.new_zeros(shape)
            round_ids = round_ids.new_zeros(shape)
            position_ids = position_ids.new_zeros(shape)
            source_mask = torch.zeros(shape, dtype=torch.bool, device=token_ids.device)
        source = (
            self.token_embedding(token_ids)
            + self.specialist_embedding(specialist_ids.clamp_min(0))
            + self.round_embedding(round_ids.clamp_min(0))
            + self.source_position_embedding(position_ids.clamp_min(0))
        )
        prompt = self.prompt_projection(prompt_context).unsqueeze(1)
        prompt = prompt + self.prompt_type.view(1, 1, -1)
        memory = torch.cat((prompt, source), dim=1)
        memory_mask = torch.cat(
            (
                torch.ones(
                    (source.shape[0], 1),
                    dtype=torch.bool,
                    device=source.device,
                ),
                source_mask,
            ),
            dim=1,
        )
        return memory, memory_mask, source, source_mask, token_ids

    def forward(
        self,
        *,
        prompt_context: torch.Tensor,
        source_token_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_specialist_ids: torch.Tensor,
        source_round_ids: torch.Tensor,
        source_position_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> AnswerComposerResult:
        if decoder_input_ids.ndim != 2:
            raise ValueError("answer-composer decoder inputs must have shape [B, T]")
        if decoder_input_ids.shape[0] != source_token_ids.shape[0]:
            raise ValueError("answer-composer source and target batch sizes differ")
        if decoder_input_ids.shape[1] > self.maximum_target_positions:
            raise ValueError("answer-composer target exceeds its positional embedding")
        memory, memory_mask, source, source_mask, normalized_source_token_ids = (
            self._source_memory(
            prompt_context,
            source_token_ids,
            source_attention_mask,
            source_specialist_ids,
            source_round_ids,
            source_position_ids,
            )
        )
        target_positions = torch.arange(
            decoder_input_ids.shape[1], device=decoder_input_ids.device
        )
        target = self.token_embedding(decoder_input_ids) + self.target_position_embedding(
            target_positions
        )
        causal_mask = torch.triu(
            torch.ones(
                decoder_input_ids.shape[1],
                decoder_input_ids.shape[1],
                dtype=torch.bool,
                device=decoder_input_ids.device,
            ),
            diagonal=1,
        )
        target_padding_mask = (
            None
            if decoder_attention_mask is None
            else ~decoder_attention_mask.to(dtype=torch.bool)
        )
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=~memory_mask,
        )
        vocabulary_logits = self.vocabulary_projection(decoded).float()
        # PAD/BOS/SEP are decoder controls, never valid output payloads.
        vocabulary_logits[..., ByteMathTokenizer.pad_token_id] = -1.0e4
        vocabulary_logits[..., ByteMathTokenizer.bos_token_id] = -1.0e4
        vocabulary_logits[..., ByteMathTokenizer.sep_token_id] = -1.0e4
        vocabulary_probabilities = torch.softmax(vocabulary_logits, dim=-1)

        pointer_scores = torch.matmul(
            self.pointer_query(decoded), self.pointer_key(source).transpose(1, 2)
        ).float() / math.sqrt(self.hidden_size)
        safe_source_mask = source_mask.clone()
        empty_rows = ~safe_source_mask.any(dim=1)
        if bool(empty_rows.any()):
            safe_source_mask[empty_rows, 0] = True
        pointer_scores = pointer_scores.masked_fill(
            ~safe_source_mask.unsqueeze(1), -1.0e4
        )
        pointer_attention = torch.softmax(pointer_scores, dim=-1)
        pointer_attention = pointer_attention * (~empty_rows).to(
            pointer_attention.dtype
        ).view(-1, 1, 1)
        pointer_vocabulary = torch.zeros_like(vocabulary_probabilities)
        pointer_vocabulary.scatter_add_(
            2,
            normalized_source_token_ids.unsqueeze(1).expand(
                -1, decoded.shape[1], -1
            ),
            pointer_attention,
        )
        copy_probabilities = torch.sigmoid(self.copy_gate(decoded).float())
        copy_probabilities = copy_probabilities * (~empty_rows).to(
            copy_probabilities.dtype
        ).view(-1, 1, 1)
        probabilities = (
            (1.0 - copy_probabilities) * vocabulary_probabilities
            + copy_probabilities * pointer_vocabulary
        )
        return AnswerComposerResult(
            log_probabilities=torch.log(probabilities.clamp_min(1.0e-9)),
            copy_probabilities=copy_probabilities,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        prompt_context: torch.Tensor,
        source_token_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_specialist_ids: torch.Tensor,
        source_round_ids: torch.Tensor,
        source_position_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> list[list[int]]:
        if max_new_tokens < 1 or max_new_tokens > self.maximum_target_positions:
            raise ValueError("answer-composer generation length is outside its runtime")
        batch_size = int(source_token_ids.shape[0])
        sequences = torch.full(
            (batch_size, 1),
            ByteMathTokenizer.bos_token_id,
            dtype=torch.long,
            device=source_token_ids.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=sequences.device)
        emitted: list[list[int]] = [[] for _ in range(batch_size)]
        for _ in range(int(max_new_tokens)):
            result = self(
                prompt_context=prompt_context,
                source_token_ids=source_token_ids,
                source_attention_mask=source_attention_mask,
                source_specialist_ids=source_specialist_ids,
                source_round_ids=source_round_ids,
                source_position_ids=source_position_ids,
                decoder_input_ids=sequences,
                decoder_attention_mask=torch.ones_like(sequences),
            )
            next_tokens = result.log_probabilities[:, -1].argmax(dim=-1)
            for row, token in enumerate(next_tokens.tolist()):
                if bool(finished[row]):
                    continue
                if int(token) == ByteMathTokenizer.eos_token_id:
                    finished[row] = True
                else:
                    emitted[row].append(int(token))
            sequences = torch.cat((sequences, next_tokens.unsqueeze(1)), dim=1)
            if bool(finished.all()):
                break
        return emitted
