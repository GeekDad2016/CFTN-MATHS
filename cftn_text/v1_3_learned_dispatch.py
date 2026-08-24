from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .v1_3_dispatch import (
    DISPATCH_INTENTS,
    DispatchError,
    DispatchPlan,
    compile_v1_3_intent,
)


CHECKPOINT_FORMAT = "cftn_text_v1_3_learned_dispatcher_v6"
PAD_BYTE_ID = 256
QUOTED_SPAN_ID = 257
INTEGER_SPAN_ID = 258
LABEL_SPAN_ID = 259
BYTE_VOCABULARY_SIZE = 260
_SOURCE_SPAN_PATTERN = re.compile(
    r"(?P<quoted>'[^']+')|"
    r"(?P<integer>(?<![A-Za-z0-9])[+-]?\d+(?![A-Za-z0-9]))|"
    r"(?P<label_prefix>(?i:\b(?:tag|name|label)\s+(?:is\s+)?))"
    r"(?P<label_value>[a-z]+)"
)


def _canonical_dispatch_tokens(prompt: str) -> list[int]:
    """Hide operand values while retaining exact span shape and language."""

    tokens: list[int] = []
    cursor = 0
    for match in _SOURCE_SPAN_PATTERN.finditer(prompt):
        tokens.extend(prompt[cursor : match.start()].encode("utf-8"))
        if match.group("quoted") is not None:
            tokens.extend((ord("'"), QUOTED_SPAN_ID, ord("'")))
        elif match.group("integer") is not None:
            tokens.append(INTEGER_SPAN_ID)
        else:
            tokens.extend(match.group("label_prefix").encode("utf-8"))
            tokens.append(LABEL_SPAN_ID)
        cursor = match.end()
    tokens.extend(prompt[cursor:].encode("utf-8"))
    return tokens


def encode_dispatch_prompts(
    prompts: Iterable[str], *, maximum_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_prompts = [str(prompt) for prompt in prompts]
    raw_longest = max(
        (len(prompt.encode("utf-8")) for prompt in raw_prompts), default=0
    )
    if raw_longest > int(maximum_length):
        raise DispatchError(
            f"dispatcher prompt has {raw_longest} bytes, exceeding {maximum_length}"
        )
    encoded = [_canonical_dispatch_tokens(prompt) for prompt in raw_prompts]
    if not encoded:
        raise ValueError("cannot encode an empty dispatcher batch")
    longest = max(len(value) for value in encoded)
    if longest > int(maximum_length):
        raise DispatchError(
            f"dispatcher prompt has {longest} bytes, exceeding {maximum_length}"
        )
    width = max(1, longest)
    input_ids = torch.full(
        (len(encoded), width), PAD_BYTE_ID, dtype=torch.long
    )
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for row, values in enumerate(encoded):
        if values:
            input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            attention_mask[row, : len(values)] = True
    return input_ids, attention_mask


class ByteIntentClassifier(nn.Module):
    """Small byte CNN that predicts a finite typed call graph.

    It predicts only intent. Exact operands remain immutable source spans and
    are recovered by the constrained compiler after classification.
    """

    def __init__(
        self,
        *,
        embedding_size: int = 48,
        channels: int = 64,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(embedding_size, channels, *kernels) < 1:
            raise ValueError("dispatcher dimensions must be positive")
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("dispatcher convolution kernels must be odd")
        self.model_config = {
            "embedding_size": int(embedding_size),
            "channels": int(channels),
            "kernels": tuple(int(value) for value in kernels),
            "dropout": float(dropout),
        }
        self.embedding = nn.Embedding(
            BYTE_VOCABULARY_SIZE,
            int(embedding_size),
            padding_idx=PAD_BYTE_ID,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                int(embedding_size),
                int(channels),
                kernel_size=int(kernel),
                padding=int(kernel) // 2,
            )
            for kernel in kernels
        )
        # The learned component still chooses the semantic call graph, but
        # exposing immutable source-shape facts makes that choice robust to
        # paraphrases. These features contain no task labels or operands: they
        # are only counts of quoted spans and integer spans in the raw prompt.
        self.structure_feature_size = 8
        feature_size = len(kernels) * int(channels) * 2 + self.structure_feature_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Dropout(float(dropout)),
            nn.Linear(feature_size, len(DISPATCH_INTENTS)),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("dispatcher inputs must be aligned rank-two tensors")
        mask = attention_mask.to(dtype=torch.bool)
        embedded = self.embedding(input_ids).transpose(1, 2)
        features: list[torch.Tensor] = []
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        for convolution in self.convolutions:
            values = torch.nn.functional.gelu(convolution(embedded))
            masked_max = values.masked_fill(~mask.unsqueeze(1), -1e4).amax(dim=2)
            masked_mean = (values * mask.unsqueeze(1)).sum(dim=2) / denominator
            features.extend((masked_max, masked_mean))
        quote_pairs = (
            ((input_ids == ord("'")).logical_and(mask)).sum(dim=1) // 2
        ).clamp(max=3)
        integer_spans = (
            (input_ids == INTEGER_SPAN_ID).logical_and(mask).sum(dim=1)
        ).clamp(max=3)
        structure = torch.cat(
            (
                torch.nn.functional.one_hot(quote_pairs, num_classes=4),
                torch.nn.functional.one_hot(integer_spans, num_classes=4),
            ),
            dim=1,
        ).to(dtype=embedded.dtype)
        features.append(structure)
        logits = self.classifier(torch.cat(features, dim=1))
        # A predicted call graph is only eligible when the immutable operand
        # shape can satisfy its compiler contract. ``unsupported`` remains a
        # candidate for every shape, so semantic rejection is still learned.
        compatible = torch.zeros_like(logits, dtype=torch.bool)
        compatible[:, DISPATCH_INTENTS.index("unsupported")] = True
        signatures = {
            "pure_language": (0, 0),
            "single_math": (0, 3),
            "string_count": (2, 0),
            "string_reverse": (1, 0),
            "string_index": (1, 1),
            "multi_parallel": (1, 3),
            "string_then_math": (2, 2),
            "math_then_string": (1, 3),
        }
        for intent, (expected_quotes, expected_integers) in signatures.items():
            compatible[:, DISPATCH_INTENTS.index(intent)] = quote_pairs.eq(
                expected_quotes
            ).logical_and(integer_spans.eq(expected_integers))
        return logits.masked_fill(~compatible, -1e4)


class LearnedV13Dispatcher:
    def __init__(
        self,
        model: ByteIntentClassifier,
        *,
        maximum_length: int,
        confidence_threshold: float,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("dispatcher confidence threshold must be in [0, 1]")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.maximum_length = int(maximum_length)
        self.confidence_threshold = float(confidence_threshold)

    @torch.no_grad()
    def predict_intents(
        self, prompts: list[str]
    ) -> list[tuple[str, float]]:
        input_ids, attention_mask = encode_dispatch_prompts(
            prompts, maximum_length=self.maximum_length
        )
        logits = self.model(
            input_ids.to(self.device), attention_mask.to(self.device)
        )
        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, indices = probabilities.max(dim=-1)
        return [
            (DISPATCH_INTENTS[int(index)], float(score))
            for index, score in zip(indices.tolist(), confidence.tolist())
        ]

    def __call__(self, prompt: str) -> DispatchPlan:
        intent, confidence = self.predict_intents([str(prompt)])[0]
        if confidence < self.confidence_threshold:
            raise DispatchError(
                f"learned dispatcher confidence {confidence:.6f} is below "
                f"{self.confidence_threshold:.6f}"
            )
        return compile_v1_3_intent(str(prompt), intent)


def save_learned_dispatcher_checkpoint(
    path: str | Path,
    model: ByteIntentClassifier,
    *,
    maximum_length: int,
    confidence_threshold: float,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "intents": list(DISPATCH_INTENTS),
            "model_config": model.model_config,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "maximum_length": int(maximum_length),
            "confidence_threshold": float(confidence_threshold),
            "metadata": dict(metadata),
        },
        destination,
    )


def load_learned_dispatcher(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> LearnedV13Dispatcher:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise DispatchError("learned dispatcher checkpoint format is invalid")
    if tuple(checkpoint.get("intents", ())) != DISPATCH_INTENTS:
        raise DispatchError("learned dispatcher intent contract changed")
    model = ByteIntentClassifier(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return LearnedV13Dispatcher(
        model,
        maximum_length=int(checkpoint["maximum_length"]),
        confidence_threshold=float(checkpoint["confidence_threshold"]),
        device=device,
    )
