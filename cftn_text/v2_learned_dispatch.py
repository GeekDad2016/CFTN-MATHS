from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .v2_dispatch import DISPATCH_INTENTS, DispatchError, DispatchPlan, compile_v2_intent


CHECKPOINT_FORMAT = "cftn_text_v2_learned_dispatcher_v1"
PAD_BYTE_ID = 256
QUOTED_SPAN_ID = 257
INTEGER_SPAN_ID = 258
LABEL_SPAN_ID = 259
BYTE_VOCABULARY_SIZE = 260
_SOURCE_SPAN_PATTERN = re.compile(
    r"(?P<quoted>'[^']+')|"
    r"(?P<integer>(?<![A-Za-z0-9])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9]))|"
    r"(?P<label_prefix>(?i:\b(?:tag|name|label)\s+(?:is\s+)?))"
    r"(?P<label_value>[a-z]+)"
)


def _canonical_dispatch_tokens(prompt: str) -> list[int]:
    """Hide values while retaining syntax and immutable source-span shape."""

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
    raw = [str(prompt) for prompt in prompts]
    if not raw:
        raise ValueError("cannot encode an empty V2 dispatcher batch")
    longest_raw = max(len(prompt.encode("utf-8")) for prompt in raw)
    if longest_raw > int(maximum_length):
        raise DispatchError(
            f"V2 dispatcher prompt has {longest_raw} bytes, exceeding {maximum_length}"
        )
    encoded = [_canonical_dispatch_tokens(prompt) for prompt in raw]
    longest = max(len(value) for value in encoded)
    if longest > int(maximum_length):
        raise DispatchError(
            f"canonical V2 dispatcher prompt has {longest} tokens, exceeding {maximum_length}"
        )
    width = max(1, longest)
    input_ids = torch.full((len(encoded), width), PAD_BYTE_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for row, values in enumerate(encoded):
        if values:
            input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            attention_mask[row, : len(values)] = True
    return input_ids, attention_mask


class ByteIntentClassifier(nn.Module):
    """Predict only a finite dispatch graph; never predict operand values."""

    def __init__(
        self,
        *,
        embedding_size: int = 64,
        channels: int = 96,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(embedding_size, channels, *kernels) < 1:
            raise ValueError("V2 dispatcher dimensions must be positive")
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("V2 dispatcher convolution kernels must be odd")
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
            raise ValueError("V2 dispatcher inputs must be aligned rank-two tensors")
        mask = attention_mask.to(dtype=torch.bool)
        embedded = self.embedding(input_ids).transpose(1, 2)
        features: list[torch.Tensor] = []
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        for convolution in self.convolutions:
            values = torch.nn.functional.gelu(convolution(embedded))
            features.append(values.masked_fill(~mask.unsqueeze(1), -1e4).amax(dim=2))
            features.append((values * mask.unsqueeze(1)).sum(dim=2) / denominator)
        quote_pairs = (
            ((input_ids == ord("'")).logical_and(mask)).sum(dim=1) // 2
        ).clamp(max=3)
        number_spans = (
            (input_ids == INTEGER_SPAN_ID).logical_and(mask).sum(dim=1)
        ).clamp(max=3)
        features.append(
            torch.cat(
                (
                    torch.nn.functional.one_hot(quote_pairs, num_classes=4),
                    torch.nn.functional.one_hot(number_spans, num_classes=4),
                ),
                dim=1,
            ).to(dtype=embedded.dtype)
        )
        logits = self.classifier(torch.cat(features, dim=1))

        # Constrain the registered graph signatures. Broad math and
        # unsupported remain semantic candidates for every shape.
        compatible = torch.zeros_like(logits, dtype=torch.bool)
        for name in ("broad_math", "unsupported"):
            compatible[:, DISPATCH_INTENTS.index(name)] = True
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
        for intent, (expected_quotes, expected_numbers) in signatures.items():
            compatible[:, DISPATCH_INTENTS.index(intent)] = quote_pairs.eq(
                expected_quotes
            ).logical_and(number_spans.eq(expected_numbers))
        return logits.masked_fill(~compatible, -1e4)


class LearnedV2Dispatcher:
    def __init__(
        self,
        model: ByteIntentClassifier,
        *,
        maximum_length: int,
        confidence_threshold: float,
        device: torch.device | str = "cpu",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("V2 dispatcher confidence threshold must be in [0, 1]")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.maximum_length = int(maximum_length)
        self.confidence_threshold = float(confidence_threshold)
        self.metadata = dict(metadata or {})

    @torch.no_grad()
    def predict_intents(self, prompts: list[str]) -> list[tuple[str, float]]:
        input_ids, attention_mask = encode_dispatch_prompts(
            prompts, maximum_length=self.maximum_length
        )
        probabilities = torch.softmax(
            self.model(
                input_ids.to(self.device), attention_mask.to(self.device)
            ).float(),
            dim=-1,
        )
        confidence, indices = probabilities.max(dim=-1)
        return [
            (DISPATCH_INTENTS[int(index)], float(score))
            for index, score in zip(indices.tolist(), confidence.tolist())
        ]

    def __call__(self, prompt: str) -> DispatchPlan:
        intent, confidence = self.predict_intents([str(prompt)])[0]
        if confidence < self.confidence_threshold:
            raise DispatchError(
                f"V2 dispatcher confidence {confidence:.6f} is below "
                f"{self.confidence_threshold:.6f}"
            )
        return compile_v2_intent(str(prompt), intent)


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
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "maximum_length": int(maximum_length),
            "confidence_threshold": float(confidence_threshold),
            "metadata": dict(metadata),
        },
        destination,
    )


def load_learned_dispatcher(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> LearnedV2Dispatcher:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise DispatchError("V2 learned dispatcher checkpoint format is invalid")
    if tuple(checkpoint.get("intents", ())) != DISPATCH_INTENTS:
        raise DispatchError("V2 learned dispatcher intent contract changed")
    model = ByteIntentClassifier(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return LearnedV2Dispatcher(
        model,
        maximum_length=int(checkpoint["maximum_length"]),
        confidence_threshold=float(checkpoint["confidence_threshold"]),
        device=device,
        metadata=dict(checkpoint.get("metadata", {})),
    )


__all__ = [
    "ByteIntentClassifier",
    "CHECKPOINT_FORMAT",
    "LearnedV2Dispatcher",
    "encode_dispatch_prompts",
    "load_learned_dispatcher",
    "save_learned_dispatcher_checkpoint",
]
