from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from .v2_dispatch import DISPATCH_INTENTS, DispatchError, DispatchPlan, compile_v2_intent


CHECKPOINT_FORMAT = "cftn_text_v2_hierarchical_dispatcher_v2"
LEGACY_CHECKPOINT_FORMAT = "cftn_text_v2_learned_dispatcher_v1"
PAD_BYTE_ID = 256
QUOTED_SPAN_ID = 257
INTEGER_SPAN_ID = 258
LABEL_SPAN_ID = 259
BYTE_VOCABULARY_SIZE = 260
DELEGATION_CLASSES = ("generalist", "specialist", "reject")
ROUND_CLASSES = (0, 1, 2)
_SOURCE_SPAN_PATTERN = re.compile(
    r"(?P<quoted>'[^']+')|"
    r"(?P<integer>(?<![A-Za-z0-9])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9]))|"
    r"(?P<label_prefix>(?i:\b(?:tag|name|label)\s+(?:is\s+)?))"
    r"(?P<label_value>[a-z]+)"
)
_INTENT_DELEGATION = {
    "pure_language": "generalist",
    "unsupported": "reject",
    **{
        name: "specialist"
        for name in DISPATCH_INTENTS
        if name not in {"pure_language", "unsupported"}
    },
}
_INTENT_ROUNDS = {
    "pure_language": 0,
    "unsupported": 0,
    "broad_math": 1,
    "single_math": 1,
    "string_count": 1,
    "string_reverse": 1,
    "string_index": 1,
    "multi_parallel": 1,
    "string_then_math": 2,
    "math_then_string": 2,
}
_INTENT_TOWERS = {
    "pure_language": (),
    "unsupported": (),
    "broad_math": ("math",),
    "single_math": ("math",),
    "string_count": ("string",),
    "string_reverse": ("string",),
    "string_index": ("string",),
    "multi_parallel": ("math", "string"),
    "string_then_math": ("math", "string"),
    "math_then_string": ("math", "string"),
}


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


def _compatibility_mask(
    logits: torch.Tensor, quote_pairs: torch.Tensor, number_spans: torch.Tensor
) -> torch.Tensor:
    """Keep source-shape-invalid graph signatures impossible."""

    compatible = torch.zeros_like(logits, dtype=torch.bool)
    # General language may legitimately contain quotes, dates, quantities, and
    # labels. Structural signatures can constrain exact specialist grammars,
    # but must never prevent the frozen coordinator from handling such prompts.
    for name in ("pure_language", "broad_math", "unsupported"):
        compatible[:, DISPATCH_INTENTS.index(name)] = True
    signatures = {
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
    return compatible


class StructuralByteEncoder(nn.Module):
    """Small value-invariant CNN used as a syntax/span safety guard."""

    def __init__(
        self,
        *,
        embedding_size: int = 64,
        channels: int = 96,
        kernels: tuple[int, ...] = (3, 5, 7),
    ) -> None:
        super().__init__()
        if min(embedding_size, channels, *kernels) < 1:
            raise ValueError("V2 dispatcher dimensions must be positive")
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("V2 dispatcher convolution kernels must be odd")
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
        self.output_size = (
            len(kernels) * int(channels) * 2 + self.structure_feature_size
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return torch.cat(features, dim=1), quote_pairs, number_spans


class ByteIntentClassifier(nn.Module):
    """Legacy 116K value-invariant classifier retained for checkpoint reading."""

    architecture = "byte_intent_v1"

    def __init__(
        self,
        *,
        embedding_size: int = 64,
        channels: int = 96,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_config = {
            "embedding_size": int(embedding_size),
            "channels": int(channels),
            "kernels": tuple(int(value) for value in kernels),
            "dropout": float(dropout),
        }
        self.structural_encoder = StructuralByteEncoder(
            embedding_size=embedding_size, channels=channels, kernels=kernels
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.structural_encoder.output_size),
            nn.Dropout(float(dropout)),
            nn.Linear(self.structural_encoder.output_size, len(DISPATCH_INTENTS)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        semantic_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del semantic_features
        features, quote_pairs, number_spans = self.structural_encoder(
            input_ids, attention_mask
        )
        logits = self.classifier(features)
        return logits.masked_fill(
            ~_compatibility_mask(logits, quote_pairs, number_spans), -1e4
        )


class HierarchicalDispatcherModel(nn.Module):
    """~5M planner over frozen Qwen semantics plus structural span features.

    The model predicts a finite intent graph, whether delegation is required,
    which registered towers are needed, and the number of dependency rounds.
    Exact operands never enter these heads: they are copied later by the typed
    compiler from immutable source spans.
    """

    architecture = "qwen_semantic_hierarchical_v2"

    def __init__(
        self,
        *,
        semantic_width: int = 2560,
        semantic_projection_size: int = 1536,
        structure_projection_size: int = 384,
        fusion_size: int = 384,
        tower_names: Sequence[str] = ("math", "string"),
        active_tower_names: Sequence[str] = ("math", "string"),
        embedding_size: int = 64,
        channels: int = 96,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
        hierarchy_weight: float = 0.25,
    ) -> None:
        super().__init__()
        dimensions = (
            semantic_width,
            semantic_projection_size,
            structure_projection_size,
            fusion_size,
        )
        if min(dimensions) < 1:
            raise ValueError("hierarchical dispatcher dimensions must be positive")
        names = tuple(str(value) for value in tower_names)
        active = tuple(str(value) for value in active_tower_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("dispatcher tower names must be ordered and unique")
        if not set(active).issubset(names):
            raise ValueError("active dispatcher towers must be registered")
        if not {"math", "string"}.issubset(active):
            raise ValueError("current V2 graph contract requires active math and string towers")
        if not 0.0 <= float(hierarchy_weight) <= 1.0:
            raise ValueError("hierarchy weight must be in [0, 1]")
        self.tower_names = names
        self.active_tower_names = active
        self.semantic_width = int(semantic_width)
        self.hierarchy_weight = float(hierarchy_weight)
        self.model_config = {
            "semantic_width": int(semantic_width),
            "semantic_projection_size": int(semantic_projection_size),
            "structure_projection_size": int(structure_projection_size),
            "fusion_size": int(fusion_size),
            "tower_names": names,
            "active_tower_names": active,
            "embedding_size": int(embedding_size),
            "channels": int(channels),
            "kernels": tuple(int(value) for value in kernels),
            "dropout": float(dropout),
            "hierarchy_weight": float(hierarchy_weight),
        }
        self.structural_encoder = StructuralByteEncoder(
            embedding_size=embedding_size,
            channels=channels,
            kernels=kernels,
        )
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(int(semantic_width)),
            nn.Linear(int(semantic_width), int(semantic_projection_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.structure_projection = nn.Sequential(
            nn.LayerNorm(self.structural_encoder.output_size),
            nn.Linear(
                self.structural_encoder.output_size,
                int(structure_projection_size),
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(int(semantic_projection_size) + int(structure_projection_size)),
            nn.Linear(
                int(semantic_projection_size) + int(structure_projection_size),
                int(fusion_size),
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.intent_head = nn.Linear(int(fusion_size), len(DISPATCH_INTENTS))
        self.delegation_head = nn.Linear(int(fusion_size), len(DELEGATION_CLASSES))
        self.tower_head = nn.Linear(int(fusion_size), len(self.tower_names))
        self.round_head = nn.Linear(int(fusion_size), len(ROUND_CLASSES))

    def hierarchical_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if semantic_features.ndim != 2:
            raise ValueError("dispatcher semantic features must be rank two")
        if semantic_features.shape != (input_ids.shape[0], self.semantic_width):
            raise ValueError(
                "dispatcher semantic feature shape differs from batch/semantic width"
            )
        structure, quote_pairs, number_spans = self.structural_encoder(
            input_ids, attention_mask
        )
        projection_dtype = self.semantic_projection[1].weight.dtype
        semantic = self.semantic_projection(
            semantic_features.to(device=input_ids.device, dtype=projection_dtype)
        )
        fused = self.fusion(
            torch.cat((semantic, self.structure_projection(structure)), dim=-1)
        )
        raw_intent = self.intent_head(fused)
        raw_intent = raw_intent.masked_fill(
            ~_compatibility_mask(raw_intent, quote_pairs, number_spans), -1e4
        )
        return {
            "intent": raw_intent,
            "delegation": self.delegation_head(fused),
            "towers": self.tower_head(fused),
            "rounds": self.round_head(fused),
        }

    def _hierarchy_scores(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        delegation = torch.log_softmax(values["delegation"], dim=-1)
        rounds = torch.log_softmax(values["rounds"], dim=-1)
        tower_log_on = torch.nn.functional.logsigmoid(values["towers"])
        tower_log_off = torch.nn.functional.logsigmoid(-values["towers"])
        rows: list[torch.Tensor] = []
        active_indices = [self.tower_names.index(name) for name in self.active_tower_names]
        for intent in DISPATCH_INTENTS:
            score = delegation[:, DELEGATION_CLASSES.index(_INTENT_DELEGATION[intent])]
            score = score + rounds[:, ROUND_CLASSES.index(_INTENT_ROUNDS[intent])]
            expected = set(_INTENT_TOWERS[intent])
            for tower_index in active_indices:
                tower_name = self.tower_names[tower_index]
                score = score + (
                    tower_log_on[:, tower_index]
                    if tower_name in expected
                    else tower_log_off[:, tower_index]
                )
            rows.append(score)
        return torch.stack(rows, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        semantic_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if semantic_features is None:
            raise ValueError("hierarchical V2 dispatcher requires frozen semantic features")
        values = self.hierarchical_logits(
            input_ids, attention_mask, semantic_features
        )
        return self.combined_intent_logits(values)

    def combined_intent_logits(
        self, values: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return values["intent"] + self.hierarchy_weight * self._hierarchy_scores(values)


def hierarchy_targets(
    labels: torch.Tensor,
    *,
    tower_names: Sequence[str],
    active_tower_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Derive auxiliary supervision from the finite graph label."""

    names = tuple(str(value) for value in tower_names)
    active = set(str(value) for value in active_tower_names)
    delegation: list[int] = []
    rounds: list[int] = []
    tower_values = torch.zeros((labels.numel(), len(names)), dtype=torch.float32)
    tower_mask = torch.zeros_like(tower_values, dtype=torch.bool)
    for row, label in enumerate(labels.detach().cpu().tolist()):
        intent = DISPATCH_INTENTS[int(label)]
        delegation.append(DELEGATION_CLASSES.index(_INTENT_DELEGATION[intent]))
        rounds.append(ROUND_CLASSES.index(_INTENT_ROUNDS[intent]))
        expected = set(_INTENT_TOWERS[intent])
        for column, name in enumerate(names):
            if name in active:
                tower_values[row, column] = float(name in expected)
                tower_mask[row, column] = True
    return {
        "delegation": torch.tensor(delegation, dtype=torch.long),
        "rounds": torch.tensor(rounds, dtype=torch.long),
        "towers": tower_values,
        "tower_mask": tower_mask,
    }


def dispatcher_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class LearnedV2Dispatcher:
    def __init__(
        self,
        model: ByteIntentClassifier | HierarchicalDispatcherModel,
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
        self.requires_semantic_features = isinstance(model, HierarchicalDispatcherModel)

    @torch.no_grad()
    def predict_intents(
        self,
        prompts: list[str],
        semantic_features: torch.Tensor | None = None,
    ) -> list[tuple[str, float]]:
        input_ids, attention_mask = encode_dispatch_prompts(
            prompts, maximum_length=self.maximum_length
        )
        if self.requires_semantic_features and semantic_features is None:
            raise DispatchError(
                "hierarchical V2 dispatcher requires frozen coordinator features"
            )
        probabilities = torch.softmax(
            self.model(
                input_ids.to(self.device),
                attention_mask.to(self.device),
                None if semantic_features is None else semantic_features.to(self.device),
            ).float(),
            dim=-1,
        )
        confidence, indices = probabilities.max(dim=-1)
        return [
            (DISPATCH_INTENTS[int(index)], float(score))
            for index, score in zip(indices.tolist(), confidence.tolist())
        ]

    def compile_prediction(
        self, prompt: str, prediction: tuple[str, float]
    ) -> DispatchPlan:
        intent, confidence = prediction
        if confidence < self.confidence_threshold:
            raise DispatchError(
                f"V2 dispatcher confidence {confidence:.6f} is below "
                f"{self.confidence_threshold:.6f}"
            )
        return compile_v2_intent(str(prompt), intent)

    def __call__(self, prompt: str) -> DispatchPlan:
        prediction = self.predict_intents([str(prompt)])[0]
        return self.compile_prediction(str(prompt), prediction)


def save_learned_dispatcher_checkpoint(
    path: str | Path,
    model: ByteIntentClassifier | HierarchicalDispatcherModel,
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
            "architecture": model.architecture,
            "intents": list(DISPATCH_INTENTS),
            "model_config": model.model_config,
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "maximum_length": int(maximum_length),
            "confidence_threshold": float(confidence_threshold),
            "parameter_count": dispatcher_parameter_count(model),
            "metadata": dict(metadata),
        },
        destination,
    )


def load_learned_dispatcher(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> LearnedV2Dispatcher:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in {CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT}:
        raise DispatchError("V2 learned dispatcher checkpoint format is invalid")
    if tuple(checkpoint.get("intents", ())) != DISPATCH_INTENTS:
        raise DispatchError("V2 learned dispatcher intent contract changed")
    architecture = checkpoint.get("architecture")
    if checkpoint_format == LEGACY_CHECKPOINT_FORMAT or architecture == "byte_intent_v1":
        model: ByteIntentClassifier | HierarchicalDispatcherModel = ByteIntentClassifier(
            **dict(checkpoint["model_config"])
        )
    elif architecture == "qwen_semantic_hierarchical_v2":
        model = HierarchicalDispatcherModel(**dict(checkpoint["model_config"]))
    else:
        raise DispatchError("V2 learned dispatcher architecture is invalid")
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
    "DELEGATION_CLASSES",
    "HierarchicalDispatcherModel",
    "LearnedV2Dispatcher",
    "ROUND_CLASSES",
    "dispatcher_parameter_count",
    "encode_dispatch_prompts",
    "hierarchy_targets",
    "load_learned_dispatcher",
    "save_learned_dispatcher_checkpoint",
]
