from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridges import BridgeOutput, ContextualMessageBridge, GatedCrossReceiver
from .gpt_receiver import FrozenCausalLMTower
from .math_tower import MathTower, MathTowerOutput
from .tokenizer import ByteMathTokenizer, pad_1d


def _permute_messages(
    message: torch.Tensor,
    *,
    shuffle: bool,
    permutation: list[int] | None,
) -> torch.Tensor:
    if shuffle and permutation is not None:
        raise ValueError("message shuffle and explicit permutation are mutually exclusive")
    if permutation is not None:
        if len(permutation) != message.shape[0] or sorted(permutation) != list(
            range(message.shape[0])
        ):
            raise ValueError("message permutation must contain every batch index once")
        indices = torch.tensor(permutation, dtype=torch.long, device=message.device)
        return message.index_select(0, indices)
    if shuffle and message.shape[0] > 1:
        return message.roll(1, dims=0)
    return message


def _row_summary(values: torch.Tensor, row: int) -> dict[str, float]:
    selected = values[row].detach().float()
    return {
        "mean": float(selected.mean()),
        "std": float(selected.std(unbiased=False)),
        "minimum": float(selected.min()),
        "maximum": float(selected.max()),
    }


def causal_language_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("language logits and labels have incompatible shapes")
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def answer_weighted_causal_language_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_labels: torch.Tensor,
    *,
    answer_weight: float,
) -> torch.Tensor:
    """Token-normalized LM loss with extra weight on the typed answer payload."""

    if logits.shape[:2] != labels.shape or labels.shape != answer_labels.shape:
        raise ValueError("weighted language logits and labels have incompatible shapes")
    weight = float(answer_weight)
    if weight < 1.0:
        raise ValueError("answer token weight must be at least one")
    shifted_labels = labels[:, 1:].contiguous().view(-1)
    shifted_focus = answer_labels[:, 1:].contiguous().view(-1).ne(-100)
    valid = shifted_labels.ne(-100)
    losses = F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        shifted_labels,
        ignore_index=-100,
        reduction="none",
    )
    weights = valid.to(losses.dtype)
    weights = weights + shifted_focus.to(losses.dtype) * (weight - 1.0)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def optional_answer_loss(
    answer_logits: torch.Tensor, answer_classes: torch.Tensor
) -> torch.Tensor:
    """Cross entropy for integer answers, or a differentiable zero otherwise."""

    valid = answer_classes.ne(-100)
    if not bool(valid.any()):
        return answer_logits.sum() * 0.0
    return F.cross_entropy(
        answer_logits[valid],
        answer_classes[valid],
    )


@dataclass
class CFTNTextOutput:
    loss: torch.Tensor
    math_loss: torch.Tensor
    gpt_loss: torch.Tensor
    answer_loss: torch.Tensor
    math_output: MathTowerOutput
    gpt_logits: torch.Tensor
    gpt_to_math: BridgeOutput
    math_to_gpt: BridgeOutput
    math_receiver_gates: dict[str, torch.Tensor]
    gpt_receiver_gates: dict[str, torch.Tensor]


class CFTNTextModel(nn.Module):
    """Persistent two-tower model with gated context exchange and no routing."""

    def __init__(
        self,
        math_tower: MathTower,
        gpt_tower: FrozenCausalLMTower,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.math_tower = math_tower
        self.gpt_tower = gpt_tower
        self.config = config
        bridge = config["bridge"]
        self.gpt_to_math = ContextualMessageBridge(
            sender_width=gpt_tower.hidden_size,
            message_width=int(bridge["message_width"]),
            message_tokens=int(bridge["message_tokens"]),
            heads=int(bridge["attention_heads"]),
            gate_hidden_size=int(bridge["gate_hidden_size"]),
            dropout=float(bridge["dropout"]),
            gate_init=float(bridge["gate_init"]),
            # Receiver residuals are zero-initialized.  Keeping the translated
            # message nonzero avoids a dead path where both ends have zero
            # Jacobians on the first optimization step.
            zero_init_output=False,
        )
        self.math_to_gpt = ContextualMessageBridge(
            sender_width=math_tower.hidden_size,
            message_width=int(bridge["message_width"]),
            message_tokens=int(bridge["message_tokens"]),
            heads=int(bridge["attention_heads"]),
            gate_hidden_size=int(bridge["gate_hidden_size"]),
            dropout=float(bridge["dropout"]),
            gate_init=float(bridge["gate_init"]),
            zero_init_output=False,
        )
        self.math_receivers = nn.ModuleDict(
            {
                str(layer): GatedCrossReceiver(
                    receiver_width=math_tower.hidden_size,
                    message_width=int(bridge["message_width"]),
                    heads=int(bridge["attention_heads"]),
                    gate_hidden_size=int(bridge["gate_hidden_size"]),
                    dropout=float(bridge["dropout"]),
                    gate_init=float(bridge["gate_init"]),
                    zero_init_output=bool(bridge["zero_init_output"]),
                )
                for layer in config["math_tower"]["receiver_layers"]
            }
        )
        layer_count = len(math_tower.blocks)
        if any(int(layer) < 0 or int(layer) >= layer_count for layer in self.math_receivers):
            raise ValueError("math receiver layer is outside the tower")
        self.gate_mode = "contextual"

    def set_gate_mode(self, mode: str) -> None:
        if mode not in {"contextual", "fixed_open"}:
            raise ValueError("gate mode must be contextual or fixed_open")
        self.gate_mode = mode

    def reset_execution_counts(self) -> None:
        self.math_tower.reset_execution_count()
        self.gpt_tower.reset_execution_count()
        self.gpt_to_math.reset_execution_count()
        self.math_to_gpt.reset_execution_count()
        for receiver in self.math_receivers.values():
            receiver.reset_execution_count()

    def execution_counts(self) -> dict[str, int]:
        return {
            "math_tower": self.math_tower.execution_count,
            "gpt_prepass": self.gpt_tower.prepass_execution_count,
            "gpt_receiver": self.gpt_tower.receiver_execution_count,
            "gpt_to_math_bridge": self.gpt_to_math.execution_count,
            "math_to_gpt_bridge": self.math_to_gpt.execution_count,
        }

    def set_trainable_stage(self, stage: str) -> None:
        if stage not in {"math", "m2g", "bidirectional"}:
            raise ValueError("stage must be math, m2g, or bidirectional")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "math":
            for parameter in self.math_tower.parameters():
                parameter.requires_grad_(True)
            return
        for parameter in self.math_to_gpt.parameters():
            parameter.requires_grad_(True)
        for parameter in self.gpt_tower.receivers.parameters():
            parameter.requires_grad_(True)
        if stage == "bidirectional":
            for parameter in self.gpt_to_math.parameters():
                parameter.requires_grad_(True)
            for parameter in self.math_receivers.parameters():
                parameter.requires_grad_(True)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in names
        }

    def load_trainable_state_dict(
        self, state: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        current = self.state_dict()
        unexpected = sorted(set(state).difference(current))
        missing = sorted(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name not in state
        )
        if strict and (unexpected or missing):
            raise ValueError(
                f"trainable state mismatch; missing={missing}, unexpected={unexpected}"
            )
        current.update({name: value for name, value in state.items() if name in current})
        self.load_state_dict(current, strict=True)

    def forward(
        self,
        batch: dict[str, Any],
        *,
        gpt_to_math_enabled: bool = True,
        math_to_gpt_enabled: bool = True,
        shuffle_gpt_to_math: bool = False,
        shuffle_math_to_gpt: bool = False,
        math_loss_weight: float = 1.0,
        gpt_loss_weight: float = 1.0,
        answer_head_weight: float = 0.25,
    ) -> CFTNTextOutput:
        gpt_hidden = self.gpt_tower.prepass(
            batch["gpt_prepass_input_ids"], batch["gpt_prepass_attention_mask"]
        )
        gpt_to_math = self.gpt_to_math(
            gpt_hidden,
            batch["gpt_prepass_attention_mask"],
            enabled=gpt_to_math_enabled,
            gate_mode=self.gate_mode,
        )
        math_message = gpt_to_math.message
        if shuffle_gpt_to_math and math_message.shape[0] > 1:
            math_message = math_message.roll(1, dims=0)
        math_output = self.math_tower(
            batch["math_input_ids"],
            batch["math_attention_mask"],
            batch["math_prefix_lengths"],
            message=math_message,
            receivers=self.math_receivers,
            receive_enabled=gpt_to_math_enabled,
            gate_mode=self.gate_mode,
        )
        math_to_gpt = self.math_to_gpt(
            math_output.hidden_states,
            batch["math_attention_mask"],
            enabled=math_to_gpt_enabled,
            gate_mode=self.gate_mode,
        )
        gpt_message = math_to_gpt.message
        if shuffle_math_to_gpt and gpt_message.shape[0] > 1:
            gpt_message = gpt_message.roll(1, dims=0)
        message_mask = torch.ones(
            gpt_message.shape[:2], dtype=torch.long, device=gpt_message.device
        )
        gpt_output = self.gpt_tower(
            batch["gpt_input_ids"],
            batch["gpt_attention_mask"],
            message=gpt_message,
            message_mask=message_mask,
            receive_enabled=math_to_gpt_enabled,
            gate_mode=self.gate_mode,
        )
        math_loss = causal_language_loss(math_output.logits, batch["math_labels"])
        gpt_loss = causal_language_loss(gpt_output.logits, batch["gpt_labels"])
        answer_classes = self.math_tower.answer_classes(batch["answer_values"])
        answer_loss = optional_answer_loss(
            math_output.answer_logits, answer_classes
        )
        total = (
            float(math_loss_weight) * math_loss
            + float(gpt_loss_weight) * gpt_loss
            + float(answer_head_weight) * answer_loss
        )
        return CFTNTextOutput(
            loss=total,
            math_loss=math_loss,
            gpt_loss=gpt_loss,
            answer_loss=answer_loss,
            math_output=math_output,
            gpt_logits=gpt_output.logits,
            gpt_to_math=gpt_to_math,
            math_to_gpt=math_to_gpt,
            math_receiver_gates={
                layer: receiver.last_gate.clone()
                for layer, receiver in self.math_receivers.items()
                if receiver.last_gate is not None
            },
            gpt_receiver_gates={
                layer: receiver.last_gate.clone()
                for layer, receiver in self.gpt_tower.receivers.items()
                if receiver.last_gate is not None
            },
        )

    @torch.no_grad()
    def generate_problems(
        self,
        problems: list[str],
        math_tokenizer: ByteMathTokenizer,
        gpt_tokenizer: Any,
        *,
        max_math_new_tokens: int,
        max_gpt_new_tokens: int,
        gpt_to_math_enabled: bool = True,
        math_to_gpt_enabled: bool = True,
        shuffle_gpt_to_math: bool = False,
        shuffle_math_to_gpt: bool = False,
        gpt_to_math_permutation: list[int] | None = None,
        math_to_gpt_permutation: list[int] | None = None,
        gpt_problems: list[str] | None = None,
        math_problems: list[str] | None = None,
        generic_answer: bool = False,
    ) -> list[dict[str, Any]]:
        if not problems:
            return []
        gpt_views = list(gpt_problems if gpt_problems is not None else problems)
        math_views = list(math_problems if math_problems is not None else problems)
        if len(gpt_views) != len(problems) or len(math_views) != len(problems):
            raise ValueError("shared and private problem-view counts differ")
        device = next(self.parameters()).device
        prompts = [
            (
                f"Problem: {problem}\n"
                + (
                    "Return only the exact result in <answer> tags.\nAnswer:"
                    if generic_answer
                    else "Return only the integer in <answer> tags.\nAnswer:"
                )
            )
            for problem in gpt_views
        ]
        gpt_prefixes = [
            list(gpt_tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ]
        gpt_pad = gpt_tokenizer.pad_token_id
        if gpt_pad is None:
            gpt_pad = gpt_tokenizer.eos_token_id
        prepass_ids, prepass_mask = pad_1d(gpt_prefixes, int(gpt_pad))
        prepass_ids = prepass_ids.to(device)
        prepass_mask = prepass_mask.to(device)
        gpt_hidden = self.gpt_tower.prepass(prepass_ids, prepass_mask)
        gpt_to_math = self.gpt_to_math(
            gpt_hidden,
            prepass_mask,
            enabled=gpt_to_math_enabled,
            gate_mode=self.gate_mode,
        )
        math_message = _permute_messages(
            gpt_to_math.message,
            shuffle=shuffle_gpt_to_math,
            permutation=gpt_to_math_permutation,
        )
        sequences = [
            math_tokenizer.encode_generation_prefix(
                problem, self.math_tower.max_sequence_length
            )
            for problem in math_views
        ]
        prefix_lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)
        finished = [False] * len(sequences)
        for _ in range(max_math_new_tokens):
            ids, mask = pad_1d(sequences, math_tokenizer.pad_token_id)
            ids = ids.to(device)
            mask = mask.to(device)
            output = self.math_tower(
                ids,
                mask,
                prefix_lengths,
                message=math_message,
                receivers=self.math_receivers,
                receive_enabled=gpt_to_math_enabled,
                gate_mode=self.gate_mode,
            )
            lengths = mask.sum(dim=1) - 1
            next_tokens = output.logits[
                torch.arange(len(sequences), device=device), lengths
            ].argmax(dim=-1)
            for row, token in enumerate(next_tokens.tolist()):
                if finished[row]:
                    continue
                if len(sequences[row]) >= self.math_tower.max_sequence_length:
                    finished[row] = True
                    continue
                sequences[row].append(int(token))
                if token == math_tokenizer.eos_token_id:
                    finished[row] = True
            if all(finished):
                break
        math_ids, math_mask = pad_1d(sequences, math_tokenizer.pad_token_id)
        math_ids = math_ids.to(device)
        math_mask = math_mask.to(device)
        math_output = self.math_tower(
            math_ids,
            math_mask,
            prefix_lengths,
            message=math_message,
            receivers=self.math_receivers,
            receive_enabled=gpt_to_math_enabled,
            gate_mode=self.gate_mode,
        )
        math_to_gpt = self.math_to_gpt(
            math_output.hidden_states,
            math_mask,
            enabled=math_to_gpt_enabled,
            gate_mode=self.gate_mode,
        )
        gpt_message = _permute_messages(
            math_to_gpt.message,
            shuffle=shuffle_math_to_gpt,
            permutation=math_to_gpt_permutation,
        )
        gpt_generated = self.gpt_tower.generate_greedy(
            gpt_prefixes,
            gpt_message,
            int(gpt_tokenizer.eos_token_id),
            max_gpt_new_tokens,
            receive_enabled=math_to_gpt_enabled,
            gate_mode=self.gate_mode,
        )
        results: list[dict[str, Any]] = []
        for row in range(len(problems)):
            math_new = sequences[row][int(prefix_lengths[row].item()) :]
            results.append(
                {
                    "problem": problems[row],
                    "gpt_problem": gpt_views[row],
                    "math_problem": math_views[row],
                    "math_generation": math_tokenizer.decode(math_new),
                    "gpt_generation": gpt_tokenizer.decode(
                        gpt_generated[row], skip_special_tokens=True
                    ),
                    "communication": {
                        "gpt_to_math_sender_gate": _row_summary(
                            gpt_to_math.gate, row
                        ),
                        "gpt_to_math_attention_entropy": _row_summary(
                            gpt_to_math.attention_entropy, row
                        ),
                        "gpt_to_math_message_norm": float(
                            math_message[row].detach().float().norm(dim=-1).mean()
                        ),
                        "math_to_gpt_sender_gate": _row_summary(
                            math_to_gpt.gate, row
                        ),
                        "math_to_gpt_attention_entropy": _row_summary(
                            math_to_gpt.attention_entropy, row
                        ),
                        "math_to_gpt_message_norm": float(
                            gpt_message[row].detach().float().norm(dim=-1).mean()
                        ),
                    },
                }
            )
        return results
