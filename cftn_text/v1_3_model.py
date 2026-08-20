from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridges import BridgeOutput, ContextualMessageBridge, GatedCrossReceiver
from .gpt_receiver import FrozenGPT2Tower
from .math_tower import MathTower, MathTowerOutput
from .model import causal_language_loss


WAKE_MODES = {
    "dense",
    "oracle",
    "soft",
    "hard_straight_through",
    "hard",
}


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _safe_causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not bool(labels[:, 1:].ne(-100).any()):
        return logits.sum() * 0.0
    return causal_language_loss(logits, labels)


class IndependentWakeGates(nn.Module):
    """Independent sigmoid decisions; no softmax, top-k, or winning expert."""

    def __init__(self, hidden_size: int, specialist_count: int, gate_hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, specialist_count),
        )
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, pooled_context: torch.Tensor) -> torch.Tensor:
        return self.network(pooled_context)


class HaltGate(nn.Module):
    def __init__(self, hidden_size: int, gate_hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.constant_(self.network[-1].bias, -1.0)

    def forward(self, pooled_context: torch.Tensor) -> torch.Tensor:
        return self.network(pooled_context).squeeze(-1)


@dataclass
class V13RoundOutput:
    wake_logits: torch.Tensor
    wake_probabilities: torch.Tensor
    wake_activations: torch.Tensor
    halt_logits: torch.Tensor
    requests: dict[str, BridgeOutput]
    specialist_outputs: dict[str, MathTowerOutput]
    returns: dict[str, BridgeOutput]


@dataclass
class V13ModelOutput:
    loss: torch.Tensor
    gpt_loss: torch.Tensor
    specialist_loss: torch.Tensor
    wake_loss: torch.Tensor
    halt_loss: torch.Tensor
    compute_loss: torch.Tensor
    gpt_logits: torch.Tensor
    rounds: list[V13RoundOutput]


class V13MultiTowerModel(nn.Module):
    """GPT workspace plus independently wakeable persistent specialists."""

    def __init__(
        self,
        *,
        gpt_tower: FrozenGPT2Tower,
        specialists: Mapping[str, MathTower],
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        configured_specialists = tuple(config["runtime"]["specialist_names"])
        if tuple(specialists) != configured_specialists:
            raise ValueError(
                f"specialists must be ordered as {configured_specialists}"
            )
        self.specialist_names = configured_specialists
        self.gpt_tower = gpt_tower
        self.specialists = nn.ModuleDict(dict(specialists))
        self.config = config
        bridge = config["bridge"]
        message_width = int(bridge["message_width"])
        self.request_bridges = nn.ModuleDict(
            {
                name: ContextualMessageBridge(
                    sender_width=gpt_tower.hidden_size,
                    message_width=message_width,
                    message_tokens=int(bridge["message_tokens"]),
                    heads=int(bridge["attention_heads"]),
                    gate_hidden_size=int(bridge["gate_hidden_size"]),
                    dropout=float(bridge["dropout"]),
                    gate_init=float(bridge["gate_init"]),
                    zero_init_output=False,
                )
                for name in self.specialist_names
            }
        )
        self.return_bridges = nn.ModuleDict(
            {
                name: ContextualMessageBridge(
                    sender_width=self.specialists[name].hidden_size,
                    message_width=message_width,
                    message_tokens=int(bridge["message_tokens"]),
                    heads=int(bridge["attention_heads"]),
                    gate_hidden_size=int(bridge["gate_hidden_size"]),
                    dropout=float(bridge["dropout"]),
                    gate_init=float(bridge["gate_init"]),
                    zero_init_output=False,
                )
                for name in self.specialist_names
            }
        )
        self.specialist_receivers = nn.ModuleDict()
        for name in self.specialist_names:
            tower = self.specialists[name]
            layers = [int(value) for value in tower.config.get("receiver_layers", [])]
            self.specialist_receivers[name] = nn.ModuleDict(
                {
                    str(layer): GatedCrossReceiver(
                        receiver_width=tower.hidden_size,
                        message_width=message_width,
                        heads=int(bridge["attention_heads"]),
                        gate_hidden_size=int(bridge["gate_hidden_size"]),
                        dropout=float(bridge["dropout"]),
                        gate_init=float(bridge["gate_init"]),
                        zero_init_output=bool(bridge["zero_init_output"]),
                    )
                    for layer in layers
                }
            )
        gate_hidden = int(bridge["gate_hidden_size"])
        self.wake_gates = IndependentWakeGates(
            gpt_tower.hidden_size, len(self.specialist_names), gate_hidden
        )
        self.wake_round_embeddings = nn.Embedding(
            int(config["runtime"]["maximum_callosal_rounds"]),
            gpt_tower.hidden_size,
        )
        nn.init.zeros_(self.wake_round_embeddings.weight)
        self.halt_gate = HaltGate(gpt_tower.hidden_size, gate_hidden)
        self.maximum_rounds = int(config["runtime"]["maximum_callosal_rounds"])
        self.wake_threshold = float(config["runtime"]["wake_threshold"])
        self.gate_mode = "contextual"
        self.trainable_phase: str | None = None

    def set_gate_mode(self, mode: str) -> None:
        if mode not in {"contextual", "fixed_open"}:
            raise ValueError("gate mode must be contextual or fixed_open")
        self.gate_mode = mode

    def set_trainable_phase(self, phase: str) -> None:
        allowed = {
            "single_specialist_capacity",
            "dense_mixed_messages",
            "dense_recurrent",
            "supervised_soft_wake",
            "hardened_wake",
            "oracle_hard_adapter_recovery",
            "hard_router_recovery",
        }
        if phase not in allowed:
            raise ValueError(f"unknown V1.3 phase: {phase}")
        self.trainable_phase = phase
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if phase not in {"hardened_wake", "hard_router_recovery"}:
            for modules in (
                self.request_bridges,
                self.return_bridges,
                self.specialist_receivers,
                self.gpt_tower.receivers,
            ):
                for parameter in modules.parameters():
                    parameter.requires_grad_(True)
        if phase == "supervised_soft_wake":
            for modules in (self.wake_gates, self.halt_gate):
                for parameter in modules.parameters():
                    parameter.requires_grad_(True)
        elif phase in {"hardened_wake", "hard_router_recovery"}:
            # Harden only the specialist routing decision.  V1.3 showed that
            # jointly updating the halt gate at the hard threshold can erase a
            # useful soft-wake policy even when the bridges stay frozen.
            for parameter in self.wake_gates.parameters():
                parameter.requires_grad_(True)
            if phase == "hard_router_recovery":
                for parameter in self.wake_round_embeddings.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.gpt_tower.model.eval()
        for tower in self.specialists.values():
            tower.eval()
        if self.trainable_phase in {"hardened_wake", "hard_router_recovery"}:
            for modules in (
                self.request_bridges,
                self.return_bridges,
                self.specialist_receivers,
                self.gpt_tower.receivers,
            ):
                modules.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @staticmethod
    def _collaboration_prefixes() -> tuple[str, ...]:
        return (
            "request_bridges.",
            "return_bridges.",
            "specialist_receivers.",
            "gpt_tower.receivers.",
            "wake_gates.",
            "wake_round_embeddings.",
            "halt_gate.",
        )

    def collaboration_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = self._collaboration_prefixes()
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name.startswith(prefixes)
        }

    def load_collaboration_state_dict(
        self, state: Mapping[str, torch.Tensor], *, strict: bool = True
    ) -> None:
        current = self.state_dict()
        expected = {
            name for name in current if name.startswith(self._collaboration_prefixes())
        }
        optional_legacy = {"wake_round_embeddings.weight"}
        missing = sorted(expected.difference(state).difference(optional_legacy))
        unexpected = sorted(set(state).difference(expected))
        incompatible = sorted(
            name
            for name in expected.intersection(state)
            if current[name].shape != state[name].shape
        )
        if strict and (missing or unexpected or incompatible):
            raise ValueError(
                "V1.3 collaboration state mismatch; "
                f"missing={missing}, unexpected={unexpected}, incompatible={incompatible}"
            )
        current.update(
            {
                name: value
                for name, value in state.items()
                if name in expected and current[name].shape == value.shape
            }
        )
        self.load_state_dict(current, strict=True)

    def load_v1_2_bridge_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Transfer the proven math path and GPT receiver without renaming files."""

        mappings = (
            ("gpt_to_math.", "request_bridges.math."),
            ("math_to_gpt.", "return_bridges.math."),
            ("math_receivers.", "specialist_receivers.math."),
            ("gpt_tower.receivers.", "gpt_tower.receivers."),
        )
        translated: dict[str, torch.Tensor] = {}
        for old_prefix, new_prefix in mappings:
            for name, value in state.items():
                if name.startswith(old_prefix):
                    translated[new_prefix + name[len(old_prefix) :]] = value
        current = self.state_dict()
        missing = sorted(name for name in translated if name not in current)
        incompatible = sorted(
            name
            for name, value in translated.items()
            if name in current and current[name].shape != value.shape
        )
        if missing or incompatible:
            raise ValueError(
                f"V1.2 bridge transfer mismatch; missing={missing}, incompatible={incompatible}"
            )
        expected_prefixes = (
            "request_bridges.math.",
            "return_bridges.math.",
            "specialist_receivers.math.",
            "gpt_tower.receivers.",
        )
        absent = [prefix for prefix in expected_prefixes if not any(name.startswith(prefix) for name in translated)]
        if absent:
            raise ValueError(f"V1.2 bridge checkpoint lacks paths: {absent}")
        current.update(translated)
        self.load_state_dict(current, strict=True)

    def _wake_activation(
        self, logits: torch.Tensor, targets: torch.Tensor, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in WAKE_MODES:
            raise ValueError(f"unsupported wake mode: {mode}")
        probabilities = torch.sigmoid(logits)
        if mode == "dense":
            activation = torch.ones_like(probabilities)
        elif mode == "oracle":
            activation = targets
        elif mode == "soft":
            activation = probabilities
        else:
            hard = probabilities.ge(self.wake_threshold).to(probabilities.dtype)
            activation = (
                hard + probabilities - probabilities.detach()
                if mode == "hard_straight_through"
                else hard
            )
        return probabilities, activation

    def _zero_specialist_output(
        self,
        tower: MathTower,
        batch: dict[str, torch.Tensor],
        *,
        reference: MathTowerOutput | None = None,
    ) -> MathTowerOutput:
        batch_size, length = batch["input_ids"].shape
        device = batch["input_ids"].device
        logits_dtype = (
            reference.logits.dtype
            if reference is not None
            else tower.token_embedding.weight.dtype
        )
        hidden_dtype = (
            reference.hidden_states.dtype
            if reference is not None
            else tower.token_embedding.weight.dtype
        )
        answer_dtype = (
            reference.answer_logits.dtype
            if reference is not None
            else tower.token_embedding.weight.dtype
        )
        return MathTowerOutput(
            logits=torch.zeros(
                batch_size,
                length,
                tower.vocabulary_size,
                device=device,
                dtype=logits_dtype,
            ),
            hidden_states=torch.zeros(
                batch_size,
                length,
                tower.hidden_size,
                device=device,
                dtype=hidden_dtype,
            ),
            answer_logits=torch.zeros(
                batch_size,
                tower.answer_max - tower.answer_min + 1,
                device=device,
                dtype=answer_dtype,
            ),
        )

    def _run_specialist(
        self,
        name: str,
        batch: dict[str, torch.Tensor],
        request_message: torch.Tensor,
        activation: torch.Tensor,
        *,
        conditional_execution: bool,
    ) -> MathTowerOutput:
        tower = self.specialists[name]
        receivers = self.specialist_receivers[name]
        if not conditional_execution:
            return tower(
                batch["input_ids"],
                batch["attention_mask"],
                batch["prefix_lengths"],
                message=request_message,
                receivers=receivers,
                receive_enabled=True,
                gate_mode=self.gate_mode,
            )
        active = activation.detach().ge(self.wake_threshold)
        indices = active.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            return self._zero_specialist_output(tower, batch)
        selected = {
            key: value.index_select(0, indices)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        selected_output = tower(
            selected["input_ids"],
            selected["attention_mask"],
            selected["prefix_lengths"],
            message=request_message.index_select(0, indices),
            receivers=receivers,
            receive_enabled=True,
            gate_mode=self.gate_mode,
        )
        # Autocast can make the selected output BF16 while the frozen tower's
        # parameters remain FP32.  Match the produced tensors before scatter.
        output = self._zero_specialist_output(
            tower, batch, reference=selected_output
        )
        output.logits.index_copy_(0, indices, selected_output.logits)
        output.hidden_states.index_copy_(0, indices, selected_output.hidden_states)
        output.answer_logits.index_copy_(0, indices, selected_output.answer_logits)
        return output

    def forward(
        self,
        batch: dict[str, Any],
        *,
        wake_mode: str,
        maximum_rounds: int | None = None,
        conditional_execution: bool | None = None,
        apply_halt: bool | None = None,
        loss_weights: Mapping[str, float] | None = None,
        disabled_specialists: set[str] | None = None,
        shuffled_requests: set[str] | None = None,
        shuffled_returns: set[str] | None = None,
        disable_all_communication: bool = False,
    ) -> V13ModelOutput:
        rounds_to_run = int(maximum_rounds or self.maximum_rounds)
        if rounds_to_run < 1 or rounds_to_run > self.maximum_rounds:
            raise ValueError("requested callosal rounds are outside the model runtime")
        if conditional_execution is None:
            conditional_execution = bool(
                self.config["runtime"].get(
                    "conditional_execution_in_hard_mode", False
                )
            ) and wake_mode in {"hard", "hard_straight_through"}
        if apply_halt is None:
            # Hard wake and hard halt are separate transitions.  The halt gate
            # remains diagnostic until a later, explicitly calibrated phase.
            apply_halt = bool(self.config["runtime"].get("hard_halt_enabled", False))
        gpt_hidden = self.gpt_tower.prepass(
            batch["gpt_prepass_input_ids"], batch["gpt_prepass_attention_mask"]
        )
        disabled = set(disabled_specialists or ())
        request_shuffle = set(shuffled_requests or ())
        return_shuffle = set(shuffled_returns or ())
        unknown = (disabled | request_shuffle | return_shuffle).difference(
            self.specialist_names
        )
        if unknown:
            raise ValueError(f"unknown specialists in ablation: {sorted(unknown)}")
        accumulated_returns: list[torch.Tensor] = []
        accumulated_return_masks: list[torch.Tensor] = []
        round_outputs: list[V13RoundOutput] = []
        specialist_losses: list[torch.Tensor] = []
        wake_logits_all: list[torch.Tensor] = []
        halt_logits_all: list[torch.Tensor] = []
        activations_all: list[torch.Tensor] = []
        halted = torch.zeros(
            batch["gpt_prepass_input_ids"].shape[0],
            dtype=torch.bool,
            device=gpt_hidden.device,
        )
        halt_runtime = (
            wake_mode in {"hard", "hard_straight_through"} and bool(apply_halt)
        )
        for round_index in range(rounds_to_run):
            pooled = _masked_mean(gpt_hidden, batch["gpt_prepass_attention_mask"])
            round_ids = torch.full(
                (pooled.shape[0],),
                round_index,
                dtype=torch.long,
                device=pooled.device,
            )
            wake_logits = self.wake_gates(
                pooled + self.wake_round_embeddings(round_ids)
            )
            targets = batch["wake_targets"][:, round_index]
            wake_probabilities, wake_activations = self._wake_activation(
                wake_logits, targets, wake_mode
            )
            if disable_all_communication:
                wake_activations = torch.zeros_like(wake_activations)
            if halt_runtime:
                wake_activations = wake_activations * (
                    ~halted
                ).to(wake_activations.dtype).unsqueeze(1)
            requests: dict[str, BridgeOutput] = {}
            specialist_outputs: dict[str, MathTowerOutput] = {}
            returns: dict[str, BridgeOutput] = {}
            round_return_messages: list[torch.Tensor] = []
            round_return_masks: list[torch.Tensor] = []
            for specialist_index, name in enumerate(self.specialist_names):
                activation = wake_activations[:, specialist_index]
                if name in disabled:
                    activation = torch.zeros_like(activation)
                binary_routing = wake_mode in {
                    "oracle",
                    "hard",
                    "hard_straight_through",
                }
                active_rows = (
                    activation.detach().ge(self.wake_threshold)
                    if binary_routing
                    else torch.ones_like(activation, dtype=torch.bool)
                )
                request = self.request_bridges[name](
                    gpt_hidden,
                    batch["gpt_prepass_attention_mask"],
                    enabled=True,
                    gate_mode=self.gate_mode,
                )
                gated_request = request.message * activation[:, None, None]
                if name in request_shuffle and gated_request.shape[0] > 1:
                    gated_request = gated_request.roll(1, dims=0)
                specialist_batch = batch["specialists"][name][round_index]
                specialist_output = self._run_specialist(
                    name,
                    specialist_batch,
                    gated_request,
                    activation,
                    conditional_execution=conditional_execution,
                )
                returned = self.return_bridges[name](
                    specialist_output.hidden_states,
                    specialist_batch["attention_mask"],
                    enabled=True,
                    gate_mode=self.gate_mode,
                )
                gated_return = returned.message * activation[:, None, None]
                if name in return_shuffle and gated_return.shape[0] > 1:
                    gated_return = gated_return.roll(1, dims=0)
                requests[name] = BridgeOutput(
                    message=gated_request,
                    gate=request.gate * activation[:, None, None],
                    attention_entropy=request.attention_entropy,
                )
                returns[name] = BridgeOutput(
                    message=gated_return,
                    gate=returned.gate * activation[:, None, None],
                    attention_entropy=returned.attention_entropy,
                )
                specialist_outputs[name] = specialist_output
                round_return_messages.append(gated_return)
                round_return_masks.append(
                    active_rows.unsqueeze(1).expand(-1, gated_return.shape[1])
                )
                specialist_losses.append(
                    _safe_causal_loss(specialist_output.logits, specialist_batch["labels"])
                )
            accumulated_returns.extend(round_return_messages)
            accumulated_return_masks.extend(round_return_masks)
            combined = torch.cat(accumulated_returns, dim=1)
            message_mask = torch.cat(accumulated_return_masks, dim=1)
            gpt_update = self.gpt_tower(
                batch["gpt_prepass_input_ids"],
                batch["gpt_prepass_attention_mask"],
                message=combined,
                message_mask=message_mask,
                receive_enabled=not disable_all_communication,
                gate_mode=self.gate_mode,
            )
            gpt_hidden = gpt_update.hidden_states[-1]
            halt_logits = self.halt_gate(
                _masked_mean(gpt_hidden, batch["gpt_prepass_attention_mask"])
            )
            if halt_runtime:
                halted |= torch.sigmoid(halt_logits.detach()).ge(0.5)
            round_outputs.append(
                V13RoundOutput(
                    wake_logits=wake_logits,
                    wake_probabilities=wake_probabilities,
                    wake_activations=wake_activations,
                    halt_logits=halt_logits,
                    requests=requests,
                    specialist_outputs=specialist_outputs,
                    returns=returns,
                )
            )
            wake_logits_all.append(wake_logits)
            halt_logits_all.append(halt_logits)
            activations_all.append(wake_activations)
        combined = torch.cat(accumulated_returns, dim=1)
        message_mask = torch.cat(accumulated_return_masks, dim=1)
        gpt_output = self.gpt_tower(
            batch["gpt_input_ids"],
            batch["gpt_attention_mask"],
            message=combined,
            message_mask=message_mask,
            receive_enabled=not disable_all_communication,
            gate_mode=self.gate_mode,
        )
        gpt_loss = _safe_causal_loss(gpt_output.logits, batch["gpt_labels"])
        specialist_loss = torch.stack(specialist_losses).mean()
        stacked_wake_logits = torch.stack(wake_logits_all, dim=1)
        wake_targets = batch["wake_targets"][:, :rounds_to_run]
        reachable_rounds = batch["halt_targets"][:, :rounds_to_run].ge(0)
        wake_strata = []
        for round_index in range(rounds_to_run):
            valid = reachable_rounds[:, round_index]
            if not bool(valid.any()):
                continue
            for specialist_index in range(len(self.specialist_names)):
                wake_strata.append(
                    F.binary_cross_entropy_with_logits(
                        stacked_wake_logits[valid, round_index, specialist_index],
                        wake_targets[valid, round_index, specialist_index],
                    )
                )
        wake_loss = (
            torch.stack(wake_strata).mean()
            if wake_strata
            else stacked_wake_logits.sum() * 0.0
        )
        stacked_halt_logits = torch.stack(halt_logits_all, dim=1)
        halt_targets = batch["halt_targets"][:, :rounds_to_run]
        valid_halt = halt_targets.ge(0)
        halt_loss = (
            F.binary_cross_entropy_with_logits(
                stacked_halt_logits[valid_halt], halt_targets[valid_halt]
            )
            if bool(valid_halt.any())
            else stacked_halt_logits.sum() * 0.0
        )
        compute_loss = torch.stack(activations_all, dim=1).mean()
        weights = {
            "task": 1.0,
            "specialist": 1.0,
            "wake_required_set": 1.0,
            "halt": 0.25,
            "active_compute": 0.0,
            **dict(loss_weights or {}),
        }
        total = (
            float(weights["task"]) * gpt_loss
            + float(weights["specialist"]) * specialist_loss
            + float(weights["wake_required_set"]) * wake_loss
            + float(weights["halt"]) * halt_loss
            + float(weights["active_compute"]) * compute_loss
        )
        return V13ModelOutput(
            loss=total,
            gpt_loss=gpt_loss,
            specialist_loss=specialist_loss,
            wake_loss=wake_loss,
            halt_loss=halt_loss,
            compute_loss=compute_loss,
            gpt_logits=gpt_output.logits,
            rounds=round_outputs,
        )
