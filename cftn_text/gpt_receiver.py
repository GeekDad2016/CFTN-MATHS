from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn

from .bridges import GatedCrossReceiver


class FrozenGPT2Tower(nn.Module):
    """Frozen GPT-2 with trainable message receivers attached by block hooks."""

    def __init__(
        self,
        model: nn.Module,
        receiver_layers: list[int],
        bridge_config: dict,
    ) -> None:
        super().__init__()
        if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
            raise TypeError("FrozenGPT2Tower requires a GPT2LMHeadModel-compatible model")
        self.model = model
        self.hidden_size = int(model.config.n_embd)
        self.receiver_layers = tuple(int(layer) for layer in receiver_layers)
        block_count = len(model.transformer.h)
        if any(layer < 0 or layer >= block_count for layer in self.receiver_layers):
            raise ValueError(
                f"GPT receiver layer must be within 0..{block_count - 1}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.receivers = nn.ModuleDict(
            {
                str(layer): GatedCrossReceiver(
                    receiver_width=self.hidden_size,
                    message_width=int(bridge_config["message_width"]),
                    heads=int(bridge_config["attention_heads"]),
                    gate_hidden_size=int(bridge_config["gate_hidden_size"]),
                    dropout=float(bridge_config["dropout"]),
                    gate_init=float(bridge_config["gate_init"]),
                    zero_init_output=bool(bridge_config["zero_init_output"]),
                )
                for layer in self.receiver_layers
            }
        )
        self._active_message: torch.Tensor | None = None
        self._active_message_mask: torch.Tensor | None = None
        self._active_enabled = False
        self._active_gate_mode = "contextual"
        self._hooks = [
            self.model.transformer.h[layer].register_forward_hook(
                self._make_hook(layer)
            )
            for layer in self.receiver_layers
        ]
        self.prepass_execution_count = 0
        self.receiver_execution_count = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        receiver_layers: list[int],
        bridge_config: dict,
        *,
        local_files_only: bool = True,
    ) -> "FrozenGPT2Tower":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        return cls(model, receiver_layers, bridge_config)

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if not self._active_enabled or self._active_message is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            updated = self.receivers[str(layer)](
                hidden,
                self._active_message,
                self._active_message_mask,
                enabled=True,
                gate_mode=self._active_gate_mode,
            )
            if isinstance(output, tuple):
                return (updated, *output[1:])
            return updated

        return hook

    @contextmanager
    def _message_context(
        self,
        message: torch.Tensor | None,
        message_mask: torch.Tensor | None,
        enabled: bool,
        gate_mode: str,
    ) -> Iterator[None]:
        previous = (
            self._active_message,
            self._active_message_mask,
            self._active_enabled,
            self._active_gate_mode,
        )
        self._active_message = message
        self._active_message_mask = message_mask
        self._active_enabled = bool(enabled)
        self._active_gate_mode = gate_mode
        try:
            yield
        finally:
            (
                self._active_message,
                self._active_message_mask,
                self._active_enabled,
                self._active_gate_mode,
            ) = previous

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        self.receivers.train(mode)
        return self

    def reset_execution_count(self) -> None:
        self.prepass_execution_count = 0
        self.receiver_execution_count = 0
        for receiver in self.receivers.values():
            receiver.reset_execution_count()

    def prepass(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.prepass_execution_count += 1
        with torch.no_grad(), self._message_context(None, None, False, "contextual"):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return output.hidden_states[-1]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        message: torch.Tensor | None = None,
        message_mask: torch.Tensor | None = None,
        receive_enabled: bool = True,
        gate_mode: str = "contextual",
    ):
        self.receiver_execution_count += 1
        with self._message_context(message, message_mask, receive_enabled, gate_mode):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

    @torch.no_grad()
    def generate_greedy(
        self,
        prefix_ids: list[list[int]],
        messages: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
        *,
        receive_enabled: bool = True,
        gate_mode: str = "contextual",
    ) -> list[list[int]]:
        if len(prefix_ids) != messages.shape[0]:
            raise ValueError("prefix/message batch sizes differ")
        if not prefix_ids or any(not prefix for prefix in prefix_ids):
            raise ValueError("generation prefixes must be nonempty")
        device = messages.device
        batch = len(prefix_ids)
        maximum_prefix = max(len(prefix) for prefix in prefix_ids)
        maximum_context = int(
            getattr(
                self.model.config,
                "n_positions",
                getattr(self.model.config, "max_position_embeddings", 1024),
            )
        )
        if maximum_prefix + int(max_new_tokens) > maximum_context:
            raise ValueError("GPT generation exceeds the model context window")
        pad_token_id = getattr(self.model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        input_ids = torch.full(
            (batch, maximum_prefix),
            int(pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, prefix in enumerate(prefix_ids):
            input_ids[row, maximum_prefix - len(prefix) :] = torch.tensor(
                prefix, dtype=torch.long, device=device
            )
            attention_mask[row, maximum_prefix - len(prefix) :] = 1
        message_mask = torch.ones(
            messages.shape[:2], dtype=torch.long, device=device
        )
        generated: list[list[int]] = [[] for _ in prefix_ids]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        for _ in range(int(max_new_tokens)):
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)
            output = self.forward(
                input_ids,
                attention_mask,
                position_ids=position_ids,
                message=messages,
                message_mask=message_mask,
                receive_enabled=receive_enabled,
                gate_mode=gate_mode,
            )
            next_tokens = output.logits[:, -1].argmax(dim=-1)
            active = ~finished
            for row, token in enumerate(next_tokens.tolist()):
                if bool(active[row]):
                    generated[row].append(int(token))
            appended = torch.where(
                active,
                next_tokens,
                torch.full_like(next_tokens, int(pad_token_id)),
            )
            input_ids = torch.cat((input_ids, appended.unsqueeze(1)), dim=1)
            attention_mask = torch.cat(
                (attention_mask, active.to(dtype=torch.long).unsqueeze(1)), dim=1
            )
            finished = finished | (active & next_tokens.eq(int(eos_token_id)))
            if bool(finished.all()):
                break
        return generated

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
