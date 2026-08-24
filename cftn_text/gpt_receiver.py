from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn

from .bridges import GatedCrossReceiver


_DENSE_BLOCK_PATHS = (
    ("transformer", "h"),       # GPT-2
    ("model", "layers"),        # Qwen, Llama, Mistral
    ("model", "decoder", "layers"),
    ("gpt_neox", "layers"),
)
_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "experts_per_token",
    "moe_intermediate_size",
)


def resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    """Resolve the small, explicit dtype vocabulary accepted by V2 configs."""

    if value is None or isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower().replace("torch.", "")
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported causal-LM dtype: {value}")
    return aliases[normalized]


def pretrained_dtype_kwargs(
    value: str | torch.dtype | None, transformers_version: str
) -> dict[str, torch.dtype]:
    """Use the non-deprecated dtype keyword for the installed Transformers major."""

    resolved = resolve_torch_dtype(value)
    if resolved is None:
        return {}
    match = re.match(r"\d+", str(transformers_version))
    if match is None:
        raise ValueError(f"invalid Transformers version: {transformers_version!r}")
    keyword = "dtype" if int(match.group()) >= 5 else "torch_dtype"
    return {keyword: resolved}


def validate_dense_causal_lm_config(
    config: Any,
    *,
    expected_model_type: str | None = None,
    expected_hidden_size: int | None = None,
    expected_layers: int | None = None,
    require_dense: bool = True,
) -> None:
    """Fail closed if a configured coordinator is not the pinned dense model.

    V2 deliberately excludes mixture-of-experts coordinators.  Checking both
    architecture names and known router/expert fields protects against a model
    repository changing underneath an otherwise familiar identifier.
    """

    model_type = str(getattr(config, "model_type", ""))
    architectures = tuple(str(value) for value in getattr(config, "architectures", ()) or ())
    if expected_model_type and model_type != str(expected_model_type):
        raise ValueError(
            f"causal-LM model_type {model_type!r} differs from expected "
            f"{expected_model_type!r}"
        )
    if require_dense:
        names = (model_type, *architectures)
        if any("moe" in name.casefold() or "mixtral" in name.casefold() for name in names):
            raise ValueError("V2 coordinator must be dense; an MoE architecture was found")
        for key in _MOE_CONFIG_KEYS:
            value = getattr(config, key, None)
            if value not in (None, False, 0, 1):
                raise ValueError(
                    f"V2 coordinator must be dense; config.{key}={value!r}"
                )
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    layer_count = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    if expected_hidden_size is not None and int(hidden_size or -1) != int(
        expected_hidden_size
    ):
        raise ValueError(
            f"causal-LM hidden size {hidden_size!r} differs from expected "
            f"{expected_hidden_size}"
        )
    if expected_layers is not None and int(layer_count or -1) != int(expected_layers):
        raise ValueError(
            f"causal-LM layer count {layer_count!r} differs from expected {expected_layers}"
        )


def _resolve_decoder_blocks(model: nn.Module) -> nn.ModuleList | Sequence[nn.Module]:
    for path in _DENSE_BLOCK_PATHS:
        value: Any = model
        for attribute in path:
            if not hasattr(value, attribute):
                break
            value = getattr(value, attribute)
        else:
            if isinstance(value, (nn.ModuleList, list, tuple)):
                return value
    raise TypeError(
        "FrozenCausalLMTower requires a supported dense decoder block layout"
    )


class FrozenCausalLMTower(nn.Module):
    """Frozen dense causal LM with trainable CFTN receivers on decoder blocks."""

    def __init__(
        self,
        model: nn.Module,
        receiver_layers: list[int],
        bridge_config: dict,
    ) -> None:
        super().__init__()
        self.model = model
        validate_dense_causal_lm_config(model.config)
        self.hidden_size = int(
            getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
        )
        if self.hidden_size < 1:
            raise TypeError("causal LM config exposes no positive hidden size")
        self._decoder_blocks = _resolve_decoder_blocks(model)
        self.receiver_layers = tuple(int(layer) for layer in receiver_layers)
        block_count = len(self._decoder_blocks)
        if any(layer < 0 or layer >= block_count for layer in self.receiver_layers):
            raise ValueError(
                f"causal-LM receiver layer must be within 0..{block_count - 1}"
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
            self._decoder_blocks[layer].register_forward_hook(
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
        revision: str | None = None,
        dtype: str | torch.dtype | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        expected_model_type: str | None = None,
        expected_hidden_size: int | None = None,
        expected_layers: int | None = None,
        require_dense: bool = True,
    ) -> "FrozenCausalLMTower":
        from transformers import AutoModelForCausalLM, __version__ as transformers_version

        load_kwargs: dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
        }
        if revision is not None:
            load_kwargs["revision"] = str(revision)
        load_kwargs.update(pretrained_dtype_kwargs(dtype, transformers_version))
        if attn_implementation:
            load_kwargs["attn_implementation"] = str(attn_implementation)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        validate_dense_causal_lm_config(
            model.config,
            expected_model_type=expected_model_type,
            expected_hidden_size=expected_hidden_size,
            expected_layers=expected_layers,
            require_dense=require_dense,
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
        eos_token_id: int | Sequence[int],
        max_new_tokens: int,
        *,
        message_mask: torch.Tensor | None = None,
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
        eos_ids = (
            (int(eos_token_id),)
            if isinstance(eos_token_id, int)
            else tuple(int(value) for value in eos_token_id)
        )
        if not eos_ids:
            raise ValueError("generation requires at least one EOS token ID")
        pad_token_id = getattr(self.model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_ids[0]
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
        if message_mask is None:
            message_mask = torch.ones(
                messages.shape[:2], dtype=torch.long, device=device
            )
        elif message_mask.shape != messages.shape[:2]:
            raise ValueError("generation message mask shape differs from messages")
        else:
            message_mask = message_mask.to(device=device, dtype=torch.long)
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
            reached_eos = torch.zeros_like(finished)
            for value in eos_ids:
                reached_eos.logical_or_(next_tokens.eq(value))
            finished = finished | (active & reached_eos)
            if bool(finished.all()):
                break
        return generated

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


# Historical configurations and tests import this name.  Keeping the alias is
# intentionally source-compatible while V2 uses the model-neutral class.
FrozenGPT2Tower = FrozenCausalLMTower


__all__ = [
    "FrozenCausalLMTower",
    "FrozenGPT2Tower",
    "pretrained_dtype_kwargs",
    "resolve_torch_dtype",
    "validate_dense_causal_lm_config",
]
