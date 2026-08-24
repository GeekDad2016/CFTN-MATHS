from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import canonical_json
from .gpt_receiver import pretrained_dtype_kwargs, validate_dense_causal_lm_config


SEMANTIC_CACHE_FORMAT = "cftn_text_frozen_semantic_features_v1"


def normalize_token_ids(values: Any) -> list[int]:
    """Normalize tokenizer outputs across Transformers 4.x and 5.x."""

    if isinstance(values, Mapping):
        if "input_ids" not in values:
            raise ValueError("tokenizer mapping exposes no input_ids")
        values = values["input_ids"]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    elif not isinstance(values, (list, tuple)) and callable(
        getattr(values, "tolist", None)
    ):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if isinstance(values, list) and values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError("tokenizer returned multiple sequences for one prompt")
        values = list(values[0])
    if not isinstance(values, list):
        raise TypeError("tokenizer input_ids must be a one-dimensional sequence")
    return [int(value) for value in values]


def semantic_prompt_ids(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
) -> list[int]:
    """Tokenize a dispatcher prompt exactly as a frozen coordinator request."""

    if use_chat_template:
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise ValueError("semantic tokenizer exposes no chat template")
        values = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return normalize_token_ids(values)
    try:
        values = tokenizer.encode(str(prompt), add_special_tokens=False)
    except TypeError:
        values = tokenizer.encode(str(prompt))
    return [int(value) for value in values]


def causal_prompt_and_target_ids(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    use_chat_template: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Return prefix, full sequence, and masked-label suffix for LM training."""

    prefix = semantic_prompt_ids(
        tokenizer, prompt, use_chat_template=use_chat_template
    )
    if use_chat_template:
        full_values = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(target)},
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        full = normalize_token_ids(full_values)
        if full[: len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(
                "chat template assistant sequence is not an extension of its prompt"
            )
        target_ids = full[len(prefix) :]
    else:
        try:
            target_ids = list(tokenizer.encode(str(target), add_special_tokens=False))
        except TypeError:
            target_ids = list(tokenizer.encode(str(target)))
        if tokenizer.eos_token_id is None:
            raise ValueError("causal tokenizer exposes no EOS token")
        target_ids.append(int(tokenizer.eos_token_id))
        full = [*prefix, *target_ids]
    return prefix, full, [-100] * len(prefix) + target_ids


def semantic_cache_contract(
    prompts: Iterable[str],
    *,
    model_name: str,
    revision: str | None,
    maximum_length: int,
    use_chat_template: bool,
    semantic_width: int,
) -> dict[str, Any]:
    prompt_values = [str(value) for value in prompts]
    prompt_hash = hashlib.sha256(
        canonical_json(prompt_values).encode("utf-8")
    ).hexdigest()
    contract = {
        "model_name": str(model_name),
        "revision": None if revision is None else str(revision),
        "maximum_length": int(maximum_length),
        "use_chat_template": bool(use_chat_template),
        "semantic_width": int(semantic_width),
        "examples": len(prompt_values),
        "prompts_sha256": prompt_hash,
        "pooling": "attention_mask_mean_final_hidden_v1",
    }
    contract["contract_sha256"] = hashlib.sha256(
        canonical_json(contract).encode("utf-8")
    ).hexdigest()
    return contract


class FrozenSemanticFeatureExtractor:
    """One-pass frozen dense-LM encoder used to prepare dispatcher features."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        device: torch.device | str,
        dtype: str | torch.dtype | None,
        local_files_only: bool,
        trust_remote_code: bool,
        attn_implementation: str | None,
        expected_model_type: str | None,
        expected_hidden_size: int | None,
        expected_layers: int | None,
        require_dense: bool,
        maximum_length: int,
        use_chat_template: bool,
    ) -> None:
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoTokenizer,
            __version__ as transformers_version,
        )

        self.device = torch.device(device)
        self.maximum_length = int(maximum_length)
        self.use_chat_template = bool(use_chat_template)
        common: dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
        }
        if revision:
            common["revision"] = str(revision)
        model_config = AutoConfig.from_pretrained(model_name, **common)
        validate_dense_causal_lm_config(
            model_config,
            expected_model_type=expected_model_type,
            expected_hidden_size=expected_hidden_size,
            expected_layers=expected_layers,
            require_dense=require_dense,
        )
        self.semantic_width = int(
            getattr(model_config, "hidden_size", getattr(model_config, "n_embd", 0))
        )
        if self.semantic_width < 1:
            raise ValueError("semantic model config exposes no hidden size")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("semantic tokenizer exposes no EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = dict(common)
        model_kwargs.update(pretrained_dtype_kwargs(dtype, transformers_version))
        if attn_implementation:
            model_kwargs["attn_implementation"] = str(attn_implementation)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def encode(self, prompts: list[str], *, batch_size: int) -> torch.Tensor:
        if not prompts:
            raise ValueError("cannot encode an empty semantic feature panel")
        rows: list[torch.Tensor] = []
        for start in range(0, len(prompts), int(batch_size)):
            token_rows = [
                semantic_prompt_ids(
                    self.tokenizer,
                    prompt,
                    use_chat_template=self.use_chat_template,
                )
                for prompt in prompts[start : start + int(batch_size)]
            ]
            if any(not values for values in token_rows):
                raise ValueError("semantic prompt tokenized to an empty sequence")
            longest = max(len(values) for values in token_rows)
            if longest > self.maximum_length:
                raise ValueError(
                    f"semantic prompt has {longest} tokens, exceeding "
                    f"{self.maximum_length}"
                )
            input_ids = torch.full(
                (len(token_rows), longest),
                int(self.tokenizer.pad_token_id),
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, values in enumerate(token_rows):
                input_ids[row, : len(values)] = torch.tensor(
                    values, dtype=torch.long, device=self.device
                )
                attention_mask[row, : len(values)] = 1
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.last_hidden_state
            denominator = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / denominator
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError("frozen semantic encoder produced non-finite features")
            rows.append(pooled.detach().cpu().to(dtype=torch.float16))
        return torch.cat(rows, dim=0)


def load_or_create_semantic_cache(
    path: str | Path,
    prompts: list[str],
    *,
    extractor: FrozenSemanticFeatureExtractor,
    model_name: str,
    revision: str | None,
    maximum_length: int,
    use_chat_template: bool,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    destination = Path(path)
    contract = semantic_cache_contract(
        prompts,
        model_name=model_name,
        revision=revision,
        maximum_length=maximum_length,
        use_chat_template=use_chat_template,
        semantic_width=extractor.semantic_width,
    )
    if destination.is_file():
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        if (
            payload.get("format") == SEMANTIC_CACHE_FORMAT
            and payload.get("contract") == contract
            and torch.is_tensor(payload.get("features"))
            and tuple(payload["features"].shape)
            == (len(prompts), extractor.semantic_width)
        ):
            return payload["features"], contract
        raise RuntimeError(
            f"semantic cache contract differs at {destination}; remove that cache explicitly"
        )
    features = extractor.encode(prompts, batch_size=int(batch_size))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {
            "format": SEMANTIC_CACHE_FORMAT,
            "contract": contract,
            "features": features,
        },
        temporary,
    )
    temporary.replace(destination)
    return features, contract


__all__ = [
    "FrozenSemanticFeatureExtractor",
    "SEMANTIC_CACHE_FORMAT",
    "causal_prompt_and_target_ids",
    "load_or_create_semantic_cache",
    "semantic_cache_contract",
    "semantic_prompt_ids",
]
