from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteMathTokenizer, SequenceTooLongError, pad_1d
from .v1_3_data import JOINT_SCHEMA, SPECIALISTS, STRING_SCHEMA, load_v1_3_records


class ExternalTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


class V13Dataset(Dataset[dict[str, Any]]):
    def __init__(self, records_or_path: list[dict[str, Any]] | str | Path) -> None:
        self.records = (
            load_v1_3_records(records_or_path)
            if isinstance(records_or_path, (str, Path))
            else list(records_or_path)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _external_encode(tokenizer: ExternalTokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


class V13StringCollator:
    def __init__(self, tokenizer: ByteMathTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if any(record.get("schema_version") != STRING_SCHEMA for record in records):
            raise ValueError("native string collator received a non-string record")
        encoded = [
            self.tokenizer.encode_training_example(
                str(record["problem"]), str(record["target_trace"]), self.max_length
            )
            for record in records
        ]
        input_ids, attention_mask = pad_1d(
            [item.input_ids for item in encoded], self.tokenizer.pad_token_id
        )
        labels, _ = pad_1d([item.labels for item in encoded], -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prefix_lengths": torch.tensor(
                [item.prefix_length for item in encoded], dtype=torch.long
            ),
            "records": records,
        }


class V13JointCollator:
    """Build GPT-only raw prompts and neutral specialist workspaces."""

    def __init__(
        self,
        math_tokenizer: ByteMathTokenizer,
        gpt_tokenizer: ExternalTokenizer,
        *,
        maximum_gpt_length: int,
        maximum_specialist_length: int,
        maximum_rounds: int,
        neutral_workspaces: dict[str, str],
    ) -> None:
        self.math_tokenizer = math_tokenizer
        self.gpt_tokenizer = gpt_tokenizer
        self.maximum_gpt_length = int(maximum_gpt_length)
        self.maximum_specialist_length = int(maximum_specialist_length)
        self.maximum_rounds = int(maximum_rounds)
        self.neutral_workspaces = dict(neutral_workspaces)
        self.gpt_pad_id = (
            gpt_tokenizer.pad_token_id
            if gpt_tokenizer.pad_token_id is not None
            else gpt_tokenizer.eos_token_id
        )
        if self.gpt_pad_id is None or gpt_tokenizer.eos_token_id is None:
            raise ValueError("GPT tokenizer must expose EOS and pad/EOS IDs")
        if set(self.neutral_workspaces) != set(SPECIALISTS):
            raise ValueError("neutral workspaces must be defined for math and string")

    @staticmethod
    def gpt_prompt(record: Mapping[str, Any] | str) -> str:
        """Return the registered completion prompt, with a legacy string fallback."""

        if isinstance(record, Mapping):
            prompt = record.get("gpt_prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("V1.3 record has no registered GPT prompt")
            return prompt
        return f"Problem: {record}\nExact result:"

    @staticmethod
    def gpt_target(record: Mapping[str, Any]) -> str:
        target = record.get("gpt_target")
        if not isinstance(target, str) or not target or "\n" in target or "\r" in target:
            raise ValueError("V1.3 record has an invalid GPT completion target")
        return target + "\n"

    def _specialist_batch(
        self, records: list[dict[str, Any]], specialist: str, round_index: int
    ) -> dict[str, torch.Tensor]:
        input_sequences: list[list[int]] = []
        label_sequences: list[list[int]] = []
        prefix_lengths: list[int] = []
        for record in records:
            target = record["specialist_targets_by_round"][specialist][round_index]
            workspace = self.neutral_workspaces[specialist]
            prefix = self.math_tokenizer.encode_generation_prefix(
                workspace, self.maximum_specialist_length
            )
            if target is None:
                sequence = prefix
                labels = [-100] * len(sequence)
            else:
                target_ids = self.math_tokenizer.encode(str(target), add_eos=True)
                sequence = prefix + target_ids
                labels = [-100] * len(prefix) + target_ids
            if len(sequence) > self.maximum_specialist_length:
                raise SequenceTooLongError(
                    f"{specialist} round {round_index + 1} sequence has "
                    f"{len(sequence)} tokens, exceeding {self.maximum_specialist_length}"
                )
            input_sequences.append(sequence)
            label_sequences.append(labels)
            prefix_lengths.append(len(prefix))
        input_ids, attention_mask = pad_1d(
            input_sequences, self.math_tokenizer.pad_token_id
        )
        labels, _ = pad_1d(label_sequences, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prefix_lengths": torch.tensor(prefix_lengths, dtype=torch.long),
        }

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if any(record.get("schema_version") != JOINT_SCHEMA for record in records):
            raise ValueError("V1.3 joint collator received a non-joint record")
        gpt_prefixes: list[list[int]] = []
        gpt_sequences: list[list[int]] = []
        gpt_labels: list[list[int]] = []
        wake_targets = torch.zeros(
            (len(records), self.maximum_rounds, len(SPECIALISTS)), dtype=torch.float32
        )
        halt_targets = torch.full(
            (len(records), self.maximum_rounds), -100.0, dtype=torch.float32
        )
        for row, record in enumerate(records):
            prompt_ids = _external_encode(
                self.gpt_tokenizer, self.gpt_prompt(record)
            )
            target_ids = _external_encode(
                self.gpt_tokenizer, self.gpt_target(record)
            ) + [int(self.gpt_tokenizer.eos_token_id)]
            combined = prompt_ids + target_ids
            if len(combined) > self.maximum_gpt_length:
                raise SequenceTooLongError(
                    f"GPT sequence has {len(combined)} tokens, exceeding "
                    f"{self.maximum_gpt_length}"
                )
            gpt_prefixes.append(prompt_ids)
            gpt_sequences.append(combined)
            gpt_labels.append([-100] * len(prompt_ids) + target_ids)
            required_by_round = record["required_specialists_by_round"]
            if len(required_by_round) != self.maximum_rounds:
                raise ValueError("required-specialist rounds differ from configuration")
            for round_index, required in enumerate(required_by_round):
                if len(set(required)) != len(required) or not set(required).issubset(SPECIALISTS):
                    raise ValueError("record contains an invalid required-specialist set")
                for name in required:
                    wake_targets[row, round_index, SPECIALISTS.index(name)] = 1.0
                for name in SPECIALISTS:
                    has_target = (
                        record["specialist_targets_by_round"][name][round_index]
                        is not None
                    )
                    if has_target != (name in required):
                        raise ValueError(
                            "specialist target and required-set labels disagree"
                        )
            halt_index = int(record["halt_round"]) - 1
            if halt_index < 0 or halt_index >= self.maximum_rounds:
                raise ValueError("record halt round is outside the configured runtime")
            halt_targets[row, :halt_index] = 0.0
            halt_targets[row, halt_index] = 1.0
        gpt_prepass_ids, gpt_prepass_mask = pad_1d(
            gpt_prefixes, int(self.gpt_pad_id)
        )
        gpt_input_ids, gpt_attention_mask = pad_1d(
            gpt_sequences, int(self.gpt_pad_id)
        )
        padded_gpt_labels, _ = pad_1d(gpt_labels, -100)
        specialist_batches = {
            name: [
                self._specialist_batch(records, name, round_index)
                for round_index in range(self.maximum_rounds)
            ]
            for name in SPECIALISTS
        }
        return {
            "gpt_prepass_input_ids": gpt_prepass_ids,
            "gpt_prepass_attention_mask": gpt_prepass_mask,
            "gpt_input_ids": gpt_input_ids,
            "gpt_attention_mask": gpt_attention_mask,
            "gpt_labels": padded_gpt_labels,
            "specialists": specialist_batches,
            "wake_targets": wake_targets,
            "halt_targets": halt_targets,
            "task_classes": [str(record["task_class"]) for record in records],
            "records": records,
        }


def move_v1_3_batch(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_v1_3_batch(item, device) for key, item in value.items()}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [move_v1_3_batch(item, device) for item in value]
    return value
