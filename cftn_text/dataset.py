from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset

from .data_generator import load_records
from .tokenizer import ByteMathTokenizer, SequenceTooLongError, pad_1d


class ExternalTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


class EquationDataset(Dataset[dict[str, Any]]):
    def __init__(self, records_or_path: list[dict[str, Any]] | str | Path) -> None:
        if isinstance(records_or_path, (str, Path)):
            path = Path(records_or_path)
            first: dict[str, Any] | None = None
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        import json

                        first = json.loads(line)
                        break
            if first and first.get("schema_version") == "cftn_math_record_v2":
                from .v2_data import load_v2_records

                self.records = load_v2_records(path)
            elif first and first.get("schema_version") == "cftn_linear_equation_v1_1":
                from .algorithmic_data_generator import load_algorithmic_records

                self.records = load_algorithmic_records(path)
            else:
                self.records = load_records(path)
        else:
            self.records = list(records_or_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _external_encode(tokenizer: ExternalTokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


class MathCollator:
    def __init__(self, tokenizer: ByteMathTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = [
            self.tokenizer.encode_training_example(
                record.get("math_problem", record["problem"]),
                record["target_trace"],
                self.max_length,
            )
            for record in records
        ]
        input_ids, attention_mask = pad_1d(
            [item.input_ids for item in encoded],
            self.tokenizer.pad_token_id,
            self.max_length,
        )
        labels, _ = pad_1d(
            [item.labels for item in encoded], -100, self.max_length
        )
        return {
            "math_input_ids": input_ids,
            "math_attention_mask": attention_mask,
            "math_labels": labels,
            "math_prefix_lengths": torch.tensor(
                [item.prefix_length for item in encoded], dtype=torch.long
            ),
            "answer_values": torch.tensor(
                [
                    (
                        record.get("answer_value")
                        if record.get("answer_value") is not None
                        else record.get("x", -2_000_000_000)
                    )
                    for record in records
                ],
                dtype=torch.long,
            ),
            "records": records,
        }


class CFTNCollator(MathCollator):
    def __init__(
        self,
        math_tokenizer: ByteMathTokenizer,
        gpt_tokenizer: ExternalTokenizer,
        max_math_length: int,
        max_gpt_length: int,
    ) -> None:
        super().__init__(math_tokenizer, max_math_length)
        self.gpt_tokenizer = gpt_tokenizer
        self.max_gpt_length = int(max_gpt_length)
        self.gpt_pad_id = (
            gpt_tokenizer.pad_token_id
            if gpt_tokenizer.pad_token_id is not None
            else gpt_tokenizer.eos_token_id
        )
        if self.gpt_pad_id is None:
            raise ValueError("GPT tokenizer must expose a pad or EOS token ID")
        if gpt_tokenizer.eos_token_id is None:
            raise ValueError("GPT tokenizer must expose an EOS token ID")

    @staticmethod
    def gpt_prompt(problem: str, *, generic_answer: bool = False) -> str:
        instruction = (
            "Return only the exact result in <answer> tags."
            if generic_answer
            else "Return only the integer in <answer> tags."
        )
        return (
            f"Problem: {problem}\n"
            f"{instruction}\nAnswer:"
        )

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().__call__(records)
        prepass_ids: list[list[int]] = []
        final_ids: list[list[int]] = []
        final_labels: list[list[int]] = []
        final_prefix_lengths: list[int] = []
        for record in records:
            prompt = self.gpt_prompt(
                record.get("gpt_problem", record["problem"]),
                generic_answer=(
                    record.get("schema_version") == "cftn_math_record_v2"
                ),
            )
            prompt_ids = _external_encode(self.gpt_tokenizer, prompt)
            target_ids = _external_encode(self.gpt_tokenizer, record["target_answer"])
            target_ids.append(int(self.gpt_tokenizer.eos_token_id))
            combined = prompt_ids + target_ids
            if len(combined) > self.max_gpt_length:
                raise SequenceTooLongError(
                    f"GPT sequence has {len(combined)} tokens, exceeding "
                    f"{self.max_gpt_length}"
                )
            prepass_ids.append(prompt_ids)
            final_ids.append(combined)
            final_labels.append([-100] * len(prompt_ids) + target_ids)
            final_prefix_lengths.append(len(prompt_ids))
        gpt_prepass_ids, gpt_prepass_mask = pad_1d(
            prepass_ids, int(self.gpt_pad_id), self.max_gpt_length
        )
        gpt_input_ids, gpt_attention_mask = pad_1d(
            final_ids, int(self.gpt_pad_id), self.max_gpt_length
        )
        gpt_labels, _ = pad_1d(final_labels, -100, self.max_gpt_length)
        batch.update(
            {
                "gpt_prepass_input_ids": gpt_prepass_ids,
                "gpt_prepass_attention_mask": gpt_prepass_mask,
                "gpt_input_ids": gpt_input_ids,
                "gpt_attention_mask": gpt_attention_mask,
                "gpt_labels": gpt_labels,
                "gpt_prefix_lengths": torch.tensor(
                    final_prefix_lengths, dtype=torch.long
                ),
            }
        )
        return batch
