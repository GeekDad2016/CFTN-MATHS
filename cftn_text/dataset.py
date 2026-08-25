from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset

from .data_generator import load_records
from .semantic_features import causal_prompt_and_target_ids
from .tokenizer import ByteMathTokenizer, SequenceTooLongError, pad_1d


_ANSWER_VALUE_SENTINEL = -2_000_000_000
_MIN_SIGNED_INT64 = -(1 << 63)
_MAX_SIGNED_INT64 = (1 << 63) - 1
SHARED_MATH_INPUT_VIEW = "shared_problem_v1"
PRIVATE_MATH_INPUT_VIEW = "private_math_problem_v1"
MATH_INPUT_VIEWS = {SHARED_MATH_INPUT_VIEW, PRIVATE_MATH_INPUT_VIEW}


def _tensor_safe_answer_value(record: dict[str, Any]) -> int:
    value = record.get("answer_value")
    if value is None:
        value = record.get("x", _ANSWER_VALUE_SENTINEL)
    if not isinstance(value, int) or isinstance(value, bool):
        return _ANSWER_VALUE_SENTINEL
    if value < _MIN_SIGNED_INT64 or value > _MAX_SIGNED_INT64:
        return _ANSWER_VALUE_SENTINEL
    return value


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
            if first and str(first.get("schema_version", "")).startswith(
                "cftn_math_record_v2"
            ):
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


def math_problem_for_view(record: dict[str, Any], input_view: str) -> str:
    """Resolve the exact text contract presented to the standalone math tower."""

    view = str(input_view)
    if view == SHARED_MATH_INPUT_VIEW:
        problem = record.get("problem")
    elif view == PRIVATE_MATH_INPUT_VIEW:
        problem = record.get("math_problem") or record.get("problem")
    else:
        raise ValueError(f"unsupported math input view: {view}")
    if problem is None or not str(problem).strip():
        raise ValueError(f"math input view {view} resolved to an empty problem")
    return str(problem)


class MathCollator:
    def __init__(
        self,
        tokenizer: ByteMathTokenizer,
        max_length: int,
        *,
        target_mode: str = "full_trace_v1",
        input_view: str = SHARED_MATH_INPUT_VIEW,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.target_mode = str(target_mode)
        self.input_view = str(input_view)
        if self.target_mode not in {"full_trace_v1", "answer_only_v1"}:
            raise ValueError(f"unsupported math target mode: {self.target_mode}")
        if self.input_view not in MATH_INPUT_VIEWS:
            raise ValueError(f"unsupported math input view: {self.input_view}")

    def _target(self, record: dict[str, Any]) -> str:
        if self.target_mode == "full_trace_v1":
            return str(record["target_trace"])
        answer = record.get("normalized_answer", record.get("target_answer"))
        if answer is None:
            raise ValueError("answer-only math training requires a normalized answer")
        return f"<answer>{answer}</answer>"

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        targets = [self._target(record) for record in records]
        encoded = [
            self.tokenizer.encode_training_example(
                math_problem_for_view(record, self.input_view),
                target,
                self.max_length,
            )
            for record, target in zip(records, targets)
        ]
        input_ids, attention_mask = pad_1d(
            [item.input_ids for item in encoded],
            self.tokenizer.pad_token_id,
            self.max_length,
        )
        labels, _ = pad_1d(
            [item.labels for item in encoded], -100, self.max_length
        )
        answer_labels_rows: list[list[int]] = []
        for item, target in zip(encoded, targets):
            answer_start = target.rfind("<answer>")
            if answer_start < 0:
                raise ValueError("math target is missing an <answer> payload")
            prefix_bytes = len(self.tokenizer.encode(target[:answer_start]))
            focused = [-100] * len(item.labels)
            suffix_start = item.prefix_length + prefix_bytes
            focused[suffix_start:] = item.labels[suffix_start:]
            answer_labels_rows.append(focused)
        answer_labels, _ = pad_1d(answer_labels_rows, -100, self.max_length)
        return {
            "math_input_ids": input_ids,
            "math_attention_mask": attention_mask,
            "math_labels": labels,
            "math_answer_labels": answer_labels,
            "math_prefix_lengths": torch.tensor(
                [item.prefix_length for item in encoded], dtype=torch.long
            ),
            "answer_values": torch.tensor(
                [_tensor_safe_answer_value(record) for record in records],
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
        use_chat_template: bool | None = None,
    ) -> None:
        # Joint CFTN training intentionally uses the complementary private
        # math view. Standalone specialist training uses the shared view.
        super().__init__(
            math_tokenizer,
            max_math_length,
            input_view=PRIVATE_MATH_INPUT_VIEW,
        )
        self.gpt_tokenizer = gpt_tokenizer
        self.max_gpt_length = int(max_gpt_length)
        self.use_chat_template = (
            bool(getattr(gpt_tokenizer, "_cftn_use_chat_template", False))
            if use_chat_template is None
            else bool(use_chat_template)
        )
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
                    str(record.get("schema_version", "")).startswith(
                        "cftn_math_record_v2"
                    )
                ),
            )
            prompt_ids, combined, labels = causal_prompt_and_target_ids(
                self.gpt_tokenizer,
                prompt,
                str(record["target_answer"]),
                use_chat_template=self.use_chat_template,
            )
            if len(combined) > self.max_gpt_length:
                raise SequenceTooLongError(
                    f"GPT sequence has {len(combined)} tokens, exceeding "
                    f"{self.max_gpt_length}"
                )
            prepass_ids.append(prompt_ids)
            final_ids.append(combined)
            final_labels.append(labels)
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
