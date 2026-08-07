from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


class SequenceTooLongError(ValueError):
    pass


@dataclass(frozen=True)
class EncodedMathExample:
    input_ids: list[int]
    labels: list[int]
    prefix_length: int


class ByteMathTokenizer:
    """Lossless UTF-8 byte tokenizer with four reserved control tokens."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    sep_token_id = 3
    byte_offset = 4
    vocab_size = 260

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        add_special_tokens: bool | None = None,
        max_length: int | None = None,
    ) -> list[int]:
        if add_special_tokens:
            add_bos = True
            add_eos = True
        ids = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length is not None and len(ids) > max_length:
            raise SequenceTooLongError(
                f"encoded sequence has {len(ids)} tokens, exceeding {max_length}"
            )
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        data = bytearray()
        for token in ids:
            value = int(token)
            if value < self.byte_offset:
                if skip_special_tokens:
                    continue
                raise ValueError("special tokens cannot be decoded as UTF-8 bytes")
            if value >= self.vocab_size:
                raise ValueError(f"token ID {value} is outside the byte vocabulary")
            data.append(value - self.byte_offset)
        return data.decode("utf-8", errors="replace")

    def encode_training_example(
        self,
        problem: str,
        target_trace: str,
        max_length: int,
    ) -> EncodedMathExample:
        prefix = self.encode(f"Problem: {problem}\nSolution:", add_bos=True)
        prefix.append(self.sep_token_id)
        target = self.encode(target_trace, add_eos=True)
        input_ids = prefix + target
        if len(input_ids) > max_length:
            raise SequenceTooLongError(
                f"math sequence has {len(input_ids)} tokens, exceeding {max_length}"
            )
        labels = [-100] * len(prefix) + target
        return EncodedMathExample(input_ids, labels, len(prefix))

    def encode_generation_prefix(self, problem: str, max_length: int) -> list[int]:
        ids = self.encode(f"Problem: {problem}\nSolution:", add_bos=True)
        ids.append(self.sep_token_id)
        if len(ids) > max_length:
            raise SequenceTooLongError(
                f"math prefix has {len(ids)} tokens, exceeding {max_length}"
            )
        return ids


def pad_1d(
    sequences: list[list[int]],
    pad_value: int,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty sequence collection")
    width = max(len(sequence) for sequence in sequences)
    if max_length is not None and width > max_length:
        raise SequenceTooLongError(f"batch sequence length {width} exceeds {max_length}")
    values = torch.full((len(sequences), width), int(pad_value), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        values[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, : len(sequence)] = 1
    return values, mask
