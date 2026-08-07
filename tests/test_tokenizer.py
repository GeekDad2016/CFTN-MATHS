from __future__ import annotations

import pytest

from cftn_text.tokenizer import ByteMathTokenizer, SequenceTooLongError, pad_1d


def test_byte_tokenizer_round_trip_preserves_signs_and_unicode():
    tokenizer = ByteMathTokenizer()
    text = "Solve -12*x + (7) = −41."
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert tokenizer.decode(ids) == text


def test_training_labels_hide_the_problem_prefix():
    tokenizer = ByteMathTokenizer()
    item = tokenizer.encode_training_example(
        "Solve 7*x + (4) = 53.",
        "<work>7*x=49</work><answer>7</answer>",
        256,
    )
    assert all(label == -100 for label in item.labels[: item.prefix_length])
    assert item.labels[item.prefix_length] != -100
    assert item.input_ids[-1] == tokenizer.eos_token_id


def test_sequence_limit_fails_before_model_indexing():
    tokenizer = ByteMathTokenizer()
    with pytest.raises(SequenceTooLongError):
        tokenizer.encode_training_example("x" * 300, "<answer>1</answer>", 64)


def test_padding_returns_explicit_mask():
    values, mask = pad_1d([[1, 2], [3]], 0)
    assert values.tolist() == [[1, 2], [3, 0]]
    assert mask.tolist() == [[1, 1], [1, 0]]
