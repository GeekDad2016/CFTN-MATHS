from __future__ import annotations

import copy
import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from cftn_text.config import load_config
from cftn_text.v2_data import (
    _IntegralFloatRandintProxy,
    _atomic_write_records,
    LOCAL_FAMILIES,
    V2_FORMAT,
    audit_v2_manifest,
    iter_deepmind_records,
    iter_gsm8k_records,
    iter_gsm_symbolic_records,
    iter_local_records,
    iter_mathqa_records,
    prepare_v2_manifests,
    validate_v2_record,
)


def test_atomic_writer_recovers_from_partial_fuse_eio_without_duplicate_rows(
    tmp_path: Path, monkeypatch
):
    destination = tmp_path / "train.jsonl"
    records = list(iter_local_records(count=3, split="train", seed=719))
    real_open = Path.open
    fault = {"injected": False}

    class PartialEioWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def write(self, payload):
            if not fault["injected"]:
                fault["injected"] = True
                self.wrapped.write(bytes(payload[:7]))
                self.wrapped.flush()
                raise OSError(errno.EIO, "simulated transient FUSE failure")
            return self.wrapped.write(payload)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    def flaky_open(
        path,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
    ):
        handle = real_open(path, mode, buffering, encoding, errors, newline)
        if mode == "w+b" and path.name.startswith(".train.jsonl."):
            return PartialEioWriter(handle)
        return handle

    monkeypatch.setattr(Path, "open", flaky_open)
    monkeypatch.setattr("cftn_text.v2_data.time.sleep", lambda _delay: None)

    metadata = _atomic_write_records(destination, records, expected_count=3)
    payload = destination.read_bytes()
    decoded = [json.loads(line) for line in payload.splitlines()]

    assert fault["injected"] is True
    assert [row["record_id"] for row in decoded] == [
        row["record_id"] for row in records
    ]
    assert metadata["count"] == 3
    assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not list(tmp_path.glob(".train.jsonl.*.tmp"))


def test_legacy_randint_proxy_converts_only_integral_float_bounds():
    class Delegate:
        sentinel = object()

        def __init__(self):
            self.bounds = None

        def randint(self, lower, upper):
            self.bounds = (lower, upper)
            return 7

    delegate = Delegate()
    proxy = _IntegralFloatRandintProxy(delegate)

    assert proxy.randint(-50.0, 49.0) == 7
    assert delegate.bounds == (-50, 49)
    assert proxy.sentinel is delegate.sentinel

    try:
        proxy.randint(0.5, 2.0)
    except TypeError as exc:
        assert "non-integral randint bound" in str(exc)
    else:
        raise AssertionError("fractional randint bound was silently accepted")


def test_deepmind_generator_retries_transient_internal_assertions(monkeypatch):
    attempts = 0

    def flaky_generator():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AssertionError("stochastic degenerate sample")
        return SimpleNamespace(question="What is 1 + 1?", answer="2")

    monkeypatch.setattr(
        "cftn_text.v2_data._deepmind_module_pool",
        lambda mode, selected_names: [("arithmetic__add_or_sub", 1, flaky_generator)],
    )
    records = list(
        iter_deepmind_records(
            count=1,
            split="train",
            seed=719,
            mode="train",
            selected_modules=["arithmetic__add_or_sub"],
        )
    )

    assert attempts == 2
    assert len(records) == 1
    assert records[0]["normalized_answer"] == "2"


def test_local_v2_generator_covers_every_requested_family_without_overlap():
    records = list(iter_local_records(count=12, split="train", seed=719))
    assert set(record["family"] for record in records) == set(LOCAL_FAMILIES)
    assert len({record["record_id"] for record in records}) == 12
    assert len({record["content_id"] for record in records}) == 12
    for record in records:
        validate_v2_record(record)
        assert record["gpt_problem"]
        assert record["math_problem"]
        assert record["target_trace"].endswith(record["target_answer"])
        assert record["raw_problem"]
        assert record["native_program"]
        assert record["execution_trace"]


def test_gsm_adapters_keep_training_and_symbolic_evaluation_contracts():
    gsm8k = list(
        iter_gsm8k_records(
            hf_split="train",
            output_split="train",
            count=1,
            dataset_rows=[
                {
                    "question": "Mira has 4 bags with 3 marbles each. How many?",
                    "answer": "There are <<4*3=12>>12 marbles. #### 12",
                }
            ],
        )
    )[0]
    assert gsm8k["normalized_answer"] == "12"
    assert gsm8k["native_program"] == "4*3"
    assert gsm8k["execution_trace"] == "<work>4*3=12</work>"
    assert gsm8k["target_trace"] == "<work>4*3=12</work><answer>12</answer>"
    assert gsm8k["metadata"]["official_split"] == "train"
    mathqa = list(
        iter_mathqa_records(
            hf_split="train",
            output_split="train",
            count=1,
            dataset_rows=[
                {
                    "Problem": "How many ways are there?",
                    "Rationale": "Five choices for four questions.",
                    "options": (
                        "['a ) 24', 'b ) 120', 'c ) 625', "
                        "'d ) 720', 'e ) 1024']"
                    ),
                    "correct": "c",
                    "annotated_formula": "power(5, 4)",
                    "linear_formula": "power(n1,n0)|",
                    "category": "general",
                }
            ],
        )
    )[0]
    assert mathqa["normalized_answer"] == "625"
    assert mathqa["native_program"] == "power(n1,n0)"
    assert mathqa["execution_trace"] == "<program>power(n1,n0)</program>"
    assert mathqa["target_trace"].endswith("<answer>625</answer>")
    assert mathqa["metadata"]["license"] == "Apache-2.0"
    symbolic = list(
        iter_gsm_symbolic_records(
            variant="gsm_symbolic",
            cache_root="unused",
            count=1,
            rows=[
                {
                    "id": "a",
                    "original_id": "b",
                    "question": "A symbolic test asks for 7+5. What is it?",
                    "answer": "Compute it. #### 12",
                    "canary": "do-not-copy",
                }
            ],
        )
    )[0]
    assert symbolic["metadata"]["evaluation_only"] is True
    assert symbolic["metadata"]["license"] == "Apple-Sample-Code-License"
    assert "do-not-copy" not in str(symbolic)


def test_gsm_symbolic_retains_requested_unique_rows_after_source_duplicates():
    rows = [
        {
            "id": "a",
            "instance": index,
            "original_id": "source",
            "question": question,
            "answer": f"Compute it. #### {answer}",
        }
        for index, (question, answer) in enumerate(
            (
                ("What is 2 + 2?", "4"),
                ("What is 2 + 2?", "4"),
                ("What is 3 + 3?", "6"),
            )
        )
    ]

    records = list(
        iter_gsm_symbolic_records(
            variant="gsm_symbolic",
            cache_root="unused",
            count=2,
            rows=rows,
        )
    )

    assert [record["normalized_answer"] for record in records] == ["4", "6"]
    assert [record["metadata"]["official_index"] for record in records] == [0, 2]


def test_small_local_only_v2_manifest_is_immutable_and_auditable(tmp_path: Path):
    source = Path(__file__).parents[1] / "config" / "v2_broad_math.yaml"
    config = copy.deepcopy(load_config(source))
    config.pop("_meta", None)
    config["project"]["data_root"] = str(tmp_path / "data")
    config["project"]["artifact_root"] = str(tmp_path / "artifacts")
    config["data"]["train_examples"] = 12
    config["data"]["training_sources"] = {"local": 12}
    for split, key in (
        ("calibration", "calibration_examples"),
        ("validation", "validation_examples"),
        ("test", "test_examples"),
        ("heldout_language", "heldout_language_examples"),
        ("extrapolation", "extrapolation_examples"),
        ("compositional", "compositional_examples"),
    ):
        config["data"][key] = 6
        config["data"]["split_sources"][split] = {"local": 6}
    config["data"]["curriculum"]["examples_per_epoch"] = 12
    manifest = prepare_v2_manifests(
        config, include_external_benchmarks=False
    )
    assert manifest["format"] == V2_FORMAT
    assert manifest["train_records"] == 12
    result = audit_v2_manifest(manifest, config["project"]["data_root"])
    assert result["pass"] is True
    assert result["training_overlap"] == 0


def test_v2_recommended_400k_mix_and_sealed_mathqa_splits_are_fixed():
    source = Path(__file__).parents[1] / "config" / "v2_broad_math.yaml"
    config = load_config(source)
    assert config["data"]["training_sources"] == {
        "local": 150000,
        "deepmind": 212690,
        "mathqa": 29837,
        "gsm8k": 7473,
    }
    assert sum(config["data"]["training_sources"].values()) == 400000
    assert config["data"]["mathqa_validation_examples"] == 4475
    assert config["data"]["mathqa_test_examples"] == 2985
    assert config["data"]["gsm_symbolic_examples"] == {
        "gsm_symbolic": 4999,
        "gsm_symbolic_p1": 5000,
        "gsm_symbolic_p2": 2492,
    }
    assert config["data"]["max_math_length"] == 4096
    assert config["math_tower"]["max_sequence_length"] == 4096
    assert config["math_tower"]["tokenizer_kind"] == "lossless_utf8_bytes_v1"
