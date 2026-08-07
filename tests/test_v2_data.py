from __future__ import annotations

import copy
from pathlib import Path

from cftn_text.config import load_config
from cftn_text.v2_data import (
    LOCAL_FAMILIES,
    V2_FORMAT,
    audit_v2_manifest,
    iter_gsm8k_records,
    iter_gsm_symbolic_records,
    iter_local_records,
    prepare_v2_manifests,
    validate_v2_record,
)


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
    assert gsm8k["metadata"]["official_split"] == "train"
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
    assert symbolic["metadata"]["license"] == "CC-BY-NC-ND-4.0"
    assert "do-not-copy" not in str(symbolic)


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
