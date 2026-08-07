from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from cftn_text.config import load_config
from cftn_text.gpt_receiver import FrozenGPT2Tower
from cftn_text.math_tower import MathTower
from cftn_text.model import CFTNTextModel
from cftn_text.tokenizer import ByteMathTokenizer


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict:
    source = Path(__file__).parents[1] / "config" / "v1_linear_equations.yaml"
    config = copy.deepcopy(load_config(source))
    config.pop("_meta", None)
    config["project"]["artifact_root"] = str(tmp_path / "artifacts")
    config["project"]["data_root"] = str(tmp_path / "data")
    for key in (
        "calibration_examples",
        "train_examples",
        "validation_examples",
        "test_examples",
        "heldout_language_examples",
        "extrapolation_examples",
        "compositional_examples",
    ):
        config["data"][key] = 8
    config["math_tower"].update(
        {
            "layers": 2,
            "hidden_size": 32,
            "attention_heads": 4,
            "feed_forward_size": 64,
            "dropout": 0.0,
            "receiver_layers": [0, 1],
        }
    )
    config["bridge"].update(
        {
            "message_tokens": 4,
            "message_width": 32,
            "attention_heads": 4,
            "dropout": 0.0,
            "gate_hidden_size": 32,
        }
    )
    config["gpt"]["receiver_layers"] = [0, 1]
    config["math_training"].update(
        {
            "batch_size": 4,
            "eval_batch_size": 4,
            "max_epochs": 2,
            "minimum_epochs": 1,
            "early_stop_patience": 2,
            "num_workers": 0,
            "precision": "fp32",
        }
    )
    config["bridge_training"].update(
        {
            "batch_size": 2,
            "eval_batch_size": 2,
            "max_epochs": 2,
            "minimum_epochs": 1,
            "early_stop_patience": 2,
            "num_workers": 0,
            "precision": "fp32",
        }
    )
    config["evaluation"].update(
        {
            "batch_size": 2,
            "maximum_generation_examples": 2,
            "max_math_new_tokens": 8,
            "max_gpt_new_tokens": 4,
            "bootstrap_samples": 100,
        }
    )
    return config


@pytest.fixture
def tiny_model(tiny_config: dict) -> CFTNTextModel:
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(3)
    gpt_model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=ByteMathTokenizer.vocab_size,
            n_positions=256,
            n_ctx=256,
            n_embd=32,
            n_layer=2,
            n_head=4,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=ByteMathTokenizer.bos_token_id,
            eos_token_id=ByteMathTokenizer.eos_token_id,
            pad_token_id=ByteMathTokenizer.pad_token_id,
        )
    )
    gpt = FrozenGPT2Tower(
        gpt_model,
        tiny_config["gpt"]["receiver_layers"],
        tiny_config["bridge"],
    )
    math = MathTower(tiny_config["math_tower"], ByteMathTokenizer.vocab_size)
    return CFTNTextModel(math, gpt, tiny_config)
