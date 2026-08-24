from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {
    "project",
    "data",
    "gpt",
    "math_tower",
    "bridge",
    "math_training",
    "bridge_training",
    "evaluation",
    "monitoring",
    "gpt_calibration",
}

_ENVIRONMENT_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-?(?P<default>[^}]*))?\}"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_sha256(config: dict[str, Any]) -> str:
    clean = copy.deepcopy(config)
    clean.pop("_meta", None)
    return hashlib.sha256(canonical_json(clean).encode("utf-8")).hexdigest()


def _expand_environment(value: Any) -> Any:
    """Expand ${NAME} and ${NAME:-default} without ever persisting secrets."""

    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        default = match.group("default")
        resolved = os.environ.get(name)
        if resolved is not None and resolved != "":
            return resolved
        if default is not None:
            return default
        raise ValueError(f"required environment variable is not set: {name}")

    return _ENVIRONMENT_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    config = _expand_environment(config)
    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"configuration is missing sections: {sorted(missing)}")
    validate_config(config)
    config["_meta"] = {
        "path": str(config_path),
        "sha256": config_sha256(config),
    }
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    format_name = str(data.get("format", "cftn_text_linear_equations_v1"))
    size_keys = [
        "calibration_examples",
        "train_examples",
        "validation_examples",
        "test_examples",
        "heldout_language_examples",
        "extrapolation_examples",
        "compositional_examples",
    ]
    if format_name == "cftn_text_linear_equations_v1_1":
        size_keys.append("answer_extrapolation_examples")
    for key in size_keys:
        if key not in data:
            raise ValueError(f"data.{key} is required for {format_name}")
        if int(data[key]) < 1:
            raise ValueError(f"data.{key} must be positive")
    if format_name == "cftn_text_broad_math_v2":
        gpt = config["gpt"]
        if gpt.get("model_name") != "Qwen/Qwen3-4B-Instruct-2507":
            raise ValueError("V2 coordinator must use the registered dense Qwen checkpoint")
        revision = str(gpt.get("revision", ""))
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision.casefold()
        ):
            raise ValueError("V2 coordinator revision must be a full immutable Git SHA")
        if gpt.get("architecture") != "dense" or gpt.get("require_dense") is not True:
            raise ValueError("V2 coordinator must be explicitly pinned as dense")
        if gpt.get("expected_model_type") != "qwen3":
            raise ValueError("V2 coordinator must use the registered Qwen3 model type")
        if int(gpt.get("expected_hidden_size", 0)) != 2560:
            raise ValueError("V2 Qwen coordinator hidden size differs from the target")
        if int(gpt.get("expected_layers", 0)) != 36:
            raise ValueError("V2 Qwen coordinator layer count differs from the target")
        if gpt.get("use_chat_template") is not True:
            raise ValueError("V2 Qwen coordinator must use its tokenizer chat template")
        receiver_layers = [int(value) for value in gpt.get("receiver_layers", [])]
        if not receiver_layers or any(value < 0 or value >= 36 for value in receiver_layers):
            raise ValueError("V2 Qwen receiver layers are outside the decoder")
        sources = data.get("training_sources", {})
        expected = int(data["train_examples"])
        allocated = sum(int(value) for value in sources.values())
        if allocated != expected:
            raise ValueError(
                "V2 training source allocations must sum to data.train_examples "
                f"({allocated} != {expected})"
            )
        curriculum = data.get("curriculum", {})
        phases = curriculum.get("phases", [])
        if not phases:
            raise ValueError("V2 requires at least one curriculum phase")
        through_epochs = [int(phase["through_epoch"]) for phase in phases]
        if through_epochs != sorted(through_epochs) or len(set(through_epochs)) != len(
            through_epochs
        ):
            raise ValueError("V2 curriculum through_epoch values must increase")
        if through_epochs[-1] != int(config["math_training"]["max_epochs"]):
            raise ValueError("the final V2 curriculum phase must reach max_epochs")
        tower = config["math_tower"]
        if tower.get("tokenizer_kind") != "lossless_utf8_bytes_v1":
            raise ValueError("V2 proof math tower must register its byte tokenizer")
        if tower.get("request_contract") != "raw_utf8_problem_v1":
            raise ValueError("V2 math request contract must preserve the raw prompt")
        if tower.get("result_contract") != "typed_answer_payload_v1":
            raise ValueError("V2 math result contract is not recognized")
        if int(tower["max_sequence_length"]) < 4096:
            raise ValueError("V2 proof math context must be at least 4096 byte tokens")
        generation_validation = config["math_training"].get(
            "generation_validation", {}
        )
        if generation_validation.get("enabled") is not True:
            raise ValueError("V2 math training must enable native generation validation")
        for key in (
            "every_epochs",
            "examples",
            "batch_size",
            "max_new_tokens",
            "failure_examples",
        ):
            if int(generation_validation.get(key, 0)) <= 0:
                raise ValueError(f"V2 generation_validation.{key} must be positive")
        if int(generation_validation["max_new_tokens"]) >= int(
            tower["max_sequence_length"]
        ):
            raise ValueError(
                "V2 generation validation must leave context room for the request"
            )
    elif format_name == "cftn_text_linear_equations_v1_1":
        bands = data.get("numeric_curriculum_bands", [])
        if len(bands) < 2:
            raise ValueError("V1.1 requires at least two numeric curriculum bands")
        previous = (0, 0, 0)
        for index, band in enumerate(bands, start=1):
            maxima = (
                int(band["max_abs_a"]),
                int(band["max_abs_x"]),
                int(band["max_abs_b"]),
            )
            if any(value < prior for value, prior in zip(maxima, previous)):
                raise ValueError("V1.1 numeric curriculum bounds must not decrease")
            if maxima == previous or min(maxima) < 1:
                raise ValueError(
                    f"V1.1 numeric curriculum band {index} must expand the bounds"
                )
            if float(band.get("weight", 0.0)) <= 0.0:
                raise ValueError("V1.1 curriculum weights must be positive")
            if float(band.get("evaluation_weight", band.get("weight", 0.0))) <= 0.0:
                raise ValueError("V1.1 evaluation weights must be positive")
            previous = maxima
        curriculum = data.get("curriculum", {})
        phases = curriculum.get("phases", [])
        if not curriculum.get("enabled", False) or not phases:
            raise ValueError("V1.1 requires an enabled epoch curriculum")
        through_epochs = [int(phase["through_epoch"]) for phase in phases]
        if through_epochs != sorted(through_epochs) or len(set(through_epochs)) != len(
            through_epochs
        ):
            raise ValueError("V1.1 curriculum through_epoch values must increase")
        if through_epochs[-1] != int(config["math_training"]["max_epochs"]):
            raise ValueError("the final V1.1 curriculum phase must reach max_epochs")
        if any(
            int(phase["max_difficulty"]) < 1
            or int(phase["max_difficulty"]) > len(bands)
            for phase in phases
        ):
            raise ValueError("V1.1 curriculum max_difficulty is outside its bands")
    if int(data["max_math_length"]) > int(config["math_tower"]["max_sequence_length"]):
        raise ValueError("data.max_math_length exceeds the math tower context")
    width = int(config["bridge"]["message_width"])
    heads = int(config["bridge"]["attention_heads"])
    if width < 1 or heads < 1 or width % heads:
        raise ValueError("bridge message_width must be divisible by attention_heads")
    math_width = int(config["math_tower"]["hidden_size"])
    math_heads = int(config["math_tower"]["attention_heads"])
    if math_width < 1 or math_width % math_heads:
        raise ValueError("math hidden_size must be divisible by attention_heads")
    if int(config["bridge"]["message_tokens"]) < 1:
        raise ValueError("bridge.message_tokens must be positive")
    if int(config["math_tower"]["answer_min"]) >= int(config["math_tower"]["answer_max"]):
        raise ValueError("math answer range is invalid")
    answer_head_mode = str(config["math_tower"].get("answer_head_mode", "categorical"))
    if answer_head_mode not in {"categorical", "disabled"}:
        raise ValueError("math_tower.answer_head_mode must be categorical or disabled")
    for section_name in ("specialist_acceptance", "collaboration_acceptance"):
        allowed_metrics = (
            {"exact_accuracy", "valid_rate", "trace_exact_rate"}
            if section_name == "specialist_acceptance"
            else {"exact_accuracy", "valid_rate"}
        )
        for criteria in config["evaluation"].get(section_name, {}).values():
            for metric, threshold in criteria.items():
                if metric not in allowed_metrics:
                    raise ValueError(f"unsupported {section_name} metric: {metric}")
                if not 0.0 <= float(threshold) <= 1.0:
                    raise ValueError(f"{section_name} thresholds must be in [0, 1]")


def with_data_sizes(config: dict[str, Any], size: int) -> dict[str, Any]:
    """Return a validated small-data copy used by smoke tests and capacity runs."""
    if size < 1:
        raise ValueError("size must be positive")
    result = copy.deepcopy(config)
    keys = [
        "calibration_examples",
        "train_examples",
        "validation_examples",
        "test_examples",
        "heldout_language_examples",
        "extrapolation_examples",
        "compositional_examples",
    ]
    if "answer_extrapolation_examples" in result["data"]:
        keys.append("answer_extrapolation_examples")
    for key in keys:
        result["data"][key] = size
    result.pop("_meta", None)
    validate_config(result)
    return result
