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
    size_keys = (
        "calibration_examples",
        "train_examples",
        "validation_examples",
        "test_examples",
        "heldout_language_examples",
        "extrapolation_examples",
        "compositional_examples",
    )
    for key in size_keys:
        if key not in data:
            raise ValueError(f"data.{key} is required for {format_name}")
        if int(data[key]) < 1:
            raise ValueError(f"data.{key} must be positive")
    if format_name == "cftn_text_broad_math_v2":
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


def with_data_sizes(config: dict[str, Any], size: int) -> dict[str, Any]:
    """Return a validated small-data copy used by smoke tests and capacity runs."""
    if size < 1:
        raise ValueError("size must be positive")
    result = copy.deepcopy(config)
    for key in (
        "calibration_examples",
        "train_examples",
        "validation_examples",
        "test_examples",
        "heldout_language_examples",
        "extrapolation_examples",
        "compositional_examples",
    ):
        result["data"][key] = size
    result.pop("_meta", None)
    validate_config(result)
    return result
