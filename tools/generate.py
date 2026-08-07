from __future__ import annotations

import argparse
import json
from pathlib import Path

from cftn_text.checkpoint import load_checkpoint
from cftn_text.config import config_sha256, load_config
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.training import build_cftn_model, load_data_contract, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CFTN-Text math responses")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-gpt-to-math", action="store_true")
    parser.add_argument("--disable-math-to-gpt", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = Path(
        args.checkpoint
        or Path(config["project"]["artifact_root"])
        / "bridge_bidirectional_contextual"
        / "bridge_bidirectional.best.pth"
    )
    device = resolve_device(args.device)
    _, manifest = load_data_contract(config)
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_config_sha256=config_sha256(config),
        expected_manifest_sha256=manifest["manifest_sha256"],
        map_location=device,
    )
    math_checkpoint = checkpoint["extra"]["math_checkpoint"]
    model, gpt_tokenizer = build_cftn_model(
        config, math_checkpoint, manifest, device
    )
    model.set_trainable_stage(checkpoint["stage"])
    model.load_trainable_state_dict(checkpoint["model_state"], strict=True)
    model.set_gate_mode(checkpoint["extra"].get("gate_mode", "contextual"))
    model.eval()
    outputs = model.generate_problems(
        args.prompt,
        ByteMathTokenizer(),
        gpt_tokenizer,
        max_math_new_tokens=int(config["evaluation"]["max_math_new_tokens"]),
        max_gpt_new_tokens=int(config["evaluation"]["max_gpt_new_tokens"]),
        gpt_to_math_enabled=not args.disable_gpt_to_math,
        math_to_gpt_enabled=not args.disable_math_to_gpt,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
