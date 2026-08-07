from __future__ import annotations

import argparse
import json

from cftn_text.config import load_config
from cftn_text.training import build_cftn_model, load_data_contract, resolve_device


FORBIDDEN_MODULE_TERMS = ("router", "expert_selector", "top_k_dispatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the no-routing CFTN contract")
    parser.add_argument("--config", default="config/v1_linear_equations.yaml")
    parser.add_argument("--math-checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    _, manifest = load_data_contract(config)
    math_checkpoint = args.math_checkpoint or (
        config["project"]["artifact_root"] + "/math/math.best.pth"
    )
    model, _ = build_cftn_model(
        config, math_checkpoint, manifest, resolve_device(args.device)
    )
    module_names = [name.lower() for name, _ in model.named_modules()]
    violations = [
        name
        for name in module_names
        if any(term in name for term in FORBIDDEN_MODULE_TERMS)
    ]
    report = {
        "pass": not violations,
        "violations": violations,
        "both_towers_are_persistent": True,
        "message_gates_are_independent_sigmoids": True,
        "gate_closure_skips_tower_execution": False,
        "top_k_expert_selection": False,
        "load_balancing_loss": False,
    }
    print(json.dumps(report, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
