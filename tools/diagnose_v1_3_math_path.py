from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.data_generator import file_sha256
from cftn_text.tokenizer import ByteMathTokenizer
from cftn_text.training import resolve_device
from cftn_text.v1_3_answer_bus import extract_answer_payload
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_dataset import V13Dataset, V13JointCollator, move_v1_3_batch
from cftn_text.v1_3_evaluation import (
    generate_joint_batch,
    generate_native_specialist,
    resolve_specialist_generation_budget,
)
from cftn_text.v1_3_training import build_v1_3_model, load_v1_3_data_contract
from tools.recover_v1_3_answer_bus import configure_answer_bus_recovery


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standalone math inference with the same record through V1.3 CFTN"
    )
    parser.add_argument("--config", default="G:/ctfn-text/config/v1_3_multi_specialist.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-candidates", type=int, default=64)
    parser.add_argument("--task-class", default="explicit_math")
    parser.add_argument("--specialist", choices=("math", "string"), default="math")
    parser.add_argument(
        "--output",
        default="G:/ctfn-text/artifacts/v1_3_multi_specialist/math_path_diagnostic.json",
    )
    args = parser.parse_args()

    config = load_v1_3_config(args.config)
    root = Path(config["paths"]["artifact_root"])
    native_report = _load_json(
        root / "oracle_hard_answer_bus_recovery" / "native_answer_bus_report.json"
    )
    source = Path(native_report["checkpoint"]).resolve()
    if file_sha256(source) != native_report["checkpoint_sha256"]:
        raise RuntimeError("protected answer-bus checkpoint hash changed")
    configure_answer_bus_recovery(config, source)
    device = resolve_device(args.device)
    model, gpt_tokenizer, provenance = build_v1_3_model(
        config, device=device, collaboration_checkpoint=source
    )
    model.eval()
    tokenizer = ByteMathTokenizer()
    data_root, manifest = load_v1_3_data_contract(config)
    specialist = str(args.specialist)
    candidates = [
        record
        for record in V13Dataset(
            data_root / manifest["splits"]["joint_validation"]["path"]
        ).records
        if record["task_class"] == str(args.task_class)
        and any(
            specialist in required
            for required in record["required_specialists_by_round"][:2]
        )
    ][: int(args.maximum_candidates)]
    round_indices = [
        next(
            index
            for index, required in enumerate(record["required_specialists_by_round"][:2])
            if specialist in required
        )
        for record in candidates
    ]
    direct_items = [
        {
            "problem": record["specialist_oracle_problems_by_round"][specialist][round_index],
            "target_answer": record["specialist_targets_by_round"][specialist][round_index],
        }
        for record, round_index in zip(candidates, round_indices)
    ]
    expected_payloads = [
        extract_answer_payload(str(item["target_answer"]), strict=True)
        for item in direct_items
    ]
    budget = resolve_specialist_generation_budget(
        config, model.specialists[specialist], "configured"
    )
    direct_generations = generate_native_specialist(
        model.specialists[specialist],
        direct_items,
        tokenizer,
        device=device,
        max_new_tokens=budget,
    )
    direct_correct = [
        extract_answer_payload(generation, strict=False) == expected
        for expected, generation in zip(expected_payloads, direct_generations)
    ]
    raw_generations = generate_native_specialist(
        model.specialists[specialist],
        [{"problem": record["problem"]} for record in candidates],
        tokenizer,
        device=device,
        max_new_tokens=budget,
    )
    raw_correct = [
        extract_answer_payload(generation, strict=False) == expected
        for expected, generation in zip(expected_payloads, raw_generations)
    ]
    collator = V13JointCollator(
        tokenizer,
        gpt_tokenizer,
        maximum_gpt_length=int(config["data"]["maximum_gpt_length"]),
        maximum_specialist_length=int(config["data"]["maximum_specialist_length"]),
        maximum_rounds=int(config["runtime"]["maximum_callosal_rounds"]),
        neutral_workspaces=config["runtime"]["neutral_workspaces"],
    )
    current_rows: list[dict] = []
    hybrid_rows: list[dict] = []
    lossless_only_rows: list[dict] = []
    for start in range(0, len(candidates), 8):
        batch_records = candidates[start : start + 8]
        batch = move_v1_3_batch(collator(batch_records), device)
        results = generate_joint_batch(
            model,
            batch,
            tokenizer,
            gpt_tokenizer,
            wake_mode="oracle",
            maximum_rounds=2,
            max_specialist_new_tokens=budget,
            max_gpt_new_tokens=1,
        )
        hybrid = generate_joint_batch(
            model,
            batch,
            tokenizer,
            gpt_tokenizer,
            wake_mode="oracle",
            maximum_rounds=2,
            max_specialist_new_tokens=budget,
            max_gpt_new_tokens=1,
            lossless_request_mode="raw_problem",
        )
        lossless_only = generate_joint_batch(
            model,
            batch,
            tokenizer,
            gpt_tokenizer,
            wake_mode="oracle",
            maximum_rounds=2,
            max_specialist_new_tokens=budget,
            max_gpt_new_tokens=1,
            lossless_request_mode="raw_problem_no_latent",
        )
        current_rows.extend(results)
        hybrid_rows.extend(hybrid)
        lossless_only_rows.extend(lossless_only)
    current_payloads = [
        extract_answer_payload(
            row["specialist_generations"][specialist][round_index], strict=False
        )
        for row, round_index in zip(current_rows, round_indices)
    ]
    hybrid_payloads = [
        extract_answer_payload(
            row["specialist_generations"][specialist][round_index], strict=False
        )
        for row, round_index in zip(hybrid_rows, round_indices)
    ]
    lossless_only_payloads = [
        extract_answer_payload(
            row["specialist_generations"][specialist][round_index], strict=False
        )
        for row, round_index in zip(lossless_only_rows, round_indices)
    ]
    current_correct = [
        payload == expected
        for expected, payload in zip(expected_payloads, current_payloads)
    ]
    hybrid_correct = [
        payload == expected
        for expected, payload in zip(expected_payloads, hybrid_payloads)
    ]
    lossless_only_correct = [
        payload == expected
        for expected, payload in zip(expected_payloads, lossless_only_payloads)
    ]
    selected_index = next(
        (
            index
            for index in range(len(candidates))
            if direct_correct[index]
            and not current_correct[index]
            and lossless_only_correct[index]
        ),
        next(
            (
                index
                for index in range(len(candidates))
                if direct_correct[index] and not current_correct[index]
            ),
            None,
        ),
    )
    selected = None
    if selected_index is not None:
        index = int(selected_index)
        record = candidates[index]
        round_index = round_indices[index]
        selected = {
            "record_id": record["record_id"],
            "task_class": record["task_class"],
            "gpt_prompt": record["gpt_prompt"],
            "specialist": specialist,
            "round": round_index + 1,
            "specialist_oracle_problem": record["specialist_oracle_problems_by_round"][specialist][round_index],
            "expected_trace": record["specialist_targets_by_round"][specialist][round_index],
            "expected_payload": expected_payloads[index],
            "canonical_standalone": {
                "generation": direct_generations[index],
                "payload": extract_answer_payload(direct_generations[index], strict=False),
                "correct": direct_correct[index],
            },
            "raw_prompt_standalone": {
                "generation": raw_generations[index],
                "payload": extract_answer_payload(raw_generations[index], strict=False),
                "correct": raw_correct[index],
            },
            "current_cftn": {
                "generation": current_rows[index]["specialist_generations"][specialist][round_index],
                "payload": current_payloads[index],
                "correct": current_correct[index],
                "wake_probabilities": current_rows[index]["wake"]["probabilities"],
                "wake_activations": current_rows[index]["wake"]["activations"],
            },
            "hybrid_lossless_request_cftn": {
                "generation": hybrid_rows[index]["specialist_generations"][specialist][round_index],
                "payload": hybrid_payloads[index],
                "correct": hybrid_correct[index],
                "request_mode": hybrid_rows[index]["lossless_request_mode"],
                "wake_probabilities": hybrid_rows[index]["wake"]["probabilities"],
                "wake_activations": hybrid_rows[index]["wake"]["activations"],
            },
            "lossless_request_only_cftn": {
                "generation": lossless_only_rows[index]["specialist_generations"][specialist][round_index],
                "payload": lossless_only_payloads[index],
                "correct": lossless_only_correct[index],
                "request_mode": lossless_only_rows[index]["lossless_request_mode"],
                "wake_probabilities": lossless_only_rows[index]["wake"]["probabilities"],
                "wake_activations": lossless_only_rows[index]["wake"]["activations"],
            },
        }
    report = {
        "format": "cftn_text_v1_3_math_path_diagnostic_v1",
        "state": "confirmed_path_failure" if selected is not None else "no_contrast_found",
        "checkpoint": str(source),
        "checkpoint_sha256": file_sha256(source),
        "candidates": len(candidates),
        "task_class": str(args.task_class),
        "specialist": specialist,
        "arms": {
            "canonical_standalone": {
                "correct": sum(direct_correct),
                "accuracy": sum(direct_correct) / max(1, len(direct_correct)),
            },
            "raw_prompt_standalone": {
                "correct": sum(raw_correct),
                "accuracy": sum(raw_correct) / max(1, len(raw_correct)),
            },
            "current_latent_cftn": {
                "correct": sum(current_correct),
                "accuracy": sum(current_correct) / max(1, len(current_correct)),
            },
            "hybrid_lossless_request_cftn": {
                "correct": sum(hybrid_correct),
                "accuracy": sum(hybrid_correct) / max(1, len(hybrid_correct)),
            },
            "lossless_request_only_cftn": {
                "correct": sum(lossless_only_correct),
                "accuracy": sum(lossless_only_correct)
                / max(1, len(lossless_only_correct)),
            },
        },
        "contrast": selected,
        "provenance": provenance,
    }
    atomic_json_dump(report, Path(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
