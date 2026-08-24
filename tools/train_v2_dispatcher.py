from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from cftn_text.checkpoint import append_jsonl, atomic_json_dump
from cftn_text.config import load_config
from cftn_text.data_generator import file_sha256
from cftn_text.dataset import CFTNCollator
from cftn_text.semantic_features import (
    FrozenSemanticFeatureExtractor,
    load_or_create_semantic_cache,
)
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_data import audit_v1_3_manifest
from cftn_text.v1_3_dataset import V13Dataset
from cftn_text.v2_data import audit_v2_manifest, validate_v2_record
from cftn_text.v2_dispatch import (
    DISPATCH_INTENTS,
    compile_v2_intent,
    dispatch_v2_intent_from_registered_prompt,
)
from cftn_text.v2_learned_dispatch import (
    HierarchicalDispatcherModel,
    dispatcher_parameter_count,
    encode_dispatch_prompts,
    hierarchy_targets,
    load_learned_dispatcher,
    save_learned_dispatcher_checkpoint,
)


DispatcherRow = tuple[str, str, str]


def _default_semantic_prompt(prompt: str) -> str:
    return f"Problem: {prompt}\nExact result:"


class PromptIntentDataset(Dataset[tuple[str, int, torch.Tensor]]):
    def __init__(self, values: Iterable[DispatcherRow | tuple[str, str]]) -> None:
        rows: list[DispatcherRow] = []
        for value in values:
            if len(value) == 2:
                prompt, intent = value
                semantic_prompt = _default_semantic_prompt(str(prompt))
            else:
                prompt, intent, semantic_prompt = value
            rows.append((str(prompt), str(intent), str(semantic_prompt)))
        self.prompts = [prompt for prompt, _, _ in rows]
        self.labels = [DISPATCH_INTENTS.index(intent) for _, intent, _ in rows]
        self.semantic_prompts = [semantic for _, _, semantic in rows]
        self.semantic_features: torch.Tensor | None = None

    def attach_semantic_features(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[0] != len(self):
            raise ValueError("semantic feature cache does not align with dispatcher rows")
        self.semantic_features = features.detach().cpu()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> tuple[str, int, torch.Tensor]:
        if self.semantic_features is None:
            raise RuntimeError("dispatcher semantic features have not been attached")
        return self.prompts[index], self.labels[index], self.semantic_features[index]


def _collate(
    items: list[tuple[str, int, torch.Tensor]], *, maximum_length: int
) -> dict[str, torch.Tensor]:
    prompts, labels, semantic_features = zip(*items)
    input_ids, attention_mask = encode_dispatch_prompts(
        prompts, maximum_length=maximum_length
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "semantic_features": torch.stack(semantic_features),
    }


def _read_v2_split(
    root: Path, manifest: dict[str, Any], split: str, maximum: int
) -> list[DispatcherRow]:
    path = root / str(manifest["splits"][split]["path"])
    rows: list[DispatcherRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= int(maximum):
                break
            record = json.loads(line)
            validate_v2_record(record)
            problem = str(record["problem"])
            rows.append(
                (
                    problem,
                    "broad_math",
                    CFTNCollator.gpt_prompt(problem, generic_answer=True),
                )
            )
    if len(rows) != min(int(maximum), int(manifest["splits"][split]["examples"])):
        raise RuntimeError(f"V2 dispatcher could not read requested {split} records")
    return rows


def _read_joint_split(
    root: Path, manifest: dict[str, Any], split: str, maximum: int
) -> list[DispatcherRow]:
    records = V13Dataset(root / str(manifest["splits"][split]["path"])).records
    selected = records[: int(maximum)]
    return [
        (
            str(record["problem"]),
            dispatch_v2_intent_from_registered_prompt(str(record["problem"])),
            str(record["gpt_prompt"]),
        )
        for record in selected
    ]


def _semantic_rows(*, count: int, seed: int, heldout: bool) -> list[DispatcherRow]:
    """Generate semantic controls independent of official validation text."""

    rng = random.Random(int(seed))
    labels = ("amber", "birch", "cobalt", "delta", "ember", "fable")
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    train_templates: dict[str, tuple[str, ...]] = {
        "pure_language": (
            "Recall registry name {label} and ignore the blue marker.",
            "Report archive tag {label}, excluding the red annotation.",
        ),
        "broad_math": (
            "Compute the greatest common divisor of {a} and {c}.",
            "What is {a} multiplied by {b}, then increased by {c}?",
            "Differentiate {a}*x^2 + {b}*x + {c} with respect to x.",
        ),
        "single_math": (
            "Find x when {a} times x plus {b} equals {c}.",
            "A hidden integer scaled by {a} then offset by {b} gives {c}; recover it.",
        ),
        "string_count": (
            "Tally occurrences of '{char}' inside '{text}'.",
            "How often is '{char}' present in sequence '{text}'?",
        ),
        "string_reverse": (
            "Produce '{text}' in reverse character order.",
            "Read '{text}' from right to left.",
        ),
        "string_index": (
            "Fetch zero-origin offset {index} from '{text}'.",
            "Return position {index} in '{text}', counting from zero.",
        ),
        "multi_parallel": (
            "Independently solve {a}*x+({b})={c} and reverse '{text}'; emit x|reversal.",
            "Find x for coefficient {a}, offset {b}, total {c}; also mirror '{text}'. Join with |.",
        ),
        "string_then_math": (
            "First tally '{char}' in '{text}' and call the count n; then solve {a}*x+n={c}.",
            "Use the frequency of '{char}' within '{text}' as n in {a}*x+n={c}; output x.",
        ),
        "math_then_string": (
            "Solve {a}*x+({b})={c}; use x as a zero-based offset into '{text}'.",
            "Find x from coefficient {a}, shift {b}, total {c}; fetch index x in '{text}'.",
        ),
        "unsupported": (
            "Explain why leaves change colour.",
            "Translate a greeting into French.",
            "Alphabetize '{text}' instead of reversing it.",
            "Convert every letter of '{text}' to uppercase.",
            "Remove the symbol at offset {index} from '{text}'.",
            "Discuss whether '{char}' rhymes with '{text}'.",
        ),
    }
    heldout_templates = {
        "pure_language": ("Give registry name {label} and omit the orange indicator.",),
        "broad_math": ("Determine the least common multiple of {a} and {c}.",),
        "single_math": ("An unknown multiplied by {a}, shifted by {b}, reaches {c}. Find it.",),
        "string_count": ("Compute the frequency of '{char}' throughout '{text}'.",),
        "string_reverse": ("Output the symbols of '{text}' in opposite order.",),
        "string_index": ("Retrieve offset {index} from '{text}' with origin zero.",),
        "multi_parallel": ("Simultaneously resolve {a}*x+({b})={c} and flip '{text}'; answer x|flipped.",),
        "string_then_math": ("Let n be how often '{char}' occurs in '{text}', then solve {a}*x+n={c}.",),
        "math_then_string": ("Resolve {a}*x+({b})={c}; use the result as a zero-origin position in '{text}'.",),
        "unsupported": ("Sort '{text}' lexically rather than mirror it.",),
    }
    templates = heldout_templates if heldout else train_templates
    rows: list[DispatcherRow] = []
    for index in range(int(count)):
        intent = DISPATCH_INTENTS[index % len(DISPATCH_INTENTS)]
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(6, 24)))
        char = rng.choice(text)
        position = rng.randrange(len(text))
        a = rng.choice([value for value in range(-20, 21) if value])
        b = rng.randint(-100, 100)
        x = position if intent == "math_then_string" else rng.randint(-40, 40)
        c = a * x + b
        if intent == "string_then_math":
            c = a * x + text.count(char)
        values = {
            "label": rng.choice(labels),
            "text": text,
            "char": char,
            "index": position,
            "a": a,
            "b": b,
            "c": c,
        }
        choices = templates[intent]
        prompt = choices[(index // len(DISPATCH_INTENTS)) % len(choices)].format(
            **values
        )
        if intent != "unsupported":
            compile_v2_intent(prompt, intent).validate()
        rows.append((prompt, intent, _default_semantic_prompt(prompt)))
    return rows


@torch.no_grad()
def evaluate_classifier(
    model: HierarchicalDispatcherModel,
    dataset: PromptIntentDataset,
    *,
    device: torch.device,
    maximum_length: int,
    batch_size: int,
    confidence_threshold: float,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda values: _collate(values, maximum_length=maximum_length),
    )
    total = correct = accepted = accepted_correct = 0
    confidence_sum = 0.0
    confusion = torch.zeros(
        len(DISPATCH_INTENTS), len(DISPATCH_INTENTS), dtype=torch.long
    )
    for batch in loader:
        labels = batch["labels"].to(device)
        probabilities = torch.softmax(
            model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["semantic_features"].to(device),
            ).float(),
            dim=-1,
        )
        confidence, prediction = probabilities.max(dim=-1)
        matches = prediction.eq(labels)
        accepted_mask = confidence.ge(float(confidence_threshold))
        total += int(labels.numel())
        correct += int(matches.sum())
        accepted += int(accepted_mask.sum())
        accepted_correct += int(matches.logical_and(accepted_mask).sum())
        confidence_sum += float(confidence.sum())
        for target, predicted in zip(labels.tolist(), prediction.tolist()):
            confusion[int(target), int(predicted)] += 1
    return {
        "examples": total,
        "accuracy": correct / max(1, total),
        "coverage_at_threshold": accepted / max(1, total),
        "correct_and_accepted_rate": accepted_correct / max(1, total),
        "mean_confidence": confidence_sum / max(1, total),
        "confusion": {
            DISPATCH_INTENTS[row]: {
                DISPATCH_INTENTS[column]: int(confusion[row, column])
                for column in range(len(DISPATCH_INTENTS))
                if int(confusion[row, column])
            }
            for row in range(len(DISPATCH_INTENTS))
        },
    }


def _contracts(base: dict[str, Any], revision: dict[str, Any]) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    v2_root = Path(base["project"]["data_root"])
    if not v2_root.is_absolute():
        v2_root = Path(base["_meta"]["path"]).parent.parent / v2_root
    v2_root = v2_root.resolve()
    joint_root = Path(revision["paths"]["data_root"])
    v2_manifest = json.loads((v2_root / "manifest.json").read_text(encoding="utf-8"))
    joint_manifest = json.loads((joint_root / "manifest.json").read_text(encoding="utf-8"))
    audit_v2_manifest(v2_manifest, v2_root)
    audit_v1_3_manifest(revision, joint_root)
    return v2_root, v2_manifest, joint_root, joint_manifest


def _dispatcher_model(settings: dict[str, Any]) -> HierarchicalDispatcherModel:
    model = dict(settings["model"])
    return HierarchicalDispatcherModel(
        semantic_width=int(model["semantic_width"]),
        semantic_projection_size=int(model["semantic_projection_size"]),
        structure_projection_size=int(model["structure_projection_size"]),
        fusion_size=int(model["fusion_size"]),
        tower_names=tuple(model["tower_names"]),
        active_tower_names=tuple(model["active_tower_names"]),
        embedding_size=int(model.get("embedding_size", 64)),
        channels=int(model.get("channels", 96)),
        kernels=tuple(int(value) for value in model.get("kernels", (3, 5, 7))),
        dropout=float(model.get("dropout", 0.1)),
        hierarchy_weight=float(model.get("hierarchy_weight", 0.25)),
    )


def _training_loss(
    model: HierarchicalDispatcherModel,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    weights: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    labels = batch["labels"].to(device)
    values = model.hierarchical_logits(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["semantic_features"].to(device),
    )
    logits = model.combined_intent_logits(values)
    targets = hierarchy_targets(
        labels,
        tower_names=model.tower_names,
        active_tower_names=model.active_tower_names,
    )
    intent_loss = torch.nn.functional.cross_entropy(logits, labels)
    delegation_loss = torch.nn.functional.cross_entropy(
        values["delegation"], targets["delegation"].to(device)
    )
    round_loss = torch.nn.functional.cross_entropy(
        values["rounds"], targets["rounds"].to(device)
    )
    tower_values = torch.nn.functional.binary_cross_entropy_with_logits(
        values["towers"], targets["towers"].to(device), reduction="none"
    )
    tower_mask = targets["tower_mask"].to(device)
    tower_loss = tower_values.masked_select(tower_mask).mean()
    total = (
        float(weights["intent"]) * intent_loss
        + float(weights["delegation"]) * delegation_loss
        + float(weights["towers"]) * tower_loss
        + float(weights["rounds"]) * round_loss
    )
    components = {
        "intent": float(intent_loss.detach()),
        "delegation": float(delegation_loss.detach()),
        "towers": float(tower_loss.detach()),
        "rounds": float(round_loss.detach()),
    }
    return total, logits, components


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the constrained hierarchical V2 prompt dispatcher"
    )
    parser.add_argument("--config", default="config/v2_broad_math.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = load_config(args.config)
    revision_path = Path(base["multi_specialist"]["revision_config"])
    if not revision_path.is_absolute():
        revision_path = Path(base["_meta"]["path"]).parent.parent / revision_path
    revision = load_v1_3_config(revision_path)
    settings = revision["dispatcher"]
    seed = int(revision["revision"]["seed"]) + 313
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA V2 dispatcher training requested but unavailable")

    artifact = Path(revision["paths"]["artifact_root"]) / str(
        settings["artifact_directory"]
    )
    artifact.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact / "learned_dispatcher.best.pth"
    summary_path = artifact / "summary.json"
    status_path = artifact / "status.json"
    metrics_path = artifact / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    v2_root, v2_manifest, joint_root, joint_manifest = _contracts(base, revision)

    train_rows = _read_v2_split(
        v2_root, v2_manifest, "train", int(settings["broad_train_examples"])
    ) + _read_joint_split(
        joint_root,
        joint_manifest,
        "joint_train",
        int(settings["joint_train_examples"]),
    ) + _semantic_rows(
        count=int(settings["semantic_train_examples"]), seed=seed + 40_000, heldout=False
    )
    panels = {
        "registered_validation": PromptIntentDataset(
            _read_joint_split(
                joint_root,
                joint_manifest,
                "joint_validation",
                int(settings["registered_validation_examples"]),
            )
        ),
        "registered_heldout": PromptIntentDataset(
            _read_joint_split(
                joint_root,
                joint_manifest,
                "joint_heldout_paraphrase",
                int(settings["registered_heldout_examples"]),
            )
        ),
        "broad_validation": PromptIntentDataset(
            _read_v2_split(
                v2_root,
                v2_manifest,
                "validation",
                int(settings["broad_validation_examples"]),
            )
        ),
        "broad_heldout": PromptIntentDataset(
            _read_v2_split(
                v2_root,
                v2_manifest,
                "heldout_language",
                int(settings["broad_heldout_examples"]),
            )
        ),
        "semantic_holdout": PromptIntentDataset(
            _semantic_rows(
                count=int(settings["semantic_holdout_examples"]),
                seed=seed + 50_000,
                heldout=True,
            )
        ),
    }
    train_dataset = PromptIntentDataset(train_rows)
    maximum_length = int(settings["maximum_length"])
    batch_size = int(settings["batch_size"])
    confidence_threshold = float(settings["confidence_threshold"])
    model = _dispatcher_model(settings)
    parameter_count = dispatcher_parameter_count(model)
    parameter_target = int(settings["model"]["parameter_target"])
    parameter_tolerance = int(settings["model"]["parameter_tolerance"])
    if abs(parameter_count - parameter_target) > parameter_tolerance:
        raise RuntimeError(
            f"dispatcher has {parameter_count:,} parameters, outside target "
            f"{parameter_target:,} +/- {parameter_tolerance:,}"
        )
    semantic_settings = settings["semantic_encoder"]
    gpt = base["gpt"]
    started = time.time()
    atomic_json_dump(
        {
            "format": "cftn_text_v2_dispatcher_status_v2",
            "state": "extracting_frozen_semantic_features",
            "pid": __import__("os").getpid(),
            "model_name": gpt["model_name"],
            "revision": gpt.get("revision"),
            "parameter_count": parameter_count,
        },
        status_path,
    )
    extractor = FrozenSemanticFeatureExtractor(
        model_name=str(gpt["model_name"]),
        revision=gpt.get("revision"),
        device=device,
        dtype=gpt.get("dtype"),
        local_files_only=bool(gpt.get("local_files_only", True)),
        trust_remote_code=bool(gpt.get("trust_remote_code", False)),
        attn_implementation=gpt.get("attn_implementation"),
        expected_model_type=gpt.get("expected_model_type"),
        expected_hidden_size=gpt.get("expected_hidden_size"),
        expected_layers=gpt.get("expected_layers"),
        require_dense=bool(gpt.get("require_dense", True)),
        maximum_length=int(semantic_settings["maximum_length"]),
        use_chat_template=bool(semantic_settings["use_chat_template"]),
    )
    if extractor.semantic_width != model.semantic_width:
        raise RuntimeError(
            "dispatcher semantic width differs from the frozen coordinator"
        )
    feature_contracts: dict[str, Any] = {}
    feature_datasets = {"train": train_dataset, **panels}
    for name, dataset in feature_datasets.items():
        features, contract = load_or_create_semantic_cache(
            artifact / "semantic_cache" / f"{name}.pth",
            dataset.semantic_prompts,
            extractor=extractor,
            model_name=str(gpt["model_name"]),
            revision=gpt.get("revision"),
            maximum_length=int(semantic_settings["maximum_length"]),
            use_chat_template=bool(semantic_settings["use_chat_template"]),
            batch_size=int(semantic_settings["batch_size"]),
        )
        dataset.attach_semantic_features(features)
        feature_contracts[name] = contract
    del extractor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=lambda values: _collate(values, maximum_length=maximum_length),
    )
    best_score: tuple[float, ...] = (-1.0,)
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        component_sums = {name: 0.0 for name in ("intent", "delegation", "towers", "rounds")}
        examples = correct = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, logits, components = _training_loss(
                model,
                batch,
                device=device,
                weights=settings["losses"],
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("V2 dispatcher produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(labels.numel())
            for name, value in components.items():
                component_sums[name] += value * int(labels.numel())
            examples += int(labels.numel())
            correct += int(logits.detach().argmax(dim=-1).eq(labels).sum())
        evaluations = {
            name: evaluate_classifier(
                model,
                panel,
                device=device,
                maximum_length=maximum_length,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
            )
            for name, panel in panels.items()
        }
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, examples),
            "train_loss_components": {
                name: value / max(1, examples)
                for name, value in component_sums.items()
            },
            "train_accuracy": correct / max(1, examples),
            "evaluations": evaluations,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        append_jsonl(row, metrics_path)
        atomic_json_dump(
            {
                "format": "cftn_text_v2_dispatcher_status_v2",
                "state": "training",
                "pid": __import__("os").getpid(),
                "epoch": epoch,
                "global_step": epoch * len(train_loader),
                "metrics": row,
                "elapsed_seconds": time.time() - started,
            },
            status_path,
        )
        print(json.dumps(row), flush=True)
        score = tuple(
            min(
                float(value["accuracy"]),
                float(value["coverage_at_threshold"]),
            )
            for value in evaluations.values()
        )
        if min(score) > min(best_score) or (min(score) == min(best_score) and score > best_score):
            best_score = score
            best_epoch = epoch
            stale = 0
            save_learned_dispatcher_checkpoint(
                checkpoint_path,
                model,
                maximum_length=maximum_length,
                confidence_threshold=confidence_threshold,
                metadata={
                    "epoch": epoch,
                    "seed": seed,
                    "base_manifest_sha256": v2_manifest["manifest_sha256"],
                    "joint_manifest_sha256": joint_manifest["manifest_sha256"],
                    "base_config_sha256": base["_meta"]["sha256"],
                    "revision_sha256": revision["_meta"]["sha256"],
                    "semantic_encoder": {
                        "model_name": gpt["model_name"],
                        "revision": gpt.get("revision"),
                        "use_chat_template": bool(
                            semantic_settings["use_chat_template"]
                        ),
                        "pooling": "attention_mask_mean_final_hidden_v1",
                    },
                    "feature_contracts": feature_contracts,
                    "parameter_count": parameter_count,
                    "evaluations": evaluations,
                },
            )
        else:
            stale += 1
        if epoch >= 2 and min(best_score) >= 1.0 and stale >= int(settings["patience"]):
            break

    dispatcher = load_learned_dispatcher(checkpoint_path, device=device)
    final: dict[str, Any] = {}
    for name, panel in panels.items():
        predictions = dispatcher.predict_intents(
            panel.prompts, panel.semantic_features
        )
        correct = sum(
            DISPATCH_INTENTS.index(intent) == label
            for (intent, _), label in zip(predictions, panel.labels)
        )
        covered = sum(score >= confidence_threshold for _, score in predictions)
        final[name] = {
            "examples": len(panel),
            "accuracy": correct / max(1, len(panel)),
            "coverage_at_threshold": covered / max(1, len(panel)),
        }
    acceptance = settings["acceptance"]
    gates = {
        "registered_validation_accuracy": final["registered_validation"]["accuracy"]
        >= float(acceptance["minimum_registered_accuracy"]),
        "registered_validation_coverage": final["registered_validation"]["coverage_at_threshold"]
        >= float(acceptance["minimum_registered_coverage"]),
        "registered_heldout_accuracy": final["registered_heldout"]["accuracy"]
        >= float(acceptance["minimum_registered_accuracy"]),
        "registered_heldout_coverage": final["registered_heldout"]["coverage_at_threshold"]
        >= float(acceptance["minimum_registered_coverage"]),
        "broad_validation_accuracy": final["broad_validation"]["accuracy"]
        >= float(acceptance["minimum_broad_accuracy"]),
        "broad_validation_coverage": final["broad_validation"]["coverage_at_threshold"]
        >= float(acceptance["minimum_broad_coverage"]),
        "broad_heldout_accuracy": final["broad_heldout"]["accuracy"]
        >= float(acceptance["minimum_broad_accuracy"]),
        "broad_heldout_coverage": final["broad_heldout"]["coverage_at_threshold"]
        >= float(acceptance["minimum_broad_coverage"]),
        "semantic_accuracy": final["semantic_holdout"]["accuracy"]
        >= float(acceptance["minimum_semantic_accuracy"]),
        "semantic_coverage": final["semantic_holdout"]["coverage_at_threshold"]
        >= float(acceptance["minimum_semantic_coverage"]),
    }
    gates["pass"] = all(gates.values())
    summary = {
        "format": "cftn_text_v2_hierarchical_dispatcher_training_v2",
        "state": "passed" if gates["pass"] else "failed_acceptance",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": best_epoch,
        "confidence_threshold": confidence_threshold,
        "architecture": model.architecture,
        "parameter_count": parameter_count,
        "semantic_encoder": {
            "model_name": gpt["model_name"],
            "revision": gpt.get("revision"),
            "use_chat_template": bool(semantic_settings["use_chat_template"]),
            "pooling": "attention_mask_mean_final_hidden_v1",
        },
        "feature_contracts": feature_contracts,
        "train_examples": len(train_dataset),
        "base_manifest_sha256": v2_manifest["manifest_sha256"],
        "joint_manifest_sha256": joint_manifest["manifest_sha256"],
        "base_config_sha256": base["_meta"]["sha256"],
        "revision_sha256": revision["_meta"]["sha256"],
        "final_panels": final,
        "acceptance": {"gates": gates, "thresholds": acceptance},
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_dump(summary, summary_path)
    atomic_json_dump(
        {
            "format": "cftn_text_v2_dispatcher_status_v2",
            "state": summary["state"],
            "epoch": history[-1]["epoch"],
            "global_step": history[-1]["epoch"] * len(train_loader),
            "elapsed_seconds": time.time() - started,
            "summary": str(summary_path.resolve()),
        },
        status_path,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not gates["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
