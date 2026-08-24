from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from cftn_text.checkpoint import atomic_json_dump
from cftn_text.data_generator import file_sha256
from cftn_text.v1_3_config import load_v1_3_config
from cftn_text.v1_3_dataset import V13Dataset
from cftn_text.v1_3_dispatch import (
    DISPATCH_INTENTS,
    compile_v1_3_intent,
    dispatch_intent_from_plan,
    dispatch_v1_3_prompt,
)
from cftn_text.v1_3_data import generate_joint_record
from cftn_text.v1_3_learned_dispatch import (
    ByteIntentClassifier,
    encode_dispatch_prompts,
    load_learned_dispatcher,
    save_learned_dispatcher_checkpoint,
)
from cftn_text.v1_3_training import load_v1_3_data_contract


class PromptIntentDataset(Dataset[tuple[str, int]]):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.prompts = [str(record["problem"]) for record in records]
        self.labels = [
            DISPATCH_INTENTS.index(
                str(record["_dispatch_intent"])
                if "_dispatch_intent" in record
                else dispatch_intent_from_plan(dispatch_v1_3_prompt(prompt))
            )
            for prompt, record in zip(self.prompts, records)
        ]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.prompts[index], self.labels[index]


def _collate(
    items: list[tuple[str, int]], *, maximum_length: int
) -> dict[str, torch.Tensor]:
    prompts, labels = zip(*items)
    input_ids, attention_mask = encode_dispatch_prompts(
        prompts, maximum_length=maximum_length
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@torch.no_grad()
def evaluate_classifier(
    model: ByteIntentClassifier,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda values: _collate(values, maximum_length=maximum_length),
    )
    total = 0
    correct = 0
    confidence_sum = 0.0
    minimum_confidence = 1.0
    minimum_correct_confidence = 1.0
    accepted = 0
    accepted_and_correct = 0
    confusion = torch.zeros(
        len(DISPATCH_INTENTS), len(DISPATCH_INTENTS), dtype=torch.long
    )
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        probabilities = torch.softmax(
            model(input_ids, attention_mask).float(), dim=-1
        )
        confidence, predicted = probabilities.max(dim=-1)
        matches = predicted.eq(labels)
        accepted_mask = confidence >= float(confidence_threshold)
        total += int(labels.numel())
        correct += int(matches.sum())
        accepted += int(accepted_mask.sum())
        accepted_and_correct += int(matches.logical_and(accepted_mask).sum())
        confidence_sum += float(confidence.sum())
        minimum_confidence = min(minimum_confidence, float(confidence.min()))
        if matches.any():
            minimum_correct_confidence = min(
                minimum_correct_confidence,
                float(confidence[matches].min()),
            )
        for target, prediction in zip(labels.tolist(), predicted.tolist()):
            confusion[int(target), int(prediction)] += 1
    return {
        "examples": total,
        "accuracy": correct / max(1, total),
        "coverage_at_threshold": accepted / max(1, total),
        "correct_and_accepted_rate": accepted_and_correct / max(1, total),
        "mean_confidence": confidence_sum / max(1, total),
        "minimum_confidence": minimum_confidence if total else None,
        "minimum_correct_confidence": (
            minimum_correct_confidence if correct else None
        ),
        "confusion": {
            DISPATCH_INTENTS[row]: {
                DISPATCH_INTENTS[column]: int(confusion[row, column])
                for column in range(len(DISPATCH_INTENTS))
                if int(confusion[row, column])
            }
            for row in range(len(DISPATCH_INTENTS))
        },
    }


def _records(
    data_root: Path,
    manifest: dict[str, Any],
    split: str,
    maximum: int | None,
) -> list[dict[str, Any]]:
    values = V13Dataset(data_root / manifest["splits"][split]["path"]).records
    return values if maximum is None else values[: int(maximum)]


def _math_then_string_records(
    config: dict[str, Any], *, split: str, count: int, seed_offset: int
) -> list[dict[str, Any]]:
    seed = int(config["revision"]["seed"]) + int(seed_offset)
    # Indices 20*k+9 are multi-sequential examples in an even ten-record
    # cycle, which selects the repaired math->string dependency order.
    return [
        generate_joint_record(
            seed=seed,
            split=split,
            index=20 * index + 9,
            config=config,
        )
        for index in range(int(count))
    ]


def _semantic_dispatch_records(
    *, count: int, seed: int, evaluation: bool
) -> list[dict[str, Any]]:
    """Create labeled semantic variety without copying official held-out text."""

    rng = random.Random(int(seed))
    labels = ("amber", "birch", "cobalt", "delta", "ember", "fable")
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    training_templates: dict[str, tuple[str, ...]] = {
        "pure_language": (
            "Recall the stored tag {label} and report it, ignoring the blue marker.",
            "The record name is {label}; provide that name and disregard the red swatch.",
            "Return catalog label {label}, not the mentioned colour green.",
            "State archive tag {label} while omitting the yellow annotation.",
            "Provide the registry name {label} while excluding the violet signal.",
            "Give registry name {label} and ignore the grey marker.",
        ),
        "single_math": (
            "Find x when {a} times x plus {b} equals {c}.",
            "Determine the unknown in {a}*x + ({b}) = {c}.",
            "Multiplying an unknown by {a}, then shifting it by {b}, produces {c}; recover the unknown.",
            "The coefficient is {a}, the additive term is {b}, and the total is {c}; solve for x.",
            "Scale a hidden integer by {a}, translate it by {b}, and obtain {c}; identify the integer.",
            "A concealed number scaled by {a} then offset by {b} results in {c}; recover it.",
        ),
        "string_count": (
            "Find the frequency of '{char}' across '{text}'.",
            "Tally occurrences of '{char}' inside '{text}'.",
            "How often is symbol '{char}' present in sequence '{text}'?",
            "Measure the repetition count for '{char}' throughout '{text}'.",
            "Calculate symbol frequency: '{char}' within '{text}'.",
        ),
        "string_reverse": (
            "Mirror the character sequence '{text}'.",
            "Produce '{text}' in reverse character order.",
            "Read '{text}' from right to left.",
            "Invert the symbol order in '{text}'.",
            "Emit the mirrored ordering of characters in '{text}'.",
            "Place the symbols of '{text}' in their opposite order.",
        ),
        "string_index": (
            "Select zero-origin offset {index} from '{text}'.",
            "Fetch the symbol at position {index} in '{text}', counting from zero.",
            "From '{text}', return its zero-based character {index}.",
            "Look up index {index} of '{text}' under zero-based indexing.",
            "Extract zero-based offset {index} from sequence '{text}'.",
            "Index from zero and retrieve offset {index} in '{text}'.",
        ),
        "multi_parallel": (
            "Independently solve {a}*x+({b})={c} and reverse '{text}'; emit x|reversal.",
            "Do two parallel jobs: find x for coefficient {a}, offset {b}, total {c}; mirror '{text}'. Join with |.",
            "Return the linear answer and backwards text: {a}*x+({b})={c}; text '{text}'; format x|text.",
            "Without dependency, solve {a}*x plus {b} equals {c}, and invert '{text}', separated by |.",
        ),
        "string_then_math": (
            "First tally '{char}' in '{text}' and call the count n; then solve {a}*x+n={c}.",
            "Let n be the frequency of '{char}' within '{text}'. Determine x from {a}*x+n={c}.",
            "Count symbol '{char}' across '{text}'. Use that result as the offset in {a}*x+offset={c}; output x.",
            "Compute occurrences of '{char}' throughout '{text}', insert the count into {a}*x+count={c}, and return x.",
        ),
        "math_then_string": (
            "Solve {a}*x+({b})={c}; use x as a zero-based offset into '{text}'.",
            "First determine x from coefficient {a}, shift {b}, and total {c}; fetch index x in '{text}'.",
            "Find integer x satisfying {a} times x plus {b} equals {c}, then select that zero-origin character of '{text}'.",
            "Calculate x for {a}*x+({b})={c}; return '{text}' at position x.",
            "Resolve {a}*x+({b})={c}; treat x as a position counted from zero in '{text}'.",
            "After resolving {a}*x+({b})={c}, interpret the result as a zero-based position in '{text}'.",
        ),
    }
    evaluation_templates: dict[str, tuple[str, ...]] = {
        "pure_language": (
            "Give the registry name {label} and leave out the orange indicator.",
        ),
        "single_math": (
            "A linear unknown is multiplied by {a}, receives {b}, and ends at {c}. What is it?",
        ),
        "string_count": (
            "Compute how frequently '{char}' is found throughout '{text}'.",
        ),
        "string_reverse": (
            "Output the symbols of '{text}' in opposite order.",
        ),
        "string_index": (
            "Retrieve offset {index} from '{text}' with the initial offset equal to zero.",
        ),
        "multi_parallel": (
            "Simultaneously resolve {a}*x+({b})={c} and flip '{text}'; answer x|flipped.",
        ),
        "string_then_math": (
            "Obtain the frequency of '{char}' in '{text}', denote it n, and solve {a}*x+n={c}.",
        ),
        "math_then_string": (
            "Resolve {a}*x+({b})={c}; treat the result as a position in '{text}' counted from zero.",
        ),
    }
    unsupported_training = (
        "Explain why leaves change colour.",
        "Describe the history of the archive term amber.",
        "Discuss the etymology and origin of archive names.",
        "Explain where the term amber originated.",
        "Summarize the background of a colour word.",
        "Translate a greeting into French.",
        "Name a shade resembling cobalt.",
        "Rank the values {a}, {b}, and {c} from largest to smallest.",
        "Compute the product of {a}, {b}, and {c}.",
        "Multiply {a}, {b}, and {c}; no equation or unknown needs solving.",
        "Take the product of {a}, {b}, and {c} rather than recover x.",
        "Find the product of {a}, {b}, and {c}; do not solve a linear variable.",
        "Report the median among {a}, {b}, and {c}.",
        "Sort the letters of '{text}' alphabetically.",
        "Arrange '{text}' in dictionary order, not reverse order.",
        "Alphabetize '{text}' rather than reverse its characters.",
        "Put '{text}' in alphabetical order instead of mirroring it.",
        "Lexically sort '{text}'; reversal is not requested.",
        "Convert every letter of '{text}' to uppercase.",
        "Remove all vowels from '{text}'.",
        "Decide whether '{char}' rhymes with '{text}'.",
        "Concatenate '{char}' and '{text}' without counting them.",
        "Compare '{char}' with '{text}' for equality.",
        "Delete {index} characters from '{text}'.",
        "Erase the symbol at position {index} in '{text}'.",
        "Remove, rather than return, the symbol at offset {index} within '{text}'.",
        "At offset {index} of '{text}', erase the character instead of selecting it.",
        "Drop the character at zero-based offset {index} from '{text}'.",
        "Rotate '{text}' by {index} places.",
        "Repeat '{text}' exactly {index} times.",
        "Add {a}, {b}, and {c}, then append '{text}'.",
        "Total {a}, {b}, and {c}, then prepend that sum to '{text}'.",
        "Use {a}, {b}, and {c} as slice boundaries for '{text}'.",
        "Replace '{char}' in '{text}' exactly {a} times and repeat the result {c} times.",
    )
    unsupported_evaluation = (
        "Discuss the origin of the word amber.",
        "Multiply {a}, {b}, and {c} without solving an equation.",
        "Alphabetize '{text}' rather than reversing it.",
        "Remove the symbol at offset {index} from '{text}'.",
        "Sum {a}, {b}, and {c}, then prefix it to '{text}'.",
    )
    supported = tuple(intent for intent in DISPATCH_INTENTS if intent != "unsupported")
    intents = supported + ("unsupported",)
    records: list[dict[str, Any]] = []
    for index in range(int(count)):
        intent = intents[index % len(intents)]
        length = rng.randint(5, 24)
        text = "".join(rng.choice(alphabet) for _ in range(length))
        char = rng.choice(text)
        selected_index = rng.randrange(length)
        a = rng.choice([value for value in range(-20, 21) if value])
        b = rng.randint(-100, 100)
        x = selected_index if intent == "math_then_string" else rng.randint(-40, 40)
        c = a * x + b
        values = {
            "label": rng.choice(labels),
            "text": text,
            "char": char,
            "index": selected_index,
            "a": a,
            "b": b,
            "c": c,
        }
        if intent == "string_then_math":
            values["c"] = a * x + text.count(char)
        if intent == "unsupported":
            choices = unsupported_evaluation if evaluation else unsupported_training
        else:
            families = evaluation_templates if evaluation else training_templates
            choices = families[intent]
        prompt = choices[(index // len(intents)) % len(choices)].format(**values)
        if intent == "unsupported":
            try:
                dispatch_v1_3_prompt(prompt)
            except Exception:
                pass
            else:
                raise AssertionError("unsupported augmentation entered registered grammar")
        else:
            compile_v1_3_intent(prompt, intent).validate()
        records.append({"problem": prompt, "_dispatch_intent": intent})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the constrained V1.3 learned intent dispatcher"
    )
    parser.add_argument(
        "--config", default="G:/ctfn-text/config/v1_3_multi_specialist.yaml"
    )
    parser.add_argument("--output", default="C:/CFTN/learned_dispatcher_v1_3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-length", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--train-examples", type=int, default=100000)
    parser.add_argument("--validation-examples", type=int, default=5000)
    parser.add_argument("--heldout-examples", type=int, default=5000)
    parser.add_argument(
        "--sequential-augmentation-examples", type=int, default=10000
    )
    parser.add_argument(
        "--evaluation-sequential-augmentation-examples", type=int, default=500
    )
    parser.add_argument(
        "--semantic-augmentation-examples", type=int, default=45000
    )
    parser.add_argument(
        "--semantic-holdout-examples", type=int, default=4500
    )
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    config = load_v1_3_config(args.config)
    seed = int(config["revision"]["seed"]) + 313
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dispatcher training requested but unavailable")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "learned_dispatcher.best.pth"
    summary_path = output / "summary.json"
    data_root, manifest = load_v1_3_data_contract(config)
    train_records = _records(
        data_root, manifest, "joint_train", args.train_examples
    ) + _math_then_string_records(
        config,
        split="joint_train",
        count=int(args.sequential_augmentation_examples),
        seed_offset=10_000,
    ) + _semantic_dispatch_records(
        count=int(args.semantic_augmentation_examples),
        seed=seed + 40_000,
        evaluation=False,
    )
    validation_records = _records(
        data_root, manifest, "joint_validation", args.validation_examples
    ) + _math_then_string_records(
        config,
        split="joint_validation",
        count=int(args.evaluation_sequential_augmentation_examples),
        seed_offset=20_000,
    )
    heldout_records = _records(
        data_root,
        manifest,
        "joint_heldout_paraphrase",
        args.heldout_examples,
    ) + _math_then_string_records(
        config,
        split="joint_heldout_paraphrase",
        count=int(args.evaluation_sequential_augmentation_examples),
        seed_offset=30_000,
    )
    train_dataset = PromptIntentDataset(train_records)
    validation_dataset = PromptIntentDataset(validation_records)
    heldout_dataset = PromptIntentDataset(heldout_records)
    semantic_holdout_dataset = PromptIntentDataset(
        _semantic_dispatch_records(
            count=int(args.semantic_holdout_examples),
            seed=seed + 50_000,
            evaluation=True,
        )
    )
    model = ByteIntentClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        collate_fn=lambda values: _collate(
            values, maximum_length=int(args.maximum_length)
        ),
    )
    started = time.time()
    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0, -1.0, -1.0)
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        correct = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            # The model masks structurally impossible intents. Conventional
            # label smoothing would allocate target mass to those -inf-like
            # classes and create a large constant loss, so use exact labels.
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(labels.numel())
            examples += int(labels.numel())
            correct += int(logits.detach().argmax(dim=-1).eq(labels).sum())
        validation = evaluate_classifier(
            model,
            validation_dataset,
            device=device,
            maximum_length=int(args.maximum_length),
            batch_size=int(args.batch_size),
            confidence_threshold=float(args.confidence_threshold),
        )
        heldout = evaluate_classifier(
            model,
            heldout_dataset,
            device=device,
            maximum_length=int(args.maximum_length),
            batch_size=int(args.batch_size),
            confidence_threshold=float(args.confidence_threshold),
        )
        semantic_holdout = evaluate_classifier(
            model,
            semantic_holdout_dataset,
            device=device,
            maximum_length=int(args.maximum_length),
            batch_size=int(args.batch_size),
            confidence_threshold=float(args.confidence_threshold),
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, examples),
            "train_accuracy": correct / max(1, examples),
            "validation": validation,
            "heldout_paraphrase": heldout,
            "synthetic_semantic_holdout": semantic_holdout,
            "elapsed_seconds": time.time() - started,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        score = (
            min(
                float(validation["correct_and_accepted_rate"]),
                float(heldout["correct_and_accepted_rate"]),
                float(semantic_holdout["correct_and_accepted_rate"]),
            ),
            min(
                float(validation["accuracy"]),
                float(heldout["accuracy"]),
                float(semantic_holdout["accuracy"]),
            ),
            float(heldout["accuracy"]),
            float(validation["mean_confidence"]),
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_learned_dispatcher_checkpoint(
                checkpoint_path,
                model,
                maximum_length=int(args.maximum_length),
                confidence_threshold=float(args.confidence_threshold),
                metadata={
                    "epoch": epoch,
                    "seed": seed,
                    "validation": validation,
                    "heldout_paraphrase": heldout,
                    "synthetic_semantic_holdout": semantic_holdout,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "sequential_augmentation_examples": int(
                        args.sequential_augmentation_examples
                    ),
                },
            )
        else:
            stale_epochs += 1
        if (
            epoch >= 2
            and best_score[0] >= 1.0
            and stale_epochs >= int(args.patience)
        ):
            break

    dispatcher = load_learned_dispatcher(checkpoint_path, device=device)
    final_validation_predictions = dispatcher.predict_intents(
        validation_dataset.prompts
    )
    final_heldout_predictions = dispatcher.predict_intents(heldout_dataset.prompts)
    final_semantic_predictions = dispatcher.predict_intents(
        semantic_holdout_dataset.prompts
    )
    final_validation_accuracy = sum(
        DISPATCH_INTENTS.index(intent) == label
        for (intent, _), label in zip(
            final_validation_predictions, validation_dataset.labels
        )
    ) / len(validation_dataset)
    final_heldout_accuracy = sum(
        DISPATCH_INTENTS.index(intent) == label
        for (intent, _), label in zip(
            final_heldout_predictions, heldout_dataset.labels
        )
    ) / len(heldout_dataset)
    final_validation_coverage = sum(
        score >= float(args.confidence_threshold)
        for _, score in final_validation_predictions
    ) / len(validation_dataset)
    final_heldout_coverage = sum(
        score >= float(args.confidence_threshold)
        for _, score in final_heldout_predictions
    ) / len(heldout_dataset)
    final_semantic_accuracy = sum(
        DISPATCH_INTENTS.index(intent) == label
        for (intent, _), label in zip(
            final_semantic_predictions, semantic_holdout_dataset.labels
        )
    ) / len(semantic_holdout_dataset)
    final_semantic_coverage = sum(
        score >= float(args.confidence_threshold)
        for _, score in final_semantic_predictions
    ) / len(semantic_holdout_dataset)
    summary = {
        "format": "cftn_text_v1_3_learned_dispatcher_training_v1",
        "state": (
            "passed"
            if (
                final_validation_accuracy == 1.0
                and final_heldout_accuracy == 1.0
                and final_validation_coverage == 1.0
                and final_heldout_coverage == 1.0
                and final_semantic_accuracy == 1.0
                and final_semantic_coverage == 1.0
            )
            else "failed_acceptance"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": best_epoch,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "heldout_examples": len(heldout_dataset),
        "semantic_holdout_examples": len(semantic_holdout_dataset),
        "sequential_augmentation_examples": int(
            args.sequential_augmentation_examples
        ),
        "semantic_augmentation_examples": int(
            args.semantic_augmentation_examples
        ),
        "validation_accuracy": final_validation_accuracy,
        "validation_coverage_at_threshold": final_validation_coverage,
        "heldout_paraphrase_accuracy": final_heldout_accuracy,
        "heldout_coverage_at_threshold": final_heldout_coverage,
        "synthetic_semantic_holdout_accuracy": final_semantic_accuracy,
        "synthetic_semantic_holdout_coverage_at_threshold": (
            final_semantic_coverage
        ),
        "confidence_threshold": float(args.confidence_threshold),
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_dump(summary, summary_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
