"""Per-example, per-role loss for the opt-in verified-procedure experiment."""
from __future__ import annotations

import math
import re
import torch
import torch.nn.functional as F

from .dataset import MathCollator
from .verified_math_data import GROUPS, VERSION, legacy_spans, validate_verified_record


class ComputationCollator(MathCollator):
    def supervision_spans(self, row: dict) -> list[dict]:
        if row.get("schema_version") == VERSION:
            validate_verified_record(row)
            return row["supervision_spans"]
        return legacy_spans(row)

    def __call__(self, records: list[dict]) -> dict:
        if self.target_mode != "full_trace_v1":
            raise ValueError("computation supervision requires full trace targets")
        batch = super().__call__(records)
        roles = torch.full_like(batch["math_labels"], -100)
        for index, row in enumerate(records):
            target = row["target_trace"]
            spans = self.supervision_spans(row)
            prefix = int(batch["math_prefix_lengths"][index])
            size = len(self.tokenizer.encode(target))
            roles[index, prefix:prefix + size + 1] = GROUPS["format"]  # includes EOS
            # Numeric operands/constants are copies unless a verified result span
            # overrides them. UTF-8 byte offsets, not character/token assumptions.
            for match in re.finditer(r"-?\d+(?:\.\d+)?(?:/\d+)?", target):
                start = prefix + len(self.tokenizer.encode(target[:match.start()]))
                end = prefix + len(self.tokenizer.encode(target[:match.end()]))
                roles[index, start:end] = GROUPS["copy"]
            previous_end = 0
            for span in spans:
                start, end = span["start"], span["end"]
                if not (previous_end <= start < end <= len(target)):
                    raise ValueError("overlapping or invalid supervision spans")
                previous_end = end
                begin = prefix + len(self.tokenizer.encode(target[:start]))
                finish = prefix + len(self.tokenizer.encode(target[:end]))
                roles[index, begin:finish] = GROUPS[span["kind"]]
        if not torch.equal(roles.ne(-100), batch["math_labels"].ne(-100)):
            raise ValueError("supervision roles must cover exactly the target tokens")
        batch["math_roles"] = roles
        return batch


def computation_loss(logits: torch.Tensor, labels: torch.Tensor, roles: torch.Tensor,
                     *, weights: tuple[float, float, float] = (0.1, 0.7, 0.2),
                     require_computation: bool = True) -> torch.Tensor:
    """70% newly computed results, 20% copies, 10% syntax; equal example weight.

    Each present role is averaged within an example first. Missing roles are
    renormalized; long traces cannot dominate simply by containing more bytes.
    This is an experimental objective, not a change to the production default.
    """
    if len(weights) != 3 or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("three finite positive role weights required")
    if logits.shape[:2] != labels.shape or roles.shape != labels.shape:
        raise ValueError("incompatible computation-loss shapes")
    targets, groups = labels[:, 1:], roles[:, 1:]
    if not torch.equal(targets.ne(-100), groups.ne(-100)):
        raise ValueError("role mask differs from supervised label mask")
    if bool(((groups != -100) & ((groups < 0) | (groups > 2))).any()):
        raise ValueError("invalid supervision role")
    if require_computation and bool(~groups.eq(GROUPS["compute"]).any(dim=1).all()):
        raise ValueError("every example needs a computation target")
    if not bool(groups.ne(-100).any(dim=1).all()):
        raise ValueError("every example needs a supervised target")
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                            ignore_index=-100, reduction="none")
    total = losses.new_zeros(losses.shape[0])
    present_weights = losses.new_zeros(losses.shape[0])
    for role, weight in enumerate(weights):
        mask = groups.eq(role)
        count = mask.sum(dim=1)
        total += weight * (losses * mask).sum(dim=1) / count.clamp_min(1)
        present_weights += weight * count.gt(0)
    return (total / present_weights).mean()
