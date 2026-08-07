from __future__ import annotations

import copy
import json

import pytest

from cftn_text.data_generator import (
    HELDOUT_TEMPLATE_IDS,
    TRAIN_TEMPLATE_IDS,
    audit_manifest,
    build_records,
    prepare_manifests,
    validate_record,
)


def test_generation_is_deterministic_valid_and_disjoint(tiny_config):
    first = build_records(tiny_config)
    second = build_records(copy.deepcopy(tiny_config))
    assert {
        split: [record.to_dict() for record in rows]
        for split, rows in first.items()
    } == {
        split: [record.to_dict() for record in rows]
        for split, rows in second.items()
    }
    seen = set()
    for rows in first.values():
        for record in rows:
            validate_record(record)
            assert record.equation_id not in seen
            seen.add(record.equation_id)


def test_heldout_language_templates_are_not_training_templates(tiny_config):
    records = build_records(tiny_config)
    train_templates = {record.template_id for record in records["train"]}
    heldout_templates = {record.template_id for record in records["heldout_language"]}
    assert train_templates.issubset(TRAIN_TEMPLATE_IDS)
    assert heldout_templates.issubset(HELDOUT_TEMPLATE_IDS)
    assert train_templates.isdisjoint(heldout_templates)


def test_manifest_hashes_and_split_audit(tiny_config):
    manifest = prepare_manifests(tiny_config)
    audit = audit_manifest(manifest, tiny_config["project"]["data_root"])
    assert audit["pass"]
    assert audit["overlap"] == 0
    assert audit["total_unique_equations"] == 56


def test_manifest_tampering_is_rejected(tiny_config):
    manifest = prepare_manifests(tiny_config)
    changed = dict(manifest)
    changed["total_records"] += 1
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        audit_manifest(changed, tiny_config["project"]["data_root"])
