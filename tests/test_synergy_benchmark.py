from __future__ import annotations

from cftn_text.data_generator import load_records, prepare_manifests
from cftn_text.synergy_benchmark import (
    SYNERGY_BENCHMARK_FORMAT,
    audit_synergy_benchmark,
    load_synergy_rows,
    prepare_synergy_benchmark,
)


def test_synergy_benchmark_is_paired_immutable_and_source_disjoint(
    tiny_config, tmp_path
):
    source_manifest = prepare_manifests(tiny_config)
    protocol = {
        "format": SYNERGY_BENCHMARK_FORMAT,
        "seed": 719,
        "benchmark": {
            "source_splits": [
                "test",
                "heldout_language",
                "extrapolation",
                "compositional",
            ],
            "examples_per_split": 2,
            "distractor_splits": ["compositional"],
        },
    }
    output = tmp_path / "synergy"
    manifest = prepare_synergy_benchmark(
        tiny_config, protocol, output_root=output
    )
    assert manifest["audit"]["pass"]
    assert manifest["total_pairs"] == 8
    assert manifest["total_records"] == 16
    source_equations = set()
    for metadata in source_manifest["splits"].values():
        source_equations.update(
            row["equation_id"]
            for row in load_records(
                tiny_config["project"]["data_root"] + "/" + metadata["path"]
            )
        )
    for metadata in manifest["splits"].values():
        rows = load_synergy_rows(output / metadata["path"])
        for index in range(0, len(rows), 2):
            base, changed = rows[index : index + 2]
            assert base["gpt_problem"] == changed["gpt_problem"]
            assert base["math_problem"] != changed["math_problem"]
            assert changed["equation_id"] not in source_equations
    audited = audit_synergy_benchmark(output / "manifest.json")
    assert audited["manifest_sha256"] == manifest["manifest_sha256"]
