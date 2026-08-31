from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.assemble_quantized_draft_int4_microgate import assemble_bundle


SOURCE_REVISION = "a" * 40
RUN_TAG = "20260831-quantized-draft-int4-stage0-fixture-r1"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_raw_evidence(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir()
    shape = {
        "shape_id": "m1_k1024_n2048_g128",
        "input_features": 1024,
        "output_features": 2048,
        "execution_count": 2,
        "group_size": 128,
        "module_names_sha256": "b" * 64,
    }
    _write_json(
        raw / "environment.json",
        {
            "schema_version": 1,
            "python": "3.11.9",
            "torch": "2.4.1+cu121",
            "cuda": "12.1",
            "device_name": "NVIDIA A100-SXM4-80GB",
        },
    )
    _write_json(
        raw / "shape_manifest.json",
        {
            "schema_version": 1,
            "checkpoint": {
                "path": "/models/Qwen3-0___6B",
                "composite_sha256": "c" * 64,
                "files": [],
            },
            "shapes": [shape],
        },
    )
    with (raw / "microgate_rows.jsonl").open(
        "w",
        encoding="utf-8",
    ) as stream:
        for pair_index in range(200):
            stream.write(json.dumps({
                "shape_id": shape["shape_id"],
                "pair_index": pair_index,
                "arm_order": (
                    ["bf16", "dequant", "fused_int4"]
                    if pair_index % 2 == 0
                    else ["fused_int4", "dequant", "bf16"]
                ),
                "bf16_cuda_ns": 100_000 + pair_index,
                "dequant_cuda_ns": 110_000 + pair_index,
                "fused_int4_cuda_ns": 70_000 + pair_index,
                "bf16_host_submission_ns": 10_000,
                "dequant_host_submission_ns": 12_000,
                "fused_int4_host_submission_ns": 11_000,
                "maximum_absolute_error": 0.01,
                "maximum_relative_error": 0.01,
                "fallback_reason": None,
                "full_dequant_allocation_observed": False,
            }, sort_keys=True))
            stream.write("\n")
    bf16 = 2048 * 1024 * 2 * 2
    packed = (
        2048 * (1024 // 2)
        + 2048 * (1024 // 128) * 4
    ) * 2
    _write_json(
        raw / "memory.json",
        {
            "classification": "PASS",
            "observed_bf16_weight_bytes": bf16,
            "observed_candidate_weight_bytes": packed,
            "minimum_packed_weight_bytes": packed,
            "maximum_candidate_allocated_delta_bytes": 0,
            "full_dequant_allocation_observed": False,
        },
    )
    _write_json(
        raw / "graph.json",
        {
            "classification": "PASS",
            "shapes": [{
                "shape_id": shape["shape_id"],
                "capture_succeeded": True,
                "replay_count": 2,
                "static_pointers_stable": True,
                "maximum_absolute_error": 0.01,
                "maximum_relative_error": 0.01,
            }],
        },
    )
    _write_json(raw / "cleanup.json", {"classification": "CLEAN"})
    return raw


def test_assembler_builds_complete_passing_bundle(tmp_path):
    raw = make_raw_evidence(tmp_path)
    output = tmp_path / "bundle"

    result = assemble_bundle(
        raw_dir=raw,
        output_dir=output,
        source_revision=SOURCE_REVISION,
        run_tag=RUN_TAG,
    )

    assert result["classification"] == "GO_FUSED_INT4_DRAFT_KERNEL"
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["classification"] == result["classification"]
    assert summary["shape_summaries"][0]["pair_count"] == 200
    assert summary["weighted_summary"][
        "candidate_to_bf16_median_ratio"
    ] < 0.75
    for name in (
        "source_identity.json",
        "summary.json",
        "classification.json",
        "independent_verification.json",
    ):
        payload = json.loads((output / name).read_text(encoding="utf-8"))
        assert payload["source_revision"] == SOURCE_REVISION
        assert payload["run_tag"] == RUN_TAG

    entries = (output / "manifest.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    names = [line.split("  ", 1)[1] for line in entries]
    assert sorted(names) == sorted(
        path.name
        for path in output.iterdir()
        if path.name != "manifest.sha256"
    )
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "mutation",
    ("missing", "symlink", "duplicate", "nonfinite"),
)
def test_assembler_rejects_invalid_raw_evidence(tmp_path, mutation):
    raw = make_raw_evidence(tmp_path)
    if mutation == "missing":
        (raw / "memory.json").unlink()
    elif mutation == "symlink":
        target = tmp_path / "outside.json"
        target.write_text("{}\n", encoding="utf-8")
        (raw / "environment.json").unlink()
        (raw / "environment.json").symlink_to(target)
    elif mutation == "duplicate":
        first = (raw / "microgate_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        with (raw / "microgate_rows.jsonl").open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(first + "\n")
    elif mutation == "nonfinite":
        memory = json.loads(
            (raw / "memory.json").read_text(encoding="utf-8")
        )
        memory["maximum_candidate_allocated_delta_bytes"] = float("nan")
        (raw / "memory.json").write_text(
            json.dumps(memory) + "\n",
            encoding="utf-8",
        )
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        assemble_bundle(
            raw_dir=raw,
            output_dir=tmp_path / "bundle",
            source_revision=SOURCE_REVISION,
            run_tag=RUN_TAG,
        )
