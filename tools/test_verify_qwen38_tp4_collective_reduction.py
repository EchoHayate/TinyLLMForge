from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tools.assemble_qwen38_tp4_collective_reduction import (
    MANIFEST_SCHEMA,
    assemble_bundle,
)
from tools.test_assemble_qwen38_tp4_collective_reduction import (
    _inputs,
)
from tools.verify_qwen38_tp4_collective_reduction import (
    verify_bundle,
)
import tools.verify_qwen38_tp4_collective_reduction as verifier_module


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line
    ]


def _write_jsonl(path, rows):
    Path(path).write_text("".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ))


def _rewrite_manifest(root):
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.sha256"
    }
    _write_json(root / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _mutate_json(root, name, mutate):
    path = root / name
    payload = _read_json(path)
    mutate(payload)
    _write_json(path, payload)


def _mutate_jsonl(root, name, mutate):
    path = root / name
    rows = _read_jsonl(path)
    mutate(rows)
    _write_jsonl(path, rows)


def test_independent_verifier_reconstructs_classification_and_manifest(
    tmp_path,
):
    assemble_bundle(output_root=tmp_path, **_inputs())

    first = verify_bundle(tmp_path)
    first_bytes = (
        tmp_path / "independent_verification.json"
    ).read_bytes()
    second = verify_bundle(tmp_path)

    assert first == second
    assert first["status"] == "PASS"
    assert first["producer_classification"] == (
        "GO_SYNC_COLLECTIVE_REDUCTION"
    )
    assert first["reconstructed_classification"] == (
        first["producer_classification"]
    )
    assert (
        tmp_path / "independent_verification.json"
    ).read_bytes() == first_bytes
    manifest = _read_json(tmp_path / "manifest.sha256")
    assert "independent_verification.json" in manifest["artifacts"]


def test_independent_verifier_does_not_import_producer_assembler():
    source = inspect.getsource(verifier_module)

    assert "import assemble_qwen38_tp4_collective_reduction" not in source
    assert "from tools.assemble_qwen38_tp4_collective_reduction" not in source


def test_verifier_accepts_terminal_profiler_overhead_stop(tmp_path):
    inputs = _inputs()
    inputs["cases"] = [
        case
        for case in inputs["cases"]
        if case["campaign_phase"] == "calibration"
    ]
    for case in inputs["cases"]:
        if case["budget"] != 0:
            by_arm = {row["arm"]: row for row in case["arms"]}
            by_arm["instrumented"]["decode_time_ns"] = 1_060_000
    case_ids = {case["case_id"] for case in inputs["cases"]}
    inputs["resource_samples"] = [
        row
        for row in inputs["resource_samples"]
        if row["case_id"] in case_ids
    ]
    assemble_bundle(output_root=tmp_path, **inputs)

    result = verify_bundle(tmp_path)

    assert result["status"] == "PASS"
    assert result["selected_event_budget"] is None
    assert result["producer_classification"] == (
        "INCONCLUSIVE_PROFILER_OVERHEAD"
    )


def test_verifier_rejects_every_evidence_authority_mutation(tmp_path):
    assemble_bundle(output_root=tmp_path, **_inputs())
    original = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    mutations = (
        (
            "source identity",
            lambda: _mutate_json(
                tmp_path,
                "source_identity.json",
                lambda payload: payload.__setitem__(
                    "source_revision",
                    "d" * 40,
                ),
            ),
        ),
        (
            "model identity",
            lambda: _mutate_json(
                tmp_path,
                "model_manifest.json",
                lambda payload: payload["text_profile"].__setitem__(
                    "hidden_size",
                    4096,
                ),
            ),
        ),
        (
            "GPU rank map",
            lambda: _mutate_json(
                tmp_path,
                "gpu_topology.json",
                lambda payload: payload["rank_mapping"][3].__setitem__(
                    "gpu_uuid",
                    "GPU-0",
                ),
            ),
        ),
        (
            "workload matrix",
            lambda: _mutate_json(
                tmp_path,
                "workload_manifest.json",
                lambda payload: payload["case_ids"].pop(),
            ),
        ),
        (
            "static catalog",
            lambda: _mutate_json(
                tmp_path,
                "static_collective_catalog.json",
                lambda payload: payload["rows"][0].__setitem__(
                    "site_id",
                    "changed",
                ),
            ),
        ),
        (
            "consumer proof",
            lambda: _mutate_json(
                tmp_path,
                "consumer_dependency_proofs.json",
                lambda payload: payload["rows"][0].__setitem__(
                    "status",
                    "FAIL_IMMEDIATE_CONSUMER",
                ),
            ),
        ),
        (
            "count byte sequence",
            lambda: _mutate_jsonl(
                tmp_path,
                "collective_census.jsonl",
                lambda rows: rows[3]["collectives"][7].__setitem__(
                    "tensor_bytes",
                    10242,
                ),
            ),
        ),
        (
            "timing cohort",
            lambda: _mutate_jsonl(
                tmp_path,
                "collective_timing_samples.jsonl",
                lambda rows: rows[0].__setitem__("cuda_ns", 999_999),
            ),
        ),
        (
            "profiler overhead",
            lambda: _mutate_json(
                tmp_path,
                "profiler_calibration.json",
                lambda payload: payload["rows"][0].__setitem__(
                    "median_overhead_ratio",
                    0.0,
                ),
            ),
        ),
        (
            "correctness",
            lambda: _mutate_jsonl(
                tmp_path,
                "correctness.jsonl",
                lambda rows: rows[-1]["output_token_ids"].__setitem__(
                    -1,
                    8,
                ),
            ),
        ),
        (
            "resource identity",
            lambda: _mutate_jsonl(
                tmp_path,
                "resource_samples.jsonl",
                lambda rows: rows[0]["selected_gpus"][0].__setitem__(
                    "memory_used_mib",
                    1025,
                ),
            ),
        ),
        (
            "cleanup",
            lambda: _mutate_json(
                tmp_path,
                "cleanup.json",
                lambda payload: payload[
                    "owned_children_remaining"
                ].append(123),
            ),
        ),
        (
            "classification",
            lambda: _mutate_json(
                tmp_path,
                "classification.json",
                lambda payload: payload.__setitem__(
                    "classification",
                    "NO_GO_NO_REDUCIBLE_COLLECTIVE",
                ),
            ),
        ),
        (
            "manifest",
            lambda: _mutate_json(
                tmp_path,
                "manifest.sha256",
                lambda payload: payload["artifacts"].__setitem__(
                    "source_identity.json",
                    "0" * 64,
                ),
            ),
        ),
    )

    for authority, mutate in mutations:
        for path in tuple(tmp_path.iterdir()):
            if path.is_file() and path.name not in original:
                path.unlink()
        for name, content in original.items():
            (tmp_path / name).write_bytes(content)
        mutate()
        if authority != "manifest":
            _rewrite_manifest(tmp_path)
        with pytest.raises(ValueError):
            verify_bundle(tmp_path)
