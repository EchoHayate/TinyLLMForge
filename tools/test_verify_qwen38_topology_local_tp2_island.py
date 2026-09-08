from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import pytest

from tools.verify_qwen38_topology_local_tp2_island import (
    LOCAL_RECEIPT_NAME,
    REMOTE_RECEIPT_NAME,
    TERMINAL_MANIFEST_NAME,
    verify_bundle,
)
import tools.verify_qwen38_topology_local_tp2_island as verifier_module


MANIFEST_SCHEMA = "qwen38.topology-local-tp2-island-manifest.v1"
ASSEMBLER_MODULE = "tools.assemble_qwen38_topology_local_tp2_island"


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


def _rewrite_manifest(root):
    artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.sha256"
    }
    _write_json(root / "manifest.sha256", {
        "schema": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _passing_inputs():
    fixture_module = importlib.import_module(
        "tools.test_assemble_qwen38_topology_local_tp2_island"
    )
    return fixture_module.passing_inputs()


def _mutated_inputs(mutation):
    fixture_module = importlib.import_module(
        "tools.test_assemble_qwen38_topology_local_tp2_island"
    )
    return fixture_module.mutate(
        fixture_module.passing_inputs(), mutation
    )


def _write_bundle(root, inputs=None):
    assembler = importlib.import_module(ASSEMBLER_MODULE)
    assembler.assemble_bundle(root, **(inputs or _passing_inputs()))
    sys.modules.pop(ASSEMBLER_MODULE, None)
    return root


def test_verifier_reconstructs_go_without_assembler_import(tmp_path):
    root = _write_bundle(tmp_path)

    result = verify_bundle(root)

    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
    )
    assert ASSEMBLER_MODULE not in sys.modules
    source = inspect.getsource(verifier_module)
    assert "assemble_qwen38_topology_local_tp2_island" not in source


def test_verifier_rejects_mutated_row_after_manifest(tmp_path):
    root = _write_bundle(tmp_path)
    path = root / "paired_timing_rows.jsonl"
    path.write_text(path.read_text().replace(
        '"candidate_cuda_ns":900',
        '"candidate_cuda_ns":901',
        1,
    ))

    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(root)


def test_verifier_rejects_producer_classifier_disagreement(tmp_path):
    root = _write_bundle(tmp_path)
    producer = _read_json(root / "producer_result.json")
    producer["classification"] = "NO_GO_PERFORMANCE"
    _write_json(root / "producer_result.json", producer)
    _rewrite_manifest(root)

    with pytest.raises(ValueError, match="classification"):
        verify_bundle(root)


@pytest.mark.parametrize(
    "mutation",
    (
        "pair_disagreement",
        "baseline_numeric_failure",
        "token1_speedup_0049",
        "token48_geomean_0049",
        "p99_regression_0031",
        "improving_pairs_10",
        "host_regression_0101",
        "break_even_33",
        "steady_increment_over_1920_mib",
        "peak_ratio_09801",
    ),
)
def test_verifier_reconstructs_every_frozen_no_go(tmp_path, mutation):
    root = _write_bundle(tmp_path, _mutated_inputs(mutation))
    producer = _read_json(root / "producer_result.json")

    result = verify_bundle(root)

    assert result["classification"] == producer["classification"]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_rank",
        "wrong_row_count",
        "unfrozen_pair_map",
        "source_drift",
        "model_drift",
        "candidate_global_collective",
        "fallback",
        "temporary_tensor_live",
        "incomplete_cleanup",
        "parameter_reconstruction_mismatch",
    ),
)
def test_verifier_reconstructs_invalid_evidence(tmp_path, mutation):
    inputs = _passing_inputs()
    if mutation == "missing_rank":
        inputs["timing_rows"] = [
            row for row in inputs["timing_rows"] if row["rank"] != 3
        ]
    elif mutation == "wrong_row_count":
        inputs["timing_rows"].pop()
    elif mutation == "unfrozen_pair_map":
        inputs["topology"]["selection_frozen"] = False
    elif mutation == "source_drift":
        inputs["timing_rows"][0]["source_revision"] = "f" * 40
    elif mutation == "model_drift":
        inputs["model_identity"]["model_revision"] = "f" * 40
    elif mutation == "candidate_global_collective":
        inputs["timing_rows"][0][
            "candidate_global_collective_count"
        ] = 1
    elif mutation == "fallback":
        inputs["timing_rows"][0]["fallback_count"] = 1
    elif mutation == "temporary_tensor_live":
        inputs["migration_rows"][0][
            "temporary_live_tensor_count_after_release"
        ] = 1
    elif mutation == "incomplete_cleanup":
        inputs["cleanup"]["rank_rows"][0][
            "candidate_state_unpublished"
        ] = False
    elif mutation == "parameter_reconstruction_mismatch":
        inputs["parameter_slices"]["rank_parameter_evidence"][3][
            "reconstructed_full_parameter_digests"
        ]["full"] = "d" * 64
    root = _write_bundle(tmp_path, inputs)

    assert verify_bundle(root)["classification"] == "INVALID_EVIDENCE"


def test_verifier_rejects_missing_and_extra_files(tmp_path):
    missing = _write_bundle(tmp_path / "missing")
    (missing / "cleanup.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(missing)

    extra = _write_bundle(tmp_path / "extra")
    (extra / "unexpected.json").write_text("{}\n")
    _rewrite_manifest(extra)
    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(extra)


@pytest.mark.parametrize("payload", ('{"a":1,"a":2}', '{"a":NaN}'))
def test_verifier_rejects_duplicate_and_nonfinite_json(tmp_path, payload):
    root = _write_bundle(tmp_path)
    (root / "admission.json").write_text(payload)
    _rewrite_manifest(root)

    with pytest.raises(ValueError):
        verify_bundle(root)


def test_verifier_writes_both_receipts_and_seals_terminal(tmp_path):
    root = _write_bundle(tmp_path)

    remote = verify_bundle(root, receipt_name=REMOTE_RECEIPT_NAME)
    local = verify_bundle(
        root,
        receipt_name=LOCAL_RECEIPT_NAME,
        seal_terminal=True,
    )
    before = {
        path.name: path.read_bytes()
        for path in root.iterdir()
        if path.is_file()
    }
    checked = verify_bundle(root, receipt_name=None, check_only=True)

    assert remote["classification"] == local["classification"]
    assert checked["classification"] == local["classification"]
    assert (root / REMOTE_RECEIPT_NAME).is_file()
    assert (root / LOCAL_RECEIPT_NAME).is_file()
    assert (root / TERMINAL_MANIFEST_NAME).is_file()
    assert before == {
        path.name: path.read_bytes()
        for path in root.iterdir()
        if path.is_file()
    }


def test_terminal_seal_requires_remote_receipt(tmp_path):
    root = _write_bundle(tmp_path)

    with pytest.raises(ValueError, match="remote"):
        verify_bundle(
            root,
            receipt_name=LOCAL_RECEIPT_NAME,
            seal_terminal=True,
        )
