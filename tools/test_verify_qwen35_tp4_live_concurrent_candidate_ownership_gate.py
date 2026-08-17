from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT
    / "tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools/verify_qwen35_tp4_live_concurrent_candidate_ownership_gate.py"
)
PRISTINE_ORACLE_PATH = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-real-candidate-replay-20260728-145713/"
    "tp4_real_candidate_provenance_oracle.json"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ready_row(module, rank):
    pristine = json.loads(PRISTINE_ORACLE_PATH.read_text())[
        "producer_rows"
    ][rank]
    return {
        "schema_version": module.READY_ROW_SCHEMA_VERSION,
        "status": "READY",
        "provenance": module.PROVENANCE,
        "claim_boundary": module.CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "process_id": 60_000_000 + rank,
        "method_row": {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": pristine["model_manifest_sha256"],
            "layout_fingerprint": pristine["layout_fingerprint"],
            "dtype": pristine["dtype"],
            "detail": "",
        },
        "binding_hash_count": 320,
        "binding_destination_sha256": pristine[
            "binding_destination_sha256"
        ],
        "phase_hash_count": 26,
        "phase_destination_sha256": pristine[
            "phase_destination_sha256"
        ],
        "aggregate_destination_sha256": pristine[
            "aggregate_destination_sha256"
        ],
        "alias_groups": pristine["alias_groups"],
        "loader_stats": pristine["loader_stats"],
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "ready_memory": {
            "vmrss_kib": 1_700_000,
            "vmhwm_kib": 2_900_000,
        },
        "memory": {
            "before": {
                "vmrss_kib": 300_000,
                "vmhwm_kib": 300_000,
            },
            "ready": {
                "vmrss_kib": 1_700_000,
                "vmhwm_kib": 2_900_000,
            },
        },
        "all_private_objects_retained": True,
        "retained_private_objects": {
            name: True for name in module.PRIVATE_OBJECT_NAMES
        },
    }


def _released_row(module, rank):
    return {
        "schema_version": module.RELEASED_ROW_SCHEMA_VERSION,
        "status": "RELEASED",
        "provenance": module.PROVENANCE,
        "claim_boundary": module.CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "collected_private_objects": {
            name: True for name in module.PRIVATE_OBJECT_NAMES
        },
        "all_private_objects_collected": True,
    }


def _build_run(directory):
    module = _load(PREFLIGHT_PATH, "live_concurrent_fixture")
    run_dir = Path(directory) / "live-concurrent-test-run"
    run_dir.mkdir()
    source_root = Path(directory) / "source"
    source = (
        source_root
        / "tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py"
    )
    source.parent.mkdir(parents=True)
    shutil.copy2(PREFLIGHT_PATH, source)
    source_hashes = {
        "tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py": (
            hashlib.sha256(source.read_bytes()).hexdigest()
        )
    }
    source_tree = module._sha256(module._canonical(source_hashes))
    ready_rows = [_ready_row(module, rank) for rank in range(4)]
    snapshot = {
        "schema_version": (
            "qwen35.tp4-live-concurrent-candidate-snapshot.v1"
        ),
        "coordinator_process_id": 59_999_999,
        "snapshot_unix_time_ns": 1_785_223_200_000_000_000,
        "start_order": [0, 1, 2, 3],
        "ready_order": [0, 1, 2, 3],
        "live_process_ids": [
            row["process_id"] for row in ready_rows
        ],
        "ready_row_sha256": [
            module._sha256(module._canonical(row))
            for row in ready_rows
        ],
        "process_status": [
            {
                "rank": rank,
                "process_id": row["process_id"],
                "state": "S",
                "vmrss_kib": 1_700_000,
                "vmhwm_kib": 2_900_000,
            }
            for rank, row in enumerate(ready_rows)
        ],
        "release_acknowledgement_count": 0,
        "all_workers_live_concurrently": True,
    }
    released_rows = [
        _released_row(module, rank) for rank in (3, 2, 1, 0)
    ]
    artifact = module.build_ownership_artifact(
        ready_rows=ready_rows,
        concurrent_snapshot=snapshot,
        released_rows=released_rows,
        host_memory_before={
            "mem_available_kib": 20_000_000,
            "swap_total_kib": 0,
            "swap_free_kib": 0,
        },
        host_memory_ready={
            "mem_available_kib": 13_000_000,
            "swap_total_kib": 0,
            "swap_free_kib": 0,
        },
        source_file_sha256=source_hashes,
        source_tree_sha256=source_tree,
        prerequisite_oracle_sha256=(
            "d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef"
        ),
    )
    artifact["worker_process_ids"] = [
        row["process_id"] for row in ready_rows
    ]
    artifact["residual_worker_process_ids"] = []
    artifact["all_worker_process_ids_absent"] = True
    artifact_path = (
        run_dir / "tp4_live_concurrent_candidate_ownership.json"
    )
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    manifest = {
        "schema_version": (
            "qwen35.tp4-live-concurrent-candidate-ownership.v1"
        ),
        "run_tag": run_dir.name,
        "remote_target": "sitian@10.232.195.203",
        "remote_python": (
            "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
        ),
        "source_file_sha256": source_hashes,
        "source_tree_sha256": source_tree,
        "prerequisite_oracle_sha256": (
            "d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef"
        ),
        "result_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
    }
    (run_dir / "source_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    return run_dir, source_root


def test_generated_artifact_passes_independent_verification():
    verifier = _load(VERIFIER_PATH, "live_concurrent_verifier")
    with tempfile.TemporaryDirectory() as directory:
        run_dir, source_root = _build_run(directory)
        result = verifier.verify_run(
            run_dir,
            source_root=source_root,
            prerequisite_oracle=PRISTINE_ORACLE_PATH,
        )
    assert result["status"] == "PASS"
    assert result["checks"] >= 100
    assert result["ready_count"] == 4
    assert result["released_count"] == 4
    assert result["unique_process_count"] == 4


def _tamper(mutator, *, source_mutator=None):
    verifier = _load(
        VERIFIER_PATH,
        "live_concurrent_verifier_tamper",
    )
    with tempfile.TemporaryDirectory() as directory:
        run_dir, source_root = _build_run(directory)
        artifact_path = (
            run_dir / "tp4_live_concurrent_candidate_ownership.json"
        )
        manifest_path = run_dir / "source_manifest.json"
        artifact = json.loads(artifact_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        mutator(artifact)
        if source_mutator is not None:
            source_mutator(source_root, artifact, manifest)
        artifact_path.write_text(
            json.dumps(artifact, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        manifest["result_sha256"] = hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        return verifier.verify_run(
            run_dir,
            source_root=source_root,
            prerequisite_oracle=PRISTINE_ORACLE_PATH,
        )


def test_missing_live_pid_is_rejected():
    result = _tamper(
        lambda artifact: artifact["concurrent_snapshot"][
            "live_process_ids"
        ].pop()
    )
    assert result["status"] == "FAIL"
    assert "process" in result["detail"].lower()


def test_resigned_memory_breach_is_rejected():
    def mutate(artifact):
        row = artifact["ready_rows"][3]
        row["memory"]["ready"]["vmhwm_kib"] = 3_600_000
        row["ready_memory"]["vmhwm_kib"] = 3_600_000
        status = artifact["concurrent_snapshot"]["process_status"][3]
        status["vmhwm_kib"] = 3_600_000
        artifact["concurrent_snapshot"]["ready_row_sha256"][3] = (
            hashlib.sha256(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        artifact["ready_rows_sha256"] = hashlib.sha256(
            json.dumps(
                artifact["ready_rows"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    result = _tamper(mutate)
    assert result["status"] == "FAIL"
    assert "3145728" in result["detail"]


def test_resigned_production_engine_import_is_rejected():
    def mutate_source(source_root, artifact, manifest):
        name = (
            "tools/"
            "qwen35_tp4_live_concurrent_candidate_ownership_preflight.py"
        )
        path = source_root / name
        path.write_text(
            path.read_text()
            + "\nfrom tinyvllm.engine.llm_engine import LLMEngine\n"
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact["source_file_sha256"][name] = digest
        manifest["source_file_sha256"][name] = digest
        tree = hashlib.sha256(
            json.dumps(
                artifact["source_file_sha256"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        artifact["source_tree_sha256"] = tree
        manifest["source_tree_sha256"] = tree

    result = _tamper(
        lambda artifact: None,
        source_mutator=mutate_source,
    )
    assert result["status"] == "FAIL"
    assert "import" in result["detail"].lower()


def test_resigned_start_order_change_is_rejected():
    def mutate(artifact):
        artifact["start_order"] = [0, 2, 1, 3]
        artifact["concurrent_snapshot"]["start_order"] = [0, 2, 1, 3]

    result = _tamper(mutate)
    assert result["status"] == "FAIL"
    assert "start order" in result["detail"].lower()


def test_resigned_premature_release_is_rejected():
    result = _tamper(
        lambda artifact: artifact["concurrent_snapshot"].__setitem__(
            "release_acknowledgement_count",
            1,
        )
    )
    assert result["status"] == "FAIL"
    assert "release" in result["detail"].lower()


def test_resigned_incomplete_collection_is_rejected():
    def mutate(artifact):
        row = artifact["released_rows"][0]
        row["collected_private_objects"]["owner"] = False
        row["all_private_objects_collected"] = False
        artifact["released_rows_sha256"] = hashlib.sha256(
            json.dumps(
                artifact["released_rows"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    result = _tamper(mutate)
    assert result["status"] == "FAIL"
    assert "released row" in result["detail"].lower()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 live concurrent independent verifier tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
