from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools/qwen35_tp4_synthetic_binding_oracle_preflight.py"
)
TP4_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-shm-fanout-20260728-115046/"
    "tp4_shared_memory_fanout_preflight.json"
)
ORACLE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-synthetic-binding-oracle-v1/"
    "synthetic_binding_oracle.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_synthetic_binding_oracle_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dual_prerequisites_oracle_reconstruction_and_source_closure():
    module = _load_module()
    prerequisite = module.load_synthetic_binding_prerequisites(
        TP4_ARTIFACT,
        ORACLE_ARTIFACT,
    )

    assert prerequisite.tp4_artifact_sha256 == (
        "ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a"
    )
    assert prerequisite.tp4_source_tree_sha256 == (
        "ec7b0dee43a06c47b72f8ac14ab26518845f57f070e6c27d394bb4c328644403"
    )
    assert prerequisite.oracle_artifact_sha256 == (
        "1fdc3d64178308d6d26242805433ee31012890f6824a13826075db1e6937431e"
    )
    assert prerequisite.oracle["provenance"] == (
        "synthetic-construction-free-oracle"
    )
    assert prerequisite.oracle["claim_boundary"] == (
        "not-real-checkpoint-binding"
    )
    assert prerequisite.oracle["tensor_payload"] == "absent"
    assert prerequisite.oracle["model_fingerprint"] == (
        "b48e29c9ea266197c56fe3e9133c65d378c682e9437d9e1299c4fa93d0241bd9"
    )
    assert prerequisite.oracle["layout_fingerprint"] == (
        "fe2db881dc909cbd05894a4e194273f4692caed1c4df967e810330324248e2c6"
    )
    assert tuple(prerequisite.cases) == (
        "tp4_synthetic_binding_success",
        "tp4_synthetic_rank2_model_mismatch",
        "tp4_synthetic_rank2_layout_mismatch",
        "tp4_synthetic_rank2_dtype_mismatch",
    )
    assert len(prerequisite.source_file_sha256) == 56
    assert len(module.SOURCE_FILES) == 57
    assert set(module.SOURCE_FILES) - set(
        prerequisite.source_file_sha256
    ) == {
        "tools/qwen35_tp4_synthetic_binding_oracle_preflight.py",
    }

    methods = module.load_frozen_synthetic_binding_methods(ROOT)
    assert set(methods) == {
        "write_shm",
        "read_shm",
        "loop",
        "dispatch_command",
        "call_model_runner_acknowledged",
        "bind_qwen35_loaded_checkpoint_candidates",
    }


def test_synthetic_tp4_binding_success_commits_and_replays():
    module = _load_module()
    result = module.execute_tp4_synthetic_binding_attempt(
        source_root=ROOT,
        tp4_artifact=TP4_ARTIFACT,
        oracle_artifact=ORACLE_ARTIFACT,
        mode="tp4_synthetic_binding_success",
        timeout_s=4.0,
        name_prefix="qwen35-synthetic-success",
    )

    assert result["status"] == "PASS"
    assert result["ack_send_order"] == [3, 2, 1]
    assert result["collector_return_order"] == [1, 2, 3]
    assert result["ack_status_by_rank"] == {
        "1": "ok",
        "2": "ok",
        "3": "ok",
    }
    assert result["binding_rows"] == result["oracle_rows"]
    assert result["completion_configuration"] == [
        "b48e29c9ea266197c56fe3e9133c65d378c682e9437d9e1299c4fa93d0241bd9",
        "fe2db881dc909cbd05894a4e194273f4692caed1c4df967e810330324248e2c6",
        "bfloat16",
        4.0,
    ]
    assert result["completion_committed"] is True
    assert result["repeat_zero_binding_dispatch"] is True
    assert result["binding_dispatch_count"] == 1
    assert result["dispatch_count"] == 2
    assert result["collector_poisoned"] is False
    assert result["child_exitcodes"] == {"1": 0, "2": 0, "3": 0}
    assert result["segment_unlinked"] is True
    assert result["post_unlink_attach_failed"] is True


def test_synthetic_rank2_mismatches_are_transport_ok_and_binder_closed():
    module = _load_module()
    cases = (
        (
            "tp4_synthetic_rank2_model_mismatch",
            "mismatch: model_fingerprint",
            "model_fingerprint",
        ),
        (
            "tp4_synthetic_rank2_layout_mismatch",
            "mismatch: layout_fingerprint",
            "layout_fingerprint",
        ),
        (
            "tp4_synthetic_rank2_dtype_mismatch",
            "mismatch: dtype",
            "dtype",
        ),
    )
    for mode, detail, changed_field in cases:
        result = module.execute_tp4_synthetic_binding_attempt(
            source_root=ROOT,
            tp4_artifact=TP4_ARTIFACT,
            oracle_artifact=ORACLE_ARTIFACT,
            mode=mode,
            timeout_s=4.0,
            name_prefix=f"qwen35-{mode}",
        )

        assert result["status"] == "PASS"
        assert detail in result["error_detail"]
        assert result["ack_send_order"] == [3, 2, 1]
        assert result["collector_return_order"] == [1, 2, 3]
        assert result["ack_status_by_rank"] == {
            "1": "ok",
            "2": "ok",
            "3": "ok",
        }
        assert result["collector_poisoned"] is False
        assert result["completion_committed"] is False
        assert result["completion_configuration"] is None
        assert result["binding_rows"] is None
        assert result["repeat_zero_binding_dispatch"] is False
        assert result["authorized_changed_field"] == changed_field
        assert result["child_exitcodes"] == {
            "1": 0,
            "2": 0,
            "3": 0,
        }
        assert result["segment_unlinked"] is True
        assert result["post_unlink_attach_failed"] is True


def test_orchestration_contract_and_static_safety():
    module = _load_module()
    prerequisite = module.load_synthetic_binding_prerequisites(
        TP4_ARTIFACT,
        ORACLE_ARTIFACT,
    )
    hashes = module._source_hashes(ROOT)
    assert len(hashes) == 57
    assert {
        name: hashes[name]
        for name in prerequisite.source_file_sha256
    } == dict(prerequisite.source_file_sha256)
    audit = module.audit_synthetic_binding_source(ROOT)
    assert audit == {
        "llm_engine_import_count": 0,
        "model_runner_import_count": 0,
        "llm_engine_construction_count": 0,
        "model_runner_construction_count": 0,
        "fixed_tinyvllm_shared_memory_count": 0,
        "checkpoint_call_count": 0,
        "model_construction_count": 0,
        "scheduler_call_count": 0,
        "step_call_count": 0,
        "cuda_call_count": 0,
        "forward_call_count": 0,
        "inference_call_count": 0,
    }
    try:
        module._aggregate([], ROOT)
    except ValueError as error:
        assert "attempt rows" in str(error)
    else:
        raise AssertionError("partial finalization must fail")


def test_deterministic_source_archive_and_dual_prerequisite_manifest():
    module = _load_module()
    archive = module.build_source_tar(ROOT)
    assert isinstance(archive, bytes)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        members = bundle.getmembers()
        assert [member.name for member in members] == list(
            module.SOURCE_FILES
        )
        assert all(member.uid == 0 and member.gid == 0 for member in members)
        assert all(member.mtime == 0 for member in members)

    hashes = module._source_hashes(ROOT)
    staged = {
        "remote_source_dir": "/remote/run/source",
        "remote_tp4_artifact": "/remote/run/tp4.json",
        "remote_oracle_artifact": "/remote/run/oracle.json",
        "tp4_artifact_sha256": module.TP4_ARTIFACT_SHA256,
        "oracle_artifact_sha256": module.ORACLE_ARTIFACT_SHA256,
        "method_source_sha256": dict(module.METHOD_SOURCE_SHA256),
        "source_tree_sha256": module.tp4_gate._source_tree_sha256(hashes),
        "local_file_sha256": hashes,
        "remote_file_sha256": hashes,
    }
    manifest = module._source_manifest(
        "qwen35-tp4-synthetic-binding-test",
        staged,
    )
    assert manifest["remote_tp4_artifact"] == "/remote/run/tp4.json"
    assert manifest["remote_oracle_artifact"] == "/remote/run/oracle.json"
    assert manifest["tp4_artifact_sha256"] == (
        module.TP4_ARTIFACT_SHA256
    )
    assert manifest["oracle_artifact_sha256"] == (
        module.ORACLE_ARTIFACT_SHA256
    )
    assert manifest["local_file_sha256"] == hashes
    assert manifest["remote_file_sha256"] == hashes


def test_internal_finalizer_passes_path_to_atomic_writer():
    module = _load_module()
    observed = {}
    original_aggregate = module._aggregate
    original_writer = module._atomic_write_json
    original_stdin = sys.stdin
    try:
        module._aggregate = lambda rows, source_root: {"rows": rows}

        def atomic_writer(path, record):
            assert isinstance(path, Path)
            observed["path"] = path
            observed["record"] = record

        module._atomic_write_json = atomic_writer
        sys.stdin = io.StringIO('{"rows":[]}')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            assert module.main([
                "internal-finalize",
                "--source-root",
                str(ROOT),
                "--output",
                str(output),
            ]) == 0
            assert observed == {
                "path": output,
                "record": {"rows": []},
            }
    finally:
        module._aggregate = original_aggregate
        module._atomic_write_json = original_writer
        sys.stdin = original_stdin


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 synthetic binding oracle tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
