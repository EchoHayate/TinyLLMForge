from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools/qwen35_tp4_shared_memory_fanout_preflight.py"
)
PREREQUISITE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-live-shm-engine-ack-20260728-110846/"
    "live_shared_memory_engine_ack_dispatch_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_shared_memory_fanout_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prerequisite_source_closure_and_method_identities():
    module = _load_module()
    prerequisite = module.load_tp4_fanout_prerequisite(
        PREREQUISITE_ARTIFACT
    )

    assert prerequisite.artifact_sha256 == (
        "11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57"
    )
    assert prerequisite.source_tree_sha256 == (
        "6cc9672dbd80c211ccd64371573fd8de463b773fc5cc3ae7286ad21c9c664572"
    )
    assert tuple(prerequisite.rows) == (
        "tp2_shm_success",
        "tp2_shm_worker_binding_error",
        "tp2_shm_worker_ack_exception",
        "tp2_shm_worker_exit_without_ack",
    )
    assert len(prerequisite.source_file_sha256) == 55
    assert len(module.SOURCE_FILES) == 56
    assert len(set(module.SOURCE_FILES)) == 56
    assert set(module.SOURCE_FILES) - set(
        prerequisite.source_file_sha256
    ) == {
        "tools/qwen35_tp4_shared_memory_fanout_preflight.py",
    }

    methods = module.load_frozen_tp4_fanout_methods(ROOT)
    assert set(methods) == {
        "write_shm",
        "read_shm",
        "loop",
        "dispatch_command",
        "call_model_runner_acknowledged",
    }
    assert all(callable(method) for method in methods.values())


def test_tp4_identity_rows_are_transport_only_and_fail_closed():
    module = _load_module()
    nonce = "nonce-0123456789abcdef"
    rows = tuple(
        module.build_tp4_identity_row(
            participant_id=rank,
            attempt_nonce=nonce,
        )
        for rank in range(4)
    )

    assert module.validate_tp4_identity_rows(
        rows,
        attempt_nonce=nonce,
    ) == rows
    assert all(
        row["operation"]
        == "report_tp4_shared_memory_fanout_identity"
        for row in rows
    )
    assert all("checkpoint" not in row["operation"] for row in rows)

    invalid = list(rows)
    invalid[2] = module.build_tp4_identity_row(
        participant_id=2,
        attempt_nonce=nonce,
        status="error",
        detail="injected rank2 inner error",
    )
    try:
        module.validate_tp4_identity_rows(
            tuple(invalid),
            attempt_nonce=nonce,
        )
    except RuntimeError as error:
        assert "rank=2" in str(error)
        assert "injected rank2 inner error" in str(error)
    else:
        raise AssertionError("inner row error must fail closed")


def test_tp4_reverse_completion_returns_ranked_acknowledgements():
    module = _load_module()
    result = module.execute_tp4_shared_memory_fanout_attempt(
        source_root=ROOT,
        prerequisite_artifact=PREREQUISITE_ARTIFACT,
        mode="tp4_fanout_success_reverse_completion",
        timeout_s=4.0,
        name_prefix="qwen35-tp4-success",
    )

    assert result["status"] == "PASS"
    assert result["worker_ranks"] == [1, 2, 3]
    assert result["ack_send_order"] == [3, 2, 1]
    assert result["collector_return_order"] == [1, 2, 3]
    assert result["collector_result_participants"] == [1, 2, 3]
    assert result["fanout_validated"] is True
    assert result["collector_poisoned"] is False
    assert result["dispatch_count"] == 2
    assert result["write_count"] == 2
    assert result["read_count_by_rank"] == {"1": 2, "2": 2, "3": 2}
    assert result["executor_count_by_rank"] == {
        "1": 2,
        "2": 2,
        "3": 2,
    }
    assert result["event_set_count_by_rank"] == {
        "1": 2,
        "2": 2,
        "3": 2,
    }
    assert result["event_wait_count_by_rank"] == {
        "1": 2,
        "2": 2,
        "3": 2,
    }
    assert result["event_clear_count_by_rank"] == {
        "1": 2,
        "2": 2,
        "3": 2,
    }
    assert result["child_exitcodes"] == {"1": 0, "2": 0, "3": 0}
    assert result["child_collected_by_rank"] == {
        "1": True,
        "2": True,
        "3": True,
    }
    assert result["segment_unlinked"] is True
    assert result["post_unlink_attach_failed"] is True
    assert len(set(result["child_process_ids"].values())) == 3


def test_tp4_failure_modes_are_distinct_and_fail_closed():
    module = _load_module()
    cases = (
        (
            "tp4_fanout_rank2_inner_error",
            "injected rank2 inner error",
            {"1": "ok", "2": "ok", "3": "ok"},
            False,
            0,
        ),
        (
            "tp4_fanout_rank2_ack_exception",
            "injected rank2 acknowledgement exception",
            {"1": "ok", "2": "error", "3": "ok"},
            True,
            0,
        ),
        (
            "tp4_fanout_rank2_exit_without_ack",
            "acknowledgement",
            {"1": "ok", "2": "absent", "3": "ok"},
            True,
            9,
        ),
    )
    for mode, detail, ack_statuses, poisoned, rank2_exitcode in cases:
        result = module.execute_tp4_shared_memory_fanout_attempt(
            source_root=ROOT,
            prerequisite_artifact=PREREQUISITE_ARTIFACT,
            mode=mode,
            timeout_s=4.0,
            name_prefix=f"qwen35-{mode}",
        )

        assert result["status"] == "PASS"
        assert detail in result["error_detail"]
        assert result["ack_status_by_rank"] == ack_statuses
        assert result["collector_poisoned"] is poisoned
        assert result["fanout_validated"] is False
        assert result["exit_envelope_sent"] is True
        assert result["child_exitcodes"]["2"] == rank2_exitcode
        assert result["child_collected_by_rank"] == {
            "1": True,
            "2": True,
            "3": True,
        }
        assert result["segment_unlinked"] is True
        assert result["post_unlink_attach_failed"] is True


def test_orchestration_contract_and_partial_rejection():
    module = _load_module()
    assert module.ATTEMPT_MODES == (
        "tp4_fanout_success_reverse_completion",
        "tp4_fanout_rank2_inner_error",
        "tp4_fanout_rank2_ack_exception",
        "tp4_fanout_rank2_exit_without_ack",
    )
    prerequisite = module.load_tp4_fanout_prerequisite(
        PREREQUISITE_ARTIFACT
    )
    hashes = module._source_hashes(ROOT)
    assert len(hashes) == 56
    assert {
        name: hashes[name]
        for name in prerequisite.source_file_sha256
    } == dict(prerequisite.source_file_sha256)
    archive = module.build_source_tar(ROOT)
    assert isinstance(archive, bytes)
    assert len(archive) > 0
    try:
        module._aggregate([], ROOT)
    except ValueError as error:
        assert "attempt rows" in str(error)
    else:
        raise AssertionError("partial finalization must fail")


def test_static_safety_audit():
    module = _load_module()
    audit = module.audit_tp4_fanout_source(ROOT)
    assert audit["llm_engine_import_count"] == 0
    assert audit["model_runner_import_count"] == 0
    assert audit["llm_engine_construction_count"] == 0
    assert audit["model_runner_construction_count"] == 0
    assert audit["fixed_tinyvllm_shared_memory_count"] == 0
    assert audit["shared_memory_create_count"] == 1
    assert audit["shared_memory_attach_count"] == 2
    assert audit["checkpoint_call_count"] == 0
    assert audit["scheduler_call_count"] == 0
    assert audit["step_call_count"] == 0
    assert audit["cuda_call_count"] == 0
    assert audit["forward_call_count"] == 0
    assert audit["inference_call_count"] == 0


def test_internal_finalizer_passes_a_path_to_atomic_writer():
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
        "qwen35 TP4 shared-memory fan-out tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
