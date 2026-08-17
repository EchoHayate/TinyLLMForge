from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_live_shared_memory_engine_ack_dispatch_preflight.py"
)
PREREQUISITE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-engine-ack-transport-20260728-102828/"
    "engine_ack_transport_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_live_shared_memory_engine_ack_dispatch_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prerequisite_source_closure_and_method_identities():
    module = _load_module()
    prerequisite = module.load_live_shared_memory_prerequisite(
        PREREQUISITE_ARTIFACT
    )

    assert prerequisite.artifact_sha256 == (
        "8aeb571c3d56641e747a0d5c5e66314efe6b35b73320cb49e0340c0fe5fd42fb"
    )
    assert prerequisite.source_tree_sha256 == (
        "a041ebf7653e141dd96ebe31143ba00e5634c61c1a4bec68f17e7a7c6bba5cc8"
    )
    assert tuple(prerequisite.rows) == (
        "tp1_success",
        "tp1_local_binding_error",
        "tp2_success",
        "tp2_worker_binding_error",
        "tp2_worker_ack_exception",
        "tp2_worker_exit_without_ack",
    )
    assert len(prerequisite.source_file_sha256) == 54
    assert len(module.SOURCE_FILES) == 55
    assert len(set(module.SOURCE_FILES)) == 55
    assert set(module.SOURCE_FILES) - set(
        prerequisite.source_file_sha256
    ) == {
        "tools/qwen35_live_shared_memory_engine_ack_dispatch_preflight.py",
    }

    methods = module.load_frozen_live_shared_memory_methods(
        ROOT,
        fingerprint_validator=lambda value: value,
    )
    assert set(methods) == {
        "write_shm",
        "read_shm",
        "loop",
        "dispatch_command",
        "call_model_runner_acknowledged",
        "bind_qwen35_loaded_checkpoint_candidates",
    }
    assert all(callable(method) for method in methods.values())


def test_real_shared_memory_codec_round_trip_and_unlink():
    module = _load_module()
    result = module.execute_live_shared_memory_codec_round_trip(
        source_root=ROOT,
        name_prefix="qwen35-shm-codec-test",
    )

    assert result["status"] == "PASS"
    assert result["shared_memory_name"].startswith("qwen35-shm-")
    assert len(result["shared_memory_name"]) <= 30
    assert result["shared_memory_name"] != "tinyvllm"
    assert result["shared_memory_capacity"] == 2**20
    assert result["payload_bytes"] > 0
    assert result["envelope"] == {
        "command_id": 7,
        "method_name": (
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ),
        "args": [],
        "requires_ack": True,
    }
    assert result["event_set_count"] == 1
    assert result["event_wait_count"] == 1
    assert result["event_clear_count"] == 1
    assert result["segment_unlinked"] is True
    assert result["post_unlink_attach_failed"] is True


def test_live_shared_memory_tp2_success_and_replay():
    module = _load_module()
    result = module.execute_live_shared_memory_engine_ack_attempt(
        source_root=ROOT,
        prerequisite_artifact=PREREQUISITE_ARTIFACT,
        mode="tp2_shm_success",
        timeout_s=2.0,
        name_prefix="qwen35-shm-success",
    )

    assert result["status"] == "PASS"
    assert result["dispatch_count"] == 2
    assert result["binding_dispatch_count"] == 1
    assert result["write_count"] == 2
    assert result["read_count"] == 2
    assert result["executor_count"] == 2
    assert result["collector_call_count"] == 1
    assert result["acknowledgement_status"] == "ok"
    assert result["completion_committed"] is True
    assert result["repeat_zero_binding_dispatch"] is True
    assert result["exit_envelope_sent"] is True
    assert result["child_exitcode"] == 0
    assert result["child_collected"] is True
    assert result["segment_unlinked"] is True
    assert result["post_unlink_attach_failed"] is True
    assert result["collector_poisoned"] is False
    assert result["event_set_count"] == 2
    assert result["child_event_wait_count"] == 2
    assert result["child_event_clear_count"] == 2
    assert [row["participant_id"] for row in result["binding_rows"]] == [
        0,
        1,
    ]


def test_live_shared_memory_failure_modes_are_fail_closed():
    module = _load_module()
    cases = (
        (
            "tp2_shm_worker_binding_error",
            "loaded checkpoint candidate binding failed: rank=1",
            "ok",
            False,
            True,
            0,
        ),
        (
            "tp2_shm_worker_ack_exception",
            "injected worker acknowledgement exception",
            "error",
            True,
            True,
            0,
        ),
        (
            "tp2_shm_worker_exit_without_ack",
            "acknowledgement",
            "absent",
            True,
            False,
            9,
        ),
    )
    for (
        mode,
        detail,
        acknowledgement_status,
        poisoned,
        exit_sent,
        child_exitcode,
    ) in cases:
        result = module.execute_live_shared_memory_engine_ack_attempt(
            source_root=ROOT,
            prerequisite_artifact=PREREQUISITE_ARTIFACT,
            mode=mode,
            timeout_s=2.0,
            name_prefix=f"qwen35-shm-{mode}",
        )
        assert result["status"] == "PASS"
        assert detail in result["error_detail"]
        assert (
            result["acknowledgement_status"]
            == acknowledgement_status
        )
        assert result["collector_poisoned"] is poisoned
        assert result["completion_committed"] is False
        assert result["completion_configuration"] is None
        assert result["binding_rows"] is None
        assert result["exit_envelope_sent"] is exit_sent
        assert result["child_exitcode"] == child_exitcode
        assert result["child_collected"] is True
        assert result["segment_unlinked"] is True
        assert result["post_unlink_attach_failed"] is True
        assert result["dispatch_count"] == (2 if exit_sent else 1)
        assert result["binding_dispatch_count"] == 1
        assert result["collector_call_count"] == 1


def test_orchestration_contract_and_partial_rejection():
    module = _load_module()
    assert module.ATTEMPT_MODES == (
        "tp2_shm_success",
        "tp2_shm_worker_binding_error",
        "tp2_shm_worker_ack_exception",
        "tp2_shm_worker_exit_without_ack",
    )
    prerequisite = module.load_live_shared_memory_prerequisite(
        PREREQUISITE_ARTIFACT
    )
    hashes = module._source_hashes(ROOT)
    assert len(hashes) == 55
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
    audit = module.audit_live_shared_memory_source(ROOT)
    assert audit["llm_engine_import_count"] == 0
    assert audit["model_runner_import_count"] == 0
    assert audit["llm_engine_construction_count"] == 0
    assert audit["model_runner_construction_count"] == 0
    assert audit["fixed_tinyvllm_shared_memory_count"] == 0
    assert audit["shared_memory_create_count"] == 2
    assert audit["shared_memory_attach_count"] == 4
    assert audit["checkpoint_call_count"] == 0
    assert audit["scheduler_call_count"] == 0
    assert audit["step_call_count"] == 0
    assert audit["cuda_call_count"] == 0
    assert audit["forward_call_count"] == 0
    assert audit["inference_call_count"] == 0


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 live shared-memory Engine ack dispatch tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
