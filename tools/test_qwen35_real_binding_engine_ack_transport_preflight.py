from __future__ import annotations

import importlib.util
from itertools import count
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_real_binding_engine_ack_transport_preflight.py"
)
PREREQUISITE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-model-runner-published-binding-20260728-100419/"
    "model_runner_published_candidate_binding_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_engine_ack_transport_preflight_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prerequisite_source_closure_and_method_identities():
    module = _load_module()
    prerequisite = module.load_engine_ack_transport_prerequisite(
        PREREQUISITE_ARTIFACT
    )

    assert prerequisite.artifact_sha256 == (
        "79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a"
    )
    assert prerequisite.source_tree_sha256 == (
        "0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785"
    )
    assert tuple(prerequisite.rows) == (
        (1, 0, "success"),
        (1, 0, "injected_bridge_conflict"),
        (2, 0, "success"),
        (2, 0, "injected_bridge_conflict"),
        (2, 1, "success"),
        (2, 1, "injected_bridge_conflict"),
    )
    assert len(prerequisite.source_file_sha256) == 51
    assert len(module.SOURCE_FILES) == 54
    assert len(set(module.SOURCE_FILES)) == 54
    assert set(module.SOURCE_FILES) - set(
        prerequisite.source_file_sha256
    ) == {
        "tinyvllm/engine/llm_engine.py",
        "tinyvllm/engine/model_runner_command_ack.py",
        "tools/qwen35_real_binding_engine_ack_transport_preflight.py",
    }

    methods = module.load_frozen_engine_ack_transport_methods(
        ROOT,
        fingerprint_validator=lambda value: value,
    )
    assert set(methods) == {
        "call_model_runner_acknowledged",
        "bind_qwen35_loaded_checkpoint_candidates",
        "dispatch_command",
    }
    assert all(callable(method) for method in methods.values())


def _validate_fingerprint(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("model_fingerprint must be a lowercase SHA256")
    return value


def _success_rows():
    record = json.loads(PREREQUISITE_ARTIFACT.read_text())
    return {
        (row["tp_size"], row["tp_rank"]): row["method_row"]
        for row in record["rows"]
        if row["mode"] == "success"
    }


def _rows_by_mode():
    record = json.loads(PREREQUISITE_ARTIFACT.read_text())
    return {
        (
            row["tp_size"],
            row["tp_rank"],
            row["mode"],
        ): row["method_row"]
        for row in record["rows"]
    }


def _bind_methods(module):
    return module.load_frozen_engine_ack_transport_methods(
        ROOT,
        fingerprint_validator=_validate_fingerprint,
    )


def test_frozen_engine_methods_compose_tp1_local_commit():
    module = _load_module()
    methods = _bind_methods(module)
    row = _success_rows()[(1, 0)]

    class Runner:
        world_size = 1

        def bind_published_qwen35_loaded_checkpoint_candidate(self):
            return dict(row)

    engine = SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=None,
        qwen35_loaded_checkpoint_candidate_binding_configuration=None,
        qwen35_loaded_checkpoint_candidate_binding_rows=None,
    )
    engine.call_model_runner_acknowledged = (
        lambda method_name, *args, timeout_s: methods[
            "call_model_runner_acknowledged"
        ](
            engine,
            method_name,
            *args,
            timeout_s=timeout_s,
        )
    )

    result = methods["bind_qwen35_loaded_checkpoint_candidates"](
        engine,
        timeout_s=0.25,
    )

    assert result == (row,)
    assert (
        engine.qwen35_loaded_checkpoint_candidate_binding_configuration
        == (
            row["model_fingerprint"],
            row["layout_fingerprint"],
            row["dtype"],
            0.25,
        )
    )


def test_frozen_engine_methods_compose_tp2_dispatch_collect_and_replay():
    module = _load_module()
    methods = _bind_methods(module)
    rows = _success_rows()
    envelopes = []
    collect_calls = []

    class Runner:
        world_size = 2
        rank = 0
        _command_ids = count()

        def write_shm(self, envelope):
            envelopes.append(envelope)

        def dispatch_command(
            self,
            method_name,
            *args,
            requires_ack,
        ):
            return methods["dispatch_command"](
                self,
                method_name,
                *args,
                requires_ack=requires_ack,
            )

        def bind_published_qwen35_loaded_checkpoint_candidate(self):
            return dict(rows[(2, 0)])

    class Collector:
        def collect(
            self,
            command_id,
            *,
            expected_ranks,
            timeout_s,
            is_rank_alive,
        ):
            collect_calls.append(
                (
                    command_id,
                    expected_ranks,
                    timeout_s,
                    is_rank_alive(1),
                )
            )
            return (
                module.ack_module.ModelRunnerCommandAck(
                    command_id=command_id,
                    rank=1,
                    status="ok",
                    result=dict(rows[(2, 1)]),
                ),
            )

    engine = SimpleNamespace(
        model_runner=Runner(),
        model_runner_ack_collector=Collector(),
        qwen35_loaded_checkpoint_candidate_binding_configuration=None,
        qwen35_loaded_checkpoint_candidate_binding_rows=None,
        _is_worker_rank_alive=lambda rank: rank == 1,
    )
    engine.call_model_runner_acknowledged = (
        lambda method_name, *args, timeout_s: methods[
            "call_model_runner_acknowledged"
        ](
            engine,
            method_name,
            *args,
            timeout_s=timeout_s,
        )
    )

    first = methods["bind_qwen35_loaded_checkpoint_candidates"](
        engine,
        timeout_s=0.5,
    )
    assert first == (rows[(2, 0)], rows[(2, 1)])
    assert len(envelopes) == 1
    assert envelopes[0] == module.ack_module.ModelRunnerCommandEnvelope(
        command_id=0,
        method_name=(
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ),
        args=(),
        requires_ack=True,
    )
    assert collect_calls == [(0, (1,), 0.5, True)]

    second = methods["bind_qwen35_loaded_checkpoint_candidates"](
        engine,
        timeout_s=0.5,
    )
    assert second is first
    assert len(envelopes) == 1
    assert len(collect_calls) == 1


def test_real_pipe_tp2_success_transport_and_replay():
    module = _load_module()
    rows = _rows_by_mode()
    result = module.execute_engine_ack_transport_attempt(
        source_root=ROOT,
        mode="tp2_success",
        local_row=rows[(2, 0, "success")],
        worker_row=rows[(2, 1, "success")],
        timeout_s=2.0,
    )

    assert result["status"] == "PASS"
    assert result["dispatch_count"] == 1
    assert result["collector_call_count"] == 1
    assert result["command_send_count"] == 1
    assert result["child_receive_count"] == 1
    assert result["envelope"] == {
        "command_id": 0,
        "method_name": (
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ),
        "args": [],
        "requires_ack": True,
    }
    assert result["acknowledgement_status"] == "ok"
    assert result["binding_rows"] == [
        rows[(2, 0, "success")],
        rows[(2, 1, "success")],
    ]
    assert result["completion_committed"] is True
    assert result["repeat_zero_dispatch"] is True
    assert result["collector_poisoned"] is False
    assert result["child_process_id"] > 0
    assert result["child_exitcode"] == 0
    assert result["child_collected"] is True


def test_real_pipe_failure_modes_are_fail_closed():
    module = _load_module()
    rows = _rows_by_mode()
    cases = (
        (
            "tp1_local_binding_error",
            rows[(1, 0, "injected_bridge_conflict")],
            None,
            "loaded checkpoint candidate binding failed: rank=0",
            False,
        ),
        (
            "tp2_worker_binding_error",
            rows[(2, 0, "success")],
            rows[(2, 1, "injected_bridge_conflict")],
            "loaded checkpoint candidate binding failed: rank=1",
            False,
        ),
        (
            "tp2_worker_ack_exception",
            rows[(2, 0, "success")],
            rows[(2, 1, "success")],
            "injected worker acknowledgement exception",
            True,
        ),
        (
            "tp2_worker_exit_without_ack",
            rows[(2, 0, "success")],
            rows[(2, 1, "success")],
            "acknowledgement receive failed",
            True,
        ),
    )
    for (
        mode,
        local_row,
        worker_row,
        detail,
        poisoned,
    ) in cases:
        result = module.execute_engine_ack_transport_attempt(
            source_root=ROOT,
            mode=mode,
            local_row=local_row,
            worker_row=worker_row,
            timeout_s=2.0,
        )
        assert result["status"] == "PASS"
        assert detail in result["error_detail"]
        assert result["completion_committed"] is False
        assert result["completion_configuration"] is None
        assert result["binding_rows"] is None
        assert result["collector_poisoned"] is poisoned
        assert result["child_collected"] is True
        if mode == "tp1_local_binding_error":
            assert result["dispatch_count"] == 0
            assert result["child_process_id"] is None
        else:
            assert result["dispatch_count"] == 1
            assert result["child_process_id"] > 0
            assert result["child_exitcode"] == 0
        if mode == "tp2_worker_binding_error":
            assert result["acknowledgement_status"] == "ok"
        if mode == "tp2_worker_ack_exception":
            assert result["acknowledgement_status"] == "error"
        if mode == "tp2_worker_exit_without_ack":
            assert result["acknowledgement_status"] == "absent"


def test_orchestration_contract_and_partial_rejection():
    module = _load_module()
    assert module.WORKER_MODES == (
        "tp1_success",
        "tp1_local_binding_error",
        "tp2_success",
        "tp2_worker_binding_error",
        "tp2_worker_ack_exception",
        "tp2_worker_exit_without_ack",
    )
    prerequisite = module.load_engine_ack_transport_prerequisite(
        PREREQUISITE_ARTIFACT
    )
    hashes = module._source_hashes(ROOT)
    assert len(hashes) == 54
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
        assert "worker rows" in str(error)
    else:
        raise AssertionError("partial finalization must fail")


def test_static_safety_audit():
    module = _load_module()
    audit = module.audit_engine_ack_transport_source(ROOT)
    assert audit == {
        "llm_engine_import_count": 0,
        "model_runner_import_count": 0,
        "llm_engine_construction_count": 0,
        "model_runner_construction_count": 0,
        "frozen_method_invocation_count": {
            name: 1 for name in module.METHOD_SOURCE_SHA256
        },
        "ack_collector_constructor_count": 1,
        "ack_executor_call_count": 1,
        "checkpoint_call_count": 0,
        "scheduler_call_count": 0,
        "step_call_count": 0,
        "cuda_call_count": 0,
        "forward_call_count": 0,
        "inference_call_count": 0,
    }


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 real binding Engine ack transport tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
