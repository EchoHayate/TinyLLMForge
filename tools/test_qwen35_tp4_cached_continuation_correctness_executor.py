from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_cached_continuation_contract_for_executor_test",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
engine_executor = _load(
    "qwen35_tp4_engine_correctness_executor_for_cached_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
executor = _load(
    "qwen35_tp4_cached_continuation_correctness_executor",
    "qwen35_tp4_cached_continuation_correctness_executor.py",
)


def _configuration():
    return engine_executor.ExecutorConfiguration(
        model_dir="/models/qwen35",
        model_manifest_path="/authority/model_manifest.json",
        model_manifest_sha256="a" * 64,
        source_tree_sha256="b" * 64,
        workload_manifest_path="/authority/workload_manifest.json",
        workload_manifest_sha256=contract.WORKLOAD_MANIFEST_SHA256,
        model_fingerprint="qwen35-m8-authority",
        gpu_indices=(0, 1, 2, 3),
        dist_port=31001,
        master_port=31002,
        max_cache_entries=32,
        max_cache_bytes=1 << 30,
        timeout_s=600.0,
    )


def _valid_row(workload, request_index, payload):
    spec = payload["spec"]
    hit = workload in contract.HIT_WORKLOADS
    generated = spec["generated_tokens"]
    return {
        "workload": workload,
        "request_index": request_index,
        "outcome": "continuation",
        "restore_hit": hit,
        "restore_reason": (
            "exact_hit"
            if hit
            else contract.W4_EXPECTED_REASONS[request_index]
        ),
        "prompt_tokens": (
            spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        ),
        "reused_tokens": spec["shared_prefix_tokens"] if hit else 0,
        "executed_prefill_tokens": (
            spec["suffix_tokens"]
            if hit
            else spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        ),
        "output_token_ids": list(range(generated)),
        "reference_output_token_ids": list(range(generated)),
        "logits_max_abs_diff": 0.0,
        "logits_allclose": True,
        "cache_identity_match": True,
        "rank_inventory": [0, 1, 2, 3],
        "process_group_destroyed": True,
        "owned_children_remaining": [],
    }


class FakeSession:
    def __init__(
        self,
        *,
        workload,
        request_index,
        payload,
        corrupt=None,
    ):
        self.workload = workload
        self.request_index = request_index
        self.payload = payload
        self.corrupt = corrupt
        self.closed = False

    def run(self):
        row = _valid_row(
            self.workload,
            self.request_index,
            self.payload,
        )
        if self.corrupt == "logits":
            row["logits_allclose"] = False
            row["logits_max_abs_diff"] = 1.0
        if self.corrupt == "cleanup":
            row["process_group_destroyed"] = False
        return row

    def close(self):
        self.closed = True


def test_executor_runs_fresh_session_per_row_and_validates_evidence():
    sessions = []

    def factory(
        configuration,
        *,
        workload,
        request_index,
        payload,
    ):
        assert configuration.to_payload() == _configuration().to_payload()
        session = FakeSession(
            workload=workload,
            request_index=request_index,
            payload=payload,
        )
        sessions.append(session)
        return session

    runtime = executor.CachedContinuationExecutor(
        _configuration(),
        session_factory=factory,
    )
    rows = []
    for workload in contract.WORKLOADS:
        payload = contract.workload_payload(workload)
        for request_index in range(payload["spec"]["continuations"]):
            rows.append(runtime.run_continuation(
                workload=workload,
                request_index=request_index,
                payload=payload,
            ))
    assert contract.classify_rows(rows)["classification"] == "PASS"
    assert len(sessions) == 19
    assert all(session.closed for session in sessions)
    runtime.close()


def test_executor_rejects_invalid_session_evidence_and_closes():
    for corruption in ("logits", "cleanup"):
        sessions = []

        def factory(
            configuration,
            *,
            workload,
            request_index,
            payload,
        ):
            session = FakeSession(
                workload=workload,
                request_index=request_index,
                payload=payload,
                corrupt=corruption,
            )
            sessions.append(session)
            return session

        runtime = executor.CachedContinuationExecutor(
            _configuration(),
            session_factory=factory,
        )
        payload = contract.workload_payload("w1_medium_reuse")
        try:
            runtime.run_continuation(
                workload="w1_medium_reuse",
                request_index=0,
                payload=payload,
            )
        except ValueError as error:
            assert "classification" in str(error)
        else:
            raise AssertionError(
                f"{corruption} evidence was accepted"
            )
        assert sessions[0].closed is True


def test_executor_rejects_payload_or_order_mismatch_before_session():
    calls = []

    def factory(
        configuration,
        *,
        workload,
        request_index,
        payload,
    ):
        calls.append((workload, request_index))
        return FakeSession(
            workload=workload,
            request_index=request_index,
            payload=payload,
        )

    runtime = executor.CachedContinuationExecutor(
        _configuration(),
        session_factory=factory,
    )
    payload = contract.workload_payload("w1_medium_reuse")
    changed = {**payload, "workload": "wrong"}
    try:
        runtime.run_continuation(
            workload="w1_medium_reuse",
            request_index=0,
            payload=changed,
        )
    except ValueError as error:
        assert "payload" in str(error)
    else:
        raise AssertionError("changed workload payload was accepted")
    assert calls == []

    runtime.run_continuation(
        workload="w1_medium_reuse",
        request_index=0,
        payload=payload,
    )
    try:
        runtime.run_continuation(
            workload="w1_medium_reuse",
            request_index=0,
            payload=payload,
        )
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("duplicate row order was accepted")


def test_configured_factory_bridges_producer_without_environment_reads():
    sessions = []

    def session_factory(
        configuration,
        *,
        workload,
        request_index,
        payload,
    ):
        session = FakeSession(
            workload=workload,
            request_index=request_index,
            payload=payload,
        )
        sessions.append(session)
        return session

    factory = executor.build_configured_executor_factory(
        _configuration().to_payload(),
        session_factory=session_factory,
    )
    runtime = factory()
    payload = contract.workload_payload("w1_medium_reuse")
    row = runtime.run_continuation(
        workload="w1_medium_reuse",
        request_index=0,
        payload=payload,
    )
    assert row["restore_hit"] is True
    assert len(sessions) == 1


def test_default_session_factory_is_fail_closed():
    runtime = executor.CachedContinuationExecutor(_configuration())
    payload = contract.workload_payload("w1_medium_reuse")
    try:
        runtime.run_continuation(
            workload="w1_medium_reuse",
            request_index=0,
            payload=payload,
        )
    except RuntimeError as error:
        assert "session is not implemented" in str(error)
    else:
        raise AssertionError("real cached runtime was silently enabled")


def test_executor_close_propagates_to_stateful_session_factory():
    class StatefulFactory:
        def __init__(self):
            self.closed = False

        def __call__(self, *args, **kwargs):
            raise AssertionError("session creation was not expected")

        def close(self):
            self.closed = True

    session_factory = StatefulFactory()
    runtime = executor.CachedContinuationExecutor(
        _configuration(),
        session_factory=session_factory,
    )
    runtime.close()
    runtime.close()
    assert session_factory.closed is True


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cached-continuation executor tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
