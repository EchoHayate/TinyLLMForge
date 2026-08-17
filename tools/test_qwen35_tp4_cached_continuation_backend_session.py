from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
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
    "qwen35_tp4_cached_continuation_contract_for_backend_test",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
engine_executor = _load(
    "qwen35_tp4_engine_correctness_executor_for_cached_backend_test",
    "qwen35_tp4_engine_correctness_executor.py",
)
backend = _load(
    "qwen35_tp4_cached_continuation_backend_session",
    "qwen35_tp4_cached_continuation_backend_session.py",
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


class FakeLogits:
    def __init__(self, value):
        self.value = float(value)

    def clone(self):
        return FakeLogits(self.value)


class FakeEngine:

    def __init__(
        self,
        *,
        hit,
        generated_tokens,
        executed_prefill_tokens,
        sample_on_prefill=False,
        count_restore_miss=True,
    ):
        self.hit = hit
        self.generated_tokens = generated_tokens
        self.executed_prefill_tokens = executed_prefill_tokens
        self.sample_on_prefill = sample_on_prefill
        self.count_restore_miss = count_restore_miss
        self.model_runner = SimpleNamespace(world_size=4, rank=0)
        self.ps = [SimpleNamespace() for _ in range(3)]
        self.calls = []
        self.phase = None
        self.request_calls = 0
        self.steps = []
        self.recording = False
        self.last_logits = None
        self.last_step_observation = None
        self.exit_calls = 0
        self.snapshot = {
            "current_entries": 0,
            "hits": 0,
            "misses": 0,
            "publication_commits": 0,
            "invalidations": 0,
            "clears": 0,
            "last_publication_block_identities": [],
        }

    def configure_qwen35_hybrid_prefix_publication_runtime(self, **kwargs):
        self.calls.append(("configure", kwargs))

    def qwen35_hybrid_prefix_authority_snapshots(self, *, timeout_s):
        return tuple(
            {"rank": rank, **self.snapshot}
            for rank in range(4)
        )

    def add_request(self, prompt, sampling_params):
        self.calls.append(("add", list(prompt), sampling_params.max_tokens))
        self.request_calls += 1
        if self.request_calls % 2 == 1:
            self.phase = "source"
            self.steps = [(1, [0])]
        else:
            self.phase = "continuation"
            if self.sample_on_prefill:
                self.steps = [
                    (self.executed_prefill_tokens, [0]),
                    *[
                        (-1, list(range(index + 1)))
                        for index in range(1, self.generated_tokens)
                    ],
                ]
            else:
                self.steps = [
                    (self.executed_prefill_tokens, []),
                    *[
                        (-1, list(range(index + 1)))
                        for index in range(self.generated_tokens)
                    ],
                ]

    def is_finished(self):
        return not self.steps

    def step(self):
        num_tokens, output = self.steps.pop(0)
        if self.phase == "source" and not self.steps:
            self.snapshot.update({
                "current_entries": 1,
                "publication_commits": 1,
                "last_publication_block_identities": [[7, 2, 99]],
            })
        sampled = self.phase == "continuation" and bool(output)
        if sampled:
            self.last_logits = FakeLogits(len(output))
        self.last_step_observation = {
            "do_sample": sampled,
            "new_completion_tokens_by_seq": {
                17: [output[-1]] if output else [],
            },
        }
        if self.phase == "continuation" and sampled:
            if not self.steps:
                if self.hit:
                    self.snapshot["hits"] += 1
                elif self.count_restore_miss:
                    self.snapshot["misses"] += 1
        return ([(17, output)] if output else [], num_tokens)

    def enable_step_logits_authority_recording(
        self,
        enabled,
        *,
        timeout_s,
    ):
        self.recording = bool(enabled)
        self.last_logits = None
        return {
            "enabled": self.recording,
            "rank_inventory": [0, 1, 2, 3],
        }

    def read_step_logits_authority(self):
        return self.last_logits.clone()

    def invalidate_qwen35_hybrid_prefix_blocks(
        self,
        block_identities,
        *,
        timeout_s,
    ):
        self.snapshot["invalidations"] += 1
        self.snapshot["current_entries"] = 0
        return tuple(
            {"rank": rank, "invalidated_entries": 1}
            for rank in range(4)
        )

    def clear_qwen35_hybrid_prefix_caches(self, *, timeout_s):
        self.snapshot["clears"] += 1
        self.snapshot["current_entries"] = 0
        return tuple(
            {"rank": rank, "cleared_entries": 1}
            for rank in range(4)
        )

    def exit(self):
        self.exit_calls += 1
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {"rank": rank, "process_group_destroyed": True}
                for rank in range(4)
            ],
        }


def _reference_provider(**kwargs):
    generated = kwargs["generated_tokens"]
    return {
        "output_token_ids": list(range(generated)),
        "step_logits": [
            FakeLogits(index + 1)
            for index in range(generated)
        ],
    }


def _compare(engine_logits, reference_logits, *, atol):
    diffs = [
        abs(left.value - right.value)
        for left, right in zip(engine_logits, reference_logits)
    ]
    maximum = max(diffs, default=0.0)
    return {
        "max_abs_diff": maximum,
        "allclose": maximum <= atol,
        "first_mismatch_step": next(
            (
                index
                for index, difference in enumerate(diffs)
                if difference > atol
            ),
            None,
        ),
    }


def _run_case(workload, request_index):
    payload = contract.workload_payload(workload)
    spec = payload["spec"]
    hit = workload in contract.HIT_WORKLOADS
    engine = FakeEngine(
        hit=hit,
        generated_tokens=spec["generated_tokens"],
        executed_prefill_tokens=(
            spec["suffix_tokens"]
            if hit
            else spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        ),
        count_restore_miss=not (
            workload == "w4_miss_invalidation"
            and request_index == 0
        ),
    )
    session = backend.CachedContinuationBackendSession(
        _configuration(),
        workload=workload,
        request_index=request_index,
        payload=payload,
        engine_factory=lambda configuration: engine,
        reference_provider=_reference_provider,
        logits_comparator=_compare,
    )
    return session.run(), engine


def test_backend_session_proves_hit_from_runtime_deltas_and_logits():
    row, engine = _run_case("w1_medium_reuse", 0)
    spec = contract.workload_payload("w1_medium_reuse")["spec"]
    assert row["restore_hit"] is True
    assert row["restore_reason"] == "exact_hit"
    assert row["reused_tokens"] == spec["shared_prefix_tokens"]
    assert row["executed_prefill_tokens"] == spec["suffix_tokens"]
    assert row["logits_allclose"] is True
    assert row["logits_max_abs_diff"] == 0.0
    assert engine.recording is False
    assert row["process_group_destroyed"] is True


def test_backend_session_records_first_token_logits_from_prefill_sample():
    payload = contract.workload_payload("w1_medium_reuse")
    spec = payload["spec"]
    engine = FakeEngine(
        hit=True,
        generated_tokens=spec["generated_tokens"],
        executed_prefill_tokens=spec["suffix_tokens"],
        sample_on_prefill=True,
    )
    session = backend.CachedContinuationBackendSession(
        _configuration(),
        workload="w1_medium_reuse",
        request_index=0,
        payload=payload,
        engine_factory=lambda configuration: engine,
        reference_provider=_reference_provider,
        logits_comparator=_compare,
    )
    row = session.run()
    assert row["logits_allclose"] is True
    assert row["output_token_ids"] == list(
        range(spec["generated_tokens"])
    )


def test_backend_session_applies_each_w4_invalidation():
    expected_calls = {
        0: None,
        1: "invalidate",
        2: "clear",
    }
    for request_index, expected_call in expected_calls.items():
        row, engine = _run_case(
            "w4_miss_invalidation",
            request_index,
        )
        assert row["restore_hit"] is False
        assert row["restore_reason"] == (
            contract.W4_EXPECTED_REASONS[request_index]
        )
        names = [call[0] for call in engine.calls]
        if expected_call is not None:
            assert expected_call in names or (
                expected_call == "invalidate"
                and engine.snapshot["invalidations"] == 1
            ) or (
                expected_call == "clear"
                and engine.snapshot["clears"] == 1
            )


def test_backend_session_rejects_counter_or_logits_mismatch():
    payload = contract.workload_payload("w1_medium_reuse")
    spec = payload["spec"]
    engine = FakeEngine(
        hit=False,
        generated_tokens=spec["generated_tokens"],
        executed_prefill_tokens=spec["suffix_tokens"],
    )
    session = backend.CachedContinuationBackendSession(
        _configuration(),
        workload="w1_medium_reuse",
        request_index=0,
        payload=payload,
        engine_factory=lambda configuration: engine,
        reference_provider=_reference_provider,
        logits_comparator=_compare,
    )
    try:
        session.run()
    except ValueError as error:
        message = str(error)
        assert "restore counters" in message
        assert "workload=w1_medium_reuse" in message
        assert "request_index=0" in message
        assert "expected_delta=(1, 0)" in message
        assert "observed_delta=(0, 1)" in message
    else:
        raise AssertionError("runtime miss was reported as hit")

    bad = FakeEngine(
        hit=True,
        generated_tokens=spec["generated_tokens"],
        executed_prefill_tokens=spec["suffix_tokens"],
    )

    def shifted_reference(**kwargs):
        result = _reference_provider(**kwargs)
        result["step_logits"][0] = FakeLogits(999)
        return result

    session = backend.CachedContinuationBackendSession(
        _configuration(),
        workload="w1_medium_reuse",
        request_index=0,
        payload=payload,
        engine_factory=lambda configuration: bad,
        reference_provider=shifted_reference,
        logits_comparator=_compare,
    )
    try:
        session.run()
    except ValueError as error:
        message = str(error)
        assert "logits" in message
        assert "workload=w1_medium_reuse" in message
        assert "request_index=0" in message
        assert "max_abs_diff=998.0" in message
        assert "first_mismatch_step=0" in message
    else:
        raise AssertionError("mismatched logits were accepted")


def test_default_engine_and_reference_boundaries_are_fail_closed():
    payload = contract.workload_payload("w1_medium_reuse")
    session = backend.CachedContinuationBackendSession(
        _configuration(),
        workload="w1_medium_reuse",
        request_index=0,
        payload=payload,
    )
    try:
        session.run()
    except RuntimeError as error:
        assert "reference provider" in str(error) or "Engine" in str(error)
    else:
        raise AssertionError("unconfigured real backend was enabled")


def test_default_engine_factory_reserves_a_release_handoff_slot():
    observed = {}
    original_import = importlib.import_module

    def engine_factory(configuration, **kwargs):
        observed["configuration"] = configuration
        observed["kwargs"] = dict(kwargs)
        return "engine"

    def fake_import(name):
        if name == "qwen35_tp4_engine_backend_session":
            return SimpleNamespace(
                _default_engine_factory=engine_factory,
            )
        return original_import(name)

    backend.importlib.import_module = fake_import
    try:
        configuration = _configuration()
        result = backend._default_engine_factory(configuration)
    finally:
        backend.importlib.import_module = original_import

    assert result == "engine"
    assert observed == {
        "configuration": configuration,
        "kwargs": {
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 2,
        },
    }


def test_session_factory_precomputes_all_references_once_and_releases_gpu():
    reference_executors = []
    engine_calls = []

    class FakeReferenceExecutor:
        def __init__(self):
            self.calls = []
            self.closed = False

        def generate_reference_with_step_logits(self, **kwargs):
            self.calls.append(kwargs)
            generated = kwargs["generated_tokens"]
            return {
                "output_token_ids": list(range(generated)),
                "step_logits": [
                    FakeLogits(index)
                    for index in range(generated)
                ],
            }

        def close(self):
            self.closed = True

    def reference_executor_factory():
        value = FakeReferenceExecutor()
        reference_executors.append(value)
        return value

    def engine_factory(configuration):
        engine_calls.append(configuration)
        return object()

    factory = backend.CachedContinuationSessionFactory(
        _configuration(),
        engine_factory=engine_factory,
        reference_executor_factory=reference_executor_factory,
    )
    assert reference_executors == []
    payload = contract.workload_payload("w1_medium_reuse")
    first = factory(
        _configuration(),
        workload="w1_medium_reuse",
        request_index=0,
        payload=payload,
    )
    second = factory(
        _configuration(),
        workload="w1_medium_reuse",
        request_index=1,
        payload=payload,
    )
    assert len(reference_executors) == 1
    assert reference_executors[0].closed is True
    assert len(reference_executors[0].calls) == 19
    first_prompt = backend._request_prompt(payload, 0)
    second_prompt = backend._request_prompt(payload, 1)
    first_reference = first.reference_provider(
        workload="w1_medium_reuse",
        request_index=0,
        prompt_token_ids=first_prompt,
        generated_tokens=payload["spec"]["generated_tokens"],
    )
    second_reference = second.reference_provider(
        workload="w1_medium_reuse",
        request_index=1,
        prompt_token_ids=second_prompt,
        generated_tokens=payload["spec"]["generated_tokens"],
    )
    assert len(reference_executors) == 1
    assert first_reference["output_token_ids"] == list(
        range(payload["spec"]["generated_tokens"])
    )
    assert second_reference["output_token_ids"] == list(
        range(payload["spec"]["generated_tokens"])
    )
    assert all(
        executor.closed is True
        for executor in reference_executors
    )
    assert {
        call["scenario"]
        for call in reference_executors[0].calls
    } == {"publish_source"}
    factory.close()
    factory.close()


def test_session_factory_reuses_one_engine_and_cleans_up_only_at_close():
    payload = contract.workload_payload("w1_medium_reuse")
    spec = payload["spec"]
    engine = FakeEngine(
        hit=True,
        generated_tokens=spec["generated_tokens"],
        executed_prefill_tokens=spec["suffix_tokens"],
    )
    engine_calls = []

    class FakeReferenceExecutor:
        def generate_reference_with_step_logits(self, **kwargs):
            generated = kwargs["generated_tokens"]
            return {
                "output_token_ids": list(range(generated)),
                "step_logits": [
                    FakeLogits(index + 1)
                    for index in range(generated)
                ],
            }

        def close(self):
            pass

    def engine_factory(configuration):
        engine_calls.append(configuration)
        return engine

    factory = backend.CachedContinuationSessionFactory(
        _configuration(),
        engine_factory=engine_factory,
        reference_executor_factory=FakeReferenceExecutor,
        logits_comparator=_compare,
    )
    rows = []
    for request_index in (0, 1):
        session = factory(
            _configuration(),
            workload="w1_medium_reuse",
            request_index=request_index,
            payload=payload,
        )
        rows.append(session.run())
        session.close()

    assert len(engine_calls) == 1
    assert engine.exit_calls == 0
    assert [row["process_group_destroyed"] for row in rows] == [
        False,
        False,
    ]
    assert engine.snapshot["clears"] == 2

    factory.close()
    factory.close()

    assert engine.exit_calls == 1
    assert [row["process_group_destroyed"] for row in rows] == [
        True,
        True,
    ]
    assert all(row["owned_children_remaining"] == [] for row in rows)


def test_session_factory_rejects_configuration_before_reference_loading():
    reference_calls = []
    factory = backend.CachedContinuationSessionFactory(
        _configuration(),
        engine_factory=lambda configuration: object(),
        reference_executor_factory=lambda: (
            reference_calls.append("loaded") or object()
        ),
    )
    payload = _configuration().to_payload()
    payload.pop("world_size")
    payload["gpu_indices"] = tuple(payload["gpu_indices"])
    payload["dist_port"] += 10
    changed = engine_executor.ExecutorConfiguration(**payload)
    try:
        factory(
            changed,
            workload="w1_medium_reuse",
            request_index=0,
            payload=contract.workload_payload("w1_medium_reuse"),
        )
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("changed configuration was accepted")
    assert reference_calls == []


def test_reference_corpus_failure_closes_executor_and_stays_retryable():
    executors = []

    class FailingReferenceExecutor:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def generate_reference_with_step_logits(self, **kwargs):
            self.calls += 1
            raise RuntimeError("synthetic reference failure")

        def close(self):
            self.closed = True

    def reference_executor_factory():
        value = FailingReferenceExecutor()
        executors.append(value)
        return value

    factory = backend.CachedContinuationSessionFactory(
        _configuration(),
        engine_factory=lambda configuration: object(),
        reference_executor_factory=reference_executor_factory,
    )
    for _ in range(2):
        try:
            factory(
                _configuration(),
                workload="w1_medium_reuse",
                request_index=0,
                payload=contract.workload_payload(
                    "w1_medium_reuse"
                ),
            )
        except RuntimeError as error:
            assert "synthetic" in str(error)
        else:
            raise AssertionError("reference failure was hidden")
    assert len(executors) == 2
    assert all(value.closed for value in executors)
    assert factory.reference_corpus is None


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cached-continuation backend session tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
