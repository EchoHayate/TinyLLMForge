from __future__ import annotations

import importlib.util
import inspect
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
    "qwen35_tp4_hybrid_prefix_contract_for_engine_adapter_test",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
adapter = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_engine_adapter",
    "qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py",
)


class FakeSamplingParams:

    def __init__(self, *, temperature, max_tokens, ignore_eos):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos


class FakeSequence:

    def __init__(self, seq_id, prompt, sampling_params):
        self.seq_id = seq_id
        self.prompt = list(prompt)
        self.max_tokens = sampling_params.max_tokens
        self.output = []


class FakeScheduler:

    def __init__(self):
        self.waiting = []


class FakeEngine:

    def __init__(self):
        self.scheduler = FakeScheduler()
        self.active = []
        self.sequences = {}
        self.next_seq_id = 0
        self.last_step_observation = None
        self.configured = False
        self.cache_present = False
        self.cache_source_prompt = None
        self.cache_invalidated = False
        self.cache_cleared = False
        self.add_batches = []
        self.configure_calls = []
        self.capture_configure_calls = []
        self.capture_arm_calls = []
        self.capture_finish_calls = []
        self.invalidate_calls = []
        self.clear_calls = 0
        self.reusable_cache_clear_calls = 0
        self.closed = False
        self.record_logits = False
        self.now = 1000
        self.restore_events = []
        self.decode_profile_enabled = False
        self.decode_profile_configurations = []
        self.decode_profile_finalize_calls = 0

    def configure_qwen35_hybrid_prefix_publication_runtime(self, **kwargs):
        self.configured = True
        self.configure_calls.append(kwargs)

    def configure_qwen35_recurrent_capture(self, **kwargs):
        self.capture_configure_calls.append(kwargs)

    def arm_qwen35_recurrent_capture(
        self,
        workload_id,
        *,
        timeout_s,
    ):
        self.capture_arm_calls.append((workload_id, timeout_s))
        return tuple(
            {
                "rank": rank,
                "workload_id": workload_id,
                "armed": True,
            }
            for rank in range(4)
        )

    def finish_qwen35_recurrent_capture_workload(
        self,
        workload_id,
        *,
        timeout_s,
    ):
        self.capture_finish_calls.append((workload_id, timeout_s))
        return tuple(
            {
                "rank": rank,
                "workload_id": workload_id,
                "complete": True,
            }
            for rank in range(4)
        )

    def add_request(self, prompt, sampling_params):
        sequence = FakeSequence(
            self.next_seq_id,
            prompt,
            sampling_params,
        )
        self.next_seq_id += 1
        self.scheduler.waiting.append(sequence)
        self.active.append(sequence)
        self.sequences[sequence.seq_id] = sequence
        if self.configured:
            self.flush_pending_hybrid_state_releases(
                timeout_s=120.0,
            )
            self.acquire_qwen35_hybrid_prefix(
                sequence,
                object(),
                tuple(prompt[:-64]),
            )

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        self.restore_events.append(("release_flush", timeout_s))
        return ()

    def acquire_qwen35_hybrid_prefix(
        self,
        sequence,
        key,
        token_ids,
    ):
        payload = type("Payload", (), {
            "request_id": sequence.seq_id,
        })()
        self.prepare_model_runner_hybrid_prefix_restore(
            payload,
            timeout_s=120.0,
        )
        self.validate_model_runner_hybrid_prefix_restore(
            payload,
            timeout_s=120.0,
        )
        self.commit_model_runner_hybrid_prefix_restore(
            payload,
            timeout_s=120.0,
        )
        return self._restore_hit(sequence)

    def prepare_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.restore_events.append(("prepare", payload.request_id))
        return ()

    def validate_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.restore_events.append(("validate", payload.request_id))
        return ()

    def commit_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.restore_events.append(("commit", payload.request_id))
        return ()

    def rollback_model_runner_hybrid_prefix_restore(
        self,
        payload,
        *,
        timeout_s,
    ):
        self.restore_events.append(("rollback", payload.request_id))
        return ()

    def _restore_hit(self, sequence):
        if self.cache_source_prompt is None:
            return False
        shared = len(sequence.prompt) - 64
        return (
            self.configured
            and self.cache_present
            and not self.cache_invalidated
            and not self.cache_cleared
            and sequence.prompt[:shared]
            == self.cache_source_prompt[:shared]
        )

    def step(self):
        if self.scheduler.waiting:
            admitted = list(self.scheduler.waiting)
            self.scheduler.waiting.clear()
            self.add_batches.append([seq.seq_id for seq in admitted])
        scheduled = list(self.active)
        first_step = all(not sequence.output for sequence in scheduled)
        scheduled_rows = []
        token_deltas = {}
        prefill_tokens = 0
        for sequence in scheduled:
            restored = first_step and self._restore_hit(sequence)
            prefill_start = (
                len(sequence.prompt) - 64 if restored else 0
            )
            scheduled_rows.append({
                "seq_id": sequence.seq_id,
                "is_decode": not first_step,
                "do_sample": True,
                "prefill_chunk_start": prefill_start,
                "prefill_chunk_end": (
                    len(sequence.prompt) if first_step else 0
                ),
                "prefill_chunk_final": first_step,
            })
            if first_step:
                prefill_tokens += len(sequence.prompt) - prefill_start
            token = 7 + len(sequence.output)
            sequence.output.append(token)
            token_deltas[sequence.seq_id] = [token]
        self.now += 100
        finished = [
            sequence
            for sequence in scheduled
            if len(sequence.output) == sequence.max_tokens
        ]
        outputs = [
            (sequence.seq_id, list(sequence.output))
            for sequence in finished
        ]
        self.active = [
            sequence for sequence in self.active
            if sequence not in finished
        ]
        self.last_step_observation = {
            "step_end_ns": self.now,
            "scheduled": scheduled_rows,
            "new_completion_tokens_by_seq": token_deltas,
            "finished_seq_ids": [
                sequence.seq_id for sequence in finished
            ],
        }
        for sequence in finished:
            if sequence.max_tokens == 1 and self.configured:
                self.cache_present = True
                self.cache_source_prompt = list(sequence.prompt)
                self.cache_invalidated = False
                self.cache_cleared = False
        return outputs, prefill_tokens if first_step else -len(scheduled)

    def is_finished(self):
        return not self.scheduler.waiting and not self.active

    def enable_step_logits_authority_recording(
        self,
        enabled,
        *,
        timeout_s,
    ):
        self.record_logits = enabled
        return {
            "enabled": enabled,
            "rank_inventory": [0, 1, 2, 3],
        }

    def read_step_logits_authority(self):
        assert self.record_logits
        return [
            [float(sequence.seq_id), float(len(sequence.output))]
            for sequence in self.active
        ] or [[0.0, 1.0]]

    def configure_decode_internal_profile(
        self,
        enabled,
        profile_label,
        *,
        timeout_s,
    ):
        self.decode_profile_enabled = enabled
        self.decode_profile_configurations.append({
            "enabled": enabled,
            "profile_label": profile_label,
        })
        return {
            "enabled": enabled,
            "rank_inventory": [0, 1, 2, 3],
        }

    def finalize_decode_internal_profile(self, *, timeout_s):
        self.decode_profile_finalize_calls += 1
        ranks = []
        for rank in range(4):
            steps = [{
                "rank": rank,
                "step_index": 0,
                "batch_kind": "prefill",
                "is_decode": False,
                "decode_ordinal": None,
                "active_sequence_count": 4,
                "request_set_sha256": "a" * 64,
                "wall_ns": 100,
                "cuda_ns": 80,
                "non_cuda_upper_bound_ns": 20,
                "dispatch": "eager",
            }]
            for decode_ordinal in range(63):
                steps.append({
                    "rank": rank,
                    "step_index": decode_ordinal + 1,
                    "batch_kind": "decode",
                    "is_decode": True,
                    "decode_ordinal": decode_ordinal,
                    "active_sequence_count": 4,
                    "request_set_sha256": "a" * 64,
                    "wall_ns": 100,
                    "cuda_ns": 80,
                    "non_cuda_upper_bound_ns": 20,
                    "dispatch": "eager",
                })
            ranks.append({
                "rank": rank,
                "enabled": True,
                "finalization_status": "complete",
                "steps": steps,
                "collectives": [],
            })
        return {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
            "ranks": ranks,
        }

    def qwen35_hybrid_prefix_authority_snapshots(self, *, timeout_s):
        rows = []
        for rank in range(4):
            rows.append({
                "rank": rank,
                "current_entries": int(self.cache_present),
                "hits": 0,
                "misses": 0,
                "publication_commits": int(self.cache_present),
                "invalidations": int(self.cache_invalidated),
                "clears": int(self.cache_cleared),
                "last_publication_block_identities": (
                    [[1, 2, 3]] if self.cache_present else []
                ),
            })
        return tuple(rows)

    def invalidate_qwen35_hybrid_prefix_blocks(
        self,
        block_identities,
        *,
        timeout_s,
    ):
        self.invalidate_calls.append(block_identities)
        self.cache_invalidated = True

    def clear_qwen35_hybrid_prefix_caches(self, *, timeout_s):
        self.clear_calls += 1
        self.cache_cleared = True
        self.cache_present = False

    def clear_reusable_prefix_cache(self):
        self.reusable_cache_clear_calls += 1
        return 1

    def capacity_snapshot(self):
        return {"num_kvcache_blocks": 64, "block_size": 256}

    def memory_snapshots(self, *, timeout_s):
        return tuple({
            "rank": rank,
            "cuda_allocated_bytes": 1,
            "cuda_reserved_bytes": 2,
            "cuda_peak_allocated_bytes": 3,
            "cuda_peak_reserved_bytes": 4,
            "kv_capacity_bytes": 5,
        } for rank in range(4))

    def qwen35_hybrid_prefix_cache_snapshots(self, *, timeout_s):
        return tuple({
            "rank": rank,
            "current_entries": int(self.cache_present),
            "current_bytes": 100 if self.cache_present else 0,
            "current_logical_bytes": 120 if self.cache_present else 0,
            "deduplicated_bytes": 20 if self.cache_present else 0,
            "peak_entries": int(self.cache_present),
            "peak_bytes": 100 if self.cache_present else 0,
            "hits": 1 if self.cache_present else 0,
            "misses": 0,
            "evictions": 0,
            "validation_failures": 0,
            "failed_restores": 0,
        } for rank in range(4))

    def exit(self):
        self.closed = True
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {
                    "rank": rank,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
        }


class FakeClock:

    def __init__(self):
        self.value = 900

    def __call__(self):
        self.value += 10
        return self.value


def _configuration(policy, *, decode_internal=False):
    return {
        "policy": policy,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": 64,
        },
        "hybrid_prefix": {
            "enabled": policy == "exact_restore",
            "representation": (
                "exact_full_fidelity"
                if policy == "exact_restore"
                else "none"
            ),
        },
        "profiling": {
            "enabled": decode_internal,
            "decode_internal": decode_internal,
        },
    }


def _capture_configuration():
    return {
        "capture_root": "/tmp/recurrent-capture",
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "world_size": 4,
        "workload_ids": [
            "w0_short_control",
            "w1_medium_reuse",
            "w2_long_reuse",
            "w3_batched_fanout",
            "w4_miss_invalidation",
        ],
    }


def _adapter(
    policy,
    *,
    recurrent_capture=False,
    profiling=False,
    decode_internal=False,
):
    engine = FakeEngine()
    configuration = _configuration(
        policy,
        decode_internal=decode_internal,
    )
    if profiling:
        configuration["profiling"]["enabled"] = True
    if recurrent_capture:
        configuration["recurrent_calibration_capture"] = (
            _capture_configuration()
        )
    runtime = adapter.BenchmarkEngineAdapter(
        configuration,
        {"model_dir": "/model"},
        engine_factory=lambda configuration, authorized: engine,
        sampling_params_factory=FakeSamplingParams,
        clock_ns=FakeClock(),
    )
    if policy == "exact_restore":
        runtime.configure_qwen35_hybrid_prefix_publication_runtime(
            model_fingerprint="model",
            max_entries=16,
            max_bytes=2 * 1024**3,
            timeout_s=120.0,
        )
    return runtime, engine


def test_decode_internal_profile_is_opt_in_and_finalized():
    runtime, engine = _adapter(
        "recompute",
        profiling=True,
        decode_internal=True,
    )

    runtime.run_benchmark_workload(
        workload="w2_long_reuse",
        workload_spec=contract.workload_payload("w2_long_reuse"),
        phase="measured",
        repetition=0,
        policy="recompute",
    )
    profile = runtime.profile_snapshot()

    assert engine.decode_profile_enabled is True
    assert engine.decode_profile_configurations == [{
        "enabled": True,
        "profile_label": (
            "policy=recompute/"
            "case=w2_long_reuse__measured__r0__recompute"
        ),
    }]
    assert engine.decode_profile_finalize_calls == 1
    assert profile["decode_internal"]["enabled"] is True
    assert profile["decode_internal"]["rank_inventory"] == [0, 1, 2, 3]
    assert len(profile["decode_internal"]["ranks"]) == 4


def test_opt_in_profile_records_restore_phases_and_request_lifecycle():
    runtime, _ = _adapter("exact_restore", profiling=True)

    runtime.run_benchmark_workload(
        workload="w2_long_reuse",
        workload_spec=contract.workload_payload("w2_long_reuse"),
        phase="measured",
        repetition=0,
        policy="exact_restore",
    )

    profile = runtime.profile_snapshot()
    assert profile["enabled"] is True
    names = [event["name"] for event in profile["events"]]
    assert "release_flush" in names
    assert "restore_total" in names
    assert "restore_prepare" in names
    assert "restore_validate" in names
    assert "restore_commit" in names
    assert all(
        isinstance(event["duration_ns"], int)
        and event["duration_ns"] >= 0
        for event in profile["events"]
    )
    continuation = [
        row
        for row in profile["requests"]
        if row["request_id"].startswith("request-")
    ]
    assert len(continuation) == 4
    assert all(
        row["ttft_ns"] > 0
        and row["decode_ns"] > 0
        and row["e2e_ns"] >= row["ttft_ns"]
        for row in continuation
    )


def test_capture_absent_preserves_zero_capture_calls():
    runtime, engine = _adapter("exact_restore")

    runtime.run_benchmark_workload(
        workload="w1_medium_reuse",
        workload_spec=contract.workload_payload("w1_medium_reuse"),
        phase="measured",
        repetition=0,
        policy="exact_restore",
    )

    assert engine.capture_configure_calls == []
    assert engine.capture_arm_calls == []
    assert engine.capture_finish_calls == []


def test_capture_configuration_is_forwarded_once_at_construction():
    runtime, engine = _adapter(
        "exact_restore",
        recurrent_capture=True,
    )

    assert engine.capture_configure_calls == [{
        **_capture_configuration(),
        "timeout_s": 120.0,
    }]
    assert runtime.recurrent_calibration_capture == (
        _capture_configuration()
    )


def test_capture_source_failure_does_not_finish_workload():
    runtime, engine = _adapter(
        "exact_restore",
        recurrent_capture=True,
    )

    def fail_requests(*args, **kwargs):
        raise RuntimeError("synthetic source failure")

    runtime._run_requests = fail_requests
    try:
        runtime._run_source(
            contract.workload_payload("w0_short_control"),
            capture_workload_id="w0_short_control",
        )
    except RuntimeError as error:
        assert str(error) == "synthetic source failure"
    else:
        raise AssertionError("source failure was swallowed")

    assert engine.capture_arm_calls == [
        ("w0_short_control", 120.0)
    ]
    assert engine.capture_finish_calls == []


def test_capture_arms_only_first_w4_source_request():
    runtime, engine = _adapter(
        "exact_restore",
        recurrent_capture=True,
    )

    runtime.run_benchmark_workload(
        workload="w4_miss_invalidation",
        workload_spec=contract.workload_payload(
            "w4_miss_invalidation"
        ),
        phase="measured",
        repetition=0,
        policy="exact_restore",
    )

    assert engine.capture_arm_calls == [
        ("w4_miss_invalidation", 120.0)
    ]
    assert engine.capture_finish_calls == [
        ("w4_miss_invalidation", 120.0)
    ]
    assert len(engine.add_batches) == 6


def test_capture_arms_each_regular_workload_once_and_not_continuations():
    for workload in (
        "w0_short_control",
        "w1_medium_reuse",
        "w2_long_reuse",
        "w3_batched_fanout",
    ):
        runtime, engine = _adapter(
            "exact_restore",
            recurrent_capture=True,
        )

        runtime.run_benchmark_workload(
            workload=workload,
            workload_spec=contract.workload_payload(workload),
            phase="measured",
            repetition=0,
            policy="exact_restore",
        )

        assert engine.capture_arm_calls == [(workload, 120.0)]
        assert engine.capture_finish_calls == [(workload, 120.0)]


def test_recompute_executes_full_prefill_without_restore():
    runtime, engine = _adapter("recompute")
    payload = runtime.run_benchmark_workload(
        workload="w1_medium_reuse",
        workload_spec=contract.workload_payload("w1_medium_reuse"),
        phase="measured",
        repetition=0,
        policy="recompute",
    )

    assert engine.configure_calls == []
    assert engine.reusable_cache_clear_calls == 4
    assert len(payload["requests"]) == 4
    assert all(
        row["restored_hybrid_state"] is False
        and row["reused_kv_tokens"] == 0
        and row["executed_prefill_tokens"] == 1088
        for row in payload["requests"]
    )
    continuation_ids = [
        seq_id
        for batch in engine.add_batches[-4:]
        for seq_id in batch
    ]
    assert all(
        engine.sequences[seq_id].hybrid_prefix_restore_attempted is True
        and engine.sequences[seq_id].hybrid_prefix_restore_hit is False
        for seq_id in continuation_ids
    )


def test_exact_restore_excludes_source_and_reuses_registered_prefix():
    runtime, engine = _adapter("exact_restore")
    payload = runtime.run_benchmark_workload(
        workload="w1_medium_reuse",
        workload_spec=contract.workload_payload("w1_medium_reuse"),
        phase="measured",
        repetition=0,
        policy="exact_restore",
    )

    assert len(engine.configure_calls) == 1
    assert engine.reusable_cache_clear_calls == 0
    assert len(payload["requests"]) == 4
    assert all(
        row["restored_hybrid_state"] is True
        and row["reused_kv_tokens"] == 1024
        and row["executed_prefill_tokens"] == 64
        and row["ttft_ns"] > 0
        and len(row["decode_step_ns"]) == 63
        for row in payload["requests"]
    )


def test_adapter_configure_accepts_and_forwards_exact_restore_representation():
    signature = inspect.signature(
        adapter.BenchmarkEngineAdapter
        .configure_qwen35_hybrid_prefix_publication_runtime
    )
    assert "representation" in signature.parameters, (
        "adapter configure method lacks explicit representation parameter"
    )

    runtime, engine = _adapter("recompute")
    runtime.configure_qwen35_hybrid_prefix_publication_runtime(
        model_fingerprint="model",
        max_entries=16,
        max_bytes=2 * 1024**3,
        timeout_s=120.0,
        representation="exact_restore",
    )

    assert engine.configure_calls == [{
        "model_fingerprint": "model",
        "max_entries": 16,
        "max_bytes": 2 * 1024**3,
        "timeout_s": 120.0,
        "representation": "exact_restore",
    }], engine.configure_calls


def test_schema_v1_exact_restore_branch_passes_explicit_representation():
    runtime, engine = _adapter("exact_restore")

    assert engine.configure_calls == [{
        "model_fingerprint": "model",
        "max_entries": 16,
        "max_bytes": 2 * 1024**3,
        "timeout_s": 120.0,
        "representation": "exact_restore",
    }], engine.configure_calls


def test_schema_v1_policy_rejects_recurrent_int8_representation():
    configuration = _configuration("exact_restore")
    configuration["hybrid_prefix"]["representation"] = (
        "recurrent_int8_per_row"
    )
    factory_calls = []

    def fail_if_called(configuration, authorized):
        factory_calls.append((configuration, authorized))
        raise AssertionError(
            "engine_factory must not run for invalid schema-v1"
        )

    try:
        adapter.BenchmarkEngineAdapter(
            configuration,
            {"model_dir": "/model"},
            engine_factory=fail_if_called,
            sampling_params_factory=FakeSamplingParams,
            clock_ns=FakeClock(),
        )
    except ValueError as error:
        assert "representation" in str(error), str(error)
    else:
        raise AssertionError(
            "schema-v1 accepted recurrent_int8_per_row representation"
        )
    assert factory_calls == []


def test_batched_fanout_admits_all_continuations_before_first_step():
    runtime, engine = _adapter("exact_restore")
    payload = runtime.run_benchmark_workload(
        workload="w3_batched_fanout",
        workload_spec=contract.workload_payload("w3_batched_fanout"),
        phase="measured",
        repetition=0,
        policy="exact_restore",
    )

    assert len(payload["requests"]) == 8
    assert len(engine.add_batches[-1]) == 8


def test_recompute_batched_fanout_clears_reusable_prefix_before_admission():
    runtime, engine = _adapter("recompute")

    payload = runtime.run_benchmark_workload(
        workload="w3_batched_fanout",
        workload_spec=contract.workload_payload(
            "w3_batched_fanout"
        ),
        phase="measured",
        repetition=0,
        policy="recompute",
    )

    assert engine.reusable_cache_clear_calls == 1
    assert len(payload["requests"]) == 8
    assert len(engine.add_batches[-1]) == 8
    admitted = engine.add_batches[-1]
    assert all(
        engine.sequences[seq_id].hybrid_prefix_restore_attempted is True
        and engine.sequences[seq_id].hybrid_prefix_restore_hit is False
        for seq_id in admitted
    )


def test_w4_applies_token_stale_and_clear_controls_as_misses():
    runtime, engine = _adapter("exact_restore")
    payload = runtime.run_benchmark_workload(
        workload="w4_miss_invalidation",
        workload_spec=contract.workload_payload(
            "w4_miss_invalidation"
        ),
        phase="measured",
        repetition=0,
        policy="exact_restore",
    )

    assert len(payload["requests"]) == 3
    assert all(
        row["restored_hybrid_state"] is False
        and row["reused_kv_tokens"] == 0
        for row in payload["requests"]
    )
    assert engine.invalidate_calls == [[[1, 2, 3]]]
    assert engine.clear_calls == 1


def test_correctness_records_final_logits_and_close_proves_cleanup():
    runtime, engine = _adapter("exact_restore")
    payload = runtime.run_benchmark_workload(
        workload="w0_short_control",
        workload_spec=contract.workload_payload("w0_short_control"),
        phase="correctness",
        repetition=0,
        policy="exact_restore",
    )

    assert payload["requests"][0]["final_logits"] is not None
    runtime.close()
    assert engine.closed is True


def test_default_engine_factory_forwards_registered_capacity(monkeypatch=None):
    calls = []

    class Engine:

        def __init__(self, model_dir, **kwargs):
            calls.append((model_dir, kwargs))

    original = adapter.importlib.import_module
    adapter.importlib.import_module = lambda name: type(
        "Module",
        (),
        {"LLMEngine": Engine},
    )
    try:
        configuration = {
            "engine": {
                "tensor_parallel_size": 4,
                "num_kvcache_blocks": 64,
                "kvcache_block_size": 256,
                "enforce_eager": True,
                "max_model_len": 4352,
                "max_num_batched_tokens": 17408,
                "max_num_seqs": 8,
            },
        }
        adapter._default_engine_factory(
            configuration,
            {"model_dir": "/model"},
        )
    finally:
        adapter.importlib.import_module = original

    assert calls == [(
        "/model",
        configuration["engine"],
    )]


def test_close_rejects_incomplete_rank_cleanup():
    runtime, engine = _adapter("recompute")
    engine.exit = lambda: {
        "process_group_destroyed": False,
        "rank_exit_codes": [0, 0, 1, 0],
        "owned_children_remaining": [2],
        "rank_cleanup_receipts": [],
    }

    try:
        runtime.close()
    except ValueError as error:
        assert "cleanup receipt" in str(error), str(error)
    else:
        raise AssertionError("incomplete rank cleanup was accepted")


def test_malformed_step_observation_fails_closed():
    runtime, engine = _adapter("recompute")
    original = engine.step

    def malformed():
        outputs, num_tokens = original()
        engine.last_step_observation.pop("step_end_ns")
        return outputs, num_tokens

    engine.step = malformed
    try:
        runtime.run_benchmark_workload(
            workload="w0_short_control",
            workload_spec=contract.workload_payload(
                "w0_short_control"
            ),
            phase="measured",
            repetition=0,
            policy="recompute",
        )
    except ValueError as error:
        assert "step observation" in str(error), str(error)
    else:
        raise AssertionError("malformed step observation was accepted")


def test_non_monotonic_step_observation_fails_closed():
    runtime, engine = _adapter("recompute")
    original = engine.step

    def non_monotonic():
        outputs, num_tokens = original()
        engine.last_step_observation["step_end_ns"] = 800
        return outputs, num_tokens

    engine.step = non_monotonic
    try:
        runtime.run_benchmark_workload(
            workload="w0_short_control",
            workload_spec=contract.workload_payload(
                "w0_short_control"
            ),
            phase="measured",
            repetition=0,
            policy="recompute",
        )
    except ValueError as error:
        assert "step timestamp" in str(error), str(error)
    else:
        raise AssertionError(
            "non-monotonic step observation was accepted"
        )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark Engine adapter tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
