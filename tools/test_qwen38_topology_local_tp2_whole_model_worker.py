from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from tools.assemble_qwen38_topology_local_tp2_whole_model import (
    assemble_attempt,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools/qwen38_topology_local_tp2_whole_model_worker.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen38_topology_local_tp2_whole_model_worker_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_freezes_workloads_epochs_and_checkpoints():
    worker = _load()

    assert worker.WORKLOADS == {
        "P0": ("causal", 256, 128, 1),
        "P1": ("causal", 2048, 128, 1),
        "Q0": ("online", 256, 128, 4),
        "Q1": ("online", 256, 128, 8),
        "Q2": ("online", 2048, 128, 4),
    }
    assert worker.EPOCH_ARMS == (
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    )
    assert worker.MEASURED_REPETITIONS == 5
    assert worker.WARMUP_REPETITIONS == 2
    assert worker.STATE_CHECKPOINTS == (
        "pre_migration",
        "post_migration",
        "token_1",
        "token_4",
        "token_8",
        "token_32",
        "token_128",
    )


def test_request_specs_are_deterministic_exact_and_namespace_disjoint():
    worker = _load()

    first = worker.build_request_specs(256, 128, 4, "timing/P0/r0")
    repeated = worker.build_request_specs(
        256,
        128,
        4,
        "timing/P0/r0",
    )
    correctness = worker.build_request_specs(
        256,
        128,
        4,
        "correctness/P0/r0",
    )

    assert first == repeated
    assert len(first) == 4
    assert all(len(row["prompt_token_ids"]) == 256 for row in first)
    assert all(row["output_tokens"] == 128 for row in first)
    assert len({row["request_id"] for row in first}) == 4
    assert {
        tuple(row["prompt_token_ids"]) for row in first
    }.isdisjoint({
        tuple(row["prompt_token_ids"]) for row in correctness
    })


def test_request_metrics_use_first_to_last_tpot_and_all_token_gaps():
    worker = _load()
    timestamps = tuple(index * index + 100 for index in range(128))

    metrics = worker.reconstruct_request_metrics(
        admitted_ns=10,
        token_timestamps_ns=timestamps,
    )

    assert metrics["ttft_ns"] == 90
    assert metrics["tpot_ns"] == (
        timestamps[-1] - timestamps[0]
    ) / 127
    assert metrics["token_gaps_ns"] == [
        current - previous
        for previous, current in zip(
            timestamps,
            timestamps[1:],
        )
    ]
    assert metrics["e2e_ns"] == timestamps[-1] - 10


def _model_manifest():
    return {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "pair_local_accumulation_dtype_bytes": 4,
        "full_attention_collective_dtype_bytes": 2,
        "layer_types": [
            (
                "full_attention"
                if index % 4 == 3
                else "linear_attention"
            )
            for index in range(64)
        ],
    }


def _raw_candidate_evidence(worker, *, concurrency=2):
    request_rows = [
        {
            "request_id": f"request-{index}",
            "output_token_ids": list(range(128)),
            "rank_token_agreement": True,
            "finite_logits": True,
            "stop_position": 128,
            "stop_reason": "length",
        }
        for index in range(concurrency)
    ]
    scheduler_step_rows = [{
        "step_index": 0,
        "is_prefill": True,
        "batch_kind": "prefill",
        "request_ids": [
            row["request_id"] for row in request_rows
        ],
    }]
    token_count_rows = []
    collective_rows = []
    for step_index in range(1, 128):
        scheduler_step_rows.append({
            "step_index": step_index,
            "is_prefill": False,
            "batch_kind": "decode",
            "request_ids": [
                row["request_id"] for row in request_rows
            ],
        })
        for request in request_rows:
            token_count_rows.append({
                "step_index": step_index,
                "request_id": request["request_id"],
                "token_count": 1,
            })
        for rank in range(4):
            for layer_index in range(64):
                layer_role = (
                    "full_attention"
                    if layer_index % 4 == 3
                    else "linear_attention"
                )
                scope = (
                    "global_tp4"
                    if layer_role == "full_attention"
                    else "pair_local"
                )
                dtype_bytes = (
                    2 if layer_role == "full_attention" else 4
                )
                collective_rows.append({
                    "rank": rank,
                    "step_index": step_index,
                    "layer_index": layer_index,
                    "layer_role": layer_role,
                    "scope": scope,
                    "calls": concurrency,
                    "bytes": (
                        concurrency
                        * 5120
                        * dtype_bytes
                    ),
                    "sequence": [
                        f"{step_index}:{layer_index}:{request_id}"
                        for request_id in (
                            row["request_id"]
                            for row in request_rows
                        )
                    ],
                })

    expected_segments = concurrency * 127
    before = []
    after = []
    linear_indices = tuple(
        index for index in range(64) if index % 4 != 3
    )
    for rank in range(4):
        pair_id = 0 if rank < 2 else 1
        before.append({
            "rank": rank,
            "enabled": True,
            "pair_id": pair_id,
            "linear_layer_indices": linear_indices,
            "transition_count": 0,
            "fixed_cohort": None,
            "state": {
                "publication_count": 0,
                "temporary_live_tensors": 0,
                "rollback_count": 0,
            },
            "mixers": tuple({
                "tp2_decode_calls": 0,
                "recurrent_token_one_calls": 0,
                "short_chunk_calls": 0,
                "chunk_64_calls": 0,
                "pair_local_all_reduce_calls": 0,
                "global_tp4_decode_all_reduce_calls": 0,
            } for _ in linear_indices),
            "fallback_calls": 0,
            "post_warmup_request_path_allocations": 0,
            "prefix_restore_calls": 0,
            "prefix_publication_calls": 0,
            "retry_after_mutation_calls": 0,
            "duplicate_commit_calls": 0,
            "pair_replica_comparison_failures": 0,
        })
        after.append({
            **before[-1],
            "transition_count": 1,
            "fixed_cohort": tuple(
                (index, 1, index)
                for index in range(concurrency)
            ),
            "state": {
                "publication_count": concurrency,
                "temporary_live_tensors": 0,
                "rollback_count": 0,
            },
            "mixers": tuple({
                "tp2_decode_calls": expected_segments,
                "recurrent_token_one_calls": expected_segments,
                "short_chunk_calls": 0,
                "chunk_64_calls": 0,
                "pair_local_all_reduce_calls": expected_segments,
                "global_tp4_decode_all_reduce_calls": 0,
            } for _ in linear_indices),
        })
    cleanup = {
        "process_group_destroyed": True,
        "rank_exit_codes": [0, 0, 0, 0],
        "owned_children_remaining": [],
        "cleanup_started_ns": 10,
        "cleanup_finished_ns": 110,
        "cleanup_duration_ns": 100,
        "rank_cleanup_receipts": [{
            "rank": rank,
            "process_group_destroyed": True,
            "qwen38_topology_local_tp2_cleanup": {
                "pair_groups_destroyed": 2,
                "candidate_state_released": True,
                "published_generations_remaining": 0,
                "temporary_live_tensors": 0,
            },
        } for rank in range(4)],
    }
    return {
        "request_rows": request_rows,
        "scheduler_step_rows": scheduler_step_rows,
        "token_count_rows": token_count_rows,
        "collective_rows": collective_rows,
        "model_manifest": _model_manifest(),
        "before_snapshots": tuple(before),
        "after_snapshots": tuple(after),
        "cleanup": cleanup,
    }


def _timing_correctness_replay(requests):
    replay_requests = [{
        "request_id": request["request_id"],
        "runtime_request_id": request_index,
        "output_token_ids": list(request["output_token_ids"]),
        "stop_position": request.get("stop_position", 128),
        "stop_reason": request.get("stop_reason", "length"),
        "decoded_text": request["decoded_text"],
        "decoded_text_sha256": request["decoded_text_sha256"],
    } for request_index, request in enumerate(requests)]
    proofs = [[{
        "rank": rank,
        "sequence_ids": list(range(len(replay_requests))),
        "finite_logits": True,
        "token_ids": [
            request["output_token_ids"][step]
            for request in replay_requests
        ],
        "top_logit_values": [
            float(request["output_token_ids"][step])
            for request in replay_requests
        ],
    } for rank in range(4)] for step in range(128)]
    return {
        "requests": replay_requests,
        "correctness_step_proofs": proofs,
        "timing_authority": False,
    }


def test_candidate_evidence_is_rederived_from_raw_rows():
    worker = _load()
    raw = _raw_candidate_evidence(worker)

    result = worker.validate_candidate_evidence(**raw)

    expected_segments = 2 * 127
    assert result["expected_segments"] == expected_segments
    assert result["tp2_decode_calls"] == expected_segments * 48
    assert (
        result["recurrent_token_one_calls"]
        == expected_segments * 48
    )
    assert result["short_chunk_calls"] == 0
    assert result["ordinary_chunk_calls"] == 0
    assert (
        result["global_tp4_linear_decode_all_reduce_calls"]
        == 0
    )
    assert result["fallback_calls"] == 0
    assert result["migration_publications"] == 2
    assert result["full_attention_layer_count"] == 16
    assert (
        result["full_attention_tp4_collective_calls"]
        == expected_segments * 16
    )
    assert result["pair_local_collective_sequence_match"] is True
    assert result["post_warmup_request_path_allocations"] == 0
    assert result["prefix_restore_calls"] == 0
    assert result["prefix_publication_calls"] == 0
    assert result["retry_after_mutation_calls"] == 0
    assert result["duplicate_commit_calls"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("token_mismatch", "output token"),
        ("changed_cohort", "cohort"),
        ("missing_rank", "rank"),
        ("duplicate_commit", "duplicate commit"),
        ("nonfinite_logits", "finite"),
        ("missing_layer", "layer"),
        ("retained_temporary", "temporary"),
        ("incomplete_cleanup", "cleanup"),
        ("missing_zero_counter", "fallback_calls"),
    ),
)
def test_candidate_evidence_rejects_incomplete_or_invalid_proof(
    mutation,
    message,
):
    worker = _load()
    raw = _raw_candidate_evidence(worker)
    if mutation == "token_mismatch":
        raw["request_rows"][0]["output_token_ids"] = list(range(127))
    elif mutation == "changed_cohort":
        raw["after_snapshots"][0]["fixed_cohort"] = ((9, 1, 9),)
    elif mutation == "missing_rank":
        raw["after_snapshots"] = raw["after_snapshots"][:-1]
    elif mutation == "duplicate_commit":
        raw["after_snapshots"][0]["duplicate_commit_calls"] = 1
    elif mutation == "nonfinite_logits":
        raw["request_rows"][0]["finite_logits"] = False
    elif mutation == "missing_layer":
        raw["model_manifest"]["layer_types"].pop()
    elif mutation == "retained_temporary":
        raw["after_snapshots"][0]["state"][
            "temporary_live_tensors"
        ] = 1
    elif mutation == "incomplete_cleanup":
        raw["cleanup"]["rank_cleanup_receipts"].pop()
    elif mutation == "missing_zero_counter":
        raw["after_snapshots"][0].pop("fallback_calls")

    with pytest.raises((ValueError, RuntimeError), match=message):
        worker.validate_candidate_evidence(**raw)


def test_resource_summary_rebuilds_cleanliness_and_gpu_identity():
    worker = _load()
    plan = {
        "attempt_tag": "attempt-r1",
        "gpu_rank_mapping": [{
            "rank": rank,
            "gpu_index": rank + 2,
            "gpu_uuid": f"GPU-{rank}",
        } for rank in range(4)],
    }
    sample = {
        "stage": "entry",
        "measurement_scope": "boundary",
        "gpu_inventory": [{
            "gpu_index": rank + 2,
            "gpu_uuid": f"GPU-{rank}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        } for rank in range(4)],
    }

    summary = worker._summarize_resource_sample(plan, sample)

    assert summary["strict_clean"] is True
    assert summary["identity_match"] is True
    assert summary["attempt_tag"] == "attempt-r1"
    assert summary["measurement_scope"] == "boundary"
    assert summary["run_label"] is None
    assert summary["sample_index"] is None
    assert summary["gpu_inventory"] == sample["gpu_inventory"]
    assert summary["process_rows"] == []

    empty = {**sample, "gpu_inventory": []}
    assert worker._summarize_resource_sample(
        plan,
        empty,
    )["strict_clean"] is False
    assert worker._summarize_resource_sample(
        plan,
        empty,
    )["identity_match"] is False

    drifted = {
        **sample,
        "gpu_inventory": [
            {**row, "gpu_uuid": "GPU-drift"}
            if row["gpu_index"] == 2
            else row
            for row in sample["gpu_inventory"]
        ],
    }
    assert worker._summarize_resource_sample(
        plan,
        drifted,
    )["identity_match"] is False


def test_cleanup_summary_is_derived_from_rank_receipts():
    worker = _load()
    candidate = _raw_candidate_evidence(worker)["cleanup"]
    baseline = copy.deepcopy(candidate)
    for row in baseline["rank_cleanup_receipts"]:
        row["qwen38_topology_local_tp2_cleanup"] = None
    records = [
        ("correctness/baseline", baseline, False),
        ("correctness/candidate", candidate, True),
    ]

    summary = worker._summarize_cleanup_receipts(
        records,
        task_paths=("/data00/home/sitian/task",),
    )

    assert summary["complete"] is True
    assert summary["retained_generations"] == 0
    assert summary["retained_leases"] == 0
    assert summary["retained_tensors"] == 0
    assert summary["retained_process_groups"] == 0
    assert summary["validated_worker_cleanups"] == 2
    assert summary["validated_rank_cleanup_receipts"] == 8
    assert summary["cleanup_durations_ns"] == [100, 100]
    assert summary["worker_cleanup_receipts"] == [
        {
            "label": label,
            "candidate_enabled": candidate_enabled,
            "receipt": receipt,
        }
        for label, receipt, candidate_enabled in records
    ]

    missing_candidate_receipt = copy.deepcopy(records)
    missing_candidate_receipt[1][1]["rank_cleanup_receipts"][0][
        "qwen38_topology_local_tp2_cleanup"
    ] = None
    with pytest.raises(RuntimeError, match="candidate cleanup"):
        worker._summarize_cleanup_receipts(
            missing_candidate_receipt,
            task_paths=("/data00/home/sitian/task",),
        )


def test_weight_layout_is_rebuilt_from_candidate_runtime_snapshots():
    worker = _load()
    evidence = _raw_candidate_evidence(worker)
    released_bytes = 48 * 1024
    for row in evidence["after_snapshots"]:
        row["released_layer_count"] = 48
        row["released_bytes"] = released_bytes
    epoch_results = ({
        "epoch": 1,
        "arm": "candidate",
        "rows": [{
            "after_snapshots": evidence["after_snapshots"],
        }],
    },)

    summary = worker._candidate_weight_layout(
        epoch_results,
        steady_increment_bytes_per_rank=123,
    )

    assert (
        summary["baseline_tp4_decode_accumulation_retained"]
        is False
    )
    assert summary["released_layer_count_per_rank"] == 48
    assert summary["released_bytes_per_rank"] == released_bytes
    assert summary["steady_increment_bytes_per_rank"] == 123

    incomplete = copy.deepcopy(epoch_results)
    incomplete[0]["rows"][0]["after_snapshots"][0].pop(
        "released_layer_count"
    )
    with pytest.raises(RuntimeError, match="weight release"):
        worker._candidate_weight_layout(
            incomplete,
            steady_increment_bytes_per_rank=123,
        )


class _FakeEngine:

    def __init__(self, request_specs, *, candidate, seq_offset=0):
        self.request_specs = request_specs
        self.candidate = candidate
        self.seq_offset = seq_offset
        self.model_runner = SimpleNamespace(rank=0, world_size=4)
        self.tokenizer = SimpleNamespace(
            decode=lambda tokens, **_kwargs: ",".join(
                str(token) for token in tokens
            )
        )
        self.last_step_observation = None
        self._step = 0
        self._finished = False
        self._admitted = []
        self.exit_calls = 0
        self.flush_calls = 0
        self.proof_recording = False
        self._last_tokens = ()
        self.correctness_checkpoint_steps = []

    def add_request(self, prompt, sampling):
        self._admitted.append((list(prompt), sampling))
        return self.seq_offset + len(self._admitted) - 1

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        self.flush_calls += 1
        return ()

    def is_finished(self):
        return self._finished

    def step(self):
        tokens = {
            self.seq_offset + index: [1000 + self._step + index]
            for index in range(len(self.request_specs))
        }
        self._last_tokens = tuple(
            tokens[self.seq_offset + index][0]
            for index in range(len(self.request_specs))
        )
        self.last_step_observation = {
            "step_start_ns": 999_950 + self._step * 100,
            "step_end_ns": 1_000_000 + self._step * 100,
            "is_prefill": self._step == 0,
            "batch_kind": (
                "prefill" if self._step == 0 else "decode"
            ),
            "scheduled": [
                {"seq_id": self.seq_offset + index}
                for index in range(len(self.request_specs))
            ],
            "new_completion_tokens_by_seq": tokens,
            "memory": {},
            "command_timeline_step": {
                "phases": {
                    "ordinary_or_first_target_dispatch": {
                        "duration_ns": 40,
                    },
                },
            },
        }
        self._step += 1
        if self._step == 128:
            self._finished = True
            return [
                (
                    self.seq_offset + index,
                    [
                        1000 + step + index
                        for step in range(128)
                    ],
                )
                for index in range(len(self.request_specs))
            ], 0
        return [], 0

    def enable_qwen38_correctness_proof(self, enabled, *, timeout_s):
        self.proof_recording = bool(enabled)
        return {"enabled": self.proof_recording}

    def qwen38_correctness_step_proofs(self, *, timeout_s):
        assert self.proof_recording
        return tuple({
            "rank": rank,
            "finite_logits": True,
            "sequence_ids": [
                self.seq_offset + index
                for index in range(len(self.request_specs))
            ],
            "token_ids": list(self._last_tokens),
            "top_logit_values": [
                float(token) for token in self._last_tokens
            ],
        } for rank in range(4))

    def qwen38_correctness_state_checkpoints(self, *, timeout_s):
        self.correctness_checkpoint_steps.append(self._step)
        linear_layer_indices = tuple(
            index for index in range(64) if index % 4 != 3
        )
        candidate_active = self.candidate and self._step >= 2

        def component(layer, source_rank):
            return {
                "layer_index": layer,
                "logical_rank": source_rank // 2,
                "source_rank": source_rank,
                "convolution_query_sha256": (
                    f"{layer * 16 + source_rank * 4:064x}"
                ),
                "convolution_key_sha256": (
                    f"{layer * 16 + source_rank * 4 + 1:064x}"
                ),
                "convolution_value_sha256": (
                    f"{layer * 16 + source_rank * 4 + 2:064x}"
                ),
                "recurrent_sha256": (
                    f"{layer * 16 + source_rank * 4 + 3:064x}"
                ),
            }

        return tuple({
            "rank": rank,
            "pair_id": 0 if rank < 2 else 1,
            "logical_rank": rank % 2,
            "state_layout": (
                "tp2_logical_half"
                if candidate_active
                else "tp4_source_quarter"
            ),
            "cohort": [{
                "slot_id": index,
                "generation": 1,
                "request_id": self.seq_offset + index,
            } for index in range(len(self.request_specs))],
            "output_digests": [{
                "layer_index": layer,
                "sha256": f"{layer:064x}",
            } for layer in linear_layer_indices] if candidate_active else [],
            "state_digests": [{
                "layer_index": layer,
                "convolution_sha256": (
                    f"{layer * 2 + rank % 2:064x}"
                ),
                "recurrent_sha256": (
                    f"{layer * 2 + rank % 2 + 1:064x}"
                ),
            } for layer in linear_layer_indices] if candidate_active else [],
            "canonical_state_components": [
                component(layer, source_rank)
                for layer in linear_layer_indices
                for source_rank in (
                    (
                        2 * (rank % 2),
                        2 * (rank % 2) + 1,
                    )
                    if candidate_active
                    else (rank,)
                )
            ],
        } for rank in range(4))

    def qwen38_topology_local_tp2_snapshots(self, timeout_s):
        raw = _raw_candidate_evidence(
            SimpleNamespace(),
            concurrency=len(self.request_specs),
        )
        return (
            raw["after_snapshots"]
            if self._step
            else raw["before_snapshots"]
        )

    def memory_snapshots(self, *, timeout_s):
        return tuple({
            "rank": rank,
            "cuda_peak_allocated_bytes": 100 + rank,
            "cuda_peak_reserved_bytes": 200 + rank,
            "physical_memory_bytes": 80 * 1024**3,
        } for rank in range(4))

    def exit(self):
        self.exit_calls += 1
        return _raw_candidate_evidence(
            SimpleNamespace(),
            concurrency=len(self.request_specs),
        )["cleanup"]


def test_run_engine_case_uses_frozen_engine_shape_and_exact_timing():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "timing/P0/r0",
    )
    factory_calls = []
    fake = _FakeEngine(specs, candidate=True)

    def factory(model_root, **kwargs):
        factory_calls.append((model_root, kwargs))
        return fake

    ticks = iter(range(10, 1000))
    result = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=False,
        epoch=1,
        repetition=0,
        engine_factory=factory,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=lambda: next(ticks),
        timeout_s=30.0,
    )

    assert factory_calls == [(
        Path("/model"),
        {
            "tensor_parallel_size": 4,
            "enforce_eager": True,
            "max_num_seqs": 8,
            "max_model_len": 384,
            "max_num_batched_tokens": 512,
            "qwen38_topology_local_tp2_islands": True,
        },
    )]
    assert len(result["requests"]) == 2
    assert all(
        len(row["output_token_ids"]) == 128
        for row in result["requests"]
    )
    assert all(
        len(row["token_gaps_ns"]) == 127
        for row in result["requests"]
    )
    assert len(result["request_set_digest"]) == 64
    assert result["cohort_makespan_ns"] >= 0
    assert all(row["queueing_ns"] >= 0 for row in result["requests"])
    assert all(
        row["step_duration_ns"] == 50
        and row["host_submission_ns"] == 40
        for row in result["scheduler_step_rows"]
    )
    assert all(
        row["decoded_text_sha256"]
        == hashlib.sha256(
            row["decoded_text"].encode("utf-8")
        ).hexdigest()
        for row in result["requests"]
    )
    assert all(
        row["rank_token_agreement"] is None
        and row["finite_logits"] is None
        and row["stop_position"] == 128
        and row["stop_reason"] == "length"
        for row in result["requests"]
    )
    assert result["cleanup"]["process_group_destroyed"] is True
    assert result["cleanup"]["cleanup_duration_ns"] >= 0
    assert fake.exit_calls == 1


def test_shared_engine_case_flushes_and_uses_returned_sequence_ids():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "timing/P0/r1",
    )
    fake = _FakeEngine(specs, candidate=False, seq_offset=40)

    result = worker.run_engine_case(
        model_root=Path("/model"),
        arm="baseline",
        workload_id="P0",
        request_specs=specs,
        warmup=True,
        epoch=0,
        repetition=1,
        engine=fake,
        close_engine=False,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
        timeout_s=30.0,
    )

    assert fake.flush_calls == 1
    assert fake.exit_calls == 0
    assert result["cleanup"] is None
    assert [row["request_id"] for row in result["requests"]] == [
        row["request_id"] for row in specs
    ]


def test_correctness_case_uses_rank_local_logit_and_token_proofs():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "correctness/P0/r0",
    )
    fake = _FakeEngine(specs, candidate=True)

    result = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=True,
        epoch=-1,
        repetition=0,
        engine=fake,
        close_engine=False,
        correctness_authority=True,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )

    assert all(row["rank_token_agreement"] for row in result["requests"])
    assert all(row["finite_logits"] for row in result["requests"])
    assert result["correctness_step_proofs"]
    assert set(result["correctness_state_checkpoints"]) == set(
        worker.STATE_CHECKPOINTS
    )
    assert fake.correctness_checkpoint_steps == [
        1,
        2,
        4,
        8,
        32,
        128,
    ]


def test_baseline_correctness_case_captures_matching_state_checkpoints():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "correctness/P0/baseline-r0",
    )
    fake = _FakeEngine(specs, candidate=False)

    result = worker.run_engine_case(
        model_root=Path("/model"),
        arm="baseline",
        workload_id="P0",
        request_specs=specs,
        warmup=True,
        epoch=-1,
        repetition=0,
        engine=fake,
        close_engine=False,
        correctness_authority=True,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )

    assert set(result["correctness_state_checkpoints"]) == set(
        worker.STATE_CHECKPOINTS
    )
    assert fake.correctness_checkpoint_steps == [
        1,
        2,
        4,
        8,
        32,
        128,
    ]


def test_service_replica_case_uses_a_real_tp2_engine_shape():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "service/Q0/r0",
    )
    fake = _FakeEngine(specs, candidate=False)
    fake.model_runner.world_size = 2
    factory_calls = []

    result = worker.run_service_replica_case(
        model_root=Path("/model"),
        workload_id="Q0",
        request_specs=specs,
        engine_factory=lambda model_root, **kwargs: (
            factory_calls.append((model_root, kwargs)) or fake
        ),
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )

    assert factory_calls[0][1]["tensor_parallel_size"] == 2
    assert factory_calls[0][1]["qwen38_topology_local_tp2_islands"] is False
    assert result["replica_tensor_parallel_size"] == 2
    assert len(result["requests"]) == 2


def test_performance_epoch_uses_two_warmups_and_five_measurements():
    worker = _load()
    calls = []

    def case_runner(**kwargs):
        calls.append(kwargs)
        return {
            "workload_id": kwargs["workload_id"],
            "warmup": kwargs["warmup"],
            "repetition": kwargs["repetition"],
        }

    result = worker.run_performance_epoch(
        model_root=Path("/model"),
        output_root=Path("/output"),
        epoch=1,
        arm="candidate",
        workload_order=("P0", "Q0"),
        case_runner=case_runner,
        row_sink=lambda _row: None,
    )

    assert len(calls) == 2 * (2 + 5 + 5)
    assert [row["workload_id"] for row in calls[:7]] == ["P0"] * 7
    assert sum(not row["warmup"] for row in calls) == 10
    assert sum(
        row.get("correctness_authority") is True for row in calls
    ) == 10
    assert all(
        "timing_correctness_replay" in row
        for row in result["rows"]
        if row["warmup"] is False
    )
    assert result["epoch"] == 1
    assert result["arm"] == "candidate"


def test_performance_epoch_reuses_one_engine_when_factory_is_supplied():
    worker = _load()
    engine = SimpleNamespace(exit=lambda: {"clean": True})
    factory_calls = []
    case_engines = []

    def factory(model_root, **kwargs):
        factory_calls.append((model_root, kwargs))
        return engine

    def case_runner(**kwargs):
        case_engines.append(kwargs["engine"])
        assert kwargs["close_engine"] is False
        return {
            "workload_id": kwargs["workload_id"],
            "warmup": kwargs["warmup"],
            "repetition": kwargs["repetition"],
        }

    result = worker.run_performance_epoch(
        model_root=Path("/model"),
        output_root=Path("/output"),
        epoch=2,
        arm="candidate",
        workload_order=("P0",),
        case_runner=case_runner,
        engine_factory=factory,
        row_sink=lambda _row: None,
    )

    assert len(factory_calls) == 1
    assert factory_calls[0][1] == {
        "tensor_parallel_size": 4,
        "enforce_eager": True,
        "max_num_seqs": 8,
        "max_model_len": 2176,
        "max_num_batched_tokens": 8192,
        "qwen38_topology_local_tp2_islands": True,
    }
    assert case_engines == [engine] * 12
    assert result["cleanup"]["clean"] is True
    assert (
        result["cleanup"]["cleanup_duration_ns"]
        == result["cleanup"]["cleanup_finished_ns"]
        - result["cleanup"]["cleanup_started_ns"]
    )


def test_worker_cli_dispatches_one_frozen_performance_epoch(tmp_path):
    worker = _load()
    calls = []

    def performance_runner(**kwargs):
        calls.append(kwargs)
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "phase": "performance_epoch",
            "epoch": kwargs["epoch"],
            "arm": kwargs["arm"],
        }

    exit_code = worker.main(
        [
            "performance-epoch",
            "--model-root",
            "/model",
            "--output-root",
            str(tmp_path),
            "--epoch",
            "2",
            "--arm",
            "candidate",
            "--workload-order",
            "P0,P1,Q0,Q1,Q2",
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
        ],
        performance_runner=performance_runner,
    )

    assert exit_code == 0
    assert calls == [{
        "model_root": Path("/model"),
        "output_root": tmp_path,
        "epoch": 2,
        "arm": "candidate",
        "workload_order": ("P0", "P1", "Q0", "Q1", "Q2"),
        "engine_factory": worker._default_engine_factory,
    }]
    payload = json.loads(
        (tmp_path / "worker-receipt.json").read_text(encoding="utf-8")
    )
    assert payload["phase"] == "performance_epoch"
    assert payload["epoch"] == 2
    artifacts = json.loads(
        (tmp_path / "epoch-2-artifact-rows.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(artifacts) == {
        "request_rows.jsonl",
        "scheduler_step_rows.jsonl",
        "candidate_hit_rows.jsonl",
        "collective_rows.jsonl",
        "migration_rows.jsonl",
        "memory_rows.jsonl",
    }


def test_worker_cli_dispatches_correctness_and_service_control(tmp_path):
    worker = _load()
    calls = []

    def correctness_runner(**kwargs):
        calls.append(("correctness", kwargs))
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "phase": "correctness",
        }

    correctness_root = tmp_path / "correctness"
    assert worker.main(
        [
            "correctness",
            "--model-root",
            "/model",
            "--output-root",
            str(correctness_root),
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
        ],
        correctness_runner=correctness_runner,
    ) == 0

    def service_runner(**kwargs):
        calls.append(("service", kwargs))
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "arm": "TP2_X2_SERVICE_CONTROL",
        }

    service_root = tmp_path / "service"
    assert worker.main(
        [
            "service-control",
            "--model-root",
            "/model",
            "--output-root",
            str(service_root),
            "--pair-devices",
            "0,1;2,3",
            "--workloads",
            "Q0,Q1,Q2",
        ],
        service_runner=service_runner,
        replica_runner=lambda **_kwargs: {},
    ) == 0

    assert calls[0] == (
        "correctness",
        {
            "model_root": Path("/model"),
            "output_root": correctness_root,
            "engine_factory": worker._default_engine_factory,
        },
    )
    assert calls[1][0] == "service"
    assert calls[1][1]["model_root"] == Path("/model")
    assert (
        correctness_root / "correctness-artifact-rows.json"
    ).is_file()
    assert calls[1][1]["output_root"] == service_root
    assert calls[1][1]["pair_devices"] == ((0, 1), (2, 3))
    assert calls[1][1]["workloads"] == ("Q0", "Q1", "Q2")
    assert callable(calls[1][1]["replica_runner"])


def test_correctness_campaign_loads_one_engine_per_arm():
    worker = _load()
    engines = {
        "baseline": SimpleNamespace(exit=lambda: {"arm": "baseline"}),
        "candidate": SimpleNamespace(exit=lambda: {"arm": "candidate"}),
    }
    factory_calls = []
    case_calls = []

    def factory(_model_root, **kwargs):
        arm = (
            "candidate"
            if kwargs["qwen38_topology_local_tp2_islands"]
            else "baseline"
        )
        factory_calls.append(arm)
        return engines[arm]

    def case_runner(**kwargs):
        case_calls.append(kwargs)
        return {
            "requests": [{
                "output_token_ids": list(range(128)),
            } for _ in kwargs["request_specs"]],
            "timing_authority": False,
        }

    result = worker.run_correctness_campaign(
        model_root=Path("/model"),
        output_root=Path("/output"),
        engine_factory=factory,
        case_runner=case_runner,
        row_sink=lambda _row: None,
    )

    assert factory_calls == ["baseline", "candidate"]
    assert len(case_calls) == 50
    assert all(call["close_engine"] is False for call in case_calls)
    assert result["cleanup"]["baseline"]["arm"] == "baseline"
    assert result["cleanup"]["candidate"]["arm"] == "candidate"
    for receipt in result["cleanup"].values():
        assert (
            receipt["cleanup_duration_ns"]
            == receipt["cleanup_finished_ns"]
            - receipt["cleanup_started_ns"]
        )


def test_service_control_splits_requests_and_is_non_authoritative():
    worker = _load()
    calls = []

    def replica_runner(*, pair_devices, request_specs, **kwargs):
        calls.append((pair_devices, request_specs))
        requests = []
        for request_index, row in enumerate(request_specs):
            admitted_ns = 100 + request_index
            token_timestamps_ns = [
                admitted_ns + 10 + step * 10 for step in range(128)
            ]
            requests.append({
                "request_id": row["request_id"],
                "runtime_request_id": request_index,
                "output_token_ids": list(range(128)),
                "admitted_ns": admitted_ns,
                "token_timestamps_ns": token_timestamps_ns,
                "completion_ns": token_timestamps_ns[-1],
                "token_gaps_ns": [10] * 127,
                "ttft_ns": 10,
                "tpot_ns": 10,
                "e2e_ns": 1280,
                "stop_position": 128,
                "stop_reason": "length",
                "decoded_text_sha256": "a" * 64,
            })
        return {
            "requests": requests,
            "memory": [{
                "rank": rank,
                "cuda_peak_allocated_bytes": 100,
                "cuda_peak_reserved_bytes": 120,
                "physical_memory_bytes": 1_000,
            } for rank in (0, 1)],
            "replica_tensor_parallel_size": 2,
            "request_set_digest": worker._request_set_digest(request_specs),
            "cleanup": {},
        }

    result = worker.run_service_control(
        model_root=Path("/model"),
        output_root=Path("/output"),
        pair_devices=((0, 1), (2, 3)),
        workloads=("Q0",),
        replica_runner=replica_runner,
        row_sink=lambda _row: None,
    )

    assert len(calls) == 2
    assert {
        int(row["request_id"].rsplit("-", 1)[-1]) % 2
        for row in calls[0][1]
    } == {0}
    assert {
        int(row["request_id"].rsplit("-", 1)[-1]) % 2
        for row in calls[1][1]
    } == {1}
    assert all(
        row["request_id"].startswith("timing-Q0-r0-")
        for _, request_specs in calls
        for row in request_specs
    )
    assert result["arm"] == "TP2_X2_SERVICE_CONTROL"
    assert result["classification_authority"] is False
    assert result["rows"][0]["output_tokens_per_second"] > 0
    assert result["rows"][0]["ttft_ns"] == {
        "p50": 10.0,
        "p95": 10.0,
        "p99": 10.0,
    }
    assert result["rows"][0]["tpot_ns"] == {
        "p50": 10.0,
        "p95": 10.0,
        "p99": 10.0,
    }
    assert result["rows"][0]["replica_balance"]["request_counts"] == [2, 2]
    assert [
        row["gpu_index"]
        for replica in result["rows"][0]["replicas"]
        for row in replica["peak_memory_by_gpu"]
    ] == [0, 1, 2, 3]


def test_service_control_can_launch_both_replicas_as_one_parallel_batch():
    worker = _load()
    calls = []

    def parallel_runner(
        *,
        pair_devices,
        request_specs_by_replica,
        **kwargs,
    ):
        calls.append((pair_devices, request_specs_by_replica, kwargs))
        replicas = []
        for replica_specs in request_specs_by_replica:
            requests = []
            for request_index, row in enumerate(replica_specs):
                admitted_ns = 10 + request_index
                timestamps = [
                    admitted_ns + 10 + step * 10
                    for step in range(128)
                ]
                requests.append({
                    "request_id": row["request_id"],
                    "runtime_request_id": request_index,
                    "output_token_ids": list(range(128)),
                    "admitted_ns": admitted_ns,
                    "token_timestamps_ns": timestamps,
                    "completion_ns": timestamps[-1],
                    "token_gaps_ns": [10] * 127,
                    "ttft_ns": 10,
                    "tpot_ns": 10,
                    "e2e_ns": 1280,
                })
            replicas.append({
                "requests": requests,
                "memory": [{
                    "rank": rank,
                    "cuda_peak_allocated_bytes": 100,
                    "cuda_peak_reserved_bytes": 120,
                    "physical_memory_bytes": 1_000,
                } for rank in (0, 1)],
                "replica_tensor_parallel_size": 2,
                "request_set_digest":
                    worker._request_set_digest(replica_specs),
            })
        return tuple(replicas)

    result = worker.run_service_control(
        model_root=Path("/model"),
        output_root=Path("/output"),
        pair_devices=((0, 1), (2, 3)),
        workloads=("Q0",),
        parallel_runner=parallel_runner,
        row_sink=lambda _row: None,
    )

    assert len(calls) == 1
    assert calls[0][0] == ((0, 1), (2, 3))
    assert [len(rows) for rows in calls[0][1]] == [2, 2]
    assert result["rows"][0]["request_qps"] > 0


def test_parallel_service_runner_starts_both_children_before_waiting(
    tmp_path,
):
    worker = _load()
    events = []
    processes = []
    specs = (
        worker.build_request_specs(256, 128, 2, "service/Q0/a"),
        worker.build_request_specs(256, 128, 2, "service/Q0/b"),
    )

    class Process:
        returncode = 0

        def __init__(self, argv, env):
            self.argv = argv
            self.env = env
            processes.append(self)
            events.append(("start", env["CUDA_VISIBLE_DEVICES"]))
            output_path = Path(
                argv[argv.index("--output-path") + 1]
            )
            input_path = Path(
                argv[argv.index("--request-specs-path") + 1]
            )
            requests = json.loads(input_path.read_text(encoding="utf-8"))
            output_path.write_text(json.dumps({
                "requests": [{
                    "request_id": row["request_id"],
                    "output_token_ids": list(range(128)),
                    "admitted_ns": 1,
                    "completion_ns": 2,
                } for row in requests],
            }))

        def communicate(self, timeout):
            events.append(("wait", self.env["CUDA_VISIBLE_DEVICES"], timeout))
            return "", ""

    result = worker._run_service_replicas_in_subprocesses(
        model_root=Path("/model"),
        output_root=tmp_path,
        pair_devices=((3, 4), (6, 7)),
        workload_id="Q0",
        request_specs_by_replica=specs,
        shared_start_ns=0,
        popen_factory=lambda argv, **kwargs: Process(argv, kwargs["env"]),
    )

    assert [event[0] for event in events] == [
        "start",
        "start",
        "wait",
        "wait",
    ]
    assert [process.env["CUDA_VISIBLE_DEVICES"] for process in processes] == [
        "3,4",
        "6,7",
    ]
    assert len(result) == 2


def test_parallel_service_runner_cleans_all_owned_groups_on_timeout(
    tmp_path,
):
    worker = _load()
    processes = []
    signals = []
    specs = (
        worker.build_request_specs(256, 128, 2, "service/Q0/a"),
        worker.build_request_specs(256, 128, 2, "service/Q0/b"),
    )

    class Process:
        returncode = None

        def __init__(self, *_args, **_kwargs):
            self.pid = 100 + len(processes)
            self.alive = True
            processes.append(self)

        def communicate(self, timeout):
            raise subprocess.TimeoutExpired("service-replica", timeout)

        def poll(self):
            return None if self.alive else -15

        def wait(self, timeout):
            if self.alive:
                raise subprocess.TimeoutExpired("service-replica", timeout)
            self.returncode = -15
            return self.returncode

    def killpg(pgid, signum):
        signals.append((pgid, signum))
        next(process for process in processes if process.pid == pgid).alive = (
            False
        )

    with pytest.raises(subprocess.TimeoutExpired):
        worker._run_service_replicas_in_subprocesses(
            model_root=Path("/model"),
            output_root=tmp_path,
            pair_devices=((3, 4), (6, 7)),
            workload_id="Q0",
            request_specs_by_replica=specs,
            shared_start_ns=0,
            popen_factory=lambda *args, **kwargs: Process(*args, **kwargs),
            killpg=killpg,
        )

    assert signals == [
        (100, signal.SIGTERM),
        (101, signal.SIGTERM),
    ]
    assert all(process.alive is False for process in processes)


def test_performance_artifacts_are_rebuilt_from_measured_raw_case():
    worker = _load()
    specs = worker.build_request_specs(256, 128, 2, "timing/P0/r0")
    fake = _FakeEngine(specs, candidate=True)
    case = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=False,
        epoch=1,
        repetition=0,
        engine_factory=lambda *_args, **_kwargs: fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )
    for snapshot in case["after_snapshots"]:
        snapshot["last_transition_latency_ns"] = 1234
    case["timing_correctness_replay"] = _timing_correctness_replay(
        case["requests"]
    )
    artifacts = worker.build_performance_artifact_rows(
        {
            "epoch": 1,
            "arm": "candidate",
            "rows": [case],
            "cleanup": case["cleanup"],
        },
        source_revision="a" * 40,
        model_revision="b" * 40,
    )

    assert len(artifacts["request_rows.jsonl"]) == 1
    request = artifacts["request_rows.jsonl"][0]
    assert request["request_set_digest"] == case["request_set_digest"]
    assert request["cohort_makespan_ns"] == case["cohort_makespan_ns"]
    assert request["rank_token_agreement"] is True
    assert request["finite_logits"] is True
    assert request["timing_correctness_replay"]["requests"]
    assert request["timing_correctness_replay"]["step_proofs"]
    scheduler = artifacts["scheduler_step_rows.jsonl"][0]
    assert scheduler["decode_steps"] == 127
    assert scheduler["token_one_segments"] == 2 * 127
    hit = artifacts["candidate_hit_rows.jsonl"][0]
    assert hit["tp2_decode_calls"] == 2 * 127 * 48
    assert hit["short_chunk_calls"] == 0
    collective = artifacts["collective_rows.jsonl"][0]
    assert collective["pair_local_calls"] == 2 * 127 * 48
    assert collective["pair_local_bytes"] == 2 * 127 * 48 * 5120 * 4
    assert artifacts["migration_rows.jsonl"][0]["latency_ns"] == 1234
    assert len(artifacts["memory_rows.jsonl"]) == 4


def test_performance_artifacts_preserve_independent_allocator_peaks():
    worker = _load()
    specs = worker.build_request_specs(256, 128, 2, "timing/P0/r0")
    fake = _FakeEngine(specs, candidate=True)
    first = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=False,
        epoch=1,
        repetition=0,
        engine_factory=lambda *_args, **_kwargs: fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )
    for snapshot in first["after_snapshots"]:
        snapshot["last_transition_latency_ns"] = 1234
    first["timing_correctness_replay"] = _timing_correctness_replay(
        first["requests"]
    )
    second = copy.deepcopy(first)
    second["repetition"] = 1
    second["request_set_digest"] = "c" * 64
    for memory in second["memory"]:
        memory["cuda_peak_reserved_bytes"] += 1_000

    artifacts = worker.build_performance_artifact_rows(
        {
            "epoch": 1,
            "arm": "candidate",
            "rows": [first, second],
            "cleanup": first["cleanup"],
        },
        source_revision="a" * 40,
        model_revision="b" * 40,
    )

    assert {
        row["rank"]: row["peak_reserved_bytes"]
        for row in artifacts["memory_rows.jsonl"]
    } == {
        rank: 1_200 + rank
        for rank in range(4)
    }


def test_performance_artifacts_reject_missing_timing_correctness_replay():
    worker = _load()
    specs = worker.build_request_specs(256, 128, 2, "timing/P0/r0")
    fake = _FakeEngine(specs, candidate=True)
    case = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=False,
        epoch=1,
        repetition=0,
        engine_factory=lambda *_args, **_kwargs: fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )
    for snapshot in case["after_snapshots"]:
        snapshot["last_transition_latency_ns"] = 1234

    with pytest.raises(RuntimeError, match="timing correctness"):
        worker.build_performance_artifact_rows(
            {
                "epoch": 1,
                "arm": "candidate",
                "rows": [case],
                "cleanup": case["cleanup"],
            },
            source_revision="a" * 40,
            model_revision="b" * 40,
        )


@pytest.mark.parametrize(
    ("scope", "field"),
    (
        ("runtime", "fallback_calls"),
        ("mixer", "short_chunk_calls"),
    ),
)
def test_performance_artifacts_reject_missing_runtime_counter(
    scope,
    field,
):
    worker = _load()
    specs = worker.build_request_specs(256, 128, 2, "timing/P0/r0")
    fake = _FakeEngine(specs, candidate=True)
    case = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=False,
        epoch=1,
        repetition=0,
        engine_factory=lambda *_args, **_kwargs: fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )
    for snapshot in case["after_snapshots"]:
        snapshot["last_transition_latency_ns"] = 1234
    case["timing_correctness_replay"] = _timing_correctness_replay(
        case["requests"]
    )
    if scope == "runtime":
        case["after_snapshots"][0].pop(field)
    else:
        case["after_snapshots"][0]["mixers"][0].pop(field)

    with pytest.raises(RuntimeError, match=field):
        worker.build_performance_artifact_rows(
            {
                "epoch": 1,
                "arm": "candidate",
                "rows": [case],
                "cleanup": case["cleanup"],
            },
            source_revision="a" * 40,
            model_revision="b" * 40,
        )


def test_correctness_artifacts_require_real_rank_and_state_proofs():
    worker = _load()
    fake = _FakeEngine((
        {"request_id": "r0", "prompt_token_ids": [1], "output_tokens": 128},
    ), candidate=True)
    baseline_fake = _FakeEngine((
        {"request_id": "r0", "prompt_token_ids": [1], "output_tokens": 128},
    ), candidate=False)
    baseline_fake._step = 1
    baseline_checkpoints = {
        name: baseline_fake.qwen38_correctness_state_checkpoints(
            timeout_s=1
        )
        for name in worker.STATE_CHECKPOINTS
    }
    checkpoints = {}
    for name in worker.STATE_CHECKPOINTS:
        fake._step = (
            1
            if name in {"pre_migration", "token_1"}
            else 2
        )
        checkpoints[name] = (
            fake.qwen38_correctness_state_checkpoints(timeout_s=1)
        )
    commit_counts = {
        "pre_migration": 0,
        "token_1": 0,
        "post_migration": 1,
        "token_4": 3,
        "token_8": 7,
        "token_32": 31,
        "token_128": 127,
    }
    checkpoints = {
        name: tuple({
            **row,
            "runtime_snapshot": {
                "state": {
                    "commit_count": commit_counts[name],
                    "rollback_count": 0,
                    "temporary_live_tensors": 0,
                },
            },
        } for row in rows)
        for name, rows in checkpoints.items()
    }
    request = {
        "request_id": "r0",
        "runtime_request_id": 0,
        "output_token_ids": list(range(128)),
        "rank_token_agreement": True,
        "finite_logits": True,
        "stop_position": 128,
        "stop_reason": "length",
    }
    proofs = [[{
        "rank": rank,
        "sequence_ids": [0],
        "finite_logits": True,
        "token_ids": [step],
        "top_logit_values": [float(step)],
    } for rank in range(4)] for step in range(128)]
    result = {
        "rows": [{
            "workload_id": "P0",
            "repetition": 0,
            "baseline": {
                "requests": [request],
                "correctness_step_proofs": proofs,
                "correctness_state_checkpoints": baseline_checkpoints,
            },
            "candidate": {
                "requests": [dict(request)],
                "correctness_step_proofs": proofs,
                "correctness_state_checkpoints": checkpoints,
            },
        }],
    }

    rows = worker.build_correctness_artifact_rows(
        result,
        source_revision="a" * 40,
        model_revision="b" * 40,
    )

    assert [{
        key: row[key]
        for key in (
            "source_revision",
            "model_revision",
            "workload_id",
            "repetition",
            "output_tokens_match",
            "rank_token_agreement",
            "finite_logits",
            "top_logit_values_match",
            "state_checkpoints_complete",
            "single_commit_per_step",
            "pair_replica_digest_match",
            "baseline_candidate_state_match",
        )
    } for row in rows] == [{
        "source_revision": "a" * 40,
        "model_revision": "b" * 40,
        "workload_id": "P0",
        "repetition": 0,
        "output_tokens_match": True,
        "rank_token_agreement": True,
        "finite_logits": True,
        "top_logit_values_match": True,
        "state_checkpoints_complete": True,
        "single_commit_per_step": True,
        "pair_replica_digest_match": True,
        "baseline_candidate_state_match": True,
    }]
    assert rows[0]["baseline_requests"] == [{
        "request_id": "r0",
        "runtime_request_id": 0,
        "output_token_ids": list(range(128)),
    }]
    assert rows[0]["candidate_requests"] == rows[0]["baseline_requests"]
    assert rows[0]["baseline_step_proofs"] == proofs
    assert rows[0]["candidate_step_proofs"] == proofs
    assert (
        rows[0]["baseline_state_checkpoints"]
        == baseline_checkpoints
    )
    assert rows[0]["candidate_state_checkpoints"] == checkpoints

    cohort_drift = copy.deepcopy(result)
    cohort_drift["rows"][0]["candidate"][
        "correctness_state_checkpoints"
    ]["token_32"][0]["cohort"][0]["request_id"] += 1
    invalid = worker.build_correctness_artifact_rows(
        cohort_drift,
        source_revision="a" * 40,
        model_revision="b" * 40,
    )
    assert invalid[0]["state_checkpoints_complete"] is False
    assert invalid[0]["baseline_candidate_state_match"] is False

    result["rows"][0]["candidate"]["correctness_step_proofs"] = []
    invalid = worker.build_correctness_artifact_rows(
        result,
        source_revision="a" * 40,
        model_revision="b" * 40,
    )
    assert invalid[0]["rank_token_agreement"] is False
    assert invalid[0]["finite_logits"] is False
    assert invalid[0]["top_logit_values_match"] is False


def test_raw_finalizer_payloads_are_accepted_by_real_assembler(tmp_path):
    worker = _load()
    source_revision = "a" * 40
    model_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    attempt_root = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/attempts/finalizer-e2e"
    )
    plan = {
        "attempt_tag": "finalizer-e2e",
        "source_revision": source_revision,
        "source_tree_sha256": "b" * 64,
        "model_repository": "Qwen/Qwen3.8-27B",
        "model_revision": model_revision,
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "controller_root": f"{attempt_root}/controller",
        "bundle_root": f"{attempt_root}/final_bundle",
        "environment": {
            "TMPDIR": f"{attempt_root}/runtime/tmp",
        },
            "topology": {
                "rows": [
                    {
                        "left_rank": left,
                        "right_rank": right,
                        "link": (
                            "PIX"
                            if tuple(sorted((left, right)))
                            in {(0, 1), (2, 3)}
                            else "SYS"
                        ),
                    }
                    for left in range(4)
                    for right in range(4)
                    if left != right
                ],
            },
        "gpu_rank_mapping": [
            {
                "rank": rank,
                "gpu_index": rank,
                "gpu_uuid": f"GPU-{rank}",
            }
            for rank in range(4)
        ],
        "pair_groups": [[0, 1], [2, 3]],
        "campaign_epochs": [
            {
                "epoch": epoch,
                "arm": arm,
                "workload_order": (
                    list(worker.WORKLOADS)
                    if epoch in (0, 2)
                    else list(reversed(worker.WORKLOADS))
                ),
            }
            for epoch, arm in enumerate(worker.EPOCH_ARMS)
        ],
    }
    cleanup = _raw_candidate_evidence(worker)["cleanup"]
    baseline_cleanup = copy.deepcopy(cleanup)
    for receipt in baseline_cleanup["rank_cleanup_receipts"]:
        receipt["qwen38_topology_local_tp2_cleanup"] = None
    service_cleanup = copy.deepcopy(baseline_cleanup)
    service_cleanup["rank_exit_codes"] = [0, 0]
    service_cleanup["rank_cleanup_receipts"] = [
        receipt
        for receipt in service_cleanup["rank_cleanup_receipts"]
        if receipt["rank"] in (0, 1)
    ]
    epoch_results = []
    for epoch, arm in enumerate(worker.EPOCH_ARMS):
        cases = []
        for workload_id, (
            _,
            prompt_tokens,
            output_tokens,
            concurrency,
        ) in worker.WORKLOADS.items():
            for repetition in range(worker.MEASURED_REPETITIONS):
                request_specs = worker.build_request_specs(
                    prompt_tokens,
                    output_tokens,
                    concurrency,
                    f"timing/{workload_id}/r{repetition}",
                )
                digest = worker._request_set_digest(request_specs)
                tpot_ns = 94.0 if arm == "candidate" else 100.0
                requests = []
                for request_index, request_spec in enumerate(request_specs):
                    admitted_ns = 1_000_000 + request_index
                    token_timestamps_ns = [
                        admitted_ns + 1_000 + step * int(tpot_ns)
                        for step in range(128)
                    ]
                    decoded_text = ",".join(
                        str(token) for token in range(128)
                    )
                    requests.append({
                        "request_id": request_spec["request_id"],
                        "runtime_request_id": request_index,
                        "admitted_ns": admitted_ns,
                        "first_scheduled_ns": 1_000_500,
                        "queueing_ns": 1_000_500 - admitted_ns,
                        "token_timestamps_ns": token_timestamps_ns,
                        "completion_ns": token_timestamps_ns[-1],
                        "output_token_ids": list(range(128)),
                        "token_gaps_ns": [int(tpot_ns)] * 127,
                        "ttft_ns": 1_000.0,
                        "tpot_ns": tpot_ns,
                        "e2e_ns": 1_000.0 + tpot_ns * 127,
                        "complete": True,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": output_tokens,
                        "stop_position": 128,
                        "stop_reason": "length",
                        "decoded_text": decoded_text,
                        "decoded_text_sha256": hashlib.sha256(
                            decoded_text.encode("utf-8")
                        ).hexdigest(),
                    })
                case = {
                    "warmup": False,
                    "workload_id": workload_id,
                    "repetition": repetition,
                    "request_set_digest": digest,
                    "requests": requests,
                    "cohort_makespan_ns": (
                        max(row["completion_ns"] for row in requests)
                        - min(row["admitted_ns"] for row in requests)
                    ),
                    "memory": [{
                        "rank": rank,
                        "cuda_peak_allocated_bytes": (
                            70 * 1024**3
                            + (1024**3 if arm == "candidate" else 0)
                        ),
                        "cuda_peak_reserved_bytes": 72 * 1024**3,
                        "physical_memory_bytes": 80 * 1024**3,
                    } for rank in range(4)],
                    "scheduler_step_rows": [{
                        "step_index": step,
                        "is_prefill": step == 0,
                        "batch_kind": (
                            "prefill" if step == 0 else "decode"
                        ),
                        "request_ids": [
                            request["request_id"] for request in requests
                        ],
                        "step_start_ns": 1_000_500 + step * 100,
                        "step_end_ns": 1_000_580 + step * 100,
                        "step_duration_ns": 80,
                        "host_submission_ns": 60,
                    } for step in range(128)],
                    "token_count_rows": [{
                        "step_index": step,
                        "token_count": concurrency,
                    } for step in range(128)],
                }
                case["timing_correctness_replay"] = (
                    _timing_correctness_replay(requests)
                )
                if arm == "candidate":
                    evidence = _raw_candidate_evidence(
                        worker,
                        concurrency=concurrency,
                    )
                    for snapshot in evidence["after_snapshots"]:
                        snapshot["last_transition_latency_ns"] = 100
                        snapshot["released_layer_count"] = 48
                        snapshot["released_bytes"] = 48 * 1024
                    case.update({
                        "before_snapshots": evidence["before_snapshots"],
                        "after_snapshots": evidence["after_snapshots"],
                    })
                cases.append(case)
        epoch_results.append({
            "epoch": epoch,
            "arm": arm,
            "rows": cases,
            "cleanup": (
                cleanup if arm == "candidate" else baseline_cleanup
            ),
            "startup_model_load_started_ns": 100,
            "startup_model_load_finished_ns": 1_000_100,
            "startup_model_load_duration_ns": 1_000_000,
        })

    fake = _FakeEngine((
        {
            "request_id": "correctness",
            "prompt_token_ids": [1],
            "output_tokens": 128,
        },
    ), candidate=True)
    baseline_fake = _FakeEngine((
        {
            "request_id": "correctness",
            "prompt_token_ids": [1],
            "output_tokens": 128,
        },
    ), candidate=False)
    baseline_fake._step = 1
    baseline_checkpoints = {
        name: baseline_fake.qwen38_correctness_state_checkpoints(
            timeout_s=1,
        )
        for name in worker.STATE_CHECKPOINTS
    }
    checkpoints = {}
    for name in worker.STATE_CHECKPOINTS:
        fake._step = (
            1
            if name in {"pre_migration", "token_1"}
            else 2
        )
        checkpoints[name] = tuple({
            **row,
            "runtime_snapshot": {
                "state": {
                    "commit_count": {
                        "pre_migration": 0,
                        "token_1": 0,
                        "post_migration": 1,
                        "token_4": 3,
                        "token_8": 7,
                        "token_32": 31,
                        "token_128": 127,
                    }[name],
                    "rollback_count": 0,
                    "temporary_live_tensors": 0,
                },
            },
        } for row in fake.qwen38_correctness_state_checkpoints(
            timeout_s=1,
        ))
    correctness_request = {
        "request_id": "correctness",
        "runtime_request_id": 0,
        "output_token_ids": list(range(128)),
        "rank_token_agreement": True,
        "finite_logits": True,
    }
    correctness_proofs = [[{
        "rank": rank,
        "sequence_ids": [0],
        "finite_logits": True,
        "token_ids": [step],
        "top_logit_values": [float(step)],
    } for rank in range(4)] for step in range(128)]
    correctness_result = {
        "rows": [{
            "workload_id": workload_id,
            "repetition": repetition,
            "baseline": {
                "requests": [correctness_request],
                "correctness_step_proofs": correctness_proofs,
                    "correctness_state_checkpoints":
                        baseline_checkpoints,
            },
            "candidate": {
                "requests": [dict(correctness_request)],
                "correctness_step_proofs": correctness_proofs,
                "correctness_state_checkpoints": checkpoints,
            },
        } for workload_id in worker.WORKLOADS
        for repetition in range(worker.MEASURED_REPETITIONS)],
        "cleanup": {
            "baseline": baseline_cleanup,
            "candidate": cleanup,
        },
    }
    baseline_service_requests = {
        row["workload_id"]: {
            request["request_id"]: request
            for request in row["requests"]
        }
        for row in epoch_results[0]["rows"]
        if row["repetition"] == 0
    }

    def service_parallel_runner(
        *,
        workload_id,
        request_specs_by_replica,
        **_kwargs,
    ):
        replicas = []
        for request_specs in request_specs_by_replica:
            replicas.append({
                "requests": [
                    copy.deepcopy(
                        baseline_service_requests[workload_id][
                            request["request_id"]
                        ]
                    )
                    for request in request_specs
                ],
                "memory": [{
                    "rank": rank,
                    "cuda_peak_allocated_bytes": 35 * 1024**3,
                    "cuda_peak_reserved_bytes": 36 * 1024**3,
                    "physical_memory_bytes": 80 * 1024**3,
                } for rank in range(2)],
                "replica_tensor_parallel_size": 2,
                "request_set_digest":
                    worker._request_set_digest(request_specs),
                "cleanup": service_cleanup,
            })
        return tuple(replicas)

    service_result = worker.run_service_control(
        model_root=Path("/model"),
        output_root=tmp_path,
        pair_devices=((0, 1), (2, 3)),
        workloads=("Q0", "Q1", "Q2"),
        parallel_runner=service_parallel_runner,
        row_sink=lambda _row: None,
    )
    stages = (
        "entry",
        "pre_correctness",
        "post_correctness",
        *(f"pre_epoch_{index}" for index in range(4)),
        *(f"post_launch_{index}" for index in range(4)),
        "pre_service_control",
        "post_service_control",
        "terminal",
    )
    resource_samples = tuple({
        "stage": stage,
        "measurement_scope": "boundary",
        "gpu_inventory": [{
            "gpu_index": rank,
            "gpu_uuid": f"GPU-{rank}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
                "power_watts": 70.0 + rank,
            "compute_processes": [],
        } for rank in range(4)],
        "process_rows": [],
    } for stage in stages)
    resource_samples += tuple({
        "stage": f"runtime_{run_label}_0000",
        "measurement_scope": "runtime",
        "run_label": run_label,
        "sample_index": 0,
        "gpu_inventory": [{
            "gpu_index": rank,
            "gpu_uuid": f"GPU-{rank}",
            "memory_used_mib": 4096,
            "utilization_percent": 80,
            "power_watts": 250.0 + rank,
            "compute_processes": [{"pid": 1000 + rank}],
        } for rank in range(4)],
        "process_rows": [],
    } for run_label in (
        "correctness",
        *(f"epoch_{index}" for index in range(4)),
        "service_control",
    ))

    payloads = worker.build_raw_artifact_payloads(
        plan=plan,
        correctness_result=correctness_result,
        epoch_results=tuple(epoch_results),
        service_result=service_result,
        resource_samples=resource_samples,
    )
    raw_root = tmp_path / "attempt" / "raw"
    raw_root.mkdir(parents=True)
    for name, payload in payloads.items():
        path = raw_root / name
        if name.endswith(".jsonl"):
            worker._atomic_write_jsonl(path, payload)
        else:
            worker._atomic_write_json(path, payload)

    result = assemble_attempt(
        raw_root.parent,
        tmp_path / "final_bundle",
    )

    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"
    )
