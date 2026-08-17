from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    name = "qwen35_tp4_cached_partition_diagnostic"
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_cached_partition_diagnostic.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = _load_module()


class FakeConfiguration:

    gpu_indices = (2, 4, 5, 6)
    model_fingerprint = "f" * 64
    max_cache_entries = 8
    max_cache_bytes = 1 << 30
    timeout_s = 30.0


class FakeReferenceExecutor:

    def __init__(self):
        self.closed = False

    def generate_reference_with_step_logits(self, **kwargs):
        assert kwargs["scenario"] == "publish_source"
        assert kwargs["generated_tokens"] == 2
        return {
            "output_token_ids": [7, 8],
            "step_logits": [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        }

    def close(self):
        self.closed = True


class FakeEngine:

    def __init__(self, runs, *, configure_after_requests):
        self.configured = False
        self.request_index = -1
        self.step_index = 0
        self.logits_read_index = 0
        self.current = None
        self.recording = False
        self.exited = False
        self.configure_after_requests = configure_after_requests
        self.snapshots = {
            "current_entries": 0,
            "hits": 0,
            "misses": 0,
            "publication_commits": 0,
            "invalidations": 0,
            "clears": 0,
            "last_publication_block_identities": [],
        }
        self.runs = runs

    def configure_qwen35_hybrid_prefix_publication_runtime(self, **kwargs):
        assert self.request_index + 1 == self.configure_after_requests
        self.configured = True

    def clear_qwen35_hybrid_prefix_caches(self, **kwargs):
        self.snapshots["current_entries"] = 0

    def qwen35_hybrid_prefix_authority_snapshots(self, **kwargs):
        return tuple(
            {"rank": rank, **self.snapshots}
            for rank in range(4)
        )

    def enable_step_logits_authority_recording(self, enabled, **kwargs):
        self.recording = enabled
        return {
            "enabled": enabled,
            "rank_inventory": [0, 1, 2, 3],
        }

    def add_request(self, prompt, sampling_params):
        self.request_index += 1
        self.step_index = 0
        self.logits_read_index = 0
        self.current = self.runs[self.request_index]
        assert len(prompt) == 1088

    def is_finished(self):
        return self.step_index >= len(self.current["chunks"]) + 1

    def step(self):
        chunks = self.current["chunks"]
        if self.step_index < len(chunks):
            start, end = chunks[self.step_index]
            sampled = self.step_index == len(chunks) - 1
            self.last_step_observation = {
                "do_sample": sampled,
                "scheduled": [{
                    "seq_id": self.request_index,
                    "is_decode": False,
                    "do_sample": sampled,
                    "prefill_chunk_start": start,
                    "prefill_chunk_end": end,
                    "prefill_chunk_final": end == 1088,
                }],
                "new_completion_tokens_by_seq": (
                    {self.request_index: [7]} if sampled else {}
                ),
            }
            result = ([], end - start)
        else:
            token_offset = self.step_index - len(chunks)
            self.last_step_observation = {
                "do_sample": True,
                "scheduled": [],
                "new_completion_tokens_by_seq": {
                    self.request_index: [7 + token_offset]
                },
            }
            result = (
                (
                    [(self.request_index, [7, 8])]
                    if token_offset == 0
                    else []
                ),
                -1,
            )
        self.step_index += 1
        if (
            self.configured
            and self.request_index == 0
            and self.step_index == len(chunks)
        ):
            self.snapshots.update({
                "current_entries": 1,
                "misses": 1,
                "publication_commits": 1,
                "last_publication_block_identities": [[1, 2, 3]],
            })
        if self.configured and self.request_index == 1:
            self.snapshots["hits"] = 1
        return result

    def read_step_logits_authority(self):
        row = list(
            self.current["logits"][self.logits_read_index]
        )
        self.logits_read_index += 1
        return row

    def exit(self):
        self.exited = True
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [],
        }


def _compare(left, right, *, atol):
    per_step = [
        max(abs(a - b) for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    ]
    return {
        "max_abs_diff": max(per_step),
        "per_step_max_abs_diff": per_step,
        "allclose": all(value <= atol for value in per_step),
        "first_mismatch_step": next(
            (
                index
                for index, value in enumerate(per_step)
                if value > atol
            ),
            None,
        ),
    }


def _full_engine():
    return FakeEngine(
        [{
            "chunks": [(0, 1088)],
            "prefill": 1088,
            "logits": [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        }],
        configure_after_requests=1,
    )


def _partition_engine():
    return FakeEngine(
        [
            {
                "chunks": [(0, 1024), (1024, 1088)],
                "prefill": 1088,
                "logits": [
                    [0.0, 1.125],
                    [1.25, 0.0],
                ],
            },
            {
                "chunks": [(1024, 1088)],
                "prefill": 64,
                "logits": [
                    [0.0, 1.125],
                    [1.25, 0.0],
                ],
            },
        ],
        configure_after_requests=0,
    )


def test_diagnostic_isolates_full_partition_and_restore_paths():
    engines = [_full_engine(), _partition_engine()]
    result = diagnostic.run_diagnostic(
        configuration=FakeConfiguration(),
        prompt_token_ids=[3] * 1088,
        generated_tokens=2,
        engine_factory=lambda configuration: engines.pop(0),
        reference_executor_factory=FakeReferenceExecutor,
        logits_comparator=_compare,
    )

    assert result["schema_version"] == (
        "qwen35.tp4-cached-partition-diagnostic.v1"
    )
    assert [
        row["name"] for row in result["runs"]
    ] == [
        "native_full",
        "native_partitioned_miss",
        "native_restored_hit",
    ]
    assert [
        row["executed_prefill_tokens"] for row in result["runs"]
    ] == [1088, 1088, 64]
    assert [
        row["prefill_chunks"] for row in result["runs"]
    ] == [
        [[0, 1088]],
        [[0, 1024], [1024, 1088]],
        [[1024, 1088]],
    ]
    comparisons = result["comparisons"]
    assert comparisons["native_full_vs_native_partitioned_miss"][
        "per_step_max_abs_diff"
    ] == [0.125, 0.25]
    assert comparisons[
        "native_partitioned_miss_vs_native_restored_hit"
    ]["max_abs_diff"] == 0.0
    assert comparisons["official_vs_native_full"][
        "max_abs_diff"
    ] == 0.0
    assert result["cache_deltas"] == {
        "partitioned_miss": {
            "hits": 0,
            "misses": 1,
            "publication_commits": 1,
        },
        "restored_hit": {
            "hits": 1,
            "misses": 0,
            "publication_commits": 0,
        },
    }
    assert result["classification"] == (
        "PARTITION_NON_EQUIVALENCE_RESTORE_EXACT"
    )
    assert not engines


def test_diagnostic_keeps_native_evidence_when_reference_is_unavailable():
    class FailingReferenceExecutor:

        def generate_reference_with_step_logits(self, **kwargs):
            raise RuntimeError("reference worker unavailable")

        def close(self):
            pass

    engines = [_full_engine(), _partition_engine()]
    result = diagnostic.run_diagnostic(
        configuration=FakeConfiguration(),
        prompt_token_ids=[3] * 1088,
        generated_tokens=2,
        engine_factory=lambda configuration: engines.pop(0),
        reference_executor_factory=FailingReferenceExecutor,
        logits_comparator=_compare,
    )

    assert result["classification"] == (
        "PARTITION_NON_EQUIVALENCE_RESTORE_EXACT"
    )
    assert result["official_reference"] == {
        "status": "unavailable",
        "error_type": "RuntimeError",
        "error_detail": "reference worker unavailable",
    }
    assert set(result["comparisons"]) == {
        "native_full_vs_native_partitioned_miss",
        "native_partitioned_miss_vs_native_restored_hit",
    }


def test_merge_phase_artifacts_classifies_without_live_engines():
    result = diagnostic.merge_phase_artifacts(
        full_phase={
            "run": {
                "name": "native_full",
                "executed_prefill_tokens": 1088,
                "prefill_chunks": [[0, 1088]],
                "output_token_ids": [7, 8],
                "_step_logits": [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
            },
        },
        partition_phase={
            "runs": [
                {
                    "name": "native_partitioned_miss",
                    "executed_prefill_tokens": 1088,
                    "prefill_chunks": [[0, 1024], [1024, 1088]],
                    "output_token_ids": [7, 8],
                    "_step_logits": [
                        [0.0, 1.125],
                        [1.25, 0.0],
                    ],
                },
                {
                    "name": "native_restored_hit",
                    "executed_prefill_tokens": 64,
                    "prefill_chunks": [[1024, 1088]],
                    "output_token_ids": [7, 8],
                    "_step_logits": [
                        [0.0, 1.125],
                        [1.25, 0.0],
                    ],
                },
            ],
            "cache_deltas": {
                "partitioned_miss": {
                    "hits": 0,
                    "misses": 1,
                    "publication_commits": 1,
                },
                "restored_hit": {
                    "hits": 1,
                    "misses": 0,
                    "publication_commits": 0,
                },
            },
        },
        prompt_tokens=1088,
        generated_tokens=2,
        logits_comparator=_compare,
    )

    assert result["classification"] == (
        "PARTITION_NON_EQUIVALENCE_RESTORE_EXACT"
    )
    assert result["comparisons"][
        "native_full_vs_native_partitioned_miss"
    ]["per_step_max_abs_diff"] == [0.125, 0.25]

