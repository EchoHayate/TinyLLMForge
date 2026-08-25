from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.profile_context_gated_elastic_exact_burst import (
    CONTEXT_LENGTHS,
    GENERATED_TOKENS,
    POLICIES,
    POLICY_CONFIGS,
    REPETITIONS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    WARMUP_REPETITIONS,
    _construct_llm,
    build_workload_manifest,
    correctness_identities,
    performance_identities,
    policy_order,
    summarize_rows,
    validate_case_row,
    validate_correctness_rows,
    write_float32_sidecar,
)


RUN_TAG = "20260825-context-gated-elastic-k16-fixture"
SOURCE_COMMIT = "a" * 40


def _receipt(*, correctness_trace: bool = False) -> dict:
    return {
        "graph_identity_sha256": "c" * 64,
        "graph_generation": 7,
        "capture_duration_ns": 1_000_000,
        "allocated_delta_bytes": 400_000,
        "reserved_delta_bytes": 2_000_000,
        "retained_static_bytes": 900_008,
        "scratch_block_count": 1,
        "correctness_trace": correctness_trace,
        "flash_attn_num_splits": 0,
    }


def _summary(policy: str, context_length: int) -> dict:
    elastic = policy == "context_gated_elastic_k16"
    eligible = elastic and context_length <= 2048
    committed_width = 16 if eligible else 8
    commits = 8 if committed_width == 16 else 16
    return {
        "attempts": commits,
        "acceptances": commits,
        "target_model_forwards": 127,
        "graph_replays": 127,
        "intermediate_token_d2h_calls": 0,
        "final_token_d2h_calls": commits,
        "final_token_d2h_bytes": 1_016,
        "sampled_logit_d2h_calls": 0,
        "output_budget_clipped": 1,
        "block_boundary_clipped": 0,
        "commits": commits,
        "committed_tokens": 127,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
        "maximum_host_visible_gap_ns": 4_000_000,
        "k16_attempts": commits if elastic else 0,
        "k16_acceptances": commits if eligible else 0,
        "k8_fallbacks": commits if elastic and not eligible else 0,
        "requested_width_histogram": {
            str(committed_width): commits,
        },
        "authorized_width_histogram": {
            "7": 1,
            str(committed_width): commits - 1,
        },
        "elastic_k16_fallback_counts": (
            {}
            if not elastic or eligible
            else {"context_above_2048": commits}
        ),
        "per_width_commits": {
            str(committed_width): commits,
        },
        "fallback_counts": {},
        "lease_local_delta_journal_attempts": commits,
        "lease_local_delta_journal_captures": commits,
        "lease_local_delta_journal_commits": commits,
        "lease_local_delta_journal_fallback_counts": {},
        "lease_local_delta_journal_rollbacks": 0,
        "lease_local_delta_journal_one_phase_attempts": commits,
        "lease_local_delta_journal_one_phase_captures": commits,
        "lease_local_delta_journal_one_phase_commits": commits,
        "lease_local_delta_journal_one_phase_fallback_counts": {},
        "lease_local_delta_journal_one_phase_rollbacks": 0,
        "quarantine_reason": None,
        "capture_receipts": [_receipt()],
    }


def _case_row(
    policy: str,
    *,
    context_length: int = 2048,
    repetition: int = 0,
    order_position: int | None = None,
) -> dict:
    if order_position is None:
        order_position = policy_order(
            repetition,
            CONTEXT_LENGTHS.index(context_length),
        ).index(policy)
    samples = [1_000_000.0] * (GENERATED_TOKENS - 1)
    summary = _summary(policy, context_length)
    return {
        "schema_version":
            "context-gated-elastic-exact-burst.case.v1",
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "policy": policy,
        "repetition": repetition,
        "order_position": order_position,
        "context_length": context_length,
        "prompt_tokens": context_length,
        "generated_tokens": GENERATED_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "completion_only": True,
        "prompt_sha256": "b" * 64,
        "output_token_ids": list(range(GENERATED_TOKENS)),
        "output_text_sha256": "e" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "amortized_tpot_samples_ns": samples,
        "amortized_tpot_median_ns": 1_000_000.0,
        "amortized_tpot_p95_ns": 1_000_000.0,
        "amortized_tpot_p99_ns": 1_000_000.0,
        "decode_host_ns": [900_000] * summary["commits"],
        "decode_cuda_ns": [700_000] * summary["commits"],
        "output_tokens_per_second": 914.285714,
        "host_visible_burst_gaps_ns": [4_000_000] * summary["commits"],
        "maximum_host_visible_burst_gap_ns": 4_000_000,
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "shared_capture_duration_ns": 1_000_000,
        "shared_capture_allocated_delta_bytes": 400_000,
        "shared_capture_reserved_delta_bytes": 2_000_000,
        "shared_capture_retained_static_bytes": 900_008,
        "elastic_incremental_allocated_bytes": 0,
        "elastic_incremental_reserved_bytes": 0,
        "elastic_incremental_retained_static_bytes": 0,
        "correctness_trace": False,
        "exact_greedy_decode_burst_summary": summary,
    }


def test_frozen_workload_and_alternating_order() -> None:
    assert POLICIES == (
        "fixed_k8",
        "context_gated_elastic_k16",
    )
    assert POLICY_CONFIGS == {
        "fixed_k8": {
            "exact_greedy_decode_burst_elastic_k16": False,
        },
        "context_gated_elastic_k16": {
            "exact_greedy_decode_burst_elastic_k16": True,
        },
    }
    assert CONTEXT_LENGTHS == (256, 2048, 4096, 8192)
    assert GENERATED_TOKENS == 128
    assert REPETITIONS == 5
    assert WARMUP_REPETITIONS == 2
    assert SAMPLING_POINTS == (
        "prefill-final",
        "decode-first",
        "decode-middle",
        "decode-final",
    )
    for repetition in range(REPETITIONS):
        for context_index in range(len(CONTEXT_LENGTHS)):
            expected = (
                POLICIES
                if (repetition + context_index) % 2 == 0
                else tuple(reversed(POLICIES))
            )
            assert policy_order(repetition, context_index) == expected


def test_manifest_binds_inventory_execution_and_source() -> None:
    manifest = build_workload_manifest(
        model="/models/Qwen3-0.6B",
        device="cuda:0",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        gpu_memory_utilization=0.5,
        environment={"fixture": True},
    )

    assert manifest["contexts"] == list(CONTEXT_LENGTHS)
    assert manifest["policies"] == list(POLICIES)
    assert manifest["performance_row_count"] == 40
    assert manifest["correctness_row_count"] == 32
    assert manifest["generated_tokens"] == 128
    assert manifest["temperature"] == 0.0
    assert manifest["ignore_eos"] is True
    assert manifest["tensor_parallel_size"] == 1
    assert manifest["max_num_seqs"] == 1
    assert manifest["completion_only"] is True
    assert manifest["device"] == "cuda:0"
    assert len(performance_identities(repetitions=REPETITIONS)) == 40
    assert len(correctness_identities()) == 32
    assert (
        "tools/profile_context_gated_elastic_exact_burst.py"
        in SOURCE_FILES
    )


def test_llm_arms_only_differ_by_elastic_k16_flag() -> None:
    calls = []

    def fake_llm(model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace()

    with patch.dict(
        sys.modules,
        {"tinyvllm": SimpleNamespace(LLM=fake_llm)},
    ):
        for policy in POLICIES:
            _construct_llm(
                model="/models/qwen3",
                device="cuda:0",
                prompt_tokens=2048,
                generated_tokens=GENERATED_TOKENS,
                gpu_memory_utilization=0.5,
                policy=policy,
            )

    control = calls[0][1]
    candidate = calls[1][1]
    differing = {
        key
        for key in set(control) | set(candidate)
        if control.get(key) != candidate.get(key)
    }
    assert differing == {
        "exact_greedy_decode_burst_elastic_k16"
    }
    assert control["exact_greedy_decode_burst"] is True
    assert control["exact_greedy_decode_burst_tokens"] == 8
    assert control["tensor_parallel_size"] == 1
    assert control["max_num_seqs"] == 1


def test_case_rows_bind_raw_metrics_costs_and_width_policy() -> None:
    for policy in POLICIES:
        for context_length in CONTEXT_LENGTHS:
            row = validate_case_row(
                _case_row(policy, context_length=context_length)
            )
            assert len(row["amortized_tpot_samples_ns"]) == 127
            assert math.isfinite(row["output_tokens_per_second"])

    illegal_long_k16 = _case_row(
        "context_gated_elastic_k16",
        context_length=4096,
    )
    summary = illegal_long_k16[
        "exact_greedy_decode_burst_summary"
    ]
    summary["k16_acceptances"] = 1
    summary["authorized_width_histogram"] = {"16": 1, "8": 15}
    with pytest.raises(ValueError, match="K16 selection"):
        validate_case_row(illegal_long_k16)

    missing_cost = _case_row("fixed_k8")
    del missing_cost["elastic_incremental_retained_static_bytes"]
    with pytest.raises(ValueError, match="fields"):
        validate_case_row(missing_cost)

    non_finite = _case_row("fixed_k8")
    non_finite["ttft_ns"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_case_row(non_finite)


def test_summary_rejects_missing_duplicate_and_mixed_source_rows() -> None:
    rows = [
        _case_row(
            policy,
            context_length=context_length,
            repetition=repetition,
            order_position=order_position,
        )
        for repetition in range(REPETITIONS)
        for context_index, context_length
        in enumerate(CONTEXT_LENGTHS)
        for order_position, policy in enumerate(
            policy_order(repetition, context_index)
        )
    ]
    summary = summarize_rows(rows, expected_repetitions=REPETITIONS)
    assert summary["row_count"] == 40
    assert summary["comparison_set_count"] == 20
    assert summary["all_outputs_exact"] is True

    with pytest.raises(ValueError, match="inventory"):
        summarize_rows(rows[:-1], expected_repetitions=REPETITIONS)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_rows(
            rows + [deepcopy(rows[0])],
            expected_repetitions=REPETITIONS,
        )
    mixed = deepcopy(rows)
    mixed[-1]["source_commit"] = "f" * 40
    with pytest.raises(ValueError, match="source identity"):
        summarize_rows(mixed, expected_repetitions=REPETITIONS)


def test_correctness_rows_bind_sidecars_tokens_argmax_and_logits(
    tmp_path: Path,
) -> None:
    rows = []
    for context_length in CONTEXT_LENGTHS:
        for policy in POLICIES:
            summary = _summary(policy, context_length)
            summary["sampled_logit_d2h_calls"] = 3
            summary["capture_receipts"][0][
                "correctness_trace"
            ] = True
            for point in SAMPLING_POINTS:
                sidecar = write_float32_sidecar(
                    tmp_path,
                    (
                        f"logits/{context_length}-{policy}-"
                        f"{point}.f32"
                    ),
                    (1.0, 4.0, 3.0),
                )
                rows.append({
                    "schema_version": (
                        "context-gated-elastic-exact-burst."
                        "correctness.v1"
                    ),
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "context_length": context_length,
                    "generated_tokens": GENERATED_TOKENS,
                    "sampling_point": point,
                    "prompt_sha256": "b" * 64,
                    "output_token_ids": list(
                        range(GENERATED_TOKENS)
                    ),
                    "output_text_sha256": "e" * 64,
                    "argmax_token_id": 1,
                    "logits_path": sidecar["path"],
                    "logits_shape": [1, 3],
                    "logits_element_count":
                        sidecar["element_count"],
                    "logits_byte_length": sidecar["byte_length"],
                    "logits_sha256": sidecar["sha256"],
                    "correctness_trace": True,
                    "exact_greedy_decode_burst_summary":
                        deepcopy(summary),
                })

    validated = validate_correctness_rows(rows, run_dir=tmp_path)
    assert len(validated) == 32

    wrong_argmax = deepcopy(rows)
    wrong_argmax[0]["argmax_token_id"] = 2
    with pytest.raises(ValueError, match="argmax"):
        validate_correctness_rows(wrong_argmax, run_dir=tmp_path)
    with pytest.raises(ValueError, match="inventory"):
        validate_correctness_rows(rows[:-1], run_dir=tmp_path)


def test_output_directory_is_exclusive(tmp_path: Path) -> None:
    from tools import profile_context_gated_elastic_exact_burst as profile

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        profile.create_output_directory(existing)

    created = profile.create_output_directory(tmp_path / "new")
    assert created.is_dir()
