#!/usr/bin/env python3
"""Dependency-light contracts for the exact greedy decode burst profiler."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

try:
    import pytest
except ModuleNotFoundError:
    class _Raises:
        def __init__(self, expected, *, match=None):
            self.expected = expected
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exception_type, exception, _traceback):
            if exception_type is None:
                raise AssertionError(
                    f"did not raise {self.expected!r}"
                )
            if not issubclass(exception_type, self.expected):
                return False
            if (
                self.match is not None
                and re.search(self.match, str(exception)) is None
            ):
                raise AssertionError(
                    f"{exception!r} does not match {self.match!r}"
                )
            return True

    class _PytestCompat:
        @staticmethod
        def raises(expected, *, match=None):
            return _Raises(expected, match=match)

    pytest = _PytestCompat()


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.profile_exact_greedy_decode_burst import (
    CASE_SCHEMA_VERSION,
    CONTEXT_CASES,
    CORRECTNESS_SCHEMA_VERSION,
    POLICIES,
    POLICY_CONFIGS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    SOURCE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
    build_workload_manifest,
    correctness_trace_for_step,
    correctness_uses_burst_trace,
    correctness_identities,
    performance_identities,
    policy_order,
    read_float32_sidecar,
    summarize_rows,
    validate_case_row,
    validate_correctness_rows,
    write_float32_sidecar,
)


RUN_TAG = "20260822-qwen3-06b-exact-burst-test"
SOURCE_COMMIT = "a" * 40


def _capture_receipt(*, correctness_trace: bool = False) -> dict:
    return {
        "graph_identity_sha256": "c" * 64,
        "graph_generation": 7,
        "capture_duration_ns": 1_000_000,
        "allocated_delta_bytes": 400_000,
        "reserved_delta_bytes": 2_000_000,
        "retained_static_bytes": 900_008,
        "scratch_block_count": 1,
        "correctness_trace": correctness_trace,
    }


def _burst_summary(
    policy: str,
    *,
    correctness_trace: bool = False,
) -> dict:
    width = POLICY_CONFIGS[policy]["width"]
    enabled = POLICY_CONFIGS[policy]["enabled"]
    replay_count = 0 if not enabled else 127
    commits = 0 if not enabled else math.ceil(127 / width)
    final_d2h_calls = commits
    sampled_d2h_calls = (
        3 if correctness_trace and enabled else 0
    )
    return {
        "attempts": commits,
        "acceptances": commits,
        "target_model_forwards": replay_count,
        "graph_replays": replay_count,
        "intermediate_token_d2h_calls": 0,
        "final_token_d2h_calls": final_d2h_calls,
        "final_token_d2h_bytes": replay_count * 8,
        "sampled_logit_d2h_calls": sampled_d2h_calls,
        "output_budget_clipped": int(enabled),
        "block_boundary_clipped": int(enabled),
        "commits": commits,
        "committed_tokens": replay_count,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
        "maximum_host_visible_gap_ns": (
            4_000_000 if enabled else 0
        ),
        "requested_width_histogram": (
            {str(width): commits} if enabled else {}
        ),
        "authorized_width_histogram": (
            {str(width): commits} if enabled else {}
        ),
        "fallback_counts": {},
        "quarantine_reason": None,
        "capture_receipts": (
            [_capture_receipt(
                correctness_trace=correctness_trace
            )]
            if enabled
            else []
        ),
    }


def _case_row(
    policy: str,
    *,
    bucket: str = "short",
    repetition: int = 0,
) -> dict:
    prompt_tokens, generated_tokens = {
        name: (prompt, generated)
        for name, prompt, generated in CONTEXT_CASES
    }[bucket]
    enabled = POLICY_CONFIGS[policy]["enabled"]
    width = POLICY_CONFIGS[policy]["width"]
    decode_step_count = (
        math.ceil(127 / width) if enabled else 127
    )
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "policy": policy,
        "selectable": POLICY_CONFIGS[policy]["selectable"],
        "burst_width": width,
        "repetition": repetition,
        "context_bucket": bucket,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "output_token_ids": list(range(generated_tokens)),
        "output_text_sha256": "b" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "amortized_tpot_samples_ns": [1_000_000] * 127,
        "amortized_tpot_median_ns": 1_000_000,
        "amortized_tpot_p95_ns": 1_000_000,
        "amortized_tpot_p99_ns": 1_000_000,
        "decode_host_ns": [900_000] * decode_step_count,
        "decode_cuda_ns": [700_000] * decode_step_count,
        "output_tokens_per_second": 914.285714,
        "host_visible_burst_gaps_ns": (
            [4_000_000] * math.ceil(127 / width)
            if enabled
            else []
        ),
        "maximum_host_visible_burst_gap_ns": (
            4_000_000 if enabled else 0
        ),
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "capture_duration_ns": (
            1_000_000 if enabled else 0
        ),
        "capture_allocated_delta_bytes": (
            400_000 if enabled else 0
        ),
        "capture_reserved_delta_bytes": (
            2_000_000 if enabled else 0
        ),
        "capture_retained_static_bytes": (
            900_008 if enabled else 0
        ),
        "reserved_scratch_blocks": 1 if enabled else 0,
        "correctness_trace": False,
        "exact_greedy_decode_burst_summary":
            _burst_summary(policy),
    }


def test_case_row_accepts_runtime_flash_attention_split_receipt():
    row = _case_row("decode_burst_k8")
    row["exact_greedy_decode_burst_summary"][
        "capture_receipts"
    ][0]["flash_attn_num_splits"] = 0

    validated = validate_case_row(row)

    assert validated["exact_greedy_decode_burst_summary"][
        "capture_receipts"
    ][0]["flash_attn_num_splits"] == 0


def _correctness_row(
    *,
    policy: str,
    bucket: str,
    sampling_point: str,
    sidecar: dict,
) -> dict:
    prompt_tokens, generated_tokens = {
        name: (prompt, generated)
        for name, prompt, generated in CONTEXT_CASES
    }[bucket]
    return {
        "schema_version": CORRECTNESS_SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "policy": policy,
        "context_bucket": bucket,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "sampling_point": sampling_point,
        "output_token_ids": list(range(generated_tokens)),
        "output_text_sha256": "b" * 64,
        "logits_path": sidecar["path"],
        "logits_shape": [1, 4],
        "logits_element_count": sidecar["element_count"],
        "logits_byte_length": sidecar["byte_length"],
        "logits_sha256": sidecar["sha256"],
        "correctness_trace": True,
        "trace_identity": "gate-only-exact-burst-correctness-v1",
        "trace_graph_identity_sha256": (
            "c" * 64
            if policy != "host_greedy"
            and sampling_point != "prefill-final"
            else None
        ),
        "selected_replay_ordinal": (
            _selected_replay_ordinal(
                policy,
                sampling_point,
                generated_tokens,
            )
            if policy != "host_greedy"
            and sampling_point != "prefill-final"
            else None
        ),
        "sampled_logit_d2h_calls": (
            1
            if policy != "host_greedy"
            and sampling_point != "prefill-final"
            else 0
        ),
        "exact_greedy_decode_burst_summary": _burst_summary(
            policy,
            correctness_trace=True,
        ),
    }


def _selected_replay_ordinal(
    policy: str,
    sampling_point: str,
    generated_tokens: int,
) -> int:
    output_index = {
        "decode-first": 1,
        "decode-middle": generated_tokens // 2,
        "decode-final": generated_tokens - 1,
    }[sampling_point]
    return (output_index - 1) % POLICY_CONFIGS[policy]["width"]


def test_schema_policy_and_context_contracts_are_exact() -> None:
    assert CASE_SCHEMA_VERSION == (
        "exact-greedy-decode-burst.case.v1"
    )
    assert CORRECTNESS_SCHEMA_VERSION == (
        "exact-greedy-decode-burst.correctness.v1"
    )
    assert SUMMARY_SCHEMA_VERSION == (
        "exact-greedy-decode-burst.summary.v1"
    )
    assert WORKLOAD_SCHEMA_VERSION == (
        "exact-greedy-decode-burst.workload.v1"
    )
    assert SOURCE_SCHEMA_VERSION == (
        "exact-greedy-decode-burst.source.v1"
    )
    assert POLICIES == (
        "host_greedy",
        "full_step_graph_k1",
        "decode_burst_k4",
        "decode_burst_k8",
    )
    assert CONTEXT_CASES == (
        ("short", 256, 128),
        ("medium", 2048, 128),
        ("long", 8192, 128),
    )
    assert SAMPLING_POINTS == (
        "prefill-final",
        "decode-first",
        "decode-middle",
        "decode-final",
    )


def test_four_order_latin_rotation_reverses_odd_contexts() -> None:
    expected = (
        POLICIES,
        POLICIES[1:] + POLICIES[:1],
        POLICIES[2:] + POLICIES[:2],
        POLICIES[3:] + POLICIES[:3],
    )
    for repetition in range(5):
        assert policy_order(repetition, 0) == expected[
            repetition % 4
        ]
        assert policy_order(repetition, 1) == tuple(
            reversed(expected[repetition % 4])
        )
        assert policy_order(repetition, 2) == expected[
            repetition % 4
        ]


def test_exact_performance_and_correctness_identities() -> None:
    performance = performance_identities(repetitions=5)
    assert len(performance) == 60
    assert len(set(performance)) == 60
    assert performance[0] == (
        0,
        "short",
        "host_greedy",
    )

    correctness = correctness_identities()
    assert len(correctness) == 48
    assert len(set(correctness)) == 48
    assert (
        "long",
        "decode_burst_k8",
        "decode-final",
    ) in correctness


def test_policy_promotion_contract_excludes_k1() -> None:
    assert POLICY_CONFIGS["host_greedy"] == {
        "enabled": False,
        "width": 1,
        "selectable": False,
        "entrypoint": "ordinary",
    }
    assert POLICY_CONFIGS["full_step_graph_k1"][
        "selectable"
    ] is False
    assert POLICY_CONFIGS["full_step_graph_k1"][
        "entrypoint"
    ] == "gate_direct"
    assert POLICY_CONFIGS["decode_burst_k4"][
        "selectable"
    ] is True
    assert POLICY_CONFIGS["decode_burst_k8"][
        "selectable"
    ] is True
    assert correctness_uses_burst_trace("host_greedy") is False
    assert correctness_uses_burst_trace(
        "full_step_graph_k1"
    ) is True
    assert correctness_uses_burst_trace(
        "decode_burst_k4"
    ) is True


def test_workload_manifest_freezes_stage_one_contract() -> None:
    environment = {
        "python_version": "3.12.1",
        "platform": "Linux-test",
        "torch_version": "2.7.0",
        "cuda_runtime_version": "12.8",
        "cuda_device_name": "NVIDIA A100",
    }
    manifest = build_workload_manifest(
        model="/models/Qwen3-0.6B",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        gpu_memory_utilization=0.5,
        environment=environment,
    )
    assert manifest["schema_version"] == WORKLOAD_SCHEMA_VERSION
    assert manifest["warmup_repetitions"] == 2
    assert manifest["repetitions"] == 5
    assert manifest["batch_size"] == 1
    assert manifest["temperature"] == 0.0
    assert manifest["ignore_eos"] is True
    assert manifest["generated_tokens"] == 128
    assert manifest["performance_row_count"] == 60
    assert manifest["correctness_row_count"] == 48
    assert manifest["performance_correctness_trace"] is False
    assert manifest["correctness_trace_identity"] == (
        "gate-only-exact-burst-correctness-v1"
    )
    assert manifest["environment"] == environment
    assert manifest["policy_order"]["0"]["short"] == list(
        POLICIES
    )
    assert manifest["policy_order"]["0"]["medium"] == list(
        reversed(POLICIES)
    )


def test_case_row_preserves_benefit_cost_and_inventory() -> None:
    for policy in POLICIES:
        validated = validate_case_row(_case_row(policy))
        assert validated["correctness_trace"] is False
        assert len(
            validated["amortized_tpot_samples_ns"]
        ) == 127

    traced = _case_row("decode_burst_k4")
    traced["correctness_trace"] = True
    with pytest.raises(
        ValueError,
        match="performance row cannot enable correctness tracing",
    ):
        validate_case_row(traced)

    bad_d2h = _case_row("decode_burst_k4")
    bad_d2h[
        "exact_greedy_decode_burst_summary"
    ]["intermediate_token_d2h_calls"] = 1
    with pytest.raises(
        ValueError,
        match="intermediate token D2H",
    ):
        validate_case_row(bad_d2h)

    duplicate_receipt = _case_row("decode_burst_k4")
    duplicate_receipt[
        "exact_greedy_decode_burst_summary"
    ]["capture_receipts"].append(_capture_receipt())
    with pytest.raises(
        ValueError,
        match="exactly one capture receipt",
    ):
        validate_case_row(duplicate_receipt)

    missing_decode_profile = _case_row("decode_burst_k8")
    missing_decode_profile["decode_host_ns"] = []
    missing_decode_profile["decode_cuda_ns"] = []
    with pytest.raises(
        ValueError,
        match=(
            "decode profile inventory mismatch: "
            "policy=decode_burst_k8, expected_steps=16, "
            "host_steps=0, cuda_steps=0, "
            "commits=16, attempts=16, fallbacks=\\{\\}"
        ),
    ):
        validate_case_row(missing_decode_profile)

    polluted_gap = _case_row("decode_burst_k4")
    polluted_gap[
        "exact_greedy_decode_burst_summary"
    ]["maximum_host_visible_gap_ns"] = 9_000_000
    with pytest.raises(
        ValueError,
        match="summary host-visible gap",
    ):
        validate_case_row(polluted_gap)


def test_summary_requires_exact_60_row_inventory() -> None:
    rows = [
        _case_row(
            policy,
            bucket=bucket,
            repetition=repetition,
        )
        for repetition in range(5)
        for bucket, _prompt, _generated in CONTEXT_CASES
        for policy in POLICIES
    ]
    summary = summarize_rows(rows, expected_repetitions=5)
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["row_count"] == 60
    assert summary["comparison_set_count"] == 15
    assert summary["all_outputs_exact"] is True

    with pytest.raises(ValueError, match="case row inventory"):
        summarize_rows(rows[:-1], expected_repetitions=5)

    duplicate = rows + [deepcopy(rows[0])]
    with pytest.raises(
        ValueError,
        match="duplicate case identity",
    ):
        summarize_rows(duplicate, expected_repetitions=5)

    mixed_run = deepcopy(rows)
    mixed_run[-1]["run_tag"] = RUN_TAG + "-other"
    with pytest.raises(
        ValueError,
        match="performance rows do not share source identity",
    ):
        summarize_rows(mixed_run, expected_repetitions=5)

    mixed_source = deepcopy(rows)
    mixed_source[-1]["source_commit"] = "d" * 40
    with pytest.raises(
        ValueError,
        match="performance rows do not share source identity",
    ):
        summarize_rows(mixed_source, expected_repetitions=5)


def test_correctness_trace_only_selects_containing_bursts() -> None:
    expected_starts = {
        "full_step_graph_k1": {1, 64, 127},
        "decode_burst_k4": {1, 61, 125},
        "decode_burst_k8": {1, 57, 121},
    }
    for policy, expected in expected_starts.items():
        emitted_total = 1
        selected = set()
        while emitted_total < 128:
            if correctness_trace_for_step(
                policy,
                emitted_total=emitted_total,
                generated_tokens=128,
            ):
                selected.add(emitted_total)
            emitted_total += min(
                POLICY_CONFIGS[policy]["width"],
                128 - emitted_total,
            )
        assert selected == expected
    assert correctness_trace_for_step(
        "host_greedy",
        emitted_total=1,
        generated_tokens=128,
    ) is False


def test_correctness_rows_require_exact_48_float32_sidecars() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for bucket, _prompt, _generated in CONTEXT_CASES:
            for policy in POLICIES:
                for point in SAMPLING_POINTS:
                    sidecar = write_float32_sidecar(
                        root,
                        f"logits/{bucket}-{policy}-{point}.f32",
                        (1.0, 2.0, 4.0, 3.0),
                    )
                    rows.append(
                        _correctness_row(
                            policy=policy,
                            bucket=bucket,
                            sampling_point=point,
                            sidecar=sidecar,
                        )
                    )
        validated = validate_correctness_rows(
            rows,
            run_dir=root,
        )
        assert len(validated) == 48
        restored = read_float32_sidecar(
            root,
            path=rows[0]["logits_path"],
            expected_element_count=4,
            expected_byte_length=16,
            expected_sha256=rows[0]["logits_sha256"],
        )
        assert restored == (1.0, 2.0, 4.0, 3.0)

        untraced = deepcopy(rows)
        untraced[0]["correctness_trace"] = False
        with pytest.raises(
            ValueError,
            match="gate-only correctness trace",
        ):
            validate_correctness_rows(
                untraced,
                run_dir=root,
            )

        with pytest.raises(
            ValueError,
            match="correctness row inventory",
        ):
            validate_correctness_rows(
                rows[:-1],
                run_dir=root,
            )

        missing_run_tag = deepcopy(rows)
        for row in missing_run_tag:
            row["run_tag"] = None
        with pytest.raises(
            ValueError,
            match="correctness run tag",
        ):
            validate_correctness_rows(
                missing_run_tag,
                run_dir=root,
            )

        wrong_graph = deepcopy(rows)
        burst_row = next(
            row
            for row in wrong_graph
            if row["policy"] == "decode_burst_k4"
            and row["sampling_point"] == "decode-first"
        )
        burst_row["trace_graph_identity_sha256"] = "d" * 64
        with pytest.raises(
            ValueError,
            match="correctness graph identity",
        ):
            validate_correctness_rows(
                wrong_graph,
                run_dir=root,
            )

        wrong_ordinal = deepcopy(rows)
        burst_row = next(
            row
            for row in wrong_ordinal
            if row["policy"] == "decode_burst_k8"
            and row["sampling_point"] == "decode-final"
        )
        burst_row["selected_replay_ordinal"] = 7
        with pytest.raises(
            ValueError,
            match="selected replay ordinal",
        ):
            validate_correctness_rows(
                wrong_ordinal,
                run_dir=root,
            )

        extra_d2h = deepcopy(rows)
        host_row = next(
            row
            for row in extra_d2h
            if row["policy"] == "host_greedy"
            and row["sampling_point"] == "decode-first"
        )
        host_row["sampled_logit_d2h_calls"] = 1
        with pytest.raises(
            ValueError,
            match="sampled-logit D2H",
        ):
            validate_correctness_rows(
                extra_d2h,
                run_dir=root,
            )

        missing_burst_d2h = deepcopy(rows)
        for row in missing_burst_d2h:
            if row["policy"] == "decode_burst_k4":
                row[
                    "exact_greedy_decode_burst_summary"
                ]["sampled_logit_d2h_calls"] = 2
        with pytest.raises(
            ValueError,
            match="correctness sampled-logit D2H inventory",
        ):
            validate_correctness_rows(
                missing_burst_d2h,
                run_dir=root,
            )


def test_source_manifest_inventory_covers_runtime_and_gate_chain() -> None:
    required = {
        "tinyvllm/config.py",
        "tinyvllm/engine/exact_greedy_decode_burst.py",
        "tinyvllm/engine/model_runner.py",
        "tinyvllm/engine/scheduler.py",
        "tinyvllm/engine/llm_engine.py",
        "tools/profile_exact_greedy_decode_burst.py",
        "tools/test_profile_exact_greedy_decode_burst.py",
        "tools/exact_greedy_decode_burst_gate.py",
        "tools/test_exact_greedy_decode_burst_gate.py",
        "tools/exact_greedy_decode_burst_verify.py",
        "tools/test_exact_greedy_decode_burst_verify.py",
        "tools/run_exact_greedy_decode_burst_remote.py",
        "tools/test_run_exact_greedy_decode_burst_remote.py",
    }
    assert required <= set(SOURCE_FILES)


def main() -> None:
    test_schema_policy_and_context_contracts_are_exact()
    test_four_order_latin_rotation_reverses_odd_contexts()
    test_exact_performance_and_correctness_identities()
    test_policy_promotion_contract_excludes_k1()
    test_workload_manifest_freezes_stage_one_contract()
    test_case_row_preserves_benefit_cost_and_inventory()
    test_summary_requires_exact_60_row_inventory()
    test_correctness_trace_only_selects_containing_bursts()
    test_correctness_rows_require_exact_48_float32_sidecars()
    test_source_manifest_inventory_covers_runtime_and_gate_chain()
    print("exact greedy decode burst profile tests passed")


if __name__ == "__main__":
    main()
