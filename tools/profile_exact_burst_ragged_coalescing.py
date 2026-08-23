#!/usr/bin/env python3
"""Source-bound gate producer for split-phase ragged coalescing."""

from __future__ import annotations

from pathlib import Path

from tools import profile_exact_burst_split_phase as _split


_base = _split._base
_ORIGINAL_SPLIT_CORRECTNESS_TRACE_FOR_STEP = (
    _split.correctness_trace_for_step
)
_ORIGINAL_SPLIT_SAMPLED_LOCAL_ORDINALS = (
    _split._sampled_local_ordinals
)
_ORIGINAL_EXPECTED_SELECTED_REPLAY_ORDINAL = (
    _base._expected_selected_replay_ordinal
)

CASE_SCHEMA_VERSION = "exact-burst-ragged-coalescing.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "exact-burst-ragged-coalescing.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = (
    "exact-burst-ragged-coalescing.summary.v1"
)
WORKLOAD_SCHEMA_VERSION = (
    "exact-burst-ragged-coalescing.workload.v1"
)
SOURCE_SCHEMA_VERSION = (
    "exact-burst-ragged-coalescing.source.v1"
)
CORRECTNESS_TRACE_IDENTITY = (
    "gate-only-exact-burst-ragged-coalescing-correctness-v1"
)

POLICIES = (
    "decode_burst_k4",
    "decode_burst_k8_split_phase",
    "decode_burst_k8_split_phase_ragged",
)
POLICY_CONFIGS = {
    "decode_burst_k4": dict(
        _split.POLICY_CONFIGS["decode_burst_k4"]
    ),
    "decode_burst_k8_split_phase": dict(
        _split.POLICY_CONFIGS["decode_burst_k8_split_phase"]
    ),
    "decode_burst_k8_split_phase_ragged": {
        "enabled": True,
        "split": True,
        "ragged_coalescing": True,
        "width": 8,
        "selectable": True,
        "entrypoint": "production",
        "correctness_sampled_logit_d2h_calls": 3,
        "ordinary_tail_sampling_points": (),
    },
}
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/exact_greedy_decode_burst_split_phase.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/profile_exact_burst_split_phase.py",
    "tools/profile_exact_burst_ragged_coalescing.py",
    "tools/test_profile_exact_burst_ragged_coalescing.py",
    "tools/exact_burst_ragged_coalescing_gate.py",
    "tools/test_exact_burst_ragged_coalescing_gate.py",
    "tools/exact_burst_ragged_coalescing_verify.py",
    "tools/test_exact_burst_ragged_coalescing_verify.py",
    "tools/run_exact_burst_ragged_coalescing_remote.py",
    "tools/test_run_exact_burst_ragged_coalescing_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)
CONTEXT_CASES = _split.CONTEXT_CASES
SAMPLING_POINTS = _split.SAMPLING_POINTS


def correctness_trace_for_step(
    policy: str,
    *,
    emitted_total: int,
    generated_tokens: int,
) -> bool:
    if policy != "decode_burst_k8_split_phase_ragged":
        return _ORIGINAL_SPLIT_CORRECTNESS_TRACE_FOR_STEP(
            policy,
            emitted_total=emitted_total,
            generated_tokens=generated_tokens,
        )
    if emitted_total <= 0 or emitted_total >= generated_tokens:
        return False
    remaining = generated_tokens - emitted_total
    if remaining >= 8:
        width = 8
    elif remaining >= 2:
        width = min(4, remaining)
    else:
        width = 1
    sampled_indices = {
        _base._sampling_output_index(point, generated_tokens)
        for point in SAMPLING_POINTS[1:]
    }
    return any(
        emitted_total <= output_index < emitted_total + width
        for output_index in sampled_indices
    )


def _sampled_local_ordinals(policy: str) -> tuple[int, ...]:
    if policy == "decode_burst_k8_split_phase_ragged":
        return (0, 2, 7)
    return _ORIGINAL_SPLIT_SAMPLED_LOCAL_ORDINALS(policy)


def _expected_selected_replay_ordinal(
    policy: str,
    point: str,
    generated_tokens: int,
) -> int:
    if policy != "decode_burst_k8_split_phase_ragged":
        return _ORIGINAL_EXPECTED_SELECTED_REPLAY_ORDINAL(
            policy,
            point,
            generated_tokens,
        )
    output_index = _base._sampling_output_index(
        point,
        generated_tokens,
    )
    emitted_total = 1
    while emitted_total < generated_tokens:
        remaining = generated_tokens - emitted_total
        if remaining >= 8:
            width = 8
        elif remaining >= 2:
            width = min(4, remaining)
        else:
            width = 1
        if output_index < emitted_total + width:
            return output_index - emitted_total
        emitted_total += width
    raise ValueError("sampling point is outside ragged decode bursts")


def validate_case_row(row) -> dict:
    normalized = _split.validate_case_row(row)
    samples = normalized["amortized_tpot_samples_ns"]
    expected_tail = sum(samples[-7:])
    actual_tail = row.get(
        "tail_seven_elapsed_ns",
        expected_tail,
    )
    actual_tail = _base._require_finite_non_negative(
        actual_tail,
        "tail_seven_elapsed_ns",
    )
    if actual_tail != expected_tail:
        raise ValueError(
            "tail-seven latency does not match TPOT samples"
        )
    normalized["tail_seven_elapsed_ns"] = actual_tail
    return normalized


def _activate_contract() -> None:
    for module in (_split, _base):
        module.CASE_SCHEMA_VERSION = CASE_SCHEMA_VERSION
        module.CORRECTNESS_SCHEMA_VERSION = (
            CORRECTNESS_SCHEMA_VERSION
        )
        module.SUMMARY_SCHEMA_VERSION = SUMMARY_SCHEMA_VERSION
        module.WORKLOAD_SCHEMA_VERSION = WORKLOAD_SCHEMA_VERSION
        module.SOURCE_SCHEMA_VERSION = SOURCE_SCHEMA_VERSION
        module.CORRECTNESS_TRACE_IDENTITY = (
            CORRECTNESS_TRACE_IDENTITY
        )
        module.POLICIES = POLICIES
        module.POLICY_CONFIGS = POLICY_CONFIGS
        module.SOURCE_FILES = SOURCE_FILES
    _base._validate_burst_summary = (
        _split._validate_burst_summary
    )
    _base.validate_case_row = validate_case_row
    _base._counter_delta = _split._counter_delta
    _base._combined_summary = _split._combined_summary
    _base._construct_llm = _split._construct_llm
    _base._run_request = _split._run_request
    _base.correctness_trace_for_step = (
        correctness_trace_for_step
    )
    _base._expected_selected_replay_ordinal = (
        _expected_selected_replay_ordinal
    )
    _base._sampled_local_ordinals = _sampled_local_ordinals
    _base.run_correctness_probe = _split.run_correctness_probe
    _split.correctness_trace_for_step = (
        correctness_trace_for_step
    )
    _split._expected_selected_replay_ordinal = (
        _expected_selected_replay_ordinal
    )
    _split._sampled_local_ordinals = _sampled_local_ordinals


_activate_contract()

policy_order = _base.policy_order
performance_identities = _base.performance_identities
correctness_identities = _base.correctness_identities
correctness_uses_burst_trace = _base.correctness_uses_burst_trace
correctness_point_uses_burst_trace = (
    _base.correctness_point_uses_burst_trace
)
runtime_environment_manifest = _base.runtime_environment_manifest
validate_correctness_rows = _base.validate_correctness_rows
summarize_rows = _base.summarize_rows
_construct_llm = _split._construct_llm
read_float32_sidecar = _base.read_float32_sidecar
write_float32_sidecar = _base.write_float32_sidecar
sha256_file = _base.sha256_file
run_case = _base.run_case


def build_workload_manifest(**kwargs) -> dict:
    manifest = _base.build_workload_manifest(**kwargs)
    manifest["performance_row_count"] = 45
    manifest["correctness_row_count"] = 36
    return manifest


def source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    run_tag: str,
) -> dict:
    return _base._source_manifest(
        repo_root=repo_root,
        source_commit=source_commit,
        run_tag=run_tag,
    )


def main(argv=None) -> int:
    _activate_contract()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
