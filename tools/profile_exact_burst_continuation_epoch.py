#!/usr/bin/env python3
"""Source-bound four-arm benchmark for exact-burst continuation epochs."""

from __future__ import annotations

from pathlib import Path

from tools import profile_exact_greedy_decode_burst as _base


CASE_SCHEMA_VERSION = "exact-burst-continuation-epoch.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "exact-burst-continuation-epoch.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = (
    "exact-burst-continuation-epoch.summary.v1"
)
WORKLOAD_SCHEMA_VERSION = (
    "exact-burst-continuation-epoch.workload.v1"
)
SOURCE_SCHEMA_VERSION = (
    "exact-burst-continuation-epoch.source.v1"
)
CORRECTNESS_TRACE_IDENTITY = (
    "gate-only-exact-burst-continuation-correctness-v1"
)

POLICIES = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k4_continuation",
    "decode_burst_k8",
)
CONTEXT_CASES = _base.CONTEXT_CASES
SAMPLING_POINTS = _base.SAMPLING_POINTS
POLICY_CONFIGS = {
    "host_greedy": {
        "enabled": False,
        "continuation": False,
        "epoch_relative_sampling": False,
        "width": 1,
        "selectable": False,
        "entrypoint": "ordinary",
    },
    "decode_burst_k4": {
        "enabled": True,
        "continuation": False,
        "epoch_relative_sampling": False,
        "width": 4,
        "selectable": False,
        "entrypoint": "production",
    },
    "decode_burst_k4_continuation": {
        "enabled": True,
        "continuation": True,
        "epoch_relative_sampling": True,
        "width": 4,
        "selectable": True,
        "entrypoint": "production",
    },
    "decode_burst_k8": {
        "enabled": True,
        "continuation": False,
        "epoch_relative_sampling": False,
        "width": 8,
        "selectable": False,
        "entrypoint": "production",
    },
}

CONTINUATION_COUNTER_FIELDS = (
    "continuation_attempts",
    "continuation_hits",
    "cold_binds",
    "continuation_tokens",
    "continuation_bursts",
    "skipped_static_reset_operations",
    "skipped_scalar_bind_operations",
    "skipped_block_table_constructions",
    "skipped_block_table_copy_calls",
    "skipped_block_table_bytes",
)
CONTINUATION_MAP_FIELDS = (
    "continuation_miss_counts",
    "continuation_invalidation_counts",
)
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/profile_exact_burst_continuation_epoch.py",
    "tools/test_profile_exact_burst_continuation_epoch.py",
    "tools/exact_burst_continuation_epoch_gate.py",
    "tools/test_exact_burst_continuation_epoch_gate.py",
    "tools/exact_burst_continuation_epoch_verify.py",
    "tools/test_exact_burst_continuation_epoch_verify.py",
    "tools/run_exact_burst_continuation_epoch_remote.py",
    "tools/test_run_exact_burst_continuation_epoch_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)

_ORIGINAL_BURST_COUNTER_FIELDS = _base.BURST_COUNTER_FIELDS
_ORIGINAL_VALIDATE_BURST_SUMMARY = _base._validate_burst_summary
_ORIGINAL_COUNTER_DELTA = _base._counter_delta
_ORIGINAL_COMBINED_SUMMARY = _base._combined_summary


def _activate_contract() -> None:
    _base.CASE_SCHEMA_VERSION = CASE_SCHEMA_VERSION
    _base.CORRECTNESS_SCHEMA_VERSION = CORRECTNESS_SCHEMA_VERSION
    _base.SUMMARY_SCHEMA_VERSION = SUMMARY_SCHEMA_VERSION
    _base.WORKLOAD_SCHEMA_VERSION = WORKLOAD_SCHEMA_VERSION
    _base.SOURCE_SCHEMA_VERSION = SOURCE_SCHEMA_VERSION
    _base.CORRECTNESS_TRACE_IDENTITY = (
        CORRECTNESS_TRACE_IDENTITY
    )
    _base.POLICIES = POLICIES
    _base.POLICY_CONFIGS = POLICY_CONFIGS
    _base.SOURCE_FILES = SOURCE_FILES
    _base.BURST_COUNTER_FIELDS = (
        _ORIGINAL_BURST_COUNTER_FIELDS
        + CONTINUATION_COUNTER_FIELDS
    )


def _validate_reason_counts(value, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(reason, str)
        or not reason
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for reason, count in value.items()
    ):
        raise ValueError(f"{name} is invalid")
    return dict(sorted(value.items()))


def _validate_burst_summary(
    summary,
    *,
    policy: str,
    correctness_trace: bool,
) -> dict:
    required = set(CONTINUATION_COUNTER_FIELDS) | set(
        CONTINUATION_MAP_FIELDS
    )
    if not isinstance(summary, dict) or required - set(summary):
        raise ValueError(
            "exact burst continuation summary fields are missing"
        )
    normalized = _ORIGINAL_VALIDATE_BURST_SUMMARY(
        summary,
        policy=policy,
        correctness_trace=correctness_trace,
    )
    for field in CONTINUATION_COUNTER_FIELDS:
        normalized[field] = _base._require_non_negative_int(
            summary[field],
            f"exact burst continuation summary {field}",
        )
    for field in CONTINUATION_MAP_FIELDS:
        normalized[field] = _validate_reason_counts(
            summary[field],
            field.replace("_", " "),
        )

    continuation = POLICY_CONFIGS[policy]["continuation"]
    if not continuation:
        if any(
            normalized[field]
            for field in CONTINUATION_COUNTER_FIELDS
        ) or any(
            normalized[field] for field in CONTINUATION_MAP_FIELDS
        ):
            raise ValueError(
                "non-continuation policy reported continuation activity"
            )
        return normalized

    attempts = normalized["continuation_attempts"]
    hits = normalized["continuation_hits"]
    cold_binds = normalized["cold_binds"]
    if attempts != normalized["commits"]:
        raise ValueError(
            "continuation attempt inventory mismatch"
        )
    if hits + cold_binds != attempts:
        raise ValueError("continuation outcome inventory mismatch")
    if normalized["continuation_bursts"] != hits:
        raise ValueError("continuation burst inventory mismatch")
    if normalized["skipped_static_reset_operations"] != hits * 7:
        raise ValueError("continuation reset savings mismatch")
    if normalized["skipped_scalar_bind_operations"] != hits * 5:
        raise ValueError("continuation bind savings mismatch")
    if normalized["skipped_block_table_constructions"] != hits:
        raise ValueError(
            "continuation construction savings mismatch"
        )
    if normalized["skipped_block_table_copy_calls"] != hits:
        raise ValueError("continuation copy savings mismatch")
    if normalized["continuation_tokens"] < hits:
        raise ValueError("continuation token inventory mismatch")
    if normalized["continuation_invalidation_counts"]:
        raise ValueError(
            "continuation invalidation inventory is nonzero"
        )
    return normalized


def _counter_delta(before: dict, after: dict) -> dict:
    result = _ORIGINAL_COUNTER_DELTA(before, after)
    for field in CONTINUATION_MAP_FIELDS:
        before_map = before.get(field, {})
        after_map = after.get(field, {})
        result[field] = {}
        for key in sorted(set(before_map) | set(after_map)):
            difference = int(after_map.get(key, 0)) - int(
                before_map.get(key, 0)
            )
            if difference < 0:
                raise RuntimeError(
                    f"exact burst map counter decreased: {field}"
                )
            if difference:
                result[field][str(key)] = difference
    return result


def _combined_summary(
    llm,
    before: tuple[dict, dict],
    *,
    correctness_trace: bool = False,
) -> dict:
    result = _ORIGINAL_COMBINED_SUMMARY(
        llm,
        before,
        correctness_trace=correctness_trace,
    )
    runner = _counter_delta(
        before[0],
        llm.model_runner.exact_greedy_decode_burst_summary(),
    )
    for field in CONTINUATION_COUNTER_FIELDS:
        result[field] = runner[field]
    for field in CONTINUATION_MAP_FIELDS:
        result[field] = runner[field]
    return result


def _sampled_local_ordinals(policy: str) -> tuple[int, ...]:
    if not POLICY_CONFIGS[policy]["enabled"]:
        return ()
    decode_ordinals = (0, 63, 126)
    if POLICY_CONFIGS[policy]["continuation"]:
        return decode_ordinals
    width = POLICY_CONFIGS[policy]["width"]
    return tuple(sorted({ordinal % width for ordinal in decode_ordinals}))


_activate_contract()
_base._validate_burst_summary = _validate_burst_summary
_base._counter_delta = _counter_delta
_base._combined_summary = _combined_summary
_base._sampled_local_ordinals = _sampled_local_ordinals

policy_order = _base.policy_order
performance_identities = _base.performance_identities
correctness_identities = _base.correctness_identities
correctness_uses_burst_trace = _base.correctness_uses_burst_trace
correctness_trace_for_step = _base.correctness_trace_for_step
runtime_environment_manifest = _base.runtime_environment_manifest
build_workload_manifest = _base.build_workload_manifest
validate_case_row = _base.validate_case_row
validate_correctness_rows = _base.validate_correctness_rows
summarize_rows = _base.summarize_rows
read_float32_sidecar = _base.read_float32_sidecar
write_float32_sidecar = _base.write_float32_sidecar
run_case = _base.run_case
run_correctness_probe = _base.run_correctness_probe


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
