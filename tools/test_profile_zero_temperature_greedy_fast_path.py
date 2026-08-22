#!/usr/bin/env python3
"""Contracts for the zero-temperature greedy fast-path worker."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.profile_zero_temperature_greedy_fast_path import (
    CASE_SCHEMA_VERSION,
    CORRECTNESS_SCHEMA_VERSION,
    context_cases,
    policy_order,
    read_float32_sidecar,
    summarize_rows,
    validate_case_row,
    validate_correctness_rows,
    write_float32_sidecar,
)


def _summary(policy: str) -> dict:
    optimized_steps = 128 if policy == "on" else 0
    return {
        "eligible_steps": optimized_steps,
        "optimized_steps": optimized_steps,
        "avoided_temperature_h2d_bytes": 4 * optimized_steps,
        "avoided_softmax_calls": optimized_steps,
        "avoided_gumbel_rng_calls": optimized_steps,
        "avoided_stochastic_divisions": 2 * optimized_steps,
        "avoided_stochastic_argmax_calls": optimized_steps,
        "avoided_where_calls": optimized_steps,
        "fallback_counts": (
            {"disabled": 128}
            if policy == "off"
            else {}
        ),
    }


def _case_row(policy: str) -> dict:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "run_tag": "20260822-qwen3-06b-greedy-fast-test",
        "source_commit": "a" * 40,
        "policy": policy,
        "repetition": 0,
        "context_bucket": "short",
        "prompt_tokens": 256,
        "generated_tokens": 128,
        "output_token_ids": list(range(128)),
        "output_text_sha256": "b" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "tpot_samples_ns": [1_000_000] * 127,
        "decode_host_ns": [900_000] * 127,
        "decode_cuda_ns": [700_000] * 127,
        "output_tokens_per_second": 914.285714,
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "greedy_fast_path_summary": _summary(policy),
    }


def _correctness_row(
    *,
    policy: str,
    sampling_point: str,
    sidecar: dict,
) -> dict:
    return {
        "schema_version": CORRECTNESS_SCHEMA_VERSION,
        "run_tag": "20260822-qwen3-06b-greedy-fast-test",
        "source_commit": "a" * 40,
        "policy": policy,
        "context_bucket": "short",
        "prompt_tokens": 256,
        "generated_tokens": 128,
        "sampling_point": sampling_point,
        "output_token_ids": list(range(128)),
        "output_text_sha256": "b" * 64,
        "logits_path": sidecar["path"],
        "logits_shape": [1, 4],
        "logits_element_count": sidecar["element_count"],
        "logits_byte_length": sidecar["byte_length"],
        "logits_sha256": sidecar["sha256"],
        "greedy_fast_path_summary": _summary(policy),
    }


def test_worker_contract_helpers_are_deterministic() -> None:
    assert context_cases() == (
        ("short", 256, 128),
        ("medium", 2048, 128),
        ("long", 8192, 128),
    )
    assert policy_order(0) == ("off", "on")
    assert policy_order(1) == ("on", "off")


def test_float32_sidecar_round_trip_and_integrity() -> None:
    values = (1.25, -2.5, 3.75, 0.0)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata = write_float32_sidecar(
            root,
            "logits/short-off-prefill-final.f32",
            values,
        )
        restored = read_float32_sidecar(
            root,
            path=metadata["path"],
            expected_element_count=4,
            expected_byte_length=16,
            expected_sha256=metadata["sha256"],
        )
        assert restored == values

        with pytest.raises(ValueError, match="SHA256 mismatch"):
            read_float32_sidecar(
                root,
                path=metadata["path"],
                expected_element_count=4,
                expected_byte_length=16,
                expected_sha256="0" * 64,
            )
        with pytest.raises(ValueError, match="byte length mismatch"):
            read_float32_sidecar(
                root,
                path=metadata["path"],
                expected_element_count=4,
                expected_byte_length=12,
                expected_sha256=metadata["sha256"],
            )


def test_sidecar_rejects_non_finite_values() -> None:
    with TemporaryDirectory() as temporary:
        with pytest.raises(ValueError, match="finite"):
            write_float32_sidecar(
                Path(temporary),
                "logits/non-finite.f32",
                (1.0, math.nan),
            )


def test_case_rows_require_exact_outputs_and_128_on_steps() -> None:
    validate_case_row(_case_row("off"))
    validate_case_row(_case_row("on"))

    wrong_output = _case_row("on")
    wrong_output["output_token_ids"].pop()
    with pytest.raises(ValueError, match="output token inventory"):
        validate_case_row(wrong_output)

    incomplete = _case_row("on")
    incomplete["greedy_fast_path_summary"]["optimized_steps"] = 127
    with pytest.raises(
        ValueError,
        match="optimized sampling step inventory mismatch",
    ):
        validate_case_row(incomplete)


def test_summary_rejects_reused_case_identity() -> None:
    row = _case_row("off")
    with pytest.raises(ValueError, match="duplicate case identity"):
        summarize_rows([row, deepcopy(row)])


def test_correctness_rows_require_six_unique_exact_sidecars() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for policy in ("off", "on"):
            for point in (
                "prefill-final",
                "decode-first",
                "decode-final",
            ):
                values = (
                    (1.0, 2.0, 4.0, 3.0)
                    if policy == "off"
                    else (1.0, 2.0, 4.0, 3.0)
                )
                sidecar = write_float32_sidecar(
                    root,
                    f"logits/short-{policy}-{point}.f32",
                    values,
                )
                rows.append(
                    _correctness_row(
                        policy=policy,
                        sampling_point=point,
                        sidecar=sidecar,
                    )
                )

        validated = validate_correctness_rows(
            rows,
            run_dir=root,
            expected_buckets=("short",),
        )
        assert len(validated) == 6

        duplicate = rows + [deepcopy(rows[0])]
        with pytest.raises(
            ValueError,
            match="duplicate correctness identity",
        ):
            validate_correctness_rows(
                duplicate,
                run_dir=root,
                expected_buckets=("short",),
            )


def main() -> None:
    test_worker_contract_helpers_are_deterministic()
    test_float32_sidecar_round_trip_and_integrity()
    test_sidecar_rejects_non_finite_values()
    test_case_rows_require_exact_outputs_and_128_on_steps()
    test_summary_rejects_reused_case_identity()
    test_correctness_rows_require_six_unique_exact_sidecars()
    print("zero-temperature greedy profile tests passed")


if __name__ == "__main__":
    main()
