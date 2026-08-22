#!/usr/bin/env python3
"""Contracts for the graph-resident greedy-tail profile worker."""

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

from tools.profile_graph_resident_greedy_tail import (
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


def _greedy_summary(policy: str) -> dict:
    optimized_steps = {
        "legacy": 0,
        "host_greedy": 128,
        "graph_greedy": 1,
    }[policy]
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
            if policy == "legacy"
            else {}
        ),
    }


def _capture_receipt() -> dict:
    return {
        "source_identity": {
            "data_ptr": 4096,
            "shape": [1, 1024],
            "stride": [1024, 1],
            "storage_offset": 0,
            "dtype": "torch.float16",
            "device": "cuda:0",
        },
        "graph_generation": 1,
        "rank": 0,
        "capture_duration_ns": 1_000_000,
        "allocated_delta_bytes": 400_000,
        "reserved_delta_bytes": 2_000_000,
        "retained_logits_bytes": 300_000,
        "retained_float32_bytes": 600_000,
        "retained_token_bytes": 8,
        "retained_static_bytes": 900_008,
    }


def _graph_summary(policy: str) -> dict:
    optimized_steps = 127 if policy == "graph_greedy" else 0
    return {
        "eligible_steps": optimized_steps,
        "captured_graphs": 1 if policy == "graph_greedy" else 0,
        "replayed_steps": optimized_steps,
        "final_token_d2h_calls": optimized_steps,
        "avoided_external_compute_logits_calls": optimized_steps,
        "avoided_external_float32_conversions": optimized_steps,
        "avoided_external_argmax_calls": optimized_steps,
        "fallback_counts": {},
        "quarantine_reason": None,
        "capture_receipt": (
            _capture_receipt()
            if policy == "graph_greedy"
            else None
        ),
    }


def _case_row(
    policy: str,
    *,
    bucket: str = "short",
    repetition: int = 0,
) -> dict:
    case = {
        name: (prompt_tokens, generated_tokens)
        for name, prompt_tokens, generated_tokens in context_cases()
    }[bucket]
    prompt_tokens, generated_tokens = case
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "run_tag": "20260822-qwen3-06b-graph-greedy-tail-test",
        "source_commit": "a" * 40,
        "policy": policy,
        "repetition": repetition,
        "context_bucket": bucket,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "output_token_ids": list(range(generated_tokens)),
        "output_text_sha256": "b" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "tpot_samples_ns": [1_000_000] * 127,
        "decode_host_ns": [900_000] * 127,
        "decode_cuda_ns": [700_000] * 127,
        "output_tokens_per_second": 914.285714,
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "graph_capture_duration_ns": (
            1_000_000 if policy == "graph_greedy" else 0
        ),
        "graph_allocated_delta_bytes": (
            400_000 if policy == "graph_greedy" else 0
        ),
        "graph_reserved_delta_bytes": (
            2_000_000 if policy == "graph_greedy" else 0
        ),
        "graph_retained_static_bytes": (
            900_008 if policy == "graph_greedy" else 0
        ),
        "greedy_fast_path_summary": _greedy_summary(policy),
        "graph_resident_greedy_tail_summary": _graph_summary(policy),
    }


def _correctness_row(
    *,
    policy: str,
    bucket: str,
    sampling_point: str,
    sidecar: dict,
) -> dict:
    prompt_tokens, generated_tokens = {
        name: (prompt, generated)
        for name, prompt, generated in context_cases()
    }[bucket]
    return {
        "schema_version": CORRECTNESS_SCHEMA_VERSION,
        "run_tag": "20260822-qwen3-06b-graph-greedy-tail-test",
        "source_commit": "a" * 40,
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
        "greedy_fast_path_summary": _greedy_summary(policy),
        "graph_resident_greedy_tail_summary": _graph_summary(policy),
    }


def test_worker_contract_helpers_are_deterministic() -> None:
    assert context_cases() == (
        ("short", 256, 128),
        ("medium", 2048, 128),
        ("long", 8192, 128),
    )
    assert policy_order(0) == (
        "legacy",
        "host_greedy",
        "graph_greedy",
    )
    assert policy_order(1) == (
        "graph_greedy",
        "host_greedy",
        "legacy",
    )


def test_float32_sidecar_round_trip_and_integrity() -> None:
    values = (1.25, -2.5, 3.75, 0.0)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata = write_float32_sidecar(
            root,
            "logits/short-legacy-prefill-final.f32",
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


def test_case_rows_require_exact_outputs_and_tail_inventory() -> None:
    for policy in (
        "legacy",
        "host_greedy",
        "graph_greedy",
    ):
        validate_case_row(_case_row(policy))

    wrong_output = _case_row("graph_greedy")
    wrong_output["output_token_ids"].pop()
    with pytest.raises(ValueError, match="output token inventory"):
        validate_case_row(wrong_output)

    wrong_replay = _case_row("graph_greedy")
    wrong_replay[
        "graph_resident_greedy_tail_summary"
    ]["replayed_steps"] = 126
    with pytest.raises(ValueError, match="tail replay inventory"):
        validate_case_row(wrong_replay)

    wrong_d2h = _case_row("graph_greedy")
    wrong_d2h[
        "graph_resident_greedy_tail_summary"
    ]["final_token_d2h_calls"] = 126
    with pytest.raises(ValueError, match="token D2H inventory"):
        validate_case_row(wrong_d2h)


def test_summary_requires_exact_45_row_inventory() -> None:
    rows = [
        _case_row(
            policy,
            bucket=bucket,
            repetition=repetition,
        )
        for repetition in range(5)
        for bucket, _prompt_tokens, _generated_tokens in context_cases()
        for policy in (
            "legacy",
            "host_greedy",
            "graph_greedy",
        )
    ]
    summary = summarize_rows(rows, expected_repetitions=5)
    assert summary["row_count"] == 45
    assert summary["triple_count"] == 15
    assert summary["all_outputs_exact"] is True
    assert summary["all_graph_decode_steps_optimized"] is True

    with pytest.raises(ValueError, match="case row inventory"):
        summarize_rows(rows[:-1], expected_repetitions=5)

    duplicate = rows + [deepcopy(rows[0])]
    with pytest.raises(ValueError, match="duplicate case identity"):
        summarize_rows(duplicate, expected_repetitions=5)


def test_correctness_rows_require_exact_27_sidecars() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for bucket, _prompt_tokens, _generated_tokens in context_cases():
            for policy in (
                "legacy",
                "host_greedy",
                "graph_greedy",
            ):
                for point in (
                    "prefill-final",
                    "decode-first",
                    "decode-final",
                ):
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
        assert len(validated) == 27

        with pytest.raises(
            ValueError,
            match="correctness row inventory",
        ):
            validate_correctness_rows(
                rows[:-1],
                run_dir=root,
            )

        duplicate = rows + [deepcopy(rows[0])]
        with pytest.raises(
            ValueError,
            match="duplicate correctness identity",
        ):
            validate_correctness_rows(
                duplicate,
                run_dir=root,
            )


def main() -> None:
    test_worker_contract_helpers_are_deterministic()
    test_float32_sidecar_round_trip_and_integrity()
    test_sidecar_rejects_non_finite_values()
    test_case_rows_require_exact_outputs_and_tail_inventory()
    test_summary_requires_exact_45_row_inventory()
    test_correctness_rows_require_exact_27_sidecars()
    print("graph-resident greedy-tail profile tests passed")


if __name__ == "__main__":
    main()
