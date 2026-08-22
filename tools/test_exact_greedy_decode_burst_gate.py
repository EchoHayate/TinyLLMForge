#!/usr/bin/env python3
"""Producer-gate tests for exact greedy decode-burst evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
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

from tools.exact_greedy_decode_burst_gate import (
    classify,
    produce_gate,
)
from tools.profile_exact_greedy_decode_burst import (
    POLICY_CONFIGS,
    POLICIES,
    SAMPLING_POINTS,
    SOURCE_FILES,
    build_workload_manifest,
    context_cases,
    policy_order,
    summarize_rows,
    write_float32_sidecar,
)


RUN_TAG = "20260822-qwen3-06b-exact-burst-fixture"
SOURCE_COMMIT = "a" * 40
GRAPH_IDS = {
    policy: hashlib.sha256(policy.encode("utf-8")).hexdigest()
    for policy in POLICIES
    if policy != "host_greedy"
}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _capture_receipt(
    policy: str,
    *,
    correctness_trace: bool,
) -> dict:
    return {
        "graph_identity_sha256": GRAPH_IDS[policy],
        "graph_generation": 1,
        "capture_duration_ns": 1_000_000,
        "allocated_delta_bytes": 400_000,
        "reserved_delta_bytes": 2_000_000,
        "retained_static_bytes": 900_000,
        "scratch_block_count": 1,
        "correctness_trace": correctness_trace,
    }


def _burst_summary(
    policy: str,
    *,
    correctness_trace: bool = False,
    maximum_gap_ns: int = 0,
) -> dict:
    config = POLICY_CONFIGS[policy]
    if not config["enabled"]:
        return {
            "attempts": 0,
            "acceptances": 0,
            "target_model_forwards": 0,
            "graph_replays": 0,
            "intermediate_token_d2h_calls": 0,
            "final_token_d2h_calls": 0,
            "final_token_d2h_bytes": 0,
            "sampled_logit_d2h_calls": 0,
            "output_budget_clipped": 0,
            "block_boundary_clipped": 0,
            "commits": 0,
            "committed_tokens": 0,
            "failures": 0,
            "quarantines": 0,
            "pending_leases": 0,
            "maximum_host_visible_gap_ns": maximum_gap_ns,
            "requested_width_histogram": {},
            "authorized_width_histogram": {},
            "fallback_counts": {},
            "quarantine_reason": None,
            "capture_receipts": [],
        }
    width = config["width"]
    commits = math.ceil(127 / width)
    partial = 127 % width
    authorized = {str(width): commits}
    if partial:
        authorized[str(width)] -= 1
        authorized[str(partial)] = 1
    return {
        "attempts": commits,
        "acceptances": commits,
        "target_model_forwards": 127,
        "graph_replays": 127,
        "intermediate_token_d2h_calls": 0,
        "final_token_d2h_calls": commits,
        "final_token_d2h_bytes": 127 * 8,
        "sampled_logit_d2h_calls": (
            3 if correctness_trace else 0
        ),
        "output_budget_clipped": 0 if width == 1 else 1,
        "block_boundary_clipped": 0,
        "commits": commits,
        "committed_tokens": 127,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
        "maximum_host_visible_gap_ns": maximum_gap_ns,
        "requested_width_histogram": {str(width): commits},
        "authorized_width_histogram": authorized,
        "fallback_counts": {},
        "quarantine_reason": None,
        "capture_receipts": [
            _capture_receipt(
                policy,
                correctness_trace=correctness_trace,
            )
        ],
    }


def make_fixture_rows(
    *,
    policy_scales: dict[str, float] | None = None,
) -> list[dict]:
    scales = {
        "host_greedy": 1.0,
        "full_step_graph_k1": 0.96,
        "decode_burst_k4": 0.86,
        "decode_burst_k8": 0.84,
    }
    if policy_scales is not None:
        scales.update(policy_scales)
    rows = []
    for repetition in range(5):
        for bucket, prompt_tokens, generated_tokens in context_cases():
            base_tpot = {
                "short": 1_000_000,
                "medium": 1_200_000,
                "long": 1_400_000,
            }[bucket]
            output_ids = [
                10 + ((index + repetition) % 100)
                for index in range(generated_tokens)
            ]
            output_digest = hashlib.sha256(
                json.dumps(output_ids).encode("utf-8")
            ).hexdigest()
            for policy in POLICIES:
                scale = scales[policy]
                tpot = round(base_tpot * scale)
                config = POLICY_CONFIGS[policy]
                gap = (
                    0
                    if not config["enabled"]
                    else min(35_000_000, tpot * config["width"])
                )
                summary = _burst_summary(
                    policy,
                    maximum_gap_ns=gap,
                )
                receipt = (
                    summary["capture_receipts"][0]
                    if summary["capture_receipts"]
                    else None
                )
                e2e = 20_000_000 + tpot * 127
                rows.append({
                    "schema_version":
                        "exact-greedy-decode-burst.case.v1",
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "selectable": config["selectable"],
                    "burst_width": config["width"],
                    "repetition": repetition,
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "output_token_ids": list(output_ids),
                    "output_text_sha256": output_digest,
                    "ttft_ns": 20_000_000,
                    "e2e_ns": e2e,
                    "amortized_tpot_samples_ns": [tpot] * 127,
                    "amortized_tpot_median_ns": tpot,
                    "amortized_tpot_p95_ns": tpot,
                    "amortized_tpot_p99_ns": tpot,
                    "decode_host_ns": [round(tpot * 0.85)] * (
                        summary["commits"]
                        if config["enabled"]
                        else 127
                    ),
                    "decode_cuda_ns": [round(tpot * 0.70)] * (
                        summary["commits"]
                        if config["enabled"]
                        else 127
                    ),
                    "output_tokens_per_second":
                        generated_tokens * 1_000_000_000 / e2e,
                    "host_visible_burst_gaps_ns": (
                        [gap] * summary["commits"]
                        if gap
                        else []
                    ),
                    "maximum_host_visible_burst_gap_ns": gap,
                    "cuda_peak_allocated_bytes":
                        1_000_000_000,
                    "cuda_peak_reserved_bytes": (
                        1_020_000_000
                        if config["enabled"]
                        else 1_000_000_000
                    ),
                    "capture_duration_ns": (
                        receipt["capture_duration_ns"]
                        if receipt else 0
                    ),
                    "capture_allocated_delta_bytes": (
                        receipt["allocated_delta_bytes"]
                        if receipt else 0
                    ),
                    "capture_reserved_delta_bytes": (
                        receipt["reserved_delta_bytes"]
                        if receipt else 0
                    ),
                    "capture_retained_static_bytes": (
                        receipt["retained_static_bytes"]
                        if receipt else 0
                    ),
                    "reserved_scratch_blocks": (
                        receipt["scratch_block_count"]
                        if receipt else 0
                    ),
                    "correctness_trace": False,
                    "exact_greedy_decode_burst_summary": summary,
                })
    return rows


def _sampling_output_index(point: str) -> int:
    return {
        "prefill-final": 0,
        "decode-first": 1,
        "decode-middle": 64,
        "decode-final": 127,
    }[point]


def _write_correctness_rows(
    run_dir: Path,
    *,
    mutation=None,
) -> list[dict]:
    rows = []
    for bucket, prompt_tokens, generated_tokens in context_cases():
        output_ids = list(range(generated_tokens))
        output_digest = hashlib.sha256(
            json.dumps(output_ids).encode("utf-8")
        ).hexdigest()
        for policy in POLICIES:
            summary = _burst_summary(
                policy,
                correctness_trace=POLICY_CONFIGS[policy]["enabled"],
            )
            for point in SAMPLING_POINTS:
                values = [1.0, 2.0, 5.0, 3.0]
                if mutation is not None:
                    values = mutation(
                        bucket,
                        policy,
                        point,
                        list(values),
                    )
                sidecar = write_float32_sidecar(
                    run_dir,
                    f"logits/{bucket}-{policy}-{point}.f32",
                    values,
                )
                burst_sample = (
                    POLICY_CONFIGS[policy]["enabled"]
                    and point != "prefill-final"
                )
                rows.append({
                    "schema_version":
                        "exact-greedy-decode-burst.correctness.v1",
                    "run_tag": RUN_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "policy": policy,
                    "context_bucket": bucket,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "sampling_point": point,
                    "output_token_ids": list(output_ids),
                    "output_text_sha256": output_digest,
                    "logits_path": sidecar["path"],
                    "logits_shape": [1, 4],
                    "logits_element_count": sidecar["element_count"],
                    "logits_byte_length": sidecar["byte_length"],
                    "logits_sha256": sidecar["sha256"],
                    "correctness_trace": True,
                    "trace_identity":
                        "gate-only-exact-burst-correctness-v1",
                    "trace_graph_identity_sha256": (
                        GRAPH_IDS[policy] if burst_sample else None
                    ),
                    "selected_replay_ordinal": (
                        (_sampling_output_index(point) - 1)
                        % POLICY_CONFIGS[policy]["width"]
                        if burst_sample
                        else None
                    ),
                    "sampled_logit_d2h_calls": (
                        1 if burst_sample else 0
                    ),
                    "exact_greedy_decode_burst_summary": summary,
                })
    return rows


def _load_correctness_rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (
            run_dir / "correctness_rows.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]


def write_fixture_bundle(
    root: Path,
    *,
    performance_rows: list[dict] | None = None,
    correctness_mutation=None,
) -> tuple[Path, Path]:
    rows = (
        make_fixture_rows()
        if performance_rows is None
        else performance_rows
    )
    run_dir = root / "run"
    repo_root = root / "repo"
    run_dir.mkdir()
    for relative in SOURCE_FILES:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"fixture source for {relative}\n",
            encoding="utf-8",
        )
    correctness_rows = _write_correctness_rows(
        run_dir,
        mutation=correctness_mutation,
    )
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version":
                "exact-greedy-decode-burst.source.v1",
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": {
                relative: hashlib.sha256(
                    (repo_root / relative).read_bytes()
                ).hexdigest()
                for relative in SOURCE_FILES
            },
        },
    )
    _write_json(
        run_dir / "workload_manifest.json",
        build_workload_manifest(
            model="/models/Qwen3-0.6B",
            run_tag=RUN_TAG,
            source_commit=SOURCE_COMMIT,
            gpu_memory_utilization=0.5,
            environment={
                "python_version": "3.12.0",
                "platform": "fixture",
                "python_executable": "/python",
                "torch_available": True,
                "torch_version": "2.8.0",
                "cuda_runtime_version": "12.8",
                "cuda_available": True,
                "cuda_device_name": "fixture-gpu",
            },
        ),
    )
    _write_jsonl(run_dir / "case_rows.jsonl", rows)
    _write_jsonl(
        run_dir / "correctness_rows.jsonl",
        correctness_rows,
    )
    summary = summarize_rows(rows, expected_repetitions=5)
    summary["correctness_row_count"] = 48
    _write_json(run_dir / "summary.json", summary)
    return run_dir, repo_root


def _classification(
    run_dir: Path,
    rows: list[dict],
) -> str:
    return classify(
        rows,
        _load_correctness_rows(run_dir),
        run_dir=run_dir,
    )["classification"]


def _set_tpot(row: dict, value: int) -> None:
    row["amortized_tpot_samples_ns"] = [value] * 127
    row["amortized_tpot_median_ns"] = value
    row["amortized_tpot_p95_ns"] = value
    row["amortized_tpot_p99_ns"] = value


def _set_tpot_samples(row: dict, values: list[int]) -> None:
    ordered = sorted(values)
    row["amortized_tpot_samples_ns"] = values
    row["amortized_tpot_median_ns"] = statistics.median(values)
    row["amortized_tpot_p95_ns"] = ordered[
        math.ceil(0.95 * len(ordered)) - 1
    ]
    row["amortized_tpot_p99_ns"] = ordered[
        math.ceil(0.99 * len(ordered)) - 1
    ]


def test_classify_selects_best_eligible_arm_and_breaks_ties_to_k4() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        result = classify(
            make_fixture_rows(),
            _load_correctness_rows(run_dir),
            run_dir=run_dir,
        )
        assert result["classification"] == (
            "GO_EXACT_GREEDY_DECODE_BURST"
        )
        assert result["selected_policy"] == "decode_burst_k8"
        assert result["correctness"]["max_abs"] == 0.0
        assert result["selected_lifecycle_complete"] is True

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows(
            policy_scales={
                "decode_burst_k4": 0.84,
                "decode_burst_k8": 0.84,
            }
        )
        result = classify(
            rows,
            _load_correctness_rows(run_dir),
            run_dir=run_dir,
        )
        assert result["selected_policy"] == "decode_burst_k4"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(
            Path(temporary),
            correctness_mutation=lambda bucket, policy, point, values: (
                [1.0, 2.0, 5.3, 3.0]
                if policy == "decode_burst_k8"
                and point == "decode-middle"
                else values
            ),
        )
        result = classify(
            make_fixture_rows(),
            _load_correctness_rows(run_dir),
            run_dir=run_dir,
        )
        assert result["classification"] == (
            "GO_EXACT_GREEDY_DECODE_BURST"
        )
        assert result["selected_policy"] == "decode_burst_k4"


def test_classify_uses_fixed_failure_precedence() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(
            Path(temporary),
            correctness_mutation=lambda bucket, policy, point, values: (
                [1.0, 2.0, 5.3, 3.0]
                if policy in {"decode_burst_k4", "decode_burst_k8"}
                and point == "decode-middle"
                else values
            ),
        )
        assert _classification(
            run_dir,
            make_fixture_rows(),
        ) == "NO_GO_CORRECTNESS"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] == "full_step_graph_k1":
                row["exact_greedy_decode_burst_summary"][
                    "graph_replays"
                ] = 126
        assert _classification(
            run_dir,
            rows,
        ) == "NO_GO_REPLAY_INCOMPLETE"

    lifecycle_mutations = (
        (
            "graph_replays",
            126,
            "NO_GO_REPLAY_INCOMPLETE",
        ),
        (
            "intermediate_token_d2h_calls",
            1,
            "NO_GO_D2H_LIFECYCLE",
        ),
        (
            "final_token_d2h_calls",
            0,
            "NO_GO_D2H_LIFECYCLE",
        ),
        (
            "final_token_d2h_bytes",
            8,
            "NO_GO_D2H_LIFECYCLE",
        ),
        (
            "pending_leases",
            1,
            "NO_GO_LEASE_LIFECYCLE",
        ),
        (
            "commits",
            0,
            "NO_GO_REPLAY_INCOMPLETE",
        ),
        (
            "failures",
            1,
            "NO_GO_EXECUTION_FAILURE",
        ),
        (
            "quarantines",
            1,
            "NO_GO_EXECUTION_FAILURE",
        ),
    )
    for field, value, expected in lifecycle_mutations:
        with TemporaryDirectory() as temporary:
            run_dir, _repo = write_fixture_bundle(Path(temporary))
            rows = make_fixture_rows()
            for target in rows:
                if target["policy"] in {
                    "decode_burst_k4",
                    "decode_burst_k8",
                }:
                    target[
                        "exact_greedy_decode_burst_summary"
                    ][field] = value
            assert _classification(run_dir, rows) == expected

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if row["policy"] in {
                "decode_burst_k4",
                "decode_burst_k8",
            }:
                row["exact_greedy_decode_burst_summary"][
                    "authorized_width_histogram"
                ] = {
                    str(row["burst_width"]): 1,
                }
        assert _classification(
            run_dir,
            rows,
        ) == "NO_GO_REPLAY_INCOMPLETE"

    threshold_cases = (
        (
            {"decode_burst_k4": 0.91, "decode_burst_k8": 0.901},
            "NO_GO_HOST_TPOT_MEDIAN",
        ),
        (
            {"decode_burst_k4": 0.94, "decode_burst_k8": 0.94},
            "NO_GO_HOST_TPOT_MEDIAN",
        ),
        (
            {
                "full_step_graph_k1": 0.89,
                "decode_burst_k4": 0.86,
                "decode_burst_k8": 0.85,
            },
            "NO_GO_K1_INCREMENTAL",
        ),
    )
    for scales, expected in threshold_cases:
        with TemporaryDirectory() as temporary:
            run_dir, _repo = write_fixture_bundle(Path(temporary))
            assert _classification(
                run_dir,
                make_fixture_rows(policy_scales=scales),
            ) == expected


def test_classify_distinguishes_p95_and_bucket_coverage() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            if (
                row["policy"] in {
                    "decode_burst_k4",
                    "decode_burst_k8",
                }
                and row["context_bucket"] == "long"
            ):
                _set_tpot(row, 1_302_000)
        assert _classification(
            run_dir,
            rows,
        ) == "NO_GO_HOST_TPOT_P95"

    with TemporaryDirectory() as temporary:
        run_dir, _repo = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        for row in rows:
            bucket = row["context_bucket"]
            base = {
                "short": 1_000_000,
                "medium": 1_200_000,
                "long": 1_400_000,
            }[bucket]
            if row["policy"] in {
                "decode_burst_k4",
                "decode_burst_k8",
            }:
                scale = 0.80 if bucket == "medium" else 0.93
                _set_tpot(row, round(base * scale))
            elif (
                row["policy"] == "host_greedy"
                and bucket == "long"
            ):
                _set_tpot_samples(
                    row,
                    [base] * 102 + [round(base * 1.10)] * 25,
                )
        assert _classification(
            run_dir,
            rows,
        ) == "NO_GO_BUCKET_COVERAGE"


def test_classify_reports_each_protected_cost_failure() -> None:
    mutations = (
        (
            lambda row: _set_tpot(row, 1_031_000),
            "NO_GO_BUCKET_REGRESSION",
            "short",
        ),
        (
            lambda row: row.__setitem__("ttft_ns", 20_600_001),
            "NO_GO_TTFT_E2E",
            None,
        ),
        (
            lambda row: row.__setitem__(
                "output_tokens_per_second",
                1.0,
            ),
            "NO_GO_THROUGHPUT",
            None,
        ),
        (
            lambda row: row.__setitem__(
                "cuda_peak_reserved_bytes",
                1_030_000_001,
            ),
            "NO_GO_MEMORY",
            None,
        ),
        (
            lambda row: (
                row.__setitem__(
                    "host_visible_burst_gaps_ns",
                    [40_000_001]
                    * row[
                        "exact_greedy_decode_burst_summary"
                    ]["commits"],
                ),
                row.__setitem__(
                    "maximum_host_visible_burst_gap_ns",
                    40_000_001,
                ),
                row["exact_greedy_decode_burst_summary"].__setitem__(
                    "maximum_host_visible_gap_ns",
                    40_000_001,
                ),
            ),
            "NO_GO_VISIBILITY_GAP",
            None,
        ),
        (
            lambda row: (
                row.__setitem__("capture_duration_ns", 0),
                row["exact_greedy_decode_burst_summary"][
                    "capture_receipts"
                ][0].__setitem__("capture_duration_ns", 0),
            ),
            "NO_GO_COST_INCOMPLETE",
            None,
        ),
    )
    for mutate, expected, bucket in mutations:
        with TemporaryDirectory() as temporary:
            run_dir, _repo = write_fixture_bundle(Path(temporary))
            rows = make_fixture_rows()
            for row in rows:
                if (
                    row["policy"] in {
                        "decode_burst_k4",
                        "decode_burst_k8",
                    }
                    and (bucket is None or row["context_bucket"] == bucket)
                ):
                    mutate(row)
            assert _classification(run_dir, rows) == expected


def test_producer_writes_complete_manifest_and_rejects_bad_evidence() -> None:
    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(Path(temporary))
        gate = produce_gate(run_dir, repo_root=repo_root)
        assert gate["classification"] == (
            "GO_EXACT_GREEDY_DECODE_BURST"
        )
        assert gate["selected_policy"] == "decode_burst_k8"
        manifest = json.loads(
            (run_dir / "manifest.sha256").read_text(
                encoding="utf-8"
            )
        )
        assert len({
            name
            for name in manifest["artifacts"]
            if name.startswith("logits/")
        }) == 48
        assert {
            "case_rows.jsonl",
            "correctness_rows.jsonl",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
            "comparison.json",
            "gate.json",
        } <= set(manifest["artifacts"])

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()[:-1]
        _write_jsonl(run_dir / "case_rows.jsonl", rows)
        with pytest.raises(ValueError, match="60 measured rows"):
            produce_gate(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(Path(temporary))
        source = json.loads(
            (run_dir / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        source["source_sha256"][SOURCE_FILES[0]] = "0" * 64
        _write_json(run_dir / "source_manifest.json", source)
        with pytest.raises(ValueError, match="source digest mismatch"):
            produce_gate(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(Path(temporary))
        workload = json.loads(
            (run_dir / "workload_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        workload["warmup_repetitions"] = 1
        _write_json(run_dir / "workload_manifest.json", workload)
        with pytest.raises(ValueError, match="workload manifest mismatch"):
            produce_gate(run_dir, repo_root=repo_root)

    for field, value in (
        ("model", "/models/not-the-stage1-model"),
        (
            "environment",
            {
                "python_version": "3.12.0",
                "platform": "fixture",
                "python_executable": "/python",
                "torch_available": True,
                "torch_version": "2.8.0",
                "cuda_runtime_version": "12.8",
                "cuda_available": False,
                "cuda_device_name": None,
            },
        ),
    ):
        with TemporaryDirectory() as temporary:
            run_dir, repo_root = write_fixture_bundle(Path(temporary))
            workload_path = run_dir / "workload_manifest.json"
            workload = json.loads(
                workload_path.read_text(encoding="utf-8")
            )
            workload[field] = value
            _write_json(workload_path, workload)
            with pytest.raises(
                ValueError,
                match="workload manifest mismatch",
            ):
                produce_gate(run_dir, repo_root=repo_root)

    with TemporaryDirectory() as temporary:
        run_dir, repo_root = write_fixture_bundle(Path(temporary))
        rows = make_fixture_rows()
        rows[0]["ttft_ns"] = math.nan
        raw = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        )
        (run_dir / "case_rows.jsonl").write_text(
            raw,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-finite"):
            produce_gate(run_dir, repo_root=repo_root)


def main() -> None:
    test_classify_selects_best_eligible_arm_and_breaks_ties_to_k4()
    test_classify_uses_fixed_failure_precedence()
    test_classify_distinguishes_p95_and_bucket_coverage()
    test_classify_reports_each_protected_cost_failure()
    test_producer_writes_complete_manifest_and_rejects_bad_evidence()
    print("exact greedy decode-burst gate tests passed")


if __name__ == "__main__":
    main()
