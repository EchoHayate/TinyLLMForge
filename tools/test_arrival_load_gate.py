"""Dependency-light tests for the production arrival-load gate."""

from __future__ import annotations

from collections import Counter
import importlib.util
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = REPO_ROOT / "tools" / "arrival_load_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "arrival_load_gate",
        GATE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arrival_load_gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


class FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return list(range(len(prompt.split())))


def _prompt_bank() -> dict:
    prompts = []
    for prompt_class, token_count in (
        ("short", 64),
        ("medium", 512),
        ("long", 1536),
    ):
        token_ids = list(range(token_count))
        prompt = " ".join(f"t{token_id}" for token_id in token_ids)
        prompts.append({
            "prompt_id": f"{prompt_class}-0",
            "prompt": prompt,
            "prompt_token_ids": token_ids,
            "prompt_token_count": token_count,
            "prompt_class": prompt_class,
            "prompt_sha256": gate.canonical_json_sha256({
                "prompt": prompt,
                "prompt_token_ids": token_ids,
            }),
        })
    bank = {
        "schema_version": gate.SCHEMA_VERSION,
        "model_id": "fake-model",
        "prompts": sorted(
            prompts,
            key=lambda record: record["prompt_id"],
        ),
    }
    bank["prompt_bank_sha256"] = gate.canonical_json_sha256(bank)
    return bank


def test_seeded_steady_and_burst_workloads_are_byte_stable():
    first = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    second = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    assert first == second
    assert gate.canonical_json_sha256(first) == (
        gate.canonical_json_sha256(second)
    )
    scenario_order = {
        name: index
        for index, name in enumerate(gate.CANONICAL_SCENARIOS)
    }
    assert first == sorted(
        first,
        key=lambda row: (
            scenario_order[row["scenario"]],
            row["arrival_offset_ns"],
            row["request_id"],
        ),
    )
    burst = [
        row for row in first if row["scenario"] == "burst"
    ]
    assert len(burst) == 64 + gate.CANONICAL_WARMUP_REQUESTS
    assert max(row["arrival_offset_ns"] for row in burst) > 6_000_000_000


def test_built_prompt_bank_is_sorted_hashed_and_valid():
    bank = gate.build_prompt_bank(
        FakeTokenizer(),
        model_id="fake-model",
    )
    gate.validate_prompt_bank(bank)
    assert [row["prompt_id"] for row in bank["prompts"]] == sorted(
        row["prompt_id"] for row in bank["prompts"]
    )
    assert {
        row["prompt_token_count"]
        for row in bank["prompts"]
    } == {64, 512, 1536}


def test_service_buckets_are_fixed_before_execution():
    rows = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    fairness = [
        row for row in rows
        if row["scenario"] == "mixed_service_fairness"
        and not row["warmup"]
    ]
    counts = Counter(row["service_time_bucket"] for row in fairness)
    assert set(counts) == {
        "short__short",
        "short__long",
        "medium__short",
        "medium__long",
        "long__short",
        "long__long",
    }
    assert set(counts.values()) == {
        gate.FAIRNESS_REQUESTS_PER_BUCKET
    }


def test_policy_identity_aliases_explicit_default_only():
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: gate.resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    aliases = gate.deduplicate_policies(resolved)
    assert aliases["canonical_policy_by_name"] == {
        "P0": "P0",
        "P1": "P0",
        "P2": "P2",
        "P3": "P3",
    }
    assert len(set(aliases["identity_by_name"].values())) == 3


def test_nearest_rank_boundaries():
    assert gate.nearest_rank([7.0], 0.50) == 7.0
    assert gate.nearest_rank([1.0, 9.0], 0.50) == 1.0
    assert gate.nearest_rank([1.0, 9.0], 0.95) == 9.0
    values = list(range(1, 21))
    assert gate.nearest_rank(values, 0.50) == 10
    assert gate.nearest_rank(values, 0.95) == 19
    assert gate.nearest_rank(values, 0.99) == 20


def test_canonical_rates_and_sampling_contract_are_frozen():
    rows = gate.build_canonical_workload(
        lambda_ref=10.0,
        prompt_bank=_prompt_bank(),
    )
    rates = {}
    for row in rows:
        rates.setdefault(row["scenario"], row["requested_rate_rps"])
        assert row["sampling"] == {
            "temperature": 0.0,
            "ignore_eos": True,
            "max_tokens": row["requested_output_tokens"],
        }
    assert rates["steady_moderate"] == 6.0
    assert rates["near_saturation"] == 9.0
    assert rates["overload"] == 12.0


def test_invalid_lambda_and_policy_fail_closed():
    for invalid in (-1.0, 0.0, float("inf"), float("nan")):
        try:
            gate.build_canonical_workload(
                lambda_ref=invalid,
                prompt_bank=_prompt_bank(),
            )
        except ValueError as exc:
            assert "lambda_ref" in str(exc)
        else:
            raise AssertionError(f"invalid lambda_ref accepted: {invalid}")

    try:
        gate.resolve_policy_config("P9", {})
    except ValueError as exc:
        assert "unknown policy" in str(exc)
    else:
        raise AssertionError("unknown policy accepted")


def test_unexpected_candidate_policy_collision_is_rejected():
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: gate.resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    resolved["P3"] = dict(resolved["P2"])
    try:
        gate.deduplicate_policies(resolved)
    except ValueError as exc:
        assert "unexpected policy identity collision" in str(exc)
    else:
        raise AssertionError("candidate collision must fail")


def test_prompt_bank_hash_detects_drift():
    bank = _prompt_bank()
    gate.validate_prompt_bank(bank)
    bank["prompts"][0]["prompt"] += " changed"
    try:
        gate.validate_prompt_bank(bank)
    except ValueError as exc:
        assert "prompt hash mismatch" in str(exc)
    else:
        raise AssertionError("prompt drift must fail")


def test_calibration_manifest_is_deterministic_and_p0_only():
    first = gate.build_calibration_manifest(_prompt_bank())
    second = gate.build_calibration_manifest(_prompt_bank())
    assert first == second
    assert all(row["policy"] == "P0" for row in first)
    rates = [row["requested_rate_rps"] for row in first]
    assert rates == sorted(rates)
    assert len(rates) == gate.CALIBRATION_MAX_DOUBLINGS + 1
    assert math.isclose(
        rates[-1],
        gate.CALIBRATION_INITIAL_RATE_RPS
        * (2 ** gate.CALIBRATION_MAX_DOUBLINGS),
    )


def _workload_row(
    request_id: str,
    *,
    output_tokens: int = 3,
    bucket: str = "short__short",
) -> dict:
    prompt_class, output_class = bucket.split("__")
    return {
        "request_id": request_id,
        "warmup": False,
        "scenario": "steady_moderate",
        "prompt_token_count": 4,
        "requested_output_tokens": output_tokens,
        "prompt_class": prompt_class,
        "output_class": output_class,
        "service_time_bucket": bucket,
    }


def _timeline_row(
    request_id: str,
    token_timestamps_ns: list[int],
    *,
    seq_id: int = 7,
    scheduled_arrival_ns: int = 100,
    actual_arrival_ns: int = 120,
    first_scheduled_ns: int = 150,
    completion_ns: int | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "seq_id": seq_id,
        "scheduled_arrival_ns": scheduled_arrival_ns,
        "actual_arrival_ns": actual_arrival_ns,
        "first_scheduled_ns": first_scheduled_ns,
        "first_token_ns": token_timestamps_ns[0],
        "token_timestamps_ns": token_timestamps_ns,
        "completion_ns": (
            token_timestamps_ns[-1]
            if completion_ns is None
            else completion_ns
        ),
        "output_token_ids": list(range(len(token_timestamps_ns))),
        "finish_reason": "length",
        "error": None,
    }


def test_reconstructs_scheduled_arrival_metrics_and_shared_step_tokens():
    metrics = gate.reconstruct_request_metrics(
        [_workload_row("r0")],
        [{
            **_timeline_row("r0", [200, 260, 260]),
            "output_token_ids": [11, 12, 13],
        }],
        [],
    )
    assert len(metrics) == 1
    assert metrics[0]["injection_lag_ns"] == 20
    assert metrics[0]["queue_delay_ns"] == 30
    assert metrics[0]["ttft_ns"] == 100
    assert metrics[0]["e2e_ns"] == 160
    assert metrics[0]["itl_ns"] == [60, 0]
    assert metrics[0]["maximum_decode_gap_ns"] == 60


def test_one_token_output_has_no_itl_sample():
    metrics = gate.reconstruct_request_metrics(
        [_workload_row("r0", output_tokens=1)],
        [_timeline_row("r0", [300])],
        [],
    )
    assert metrics[0]["itl_ns"] == []
    assert metrics[0]["maximum_decode_gap_ns"] is None


def test_lifecycle_reconstruction_rejects_duplicate_binding_and_bad_time():
    workload = [
        _workload_row("r0", output_tokens=1),
        _workload_row("r1", output_tokens=1),
    ]
    duplicate = [
        _timeline_row("r0", [300], seq_id=7),
        _timeline_row("r1", [310], seq_id=7),
    ]
    try:
        gate.reconstruct_request_metrics(workload, duplicate, [])
    except ValueError as exc:
        assert "duplicate sequence binding" in str(exc)
    else:
        raise AssertionError("duplicate sequence binding must fail")

    bad_time = [{
        **_timeline_row("r0", [300], seq_id=7),
        "actual_arrival_ns": 90,
    }]
    try:
        gate.reconstruct_request_metrics(
            [_workload_row("r0", output_tokens=1)],
            bad_time,
            [],
        )
    except ValueError as exc:
        assert "timestamp ordering" in str(exc)
    else:
        raise AssertionError("invalid timestamps must fail")


def test_repetition_summary_reports_percentiles_fairness_and_memory():
    workload = [
        _workload_row("r0", bucket="short__short"),
        _workload_row("r1", bucket="long__long"),
    ]
    timeline = [
        _timeline_row("r0", [200, 250, 300], seq_id=7),
        _timeline_row(
            "r1",
            [400, 500, 600],
            seq_id=8,
            scheduled_arrival_ns=110,
            actual_arrival_ns=130,
            first_scheduled_ns=170,
        ),
    ]
    metrics = gate.reconstruct_request_metrics(
        workload,
        timeline,
        [],
    )
    summary = gate.summarize_repetition(
        {
            "policy": "P0",
            "scenario": "steady_moderate",
            "repetition": 0,
            "measurement_start_ns": 100,
            "measurement_end_ns": 600,
            "required_service_buckets": [
                "short__short",
                "long__long",
            ],
        },
        metrics,
        [{
            "cuda_allocated_bytes": 100,
            "cuda_reserved_bytes": 200,
            "used_kv_blocks": 3,
            "kv_block_bytes": 64,
        }, {
            "cuda_allocated_bytes": 150,
            "cuda_reserved_bytes": 240,
            "used_kv_blocks": 4,
            "kv_block_bytes": 64,
        }],
    )
    assert summary["metrics"]["request_throughput_rps"] == 4_000_000.0
    assert summary["metrics"]["output_token_throughput_tps"] == 12_000_000.0
    assert summary["metrics"]["p95_ttft_ns"] == 290.0
    assert summary["metrics"]["p95_itl_ns"] == 100.0
    assert summary["metrics"]["peak_cuda_reserved_bytes"] == 240
    assert summary["metrics"]["peak_kv_bytes"] == 256
    assert set(summary["metrics"]["service_buckets"]) == {
        "short__short",
        "long__long",
    }
    assert 0.0 < summary["metrics"]["jain_service_rate_index"] <= 1.0


def test_case_aggregation_reports_median_and_worst_repetition():
    rows = [
        _case_row("P2", 0, throughput=110.0, ttft=90.0),
        _case_row("P2", 1, throughput=105.0, ttft=95.0),
        _case_row("P2", 2, throughput=99.0, ttft=110.0),
    ]
    aggregate = gate.aggregate_case_repetitions(rows)
    assert aggregate["median_metrics"]["request_throughput_rps"] == 105.0
    assert aggregate["median_metrics"]["p95_ttft_ns"] == 95.0
    assert aggregate["worst_repetition"]["repetition"] == 2
    assert (
        aggregate["worst_repetition"]["metrics"][
            "request_throughput_rps"
        ]
        == 99.0
    )


def _classification_manifest() -> dict:
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: gate.resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    aliases = gate.deduplicate_policies(resolved)
    return {
        "required_scenarios": ["steady_moderate"],
        "measured_repetitions": 3,
        "policy_identity_by_name": aliases["identity_by_name"],
        "canonical_policy_by_name": (
            aliases["canonical_policy_by_name"]
        ),
    }


def _case_row(
    policy: str,
    repetition: int,
    *,
    throughput: float = 100.0,
    ttft: float = 100.0,
    itl: float = 100.0,
    e2e: float = 100.0,
    gap: float = 100.0,
    cuda_reserved: float = 100.0,
    kv_bytes: float = 100.0,
    bucket_p95: float = 100.0,
    status: str = "PASS",
    exact_outputs: bool = True,
    complete_requests: bool = True,
    no_starvation: bool = True,
) -> dict:
    return {
        "policy": policy,
        "scenario": "steady_moderate",
        "repetition": repetition,
        "status": status,
        "correctness": {
            "exact_outputs": exact_outputs,
            "complete_requests": complete_requests,
            "no_starvation": no_starvation,
            "valid_lifecycle": True,
            "stable_p0_outputs": True,
        },
        "metrics": {
            "request_throughput_rps": throughput,
            "output_token_throughput_tps": throughput * 10.0,
            "p95_ttft_ns": ttft,
            "p95_itl_ns": itl,
            "p99_ttft_ns": ttft,
            "p99_itl_ns": itl,
            "p99_e2e_ns": e2e,
            "maximum_decode_gap_ns": gap,
            "peak_cuda_reserved_bytes": cuda_reserved,
            "peak_kv_bytes": kv_bytes,
            "service_buckets": {
                "short__short": {
                    "p95_e2e_ns": bucket_p95,
                    "completed_requests": 1,
                },
            },
        },
    }


def _rows_with_candidate(candidate_values: list[dict]) -> list[dict]:
    rows = []
    for repetition in range(3):
        rows.append(_case_row("P0", repetition))
        rows.append(_case_row(
            "P2",
            repetition,
            **candidate_values[repetition],
        ))
        rows.append(_case_row("P3", repetition))
    return rows


def test_classification_throughput_boundary_is_go():
    summary = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 105.0},
            {"throughput": 106.0},
            {"throughput": 107.0},
        ]),
    )
    assert summary["classification"] == "GO"
    assert summary["candidate_results"]["P2"]["benefit_path"] == (
        "throughput"
    )


def test_classification_favorable_but_subthreshold_is_promising():
    summary = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 104.0},
            {"throughput": 104.5},
            {"throughput": 104.999},
        ]),
    )
    assert summary["classification"] == "PROMISING_NOT_PROVEN"


def test_classification_latency_and_memory_boundaries_are_go():
    latency = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"ttft": 90.0, "throughput": 98.0},
            {"ttft": 89.0, "throughput": 99.0},
            {"ttft": 88.0, "throughput": 100.0},
        ]),
    )
    assert latency["classification"] == "GO"
    assert latency["candidate_results"]["P2"]["benefit_path"] == (
        "latency"
    )

    memory = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {
                "cuda_reserved": 95.0,
                "throughput": 98.0,
                "ttft": 102.0,
                "itl": 102.0,
            },
            {
                "cuda_reserved": 94.0,
                "throughput": 99.0,
                "ttft": 101.0,
                "itl": 101.0,
            },
            {
                "cuda_reserved": 93.0,
                "throughput": 100.0,
            },
        ]),
    )
    assert memory["classification"] == "GO"
    assert memory["candidate_results"]["P2"]["benefit_path"] == (
        "memory"
    )


def test_tail_guard_or_bad_worst_repetition_prevents_go():
    tail = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 106.0, "e2e": 110.001},
            {"throughput": 106.0},
            {"throughput": 106.0},
        ]),
    )
    assert tail["classification"] == "NO_GO"
    assert tail["candidate_results"]["P2"]["guard_failures"]

    bad_worst = gate.classify_gate(
        _classification_manifest(),
        _rows_with_candidate([
            {"throughput": 110.0},
            {"throughput": 110.0},
            {"throughput": 99.0},
        ]),
    )
    assert bad_worst["classification"] == "PROMISING_NOT_PROVEN"


def test_structural_and_correctness_failures_take_precedence():
    missing = _rows_with_candidate([
        {"throughput": 106.0},
        {"throughput": 106.0},
        {"throughput": 106.0},
    ])
    missing.pop()
    incomplete = gate.classify_gate(
        _classification_manifest(),
        missing,
    )
    assert incomplete["classification"] == "INCOMPLETE"
    assert incomplete["structural_failures"]

    incorrect_rows = _rows_with_candidate([
        {"throughput": 106.0},
        {"throughput": 106.0},
        {"throughput": 106.0},
    ])
    incorrect_rows[1]["correctness"]["exact_outputs"] = False
    no_go = gate.classify_gate(
        _classification_manifest(),
        incorrect_rows,
    )
    assert no_go["classification"] == "NO_GO"
    assert no_go["correctness_failures"]


def main():
    test_seeded_steady_and_burst_workloads_are_byte_stable()
    test_built_prompt_bank_is_sorted_hashed_and_valid()
    test_service_buckets_are_fixed_before_execution()
    test_policy_identity_aliases_explicit_default_only()
    test_nearest_rank_boundaries()
    test_canonical_rates_and_sampling_contract_are_frozen()
    test_invalid_lambda_and_policy_fail_closed()
    test_unexpected_candidate_policy_collision_is_rejected()
    test_prompt_bank_hash_detects_drift()
    test_calibration_manifest_is_deterministic_and_p0_only()
    test_reconstructs_scheduled_arrival_metrics_and_shared_step_tokens()
    test_one_token_output_has_no_itl_sample()
    test_lifecycle_reconstruction_rejects_duplicate_binding_and_bad_time()
    test_repetition_summary_reports_percentiles_fairness_and_memory()
    test_case_aggregation_reports_median_and_worst_repetition()
    test_classification_throughput_boundary_is_go()
    test_classification_favorable_but_subthreshold_is_promising()
    test_classification_latency_and_memory_boundaries_are_go()
    test_tail_guard_or_bad_worst_repetition_prevents_go()
    test_structural_and_correctness_failures_take_precedence()
    print("arrival load gate tests passed")


if __name__ == "__main__":
    main()
