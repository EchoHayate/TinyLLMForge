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
    print("arrival load gate tests passed")


if __name__ == "__main__":
    main()
