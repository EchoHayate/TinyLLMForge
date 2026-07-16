from __future__ import annotations

import statistics


def make_token_prompt(length: int, offset: int = 0) -> list[int]:
    return [100 + ((index + offset) % 1000) for index in range(length)]


def expected_reusable_tokens(prompt_tokens: int, block_size: int) -> int:
    if prompt_tokens <= 1:
        return 0
    return ((prompt_tokens - 1) // block_size) * block_size


def summarize_case_rows(rows: list[dict]) -> dict:
    return {
        "samples": len(rows),
        "median_ttft_ms": statistics.median(
            float(row["ttft_ms"]) for row in rows
        ),
        "min_ttft_ms": min(float(row["ttft_ms"]) for row in rows),
        "max_ttft_ms": max(float(row["ttft_ms"]) for row in rows),
        "median_query_tokens": statistics.median(
            int(row["query_tokens"]) for row in rows
        ),
        "median_cached_tokens": statistics.median(
            int(row["cached_tokens"]) for row in rows
        ),
        "all_correct": all(bool(row["correct"]) for row in rows),
    }


def decide_gate(correctness_rows: list[dict], performance_cases: list[dict]) -> dict:
    reasons = []
    failed = [row["case"] for row in correctness_rows if not row["correct"]]
    if failed:
        reasons.append("correctness failures: " + ", ".join(failed))
    for case in performance_cases:
        prefix = int(case["shared_prefix_tokens"])
        cold = float(case["cold"]["median_ttft_ms"])
        warm = float(case["warm"]["median_ttft_ms"])
        improvement = (cold - warm) / cold if cold > 0 else 0.0
        case["warm_ttft_improvement_fraction"] = improvement
        if not case["all_correct"]:
            reasons.append(f"{prefix}: incorrect performance sample")
        if prefix >= 1024 and improvement < 0.20:
            reasons.append(f"{prefix}: warm median TTFT improvement below 20%")
        if warm > cold * 1.05:
            reasons.append(f"{prefix}: warm median TTFT regression exceeds 5%")
        if int(case["warm_median_cached_tokens"]) != int(
            case["expected_reusable_tokens"]
        ):
            reasons.append(f"{prefix}: cached-token accounting mismatch")
        saved_queries = int(case["cold_median_query_tokens"]) - int(
            case["warm_median_query_tokens"]
        )
        if saved_queries != int(case["expected_reusable_tokens"]):
            reasons.append(f"{prefix}: executed prefill-token reduction mismatch")
    return {"decision": "NO_GO" if reasons else "GO", "reasons": reasons}
