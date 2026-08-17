from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


SCHEMA_VERSION = "qwen35.tp4-request-e2e64-comparison.v1"
CASE_SCHEMA_VERSION = "qwen35.tp4-w2-restore-profile-case.v1"
POLICIES = ("recompute", "exact_restore")
REPETITIONS = tuple(range(5))
REQUEST_COUNT = 4
GENERATED_TOKENS = 64
DECODE_STEPS = 63
OUTPUT_TOKENS_PER_CASE = REQUEST_COUNT * GENERATED_TOKENS
BASELINE_SOURCE_SHA256 = (
    "a26c543e79a9d4927fd0451d4a287363a677568a1daefe65a2a234a22f5997aa"
)
CANDIDATE_SOURCE_SHA256 = (
    "6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837"
)
METRIC_FIELDS = (
    "makespan_ns",
    "request_throughput_rps",
    "output_token_throughput_tps",
    "median_request_e2e_ns",
    "median_ttft_ns",
    "median_decode_ns",
)
LATENCY_FIELDS = {
    "makespan_ns",
    "median_request_e2e_ns",
    "median_ttft_ns",
    "median_decode_ns",
}


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def _case_id(policy: str, repetition: int) -> str:
    return (
        f"w2_long_reuse__measured__r{repetition}__{policy}"
    )


def _cases_root(root: Path) -> Path:
    candidates = (
        root / "download" / "output" / "cases",
        root / "download" / "cases",
        root / "cases",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ValueError(f"measured cases directory is missing: {root}")


def _validate_receipt(root: Path, source_sha256: str) -> dict:
    path = root / "attempt_receipt.json"
    if not path.is_file():
        raise ValueError(f"attempt receipt is missing: {root}")
    receipt = _read_json(path)
    if receipt.get("classification") != "DOWNLOADED":
        raise ValueError(f"attempt was not downloaded: {root}")
    if receipt.get("source_tree_sha256") != source_sha256:
        raise ValueError(f"attempt source identity mismatch: {root}")
    if (
        receipt.get("cleanup", {}).get("classification")
        != "CLEAN"
    ):
        raise ValueError(f"attempt cleanup is not CLEAN: {root}")
    return receipt


def _validate_profile(
    profile: dict,
    *,
    case_id: str,
    policy: str,
    repetition: int,
) -> None:
    if (
        profile.get("schema_version") != CASE_SCHEMA_VERSION
        or profile.get("case_id") != case_id
        or profile.get("workload") != "w2_long_reuse"
        or profile.get("phase") != "measured"
        or profile.get("policy") != policy
        or profile.get("repetition") != repetition
        or profile.get("generated_tokens", 64) != 64
    ):
        raise ValueError(f"invalid profile: {case_id}")
    if "generated_tokens" in profile and (
        profile.get("canonical_generated_tokens") != 64
        or profile.get("variant") != "canonical_output"
    ):
        raise ValueError(f"non-canonical profile: {case_id}")


def _validate_rows(
    rows: list[dict],
    *,
    case_id: str,
    policy: str,
    repetition: int,
    source_sha256: str,
) -> list[dict]:
    if len(rows) != REQUEST_COUNT:
        raise ValueError(
            f"{case_id} must contain four request rows"
        )
    request_ids = [row.get("request_id") for row in rows]
    expected_request_ids = [
        f"request-{index}"
        for index in range(REQUEST_COUNT)
    ]
    if sorted(request_ids) != expected_request_ids:
        raise ValueError(f"invalid request identities: {case_id}")
    ordered = sorted(rows, key=lambda row: row["request_id"])
    for row in ordered:
        if (
            row.get("case_id") != case_id
            or row.get("workload") != "w2_long_reuse"
            or row.get("phase") != "measured"
            or row.get("policy") != policy
            or row.get("repetition") != repetition
            or row.get("source_tree_sha256") != source_sha256
            or row.get("generated_tokens") != GENERATED_TOKENS
        ):
            raise ValueError(f"invalid request row: {case_id}")
        if len(row.get("output_token_ids", [])) != GENERATED_TOKENS:
            raise ValueError(
                f"{case_id} request must contain 64 output tokens"
            )
        if len(row.get("decode_step_ns", [])) != DECODE_STEPS:
            raise ValueError(
                f"{case_id} request must contain 63 decode steps"
            )
        for field in ("e2e_ns", "ttft_ns"):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"invalid {field} in {case_id}"
                )
    return ordered


def _case_metrics(
    cases_root: Path,
    *,
    policy: str,
    repetition: int,
    source_sha256: str,
) -> dict:
    case_id = _case_id(policy, repetition)
    case_root = cases_root / case_id
    profile_path = case_root / "profile.json"
    rows_path = case_root / "case_rows.jsonl"
    if not profile_path.is_file() or not rows_path.is_file():
        raise ValueError(f"missing measured artifacts: {case_id}")
    profile = _read_json(profile_path)
    _validate_profile(
        profile,
        case_id=case_id,
        policy=policy,
        repetition=repetition,
    )
    rows = _validate_rows(
        _read_jsonl(rows_path),
        case_id=case_id,
        policy=policy,
        repetition=repetition,
        source_sha256=source_sha256,
    )
    makespan_ns = max(row["e2e_ns"] for row in rows)
    return {
        "case_id": case_id,
        "policy": policy,
        "repetition": repetition,
        "makespan_ns": makespan_ns,
        "request_throughput_rps": (
            REQUEST_COUNT * 1_000_000_000 / makespan_ns
        ),
        "output_token_throughput_tps": (
            OUTPUT_TOKENS_PER_CASE * 1_000_000_000
            / makespan_ns
        ),
        "median_request_e2e_ns": statistics.median(
            row["e2e_ns"] for row in rows
        ),
        "median_ttft_ns": statistics.median(
            row["ttft_ns"] for row in rows
        ),
        "median_decode_ns": statistics.median(
            sum(row["decode_step_ns"]) for row in rows
        ),
        "requests": [
            {
                "request_id": row["request_id"],
                "output_token_ids": row["output_token_ids"],
                "e2e_ns": row["e2e_ns"],
                "ttft_ns": row["ttft_ns"],
                "decode_ns": sum(row["decode_step_ns"]),
            }
            for row in rows
        ],
    }


def _metric_summary(rows: list[dict]) -> dict:
    result = {}
    for field in METRIC_FIELDS:
        values = [row[field] for row in rows]
        result[field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "pstdev": statistics.pstdev(values),
        }
    return result


def _pooled_request_summary(rows: list[dict]) -> dict:
    requests = [
        request
        for row in rows
        for request in row["requests"]
    ]
    return {
        field: {
            "median": statistics.median(
                request[field] for request in requests
            ),
            "min": min(request[field] for request in requests),
            "max": max(request[field] for request in requests),
            "pstdev": statistics.pstdev(
                request[field] for request in requests
            ),
            "count": len(requests),
        }
        for field in ("e2e_ns", "ttft_ns", "decode_ns")
    }


def _load_attempt(
    root: Path,
    *,
    source_sha256: str,
) -> dict:
    root = Path(root)
    receipt = _validate_receipt(root, source_sha256)
    cases_root = _cases_root(root)
    cases = {
        policy: [
            _case_metrics(
                cases_root,
                policy=policy,
                repetition=repetition,
                source_sha256=source_sha256,
            )
            for repetition in REPETITIONS
        ]
        for policy in POLICIES
    }
    return {
        "root": str(root),
        "source_tree_sha256": source_sha256,
        "receipt": receipt,
        "cases": cases,
        "summary": {
            policy: _metric_summary(cases[policy])
            for policy in POLICIES
        },
        "pooled_request_summary": {
            policy: _pooled_request_summary(cases[policy])
            for policy in POLICIES
        },
    }


def _parity(
    baseline: dict,
    candidate: dict,
) -> tuple[bool, list[str]]:
    reasons = []
    for policy in POLICIES:
        for repetition in REPETITIONS:
            baseline_case = baseline["cases"][policy][repetition]
            candidate_case = candidate["cases"][policy][repetition]
            if baseline_case["case_id"] != candidate_case["case_id"]:
                reasons.append(
                    f"case alignment mismatch: {policy} r{repetition}"
                )
                continue
            baseline_requests = {
                row["request_id"]: row
                for row in baseline_case["requests"]
            }
            candidate_requests = {
                row["request_id"]: row
                for row in candidate_case["requests"]
            }
            if baseline_requests.keys() != candidate_requests.keys():
                reasons.append(
                    f"request alignment mismatch: {policy} r{repetition}"
                )
                continue
            for request_id in baseline_requests:
                if (
                    baseline_requests[request_id]["output_token_ids"]
                    != candidate_requests[request_id]["output_token_ids"]
                ):
                    reasons.append(
                        "token parity mismatch: "
                        f"{policy} r{repetition} {request_id}"
                    )
    return not reasons, reasons


def _improvement(
    baseline: float,
    candidate: float,
    *,
    latency: bool,
) -> float:
    if latency:
        return 1.0 - candidate / baseline
    return candidate / baseline - 1.0


def _build_comparison(
    baseline: dict,
    candidate: dict,
) -> dict:
    by_policy = {}
    for policy in POLICIES:
        baseline_summary = baseline["summary"][policy]
        candidate_summary = candidate["summary"][policy]
        metrics = {}
        paired = []
        for field in METRIC_FIELDS:
            baseline_median = baseline_summary[field]["median"]
            candidate_median = candidate_summary[field]["median"]
            metrics[field] = {
                "baseline_median": baseline_median,
                "candidate_median": candidate_median,
                "improvement_fraction": _improvement(
                    baseline_median,
                    candidate_median,
                    latency=field in LATENCY_FIELDS,
                ),
                "baseline_dispersion": baseline_summary[field],
                "candidate_dispersion": candidate_summary[field],
            }
        for repetition in REPETITIONS:
            baseline_case = baseline["cases"][policy][repetition]
            candidate_case = candidate["cases"][policy][repetition]
            paired.append({
                "repetition": repetition,
                "makespan_improvement_fraction": _improvement(
                    baseline_case["makespan_ns"],
                    candidate_case["makespan_ns"],
                    latency=True,
                ),
                "request_throughput_improvement_fraction": (
                    _improvement(
                        baseline_case["request_throughput_rps"],
                        candidate_case["request_throughput_rps"],
                        latency=False,
                    )
                ),
                "output_token_throughput_improvement_fraction": (
                    _improvement(
                        baseline_case[
                            "output_token_throughput_tps"
                        ],
                        candidate_case[
                            "output_token_throughput_tps"
                        ],
                        latency=False,
                    )
                ),
            })
        pooled_request_metrics = {}
        for field in ("e2e_ns", "ttft_ns", "decode_ns"):
            baseline_metric = baseline[
                "pooled_request_summary"
            ][policy][field]
            candidate_metric = candidate[
                "pooled_request_summary"
            ][policy][field]
            pooled_request_metrics[field] = {
                "baseline_median": baseline_metric["median"],
                "candidate_median": candidate_metric["median"],
                "improvement_fraction": _improvement(
                    baseline_metric["median"],
                    candidate_metric["median"],
                    latency=True,
                ),
                "baseline_dispersion": baseline_metric,
                "candidate_dispersion": candidate_metric,
            }
        by_policy[policy] = {
            "metrics": metrics,
            "pooled_request_metrics": pooled_request_metrics,
            "paired_repetitions": paired,
            "makespan_improvement_fraction": metrics[
                "makespan_ns"
            ]["improvement_fraction"],
            "request_throughput_improvement_fraction": metrics[
                "request_throughput_rps"
            ]["improvement_fraction"],
            "output_token_throughput_improvement_fraction": metrics[
                "output_token_throughput_tps"
            ]["improvement_fraction"],
        }
    return {"by_policy": by_policy}


def _classify(comparison: dict) -> str:
    improvements = [
        comparison["by_policy"][policy][
            "makespan_improvement_fraction"
        ]
        for policy in POLICIES
    ]
    if all(value >= 0.05 for value in improvements):
        return "E2E_PERFORMANCE_PASS"
    if all(abs(value) < 0.05 for value in improvements):
        return "NO_MATERIAL_E2E_CHANGE"
    if any(value <= -0.05 for value in improvements) and not any(
        value >= 0.05 for value in improvements
    ):
        return "E2E_REGRESSION"
    return "MIXED"


def compare_attempts(
    baseline_root: Path,
    candidate_root: Path,
) -> dict:
    reasons = []
    try:
        baseline = _load_attempt(
            Path(baseline_root),
            source_sha256=BASELINE_SOURCE_SHA256,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append(f"baseline invalid: {error}")
        baseline = None
    try:
        candidate = _load_attempt(
            Path(candidate_root),
            source_sha256=CANDIDATE_SOURCE_SHA256,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append(f"candidate invalid: {error}")
        candidate = None
    if baseline is None or candidate is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "classification": "NO_GO",
            "output_parity": False,
            "reasons": reasons,
        }
    output_parity, parity_reasons = _parity(
        baseline,
        candidate,
    )
    reasons.extend(parity_reasons)
    comparison = _build_comparison(baseline, candidate)
    classification = (
        _classify(comparison)
        if output_parity
        else "NO_GO"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "output_parity": output_parity,
        "reasons": reasons,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = compare_attempts(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "classification": result["classification"],
        "output": str(args.output),
        "output_parity": result["output_parity"],
        "reasons": result["reasons"],
    }, indent=2, sort_keys=True))
    return 0 if result["classification"] != "NO_GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
