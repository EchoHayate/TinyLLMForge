#!/usr/bin/env python3
"""Produce the frozen four-arm phase-stitched exact-graph gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from tools import phase_stitched_exact_graph_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[1]

GO_PHASE_STITCHED_EXACT_GRAPH = "GO_PHASE_STITCHED_EXACT_GRAPH"
NO_GO_EVIDENCE = "NO_GO_EVIDENCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_MECHANISM = "NO_GO_MECHANISM"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"

_ROW_NUMERIC_FIELDS = (
    "ttft_ns",
    "token_0_to_1_gap_ns",
    "tpot_median_ns",
    "e2e_ns",
    "output_tokens_per_second",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
)
_ROW_COUNTER_FIELDS = (
    "prefill_graph_replay_delta",
    "exact_burst_replay_delta",
    "phase_stitch_attempt_delta",
    "phase_stitch_success_delta",
    "phase_stitch_prefill_replay_delta",
    "phase_stitch_decode_replay_delta",
    "phase_stitch_target_forward_delta",
    "phase_stitch_failure_delta",
    "phase_stitch_quarantine_delta",
    "phase_stitch_prefix_d2h_calls",
    "phase_stitch_suffix_d2h_calls",
    "phase_stitch_prefix_d2h_bytes",
    "phase_stitch_suffix_d2h_bytes",
    "phase_stitch_prefix_commits",
    "phase_stitch_suffix_commits",
    "phase_stitch_pending_leases",
    "phase_stitch_fallback_count",
    "preauthorized_kv_tokens",
)


def _read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: object) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def _improvement(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return 1.0 - candidate / baseline


def _regression(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return candidate / baseline - 1.0


def _validate_run_manifest(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("run manifest must be an object")
    if value.get("schema_version") != contract.RUN_SCHEMA_VERSION:
        raise ValueError("run manifest schema drifted")
    if value.get("contract_sha256") != contract.contract_sha256():
        raise ValueError("run manifest contract hash drifted")
    if value.get("case_order") != list(contract.expected_case_ids()):
        raise ValueError("run manifest case order drifted")
    if value.get("clean_gpu_admission") is not True:
        raise ValueError("strict-clean GPU admission is absent")
    if not isinstance(value.get("run_tag"), str) or not value["run_tag"]:
        raise ValueError("run tag is absent")
    if value.get("precision") != "bfloat16":
        raise ValueError("run precision drifted")
    if not _is_commit_sha(value.get("source_base_commit")):
        raise ValueError("source commit is invalid")
    source_files = value.get("source_files")
    if (
        not isinstance(source_files, dict)
        or set(source_files) != set(contract.SOURCE_FILES)
        or not all(_is_sha256(digest) for digest in source_files.values())
    ):
        raise ValueError("source hash inventory is invalid")
    for relative, expected_hash in source_files.items():
        source_path = REPO_ROOT / relative
        if (
            not source_path.is_file()
            or _file_sha256(source_path) != expected_hash
        ):
            raise ValueError(f"source hash mismatch: {relative}")
    inventory = value.get("gpu_inventory")
    if (
        not isinstance(inventory, list)
        or len(inventory) != 1
        or "A100" not in str(inventory[0].get("name", ""))
    ):
        raise ValueError("one admitted A100 is required")
    return value


def _validate_row(row: object, case: dict, sample_index: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    expected_scalars = {
        "schema_version": contract.ROW_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "round": case["round"],
        "order_position": case["order_position"],
        "arm": case["arm"],
        "prompt_tokens": case["prompt_tokens"],
        "sample_index": sample_index,
        "generated_tokens": contract.GENERATED_TOKENS,
    }
    if any(row.get(name) != expected for name, expected in expected_scalars.items()):
        raise ValueError("row identity drifted")
    if not _is_sha256(row.get("prompt_sha256")):
        raise ValueError("prompt hash is invalid")
    if not _is_sha256(row.get("output_text_sha256")):
        raise ValueError("output text hash is invalid")
    token_ids = row.get("output_token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != contract.GENERATED_TOKENS
        or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids)
    ):
        raise ValueError("output token inventory is invalid")
    samples = row.get("tpot_samples_ns")
    if (
        not isinstance(samples, list)
        or len(samples) != contract.GENERATED_TOKENS - 1
    ):
        raise ValueError("TPOT sample inventory is invalid")
    for index, value in enumerate(samples):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"TPOT sample {index} is invalid")
    for name in _ROW_NUMERIC_FIELDS:
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"row metric {name} is invalid")
    for name in _ROW_COUNTER_FIELDS:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"row counter {name} is invalid")
    return row


def _load_results(run_dir: Path) -> tuple[dict, list[dict]]:
    root = Path(run_dir)
    run_manifest = _validate_run_manifest(
        _read_json(root / "run_manifest.json")
    )
    cases_root = root / "cases"
    actual = {
        path.name for path in cases_root.iterdir() if path.is_dir()
    }
    if actual != set(contract.expected_case_ids()):
        raise ValueError("case directory inventory is incomplete")
    results = []
    for case in contract.build_case_matrix():
        result = _read_json(cases_root / case["case_id"] / "result.json")
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != contract.RESULT_SCHEMA_VERSION
            or result.get("case") != case
            or result.get("model") != run_manifest.get("model")
            or result.get("model_dtype") != "bfloat16"
        ):
            raise ValueError("case result metadata drifted")
        rows = result.get("rows")
        if (
            not isinstance(rows, list)
            or len(rows) != contract.MEASURED_REPETITIONS
        ):
            raise ValueError("case row inventory is incomplete")
        for sample_index, row in enumerate(rows):
            _validate_row(row, case, sample_index)
        results.append(result)
    return run_manifest, results


def _aggregate(results: list[dict]) -> tuple[dict, dict, dict]:
    groups = {
        prompt_tokens: {arm: [] for arm in contract.ARMS}
        for prompt_tokens in contract.PROMPT_TOKEN_COUNTS
    }
    paired = {}
    evidence_ok = True
    mechanism_ok = True
    for result in results:
        case = result["case"]
        arm = case["arm"]
        rows = result["rows"]
        groups[case["prompt_tokens"]][arm].extend(rows)
        expected_graph = arm != "eager"
        prefill = result.get("prefill_graph_summary")
        burst = result.get("exact_burst_summary")
        phase = result.get("phase_stitch_summary")
        if not isinstance(prefill, dict):
            evidence_ok = False
        elif expected_graph:
            required_capture_metrics = (
                "capture_attempts",
                "capture_successes",
                "capture_failures",
                "replays",
                "replay_failures",
                "quarantines",
                "fallbacks",
                "static_bytes",
                "allocated_delta_bytes",
                "reserved_delta_bytes",
                "total_capture_ns",
            )
            evidence_ok = evidence_ok and (
                all(
                    isinstance(prefill.get(name), int)
                    and not isinstance(prefill.get(name), bool)
                    and prefill[name] >= 0
                    for name in required_capture_metrics
                )
                and
                prefill.get("capture_attempts", 0) >= 2
                and prefill.get("capture_successes", 0) >= 2
                and prefill.get("capture_failures") == 0
                and prefill.get("replay_failures") == 0
                and prefill.get("quarantines") == 0
            )
        if not isinstance(burst, dict) or not isinstance(phase, dict):
            evidence_ok = False
        else:
            for summary in (burst, phase):
                evidence_ok = evidence_ok and all(
                    isinstance(summary.get(name), int)
                    and not isinstance(summary.get(name), bool)
                    and summary[name] >= 0
                    for name in (
                        "failures",
                        "quarantines",
                        "pending_leases",
                        "fallback_count",
                    )
                )
                fallback_counts = summary.get("fallback_counts")
                evidence_ok = evidence_ok and (
                    isinstance(fallback_counts, dict)
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                        for value in fallback_counts.values()
                    )
                    and sum(fallback_counts.values())
                    == summary.get("fallback_count")
                )
            mechanism_ok = mechanism_ok and (
                burst.get("failures") == 0
                and burst.get("quarantines") == 0
                and burst.get("pending_leases") == 0
                and burst.get("quarantine_reason") is None
                and phase.get("failures") == 0
                and phase.get("quarantines") == 0
                and phase.get("pending_leases") == 0
                and (
                    arm != "stitched_composition"
                    or (
                        phase.get("fallback_count") == 0
                        and not phase.get("fallback_counts")
                    )
                )
            )
        for row in rows:
            key = (
                case["round"],
                case["prompt_tokens"],
                row["sample_index"],
                row["prompt_sha256"],
            )
            paired.setdefault(key, {})[arm] = (
                tuple(row["output_token_ids"]),
                row["output_text_sha256"],
            )
            if arm == "stitched_composition":
                mechanism_ok = mechanism_ok and all((
                    row["prefill_graph_replay_delta"] == 1,
                    row["exact_burst_replay_delta"] == 120,
                    row["phase_stitch_attempt_delta"] == 1,
                    row["phase_stitch_success_delta"] == 1,
                    row["phase_stitch_prefill_replay_delta"] == 1,
                    row["phase_stitch_decode_replay_delta"] == 7,
                    row["phase_stitch_target_forward_delta"] == 8,
                    row["phase_stitch_failure_delta"] == 0,
                    row["phase_stitch_quarantine_delta"] == 0,
                    row["phase_stitch_prefix_d2h_calls"] == 1,
                    row["phase_stitch_suffix_d2h_calls"] == 1,
                    row["phase_stitch_prefix_d2h_bytes"] == 8,
                    row["phase_stitch_suffix_d2h_bytes"] == 56,
                    row["phase_stitch_prefix_commits"] == 1,
                    row["phase_stitch_suffix_commits"] == 1,
                    row["phase_stitch_pending_leases"] == 0,
                    row["phase_stitch_fallback_count"] == 0,
                    row["preauthorized_kv_tokens"] == 7,
                ))
            elif arm == "independent_composition":
                mechanism_ok = mechanism_ok and (
                    row["prefill_graph_replay_delta"] == 1
                    and row["exact_burst_replay_delta"] == 127
                    and row["phase_stitch_attempt_delta"] == 0
                )
    exact_token_ids = bool(paired) and all(
        set(outputs) == set(contract.ARMS)
        and len({
            value[0] for value in outputs.values()
        }) == 1
        for outputs in paired.values()
    )
    exact_text = bool(paired) and all(
        set(outputs) == set(contract.ARMS)
        and len({
            value[1] for value in outputs.values()
        }) == 1
        for outputs in paired.values()
    )
    shapes = []
    for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
        arms = groups[prompt_tokens]
        expected_count = contract.ROUNDS * contract.MEASURED_REPETITIONS
        if any(len(rows) != expected_count for rows in arms.values()):
            raise ValueError("shape row inventory is incomplete")
        arm_metrics = {}
        for arm, rows in arms.items():
            e2e = [float(row["e2e_ns"]) for row in rows]
            arm_metrics[arm] = {
                "sample_count": len(rows),
                "ttft_median_ns": statistics.median(
                    float(row["ttft_ns"]) for row in rows
                ),
                "token_0_to_1_gap_median_ns": statistics.median(
                    float(row["token_0_to_1_gap_ns"]) for row in rows
                ),
                "tpot_median_ns": statistics.median(
                    float(row["tpot_median_ns"]) for row in rows
                ),
                "e2e_median_ns": statistics.median(e2e),
                "e2e_p95_ns": _percentile(e2e, 0.95),
                "e2e_p99_ns": _percentile(e2e, 0.99),
                "throughput_median_tokens_per_second": statistics.median(
                    float(row["output_tokens_per_second"]) for row in rows
                ),
                "peak_reserved_bytes": max(
                    int(row["cuda_peak_reserved_bytes"]) for row in rows
                ),
            }
        baseline = arm_metrics["independent_composition"]
        candidate = arm_metrics["stitched_composition"]
        shapes.append({
            "prompt_tokens": prompt_tokens,
            "arms": arm_metrics,
            "d_vs_c_e2e_improvement_fraction": _improvement(
                candidate["e2e_median_ns"],
                baseline["e2e_median_ns"],
            ),
            "d_vs_c_gap_improvement_fraction": _improvement(
                candidate["token_0_to_1_gap_median_ns"],
                baseline["token_0_to_1_gap_median_ns"],
            ),
            "d_vs_c_ttft_regression_fraction": _regression(
                candidate["ttft_median_ns"],
                baseline["ttft_median_ns"],
            ),
            "d_vs_c_p95_e2e_regression_fraction": _regression(
                candidate["e2e_p95_ns"],
                baseline["e2e_p95_ns"],
            ),
            "d_vs_c_p99_e2e_regression_fraction": _regression(
                candidate["e2e_p99_ns"],
                baseline["e2e_p99_ns"],
            ),
            "d_vs_c_peak_reserved_memory_regression_fraction": _regression(
                candidate["peak_reserved_bytes"],
                baseline["peak_reserved_bytes"],
            ),
            "a_vs_b_prefill_attribution_fraction": _improvement(
                arm_metrics["prefill_only"]["e2e_median_ns"],
                arm_metrics["eager"]["e2e_median_ns"],
            ),
        })
    all_c = [
        float(row["e2e_ns"])
        for shape in groups.values()
        for row in shape["independent_composition"]
    ]
    all_d = [
        float(row["e2e_ns"])
        for shape in groups.values()
        for row in shape["stitched_composition"]
    ]
    all_c_gap = [
        float(row["token_0_to_1_gap_ns"])
        for shape in groups.values()
        for row in shape["independent_composition"]
    ]
    all_d_gap = [
        float(row["token_0_to_1_gap_ns"])
        for shape in groups.values()
        for row in shape["stitched_composition"]
    ]
    aggregate = {
        "d_vs_c_e2e_improvement_fraction": _improvement(
            statistics.median(all_d),
            statistics.median(all_c),
        ),
        "d_vs_c_token_0_to_1_gap_improvement_fraction": _improvement(
            statistics.median(all_d_gap),
            statistics.median(all_c_gap),
        ),
    }
    checks = {
        "complete_case_inventory": len(results)
        == len(contract.build_case_matrix()),
        "complete_evidence": evidence_ok,
        "all_token_ids_exact": exact_token_ids,
        "all_text_exact": exact_text,
        "complete_accounting": mechanism_ok,
        "zero_failure_and_quarantine": mechanism_ok,
        "shape_e2e_gain": any(
            shape["d_vs_c_e2e_improvement_fraction"]
            >= contract.E2E_SHAPE_IMPROVEMENT_MINIMUM
            for shape in shapes
        ),
        "aggregate_e2e_gain": (
            aggregate["d_vs_c_e2e_improvement_fraction"]
            >= contract.E2E_AGGREGATE_IMPROVEMENT_MINIMUM
        ),
        "gap_gain": (
            aggregate[
                "d_vs_c_token_0_to_1_gap_improvement_fraction"
            ]
            >= contract.TOKEN_0_TO_1_GAP_IMPROVEMENT_MINIMUM
        ),
        "ttft_non_regression": all(
            shape["d_vs_c_ttft_regression_fraction"]
            <= contract.TTFT_REGRESSION_LIMIT
            for shape in shapes
        ),
        "tail_non_regression": all(
            shape["d_vs_c_p95_e2e_regression_fraction"]
            <= contract.E2E_TAIL_REGRESSION_LIMIT
            and shape["d_vs_c_p99_e2e_regression_fraction"]
            <= contract.E2E_TAIL_REGRESSION_LIMIT
            for shape in shapes
        ),
        "memory_non_regression": all(
            shape[
                "d_vs_c_peak_reserved_memory_regression_fraction"
            ]
            <= contract.PEAK_RESERVED_MEMORY_REGRESSION_LIMIT
            for shape in shapes
        ),
    }
    return {"shapes": shapes, "aggregate": aggregate}, checks, {
        "pair_count": len(paired),
        "all_token_ids_exact": exact_token_ids,
        "all_text_exact": exact_text,
    }


def _classification(checks: dict) -> str:
    if not checks["complete_case_inventory"] or not checks["complete_evidence"]:
        return NO_GO_EVIDENCE
    if not checks["all_token_ids_exact"] or not checks["all_text_exact"]:
        return NO_GO_CORRECTNESS
    if (
        not checks["complete_accounting"]
        or not checks["zero_failure_and_quarantine"]
    ):
        return NO_GO_MECHANISM
    if not all(
        checks[name]
        for name in (
            "shape_e2e_gain",
            "aggregate_e2e_gain",
            "gap_gain",
            "ttft_non_regression",
            "tail_non_regression",
            "memory_non_regression",
        )
    ):
        return NO_GO_PERFORMANCE
    return GO_PHASE_STITCHED_EXACT_GRAPH


def produce_gate(run_dir: Path) -> dict:
    root = Path(run_dir)
    try:
        run_manifest, results = _load_results(root)
        performance, checks, correctness = _aggregate(results)
        classification = _classification(checks)
        run_tag = run_manifest["run_tag"]
        evidence_errors = []
    except (KeyError, TypeError, ValueError) as error:
        run_manifest = _read_json(root / "run_manifest.json")
        run_tag = (
            run_manifest.get("run_tag", "invalid-run")
            if isinstance(run_manifest, dict)
            else "invalid-run"
        )
        classification = NO_GO_EVIDENCE
        performance = {"shapes": [], "aggregate": {}}
        correctness = {
            "pair_count": 0,
            "all_token_ids_exact": False,
            "all_text_exact": False,
        }
        checks = {
            "complete_case_inventory": False,
            "complete_evidence": False,
            "all_token_ids_exact": False,
            "all_text_exact": False,
            "complete_accounting": False,
            "zero_failure_and_quarantine": False,
            "shape_e2e_gain": False,
            "aggregate_e2e_gain": False,
            "gap_gain": False,
            "ttft_non_regression": False,
            "tail_non_regression": False,
            "memory_non_regression": False,
        }
        evidence_errors = [str(error)]
    summary = {
        "schema_version": contract.SUMMARY_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run_tag,
        "classification": classification,
        "correctness": correctness,
        "mechanism": {
            "complete_accounting": checks["complete_accounting"],
            "zero_failure_and_quarantine":
                checks["zero_failure_and_quarantine"],
        },
        "performance": performance,
        "checks": checks,
        "evidence_errors": evidence_errors,
    }
    gate = {
        "schema_version": contract.GATE_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run_tag,
        "classification": classification,
        "checks": checks,
        "summary_sha256": contract.canonical_json_sha256(summary),
        "correctness": correctness,
        "mechanism": summary["mechanism"],
        "performance": performance,
        "evidence_errors": evidence_errors,
    }
    _atomic_write_json(root / "summary.json", summary)
    _atomic_write_json(root / "gate.json", gate)
    manifest_paths = (
        [Path("run_manifest.json")]
        + [
            Path("cases") / case_id / "result.json"
            for case_id in contract.expected_case_ids()
        ]
        + [Path("summary.json"), Path("gate.json")]
    )
    manifest = {
        "schema_version": contract.MANIFEST_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "files": {
            path.as_posix(): _file_sha256(root / path)
            for path in manifest_paths
        },
    }
    _atomic_write_json(root / "manifest.json", manifest)
    receipt = {
        "schema_version": contract.RECEIPT_SCHEMA_VERSION,
        "producer": "phase_stitched_exact_graph_gate",
        "classification": classification,
        "manifest_sha256": _file_sha256(root / "manifest.json"),
    }
    _atomic_write_json(root / "producer_receipt.json", receipt)
    return gate


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    result = produce_gate(_parse_args(argv).run_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
