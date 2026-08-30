#!/usr/bin/env python3
"""Independently verify a four-arm phase-stitched runtime artifact."""

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


def _read(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
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
    temporary.replace(path)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    return float(ordered[max(0, math.ceil(probability * len(ordered)) - 1)])


def _gain(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("non-positive baseline")
    return 1.0 - candidate / baseline


def _regression(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("non-positive baseline")
    return candidate / baseline - 1.0


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_run_manifest(run: object) -> dict:
    if not isinstance(run, dict):
        raise ValueError("run manifest must be an object")
    source_files = run.get("source_files")
    inventory = run.get("gpu_inventory")
    if (
        run.get("schema_version") != contract.RUN_SCHEMA_VERSION
        or run.get("contract_sha256") != contract.contract_sha256()
        or run.get("case_order") != list(contract.expected_case_ids())
        or run.get("clean_gpu_admission") is not True
        or run.get("precision") != "bfloat16"
        or not isinstance(run.get("run_tag"), str)
        or not run["run_tag"]
        or not _is_lower_hex(run.get("source_base_commit"), 40)
    ):
        raise ValueError("run manifest does not satisfy frozen contract")
    if (
        not isinstance(source_files, dict)
        or set(source_files) != set(contract.SOURCE_FILES)
        or not all(
            _is_lower_hex(digest, 64)
            for digest in source_files.values()
        )
    ):
        raise ValueError("source hash inventory is invalid")
    for relative, expected_hash in source_files.items():
        source_path = REPO_ROOT / relative
        if not source_path.is_file() or _sha(source_path) != expected_hash:
            raise ValueError(f"source hash mismatch: {relative}")
    if (
        not isinstance(inventory, list)
        or len(inventory) != 1
        or not isinstance(inventory[0], dict)
        or "A100" not in str(inventory[0].get("name", ""))
    ):
        raise ValueError("one admitted A100 is required")
    return run


def _validate_raw_row(
    row: object,
    *,
    case: dict,
    sample_index: int,
) -> dict:
    if not isinstance(row, dict):
        raise ValueError("raw row must be an object")
    expected_identity = {
        "schema_version": contract.ROW_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "round": case["round"],
        "order_position": case["order_position"],
        "arm": case["arm"],
        "prompt_tokens": case["prompt_tokens"],
        "sample_index": sample_index,
        "generated_tokens": contract.GENERATED_TOKENS,
    }
    if any(
        row.get(name) != expected
        for name, expected in expected_identity.items()
    ):
        raise ValueError("raw row identity drifted")
    if (
        not _is_lower_hex(row.get("prompt_sha256"), 64)
        or not _is_lower_hex(row.get("output_text_sha256"), 64)
    ):
        raise ValueError("raw row hash inventory is invalid")
    token_ids = row.get("output_token_ids")
    tpot_samples = row.get("tpot_samples_ns")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != contract.GENERATED_TOKENS
        or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in token_ids
        )
        or not isinstance(tpot_samples, list)
        or len(tpot_samples) != contract.GENERATED_TOKENS - 1
    ):
        raise ValueError("raw row token inventory drifted")
    for name, value in (
        [(name, row.get(name)) for name in _ROW_NUMERIC_FIELDS]
        + [
            (f"tpot_samples_ns[{index}]", value)
            for index, value in enumerate(tpot_samples)
        ]
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"raw row metric is invalid: {name}")
    for name in _ROW_COUNTER_FIELDS:
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"raw row counter is invalid: {name}")
    return row


def _reconstruct(root: Path) -> tuple[dict, dict, dict]:
    run = _validate_run_manifest(_read(root / "run_manifest.json"))
    cases_root = root / "cases"
    if {
        path.name for path in cases_root.iterdir() if path.is_dir()
    } != set(contract.expected_case_ids()):
        raise ValueError("case directory inventory drifted")
    buckets = {
        count: {arm: [] for arm in contract.ARMS}
        for count in contract.PROMPT_TOKEN_COUNTS
    }
    pairs = {}
    mechanism = True
    evidence = True
    expected_result_paths = {
        f"cases/{case_id}/result.json"
        for case_id in contract.expected_case_ids()
    }
    for case in contract.build_case_matrix():
        result = _read(
            root / "cases" / case["case_id"] / "result.json"
        )
        if (
            not isinstance(result, dict)
            or result.get("schema_version") != contract.RESULT_SCHEMA_VERSION
            or result.get("case") != case
            or result.get("model") != run.get("model")
            or result.get("model_dtype") != "bfloat16"
        ):
            raise ValueError("case result metadata drifted")
        rows = result.get("rows")
        if (
            not isinstance(rows, list)
            or len(rows) != contract.MEASURED_REPETITIONS
        ):
            raise ValueError("case row inventory drifted")
        prefill = result.get("prefill_graph_summary")
        burst = result.get("exact_burst_summary")
        phase = result.get("phase_stitch_summary")
        if not isinstance(prefill, dict):
            evidence = False
        elif case["arm"] != "eager":
            capture_names = (
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
            evidence = evidence and (
                all(
                    isinstance(prefill.get(name), int)
                    and not isinstance(prefill.get(name), bool)
                    and prefill[name] >= 0
                    for name in capture_names
                )
                and
                prefill.get("capture_attempts", 0) >= 2
                and prefill.get("capture_successes", 0) >= 2
                and prefill.get("capture_failures") == 0
                and prefill.get("replay_failures") == 0
                and prefill.get("quarantines") == 0
            )
        if not isinstance(burst, dict) or not isinstance(phase, dict):
            evidence = False
        else:
            for summary in (burst, phase):
                fallback_counts = summary.get("fallback_counts")
                evidence = evidence and (
                    all(
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
                    and isinstance(fallback_counts, dict)
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 0
                        for value in fallback_counts.values()
                    )
                    and sum(fallback_counts.values())
                    == summary.get("fallback_count")
                )
            mechanism = mechanism and (
                burst.get("failures") == 0
                and burst.get("quarantines") == 0
                and burst.get("pending_leases") == 0
                and burst.get("quarantine_reason") is None
                and phase.get("failures") == 0
                and phase.get("quarantines") == 0
                and phase.get("pending_leases") == 0
                and (
                    case["arm"] != "stitched_composition"
                    or (
                        phase.get("fallback_count") == 0
                        and not phase.get("fallback_counts")
                    )
                )
            )
        for sample_index, row in enumerate(rows):
            _validate_raw_row(
                row,
                case=case,
                sample_index=sample_index,
            )
            buckets[case["prompt_tokens"]][case["arm"]].append(row)
            key = (
                case["round"],
                case["prompt_tokens"],
                sample_index,
                row.get("prompt_sha256"),
            )
            pairs.setdefault(key, {})[case["arm"]] = (
                tuple(row["output_token_ids"]),
                row.get("output_text_sha256"),
            )
            if case["arm"] == "stitched_composition":
                required = {
                    "prefill_graph_replay_delta": 1,
                    "exact_burst_replay_delta": 120,
                    "phase_stitch_attempt_delta": 1,
                    "phase_stitch_success_delta": 1,
                    "phase_stitch_prefill_replay_delta": 1,
                    "phase_stitch_decode_replay_delta": 7,
                    "phase_stitch_target_forward_delta": 8,
                    "phase_stitch_failure_delta": 0,
                    "phase_stitch_quarantine_delta": 0,
                    "phase_stitch_prefix_d2h_calls": 1,
                    "phase_stitch_suffix_d2h_calls": 1,
                    "phase_stitch_prefix_d2h_bytes": 8,
                    "phase_stitch_suffix_d2h_bytes": 56,
                    "phase_stitch_prefix_commits": 1,
                    "phase_stitch_suffix_commits": 1,
                    "phase_stitch_pending_leases": 0,
                    "phase_stitch_fallback_count": 0,
                    "preauthorized_kv_tokens": 7,
                }
                mechanism = mechanism and all(
                    row.get(name) == value
                    for name, value in required.items()
                )
            elif case["arm"] == "independent_composition":
                mechanism = mechanism and (
                    row.get("prefill_graph_replay_delta") == 1
                    and row.get("exact_burst_replay_delta") == 127
                    and row.get("phase_stitch_attempt_delta") == 0
                )
    exact_token_ids = bool(pairs) and all(
        set(outputs) == set(contract.ARMS)
        and len({value[0] for value in outputs.values()}) == 1
        for outputs in pairs.values()
    )
    exact_text = bool(pairs) and all(
        set(outputs) == set(contract.ARMS)
        and len({value[1] for value in outputs.values()}) == 1
        for outputs in pairs.values()
    )
    shapes = []
    for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
        metrics = {}
        for arm in contract.ARMS:
            rows = buckets[prompt_tokens][arm]
            if len(rows) != contract.ROUNDS * contract.MEASURED_REPETITIONS:
                raise ValueError("shape sample inventory drifted")
            e2e = [float(row["e2e_ns"]) for row in rows]
            metrics[arm] = {
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
        c = metrics["independent_composition"]
        d = metrics["stitched_composition"]
        shapes.append({
            "prompt_tokens": prompt_tokens,
            "arms": metrics,
            "d_vs_c_e2e_improvement_fraction": _gain(
                d["e2e_median_ns"], c["e2e_median_ns"]
            ),
            "d_vs_c_gap_improvement_fraction": _gain(
                d["token_0_to_1_gap_median_ns"],
                c["token_0_to_1_gap_median_ns"],
            ),
            "d_vs_c_ttft_regression_fraction": _regression(
                d["ttft_median_ns"], c["ttft_median_ns"]
            ),
            "d_vs_c_p95_e2e_regression_fraction": _regression(
                d["e2e_p95_ns"], c["e2e_p95_ns"]
            ),
            "d_vs_c_p99_e2e_regression_fraction": _regression(
                d["e2e_p99_ns"], c["e2e_p99_ns"]
            ),
            "d_vs_c_peak_reserved_memory_regression_fraction": _regression(
                d["peak_reserved_bytes"], c["peak_reserved_bytes"]
            ),
            "a_vs_b_prefill_attribution_fraction": _gain(
                metrics["prefill_only"]["e2e_median_ns"],
                metrics["eager"]["e2e_median_ns"],
            ),
        })
    c_e2e = [
        float(row["e2e_ns"])
        for shape in buckets.values()
        for row in shape["independent_composition"]
    ]
    d_e2e = [
        float(row["e2e_ns"])
        for shape in buckets.values()
        for row in shape["stitched_composition"]
    ]
    c_gap = [
        float(row["token_0_to_1_gap_ns"])
        for shape in buckets.values()
        for row in shape["independent_composition"]
    ]
    d_gap = [
        float(row["token_0_to_1_gap_ns"])
        for shape in buckets.values()
        for row in shape["stitched_composition"]
    ]
    aggregate = {
        "d_vs_c_e2e_improvement_fraction": _gain(
            statistics.median(d_e2e), statistics.median(c_e2e)
        ),
        "d_vs_c_token_0_to_1_gap_improvement_fraction": _gain(
            statistics.median(d_gap), statistics.median(c_gap)
        ),
    }
    checks = {
        "complete_case_inventory": len(expected_result_paths)
        == len(contract.expected_case_ids()),
        "complete_evidence": evidence,
        "all_token_ids_exact": exact_token_ids,
        "all_text_exact": exact_text,
        "complete_accounting": mechanism,
        "zero_failure_and_quarantine": mechanism,
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
    if not checks["complete_evidence"]:
        classification = "NO_GO_EVIDENCE"
    elif not checks["all_token_ids_exact"] or not checks["all_text_exact"]:
        classification = "NO_GO_CORRECTNESS"
    elif not checks["complete_accounting"]:
        classification = "NO_GO_MECHANISM"
    elif not all((
        checks["shape_e2e_gain"],
        checks["aggregate_e2e_gain"],
        checks["gap_gain"],
        checks["ttft_non_regression"],
        checks["tail_non_regression"],
        checks["memory_non_regression"],
    )):
        classification = "NO_GO_PERFORMANCE"
    else:
        classification = "GO_PHASE_STITCHED_EXACT_GRAPH"
    summary = {
        "schema_version": contract.SUMMARY_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run["run_tag"],
        "classification": classification,
        "correctness": {
            "pair_count": len(pairs),
            "all_token_ids_exact": exact_token_ids,
            "all_text_exact": exact_text,
        },
        "mechanism": {
            "complete_accounting": mechanism,
            "zero_failure_and_quarantine": mechanism,
        },
        "performance": {"shapes": shapes, "aggregate": aggregate},
        "checks": checks,
        "evidence_errors": [],
    }
    return run, summary, {
        "schema_version": contract.GATE_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run["run_tag"],
        "classification": classification,
        "checks": checks,
        "summary_sha256": contract.canonical_json_sha256(summary),
        "correctness": summary["correctness"],
        "mechanism": summary["mechanism"],
        "performance": summary["performance"],
        "evidence_errors": [],
    }


def verify_artifact_directory(run_dir: Path) -> dict:
    root = Path(run_dir)
    manifest = _read(root / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != contract.MANIFEST_SCHEMA_VERSION
        or manifest.get("contract_sha256") != contract.contract_sha256()
    ):
        raise ValueError("manifest metadata is invalid")
    expected_paths = (
        {"run_manifest.json", "summary.json", "gate.json"}
        | {
            f"cases/{case_id}/result.json"
            for case_id in contract.expected_case_ids()
        }
    )
    if set(manifest.get("files", {})) != expected_paths:
        raise ValueError("manifest file inventory drifted")
    for relative, expected_hash in manifest["files"].items():
        if _sha(root / relative) != expected_hash:
            raise ValueError(f"manifest hash mismatch: {relative}")
    _run, expected_summary, expected_gate = _reconstruct(root)
    if _read(root / "summary.json") != expected_summary:
        raise ValueError("summary does not match raw reconstruction")
    if _read(root / "gate.json") != expected_gate:
        raise ValueError("gate does not match raw reconstruction")
    producer = _read(root / "producer_receipt.json")
    if (
        not isinstance(producer, dict)
        or producer.get("schema_version") != contract.RECEIPT_SCHEMA_VERSION
        or producer.get("classification") != expected_gate["classification"]
        or producer.get("manifest_sha256") != _sha(root / "manifest.json")
    ):
        raise ValueError("producer receipt is invalid")
    receipt = {
        "schema_version": contract.RECEIPT_SCHEMA_VERSION,
        "verified": True,
        "classification": expected_gate["classification"],
        "raw_metrics_reconstructed": True,
        "manifest_sha256": _sha(root / "manifest.json"),
    }
    _write(root / "independent_verifier_receipt.json", receipt)
    return receipt


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    receipt = verify_artifact_directory(_parse_args(argv).run_dir)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
