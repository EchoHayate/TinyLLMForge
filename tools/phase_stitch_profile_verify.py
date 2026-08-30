#!/usr/bin/env python3
"""Independent verifier for a Phase-Stitch profile evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools import phase_stitch_profile_contract as contract


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def _nearest_rank(values, percentile):
    if not values:
        raise ValueError("cannot verify an empty percentile sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _validate_run_manifest(run_manifest):
    if run_manifest.get("schema_version") != contract.RUN_SCHEMA_VERSION:
        raise ValueError("run manifest schema is invalid")
    if run_manifest.get("contract_sha256") != contract.contract_sha256():
        raise ValueError("run manifest contract hash drifted")
    if run_manifest.get("case_order") != list(
        contract.expected_case_ids()
    ):
        raise ValueError("run manifest case order drifted")
    source_files = run_manifest.get("source_files")
    if (
        not isinstance(source_files, dict)
        or set(source_files) != set(contract.SOURCE_FILES)
        or not all(_valid_sha256(value) for value in source_files.values())
    ):
        raise ValueError("run manifest source hashes are invalid")
    if run_manifest.get("clean_gpu_admission") is not True:
        raise ValueError("clean GPU admission is absent")
    inventory = run_manifest.get("gpu_inventory")
    if (
        not isinstance(inventory, list)
        or len(inventory) != 1
        or "A100" not in str(inventory[0].get("name", ""))
    ):
        raise ValueError("GPU inventory is invalid")


def _load_raw_results(root):
    cases_root = root / "cases"
    expected_ids = contract.expected_case_ids()
    actual_ids = {
        path.name for path in cases_root.iterdir() if path.is_dir()
    }
    if actual_ids != set(expected_ids):
        raise ValueError("case directory inventory is incomplete")
    results = []
    for case_id in expected_ids:
        result = _read_json(
            cases_root / case_id / "result.json"
        )
        contract.validate_case_result(result)
        results.append(result)
    return results


def _reconstruct(results, run_tag):
    rows_by_shape = {
        prompt_tokens: {
            "instrumentation_off": [],
            "instrumentation_on": [],
        }
        for prompt_tokens in contract.PROMPT_TOKEN_COUNTS
    }
    paired = {}
    graph_ok = True
    failures_ok = True
    for result in results:
        case = result["case"]
        rows_by_shape[case["prompt_tokens"]][case["arm"]].extend(
            result["rows"]
        )
        prefill = result["prefill_graph_summary"]
        burst = result["exact_burst_summary"]
        graph_ok = graph_ok and all(
            row["prefill_graph_replay_delta"] > 0
            and row["exact_burst_replay_delta"] > 0
            and row["exact_burst_acceptance_delta"] > 0
            for row in result["rows"]
        )
        failures_ok = failures_ok and (
            prefill.get("capture_failures") == 0
            and prefill.get("replay_failures") == 0
            and prefill.get("quarantines") == 0
            and burst.get("failures") == 0
            and burst.get("quarantines") == 0
            and burst.get("pending_leases") == 0
            and burst.get("quarantine_reason") is None
        )
        for row in result["rows"]:
            identity = (
                case["round"],
                case["prompt_tokens"],
                row["sample_index"],
                row["prompt_sha256"],
            )
            paired.setdefault(identity, {})[case["arm"]] = (
                row["output_token_ids_sha256"],
                row["output_text_sha256"],
            )
    outputs_equal = bool(paired) and all(
        set(arms) == set(contract.ARMS)
        and arms["instrumentation_off"]
        == arms["instrumentation_on"]
        for arms in paired.values()
    )
    shapes = []
    coverage_ok = True
    for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
        off_rows = rows_by_shape[prompt_tokens][
            "instrumentation_off"
        ]
        on_rows = rows_by_shape[prompt_tokens]["instrumentation_on"]
        expected_count = (
            contract.ROUNDS * contract.MEASURED_REPETITIONS
        )
        if len(off_rows) != expected_count or len(on_rows) != expected_count:
            raise ValueError("shape row inventory is incomplete")
        gaps = []
        for row in on_rows:
            profile = row["phase_stitch_profile"]
            coverage_ok = coverage_ok and (
                profile["event_coverage_complete"] is True
            )
            recomputed_gap = (
                profile["first_k8_dispatch_started_ns"]
                - profile["first_token_host_available_ns"]
            )
            if recomputed_gap != profile["removable_host_gap_ns"]:
                raise ValueError("profile gap does not reconstruct")
            gaps.append(recomputed_gap)
        off_median = statistics.median(
            row["e2e_ns"] for row in off_rows
        )
        on_median = statistics.median(
            row["e2e_ns"] for row in on_rows
        )
        gap_median = statistics.median(gaps)
        shapes.append({
            "prompt_tokens": prompt_tokens,
            "sample_count_per_arm": len(on_rows),
            "median_gap_ns": gap_median,
            "p95_gap_ns": _nearest_rank(gaps, 0.95),
            "profile_off_e2e_median_ns": off_median,
            "profile_on_e2e_median_ns": on_median,
            "median_gap_e2e_fraction": gap_median / on_median,
            "profile_e2e_overhead_fraction": (
                on_median / off_median - 1.0
            ),
        })
    ceiling_ok = (
        any(
            shape["median_gap_ns"]
            >= contract.MEDIAN_GAP_MINIMUM_NS
            for shape in shapes
        )
        and any(
            shape["median_gap_e2e_fraction"]
            >= contract.MEDIAN_GAP_E2E_FRACTION_MINIMUM
            or shape["p95_gap_ns"]
            >= contract.P95_GAP_MINIMUM_NS
            for shape in shapes
        )
    )
    overhead_ok = all(
        abs(shape["profile_e2e_overhead_fraction"])
        <= contract.PROFILE_E2E_OVERHEAD_LIMIT
        for shape in shapes
    )
    checks = {
        "complete_case_inventory": len(results)
        == len(contract.build_case_matrix()),
        "exact_output_equality": outputs_equal,
        "event_coverage_pass": coverage_ok,
        "graph_evidence_pass": graph_ok,
        "zero_failures_pass": failures_ok,
        "ceiling_pass": ceiling_ok,
        "overhead_pass": overhead_ok,
    }
    controls_ok = all(
        checks[name]
        for name in (
            "complete_case_inventory",
            "exact_output_equality",
            "event_coverage_pass",
            "graph_evidence_pass",
            "zero_failures_pass",
            "overhead_pass",
        )
    )
    classification = (
        "GO_PHASE_STITCH_PROFILE"
        if controls_ok and ceiling_ok
        else "NO_GO_PHASE_STITCH_CEILING"
    )
    summary = {
        "schema_version": "phase-stitch-profile.summary.v1",
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run_tag,
        "shapes": shapes,
        "checks": checks,
    }
    gate = {
        "schema_version": contract.GATE_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run_tag,
        "classification": classification,
        "checks": checks,
        "summary_sha256": contract.canonical_json_sha256(summary),
    }
    return summary, gate


def verify_bundle(run_dir):
    root = Path(run_dir)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != contract.MANIFEST_SCHEMA_VERSION:
        raise ValueError("evidence manifest schema is invalid")
    if manifest.get("contract_sha256") != contract.contract_sha256():
        raise ValueError("evidence manifest contract hash drifted")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("evidence manifest files are missing")
    expected_files = {
        "run_manifest.json",
        "summary.json",
        "gate.json",
        *{
            f"cases/{case_id}/result.json"
            for case_id in contract.expected_case_ids()
        },
    }
    if set(files) != expected_files:
        raise ValueError("evidence manifest file inventory drifted")
    for relative_path, expected_sha256 in files.items():
        if not _valid_sha256(expected_sha256):
            raise ValueError("evidence manifest hash is invalid")
        if _file_sha256(root / relative_path) != expected_sha256:
            raise ValueError(
                f"evidence file hash drifted: {relative_path}"
            )
    run_manifest = _read_json(root / "run_manifest.json")
    _validate_run_manifest(run_manifest)
    results = _load_raw_results(root)
    expected_summary, expected_gate = _reconstruct(
        results,
        run_manifest["run_tag"],
    )
    recorded_summary = _read_json(root / "summary.json")
    recorded_gate = _read_json(root / "gate.json")
    if recorded_summary != expected_summary:
        raise ValueError("recorded summary does not reconstruct")
    if recorded_gate != expected_gate:
        raise ValueError("recorded gate does not reconstruct")
    return {
        "verified": True,
        "classification": expected_gate["classification"],
        "contract_sha256": contract.contract_sha256(),
        "manifest_sha256": _file_sha256(root / "manifest.json"),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    result = verify_bundle(args.run_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
