#!/usr/bin/env python3
"""Produce the source-bound Phase-Stitch profile gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from tools import phase_stitch_profile_contract as contract


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write_json(path, value):
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


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")


def _validate_run_manifest(run_manifest):
    if not isinstance(run_manifest, dict):
        raise ValueError("run manifest must be an object")
    if run_manifest.get("schema_version") != contract.RUN_SCHEMA_VERSION:
        raise ValueError("run manifest schema is invalid")
    if run_manifest.get("contract_sha256") != contract.contract_sha256():
        raise ValueError("run manifest contract hash drifted")
    if run_manifest.get("case_order") != list(
        contract.expected_case_ids()
    ):
        raise ValueError("run manifest case inventory drifted")
    source_files = run_manifest.get("source_files")
    if (
        not isinstance(source_files, dict)
        or set(source_files) != set(contract.SOURCE_FILES)
    ):
        raise ValueError("run manifest source inventory is incomplete")
    for relative_path, digest in source_files.items():
        _require_sha256(digest, f"source hash for {relative_path}")
    if run_manifest.get("clean_gpu_admission") is not True:
        raise ValueError("clean GPU admission is absent")
    gpu_inventory = run_manifest.get("gpu_inventory")
    if not isinstance(gpu_inventory, list) or len(gpu_inventory) != 1:
        raise ValueError("GPU inventory must contain exactly one GPU")
    if "A100" not in str(gpu_inventory[0].get("name", "")):
        raise ValueError("GPU inventory does not contain an A100")


def _percentile_nearest_rank(values, percentile):
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _load_results(run_dir):
    root = Path(run_dir)
    run_manifest = _read_json(root / "run_manifest.json")
    _validate_run_manifest(run_manifest)
    cases_root = root / "cases"
    expected = contract.build_case_matrix()
    actual_dirs = {
        path.name
        for path in cases_root.iterdir()
        if path.is_dir()
    }
    if actual_dirs != set(contract.expected_case_ids()):
        raise ValueError("case directory inventory is incomplete")
    results = []
    for case in expected:
        result = _read_json(
            cases_root / case["case_id"] / "result.json"
        )
        contract.validate_case_result(result)
        results.append(result)
    return run_manifest, results


def _aggregate(results):
    paired_outputs = {}
    shape_rows = {
        prompt_tokens: {
            "instrumentation_off": [],
            "instrumentation_on": [],
        }
        for prompt_tokens in contract.PROMPT_TOKEN_COUNTS
    }
    graph_evidence_pass = True
    zero_failures_pass = True
    for result in results:
        case = result["case"]
        arm = case["arm"]
        prompt_tokens = case["prompt_tokens"]
        shape_rows[prompt_tokens][arm].extend(result["rows"])
        prefill_summary = result["prefill_graph_summary"]
        burst_summary = result["exact_burst_summary"]
        graph_evidence_pass = graph_evidence_pass and all(
            row["prefill_graph_replay_delta"] > 0
            and row["exact_burst_replay_delta"] > 0
            and row["exact_burst_acceptance_delta"] > 0
            for row in result["rows"]
        )
        zero_failures_pass = zero_failures_pass and (
            prefill_summary.get("capture_failures") == 0
            and prefill_summary.get("replay_failures") == 0
            and prefill_summary.get("quarantines") == 0
            and burst_summary.get("failures") == 0
            and burst_summary.get("quarantines") == 0
            and burst_summary.get("pending_leases") == 0
            and burst_summary.get("quarantine_reason") is None
        )
        for row in result["rows"]:
            key = (
                case["round"],
                prompt_tokens,
                row["sample_index"],
                row["prompt_sha256"],
            )
            paired_outputs.setdefault(key, {})[arm] = (
                row["output_token_ids_sha256"],
                row["output_text_sha256"],
            )
    exact_output_equality = bool(paired_outputs) and all(
        set(arms) == set(contract.ARMS)
        and arms["instrumentation_off"]
        == arms["instrumentation_on"]
        for arms in paired_outputs.values()
    )
    shapes = []
    for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
        off_rows = shape_rows[prompt_tokens]["instrumentation_off"]
        on_rows = shape_rows[prompt_tokens]["instrumentation_on"]
        if (
            len(off_rows)
            != contract.ROUNDS * contract.MEASURED_REPETITIONS
            or len(on_rows)
            != contract.ROUNDS * contract.MEASURED_REPETITIONS
        ):
            raise ValueError("shape row inventory is incomplete")
        gaps = [
            row["phase_stitch_profile"]["removable_host_gap_ns"]
            for row in on_rows
        ]
        off_e2e = [row["e2e_ns"] for row in off_rows]
        on_e2e = [row["e2e_ns"] for row in on_rows]
        median_gap_ns = statistics.median(gaps)
        p95_gap_ns = _percentile_nearest_rank(gaps, 0.95)
        off_e2e_median_ns = statistics.median(off_e2e)
        on_e2e_median_ns = statistics.median(on_e2e)
        shapes.append({
            "prompt_tokens": prompt_tokens,
            "sample_count_per_arm": len(on_rows),
            "median_gap_ns": median_gap_ns,
            "p95_gap_ns": p95_gap_ns,
            "profile_off_e2e_median_ns": off_e2e_median_ns,
            "profile_on_e2e_median_ns": on_e2e_median_ns,
            "median_gap_e2e_fraction": (
                median_gap_ns / on_e2e_median_ns
            ),
            "profile_e2e_overhead_fraction": (
                on_e2e_median_ns / off_e2e_median_ns - 1.0
            ),
        })
    ceiling_pass = (
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
    overhead_pass = all(
        abs(shape["profile_e2e_overhead_fraction"])
        <= contract.PROFILE_E2E_OVERHEAD_LIMIT
        for shape in shapes
    )
    event_coverage_pass = all(
        row["phase_stitch_profile"]["event_coverage_complete"]
        is True
        for prompt_rows in shape_rows.values()
        for row in prompt_rows["instrumentation_on"]
    )
    checks = {
        "complete_case_inventory": len(results)
        == len(contract.build_case_matrix()),
        "exact_output_equality": exact_output_equality,
        "event_coverage_pass": event_coverage_pass,
        "graph_evidence_pass": graph_evidence_pass,
        "zero_failures_pass": zero_failures_pass,
        "ceiling_pass": ceiling_pass,
        "overhead_pass": overhead_pass,
    }
    return shapes, checks


def _build_outputs(run_manifest, results):
    shapes, checks = _aggregate(results)
    controls_pass = all(
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
        if controls_pass and checks["ceiling_pass"]
        else "NO_GO_PHASE_STITCH_CEILING"
    )
    summary = {
        "schema_version": "phase-stitch-profile.summary.v1",
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run_manifest["run_tag"],
        "shapes": shapes,
        "checks": checks,
    }
    gate = {
        "schema_version": contract.GATE_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "run_tag": run_manifest["run_tag"],
        "classification": classification,
        "checks": checks,
        "summary_sha256": contract.canonical_json_sha256(summary),
    }
    return summary, gate


def produce_gate(run_dir):
    root = Path(run_dir)
    run_manifest, results = _load_results(root)
    summary, gate = _build_outputs(run_manifest, results)
    _atomic_write_json(root / "summary.json", summary)
    _atomic_write_json(root / "gate.json", gate)
    manifest_paths = [
        Path("run_manifest.json"),
        *[
            Path("cases") / case_id / "result.json"
            for case_id in contract.expected_case_ids()
        ],
        Path("summary.json"),
        Path("gate.json"),
    ]
    manifest = {
        "schema_version": contract.MANIFEST_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256(),
        "files": {
            path.as_posix(): _file_sha256(root / path)
            for path in manifest_paths
        },
    }
    _atomic_write_json(root / "manifest.json", manifest)
    return gate


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    result = produce_gate(args.run_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
