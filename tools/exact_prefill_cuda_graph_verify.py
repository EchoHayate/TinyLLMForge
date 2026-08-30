#!/usr/bin/env python3
"""Independent artifact-only verifier for the exact-prefill graph gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from tools import exact_prefill_cuda_graph_benchmark_contract as contract


GO = "GO_EXACT_PREFILL_GRAPH"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_MECHANISM = "NO_GO_MECHANISM"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _verify_manifest(root: Path) -> dict:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != contract.MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest schema mismatch")
    artifacts = manifest.get("artifacts")
    expected = {
        *(f"cases/{case_id}/result.json"
          for case_id in contract.expected_case_ids()),
        "run_manifest.json",
        "comparison.json",
        "summary.json",
        "report.md",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("manifest artifact inventory mismatch")
    for relative, digest in artifacts.items():
        path = root / relative
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or len(digest) != 64
            or _sha256_path(path) != digest
        ):
            raise ValueError(f"manifest hash mismatch: {relative}")
    return manifest


def _verify_run_manifest(root: Path) -> dict:
    manifest = _read_json(root / "run_manifest.json")
    if (
        manifest.get("schema_version") != contract.RUN_SCHEMA_VERSION
        or manifest.get("case_order")
        != list(contract.expected_case_ids())
        or manifest.get("contract_sha256")
        != contract.contract_sha256()
        or manifest.get("clean_gpu_admission") is not True
        or re.fullmatch(
            r"[0-9a-f]{40}",
            manifest.get("source_base_commit", ""),
        )
        is None
    ):
        raise ValueError("run manifest identity mismatch")
    source_files = manifest.get("source_files")
    if (
        not isinstance(source_files, dict)
        or set(source_files) != set(contract.SOURCE_FILES)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value or "") is None
            for value in source_files.values()
        )
    ):
        raise ValueError("run manifest source inventory mismatch")
    inventory = manifest.get("gpu_inventory")
    selected_gpu = manifest.get("cuda_visible_devices")
    if (
        not isinstance(selected_gpu, str)
        or re.fullmatch(r"[0-9]+", selected_gpu) is None
        or
        not isinstance(inventory, list)
        or len(inventory) != 1
        or inventory[0].get("index") != int(selected_gpu)
        or inventory[0].get("memory_used_mb") != 0
        or inventory[0].get("utilization_gpu_percent") != 0
        or inventory[0].get("compute_processes") != []
    ):
        raise ValueError("run manifest GPU admission mismatch")
    return manifest


def _load_results(root: Path) -> list[dict]:
    results = []
    for case in contract.build_case_matrix():
        result = _read_json(
            root / "cases" / case["case_id"] / "result.json"
        )
        if (
            result.get("schema_version") != contract.RESULT_SCHEMA_VERSION
            or result.get("case") != case
            or not isinstance(result.get("rows"), list)
            or len(result["rows"]) != contract.MEASURED_REPETITIONS
        ):
            raise ValueError("raw case contract mismatch")
        results.append(result)
    return results


def _classify(results: list[dict]) -> dict:
    by_key = {}
    incomplete = []
    mechanism_failures = []
    cost_fields = (
        ("capture_duration_ns", "total_capture_ns"),
        ("static_bytes", "static_bytes"),
        ("allocated_delta_bytes", "allocated_delta_bytes"),
        ("reserved_delta_bytes", "reserved_delta_bytes"),
    )
    costs = {
        output_name: [] for output_name, _ in cost_fields
    }
    for result in results:
        case = result["case"]
        summary = result.get("prefill_graph_summary")
        if not isinstance(summary, dict):
            incomplete.append(f"{case['case_id']}:graph_summary")
            continue
        graph = case["arm"] == "exact_prefill_graph"
        if graph:
            for output_field, source_field in cost_fields:
                value = summary.get(source_field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    incomplete.append(f"cost:{output_field}")
                else:
                    costs[output_field].append(value)
            if (
                summary.get("capture_successes") != 2
                or summary.get("capture_failures") != 0
                or summary.get("replay_failures") != 0
                or summary.get("quarantines") != 0
            ):
                mechanism_failures.append(case["case_id"])
        elif (
            summary.get("capture_successes") != 0
            or summary.get("replays") != 0
        ):
            mechanism_failures.append(case["case_id"])
        for sample_index, row in enumerate(result["rows"]):
            key = (
                case["round"],
                case["prompt_tokens"],
                sample_index,
                case["arm"],
            )
            if key in by_key:
                incomplete.append("duplicate_row")
            tpot_samples = row.get("tpot_samples_ns")
            token_ids = row.get("output_token_ids")
            prompt_sha256 = row.get("prompt_sha256")
            output_text_sha256 = row.get("output_text_sha256")
            if (
                row.get("schema_version") != contract.ROW_SCHEMA_VERSION
                or row.get("case_id") != case["case_id"]
                or row.get("round") != case["round"]
                or row.get("arm") != case["arm"]
                or row.get("prompt_tokens") != case["prompt_tokens"]
                or row.get("sample_index") != sample_index
                or row.get("generated_tokens")
                != contract.GENERATED_TOKENS
                or not isinstance(token_ids, list)
                or len(token_ids)
                != contract.GENERATED_TOKENS
                or any(
                    isinstance(token, bool) or not isinstance(token, int)
                    for token in token_ids
                )
                or not isinstance(tpot_samples, list)
                or len(tpot_samples)
                != contract.GENERATED_TOKENS - 1
                or not isinstance(prompt_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", prompt_sha256)
                is None
                or not isinstance(output_text_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", output_text_sha256)
                is None
            ):
                incomplete.append(f"{case['case_id']}:row_contract")
            try:
                contract.positive_int(row.get("ttft_ns"), "ttft_ns")
                contract.positive_int(row.get("e2e_ns"), "e2e_ns")
                contract.nonnegative_int(
                    row.get("cuda_peak_allocated_bytes"),
                    "peak allocated bytes",
                )
                contract.nonnegative_int(
                    row.get("cuda_peak_reserved_bytes"),
                    "peak reserved bytes",
                )
                for value in tpot_samples or ():
                    if contract.finite_number(value, "tpot") <= 0:
                        raise ValueError("tpot must be positive")
            except ValueError as error:
                incomplete.append(f"{case['case_id']}:{error}")
            expected_replay = 1 if graph else 0
            if row.get("prefill_graph_replay_delta") != expected_replay:
                mechanism_failures.append(case["case_id"])
            by_key[key] = row
    pairs = []
    try:
        for round_index in range(contract.ROUNDS):
            for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
                for sample_index in range(
                    contract.MEASURED_REPETITIONS
                ):
                    pairs.append((
                        by_key[(
                            round_index,
                            prompt_tokens,
                            sample_index,
                            "eager",
                        )],
                        by_key[(
                            round_index,
                            prompt_tokens,
                            sample_index,
                            "exact_prefill_graph",
                        )],
                    ))
    except KeyError:
        incomplete.append("paired_row_inventory")
        pairs = []
    if incomplete:
        pairs = []
    correctness_mismatches = [
        {
            "round": eager["round"],
            "prompt_tokens": eager["prompt_tokens"],
            "sample_index": eager["sample_index"],
        }
        for eager, graph in pairs
        if (
            eager.get("prompt_sha256") != graph.get("prompt_sha256")
            or eager.get("output_token_ids")
            != graph.get("output_token_ids")
            or eager.get("output_text_sha256")
            != graph.get("output_text_sha256")
        )
    ]
    performance = {}
    failures = []
    for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
        selected = [
            pair
            for pair in pairs
            if pair[0]["prompt_tokens"] == prompt_tokens
        ]
        if not selected:
            continue
        eager_ttft = statistics.median(
            row["ttft_ns"] for row, _ in selected
        )
        graph_ttft = statistics.median(
            row["ttft_ns"] for _, row in selected
        )
        eager_tpot = statistics.median(
            value
            for row, _ in selected
            for value in row["tpot_samples_ns"]
        )
        graph_tpot = statistics.median(
            value
            for _, row in selected
            for value in row["tpot_samples_ns"]
        )
        eager_e2e = statistics.median(
            row["e2e_ns"] for row, _ in selected
        )
        graph_e2e = statistics.median(
            row["e2e_ns"] for _, row in selected
        )
        metrics = {
            "sample_count_per_arm": len(selected),
            "eager_ttft_median_ns": eager_ttft,
            "graph_ttft_median_ns": graph_ttft,
            "ttft_improvement_fraction":
                1.0 - graph_ttft / eager_ttft,
            "ttft_regression_fraction":
                graph_ttft / eager_ttft - 1.0,
            "eager_tpot_median_ns": eager_tpot,
            "graph_tpot_median_ns": graph_tpot,
            "tpot_regression_fraction":
                graph_tpot / eager_tpot - 1.0,
            "eager_e2e_median_ns": eager_e2e,
            "graph_e2e_median_ns": graph_e2e,
            "e2e_regression_fraction":
                graph_e2e / eager_e2e - 1.0,
        }
        performance[str(prompt_tokens)] = metrics
        if (
            prompt_tokens == 256
            and metrics["ttft_improvement_fraction"]
            < contract.TTFT_256_IMPROVEMENT_MINIMUM
        ):
            failures.append("256:ttft")
        if (
            prompt_tokens == 2048
            and metrics["ttft_regression_fraction"]
            > contract.TTFT_2048_REGRESSION_LIMIT
        ):
            failures.append("2048:ttft")
        if (
            metrics["tpot_regression_fraction"]
            > contract.TPOT_REGRESSION_LIMIT
        ):
            failures.append(f"{prompt_tokens}:tpot")
        if (
            metrics["e2e_regression_fraction"]
            > contract.E2E_REGRESSION_LIMIT
        ):
            failures.append(f"{prompt_tokens}:e2e")
    if incomplete:
        classification = NO_GO_EVIDENCE_INCOMPLETE
    elif correctness_mismatches:
        classification = NO_GO_CORRECTNESS
    elif mechanism_failures:
        classification = NO_GO_MECHANISM
    elif failures:
        classification = NO_GO_PERFORMANCE
    else:
        classification = GO
    return {
        "classification": classification,
        "correctness": {
            "all_token_ids_exact": not correctness_mismatches,
            "mismatches": correctness_mismatches,
        },
        "mechanism": {
            "candidate_replayed_every_sample":
                not mechanism_failures,
            "failures": sorted(set(mechanism_failures)),
        },
        "performance": performance,
        "cost": {
            field: {
                "available": len(values) == 4,
                "median": statistics.median(values) if values else None,
                "maximum": max(values) if values else None,
                "samples": len(values),
            }
            for field, values in costs.items()
        },
        "performance_failures": failures,
        "incomplete_evidence": sorted(set(incomplete)),
    }


def verify_artifact_directory(run_dir: Path) -> dict:
    root = Path(run_dir)
    manifest = _verify_manifest(root)
    _verify_run_manifest(root)
    reconstructed = _classify(_load_results(root))
    comparison = _read_json(root / "comparison.json")
    summary = _read_json(root / "summary.json")
    if comparison.get("schema_version") != (
        contract.COMPARISON_SCHEMA_VERSION
    ):
        raise ValueError("comparison schema mismatch")
    if summary.get("schema_version") != contract.GATE_SCHEMA_VERSION:
        raise ValueError("summary schema mismatch")
    for field in (
        "classification",
        "correctness",
        "mechanism",
        "performance",
        "cost",
        "performance_failures",
        "incomplete_evidence",
    ):
        if comparison.get(field) != reconstructed[field]:
            raise ValueError(
                f"reconstructed comparison mismatch: {field}"
            )
        if summary.get(field) != reconstructed[field]:
            raise ValueError(
                f"reconstructed summary mismatch: {field}"
            )
    return {
        "verified": True,
        "classification": reconstructed["classification"],
        "manifest_verified": bool(manifest),
        "raw_metrics_reconstructed": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = verify_artifact_directory(args.run_dir)
    if args.output is not None:
        args.output.write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
