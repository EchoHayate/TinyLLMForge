#!/usr/bin/env python3
"""Producer gate for exact-prefill CUDA Graph paired evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from tools import exact_prefill_cuda_graph_benchmark_contract as contract


GO_EXACT_PREFILL_GRAPH = "GO_EXACT_PREFILL_GRAPH"
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _ratio_regression(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return candidate / baseline - 1.0


def _improvement(baseline: float, candidate: float) -> float:
    return -_ratio_regression(baseline, candidate)


def _load_case_results(run_dir: Path) -> list[dict]:
    results = []
    for case in contract.build_case_matrix():
        path = run_dir / "cases" / case["case_id"] / "result.json"
        if not path.is_file():
            raise ValueError(f"missing case result: {case['case_id']}")
        result = _read_json(path)
        if (
            result.get("schema_version") != contract.RESULT_SCHEMA_VERSION
            or result.get("case") != case
        ):
            raise ValueError(f"case result contract mismatch: {case['case_id']}")
        results.append(result)
    return results


def _validate_run_manifest(root: Path) -> dict:
    manifest = _read_json(root / "run_manifest.json")
    if manifest.get("schema_version") != contract.RUN_SCHEMA_VERSION:
        raise ValueError("run manifest schema mismatch")
    if (
        not isinstance(manifest.get("run_tag"), str)
        or not manifest["run_tag"]
        or re.fullmatch(
            r"[0-9a-f]{40}",
            manifest.get("source_base_commit", ""),
        )
        is None
        or manifest.get("case_order")
        != list(contract.expected_case_ids())
        or manifest.get("contract_sha256")
        != contract.contract_sha256()
        or manifest.get("clean_gpu_admission") is not True
    ):
        raise ValueError("run manifest identity mismatch")
    source_files = manifest.get("source_files")
    if (
        not isinstance(source_files, dict)
        or set(source_files) != set(contract.SOURCE_FILES)
        or any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in source_files.values()
        )
    ):
        raise ValueError("run manifest source inventory mismatch")
    for field in ("model", "python"):
        if (
            not isinstance(manifest.get(field), str)
            or not manifest[field]
        ):
            raise ValueError(f"run manifest {field} is invalid")
    inventory = manifest.get("gpu_inventory")
    selected_gpu = manifest.get("cuda_visible_devices")
    if (
        not isinstance(selected_gpu, str)
        or re.fullmatch(r"[0-9]+", selected_gpu) is None
        or
        not isinstance(inventory, list)
        or len(inventory) != 1
        or not isinstance(inventory[0], dict)
        or inventory[0].get("index") != int(selected_gpu)
        or inventory[0].get("memory_used_mb") != 0
        or inventory[0].get("utilization_gpu_percent") != 0
        or inventory[0].get("compute_processes") != []
    ):
        raise ValueError("run manifest GPU admission mismatch")
    return manifest


def _validate_rows(results: list[dict]) -> tuple[list[dict], list[str]]:
    rows = []
    incomplete = []
    for result in results:
        case = result["case"]
        case_rows = result.get("rows")
        if (
            not isinstance(case_rows, list)
            or len(case_rows) != contract.MEASURED_REPETITIONS
        ):
            incomplete.append(f"{case['case_id']}:row_inventory")
            continue
        for sample_index, row in enumerate(case_rows):
            identity = (
                row.get("case_id"),
                row.get("round"),
                row.get("arm"),
                row.get("prompt_tokens"),
                row.get("sample_index"),
            )
            expected = (
                case["case_id"],
                case["round"],
                case["arm"],
                case["prompt_tokens"],
                sample_index,
            )
            if identity != expected:
                incomplete.append(f"{case['case_id']}:row_identity")
                continue
            prompt_sha256 = row.get("prompt_sha256")
            output_text_sha256 = row.get("output_text_sha256")
            if (
                row.get("schema_version")
                != contract.ROW_SCHEMA_VERSION
                or row.get("generated_tokens")
                != contract.GENERATED_TOKENS
                or not isinstance(prompt_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", prompt_sha256)
                is None
                or not isinstance(output_text_sha256, str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    output_text_sha256,
                )
                is None
            ):
                incomplete.append(f"{case['case_id']}:row_contract")
                continue
            try:
                contract.positive_int(row.get("ttft_ns"), "ttft_ns")
                contract.positive_int(row.get("e2e_ns"), "e2e_ns")
                samples = row.get("tpot_samples_ns")
                if (
                    not isinstance(samples, list)
                    or len(samples) != contract.GENERATED_TOKENS - 1
                ):
                    raise ValueError("tpot sample inventory mismatch")
                for value in samples:
                    if contract.finite_number(value, "tpot") <= 0:
                        raise ValueError("tpot must be positive")
                contract.nonnegative_int(
                    row.get("prefill_graph_replay_delta"),
                    "replay delta",
                )
                contract.nonnegative_int(
                    row.get("cuda_peak_allocated_bytes"),
                    "peak allocated bytes",
                )
                contract.nonnegative_int(
                    row.get("cuda_peak_reserved_bytes"),
                    "peak reserved bytes",
                )
                token_ids = row.get("output_token_ids")
                if (
                    not isinstance(token_ids, list)
                    or len(token_ids) != contract.GENERATED_TOKENS
                    or any(
                        isinstance(token, bool)
                        or not isinstance(token, int)
                        for token in token_ids
                    )
                ):
                    raise ValueError("output token inventory mismatch")
            except ValueError as error:
                incomplete.append(f"{case['case_id']}:{error}")
                continue
            rows.append(row)
    expected_count = (
        len(contract.build_case_matrix())
        * contract.MEASURED_REPETITIONS
    )
    if len(rows) != expected_count:
        incomplete.append("global_row_inventory")
    return rows, sorted(set(incomplete))


def _pair_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    by_key = {
        (
            row["round"],
            row["prompt_tokens"],
            row["sample_index"],
            row["arm"],
        ): row
        for row in rows
    }
    pairs = []
    for round_index in range(contract.ROUNDS):
        for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
            for sample_index in range(contract.MEASURED_REPETITIONS):
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
    return pairs


def _metric_summary(pairs: list[tuple[dict, dict]]) -> dict:
    result = {}
    for prompt_tokens in contract.PROMPT_TOKEN_COUNTS:
        selected = [
            pair
            for pair in pairs
            if pair[0]["prompt_tokens"] == prompt_tokens
        ]
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
        result[str(prompt_tokens)] = {
            "sample_count_per_arm": len(selected),
            "eager_ttft_median_ns": eager_ttft,
            "graph_ttft_median_ns": graph_ttft,
            "ttft_improvement_fraction": _improvement(
                eager_ttft,
                graph_ttft,
            ),
            "ttft_regression_fraction": _ratio_regression(
                eager_ttft,
                graph_ttft,
            ),
            "eager_tpot_median_ns": eager_tpot,
            "graph_tpot_median_ns": graph_tpot,
            "tpot_regression_fraction": _ratio_regression(
                eager_tpot,
                graph_tpot,
            ),
            "eager_e2e_median_ns": eager_e2e,
            "graph_e2e_median_ns": graph_e2e,
            "e2e_regression_fraction": _ratio_regression(
                eager_e2e,
                graph_e2e,
            ),
        }
    return result


def _cost(results: list[dict]) -> tuple[dict, list[str]]:
    fields = (
        ("capture_duration_ns", "total_capture_ns"),
        ("static_bytes", "static_bytes"),
        ("allocated_delta_bytes", "allocated_delta_bytes"),
        ("reserved_delta_bytes", "reserved_delta_bytes"),
    )
    candidate_summaries = [
        result.get("prefill_graph_summary")
        for result in results
        if result["case"]["arm"] == "exact_prefill_graph"
    ]
    incomplete = []
    output = {}
    for output_field, source_field in fields:
        values = []
        for summary in candidate_summaries:
            try:
                values.append(
                    contract.nonnegative_int(
                        summary.get(source_field),
                        source_field,
                    )
                )
            except (AttributeError, ValueError):
                incomplete.append(f"cost:{output_field}")
        output[output_field] = {
            "available": len(values) == len(candidate_summaries),
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None,
            "samples": len(values),
        }
    return output, sorted(set(incomplete))


def classify_results(results: list[dict]) -> dict:
    rows, incomplete = _validate_rows(results)
    try:
        pairs = _pair_rows(rows) if not incomplete else []
    except KeyError:
        pairs = []
        incomplete.append("paired_row_inventory")
    correctness_mismatches = []
    for eager, graph in pairs:
        if (
            eager["prompt_sha256"] != graph["prompt_sha256"]
            or eager["output_token_ids"] != graph["output_token_ids"]
            or eager["output_text_sha256"] != graph["output_text_sha256"]
        ):
            correctness_mismatches.append({
                "round": eager["round"],
                "prompt_tokens": eager["prompt_tokens"],
                "sample_index": eager["sample_index"],
            })
    mechanism_failures = []
    for result in results:
        summary = result.get("prefill_graph_summary")
        if not isinstance(summary, dict):
            incomplete.append(
                f"{result['case']['case_id']}:graph_summary"
            )
            continue
        if result["case"]["arm"] == "exact_prefill_graph":
            if (
                summary.get("capture_successes") != 2
                or summary.get("capture_failures") != 0
                or summary.get("replay_failures") != 0
                or summary.get("quarantines") != 0
                or any(
                    row["prefill_graph_replay_delta"] != 1
                    for row in result.get("rows", ())
                )
            ):
                mechanism_failures.append(result["case"]["case_id"])
        elif (
            summary.get("capture_successes") != 0
            or summary.get("replays") != 0
            or any(
                row["prefill_graph_replay_delta"] != 0
                for row in result.get("rows", ())
            )
        ):
            mechanism_failures.append(result["case"]["case_id"])
    performance = _metric_summary(pairs) if pairs else {}
    performance_failures = []
    if performance:
        if (
            performance["256"]["ttft_improvement_fraction"]
            < contract.TTFT_256_IMPROVEMENT_MINIMUM
        ):
            performance_failures.append("256:ttft")
        if (
            performance["2048"]["ttft_regression_fraction"]
            > contract.TTFT_2048_REGRESSION_LIMIT
        ):
            performance_failures.append("2048:ttft")
        for prompt_tokens in map(str, contract.PROMPT_TOKEN_COUNTS):
            if (
                performance[prompt_tokens]["tpot_regression_fraction"]
                > contract.TPOT_REGRESSION_LIMIT
            ):
                performance_failures.append(f"{prompt_tokens}:tpot")
            if (
                performance[prompt_tokens]["e2e_regression_fraction"]
                > contract.E2E_REGRESSION_LIMIT
            ):
                performance_failures.append(f"{prompt_tokens}:e2e")
    cost, cost_incomplete = _cost(results)
    incomplete.extend(cost_incomplete)
    if incomplete:
        classification = NO_GO_EVIDENCE_INCOMPLETE
    elif correctness_mismatches:
        classification = NO_GO_CORRECTNESS
    elif mechanism_failures:
        classification = NO_GO_MECHANISM
    elif performance_failures:
        classification = NO_GO_PERFORMANCE
    else:
        classification = GO_EXACT_PREFILL_GRAPH
    return {
        "schema_version": contract.COMPARISON_SCHEMA_VERSION,
        "classification": classification,
        "correctness": {
            "all_token_ids_exact": not correctness_mismatches,
            "mismatches": correctness_mismatches,
        },
        "mechanism": {
            "candidate_replayed_every_sample": not mechanism_failures,
            "failures": mechanism_failures,
        },
        "performance": performance,
        "cost": cost,
        "performance_failures": performance_failures,
        "incomplete_evidence": sorted(set(incomplete)),
    }


def _report(result: dict) -> str:
    lines = [
        "# Exact Prefill CUDA Graph Gate",
        "",
        f"Classification: `{result['classification']}`",
        "",
        "## Benefit and cost",
        "",
    ]
    for prompt_tokens in map(str, contract.PROMPT_TOKEN_COUNTS):
        metrics = result["performance"].get(prompt_tokens, {})
        lines.append(
            f"- {prompt_tokens} tokens: TTFT improvement "
            f"{metrics.get('ttft_improvement_fraction')!r}; "
            f"TPOT regression "
            f"{metrics.get('tpot_regression_fraction')!r}; "
            f"E2E regression "
            f"{metrics.get('e2e_regression_fraction')!r}."
        )
    lines.extend([
        (
            "- Startup capture duration median ns: "
            f"{result['cost']['capture_duration_ns']['median']!r}."
        ),
        (
            "- Reserved-memory delta maximum bytes: "
            f"{result['cost']['reserved_delta_bytes']['maximum']!r}."
        ),
        "",
    ])
    return "\n".join(lines)


def produce_gate(run_dir: Path) -> dict:
    root = Path(run_dir)
    _validate_run_manifest(root)
    results = _load_case_results(root)
    comparison = classify_results(results)
    _write_json(root / "comparison.json", comparison)
    summary = {
        **comparison,
        "schema_version": contract.GATE_SCHEMA_VERSION,
    }
    _write_json(root / "summary.json", summary)
    (root / "report.md").write_text(
        _report(comparison),
        encoding="utf-8",
    )
    artifact_paths = sorted(
        [
            *root.glob("cases/*/result.json"),
            root / "run_manifest.json",
            root / "comparison.json",
            root / "summary.json",
            root / "report.md",
        ],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    manifest = {
        "schema_version": contract.MANIFEST_SCHEMA_VERSION,
        "artifacts": {
            path.relative_to(root).as_posix(): _sha256_path(path)
            for path in artifact_paths
        },
    }
    _write_json(root / "manifest.json", manifest)
    return comparison


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = produce_gate(args.run_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["classification"] == GO_EXACT_PREFILL_GRAPH else 1


if __name__ == "__main__":
    raise SystemExit(main())
