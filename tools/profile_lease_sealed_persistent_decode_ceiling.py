#!/usr/bin/env python3
"""Produce source-bound evidence for the persistent-decode ceiling gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

from tools.lease_sealed_persistent_decode_ceiling import (
    CONTEXT_LENGTHS,
    GENERATED_TOKENS,
    REPETITIONS,
    TIMING_SCHEMA_VERSION,
    TRACE_SUMMARY_SCHEMA_VERSION,
    compute_ceiling,
)
from tools.persistent_decode_kernel_trace import (
    build_candidate_segments,
    classify_kernel_rows,
    read_decode_trace,
    summarize_trace_coverage,
)
from tools.profile_exact_greedy_decode_burst import (
    _aggregate_memory,
    _combined_summary,
    _construct_llm as _construct_exact_llm,
    _make_prompt,
    _run_request as _run_exact_request,
    _runner_summaries,
    sha256_file,
    sha256_text,
)


STRUCTURAL_SCHEMA_VERSION = (
    "lease-sealed-persistent-decode.structural.v1"
)
SOURCE_SCHEMA_VERSION = "lease-sealed-persistent-decode.source.v1"
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/persistent_decode_kernel_trace.py",
    "tools/test_persistent_decode_kernel_trace.py",
    "tools/lease_sealed_persistent_decode_ceiling.py",
    "tools/test_lease_sealed_persistent_decode_ceiling.py",
    "tools/profile_lease_sealed_persistent_decode_ceiling.py",
    "tools/test_profile_lease_sealed_persistent_decode_ceiling.py",
    "tools/verify_lease_sealed_persistent_decode_ceiling.py",
    "tools/test_verify_lease_sealed_persistent_decode_ceiling.py",
    "tools/run_lease_sealed_persistent_decode_ceiling_remote.py",
    "tools/test_run_lease_sealed_persistent_decode_ceiling_remote.py",
    (
        "docs/superpowers/specs/"
        "2026-08-30-lease-sealed-persistent-decode-"
        "megakernel-ceiling-design.md"
    ),
    (
        "docs/superpowers/plans/"
        "2026-08-30-lease-sealed-persistent-decode-"
        "megakernel-ceiling.md"
    ),
)
MANIFEST_FILES = (
    "source_manifest.json",
    "runtime_manifest.json",
    "gpu_admission.json",
    "workload_manifest.json",
    "timing_rows.jsonl",
    "structural_rows.jsonl",
    "timing_summary.json",
    "trace_inventory.json",
    "kernel_rows.jsonl",
    "segment_rows.jsonl",
    "ceiling.json",
)


def build_timing_identities() -> tuple[tuple[int, int], ...]:
    return tuple(
        (repetition, context)
        for repetition in range(REPETITIONS)
        for context in CONTEXT_LENGTHS
    )


def build_trace_identities() -> tuple[int, int, int]:
    return CONTEXT_LENGTHS


def _identity_value(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or "/" in value:
        raise ValueError(f"{field} is invalid")
    return value


def _integer(value: int, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _digest(value: str, field: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def trace_label(
    *,
    attempt: str,
    workload: str,
    repetition: int,
    context: int,
    burst: int,
    logical_tokens: int,
) -> str:
    values = (
        ("attempt", _identity_value(attempt, "attempt")),
        ("workload", _identity_value(workload, "workload")),
        ("repetition", _integer(repetition, "repetition", minimum=0)),
        ("context", _integer(context, "context", minimum=1)),
        ("burst", _integer(burst, "burst", minimum=0)),
        (
            "logical_tokens",
            _integer(logical_tokens, "logical_tokens", minimum=1),
        ),
    )
    return "persistent_decode_trace/" + "/".join(
        f"{name}={value}" for name, value in values
    )


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile input cannot be empty")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _summary_fields(summary: dict) -> dict:
    fallback_counts = summary.get("fallback_counts")
    fallback_count = (
        int(summary.get("fallback_count", 0))
        if fallback_counts is None
        else sum(int(value) for value in fallback_counts.values())
    )
    return {
        "target_model_forwards": int(summary["target_model_forwards"]),
        "committed_tokens": int(summary["committed_tokens"]),
        "fallback_count": fallback_count,
        "failure_count": int(
            summary.get("failure_count", summary.get("failures", 0))
        ),
        "rollback_count": int(
            summary.get("rollback_count", summary.get("rollbacks", 0))
        ),
        "quarantine_reason": summary.get("quarantine_reason"),
    }


def _validate_fixed_shape(
    *,
    prompt_tokens: int,
    generated_tokens: int,
) -> None:
    if prompt_tokens not in CONTEXT_LENGTHS:
        raise ValueError("context length is outside the frozen inventory")
    if generated_tokens != GENERATED_TOKENS:
        raise ValueError("generated token count must remain frozen at 128")


def run_timing_case(
    *,
    model: str,
    run_tag: str,
    source_commit: str,
    source_tree_sha256: str,
    runtime_identity_sha256: str,
    workload_identity_sha256: str,
    repetition: int,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
) -> dict:
    _validate_fixed_shape(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
    )
    _digest(source_commit, "source_commit", length=40)
    for field, value in (
        ("source_tree_sha256", source_tree_sha256),
        ("runtime_identity_sha256", runtime_identity_sha256),
        ("workload_identity_sha256", workload_identity_sha256),
    ):
        _digest(value, field)
    _integer(repetition, "repetition", minimum=0)
    llm = _construct_exact_llm(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        policy="decode_burst_k8",
    )
    try:
        for warmup_index in range(2):
            _run_exact_request(
                llm,
                prompt=_make_prompt(
                    prompt_tokens,
                    offset=50_000 + warmup_index * 2_003,
                ),
                generated_tokens=generated_tokens,
                policy="decode_burst_k8",
                profile_label=None,
            )
            llm.clear_reusable_prefix_cache()
        before = _runner_summaries(llm)
        llm.reset_peak_memory_stats(timeout_s=60.0)
        measured = _run_exact_request(
            llm,
            prompt=_make_prompt(
                prompt_tokens,
                offset=repetition * 10_007,
            ),
            generated_tokens=generated_tokens,
            policy="decode_burst_k8",
            profile_label=None,
        )
        memory = _aggregate_memory(
            llm.memory_snapshots(timeout_s=60.0)
        )
        summary = _combined_summary(llm, before)
        samples = list(measured["amortized_tpot_samples_ns"])
        if len(samples) != generated_tokens - 1:
            raise RuntimeError("amortized TPOT inventory mismatch")
        return {
            "schema_version": TIMING_SCHEMA_VERSION,
            "arm": "uninstrumented",
            "run_tag": run_tag,
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256,
            "runtime_identity_sha256": runtime_identity_sha256,
            "workload_identity_sha256": workload_identity_sha256,
            "repetition": repetition,
            "context_length": prompt_tokens,
            "generated_tokens": generated_tokens,
            "output_token_ids": list(measured["output_token_ids"]),
            "output_text_sha256": sha256_text(measured["output_text"]),
            "ttft_ns": int(measured["ttft_ns"]),
            "e2e_ns": int(measured["e2e_ns"]),
            "tpot_samples_ns": samples,
            "tpot_median_ns": statistics.median(samples),
            "tpot_p95_ns": _nearest_rank(samples, 0.95),
            **memory,
            **_summary_fields(summary),
        }
    finally:
        llm.exit()


def _sampling_params(generated_tokens: int):
    from tinyvllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=generated_tokens,
        ignore_eos=True,
    )


def _cuda_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _default_range_factory(label: str):
    import torch

    return torch.cuda.nvtx.range(label)


def _run_structural_request(
    llm,
    *,
    prompt: list[int],
    generated_tokens: int,
    run_tag: str,
    context: int,
    range_factory=_default_range_factory,
    clock_ns=time.perf_counter_ns,
) -> dict:
    llm.add_request(prompt, _sampling_params(generated_tokens))
    started_ns = clock_ns()
    first_token_ns = None
    emitted_total = 0
    burst_ordinal = 0
    burst_tokens = []
    tpot_samples = []
    final_outputs = None
    while not llm.is_finished():
        expected_emitted = (
            1
            if emitted_total == 0
            else min(8, generated_tokens - emitted_total)
        )
        if emitted_total == 0:
            step_started_ns = clock_ns()
            outputs, _num_tokens = llm.step(completion_only=True)
            _cuda_synchronize()
            step_finished_ns = clock_ns()
        else:
            _cuda_synchronize()
            step_started_ns = clock_ns()
            label = trace_label(
                attempt=run_tag,
                workload="exact_greedy_k8",
                repetition=0,
                context=context,
                burst=burst_ordinal,
                logical_tokens=expected_emitted,
            )
            with range_factory(label):
                outputs, _num_tokens = llm.step(completion_only=True)
                _cuda_synchronize()
            step_finished_ns = clock_ns()
        observation = llm.last_step_observation
        emitted = sum(
            len(tokens)
            for tokens in observation[
                "new_completion_tokens_by_seq"
            ].values()
        )
        if emitted != expected_emitted:
            raise RuntimeError(
                "expected-versus-actual emitted-token mismatch"
            )
        emitted_total += emitted
        if first_token_ns is None:
            first_token_ns = step_finished_ns
        else:
            burst_tokens.append(emitted)
            tpot_samples.extend(
                [(step_finished_ns - step_started_ns) / emitted] * emitted
            )
            burst_ordinal += 1
        if outputs:
            final_outputs = outputs
    finished_ns = clock_ns()
    if first_token_ns is None:
        raise RuntimeError("request produced no first token")
    if not isinstance(final_outputs, list) or len(final_outputs) != 1:
        raise RuntimeError("request completion output is incomplete")
    output_ids = list(final_outputs[0][1])
    if len(output_ids) != generated_tokens:
        raise RuntimeError("generated token inventory mismatch")
    if len(tpot_samples) != generated_tokens - 1:
        raise RuntimeError("profiled TPOT inventory mismatch")
    return {
        "output_token_ids": output_ids,
        "output_text": llm.tokenizer.decode(output_ids),
        "ttft_ns": first_token_ns - started_ns,
        "e2e_ns": finished_ns - started_ns,
        "tpot_samples_ns": tpot_samples,
        "burst_logical_tokens": burst_tokens,
    }


def run_structural_case(
    *,
    model: str,
    run_tag: str,
    source_commit: str,
    source_tree_sha256: str,
    runtime_identity_sha256: str,
    workload_identity_sha256: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
) -> dict:
    _validate_fixed_shape(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
    )
    llm = _construct_exact_llm(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        policy="decode_burst_k8",
    )
    try:
        _run_exact_request(
            llm,
            prompt=_make_prompt(prompt_tokens, offset=50_000),
            generated_tokens=generated_tokens,
            policy="decode_burst_k8",
            profile_label=None,
        )
        llm.clear_reusable_prefix_cache()
        before = _runner_summaries(llm)
        measured = _run_structural_request(
            llm,
            prompt=_make_prompt(prompt_tokens, offset=0),
            generated_tokens=generated_tokens,
            run_tag=run_tag,
            context=prompt_tokens,
        )
        summary = _combined_summary(llm, before)
        samples = measured["tpot_samples_ns"]
        return {
            "schema_version": STRUCTURAL_SCHEMA_VERSION,
            "arm": "nsight_structural",
            "run_tag": run_tag,
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256,
            "runtime_identity_sha256": runtime_identity_sha256,
            "workload_identity_sha256": workload_identity_sha256,
            "repetition": 0,
            "context_length": prompt_tokens,
            "generated_tokens": generated_tokens,
            "output_token_ids": measured["output_token_ids"],
            "output_text_sha256": sha256_text(measured["output_text"]),
            "ttft_ns": measured["ttft_ns"],
            "e2e_ns": measured["e2e_ns"],
            "profiled_tpot_median_ns": statistics.median(samples),
            "profiled_tpot_p95_ns": _nearest_rank(samples, 0.95),
            "burst_logical_tokens": measured["burst_logical_tokens"],
            **_summary_fields(summary),
        }
    finally:
        llm.exit()


def build_source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    run_tag: str,
) -> dict:
    root = Path(repo_root).resolve()
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "run_tag": _identity_value(run_tag, "run_tag"),
        "source_commit": _digest(
            source_commit,
            "source_commit",
            length=40,
        ),
        "source_sha256": {
            relative: sha256_file(root / relative)
            for relative in SOURCE_FILES
        },
    }


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row {line_number} must be an object"
                )
            rows.append(value)
    return rows


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_output_directory(path: Path) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ValueError("output directory is invalid")
    return output


def _output_path(output_dir: Path, name: str) -> Path:
    path = (output_dir / name).resolve()
    if output_dir not in path.parents:
        raise ValueError("output path escapes output directory")
    return path


def _assert_outputs_match(
    timing_rows: list[dict],
    structural_rows: list[dict],
) -> None:
    timing_by_context = {}
    for row in timing_rows:
        key = row.get("context_length")
        authority = (
            row.get("output_token_ids"),
            row.get("output_text_sha256"),
        )
        previous = timing_by_context.setdefault(key, authority)
        if previous != authority:
            raise ValueError("timing output mismatch")
    for row in structural_rows:
        context = row.get("context_length")
        authority = timing_by_context.get(context)
        observed = (
            row.get("output_token_ids"),
            row.get("output_text_sha256"),
        )
        if authority is None or observed != authority:
            raise ValueError("timing and structural output mismatch")


def _trace_context_summary(
    structural: dict,
    parsed: dict,
) -> tuple[dict, list[dict], list[dict]]:
    ranges = parsed["ranges"]
    expected_tokens = structural["burst_logical_tokens"]
    actual_tokens = [row["logical_tokens"] for row in ranges]
    if actual_tokens != expected_tokens:
        raise ValueError("trace logical-token inventory mismatch")
    classified = classify_kernel_rows(parsed["kernel_rows"])
    segments = build_candidate_segments(classified)
    coverage = summarize_trace_coverage(classified)
    candidate_duration = sum(
        row["kernel_duration_sum_ns"] for row in segments
    )
    eligible_zero_cost = sum(row["wall_union_ns"] for row in segments)
    logical_tokens = sum(expected_tokens)
    context_row = {
        "context_length": structural["context_length"],
        "profiled_tpot_median_ns":
            structural["profiled_tpot_median_ns"],
        "profiled_tpot_p95_ns": structural["profiled_tpot_p95_ns"],
        "output_token_ids": structural["output_token_ids"],
        "output_text_sha256": structural["output_text_sha256"],
        "transaction_count": len(ranges),
        "logical_token_count": logical_tokens,
        "eligible_zero_cost_ns_per_token":
            eligible_zero_cost / logical_tokens,
        "candidate_cuda_duration_ns": candidate_duration,
        "total_kernel_duration_ns": coverage["kernel_duration_ns"],
        "classified_launch_ratio":
            coverage["classified_launch_ratio"],
        "classified_duration_ratio":
            coverage["classified_duration_ratio"],
        "segment_signatures": sorted({
            row["normalized_kernel_signature_sha256"]
            for row in segments
        }),
        **_summary_fields(structural),
    }
    return context_row, classified, segments


def finalize_evidence(
    *,
    timing_path: Path,
    structural_path: Path,
    trace_paths: dict[int, Path],
    output_dir: Path,
) -> dict:
    timing_rows = _read_jsonl(timing_path)
    structural_rows = _read_jsonl(structural_path)
    _assert_outputs_match(timing_rows, structural_rows)
    structural_by_context = {
        int(row["context_length"]): row for row in structural_rows
    }
    if set(structural_by_context) != set(CONTEXT_LENGTHS):
        raise ValueError("structural row inventory is incomplete")
    if set(trace_paths) != set(CONTEXT_LENGTHS):
        raise ValueError("trace path inventory is incomplete")

    context_rows = []
    all_kernels = []
    all_segments = []
    trace_inventory = []
    for context in CONTEXT_LENGTHS:
        trace_path = Path(trace_paths[context])
        parsed = read_decode_trace(trace_path)
        context_row, kernels, segments = _trace_context_summary(
            structural_by_context[context],
            parsed,
        )
        context_rows.append(context_row)
        all_kernels.extend(kernels)
        all_segments.extend(segments)
        trace_inventory.append({
            "context_length": context,
            "remote_path": str(trace_path),
            "byte_length": trace_path.stat().st_size,
            "sha256": sha256_file(trace_path),
            "transaction_count": len(parsed["ranges"]),
            "kernel_count": len(kernels),
        })

    identity = {
        field: timing_rows[0][field]
        for field in (
            "source_commit",
            "source_tree_sha256",
            "runtime_identity_sha256",
            "workload_identity_sha256",
        )
    }
    trace_summary = {
        "schema_version": TRACE_SUMMARY_SCHEMA_VERSION,
        **identity,
        "contexts": context_rows,
    }
    ceiling = compute_ceiling(timing_rows, trace_summary)
    output = _ensure_output_directory(output_dir)
    _write_json(
        _output_path(output, "timing_summary.json"),
        {
            "schema_version": TIMING_SCHEMA_VERSION,
            "row_count": len(timing_rows),
            "contexts": list(CONTEXT_LENGTHS),
        },
    )
    _write_json(
        _output_path(output, "trace_inventory.json"),
        {
            "schema_version":
                "lease-sealed-persistent-decode.trace-inventory.v1",
            "raw_traces": trace_inventory,
            "trace_summary": trace_summary,
        },
    )
    _write_jsonl(
        _output_path(output, "kernel_rows.jsonl"),
        all_kernels,
    )
    _write_jsonl(
        _output_path(output, "segment_rows.jsonl"),
        all_segments,
    )
    _write_json(_output_path(output, "ceiling.json"), ceiling)
    _write_json(
        _output_path(output, "manifest.json"),
        {
            "schema_version":
                "lease-sealed-persistent-decode.manifest.v1",
            "artifacts": [
                {
                    "path": relative,
                    "byte_length": (output / relative).stat().st_size,
                    "sha256": sha256_file(output / relative),
                }
                for relative in MANIFEST_FILES
            ],
        },
    )
    return ceiling


def synthetic_timing_rows_for_test() -> list[dict]:
    rows = []
    for repetition, context in build_timing_identities():
        rows.append({
            "schema_version": TIMING_SCHEMA_VERSION,
            "arm": "uninstrumented",
            "source_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "runtime_identity_sha256": "c" * 64,
            "workload_identity_sha256": "d" * 64,
            "repetition": repetition,
            "context_length": context,
            "generated_tokens": GENERATED_TOKENS,
            "output_token_ids": [context % 31] * GENERATED_TOKENS,
            "output_text_sha256": sha256_text(f"output-{context}"),
            "tpot_median_ns": 2_000_000,
            "tpot_p95_ns": 2_100_000,
            "target_model_forwards": GENERATED_TOKENS - 1,
            "committed_tokens": GENERATED_TOKENS - 1,
            "fallback_count": 0,
            "failure_count": 0,
            "rollback_count": 0,
            "quarantine_reason": None,
        })
    return rows


def synthetic_structural_rows_for_test() -> list[dict]:
    return [
        {
            "schema_version": STRUCTURAL_SCHEMA_VERSION,
            "context_length": context,
            "output_token_ids": [context % 31] * GENERATED_TOKENS,
            "output_text_sha256": sha256_text(f"output-{context}"),
            "burst_logical_tokens": [8] * 15 + [7],
        }
        for context in CONTEXT_LENGTHS
    ]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_trace_paths(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        context_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError("trace path must use CONTEXT=PATH")
        context = int(context_text)
        if context in result:
            raise ValueError("duplicate trace path context")
        result[context] = Path(path_text)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("timing", "structural", "finalize"),
    )
    parser.add_argument("--model")
    parser.add_argument("--run-tag")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree-sha256")
    parser.add_argument("--runtime-identity-sha256")
    parser.add_argument("--workload-identity-sha256")
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument(
        "--generated-tokens",
        type=int,
        default=GENERATED_TOKENS,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    parser.add_argument("--output")
    parser.add_argument("--timing-path")
    parser.add_argument("--structural-path")
    parser.add_argument("--trace", action="append", default=[])
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    if args.mode == "finalize":
        finalize_evidence(
            timing_path=Path(args.timing_path),
            structural_path=Path(args.structural_path),
            trace_paths=_parse_trace_paths(args.trace),
            output_dir=Path(args.output_dir),
        )
        return 0

    common = {
        "model": args.model,
        "run_tag": args.run_tag,
        "source_commit": args.source_commit,
        "source_tree_sha256": args.source_tree_sha256,
        "runtime_identity_sha256": args.runtime_identity_sha256,
        "workload_identity_sha256": args.workload_identity_sha256,
        "prompt_tokens": args.prompt_tokens,
        "generated_tokens": args.generated_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    row = (
        run_timing_case(repetition=args.repetition, **common)
        if args.mode == "timing"
        else run_structural_case(**common)
    )
    _append_jsonl(Path(args.output), row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
