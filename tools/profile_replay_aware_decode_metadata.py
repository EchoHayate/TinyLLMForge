#!/usr/bin/env python3
"""Source-bound OFF/ON benchmark for replay-aware decode metadata landing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


CASE_SCHEMA_VERSION = "replay-aware-decode-metadata.case.v1"
SUMMARY_SCHEMA_VERSION = (
    "replay-aware-decode-metadata.summary.v1"
)
WORKLOAD_SCHEMA_VERSION = (
    "replay-aware-decode-metadata.workload.v1"
)
SOURCE_SCHEMA_VERSION = (
    "replay-aware-decode-metadata.source.v1"
)
POLICIES = ("off", "on")
LANDING_COUNTER_FIELDS = (
    "eligible_steps",
    "optimized_steps",
    "allocation_count",
    "growth_count",
    "staged_h2d_bytes",
    "avoided_temporary_cuda_tensors",
    "avoided_blanket_zero_bytes",
)
LANDING_CAPACITY_FIELDS = (
    "current_pinned_capacity_bytes",
    "peak_pinned_capacity_bytes",
)
LANDING_REQUIRED_FIELDS = (
    *LANDING_COUNTER_FIELDS,
    *LANDING_CAPACITY_FIELDS,
    "fallback_counts",
)
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/decode_metadata_landing.py",
    "tinyvllm/engine/model_runner.py",
    "tools/profile_replay_aware_decode_metadata.py",
    "tools/test_profile_replay_aware_decode_metadata.py",
)


def context_cases() -> tuple[tuple[str, int, int], ...]:
    return (
        ("short", 256, 128),
        ("medium", 2048, 128),
        ("long", 8192, 128),
    )


def policy_order(repetition: int) -> tuple[str, str]:
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 0
    ):
        raise ValueError(
            "repetition must be a non-negative integer"
        )
    return POLICIES if repetition % 2 == 0 else tuple(
        reversed(POLICIES)
    )


def nearest_rank_percentile(
    values,
    percentile: float,
) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError(
            "percentile requires at least one value"
        )
    if (
        not math.isfinite(percentile)
        or percentile <= 0.0
        or percentile > 1.0
    ):
        raise ValueError(
            "percentile must be finite and in (0, 1]"
        )
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _require_non_negative_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_finite_non_negative(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{name} must be finite and non-negative"
        )
    return float(value)


def _validate_landing_summary(
    summary,
    *,
    policy: str,
    expected_decode_steps: int,
) -> dict:
    if not isinstance(summary, dict):
        raise ValueError(
            "landing summary must be an object"
        )
    for field in LANDING_REQUIRED_FIELDS:
        if field not in summary:
            raise ValueError(
                f"landing summary field is missing: {field}"
            )
    normalized = {}
    for field in (
        *LANDING_COUNTER_FIELDS,
        *LANDING_CAPACITY_FIELDS,
    ):
        normalized[field] = _require_non_negative_int(
            summary[field],
            f"landing_summary.{field}",
        )
    fallback_counts = summary["fallback_counts"]
    if (
        not isinstance(fallback_counts, dict)
        or any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for reason, count in fallback_counts.items()
        )
    ):
        raise ValueError(
            "landing summary fallback counts are invalid"
        )
    normalized["fallback_counts"] = dict(
        sorted(fallback_counts.items())
    )
    if policy == "on" and (
        normalized["eligible_steps"]
        != expected_decode_steps
        or normalized["optimized_steps"]
        != expected_decode_steps
    ):
        raise ValueError(
            "optimized decode step inventory mismatch"
        )
    if policy == "off" and (
        normalized["eligible_steps"] != 0
        or normalized["optimized_steps"] != 0
    ):
        raise ValueError(
            "disabled policy reached metadata landing"
        )
    return normalized


def validate_case_row(row) -> dict:
    if not isinstance(row, dict):
        raise ValueError("case row must be an object")
    required = {
        "schema_version",
        "run_tag",
        "source_commit",
        "policy",
        "repetition",
        "context_bucket",
        "prompt_tokens",
        "generated_tokens",
        "output_token_ids",
        "output_text_sha256",
        "ttft_ns",
        "e2e_ns",
        "tpot_samples_ns",
        "decode_host_ns",
        "decode_cuda_ns",
        "output_tokens_per_second",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "landing_summary",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(
            f"case row fields are missing: {sorted(missing)}"
        )
    if row["schema_version"] != CASE_SCHEMA_VERSION:
        raise ValueError("case row schema mismatch")
    if (
        not isinstance(row["run_tag"], str)
        or not row["run_tag"]
    ):
        raise ValueError("run tag is invalid")
    source_commit = row["source_commit"]
    if (
        not isinstance(source_commit, str)
        or len(source_commit) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in source_commit
        )
    ):
        raise ValueError("source commit is invalid")
    policy = row["policy"]
    if policy not in POLICIES:
        raise ValueError("policy is invalid")
    repetition = _require_non_negative_int(
        row["repetition"],
        "repetition",
    )
    cases_by_bucket = {
        bucket: (prompt_tokens, generated_tokens)
        for bucket, prompt_tokens, generated_tokens
        in context_cases()
    }
    bucket = row["context_bucket"]
    if bucket not in cases_by_bucket:
        raise ValueError("context bucket is invalid")
    prompt_tokens = _require_non_negative_int(
        row["prompt_tokens"],
        "prompt_tokens",
    )
    generated_tokens = _require_non_negative_int(
        row["generated_tokens"],
        "generated_tokens",
    )
    if (
        (prompt_tokens, generated_tokens)
        != cases_by_bucket[bucket]
    ):
        raise ValueError(
            "context shape does not match bucket"
        )
    output_ids = row["output_token_ids"]
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != generated_tokens
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in output_ids
        )
    ):
        raise ValueError(
            "output token inventory is invalid"
        )
    text_digest = row["output_text_sha256"]
    if (
        not isinstance(text_digest, str)
        or len(text_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in text_digest
        )
    ):
        raise ValueError(
            "output text digest is invalid"
        )
    expected_decode_steps = max(0, generated_tokens - 1)
    tpot_samples = row["tpot_samples_ns"]
    decode_host = row["decode_host_ns"]
    decode_cuda = row["decode_cuda_ns"]
    for values, name in (
        (tpot_samples, "tpot_samples_ns"),
        (decode_host, "decode_host_ns"),
        (decode_cuda, "decode_cuda_ns"),
    ):
        if (
            not isinstance(values, list)
            or len(values) != expected_decode_steps
        ):
            raise ValueError(
                f"{name} inventory mismatch"
            )
        for index, value in enumerate(values):
            _require_finite_non_negative(
                value,
                f"{name}[{index}]",
            )
    for field in (
        "ttft_ns",
        "e2e_ns",
        "output_tokens_per_second",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ):
        _require_finite_non_negative(
            row[field],
            field,
        )
    normalized = dict(row)
    normalized["repetition"] = repetition
    normalized["landing_summary"] = (
        _validate_landing_summary(
            row["landing_summary"],
            policy=policy,
            expected_decode_steps=expected_decode_steps,
        )
    )
    return normalized


def summarize_rows(rows: list[dict]) -> dict:
    validated = [validate_case_row(row) for row in rows]
    if not validated:
        raise ValueError(
            "at least one case row is required"
        )
    run_tags = {row["run_tag"] for row in validated}
    commits = {row["source_commit"] for row in validated}
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError(
            "case rows do not share source identity"
        )
    by_identity = {}
    for row in validated:
        identity = (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        )
        if identity in by_identity:
            raise ValueError(
                f"duplicate case identity: {identity}"
            )
        by_identity[identity] = row
    pair_keys = sorted({
        (bucket, repetition)
        for bucket, repetition, _policy in by_identity
    })
    pairs = []
    for bucket, repetition in pair_keys:
        try:
            off = by_identity[(bucket, repetition, "off")]
            on = by_identity[(bucket, repetition, "on")]
        except KeyError as error:
            raise ValueError(
                "OFF/ON pair is incomplete"
            ) from error
        if off["output_token_ids"] != on["output_token_ids"]:
            raise ValueError(
                "output token mismatch in OFF/ON pair"
            )
        if (
            off["output_text_sha256"]
            != on["output_text_sha256"]
        ):
            raise ValueError(
                "output text mismatch in OFF/ON pair"
            )
        pairs.append({
            "context_bucket": bucket,
            "repetition": repetition,
            "off_tpot_median_ns": statistics.median(
                off["tpot_samples_ns"]
            ),
            "on_tpot_median_ns": statistics.median(
                on["tpot_samples_ns"]
            ),
            "off_tpot_p95_ns": nearest_rank_percentile(
                off["tpot_samples_ns"],
                0.95,
            ),
            "on_tpot_p95_ns": nearest_rank_percentile(
                on["tpot_samples_ns"],
                0.95,
            ),
        })
    on_rows = [
        row for row in validated if row["policy"] == "on"
    ]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_tag": next(iter(run_tags)),
        "source_commit": next(iter(commits)),
        "row_count": len(validated),
        "pair_count": len(pairs),
        "all_outputs_exact": True,
        "all_on_steps_optimized": True,
        "peak_pinned_capacity_bytes": max(
            row["landing_summary"][
                "peak_pinned_capacity_bytes"
            ]
            for row in on_rows
        ),
        "pairs": pairs,
    }


def _landing_delta(before: dict, after: dict) -> dict:
    result = {}
    for field in LANDING_COUNTER_FIELDS:
        before_value = _require_non_negative_int(
            before.get(field),
            f"before.{field}",
        )
        after_value = _require_non_negative_int(
            after.get(field),
            f"after.{field}",
        )
        if after_value < before_value:
            raise RuntimeError(
                f"landing counter decreased: {field}"
            )
        result[field] = after_value - before_value
    for field in LANDING_CAPACITY_FIELDS:
        result[field] = _require_non_negative_int(
            after.get(field),
            f"after.{field}",
        )
    before_fallbacks = before.get("fallback_counts")
    after_fallbacks = after.get("fallback_counts")
    if not isinstance(before_fallbacks, dict) or not isinstance(
        after_fallbacks,
        dict,
    ):
        raise RuntimeError(
            "landing fallback counters are unavailable"
        )
    reasons = set(before_fallbacks) | set(after_fallbacks)
    result["fallback_counts"] = {}
    for reason in sorted(reasons):
        difference = int(after_fallbacks.get(reason, 0)) - int(
            before_fallbacks.get(reason, 0)
        )
        if difference < 0:
            raise RuntimeError(
                "landing fallback counter decreased"
            )
        if difference:
            result["fallback_counts"][reason] = difference
    return result


def _make_prompt(
    prompt_tokens: int,
    *,
    offset: int,
) -> list[int]:
    return [
        100 + ((index + offset) % 1_000)
        for index in range(prompt_tokens)
    ]


def _run_request(
    llm,
    *,
    prompt: list[int],
    generated_tokens: int,
    profile_label: str | None,
) -> dict:
    from tinyvllm import SamplingParams

    if profile_label is not None:
        llm.configure_decode_internal_profile(
            True,
            profile_label,
            timeout_s=60.0,
        )
    llm.add_request(
        prompt,
        SamplingParams(
            temperature=0.0,
            max_tokens=generated_tokens,
            ignore_eos=True,
        ),
    )
    started_ns = time.perf_counter_ns()
    first_token_ns = None
    tpot_samples_ns = []
    final_outputs = None
    while not llm.is_finished():
        step_started_ns = time.perf_counter_ns()
        outputs, _num_tokens = llm.step()
        step_finished_ns = time.perf_counter_ns()
        observation = llm.last_step_observation
        emitted = sum(
            len(token_ids)
            for token_ids in observation[
                "new_completion_tokens_by_seq"
            ].values()
        )
        if emitted:
            if first_token_ns is None:
                first_token_ns = step_finished_ns
            elif not observation["is_prefill"]:
                if emitted != 1:
                    raise RuntimeError(
                        "non-speculative decode emitted "
                        "an unexpected token count"
                    )
                tpot_samples_ns.append(
                    step_finished_ns - step_started_ns
                )
        if outputs:
            final_outputs = outputs
    import torch

    torch.cuda.synchronize()
    finished_ns = time.perf_counter_ns()
    if first_token_ns is None:
        raise RuntimeError(
            "request produced no first token"
        )
    if (
        not isinstance(final_outputs, list)
        or len(final_outputs) != 1
    ):
        raise RuntimeError(
            "request completion output is incomplete"
        )
    output_ids = list(final_outputs[0][1])
    if len(output_ids) != generated_tokens:
        raise RuntimeError(
            "generated token inventory mismatch"
        )
    decode_host_ns = []
    decode_cuda_ns = []
    if profile_label is not None:
        profiles = llm.finalize_decode_internal_profile(
            already_synchronized=True,
            timeout_s=60.0,
        )
        if len(profiles) != 1:
            raise RuntimeError(
                "Stage-1 worker requires tensor parallel size one"
            )
        decode_steps = [
            row
            for row in profiles[0]["steps"]
            if row["is_decode"]
        ]
        decode_steps.sort(
            key=lambda row: row["decode_ordinal"]
        )
        decode_host_ns = [
            int(row["wall_ns"]) for row in decode_steps
        ]
        decode_cuda_ns = [
            int(row["cuda_ns"]) for row in decode_steps
        ]
    expected_decode_steps = generated_tokens - 1
    for values, name in (
        (tpot_samples_ns, "TPOT"),
        (decode_host_ns, "decode host"),
        (decode_cuda_ns, "decode CUDA"),
    ):
        if profile_label is not None and (
            len(values) != expected_decode_steps
        ):
            raise RuntimeError(
                f"{name} step inventory mismatch: "
                f"{len(values)} != {expected_decode_steps}"
            )
    return {
        "output_token_ids": output_ids,
        "output_text": llm.tokenizer.decode(output_ids),
        "ttft_ns": first_token_ns - started_ns,
        "e2e_ns": finished_ns - started_ns,
        "tpot_samples_ns": tpot_samples_ns,
        "decode_host_ns": decode_host_ns,
        "decode_cuda_ns": decode_cuda_ns,
    }


def _aggregate_memory(rows: tuple[dict, ...]) -> dict:
    if len(rows) != 1:
        raise RuntimeError(
            "Stage-1 worker requires one memory rank"
        )
    return {
        "cuda_peak_allocated_bytes": max(
            int(row["cuda_peak_allocated_bytes"])
            for row in rows
        ),
        "cuda_peak_reserved_bytes": max(
            int(row["cuda_peak_reserved_bytes"])
            for row in rows
        ),
    }


def _construct_llm(
    *,
    model: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
    enabled: bool,
):
    from tinyvllm import LLM

    return LLM(
        model,
        max_num_batched_tokens=prompt_tokens,
        max_num_seqs=1,
        max_model_len=prompt_tokens + generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=False,
        replay_aware_decode_metadata=enabled,
    )


def run_case(
    *,
    model: str,
    run_tag: str,
    source_commit: str,
    policy: str,
    repetition: int,
    context_bucket: str,
    prompt_tokens: int,
    generated_tokens: int,
    warmup_repetitions: int,
    gpu_memory_utilization: float,
) -> dict:
    enabled = policy == "on"
    llm = _construct_llm(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enabled=enabled,
    )
    try:
        for warmup_index in range(warmup_repetitions):
            _run_request(
                llm,
                prompt=_make_prompt(
                    prompt_tokens,
                    offset=50_000 + warmup_index * 2_003,
                ),
                generated_tokens=generated_tokens,
                profile_label=None,
            )
            llm.clear_reusable_prefix_cache()
        before = (
            llm.model_runner
            .replay_aware_decode_metadata_summary()
        )
        llm.reset_peak_memory_stats(timeout_s=60.0)
        measured = _run_request(
            llm,
            prompt=_make_prompt(
                prompt_tokens,
                offset=repetition * 10_007,
            ),
            generated_tokens=generated_tokens,
            profile_label=(
                f"{run_tag}/{context_bucket}/"
                f"r{repetition}/{policy}"
            ),
        )
        memory = _aggregate_memory(
            llm.memory_snapshots(timeout_s=60.0)
        )
        after = (
            llm.model_runner
            .replay_aware_decode_metadata_summary()
        )
        landing_summary = _landing_delta(before, after)
        e2e_seconds = measured["e2e_ns"] / 1_000_000_000
        row = {
            "schema_version": CASE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "source_commit": source_commit,
            "policy": policy,
            "repetition": repetition,
            "context_bucket": context_bucket,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "output_token_ids": measured[
                "output_token_ids"
            ],
            "output_text_sha256": sha256_text(
                measured["output_text"]
            ),
            "ttft_ns": measured["ttft_ns"],
            "e2e_ns": measured["e2e_ns"],
            "tpot_samples_ns": measured[
                "tpot_samples_ns"
            ],
            "decode_host_ns": measured["decode_host_ns"],
            "decode_cuda_ns": measured["decode_cuda_ns"],
            "output_tokens_per_second": (
                generated_tokens / e2e_seconds
            ),
            **memory,
            "landing_summary": landing_summary,
        }
        return validate_case_row(row)
    finally:
        llm.exit()


def _parse_prompt_lengths(raw: str) -> tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in raw.split(",")
        if item.strip()
    )
    if (
        not values
        or any(value <= 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "prompt lengths must be unique positive integers"
        )
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--prompt-lengths",
        default="256,2048,8192",
    )
    parser.add_argument(
        "--generated-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    return parser.parse_args(argv)


def _source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    run_tag: str,
) -> dict:
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if git_head != source_commit:
        raise RuntimeError(
            "source commit does not match checked-out HEAD"
        )
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_sha256": {
            relative: sha256_file(repo_root / relative)
            for relative in SOURCE_FILES
        },
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.warmup_repetitions < 0:
        raise ValueError(
            "warmup repetitions must be non-negative"
        )
    if args.generated_tokens <= 1:
        raise ValueError(
            "generated tokens must be greater than one"
        )
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError(
            "gpu memory utilization must be in (0, 1]"
        )
    prompt_lengths = _parse_prompt_lengths(
        args.prompt_lengths
    )
    expected_lengths = tuple(
        prompt_tokens
        for _bucket, prompt_tokens, _generated
        in context_cases()
    )
    if prompt_lengths != expected_lengths:
        raise ValueError(
            "Stage-1 prompt lengths must be 256,2048,8192"
        )
    if args.generated_tokens != 128:
        raise ValueError(
            "Stage-1 generated tokens must equal 128"
        )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[1]
    source_manifest = _source_manifest(
        repo_root=repo_root,
        source_commit=args.source_commit,
        run_tag=args.run_tag,
    )
    workload_manifest = {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "run_tag": args.run_tag,
        "source_commit": args.source_commit,
        "model": str(Path(args.model).resolve()),
        "context_cases": [
            {
                "context_bucket": bucket,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            }
            for bucket, prompt_tokens, generated_tokens
            in context_cases()
        ],
        "repetitions": args.repetitions,
        "warmup_repetitions": args.warmup_repetitions,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "gpu_memory_utilization":
            args.gpu_memory_utilization,
        "policy_order": {
            str(repetition): list(policy_order(repetition))
            for repetition in range(args.repetitions)
        },
    }
    _write_json(
        out_dir / "source_manifest.json",
        source_manifest,
    )
    _write_json(
        out_dir / "workload_manifest.json",
        workload_manifest,
    )
    rows = []
    case_path = out_dir / "case_rows.jsonl"
    for repetition in range(args.repetitions):
        for bucket, prompt_tokens, generated_tokens in (
            context_cases()
        ):
            for policy in policy_order(repetition):
                row = run_case(
                    model=args.model,
                    run_tag=args.run_tag,
                    source_commit=args.source_commit,
                    policy=policy,
                    repetition=repetition,
                    context_bucket=bucket,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    warmup_repetitions=(
                        args.warmup_repetitions
                    ),
                    gpu_memory_utilization=(
                        args.gpu_memory_utilization
                    ),
                )
                append_jsonl(case_path, row)
                rows.append(row)
    _write_json(
        out_dir / "summary.json",
        summarize_rows(rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
