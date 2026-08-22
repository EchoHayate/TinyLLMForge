#!/usr/bin/env python3
"""Source-bound benchmark for the zero-temperature greedy fast path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time


CASE_SCHEMA_VERSION = "zero-temperature-greedy-fast-path.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "zero-temperature-greedy-fast-path.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = (
    "zero-temperature-greedy-fast-path.summary.v1"
)
WORKLOAD_SCHEMA_VERSION = (
    "zero-temperature-greedy-fast-path.workload.v1"
)
SOURCE_SCHEMA_VERSION = (
    "zero-temperature-greedy-fast-path.source.v1"
)
POLICIES = ("off", "on")
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-final",
)
COUNTER_FIELDS = (
    "eligible_steps",
    "optimized_steps",
    "avoided_temperature_h2d_bytes",
    "avoided_softmax_calls",
    "avoided_gumbel_rng_calls",
    "avoided_stochastic_divisions",
    "avoided_stochastic_argmax_calls",
    "avoided_where_calls",
)
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/greedy_sampling_fast_path.py",
    "tinyvllm/engine/model_runner.py",
    "tools/profile_zero_temperature_greedy_fast_path.py",
    "tools/test_profile_zero_temperature_greedy_fast_path.py",
    "tools/zero_temperature_greedy_fast_path_gate.py",
    "tools/test_zero_temperature_greedy_fast_path_gate.py",
    "tools/zero_temperature_greedy_fast_path_verify.py",
    "tools/test_zero_temperature_greedy_fast_path_verify.py",
    "tools/run_zero_temperature_greedy_fast_path_remote.py",
    "tools/test_run_zero_temperature_greedy_fast_path_remote.py",
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
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_relative_path(path: str) -> Path:
    if not isinstance(path, str) or not path:
        raise ValueError("sidecar path must be a non-empty string")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("sidecar path must stay below run directory")
    return relative


def write_float32_sidecar(
    run_dir: Path,
    relative_path: str,
    values,
) -> dict:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(
        not math.isfinite(value) for value in normalized
    ):
        raise ValueError(
            "float32 sidecar values must be finite and non-empty"
        )
    relative = _safe_relative_path(relative_path)
    destination = Path(run_dir) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = struct.pack(
        f"<{len(normalized)}f",
        *normalized,
    )
    temporary = destination.with_name(
        f".{destination.name}.tmp"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return {
        "path": relative.as_posix(),
        "element_count": len(normalized),
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_float32_sidecar(
    run_dir: Path,
    *,
    path: str,
    expected_element_count: int,
    expected_byte_length: int,
    expected_sha256: str,
) -> tuple[float, ...]:
    relative = _safe_relative_path(path)
    payload = (Path(run_dir) / relative).read_bytes()
    if len(payload) != expected_byte_length:
        raise ValueError("sidecar byte length mismatch")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("sidecar SHA256 mismatch")
    if (
        isinstance(expected_element_count, bool)
        or not isinstance(expected_element_count, int)
        or expected_element_count <= 0
        or expected_byte_length != expected_element_count * 4
    ):
        raise ValueError("sidecar element inventory mismatch")
    values = struct.unpack(
        f"<{expected_element_count}f",
        payload,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sidecar values must be finite")
    return tuple(values)


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


def _validate_digest(value, name: str, lengths=(64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _validate_summary_shape(summary) -> dict:
    if not isinstance(summary, dict):
        raise ValueError(
            "greedy fast-path summary must be an object"
        )
    missing = (
        set(COUNTER_FIELDS) | {"fallback_counts"}
    ) - set(summary)
    if missing:
        raise ValueError(
            "greedy fast-path summary fields are missing: "
            f"{sorted(missing)}"
        )
    normalized = {
        field: _require_non_negative_int(
            summary[field],
            f"greedy_fast_path_summary.{field}",
        )
        for field in COUNTER_FIELDS
    }
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
            "greedy fast-path fallback counts are invalid"
        )
    normalized["fallback_counts"] = dict(
        sorted(fallback_counts.items())
    )
    return normalized


def _validate_summary(
    summary,
    *,
    policy: str,
    expected_steps: int,
) -> dict:
    normalized = _validate_summary_shape(summary)
    optimized = normalized["optimized_steps"]
    eligible = normalized["eligible_steps"]
    if policy == "on":
        if optimized != expected_steps or eligible != expected_steps:
            raise ValueError(
                "optimized sampling step inventory mismatch"
            )
        expected_avoided = {
            "avoided_temperature_h2d_bytes": 4 * expected_steps,
            "avoided_softmax_calls": expected_steps,
            "avoided_gumbel_rng_calls": expected_steps,
            "avoided_stochastic_divisions": 2 * expected_steps,
            "avoided_stochastic_argmax_calls": expected_steps,
            "avoided_where_calls": expected_steps,
        }
        for field, expected in expected_avoided.items():
            if normalized[field] != expected:
                raise ValueError(
                    f"avoided work mismatch: {field}"
                )
    elif optimized != 0 or eligible != 0:
        raise ValueError(
            "disabled policy reached greedy fast path"
        )
    return normalized


def validate_case_row(
    row,
    *,
    require_complete_optimized_path: bool = True,
) -> dict:
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
        "greedy_fast_path_summary",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(
            f"case row fields are missing: {sorted(missing)}"
        )
    if row["schema_version"] != CASE_SCHEMA_VERSION:
        raise ValueError("case row schema mismatch")
    if not isinstance(row["run_tag"], str) or not row["run_tag"]:
        raise ValueError("run tag is invalid")
    _validate_digest(
        row["source_commit"],
        "source commit",
        lengths=(40, 64),
    )
    policy = row["policy"]
    if policy not in POLICIES:
        raise ValueError("policy is invalid")
    repetition = _require_non_negative_int(
        row["repetition"],
        "repetition",
    )
    cases = {
        bucket: (prompt_tokens, generated_tokens)
        for bucket, prompt_tokens, generated_tokens
        in context_cases()
    }
    bucket = row["context_bucket"]
    if bucket not in cases:
        raise ValueError("context bucket is invalid")
    prompt_tokens = _require_non_negative_int(
        row["prompt_tokens"],
        "prompt_tokens",
    )
    generated_tokens = _require_non_negative_int(
        row["generated_tokens"],
        "generated_tokens",
    )
    if (prompt_tokens, generated_tokens) != cases[bucket]:
        raise ValueError("context shape does not match bucket")
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
        raise ValueError("output token inventory is invalid")
    _validate_digest(row["output_text_sha256"], "output text digest")
    expected_decode_steps = generated_tokens - 1
    for values, name in (
        (row["tpot_samples_ns"], "tpot_samples_ns"),
        (row["decode_host_ns"], "decode_host_ns"),
        (row["decode_cuda_ns"], "decode_cuda_ns"),
    ):
        if not isinstance(values, list) or len(values) != expected_decode_steps:
            raise ValueError(f"{name} inventory mismatch")
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
        _require_finite_non_negative(row[field], field)
    normalized = dict(row)
    normalized["repetition"] = repetition
    if require_complete_optimized_path:
        normalized["greedy_fast_path_summary"] = (
            _validate_summary(
                row["greedy_fast_path_summary"],
                policy=policy,
                expected_steps=generated_tokens,
            )
        )
    else:
        normalized["greedy_fast_path_summary"] = (
            _validate_summary_shape(
                row["greedy_fast_path_summary"]
            )
        )
    return normalized


def summarize_rows(
    rows: list[dict],
    *,
    require_complete_optimized_path: bool = True,
) -> dict:
    validated = [
        validate_case_row(
            row,
            require_complete_optimized_path=(
                require_complete_optimized_path
            ),
        )
        for row in rows
    ]
    if not validated:
        raise ValueError("at least one case row is required")
    identities = {}
    for row in validated:
        identity = (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        )
        if identity in identities:
            raise ValueError(
                f"duplicate case identity: {identity}"
            )
        identities[identity] = row
    pairs = []
    for bucket, repetition in sorted({
        (row["context_bucket"], row["repetition"])
        for row in validated
    }):
        try:
            off = identities[(bucket, repetition, "off")]
            on = identities[(bucket, repetition, "on")]
        except KeyError as error:
            raise ValueError("OFF/ON pair is incomplete") from error
        if off["output_token_ids"] != on["output_token_ids"]:
            raise ValueError("output token mismatch in OFF/ON pair")
        if off["output_text_sha256"] != on["output_text_sha256"]:
            raise ValueError("output text mismatch in OFF/ON pair")
        pairs.append({
            "context_bucket": bucket,
            "repetition": repetition,
            "off_tpot_median_ns": statistics.median(
                off["tpot_samples_ns"]
            ),
            "on_tpot_median_ns": statistics.median(
                on["tpot_samples_ns"]
            ),
        })
    run_tags = {row["run_tag"] for row in validated}
    commits = {row["source_commit"] for row in validated}
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError("case rows do not share source identity")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_tag": next(iter(run_tags)),
        "source_commit": next(iter(commits)),
        "row_count": len(validated),
        "pair_count": len(pairs),
        "all_outputs_exact": True,
        "all_on_steps_optimized": all(
            row["greedy_fast_path_summary"][
                "eligible_steps"
            ] == row["generated_tokens"]
            and row["greedy_fast_path_summary"][
                "optimized_steps"
            ] == row["generated_tokens"]
            and not row["greedy_fast_path_summary"][
                "fallback_counts"
            ]
            for row in validated
            if row["policy"] == "on"
        ),
        "pairs": pairs,
    }


def validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path,
    expected_buckets: tuple[str, ...] = (
        "short",
        "medium",
        "long",
    ),
) -> list[dict]:
    expected = {
        (bucket, policy, point)
        for bucket in expected_buckets
        for policy in POLICIES
        for point in SAMPLING_POINTS
    }
    identities = {}
    validated = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "correctness row must be an object"
            )
        if row.get("schema_version") != CORRECTNESS_SCHEMA_VERSION:
            raise ValueError("correctness row schema mismatch")
        identity = (
            row.get("context_bucket"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        if identity in identities:
            raise ValueError(
                f"duplicate correctness identity: {identity}"
            )
        if identity not in expected:
            raise ValueError(
                f"unexpected correctness identity: {identity}"
            )
        identities[identity] = row
        bucket, policy, _point = identity
        cases = {
            name: (prompt_tokens, generated_tokens)
            for name, prompt_tokens, generated_tokens
            in context_cases()
        }
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != cases[bucket]:
            raise ValueError(
                "correctness context shape mismatch"
            )
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != row["generated_tokens"]
        ):
            raise ValueError(
                "correctness output token inventory is invalid"
            )
        _validate_digest(
            row.get("output_text_sha256"),
            "correctness output text digest",
        )
        _validate_digest(
            row.get("source_commit"),
            "correctness source commit",
            lengths=(40, 64),
        )
        shape = row.get("logits_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != 1
            or isinstance(shape[1], bool)
            or not isinstance(shape[1], int)
            or shape[1] <= 0
        ):
            raise ValueError("correctness logits shape is invalid")
        values = read_float32_sidecar(
            run_dir,
            path=row.get("logits_path"),
            expected_element_count=row.get(
                "logits_element_count"
            ),
            expected_byte_length=row.get(
                "logits_byte_length"
            ),
            expected_sha256=row.get("logits_sha256"),
        )
        if len(values) != shape[0] * shape[1]:
            raise ValueError(
                "correctness logits element count mismatch"
            )
        normalized = dict(row)
        normalized["greedy_fast_path_summary"] = (
            _validate_summary(
                row.get("greedy_fast_path_summary"),
                policy=policy,
                expected_steps=row["generated_tokens"],
            )
        )
        validated.append(normalized)
    if set(identities) != expected:
        raise ValueError(
            "correctness row inventory is incomplete"
        )
    run_tags = {row.get("run_tag") for row in validated}
    commits = {row.get("source_commit") for row in validated}
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError(
            "correctness rows do not share source identity"
        )
    return validated


def _stats_delta(before: dict, after: dict) -> dict:
    result = {}
    for field in COUNTER_FIELDS:
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
                f"greedy fast-path counter decreased: {field}"
            )
        result[field] = after_value - before_value
    before_fallbacks = before.get("fallback_counts")
    after_fallbacks = after.get("fallback_counts")
    if not isinstance(before_fallbacks, dict) or not isinstance(
        after_fallbacks,
        dict,
    ):
        raise RuntimeError(
            "greedy fast-path fallback counters are unavailable"
        )
    result["fallback_counts"] = {}
    for reason in sorted(set(before_fallbacks) | set(after_fallbacks)):
        difference = int(after_fallbacks.get(reason, 0)) - int(
            before_fallbacks.get(reason, 0)
        )
        if difference < 0:
            raise RuntimeError(
                "greedy fast-path fallback counter decreased"
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


def _single_rank_profile(payload: dict) -> dict:
    if (
        not isinstance(payload, dict)
        or payload.get("enabled") is not True
        or payload.get("rank_inventory") != [0]
        or not isinstance(payload.get("ranks"), list)
        or len(payload["ranks"]) != 1
        or not isinstance(payload["ranks"][0], dict)
        or payload["ranks"][0].get("rank") != 0
    ):
        raise RuntimeError(
            "Stage-1 worker requires tensor parallel size one"
        )
    return payload["ranks"][0]


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
        max_num_batched_tokens=prompt_tokens + generated_tokens,
        max_num_seqs=1,
        max_model_len=prompt_tokens + generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=False,
        zero_temperature_greedy_fast_path=enabled,
    )


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
                        "ordinary decode emitted an unexpected token count"
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
        raise RuntimeError("request produced no first token")
    if not isinstance(final_outputs, list) or len(final_outputs) != 1:
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
        profile = _single_rank_profile(
            llm.finalize_decode_internal_profile(
                already_synchronized=True,
                timeout_s=60.0,
            )
        )
        decode_steps = sorted(
            (
                row
                for row in profile["steps"]
                if row["is_decode"]
            ),
            key=lambda row: row["decode_ordinal"],
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
        if profile_label is not None and len(values) != expected_decode_steps:
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
        "cuda_peak_allocated_bytes": int(
            rows[0]["cuda_peak_allocated_bytes"]
        ),
        "cuda_peak_reserved_bytes": int(
            rows[0]["cuda_peak_reserved_bytes"]
        ),
    }


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
            .zero_temperature_greedy_fast_path_summary()
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
            .zero_temperature_greedy_fast_path_summary()
        )
        summary = _stats_delta(before, after)
        e2e_seconds = measured["e2e_ns"] / 1_000_000_000
        return validate_case_row({
            "schema_version": CASE_SCHEMA_VERSION,
            "run_tag": run_tag,
            "source_commit": source_commit,
            "policy": policy,
            "repetition": repetition,
            "context_bucket": context_bucket,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "output_token_ids": measured["output_token_ids"],
            "output_text_sha256": sha256_text(
                measured["output_text"]
            ),
            "ttft_ns": measured["ttft_ns"],
            "e2e_ns": measured["e2e_ns"],
            "tpot_samples_ns": measured["tpot_samples_ns"],
            "decode_host_ns": measured["decode_host_ns"],
            "decode_cuda_ns": measured["decode_cuda_ns"],
            "output_tokens_per_second": (
                generated_tokens / e2e_seconds
            ),
            **memory,
            "greedy_fast_path_summary": summary,
        })
    finally:
        llm.exit()


def run_correctness_probe(
    *,
    model: str,
    run_dir: Path,
    run_tag: str,
    source_commit: str,
    policy: str,
    context_bucket: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
) -> list[dict]:
    from tinyvllm import SamplingParams

    enabled = policy == "on"
    llm = _construct_llm(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enabled=enabled,
    )
    try:
        llm.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        before = (
            llm.model_runner
            .zero_temperature_greedy_fast_path_summary()
        )
        llm.add_request(
            _make_prompt(prompt_tokens, offset=90_001),
            SamplingParams(
                temperature=0.0,
                max_tokens=generated_tokens,
                ignore_eos=True,
            ),
        )
        captured = {}
        final_outputs = None
        step_index = 0
        while not llm.is_finished():
            outputs, _num_tokens = llm.step()
            if step_index == 0:
                point = "prefill-final"
            elif step_index == 1:
                point = "decode-first"
            elif step_index == generated_tokens - 1:
                point = "decode-final"
            else:
                point = None
            if point is not None:
                logits = (
                    llm.read_step_logits_authority()
                    .detach()
                    .to(dtype=__import__("torch").float32)
                    .contiguous()
                )
                shape = [int(value) for value in logits.shape]
                values = logits.view(-1).tolist()
                sidecar = write_float32_sidecar(
                    run_dir,
                    (
                        f"logits/{context_bucket}-{policy}-"
                        f"{point}.f32"
                    ),
                    values,
                )
                captured[point] = (shape, sidecar)
            if outputs:
                final_outputs = outputs
            step_index += 1
        if step_index != generated_tokens:
            raise RuntimeError(
                "correctness sampling step inventory mismatch"
            )
        if set(captured) != set(SAMPLING_POINTS):
            raise RuntimeError(
                "correctness sampling points are incomplete"
            )
        if not isinstance(final_outputs, list) or len(final_outputs) != 1:
            raise RuntimeError(
                "correctness output is incomplete"
            )
        output_ids = list(final_outputs[0][1])
        if len(output_ids) != generated_tokens:
            raise RuntimeError(
                "correctness output token inventory mismatch"
            )
        output_text_sha256 = sha256_text(
            llm.tokenizer.decode(output_ids)
        )
        after = (
            llm.model_runner
            .zero_temperature_greedy_fast_path_summary()
        )
        summary = _stats_delta(before, after)
        rows = []
        for point in SAMPLING_POINTS:
            shape, sidecar = captured[point]
            rows.append({
                "schema_version": CORRECTNESS_SCHEMA_VERSION,
                "run_tag": run_tag,
                "source_commit": source_commit,
                "policy": policy,
                "context_bucket": context_bucket,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "sampling_point": point,
                "output_token_ids": output_ids,
                "output_text_sha256": output_text_sha256,
                "logits_path": sidecar["path"],
                "logits_shape": shape,
                "logits_element_count": sidecar["element_count"],
                "logits_byte_length": sidecar["byte_length"],
                "logits_sha256": sidecar["sha256"],
                "greedy_fast_path_summary": summary,
            })
        return rows
    finally:
        try:
            llm.enable_step_logits_authority_recording(
                False,
                timeout_s=60.0,
            )
        finally:
            llm.exit()


def _source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    run_tag: str,
) -> dict:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_sha256": {
            relative: sha256_file(repo_root / relative)
            for relative in SOURCE_FILES
        },
    }


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
    parser.add_argument("--repetitions", type=int, default=5)
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


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if args.warmup_repetitions < 0:
        raise ValueError(
            "warmup repetitions must be non-negative"
        )
    if args.generated_tokens != 128:
        raise ValueError(
            "Stage-1 generated tokens must equal 128"
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
        for _bucket, prompt_tokens, _generated in context_cases()
    )
    if prompt_lengths != expected_lengths:
        raise ValueError(
            "Stage-1 prompt lengths must be 256,2048,8192"
        )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    repo_root = Path(__file__).resolve().parents[1]
    _write_json(
        out_dir / "source_manifest.json",
        _source_manifest(
            repo_root=repo_root,
            source_commit=args.source_commit,
            run_tag=args.run_tag,
        ),
    )
    _write_json(
        out_dir / "workload_manifest.json",
        {
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
            "correctness_sampling_points": list(
                SAMPLING_POINTS
            ),
        },
    )
    case_rows = []
    case_path = out_dir / "case_rows.jsonl"
    for repetition in range(args.repetitions):
        for bucket, prompt_tokens, generated_tokens in context_cases():
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
                    warmup_repetitions=args.warmup_repetitions,
                    gpu_memory_utilization=(
                        args.gpu_memory_utilization
                    ),
                )
                append_jsonl(case_path, row)
                case_rows.append(row)
    correctness_rows = []
    correctness_path = out_dir / "correctness_rows.jsonl"
    for bucket, prompt_tokens, generated_tokens in context_cases():
        for policy in POLICIES:
            rows = run_correctness_probe(
                model=args.model,
                run_dir=out_dir,
                run_tag=args.run_tag,
                source_commit=args.source_commit,
                policy=policy,
                context_bucket=bucket,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                gpu_memory_utilization=(
                    args.gpu_memory_utilization
                ),
            )
            for row in rows:
                append_jsonl(correctness_path, row)
                correctness_rows.append(row)
    validate_correctness_rows(
        correctness_rows,
        run_dir=out_dir,
    )
    summary = summarize_rows(case_rows)
    summary["correctness_row_count"] = len(
        correctness_rows
    )
    _write_json(out_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
