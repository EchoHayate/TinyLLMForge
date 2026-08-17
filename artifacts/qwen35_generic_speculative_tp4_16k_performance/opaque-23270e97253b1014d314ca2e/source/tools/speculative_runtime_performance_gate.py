from __future__ import annotations

import hashlib
import json
import math
import statistics
import copy
import argparse
from pathlib import Path
import platform
import subprocess
import sys


SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
POLICIES = ("baseline", "ngram")
BATCH_SIZES = (1, 4)
PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 64
WARMUP_RUNS = 1
PARITY_RUNS = 1
MEASURED_RUNS = 5
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
REAL_MOVEMENT_KEYS = (
    "h2d_copies",
    "h2d_bytes",
    "d2h_copies",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "speculative_residency_committed_blocks",
    "speculative_residency_rejected_blocks",
    "speculative_residency_rejected_d2h_copies",
)
DEFAULT_PROMPT_SEEDS = (
    "In a controlled systems experiment, repeat this sequence: "
    "alpha beta gamma delta. ",
    "A deterministic runtime trace repeats: red green blue amber. ",
    "The benchmark workload cycles through: north east south west. ",
    "For exact reproducibility, echo: one two three four five. ",
)
CLAIM_SCOPE = (
    "TP1 Qwen3-0.6B 4096-token prompts, batch 1/4, "
    "64 greedy output tokens, baseline versus generic n-gram runtime"
)
LIMITATIONS = (
    "classification remains NOT_PROMOTABLE",
    "no TP4 evidence",
    "no 16K or 32K evidence",
    "no second-model evidence",
    "no learned-drafter or MTP evidence",
    "five runs do not establish statistical significance",
)
DEFAULT_SOURCE_FILES = (
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/speculative_execution.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tinyvllm/speculative/ngram_adapter.py",
    "tools/speculative_runtime_performance_gate.py",
    "tools/speculative_runtime_performance_worker.py",
    "tools/verify_speculative_runtime_performance_gate.py",
)


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(
        token_ids,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_prompt_token_batches(
    tokenizer,
    *,
    batch_size: int,
    prompt_tokens: int,
    seeds: tuple[str, ...] = DEFAULT_PROMPT_SEEDS,
) -> list[dict]:
    batch_size = _positive_integer(batch_size, "batch_size")
    prompt_tokens = _positive_integer(
        prompt_tokens,
        "prompt_tokens",
    )
    if (
        not isinstance(seeds, tuple)
        or len(seeds) < batch_size
        or any(
            not isinstance(seed, str) or not seed
            for seed in seeds[:batch_size]
        )
    ):
        raise ValueError(
            "seed inventory must cover the requested batch"
        )
    rows = []
    for prompt_index, seed in enumerate(seeds[:batch_size]):
        encoded = tokenizer.encode(
            seed,
            add_special_tokens=False,
        )
        if (
            not isinstance(encoded, (list, tuple))
            or not encoded
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in encoded
            )
        ):
            raise ValueError(
                f"seed {prompt_index} produced invalid token IDs"
            )
        repeats = (
            prompt_tokens + len(encoded) - 1
        ) // len(encoded)
        token_ids = list(encoded) * repeats
        token_ids = token_ids[:prompt_tokens]
        rows.append({
            "prompt_index": prompt_index,
            "seed": seed,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "sha256": _token_ids_sha256(token_ids),
        })
    return rows


def subtract_counter_summaries(
    before: dict,
    after: dict,
    *,
    keys: tuple[str, ...],
) -> dict[str, int]:
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("counter summaries must be mappings")
    if (
        not isinstance(keys, tuple)
        or not keys
        or any(not isinstance(key, str) or not key for key in keys)
    ):
        raise ValueError("counter keys must be a non-empty tuple")
    result = {}
    for key in keys:
        if key not in before or key not in after:
            raise ValueError(f"missing counter {key}")
        before_value = _non_negative_integer(
            before[key],
            f"{key} before",
        )
        after_value = _non_negative_integer(
            after[key],
            f"{key} after",
        )
        if after_value < before_value:
            raise ValueError(f"counter {key} decreased")
        result[key] = after_value - before_value
    return result


def build_run_metrics(
    *,
    request_start_ns: int,
    request_finish_ns: int,
    token_events: dict[int, list[tuple[int, int]]],
    finished_at_ns: dict[int, int],
    expected_output_tokens: int,
) -> dict:
    request_start_ns = _non_negative_integer(
        request_start_ns,
        "request_start_ns",
    )
    request_finish_ns = _non_negative_integer(
        request_finish_ns,
        "request_finish_ns",
    )
    expected_output_tokens = _positive_integer(
        expected_output_tokens,
        "expected_output_tokens",
    )
    if request_finish_ns <= request_start_ns:
        raise ValueError(
            "request finish must be after request start"
        )
    if not isinstance(token_events, dict) or not token_events:
        raise ValueError("token event inventory must be non-empty")
    if not isinstance(finished_at_ns, dict):
        raise ValueError("finish inventory must be a mapping")
    per_request = []
    for sequence_id in sorted(token_events):
        events = token_events[sequence_id]
        if not isinstance(events, list) or not events:
            raise ValueError(
                f"sequence {sequence_id} token event inventory is empty"
            )
        emitted_tokens = 0
        first_timestamp = None
        previous_timestamp = request_start_ns
        for event_index, event in enumerate(events):
            if (
                not isinstance(event, tuple)
                or len(event) != 2
            ):
                raise ValueError(
                    f"sequence {sequence_id} token event "
                    f"{event_index} is invalid"
                )
            timestamp_ns = _non_negative_integer(
                event[0],
                "token event timestamp",
            )
            token_count = _positive_integer(
                event[1],
                "token event count",
            )
            if timestamp_ns < previous_timestamp:
                raise ValueError(
                    "token event timestamps must be monotonic"
                )
            previous_timestamp = timestamp_ns
            if first_timestamp is None:
                first_timestamp = timestamp_ns
            emitted_tokens += token_count
        if emitted_tokens != expected_output_tokens:
            raise ValueError(
                f"sequence {sequence_id} must emit exactly "
                f"{expected_output_tokens} tokens"
            )
        if sequence_id not in finished_at_ns:
            raise ValueError(
                f"sequence {sequence_id} finish timestamp is missing"
            )
        finish_ns = _non_negative_integer(
            finished_at_ns[sequence_id],
            "finish timestamp",
        )
        if finish_ns < first_timestamp:
            raise ValueError(
                "finish timestamp precedes first token"
            )
        ttft_s = (
            first_timestamp - request_start_ns
        ) / 1_000_000_000
        completion_s = (
            finish_ns - request_start_ns
        ) / 1_000_000_000
        tpot_s = (
            completion_s - ttft_s
        ) / (expected_output_tokens - 1)
        per_request.append({
            "sequence_id": sequence_id,
            "output_tokens": emitted_tokens,
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "completion_latency_s": completion_s,
        })
    batch_elapsed_s = (
        request_finish_ns - request_start_ns
    ) / 1_000_000_000
    request_count = len(per_request)
    total_tokens = request_count * expected_output_tokens
    return {
        "request_count": request_count,
        "total_output_tokens": total_tokens,
        "batch_elapsed_s": batch_elapsed_s,
        "batch_token_throughput_tps": (
            total_tokens / batch_elapsed_s
        ),
        "request_throughput_rps": (
            request_count / batch_elapsed_s
        ),
        "per_request": per_request,
    }


def aggregate_measurements(values: list[float]) -> dict:
    if not isinstance(values, list) or not values:
        raise ValueError("measurement values must be non-empty")
    normalized = [
        _finite_number(value, f"measurement {index}")
        for index, value in enumerate(values)
    ]
    return {
        "count": len(normalized),
        "median": statistics.median(normalized),
        "min": min(normalized),
        "max": max(normalized),
        "pstdev": statistics.pstdev(normalized),
    }


def summarize_step_observations(
    observations: list[dict],
) -> dict:
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    result = {
        "engine_steps": len(observations),
        "prefill_steps": 0,
        "decode_steps": 0,
        "selected_rows": 0,
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "first_target_callbacks": 0,
        "tail_callbacks": 0,
        "speculative_output_tokens": 0,
        "target_callbacks": 0,
        "acceptance_rate": 0.0,
        "timing_ms": {},
    }
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(
                f"observation {index} must be a mapping"
            )
        if observation.get("is_prefill") is True:
            result["prefill_steps"] += 1
        else:
            result["decode_steps"] += 1
        selected = observation.get(
            "speculative_selected_seq_ids",
            [],
        )
        proposal_counts = observation.get(
            "speculative_proposal_token_counts",
            {},
        )
        accepted_counts = observation.get(
            "speculative_accepted_draft_token_counts",
            {},
        )
        output_counts = observation.get(
            "speculative_output_token_counts",
            {},
        )
        for name, value in (
            ("selected rows", selected),
            ("proposal counts", proposal_counts),
            ("accepted counts", accepted_counts),
            ("output counts", output_counts),
        ):
            if not isinstance(
                value,
                list if name == "selected rows" else dict,
            ):
                raise ValueError(
                    f"observation {index} {name} is invalid"
                )
        result["selected_rows"] += len(selected)
        proposal_rows = _non_negative_integer(
            observation.get(
                "speculative_proposal_row_count",
                0,
            ),
            "speculative proposal row count",
        )
        result["proposal_rows"] += proposal_rows
        result["proposed_tokens"] += sum(
            _non_negative_integer(
                count,
                "speculative proposed token count",
            )
            for count in proposal_counts.values()
        )
        result["accepted_draft_tokens"] += sum(
            _non_negative_integer(
                count,
                "speculative accepted token count",
            )
            for count in accepted_counts.values()
        )
        result["speculative_output_tokens"] += sum(
            _non_negative_integer(
                count,
                "speculative output token count",
            )
            for count in output_counts.values()
        )
        result["first_target_callbacks"] += (
            _non_negative_integer(
                observation.get(
                    "speculative_first_target_callback_count",
                    0,
                ),
                "first target callback count",
            )
        )
        result["tail_callbacks"] += _non_negative_integer(
            observation.get(
                "speculative_fixed_q_group_count",
                0,
            ),
            "tail callback count",
        )
        timing = observation.get(
            "speculative_runtime_timing_ms",
            {},
        )
        if not isinstance(timing, dict):
            raise ValueError(
                "speculative runtime timing must be a mapping"
            )
        for key, value in timing.items():
            normalized = _finite_number(
                value,
                f"runtime timing {key}",
            )
            result["timing_ms"][key] = (
                result["timing_ms"].get(key, 0.0)
                + normalized
            )
    result["target_callbacks"] = (
        result["first_target_callbacks"]
        + result["tail_callbacks"]
    )
    if result["proposed_tokens"]:
        result["acceptance_rate"] = (
            result["accepted_draft_tokens"]
            / result["proposed_tokens"]
        )
    return result


def classify_batch_direction(
    baseline: dict,
    candidate: dict,
) -> str:
    try:
        baseline_tpot = _finite_number(
            baseline["tpot_s"]["median"],
            "baseline median TPOT",
        )
        candidate_tpot = _finite_number(
            candidate["tpot_s"]["median"],
            "candidate median TPOT",
        )
        baseline_throughput = _finite_number(
            baseline[
                "batch_token_throughput_tps"
            ]["median"],
            "baseline median token throughput",
        )
        candidate_throughput = _finite_number(
            candidate[
                "batch_token_throughput_tps"
            ]["median"],
            "candidate median token throughput",
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            "direction inputs are incomplete"
        ) from error
    if (
        candidate_tpot < baseline_tpot
        and candidate_throughput > baseline_throughput
    ):
        return "IMPROVED"
    if (
        candidate_tpot > baseline_tpot
        and candidate_throughput < baseline_throughput
    ):
        return "REGRESSED"
    return "MIXED"


def _validate_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_prompt_rows(
    rows: object,
    *,
    batch_size: int,
) -> list[dict]:
    if (
        not isinstance(rows, list)
        or len(rows) != batch_size
    ):
        raise ValueError(
            "prompt row count must match batch size"
        )
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("prompt rows must be mappings")
        token_ids = row.get("token_ids")
        if (
            row.get("prompt_index") != index
            or row.get("token_count") != PROMPT_TOKENS
            or not isinstance(token_ids, list)
            or len(token_ids) != PROMPT_TOKENS
        ):
            raise ValueError(
                f"prompt {index} must contain exactly "
                f"{PROMPT_TOKENS} tokens"
            )
        if _token_ids_sha256(token_ids) != row.get("sha256"):
            raise ValueError(
                f"prompt {index} SHA-256 mismatch"
            )
        normalized.append(copy.deepcopy(row))
    return normalized


def _validate_movement(movement: object) -> dict:
    if not isinstance(movement, dict):
        raise ValueError("movement must be a mapping")
    ranks = movement.get("ranks")
    totals = movement.get("totals")
    if (
        not isinstance(ranks, list)
        or not ranks
        or not isinstance(totals, dict)
    ):
        raise ValueError("movement rank evidence is incomplete")
    recomputed = {key: 0 for key in REAL_MOVEMENT_KEYS}
    for rank, row in enumerate(ranks):
        if not isinstance(row, dict) or row.get("rank") != rank:
            raise ValueError("movement rank inventory mismatch")
        for key in REAL_MOVEMENT_KEYS:
            value = _non_negative_integer(
                row.get(key),
                f"movement rank {rank} {key}",
            )
            recomputed[key] += value
    normalized_totals = {
        key: _non_negative_integer(
            totals.get(key),
            f"movement total {key}",
        )
        for key in REAL_MOVEMENT_KEYS
    }
    if normalized_totals != recomputed:
        raise ValueError(
            "movement totals do not match rank deltas"
        )
    return {
        "ranks": copy.deepcopy(ranks),
        "totals": normalized_totals,
    }


def _validate_memory(memory: object) -> dict:
    if not isinstance(memory, dict):
        raise ValueError("memory must be a mapping")
    ranks = memory.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        raise ValueError("memory rank evidence is incomplete")
    peak_allocated = 0
    peak_reserved = 0
    peak_allocated_delta = 0
    peak_reserved_delta = 0
    for rank, row in enumerate(ranks):
        if not isinstance(row, dict) or row.get("rank") != rank:
            raise ValueError("memory rank inventory mismatch")
        reset = row.get("reset")
        final = row.get("final")
        if not isinstance(reset, dict) or not isinstance(final, dict):
            raise ValueError("memory reset/final rows are incomplete")
        if reset.get("rank") != rank or final.get("rank") != rank:
            raise ValueError("memory reset/final rank mismatch")
        for snapshot_name, snapshot in (
            ("reset", reset),
            ("final", final),
        ):
            for key in (
                "cuda_allocated_bytes",
                "cuda_reserved_bytes",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "kv_capacity_bytes",
            ):
                _non_negative_integer(
                    snapshot.get(key),
                    f"{snapshot_name} memory {key}",
                )
        allocated_delta = max(
            0,
            final["cuda_peak_allocated_bytes"]
            - reset["cuda_allocated_bytes"],
        )
        reserved_delta = max(
            0,
            final["cuda_peak_reserved_bytes"]
            - reset["cuda_reserved_bytes"],
        )
        if (
            row.get("peak_allocated_delta_bytes")
            != allocated_delta
            or row.get("peak_reserved_delta_bytes")
            != reserved_delta
        ):
            raise ValueError("memory peak delta mismatch")
        peak_allocated = max(
            peak_allocated,
            final["cuda_peak_allocated_bytes"],
        )
        peak_reserved = max(
            peak_reserved,
            final["cuda_peak_reserved_bytes"],
        )
        peak_allocated_delta = max(
            peak_allocated_delta,
            allocated_delta,
        )
        peak_reserved_delta = max(
            peak_reserved_delta,
            reserved_delta,
        )
    expected = {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak_allocated_delta,
        "peak_reserved_delta_bytes": peak_reserved_delta,
    }
    for key, value in expected.items():
        if memory.get(key) != value:
            raise ValueError(f"memory {key} mismatch")
    return copy.deepcopy(memory)


def _validate_runtime(
    runtime: object,
    *,
    candidate: bool,
) -> dict:
    if not isinstance(runtime, dict):
        raise ValueError("runtime summary must be a mapping")
    normalized = {}
    integer_keys = (
        "engine_steps",
        "prefill_steps",
        "decode_steps",
        "selected_rows",
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "first_target_callbacks",
        "tail_callbacks",
        "speculative_output_tokens",
        "target_callbacks",
    )
    for key in integer_keys:
        normalized[key] = _non_negative_integer(
            runtime.get(key),
            f"runtime {key}",
        )
    proposed = normalized["proposed_tokens"]
    accepted = normalized["accepted_draft_tokens"]
    if accepted > proposed:
        raise ValueError(
            "runtime accepted tokens exceed proposed tokens"
        )
    expected_rate = (
        accepted / proposed if proposed else 0.0
    )
    actual_rate = _finite_number(
        runtime.get("acceptance_rate"),
        "runtime acceptance rate",
    )
    if not math.isclose(
        actual_rate,
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "runtime accepted-token acceptance rate mismatch"
        )
    if normalized["target_callbacks"] != (
        normalized["first_target_callbacks"]
        + normalized["tail_callbacks"]
    ):
        raise ValueError("runtime target callback mismatch")
    timing_ms = runtime.get("timing_ms")
    if not isinstance(timing_ms, dict):
        raise ValueError("runtime timing must be a mapping")
    normalized_timing = {
        key: _finite_number(
            value,
            f"runtime timing {key}",
        )
        for key, value in timing_ms.items()
    }
    if candidate and (
        normalized["selected_rows"] <= 0
        or normalized["proposal_rows"] <= 0
        or proposed <= 0
        or accepted <= 0
        or normalized["first_target_callbacks"] <= 0
        or normalized["tail_callbacks"] <= 0
    ):
        raise ValueError(
            "candidate accepted proposal evidence is incomplete"
        )
    normalized["acceptance_rate"] = actual_rate
    normalized["timing_ms"] = normalized_timing
    return normalized


def _validate_run(
    run: object,
    *,
    policy: str,
    batch_size: int,
) -> dict:
    if not isinstance(run, dict):
        raise ValueError("run must be a mapping")
    outputs = run.get("outputs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != batch_size
        or any(
            not isinstance(row, list)
            or len(row) != MAX_OUTPUT_TOKENS
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                for token_id in row
            )
            for row in outputs
        )
    ):
        raise ValueError(
            f"run must contain {batch_size} outputs of "
            f"{MAX_OUTPUT_TOKENS} tokens"
        )
    timing = run.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("run timing must be a mapping")
    if timing.get("request_count") != batch_size:
        raise ValueError("run request count mismatch")
    if (
        timing.get("total_output_tokens")
        != batch_size * MAX_OUTPUT_TOKENS
    ):
        raise ValueError("run output token total mismatch")
    per_request = timing.get("per_request")
    if (
        not isinstance(per_request, list)
        or len(per_request) != batch_size
    ):
        raise ValueError("per-request timing is incomplete")
    for index, row in enumerate(per_request):
        if (
            not isinstance(row, dict)
            or row.get("output_tokens") != MAX_OUTPUT_TOKENS
        ):
            raise ValueError("per-request token count mismatch")
        for key in (
            "ttft_s",
            "tpot_s",
            "completion_latency_s",
        ):
            if _finite_number(
                row.get(key),
                f"per-request {key}",
            ) < 0.0:
                raise ValueError(
                    f"per-request {key} must be non-negative"
                )
    for key in (
        "batch_elapsed_s",
        "batch_token_throughput_tps",
        "request_throughput_rps",
    ):
        if _finite_number(
            timing.get(key),
            f"run timing {key}",
        ) <= 0.0:
            raise ValueError(
                f"run timing {key} must be positive"
            )
    movement = _validate_movement(run.get("movement"))
    memory = _validate_memory(run.get("memory"))
    runtime = _validate_runtime(
        run.get("runtime"),
        candidate=policy == "ngram",
    )
    evicted_block_identities = run.get(
        "evicted_block_identities"
    )
    if (
        not isinstance(evicted_block_identities, list)
        or not evicted_block_identities
        or any(
            not isinstance(identity, list)
            or len(identity) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in identity
            )
            for identity in evicted_block_identities
        )
    ):
        raise ValueError(
            "run must record real evicted block identities"
        )
    return {
        "outputs": copy.deepcopy(outputs),
        "timing": copy.deepcopy(timing),
        "runtime": runtime,
        "movement": movement,
        "memory": memory,
        "evicted_block_identities": copy.deepcopy(
            evicted_block_identities
        ),
        "observations": copy.deepcopy(
            run.get("observations", [])
        ),
    }


def validate_worker_result(worker_result: object) -> dict:
    if not isinstance(worker_result, dict):
        raise ValueError("worker result must be a mapping")
    policy = worker_result.get("policy")
    batch_size = worker_result.get("batch_size")
    if policy not in POLICIES or batch_size not in BATCH_SIZES:
        raise ValueError("worker policy/batch cell is invalid")
    prompts = _validate_prompt_rows(
        worker_result.get("prompt_rows"),
        batch_size=batch_size,
    )
    run_groups = {}
    for name, expected_count in (
        ("warmup_runs", WARMUP_RUNS),
        ("parity_runs", PARITY_RUNS),
        ("measured_runs", MEASURED_RUNS),
    ):
        rows = worker_result.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"{name} must be a list")
        if len(rows) != expected_count:
            count_word = (
                "five" if expected_count == 5 else "one"
            )
            raise ValueError(
                f"{name} must contain exactly {count_word} runs"
            )
        run_groups[name] = [
            _validate_run(
                run,
                policy=policy,
                batch_size=batch_size,
            )
            for run in rows
        ]
    tokenizer_identifier = worker_result.get(
        "tokenizer_identifier"
    )
    dtype = worker_result.get("dtype")
    if (
        not isinstance(tokenizer_identifier, str)
        or not tokenizer_identifier
        or not isinstance(dtype, str)
        or not dtype
    ):
        raise ValueError(
            "worker tokenizer/dtype identity is incomplete"
        )
    return {
        "policy": policy,
        "batch_size": batch_size,
        "prompt_rows": prompts,
        **run_groups,
        "tokenizer_identifier": tokenizer_identifier,
        "dtype": dtype,
    }


def _aggregate_worker(worker: dict) -> dict:
    runs = worker["measured_runs"]
    ttft = []
    tpot = []
    completion = []
    throughput = []
    request_throughput = []
    peak_allocated = []
    peak_reserved = []
    movement = {key: 0 for key in REAL_MOVEMENT_KEYS}
    runtime = {
        "selected_rows": 0,
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "first_target_callbacks": 0,
        "tail_callbacks": 0,
        "target_callbacks": 0,
    }
    for run in runs:
        timing = run["timing"]
        ttft.append(statistics.median(
            row["ttft_s"]
            for row in timing["per_request"]
        ))
        tpot.append(statistics.median(
            row["tpot_s"]
            for row in timing["per_request"]
        ))
        completion.append(statistics.median(
            row["completion_latency_s"]
            for row in timing["per_request"]
        ))
        throughput.append(
            timing["batch_token_throughput_tps"]
        )
        request_throughput.append(
            timing["request_throughput_rps"]
        )
        peak_allocated.append(
            run["memory"]["peak_allocated_bytes"]
        )
        peak_reserved.append(
            run["memory"]["peak_reserved_bytes"]
        )
        for key in REAL_MOVEMENT_KEYS:
            movement[key] += run["movement"]["totals"][key]
        for key in runtime:
            runtime[key] += run["runtime"][key]
    proposed = runtime["proposed_tokens"]
    runtime["acceptance_rate"] = (
        runtime["accepted_draft_tokens"] / proposed
        if proposed
        else 0.0
    )
    return {
        "ttft_s": aggregate_measurements(ttft),
        "tpot_s": aggregate_measurements(tpot),
        "completion_latency_s": (
            aggregate_measurements(completion)
        ),
        "batch_token_throughput_tps": (
            aggregate_measurements(throughput)
        ),
        "request_throughput_rps": (
            aggregate_measurements(request_throughput)
        ),
        "peak_allocated_bytes": (
            aggregate_measurements(peak_allocated)
        ),
        "peak_reserved_bytes": (
            aggregate_measurements(peak_reserved)
        ),
        "movement_totals": movement,
        "runtime_totals": runtime,
    }


def _derive_artifact(worker_results: object) -> dict:
    if (
        not isinstance(worker_results, list)
        or len(worker_results) != 4
    ):
        raise ValueError(
            "artifact requires exactly four worker cells"
        )
    cells = {}
    tokenizer_identifier = None
    dtype = None
    for worker_result in worker_results:
        worker = validate_worker_result(worker_result)
        cell_key = (
            f"{worker['policy']}:b{worker['batch_size']}"
        )
        if cell_key in cells:
            raise ValueError("duplicate worker cell")
        if tokenizer_identifier is None:
            tokenizer_identifier = worker[
                "tokenizer_identifier"
            ]
            dtype = worker["dtype"]
        elif (
            worker["tokenizer_identifier"]
            != tokenizer_identifier
            or worker["dtype"] != dtype
        ):
            raise ValueError(
                "worker tokenizer/dtype identities differ"
            )
        aggregate = _aggregate_worker(worker)
        if (
            aggregate["movement_totals"]["h2d_copies"] <= 0
            or aggregate["movement_totals"]["h2d_bytes"] <= 0
        ):
            raise ValueError(
                f"worker cell {cell_key} lacks real H2D movement"
            )
        cells[cell_key] = {
            **worker,
            "aggregate": aggregate,
        }
    expected_cells = {
        f"{policy}:b{batch_size}"
        for policy in POLICIES
        for batch_size in BATCH_SIZES
    }
    if set(cells) != expected_cells:
        raise ValueError("worker cell inventory mismatch")
    for batch_size in BATCH_SIZES:
        baseline = cells[f"baseline:b{batch_size}"]
        candidate = cells[f"ngram:b{batch_size}"]
        if baseline["prompt_rows"] != candidate["prompt_rows"]:
            raise ValueError(
                f"batch {batch_size} prompt parity failed"
            )
        reference_outputs = baseline["parity_runs"][0][
            "outputs"
        ]
        for cell in (baseline, candidate):
            for run in (
                cell["parity_runs"]
                + cell["measured_runs"]
            ):
                if run["outputs"] != reference_outputs:
                    raise ValueError(
                        f"batch {batch_size} exact token parity failed"
                    )
    batch_directions = {
        str(batch_size): classify_batch_direction(
            cells[f"baseline:b{batch_size}"][
                "aggregate"
            ],
            cells[f"ngram:b{batch_size}"]["aggregate"],
        )
        for batch_size in BATCH_SIZES
    }
    if all(
        value == "IMPROVED"
        for value in batch_directions.values()
    ):
        direction = "POSITIVE"
    elif any(
        value == "REGRESSED"
        for value in batch_directions.values()
    ):
        direction = "NEGATIVE"
    else:
        direction = "MIXED"
    return {
        "cells": cells,
        "batch_directions": batch_directions,
        "direction": direction,
        "tokenizer_identifier": tokenizer_identifier,
        "dtype": dtype,
    }


def build_performance_artifact(
    *,
    worker_results: list[dict],
    environment: dict,
    source_files: dict[str, str],
) -> dict:
    if not isinstance(environment, dict):
        raise ValueError("environment must be a mapping")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source files must be a non-empty mapping")
    normalized_sources = {}
    for path, digest in source_files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("source path must be non-empty")
        normalized_sources[path] = _validate_sha256(
            digest,
            f"source hash {path}",
        )
    derived = _derive_artifact(worker_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "claim_scope": CLAIM_SCOPE,
        "direction": derived["direction"],
        "batch_directions": derived["batch_directions"],
        "environment": {
            **copy.deepcopy(environment),
            "tokenizer_identifier": (
                derived["tokenizer_identifier"]
            ),
            "dtype": derived["dtype"],
        },
        "campaign": {
            "tensor_parallel_size": 1,
            "prompt_tokens": PROMPT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "batch_sizes": list(BATCH_SIZES),
            "policies": list(POLICIES),
            "temperature": 0.0,
            "ignore_eos": True,
            "warmup_runs": WARMUP_RUNS,
            "parity_runs": PARITY_RUNS,
            "measured_runs": MEASURED_RUNS,
            "ngram_size": NGRAM_SIZE,
            "max_proposal_tokens": (
                MAX_PROPOSAL_TOKENS
            ),
        },
        "cells": derived["cells"],
        "source_files": normalized_sources,
        "limitations": list(LIMITATIONS),
    }


def _artifact_values_equivalent(
    stored: object,
    derived: object,
) -> bool:
    if isinstance(stored, bool) or isinstance(derived, bool):
        return stored is derived
    if isinstance(stored, int) or isinstance(derived, int):
        return (
            isinstance(stored, int)
            and not isinstance(stored, bool)
            and isinstance(derived, int)
            and not isinstance(derived, bool)
            and stored == derived
        )
    if isinstance(stored, float) or isinstance(derived, float):
        if not isinstance(stored, (int, float)) or not isinstance(
            derived,
            (int, float),
        ):
            return False
        return math.isclose(
            float(stored),
            float(derived),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if isinstance(stored, dict) or isinstance(derived, dict):
        if not isinstance(stored, dict) or not isinstance(
            derived,
            dict,
        ):
            return False
        if set(stored) != set(derived):
            return False
        return all(
            _artifact_values_equivalent(
                stored[key],
                derived[key],
            )
            for key in stored
        )
    if isinstance(stored, list) or isinstance(derived, list):
        if not isinstance(stored, list) or not isinstance(
            derived,
            list,
        ):
            return False
        return (
            len(stored) == len(derived)
            and all(
                _artifact_values_equivalent(left, right)
                for left, right in zip(stored, derived)
            )
        )
    return stored == derived


def validate_performance_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a mapping")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("artifact schema version mismatch")
    if artifact.get("status") != "PASS":
        raise ValueError("artifact status must be PASS")
    if artifact.get("classification") != CLASSIFICATION:
        raise ValueError(
            "artifact classification must remain NOT_PROMOTABLE"
        )
    cells = artifact.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("artifact cells must be a mapping")
    worker_results = []
    for key in sorted(cells):
        cell = cells[key]
        if not isinstance(cell, dict):
            raise ValueError("artifact cell must be a mapping")
        worker_result = {
            field: copy.deepcopy(cell[field])
            for field in (
                "policy",
                "batch_size",
                "prompt_rows",
                "warmup_runs",
                "parity_runs",
                "measured_runs",
                "tokenizer_identifier",
                "dtype",
            )
        }
        worker_results.append(worker_result)
    derived = _derive_artifact(worker_results)
    for key, cell in derived["cells"].items():
        if not _artifact_values_equivalent(
            cells[key].get("aggregate"),
            cell["aggregate"],
        ):
            raise ValueError(
                f"artifact aggregate mismatch for {key}"
            )
    if artifact.get("batch_directions") != (
        derived["batch_directions"]
    ):
        raise ValueError("artifact batch direction mismatch")
    if artifact.get("direction") != derived["direction"]:
        raise ValueError("artifact direction mismatch")
    source_files = artifact.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError(
            "artifact source files must be non-empty"
        )
    for path, digest in source_files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("artifact source path is invalid")
        _validate_sha256(
            digest,
            f"artifact source hash {path}",
        )
    return {
        "status": "PASS",
        "classification": CLASSIFICATION,
        "direction": derived["direction"],
        "batch_directions": derived["batch_directions"],
    }


def hash_source_files(
    *,
    repo_root: Path,
    source_files: tuple[str, ...],
) -> dict[str, str]:
    repo_root = Path(repo_root)
    if (
        not isinstance(source_files, tuple)
        or not source_files
    ):
        raise ValueError(
            "source_files must be a non-empty tuple"
        )
    result = {}
    for relative_path in source_files:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError("source path must be safe and relative")
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        result[relative_path] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    return result


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _subprocess_worker_runner(
    command,
    *,
    log_path,
    cwd,
) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _default_environment(
    *,
    model_path: str,
    command: list[str],
) -> dict:
    try:
        import torch

        torch_version = str(torch.__version__)
        device_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "unavailable"
        )
    except Exception:
        torch_version = "unavailable"
        device_name = "unavailable"
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_identifier": Path(model_path).name,
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "device_name": device_name,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "command": list(command),
    }


def run_performance_gate(
    *,
    model_path: str,
    output_path: Path,
    repo_root: Path,
    worker_script: Path,
    worker_runner=_subprocess_worker_runner,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
    environment: dict | None = None,
) -> dict:
    output_path = Path(output_path)
    repo_root = Path(repo_root)
    worker_script = Path(worker_script)
    worker_directory = output_path.parent / "workers"
    worker_directory.mkdir(parents=True, exist_ok=True)
    worker_results = []
    commands = []
    for policy in POLICIES:
        for batch_size in BATCH_SIZES:
            worker_output = (
                worker_directory
                / f"worker-{policy}-b{batch_size}.json"
            )
            worker_log = (
                worker_directory
                / f"worker-{policy}-b{batch_size}.log"
            )
            command = [
                python_executable,
                str(worker_script),
                "--model",
                model_path,
                "--policy",
                policy,
                "--batch-size",
                str(batch_size),
                "--out",
                str(worker_output),
            ]
            commands.append(command)
            status = worker_runner(
                command,
                log_path=worker_log,
                cwd=repo_root,
            )
            if status != 0:
                diagnostic = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "FAIL",
                    "classification": CLASSIFICATION,
                    "failure_reason": (
                        f"worker_{policy}_b{batch_size}_failed"
                    ),
                    "worker_status": status,
                    "worker_log": str(worker_log),
                    "commands": commands,
                    "limitations": list(LIMITATIONS),
                }
                write_json_atomic(output_path, diagnostic)
                raise RuntimeError(
                    f"worker {policy} b{batch_size} failed "
                    f"with status {status}"
                )
            if not worker_output.is_file():
                raise RuntimeError(
                    f"worker output is missing: {worker_output}"
                )
            worker_results.append(
                json.loads(
                    worker_output.read_text(
                        encoding="utf-8"
                    )
                )
            )
    command_environment = (
        _default_environment(
            model_path=model_path,
            command=[
                python_executable,
                str(Path(__file__).resolve()),
                "run",
                "--model",
                model_path,
                "--out",
                str(output_path),
            ],
        )
        if environment is None
        else copy.deepcopy(environment)
    )
    command_environment["worker_commands"] = commands
    artifact = build_performance_artifact(
        worker_results=worker_results,
        environment=command_environment,
        source_files=hash_source_files(
            repo_root=repo_root,
            source_files=source_files,
        ),
    )
    write_json_atomic(output_path, artifact)
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command != "run":
        raise ValueError(f"unsupported command: {args.command}")
    repo_root = Path(__file__).resolve().parents[1]
    run_performance_gate(
        model_path=args.model,
        output_path=Path(args.out),
        repo_root=repo_root,
        worker_script=(
            repo_root
            / "tools"
            / "speculative_runtime_performance_worker.py"
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
