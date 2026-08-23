#!/usr/bin/env python3
"""Source-bound four-arm benchmark for split-phase K8 exact bursts."""

from __future__ import annotations

from pathlib import Path
import statistics
import time

from tools import profile_exact_greedy_decode_burst as _base


CASE_SCHEMA_VERSION = "exact-burst-split-phase.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "exact-burst-split-phase.correctness.v1"
)
SUMMARY_SCHEMA_VERSION = "exact-burst-split-phase.summary.v1"
WORKLOAD_SCHEMA_VERSION = "exact-burst-split-phase.workload.v1"
SOURCE_SCHEMA_VERSION = "exact-burst-split-phase.source.v1"
CORRECTNESS_TRACE_IDENTITY = (
    "gate-only-exact-burst-split-phase-correctness-v1"
)

POLICIES = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k8",
    "decode_burst_k8_split_phase",
)
CONTEXT_CASES = _base.CONTEXT_CASES
SAMPLING_POINTS = _base.SAMPLING_POINTS
POLICY_CONFIGS = {
    "host_greedy": {
        "enabled": False,
        "split": False,
        "width": 1,
        "selectable": False,
        "entrypoint": "ordinary",
    },
    "decode_burst_k4": {
        "enabled": True,
        "split": False,
        "width": 4,
        "selectable": False,
        "entrypoint": "production",
    },
    "decode_burst_k8": {
        "enabled": True,
        "split": False,
        "width": 8,
        "selectable": False,
        "entrypoint": "production",
    },
    "decode_burst_k8_split_phase": {
        "enabled": True,
        "split": True,
        "width": 8,
        "selectable": True,
        "entrypoint": "production",
        "profile_ordinary_tail_after_full_bursts": True,
        "correctness_sampled_logit_d2h_calls": 2,
        "ordinary_tail_sampling_points": ("decode-final",),
    },
}

SPLIT_COUNTER_FIELDS = (
    "prefix_commits",
    "suffix_commits",
    "prefix_committed_tokens",
    "suffix_committed_tokens",
    "prefix_publication_tickets",
    "suffix_publication_tickets",
    "prefix_token_d2h_calls",
    "suffix_token_d2h_calls",
    "prefix_token_d2h_bytes",
    "suffix_token_d2h_bytes",
    "prefix_phase_waits",
    "suffix_phase_waits",
    "suffix_drains",
)
SPLIT_MAP_FIELDS = ("split_phase_failure_counts",)
SPLIT_INVENTORY_FIELDS = (
    "parent_lease_count",
    "prefix_row_count",
    "suffix_row_count",
    "prefix_ticket_count",
    "suffix_ticket_count",
    "replay_count",
    "prefix_d2h_calls",
    "suffix_d2h_calls",
    "prefix_d2h_bytes",
    "suffix_d2h_bytes",
    "prefix_pending_suffix_count",
    "suffix_cleared_count",
    "unexpected_scheduler_calls",
)

CAPTURE_COST_FIELDS = _base.CAPTURE_COST_FIELDS
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/exact_greedy_decode_burst_split_phase.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/profile_exact_burst_split_phase.py",
    "tools/test_profile_exact_burst_split_phase.py",
    "tools/exact_burst_split_phase_gate.py",
    "tools/test_exact_burst_split_phase_gate.py",
    "tools/exact_burst_split_phase_verify.py",
    "tools/test_exact_burst_split_phase_verify.py",
    "tools/run_exact_burst_split_phase_remote.py",
    "tools/test_run_exact_burst_split_phase_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)

_ORIGINAL_BURST_COUNTER_FIELDS = _base.BURST_COUNTER_FIELDS
_ORIGINAL_VALIDATE_BURST_SUMMARY = _base._validate_burst_summary
_ORIGINAL_VALIDATE_CASE_ROW = _base.validate_case_row
_ORIGINAL_COUNTER_DELTA = _base._counter_delta
_ORIGINAL_COMBINED_SUMMARY = _base._combined_summary
_ORIGINAL_RUN_REQUEST = _base._run_request
_ORIGINAL_RUN_CORRECTNESS_PROBE = _base.run_correctness_probe
_ORIGINAL_CORRECTNESS_TRACE_FOR_STEP = (
    _base.correctness_trace_for_step
)


def _empty_split_inventory() -> dict:
    return {
        **{field: 0 for field in SPLIT_INVENTORY_FIELDS},
        "host_visible_gaps_ns": [],
    }


def _phase_gap(row: dict) -> int:
    value = row.get(
        "host_visible_gap_ns",
        row.get(
            "exact_greedy_decode_burst_host_visible_gap_ns",
            0,
        ),
    )
    return _base._require_non_negative_int(
        value,
        "split phase host-visible gap",
    )


def summarize_split_phase_observations(
    observations: list[dict],
) -> dict:
    if not isinstance(observations, list):
        raise ValueError("split phase observations must be a list")
    if not observations:
        return _empty_split_inventory()
    if len(observations) % 2:
        raise ValueError(
            "split phase observations are not ordered parent pairs"
        )
    inventory = _empty_split_inventory()
    parents = set()
    prefix_tickets = set()
    suffix_tickets = set()
    for index in range(0, len(observations), 2):
        prefix = observations[index]
        suffix = observations[index + 1]
        if (
            not isinstance(prefix, dict)
            or not isinstance(suffix, dict)
            or prefix.get("phase_published") != "prefix"
            or suffix.get("phase_published") != "suffix"
            or prefix.get("parent_lease_identity_sha256")
            != suffix.get("parent_lease_identity_sha256")
        ):
            raise ValueError(
                "split phase observations are not ordered parent pairs"
            )
        parent = prefix.get("parent_lease_identity_sha256")
        prefix_ticket = prefix.get(
            "prefix_ticket_identity_sha256"
        )
        suffix_ticket = prefix.get(
            "suffix_ticket_identity_sha256"
        )
        for value, name in (
            (parent, "split parent lease identity"),
            (prefix_ticket, "split prefix ticket identity"),
            (suffix_ticket, "split suffix ticket identity"),
        ):
            _base._validate_digest(value, name)
        if (
            suffix.get("prefix_ticket_identity_sha256")
            != prefix_ticket
            or suffix.get("suffix_ticket_identity_sha256")
            != suffix_ticket
            or prefix_ticket == suffix_ticket
            or parent in parents
            or prefix_ticket in prefix_tickets
            or suffix_ticket in suffix_tickets
        ):
            raise ValueError(
                "split phase observations are not ordered parent pairs"
            )
        for row, phase in ((prefix, "prefix"), (suffix, "suffix")):
            if (
                row.get("split_phase_attempted") is not True
                or row.get("split_phase_accepted") is not True
                or row.get("phase_token_count") != 4
                or row.get("replay_count") != 8
                or row.get("prefix_d2h_calls") != 1
                or row.get("suffix_d2h_calls") != 1
                or row.get("prefix_d2h_bytes") != 32
                or row.get("suffix_d2h_bytes") != 32
            ):
                raise ValueError(
                    f"split {phase} observation inventory mismatch"
                )
        if (
            prefix.get("pending_suffix") is not True
            or suffix.get("pending_suffix") is not False
        ):
            raise ValueError(
                "split phase pending-suffix lifecycle mismatch"
            )
        suffix_schedule_calls = _base._require_non_negative_int(
            suffix.get("scheduler_schedule_calls"),
            "suffix scheduler schedule calls",
        )
        parents.add(parent)
        prefix_tickets.add(prefix_ticket)
        suffix_tickets.add(suffix_ticket)
        inventory["prefix_row_count"] += 1
        inventory["suffix_row_count"] += 1
        inventory["replay_count"] += 8
        inventory["prefix_d2h_calls"] += 1
        inventory["suffix_d2h_calls"] += 1
        inventory["prefix_d2h_bytes"] += 32
        inventory["suffix_d2h_bytes"] += 32
        inventory["prefix_pending_suffix_count"] += 1
        inventory["suffix_cleared_count"] += 1
        inventory["unexpected_scheduler_calls"] += (
            suffix_schedule_calls
        )
        inventory["host_visible_gaps_ns"].extend(
            (_phase_gap(prefix), _phase_gap(suffix))
        )
    inventory["parent_lease_count"] = len(parents)
    inventory["prefix_ticket_count"] = len(prefix_tickets)
    inventory["suffix_ticket_count"] = len(suffix_tickets)
    return inventory


def _activate_contract() -> None:
    _base.CASE_SCHEMA_VERSION = CASE_SCHEMA_VERSION
    _base.CORRECTNESS_SCHEMA_VERSION = CORRECTNESS_SCHEMA_VERSION
    _base.SUMMARY_SCHEMA_VERSION = SUMMARY_SCHEMA_VERSION
    _base.WORKLOAD_SCHEMA_VERSION = WORKLOAD_SCHEMA_VERSION
    _base.SOURCE_SCHEMA_VERSION = SOURCE_SCHEMA_VERSION
    _base.CORRECTNESS_TRACE_IDENTITY = (
        CORRECTNESS_TRACE_IDENTITY
    )
    _base.POLICIES = POLICIES
    _base.POLICY_CONFIGS = POLICY_CONFIGS
    _base.SOURCE_FILES = SOURCE_FILES
    _base.BURST_COUNTER_FIELDS = (
        _ORIGINAL_BURST_COUNTER_FIELDS + SPLIT_COUNTER_FIELDS
    )


def _validate_reason_counts(value, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(reason, str)
        or not reason
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for reason, count in value.items()
    ):
        raise ValueError(f"{name} is invalid")
    return dict(sorted(value.items()))


def _validate_burst_summary(
    summary,
    *,
    policy: str,
    correctness_trace: bool,
) -> dict:
    required = set(SPLIT_COUNTER_FIELDS) | set(SPLIT_MAP_FIELDS)
    if not isinstance(summary, dict) or required - set(summary):
        raise ValueError("split phase summary fields are missing")
    normalized = _ORIGINAL_VALIDATE_BURST_SUMMARY(
        summary,
        policy=policy,
        correctness_trace=correctness_trace,
    )
    for field in SPLIT_COUNTER_FIELDS:
        normalized[field] = _base._require_non_negative_int(
            summary[field],
            f"split phase summary {field}",
        )
    normalized["split_phase_failure_counts"] = (
        _validate_reason_counts(
            summary["split_phase_failure_counts"],
            "split phase failure counts",
        )
    )
    split = POLICY_CONFIGS[policy]["split"]
    split_activity = any(
        normalized[field] for field in SPLIT_COUNTER_FIELDS
    ) or bool(normalized["split_phase_failure_counts"])
    if not split:
        if split_activity:
            raise ValueError(
                "non-split policy reported split phase activity"
            )
        return normalized
    commits = normalized["commits"]
    expected_tokens = commits * 4
    for field in (
        "prefix_commits",
        "suffix_commits",
        "prefix_publication_tickets",
        "suffix_publication_tickets",
        "prefix_token_d2h_calls",
        "suffix_token_d2h_calls",
        "prefix_phase_waits",
        "suffix_phase_waits",
        "suffix_drains",
    ):
        if normalized[field] != commits:
            raise ValueError(
                f"split phase commit inventory mismatch: {field}"
            )
    for field in (
        "prefix_committed_tokens",
        "suffix_committed_tokens",
    ):
        if normalized[field] != expected_tokens:
            raise ValueError(
                f"split phase token inventory mismatch: {field}"
            )
    for field in (
        "prefix_token_d2h_bytes",
        "suffix_token_d2h_bytes",
    ):
        if normalized[field] != commits * 32:
            raise ValueError(
                f"split phase byte inventory mismatch: {field}"
            )
    if (
        normalized["committed_tokens"] != commits * 8
        or normalized["target_model_forwards"] != commits * 8
        or normalized["graph_replays"] != commits * 8
    ):
        raise ValueError("split phase replay inventory mismatch")
    if (
        normalized["final_token_d2h_calls"]
        or normalized["final_token_d2h_bytes"]
    ):
        raise ValueError(
            "split phase used one-phase final token D2H"
        )
    if normalized["split_phase_failure_counts"]:
        raise ValueError("split phase failure inventory is nonzero")
    tail_tokens = 127 - normalized["committed_tokens"]
    accepted_tail_leases = max(0, tail_tokens - 1)
    expected_fallback_counts = {}
    if accepted_tail_leases:
        expected_fallback_counts["split_phase_requires_k8"] = (
            accepted_tail_leases
        )
    if tail_tokens:
        expected_fallback_counts["insufficient_output_budget"] = 1
    if (
        tail_tokens < 0
        or normalized["fallback_counts"]
        != expected_fallback_counts
    ):
        raise ValueError("split phase ordinary-tail inventory mismatch")
    return normalized


def _validate_split_inventory(
    value,
    summary: dict,
    *,
    split: bool,
) -> dict:
    if (
        not isinstance(value, dict)
        or set(value)
        != set(SPLIT_INVENTORY_FIELDS)
        | {"host_visible_gaps_ns"}
    ):
        raise ValueError("split phase inventory fields mismatch")
    normalized = {
        field: _base._require_non_negative_int(
            value[field],
            f"split phase inventory {field}",
        )
        for field in SPLIT_INVENTORY_FIELDS
    }
    gaps = value["host_visible_gaps_ns"]
    if not isinstance(gaps, list):
        raise ValueError(
            "split phase host-visible gaps must be a list"
        )
    normalized["host_visible_gaps_ns"] = [
        _base._require_non_negative_int(
            gap,
            f"split phase host-visible gap {index}",
        )
        for index, gap in enumerate(gaps)
    ]
    if not split:
        if (
            any(
                normalized[field]
                for field in SPLIT_INVENTORY_FIELDS
            )
            or normalized["host_visible_gaps_ns"]
        ):
            raise ValueError(
                "non-split policy reported split phase activity"
            )
        return normalized
    commits = summary["commits"]
    expected = {
        "parent_lease_count": commits,
        "prefix_row_count": commits,
        "suffix_row_count": commits,
        "prefix_ticket_count": commits,
        "suffix_ticket_count": commits,
        "replay_count": commits * 8,
        "prefix_d2h_calls": commits,
        "suffix_d2h_calls": commits,
        "prefix_d2h_bytes": commits * 32,
        "suffix_d2h_bytes": commits * 32,
        "prefix_pending_suffix_count": commits,
        "suffix_cleared_count": commits,
    }
    for field, expected_value in expected.items():
        if normalized[field] != expected_value:
            raise ValueError(
                f"split phase observation inventory mismatch: {field}"
            )
    if normalized["unexpected_scheduler_calls"]:
        raise ValueError(
            "split phase scheduled during suffix drain"
        )
    if len(normalized["host_visible_gaps_ns"]) != commits * 2:
        raise ValueError(
            "split phase host-visible gap inventory mismatch"
        )
    return normalized


def validate_case_row(row) -> dict:
    normalized = _ORIGINAL_VALIDATE_CASE_ROW(row)
    if "split_phase_inventory" not in row:
        raise ValueError("split phase inventory is missing")
    split = POLICY_CONFIGS[row["policy"]]["split"]
    inventory = _validate_split_inventory(
        row["split_phase_inventory"],
        normalized["exact_greedy_decode_burst_summary"],
        split=split,
    )
    if split:
        if (
            inventory["host_visible_gaps_ns"]
            != row["host_visible_burst_gaps_ns"]
        ):
            raise ValueError(
                "split phase gap inventory does not match request"
            )
    normalized["split_phase_inventory"] = inventory
    return normalized


def _counter_delta(before: dict, after: dict) -> dict:
    result = _ORIGINAL_COUNTER_DELTA(before, after)
    for field in SPLIT_MAP_FIELDS:
        before_map = before.get(field, {})
        after_map = after.get(field, {})
        result[field] = {}
        for key in sorted(set(before_map) | set(after_map)):
            difference = int(after_map.get(key, 0)) - int(
                before_map.get(key, 0)
            )
            if difference < 0:
                raise RuntimeError(
                    f"split phase map counter decreased: {field}"
                )
            if difference:
                result[field][str(key)] = difference
    return result


def _combined_summary(
    llm,
    before: tuple[dict, dict],
    *,
    correctness_trace: bool = False,
) -> dict:
    result = _ORIGINAL_COMBINED_SUMMARY(
        llm,
        before,
        correctness_trace=correctness_trace,
    )
    runner = _counter_delta(
        before[0],
        llm.model_runner.exact_greedy_decode_burst_summary(),
    )
    scheduler = _counter_delta(
        before[1],
        llm.scheduler.exact_greedy_decode_burst_summary(),
    )
    for field in SPLIT_COUNTER_FIELDS:
        result[field] = scheduler[field]
    result["sampled_logit_d2h_calls"] = runner[
        "sampled_logit_d2h_calls"
    ]
    result["split_phase_failure_counts"] = {
        reason: (
            runner["split_phase_failure_counts"].get(reason, 0)
            + scheduler["split_phase_failure_counts"].get(reason, 0)
        )
        for reason in sorted(
            set(runner["split_phase_failure_counts"])
            | set(scheduler["split_phase_failure_counts"])
        )
    }
    return result


def _construct_llm(
    *,
    model: str,
    prompt_tokens: int,
    generated_tokens: int,
    gpu_memory_utilization: float,
    policy: str,
):
    from tinyvllm import LLM

    config = POLICY_CONFIGS[policy]
    return LLM(
        model,
        max_num_batched_tokens=prompt_tokens + generated_tokens,
        max_num_seqs=1,
        max_model_len=prompt_tokens + generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=False,
        zero_temperature_greedy_fast_path=True,
        graph_resident_greedy_tail=False,
        exact_greedy_decode_burst=config["enabled"],
        exact_greedy_decode_burst_split_phase=config["split"],
        exact_greedy_decode_burst_continuation=False,
        exact_greedy_decode_burst_tokens=max(2, config["width"]),
    )


def _run_request(
    llm,
    *,
    prompt: list[int],
    generated_tokens: int,
    policy: str,
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
    amortized_tpot = []
    burst_gaps = []
    phase_observations = []
    final_outputs = None
    while not llm.is_finished():
        step_started_ns = time.perf_counter_ns()
        outputs, _num_tokens = llm.step(
            **_base._step_kwargs(
                policy,
                correctness_trace=False,
            )
        )
        step_finished_ns = time.perf_counter_ns()
        observation = llm.last_step_observation
        emitted = sum(
            len(tokens)
            for tokens in observation[
                "new_completion_tokens_by_seq"
            ].values()
        )
        if observation.get("split_phase_accepted") is True:
            phase_observations.append(dict(observation))
        if emitted:
            if first_token_ns is None:
                first_token_ns = step_finished_ns
            elif not observation["is_prefill"]:
                per_token = (
                    step_finished_ns - step_started_ns
                ) / emitted
                amortized_tpot.extend([per_token] * emitted)
                gap = (
                    _phase_gap(observation)
                    if observation.get("phase_published")
                    in ("prefix", "suffix")
                    else observation.get(
                        "exact_greedy_decode_burst_"
                        "host_visible_gap_ns",
                        0,
                    )
                )
                if gap:
                    burst_gaps.append(int(gap))
        if outputs:
            final_outputs = outputs
    import torch

    torch.cuda.synchronize()
    finished_ns = time.perf_counter_ns()
    if first_token_ns is None:
        raise RuntimeError("request produced no first token")
    if not isinstance(final_outputs, list) or len(final_outputs) != 1:
        raise RuntimeError("request completion output is incomplete")
    output_ids = list(final_outputs[0][1])
    if len(output_ids) != generated_tokens:
        raise RuntimeError("generated token inventory mismatch")
    if len(amortized_tpot) != generated_tokens - 1:
        raise RuntimeError("amortized TPOT inventory mismatch")
    decode_host_ns = []
    decode_cuda_ns = []
    if profile_label is not None:
        profile = llm.finalize_decode_internal_profile(
            already_synchronized=True,
            timeout_s=60.0,
        )
        rank_rows = profile.get("ranks", ())
        if profile.get("rank_inventory") != [0] or len(rank_rows) != 1:
            raise RuntimeError(
                "Stage-1 worker requires tensor parallel size one"
            )
        decode_steps = sorted(
            (
                row
                for row in rank_rows[0]["steps"]
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
    inventory = summarize_split_phase_observations(
        phase_observations
    )
    if POLICY_CONFIGS[policy]["split"]:
        if inventory["host_visible_gaps_ns"] != burst_gaps:
            raise RuntimeError(
                "split phase gap collection is inconsistent"
            )
    elif phase_observations:
        raise RuntimeError(
            "non-split policy produced split phase observations"
        )
    return {
        "output_token_ids": output_ids,
        "output_text": llm.tokenizer.decode(output_ids),
        "ttft_ns": first_token_ns - started_ns,
        "e2e_ns": finished_ns - started_ns,
        "amortized_tpot_samples_ns": amortized_tpot,
        "decode_host_ns": decode_host_ns,
        "decode_cuda_ns": decode_cuda_ns,
        "host_visible_burst_gaps_ns": burst_gaps,
        "split_phase_inventory": inventory,
    }


def correctness_trace_for_step(
    policy: str,
    *,
    emitted_total: int,
    generated_tokens: int,
) -> bool:
    if policy != "decode_burst_k8_split_phase":
        return _ORIGINAL_CORRECTNESS_TRACE_FOR_STEP(
            policy,
            emitted_total=emitted_total,
            generated_tokens=generated_tokens,
        )
    if emitted_total <= 0 or emitted_total >= generated_tokens:
        return False
    remaining = generated_tokens - emitted_total
    if remaining < 8:
        return False
    sampled_indices = {
        _base._sampling_output_index(point, generated_tokens)
        for point in SAMPLING_POINTS[1:]
    }
    return any(
        emitted_total <= output_index < emitted_total + 8
        for output_index in sampled_indices
    )


def _sampled_local_ordinals(policy: str) -> tuple[int, ...]:
    if not POLICY_CONFIGS[policy]["enabled"]:
        return ()
    decode_ordinals = (0, 63, 126)
    width = POLICY_CONFIGS[policy]["width"]
    return tuple(sorted({ordinal % width for ordinal in decode_ordinals}))


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
    if policy != "decode_burst_k8_split_phase":
        return _ORIGINAL_RUN_CORRECTNESS_PROBE(
            model=model,
            run_dir=run_dir,
            run_tag=run_tag,
            source_commit=source_commit,
            policy=policy,
            context_bucket=context_bucket,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    from tinyvllm import SamplingParams

    llm = _construct_llm(
        model=model,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        policy=policy,
    )
    try:
        llm.model_runner.call(
            "capture_exact_greedy_decode_burst_correctness_graph",
            _sampled_local_ordinals(policy),
        )
        llm.enable_step_logits_authority_recording(
            True,
            timeout_s=60.0,
        )
        before = _base._runner_summaries(llm)
        llm.add_request(
            _base._make_prompt(prompt_tokens, offset=90_001),
            SamplingParams(
                temperature=0.0,
                max_tokens=generated_tokens,
                ignore_eos=True,
            ),
        )
        captured = {}
        final_outputs = None
        emitted_total = 0
        while not llm.is_finished():
            trace_this_step = correctness_trace_for_step(
                policy,
                emitted_total=emitted_total,
                generated_tokens=generated_tokens,
            )
            outputs, _num_tokens = llm.step(
                **_base._step_kwargs(
                    policy,
                    correctness_trace=trace_this_step,
                )
            )
            observation = llm.last_step_observation
            emitted = sum(
                len(tokens)
                for tokens in observation[
                    "new_completion_tokens_by_seq"
                ].values()
            )
            if emitted_total == 0:
                logits = (
                    llm.read_step_logits_authority()
                    .detach()
                    .to(dtype=__import__("torch").float32)
                    .contiguous()
                )
                captured["prefill-final"] = {
                    "logits": logits,
                    "trace_graph_identity_sha256": None,
                    "selected_replay_ordinal": None,
                    "sampled_logit_d2h_calls": 0,
                }
            sampled = observation.get(
                "exact_greedy_decode_burst_sampled_logits",
                (),
            )
            for local_ordinal, values in sampled:
                output_index = emitted_total + int(local_ordinal)
                for point in SAMPLING_POINTS[1:]:
                    if output_index == _base._sampling_output_index(
                        point,
                        generated_tokens,
                    ):
                        captured[point] = {
                            "logits": values,
                            "trace_graph_identity_sha256": (
                                observation[
                                    "exact_greedy_decode_burst_"
                                    "graph_identity_sha256"
                                ]
                            ),
                            "selected_replay_ordinal": int(
                                local_ordinal
                            ),
                            "sampled_logit_d2h_calls": int(
                                observation[
                                    "exact_greedy_decode_burst_"
                                    "sampled_logit_d2h_calls"
                                ]
                            ),
                        }
            output_index = emitted_total
            if (
                not sampled
                and emitted == 1
                and output_index
                in {
                    _base._sampling_output_index(
                        point,
                        generated_tokens,
                    )
                    for point in SAMPLING_POINTS[1:]
                }
            ):
                logits = (
                    llm.read_step_logits_authority()
                    .detach()
                    .to(dtype=__import__("torch").float32)
                    .contiguous()
                )
                for point in SAMPLING_POINTS[1:]:
                    if output_index == _base._sampling_output_index(
                        point,
                        generated_tokens,
                    ):
                        captured[point] = {
                            "logits": logits,
                            "trace_graph_identity_sha256": None,
                            "selected_replay_ordinal": None,
                            "sampled_logit_d2h_calls": 0,
                        }
            emitted_total += emitted
            if outputs:
                final_outputs = outputs
        if set(captured) != set(SAMPLING_POINTS):
            raise RuntimeError(
                "correctness sampling points are incomplete"
            )
        if not isinstance(final_outputs, list) or len(final_outputs) != 1:
            raise RuntimeError("correctness output is incomplete")
        output_ids = list(final_outputs[0][1])
        output_text_sha256 = _base.sha256_text(
            llm.tokenizer.decode(output_ids)
        )
        summary = _combined_summary(
            llm,
            before,
            correctness_trace=True,
        )
        rows = []
        for point in SAMPLING_POINTS:
            sample = captured[point]
            logits = sample["logits"]
            if hasattr(logits, "view"):
                shape = [int(value) for value in logits.shape]
                values = logits.view(-1).tolist()
            else:
                values = list(logits)
                shape = [1, len(values)]
            sidecar = _base.write_float32_sidecar(
                run_dir,
                f"logits/{context_bucket}-{policy}-{point}.f32",
                values,
            )
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
                "correctness_trace": True,
                "trace_identity": CORRECTNESS_TRACE_IDENTITY,
                "trace_graph_identity_sha256": sample[
                    "trace_graph_identity_sha256"
                ],
                "selected_replay_ordinal": sample[
                    "selected_replay_ordinal"
                ],
                "sampled_logit_d2h_calls": sample[
                    "sampled_logit_d2h_calls"
                ],
                "exact_greedy_decode_burst_summary": summary,
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


_activate_contract()
_base._validate_burst_summary = _validate_burst_summary
_base.validate_case_row = validate_case_row
_base._counter_delta = _counter_delta
_base._combined_summary = _combined_summary
_base._construct_llm = _construct_llm
_base._run_request = _run_request
_base.correctness_trace_for_step = correctness_trace_for_step
_base._sampled_local_ordinals = _sampled_local_ordinals
_base.run_correctness_probe = run_correctness_probe

policy_order = _base.policy_order
context_cases = _base.context_cases
performance_identities = _base.performance_identities
correctness_identities = _base.correctness_identities
correctness_uses_burst_trace = _base.correctness_uses_burst_trace
correctness_point_uses_burst_trace = (
    _base.correctness_point_uses_burst_trace
)
runtime_environment_manifest = _base.runtime_environment_manifest
build_workload_manifest = _base.build_workload_manifest
validate_correctness_rows = _base.validate_correctness_rows
summarize_rows = _base.summarize_rows
read_float32_sidecar = _base.read_float32_sidecar
write_float32_sidecar = _base.write_float32_sidecar
sha256_file = _base.sha256_file
run_case = _base.run_case


def source_manifest(
    *,
    repo_root: Path,
    source_commit: str,
    run_tag: str,
) -> dict:
    return _base._source_manifest(
        repo_root=repo_root,
        source_commit=source_commit,
        run_tag=run_tag,
    )


def main(argv=None) -> int:
    _activate_contract()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
