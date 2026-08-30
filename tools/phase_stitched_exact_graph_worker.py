#!/usr/bin/env python3
"""Isolated worker for one phase-stitched exact-graph case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from tools import phase_stitched_exact_graph_contract as contract


def _atomic_write_json(path: Path, value: object) -> None:
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _make_prompt(prompt_tokens: int, sample_index: int) -> list[int]:
    offset = sample_index * 7919
    return [
        100 + ((index + offset) % 1000)
        for index in range(prompt_tokens)
    ]


def _default_engine_factory(model: str):
    def create(spec: dict):
        from tinyvllm.engine.llm_engine import LLMEngine

        return LLMEngine(model, **spec["engine_config"])

    return create


def _default_sampling_params(spec: dict):
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams(**spec["sampling"])


def _default_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _summary(engine, name: str) -> dict:
    if name == "prefill":
        cache = getattr(
            engine.model_runner,
            "exact_prefill_cuda_graph_cache",
            None,
        )
        return {} if cache is None else dict(cache.summary())
    function = getattr(engine.model_runner, name, None)
    return {} if function is None else dict(function())


def _scheduler_summary(engine, name: str) -> dict:
    scheduler = getattr(engine, "scheduler", None)
    function = getattr(scheduler, name, None)
    return {} if function is None else dict(function())


def _counter(summary: dict, name: str) -> int:
    value = summary.get(name, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _fallback_total(summary: dict) -> int:
    values = summary.get("fallback_counts", {})
    if not isinstance(values, dict):
        return 0
    return sum(int(value) for value in values.values())


def _quarantine_count(summary: dict) -> int:
    values = summary.get("quarantined_joint_identities", ())
    return len(values) if isinstance(values, (list, tuple, set)) else 0


def _phase_quarantine_count(summary: dict) -> int:
    return (
        _quarantine_count(summary)
        + len(summary.get("quarantined_parent_identities", ()))
    )


def _combined_exact_burst_summary(engine) -> dict:
    runner = _summary(
        engine,
        "exact_greedy_decode_burst_summary",
    )
    scheduler = _scheduler_summary(
        engine,
        "exact_greedy_decode_burst_summary",
    )
    fallback_counts = {
        **{
            f"runner:{name}": int(value)
            for name, value in runner.get("fallback_counts", {}).items()
        },
        **{
            f"scheduler:{name}": int(value)
            for name, value in scheduler.get(
                "fallback_counts",
                {},
            ).items()
        },
    }
    return {
        "failures": (
            _counter(runner, "failures")
            + _counter(scheduler, "failures")
        ),
        "quarantines": _counter(runner, "quarantines"),
        "pending_leases": _counter(scheduler, "pending_leases"),
        "fallback_count": (
            _fallback_total(runner) + _fallback_total(scheduler)
        ),
        "fallback_counts": fallback_counts,
        "quarantine_reason": runner.get("quarantine_reason"),
        "runner": json.loads(json.dumps(runner)),
        "scheduler": json.loads(json.dumps(scheduler)),
    }


def _combined_phase_stitch_summary(engine) -> dict:
    runner = _summary(engine, "phase_stitch_summary")
    scheduler = _scheduler_summary(engine, "phase_stitch_summary")
    fallback_counts = {
        **{
            f"runner:{name}": int(value)
            for name, value in runner.get("fallback_counts", {}).items()
        },
        **{
            f"scheduler:{name}": int(value)
            for name, value in scheduler.get(
                "fallback_counts",
                {},
            ).items()
        },
    }
    return {
        "failures": (
            _counter(runner, "failures")
            + _counter(scheduler, "failures_before_prefix")
            + _counter(scheduler, "failures_after_prefix")
        ),
        "quarantines": (
            _phase_quarantine_count(runner)
            + _phase_quarantine_count(scheduler)
        ),
        "pending_leases": int(
            scheduler.get("pending_parent_identity_sha256") is not None
        ),
        "fallback_count": (
            _fallback_total(runner) + _fallback_total(scheduler)
        ),
        "fallback_counts": fallback_counts,
        "runner": json.loads(json.dumps(runner)),
        "scheduler": json.loads(json.dumps(scheduler)),
    }


def _model_dtype(engine) -> str:
    config = getattr(engine.model_runner, "config", None)
    hf_config = getattr(config, "hf_config", None)
    value = getattr(hf_config, "torch_dtype", None)
    normalized = str(value).removeprefix("torch.")
    if normalized != "bfloat16":
        raise RuntimeError(
            "phase-stitched benchmark requires bfloat16 model dtype"
        )
    return normalized


def _run_request(
    engine,
    *,
    prompt: list[int],
    sampling_params,
    clock_ns,
    synchronize,
) -> dict:
    synchronize()
    before_prefill = _summary(engine, "prefill")
    before_burst = _summary(
        engine,
        "exact_greedy_decode_burst_summary",
    )
    before_phase = _summary(engine, "phase_stitch_summary")
    before_phase_scheduler = _scheduler_summary(
        engine,
        "phase_stitch_summary",
    )
    started_ns = clock_ns()
    engine.add_request(prompt, sampling_params)
    first_token_ns = None
    second_token_ns = None
    previous_emission_ns = None
    tpot_samples_ns = []
    final_outputs = None
    prefix_d2h_calls = 0
    suffix_d2h_calls = 0
    prefix_d2h_bytes = 0
    suffix_d2h_bytes = 0
    prefix_commits = 0
    suffix_commits = 0
    pending_suffix = False
    while not engine.is_finished():
        outputs, _ = engine.step(completion_only=True)
        observation = engine.last_step_observation
        emitted = sum(
            len(token_ids)
            for token_ids in observation[
                "new_completion_tokens_by_seq"
            ].values()
        )
        prefix_d2h_calls += int(
            observation.get("prefix_d2h_calls", 0)
        )
        suffix_d2h_calls += int(
            observation.get("suffix_d2h_calls", 0)
        )
        prefix_d2h_bytes += int(
            observation.get("prefix_d2h_bytes", 0)
        )
        suffix_d2h_bytes += int(
            observation.get("suffix_d2h_bytes", 0)
        )
        prefix_commits += int(
            observation.get("phase_published") == "prefix"
        )
        suffix_commits += int(
            observation.get("phase_published") == "suffix"
        )
        pending_suffix = bool(
            observation.get("pending_suffix", False)
        )
        if emitted:
            emitted_ns = clock_ns()
            if first_token_ns is None:
                first_token_ns = emitted_ns
                if emitted > 1:
                    second_token_ns = emitted_ns
            else:
                elapsed = emitted_ns - previous_emission_ns
                tpot_samples_ns.extend(
                    [float(elapsed) / emitted] * emitted
                )
                if second_token_ns is None:
                    second_token_ns = emitted_ns
            if first_token_ns == emitted_ns and emitted > 1:
                tpot_samples_ns.extend([0.0] * (emitted - 1))
            previous_emission_ns = emitted_ns
        if outputs:
            final_outputs = outputs
    synchronize()
    finished_ns = clock_ns()
    if first_token_ns is None or second_token_ns is None:
        raise RuntimeError("request produced fewer than two tokens")
    if not isinstance(final_outputs, list) or len(final_outputs) != 1:
        raise RuntimeError("request completion output is incomplete")
    output_token_ids = list(final_outputs[0][1])
    generated_tokens = int(sampling_params.max_tokens)
    if len(output_token_ids) != generated_tokens:
        raise RuntimeError("generated token inventory mismatch")
    if len(tpot_samples_ns) != generated_tokens - 1:
        raise RuntimeError("TPOT sample inventory mismatch")
    after_prefill = _summary(engine, "prefill")
    after_burst = _summary(
        engine,
        "exact_greedy_decode_burst_summary",
    )
    after_phase = _summary(engine, "phase_stitch_summary")
    after_phase_scheduler = _scheduler_summary(
        engine,
        "phase_stitch_summary",
    )
    return {
        "output_token_ids": output_token_ids,
        "output_text_sha256": _sha256_text(
            engine.tokenizer.decode(output_token_ids)
        ),
        "ttft_ns": first_token_ns - started_ns,
        "token_0_to_1_gap_ns": second_token_ns - first_token_ns,
        "tpot_samples_ns": tpot_samples_ns,
        "e2e_ns": finished_ns - started_ns,
        "output_tokens_per_second": (
            generated_tokens
            / ((finished_ns - started_ns) / 1_000_000_000)
        ),
        "prefill_graph_replay_delta": (
            _counter(after_prefill, "replays")
            - _counter(before_prefill, "replays")
        ),
        "exact_burst_replay_delta": (
            _counter(after_burst, "graph_replays")
            - _counter(before_burst, "graph_replays")
        ),
        "phase_stitch_attempt_delta": (
            _counter(after_phase, "attempts")
            - _counter(before_phase, "attempts")
        ),
        "phase_stitch_success_delta": (
            _counter(after_phase, "successes")
            - _counter(before_phase, "successes")
        ),
        "phase_stitch_prefill_replay_delta": (
            _counter(after_phase, "prefill_graph_replays")
            - _counter(before_phase, "prefill_graph_replays")
        ),
        "phase_stitch_decode_replay_delta": (
            _counter(after_phase, "decode_graph_replays")
            - _counter(before_phase, "decode_graph_replays")
        ),
        "phase_stitch_target_forward_delta": (
            _counter(after_phase, "target_model_forwards")
            - _counter(before_phase, "target_model_forwards")
        ),
        "phase_stitch_failure_delta": (
            _counter(after_phase, "failures")
            - _counter(before_phase, "failures")
            + _counter(
                after_phase_scheduler,
                "failures_before_prefix",
            )
            - _counter(
                before_phase_scheduler,
                "failures_before_prefix",
            )
            + _counter(
                after_phase_scheduler,
                "failures_after_prefix",
            )
            - _counter(
                before_phase_scheduler,
                "failures_after_prefix",
            )
        ),
        "phase_stitch_quarantine_delta": (
            _phase_quarantine_count(after_phase)
            - _phase_quarantine_count(before_phase)
            + _phase_quarantine_count(after_phase_scheduler)
            - _phase_quarantine_count(before_phase_scheduler)
        ),
        "phase_stitch_fallback_count": (
            _fallback_total(after_phase)
            - _fallback_total(before_phase)
            + _fallback_total(after_phase_scheduler)
            - _fallback_total(before_phase_scheduler)
        ),
        "phase_stitch_prefix_d2h_calls": prefix_d2h_calls,
        "phase_stitch_suffix_d2h_calls": suffix_d2h_calls,
        "phase_stitch_prefix_d2h_bytes": prefix_d2h_bytes,
        "phase_stitch_suffix_d2h_bytes": suffix_d2h_bytes,
        "phase_stitch_prefix_commits": prefix_commits,
        "phase_stitch_suffix_commits": suffix_commits,
        "phase_stitch_pending_leases": max(
            int(pending_suffix),
            int(
                after_phase_scheduler.get(
                    "pending_parent_identity_sha256"
                )
                is not None
            ),
        ),
        "preauthorized_kv_tokens": (
            7 if _counter(after_phase, "successes")
            > _counter(before_phase, "successes")
            else 0
        ),
    }


def run_worker(
    spec: dict,
    *,
    model: str,
    output_dir: Path,
    engine_factory=None,
    sampling_params_factory=None,
    clock_ns=None,
    synchronize=None,
) -> dict:
    frozen = contract.validate_case_spec(spec)
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    create_engine = (
        _default_engine_factory(model)
        if engine_factory is None
        else engine_factory
    )
    create_sampling_params = (
        _default_sampling_params
        if sampling_params_factory is None and engine_factory is None
        else (
            (lambda value: SimpleNamespace(**value["sampling"]))
            if sampling_params_factory is None
            else sampling_params_factory
        )
    )
    clock = time.perf_counter_ns if clock_ns is None else clock_ns
    sync = _default_synchronize if synchronize is None else synchronize
    engine = create_engine(frozen)
    rows = []
    try:
        model_dtype = _model_dtype(engine)
        for warmup_index in range(frozen["warmup_repetitions"]):
            _run_request(
                engine,
                prompt=_make_prompt(
                    frozen["prompt_tokens"],
                    -1 - warmup_index,
                ),
                sampling_params=create_sampling_params(frozen),
                clock_ns=clock,
                synchronize=sync,
            )
            engine.clear_reusable_prefix_cache()
        for sample_index in range(frozen["measured_repetitions"]):
            engine.model_runner.reset_peak_memory_stats()
            prompt = _make_prompt(
                frozen["prompt_tokens"],
                sample_index,
            )
            measured = _run_request(
                engine,
                prompt=prompt,
                sampling_params=create_sampling_params(frozen),
                clock_ns=clock,
                synchronize=sync,
            )
            memory = engine.model_runner.memory_snapshot()
            rows.append({
                "schema_version": contract.ROW_SCHEMA_VERSION,
                "case_id": frozen["case_id"],
                "round": frozen["round"],
                "order_position": frozen["order_position"],
                "arm": frozen["arm"],
                "prompt_tokens": frozen["prompt_tokens"],
                "sample_index": sample_index,
                "generated_tokens": contract.GENERATED_TOKENS,
                "prompt_sha256": contract.canonical_json_sha256(prompt),
                **measured,
                "tpot_median_ns": statistics.median(
                    measured["tpot_samples_ns"]
                ),
                "cuda_peak_allocated_bytes": int(
                    memory["cuda_peak_allocated_bytes"]
                ),
                "cuda_peak_reserved_bytes": int(
                    memory["cuda_peak_reserved_bytes"]
                ),
            })
            engine.clear_reusable_prefix_cache()
        result = {
            "schema_version": contract.RESULT_SCHEMA_VERSION,
            "case": frozen,
            "model": model,
            "model_dtype": model_dtype,
            "rows": rows,
            "prefill_graph_summary": _summary(engine, "prefill"),
            "exact_burst_summary": _summary(
                engine,
                "exact_greedy_decode_burst_summary",
            ),
            "phase_stitch_summary": _combined_phase_stitch_summary(
                engine
            ),
        }
        result["exact_burst_summary"] = (
            _combined_exact_burst_summary(engine)
        )
        _atomic_write_json(destination / "result.json", result)
        return result
    finally:
        engine.exit()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    run_worker(
        spec,
        model=args.model,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
