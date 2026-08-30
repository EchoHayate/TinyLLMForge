#!/usr/bin/env python3
"""Isolated worker for one exact-prefill CUDA Graph benchmark case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from tools import exact_prefill_cuda_graph_benchmark_contract as contract


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


def _default_synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def _default_sampling_params(spec: dict):
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams(**spec["sampling"])


def _run_request(
    engine,
    *,
    prompt: list[int],
    sampling_params,
    clock_ns,
    synchronize,
) -> dict:
    synchronize()
    replay_before = int(
        engine.model_runner.exact_prefill_cuda_graph_cache.summary()[
            "replays"
        ]
    )
    started_ns = clock_ns()
    engine.add_request(prompt, sampling_params)
    first_token_ns = None
    previous_emission_ns = None
    tpot_samples_ns = []
    final_outputs = None
    while not engine.is_finished():
        outputs, _ = engine.step(completion_only=True)
        observation = engine.last_step_observation
        emitted = sum(
            len(token_ids)
            for token_ids in observation[
                "new_completion_tokens_by_seq"
            ].values()
        )
        if emitted:
            emitted_ns = clock_ns()
            if first_token_ns is None:
                first_token_ns = emitted_ns
            else:
                elapsed = emitted_ns - previous_emission_ns
                tpot_samples_ns.extend(
                    [float(elapsed) / emitted] * emitted
                )
            previous_emission_ns = emitted_ns
        if outputs:
            final_outputs = outputs
    synchronize()
    finished_ns = clock_ns()
    if first_token_ns is None:
        raise RuntimeError("request produced no first token")
    if not isinstance(final_outputs, list) or len(final_outputs) != 1:
        raise RuntimeError("request completion output is incomplete")
    output_token_ids = list(final_outputs[0][1])
    generated_tokens = int(sampling_params.max_tokens)
    if len(output_token_ids) != generated_tokens:
        raise RuntimeError("generated token inventory mismatch")
    if len(tpot_samples_ns) != generated_tokens - 1:
        raise RuntimeError("TPOT sample inventory mismatch")
    replay_after = int(
        engine.model_runner.exact_prefill_cuda_graph_cache.summary()[
            "replays"
        ]
    )
    return {
        "output_token_ids": output_token_ids,
        "output_text_sha256": _sha256_text(
            engine.tokenizer.decode(output_token_ids)
        ),
        "ttft_ns": first_token_ns - started_ns,
        "tpot_samples_ns": tpot_samples_ns,
        "e2e_ns": finished_ns - started_ns,
        "output_tokens_per_second": (
            generated_tokens
            / ((finished_ns - started_ns) / 1_000_000_000)
        ),
        "prefill_graph_replay_delta": replay_after - replay_before,
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
        engine.model_runner.reset_peak_memory_stats()
        for sample_index in range(frozen["measured_repetitions"]):
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
        summary = (
            engine.model_runner.exact_prefill_cuda_graph_cache.summary()
        )
        result = {
            "schema_version": contract.RESULT_SCHEMA_VERSION,
            "case": frozen,
            "model": model,
            "rows": rows,
            "prefill_graph_summary": json.loads(json.dumps(summary)),
        }
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
