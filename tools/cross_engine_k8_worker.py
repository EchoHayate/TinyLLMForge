from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Callable, Mapping, Optional, Sequence

from tools.cross_engine_k8_resources import NOT_EXPOSED, ProcessResourceSession
from tools.cross_engine_k8_workload import (
    OPTIONAL_ARM,
    REQUIRED_ARMS,
    reconstruct_metrics,
)


WORKER_PLAN_SCHEMA_VERSION = "cross-engine-k8.worker-plan.v1"
CASE_SCHEMA_VERSION = "cross-engine-k8.case.v1"
CORRECTNESS_SCHEMA_VERSION = "cross-engine-k8.correctness.v1"
WORKER_RECEIPT_SCHEMA_VERSION = "cross-engine-k8.worker-receipt.v1"
ALL_ARMS = REQUIRED_ARMS + (OPTIONAL_ARM,)


def _token_digest(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        list(token_ids),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class EngineResult:
    token_ids: Sequence[int]
    token_timestamps_ns: Sequence[int]
    request_start_ns: int
    request_end_ns: int
    engine_metrics: Mapping
    retained_logits: Mapping
    resource_summary: Mapping

    def validate(self, *, expected_output_tokens: int) -> "EngineResult":
        if (
            isinstance(expected_output_tokens, bool)
            or not isinstance(expected_output_tokens, int)
            or expected_output_tokens <= 0
        ):
            raise ValueError("expected output token count is invalid")
        if len(self.token_ids) != expected_output_tokens:
            raise ValueError("output token count mismatch")
        if len(self.token_timestamps_ns) != expected_output_tokens:
            raise ValueError("token timestamp count mismatch")
        timeline = [
            self.request_start_ns,
            *self.token_timestamps_ns,
            self.request_end_ns,
        ]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in timeline
        ):
            raise ValueError("timestamps must be integers")
        if any(left > right for left, right in zip(timeline, timeline[1:])):
            raise ValueError("token timestamps must be monotonic")
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in self.token_ids
        ):
            raise ValueError("output token IDs are invalid")
        return self


def discover_vllm_public_multi_step(engine_args_type) -> dict:
    annotations = getattr(engine_args_type, "__annotations__", {})
    try:
        signature = inspect.signature(engine_args_type)
        parameters = set(signature.parameters)
    except (TypeError, ValueError):
        parameters = set()
    public_names = set(annotations) | parameters
    if "num_scheduler_steps" in public_names:
        return {
            "available": True,
            "parameter": "num_scheduler_steps",
            "value": 8,
        }
    return {
        "available": False,
        "parameter": None,
        "value": None,
    }


class TinyLLMForgeAdapter:
    def __init__(self, plan: Mapping):
        from tools.profile_exact_greedy_decode_burst import _construct_llm

        arm = plan["arm"]
        policy = {
            "tinyllmforge_host_greedy": "host_greedy",
            "tinyllmforge_exact_k8": "decode_burst_k8",
        }.get(arm)
        if policy is None:
            raise ValueError("TinyLLMForge adapter received a foreign arm")
        self._policy = policy
        self._llm = _construct_llm(
            model=plan["model_path"],
            prompt_tokens=max(
                case["prompt_tokens"] for case in plan["cases"]
            ),
            generated_tokens=max(
                case["output_tokens"] for case in plan["cases"]
            ),
            gpu_memory_utilization=float(
                plan.get("gpu_memory_utilization", 0.8)
            ),
            policy=policy,
        )

    def run_case(self, case: Mapping, arm: str) -> EngineResult:
        from tools.profile_exact_greedy_decode_burst import _run_request

        measured = _run_request(
            self._llm,
            prompt=list(case["prompt_token_ids"]),
            generated_tokens=case["output_tokens"],
            policy=self._policy,
            profile_label=None,
        )
        first = int(measured["ttft_ns"])
        token_timestamps = [first]
        for duration in measured["amortized_tpot_samples_ns"]:
            token_timestamps.append(
                token_timestamps[-1] + int(round(duration))
            )
        runner_summary = (
            self._llm.model_runner.exact_greedy_decode_burst_summary()
        )
        return EngineResult(
            token_ids=list(measured["output_token_ids"]),
            token_timestamps_ns=token_timestamps,
            request_start_ns=0,
            request_end_ns=int(measured["e2e_ns"]),
            engine_metrics={
                "timing_method": "step_amortized_monotonic",
                "graph_replays": runner_summary.get(
                    "graph_replays",
                    NOT_EXPOSED,
                ),
                "intermediate_token_d2h_calls": runner_summary.get(
                    "intermediate_token_d2h_calls",
                    NOT_EXPOSED,
                ),
                "final_token_d2h_calls": runner_summary.get(
                    "final_token_d2h_calls",
                    NOT_EXPOSED,
                ),
            },
            retained_logits={},
            resource_summary={
                "peak_gpu_memory_bytes": NOT_EXPOSED,
                "peak_rss_bytes": NOT_EXPOSED,
            },
        )

    def close(self) -> None:
        self._llm.exit()


class VllmAdapter:
    def __init__(self, plan: Mapping):
        from vllm import EngineArgs, LLMEngine, SamplingParams

        kwargs = {
            "model": plan["model_path"],
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "enable_prefix_caching": False,
            "max_model_len": max(
                case["prompt_tokens"] + case["output_tokens"]
                for case in plan["cases"]
            ),
        }
        if plan["arm"] == OPTIONAL_ARM:
            control = discover_vllm_public_multi_step(EngineArgs)
            if not control["available"]:
                raise RuntimeError(
                    "VLLM_MULTI_STEP_NOT_PUBLICLY_AVAILABLE"
                )
            kwargs[control["parameter"]] = control["value"]
        self._sampling_params_type = SamplingParams
        self._engine = LLMEngine.from_engine_args(EngineArgs(**kwargs))
        self._request_ordinal = 0

    def run_case(self, case: Mapping, arm: str) -> EngineResult:
        from vllm.inputs import TokensPrompt

        request_id = f"cross-engine-k8-{self._request_ordinal}"
        self._request_ordinal += 1
        sampling = self._sampling_params_type(
            temperature=0.0,
            max_tokens=case["output_tokens"],
            ignore_eos=True,
        )
        self._engine.add_request(
            request_id,
            TokensPrompt(
                prompt_token_ids=list(case["prompt_token_ids"])
            ),
            sampling,
        )
        request_start_ns = time.perf_counter_ns()
        token_ids = []
        token_timestamps = []
        while self._engine.has_unfinished_requests():
            step_start_ns = time.perf_counter_ns()
            outputs = self._engine.step()
            step_end_ns = time.perf_counter_ns()
            matching = [
                output
                for output in outputs
                if str(output.request_id) == request_id
            ]
            if not matching or not matching[0].outputs:
                continue
            cumulative = list(matching[0].outputs[0].token_ids)
            emitted = len(cumulative) - len(token_ids)
            if emitted <= 0:
                continue
            interval = max(1, step_end_ns - step_start_ns)
            for index in range(emitted):
                token_timestamps.append(
                    step_start_ns
                    + int(interval * (index + 1) / emitted)
                )
            token_ids = cumulative
        request_end_ns = time.perf_counter_ns()
        return EngineResult(
            token_ids=token_ids,
            token_timestamps_ns=token_timestamps,
            request_start_ns=request_start_ns,
            request_end_ns=request_end_ns,
            engine_metrics={
                "timing_method": "step_amortized_monotonic",
                "graph_replays": NOT_EXPOSED,
                "intermediate_token_d2h_calls": NOT_EXPOSED,
                "final_token_d2h_calls": NOT_EXPOSED,
            },
            retained_logits={},
            resource_summary={
                "peak_gpu_memory_bytes": NOT_EXPOSED,
                "peak_rss_bytes": NOT_EXPOSED,
            },
        )

    def close(self) -> None:
        shutdown = getattr(self._engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _default_adapter_factory(plan: Mapping):
    if plan["arm"].startswith("tinyllmforge_"):
        return TinyLLMForgeAdapter(plan)
    if plan["arm"].startswith("vllm_"):
        return VllmAdapter(plan)
    raise ValueError("worker arm is invalid")


def _validate_plan(plan: Mapping) -> None:
    if plan.get("schema_version") != WORKER_PLAN_SCHEMA_VERSION:
        raise ValueError("worker plan schema mismatch")
    if plan.get("arm") not in ALL_ARMS:
        raise ValueError("worker arm is invalid")
    if plan.get("warmups") != 2:
        raise ValueError("worker warmup count is not frozen")
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("worker cases are invalid")
    for case in cases:
        prompt = case.get("prompt_token_ids")
        if (
            not isinstance(prompt, list)
            or len(prompt) != case.get("prompt_tokens")
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in prompt
            )
        ):
            raise ValueError("prompt_token_ids are invalid")
        if case.get("output_tokens") != 128:
            raise ValueError("worker output length is not frozen")


def run_worker(
    plan: Mapping,
    *,
    adapter_factory: Callable[[Mapping], object] = _default_adapter_factory,
    sampler_factory: Optional[Callable[[], object]] = None,
) -> dict:
    _validate_plan(plan)
    arm = plan["arm"]
    adapter = adapter_factory(plan)
    case_rows = []
    correctness_rows = []
    correctness_valid = True
    try:
        warmup_case = plan["cases"][0]
        for _warmup in range(plan["warmups"]):
            adapter.run_case(warmup_case, arm).validate(
                expected_output_tokens=warmup_case["output_tokens"]
            )
        for case in plan["cases"]:
            sampler = (
                ProcessResourceSession(
                    pid=os.getpid(),
                    gpu_uuid=plan["gpu_uuid"],
                )
                if sampler_factory is None
                else sampler_factory()
            )
            sampler.start()
            try:
                result = adapter.run_case(case, arm).validate(
                    expected_output_tokens=case["output_tokens"]
                )
            finally:
                external_resources = sampler.stop()
            metrics = reconstruct_metrics(
                request_start_ns=result.request_start_ns,
                token_timestamps_ns=result.token_timestamps_ns,
                request_end_ns=result.request_end_ns,
                output_tokens=case["output_tokens"],
            )
            expected = plan.get("expected_tokens", {}).get(case["context"])
            matches = expected is None or list(result.token_ids) == list(
                expected
            )
            correctness_valid = correctness_valid and matches
            row = {
                "schema_version": CASE_SCHEMA_VERSION,
                "run_tag": plan["run_tag"],
                "source_revision": plan["source_revision"],
                "arm": arm,
                "repetition": plan["repetition"],
                "context": case["context"],
                "prompt_tokens": case["prompt_tokens"],
                "output_tokens": case["output_tokens"],
                "token_ids": list(result.token_ids),
                "token_ids_sha256": _token_digest(result.token_ids),
                "request_start_ns": result.request_start_ns,
                "token_timestamps_ns": list(
                    result.token_timestamps_ns
                ),
                "request_end_ns": result.request_end_ns,
                **metrics,
                "peak_gpu_memory_bytes": external_resources.get(
                    "peak_gpu_memory_bytes",
                    result.resource_summary.get(
                        "peak_gpu_memory_bytes",
                        NOT_EXPOSED,
                    ),
                ),
                "peak_rss_bytes": external_resources.get(
                    "peak_rss_bytes",
                    result.resource_summary.get(
                        "peak_rss_bytes",
                        NOT_EXPOSED,
                    ),
                ),
                "resource_summary": external_resources,
                "engine_metrics": dict(result.engine_metrics),
                "performance_eligible": matches,
            }
            case_rows.append(row)
            correctness_rows.append({
                "schema_version": CORRECTNESS_SCHEMA_VERSION,
                "run_tag": plan["run_tag"],
                "source_revision": plan["source_revision"],
                "arm": arm,
                "repetition": plan["repetition"],
                "context": case["context"],
                "output_tokens": len(result.token_ids),
                "token_ids": list(result.token_ids),
                "token_ids_sha256": _token_digest(result.token_ids),
                "expected_token_ids_sha256": (
                    None if expected is None else _token_digest(expected)
                ),
                "matches_reference": matches,
                "retained_logits": dict(result.retained_logits),
            })
    finally:
        adapter.close()
    if not correctness_valid:
        for row in case_rows:
            row["performance_eligible"] = False
    return {
        "schema_version": WORKER_RECEIPT_SCHEMA_VERSION,
        "run_tag": plan["run_tag"],
        "source_revision": plan["source_revision"],
        "arm": arm,
        "repetition": plan["repetition"],
        "terminal": True,
        "measured_rows": len(case_rows),
        "correctness_valid": correctness_valid,
        "performance_eligible": correctness_valid,
        "case_rows": case_rows,
        "correctness_rows": correctness_rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = run_worker(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(args.output / "case_rows.jsonl", result["case_rows"])
    _write_jsonl(
        args.output / "correctness_rows.jsonl",
        result["correctness_rows"],
    )
    receipt = dict(result)
    receipt.pop("case_rows")
    receipt.pop("correctness_rows")
    _write_json(args.output / "worker_receipt.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
