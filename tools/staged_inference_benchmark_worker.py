"""Isolated worker entrypoint for staged inference benchmark cases."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import arrival_load_driver
import profile_prefix_cache
import staged_inference_benchmark_contract as contract
from profile_prefix_cache import make_token_prompt


PREFIX_PROFILE_DEFAULTS = {
    "mode": "full",
    "shared_prefix_tokens": "256,1024,2048",
    "batch_prefix_tokens": "1024,2048",
    "batch_size": 8,
    "suffix_tokens": 64,
    "repetitions": 7,
    "warmup_repetitions": 2,
    "max_model_len": 4096,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 8,
    "gpu_memory_utilization": 0.5,
    "enforce_eager": True,
}


def _validate_prefix_spec(spec: dict) -> dict:
    if spec.get("gate") != "prefix":
        raise ValueError("prefix worker requires gate='prefix'")
    model_tier = spec.get("model_tier")
    contract.build_prefix_case_matrix(model_tier=model_tier)
    expected_case_id = f"prefix_full__{model_tier}"
    if spec.get("case_id") != expected_case_id:
        raise ValueError(
            f"prefix case_id must be {expected_case_id!r}"
        )
    profile_args = spec.get("profile_args")
    if not isinstance(profile_args, dict):
        raise ValueError("prefix profile_args must be an object")
    expected_fields = {"model", *PREFIX_PROFILE_DEFAULTS}
    if set(profile_args) != expected_fields:
        raise ValueError("prefix profile_args fields do not match contract")
    if (
        not isinstance(profile_args["model"], str)
        or not profile_args["model"]
    ):
        raise ValueError("prefix profile model must be a non-empty string")
    for field, expected in PREFIX_PROFILE_DEFAULTS.items():
        if profile_args.get(field) != expected:
            raise ValueError(f"prefix profile {field} mismatch")
    return deepcopy(profile_args)


def _validate_chunked_spec(spec: dict) -> None:
    if spec.get("gate") != "chunked":
        raise ValueError("chunked worker requires gate='chunked'")
    model_tier = spec.get("model_tier")
    case_id = spec.get("case_id")
    expected_cases = {
        row["case_id"]: row
        for row in contract.build_chunked_case_matrix(
            model_tier=model_tier
        )
    }
    expected = expected_cases.get(case_id)
    if expected is None:
        raise ValueError(f"unknown chunked case_id: {case_id!r}")
    workload_rows = spec.get("workload_rows")
    contract._validate_chunked_workload(workload_rows)
    for field in (
        "policy",
        "repetition",
        "policy_order",
    ):
        if spec.get(field) != expected[field]:
            raise ValueError(f"chunked case {field} mismatch")
    if spec.get("engine_config") != expected["engine_config"]:
        raise ValueError("chunked engine_config must match frozen policy")
    if (
        spec.get("workload_sha256") != expected["workload_sha256"]
        or
        contract.canonical_json_sha256(workload_rows)
        != expected["workload_sha256"]
    ):
        raise ValueError("chunked workload hash mismatch")
    drain_timeout_ns = spec.get("drain_timeout_ns")
    if (
        isinstance(drain_timeout_ns, bool)
        or not isinstance(drain_timeout_ns, int)
        or drain_timeout_ns <= 0
    ):
        raise ValueError("drain_timeout_ns must be a positive integer")


def _materialize_chunked_workload(rows: list[dict]) -> list[dict]:
    materialized = []
    for index, source in enumerate(rows):
        row = deepcopy(source)
        prompt_tokens = row.pop("prompt_tokens")
        row["prompt_token_ids"] = make_token_prompt(
            prompt_tokens,
            offset=10_000 + index * 137,
        )
        row["prompt_token_count"] = prompt_tokens
        materialized.append(row)
    return materialized


def _default_chunked_engine_factory(model: str):
    def create(case_spec):
        from tinyvllm.engine.llm_engine import LLMEngine

        return LLMEngine(
            model,
            **case_spec["engine_config"],
        )

    return create


def _run_chunked(
    spec: dict,
    output_dir: Path,
    *,
    engine_factory=None,
    clock_ns=None,
) -> dict:
    _validate_chunked_spec(spec)
    case_spec = {
        "case_id": spec["case_id"],
        "gate": "chunked",
        "model_tier": spec["model_tier"],
        "policy": spec["policy"],
        "repetition": spec["repetition"],
        "policy_order": spec["policy_order"],
        "drain_timeout_ns": spec["drain_timeout_ns"],
        "engine_config": deepcopy(spec["engine_config"]),
        "resolved_config": deepcopy(spec["engine_config"]),
    }
    if engine_factory is None:
        model = spec.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("chunked worker requires a model path")
        engine_factory = _default_chunked_engine_factory(model)
    result = arrival_load_driver.run_case(
        case_spec=case_spec,
        workload_rows=_materialize_chunked_workload(
            spec["workload_rows"]
        ),
        engine_factory=engine_factory,
        clock_ns=time.monotonic_ns if clock_ns is None else clock_ns,
        output_dir=output_dir,
    )
    timeline_path = output_dir / "request_timeline.jsonl"
    lifecycle = (
        arrival_load_driver._load_jsonl(timeline_path)
        if timeline_path.is_file()
        else []
    )
    result["lifecycle_requests"] = len(lifecycle)
    result["measured_requests"] = sum(
        row.get("warmup") is False for row in lifecycle
    )
    arrival_load_driver._atomic_write_json(
        output_dir / "case_result.json",
        result,
    )
    return result


def _run_prefix(
    spec: dict,
    output_dir: Path,
    *,
    prefix_runner=None,
) -> dict:
    profile_args = _validate_prefix_spec(spec)
    profile_args["out_dir"] = str(output_dir)
    runner = (
        profile_prefix_cache.run_profile
        if prefix_runner is None
        else prefix_runner
    )
    summary = runner(SimpleNamespace(**profile_args))
    if not isinstance(summary, dict):
        raise ValueError("prefix profiler must return a summary object")
    return {
        "status": "PASS",
        "gate": "prefix",
        "case_id": spec["case_id"],
        "summary": summary,
    }


def run_worker(
    spec: dict,
    output_dir: Path,
    *,
    engine_factory=None,
    clock_ns=None,
    prefix_runner=None,
) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("worker spec must be an object")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    gate = spec.get("gate")
    if gate == "chunked":
        return _run_chunked(
            spec,
            destination,
            engine_factory=engine_factory,
            clock_ns=clock_ns,
        )
    if gate == "prefix":
        return _run_prefix(
            spec,
            destination,
            prefix_runner=prefix_runner,
        )
    raise ValueError(f"unsupported staged gate: {gate!r}")


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("worker spec file must contain an object")
    return value


def _load_worker_spec(
    spec_path: Path,
    *,
    workload_path: Path | None,
) -> dict:
    spec = _load_json(spec_path)
    if spec.get("gate") == "chunked":
        if workload_path is None:
            raise ValueError(
                "chunked worker requires --workload-manifest"
            )
        if "workload_rows" in spec:
            raise ValueError(
                "chunked CLI spec must not embed workload_rows"
            )
        spec["workload_rows"] = arrival_load_driver._load_jsonl(
            workload_path
        )
    elif workload_path is not None:
        raise ValueError(
            "--workload-manifest is only valid for chunked workers"
        )
    return spec


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one isolated staged benchmark case",
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = run_worker(
        _load_worker_spec(
            args.spec,
            workload_path=args.workload_manifest,
        ),
        args.output_dir,
    )
    sys.stdout.write(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
