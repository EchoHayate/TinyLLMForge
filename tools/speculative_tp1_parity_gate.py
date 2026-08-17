from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time


SCHEMA_VERSION = 2
CLAIM_SCOPE = (
    "Qwen3-0.6B BF16 TP1 greedy exact-token parity "
    "with MVP-0 transactional speculative residency"
)
LIMITATIONS = [
    "no TPOT or throughput improvement claim",
    "no TP4 claim",
    "no long-context claim",
    "no offload performance improvement claim",
    "no learned-drafter or MTP claim",
]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REAL_MOVEMENT_KEYS = (
    "h2d_copies",
    "d2h_copies",
    "h2d_bytes",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "evict_dirty",
)
RESIDENCY_KEYS = (
    "speculative_residency_prepares",
    "speculative_residency_precommits",
    "speculative_residency_seals",
    "speculative_residency_rollbacks",
    "speculative_residency_committed_blocks",
    "speculative_residency_rejected_blocks",
    "speculative_residency_rejected_d2h_copies",
)


DEFAULT_PROMPTS = (
    "alpha beta gamma alpha beta gamma alpha beta gamma",
    (
        "The sky is blue. The grass is green. "
        "The sky is blue. The grass is green. "
        "Continue the pattern:"
    ),
)
SOURCE_FILES = (
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/speculative_execution.py",
    "tinyvllm/engine/speculative_model_runner.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tinyvllm/engine/speculative_residency.py",
    "tinyvllm/engine/speculative_selection.py",
    "tinyvllm/speculative/adapter.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tinyvllm/speculative/ngram.py",
    "tinyvllm/speculative/ngram_adapter.py",
    "tinyvllm/speculative/runtime.py",
    "tools/speculative_tp1_parity_gate.py",
    "tools/verify_speculative_tp1_parity_gate.py",
)


def _non_negative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_integer(value: object, name: str) -> int:
    normalized = _non_negative_integer(value, name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _count_mapping(
    value: object,
    name: str,
) -> dict[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    normalized = {}
    for sequence_id, count in value.items():
        if (
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
        ):
            raise ValueError(
                f"{name} sequence IDs must be integers"
            )
        normalized[sequence_id] = _non_negative_integer(
            count,
            f"{name}[{sequence_id}]",
        )
    return normalized


def aggregate_speculative_observations(
    observations: list[dict],
) -> dict:
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    selected_rows = 0
    proposal_rows = 0
    proposed_tokens = 0
    accepted_draft_tokens = 0
    first_target_callbacks = 0
    tail_callbacks = 0
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise ValueError(
                f"observation {index} must be a mapping"
            )
        required = (
            "speculative_selected_seq_ids",
            "speculative_proposal_token_counts",
            "speculative_proposal_row_count",
            "speculative_accepted_draft_token_counts",
            "speculative_first_target_callback_count",
            "speculative_fixed_q_group_count",
        )
        missing = [
            key for key in required
            if key not in observation
        ]
        if missing:
            raise ValueError(
                f"observation {index} is missing {missing}"
            )
        selected_ids = observation[
            "speculative_selected_seq_ids"
        ]
        if (
            not isinstance(selected_ids, list)
            or any(
                isinstance(sequence_id, bool)
                or not isinstance(sequence_id, int)
                for sequence_id in selected_ids
            )
        ):
            raise ValueError(
                "speculative selected sequence IDs "
                "must be integer lists"
            )
        proposal_counts = _count_mapping(
            observation[
                "speculative_proposal_token_counts"
            ],
            "speculative_proposal_token_counts",
        )
        accepted_counts = _count_mapping(
            observation[
                "speculative_accepted_draft_token_counts"
            ],
            "speculative_accepted_draft_token_counts",
        )
        row_proposal_count = _non_negative_integer(
            observation[
                "speculative_proposal_row_count"
            ],
            "speculative_proposal_row_count",
        )
        actual_proposal_rows = sum(
            1 for count in proposal_counts.values()
            if count > 0
        )
        if row_proposal_count != actual_proposal_rows:
            raise ValueError(
                "speculative proposal row count does not "
                "match direct proposal token counts"
            )
        for sequence_id, accepted_count in (
            accepted_counts.items()
        ):
            proposal_count = proposal_counts.get(
                sequence_id,
                0,
            )
            if accepted_count > proposal_count:
                raise ValueError(
                    "accepted draft token count exceeds "
                    "proposed token count"
                )
        selected_rows += len(selected_ids)
        proposal_rows += row_proposal_count
        proposed_tokens += sum(proposal_counts.values())
        accepted_draft_tokens += sum(
            accepted_counts.values()
        )
        first_target_callbacks += _non_negative_integer(
            observation[
                "speculative_first_target_callback_count"
            ],
            "speculative_first_target_callback_count",
        )
        tail_callbacks += _non_negative_integer(
            observation[
                "speculative_fixed_q_group_count"
            ],
            "speculative_fixed_q_group_count",
        )
    target_invocations = (
        first_target_callbacks + tail_callbacks
    )
    return {
        "selected_rows": selected_rows,
        "proposal_rows": proposal_rows,
        "proposed_tokens": proposed_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "first_target_callbacks": first_target_callbacks,
        "tail_callbacks": tail_callbacks,
        "target_invocations": target_invocations,
        "acceptance_rate": (
            accepted_draft_tokens / proposed_tokens
            if proposed_tokens
            else 0.0
        ),
        "accepted_tokens_per_target_invocation": (
            accepted_draft_tokens / target_invocations
            if target_invocations
            else 0.0
        ),
    }


def _load_runtime_dependencies():
    from tinyvllm import SamplingParams
    from tinyvllm.engine.speculative_runtime import (
        EngineSpeculativeRuntime,
    )
    from tinyvllm.speculative.ngram_adapter import (
        NGramDraftAdapter,
    )

    return (
        SamplingParams,
        EngineSpeculativeRuntime,
        NGramDraftAdapter,
    )


def run_engine_case(
    *,
    engine_factory,
    model_path: str,
    prompts: tuple[str, ...],
    max_tokens: int,
    activate: bool,
    ngram_size: int,
    max_proposal_tokens: int,
) -> dict:
    if not callable(engine_factory):
        raise ValueError("engine_factory must be callable")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be non-empty")
    if (
        not isinstance(prompts, tuple)
        or not prompts
        or any(
            not isinstance(prompt, str) or not prompt
            for prompt in prompts
        )
    ):
        raise ValueError(
            "prompts must be a non-empty string tuple"
        )
    max_tokens = _positive_integer(
        max_tokens,
        "max_tokens",
    )
    ngram_size = _positive_integer(
        ngram_size,
        "ngram_size",
    )
    max_proposal_tokens = _positive_integer(
        max_proposal_tokens,
        "max_proposal_tokens",
    )
    if not isinstance(activate, bool):
        raise ValueError("activate must be a bool")
    (
        sampling_params_type,
        runtime_type,
        adapter_type,
    ) = _load_runtime_dependencies()
    engine = engine_factory(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=max(4, len(prompts)),
        kv_offload_mvp0=True,
        kv_offload_gpu_blocks=8,
        kv_offload_logical_blocks=64,
    )
    observations = []
    outputs_by_id = {}
    prompt_token_ids = [
        list(engine.tokenizer.encode(prompt))
        for prompt in prompts
    ]
    started = time.perf_counter()
    kv_offload_summaries = None
    try:
        if activate:
            adapter = adapter_type(
                ngram_size=ngram_size,
                max_proposal_tokens=(
                    max_proposal_tokens
                ),
            )
            engine.activate_speculative_runtime(
                runtime_type(adapter)
            )
        sampling_params = sampling_params_type(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=False,
        )
        for prompt in prompts:
            engine.add_request(prompt, sampling_params)
        while not engine.is_finished():
            step_outputs, _ = engine.step()
            observation = getattr(
                engine,
                "last_step_observation",
                None,
            )
            if observation is not None:
                observations.append(
                    copy.deepcopy(observation)
                )
            for sequence_id, token_ids in step_outputs:
                outputs_by_id[sequence_id] = list(token_ids)
    finally:
        try:
            kv_offload_summaries = (
                engine.kv_offload_summaries(
                    timeout_s=60.0
                )
            )
        finally:
            engine.exit()
    outputs = [
        outputs_by_id[sequence_id]
        for sequence_id in sorted(outputs_by_id)
    ]
    if len(outputs) != len(prompts):
        raise RuntimeError(
            "engine did not return one output per prompt"
        )
    summary = aggregate_speculative_observations(
        observations
    )
    if activate and (
        summary["selected_rows"] <= 0
        or summary["proposal_rows"] <= 0
        or summary["proposed_tokens"] <= 0
        or summary["first_target_callbacks"] <= 0
        or summary["tail_callbacks"] <= 0
    ):
        raise RuntimeError(
            "speculative path did not execute proposals "
            "and verifier callbacks"
        )
    if (
        not isinstance(kv_offload_summaries, tuple)
        or not kv_offload_summaries
        or not isinstance(kv_offload_summaries[0], dict)
    ):
        raise RuntimeError(
            "rank-0 KV offload summary is unavailable"
        )
    rank_zero_summary = kv_offload_summaries[0]
    movement = {
        key: rank_zero_summary.get(key)
        for key in REAL_MOVEMENT_KEYS
    }
    residency = {
        key: rank_zero_summary.get(key)
        for key in RESIDENCY_KEYS
    }
    config = getattr(engine, "config", None)
    if config is None:
        config = getattr(
            getattr(engine, "model_runner", None),
            "config",
            None,
        )
    return {
        "outputs": outputs,
        "prompt_token_ids": prompt_token_ids,
        "observations": observations,
        "summary": summary,
        "elapsed_s": time.perf_counter() - started,
        "movement": movement,
        "residency": residency,
        "tokenizer_identifier": str(
            getattr(
                engine.tokenizer,
                "name_or_path",
                type(engine.tokenizer).__name__,
            )
        ),
        "dtype": str(getattr(config, "dtype", "unknown")),
    }


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            "source file path must be a non-empty relative path"
        )
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "source file path must be a safe relative path"
        )
    return path.as_posix()


def hash_source_files(
    *,
    repo_root: Path,
    source_files: tuple[str, ...] = SOURCE_FILES,
) -> dict[str, str]:
    repo_root = Path(repo_root)
    if not isinstance(source_files, tuple) or not source_files:
        raise ValueError(
            "source_files must be a non-empty tuple"
        )
    result = {}
    for value in source_files:
        relative_path = _safe_relative_path(value)
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"source file is missing: {relative_path}"
            )
        result[relative_path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"artifact path already exists: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"temporary artifact path already exists: "
            f"{temporary}"
        )
    try:
        with temporary.open(
            "x",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _device_environment() -> dict:
    import torch

    return {
        "device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
    }


def run_live_gate(
    *,
    engine_factory,
    repo_root: Path,
    model_path: str,
    prompts: tuple[str, ...],
    max_tokens: int,
    ngram_size: int,
    max_proposal_tokens: int,
    output_path: Path,
    command: list[str],
    source_files: tuple[str, ...] = SOURCE_FILES,
) -> dict:
    baseline_case = run_engine_case(
        engine_factory=engine_factory,
        model_path=model_path,
        prompts=prompts,
        max_tokens=max_tokens,
        activate=False,
        ngram_size=ngram_size,
        max_proposal_tokens=max_proposal_tokens,
    )
    speculative_case = run_engine_case(
        engine_factory=engine_factory,
        model_path=model_path,
        prompts=prompts,
        max_tokens=max_tokens,
        activate=True,
        ngram_size=ngram_size,
        max_proposal_tokens=max_proposal_tokens,
    )
    if (
        baseline_case["tokenizer_identifier"]
        != speculative_case["tokenizer_identifier"]
    ):
        raise RuntimeError(
            "baseline and speculative tokenizer identities differ"
        )
    if baseline_case["dtype"] != speculative_case["dtype"]:
        raise RuntimeError(
            "baseline and speculative dtypes differ"
        )
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(item, str) or not item
            for item in command
        )
    ):
        raise ValueError(
            "command must be a non-empty string list"
        )
    device = _device_environment()
    resolved_model_path = str(Path(model_path).resolve())
    environment = {
        "model_path": resolved_model_path,
        "model_identifier": Path(
            resolved_model_path
        ).name,
        "tokenizer_identifier": (
            speculative_case["tokenizer_identifier"]
        ),
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "kv_offload_mvp0": True,
        "dtype": speculative_case["dtype"],
        "device_name": device["device_name"],
        "python_version": device["python_version"],
        "torch_version": device["torch_version"],
        "command": list(command),
    }
    baseline = {
        "outputs": baseline_case["outputs"],
        "prompt_token_ids": (
            baseline_case["prompt_token_ids"]
        ),
        "elapsed_s": baseline_case["elapsed_s"],
        "movement": baseline_case["movement"],
    }
    speculative = {
        "outputs": speculative_case["outputs"],
        "prompt_token_ids": (
            speculative_case["prompt_token_ids"]
        ),
        "summary": speculative_case["summary"],
        "elapsed_s": speculative_case["elapsed_s"],
        "movement": speculative_case["movement"],
        "residency": speculative_case["residency"],
    }
    source_hashes = hash_source_files(
        repo_root=repo_root,
        source_files=source_files,
    )
    if baseline["outputs"] != speculative["outputs"]:
        diagnostic = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "failure_reason": (
                "exact_token_parity_failed"
            ),
            "claim_scope": CLAIM_SCOPE,
            "baseline": copy.deepcopy(baseline),
            "speculative": copy.deepcopy(speculative),
            "environment": copy.deepcopy(environment),
            "source_files": dict(source_hashes),
            "limitations": list(LIMITATIONS),
        }
        write_json_atomic(output_path, diagnostic)
        raise RuntimeError("exact token parity failed")
    artifact = build_parity_artifact(
        baseline=baseline,
        speculative=speculative,
        environment=environment,
        source_files=source_hashes,
    )
    write_json_atomic(output_path, artifact)
    return artifact


def _default_engine_factory():
    from tinyvllm import LLM

    return LLM


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument(
        "--prompt",
        action="append",
        default=None,
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
    )
    run_parser.add_argument(
        "--ngram-size",
        type=int,
        default=3,
    )
    run_parser.add_argument(
        "--max-proposal-tokens",
        type=int,
        default=4,
    )
    run_parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.command != "run":
        raise ValueError(
            f"unsupported command: {args.command}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    command = list(sys.argv if argv is None else argv)
    artifact = run_live_gate(
        engine_factory=_default_engine_factory(),
        repo_root=repo_root,
        model_path=args.model,
        prompts=tuple(args.prompt or DEFAULT_PROMPTS),
        max_tokens=args.max_tokens,
        ngram_size=args.ngram_size,
        max_proposal_tokens=args.max_proposal_tokens,
        output_path=args.out,
        command=command,
    )
    print(
        json.dumps(
            validate_parity_artifact(artifact),
            sort_keys=True,
        )
    )


def build_parity_artifact(
    *,
    baseline: dict,
    speculative: dict,
    environment: dict,
    source_files: dict[str, str],
) -> dict:
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "claim_scope": CLAIM_SCOPE,
        "baseline": copy.deepcopy(baseline),
        "speculative": copy.deepcopy(speculative),
        "environment": copy.deepcopy(environment),
        "source_files": dict(source_files),
        "limitations": list(LIMITATIONS),
    }
    validate_parity_artifact(artifact)
    return artifact


def _token_matrix(value: object, name: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    normalized = []
    for row in value:
        if (
            not isinstance(row, list)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                for token_id in row
            )
        ):
            raise ValueError(
                f"{name} rows must contain integer token IDs"
            )
        normalized.append(list(row))
    return normalized


def _required_non_empty_string(
    mapping: dict,
    name: str,
) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _exact_counter_mapping(
    value: object,
    keys: tuple[str, ...],
    name: str,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if set(value) != set(keys):
        missing = [
            key for key in keys
            if key not in value
        ]
        extra = [
            key for key in value
            if key not in keys
        ]
        raise ValueError(
            f"{name} counter keys mismatch: "
            f"missing={missing}, extra={extra}"
        )
    return {
        key: _non_negative_integer(
            value[key],
            f"{name} {key}",
        )
        for key in keys
    }


def validate_parity_artifact(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("artifact must be a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("artifact schema version mismatch")
    if payload.get("status") != "PASS":
        raise ValueError("artifact status must be PASS")
    if payload.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("artifact claim scope mismatch")
    if payload.get("performance_improvement_claim"):
        raise ValueError(
            "performance improvement is not proven by "
            "this parity gate"
        )
    baseline = payload.get("baseline")
    speculative = payload.get("speculative")
    environment = payload.get("environment")
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be a mapping")
    if not isinstance(speculative, dict):
        raise ValueError("speculative must be a mapping")
    if not isinstance(environment, dict):
        raise ValueError("environment must be a mapping")
    baseline_outputs = _token_matrix(
        baseline.get("outputs"),
        "baseline outputs",
    )
    speculative_outputs = _token_matrix(
        speculative.get("outputs"),
        "speculative outputs",
    )
    if baseline_outputs != speculative_outputs:
        raise ValueError("exact token parity failed")
    if not baseline_outputs:
        raise ValueError("output token matrix is empty")
    baseline_prompts = _token_matrix(
        baseline.get("prompt_token_ids"),
        "baseline prompt token IDs",
    )
    speculative_prompts = _token_matrix(
        speculative.get("prompt_token_ids"),
        "speculative prompt token IDs",
    )
    if baseline_prompts != speculative_prompts:
        raise ValueError("prompt token IDs differ")
    if len(baseline_prompts) != len(baseline_outputs):
        raise ValueError(
            "prompt and output sequence counts differ"
        )
    baseline_movement = _exact_counter_mapping(
        baseline.get("movement"),
        REAL_MOVEMENT_KEYS,
        "baseline movement",
    )
    speculative_movement = _exact_counter_mapping(
        speculative.get("movement"),
        REAL_MOVEMENT_KEYS,
        "speculative movement",
    )
    residency = _exact_counter_mapping(
        speculative.get("residency"),
        RESIDENCY_KEYS,
        "speculative residency",
    )
    for key in (
        "speculative_residency_prepares",
        "speculative_residency_precommits",
        "speculative_residency_seals",
    ):
        if residency[key] <= 0:
            raise ValueError(
                f"speculative residency {key} must be positive"
            )
    if (
        residency[
            "speculative_residency_rejected_d2h_copies"
        ]
        != 0
    ):
        raise ValueError(
            "speculative residency rejected_d2h copies "
            "must be zero"
        )

    summary = speculative.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(
            "speculative summary must be a mapping"
        )
    selected_rows = _positive_integer(
        summary.get("selected_rows"),
        "selected speculative rows",
    )
    proposal_rows = _positive_integer(
        summary.get("proposal_rows"),
        "proposal rows",
    )
    proposed_tokens = _positive_integer(
        summary.get("proposed_tokens"),
        "proposed tokens",
    )
    accepted_tokens = _non_negative_integer(
        summary.get("accepted_draft_tokens"),
        "accepted draft tokens",
    )
    if accepted_tokens > proposed_tokens:
        raise ValueError(
            "accepted draft tokens exceed proposed tokens"
        )
    first_target_callbacks = _positive_integer(
        summary.get("first_target_callbacks"),
        "first-target callbacks",
    )
    tail_callbacks = _positive_integer(
        summary.get("tail_callbacks"),
        "tail callbacks",
    )
    target_invocations = _positive_integer(
        summary.get("target_invocations"),
        "target invocations",
    )
    if target_invocations != (
        first_target_callbacks + tail_callbacks
    ):
        raise ValueError(
            "target invocation count is inconsistent"
        )
    acceptance_rate = summary.get("acceptance_rate")
    accepted_per_invocation = summary.get(
        "accepted_tokens_per_target_invocation"
    )
    for value, name in (
        (acceptance_rate, "acceptance_rate"),
        (
            accepted_per_invocation,
            "accepted_tokens_per_target_invocation",
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"{name} must be a finite non-negative number"
            )
    if not math.isclose(
        float(acceptance_rate),
        accepted_tokens / proposed_tokens,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("acceptance rate is inconsistent")
    if not math.isclose(
        float(accepted_per_invocation),
        accepted_tokens / target_invocations,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "accepted tokens per target invocation "
            "is inconsistent"
        )

    for name in (
        "model_path",
        "model_identifier",
        "tokenizer_identifier",
        "dtype",
        "device_name",
        "python_version",
        "torch_version",
    ):
        _required_non_empty_string(environment, name)
    if environment.get("tensor_parallel_size") != 1:
        raise ValueError("TP1 is required")
    if environment.get("kv_offload_mvp0") is not True:
        raise ValueError("kv_offload_mvp0 must be enabled")
    temperature = environment.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) != 0.0
    ):
        raise ValueError(
            "temperature must be exactly 0.0"
        )
    command = environment.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(item, str) or not item
            for item in command
        )
    ):
        raise ValueError(
            "environment command must be a non-empty "
            "string list"
        )

    source_files = payload.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError(
            "source_files must be a non-empty mapping"
        )
    for relative_path, digest in source_files.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith("/")
            or ".." in relative_path.split("/")
        ):
            raise ValueError(
                "source file paths must be safe relative paths"
            )
        if (
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError(
                f"invalid SHA-256 for {relative_path}"
            )
    if payload.get("limitations") != LIMITATIONS:
        raise ValueError("artifact limitations mismatch")
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "output_sequences": len(baseline_outputs),
        "selected_rows": selected_rows,
        "proposal_rows": proposal_rows,
        "proposed_tokens": proposed_tokens,
        "accepted_draft_tokens": accepted_tokens,
        "target_invocations": target_invocations,
        "baseline_movement": baseline_movement,
        "speculative_movement": speculative_movement,
        "residency": residency,
    }


if __name__ == "__main__":
    main()
