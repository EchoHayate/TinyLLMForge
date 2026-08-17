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


_AdapterTypes = tuple[type, type]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MOVEMENT_KEYS = (
    "h2d_copies",
    "d2h_copies",
    "h2d_bytes",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "evict_dirty",
)
_RESIDENCY_KEYS = (
    "speculative_residency_prepares",
    "speculative_residency_precommits",
    "speculative_residency_seals",
    "speculative_residency_rollbacks",
    "speculative_residency_committed_blocks",
    "speculative_residency_rejected_blocks",
    "speculative_residency_rejected_d2h_copies",
)
SOURCE_FILES = (
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/speculative_execution.py",
    "tinyvllm/engine/speculative_model_runner.py",
    "tinyvllm/engine/speculative_residency.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tinyvllm/speculative/adapter.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tools/speculative_residency_boundary_gate.py",
    "tools/verify_speculative_residency_boundary_gate.py",
    "tools/run_speculative_residency_boundary_gate_remote.sh",
)


class BoundaryDraftAdapter:
    def __init__(
        self,
        mode: str,
        *,
        accepted_token_ids: tuple[int, ...],
        adapter_types: _AdapterTypes | None = None,
    ):
        if mode not in {"accept", "reject"}:
            raise ValueError(
                "boundary fixture mode must be accept or reject"
            )
        if (
            not isinstance(accepted_token_ids, tuple)
            or len(accepted_token_ids) != 3
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in accepted_token_ids
            )
        ):
            raise ValueError(
                "accepted_token_ids must contain exactly "
                "three non-negative integers"
            )
        if adapter_types is None:
            from tinyvllm.speculative.adapter import (
                DraftCapabilities,
                DraftProposal,
            )

            adapter_types = (
                DraftCapabilities,
                DraftProposal,
            )
        capability_type, proposal_type = adapter_types
        self._mode = mode
        self._accepted_token_ids = accepted_token_ids
        self._proposal_type = proposal_type
        self._capabilities = capability_type(
            source_type="boundary_fixture",
            supports_batch=True,
            requires_target_hidden=False,
            requires_target_logits=False,
            max_proposal_tokens=3,
        )

    @property
    def capabilities(self):
        return self._capabilities

    def propose_batch(self, contexts: tuple) -> tuple:
        proposals = []
        for context in contexts:
            target_token = context.first_target_token
            if target_token != self._accepted_token_ids[0]:
                raise RuntimeError(
                    "boundary fixture baseline token drifted"
                )
            token_ids = self._accepted_token_ids
            if self._mode == "reject":
                token_ids = (
                    0 if target_token != 0 else 1,
                    *token_ids[1:],
                )
            proposals.append(
                self._proposal_type(
                    sequence_id=context.sequence_id,
                    token_ids=token_ids,
                    source_type="boundary_fixture",
                    metadata={"mode": self._mode},
                )
            )
        return tuple(proposals)


def validate_boundary_prompt_token_ids(
    value: object,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(
            "boundary prompt token_ids must be a list"
        )
    if len(value) != 254:
        raise ValueError(
            "boundary prompt must contain exactly 254 tokens"
        )
    for token_id in value:
        if isinstance(token_id, bool) or not isinstance(
            token_id,
            int,
        ):
            raise ValueError(
                "boundary prompt tokens must be integers"
            )
    return tuple(value)


def require_prefill_boundary_sequence(engine):
    running = tuple(engine.scheduler.running)
    if len(running) != 1:
        raise RuntimeError(
            "boundary gate requires exactly one running sequence"
        )
    sequence = running[0]
    if sequence.num_tokens != 255:
        raise RuntimeError(
            "boundary gate requires live sequence length 255"
        )
    if sequence.num_prompt_tokens != 254:
        raise RuntimeError(
            "boundary gate requires 254 prompt tokens"
        )
    return sequence


def evict_boundary_history(
    engine,
    sequence,
) -> tuple[tuple[int, int], ...]:
    manager = engine.model_runner.kv_offload
    logical_blocks = [
        int(logical_block)
        for logical_block in sequence.block_table
    ]
    manager.writeback_dirty(logical_blocks)
    manager.synchronize_copies()
    identities = tuple(
        (
            logical_block,
            manager.bound_generations[logical_block],
        )
        for logical_block in logical_blocks
    )
    return manager.evict_clean_resident_blocks(identities)


def _load_runtime_types():
    from tinyvllm import SamplingParams
    from tinyvllm.engine.speculative_runtime import (
        EngineSpeculativeRuntime,
    )

    return SamplingParams, EngineSpeculativeRuntime


def _aggregate_observations(
    observations: list[dict],
) -> dict:
    proposed_tokens = 0
    accepted_draft_tokens = 0
    selected_rows = 0
    proposal_rows = 0
    first_target_callbacks = 0
    tail_callbacks = 0
    for observation in observations:
        selected_rows += len(
            observation[
                "speculative_selected_seq_ids"
            ]
        )
        proposal_counts = observation[
            "speculative_proposal_token_counts"
        ]
        accepted_counts = observation[
            "speculative_accepted_draft_token_counts"
        ]
        proposal_rows += observation[
            "speculative_proposal_row_count"
        ]
        proposed_tokens += sum(
            proposal_counts.values()
        )
        accepted_draft_tokens += sum(
            accepted_counts.values()
        )
        first_target_callbacks += observation[
            "speculative_first_target_callback_count"
        ]
        tail_callbacks += observation[
            "speculative_fixed_q_group_count"
        ]
    return {
        "selected_rows": selected_rows,
        "proposal_rows": proposal_rows,
        "proposed_tokens": proposed_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "first_target_callbacks": first_target_callbacks,
        "tail_callbacks": tail_callbacks,
    }


def run_boundary_case(
    *,
    engine_factory,
    model_path: str,
    prompt_token_ids: object,
    mode: str | None,
    accepted_token_ids: tuple[int, ...] | None,
    runtime_types: tuple[type, type] | None = None,
    adapter_types: _AdapterTypes | None = None,
) -> dict:
    if not callable(engine_factory):
        raise ValueError("engine_factory must be callable")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be non-empty")
    if mode not in {None, "accept", "reject"}:
        raise ValueError(
            "boundary case mode must be accept, reject, or None"
        )
    prompt_token_ids = validate_boundary_prompt_token_ids(
        prompt_token_ids
    )
    if runtime_types is None:
        runtime_types = _load_runtime_types()
    sampling_params_type, runtime_type = runtime_types
    engine = engine_factory(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=1,
        kv_offload_mvp0=True,
        kv_offload_gpu_blocks=2,
        kv_offload_logical_blocks=64,
    )
    observations = []
    outputs_by_id = {}
    evicted_identities = ()
    kv_offload_summaries = None
    started = time.perf_counter()
    try:
        if mode is not None:
            if accepted_token_ids is None:
                raise ValueError(
                    "speculative boundary cases require "
                    "accepted_token_ids"
                )
            adapter = BoundaryDraftAdapter(
                mode,
                accepted_token_ids=accepted_token_ids,
                adapter_types=adapter_types,
            )
            engine.activate_speculative_runtime(
                runtime_type(adapter)
            )
        sampling_params = sampling_params_type(
            temperature=0.0,
            max_tokens=4,
            ignore_eos=True,
        )
        engine.add_request(
            list(prompt_token_ids),
            sampling_params,
        )
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
        sequence = require_prefill_boundary_sequence(engine)
        evicted_identities = evict_boundary_history(
            engine,
            sequence,
        )
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
    if len(outputs) != 1:
        raise RuntimeError(
            "boundary case did not return exactly one output"
        )
    if (
        not isinstance(kv_offload_summaries, tuple)
        or not kv_offload_summaries
        or not isinstance(kv_offload_summaries[0], dict)
    ):
        raise RuntimeError(
            "rank-0 KV offload summary is unavailable"
        )
    summary = _aggregate_observations(observations)
    if mode is not None and (
        summary["selected_rows"] <= 0
        or summary["proposal_rows"] <= 0
        or summary["proposed_tokens"] <= 0
        or summary["first_target_callbacks"] <= 0
        or summary["tail_callbacks"] <= 0
    ):
        raise RuntimeError(
            "boundary speculative path did not execute"
        )
    rank_zero_summary = kv_offload_summaries[0]
    return {
        "mode": "baseline" if mode is None else mode,
        "outputs": outputs,
        "prompt_token_ids": list(prompt_token_ids),
        "evicted_block_identities": [
            list(identity)
            for identity in evicted_identities
        ],
        "observations": observations,
        "summary": summary,
        "movement": {
            key: rank_zero_summary.get(key)
            for key in _MOVEMENT_KEYS
        },
        "residency": {
            key: rank_zero_summary.get(key)
            for key in _RESIDENCY_KEYS
        },
        "elapsed_s": time.perf_counter() - started,
    }


def _case_counter(
    case: dict,
    section: str,
    key: str,
) -> int:
    value = case.get(section, {}).get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{section}.{key} must be a non-negative integer"
        )
    return value


def _validate_source_hashes(
    source_hashes: object,
) -> dict[str, str]:
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(
            "source_hashes must be a non-empty mapping"
        )
    normalized = {}
    for raw_path, digest in source_hashes.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(
                "source hash paths must be non-empty strings"
            )
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "source hash paths must be safe relative paths"
            )
        if (
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError(
                "source hashes must be lowercase SHA-256 values"
            )
        normalized[path.as_posix()] = digest
    return normalized


def _validate_outputs(
    outputs: object,
    name: str,
) -> list[list[int]]:
    if (
        not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0], list)
    ):
        raise ValueError(
            f"{name} outputs must contain one token list"
        )
    for token_id in outputs[0]:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
        ):
            raise ValueError(
                f"{name} output tokens must be "
                "non-negative integers"
            )
    return outputs


def _validate_boundary_case(
    case: object,
    name: str,
) -> dict:
    if not isinstance(case, dict):
        raise ValueError(f"{name} case must be a mapping")
    _validate_outputs(case.get("outputs"), name)
    identities = case.get("evicted_block_identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError(
            f"{name} evicted block identities are required"
        )
    for identity in identities:
        if (
            not isinstance(identity, list)
            or len(identity) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in identity
            )
        ):
            raise ValueError(
                f"{name} evicted block identities are invalid"
            )
    elapsed_s = case.get("elapsed_s")
    if (
        isinstance(elapsed_s, bool)
        or not isinstance(elapsed_s, (int, float))
        or not math.isfinite(elapsed_s)
        or elapsed_s < 0
    ):
        raise ValueError(
            f"{name} elapsed_s must be finite and non-negative"
        )
    movement = case.get("movement")
    if (
        not isinstance(movement, dict)
        or set(movement) != set(_MOVEMENT_KEYS)
    ):
        raise ValueError(
            f"{name} movement counters are incomplete"
        )
    for key in _MOVEMENT_KEYS:
        _case_counter(case, "movement", key)
    residency = case.get("residency")
    if (
        not isinstance(residency, dict)
        or set(residency) != set(_RESIDENCY_KEYS)
    ):
        raise ValueError(
            f"{name} residency counters are incomplete"
        )
    for key in _RESIDENCY_KEYS:
        _case_counter(case, "residency", key)
    summary = case.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(
            f"{name} summary must be a mapping"
        )
    for key in (
        "proposed_tokens",
        "accepted_draft_tokens",
    ):
        _case_counter(case, "summary", key)
    return case


def validate_boundary_artifact(
    artifact: object,
) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("artifact must be a mapping")
    if any(
        "performance" in str(key).lower()
        for key in artifact
    ):
        raise ValueError(
            "performance claims are forbidden in this artifact"
        )
    required_keys = {
        "schema_version",
        "status",
        "classification",
        "claim_scope",
        "limitations",
        "environment",
        "source_hashes",
        "cases",
    }
    if set(artifact) != required_keys:
        raise ValueError(
            "artifact schema keys are invalid"
        )
    if artifact["schema_version"] != 1:
        raise ValueError(
            "artifact schema_version must be 1"
        )
    if artifact["status"] != "PASS":
        raise ValueError("artifact status must be PASS")
    if artifact["classification"] != "NOT_PROMOTABLE":
        raise ValueError(
            "artifact classification must be NOT_PROMOTABLE"
        )
    if (
        not isinstance(artifact["claim_scope"], str)
        or not artifact["claim_scope"]
    ):
        raise ValueError(
            "artifact claim_scope must be non-empty"
        )
    limitations = artifact["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(
            not isinstance(value, str) or not value
            for value in limitations
        )
    ):
        raise ValueError(
            "artifact limitations must be non-empty strings"
        )
    environment = artifact["environment"]
    if not isinstance(environment, dict) or not environment:
        raise ValueError(
            "artifact environment must be a non-empty mapping"
        )
    _validate_source_hashes(artifact["source_hashes"])
    cases = artifact["cases"]
    if (
        not isinstance(cases, dict)
        or set(cases)
        != {
            "baseline",
            "accepted_boundary",
            "rejected_boundary",
        }
    ):
        raise ValueError(
            "artifact cases are incomplete"
        )
    baseline_case = _validate_boundary_case(
        cases["baseline"],
        "baseline",
    )
    accept_case = _validate_boundary_case(
        cases["accepted_boundary"],
        "accepted",
    )
    reject_case = _validate_boundary_case(
        cases["rejected_boundary"],
        "rejected",
    )
    baseline_outputs = baseline_case["outputs"]
    if (
        accept_case["outputs"] != baseline_outputs
        or reject_case["outputs"] != baseline_outputs
    ):
        raise ValueError(
            "boundary cases failed exact output parity"
        )
    if (
        _case_counter(
            accept_case,
            "summary",
            "proposed_tokens",
        )
        <= 0
        or _case_counter(
            accept_case,
            "summary",
            "accepted_draft_tokens",
        )
        <= 0
        or _case_counter(
            accept_case,
            "residency",
            "speculative_residency_committed_blocks",
        )
        <= 0
    ):
        raise ValueError(
            "accepted case lacks committed accepted evidence"
        )
    if (
        _case_counter(
            accept_case,
            "residency",
            "speculative_residency_rejected_blocks",
        )
        != 0
    ):
        raise ValueError(
            "accepted case unexpectedly rejected blocks"
        )
    if (
        _case_counter(
            reject_case,
            "summary",
            "proposed_tokens",
        )
        <= 0
        or _case_counter(
            reject_case,
            "summary",
            "accepted_draft_tokens",
        )
        != 0
        or _case_counter(
            reject_case,
            "residency",
            "speculative_residency_committed_blocks",
        )
        != 0
        or _case_counter(
            reject_case,
            "residency",
            "speculative_residency_rejected_blocks",
        )
        <= 0
    ):
        raise ValueError(
            "rejected case lacks rejected blocks"
        )
    if (
        _case_counter(
            reject_case,
            "residency",
            "speculative_residency_rejected_d2h_copies",
        )
        != 0
    ):
        raise ValueError(
            "rejected D2H copies must remain zero"
        )
    for name, case in (
        ("accepted", accept_case),
        ("rejected", reject_case),
    ):
        if (
            _case_counter(
                case,
                "movement",
                "h2d_copies",
            )
            <= 0
            or _case_counter(
                case,
                "movement",
                "h2d_bytes",
            )
            <= 0
        ):
            raise ValueError(
                f"{name} case lacks positive H2D evidence"
            )
    return artifact


def build_boundary_artifact(
    *,
    baseline_case: dict,
    accept_case: dict,
    reject_case: dict,
    source_hashes: dict[str, str],
    environment: dict,
) -> dict:
    baseline_outputs = baseline_case.get("outputs")
    if (
        accept_case.get("outputs") != baseline_outputs
        or reject_case.get("outputs") != baseline_outputs
    ):
        raise ValueError(
            "boundary cases failed exact output parity"
        )
    if (
        _case_counter(
            accept_case,
            "summary",
            "proposed_tokens",
        )
        <= 0
        or _case_counter(
            accept_case,
            "summary",
            "accepted_draft_tokens",
        )
        <= 0
    ):
        raise ValueError(
            "accepted case lacks accepted proposals"
        )
    if (
        _case_counter(
            accept_case,
            "residency",
            "speculative_residency_committed_blocks",
        )
        <= 0
    ):
        raise ValueError(
            "accepted case lacks committed blocks"
        )
    if (
        _case_counter(
            reject_case,
            "summary",
            "proposed_tokens",
        )
        <= 0
        or _case_counter(
            reject_case,
            "summary",
            "accepted_draft_tokens",
        )
        != 0
    ):
        raise ValueError(
            "rejected case acceptance evidence is invalid"
        )
    if (
        _case_counter(
            reject_case,
            "residency",
            "speculative_residency_rejected_blocks",
        )
        <= 0
    ):
        raise ValueError(
            "rejected case lacks rejected blocks"
        )
    if (
        _case_counter(
            reject_case,
            "residency",
            "speculative_residency_rejected_d2h_copies",
        )
        != 0
    ):
        raise ValueError(
            "rejected D2H copies must remain zero"
        )
    for name, case in (
        ("accepted", accept_case),
        ("rejected", reject_case),
    ):
        if (
            _case_counter(
                case,
                "movement",
                "h2d_copies",
            )
            <= 0
            or _case_counter(
                case,
                "movement",
                "h2d_bytes",
            )
            <= 0
        ):
            raise ValueError(
                f"{name} case lacks positive H2D evidence"
            )
    artifact = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "NOT_PROMOTABLE",
        "claim_scope": (
            "TP1 loaded-model speculative residency "
            "boundary correctness"
        ),
        "limitations": [
            "no TPOT or throughput improvement claim",
            "no long-context claim",
            "no TP4 claim",
            "no learned-drafter or MTP claim",
        ],
        "environment": dict(environment),
        "source_hashes": dict(source_hashes),
        "cases": {
            "baseline": baseline_case,
            "accepted_boundary": accept_case,
            "rejected_boundary": reject_case,
        },
    }
    return validate_boundary_artifact(artifact)


def hash_source_files(
    *,
    repo_root: Path,
    source_files: tuple[str, ...] = SOURCE_FILES,
) -> dict[str, str]:
    if not isinstance(source_files, tuple) or not source_files:
        raise ValueError(
            "source_files must be a non-empty tuple"
        )
    repo_root = Path(repo_root)
    result = {}
    for relative_path in source_files:
        normalized = next(
            iter(
                _validate_source_hashes(
                    {relative_path: "0" * 64}
                )
            )
        )
        path = repo_root / normalized
        if not path.is_file():
            raise FileNotFoundError(
                f"source file is missing: {normalized}"
            )
        result[normalized] = hashlib.sha256(
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
            )
            handle.write("\n")
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
    prompt_token_ids: object,
    output_path: Path,
    command: list[str],
    source_files: tuple[str, ...] = SOURCE_FILES,
    runtime_types: tuple[type, type] | None = None,
    adapter_types: _AdapterTypes | None = None,
    environment: dict | None = None,
) -> dict:
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(value, str) or not value
            for value in command
        )
    ):
        raise ValueError(
            "command must be a non-empty string list"
        )
    baseline_case = run_boundary_case(
        engine_factory=engine_factory,
        model_path=model_path,
        prompt_token_ids=prompt_token_ids,
        mode=None,
        accepted_token_ids=None,
        runtime_types=runtime_types,
        adapter_types=adapter_types,
    )
    baseline_outputs = baseline_case["outputs"]
    if (
        len(baseline_outputs) != 1
        or len(baseline_outputs[0]) != 4
    ):
        raise RuntimeError(
            "baseline boundary case must return four tokens"
        )
    accepted_token_ids = tuple(
        baseline_outputs[0][1:]
    )
    accept_case = run_boundary_case(
        engine_factory=engine_factory,
        model_path=model_path,
        prompt_token_ids=prompt_token_ids,
        mode="accept",
        accepted_token_ids=accepted_token_ids,
        runtime_types=runtime_types,
        adapter_types=adapter_types,
    )
    reject_case = run_boundary_case(
        engine_factory=engine_factory,
        model_path=model_path,
        prompt_token_ids=prompt_token_ids,
        mode="reject",
        accepted_token_ids=accepted_token_ids,
        runtime_types=runtime_types,
        adapter_types=adapter_types,
    )
    if environment is None:
        device = _device_environment()
        environment = {
            "model_path": str(Path(model_path).resolve()),
            "model_identifier": Path(model_path).name,
            "tensor_parallel_size": 1,
            "temperature": 0.0,
            "max_tokens": 4,
            "ignore_eos": True,
            "kv_offload_mvp0": True,
            "kv_offload_gpu_blocks": 2,
            "kv_offload_logical_blocks": 64,
            **device,
        }
    else:
        environment = dict(environment)
    environment["command"] = list(command)
    artifact = build_boundary_artifact(
        baseline_case=baseline_case,
        accept_case=accept_case,
        reject_case=reject_case,
        source_hashes=hash_source_files(
            repo_root=repo_root,
            source_files=source_files,
        ),
        environment=environment,
    )
    artifact = json.loads(
        json.dumps(
            artifact,
            sort_keys=True,
        )
    )
    validate_boundary_artifact(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def _default_engine_factory():
    from tinyvllm import LLM

    return LLM


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
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
        prompt_token_ids=list(range(254)),
        output_path=args.out,
        command=command,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "classification": artifact[
                    "classification"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
