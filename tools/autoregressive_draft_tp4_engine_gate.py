from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Callable

from tools.autoregressive_draft_tp1_engine_gate import (
    MAX_PROPOSAL_TOKENS,
    _default_identity_provider,
    _normalize_prompts,
    _positive_integer,
    _workload_configuration,
    load_prompt_file,
)
from tools.autoregressive_draft_tp4_local_gate import (
    validate_autoregressive_draft_tp4_local_evidence,
)


SCHEMA_VERSION = 2
GATE_NAME = "autoregressive_draft_tp4_engine"
DEFAULT_MAX_OUTPUT_TOKENS = 32
_EXPECTED_CONFIGURATION = {
    "tensor_parallel_size": 4,
    "allocator_mode": "direct",
    "dtype": "bfloat16",
    "temperature": 0.0,
    "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
}
DEFAULT_SOURCE_FILES = tuple(sorted((
    "tinyvllm/__init__.py",
    "tinyvllm/llm.py",
    "tinyvllm/config.py",
    "tinyvllm/sampling_params.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/model_runner_command_ack.py",
    "tinyvllm/engine/autoregressive_draft_registration.py",
    "tinyvllm/engine/autoregressive_draft_tp.py",
    "tinyvllm/engine/autoregressive_draft_executor.py",
    "tinyvllm/engine/qwen3_draft_backend.py",
    "tinyvllm/engine/qwen3_draft_proposal_kv.py",
    "tinyvllm/engine/proposal_kv_allocator.py",
    "tinyvllm/engine/proposal_kv_cache.py",
    "tinyvllm/engine/proposal_kv_lifecycle.py",
    "tinyvllm/engine/proposal_kv_residency.py",
    "tinyvllm/engine/speculative_proposal_executor.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tinyvllm/engine/speculative_selection.py",
    "tinyvllm/engine/tensor_parallel_greedy.py",
    "tinyvllm/models/qwen3.py",
    "tinyvllm/speculative/adapter.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tinyvllm/speculative/verifier.py",
    "tinyvllm/utils/context.py",
    "tinyvllm/utils/loader.py",
    "tools/autoregressive_draft_tp1_engine_gate.py",
    "tools/autoregressive_draft_tp4_engine_gate.py",
    "tools/autoregressive_draft_tp4_local_gate.py",
    "tools/verify_autoregressive_draft_tp4_engine_gate.py",
)))


def _validate_gpu_indices(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != 4
        or len(set(value)) != 4
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in value
        )
    ):
        raise ValueError(
            "GPU indices must contain four distinct "
            "non-negative integers"
        )
    return value


@contextmanager
def distributed_environment(
    *,
    gpu_indices: tuple[int, ...],
    dist_port: int,
    master_port: int,
):
    gpu_indices = _validate_gpu_indices(gpu_indices)
    for value, name in (
        (dist_port, "distributed port"),
        (master_port, "master port"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    names = (
        "CUDA_VISIBLE_DEVICES",
        "TINYVLLM_DIST_PORT",
        "MASTER_PORT",
    )
    previous = {
        name: os.environ.get(name)
        for name in names
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index in gpu_indices
    )
    os.environ["TINYVLLM_DIST_PORT"] = str(dist_port)
    os.environ["MASTER_PORT"] = str(master_port)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _TinyVLLMTP4EngineAdapter:

    def __init__(
        self,
        mode,
        *,
        target_model,
        draft_model,
        tensor_parallel_size,
        max_num_seqs,
        max_model_len,
        max_num_batched_tokens,
        proposal_slot_capacity,
        learned_enabled,
        cuda_graph_enabled=False,
        cuda_graph_max_reserved_bytes=None,
        cuda_graph_max_total_capture_ns=None,
        cuda_graph_max_single_capture_ns=None,
        llm_type=None,
        sampling_params_type=None,
        runtime_type=None,
    ):
        if mode not in ("target", "learned"):
            raise ValueError("engine mode is invalid")
        if tensor_parallel_size != 4:
            raise ValueError("TP4 adapter requires tensor parallel four")
        if learned_enabled is not (mode == "learned"):
            raise ValueError("learned engine mode mismatch")
        if not isinstance(cuda_graph_enabled, bool):
            raise ValueError("CUDA graph enable flag must be bool")
        if cuda_graph_enabled and not learned_enabled:
            raise ValueError(
                "draft CUDA graph requires learned mode"
            )
        graph_budget_overrides = {
            "autoregressive_draft_cuda_graph_max_reserved_bytes": (
                cuda_graph_max_reserved_bytes
            ),
            "autoregressive_draft_cuda_graph_max_total_capture_ns": (
                cuda_graph_max_total_capture_ns
            ),
            "autoregressive_draft_cuda_graph_max_single_capture_ns": (
                cuda_graph_max_single_capture_ns
            ),
        }
        for name, value in graph_budget_overrides.items():
            if value is None:
                continue
            if (
                not cuda_graph_enabled
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} override requires enabled CUDA graphs "
                    "and a positive integer"
                )
        if (
            llm_type is None
            or sampling_params_type is None
            or runtime_type is None
        ):
            from tinyvllm import LLM, SamplingParams
            from tinyvllm.engine.speculative_runtime import (
                EngineSpeculativeRuntime,
            )

            llm_type = LLM if llm_type is None else llm_type
            sampling_params_type = (
                SamplingParams
                if sampling_params_type is None
                else sampling_params_type
            )
            runtime_type = (
                EngineSpeculativeRuntime
                if runtime_type is None
                else runtime_type
            )
        self.mode = mode
        self._sampling_params_type = sampling_params_type
        engine_kwargs = {
            "tensor_parallel_size": 4,
            "enforce_eager": True,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "autoregressive_draft_enabled": learned_enabled,
            "autoregressive_draft_model": (
                draft_model if learned_enabled else None
            ),
            "autoregressive_draft_backend": "qwen3",
            "autoregressive_draft_max_proposal_tokens": (
                MAX_PROPOSAL_TOKENS
            ),
            "autoregressive_draft_gpu_slot_capacity": (
                proposal_slot_capacity if learned_enabled else 0
            ),
            "autoregressive_draft_proposal_kv_offload_enabled": False,
            "autoregressive_draft_cuda_graphs": (
                cuda_graph_enabled
            ),
            "autoregressive_draft_logical_entry_capacity": 0,
            "autoregressive_draft_cpu_backing_capacity": 0,
            "proposal_kv_async_copy": True,
            "proposal_kv_batch_copy": True,
        }
        engine_kwargs.update({
            name: value
            for name, value in graph_budget_overrides.items()
            if value is not None
        })
        self.engine = llm_type(target_model, **engine_kwargs)
        if learned_enabled:
            descriptor = getattr(
                self.engine.model_runner,
                "autoregressive_draft_executor_descriptor",
                None,
            )
            if descriptor is None:
                error = getattr(
                    self.engine.model_runner,
                    "autoregressive_draft_registration_error",
                    None,
                )
                self.engine.exit()
                raise RuntimeError(
                    "autoregressive draft registration failed: "
                    f"{error}"
                )
            self.engine.activate_speculative_runtime(
                runtime_type(
                    model_runner_executor=descriptor
                )
            )

    def run_case(self, prompts, *, max_output_tokens):
        prompts = tuple(tuple(prompt) for prompt in prompts)
        sampling_configuration = {
            "temperature": 0.0,
            "max_tokens": max_output_tokens,
            "ignore_eos": True,
        }
        sampling_params = self._sampling_params_type(
            **sampling_configuration
        )
        if not self.engine.is_finished():
            raise RuntimeError("engine must be idle before generation")
        for prompt in prompts:
            self.engine.add_request(
                list(prompt),
                sampling_params,
            )
        outputs = {}
        observations = []
        while not self.engine.is_finished():
            rows, _ = self.engine.step()
            observation = getattr(
                self.engine,
                "last_step_observation",
                None,
            )
            if isinstance(observation, dict):
                observations.append(dict(observation))
            for sequence_id, token_ids in rows:
                outputs[int(sequence_id)] = list(token_ids)
        output_token_ids = [
            outputs[sequence_id]
            for sequence_id in sorted(outputs)
        ]
        if len(output_token_ids) != len(prompts):
            raise RuntimeError(
                "engine did not return one output per prompt"
            )
        ordered_sequence_ids = tuple(sorted(outputs))
        sequence_to_prompt = {
            sequence_id: prompt_index
            for prompt_index, sequence_id in enumerate(
                ordered_sequence_ids
            )
        }
        acceptance_rows = []
        completion_counts = {
            sequence_id: 0
            for sequence_id in ordered_sequence_ids
        }
        event_index = 0
        for step_index, observation in enumerate(observations):
            proposal_ids = observation.get(
                "speculative_proposal_token_ids_by_seq",
                {},
            )
            accepted_counts = observation.get(
                "speculative_accepted_draft_token_counts",
                {},
            )
            accepted_ids = observation.get(
                "speculative_accepted_draft_token_ids_by_seq",
                {},
            )
            completion_deltas = observation.get(
                "new_completion_tokens_by_seq",
                {},
            )
            for sequence_id in sorted(proposal_ids):
                if sequence_id not in sequence_to_prompt:
                    raise RuntimeError(
                        "acceptance sequence identity is unknown"
                    )
                prompt_index = sequence_to_prompt[sequence_id]
                token_ids = list(proposal_ids[sequence_id])
                accepted_prefix_count = int(
                    accepted_counts.get(sequence_id, 0)
                )
                accepted_prefix_token_ids = list(
                    accepted_ids.get(
                        sequence_id,
                        token_ids[:accepted_prefix_count],
                    )
                )
                acceptance_rows.append({
                    "event_index": event_index,
                    "step_index": step_index,
                    "sequence_id": int(sequence_id),
                    "prompt_index": prompt_index,
                    "prompt_token_ids": list(
                        prompts[prompt_index]
                    ),
                    "output_token_count_before_step": (
                        completion_counts[sequence_id]
                    ),
                    "proposal_token_ids": token_ids,
                    "accepted_prefix_count": accepted_prefix_count,
                    "accepted_prefix_token_ids": (
                        accepted_prefix_token_ids
                    ),
                })
                event_index += 1
            for sequence_id, token_ids in completion_deltas.items():
                if sequence_id not in completion_counts:
                    raise RuntimeError(
                        "completion sequence identity is unknown"
                    )
                completion_counts[sequence_id] += len(token_ids)
        rank_snapshots = ()
        if self.mode == "learned":
            self.engine.flush_pending_hybrid_state_releases(
                timeout_s=60.0
            )
            rank_snapshots = (
                self.engine
                .autoregressive_draft_authority_snapshots(
                    timeout_s=60.0
                )
            )
        return {
            "output_token_ids": output_token_ids,
            "acceptance_rows": acceptance_rows,
            "rank_snapshots": rank_snapshots,
            "sampling_params": sampling_configuration,
        }

    def close(self):
        self.engine.exit()


def _default_engine_factory(mode, **kwargs):
    return _TinyVLLMTP4EngineAdapter(mode, **kwargs)


def _case_payload(
    target_result,
    learned_result,
    prompts,
) -> dict:
    target_outputs = target_result["output_token_ids"]
    learned_outputs = learned_result["output_token_ids"]
    rank_snapshots = tuple(learned_result.get("rank_snapshots", ()))
    rank_summary = validate_autoregressive_draft_tp4_local_evidence(
        rank_snapshots
    )
    if target_outputs != learned_outputs:
        raise ValueError("exact output parity failed")
    return {
        "prompts": [list(prompt) for prompt in prompts],
        "target_output_token_ids": target_outputs,
        "learned_output_token_ids": learned_outputs,
        "exact_output_parity": True,
        "acceptance_rows": list(
            learned_result.get("acceptance_rows", ())
        ),
        "rank_snapshots": list(rank_snapshots),
        "rank_summary": rank_summary,
    }


def run_gate(
    *,
    target_model,
    draft_model,
    prompts,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    engine_factory: Callable = _default_engine_factory,
    identity_provider: Callable = _default_identity_provider,
) -> dict:
    normalized_prompts = _normalize_prompts(prompts)
    max_output_tokens = _positive_integer(
        max_output_tokens,
        "max_output_tokens",
    )
    checkpoint_identity, tokenizer_contract = identity_provider(
        target_model,
        draft_model,
    )
    workload = _workload_configuration(
        normalized_prompts,
        max_output_tokens=max_output_tokens,
    )
    case_rows = (
        ("batch_1", normalized_prompts[:1]),
        ("batch_4", normalized_prompts[:4]),
    )
    engine_kwargs = {
        "target_model": target_model,
        "draft_model": draft_model,
        "tensor_parallel_size": 4,
        "max_num_seqs": 4,
        "max_model_len": workload["max_model_len"],
        "max_num_batched_tokens": workload[
            "max_num_batched_tokens"
        ],
        "proposal_slot_capacity": workload[
            "proposal_slot_capacity"
        ],
    }

    target_results = {}
    target_engine = None
    try:
        target_engine = engine_factory(
            "target",
            **engine_kwargs,
            learned_enabled=False,
        )
        for name, case_prompts in case_rows:
            target_results[name] = target_engine.run_case(
                case_prompts,
                max_output_tokens=max_output_tokens,
            )
    finally:
        if target_engine is not None:
            target_engine.close()

    learned_results = {}
    learned_engine = None
    try:
        learned_engine = engine_factory(
            "learned",
            **engine_kwargs,
            learned_enabled=True,
        )
        for name, case_prompts in case_rows:
            learned_results[name] = learned_engine.run_case(
                case_prompts,
                max_output_tokens=max_output_tokens,
            )
    finally:
        if learned_engine is not None:
            learned_engine.close()

    cases = {
        name: _case_payload(
            target_results[name],
            learned_results[name],
            case_prompts,
        )
        for name, case_prompts in case_rows
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "configuration": dict(_EXPECTED_CONFIGURATION),
        "checkpoint_identity": checkpoint_identity,
        "tokenizer_contract": tokenizer_contract,
        "workload": workload,
        "cases": cases,
        "performance_pass_criterion": False,
        "real_proposal_kv_movement": False,
        "gate_pass": True,
    }
    validate_gate_payload(payload)
    return payload


def validate_gate_payload(payload) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("gate schema version mismatch")
    if payload.get("gate") != GATE_NAME:
        raise ValueError("gate name mismatch")
    if payload.get("configuration") != _EXPECTED_CONFIGURATION:
        raise ValueError("gate configuration mismatch")
    if payload.get("performance_pass_criterion") is not False:
        raise ValueError("performance claim is not permitted")
    if payload.get("real_proposal_kv_movement") is not False:
        raise ValueError("movement claim is not permitted")
    if payload.get("gate_pass") is not True:
        raise ValueError("gate pass classification is invalid")

    checkpoint_identity = payload.get("checkpoint_identity", {})
    for name in ("target", "draft"):
        if not checkpoint_identity.get(name, {}).get(
            "composite_sha256"
        ):
            raise ValueError(
                f"{name} checkpoint identity is missing"
            )
    tokenizer_contract = payload.get("tokenizer_contract", {})
    if tokenizer_contract.get("compatible") is not True:
        raise ValueError("tokenizer compatibility is not established")
    for name in ("target", "draft"):
        row = tokenizer_contract.get(name, {})
        if (
            not row.get("ordered_token_to_id_sha256")
            or "artifact_sha256" not in row
        ):
            raise ValueError("tokenizer identity evidence is missing")

    cases = payload.get("cases", {})
    for name in ("batch_1", "batch_4"):
        row = cases.get(name, {})
        if (
            row.get("exact_output_parity") is not True
            or row.get("target_output_token_ids")
            != row.get("learned_output_token_ids")
        ):
            raise ValueError(
                f"{name} exact output parity failed"
            )
        acceptance_rows = row.get("acceptance_rows")
        if not isinstance(acceptance_rows, list) or not acceptance_rows:
            raise ValueError(
                f"{name} acceptance evidence is missing"
            )
        expected_event_index = 0
        last_step_index = -1
        sequence_to_prompt = {}
        prompt_to_sequence = {}
        last_output_count = {}
        prompts = row.get("prompts")
        if not isinstance(prompts, list):
            raise ValueError(f"{name} prompt evidence is invalid")
        for acceptance in acceptance_rows:
            if not isinstance(acceptance, dict):
                raise ValueError(
                    f"{name} acceptance evidence is invalid"
                )
            event_index = acceptance.get("event_index")
            step_index = acceptance.get("step_index")
            sequence_id = acceptance.get("sequence_id")
            prompt_index = acceptance.get("prompt_index")
            prompt_token_ids = acceptance.get("prompt_token_ids")
            output_count = acceptance.get(
                "output_token_count_before_step"
            )
            proposal_token_ids = acceptance.get(
                "proposal_token_ids"
            )
            accepted_prefix_count = acceptance.get(
                "accepted_prefix_count"
            )
            accepted_prefix_token_ids = acceptance.get(
                "accepted_prefix_token_ids"
            )
            if event_index != expected_event_index:
                raise ValueError(
                    f"{name} acceptance event ordering is invalid"
                )
            if (
                isinstance(step_index, bool)
                or not isinstance(step_index, int)
                or step_index < last_step_index
                or isinstance(sequence_id, bool)
                or not isinstance(sequence_id, int)
                or sequence_id < 0
                or isinstance(prompt_index, bool)
                or not isinstance(prompt_index, int)
                or prompt_index < 0
                or prompt_index >= len(prompts)
                or not isinstance(prompt_token_ids, list)
                or prompt_token_ids != prompts[prompt_index]
                or isinstance(output_count, bool)
                or not isinstance(output_count, int)
                or output_count < 0
                or not isinstance(proposal_token_ids, list)
                or not proposal_token_ids
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in proposal_token_ids
                )
                or isinstance(accepted_prefix_count, bool)
                or not isinstance(accepted_prefix_count, int)
                or accepted_prefix_count < 0
                or accepted_prefix_count > len(
                    proposal_token_ids
                )
                or not isinstance(
                    accepted_prefix_token_ids,
                    list,
                )
            ):
                raise ValueError(
                    f"{name} acceptance evidence is invalid"
                )
            if (
                accepted_prefix_token_ids
                != proposal_token_ids[:accepted_prefix_count]
            ):
                raise ValueError(
                    f"{name} accepted prefix identity is invalid"
                )
            previous_prompt = sequence_to_prompt.setdefault(
                sequence_id,
                prompt_index,
            )
            previous_sequence = prompt_to_sequence.setdefault(
                prompt_index,
                sequence_id,
            )
            if (
                previous_prompt != prompt_index
                or previous_sequence != sequence_id
            ):
                raise ValueError(
                    f"{name} acceptance prompt identity is invalid"
                )
            previous_output_count = last_output_count.get(
                sequence_id,
                -1,
            )
            if output_count < previous_output_count:
                raise ValueError(
                    f"{name} acceptance output boundary is invalid"
                )
            last_output_count[sequence_id] = output_count
            expected_event_index += 1
            last_step_index = step_index
        snapshots = tuple(row.get("rank_snapshots", ()))
        summary = validate_autoregressive_draft_tp4_local_evidence(
            snapshots
        )
        if row.get("rank_summary") != summary:
            raise ValueError(f"{name} rank summary mismatch")
        if summary.get("classification") != "NOT_PROMOTABLE":
            raise ValueError(
                f"{name} promotion classification is invalid"
            )
        if summary.get("promotion_boundary", {}).get(
            "phase_1"
        ) != "NOT_ACHIEVED":
            raise ValueError(
                f"{name} Phase 1 boundary is invalid"
            )


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        return _validate_gpu_indices(
            tuple(int(item) for item in value.split(","))
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_name(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("source inventory path is invalid")
    parts = name.split("/")
    pure = PurePosixPath(name)
    if (
        name.startswith("/")
        or name.endswith("/")
        or "\\" in name
        or any(part in ("", ".", "..") for part in parts)
        or pure.is_absolute()
        or pure.as_posix() != name
    ):
        raise ValueError("source inventory path is invalid")
    return name


def _validate_source_files(
    source_files: tuple[str, ...],
) -> tuple[str, ...]:
    if (
        not isinstance(source_files, tuple)
        or not source_files
    ):
        raise ValueError(
            "source inventory must be a non-empty tuple"
        )
    normalized = tuple(
        _validate_source_name(name)
        for name in source_files
    )
    if normalized != tuple(sorted(normalized)):
        raise ValueError("source inventory must be sorted")
    if len(normalized) != len(set(normalized)):
        raise ValueError("source inventory contains duplicates")
    return normalized


def _source_file(root: Path, name: str) -> Path:
    root = Path(root).resolve()
    path = root / _validate_source_name(name)
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"source inventory member is not a regular file: {name}"
        )
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"source inventory member escapes source root: {name}"
        ) from error
    return path


def hash_source_files(
    root: Path,
    source_files: tuple[str, ...],
) -> dict[str, str]:
    source_files = _validate_source_files(source_files)
    return {
        name: sha256_file(_source_file(root, name))
        for name in source_files
    }


def source_tree_sha256(
    root: Path,
    source_files: tuple[str, ...],
) -> str:
    source_files = _validate_source_files(source_files)
    digest = hashlib.sha256()
    for name in source_files:
        payload = _source_file(root, name).read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_source_archive(
    root: Path,
    archive_path: Path,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
) -> None:
    source_files = _validate_source_files(source_files)
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:") as archive:
        for name in source_files:
            path = _source_file(root, name)
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)


def safe_extract_source_archive(
    archive_path: Path,
    destination: Path,
    expected_hashes: dict[str, str],
) -> tuple[str, ...]:
    if not isinstance(expected_hashes, dict):
        raise ValueError("source archive hashes are invalid")
    expected_names = tuple(sorted(expected_hashes))
    try:
        _validate_source_files(expected_names)
    except ValueError as error:
        raise ValueError(
            "unsafe source archive inventory"
        ) from error
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"source extraction destination exists: {destination}"
        )
    buffered = []
    try:
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            names = tuple(member.name for member in members)
            if (
                names != expected_names
                or len(names) != len(set(names))
            ):
                raise ValueError(
                    "unsafe source archive inventory"
                )
            for member in members:
                try:
                    name = _validate_source_name(member.name)
                except ValueError as error:
                    raise ValueError(
                        "unsafe source archive member"
                    ) from error
                if (
                    not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                ):
                    raise ValueError(
                        "unsafe source archive member"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(
                        "unsafe source archive member"
                    )
                payload = source.read()
                digest = hashlib.sha256(payload).hexdigest()
                if digest != expected_hashes.get(name):
                    raise ValueError(
                        "source archive member SHA-256 mismatch"
                    )
                buffered.append((name, payload))
    except (OSError, tarfile.TarError) as error:
        raise ValueError("unsafe source archive") from error
    try:
        destination.mkdir(parents=True)
        for name, payload in buffered:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(0o644)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return expected_names


def _run_verifier_process(
    verifier_path: Path,
    run_dir: Path,
    source_root: Path,
) -> dict:
    source_root = Path(source_root).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(source_root)
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(verifier_path),
            os.fspath(Path(run_dir).resolve()),
            "--source-root",
            os.fspath(source_root),
        ],
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "independent verifier did not emit valid JSON"
        ) from error
    if not isinstance(receipt, dict):
        raise RuntimeError(
            "independent verifier receipt is invalid"
        )
    return receipt


def publish_authority_bundle(
    payload: dict,
    output_dir: Path,
    *,
    source_root: Path | None = None,
    verification_runner: Callable = _run_verifier_process,
) -> dict:
    validate_gate_payload(payload)
    source_root = (
        Path(__file__).resolve().parents[1]
        if source_root is None
        else Path(source_root).resolve()
    )
    output_dir = Path(output_dir)
    failed_dir = output_dir.with_name(
        f"{output_dir.name}.failed"
    )
    if output_dir.exists():
        raise FileExistsError(
            f"output already exists: {output_dir}"
        )
    if failed_dir.exists():
        raise FileExistsError(
            f"failed output already exists: {failed_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        result_path = temporary_dir / "result.json"
        result_path.write_bytes(_canonical_json_bytes(payload))
        archive_path = temporary_dir / "source.tar"
        build_source_archive(
            source_root,
            archive_path,
            DEFAULT_SOURCE_FILES,
        )
        source_hashes = hash_source_files(
            source_root,
            DEFAULT_SOURCE_FILES,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_tree_sha256": source_tree_sha256(
                source_root,
                DEFAULT_SOURCE_FILES,
            ),
            "source_files": source_hashes,
            "artifacts": {
                "result.json": sha256_file(result_path),
                "source.tar": sha256_file(archive_path),
            },
        }
        (temporary_dir / "source_manifest.json").write_bytes(
            _canonical_json_bytes(manifest)
        )
        current_receipt = verification_runner(
            source_root
            / "tools"
            / "verify_autoregressive_draft_tp4_engine_gate.py",
            temporary_dir,
            source_root,
        )
        if current_receipt.get("classification") != "PASS":
            raise RuntimeError(
                "current-source independent verification failed"
            )
        with tempfile.TemporaryDirectory(
            prefix="autoregressive-draft-tp4-source-"
        ) as extraction_name:
            extracted_root = Path(extraction_name) / "source"
            safe_extract_source_archive(
                archive_path,
                extracted_root,
                source_hashes,
            )
            archived_receipt = verification_runner(
                extracted_root
                / "tools"
                / "verify_autoregressive_draft_tp4_engine_gate.py",
                temporary_dir,
                extracted_root,
            )
        receipts_match = current_receipt == archived_receipt
        verification = {
            "schema_version": SCHEMA_VERSION,
            "classification": (
                "PASS"
                if (
                    archived_receipt.get("classification") == "PASS"
                    and receipts_match
                )
                else "FAIL"
            ),
            "receipts_match": receipts_match,
            "current_receipt": current_receipt,
            "archived_receipt": archived_receipt,
        }
        (temporary_dir / "verify.json").write_bytes(
            _canonical_json_bytes(verification)
        )
        if verification["classification"] != "PASS":
            raise RuntimeError(
                "archived-source independent verification failed"
            )
        os.rename(temporary_dir, output_dir)
        return verification
    except BaseException as error:
        if temporary_dir.exists():
            failure_path = temporary_dir / "failure.json"
            failure_path.write_bytes(_canonical_json_bytes({
                "classification": "FAIL",
                "error": str(error)[:1000],
            }))
            os.rename(temporary_dir, failed_dir)
        raise
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt-file", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output")
    output.add_argument("--output-dir")
    parser.add_argument(
        "--gpu-indices",
        required=True,
        type=_gpu_indices,
    )
    parser.add_argument("--dist-port", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    return parser.parse_args(argv)


def _atomic_write_json_exclusive(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"output already exists: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    args = _parse_args(argv)
    output_path = (
        None if args.output is None else Path(args.output)
    )
    output_dir = (
        None
        if args.output_dir is None
        else Path(args.output_dir)
    )
    for path in (output_path, output_dir):
        if path is not None and path.exists():
            raise FileExistsError(
                f"output already exists: {path}"
            )
    prompts = load_prompt_file(
        args.prompt_file,
        target_model=args.target_model,
    )
    with distributed_environment(
        gpu_indices=args.gpu_indices,
        dist_port=args.dist_port,
        master_port=args.master_port,
    ):
        payload = run_gate(
            target_model=args.target_model,
            draft_model=args.draft_model,
            prompts=prompts,
            max_output_tokens=args.max_output_tokens,
        )
    if output_dir is not None:
        publish_authority_bundle(payload, output_dir)
    else:
        _atomic_write_json_exclusive(output_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
