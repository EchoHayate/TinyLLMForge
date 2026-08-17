from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Callable


SCHEMA_VERSION = 2
GATE_NAME = "autoregressive_draft_tp1_engine"
MAX_PROPOSAL_TOKENS = 4
DEFAULT_MAX_OUTPUT_TOKENS = 32
_ALLOCATOR_COUNTER_NAMES = (
    "h2d_operation_count",
    "h2d_entry_count",
    "h2d_bytes",
    "d2h_operation_count",
    "d2h_entry_count",
    "d2h_bytes",
    "accepted_entry_copy_count",
    "accepted_entry_replay_count",
    "accepted_entry_rematerialization_count",
)


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _positive_integer(value, name: str) -> int:
    value = _integer(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _normalize_prompts(prompts) -> tuple[tuple[int, ...], ...]:
    if not isinstance(prompts, (tuple, list)) or len(prompts) < 4:
        raise ValueError("gate requires at least four prompts")
    normalized = []
    for prompt in prompts:
        if not isinstance(prompt, (tuple, list)) or not prompt:
            raise ValueError("each prompt must contain token IDs")
        token_ids = tuple(
            _integer(token_id, "prompt token ID")
            for token_id in prompt
        )
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("prompt token IDs must be nonnegative")
        normalized.append(token_ids)
    return tuple(normalized)


def _default_identity_provider(target_model, draft_model):
    from transformers import AutoTokenizer

    from tinyvllm.engine.autoregressive_draft_registration import (
        build_checkpoint_fingerprint,
        build_tokenizer_contract,
        validate_tokenizer_compatibility,
    )

    def tokenizer_contract(path):
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
        )
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            stop_token_ids = ()
        elif isinstance(eos_token_id, int):
            stop_token_ids = (eos_token_id,)
        else:
            stop_token_ids = tuple(eos_token_id)
        return build_tokenizer_contract(
            path,
            tokenizer,
            stop_token_ids=stop_token_ids,
        )

    target_checkpoint = build_checkpoint_fingerprint(target_model)
    draft_checkpoint = build_checkpoint_fingerprint(draft_model)
    target_tokenizer = tokenizer_contract(target_model)
    draft_tokenizer = tokenizer_contract(draft_model)
    validate_tokenizer_compatibility(
        target_tokenizer,
        draft_tokenizer,
    )
    return (
        {
            "target": asdict(target_checkpoint),
            "draft": asdict(draft_checkpoint),
        },
        {
            "compatible": True,
            "target": asdict(target_tokenizer),
            "draft": asdict(draft_tokenizer),
        },
    )


def _sum_timing_rows(observations) -> dict[str, float]:
    totals = {}
    for observation in observations:
        timing = observation.get(
            "speculative_runtime_timing_ms",
            {},
        )
        if not isinstance(timing, dict):
            continue
        for name, value in timing.items():
            if isinstance(value, (int, float)) and not isinstance(
                value,
                bool,
            ):
                totals[name] = totals.get(name, 0.0) + float(value)
    return totals


def _allocator_snapshot(backend_snapshot) -> dict:
    if not isinstance(backend_snapshot, dict):
        raise ValueError("draft backend snapshot is invalid")
    try:
        allocator = backend_snapshot["proposal_kv_cache"][
            "entry_allocator"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "proposal KV allocator snapshot is missing"
        ) from error
    if not isinstance(allocator, dict):
        raise ValueError("proposal KV allocator snapshot is invalid")
    mode = allocator.get("allocator_mode")
    if mode not in ("direct", "residency"):
        raise ValueError("proposal KV allocator mode is invalid")
    logical_capacity = _positive_integer(
        allocator.get("logical_entry_capacity"),
        "proposal KV logical entry capacity",
    )
    gpu_capacity = allocator.get("gpu_slot_capacity")
    if gpu_capacity is None:
        gpu_capacity = allocator.get(
            "physical_store",
            {},
        ).get("gpu_capacity")
    gpu_capacity = _positive_integer(
        gpu_capacity,
        "proposal KV GPU slot capacity",
    )
    result = {
        "allocator_mode": mode,
        "logical_entry_capacity": logical_capacity,
        "gpu_slot_capacity": gpu_capacity,
    }
    for name in _ALLOCATOR_COUNTER_NAMES:
        value = allocator.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"proposal KV allocator {name} is invalid"
            )
        result[name] = value
    return result


def _allocator_evidence_delta(before, after) -> dict:
    before = _allocator_snapshot(before)
    after = _allocator_snapshot(after)
    for name in (
        "allocator_mode",
        "logical_entry_capacity",
        "gpu_slot_capacity",
    ):
        if before[name] != after[name]:
            raise ValueError(
                f"proposal KV allocator {name} changed during case"
            )
    result = {
        name: after[name]
        for name in (
            "allocator_mode",
            "logical_entry_capacity",
            "gpu_slot_capacity",
        )
    }
    for name in _ALLOCATOR_COUNTER_NAMES:
        delta = after[name] - before[name]
        if delta < 0:
            raise ValueError(
                f"proposal KV allocator {name} counter regressed"
            )
        result[name] = delta
    return result


def _owned_proposal_entry_count(backend_snapshot) -> int:
    if not isinstance(backend_snapshot, dict):
        raise ValueError("draft backend snapshot is invalid")
    try:
        value = backend_snapshot["proposal_kv_cache"][
            "owned_entry_count"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "proposal KV owned entry count is missing"
        ) from error
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            "proposal KV owned entry count is invalid"
        )
    return value


class _TinyVLLMEngineAdapter:

    def __init__(
        self,
        mode,
        *,
        target_model,
        draft_model,
        max_num_seqs=4,
        max_model_len,
        max_num_batched_tokens,
        proposal_slot_capacity,
        proposal_kv_configuration=None,
    ):
        from tinyvllm import LLM, SamplingParams
        from tinyvllm.engine.speculative_runtime import (
            EngineSpeculativeRuntime,
        )

        self._sampling_params_type = SamplingParams
        self.mode = mode
        if mode == "learned":
            if not isinstance(proposal_kv_configuration, dict):
                raise ValueError(
                    "learned mode requires proposal KV configuration"
                )
            proposal_kv = dict(proposal_kv_configuration)
        else:
            proposal_kv = {
                "offload_enabled": False,
                "logical_entry_capacity": 0,
                "gpu_slot_capacity": 0,
                "cpu_backing_capacity": 0,
                "async_copy": True,
                "batch_copy": True,
            }
        start = perf_counter()
        kwargs = {
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "autoregressive_draft_enabled": mode == "learned",
            "autoregressive_draft_model": (
                draft_model if mode == "learned" else None
            ),
            "autoregressive_draft_backend": "qwen3",
            "autoregressive_draft_max_proposal_tokens": (
                MAX_PROPOSAL_TOKENS
            ),
            "autoregressive_draft_gpu_slot_capacity": (
                proposal_kv["gpu_slot_capacity"]
                if mode == "learned"
                else 0
            ),
            "autoregressive_draft_proposal_kv_offload_enabled": (
                proposal_kv["offload_enabled"]
                if mode == "learned"
                else False
            ),
            "autoregressive_draft_logical_entry_capacity": (
                proposal_kv["logical_entry_capacity"]
                if mode == "learned"
                and proposal_kv["offload_enabled"]
                else 0
            ),
            "autoregressive_draft_cpu_backing_capacity": (
                proposal_kv["cpu_backing_capacity"]
                if mode == "learned"
                and proposal_kv["offload_enabled"]
                else 0
            ),
            "proposal_kv_async_copy": proposal_kv["async_copy"],
            "proposal_kv_batch_copy": proposal_kv["batch_copy"],
        }
        self.engine = LLM(target_model, **kwargs)
        self.model_load_ms = (perf_counter() - start) * 1000.0
        if mode == "learned":
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
                raise RuntimeError(
                    "autoregressive draft registration failed: "
                    f"{error}"
                )
            self.engine.activate_speculative_runtime(
                EngineSpeculativeRuntime(
                    model_runner_executor=descriptor
                )
            )

    def _backend_snapshot(self):
        executor = getattr(
            self.engine.model_runner,
            "autoregressive_draft_executor",
            None,
        )
        backend = getattr(executor, "backend", None)
        snapshot = getattr(backend, "authority_snapshot", None)
        return {} if not callable(snapshot) else snapshot()

    def run_case(self, prompts, *, max_output_tokens):
        sampling_params = self._sampling_params_type(
            temperature=0.0,
            max_tokens=max_output_tokens,
            ignore_eos=True,
        )
        model_runner = self.engine.model_runner
        original_call = model_runner.call
        method_names = []
        ordinary_decode_target_forwards = 0

        def recorded_call(method_name, *args, **kwargs):
            nonlocal ordinary_decode_target_forwards
            result = original_call(method_name, *args, **kwargs)
            method_names.append(method_name)
            if (
                method_name == "run"
                and len(args) >= 2
                and not bool(args[1])
            ):
                ordinary_decode_target_forwards += 1
            return result

        before_backend = self._backend_snapshot()
        model_runner.call = recorded_call
        observations = []
        outputs = {}
        max_live_slots = 0
        start = perf_counter()
        try:
            for prompt in prompts:
                self.engine.add_request(
                    list(prompt),
                    sampling_params,
                )
            while not self.engine.is_finished():
                rows, _ = self.engine.step()
                observation = self.engine.last_step_observation
                if isinstance(observation, dict):
                    observations.append(dict(observation))
                for sequence_id, token_ids in rows:
                    outputs[int(sequence_id)] = list(token_ids)
                backend_snapshot = self._backend_snapshot()
                if self.mode == "learned":
                    max_live_slots = max(
                        max_live_slots,
                        _owned_proposal_entry_count(
                            backend_snapshot
                        ),
                    )
        finally:
            model_runner.call = original_call
        generation_ms = (perf_counter() - start) * 1000.0
        after_backend = self._backend_snapshot()
        output_token_ids = [
            outputs[sequence_id]
            for sequence_id in sorted(outputs)
        ]
        acceptance_rows = []
        for observation in observations:
            proposal_ids = observation.get(
                "speculative_proposal_token_ids_by_seq",
                {},
            )
            accepted_counts = observation.get(
                "speculative_accepted_draft_token_counts",
                {},
            )
            for sequence_id, token_ids in proposal_ids.items():
                acceptance_rows.append({
                    "sequence_id": int(sequence_id),
                    "proposal_token_ids": list(token_ids),
                    "accepted_prefix_count": int(
                        accepted_counts.get(sequence_id, 0)
                    ),
                })
        memory = model_runner.memory_snapshot()
        kv_cache = getattr(model_runner, "kv_cache", None)
        target_storage_id = (
            ""
            if kv_cache is None
            else str(kv_cache.data_ptr())
        )
        before_forward_count = int(
            before_backend.get("real_draft_forward_count", 0)
        )
        after_forward_count = int(
            after_backend.get("real_draft_forward_count", 0)
        )
        evidence = {
            "real_draft_forward_count": max(
                0,
                after_forward_count - before_forward_count,
            ),
            "first_target_forward_count": method_names.count(
                "run_spec_first_target_and_proposal_batch"
            ),
            "tail_verification_forward_count": method_names.count(
                "run_spec_verify_batch"
            ),
            "extra_target_forward_count": (
                ordinary_decode_target_forwards
                if self.mode == "learned"
                else 0
            ),
            "proposal_kv_bytes": int(
                after_backend.get("proposal_kv_bytes", 0)
            ),
            "target_kv_bytes": int(
                memory.get("kv_capacity_bytes", 0)
            ),
            "proposal_kv_storage_id": str(
                after_backend.get(
                    "proposal_kv_storage_id",
                    "",
                )
            ),
            "target_kv_storage_id": target_storage_id,
            "proposal_kv_live_slots_before_release": max_live_slots,
            "proposal_kv_live_slots_after_release": int(
                _owned_proposal_entry_count(after_backend)
                if self.mode == "learned"
                else 0
            ),
        }
        if self.mode == "learned":
            evidence.update(
                _allocator_evidence_delta(
                    before_backend,
                    after_backend,
                )
            )
        return {
            "output_token_ids": output_token_ids,
            "observations": acceptance_rows,
            "timing_ms": {
                **_sum_timing_rows(observations),
                "generation": generation_ms,
            },
            "evidence": evidence,
        }

    def close(self):
        self.engine.exit()


def _default_engine_factory(mode, **kwargs):
    return _TinyVLLMEngineAdapter(mode, **kwargs)


def _case_payload(
    target_result,
    learned_result,
    prompts,
) -> dict:
    target_outputs = target_result["output_token_ids"]
    learned_outputs = learned_result["output_token_ids"]
    return {
        "prompts": [list(prompt) for prompt in prompts],
        "target_output_token_ids": target_outputs,
        "learned_output_token_ids": learned_outputs,
        "exact_output_parity": target_outputs == learned_outputs,
        "acceptance_rows": list(
            learned_result.get("observations", ())
        ),
        "timing_ms": {
            "target": dict(target_result.get("timing_ms", {})),
            "learned": dict(learned_result.get("timing_ms", {})),
        },
    }


def _merge_evidence(rows) -> dict:
    additive = (
        "real_draft_forward_count",
        "first_target_forward_count",
        "tail_verification_forward_count",
        "extra_target_forward_count",
        *_ALLOCATOR_COUNTER_NAMES,
    )
    merged = {
        name: sum(int(row.get(name, 0)) for row in rows)
        for name in additive
    }
    latest = rows[-1]
    for name in (
        "allocator_mode",
        "logical_entry_capacity",
        "gpu_slot_capacity",
    ):
        values = tuple(row.get(name) for row in rows)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"proposal KV allocator {name} mismatch across cases"
            )
    merged.update({
        "allocator_mode": latest.get("allocator_mode"),
        "logical_entry_capacity": latest.get(
            "logical_entry_capacity"
        ),
        "gpu_slot_capacity": latest.get("gpu_slot_capacity"),
        "proposal_kv_bytes": int(
            latest.get("proposal_kv_bytes", 0)
        ),
        "target_kv_bytes": int(latest.get("target_kv_bytes", 0)),
        "proposal_kv_storage_id": str(
            latest.get("proposal_kv_storage_id", "")
        ),
        "target_kv_storage_id": str(
            latest.get("target_kv_storage_id", "")
        ),
        "proposal_kv_live_slots_before_release": max(
            int(
                row.get(
                    "proposal_kv_live_slots_before_release",
                    0,
                )
            )
            for row in rows
        ),
        "proposal_kv_live_slots_after_release": int(
            latest.get(
                "proposal_kv_live_slots_after_release",
                -1,
            )
        ),
    })
    return merged


def _real_bidirectional_movement(evidence) -> bool:
    if not isinstance(evidence, dict):
        raise ValueError("proposal KV movement evidence is invalid")
    return all(
        (
            not isinstance(evidence.get(name), bool)
            and isinstance(evidence.get(name), int)
            and evidence[name] > 0
        )
        for name in (
            "h2d_entry_count",
            "h2d_bytes",
            "d2h_entry_count",
            "d2h_bytes",
        )
    )


def _workload_configuration(
    prompts,
    *,
    max_output_tokens,
) -> dict:
    batch_four_prompts = prompts[:4]
    max_model_len = (
        max(len(prompt) for prompt in batch_four_prompts)
        + max_output_tokens
        + MAX_PROPOSAL_TOKENS
        - 1
    )
    max_num_batched_tokens = max(
        max_model_len,
        sum(len(prompt) for prompt in batch_four_prompts),
        len(batch_four_prompts) * MAX_PROPOSAL_TOKENS,
    )
    proposal_slot_capacity = sum(
        len(prompt)
        + max_output_tokens
        + MAX_PROPOSAL_TOKENS
        for prompt in batch_four_prompts
    )
    return {
        "prompt_count": len(prompts),
        "batch_1_prompt_lengths": [
            len(prompt) for prompt in prompts[:1]
        ],
        "batch_4_prompt_lengths": [
            len(prompt) for prompt in batch_four_prompts
        ],
        "max_output_tokens": max_output_tokens,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "proposal_slot_capacity": proposal_slot_capacity,
    }


def _proposal_kv_configuration(
    workload,
    *,
    offload_enabled,
    gpu_slot_capacity,
    async_copy,
    batch_copy,
) -> dict:
    offload_enabled = _boolean(
        offload_enabled,
        "proposal KV offload enabled",
    )
    async_copy = _boolean(
        async_copy,
        "proposal KV async copy",
    )
    batch_copy = _boolean(
        batch_copy,
        "proposal KV batch copy",
    )
    logical_entry_capacity = _positive_integer(
        workload["proposal_slot_capacity"],
        "proposal KV logical entry capacity",
    )
    if not offload_enabled:
        if gpu_slot_capacity is not None:
            raise ValueError(
                "proposal KV GPU slot capacity must be omitted "
                "when offload is disabled"
            )
        return {
            "offload_enabled": False,
            "allocator_mode": "direct",
            "logical_entry_capacity": logical_entry_capacity,
            "gpu_slot_capacity": logical_entry_capacity,
            "cpu_backing_capacity": 0,
            "async_copy": async_copy,
            "batch_copy": batch_copy,
        }
    if gpu_slot_capacity is None:
        raise ValueError(
            "proposal KV GPU slot capacity is required "
            "when offload is enabled"
        )
    gpu_slot_capacity = _positive_integer(
        gpu_slot_capacity,
        "proposal KV GPU slot capacity",
    )
    if gpu_slot_capacity >= logical_entry_capacity:
        raise ValueError(
            "proposal KV GPU slot capacity must be smaller "
            "than logical entry capacity"
        )
    return {
        "offload_enabled": True,
        "allocator_mode": "residency",
        "logical_entry_capacity": logical_entry_capacity,
        "gpu_slot_capacity": gpu_slot_capacity,
        "cpu_backing_capacity": logical_entry_capacity,
        "async_copy": async_copy,
        "batch_copy": batch_copy,
    }


def run_preflight(
    *,
    target_model,
    draft_model,
    prompts,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    proposal_kv_offload_enabled=False,
    proposal_kv_gpu_slot_capacity=None,
    proposal_kv_async_copy=True,
    proposal_kv_batch_copy=True,
    identity_provider: Callable = _default_identity_provider,
) -> dict:
    normalized_prompts = _normalize_prompts(prompts)
    max_output_tokens = _positive_integer(
        max_output_tokens,
        "max_output_tokens",
    )
    workload = _workload_configuration(
        normalized_prompts,
        max_output_tokens=max_output_tokens,
    )
    proposal_kv = _proposal_kv_configuration(
        workload,
        offload_enabled=proposal_kv_offload_enabled,
        gpu_slot_capacity=proposal_kv_gpu_slot_capacity,
        async_copy=proposal_kv_async_copy,
        batch_copy=proposal_kv_batch_copy,
    )
    checkpoint_identity, tokenizer_contract = identity_provider(
        target_model, draft_model
    )
    for name in ("target", "draft"):
        if not checkpoint_identity.get(name, {}).get(
            "composite_sha256"
        ):
            raise ValueError(
                f"{name} checkpoint identity is missing"
            )
    if tokenizer_contract.get("compatible") is not True:
        raise ValueError("tokenizer compatibility is not established")
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "autoregressive_draft_tp1_preflight",
        "checkpoint_identity": checkpoint_identity,
        "tokenizer_contract": tokenizer_contract,
        "workload": workload,
        "proposal_kv": proposal_kv,
        "input_ready": True,
        "correctness_established": False,
        "performance_pass_criterion": False,
    }


def run_gate(
    *,
    target_model,
    draft_model,
    prompts,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    proposal_kv_offload_enabled=False,
    proposal_kv_gpu_slot_capacity=None,
    proposal_kv_async_copy=True,
    proposal_kv_batch_copy=True,
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
    proposal_kv = _proposal_kv_configuration(
        workload,
        offload_enabled=proposal_kv_offload_enabled,
        gpu_slot_capacity=proposal_kv_gpu_slot_capacity,
        async_copy=proposal_kv_async_copy,
        batch_copy=proposal_kv_batch_copy,
    )
    max_model_len = workload["max_model_len"]
    max_num_batched_tokens = workload[
        "max_num_batched_tokens"
    ]
    proposal_slot_capacity = workload[
        "proposal_slot_capacity"
    ]
    case_rows = (
        ("batch_1", normalized_prompts[:1]),
        ("batch_4", normalized_prompts[:4]),
    )
    target_results = {}
    target_engine = None
    try:
        target_engine = engine_factory(
            "target",
            target_model=target_model,
            draft_model=draft_model,
            max_num_seqs=4,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            proposal_slot_capacity=proposal_slot_capacity,
            proposal_kv_configuration=None,
        )
        target_model_load_ms = float(
            getattr(target_engine, "model_load_ms", 0.0)
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
            target_model=target_model,
            draft_model=draft_model,
            max_num_seqs=4,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            proposal_slot_capacity=proposal_slot_capacity,
            proposal_kv_configuration=proposal_kv,
        )
        learned_model_load_ms = float(
            getattr(learned_engine, "model_load_ms", 0.0)
        )
        for name, case_prompts in case_rows:
            learned_results[name] = learned_engine.run_case(
                case_prompts,
                max_output_tokens=max_output_tokens,
            )
    finally:
        if learned_engine is not None:
            learned_engine.close()

    cases = {}
    evidence_rows = []
    for name, case_prompts in case_rows:
        target_result = target_results[name]
        learned_result = learned_results[name]
        cases[name] = _case_payload(
            target_result,
            learned_result,
            case_prompts,
        )
        evidence_rows.append(learned_result["evidence"])
    evidence = _merge_evidence(evidence_rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "configuration": {
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "temperature": 0.0,
            "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        },
        "checkpoint_identity": checkpoint_identity,
        "tokenizer_contract": tokenizer_contract,
        "proposal_kv": proposal_kv,
        "model_load_ms": {
            "target": target_model_load_ms,
            "learned": learned_model_load_ms,
        },
        "cases": cases,
        "evidence": evidence,
        "proposal_kv_offload_enabled": proposal_kv[
            "offload_enabled"
        ],
        "real_proposal_kv_bidirectional_movement": (
            _real_bidirectional_movement(evidence)
        ),
        "performance_pass_criterion": False,
        "gate_pass": True,
    }
    validate_gate_payload(payload)
    return payload


def validate_gate_payload(payload) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("gate schema version mismatch")
    if payload.get("gate") != GATE_NAME:
        raise ValueError("gate name mismatch")
    expected_configuration = {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "temperature": 0.0,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
    }
    if payload.get("configuration") != expected_configuration:
        raise ValueError("gate configuration mismatch")
    proposal_kv = payload.get("proposal_kv")
    if not isinstance(proposal_kv, dict):
        raise ValueError("proposal KV configuration is missing")
    required_proposal_kv = {
        "offload_enabled",
        "allocator_mode",
        "logical_entry_capacity",
        "gpu_slot_capacity",
        "cpu_backing_capacity",
        "async_copy",
        "batch_copy",
    }
    if set(proposal_kv) != required_proposal_kv:
        raise ValueError("proposal KV configuration is invalid")
    offload_enabled = proposal_kv.get("offload_enabled")
    if not isinstance(offload_enabled, bool):
        raise ValueError("proposal KV offload flag is invalid")
    if proposal_kv.get("allocator_mode") != (
        "residency" if offload_enabled else "direct"
    ):
        raise ValueError("proposal KV allocator mode is invalid")
    logical_capacity = _positive_integer(
        proposal_kv.get("logical_entry_capacity"),
        "proposal KV logical entry capacity",
    )
    gpu_capacity = _positive_integer(
        proposal_kv.get("gpu_slot_capacity"),
        "proposal KV GPU slot capacity",
    )
    cpu_capacity = _integer(
        proposal_kv.get("cpu_backing_capacity"),
        "proposal KV CPU backing capacity",
    )
    _boolean(
        proposal_kv.get("async_copy"),
        "proposal KV async copy",
    )
    _boolean(
        proposal_kv.get("batch_copy"),
        "proposal KV batch copy",
    )
    if offload_enabled:
        if (
            cpu_capacity != logical_capacity
            or logical_capacity <= gpu_capacity
        ):
            raise ValueError(
                "proposal KV offload capacity tuple is invalid"
            )
    elif (
        cpu_capacity != 0
        or logical_capacity != gpu_capacity
    ):
        raise ValueError(
            "proposal KV direct capacity tuple is invalid"
        )
    if payload.get(
        "proposal_kv_offload_enabled"
    ) is not offload_enabled:
        raise ValueError("proposal KV offload classification mismatch")
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
        if not row.get("acceptance_rows"):
            raise ValueError(f"{name} acceptance rows are missing")
    evidence = payload.get("evidence", {})
    if int(evidence.get("real_draft_forward_count", 0)) <= 0:
        raise ValueError("real draft forward evidence is missing")
    if int(evidence.get("extra_target_forward_count", -1)) != 0:
        raise ValueError("extra target forward count is nonzero")
    if int(
        evidence.get(
            "proposal_kv_live_slots_after_release",
            -1,
        )
    ) != 0:
        raise ValueError("proposal KV leak remains after release")
    if int(evidence.get("proposal_kv_bytes", 0)) <= 0:
        raise ValueError("proposal KV byte evidence is missing")
    if int(evidence.get("target_kv_bytes", 0)) <= 0:
        raise ValueError("target KV byte evidence is missing")
    proposal_storage = evidence.get("proposal_kv_storage_id")
    target_storage = evidence.get("target_kv_storage_id")
    if (
        not proposal_storage
        or not target_storage
        or proposal_storage == target_storage
    ):
        raise ValueError(
            "proposal and target KV require distinct storage"
        )
    if evidence.get("allocator_mode") != proposal_kv[
        "allocator_mode"
    ]:
        raise ValueError("proposal KV allocator mode mismatch")
    if evidence.get("logical_entry_capacity") != logical_capacity:
        raise ValueError(
            "proposal KV logical entry capacity mismatch"
        )
    if evidence.get("gpu_slot_capacity") != gpu_capacity:
        raise ValueError("proposal KV GPU slot capacity mismatch")
    for name in _ALLOCATOR_COUNTER_NAMES:
        value = evidence.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"proposal KV allocator {name} evidence is invalid"
            )
    if evidence["accepted_entry_copy_count"] != 0:
        raise ValueError("accepted proposal KV copy is nonzero")
    if evidence["accepted_entry_replay_count"] != 0:
        raise ValueError("accepted proposal KV replay is nonzero")
    if evidence["accepted_entry_rematerialization_count"] != 0:
        raise ValueError(
            "accepted proposal KV rematerialization is nonzero"
        )
    if not offload_enabled and any(
        evidence[name] != 0
        for name in (
            "h2d_operation_count",
            "h2d_entry_count",
            "h2d_bytes",
            "d2h_operation_count",
            "d2h_entry_count",
            "d2h_bytes",
        )
    ):
        raise ValueError(
            "direct proposal KV movement must remain zero"
        )
    movement = _real_bidirectional_movement(evidence)
    if payload.get(
        "real_proposal_kv_bidirectional_movement"
    ) is not movement:
        raise ValueError(
            "proposal KV movement classification mismatch"
        )
    if payload.get("performance_pass_criterion") is not False:
        raise ValueError(
            "performance pass criterion must remain false"
        )


def _default_tokenizer_loader(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )


def load_prompt_file(
    path,
    *,
    target_model,
    tokenizer_loader=_default_tokenizer_loader,
) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "prompts" in payload:
            payload = payload["prompts"]
        elif "targets" in payload:
            payload = payload["targets"]
        else:
            raise ValueError(
                "prompt file must contain prompts or targets"
            )
    if not isinstance(payload, list):
        raise ValueError("prompt file rows must be a list")
    tokenizer = None
    rows = []
    for row in payload:
        prompt = row
        if isinstance(row, dict):
            if "prompt_token_ids" in row:
                prompt = row["prompt_token_ids"]
            elif "token_ids" in row:
                prompt = row["token_ids"]
            elif "prompt" in row:
                prompt = row["prompt"]
            elif "text" in row:
                prompt = row["text"]
            else:
                raise ValueError(
                    "prompt row must contain token IDs or text"
                )
        if isinstance(prompt, str):
            if tokenizer is None:
                tokenizer = tokenizer_loader(target_model)
            encode = getattr(tokenizer, "encode", None)
            if not callable(encode):
                raise ValueError(
                    "target tokenizer must expose callable encode"
                )
            prompt = encode(prompt)
        rows.append(prompt)
    return _normalize_prompts(rows)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )
    parser.add_argument(
        "--proposal-kv-offload",
        action="store_true",
    )
    parser.add_argument(
        "--proposal-kv-gpu-slot-capacity",
        type=int,
    )
    parser.add_argument(
        "--proposal-kv-sync-copy",
        action="store_true",
    )
    parser.add_argument(
        "--proposal-kv-no-batch-copy",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    output_path = Path(args.output)
    try:
        prompts = load_prompt_file(
            args.prompt_file,
            target_model=args.target_model,
        )
        run = run_preflight if args.preflight_only else run_gate
        payload = run(
            target_model=args.target_model,
            draft_model=args.draft_model,
            prompts=prompts,
            max_output_tokens=args.max_output_tokens,
            proposal_kv_offload_enabled=args.proposal_kv_offload,
            proposal_kv_gpu_slot_capacity=(
                args.proposal_kv_gpu_slot_capacity
            ),
            proposal_kv_async_copy=not args.proposal_kv_sync_copy,
            proposal_kv_batch_copy=(
                not args.proposal_kv_no_batch_copy
            ),
        )
    except Exception as error:
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "gate": GATE_NAME,
                    "gate_pass": False,
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return 1
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
