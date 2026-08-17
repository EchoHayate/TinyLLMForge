from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time


SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
DEFAULT_BATCH_SIZES = (1, 4)
DEFAULT_QUERY_LENGTHS = (1, 3)
DEFAULT_PAGE_TABLE_WIDTHS = (1, 2)
PERFORMANCE_MEASUREMENT_REPEATS = 5
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py",
    "tinyvllm/engine/speculative_model_runner.py",
    "tinyvllm/speculative/adapter.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tools/run_spec_verify_cuda_graph_gate_remote.py",
    "tools/spec_verify_cuda_graph_smoke.py",
    "tools/verify_spec_verify_cuda_graph_gate.py",
)


def _positive_int(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def prompt_length_for_family(
    *,
    query_len: int,
    page_table_width: int,
    block_size: int,
) -> int:
    query_len = _positive_int(query_len, "query_len")
    page_table_width = _positive_int(
        page_table_width,
        "page_table_width",
    )
    block_size = _positive_int(block_size, "block_size")
    terminal_context_len = (
        block_size
        if page_table_width == 1
        else (page_table_width - 1) * block_size + 1
    )
    prompt_len = terminal_context_len - query_len
    if prompt_len <= 0:
        raise ValueError(
            "query_len is too large for page_table_width"
        )
    return prompt_len


def build_prompt_token_batch(
    *,
    seed_token_ids: tuple[int, ...],
    batch_size: int,
    query_len: int,
    page_table_width: int,
    block_size: int,
) -> tuple[tuple[int, ...], ...]:
    if (
        not isinstance(seed_token_ids, tuple)
        or not seed_token_ids
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in seed_token_ids
        )
    ):
        raise ValueError(
            "seed_token_ids must be a non-empty integer tuple"
        )
    batch_size = _positive_int(batch_size, "batch_size")
    if len(set(seed_token_ids)) < batch_size:
        raise ValueError(
            "seed_token_ids must contain one distinct token per row"
        )
    prompt_len = prompt_length_for_family(
        query_len=query_len,
        page_table_width=page_table_width,
        block_size=block_size,
    )
    repeated = (
        seed_token_ids
        * ((prompt_len + len(seed_token_ids) - 1)
           // len(seed_token_ids))
    )[:prompt_len]
    prompts = []
    distinct_tokens = tuple(dict.fromkeys(seed_token_ids))
    for row_index in range(batch_size):
        prompt = list(repeated)
        prompt[-1] = distinct_tokens[row_index]
        prompts.append(tuple(prompt))
    return tuple(prompts)


class OracleMismatchDraftAdapter:
    def __init__(
        self,
        *,
        oracle_tokens_by_prompt: dict[
            tuple[int, ...],
            tuple[int, ...],
        ],
        query_len: int,
        vocab_size: int,
        adapter_types: tuple[type, type] | None = None,
    ):
        query_len = _positive_int(query_len, "query_len")
        vocab_size = _positive_int(vocab_size, "vocab_size")
        if (
            not isinstance(oracle_tokens_by_prompt, dict)
            or not oracle_tokens_by_prompt
        ):
            raise ValueError(
                "oracle_tokens_by_prompt must be non-empty"
            )
        normalized = {}
        for prompt, tokens in oracle_tokens_by_prompt.items():
            if (
                not isinstance(prompt, tuple)
                or not prompt
                or not isinstance(tokens, tuple)
                or len(tokens) < query_len
            ):
                raise ValueError(
                    "oracle prompt rows are invalid"
                )
            normalized[prompt] = tokens
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
        self._oracle_tokens_by_prompt = normalized
        self._query_len = query_len
        self._vocab_size = vocab_size
        self._proposal_type = proposal_type
        self._capabilities = capability_type(
            source_type="oracle_mismatch_fixture",
            supports_batch=True,
            requires_target_hidden=False,
            requires_target_logits=False,
            max_proposal_tokens=query_len,
        )

    @property
    def capabilities(self):
        return self._capabilities

    def propose_batch(self, contexts: tuple) -> tuple:
        proposals = []
        for context in contexts:
            prompt = tuple(context.token_ids)
            oracle = self._oracle_tokens_by_prompt.get(prompt)
            if oracle is None:
                raise RuntimeError(
                    "oracle fixture prompt identity drifted"
                )
            if context.max_proposal_tokens < self._query_len:
                raise RuntimeError(
                    "oracle fixture proposal budget is too small"
                )
            if context.first_target_token != oracle[0]:
                raise RuntimeError(
                    "oracle fixture first target drifted"
                )
            accepted_prefix_length = self._query_len - 1
            wrong_token = (
                int(oracle[accepted_prefix_length]) + 1
            ) % self._vocab_size
            if wrong_token == oracle[accepted_prefix_length]:
                wrong_token = (
                    wrong_token + 1
                ) % self._vocab_size
            token_ids = (
                tuple(oracle[:accepted_prefix_length])
                + (wrong_token,)
            )
            proposals.append(
                self._proposal_type(
                    sequence_id=context.sequence_id,
                    token_ids=token_ids,
                    source_type="oracle_mismatch_fixture",
                    metadata={
                        "accepted_prefix_length": (
                            accepted_prefix_length
                        ),
                        "query_len": self._query_len,
                    },
                )
            )
        return tuple(proposals)


def _block_ids(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    normalized = tuple(
        _non_negative_int(block_id, name)
        for block_id in value
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def build_transaction_results(
    rows: tuple[dict, ...],
    *,
    accepted_prefix_kv_parity: bool,
) -> dict:
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("transaction rows must be non-empty")
    if not isinstance(accepted_prefix_kv_parity, bool):
        raise ValueError(
            "accepted_prefix_kv_parity must be a bool"
        )
    states = []
    materialized_counts = []
    committed_counts = []
    rejected_counts = []
    unused_block_ids = []
    released_block_ids = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "transaction row must be a dictionary"
            )
        state = row.get("transaction_state")
        if state != "committed":
            raise ValueError(
                "transaction state must be committed"
            )
        materialized_count = _positive_int(
            row.get("materialized_token_count"),
            "materialized_token_count",
        )
        accepted_count = _non_negative_int(
            row.get("accepted_token_count"),
            "accepted_token_count",
        )
        committed_count = max(0, accepted_count - 1)
        rejected_count = materialized_count - committed_count
        if rejected_count <= 0:
            raise ValueError(
                "transaction must contain a rejected "
                "materialized suffix"
            )
        unused = _block_ids(
            row.get("unused_block_ids"),
            "unused_block_ids",
        )
        released = _block_ids(
            row.get("released_block_ids"),
            "released_block_ids",
        )
        if released != unused:
            raise ValueError(
                "released block IDs must match unused block IDs"
            )
        states.append(state)
        materialized_counts.append(materialized_count)
        committed_counts.append(committed_count)
        rejected_counts.append(rejected_count)
        unused_block_ids.extend(unused)
        released_block_ids.extend(released)
    if len(set(unused_block_ids)) != len(unused_block_ids):
        raise ValueError(
            "unused block IDs must be globally unique"
        )
    if len(set(released_block_ids)) != len(released_block_ids):
        raise ValueError(
            "released block IDs must be globally unique"
        )
    return {
        "accepted_prefix_kv_parity": (
            accepted_prefix_kv_parity
        ),
        "rejected_suffix_released": True,
        "transaction_states": states,
        "materialized_token_counts": materialized_counts,
        "committed_materialized_token_counts": committed_counts,
        "rejected_materialized_token_counts": rejected_counts,
        "unused_block_ids": sorted(unused_block_ids),
        "released_block_ids": sorted(released_block_ids),
        "all_unused_blocks_released": True,
    }


def _aggregate_request_metrics(rows: tuple[dict, ...]) -> dict:
    if not isinstance(rows, tuple) or not rows:
        raise ValueError(
            "request metric rows must be a non-empty tuple"
        )
    elapsed_ns = 0
    ttft_ns = 0
    output_token_count = 0
    gpu_allocated_bytes = 0
    gpu_reserved_bytes = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "request metric row must be a dictionary"
            )
        row_elapsed = _positive_int(
            row.get("elapsed_ns"),
            "elapsed_ns",
        )
        row_ttft = _positive_int(
            row.get("ttft_ns"),
            "ttft_ns",
        )
        row_tokens = _positive_int(
            row.get("output_token_count"),
            "output_token_count",
        )
        row_allocated = _non_negative_int(
            row.get("gpu_allocated_bytes"),
            "gpu_allocated_bytes",
        )
        row_reserved = _non_negative_int(
            row.get("gpu_reserved_bytes"),
            "gpu_reserved_bytes",
        )
        if row_reserved < row_allocated:
            raise ValueError(
                "GPU reserved bytes must cover allocated bytes"
            )
        elapsed_ns += row_elapsed
        ttft_ns += row_ttft
        output_token_count += row_tokens
        gpu_allocated_bytes = max(
            gpu_allocated_bytes,
            row_allocated,
        )
        gpu_reserved_bytes = max(
            gpu_reserved_bytes,
            row_reserved,
        )
    return {
        "ttft_ns": max(1, ttft_ns // len(rows)),
        "tpot_ns": max(1, elapsed_ns // output_token_count),
        "throughput_tokens_per_second": (
            output_token_count * 1_000_000_000 / elapsed_ns
        ),
        "gpu_allocated_bytes": gpu_allocated_bytes,
        "gpu_reserved_bytes": gpu_reserved_bytes,
    }


def build_performance_evidence(
    rows: tuple[dict, ...],
    *,
    cache_counts: dict,
) -> dict:
    if not isinstance(rows, tuple) or not rows:
        raise ValueError(
            "performance rows must be a non-empty tuple"
        )
    family_ids = []
    prompt_lengths = {}
    proposal_distribution = {}
    batch_distribution = {}
    eager_metrics = []
    warmed_metrics = []
    mixed_metrics = []
    warmed_latencies = {}
    capture_durations = {}
    capture_allocated = {}
    capture_reserved = {}
    proposed_tokens = 0
    accepted_tokens = 0
    warmed_measurement_count = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "performance row must be a dictionary"
            )
        family_id = row.get("family_id")
        if (
            not isinstance(family_id, str)
            or not family_id
            or family_id in family_ids
        ):
            raise ValueError(
                "performance family IDs must be unique strings"
            )
        batch_size = _positive_int(
            row.get("batch_size"),
            "batch_size",
        )
        query_len = _positive_int(
            row.get("query_len"),
            "query_len",
        )
        prompt_length = _positive_int(
            row.get("prompt_length"),
            "prompt_length",
        )
        accepted = _non_negative_int(
            row.get("accepted_draft_tokens"),
            "accepted_draft_tokens",
        )
        family_proposed = batch_size * query_len
        if accepted > family_proposed:
            raise ValueError(
                "accepted draft tokens exceed proposals"
            )
        family_ids.append(family_id)
        prompt_lengths[family_id] = prompt_length
        proposal_distribution[str(query_len)] = (
            proposal_distribution.get(str(query_len), 0)
            + family_proposed
        )
        batch_distribution[str(batch_size)] = (
            batch_distribution.get(str(batch_size), 0)
            + 1
        )
        eager = row.get("eager_request_metrics")
        cold = row.get("cold_request_metrics")
        captured = row.get("capture_request_metrics")
        warmed = row.get("warmed_request_metrics")
        warmed_rows = row.get(
            "warmed_request_metrics_rows",
            (warmed,),
        )
        if (
            not isinstance(warmed_rows, tuple)
            or not warmed_rows
        ):
            raise ValueError(
                "warmed request metric rows must be non-empty"
            )
        warmed_latency_rows = row.get(
            "warmed_verifier_latency_ns_rows",
            (
                row.get("warmed_verifier_latency_ns"),
            ),
        )
        if (
            not isinstance(warmed_latency_rows, tuple)
            or len(warmed_latency_rows) != len(warmed_rows)
        ):
            raise ValueError(
                "warmed latency rows must match measurements"
            )
        eager_metrics.append(eager)
        warmed_metrics.extend(warmed_rows)
        mixed_metrics.extend((cold, captured, *warmed_rows))
        warmed_latencies[family_id] = [
            _positive_int(
                latency,
                "warmed_verifier_latency_ns",
            )
            for latency in warmed_latency_rows
        ]
        warmed_measurement_count += len(warmed_rows)
        capture_durations[family_id] = _positive_int(
            row.get("capture_duration_ns"),
            "capture_duration_ns",
        )
        capture_allocated[family_id] = _non_negative_int(
            row.get("capture_allocated_delta_bytes"),
            "capture_allocated_delta_bytes",
        )
        capture_reserved[family_id] = _non_negative_int(
            row.get("capture_reserved_delta_bytes"),
            "capture_reserved_delta_bytes",
        )
        proposed_tokens += family_proposed
        accepted_tokens += accepted

    cache_counts = dict(cache_counts)
    for name in (
        "hits",
        "misses",
        "evictions",
        "quarantines",
    ):
        cache_counts[name] = _non_negative_int(
            cache_counts.get(name),
            f"cache_{name}",
        )
    warmed_aggregate = _aggregate_request_metrics(
        tuple(warmed_metrics)
    )
    mixed_aggregate = _aggregate_request_metrics(
        tuple(mixed_metrics)
    )
    return {
        "warmup_count": 2,
        "measurement_count": warmed_measurement_count,
        "prompt_lengths": prompt_lengths,
        "proposal_length_distribution": (
            proposal_distribution
        ),
        "batch_distribution": batch_distribution,
        "eager_baseline": _aggregate_request_metrics(
            tuple(eager_metrics)
        ),
        "warmed_exact_graph_hits": {
            "measurement_count": warmed_measurement_count,
            "latency_ns_by_family": warmed_latencies,
            "gpu_allocated_bytes": (
                warmed_aggregate["gpu_allocated_bytes"]
            ),
            "gpu_reserved_bytes": (
                warmed_aggregate["gpu_reserved_bytes"]
            ),
        },
        "mixed_hit_rate": {
            "measurement_count": len(mixed_metrics),
            "hit_count": warmed_measurement_count,
            "miss_count": len(rows) * 2,
            "end_to_end_tpot_ns": mixed_aggregate["tpot_ns"],
            "ttft_ns": mixed_aggregate["ttft_ns"],
            "throughput_tokens_per_second": (
                mixed_aggregate[
                    "throughput_tokens_per_second"
                ]
            ),
        },
        "capture": {
            "duration_ns_by_family": capture_durations,
            "allocated_delta_bytes_by_family": (
                capture_allocated
            ),
            "reserved_delta_bytes_by_family": (
                capture_reserved
            ),
        },
        "cache_counts": cache_counts,
        "acceptance": {
            "proposed_tokens": proposed_tokens,
            "accepted_draft_tokens": accepted_tokens,
            "acceptance_rate": (
                accepted_tokens / proposed_tokens
            ),
        },
    }


def _sha256_tensor(tensor) -> str:
    import torch

    contiguous = (
        tensor.detach().contiguous().view(torch.uint8)
    )
    return hashlib.sha256(
        contiguous.cpu().numpy().tobytes()
    ).hexdigest()


def _sha256_kv_rows(runner, plans: tuple) -> str:
    import torch

    digest = hashlib.sha256()
    for row_index, plan in enumerate(sorted(
        plans,
        key=lambda row: int(row.sequence_id),
    )):
        digest.update(
            f"row:{row_index}\n".encode()
        )
        start = int(plan.transaction.original_num_tokens)
        end = int(plan.materialized_end)
        for position in range(start, end):
            block_id = int(
                plan.sequence.block_table[
                    position // runner.block_size
                ]
            )
            offset = position % runner.block_size
            row = runner.kv_cache[
                :,
                :,
                block_id,
                offset,
            ].detach().contiguous()
            digest.update(
                row.view(torch.uint8).cpu().numpy().tobytes()
            )
    return digest.hexdigest()


class FamilyRunRecorder:
    def __init__(self, engine):
        self.engine = engine
        self.runner = engine.model_runner
        self.manager = engine.scheduler.block_manager
        self._original_run_model = self.runner.run_model
        self._original_prepare = (
            self.manager.prepare_speculative_kv_commit
        )
        self._original_commit_batch = (
            self.manager.commit_speculative_kv_commit_batch
        )
        self._installed = False
        self.reset()

    def reset(self) -> None:
        self.logits = []
        self.latencies_ns = []
        self.dispatch_events = []
        self.plans = {}
        self.transaction_rows = []
        self.accepted_prefix_kv_sha256 = None

    def install(self) -> None:
        if self._installed:
            return

        def run_model(*args, **kwargs):
            is_spec_verify = (
                kwargs.get("execution_mode") == "spec_verify"
            )
            started_ns = time.perf_counter_ns()
            result = self._original_run_model(*args, **kwargs)
            if is_spec_verify:
                import torch

                torch.cuda.synchronize()
                self.latencies_ns.append(
                    time.perf_counter_ns() - started_ns
                )
                self.logits.append(
                    result.detach().float().cpu().contiguous()
                )
                self.dispatch_events.append(
                    copy.deepcopy(
                        self.runner
                        .last_spec_verify_cuda_graph_dispatch_event
                    )
                )
            return result

        def prepare(transaction, sequence, accepted_tokens):
            plan = self._original_prepare(
                transaction,
                sequence,
                accepted_tokens,
            )
            self.plans[int(plan.sequence_id)] = plan
            return plan

        def commit_batch(plans):
            result = self._original_commit_batch(plans)
            rows = []
            for plan in sorted(
                plans,
                key=lambda row: int(row.sequence_id),
            ):
                released = tuple(
                    int(block_id)
                    for block_id in plan.unused_block_ids
                    if (
                        int(block_id)
                        not in self.manager.used_block_ids
                        and self.manager.blocks[
                            int(block_id)
                        ].ref_count == 0
                    )
                )
                rows.append({
                    "transaction_state": (
                        plan.transaction.state
                    ),
                    "materialized_token_count": int(
                        plan.transaction.materialized_token_count
                    ),
                    "accepted_token_count": len(
                        plan.accepted_tokens
                    ),
                    "unused_block_ids": tuple(
                        int(block_id)
                        for block_id in plan.unused_block_ids
                    ),
                    "released_block_ids": released,
                })
            self.transaction_rows = rows
            self.accepted_prefix_kv_sha256 = (
                _sha256_kv_rows(self.runner, tuple(plans))
            )
            return result

        self.runner.run_model = run_model
        self.manager.prepare_speculative_kv_commit = prepare
        self.manager.commit_speculative_kv_commit_batch = (
            commit_batch
        )
        self._installed = True

    def close(self) -> None:
        if not self._installed:
            return
        self.runner.run_model = self._original_run_model
        self.manager.prepare_speculative_kv_commit = (
            self._original_prepare
        )
        self.manager.commit_speculative_kv_commit_batch = (
            self._original_commit_batch
        )
        self._installed = False

    def snapshot(
        self,
        observation: dict,
        outputs: list,
        request_metrics: dict,
    ) -> dict:
        if len(self.logits) != 1:
            raise RuntimeError(
                "family run must execute one spec-verify target forward"
            )
        if len(self.dispatch_events) != 1:
            raise RuntimeError(
                "family run must publish one dispatch event"
            )
        if not self.transaction_rows:
            raise RuntimeError(
                "family run did not commit transactions"
            )
        event = self.dispatch_events[0]
        selected_ids = tuple(
            int(sequence_id)
            for sequence_id in observation[
                "speculative_selected_seq_ids"
            ]
        )
        accepted_mapping = observation[
            "speculative_accepted_draft_token_counts"
        ]
        accepted_lengths = [
            int(accepted_mapping[sequence_id])
            for sequence_id in selected_ids
        ]
        logits = self.logits[0]
        return {
            "logits_sha256": _sha256_tensor(logits),
            "target_tokens": [
                int(token_id)
                for token_id in logits.argmax(dim=-1).tolist()
            ],
            "accepted_lengths": accepted_lengths,
            "final_tokens": [
                [int(token_id) for token_id in row]
                for row in outputs
            ],
            "accepted_prefix_kv_sha256": (
                self.accepted_prefix_kv_sha256
            ),
            "target_forward_count": 1,
            "latency_ns": int(self.latencies_ns[0]),
            "dispatch_event": event,
            "transaction_rows": copy.deepcopy(
                self.transaction_rows
            ),
            "request_metrics": dict(request_metrics),
        }


def _run_request_batch(
    engine,
    *,
    prompts: tuple[tuple[int, ...], ...],
    max_tokens: int,
    clock_ns=time.perf_counter_ns,
    sampling_params_factory=None,
) -> tuple[list[list[int]], dict, dict]:
    if sampling_params_factory is None:
        from tinyvllm import SamplingParams

        sampling_params_factory = SamplingParams
    sampling_params = sampling_params_factory(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    started_ns = clock_ns()
    first_token_ns = None
    for prompt in prompts:
        engine.add_request(list(prompt), sampling_params)
    outputs_by_id = {}
    speculative_observations = []
    while not engine.is_finished():
        outputs, _ = engine.step()
        observation = getattr(
            engine,
            "last_step_observation",
            None,
        )
        if (
            first_token_ns is None
            and isinstance(observation, dict)
            and any(
                token_ids
                for token_ids in observation.get(
                    "new_completion_tokens_by_seq",
                    {},
                ).values()
            )
        ):
            first_token_ns = clock_ns()
        if (
            isinstance(observation, dict)
            and observation.get(
                "speculative_selected_seq_ids"
            )
        ):
            speculative_observations.append(
                copy.deepcopy(observation)
            )
        for sequence_id, token_ids in outputs:
            outputs_by_id[int(sequence_id)] = [
                int(token_id)
                for token_id in token_ids
            ]
    outputs = [
        outputs_by_id[sequence_id]
        for sequence_id in sorted(outputs_by_id)
    ]
    if len(outputs) != len(prompts):
        raise RuntimeError(
            "engine did not return one output per prompt"
        )
    ended_ns = clock_ns()
    if first_token_ns is None:
        raise RuntimeError(
            "engine did not publish first-token timing"
        )
    if len(speculative_observations) > 1:
        raise RuntimeError(
            "family run produced multiple speculative steps"
        )
    observation = (
        speculative_observations[0]
        if speculative_observations
        else {}
    )
    memory = engine.model_runner.memory_snapshot()
    return outputs, observation, {
        "elapsed_ns": _positive_int(
            ended_ns - started_ns,
            "elapsed_ns",
        ),
        "ttft_ns": _positive_int(
            first_token_ns - started_ns,
            "ttft_ns",
        ),
        "output_token_count": _positive_int(
            sum(len(row) for row in outputs),
            "output_token_count",
        ),
        "gpu_allocated_bytes": _non_negative_int(
            memory.get("cuda_allocated_bytes"),
            "cuda_allocated_bytes",
        ),
        "gpu_reserved_bytes": _non_negative_int(
            memory.get("cuda_reserved_bytes"),
            "cuda_reserved_bytes",
        ),
    }


def _seed_token_ids(tokenizer, batch_size: int) -> tuple[int, ...]:
    encoded = tokenizer.encode(
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda mu"
    )
    eos_token_id = tokenizer.eos_token_id
    unique = []
    for token_id in encoded:
        token_id = int(token_id)
        if token_id == eos_token_id or token_id in unique:
            continue
        unique.append(token_id)
    if len(unique) < batch_size:
        raise RuntimeError(
            "tokenizer did not provide enough distinct seed tokens"
        )
    return tuple(unique)


def _engine_kwargs(
    *,
    context_length: int,
    batch_sizes: tuple[int, ...],
    query_len: int,
    graph_enabled: bool,
) -> dict:
    return {
        "tensor_parallel_size": 1,
        "enforce_eager": False,
        "max_model_len": context_length,
        "max_num_seqs": max(batch_sizes),
        "kv_offload_mvp0": False,
        "spec_verify_cuda_graphs": graph_enabled,
        "spec_verify_cuda_graph_batch_allowlist": batch_sizes,
        "spec_verify_cuda_graph_query_len_allowlist": (
            query_len,
        ),
        "spec_verify_cuda_graph_min_observations": 2,
        "spec_verify_cuda_graph_max_entries": 8,
    }


def _family_id(
    batch_size: int,
    query_len: int,
    width: int,
) -> str:
    return f"b{batch_size}-q{query_len}-w{width}"


def _run_oracle_cases(
    engine,
    *,
    prompts_by_family: dict[
        tuple[int, int],
        tuple[tuple[int, ...], ...],
    ],
    query_len: int,
) -> dict[tuple[int, ...], tuple[int, ...]]:
    oracle = {}
    for prompts in prompts_by_family.values():
        outputs, observation, _ = _run_request_batch(
            engine,
            prompts=prompts,
            max_tokens=query_len,
        )
        if observation:
            raise RuntimeError(
                "oracle run unexpectedly used speculative execution"
            )
        for prompt, output in zip(prompts, outputs):
            if len(output) != query_len:
                raise RuntimeError(
                    "oracle output length does not match exact Q"
                )
            existing = oracle.get(prompt)
            output_tuple = tuple(output)
            if (
                existing is not None
                and existing != output_tuple
            ):
                raise RuntimeError(
                    "oracle output is not deterministic"
                )
            oracle[prompt] = output_tuple
    return oracle


def _run_recorded_family(
    engine,
    recorder: FamilyRunRecorder,
    *,
    prompts: tuple[tuple[int, ...], ...],
    query_len: int,
) -> dict:
    recorder.reset()
    outputs, observation, request_metrics = _run_request_batch(
        engine,
        prompts=prompts,
        max_tokens=query_len,
    )
    if not observation:
        raise RuntimeError(
            "speculative family did not execute"
        )
    return recorder.snapshot(
        observation,
        outputs,
        request_metrics,
    )


class _ReplayFailureGraph:
    def __init__(self, original):
        self.original = original
        self.replay_calls = 0

    def replay(self):
        self.replay_calls += 1
        raise RuntimeError("injected graph replay failure")


def inject_replay_failure(runner, entry) -> dict:
    import torch

    from tinyvllm.engine.spec_verify_exact_cuda_graph_cache import (
        SpecVerifyGraphReplayError,
    )
    from tinyvllm.utils.context import reset_context, set_context

    wrapper = _ReplayFailureGraph(entry.graph)
    entry.graph = wrapper
    original_eager = runner._run_eager_logits
    eager_retry_count = 0

    def count_eager(*args, **kwargs):
        nonlocal eager_retry_count
        eager_retry_count += 1
        return original_eager(*args, **kwargs)

    runner._run_eager_logits = count_eager
    runner._spec_verify_transaction_authorized = True
    error_propagated = False
    try:
        tensors = entry.tensors
        set_context(
            mode="spec_verify",
            slot_mapping=tensors["slot_mapping"],
            context_lens=tensors["context_lens"],
            block_tables=tensors["block_tables"],
            spec_verify_query_lens=(
                (entry.identity.query_len,)
                * entry.identity.active_batch_size
            ),
            flash_attn_num_splits=(
                entry.identity.flash_attn_num_splits
            ),
        )
        try:
            runner.run_model(
                tensors["input_ids"],
                tensors["positions"],
                False,
                execution_mode="spec_verify",
            )
        except SpecVerifyGraphReplayError:
            error_propagated = True
        torch.cuda.synchronize()
    finally:
        reset_context()
        runner._run_eager_logits = original_eager
        runner._spec_verify_transaction_authorized = False
    summary = runner.spec_verify_exact_cuda_graph_cache.summary()
    quarantined = dict(summary["quarantined"])
    reason = quarantined.get(entry.identity_sha256)
    runner.spec_verify_exact_cuda_graph_cache.quarantine(
        entry.identity,
        "replay_failed",
    )
    stable_reason = dict(
        runner.spec_verify_exact_cuda_graph_cache
        .summary()["quarantined"]
    ).get(entry.identity_sha256)
    return {
        "graph_replay_count": wrapper.replay_calls,
        "eager_retry_count": eager_retry_count,
        "error_propagated": error_propagated,
        "quarantine_reason": reason,
        "stable_quarantine_reason": (
            reason == "replay_failed"
            and stable_reason == reason
        ),
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    result = {}
    for relative_path in SOURCE_FILES:
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"source file is missing: {relative_path}"
            )
        result[relative_path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def _source_sha256(source_files: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            source_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_live_gate(
    *,
    model_path: str,
    output_path: Path,
    context_length: int,
    batch_sizes: tuple[int, ...],
    query_lengths: tuple[int, ...],
    page_table_widths: tuple[int, ...],
    measure_performance: bool = False,
    engine_factory=None,
) -> dict:
    import torch

    from tinyvllm.engine.speculative_runtime import (
        EngineSpeculativeRuntime,
    )

    if engine_factory is None:
        from tinyvllm import LLM

        engine_factory = LLM
    repo_root = Path(__file__).resolve().parents[1]
    source_files = _source_hashes(repo_root)
    source_sha256 = _source_sha256(source_files)
    os.environ["TINYVLLM_SOURCE_SHA256"] = source_sha256
    families = []
    performance_rows = []
    performance_cache_counts = {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "quarantines": 0,
    }
    for query_len in query_lengths:
        baseline_engine = engine_factory(
            model_path,
            **_engine_kwargs(
                context_length=context_length,
                batch_sizes=batch_sizes,
                query_len=query_len,
                graph_enabled=False,
            ),
        )
        graph_engine = None
        baseline_recorder = None
        graph_recorder = None
        try:
            seed_tokens = _seed_token_ids(
                baseline_engine.tokenizer,
                max(batch_sizes),
            )
            prompts_by_family = {
                (batch_size, width):
                build_prompt_token_batch(
                    seed_token_ids=seed_tokens,
                    batch_size=batch_size,
                    query_len=query_len,
                    page_table_width=width,
                    block_size=(
                        baseline_engine
                        .scheduler.block_manager.block_size
                    ),
                )
                for batch_size in batch_sizes
                for width in page_table_widths
            }
            oracle = _run_oracle_cases(
                baseline_engine,
                prompts_by_family=prompts_by_family,
                query_len=query_len,
            )
            vocab_size = int(
                baseline_engine.model_runner
                .config.hf_config.vocab_size
            )
            baseline_adapter = OracleMismatchDraftAdapter(
                oracle_tokens_by_prompt=oracle,
                query_len=query_len,
                vocab_size=vocab_size,
            )
            baseline_engine.activate_speculative_runtime(
                EngineSpeculativeRuntime(baseline_adapter)
            )
            baseline_recorder = FamilyRunRecorder(
                baseline_engine
            )
            baseline_recorder.install()
            eager_by_family = {}
            for family_key, prompts in (
                prompts_by_family.items()
            ):
                eager_by_family[family_key] = (
                    _run_recorded_family(
                        baseline_engine,
                        baseline_recorder,
                        prompts=prompts,
                        query_len=query_len,
                    )
                )

            graph_engine = engine_factory(
                model_path,
                **_engine_kwargs(
                    context_length=context_length,
                    batch_sizes=batch_sizes,
                    query_len=query_len,
                    graph_enabled=True,
                ),
            )
            graph_adapter = OracleMismatchDraftAdapter(
                oracle_tokens_by_prompt=oracle,
                query_len=query_len,
                vocab_size=vocab_size,
            )
            graph_engine.activate_speculative_runtime(
                EngineSpeculativeRuntime(graph_adapter)
            )
            graph_recorder = FamilyRunRecorder(graph_engine)
            graph_recorder.install()
            warmed_by_family = {}
            warmed_measurements_by_family = {}
            capture_by_family = {}
            cold_by_family = {}
            for family_key, prompts in (
                prompts_by_family.items()
            ):
                cold = _run_recorded_family(
                    graph_engine,
                    graph_recorder,
                    prompts=prompts,
                    query_len=query_len,
                )
                if (
                    cold["dispatch_event"]["decision"]
                    != "cold"
                ):
                    raise RuntimeError(
                        "first family observation must be cold"
                    )
                cold_by_family[family_key] = cold
                captured = _run_recorded_family(
                    graph_engine,
                    graph_recorder,
                    prompts=prompts,
                    query_len=query_len,
                )
                if (
                    captured["dispatch_event"]["decision"]
                    != "capture"
                    or captured["dispatch_event"][
                        "cache_state"
                    ] != "ready"
                ):
                    raise RuntimeError(
                        "second family observation must capture"
                    )
                warmed_measurements = []
                repeat_count = (
                    PERFORMANCE_MEASUREMENT_REPEATS
                    if measure_performance
                    else 1
                )
                for _ in range(repeat_count):
                    warmed = _run_recorded_family(
                        graph_engine,
                        graph_recorder,
                        prompts=prompts,
                        query_len=query_len,
                    )
                    if (
                        warmed["dispatch_event"]["dispatch"]
                        != "graph"
                        or warmed["dispatch_event"]["decision"]
                        != "hit"
                    ):
                        raise RuntimeError(
                            "warmed family observation must be "
                            "an exact graph hit"
                        )
                    warmed_measurements.append(warmed)
                warmed = warmed_measurements[0]
                capture_by_family[family_key] = captured
                warmed_by_family[family_key] = warmed
                warmed_measurements_by_family[family_key] = (
                    tuple(warmed_measurements)
                )

            entries = tuple(
                graph_engine.model_runner
                .spec_verify_exact_cuda_graph_cache
                .ready_entries.values()
            )
            entry_by_family = {
                (
                    entry.identity.active_batch_size,
                    entry.identity.page_table_width,
                ): entry
                for entry in entries
                if entry.identity.query_len == query_len
            }
            if set(entry_by_family) != set(
                prompts_by_family
            ):
                raise RuntimeError(
                    "ready graph entries do not cover exact families"
                )
            pre_failure_cache_summary = (
                graph_engine.model_runner
                .spec_verify_exact_cuda_graph_cache
                .summary()
            )
            for name in performance_cache_counts:
                performance_cache_counts[name] += int(
                    pre_failure_cache_summary[name]
                )
            failure_by_family = {
                family_key: inject_replay_failure(
                    graph_engine.model_runner,
                    entry_by_family[family_key],
                )
                for family_key in prompts_by_family
            }

            for batch_size in batch_sizes:
                for width in page_table_widths:
                    family_key = (batch_size, width)
                    eager = eager_by_family[family_key]
                    warmed = warmed_by_family[family_key]
                    captured = capture_by_family[family_key]
                    event = warmed["dispatch_event"]
                    identity = {
                        "active_batch_size": batch_size,
                        "query_len": query_len,
                        "page_table_width": width,
                    }
                    if any(
                        event[name] != value
                        for name, value in identity.items()
                    ):
                        raise RuntimeError(
                            "warmed exact family identity drifted"
                        )
                    transaction_results = (
                        build_transaction_results(
                            tuple(
                                warmed["transaction_rows"]
                            ),
                            accepted_prefix_kv_parity=(
                                eager[
                                    "accepted_prefix_kv_sha256"
                                ]
                                == warmed[
                                    "accepted_prefix_kv_sha256"
                                ]
                            ),
                        )
                    )
                    family_id = _family_id(
                        batch_size,
                        query_len,
                        width,
                    )
                    families.append({
                        "family_id": family_id,
                        "identity": identity,
                        "eager_baseline": {
                            key: eager[key]
                            for key in (
                                "logits_sha256",
                                "target_tokens",
                                "accepted_lengths",
                                "final_tokens",
                                "accepted_prefix_kv_sha256",
                                "target_forward_count",
                            )
                        },
                        "warmed_graph": {
                            "identity": identity,
                            **{
                                key: warmed[key]
                                for key in (
                                    "logits_sha256",
                                    "target_tokens",
                                    "accepted_lengths",
                                    "final_tokens",
                                    "accepted_prefix_kv_sha256",
                                    "target_forward_count",
                                )
                            },
                            "eager_forward_count": 0,
                            "graph_replay_count": 1,
                            "warmed_latency_ns": (
                                warmed["latency_ns"]
                            ),
                            "capture_latency_ns": int(
                                captured["dispatch_event"][
                                    "capture_duration_ns"
                                ]
                            ),
                        },
                        "transaction_results": (
                            transaction_results
                        ),
                        "replay_failure_injection": (
                            failure_by_family[family_key]
                        ),
                    })
                    if measure_performance:
                        cold = cold_by_family[family_key]
                        capture_event = captured[
                            "dispatch_event"
                        ]
                        performance_rows.append({
                            "family_id": family_id,
                            "batch_size": batch_size,
                            "query_len": query_len,
                            "prompt_length": len(
                                prompts_by_family[
                                    family_key
                                ][0]
                            ),
                            "accepted_draft_tokens": sum(
                                int(value)
                                for value in warmed[
                                    "accepted_lengths"
                                ]
                            ),
                            "eager_request_metrics": eager[
                                "request_metrics"
                            ],
                            "cold_request_metrics": cold[
                                "request_metrics"
                            ],
                            "capture_request_metrics": (
                                captured["request_metrics"]
                            ),
                            "warmed_request_metrics": warmed[
                                "request_metrics"
                            ],
                            "warmed_request_metrics_rows": tuple(
                                measurement[
                                    "request_metrics"
                                ]
                                for measurement in (
                                    warmed_measurements_by_family[
                                        family_key
                                    ]
                                )
                            ),
                            "warmed_verifier_latency_ns": (
                                warmed["latency_ns"]
                            ),
                            "warmed_verifier_latency_ns_rows": tuple(
                                measurement["latency_ns"]
                                for measurement in (
                                    warmed_measurements_by_family[
                                        family_key
                                    ]
                                )
                            ),
                            "capture_duration_ns": int(
                                capture_event[
                                    "capture_duration_ns"
                                ]
                            ),
                            "capture_allocated_delta_bytes": int(
                                capture_event[
                                    "capture_allocated_delta_bytes"
                                ]
                            ),
                            "capture_reserved_delta_bytes": int(
                                capture_event[
                                    "capture_reserved_delta_bytes"
                                ]
                            ),
                        })
        finally:
            if graph_recorder is not None:
                graph_recorder.close()
            if baseline_recorder is not None:
                baseline_recorder.close()
            if graph_engine is not None:
                graph_engine.exit()
            baseline_engine.exit()

    family_count = len(families)
    capability = torch.cuda.get_device_capability(0)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "source_files": source_files,
        "model": Path(model_path).name,
        "checkpoint": str(Path(model_path).resolve()),
        "device_name": torch.cuda.get_device_name(0),
        "device_compute_capability": [
            int(capability[0]),
            int(capability[1]),
        ],
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "flash_attn_version": _package_version(
            "flash-attn"
        ),
        "configuration": {
            "tensor_parallel_size": 1,
            "kv_offload_mvp0": False,
            "context_length": context_length,
            "batch_sizes": list(batch_sizes),
            "query_lengths": list(query_lengths),
            "page_table_widths": list(
                page_table_widths
            ),
            "identity_policy": "exact_b_q_w",
            "capture_latency_excluded_from_warmed_hit": True,
            "measure_performance": bool(
                measure_performance
            ),
        },
        "families": families,
        "eager_baseline": {
            "family_count": family_count,
        },
        "warmed_graph": {
            "family_count": family_count,
        },
        "replay_failure_injection": {
            "family_count": family_count,
        },
        "transaction_results": {
            "family_count": family_count,
        },
        "claims": {
            "kv_offload_benefit": False,
            "h2d_d2h_benefit": False,
        },
        "classification": CLASSIFICATION,
        "runner": {
            "python_version": platform.python_version(),
            "command": list(sys.argv),
        },
    }
    if measure_performance:
        artifact["performance"] = build_performance_evidence(
            tuple(performance_rows),
            cache_counts=performance_cache_counts,
        )
    from tools.verify_spec_verify_cuda_graph_gate import (
        validate_artifact,
    )

    validate_artifact(artifact)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"output artifact already exists: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--query-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_QUERY_LENGTHS),
    )
    parser.add_argument(
        "--page-table-widths",
        type=int,
        nargs="+",
        default=list(DEFAULT_PAGE_TABLE_WIDTHS),
    )
    parser.add_argument(
        "--measure-performance",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = run_live_gate(
        model_path=args.model,
        output_path=args.output_json,
        context_length=_positive_int(
            args.context_length,
            "context_length",
        ),
        batch_sizes=tuple(sorted(set(args.batch_sizes))),
        query_lengths=tuple(
            sorted(set(args.query_lengths))
        ),
        page_table_widths=tuple(
            sorted(set(args.page_table_widths))
        ),
        measure_performance=args.measure_performance,
    )
    print(json.dumps({
        "status": "PASS",
        "classification": artifact["classification"],
        "family_count": len(artifact["families"]),
        "output_json": str(args.output_json),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
