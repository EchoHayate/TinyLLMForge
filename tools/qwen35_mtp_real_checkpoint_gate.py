from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import tempfile


SCHEMA_VERSION = 1
REQUIRED_Q_VALUES = (1, 2, 3, 4)
REQUIRED_BATCH_SIZES = (1, 4)
REQUIRED_REPORT_FIELDS = (
    "schema_version",
    "checkpoint_path",
    "checkpoint_manifest_sha256",
    "device_name",
    "torch_version",
    "cuda_version",
    "q_values",
    "batch_sizes",
    "loader_passed",
    "shared_embedding_identity",
    "shared_lm_head_identity",
    "eager_reference_max_abs_diff",
    "eager_reference_argmax_equal",
    "graph_backend_installed",
    "graph_capture_count",
    "graph_replay_count",
    "graph_eager_argmax_equal",
    "graph_eager_proposal_tokens_equal",
    "graph_transaction_commit",
    "graph_transaction_rollback",
    "replay_failure_quarantined",
    "replay_failure_eager_retry_count",
    "transaction_cases",
    "accepted_slot_identity_preserved",
    "rejected_slots_released",
    "post_rollback_continuation_equal",
    "status",
    "promotion_classification",
    "limitations",
)

_BACKEND_FAILURE_DOMAINS = (
    "load",
    "eager_reference",
    "graph_eager",
    "transaction",
)


def checkpoint_manifest_sha256(checkpoint_path) -> str:
    root = Path(checkpoint_path)
    if not root.is_dir():
        raise ValueError("checkpoint_path must be an existing directory")
    files = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
        )
    )
    if not files:
        raise ValueError("checkpoint directory must contain files")
    manifest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        payload_sha256 = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                payload_sha256.update(chunk)
                size += len(chunk)
        row = json.dumps(
            {
                "path": relative,
                "sha256": payload_sha256.hexdigest(),
                "size": size,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest.update(row.encode("utf-8"))
        manifest.update(b"\n")
    return manifest.hexdigest()


def _runtime_field(runtime, name: str, default=None):
    if isinstance(runtime, Mapping):
        return runtime.get(name, default)
    return getattr(runtime, name, default)


def _qwen35_reference_prefill_attention(
    query,
    key,
    value,
    context,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
    key_cache=None,
    value_cache=None,
):
    import torch
    import torch.nn.functional as functional

    if (
        isinstance(key_cache, torch.Tensor)
        and isinstance(value_cache, torch.Tensor)
        and key_cache.numel() > 0
        and value_cache.numel() > 0
    ):
        block_size = int(key_cache.shape[1])
        slots = context.slot_mapping.to(torch.long)
        blocks = torch.div(
            slots,
            block_size,
            rounding_mode="floor",
        )
        offsets = slots.remainder(block_size)
        key_cache[blocks, offsets] = key
        value_cache[blocks, offsets] = value
    cu_seqlens_q = getattr(context, "cu_seqlens_q", None)
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise ValueError(
            "reference prefill requires cu_seqlens_q"
        )
    boundaries = tuple(
        int(value)
        for value in cu_seqlens_q.detach().cpu().tolist()
    )
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            raise ValueError(
                "reference prefill segments must be non-empty"
            )
        segment_query = query[start:end]
        segment_key = key[start:end]
        segment_value = value[start:end]
        if num_heads % int(segment_key.shape[1]) != 0:
            raise ValueError(
                "reference prefill query heads must divide KV heads"
            )
        repeats = num_heads // int(segment_key.shape[1])
        row_query = segment_query.transpose(0, 1).unsqueeze(0)
        row_key = (
            segment_key.repeat_interleave(repeats, dim=1)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        row_value = (
            segment_value.repeat_interleave(repeats, dim=1)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        output = functional.scaled_dot_product_attention(
            row_query,
            row_key,
            row_value,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
        outputs.append(
            output.transpose(1, 2)
            .reshape(end - start, num_heads * head_dim)
        )
    return torch.cat(outputs, dim=0)


def _qwen35_reference_cached_decode_attention(
    query,
    current_key,
    current_value,
    key_cache,
    value_cache,
    context,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
):
    import torch
    import torch.nn.functional as functional

    block_size = int(key_cache.shape[1])
    slots = context.slot_mapping.to(torch.long)
    blocks = torch.div(
        slots,
        block_size,
        rounding_mode="floor",
    )
    offsets = slots.remainder(block_size)
    key_cache[blocks, offsets] = current_key
    value_cache[blocks, offsets] = current_value
    outputs = []
    for index, length_value in enumerate(context.context_lens):
        context_length = int(length_value.item())
        if context_length <= 0:
            raise ValueError(
                "reference decode context length must be positive"
            )
        block_count = (
            context_length + block_size - 1
        ) // block_size
        block_ids = context.block_tables[
            index, :block_count
        ].to(torch.long)
        key = key_cache[block_ids].reshape(
            -1,
            key_cache.shape[2],
            key_cache.shape[3],
        )[:context_length]
        value = value_cache[block_ids].reshape(
            -1,
            value_cache.shape[2],
            value_cache.shape[3],
        )[:context_length]
        if num_heads % int(key.shape[1]) != 0:
            raise ValueError(
                "reference decode query heads must divide KV heads"
            )
        repeats = num_heads // int(key.shape[1])
        row_query = query[
            index:index + 1
        ].transpose(0, 1).unsqueeze(0)
        row_key = (
            key.repeat_interleave(repeats, dim=1)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        row_value = (
            value.repeat_interleave(repeats, dim=1)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        output = functional.scaled_dot_product_attention(
            row_query,
            row_key,
            row_value,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        outputs.append(
            output.transpose(1, 2)
            .reshape(1, num_heads * head_dim)
        )
    return torch.cat(outputs, dim=0)


def _build_real_eager_reference_probe(
    *,
    executor,
    module,
    physical_store,
    hf_config,
):
    import torch

    from tinyvllm.engine.speculative_proposal_executor import (
        ModelRunnerProposalInput,
        TargetPrefillObservation,
    )
    from tinyvllm.layers import qwen35_full_attention

    config = getattr(hf_config, "text_config", hf_config)
    hidden_size = int(getattr(config, "hidden_size"))
    vocab_size = int(getattr(config, "vocab_size"))
    if hidden_size <= 0 or vocab_size <= 8:
        raise ValueError(
            "MTP eager/reference probe requires "
            "positive hidden/vocab sizes"
        )
    key_cache = physical_store.key_cache
    value_cache = physical_store.value_cache
    if (
        not isinstance(key_cache, torch.Tensor)
        or not isinstance(value_cache, torch.Tensor)
        or key_cache.shape != value_cache.shape
        or key_cache.ndim != 4
        or key_cache.shape[1] != 1
    ):
        raise ValueError(
            "MTP physical store must expose "
            "block-size-one K/V tensors"
        )
    cache = executor.proposal_kv_cache
    next_sequence_id = 2_000_000

    def hidden_rows(row_index: int, rows: int):
        values = torch.arange(
            rows * hidden_size,
            dtype=torch.float32,
            device=key_cache.device,
        ).reshape(rows, hidden_size)
        values = values + 1.0 + float(row_index) / 16.0
        return values.to(dtype=key_cache.dtype)

    def release_sequence(sequence_id: int, sequence_epoch: int):
        active_ids = [
            transaction_id
            for transaction_id, owner
            in tuple(executor._proposal_transactions.items())
            if owner[0] == sequence_id
        ]
        for transaction_id in active_ids:
            transaction = cache.transaction(transaction_id)
            if (
                transaction is not None
                and transaction.state in ("reserved", "materialized")
            ):
                cache.abort(transaction_id)
            executor._proposal_transactions.pop(
                transaction_id,
                None,
            )
        executor.release_sequence(
            sequence_id,
            sequence_epoch=sequence_epoch,
        )

    def run_scenario(
        *,
        q: int,
        batch_size: int,
        use_reference: bool,
    ):
        nonlocal next_sequence_id
        sequence_epoch = 0
        sequence_ids = tuple(
            range(next_sequence_id, next_sequence_id + batch_size)
        )
        next_sequence_id += batch_size
        original_forward_step = module.forward_step
        original_graph_runner = executor.graph_runner
        had_instance_forward_step = (
            "forward_step" in getattr(module, "__dict__", {})
        )
        instance_forward_step = getattr(
            module,
            "__dict__",
            {},
        ).get("forward_step")
        original_prefill = (
            qwen35_full_attention.qwen35_prefill_eager_attention
        )
        original_decode = (
            qwen35_full_attention
            .qwen35_cached_decode_eager_attention
        )
        captured_logits = []

        def capture_forward_step(
            input_ids,
            positions,
            hidden_states,
        ):
            next_hidden, logits = original_forward_step(
                input_ids,
                positions,
                hidden_states,
            )
            if int(input_ids.shape[0]) == 1:
                captured_logits.append(
                    logits.detach().clone()
                )
            return next_hidden, logits

        object.__setattr__(
            module,
            "forward_step",
            capture_forward_step,
        )
        executor.graph_runner = None
        if use_reference:
            qwen35_full_attention.qwen35_prefill_eager_attention = (
                _qwen35_reference_prefill_attention
            )
            qwen35_full_attention.qwen35_cached_decode_eager_attention = (
                _qwen35_reference_cached_decode_attention
            )
        try:
            observations = []
            inputs = []
            for row_index, sequence_id in enumerate(sequence_ids):
                base = 1 + row_index * 4
                prompt = (base, base + 1)
                observations.append(TargetPrefillObservation(
                    sequence_id=sequence_id,
                    sequence_epoch=sequence_epoch,
                    token_ids=prompt,
                    positions=torch.tensor(
                        [0, 1],
                        dtype=torch.int64,
                        device=key_cache.device,
                    ),
                    target_hidden=hidden_rows(row_index, 2),
                    is_final_chunk=True,
                ))
                inputs.append(ModelRunnerProposalInput(
                    sequence_id=sequence_id,
                    token_ids=prompt,
                    remaining_output_tokens=q,
                    max_proposal_tokens=q,
                    first_target_token=base + 2,
                    target_hidden=hidden_rows(
                        row_index + batch_size,
                        1,
                    ),
                ))
            executor.observe_target_prefill(tuple(observations))
            proposals = executor.propose_batch(tuple(inputs))
            expected_logit_count = batch_size * max(q - 1, 0)
            if len(captured_logits) != expected_logit_count:
                raise RuntimeError(
                    "MTP eager/reference logit capture count "
                    f"must equal {expected_logit_count}, "
                    f"got {len(captured_logits)}"
                )
            return (
                tuple(proposal.token_ids for proposal in proposals),
                tuple(captured_logits),
            )
        finally:
            executor.graph_runner = original_graph_runner
            qwen35_full_attention.qwen35_prefill_eager_attention = (
                original_prefill
            )
            qwen35_full_attention.qwen35_cached_decode_eager_attention = (
                original_decode
            )
            if had_instance_forward_step:
                object.__setattr__(
                    module,
                    "forward_step",
                    instance_forward_step,
                )
            else:
                object.__delattr__(module, "forward_step")
            for sequence_id in sequence_ids:
                release_sequence(sequence_id, sequence_epoch)
            if any(
                physical_store.is_allocated(slot_id)
                for slot_id in range(physical_store.capacity)
            ):
                raise RuntimeError(
                    "MTP eager/reference scenario leaked physical slots"
                )

    def eager_reference_probe(q: int, batch_size: int):
        if q not in REQUIRED_Q_VALUES:
            raise ValueError(
                "MTP eager/reference q is outside the gate domain"
            )
        if batch_size not in REQUIRED_BATCH_SIZES:
            raise ValueError(
                "MTP eager/reference batch size is "
                "outside the gate domain"
            )
        with torch.no_grad():
            eager_tokens, eager_logits = run_scenario(
                q=q,
                batch_size=batch_size,
                use_reference=False,
            )
            reference_tokens, reference_logits = run_scenario(
                q=q,
                batch_size=batch_size,
                use_reference=True,
            )
        if len(eager_logits) != len(reference_logits):
            raise RuntimeError(
                "MTP eager/reference logit counts differ"
            )
        max_abs_diff = 0.0
        argmax_equal = eager_tokens == reference_tokens
        for eager, reference in zip(
            eager_logits,
            reference_logits,
        ):
            if (
                eager.shape != reference.shape
                or eager.dtype != reference.dtype
                or eager.device != reference.device
            ):
                raise ValueError(
                    "MTP eager/reference logits must have "
                    "identical shape, dtype, and device"
                )
            difference = float(
                torch.max(
                    torch.abs(
                        eager.to(torch.float32)
                        - reference.to(torch.float32)
                    )
                ).item()
            )
            if not math.isfinite(difference):
                raise ValueError(
                    "MTP eager/reference logits difference "
                    "must be finite"
                )
            max_abs_diff = max(max_abs_diff, difference)
            argmax_equal = (
                argmax_equal
                and torch.equal(
                    torch.argmax(eager, dim=-1),
                    torch.argmax(reference, dim=-1),
                )
            )
        return {
            "max_abs_diff": max_abs_diff,
            "argmax_equal": argmax_equal,
        }

    return eager_reference_probe


def _build_real_graph_eager_probe(
    *,
    executor,
    module,
    physical_store,
    hf_config,
):
    import torch

    from tinyvllm.engine.qwen35_mtp_graph import (
        Qwen35MTPGraphReplayError,
    )
    from tinyvllm.engine.speculative_proposal_executor import (
        ModelRunnerProposalInput,
        ProposalFinalizeRow,
        TargetPrefillObservation,
    )

    del module
    config = getattr(hf_config, "text_config", hf_config)
    hidden_size = int(getattr(config, "hidden_size"))
    vocab_size = int(getattr(config, "vocab_size"))
    if hidden_size <= 0 or vocab_size <= 8:
        raise ValueError(
            "MTP graph/eager probe requires "
            "positive hidden/vocab sizes"
        )
    graph_runner = getattr(executor, "graph_runner", None)
    if graph_runner is None:
        raise RuntimeError(
            "Qwen3.5 MTP exact-Q graph runner is unavailable"
        )
    if not callable(getattr(graph_runner, "summary", None)):
        raise RuntimeError(
            "Qwen3.5 MTP exact-Q graph runner has no summary"
        )
    key_cache = physical_store.key_cache
    value_cache = physical_store.value_cache
    if (
        not isinstance(key_cache, torch.Tensor)
        or not isinstance(value_cache, torch.Tensor)
        or key_cache.shape != value_cache.shape
        or key_cache.ndim != 4
        or key_cache.shape[1] != 1
    ):
        raise ValueError(
            "MTP physical store must expose "
            "block-size-one K/V tensors"
        )
    cache = executor.proposal_kv_cache
    next_sequence_id = 3_000_000
    replay_failure_injected = False

    def hidden_rows(row_index: int, rows: int):
        values = torch.arange(
            rows * hidden_size,
            dtype=torch.float32,
            device=key_cache.device,
        ).reshape(rows, hidden_size)
        values = values + 1.0 + float(row_index) / 16.0
        return values.to(dtype=key_cache.dtype)

    def release_sequence(sequence_id: int, sequence_epoch: int):
        active_ids = [
            transaction_id
            for transaction_id, owner
            in tuple(executor._proposal_transactions.items())
            if owner[0] == sequence_id
        ]
        for transaction_id in active_ids:
            transaction = cache.transaction(transaction_id)
            if (
                transaction is not None
                and transaction.state in ("reserved", "materialized")
            ):
                cache.abort(transaction_id)
            executor._proposal_transactions.pop(
                transaction_id,
                None,
            )
        executor.release_sequence(
            sequence_id,
            sequence_epoch=sequence_epoch,
        )

    def build_rows(q: int, batch_size: int):
        nonlocal next_sequence_id
        sequence_epoch = 0
        sequence_ids = tuple(
            range(next_sequence_id, next_sequence_id + batch_size)
        )
        next_sequence_id += batch_size
        observations = []
        inputs = []
        for row_index, sequence_id in enumerate(sequence_ids):
            base = 1 + row_index * 4
            prompt = (base, base + 1)
            observations.append(TargetPrefillObservation(
                sequence_id=sequence_id,
                sequence_epoch=sequence_epoch,
                token_ids=prompt,
                positions=torch.tensor(
                    [0, 1],
                    dtype=torch.int64,
                    device=key_cache.device,
                ),
                target_hidden=hidden_rows(row_index, 2),
                is_final_chunk=True,
            ))
            inputs.append(ModelRunnerProposalInput(
                sequence_id=sequence_id,
                token_ids=prompt,
                remaining_output_tokens=q,
                max_proposal_tokens=q,
                first_target_token=base + 2,
                target_hidden=hidden_rows(
                    row_index + batch_size,
                    1,
                ),
            ))
        return (
            sequence_epoch,
            sequence_ids,
            tuple(observations),
            tuple(inputs),
        )

    def run_proposals(q: int, batch_size: int):
        (
            sequence_epoch,
            sequence_ids,
            observations,
            inputs,
        ) = build_rows(q, batch_size)
        try:
            executor.observe_target_prefill(observations)
            proposals = executor.propose_batch(inputs)
            return sequence_epoch, sequence_ids, proposals
        except BaseException:
            for sequence_id in sequence_ids:
                release_sequence(sequence_id, sequence_epoch)
            raise

    def release_rows(sequence_epoch, sequence_ids):
        for sequence_id in sequence_ids:
            release_sequence(sequence_id, sequence_epoch)
        if any(
            physical_store.is_allocated(slot_id)
            for slot_id in range(physical_store.capacity)
        ):
            raise RuntimeError(
                "MTP graph/eager scenario leaked physical slots"
            )

    def eager_tokens(q: int, batch_size: int):
        original_graph_runner = executor.graph_runner
        executor.graph_runner = None
        sequence_epoch = None
        sequence_ids = ()
        try:
            (
                sequence_epoch,
                sequence_ids,
                proposals,
            ) = run_proposals(q, batch_size)
            return tuple(
                proposal.token_ids for proposal in proposals
            )
        finally:
            executor.graph_runner = original_graph_runner
            if sequence_epoch is not None:
                release_rows(sequence_epoch, sequence_ids)

    def warm_until_captured(q: int, batch_size: int):
        identity = graph_runner._identity(q, batch_size)
        attempts = max(
            int(graph_runner.min_observations),
            1,
        )
        for _ in range(attempts):
            sequence_epoch = None
            sequence_ids = ()
            try:
                (
                    sequence_epoch,
                    sequence_ids,
                    _,
                ) = run_proposals(q, batch_size)
            finally:
                if sequence_epoch is not None:
                    release_rows(sequence_epoch, sequence_ids)
        if identity.sha256 not in graph_runner.ready_entries:
            raise RuntimeError(
                "Qwen3.5 MTP exact-Q graph family was not captured"
            )
        return identity

    def replay_and_finalize(
        q: int,
        batch_size: int,
        *,
        accepted: int,
    ):
        sequence_epoch = None
        sequence_ids = ()
        try:
            (
                sequence_epoch,
                sequence_ids,
                proposals,
            ) = run_proposals(q, batch_size)
            if any(
                proposal.metadata.get("execution_mode")
                != "cuda_graph"
                for proposal in proposals
            ):
                raise RuntimeError(
                    "Qwen3.5 MTP graph replay did not return "
                    "graph proposals"
                )
            accepted_snapshots = {}
            rejected_slots = []
            finalize_rows = []
            for proposal in proposals:
                transaction = cache.transaction(
                    proposal.proposal_transaction_id
                )
                if transaction is None:
                    raise RuntimeError(
                        "graph proposal transaction is unavailable"
                    )
                commit_count = max(accepted - 1, 0)
                accepted_slots = transaction.staged_slot_ids[
                    :commit_count
                ]
                rejected_slots.extend(
                    transaction.staged_slot_ids[commit_count:]
                )
                for slot_id in accepted_slots:
                    accepted_snapshots[slot_id] = (
                        physical_store.slot_identity(slot_id),
                        key_cache[slot_id].clone(),
                        value_cache[slot_id].clone(),
                    )
                finalize_rows.append(ProposalFinalizeRow(
                    sequence_id=proposal.sequence_id,
                    proposal_transaction_id=(
                        proposal.proposal_transaction_id
                    ),
                    accepted_proposal_tokens=accepted,
                ))
            ticket = executor.prepare_finalize_batch(
                tuple(finalize_rows)
            )
            executor.commit_finalize_batch(ticket)
            committed = all(
                physical_store.is_allocated(slot_id)
                and physical_store.slot_identity(slot_id)
                == snapshot[0]
                and torch.equal(key_cache[slot_id], snapshot[1])
                and torch.equal(value_cache[slot_id], snapshot[2])
                for slot_id, snapshot
                in accepted_snapshots.items()
            )
            rolled_back = all(
                not physical_store.is_allocated(slot_id)
                and torch.count_nonzero(
                    key_cache[slot_id]
                ).item() == 0
                and torch.count_nonzero(
                    value_cache[slot_id]
                ).item() == 0
                for slot_id in rejected_slots
            )
            return (
                tuple(
                    proposal.token_ids for proposal in proposals
                ),
                committed,
                rolled_back,
            )
        finally:
            if sequence_epoch is not None:
                release_rows(sequence_epoch, sequence_ids)

    def inject_replay_failure(q: int, batch_size: int, identity):
        entry = graph_runner.ready_entries[identity.sha256]
        payload = entry.graph
        original_graph = payload.graph
        eager_calls = 0
        original_eager = executor._run_exact_q_eager

        class ReplayThenFail:

            def replay(self):
                original_graph.replay()
                raise RuntimeError(
                    "injected post-replay gate failure"
                )

        def count_eager(*args, **kwargs):
            nonlocal eager_calls
            eager_calls += 1
            return original_eager(*args, **kwargs)

        payload.graph = ReplayThenFail()
        executor._run_exact_q_eager = count_eager
        sequence_epoch = None
        sequence_ids = ()
        replay_failed = False
        try:
            try:
                (
                    sequence_epoch,
                    sequence_ids,
                    _,
                ) = run_proposals(q, batch_size)
            except Qwen35MTPGraphReplayError:
                replay_failed = True
        finally:
            payload.graph = original_graph
            executor._run_exact_q_eager = original_eager
            if sequence_epoch is not None:
                release_rows(sequence_epoch, sequence_ids)
        quarantined = (
            replay_failed
            and graph_runner.quarantine_reason(identity)
            == "replay_failed"
        )
        return quarantined, eager_calls

    def graph_eager_probe(q: int, batch_size: int):
        nonlocal replay_failure_injected
        if q not in REQUIRED_Q_VALUES:
            raise ValueError(
                "MTP graph/eager q is outside the gate domain"
            )
        if batch_size not in REQUIRED_BATCH_SIZES:
            raise ValueError(
                "MTP graph/eager batch size is "
                "outside the gate domain"
            )
        if q == 1:
            return {
                "backend_installed": True,
                "capture_count": 0,
                "replay_count": 0,
                "argmax_equal": True,
                "proposal_tokens_equal": True,
                "transaction_commit": True,
                "transaction_rollback": True,
                "replay_failure_quarantined": False,
                "replay_failure_eager_retry_count": 0,
            }
        with torch.no_grad():
            baseline_tokens = eager_tokens(q, batch_size)
            summary_before = graph_runner.summary()
            identity = warm_until_captured(q, batch_size)
            (
                committed_tokens,
                committed,
                _,
            ) = replay_and_finalize(
                q,
                batch_size,
                accepted=q,
            )
            (
                rolled_back_tokens,
                _,
                rolled_back,
            ) = replay_and_finalize(
                q,
                batch_size,
                accepted=1,
            )
            quarantined = False
            eager_retry_count = 0
            if (
                not replay_failure_injected
                and (q, batch_size) == (4, 4)
            ):
                (
                    quarantined,
                    eager_retry_count,
                ) = inject_replay_failure(
                    q,
                    batch_size,
                    identity,
                )
                replay_failure_injected = True
            summary_after = graph_runner.summary()
        proposal_tokens_equal = (
            baseline_tokens == committed_tokens
            and baseline_tokens == rolled_back_tokens
        )
        return {
            "backend_installed": True,
            "capture_count": (
                summary_after["captures"]
                - summary_before["captures"]
            ),
            "replay_count": (
                summary_after["replays"]
                - summary_before["replays"]
            ),
            "argmax_equal": proposal_tokens_equal,
            "proposal_tokens_equal": proposal_tokens_equal,
            "transaction_commit": committed,
            "transaction_rollback": rolled_back,
            "replay_failure_quarantined": quarantined,
            "replay_failure_eager_retry_count": (
                eager_retry_count
            ),
        }

    return graph_eager_probe


def _build_real_transaction_probe(
    *,
    executor,
    module,
    physical_store,
    hf_config,
):
    import torch

    from tinyvllm.engine.speculative_proposal_executor import (
        ModelRunnerProposalInput,
        ProposalFinalizeRow,
        TargetPrefillObservation,
    )

    del module
    config = getattr(hf_config, "text_config", hf_config)
    hidden_size = int(getattr(config, "hidden_size"))
    vocab_size = int(getattr(config, "vocab_size"))
    if hidden_size <= 0 or vocab_size <= 8:
        raise ValueError(
            "MTP transaction probe requires positive hidden/vocab sizes"
        )
    key_cache = physical_store.key_cache
    value_cache = physical_store.value_cache
    if (
        not isinstance(key_cache, torch.Tensor)
        or not isinstance(value_cache, torch.Tensor)
        or key_cache.shape != value_cache.shape
        or key_cache.ndim != 4
        or key_cache.shape[1] != 1
    ):
        raise ValueError(
            "MTP physical store must expose block-size-one K/V tensors"
        )
    cache = executor.proposal_kv_cache
    next_sequence_id = 1_000_000

    def hidden_rows(sequence_id: int, rows: int):
        values = torch.arange(
            rows * hidden_size,
            dtype=torch.float32,
            device=key_cache.device,
        ).reshape(rows, hidden_size)
        values = values + float(sequence_id % 17) / 17.0
        return values.to(dtype=key_cache.dtype)

    def release_sequence(sequence_id: int, sequence_epoch: int):
        active_ids = [
            transaction_id
            for transaction_id, owner
            in tuple(executor._proposal_transactions.items())
            if owner[0] == sequence_id
        ]
        for transaction_id in active_ids:
            transaction = cache.transaction(transaction_id)
            if (
                transaction is not None
                and transaction.state in ("reserved", "materialized")
            ):
                cache.abort(transaction_id)
            executor._proposal_transactions.pop(
                transaction_id,
                None,
            )
        executor.release_sequence(
            sequence_id,
            sequence_epoch=sequence_epoch,
        )

    def run_continuation(
        *,
        sequence_ids,
        token_rows,
        first_target_tokens,
        sequence_epoch,
    ):
        inputs = tuple(
            ModelRunnerProposalInput(
                sequence_id=sequence_id,
                token_ids=token_ids,
                remaining_output_tokens=2,
                max_proposal_tokens=2,
                first_target_token=first_target_token,
                target_hidden=hidden_rows(
                    sequence_id + 10_000,
                    1,
                ),
            )
            for sequence_id, token_ids, first_target_token in zip(
                sequence_ids,
                token_rows,
                first_target_tokens,
            )
        )
        proposals = executor.propose_batch(inputs)
        ticket = executor.prepare_finalize_batch(tuple(
            ProposalFinalizeRow(
                sequence_id=proposal.sequence_id,
                proposal_transaction_id=(
                    proposal.proposal_transaction_id
                ),
                accepted_proposal_tokens=len(proposal.token_ids),
            )
            for proposal in proposals
        ))
        executor.rollback_finalize_batch(ticket)
        return tuple(proposal.token_ids for proposal in proposals)

    def transaction_probe(
        q: int,
        batch_size: int,
        accepted: int,
    ):
        nonlocal next_sequence_id
        original_graph_runner = executor.graph_runner
        executor.graph_runner = None
        sequence_epoch = 0
        sequence_ids = tuple(
            range(next_sequence_id, next_sequence_id + batch_size)
        )
        next_sequence_id += batch_size
        prompt_rows = []
        first_target_tokens = []
        proposals = ()
        staged_slot_ids = []
        committed_slot_ids = []
        released_slot_ids = []
        try:
            observations = []
            inputs = []
            for sequence_id in sequence_ids:
                base = 1 + sequence_id % (vocab_size - 8)
                prompt = (base, base + 1)
                first_target = base + 2
                prompt_rows.append(prompt)
                first_target_tokens.append(first_target)
                observations.append(TargetPrefillObservation(
                    sequence_id=sequence_id,
                    sequence_epoch=sequence_epoch,
                    token_ids=prompt,
                    positions=torch.tensor(
                        [0, 1],
                        dtype=torch.int64,
                        device=key_cache.device,
                    ),
                    target_hidden=hidden_rows(sequence_id, 2),
                    is_final_chunk=True,
                ))
                inputs.append(ModelRunnerProposalInput(
                    sequence_id=sequence_id,
                    token_ids=prompt,
                    remaining_output_tokens=q,
                    max_proposal_tokens=q,
                    first_target_token=first_target,
                    target_hidden=hidden_rows(
                        sequence_id + 1_000,
                        1,
                    ),
                ))
            executor.observe_target_prefill(tuple(observations))
            proposals = executor.propose_batch(tuple(inputs))

            accepted_snapshots = {}
            rejected_slots = []
            finalize_rows = []
            for proposal in proposals:
                transaction = cache.transaction(
                    proposal.proposal_transaction_id
                )
                if transaction is None:
                    raise RuntimeError(
                        "proposal transaction is unavailable"
                    )
                staged = transaction.staged_slot_ids
                staged_slot_ids.extend(staged)
                commit_count = max(accepted - 1, 0)
                accepted_slots = staged[:commit_count]
                rejected = staged[commit_count:]
                committed_slot_ids.extend(accepted_slots)
                released_slot_ids.extend(rejected)
                rejected_slots.extend(rejected)
                for slot_id in accepted_slots:
                    accepted_snapshots[slot_id] = (
                        physical_store.slot_identity(slot_id),
                        key_cache[slot_id].clone(),
                        value_cache[slot_id].clone(),
                    )
                finalize_rows.append(ProposalFinalizeRow(
                    sequence_id=proposal.sequence_id,
                    proposal_transaction_id=(
                        proposal.proposal_transaction_id
                    ),
                    accepted_proposal_tokens=accepted,
                ))

            ticket = executor.prepare_finalize_batch(
                tuple(finalize_rows)
            )
            executor.commit_finalize_batch(ticket)

            accepted_identity = all(
                physical_store.is_allocated(slot_id)
                and physical_store.slot_identity(slot_id)
                == snapshot[0]
                and torch.equal(
                    key_cache[slot_id],
                    snapshot[1],
                )
                and torch.equal(
                    value_cache[slot_id],
                    snapshot[2],
                )
                for slot_id, snapshot
                in accepted_snapshots.items()
            )
            rejected_released = all(
                not physical_store.is_allocated(slot_id)
                and torch.count_nonzero(
                    key_cache[slot_id]
                ).item() == 0
                and torch.count_nonzero(
                    value_cache[slot_id]
                ).item() == 0
                for slot_id in rejected_slots
            )

            committed_snapshots = {
                sequence_id: (
                    cache.committed_slot_ids(sequence_id),
                    tuple(
                        (
                            slot_id,
                            physical_store.slot_identity(slot_id),
                            key_cache[slot_id].clone(),
                            value_cache[slot_id].clone(),
                        )
                        for slot_id in cache.committed_slot_ids(
                            sequence_id
                        )
                    ),
                )
                for sequence_id in sequence_ids
            }
            continuation_token_rows = tuple(
                prompt + proposal.token_ids[:accepted]
                for prompt, proposal in zip(prompt_rows, proposals)
            )
            continuation_targets = tuple(
                (first_target + accepted + 1) % vocab_size
                for first_target in first_target_tokens
            )
            first_continuation = run_continuation(
                sequence_ids=sequence_ids,
                token_rows=continuation_token_rows,
                first_target_tokens=continuation_targets,
                sequence_epoch=sequence_epoch,
            )
            state_after_first = all(
                cache.committed_slot_ids(sequence_id)
                == snapshot[0]
                and all(
                    physical_store.slot_identity(slot_id)
                    == identity
                    and torch.equal(key_cache[slot_id], key)
                    and torch.equal(value_cache[slot_id], value)
                    for slot_id, identity, key, value
                    in snapshot[1]
                )
                for sequence_id, snapshot
                in committed_snapshots.items()
            )
            second_continuation = run_continuation(
                sequence_ids=sequence_ids,
                token_rows=continuation_token_rows,
                first_target_tokens=continuation_targets,
                sequence_epoch=sequence_epoch,
            )
            state_after_second = all(
                cache.committed_slot_ids(sequence_id)
                == snapshot[0]
                and all(
                    physical_store.slot_identity(slot_id)
                    == identity
                    and torch.equal(key_cache[slot_id], key)
                    and torch.equal(value_cache[slot_id], value)
                    for slot_id, identity, key, value
                    in snapshot[1]
                )
                for sequence_id, snapshot
                in committed_snapshots.items()
            )
            continuation_equal = (
                first_continuation == second_continuation
                and state_after_first
                and state_after_second
            )
            return {
                "q": q,
                "batch_size": batch_size,
                "accepted_proposal_tokens": accepted,
                "staged_slot_ids": staged_slot_ids,
                "committed_slot_ids": committed_slot_ids,
                "released_slot_ids": released_slot_ids,
                "accepted_slot_identity_preserved": (
                    accepted_identity
                ),
                "rejected_slots_released": rejected_released,
                "post_rollback_continuation_equal": (
                    continuation_equal
                ),
            }
        finally:
            executor.graph_runner = original_graph_runner
            for sequence_id in sequence_ids:
                release_sequence(sequence_id, sequence_epoch)

    return transaction_probe


class RealQwen35MTPGateBackend:

    def __init__(
        self,
        *,
        runtime_loader=None,
        runtime_metadata_loader=None,
    ):
        if runtime_loader is not None and not callable(runtime_loader):
            raise ValueError("runtime_loader must be callable")
        if (
            runtime_metadata_loader is not None
            and not callable(runtime_metadata_loader)
        ):
            raise ValueError(
                "runtime_metadata_loader must be callable"
            )
        self._runtime_loader = (
            self._load_real_runtime
            if runtime_loader is None
            else runtime_loader
        )
        self._runtime_metadata_loader = (
            self._load_runtime_metadata
            if runtime_metadata_loader is None
            else runtime_metadata_loader
        )
        self._runtime = None
        self._blockers: dict[str, str] = {}

    @staticmethod
    def _load_runtime_metadata():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return {
            "device_name": torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        }

    @staticmethod
    def _load_real_runtime(checkpoint_path: str):
        from multiprocessing import Event

        from tinyvllm.config import Config
        from tinyvllm.engine.model_runner import ModelRunner

        config = Config(
            model=checkpoint_path,
            tensor_parallel_size=1,
            enforce_eager=True,
            qwen35_mtp_enabled=True,
            qwen35_mtp_max_proposal_tokens=4,
            qwen35_mtp_cuda_graphs=True,
            qwen35_mtp_cuda_graph_q_allowlist=(2, 3, 4),
            qwen35_mtp_cuda_graph_batch_allowlist=(1, 4),
            kv_offload_mvp0=False,
        )
        runner = ModelRunner(
            config,
            rank=0,
            event=Event(),
        )
        module = getattr(runner, "qwen35_mtp_module", None)
        executor = getattr(runner, "qwen35_mtp_executor", None)
        physical_store = getattr(
            runner,
            "qwen35_mtp_physical_store",
            None,
        )
        registration_error = getattr(
            runner,
            "qwen35_mtp_registration_error",
            None,
        )
        target_model = getattr(runner, "model", None)
        loader_passed = (
            target_model is not None
            and module is not None
            and registration_error is None
        )
        blockers = {}
        eager_reference_probe = None
        graph_eager_probe = None
        transaction_probe = None
        if (
            loader_passed
            and executor is not None
            and physical_store is not None
        ):
            try:
                eager_reference_probe = (
                    _build_real_eager_reference_probe(
                        executor=executor,
                        module=module,
                        physical_store=physical_store,
                        hf_config=config.hf_config,
                    )
                )
            except BaseException as error:
                blockers["eager_reference"] = (
                    f"{type(error).__name__}: {error}"
                )
            try:
                graph_eager_probe = _build_real_graph_eager_probe(
                    executor=executor,
                    module=module,
                    physical_store=physical_store,
                    hf_config=config.hf_config,
                )
            except BaseException as error:
                blockers["graph_eager"] = (
                    f"{type(error).__name__}: {error}"
                )
            try:
                transaction_probe = _build_real_transaction_probe(
                    executor=executor,
                    module=module,
                    physical_store=physical_store,
                    hf_config=config.hf_config,
                )
            except BaseException as error:
                blockers["transaction"] = (
                    f"{type(error).__name__}: {error}"
                )
        else:
            blockers["eager_reference"] = (
                "real MTP executor and physical KV store "
                "are unavailable"
            )
            blockers["transaction"] = (
                "real MTP executor and physical KV store "
                "are unavailable"
            )
            blockers["graph_eager"] = (
                "real MTP executor and physical KV store "
                "are unavailable"
            )
        if registration_error is not None:
            blockers["load"] = registration_error
        runtime = {
            "runner": runner,
            "loader_passed": loader_passed,
            "shared_embedding_identity": (
                module is not None
                and module.embed_tokens
                is getattr(target_model, "embed_tokens", None)
            ),
            "shared_lm_head_identity": (
                module is not None
                and module.lm_head
                is getattr(target_model, "lm_head", None)
            ),
            "config_tensor_contract_passed": (
                loader_passed
                and getattr(config.hf_config, "model_type", None)
                == "qwen3_5"
            ),
            "blockers": blockers,
        }
        if eager_reference_probe is not None:
            runtime["eager_reference_probe"] = (
                eager_reference_probe
            )
        if graph_eager_probe is not None:
            runtime["graph_eager_probe"] = graph_eager_probe
        if transaction_probe is not None:
            runtime["transaction_probe"] = transaction_probe
        return runtime

    def _record_blocker(self, domain: str, reason: object) -> None:
        if domain not in _BACKEND_FAILURE_DOMAINS:
            raise ValueError("unsupported backend failure domain")
        text = str(reason).strip()
        if not text:
            text = "unspecified backend failure"
        self._blockers.setdefault(domain, text)

    def load(self, checkpoint_path: str):
        manifest_before = checkpoint_manifest_sha256(checkpoint_path)
        try:
            metadata = self._runtime_metadata_loader()
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    "runtime metadata loader must return a mapping"
                )
        except BaseException as error:
            metadata = {
                "device_name": None,
                "torch_version": None,
                "cuda_version": None,
            }
            self._record_blocker(
                "load",
                f"{type(error).__name__}: {error}",
            )
        try:
            runtime = self._runtime_loader(checkpoint_path)
            self._runtime = runtime
            runtime_blockers = _runtime_field(
                runtime,
                "blockers",
                {},
            )
            if not isinstance(runtime_blockers, Mapping):
                raise ValueError("runtime blockers must be a mapping")
            for domain, reason in runtime_blockers.items():
                self._record_blocker(domain, reason)
        except BaseException as error:
            self._runtime = None
            self._record_blocker(
                "load",
                f"{type(error).__name__}: {error}",
            )
        manifest_after = checkpoint_manifest_sha256(checkpoint_path)
        runtime = self._runtime
        return {
            "checkpoint_manifest_sha256": manifest_before,
            "device_name": metadata.get("device_name"),
            "torch_version": metadata.get("torch_version"),
            "cuda_version": metadata.get("cuda_version"),
            "loader_passed": (
                _runtime_field(runtime, "loader_passed", False)
                is True
                and "load" not in self._blockers
            ),
            "shared_embedding_identity": (
                _runtime_field(
                    runtime,
                    "shared_embedding_identity",
                    False,
                )
                is True
            ),
            "shared_lm_head_identity": (
                _runtime_field(
                    runtime,
                    "shared_lm_head_identity",
                    False,
                )
                is True
            ),
            "checkpoint_unchanged": (
                manifest_before == manifest_after
            ),
            "config_tensor_contract_passed": (
                _runtime_field(
                    runtime,
                    "config_tensor_contract_passed",
                    False,
                )
                is True
            ),
        }

    def compare_eager_reference(self, q: int, batch_size: int):
        probe = _runtime_field(
            self._runtime,
            "eager_reference_probe",
        )
        if not callable(probe):
            self._record_blocker(
                "eager_reference",
                "real eager/reference probe is unavailable",
            )
            return {
                "max_abs_diff": 0.0,
                "argmax_equal": False,
            }
        try:
            return probe(q, batch_size)
        except BaseException as error:
            self._record_blocker(
                "eager_reference",
                f"{type(error).__name__}: {error}",
            )
            return {
                "max_abs_diff": 0.0,
                "argmax_equal": False,
            }

    def compare_graph_eager(self, q: int, batch_size: int):
        probe = _runtime_field(
            self._runtime,
            "graph_eager_probe",
        )
        if not callable(probe):
            self._record_blocker(
                "graph_eager",
                "real exact-Q graph/eager probe is unavailable",
            )
            return {
                "backend_installed": False,
                "capture_count": 0,
                "replay_count": 0,
                "argmax_equal": False,
                "proposal_tokens_equal": False,
                "transaction_commit": False,
                "transaction_rollback": False,
                "replay_failure_quarantined": False,
                "replay_failure_eager_retry_count": 0,
            }
        try:
            return probe(q, batch_size)
        except BaseException as error:
            self._record_blocker(
                "graph_eager",
                f"{type(error).__name__}: {error}",
            )
            return {
                "backend_installed": False,
                "capture_count": 0,
                "replay_count": 0,
                "argmax_equal": False,
                "proposal_tokens_equal": False,
                "transaction_commit": False,
                "transaction_rollback": False,
                "replay_failure_quarantined": False,
                "replay_failure_eager_retry_count": 0,
            }

    def run_transaction_case(
        self,
        q: int,
        batch_size: int,
        accepted: int,
    ):
        probe = _runtime_field(
            self._runtime,
            "transaction_probe",
        )
        if callable(probe):
            try:
                return probe(q, batch_size, accepted)
            except BaseException as error:
                self._record_blocker(
                    "transaction",
                    f"{type(error).__name__}: {error}",
                )
        else:
            self._record_blocker(
                "transaction",
                "real physical proposal KV transaction probe "
                "is unavailable",
            )
        return {
            "q": q,
            "batch_size": batch_size,
            "accepted_proposal_tokens": accepted,
            "staged_slot_ids": [],
            "committed_slot_ids": [],
            "released_slot_ids": [],
            "accepted_slot_identity_preserved": False,
            "rejected_slots_released": False,
            "post_rollback_continuation_equal": False,
        }

    def report_metadata(self):
        return {
            "backend_failures": [
                {
                    "domain": domain,
                    "reason": self._blockers[domain],
                }
                for domain in _BACKEND_FAILURE_DOMAINS
                if domain in self._blockers
            ]
        }


def parse_integer_csv(value: str, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty CSV string")
    parts = value.split(",")
    if any(not part or not part.isdigit() for part in parts):
        raise ValueError(
            f"{name} must contain only positive integers"
        )
    parsed = tuple(int(part) for part in parts)
    if (
        any(item <= 0 for item in parsed)
        or tuple(sorted(set(parsed))) != parsed
    ):
        raise ValueError(
            f"{name} must be a canonical increasing integer tuple"
        )
    return parsed


def _canonical_positive_integer_tuple(
    value,
    *,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    result = tuple(value)
    if (
        any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in result
        )
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(
            f"{name} must be a canonical increasing integer list"
        )
    return result


def _validate_transaction_case(
    case: object,
    *,
    require_success: bool,
) -> tuple[int, int, int]:
    if not isinstance(case, Mapping):
        raise ValueError("transaction case must be a mapping")
    key = []
    for field in (
        "q",
        "batch_size",
        "accepted_proposal_tokens",
    ):
        value = case.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"transaction case {field} must be nonnegative"
            )
        key.append(value)
    q, batch_size, accepted = key
    if q <= 0 or batch_size <= 0 or accepted > q:
        raise ValueError("transaction case bounds are invalid")
    for field in (
        "staged_slot_ids",
        "committed_slot_ids",
        "released_slot_ids",
    ):
        slot_ids = case.get(field)
        if (
            not isinstance(slot_ids, list)
            or any(
                isinstance(slot_id, bool)
                or not isinstance(slot_id, int)
                or slot_id < 0
                for slot_id in slot_ids
            )
        ):
            raise ValueError(
                f"transaction case {field} must be slot IDs"
            )
    for field in (
        "accepted_slot_identity_preserved",
        "rejected_slots_released",
        "post_rollback_continuation_equal",
    ):
        value = case.get(field)
        if not isinstance(value, bool):
            raise ValueError(
                f"transaction case {field} must be a bool"
            )
        if require_success and value is not True:
            raise ValueError(
                f"transaction case {field} must be true"
            )
    return q, batch_size, accepted


def _validate_coverage(report: Mapping[str, object]) -> None:
    coverage = report.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("coverage must be a mapping")
    tensor_parallel_sizes = coverage.get(
        "tensor_parallel_sizes"
    )
    if tensor_parallel_sizes != [1]:
        raise ValueError("gate must not claim TP4 coverage")
    if coverage.get("kv_offload") is not False:
        raise ValueError("gate must not claim KV offload coverage")
    if coverage.get("architectures") != ["qwen3_5"]:
        raise ValueError(
            "gate must not claim second architecture coverage"
        )
    if coverage.get("long_context") is not False:
        raise ValueError(
            "gate must not claim long-context coverage"
        )
    if coverage.get("performance") is not False:
        raise ValueError("gate must not claim performance coverage")


def validate_gate_report(
    report: object,
    *,
    required_q_values: tuple[int, ...],
    required_batch_sizes: tuple[int, ...],
):
    if not isinstance(report, Mapping):
        raise ValueError("gate report must be a mapping")
    missing = [
        field
        for field in REQUIRED_REPORT_FIELDS
        if field not in report
    ]
    if missing:
        raise ValueError(
            "gate report is missing required fields: "
            + ", ".join(missing)
        )
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version is unsupported")
    required_q_values = _canonical_positive_integer_tuple(
        required_q_values,
        name="required_q_values",
    )
    required_batch_sizes = _canonical_positive_integer_tuple(
        required_batch_sizes,
        name="required_batch_sizes",
    )
    q_values = _canonical_positive_integer_tuple(
        report["q_values"],
        name="q_values",
    )
    batch_sizes = _canonical_positive_integer_tuple(
        report["batch_sizes"],
        name="batch_sizes",
    )
    if q_values != required_q_values:
        raise ValueError("q_values do not cover the required domain")
    if batch_sizes != required_batch_sizes:
        raise ValueError(
            "batch_sizes do not cover the required domain"
        )
    if report["promotion_classification"] != "NOT_PROMOTABLE":
        raise ValueError(
            "promotion_classification must remain NOT_PROMOTABLE"
        )
    _validate_coverage(report)
    limitations = report["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(
            not isinstance(item, str) or not item
            for item in limitations
        )
    ):
        raise ValueError("limitations must be non-empty strings")
    max_abs_diff = report["eager_reference_max_abs_diff"]
    if (
        isinstance(max_abs_diff, bool)
        or not isinstance(max_abs_diff, (int, float))
        or not math.isfinite(max_abs_diff)
        or max_abs_diff < 0
    ):
        raise ValueError(
            "eager_reference_max_abs_diff must be nonnegative"
        )
    for field in ("graph_capture_count", "graph_replay_count"):
        value = report[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"{field} must be a nonnegative integer"
            )
    retry_count = report["replay_failure_eager_retry_count"]
    if (
        isinstance(retry_count, bool)
        or not isinstance(retry_count, int)
        or retry_count < 0
    ):
        raise ValueError(
            "replay_failure_eager_retry_count must be "
            "a nonnegative integer"
        )

    cases = report["transaction_cases"]
    if not isinstance(cases, list):
        raise ValueError("transaction_cases must be a list")
    require_success = report["status"] == "PASS"
    actual_case_domain = tuple(
        _validate_transaction_case(
            case,
            require_success=require_success,
        )
        for case in cases
    )
    expected_case_domain = tuple(
        (q, batch_size, accepted)
        for batch_size in required_batch_sizes
        for q in required_q_values
        for accepted in range(q + 1)
    )
    if actual_case_domain != expected_case_domain:
        raise ValueError(
            "transaction case domain is incomplete or noncanonical"
        )

    if require_success:
        for field in ("graph_capture_count", "graph_replay_count"):
            if report[field] <= 0:
                raise ValueError(f"{field} must be positive for PASS")
        if report["replay_failure_eager_retry_count"] != 0:
            raise ValueError(
                "replay_failure_eager_retry_count "
                "must be zero for PASS"
            )
        for field in (
            "loader_passed",
            "shared_embedding_identity",
            "shared_lm_head_identity",
            "eager_reference_argmax_equal",
            "graph_backend_installed",
            "graph_eager_argmax_equal",
            "graph_eager_proposal_tokens_equal",
            "graph_transaction_commit",
            "graph_transaction_rollback",
            "replay_failure_quarantined",
            "accepted_slot_identity_preserved",
            "rejected_slots_released",
            "post_rollback_continuation_equal",
        ):
            if report[field] is not True:
                raise ValueError(f"{field} must be true for PASS")
    elif report["status"] != "FAIL":
        raise ValueError("status must be PASS or FAIL")
    return report


def _backend_method(backend, name: str):
    method = getattr(backend, name, None)
    if not callable(method):
        raise ValueError(f"backend must expose callable {name}")
    return method


def _backend_report_metadata(backend):
    method = getattr(backend, "report_metadata", None)
    if method is None:
        return {}
    if not callable(method):
        raise ValueError("backend report_metadata must be callable")
    metadata = method()
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "backend report_metadata result must be a mapping"
        )
    return dict(metadata)


def run_gate(
    *,
    checkpoint_path: str,
    q_values: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    backend,
):
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(
            "checkpoint_path must be a non-empty string"
        )
    q_values = _canonical_positive_integer_tuple(
        q_values,
        name="q_values",
    )
    batch_sizes = _canonical_positive_integer_tuple(
        batch_sizes,
        name="batch_sizes",
    )
    if q_values != REQUIRED_Q_VALUES:
        raise ValueError(
            "q_values must equal the required real gate domain"
        )
    if batch_sizes != REQUIRED_BATCH_SIZES:
        raise ValueError(
            "batch_sizes must equal the required real gate domain"
        )
    load = _backend_method(backend, "load")
    compare_eager_reference = _backend_method(
        backend,
        "compare_eager_reference",
    )
    compare_graph_eager = _backend_method(
        backend,
        "compare_graph_eager",
    )
    run_transaction_case = _backend_method(
        backend,
        "run_transaction_case",
    )
    loaded = load(checkpoint_path)
    if not isinstance(loaded, Mapping):
        raise ValueError("backend load result must be a mapping")

    eager_reference_max_abs_diff = 0.0
    eager_reference_argmax_equal = True
    graph_backend_installed = True
    graph_capture_count = 0
    graph_replay_count = 0
    graph_eager_argmax_equal = True
    graph_eager_proposal_tokens_equal = True
    graph_transaction_commit = True
    graph_transaction_rollback = True
    replay_failure_quarantined = False
    replay_failure_eager_retry_count = 0
    graph_family_counts_valid = True
    transaction_cases = []
    for batch_size in batch_sizes:
        for q in q_values:
            eager_result = compare_eager_reference(q, batch_size)
            if not isinstance(eager_result, Mapping):
                raise ValueError(
                    "eager/reference result must be a mapping"
                )
            max_abs_diff = eager_result.get("max_abs_diff")
            if (
                isinstance(max_abs_diff, bool)
                or not isinstance(max_abs_diff, (int, float))
                or not math.isfinite(max_abs_diff)
                or max_abs_diff < 0
            ):
                raise ValueError(
                    "eager/reference max_abs_diff is invalid"
                )
            eager_reference_max_abs_diff = max(
                eager_reference_max_abs_diff,
                float(max_abs_diff),
            )
            eager_reference_argmax_equal = (
                eager_reference_argmax_equal
                and eager_result.get("argmax_equal") is True
            )
            graph_result = compare_graph_eager(q, batch_size)
            if not isinstance(graph_result, Mapping):
                raise ValueError(
                    "graph/eager result must be a mapping"
                )
            graph_eager_argmax_equal = (
                graph_eager_argmax_equal
                and graph_result.get("argmax_equal") is True
            )
            graph_backend_installed = (
                graph_backend_installed
                and graph_result.get("backend_installed") is True
            )
            for field in ("capture_count", "replay_count"):
                value = graph_result.get(field, 0)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"graph/eager {field} is invalid"
                    )
            graph_capture_count += graph_result.get(
                "capture_count",
                0,
            )
            graph_replay_count += graph_result.get(
                "replay_count",
                0,
            )
            if q >= 2:
                graph_family_counts_valid = (
                    graph_family_counts_valid
                    and graph_result.get("capture_count", 0) > 0
                    and graph_result.get("replay_count", 0) > 0
                )
            graph_eager_proposal_tokens_equal = (
                graph_eager_proposal_tokens_equal
                and graph_result.get("proposal_tokens_equal") is True
            )
            graph_transaction_commit = (
                graph_transaction_commit
                and graph_result.get("transaction_commit") is True
            )
            graph_transaction_rollback = (
                graph_transaction_rollback
                and graph_result.get("transaction_rollback") is True
            )
            replay_failure_quarantined = (
                replay_failure_quarantined
                or graph_result.get("replay_failure_quarantined")
                is True
            )
            retry_count = graph_result.get(
                "replay_failure_eager_retry_count",
                0,
            )
            if (
                isinstance(retry_count, bool)
                or not isinstance(retry_count, int)
                or retry_count < 0
            ):
                raise ValueError(
                    "graph/eager replay failure retry count "
                    "is invalid"
                )
            replay_failure_eager_retry_count += retry_count
            for accepted in range(q + 1):
                transaction_cases.append(
                    run_transaction_case(
                        q,
                        batch_size,
                        accepted,
                    )
                )

    accepted_slot_identity_preserved = all(
        isinstance(case, Mapping)
        and case.get("accepted_slot_identity_preserved") is True
        for case in transaction_cases
    )
    rejected_slots_released = all(
        isinstance(case, Mapping)
        and case.get("rejected_slots_released") is True
        for case in transaction_cases
    )
    post_rollback_continuation_equal = all(
        isinstance(case, Mapping)
        and case.get("post_rollback_continuation_equal") is True
        for case in transaction_cases
    )
    correctness = (
        loaded.get("loader_passed") is True
        and loaded.get("shared_embedding_identity") is True
        and loaded.get("shared_lm_head_identity") is True
        and loaded.get("checkpoint_unchanged") is True
        and loaded.get("config_tensor_contract_passed") is True
        and eager_reference_argmax_equal
        and graph_backend_installed
        and graph_capture_count > 0
        and graph_replay_count > 0
        and graph_family_counts_valid
        and graph_eager_argmax_equal
        and graph_eager_proposal_tokens_equal
        and graph_transaction_commit
        and graph_transaction_rollback
        and replay_failure_quarantined
        and replay_failure_eager_retry_count == 0
        and accepted_slot_identity_preserved
        and rejected_slots_released
        and post_rollback_continuation_equal
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": checkpoint_path,
        "checkpoint_manifest_sha256": loaded.get(
            "checkpoint_manifest_sha256"
        ),
        "device_name": loaded.get("device_name"),
        "torch_version": loaded.get("torch_version"),
        "cuda_version": loaded.get("cuda_version"),
        "q_values": list(q_values),
        "batch_sizes": list(batch_sizes),
        "loader_passed": loaded.get("loader_passed") is True,
        "shared_embedding_identity": (
            loaded.get("shared_embedding_identity") is True
        ),
        "shared_lm_head_identity": (
            loaded.get("shared_lm_head_identity") is True
        ),
        "eager_reference_max_abs_diff": (
            eager_reference_max_abs_diff
        ),
        "eager_reference_argmax_equal": (
            eager_reference_argmax_equal
        ),
        "graph_backend_installed": graph_backend_installed,
        "graph_capture_count": graph_capture_count,
        "graph_replay_count": graph_replay_count,
        "graph_eager_argmax_equal": graph_eager_argmax_equal,
        "graph_eager_proposal_tokens_equal": (
            graph_eager_proposal_tokens_equal
        ),
        "graph_transaction_commit": graph_transaction_commit,
        "graph_transaction_rollback": graph_transaction_rollback,
        "replay_failure_quarantined": (
            replay_failure_quarantined
        ),
        "replay_failure_eager_retry_count": (
            replay_failure_eager_retry_count
        ),
        "transaction_cases": transaction_cases,
        "accepted_slot_identity_preserved": (
            accepted_slot_identity_preserved
        ),
        "rejected_slots_released": rejected_slots_released,
        "post_rollback_continuation_equal": (
            post_rollback_continuation_equal
        ),
        "status": "PASS" if correctness else "FAIL",
        "promotion_classification": "NOT_PROMOTABLE",
        "limitations": [
            "TP1 only",
            "KV offload disabled",
            "single Qwen3.5 architecture",
            "no long-context coverage",
            "no performance claim",
        ],
        "coverage": {
            "tensor_parallel_sizes": [1],
            "kv_offload": False,
            "architectures": ["qwen3_5"],
            "long_context": False,
            "performance": False,
        },
        "checkpoint_unchanged": (
            loaded.get("checkpoint_unchanged") is True
        ),
        "config_tensor_contract_passed": (
            loaded.get("config_tensor_contract_passed") is True
        ),
    }
    report.update(_backend_report_metadata(backend))
    validate_gate_report(
        report,
        required_q_values=REQUIRED_Q_VALUES,
        required_batch_sizes=REQUIRED_BATCH_SIZES,
    )
    return report


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--q-values",
        default=",".join(str(value) for value in REQUIRED_Q_VALUES),
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(
            str(value) for value in REQUIRED_BATCH_SIZES
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None, *, backend_factory=RealQwen35MTPGateBackend) -> int:
    if not callable(backend_factory):
        raise ValueError("backend_factory must be callable")
    arguments = _build_parser().parse_args(argv)
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("gate output already exists")
    report = run_gate(
        checkpoint_path=arguments.checkpoint,
        q_values=parse_integer_csv(
            arguments.q_values,
            name="q_values",
        ),
        batch_sizes=parse_integer_csv(
            arguments.batch_sizes,
            name="batch_sizes",
        ),
        backend=backend_factory(),
    )
    _write_json_atomic(output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
