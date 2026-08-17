from __future__ import annotations

from dataclasses import dataclass
import time

from tinyvllm.engine.qwen35_mtp_graph import (
    Qwen35MTPGraphEntry,
    Qwen35MTPGraphPreReplayError,
)
from tinyvllm.speculative.adapter import DraftProposal


@dataclass
class Qwen35MTPCudaGraphTensors:
    first_tokens: object
    current_tokens: object
    positions: object
    initial_hidden: object
    current_hidden: object
    next_hidden: object
    slot_mapping: object
    context_lens: object
    block_tables: object
    proposal_tokens: object


@dataclass
class Qwen35MTPCudaGraphPayload:
    graph: object
    tensors: Qwen35MTPCudaGraphTensors


class Qwen35MTPCudaGraphBackend:

    def __init__(
        self,
        *,
        module,
        proposal_kv_cache,
        device,
        compute_dtype,
        hidden_size: int,
        block_table_width: int,
        torch_module=None,
        temporary_context_factory=None,
    ):
        if torch_module is None:
            import torch as torch_module
        if temporary_context_factory is None:
            from tinyvllm.utils.context import (
                temporary_context as temporary_context_factory,
            )
        for method in (
            "begin",
            "abort",
            "committed_slot_ids",
            "mark_materialized",
        ):
            if not callable(
                getattr(proposal_kv_cache, method, None)
            ):
                raise ValueError(
                    "proposal_kv_cache must expose callable "
                    f"{method}"
                )
        for value, name in (
            (hidden_size, "hidden_size"),
            (block_table_width, "block_table_width"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        self.module = module
        self.proposal_kv_cache = proposal_kv_cache
        self.torch = torch_module
        self.device = device
        self.compute_dtype = compute_dtype
        self.hidden_size = hidden_size
        self.block_table_width = block_table_width
        self.temporary_context = temporary_context_factory
        self.graph_pool = torch_module.cuda.graph_pool_handle()
        self.capture_stream = torch_module.cuda.Stream()

    def _validate_identity(self, identity) -> None:
        if identity.device_index != 0:
            raise Qwen35MTPGraphPreReplayError(
                "graph identity device does not match backend"
            )
        if identity.compute_dtype != str(self.compute_dtype):
            raise Qwen35MTPGraphPreReplayError(
                "graph identity dtype does not match backend"
            )
        if identity.hidden_size != self.hidden_size:
            raise Qwen35MTPGraphPreReplayError(
                "graph identity hidden size does not match backend"
            )
        if identity.mtp_layer_count != 1:
            raise Qwen35MTPGraphPreReplayError(
                "graph identity MTP layer count is unsupported"
            )
        if identity.block_table_width != self.block_table_width:
            raise Qwen35MTPGraphPreReplayError(
                "graph identity block table width does not match backend"
            )

    def _allocate_tensors(self, identity):
        self._validate_identity(identity)
        batch_size = identity.exact_batch_size
        step_count = identity.exact_q - 1
        torch = self.torch
        return Qwen35MTPCudaGraphTensors(
            first_tokens=torch.zeros(
                batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            current_tokens=torch.zeros(
                batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            positions=torch.zeros(
                step_count,
                batch_size,
                dtype=torch.int64,
                device=self.device,
            ),
            initial_hidden=torch.zeros(
                batch_size,
                self.hidden_size,
                dtype=self.compute_dtype,
                device=self.device,
            ),
            current_hidden=torch.zeros(
                batch_size,
                self.hidden_size,
                dtype=self.compute_dtype,
                device=self.device,
            ),
            next_hidden=torch.zeros(
                batch_size,
                self.hidden_size,
                dtype=self.compute_dtype,
                device=self.device,
            ),
            slot_mapping=torch.zeros(
                step_count,
                batch_size,
                dtype=torch.int32,
                device=self.device,
            ),
            context_lens=torch.zeros(
                step_count,
                batch_size,
                dtype=torch.int32,
                device=self.device,
            ),
            block_tables=torch.zeros(
                step_count,
                batch_size,
                self.block_table_width,
                dtype=torch.int32,
                device=self.device,
            ),
            proposal_tokens=torch.zeros(
                batch_size,
                identity.exact_q,
                dtype=torch.int64,
                device=self.device,
            ),
        )

    def estimate_static_bytes(self, identity, rows) -> int:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError("rows must be a non-empty tuple")
        tensors = self._allocate_tensors(identity)
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors.__dict__.values()
        )

    def _abort_transactions(self, transactions) -> None:
        first_error = None
        for transaction in reversed(tuple(transactions)):
            if transaction.state not in ("reserved", "materialized"):
                continue
            try:
                self.proposal_kv_cache.abort(
                    transaction.transaction_id
                )
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _prepare_live_replay(self, identity, tensors, rows):
        self._validate_identity(identity)
        if (
            not isinstance(rows, tuple)
            or len(rows) != identity.exact_batch_size
        ):
            raise Qwen35MTPGraphPreReplayError(
                "live row count does not match exact batch size"
            )
        step_count = identity.exact_q - 1
        transactions = []
        first_tokens = []
        positions = [[] for _ in range(step_count)]
        slot_mapping = [[] for _ in range(step_count)]
        context_lens = [[] for _ in range(step_count)]
        block_tables = [[] for _ in range(step_count)]
        hidden_rows = []
        proposal_tokens = []
        try:
            for input_row, bootstrap in rows:
                hidden = input_row.target_hidden
                if (
                    tuple(hidden.shape) != (1, self.hidden_size)
                    or hidden.dtype != self.compute_dtype
                    or hidden.device != self.device
                ):
                    raise Qwen35MTPGraphPreReplayError(
                        "live hidden input does not match graph identity"
                    )
                transaction = self.proposal_kv_cache.begin(
                    input_row.sequence_id,
                    bootstrap.sequence_epoch,
                    step_count,
                )
                transactions.append(transaction)
                committed_slots = tuple(
                    self.proposal_kv_cache.committed_slot_ids(
                        input_row.sequence_id
                    )
                )
                first_tokens.append(input_row.first_target_token)
                hidden_rows.append(hidden)
                proposal_tokens.append(
                    [input_row.first_target_token]
                    + [0] * step_count
                )
                start_position = max(
                    len(input_row.token_ids) - 1,
                    0,
                )
                for step in range(step_count):
                    visible_slots = (
                        committed_slots
                        + transaction.staged_slot_ids[:step + 1]
                    )
                    if len(visible_slots) > self.block_table_width:
                        raise Qwen35MTPGraphPreReplayError(
                            "live block table exceeds graph width"
                        )
                    positions[step].append(
                        start_position + step
                    )
                    slot_mapping[step].append(
                        transaction.staged_slot_ids[step]
                    )
                    context_lens[step].append(
                        len(visible_slots)
                    )
                    block_tables[step].append(
                        list(visible_slots)
                        + [0] * (
                            self.block_table_width
                            - len(visible_slots)
                        )
                    )

            torch = self.torch
            first_token_tensor = torch.tensor(
                first_tokens,
                dtype=torch.int64,
                device=self.device,
            )
            tensors.first_tokens.copy_(first_token_tensor)
            tensors.current_tokens.copy_(first_token_tensor)
            tensors.positions.copy_(
                torch.tensor(
                    positions,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
            hidden_tensor = torch.cat(
                tuple(hidden_rows),
                dim=0,
            )
            tensors.initial_hidden.copy_(hidden_tensor)
            tensors.current_hidden.copy_(hidden_tensor)
            tensors.slot_mapping.copy_(
                torch.tensor(
                    slot_mapping,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            tensors.context_lens.copy_(
                torch.tensor(
                    context_lens,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            tensors.block_tables.copy_(
                torch.tensor(
                    block_tables,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            tensors.proposal_tokens.copy_(
                torch.tensor(
                    proposal_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
        except BaseException as error:
            self._abort_transactions(transactions)
            if isinstance(error, Qwen35MTPGraphPreReplayError):
                raise
            raise Qwen35MTPGraphPreReplayError(
                "failed to prepare graph static input"
            ) from error
        return tuple(transactions)

    def _prepare_scratch_capture(
        self,
        identity,
        tensors,
        rows,
    ) -> None:
        self._validate_identity(identity)
        if (
            not isinstance(rows, tuple)
            or len(rows) != identity.exact_batch_size
        ):
            raise ValueError(
                "scratch row count does not match exact batch size"
            )
        step_count = identity.exact_q - 1
        first_tokens = []
        positions = [[] for _ in range(step_count)]
        slot_mapping = [[] for _ in range(step_count)]
        context_lens = [[] for _ in range(step_count)]
        block_tables = [[] for _ in range(step_count)]
        hidden_rows = []
        proposal_tokens = []
        for row in rows:
            input_row = row.input_row
            transaction = row.transaction
            hidden = input_row.target_hidden
            if (
                tuple(hidden.shape) != (1, self.hidden_size)
                or hidden.dtype != self.compute_dtype
                or hidden.device != self.device
                or len(transaction.staged_slot_ids) != step_count
            ):
                raise ValueError(
                    "scratch row does not match graph identity"
                )
            first_tokens.append(input_row.first_target_token)
            hidden_rows.append(hidden)
            proposal_tokens.append(
                [input_row.first_target_token]
                + [0] * step_count
            )
            start_position = max(
                len(input_row.token_ids) - 1,
                0,
            )
            committed_slots = tuple(
                row.source_committed_slot_ids
            )
            for step in range(step_count):
                visible_slots = (
                    committed_slots
                    + transaction.staged_slot_ids[:step + 1]
                )
                if len(visible_slots) > self.block_table_width:
                    raise ValueError(
                        "scratch block table exceeds graph width"
                    )
                positions[step].append(start_position + step)
                slot_mapping[step].append(
                    transaction.staged_slot_ids[step]
                )
                context_lens[step].append(len(visible_slots))
                block_tables[step].append(
                    list(visible_slots)
                    + [0] * (
                        self.block_table_width
                        - len(visible_slots)
                    )
                )
        torch = self.torch
        first_token_tensor = torch.tensor(
            first_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        tensors.first_tokens.copy_(first_token_tensor)
        tensors.current_tokens.copy_(first_token_tensor)
        tensors.positions.copy_(
            torch.tensor(
                positions,
                dtype=torch.int64,
                device=self.device,
            )
        )
        hidden_tensor = torch.cat(tuple(hidden_rows), dim=0)
        tensors.initial_hidden.copy_(hidden_tensor)
        tensors.current_hidden.copy_(hidden_tensor)
        tensors.slot_mapping.copy_(
            torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                device=self.device,
            )
        )
        tensors.context_lens.copy_(
            torch.tensor(
                context_lens,
                dtype=torch.int32,
                device=self.device,
            )
        )
        tensors.block_tables.copy_(
            torch.tensor(
                block_tables,
                dtype=torch.int32,
                device=self.device,
            )
        )
        tensors.proposal_tokens.copy_(
            torch.tensor(
                proposal_tokens,
                dtype=torch.int64,
                device=self.device,
            )
        )

    def _run_static_chain(self, identity, tensors) -> None:
        torch = self.torch
        tensors.current_tokens.copy_(tensors.first_tokens)
        tensors.current_hidden.copy_(tensors.initial_hidden)
        tensors.proposal_tokens[:, 0].copy_(
            tensors.first_tokens
        )
        for step in range(identity.exact_q - 1):
            with self.temporary_context(
                mode="decode",
                is_prefill=False,
                slot_mapping=tensors.slot_mapping[step],
                context_lens=tensors.context_lens[step],
                block_tables=tensors.block_tables[step],
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=1,
                max_seqlen_k=self.block_table_width,
                quest_top_k_blocks=-1,
                am_compact_blocks=0,
                kv_offload_manager=None,
                kv_offload_blockwise_decode=False,
                kv_offload_blockwise_prefill=False,
                force_attention_backend=True,
            ):
                next_hidden, logits = self.module.forward_step(
                    tensors.current_tokens,
                    tensors.positions[step],
                    tensors.current_hidden,
                )
            tensors.next_hidden.copy_(next_hidden)
            next_tokens = torch.argmax(logits, dim=-1)
            tensors.proposal_tokens[:, step + 1].copy_(
                next_tokens
            )
            tensors.current_tokens.copy_(next_tokens)
            tensors.current_hidden.copy_(
                tensors.next_hidden
            )

    def capture(
        self,
        identity,
        rows,
        eager,
        scratch_lease,
    ):
        if tuple(rows) != tuple(scratch_lease.rows):
            raise ValueError(
                "capture rows must belong to scratch lease"
            )
        torch = self.torch
        tensors = self._allocate_tensors(identity)
        self._prepare_scratch_capture(
            identity,
            tensors,
            rows,
        )
        static_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors.__dict__.values()
        )
        torch.cuda.synchronize()
        allocated_before = int(torch.cuda.memory_allocated())
        reserved_before = int(torch.cuda.memory_reserved())
        self._run_static_chain(identity, tensors)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture_started_ns = time.perf_counter_ns()
        with torch.cuda.graph(
            graph,
            pool=self.graph_pool,
            stream=self.capture_stream,
        ):
            self._run_static_chain(identity, tensors)
        torch.cuda.synchronize()
        capture_duration_ns = (
            time.perf_counter_ns() - capture_started_ns
        )
        allocated_after = int(torch.cuda.memory_allocated())
        reserved_after = int(torch.cuda.memory_reserved())
        return Qwen35MTPGraphEntry(
            identity=identity,
            graph=Qwen35MTPCudaGraphPayload(
                graph=graph,
                tensors=tensors,
            ),
            static_bytes=static_bytes,
            capture_duration_ns=capture_duration_ns,
            reserved_delta_bytes=max(
                0,
                reserved_after - reserved_before,
                allocated_after - allocated_before,
            ),
        )

    def replay(self, entry, rows):
        if not isinstance(entry, Qwen35MTPGraphEntry):
            raise Qwen35MTPGraphPreReplayError(
                "graph entry type is invalid"
            )
        payload = entry.graph
        if not isinstance(payload, Qwen35MTPCudaGraphPayload):
            raise Qwen35MTPGraphPreReplayError(
                "graph payload type is invalid"
            )
        graph_replay = getattr(payload.graph, "replay", None)
        if not callable(graph_replay):
            raise Qwen35MTPGraphPreReplayError(
                "graph payload has no replay method"
            )
        identity = entry.identity
        transactions = self._prepare_live_replay(
            identity,
            payload.tensors,
            rows,
        )
        try:
            graph_replay()
            proposal_tokens = payload.tensors.proposal_tokens
            if (
                tuple(proposal_tokens.shape)
                != (
                    identity.exact_batch_size,
                    identity.exact_q,
                )
                or proposal_tokens.dtype != self.torch.int64
                or proposal_tokens.device != self.device
            ):
                raise RuntimeError(
                    "Qwen3.5 MTP CUDA graph output is invalid"
                )
            token_rows = proposal_tokens.tolist()
            if (
                not isinstance(token_rows, list)
                or len(token_rows) != len(rows)
                or any(
                    not isinstance(token_row, list)
                    or len(token_row) != identity.exact_q
                    or any(
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        for token_id in token_row
                    )
                    for token_row in token_rows
                )
            ):
                raise RuntimeError(
                    "Qwen3.5 MTP CUDA graph output tokens "
                    "are invalid"
                )
            for transaction in transactions:
                self.proposal_kv_cache.mark_materialized(
                    transaction,
                    identity.exact_q - 1,
                )
            return tuple(
                DraftProposal(
                    sequence_id=input_row.sequence_id,
                    token_ids=tuple(token_row),
                    source_type="native_model_runner",
                    metadata={
                        "exact_q": identity.exact_q,
                        "staged_entry_count": (
                            identity.exact_q - 1
                        ),
                        "execution_mode": "cuda_graph",
                    },
                    proposal_transaction_id=(
                        transaction.transaction_id
                    ),
                )
                for (
                    input_row,
                    _,
                ), transaction, token_row in zip(
                    rows,
                    transactions,
                    token_rows,
                )
            )
        except BaseException:
            self._abort_transactions(transactions)
            raise

