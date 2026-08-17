from __future__ import annotations

from dataclasses import dataclass
import time

from tinyvllm.engine.autoregressive_draft_executor import (
    AutoregressiveDraftGroupExecution,
)
from tinyvllm.engine.autoregressive_draft_graph import (
    AutoregressiveDraftGraphEntry,
    AutoregressiveDraftGraphPreReplayError,
)


@dataclass
class Qwen3DraftCudaGraphTensors:
    first_tokens: object
    current_tokens: object
    next_tokens: object
    positions: object
    slot_mapping: object
    context_lens: object
    block_tables: object
    proposal_tokens: object


@dataclass
class Qwen3DraftCudaGraphPayload:
    graph: object
    tensors: Qwen3DraftCudaGraphTensors


@dataclass
class _PreparedRows:
    transactions: tuple
    read_leases: tuple
    write_leases: tuple
    completed: bool = False


class Qwen3DraftCudaGraphBackend:

    def __init__(
        self,
        *,
        backend,
        proposal_kv_cache,
        device,
        compute_dtype,
        block_table_width: int,
        torch_module=None,
        broadcast=None,
    ):
        if torch_module is None:
            import torch as torch_module
        if broadcast is None:
            import torch.distributed as distributed
            broadcast = distributed.broadcast
        if not callable(
            getattr(backend, "decode_step_static", None)
        ):
            raise ValueError(
                "backend must expose callable decode_step_static"
            )
        for method in (
            "begin",
            "abort",
            "committed_entry_identities",
            "sequence_state",
        ):
            if not callable(
                getattr(proposal_kv_cache, method, None)
            ):
                raise ValueError(
                    "proposal_kv_cache must expose callable "
                    f"{method}"
                )
        if (
            isinstance(block_table_width, bool)
            or not isinstance(block_table_width, int)
            or block_table_width <= 0
        ):
            raise ValueError(
                "block_table_width must be a positive integer"
            )
        if not callable(broadcast):
            raise ValueError("broadcast must be callable")
        self.backend = backend
        self.proposal_kv_cache = proposal_kv_cache
        self.device = device
        self.compute_dtype = compute_dtype
        self.block_table_width = block_table_width
        self.torch = torch_module
        self.broadcast = broadcast
        self.graph_pool = torch_module.cuda.graph_pool_handle()
        self.capture_stream = torch_module.cuda.Stream()

    def _validate_identity(self, identity) -> None:
        if (
            identity.exact_q != 4
            or identity.exact_batch_size != 4
            or identity.tensor_parallel_size != 4
        ):
            raise AutoregressiveDraftGraphPreReplayError(
                "Qwen3 draft graph requires exact TP4/B4/Q4"
            )
        if identity.tensor_parallel_rank != (
            self.backend.tensor_parallel_rank
        ):
            raise AutoregressiveDraftGraphPreReplayError(
                "graph identity rank does not match backend"
            )
        if identity.compute_dtype != str(self.compute_dtype):
            raise AutoregressiveDraftGraphPreReplayError(
                "graph identity dtype does not match backend"
            )
        if identity.kv_block_table_width != (
            self.block_table_width
        ):
            raise AutoregressiveDraftGraphPreReplayError(
                "graph identity block table width does not match backend"
            )
        if identity.blockwise_offload:
            raise AutoregressiveDraftGraphPreReplayError(
                "Qwen3 draft graph does not support offload"
            )

    def _allocate_tensors(self, identity):
        self._validate_identity(identity)
        torch = self.torch
        batch_size = identity.exact_batch_size
        step_count = identity.exact_q - 1
        return Qwen3DraftCudaGraphTensors(
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
            next_tokens=torch.zeros(
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

    @staticmethod
    def _static_bytes(tensors) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors.__dict__.values()
        )

    def estimate_static_bytes(self, identity, rows) -> int:
        if (
            not isinstance(rows, tuple)
            or len(rows) != identity.exact_batch_size
        ):
            raise ValueError(
                "row count must match exact batch size"
            )
        return self._static_bytes(
            self._allocate_tensors(identity)
        )

    def _copy_metadata(
        self,
        tensors,
        *,
        first_tokens,
        positions,
        slot_mapping,
        context_lens,
        block_tables,
    ) -> None:
        torch = self.torch
        first_tensor = torch.tensor(
            first_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        tensors.first_tokens.copy_(first_tensor)
        tensors.current_tokens.copy_(first_tensor)
        tensors.next_tokens.zero_()
        tensors.positions.copy_(torch.tensor(
            positions,
            dtype=torch.int64,
            device=self.device,
        ))
        tensors.slot_mapping.copy_(torch.tensor(
            slot_mapping,
            dtype=torch.int32,
            device=self.device,
        ))
        tensors.context_lens.copy_(torch.tensor(
            context_lens,
            dtype=torch.int32,
            device=self.device,
        ))
        tensors.block_tables.copy_(torch.tensor(
            block_tables,
            dtype=torch.int32,
            device=self.device,
        ))
        tensors.proposal_tokens.zero_()
        tensors.proposal_tokens[:, 0].copy_(first_tensor)

    def _prepare(
        self,
        identity,
        tensors,
        *,
        indexed_rows,
        transactions,
        allocator,
        source_slot_rows,
    ) -> _PreparedRows:
        step_count = identity.exact_q - 1
        first_tokens = []
        positions = [[] for _ in range(step_count)]
        slot_mapping = [[] for _ in range(step_count)]
        context_lens = [[] for _ in range(step_count)]
        block_tables = [[] for _ in range(step_count)]
        write_leases = []
        try:
            for indexed_row, transaction, source_slots in zip(
                indexed_rows,
                transactions,
                source_slot_rows,
            ):
                if (
                    not isinstance(indexed_row, tuple)
                    or len(indexed_row) != 3
                ):
                    raise AutoregressiveDraftGraphPreReplayError(
                        "graph indexed row is invalid"
                    )
                _, input_row, context_count = indexed_row
                if len(source_slots) != context_count:
                    raise AutoregressiveDraftGraphPreReplayError(
                        "graph source context length is invalid"
                    )
                if len(
                    transaction.staged_entry_identities
                ) != step_count:
                    raise AutoregressiveDraftGraphPreReplayError(
                        "graph transaction width is invalid"
                    )
                first_tokens.append(
                    input_row.first_target_token
                )
                staged_slot_ids = []
                for step in range(step_count):
                    write_lease = allocator.ensure_writable((
                        transaction.staged_entry_identities[step],
                    ))
                    write_leases.append(write_lease)
                    staged_slot_ids.append(
                        write_lease.physical_slot_ids[0]
                    )
                    visible_slots = (
                        tuple(source_slots)
                        + tuple(staged_slot_ids)
                    )
                    if len(visible_slots) > self.block_table_width:
                        raise AutoregressiveDraftGraphPreReplayError(
                            "graph block table exceeds exact width"
                        )
                    positions[step].append(
                        context_count + step
                    )
                    slot_mapping[step].append(
                        write_lease.physical_slot_ids[0]
                    )
                    context_lens[step].append(
                        len(visible_slots)
                    )
                    block_tables[step].append(
                        list(visible_slots)
                        + [-1] * (
                            self.block_table_width
                            - len(visible_slots)
                        )
                    )
            self._copy_metadata(
                tensors,
                first_tokens=first_tokens,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )
        except BaseException:
            self._complete_leases(
                allocator,
                (),
                write_leases,
            )
            raise
        return _PreparedRows(
            transactions=tuple(transactions),
            read_leases=(),
            write_leases=tuple(write_leases),
        )

    @staticmethod
    def _complete_leases(
        allocator,
        read_leases,
        write_leases,
    ) -> None:
        first_error = None
        for write_lease in reversed(tuple(write_leases)):
            try:
                allocator.record_write_complete(write_lease)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        for read_lease in reversed(tuple(read_leases)):
            try:
                allocator.record_read_complete(read_lease)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _prepare_scratch_capture(
        self,
        identity,
        tensors,
        scratch_lease,
    ) -> _PreparedRows:
        if (
            len(scratch_lease.rows)
            != identity.exact_batch_size
        ):
            raise ValueError(
                "scratch row count does not match exact batch size"
            )
        scratch_cache = scratch_lease.scratch_cache
        return self._prepare(
            identity,
            tensors,
            indexed_rows=tuple(
                row.indexed_row for row in scratch_lease.rows
            ),
            transactions=tuple(
                row.transaction for row in scratch_lease.rows
            ),
            allocator=scratch_cache.entry_allocator,
            source_slot_rows=tuple(
                row.source_committed_physical_slot_ids
                for row in scratch_lease.rows
            ),
        )

    def _prepare_live_replay(
        self,
        identity,
        tensors,
        rows,
    ) -> _PreparedRows:
        if (
            not isinstance(rows, tuple)
            or len(rows) != identity.exact_batch_size
        ):
            raise AutoregressiveDraftGraphPreReplayError(
                "live row count does not match exact batch size"
            )
        allocator = self.proposal_kv_cache.entry_allocator
        transactions = []
        source_read_leases = []
        try:
            for _, input_row, context_count in rows:
                sequence_state = (
                    self.proposal_kv_cache.sequence_state(
                        input_row.sequence_id
                    )
                )
                if (
                    sequence_state is None
                    or len(
                        sequence_state.committed_entry_identities
                    ) != context_count
                ):
                    raise AutoregressiveDraftGraphPreReplayError(
                        "live committed context does not match row"
                    )
                source_read_lease = allocator.ensure_readable(
                    sequence_state.committed_entry_identities
                )
                source_read_leases.append(source_read_lease)
                transactions.append(
                    self.proposal_kv_cache.begin(
                        input_row.sequence_id,
                        sequence_state.sequence_epoch,
                        identity.exact_q - 1,
                    )
                )
            prepared = self._prepare(
                identity,
                tensors,
                indexed_rows=rows,
                transactions=tuple(transactions),
                allocator=allocator,
                source_slot_rows=tuple(
                    lease.physical_slot_ids
                    for lease in source_read_leases
                ),
            )
        except BaseException as error:
            self._abort_transactions(transactions)
            for lease in reversed(source_read_leases):
                allocator.record_read_complete(lease)
            if isinstance(
                error,
                AutoregressiveDraftGraphPreReplayError,
            ):
                raise
            raise AutoregressiveDraftGraphPreReplayError(
                "failed to prepare live graph replay"
            ) from error
        return _PreparedRows(
            transactions=prepared.transactions,
            read_leases=(
                prepared.read_leases
                + tuple(source_read_leases)
            ),
            write_leases=prepared.write_leases,
        )

    def _run_static_chain(self, identity, tensors) -> None:
        torch = self.torch
        tensors.current_tokens.copy_(tensors.first_tokens)
        tensors.next_tokens.zero_()
        tensors.proposal_tokens.zero_()
        tensors.proposal_tokens[:, 0].copy_(
            tensors.first_tokens
        )
        for step in range(identity.exact_q - 1):
            logits = self.backend.decode_step_static(
                tensors.current_tokens,
                tensors.positions[step],
                tensors.slot_mapping[step],
                tensors.context_lens[step],
                tensors.block_tables[step],
            )
            if self.backend.tensor_parallel_rank == 0:
                if logits is None:
                    raise RuntimeError(
                        "root graph decode produced no logits"
                    )
                torch.argmax(
                    logits,
                    dim=-1,
                    out=tensors.next_tokens,
                )
            else:
                if logits is not None:
                    raise RuntimeError(
                        "non-root graph decode produced logits"
                    )
                tensors.next_tokens.zero_()
            self.broadcast(tensors.next_tokens, src=0)
            tensors.proposal_tokens[:, step + 1].copy_(
                tensors.next_tokens
            )
            tensors.current_tokens.copy_(
                tensors.next_tokens
            )

    def capture(
        self,
        identity,
        rows,
        eager,
        scratch_lease,
    ):
        self._validate_identity(identity)
        if tuple(rows) != tuple(scratch_lease.rows):
            raise ValueError(
                "capture rows must belong to scratch lease"
            )
        tensors = self._allocate_tensors(identity)
        prepared = self._prepare_scratch_capture(
            identity,
            tensors,
            scratch_lease,
        )
        torch = self.torch
        try:
            torch.cuda.synchronize()
            allocated_before = int(
                torch.cuda.memory_allocated()
            )
            reserved_before = int(
                torch.cuda.memory_reserved()
            )
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
                time.perf_counter_ns()
                - capture_started_ns
            )
            allocated_after = int(
                torch.cuda.memory_allocated()
            )
            reserved_after = int(
                torch.cuda.memory_reserved()
            )
        finally:
            self._complete_leases(
                scratch_lease.scratch_cache.entry_allocator,
                prepared.read_leases,
                prepared.write_leases,
            )
        return AutoregressiveDraftGraphEntry(
            identity=identity,
            graph=Qwen3DraftCudaGraphPayload(
                graph=graph,
                tensors=tensors,
            ),
            static_bytes=self._static_bytes(tensors),
            capture_duration_ns=capture_duration_ns,
            reserved_delta_bytes=max(
                0,
                reserved_after - reserved_before,
                allocated_after - allocated_before,
            ),
        )

    def _abort_transactions(self, transactions) -> None:
        first_error = None
        for transaction in reversed(tuple(transactions)):
            if transaction.state not in (
                "reserved",
                "materialized",
            ):
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

    def replay(self, entry, rows):
        prepared = self.prepare_replay(entry, rows)
        return self.replay_prepared(entry, rows, prepared)

    def prepare_replay(self, entry, rows):
        if not isinstance(
            entry,
            AutoregressiveDraftGraphEntry,
        ):
            raise AutoregressiveDraftGraphPreReplayError(
                "graph entry type is invalid"
            )
        payload = entry.graph
        if not isinstance(
            payload,
            Qwen3DraftCudaGraphPayload,
        ):
            raise AutoregressiveDraftGraphPreReplayError(
                "graph payload type is invalid"
            )
        graph_replay = getattr(payload.graph, "replay", None)
        if not callable(graph_replay):
            raise AutoregressiveDraftGraphPreReplayError(
                "graph payload has no replay method"
            )
        identity = entry.identity
        self._validate_identity(identity)
        return self._prepare_live_replay(
            identity,
            payload.tensors,
            rows,
        )

    def _complete_prepared(
        self,
        prepared,
        *,
        abort_transactions: bool,
    ) -> None:
        if prepared.completed:
            if abort_transactions:
                self._abort_transactions(
                    prepared.transactions
                )
            return
        allocator = self.proposal_kv_cache.entry_allocator
        self._complete_leases(
            allocator,
            prepared.read_leases,
            prepared.write_leases,
        )
        prepared.completed = True
        if abort_transactions:
            self._abort_transactions(
                prepared.transactions
            )

    def abort_prepared(self, prepared) -> None:
        self._complete_prepared(
            prepared,
            abort_transactions=True,
        )

    def abort_replay_result(self, result) -> None:
        self._abort_transactions(result.transactions)

    def replay_prepared(self, entry, rows, prepared):
        payload = entry.graph
        graph_replay = payload.graph.replay
        identity = entry.identity
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
                    "Qwen3 draft CUDA graph output is invalid"
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
                        or token_id < 0
                        for token_id in token_row
                    )
                    for token_row in token_rows
                )
            ):
                raise RuntimeError(
                    "Qwen3 draft CUDA graph tokens are invalid"
                )
            self._complete_prepared(
                prepared,
                abort_transactions=False,
            )
            return AutoregressiveDraftGroupExecution(
                transactions=prepared.transactions,
                token_rows=tuple(
                    tuple(token_row)
                    for token_row in token_rows
                ),
                execution_mode="cuda_graph",
            )
        except BaseException:
            self._complete_prepared(
                prepared,
                abort_transactions=True,
            )
            raise
