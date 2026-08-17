from __future__ import annotations

import torch

from tinyvllm.engine.autoregressive_draft_executor import (
    AutoregressiveDraftDecodeRow,
    AutoregressiveDraftPrefillRow,
)
from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.qwen3_draft_proposal_kv import (
    Qwen3DraftPhysicalSlotStore,
    Qwen3DraftProposalKVStorage,
)
from tinyvllm.utils.context import temporary_context


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _identity(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


class Qwen3AutoregressiveDraftBackend:

    def __init__(
        self,
        *,
        model,
        proposal_kv_cache: ProposalKVCache,
        backend_identity: str,
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        tensor_parallel_rank: int = 0,
        tensor_parallel_size: int = 1,
    ):
        if not callable(model):
            raise ValueError("model must be callable")
        if not callable(getattr(model, "compute_logits", None)):
            raise ValueError(
                "model must expose callable compute_logits"
            )
        if not isinstance(proposal_kv_cache, ProposalKVCache):
            raise ValueError(
                "proposal_kv_cache must be a ProposalKVCache"
            )
        entry_allocator = proposal_kv_cache.entry_allocator
        physical_store = getattr(
            entry_allocator,
            "physical_store",
            None,
        )
        if physical_store is None:
            physical_store = getattr(
                entry_allocator,
                "storage",
                None,
            )
        if not isinstance(
            physical_store,
            Qwen3DraftProposalKVStorage,
        ):
            raise ValueError(
                "proposal KV cache must use "
                "Qwen3DraftProposalKVStorage"
            )
        tensor_parallel_size = _positive_integer(
            tensor_parallel_size,
            "tensor_parallel_size",
        )
        if tensor_parallel_size not in (1, 4):
            raise RuntimeError(
                "Qwen3 draft backend supports TP1 or TP4"
            )
        if (
            isinstance(tensor_parallel_rank, bool)
            or not isinstance(tensor_parallel_rank, int)
            or tensor_parallel_rank < 0
            or tensor_parallel_rank >= tensor_parallel_size
        ):
            raise ValueError(
                "tensor_parallel_rank must be in "
                "[0, tensor_parallel_size)"
            )
        layers = getattr(
            getattr(model, "model", None),
            "layers",
            None,
        )
        if layers is None or len(layers) == 0:
            raise ValueError(
                "Qwen3 model must expose non-empty model.layers"
            )
        local_query_heads = {
            _positive_integer(
                getattr(
                    getattr(layer, "self_attn", None),
                    "num_heads",
                    None,
                ),
                "local_query_heads",
            )
            for layer in layers
        }
        if len(local_query_heads) != 1:
            raise ValueError(
                "Qwen3 draft layers must use identical "
                "local query-head geometry"
            )
        parameters = getattr(model, "parameters", None)
        if not callable(parameters):
            raise ValueError(
                "model must expose callable parameters"
            )
        self.model = model
        self.proposal_kv_cache = proposal_kv_cache
        self.physical_store = physical_store
        self.device = physical_store.device
        self.backend_identity = _identity(
            backend_identity,
            "backend_identity",
        )
        self.model_fingerprint = _identity(
            model_fingerprint,
            "model_fingerprint",
        )
        self.tokenizer_fingerprint = _identity(
            tokenizer_fingerprint,
            "tokenizer_fingerprint",
        )
        self.tensor_parallel_rank = tensor_parallel_rank
        self.tensor_parallel_size = tensor_parallel_size
        self.local_query_heads = next(iter(local_query_heads))
        self.local_model_parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters()
        )
        self._prefill_forward_count = 0
        self._decode_forward_count = 0

    def _owned_transaction(self, transaction):
        owned = self.proposal_kv_cache.transaction(
            getattr(transaction, "transaction_id", "")
        )
        if owned is not transaction:
            raise ValueError(
                "transaction does not belong to proposal KV cache"
            )
        return owned

    @staticmethod
    def _validate_hidden(
        hidden,
        *,
        expected_rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor):
            raise ValueError("model hidden output must be a tensor")
        if hidden.ndim != 2:
            raise ValueError(
                "model hidden output must be rank two"
            )
        if int(hidden.shape[0]) != expected_rows:
            raise ValueError(
                "model hidden row count must match input rows"
            )
        if not hidden.is_floating_point():
            raise ValueError(
                "model hidden output must use a floating dtype"
            )
        if hidden.device != device:
            raise ValueError(
                "model hidden output must use backend device"
            )
        return hidden

    @torch.inference_mode()
    def prefill_batch(
        self,
        rows: tuple[AutoregressiveDraftPrefillRow, ...],
    ) -> None:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(
                "draft prefill rows must be a non-empty tuple"
            )
        input_ids = []
        positions = []
        slot_mapping = []
        offsets = [0]
        for row in rows:
            if not isinstance(
                row,
                AutoregressiveDraftPrefillRow,
            ):
                raise ValueError(
                    "draft prefill row type is invalid"
                )
            transaction = self._owned_transaction(
                row.transaction
            )
            if (
                not isinstance(row.token_ids, tuple)
                or not row.token_ids
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in row.token_ids
                )
            ):
                raise ValueError(
                    "draft prefill token IDs must be "
                    "nonnegative integers"
                )
            if (
                not isinstance(row.positions, tuple)
                or len(row.positions) != len(row.token_ids)
                or any(
                    isinstance(position, bool)
                    or not isinstance(position, int)
                    or position < 0
                    for position in row.positions
                )
            ):
                raise ValueError(
                    "draft prefill positions are invalid"
                )
            if len(transaction.staged_entry_identities) != len(
                row.token_ids
            ):
                raise ValueError(
                    "draft prefill token count must match staged entries"
                )
            if (
                not isinstance(row.physical_slot_ids, tuple)
                or len(row.physical_slot_ids) != len(row.token_ids)
            ):
                raise ValueError(
                    "draft prefill physical slot count must match "
                    "token count"
                )
            if (
                any(
                    isinstance(slot_id, bool)
                    or not isinstance(slot_id, int)
                    or slot_id < 0
                    for slot_id in row.physical_slot_ids
                )
                or len(set(row.physical_slot_ids))
                != len(row.physical_slot_ids)
            ):
                raise ValueError(
                    "draft prefill physical slots must be unique "
                    "nonnegative integers"
                )
            input_ids.extend(row.token_ids)
            positions.extend(row.positions)
            slot_mapping.extend(row.physical_slot_ids)
            offsets.append(len(input_ids))

        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.int64,
            device=self.device,
        )
        position_tensor = torch.tensor(
            positions,
            dtype=torch.int64,
            device=self.device,
        )
        slot_tensor = torch.tensor(
            slot_mapping,
            dtype=torch.int32,
            device=self.device,
        )
        offset_tensor = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=self.device,
        )
        max_length = max(
            len(row.token_ids) for row in rows
        )
        with temporary_context(
            mode="prefill",
            is_prefill=True,
            slot_mapping=slot_tensor,
            context_lens=None,
            block_tables=None,
            cu_seqlens_q=offset_tensor,
            cu_seqlens_k=offset_tensor,
            max_seqlen_q=max_length,
            max_seqlen_k=max_length,
            quest_top_k_blocks=-1,
            am_compact_blocks=0,
            kv_offload_manager=None,
            kv_offload_blockwise_decode=False,
            kv_offload_blockwise_prefill=False,
        ):
            hidden = self.model(
                input_tensor,
                position_tensor,
            )
            self._prefill_forward_count += 1
        self._validate_hidden(
            hidden,
            expected_rows=len(input_ids),
            device=self.device,
        )

    @torch.inference_mode()
    def decode_step_batch(
        self,
        rows: tuple[AutoregressiveDraftDecodeRow, ...],
    ) -> tuple[torch.Tensor, ...] | None:
        if not isinstance(rows, tuple) or not rows:
            raise ValueError(
                "draft decode rows must be a non-empty tuple"
            )
        input_ids = []
        positions = []
        slot_mapping = []
        visible_rows = []
        logical_rows = []
        write_blocks = []
        blockwise_offload = None
        for row in rows:
            if not isinstance(
                row,
                AutoregressiveDraftDecodeRow,
            ):
                raise ValueError(
                    "draft decode row type is invalid"
                )
            transaction = self._owned_transaction(
                row.transaction
            )
            if not isinstance(row.blockwise_offload, bool):
                raise ValueError(
                    "draft decode blockwise flag must be a bool"
                )
            if blockwise_offload is None:
                blockwise_offload = row.blockwise_offload
            elif blockwise_offload != row.blockwise_offload:
                raise ValueError(
                    "draft decode rows must use one offload mode"
                )
            if (
                isinstance(row.step, bool)
                or not isinstance(row.step, int)
                or row.step < 0
                or row.step >= len(
                    transaction.staged_entry_identities
                )
            ):
                raise ValueError("draft decode step is invalid")
            for value, name in (
                (row.input_token_id, "input_token_id"),
                (row.position, "position"),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"draft decode {name} is invalid"
                    )
            visible = row.visible_physical_slot_ids
            if not isinstance(visible, tuple):
                raise ValueError(
                    "draft decode visible physical slots must "
                    "be a tuple"
                )
            expected_visible_count = (
                self.proposal_kv_cache.committed_length(
                    transaction.sequence_id
                )
                + row.step
                + 1
            )
            expected_physical_count = (
                1 if blockwise_offload else expected_visible_count
            )
            if len(visible) != expected_physical_count:
                raise ValueError(
                    "draft decode visible physical slot count "
                    "is invalid"
                )
            if (
                any(
                    isinstance(slot_id, bool)
                    or not isinstance(slot_id, int)
                    or slot_id < 0
                    for slot_id in visible
                )
                or len(set(visible)) != len(visible)
            ):
                raise ValueError(
                    "draft decode visible physical slots must "
                    "be unique nonnegative integers"
                )
            if (
                isinstance(row.writable_physical_slot_id, bool)
                or not isinstance(
                    row.writable_physical_slot_id,
                    int,
                )
                or row.writable_physical_slot_id < 0
                or not visible
                or visible[-1]
                != row.writable_physical_slot_id
            ):
                raise ValueError(
                    "draft decode writable physical slot must "
                    "be the final visible slot"
                )
            logical_visible = row.visible_logical_entry_ids
            if blockwise_offload:
                if (
                    not isinstance(logical_visible, tuple)
                    or len(logical_visible)
                    != expected_visible_count
                    or any(
                        isinstance(logical_entry_id, bool)
                        or not isinstance(logical_entry_id, int)
                        or logical_entry_id < 0
                        for logical_entry_id in logical_visible
                    )
                    or len(set(logical_visible))
                    != len(logical_visible)
                ):
                    raise ValueError(
                        "draft decode visible logical entries "
                        "are invalid"
                    )
                write_identity = (
                    transaction.staged_entry_identities[row.step]
                )
                if (
                    logical_visible[-1]
                    != write_identity.logical_entry_id
                ):
                    raise ValueError(
                        "draft decode writable logical entry must "
                        "be the final visible entry"
                    )
                logical_rows.append(logical_visible)
                write_blocks.append(
                    write_identity.logical_entry_id
                )
            elif logical_visible:
                raise ValueError(
                    "dense draft decode must not carry logical entries"
                )
            input_ids.append(row.input_token_id)
            positions.append(row.position)
            slot_mapping.append(row.writable_physical_slot_id)
            visible_rows.append(visible)

        context_lengths = (
            [len(row) for row in logical_rows]
            if blockwise_offload
            else [len(row) for row in visible_rows]
        )
        max_visible = max(context_lengths)
        input_tensor = torch.tensor(
            input_ids,
            dtype=torch.int64,
            device=self.device,
        )
        position_tensor = torch.tensor(
            positions,
            dtype=torch.int64,
            device=self.device,
        )
        slot_tensor = torch.tensor(
            slot_mapping,
            dtype=torch.int32,
            device=self.device,
        )
        block_tables = None
        if not blockwise_offload:
            block_table_rows = [
                list(row) + [-1] * (max_visible - len(row))
                for row in visible_rows
            ]
            block_tables = torch.tensor(
                block_table_rows,
                dtype=torch.int32,
                device=self.device,
            )
        context_lens = torch.tensor(
            context_lengths,
            dtype=torch.int32,
            device=self.device,
        )
        kv_offload_manager = None
        if blockwise_offload:
            kv_offload_manager = getattr(
                self.proposal_kv_cache.entry_allocator,
                "blockwise_attention_adapter",
                None,
            )
            if kv_offload_manager is None:
                raise RuntimeError(
                    "blockwise draft decode requires a residency "
                    "allocator"
                )
        with temporary_context(
            mode="decode",
            is_prefill=False,
            slot_mapping=slot_tensor,
            context_lens=context_lens,
            block_tables=block_tables,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=1,
            max_seqlen_k=max_visible,
            quest_top_k_blocks=-1,
            am_compact_blocks=0,
            kv_offload_manager=kv_offload_manager,
            kv_offload_blockwise_decode=bool(blockwise_offload),
            kv_offload_blockwise_prefill=False,
            kv_offload_blockwise_blocks=1,
            kv_offload_logical_block_tables=(
                [list(row) for row in logical_rows]
                if blockwise_offload
                else None
            ),
            kv_offload_context_lens=(
                context_lengths if blockwise_offload else None
            ),
            kv_offload_write_blocks=(
                write_blocks if blockwise_offload else None
            ),
        ):
            hidden = self.model(
                input_tensor,
                position_tensor,
            )
            self._decode_forward_count += 1
            hidden = self._validate_hidden(
                hidden,
                expected_rows=len(rows),
                device=self.device,
            )
            logits = self.model.compute_logits(hidden)
        if self.tensor_parallel_rank != 0:
            if logits is not None:
                raise ValueError("non-root logits must be None")
            return None
        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                "root model logits must be a tensor"
            )
        if (
            logits.ndim != 2
            or int(logits.shape[0]) != len(rows)
            or int(logits.shape[1]) <= 1
        ):
            raise ValueError(
                "model logits must have exact shape "
                "[batch_size, vocab_size]"
            )
        if not logits.is_floating_point():
            raise ValueError(
                "model logits must use a floating dtype"
            )
        if logits.device != self.device:
            raise ValueError(
                "model logits must use backend device"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError(
                "model logits must contain finite values"
            )
        return tuple(logits[index] for index in range(len(rows)))

    def authority_snapshot(self) -> dict:
        key_cache = self.physical_store.key_cache
        value_cache = self.physical_store.value_cache
        snapshot = {
            "backend_identity": self.backend_identity,
            "model_fingerprint": self.model_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tensor_parallel_rank": self.tensor_parallel_rank,
            "tensor_parallel_size": self.tensor_parallel_size,
            "local_model_parameter_bytes": (
                self.local_model_parameter_bytes
            ),
            "local_proposal_kv_bytes": int(
                key_cache.numel() * key_cache.element_size()
                + value_cache.numel()
                * value_cache.element_size()
            ),
            "local_query_heads": self.local_query_heads,
            "local_kv_heads": self.physical_store.local_kv_heads,
            "local_prefill_forward_count": (
                self._prefill_forward_count
            ),
            "local_decode_forward_count": (
                self._decode_forward_count
            ),
            "prefill_forward_count": self._prefill_forward_count,
            "decode_forward_count": self._decode_forward_count,
            "real_draft_forward_count": (
                self._prefill_forward_count
                + self._decode_forward_count
            ),
            "proposal_kv_bytes": int(
                key_cache.numel() * key_cache.element_size()
                + value_cache.numel()
                * value_cache.element_size()
            ),
            "proposal_kv_storage_id": (
                f"{key_cache.data_ptr()}:{value_cache.data_ptr()}"
            ),
            "physical_store": (
                self.physical_store.authority_snapshot()
            ),
            "proposal_kv_cache": (
                self.proposal_kv_cache.authority_snapshot()
            ),
        }
        from tinyvllm.engine.speculative_proposal_executor import (
            assert_tensor_free,
        )
        assert_tensor_free(
            snapshot,
            name="Qwen3 draft backend authority snapshot",
        )
        return snapshot
