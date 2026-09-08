from __future__ import annotations

from collections.abc import Callable, Mapping
import gc
import hashlib

import torch

from tinyvllm.engine.hybrid_state import (
    HybridStateLease,
    HybridStateTensorPool,
)
from tinyvllm.engine.qwen35_hybrid_state import (
    build_qwen35_hybrid_state_layout,
)
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)
from tinyvllm.engine.topology_local_tp2_island import (
    TopologyLocalTP2RankIdentity,
    assemble_logical_state_half,
)


_SNAPSHOT_SCHEMA = "qwen38.topology-local-tp2-state-snapshot.v1"
_MIGRATION_SCHEMA = "qwen38.topology-local-tp2-state-migration.v1"
_COMMIT_SCHEMA = "qwen38.topology-local-tp2-state-commit.v1"


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )
    return digest.hexdigest()


def _lease_key(lease: HybridStateLease) -> tuple[int, int, int]:
    return (lease.slot_id, lease.generation, lease.request_id)


def _lease_record(lease: HybridStateLease) -> dict:
    return {
        "request_id": lease.request_id,
        "generation": lease.generation,
        "slot_id": lease.slot_id,
    }


def _validate_lease_batch(
    leases: tuple[HybridStateLease, ...],
) -> tuple[HybridStateLease, ...]:
    if not isinstance(leases, tuple) or not leases:
        raise ValueError("leases must be a non-empty tuple")
    if any(type(lease) is not HybridStateLease for lease in leases):
        raise ValueError(
            "leases must contain only HybridStateLease values"
        )
    slot_ids = tuple(lease.slot_id for lease in leases)
    if len(set(slot_ids)) != len(slot_ids):
        raise ValueError("leases must reference distinct slot ids")
    return leases


def _assemble_segmented_state_half(
    quarters: tuple[torch.Tensor, ...],
    *,
    logical_rank: int,
    segment_widths: tuple[int, ...],
) -> torch.Tensor:
    if len(quarters) != 4:
        raise ValueError("world gather must return four state quarters")
    reference = quarters[0]
    if (
        not isinstance(reference, torch.Tensor)
        or reference.ndim == 0
        or any(
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != reference.shape
            or tensor.dtype != reference.dtype
            or tensor.device != reference.device
            for tensor in quarters
        )
    ):
        raise ValueError("world-gathered state quarters are incompatible")
    if sum(segment_widths) != reference.shape[0]:
        raise ValueError(
            "convolution segment widths do not cover the TP4 quarter"
        )
    first = 2 * logical_rank
    selected = quarters[first:first + 2]
    offset = 0
    segments = []
    for width in segment_widths:
        segments.append(torch.cat(tuple(
            tensor.narrow(0, offset, width)
            for tensor in selected
        ), dim=0))
        offset += width
    return torch.cat(tuple(segments), dim=0)


def _state_component_digest_rows(
    state_rows,
    *,
    source_ranks: tuple[int, ...],
) -> tuple[dict, ...]:
    if (
        not isinstance(source_ranks, tuple)
        or not source_ranks
        or len(source_ranks) not in (1, 2)
        or any(
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in range(4)
            for rank in source_ranks
        )
        or tuple(source_ranks)
        != tuple(range(source_ranks[0], source_ranks[0] + len(source_ranks)))
        or source_ranks[0] % 2 != 0
        and len(source_ranks) == 2
    ):
        raise ValueError("source ranks do not identify a logical TP2 half")
    partition_count = len(source_ranks)
    logical_rank = source_ranks[0] // 2
    result = []
    for row in state_rows:
        layer_index = row.get("layer_index")
        convolution = row.get("convolution_states")
        recurrent = row.get("recurrent_states")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not isinstance(convolution, torch.Tensor)
            or not isinstance(recurrent, torch.Tensor)
            or convolution.ndim < 3
            or recurrent.ndim < 4
            or convolution.shape[0] != recurrent.shape[0]
            or convolution.shape[1] % partition_count != 0
            or recurrent.shape[1] % partition_count != 0
        ):
            raise ValueError("state component tensor layout is invalid")
        quarter_convolution_width = (
            convolution.shape[1] // partition_count
        )
        quarter_recurrent_heads = recurrent.shape[1] // partition_count
        value_width = quarter_recurrent_heads * recurrent.shape[2]
        remaining = quarter_convolution_width - value_width
        if remaining <= 0 or remaining % 2:
            raise ValueError(
                "state convolution segments do not match Qwen3.8"
            )
        key_width = remaining // 2
        segment_widths = (key_width, key_width, value_width)
        for partition_index, source_rank in enumerate(source_ranks):
            segment_digests = []
            segment_offset = 0
            for width in segment_widths:
                start = (
                    segment_offset * partition_count
                    + partition_index * width
                )
                segment_digests.append(_tensor_digest(
                    convolution.narrow(1, start, width)
                ))
                segment_offset += width
            result.append({
                "layer_index": layer_index,
                "logical_rank": logical_rank,
                "source_rank": source_rank,
                "convolution_query_sha256": segment_digests[0],
                "convolution_key_sha256": segment_digests[1],
                "convolution_value_sha256": segment_digests[2],
                "recurrent_sha256": _tensor_digest(
                    recurrent.narrow(
                        1,
                        partition_index * quarter_recurrent_heads,
                        quarter_recurrent_heads,
                    )
                ),
            })
    return tuple(result)


def build_qwen38_tp4_state_component_digests(
    *,
    source_transaction: Qwen35CrossLayerStateTransaction,
    leases: tuple[HybridStateLease, ...],
    global_rank: int,
) -> tuple[dict, ...]:
    if (
        isinstance(global_rank, bool)
        or not isinstance(global_rank, int)
        or global_rank not in range(4)
    ):
        raise ValueError("global_rank must be an integer rank in [0, 3]")
    leases = _validate_lease_batch(leases)
    source_transaction.pool.validate_leases(leases)
    gathered = source_transaction.gather(leases)
    state_rows = tuple(
        {
            "layer_index": adapter.layer_index,
            "convolution_states": state[0],
            "recurrent_states": state[1],
        }
        for adapter, state in zip(
            source_transaction.adapters,
            gathered,
        )
    )
    return _state_component_digest_rows(
        state_rows,
        source_ranks=(global_rank,),
    )


class Qwen38TopologyLocalTP2StateOwner:

    def __init__(
        self,
        *,
        source_transaction: Qwen35CrossLayerStateTransaction,
        destination_pool: HybridStateTensorPool,
        pair_identity: TopologyLocalTP2RankIdentity,
        all_gather: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
        convolution_segment_widths: tuple[int, int, int],
    ):
        self.source_transaction = source_transaction
        self.source_pool = source_transaction.pool
        self.destination_pool = destination_pool
        self.pair_identity = pair_identity
        self._all_gather = all_gather
        self._convolution_segment_widths = convolution_segment_widths
        layer_indices = tuple(
            adapter.layer_index
            for adapter in source_transaction.adapters
        )
        destination_adapters = tuple(
            Qwen35LayerStateAdapter(destination_pool, layer_index)
            for layer_index in layer_indices
        )
        self.destination_transaction = Qwen35CrossLayerStateTransaction(
            destination_adapters
        )
        self._phases: dict[tuple[int, int, int], str] = {}
        self._migration_rows: list[dict] = []
        self._publication_count = 0
        self._rollback_count = 0
        self._commit_count = 0
        self._temporary_live_tensors = 0

    def phase_for(self, lease: HybridStateLease) -> str:
        if type(lease) is not HybridStateLease:
            raise ValueError("lease must be a HybridStateLease")
        return self._phases.get(_lease_key(lease), "tp4_prefill")

    def _require_phase(
        self,
        leases: tuple[HybridStateLease, ...],
        expected: str,
    ) -> None:
        for lease in leases:
            phase = self.phase_for(lease)
            if phase != expected:
                if expected == "tp4_prefill" and phase == "tp2_decode":
                    raise RuntimeError(
                        "hybrid state lease was already migrated"
                    )
                if expected == "tp2_decode" and phase == "tp4_prefill":
                    raise RuntimeError(
                        "candidate state is not published"
                    )
                raise RuntimeError(
                    f"hybrid state lease is {phase}, expected {expected}"
                )

    def _reject_live_generation_conflicts(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> None:
        requested = {_lease_key(lease) for lease in leases}
        requested_slots = {lease.slot_id for lease in leases}
        for key, phase in self._phases.items():
            slot_id = key[0]
            if (
                slot_id in requested_slots
                and key not in requested
                and phase in {"migrating", "tp2_decode"}
            ):
                raise RuntimeError(
                    "generation-sealed destination slot is still owned"
                )

    def _gather_quarters(
        self,
        local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gathered = self._all_gather(local)
        if not isinstance(gathered, tuple) or len(gathered) != 4:
            raise ValueError(
                "world all_gather must return four tensors"
            )
        if any(not isinstance(tensor, torch.Tensor) for tensor in gathered):
            raise ValueError(
                "world all_gather must return four tensors"
            )
        return gathered

    def migrate(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[dict, ...]:
        leases = _validate_lease_batch(leases)
        self.source_pool.validate_leases(leases)
        self._reject_live_generation_conflicts(leases)
        self._require_phase(leases, "tp4_prefill")
        for lease in leases:
            self._phases[_lease_key(lease)] = "migrating"

        temporary: list[torch.Tensor] = []
        activated: list[HybridStateLease] = []
        try:
            source_layers = self.source_transaction.gather(leases)
            candidates = []
            for convolution_batch, recurrent_batch in source_layers:
                convolution_candidates = []
                recurrent_candidates = []
                for batch_index in range(len(leases)):
                    convolution_quarters = self._gather_quarters(
                        convolution_batch[batch_index]
                    )
                    recurrent_quarters = self._gather_quarters(
                        recurrent_batch[batch_index]
                    )
                    temporary.extend(convolution_quarters)
                    temporary.extend(recurrent_quarters)
                    convolution = _assemble_segmented_state_half(
                        convolution_quarters,
                        logical_rank=self.pair_identity.logical_rank,
                        segment_widths=self._convolution_segment_widths,
                    )
                    recurrent = assemble_logical_state_half(
                        recurrent_quarters,
                        self.pair_identity.logical_rank,
                    )
                    temporary.extend((convolution, recurrent))
                    convolution_candidates.append(convolution)
                    recurrent_candidates.append(recurrent)
                convolution_batch_candidate = torch.stack(
                    convolution_candidates
                )
                recurrent_batch_candidate = torch.stack(
                    recurrent_candidates
                )
                temporary.extend((
                    convolution_batch_candidate,
                    recurrent_batch_candidate,
                ))
                candidates.append((
                    convolution_batch_candidate,
                    recurrent_batch_candidate,
                ))

            self._temporary_live_tensors = len(temporary)
            for lease in leases:
                self.destination_pool.activate(lease)
                activated.append(lease)
            self.destination_transaction.commit(
                leases,
                tuple(candidates),
            )

            rows = []
            for lease in leases:
                for adapter in self.destination_transaction.adapters:
                    rows.append({
                        "schema_version": _MIGRATION_SCHEMA,
                        **_lease_record(lease),
                        "layer_index": adapter.layer_index,
                        "pair_id": self.pair_identity.pair_id,
                        "logical_rank": self.pair_identity.logical_rank,
                        "phase_from": "tp4_prefill",
                        "phase_to": "tp2_decode",
                        "published": True,
                    })
            for lease in leases:
                self._phases[_lease_key(lease)] = "tp2_decode"
            self._publication_count += len(leases)
            self._migration_rows.extend(rows)
            return tuple(rows)
        except Exception:
            for lease in reversed(activated):
                try:
                    self.destination_pool.release(lease)
                except RuntimeError:
                    pass
            for lease in leases:
                self._phases[_lease_key(lease)] = "tp4_prefill"
            self._rollback_count += 1
            raise
        finally:
            temporary.clear()
            self._temporary_live_tensors = 0
            gc.collect()

    def gather(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[dict, ...]:
        leases = _validate_lease_batch(leases)
        self.source_pool.validate_leases(leases)
        self._require_phase(leases, "tp2_decode")
        gathered = self.destination_transaction.gather(leases)
        return tuple(
            {
                "layer_index": adapter.layer_index,
                "convolution_states": candidate[0],
                "recurrent_states": candidate[1],
            }
            for adapter, candidate in zip(
                self.destination_transaction.adapters,
                gathered,
            )
        )

    def correctness_state_component_digests(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[dict, ...]:
        state_rows = self.gather(leases)
        return _state_component_digest_rows(
            state_rows,
            source_ranks=tuple(self.pair_identity.pair_ranks),
        )

    def commit(
        self,
        leases: tuple[HybridStateLease, ...],
        candidates: tuple[dict, ...],
    ) -> tuple[dict, ...]:
        leases = _validate_lease_batch(leases)
        self.source_pool.validate_leases(leases)
        self._require_phase(leases, "tp2_decode")
        if not isinstance(candidates, tuple):
            raise ValueError("candidates must be a tuple")
        if len(candidates) != len(self.destination_transaction.adapters):
            raise ValueError(
                "candidate count must match the linear layer count"
            )
        transaction_candidates = []
        for adapter, candidate in zip(
            self.destination_transaction.adapters,
            candidates,
        ):
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("layer_index") != adapter.layer_index
            ):
                raise ValueError(
                    "candidate layer inventory does not match"
                )
            transaction_candidates.append((
                candidate.get("convolution_states"),
                candidate.get("recurrent_states"),
            ))
        self.destination_transaction.commit(
            leases,
            tuple(transaction_candidates),
        )
        self._commit_count += 1
        return tuple(
            {
                "schema_version": _COMMIT_SCHEMA,
                **_lease_record(lease),
                "layer_index": adapter.layer_index,
                "committed": True,
            }
            for lease in leases
            for adapter in self.destination_transaction.adapters
        )

    def release(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[dict, ...]:
        leases = _validate_lease_batch(leases)
        self._require_phase(leases, "tp2_decode")
        released = []
        for lease in leases:
            self.destination_pool.release(lease)
            self._phases[_lease_key(lease)] = "released"
            released.append({
                **_lease_record(lease),
                "released": True,
            })
        return tuple(released)

    def snapshot(self) -> dict:
        leases = [
            {
                "request_id": request_id,
                "generation": generation,
                "slot_id": slot_id,
                "phase": phase,
            }
            for (slot_id, generation, request_id), phase
            in sorted(self._phases.items())
        ]
        return {
            "schema_version": _SNAPSHOT_SCHEMA,
            "source_layout_fingerprint": (
                self.source_pool.layout.fingerprint
            ),
            "destination_layout_fingerprint": (
                self.destination_pool.layout.fingerprint
            ),
            "capacity": self.destination_pool.capacity,
            "leases": leases,
            "migration_rows": [
                dict(row) for row in self._migration_rows
            ],
            "temporary_live_tensors": self._temporary_live_tensors,
            "publication_count": self._publication_count,
            "rollback_count": self._rollback_count,
            "commit_count": self._commit_count,
        }


def build_qwen38_topology_local_tp2_state_owner(
    *,
    hf_config,
    capacity: int,
    device,
    source_transaction: Qwen35CrossLayerStateTransaction,
    pair_identity: TopologyLocalTP2RankIdentity,
    all_gather: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
) -> Qwen38TopologyLocalTP2StateOwner:
    if not isinstance(
        source_transaction,
        Qwen35CrossLayerStateTransaction,
    ):
        raise ValueError(
            "source_transaction must be a "
            "Qwen35CrossLayerStateTransaction"
        )
    if type(pair_identity) is not TopologyLocalTP2RankIdentity:
        raise ValueError(
            "pair_identity must be a TopologyLocalTP2RankIdentity"
        )
    if not callable(all_gather):
        raise ValueError("all_gather must be callable")
    if source_transaction.pool.capacity != capacity:
        raise ValueError(
            "source TP4 state layout capacity does not match"
        )

    source_layout = build_qwen35_hybrid_state_layout(
        hf_config,
        tensor_parallel_size=4,
        dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
        speculative_tokens=1,
    )
    destination_layout = build_qwen35_hybrid_state_layout(
        hf_config,
        tensor_parallel_size=2,
        dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
        speculative_tokens=1,
    )
    source_pool = source_transaction.pool
    expected_layer_indices = tuple(sorted({
        component.layer_index
        for component in source_layout.components
    }))
    actual_layer_indices = tuple(
        adapter.layer_index
        for adapter in source_transaction.adapters
    )
    if (
        source_pool.layout.fingerprint != source_layout.fingerprint
        or actual_layer_indices != expected_layer_indices
        or len(actual_layer_indices) != 48
    ):
        raise ValueError(
            "source TP4 state layout does not match frozen Qwen3.8"
        )
    if destination_layout.bytes_per_slot != (
        2 * source_layout.bytes_per_slot
    ):
        raise ValueError(
            "destination TP2 state layout must double TP4 bytes per slot"
        )

    config = getattr(hf_config, "text_config", hf_config)
    key_width = (
        int(config.linear_num_key_heads)
        // 4
        * int(config.linear_key_head_dim)
    )
    value_width = (
        int(config.linear_num_value_heads)
        // 4
        * int(config.linear_value_head_dim)
    )
    return Qwen38TopologyLocalTP2StateOwner(
        source_transaction=source_transaction,
        destination_pool=HybridStateTensorPool(
            destination_layout,
            capacity,
            device,
        ),
        pair_identity=pair_identity,
        all_gather=all_gather,
        convolution_segment_widths=(
            key_width,
            key_width,
            value_width,
        ),
    )
