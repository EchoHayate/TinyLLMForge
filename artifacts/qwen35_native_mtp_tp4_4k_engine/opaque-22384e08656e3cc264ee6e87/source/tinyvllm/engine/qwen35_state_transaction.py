import torch

from tinyvllm.engine.hybrid_state import HybridStateLease
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter


class Qwen35CrossLayerStateTransaction:

    def __init__(
        self,
        adapters: tuple[Qwen35LayerStateAdapter, ...],
    ):
        if not isinstance(adapters, tuple) or not adapters:
            raise ValueError("adapters must be a non-empty tuple")
        if any(
            not isinstance(adapter, Qwen35LayerStateAdapter)
            for adapter in adapters
        ):
            raise ValueError(
                "adapters must contain only Qwen35LayerStateAdapter values"
            )
        pool = adapters[0].pool
        if any(adapter.pool is not pool for adapter in adapters):
            raise ValueError("adapters must reference the same pool")
        layer_indices = tuple(adapter.layer_index for adapter in adapters)
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("adapters must reference unique layer indices")
        self.adapters = adapters
        self.pool = pool

    def gather(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            adapter.gather_batch(leases)
            for adapter in self.adapters
        )

    @staticmethod
    def _validate_candidates(
        adapters: tuple[Qwen35LayerStateAdapter, ...],
        slot_ids: tuple[int, ...],
        candidates: tuple[
            tuple[torch.Tensor, torch.Tensor],
            ...,
        ],
    ) -> None:
        if not isinstance(candidates, tuple):
            raise ValueError("candidates must be a tuple")
        if len(candidates) != len(adapters):
            raise ValueError(
                "candidate count must match adapter count"
            )
        batch_size = len(slot_ids)
        for adapter, candidate_pair in zip(adapters, candidates):
            if (
                not isinstance(candidate_pair, tuple)
                or len(candidate_pair) != 2
            ):
                raise ValueError(
                    "each candidate must be a convolution/recurrent pair"
                )
            convolution_states, recurrent_states = candidate_pair
            adapter._validate_batch_candidate(
                convolution_states,
                batch_size=batch_size,
                reference=adapter.convolution[slot_ids[0]],
                name="convolution_states",
            )
            adapter._validate_batch_candidate(
                recurrent_states,
                batch_size=batch_size,
                reference=adapter.recurrent[slot_ids[0]],
                name="recurrent_states",
            )

    def commit(
        self,
        leases: tuple[HybridStateLease, ...],
        candidates: tuple[
            tuple[torch.Tensor, torch.Tensor],
            ...,
        ],
    ) -> None:
        slot_ids = tuple(
            adapter._validate_lease_batch(leases)
            for adapter in self.adapters
        )
        reference_slot_ids = slot_ids[0]
        if any(value != reference_slot_ids for value in slot_ids[1:]):
            raise RuntimeError(
                "adapters resolved inconsistent slot ids"
            )
        self._validate_candidates(
            self.adapters,
            reference_slot_ids,
            candidates,
        )

        snapshots = tuple(
            (
                torch.stack([
                    adapter.convolution[slot_id].clone()
                    for slot_id in reference_slot_ids
                ]),
                torch.stack([
                    adapter.recurrent[slot_id].clone()
                    for slot_id in reference_slot_ids
                ]),
            )
            for adapter in self.adapters
        )
        try:
            for adapter, candidate_pair in zip(
                self.adapters,
                candidates,
            ):
                convolution_states, recurrent_states = candidate_pair
                for batch_index, slot_id in enumerate(reference_slot_ids):
                    adapter._copy_component(
                        adapter.convolution[slot_id],
                        convolution_states[batch_index],
                    )
                    adapter._copy_component(
                        adapter.recurrent[slot_id],
                        recurrent_states[batch_index],
                    )
        except Exception:
            for adapter, snapshot in zip(self.adapters, snapshots):
                original_convolution, original_recurrent = snapshot
                for batch_index, slot_id in enumerate(reference_slot_ids):
                    adapter.convolution[slot_id].copy_(
                        original_convolution[batch_index]
                    )
                    adapter.recurrent[slot_id].copy_(
                        original_recurrent[batch_index]
                    )
            raise
