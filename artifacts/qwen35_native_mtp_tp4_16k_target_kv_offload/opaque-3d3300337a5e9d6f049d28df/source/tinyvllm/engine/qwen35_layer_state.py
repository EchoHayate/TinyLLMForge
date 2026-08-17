import torch

from tinyvllm.engine.hybrid_state import (
    HybridStateLease,
    HybridStateTensorPool,
)


class Qwen35LayerStateAdapter:

    def __init__(
        self,
        pool: HybridStateTensorPool,
        layer_index: int,
    ):
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer")
        self.pool = pool
        self.layer_index = layer_index
        self.convolution = pool.component_tensor(
            layer_index,
            "linear_convolution",
        )
        self.recurrent = pool.component_tensor(
            layer_index,
            "linear_recurrent",
        )

    def gather(
        self,
        lease: HybridStateLease,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_id = self.pool.validate(lease)
        return (
            self.convolution[slot_id].clone(),
            self.recurrent[slot_id].clone(),
        )

    def _validate_lease_batch(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[int, ...]:
        if not isinstance(leases, tuple) or not leases:
            raise ValueError("leases must be a non-empty tuple")
        if any(not isinstance(lease, HybridStateLease) for lease in leases):
            raise ValueError(
                "leases must contain only HybridStateLease values"
            )
        slot_ids = tuple(self.pool.validate(lease) for lease in leases)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("leases must reference distinct slot ids")
        return slot_ids

    def gather_batch(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_ids = self._validate_lease_batch(leases)
        return (
            torch.stack([
                self.convolution[slot_id]
                for slot_id in slot_ids
            ]),
            torch.stack([
                self.recurrent[slot_id]
                for slot_id in slot_ids
            ]),
        )

    @staticmethod
    def _validate_candidate(
        candidate: torch.Tensor,
        reference: torch.Tensor,
        name: str,
    ) -> None:
        if not isinstance(candidate, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if candidate.shape != reference.shape:
            raise ValueError(f"{name} shape must match the pool component")
        if candidate.dtype != reference.dtype:
            raise ValueError(f"{name} dtype must match the pool component")
        if candidate.device != reference.device:
            raise ValueError(f"{name} device must match the pool component")

    @staticmethod
    def _copy_component(
        destination: torch.Tensor,
        source: torch.Tensor,
    ) -> None:
        destination.copy_(source)

    @staticmethod
    def _validate_batch_candidate(
        candidate: torch.Tensor,
        *,
        batch_size: int,
        reference: torch.Tensor,
        name: str,
    ) -> None:
        if not isinstance(candidate, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if candidate.shape != (batch_size, *reference.shape):
            raise ValueError(f"{name} shape must match the batched component")
        if candidate.dtype != reference.dtype:
            raise ValueError(f"{name} dtype must match the pool component")
        if candidate.device != reference.device:
            raise ValueError(f"{name} device must match the pool component")

    def commit(
        self,
        lease: HybridStateLease,
        convolution_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None:
        slot_id = self.pool.validate(lease)
        destination_convolution = self.convolution[slot_id]
        destination_recurrent = self.recurrent[slot_id]
        self._validate_candidate(
            convolution_state,
            destination_convolution,
            "convolution_state",
        )
        self._validate_candidate(
            recurrent_state,
            destination_recurrent,
            "recurrent_state",
        )

        original_convolution = destination_convolution.clone()
        original_recurrent = destination_recurrent.clone()
        try:
            self._copy_component(
                destination_convolution,
                convolution_state,
            )
            self._copy_component(
                destination_recurrent,
                recurrent_state,
            )
        except Exception:
            destination_convolution.copy_(original_convolution)
            destination_recurrent.copy_(original_recurrent)
            raise

    def commit_batch(
        self,
        leases: tuple[HybridStateLease, ...],
        convolution_states: torch.Tensor,
        recurrent_states: torch.Tensor,
    ) -> None:
        slot_ids = self._validate_lease_batch(leases)
        batch_size = len(slot_ids)
        reference_convolution = self.convolution[slot_ids[0]]
        reference_recurrent = self.recurrent[slot_ids[0]]
        self._validate_batch_candidate(
            convolution_states,
            batch_size=batch_size,
            reference=reference_convolution,
            name="convolution_states",
        )
        self._validate_batch_candidate(
            recurrent_states,
            batch_size=batch_size,
            reference=reference_recurrent,
            name="recurrent_states",
        )

        original_convolution = torch.stack([
            self.convolution[slot_id].clone()
            for slot_id in slot_ids
        ])
        original_recurrent = torch.stack([
            self.recurrent[slot_id].clone()
            for slot_id in slot_ids
        ])
        try:
            for batch_index, slot_id in enumerate(slot_ids):
                self._copy_component(
                    self.convolution[slot_id],
                    convolution_states[batch_index],
                )
                self._copy_component(
                    self.recurrent[slot_id],
                    recurrent_states[batch_index],
                )
        except Exception:
            for batch_index, slot_id in enumerate(slot_ids):
                self.convolution[slot_id].copy_(
                    original_convolution[batch_index]
                )
                self.recurrent[slot_id].copy_(
                    original_recurrent[batch_index]
                )
            raise
