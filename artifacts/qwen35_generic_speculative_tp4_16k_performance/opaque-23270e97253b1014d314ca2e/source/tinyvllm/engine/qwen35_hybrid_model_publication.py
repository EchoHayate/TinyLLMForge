from __future__ import annotations

from tinyvllm.engine.qwen35_hybrid_model_owner import (
    Qwen35HybridModelOwner,
    build_qwen35_hybrid_model_owner,
)
from tinyvllm.models.qwen35_checkpoint_streaming import (
    Qwen35LoadedCheckpointCandidate,
)


class Qwen35HybridModelOwnerPublicationSlot:

    def __init__(self):
        self._publication = None

    @property
    def candidate(self) -> Qwen35LoadedCheckpointCandidate | None:
        if self._publication is None:
            return None
        return self._publication[0]

    @property
    def owner(self) -> Qwen35HybridModelOwner | None:
        if self._publication is None:
            return None
        return self._publication[1]

    @property
    def model_fingerprint(self) -> str | None:
        if self._publication is None:
            return None
        return self._publication[2]

    def publish(
        self,
        candidate: Qwen35LoadedCheckpointCandidate,
    ) -> Qwen35HybridModelOwner:
        if type(candidate) is not Qwen35LoadedCheckpointCandidate:
            raise ValueError(
                "candidate must be an exact loaded checkpoint candidate"
            )
        if self._publication is not None:
            raise RuntimeError(
                "Qwen3.5 model owner publication slot is already occupied"
            )
        owner = candidate.owner
        if type(owner) is not Qwen35HybridModelOwner:
            raise ValueError(
                "loaded checkpoint candidate owner is invalid"
            )
        validated = build_qwen35_hybrid_model_owner(owner.model)
        if (
            validated.model is not owner.model
            or validated.layer_stack is not owner.layer_stack
            or validated.state_transaction is not owner.state_transaction
            or validated.pool is not owner.pool
            or owner.runtime_bridge.pool is not owner.pool
        ):
            raise ValueError(
                "loaded checkpoint candidate owner graph is incoherent"
            )
        self._publication = (
            candidate,
            owner,
            candidate.model_fingerprint,
        )
        return owner
