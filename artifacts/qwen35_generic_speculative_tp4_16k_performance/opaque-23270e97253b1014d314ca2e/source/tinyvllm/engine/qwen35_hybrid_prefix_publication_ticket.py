from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.engine.hybrid_state import (
    HybridStateLease,
    HybridStateTensorPool,
)
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixKey,
    Qwen35HybridPrefixPreparedPublication,
    Qwen35HybridPrefixSnapshotCache,
)
from tinyvllm.engine.qwen35_hybrid_prefix_int8_cache import (
    Qwen35HybridPrefixInt8PreparedPublication,
    Qwen35HybridPrefixInt8SnapshotCache,
)


_STATUSES = {
    "prepared",
    "precommitted",
    "finalized",
    "rejected",
    "committed",
    "rolled_back",
    "error",
}


def _non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Qwen35HybridPrefixPublicationPayload:
    ticket_id: int
    participant_id: int
    request_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    lease: HybridStateLease

    def __post_init__(self):
        _non_negative_integer(self.ticket_id, "ticket_id")
        _non_negative_integer(
            self.participant_id,
            "participant_id",
        )
        _non_negative_integer(self.request_id, "request_id")
        Qwen35HybridPrefixSnapshotCache._validate_identity(
            self.key,
            self.token_ids,
            self.block_identities,
        )
        if not isinstance(self.lease, HybridStateLease):
            raise ValueError("lease must be a HybridStateLease")
        if self.lease.request_id != self.request_id:
            raise ValueError(
                "lease request_id must match payload request_id"
            )


@dataclass(frozen=True)
class Qwen35HybridPrefixPublicationAck:
    ticket_id: int
    participant_id: int
    operation: str
    status: str
    detail: str = ""

    def __post_init__(self):
        _non_negative_integer(self.ticket_id, "ticket_id")
        _non_negative_integer(
            self.participant_id,
            "participant_id",
        )
        if self.operation not in {
            "prepare",
            "precommit",
            "commit",
            "seal",
            "rollback",
        }:
            raise ValueError(
                f"unsupported publication operation: {self.operation}"
            )
        if self.status not in _STATUSES:
            raise ValueError(
                f"unsupported publication status: {self.status}"
            )
        if not isinstance(self.detail, str):
            raise ValueError("acknowledgement detail must be a string")


class Qwen35HybridPrefixPublicationParticipant:

    def __init__(
        self,
        participant_id: int,
        pool: HybridStateTensorPool,
        snapshot_cache: (
            Qwen35HybridPrefixSnapshotCache
            | Qwen35HybridPrefixInt8SnapshotCache
        ),
    ):
        self.participant_id = _non_negative_integer(
            participant_id,
            "participant_id",
        )
        if not isinstance(pool, HybridStateTensorPool):
            raise ValueError("pool must be a HybridStateTensorPool")
        if not isinstance(
            snapshot_cache,
            (
                Qwen35HybridPrefixSnapshotCache,
                Qwen35HybridPrefixInt8SnapshotCache,
            ),
        ):
            raise ValueError(
                "snapshot_cache must be a supported "
                "Qwen35 hybrid prefix snapshot cache"
            )
        if snapshot_cache.state_transaction.pool is not pool:
            raise ValueError(
                "snapshot cache transaction must use participant pool"
            )
        self.pool = pool
        self.snapshot_cache = snapshot_cache
        self._prepared = {}
        self._finalized = set()
        self._terminal = {}
        self._terminal_payloads = {}

    def _ack(self, payload, operation, status, detail=""):
        return Qwen35HybridPrefixPublicationAck(
            ticket_id=payload.ticket_id,
            participant_id=self.participant_id,
            operation=operation,
            status=status,
            detail=detail,
        )

    @staticmethod
    def _validate_payload(payload):
        if not isinstance(
            payload,
            Qwen35HybridPrefixPublicationPayload,
        ):
            raise ValueError(
                "payload must be a "
                "Qwen35HybridPrefixPublicationPayload"
            )

    def prepare(
        self,
        payload: Qwen35HybridPrefixPublicationPayload,
    ) -> Qwen35HybridPrefixPublicationAck:
        self._validate_payload(payload)
        if payload.participant_id != self.participant_id:
            return self._ack(
                payload,
                "prepare",
                "error",
                "payload participant id does not match participant",
            )
        prepared = self._prepared.get(payload.ticket_id)
        if prepared is not None:
            current_payload, _ = prepared
            if current_payload == payload:
                return self._ack(payload, "prepare", "prepared")
            return self._ack(
                payload,
                "prepare",
                "error",
                "ticket id already prepared with different payload",
            )
        terminal_payload = self._terminal_payloads.get(
            payload.ticket_id
        )
        if terminal_payload is not None:
            if terminal_payload == payload:
                terminal = self._terminal[payload.ticket_id]
                if terminal == "rejected":
                    return self._ack(
                        payload,
                        "prepare",
                        "rejected",
                        "snapshot exceeds cache max_bytes",
                    )
                return self._ack(
                    payload,
                    "prepare",
                    "error",
                    f"ticket is terminal: {terminal}",
                )
            return self._ack(
                payload,
                "prepare",
                "error",
                "ticket id is terminal with different payload",
            )
        try:
            handle = self.snapshot_cache.prepare_publication(
                payload.key,
                payload.token_ids,
                payload.block_identities,
                payload.lease,
            )
        except Exception as error:
            return self._ack(
                payload,
                "prepare",
                "error",
                str(error),
            )
        if handle is None:
            acknowledgement = self._ack(
                payload,
                "prepare",
                "rejected",
                "snapshot exceeds cache max_bytes",
            )
            self._terminal[payload.ticket_id] = "rejected"
            self._terminal_payloads[payload.ticket_id] = payload
            return acknowledgement
        if not isinstance(
            handle,
            (
                Qwen35HybridPrefixPreparedPublication,
                Qwen35HybridPrefixInt8PreparedPublication,
            ),
        ):
            try:
                self.snapshot_cache.abort_current_publication()
            except Exception as rollback_error:
                return self._ack(
                    payload,
                    "prepare",
                    "error",
                    (
                        "cache returned an invalid prepared publication; "
                        f"rollback failed: {rollback_error}"
                    ),
                )
            return self._ack(
                payload,
                "prepare",
                "error",
                "cache returned an invalid prepared publication",
            )
        self._prepared[payload.ticket_id] = (payload, handle)
        return self._ack(payload, "prepare", "prepared")

    def _terminal_operation(self, payload, operation, success_state):
        terminal_payload = self._terminal_payloads.get(
            payload.ticket_id
        )
        if terminal_payload is None:
            return None
        terminal = self._terminal[payload.ticket_id]
        if terminal_payload != payload:
            return self._ack(
                payload,
                operation,
                "error",
                "ticket id is terminal with different payload",
            )
        if terminal == success_state:
            return self._ack(
                payload,
                operation,
                success_state,
            )
        return self._ack(
            payload,
            operation,
            "error",
            f"ticket is terminal: {terminal}",
        )

    def commit(
        self,
        payload: Qwen35HybridPrefixPublicationPayload,
    ) -> Qwen35HybridPrefixPublicationAck:
        self._validate_payload(payload)
        if payload.participant_id != self.participant_id:
            return self._ack(
                payload,
                "commit",
                "error",
                "payload participant id does not match participant",
            )
        terminal = self._terminal_operation(
            payload,
            "commit",
            "committed",
        )
        if terminal is not None:
            return terminal
        prepared = self._prepared.get(payload.ticket_id)
        if prepared is None:
            return self._ack(
                payload,
                "commit",
                "error",
                "ticket is not prepared",
            )
        current_payload, handle = prepared
        if current_payload != payload:
            return self._ack(
                payload,
                "commit",
                "error",
                "ticket id is prepared with different payload",
            )
        if payload.ticket_id in self._finalized:
            return self._ack(
                payload,
                "commit",
                "finalized",
            )
        try:
            retained = self.snapshot_cache.finalize_publication(handle)
        except Exception as error:
            return self._ack(
                payload,
                "commit",
                "error",
                str(error),
            )
        self._finalized.add(payload.ticket_id)
        detail = "" if retained else "entry evicted during commit"
        return self._ack(
            payload,
            "commit",
            "finalized",
            detail,
        )

    def seal(
        self,
        payload: Qwen35HybridPrefixPublicationPayload,
    ) -> Qwen35HybridPrefixPublicationAck:
        self._validate_payload(payload)
        if payload.participant_id != self.participant_id:
            return self._ack(
                payload,
                "seal",
                "error",
                "payload participant id does not match participant",
            )
        terminal = self._terminal_operation(
            payload,
            "seal",
            "committed",
        )
        if terminal is not None:
            return terminal
        prepared = self._prepared.get(payload.ticket_id)
        if prepared is None or payload.ticket_id not in self._finalized:
            return self._ack(
                payload,
                "seal",
                "error",
                "ticket is not finalized",
            )
        current_payload, handle = prepared
        if current_payload != payload:
            return self._ack(
                payload,
                "seal",
                "error",
                "ticket id is finalized with different payload",
            )
        try:
            self.snapshot_cache.seal_publication(handle)
        except Exception as error:
            return self._ack(
                payload,
                "seal",
                "error",
                str(error),
            )
        del self._prepared[payload.ticket_id]
        self._finalized.remove(payload.ticket_id)
        self._terminal[payload.ticket_id] = "committed"
        self._terminal_payloads[payload.ticket_id] = payload
        return self._ack(
            payload,
            "seal",
            "committed",
        )

    def precommit(
        self,
        payload: Qwen35HybridPrefixPublicationPayload,
    ) -> Qwen35HybridPrefixPublicationAck:
        self._validate_payload(payload)
        if payload.participant_id != self.participant_id:
            return self._ack(
                payload,
                "precommit",
                "error",
                "payload participant id does not match participant",
            )
        terminal_payload = self._terminal_payloads.get(
            payload.ticket_id
        )
        if terminal_payload is not None:
            terminal = self._terminal[payload.ticket_id]
            return self._ack(
                payload,
                "precommit",
                "error",
                (
                    f"ticket is terminal: {terminal}"
                    if terminal_payload == payload
                    else "ticket id is terminal with different payload"
                ),
            )
        prepared = self._prepared.get(payload.ticket_id)
        if prepared is None:
            return self._ack(
                payload,
                "precommit",
                "error",
                "ticket is not prepared",
            )
        current_payload, handle = prepared
        if current_payload != payload:
            return self._ack(
                payload,
                "precommit",
                "error",
                "ticket id is prepared with different payload",
            )
        try:
            self.snapshot_cache.precommit_publication(handle)
        except Exception as error:
            return self._ack(
                payload,
                "precommit",
                "error",
                str(error),
            )
        return self._ack(
            payload,
            "precommit",
            "precommitted",
        )

    def rollback(
        self,
        payload: Qwen35HybridPrefixPublicationPayload,
    ) -> Qwen35HybridPrefixPublicationAck:
        self._validate_payload(payload)
        if payload.participant_id != self.participant_id:
            return self._ack(
                payload,
                "rollback",
                "error",
                "payload participant id does not match participant",
            )
        terminal_payload = self._terminal_payloads.get(
            payload.ticket_id
        )
        if terminal_payload is not None:
            if terminal_payload != payload:
                return self._ack(
                    payload,
                    "rollback",
                    "error",
                    "ticket id is terminal with different payload",
                )
            terminal = self._terminal[payload.ticket_id]
            if terminal == "rolled_back":
                return self._ack(
                    payload,
                    "rollback",
                    "rolled_back",
                )
            if terminal == "rejected":
                self._terminal[payload.ticket_id] = "rolled_back"
                return self._ack(
                    payload,
                    "rollback",
                    "rolled_back",
                )
            return self._ack(
                payload,
                "rollback",
                "error",
                f"ticket is terminal: {terminal}",
            )
        prepared = self._prepared.get(payload.ticket_id)
        if prepared is None:
            self._terminal[payload.ticket_id] = "rolled_back"
            self._terminal_payloads[payload.ticket_id] = payload
            return self._ack(
                payload,
                "rollback",
                "rolled_back",
            )
        current_payload, handle = prepared
        if current_payload != payload:
            return self._ack(
                payload,
                "rollback",
                "error",
                "ticket id is prepared with different payload",
            )
        try:
            self.snapshot_cache.rollback_publication(handle)
        except Exception as error:
            return self._ack(
                payload,
                "rollback",
                "error",
                str(error),
            )
        del self._prepared[payload.ticket_id]
        self._finalized.discard(payload.ticket_id)
        self._terminal[payload.ticket_id] = "rolled_back"
        self._terminal_payloads[payload.ticket_id] = payload
        return self._ack(
            payload,
            "rollback",
            "rolled_back",
        )

