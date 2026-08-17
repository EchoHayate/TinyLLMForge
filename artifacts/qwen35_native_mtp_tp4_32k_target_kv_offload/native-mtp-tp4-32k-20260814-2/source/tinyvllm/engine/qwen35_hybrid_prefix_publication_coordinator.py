from __future__ import annotations

from tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket import (
    Qwen35HybridPrefixPublicationAck,
    Qwen35HybridPrefixPublicationParticipant,
    Qwen35HybridPrefixPublicationPayload,
)


class Qwen35HybridPrefixPublicationCoordinator:

    def __init__(self, participants):
        if not isinstance(participants, tuple) or not participants:
            raise ValueError(
                "participants must be a non-empty tuple"
            )
        if any(
            not isinstance(
                participant,
                Qwen35HybridPrefixPublicationParticipant,
            )
            for participant in participants
        ):
            raise ValueError(
                "participants must contain publication participants"
            )
        participant_ids = tuple(
            participant.participant_id
            for participant in participants
        )
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("participant ids must be unique")
        self.participants = tuple(sorted(
            participants,
            key=lambda participant: participant.participant_id,
        ))
        self.participant_ids = tuple(
            participant.participant_id
            for participant in self.participants
        )
        if self.participant_ids != tuple(range(len(self.participants))):
            raise ValueError(
                "participant ids must be contiguous from zero"
            )
        self._poisoned_error = None

    def _ensure_healthy(self):
        if self._poisoned_error is not None:
            raise RuntimeError(
                "publication coordinator is poisoned: "
                f"{self._poisoned_error}"
            )

    def _poison(self, error):
        if self._poisoned_error is None:
            self._poisoned_error = str(error)

    def _validate_payloads(self, payloads):
        if (
            not isinstance(payloads, tuple)
            or len(payloads) != len(self.participants)
            or any(
                not isinstance(
                    payload,
                    Qwen35HybridPrefixPublicationPayload,
                )
                for payload in payloads
            )
        ):
            raise ValueError(
                "payload matrix must contain one payload per participant"
            )
        payloads = tuple(sorted(
            payloads,
            key=lambda payload: payload.participant_id,
        ))
        if tuple(
            payload.participant_id for payload in payloads
        ) != self.participant_ids:
            raise ValueError(
                "payload participant ids must match participants"
            )
        request_ids = {payload.request_id for payload in payloads}
        if len(request_ids) != 1:
            raise ValueError(
                "payload request identity must match across ranks"
            )
        reference = payloads[0]
        for payload in payloads[1:]:
            for name in (
                "ticket_id",
                "request_id",
                "key",
                "token_ids",
                "block_identities",
            ):
                if getattr(payload, name) != getattr(reference, name):
                    raise ValueError(
                        "payload identity mismatch across ranks: "
                        f"{name}"
                    )
        if reference.key.tensor_parallel_size != len(self.participants):
            raise ValueError(
                "payload tensor parallel size must match participants"
            )
        return payloads

    @staticmethod
    def _validate_ack(
        acknowledgement,
        payload,
        participant_id,
        operation,
        statuses,
    ):
        if not isinstance(
            acknowledgement,
            Qwen35HybridPrefixPublicationAck,
        ):
            raise RuntimeError(
                "publication acknowledgement is invalid"
            )
        if (
            acknowledgement.ticket_id != payload.ticket_id
            or acknowledgement.participant_id != participant_id
            or acknowledgement.operation != operation
            or acknowledgement.status not in statuses
            or not isinstance(acknowledgement.detail, str)
        ):
            raise RuntimeError(
                "publication acknowledgement fields are invalid"
            )
        return acknowledgement

    def _rollback(self, payloads, count):
        errors = []
        for participant, payload in reversed(tuple(zip(
            self.participants[:count],
            payloads[:count],
        ))):
            acknowledgement = participant.rollback(payload)
            try:
                self._validate_ack(
                    acknowledgement,
                    payload,
                    participant.participant_id,
                    "rollback",
                    {"rolled_back"},
                )
            except RuntimeError as error:
                errors.append(error)
        if errors:
            self._poison(errors[0])
            raise RuntimeError(
                f"publication rollback failed: {errors[0]}"
            ) from errors[0]

    def publish(self, payloads) -> bool:
        self._ensure_healthy()
        payloads = self._validate_payloads(payloads)
        prepared_count = 0
        try:
            for participant, payload in zip(
                self.participants,
                payloads,
            ):
                acknowledgement = self._validate_ack(
                    participant.prepare(payload),
                    payload,
                    participant.participant_id,
                    "prepare",
                    {"prepared", "rejected", "error"},
                )
                if acknowledgement.status == "rejected":
                    self._rollback(payloads, prepared_count)
                    return False
                if acknowledgement.status != "prepared":
                    raise RuntimeError(
                        "publication prepare failed: "
                        f"participant={participant.participant_id}, "
                        f"detail={acknowledgement.detail}"
                    )
                prepared_count += 1
        except BaseException:
            if prepared_count:
                self._rollback(payloads, prepared_count)
            raise

        try:
            for participant, payload in zip(
                self.participants,
                payloads,
            ):
                acknowledgement = self._validate_ack(
                    participant.precommit(payload),
                    payload,
                    participant.participant_id,
                    "precommit",
                    {"precommitted", "error"},
                )
                if acknowledgement.status != "precommitted":
                    raise RuntimeError(
                        "publication precommit failed: "
                        f"participant={participant.participant_id}, "
                        f"detail={acknowledgement.detail}"
                    )
        except BaseException:
            self._rollback(payloads, prepared_count)
            raise

        try:
            for participant, payload in zip(
                self.participants,
                payloads,
            ):
                acknowledgement = self._validate_ack(
                    participant.commit(payload),
                    payload,
                    participant.participant_id,
                    "commit",
                    {"finalized", "error"},
                )
                if acknowledgement.status != "finalized":
                    raise RuntimeError(
                        "publication finalize failed: "
                        f"participant={participant.participant_id}, "
                        f"detail={acknowledgement.detail}"
                    )
        except BaseException:
            self._rollback(payloads, prepared_count)
            raise
        try:
            for participant, payload in zip(
                self.participants,
                payloads,
            ):
                acknowledgement = self._validate_ack(
                    participant.seal(payload),
                    payload,
                    participant.participant_id,
                    "seal",
                    {"committed", "error"},
                )
                if acknowledgement.status != "committed":
                    raise RuntimeError(
                        "publication seal failed: "
                        f"participant={participant.participant_id}, "
                        f"detail={acknowledgement.detail}"
                    )
        except BaseException as error:
            self._poison(error)
            raise
        return True
