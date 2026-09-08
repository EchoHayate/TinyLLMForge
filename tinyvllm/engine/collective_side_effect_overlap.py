from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


TicketState = Literal[
    "launched",
    "joined",
    "sealed",
    "published",
    "aborted",
]


@dataclass(frozen=True)
class OverlapResources:
    communication_stream: object
    side_effect_stream: object
    producer_ready_event: object
    collective_visible_event: object
    side_effect_ready_event: object


@dataclass
class LeaseSealedOverlapTicket:
    commit_identity: str
    local_result: object
    collective_work: object
    collective_visible_event: object
    side_effect_ready_event: object
    collective_waited: bool = False
    collective_dependency_transferred: bool = False
    side_effect_joined: bool = False
    state: TicketState = "launched"


class LeaseSealedCollectiveSideEffect:
    def __init__(
        self,
        *,
        resources: OverlapResources,
        current_stream: Callable[[object], object],
        stream_context: Callable[[object], object],
        collective: Callable[[object], object],
    ):
        self.resources = resources
        self.current_stream = current_stream
        self.stream_context = stream_context
        self.collective = collective
        self._active_ticket = None

    def launch(
        self,
        *,
        local_result,
        side_effect_payload,
        materialize_side_effect,
        commit_identity: str,
    ) -> LeaseSealedOverlapTicket:
        if not isinstance(commit_identity, str) or not commit_identity:
            raise ValueError("commit_identity must be a non-empty string")
        if self._active_ticket is not None:
            raise RuntimeError("overlap resources already have an active ticket")
        current = self.current_stream(local_result)
        resource = self.resources
        resource.producer_ready_event.record(current)
        with self.stream_context(resource.communication_stream):
            resource.communication_stream.wait_event(
                resource.producer_ready_event
            )
            work = self.collective(local_result)
        with self.stream_context(resource.side_effect_stream):
            resource.side_effect_stream.wait_event(
                resource.producer_ready_event
            )
            materialize_side_effect(side_effect_payload)
            resource.side_effect_ready_event.record(
                resource.side_effect_stream
            )
        ticket = LeaseSealedOverlapTicket(
            commit_identity=commit_identity,
            local_result=local_result,
            collective_work=work,
            collective_visible_event=resource.collective_visible_event,
            side_effect_ready_event=resource.side_effect_ready_event,
        )
        self._active_ticket = ticket
        return ticket

    def join(self, ticket: LeaseSealedOverlapTicket):
        self._require_active(ticket, "launched")
        current = self.current_stream(ticket.local_result)
        with self.stream_context(current):
            ticket.collective_work.wait()
            ticket.collective_waited = True
            ticket.collective_visible_event.record(current)
            ticket.collective_dependency_transferred = True
            current.wait_event(ticket.side_effect_ready_event)
            ticket.side_effect_joined = True
        ticket.state = "joined"
        return ticket.local_result

    def seal(self, ticket, observed_identity: str) -> None:
        self._require_active(ticket, "joined")
        if observed_identity != ticket.commit_identity:
            raise RuntimeError("overlap commit identity mismatch")
        ticket.state = "sealed"

    def publish(self, ticket, observed_identity: str, publisher) -> None:
        self._require_active(ticket, "sealed")
        if observed_identity != ticket.commit_identity:
            raise RuntimeError("overlap commit identity mismatch")
        publisher()
        ticket.state = "published"
        self._active_ticket = None

    def abort(self, ticket, aborter) -> None:
        if ticket.state in ("published", "aborted"):
            raise RuntimeError(f"overlap ticket is already {ticket.state}")
        if self._active_ticket is not ticket:
            raise RuntimeError("overlap ticket is not active")
        if not ticket.collective_waited:
            ticket.collective_work.wait()
            ticket.collective_waited = True
        ticket.side_effect_ready_event.synchronize()
        aborter()
        ticket.state = "aborted"
        self._active_ticket = None

    def _require_active(self, ticket, expected_state: str) -> None:
        if ticket.state in ("published", "aborted"):
            raise RuntimeError(f"overlap ticket is already {ticket.state}")
        if self._active_ticket is not ticket:
            raise RuntimeError("overlap ticket is not active")
        if ticket.state != expected_state:
            raise RuntimeError(
                f"overlap ticket must be {expected_state}, "
                f"received {ticket.state}"
            )
