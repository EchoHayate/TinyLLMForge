from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


from tinyvllm.engine.collective_side_effect_overlap import (
    LeaseSealedCollectiveSideEffect,
    OverlapResources,
)


class FakeStream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_event(self, event):
        self.events.append((self.name, "wait", event.name))


class FakeEvent:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def record(self, stream):
        self.events.append((stream.name, "record", self.name))

    def synchronize(self):
        self.events.append(("host", "synchronize", self.name))


class FakeWork:
    def __init__(self, events, wait_error=None):
        self.events = events
        self.wait_error = wait_error

    def wait(self):
        self.events.append(("host", "wait", "collective"))
        if self.wait_error is not None:
            raise self.wait_error


class FakeContext:
    def __init__(self, stream, events):
        self.stream = stream
        self.events = events

    def __enter__(self):
        self.events.append(("context", "enter", self.stream.name))
        return self.stream

    def __exit__(self, exc_type, exc, traceback):
        self.events.append(("context", "exit", self.stream.name))
        return False


def resources(events):
    return OverlapResources(
        communication_stream=FakeStream("communication", events),
        side_effect_stream=FakeStream("side_effect", events),
        producer_ready_event=FakeEvent("producer_ready", events),
        collective_visible_event=FakeEvent("collective_visible", events),
        side_effect_ready_event=FakeEvent("side_effect_ready", events),
    )


def executor(events, wait_error=None):
    current = FakeStream("current", events)
    return LeaseSealedCollectiveSideEffect(
        resources=resources(events),
        current_stream=lambda _tensor: current,
        stream_context=lambda stream: FakeContext(stream, events),
        collective=lambda tensor: events.append(
            ("communication", "collective", tensor)
        )
        or FakeWork(events, wait_error=wait_error),
    )


def test_join_transfers_collective_ownership_before_side_effect_join():
    events = []
    runtime = executor(events)
    shadow = {}

    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda payload: shadow.update(value=payload),
        commit_identity="identity-a",
    )
    result = runtime.join(ticket)

    assert result == "local"
    assert shadow == {"value": "candidate"}
    assert ticket.state == "joined"
    assert events.index(("context", "enter", "current")) < events.index(
        ("host", "wait", "collective")
    )
    assert events.index(("host", "wait", "collective")) < events.index(
        ("current", "record", "collective_visible")
    )
    assert events.index(("current", "record", "collective_visible")) < (
        events.index(("current", "wait", "side_effect_ready"))
    )
    assert events.index(("current", "wait", "side_effect_ready")) < (
        events.index(("context", "exit", "current"))
    )
    assert ticket.collective_waited is True
    assert ticket.side_effect_joined is True


def test_failed_collective_wait_cannot_join_seal_or_publish():
    events = []
    runtime = executor(events, wait_error=RuntimeError("wait failed"))
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )

    with pytest.raises(RuntimeError, match="wait failed"):
        runtime.join(ticket)

    assert ticket.state == "launched"
    assert ticket.collective_waited is False
    with pytest.raises(RuntimeError, match="joined"):
        runtime.seal(ticket, "identity-a")


def test_publish_requires_join_seal_and_matching_identity():
    events = []
    runtime = executor(events)
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )

    with pytest.raises(RuntimeError, match="joined"):
        runtime.seal(ticket, "identity-a")
    runtime.join(ticket)
    with pytest.raises(RuntimeError, match="identity"):
        runtime.seal(ticket, "identity-b")
    runtime.seal(ticket, "identity-a")
    published = []
    runtime.publish(
        ticket,
        "identity-a",
        lambda: published.append("published"),
    )
    assert published == ["published"]
    assert ticket.state == "published"


def test_abort_waits_owned_work_is_terminal_and_never_publishes():
    events = []
    runtime = executor(events)
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )
    aborted = []
    runtime.abort(
        ticket,
        lambda: (
            events.append(("host", "abort", "callback")),
            aborted.append("aborted"),
        ),
    )

    assert aborted == ["aborted"]
    assert ticket.state == "aborted"
    assert events[-3:] == [
        ("host", "wait", "collective"),
        ("host", "synchronize", "side_effect_ready"),
        ("host", "abort", "callback"),
    ]
    with pytest.raises(RuntimeError, match="aborted"):
        runtime.publish(ticket, "identity-a", lambda: None)


def test_launch_rejects_empty_identity_and_reuse_of_active_resources():
    events = []
    runtime = executor(events)
    with pytest.raises(ValueError, match="commit_identity"):
        runtime.launch(
            local_result="local",
            side_effect_payload="candidate",
            materialize_side_effect=lambda _payload: None,
            commit_identity="",
        )
    runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )
    with pytest.raises(RuntimeError, match="active"):
        runtime.launch(
            local_result="local-2",
            side_effect_payload="candidate-2",
            materialize_side_effect=lambda _payload: None,
            commit_identity="identity-b",
        )


def test_publish_failure_leaves_ticket_sealed_for_explicit_abort():
    events = []
    runtime = executor(events)
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )
    runtime.join(ticket)
    runtime.seal(ticket, "identity-a")

    with pytest.raises(RuntimeError, match="publish failed"):
        runtime.publish(
            ticket,
            "identity-a",
            lambda: (_ for _ in ()).throw(RuntimeError("publish failed")),
        )
    assert ticket.state == "sealed"
    runtime.abort(ticket, lambda: None)
    assert ticket.state == "aborted"
