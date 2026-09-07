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
    def __init__(self, events):
        self.events = events

    def wait(self):
        self.events.append(("host", "wait", "collective"))


class FakeContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc, traceback):
        return False


def resources(events):
    return OverlapResources(
        communication_stream=FakeStream("communication", events),
        side_effect_stream=FakeStream("side_effect", events),
        producer_ready_event=FakeEvent("producer_ready", events),
        consumer_ready_event=FakeEvent("consumer_ready", events),
        side_effect_ready_event=FakeEvent("side_effect_ready", events),
    )


def executor(events):
    current = FakeStream("current", events)
    return LeaseSealedCollectiveSideEffect(
        resources=resources(events),
        current_stream=lambda _tensor: current,
        stream_context=lambda stream: FakeContext(stream),
        collective=lambda tensor: events.append(
            ("communication", "collective", tensor)
        )
        or FakeWork(events),
    )


def test_launch_forks_collective_and_side_effect_then_join_waits_both():
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
    assert ("current", "wait", "consumer_ready") in events
    assert ("current", "wait", "side_effect_ready") in events


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
