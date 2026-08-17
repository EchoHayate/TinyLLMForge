from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_live_concurrent_candidate_ownership_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Resource:
    pass


def _prepared_state(
    module,
    *,
    rank=2,
    escaped=False,
    cleanup_override=None,
):
    resources = {
        name: Resource()
        for name in (
            "runner",
            "production_slot",
            "request",
            "candidate",
            "owner",
            "runtime_bridge",
            "runtime_identity",
            "model",
            "pool",
            "target",
        )
    }
    selected = [types.SimpleNamespace(value=rank + 1) for rank in range(4)]
    non_selected = types.SimpleNamespace(value=37)
    released = []
    external_reference = resources["owner"] if escaped else None

    def release_graph():
        for tensor in reversed(selected):
            tensor.value = 0
        if non_selected.value != 37:
            raise RuntimeError("non-selected tensor changed")
        released.append(True)
        resources.clear()
        cleanup = {
            "all_selected_destinations_zero_after_clear": True,
            "non_selected_tensors_unchanged": True,
            "tensor_identity_preserved": True,
            "pool_unchanged": True,
        }
        if cleanup_override is not None:
            cleanup.update(cleanup_override)
        return cleanup

    ready_payload = {
        "tp_size": 4,
        "tp_rank": rank,
        "process_id": 32_000_000 + rank,
        "method_row": {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": "a" * 64,
            "layout_fingerprint": "b" * 64,
            "dtype": "bfloat16",
            "detail": "",
        },
        "binding_hash_count": 320,
        "binding_destination_sha256": ["e" * 64] * 320,
        "phase_hash_count": 26,
        "phase_destination_sha256": {
            f"phase_{index}": "f" * 64
            for index in range(26)
        },
        "aggregate_destination_sha256": "c" * 64,
        "alias_groups": [],
        "loader_stats": {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "ready_memory": {
            "vmrss_kib": 1_200_000,
            "vmhwm_kib": 2_900_000,
        },
    }
    state = module.PreparedCandidateState(
        ready_payload=ready_payload,
        retained_objects=resources,
        selected_objects=tuple(selected),
        release_graph=release_graph,
    )
    return state, selected, released, external_reference


def test_retained_scope_keeps_candidate_live_until_single_release():
    module = _load_module()
    state, selected, released, _ = _prepared_state(module)

    retained = module.prepare_retained_tp4_candidate(
        rank=2,
        state_factory=lambda: state,
    )
    ready = retained.ready_row()

    assert ready["schema_version"] == (
        "qwen35.tp4-live-concurrent-candidate-ready-rank.v1"
    )
    assert ready["status"] == "READY"
    assert ready["provenance"] == (
        "real-checkpoint-derived-live-concurrent-tp4-ownership"
    )
    assert ready["claim_boundary"] == (
        "not-constructed-engine-runtime-binding"
    )
    assert ready["tp_rank"] == 2
    assert ready["all_private_objects_retained"] is True
    assert all(tensor.value != 0 for tensor in selected)
    assert not released

    final = retained.release()

    assert final["schema_version"] == (
        "qwen35.tp4-live-concurrent-candidate-released-rank.v1"
    )
    assert final["status"] == "RELEASED"
    assert final["tp_rank"] == 2
    assert final["all_private_objects_collected"] is True
    assert all(final["collected_private_objects"].values())
    assert all(tensor.value == 0 for tensor in selected)
    assert released == [True]

    try:
        retained.release()
    except RuntimeError as error:
        assert "already released" in str(error)
    else:
        raise AssertionError("duplicate retained-candidate release must fail")


def test_retained_scope_rejects_rank_participant_mismatch():
    module = _load_module()
    state, _, _, _ = _prepared_state(module)
    state.ready_payload["method_row"]["participant_id"] = 1

    try:
        module.prepare_retained_tp4_candidate(
            rank=2,
            state_factory=lambda: state,
        )
    except ValueError as error:
        assert "participant" in str(error)
    else:
        raise AssertionError("rank/participant mismatch must fail")


def test_retained_scope_rejects_escaped_private_object_after_release():
    module = _load_module()
    state, _, _, external_reference = _prepared_state(
        module,
        escaped=True,
    )
    retained = module.prepare_retained_tp4_candidate(
        rank=2,
        state_factory=lambda: state,
    )
    assert external_reference is not None

    try:
        retained.release()
    except RuntimeError as error:
        assert "escaped" in str(error)
    else:
        raise AssertionError("escaped retained object must fail cleanup")


def test_retained_scope_rejects_incomplete_clear_invariants():
    module = _load_module()
    state, _, _, _ = _prepared_state(
        module,
        cleanup_override={
            "all_selected_destinations_zero_after_clear": False,
        },
    )
    retained = module.prepare_retained_tp4_candidate(
        rank=2,
        state_factory=lambda: state,
    )

    try:
        retained.release()
    except RuntimeError as error:
        assert "invariants" in str(error)
    else:
        raise AssertionError("incomplete clear must fail release")


def test_real_retained_state_factory_delegates_to_frozen_runtime_builder():
    module = _load_module()
    state, _, _, _ = _prepared_state(module, rank=3)
    calls = []

    def runtime_builder(**kwargs):
        calls.append(("runtime", kwargs))
        return {"scope_kwargs": {"rank": kwargs["tensor_parallel_rank"]}}

    def retained_state_builder(**kwargs):
        calls.append(("state", kwargs))
        assert kwargs["rank"] == 3
        return state

    retained = module.prepare_real_retained_tp4_candidate(
        checkpoint_dir="/approved/model",
        source_root=ROOT,
        tensor_parallel_size=4,
        tensor_parallel_rank=3,
        runtime_builder=runtime_builder,
        retained_state_builder=retained_state_builder,
    )

    assert retained.ready_row()["tp_rank"] == 3
    assert calls == [
        (
            "runtime",
            {
                "checkpoint_dir": "/approved/model",
                "source_root": ROOT,
                "tensor_parallel_size": 4,
                "tensor_parallel_rank": 3,
            },
        ),
        ("state", {"rank": 3}),
    ]


def test_real_retained_state_keeps_tensor_graph_until_release_then_collects():
    module = _load_module()

    class Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    class Tensor:
        def __init__(self, values):
            self._values = list(values)

        def detach(self):
            return self

        def clone(self):
            return Tensor(self._values)

        def zero_(self):
            self._values = [0 for _ in self._values]
            return self

        def fill_(self, value):
            self._values = [value for _ in self._values]
            return self

        def count_nonzero(self):
            return Scalar(sum(value != 0 for value in self._values))

        def equal(self, other):
            return self._values == other._values

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.no_grad = NoGrad
    fake_torch.cuda = types.SimpleNamespace(
        is_initialized=lambda: False,
    )
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch

    helpers = (
        module.serial_gate.real_binding_gate.load_publish_gate
        .publication_gate.publication.ownership.loader_core
    )
    pool_helpers = (
        module.serial_gate.real_binding_gate.load_publish_gate
        .publication_gate.publication
    )
    original_helpers = {
        "registered": helpers._registered_tensors,
        "identity": helpers._snapshot_identity,
        "require_identity": helpers._require_identity_unchanged,
        "pool_snapshot": pool_helpers._snapshot_pool,
        "pool_unchanged": pool_helpers._pool_unchanged,
    }

    class Box:
        pass

    class Stats:
        assigned_bindings = 320
        source_tensors = 320
        shard_count = 1
        loaded_bytes = 3763655360
        peak_source_bytes = 1017118720

    class Slot:
        def __init__(self):
            self.candidate = None

    def private_graph_factory():
        model = Box()
        selected = Tensor([0, 0, 0, 0])
        non_selected = Tensor([37, 37, 37, 37])
        model.registered = (selected, non_selected)
        pool = Box()
        pool.marker = 41
        assembly = Box()
        assembly.packed = Box()
        assembly.packed.model = model
        binding = Box()
        binding.destination = selected
        binding_plan = Box()
        binding_plan.bindings = (binding,)
        target = Box()
        target.assembly = assembly
        target.pool = pool
        target.binding_plan = binding_plan
        request = Box()
        request.authorization_sha256 = "a" * 64

        def installed_loader(_request):
            selected.fill_(3)
            owner = Box()
            owner.model = model
            owner.pool = pool
            owner.runtime_bridge = Box()
            candidate = Box()
            candidate.owner = owner
            candidate.binding_plan = binding_plan
            candidate.model_fingerprint = "b" * 64
            candidate.stats = Stats()
            return candidate

        return target, request, installed_loader

    def load_and_publish(runner, request):
        candidate = runner.qwen35_checkpoint_candidate_loader(request)
        runner.qwen35_loaded_checkpoint_candidate_slot.candidate = candidate
        runner.bind_qwen35_hybrid_model_owner(candidate.owner)
        return {"status": "published"}

    def bind_published(runner):
        candidate = runner.qwen35_loaded_checkpoint_candidate_slot.candidate
        runner.bind_qwen35_loaded_checkpoint_candidate(candidate)
        return {
            "participant_id": 1,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": "b" * 64,
            "layout_fingerprint": "c" * 64,
            "dtype": "bfloat16",
            "detail": "",
        }

    def bind_owner(runner, owner):
        runner.qwen35_hybrid_model_owner = owner
        runner.hybrid_state_runtime_bridge = owner.runtime_bridge

    def bind_candidate(runner, candidate):
        identity = Box()
        identity.model_fingerprint = candidate.model_fingerprint
        identity.layout_fingerprint = "c" * 64
        identity.dtype = fake_torch.bfloat16
        runner.qwen35_hybrid_prefix_runtime_identity = identity
        runner.qwen35_hybrid_prefix_runtime_identity_owner = candidate.owner

    helpers._registered_tensors = lambda model: model.registered
    helpers._snapshot_identity = lambda tensors: tuple(map(id, tensors))
    helpers._require_identity_unchanged = (
        lambda tensors, snapshot: (
            None
            if tuple(map(id, tensors)) == snapshot
            else (_ for _ in ()).throw(RuntimeError("identity changed"))
        )
    )
    pool_helpers._snapshot_pool = lambda pool: pool.marker
    pool_helpers._pool_unchanged = (
        lambda pool, snapshot: pool.marker == snapshot
    )
    try:
        state = module.prepare_real_retained_candidate_state(
            private_graph_factory=private_graph_factory,
            model_fingerprint="b" * 64,
            methods={
                "load_and_publish_qwen35_checkpoint_candidate": (
                    load_and_publish
                ),
                "bind_published_qwen35_loaded_checkpoint_candidate": (
                    bind_published
                ),
            },
            bind_owner_method=bind_owner,
            bind_candidate_method=bind_candidate,
            production_slot_factory=Slot,
            candidate_validator=lambda **_kwargs: {
                "binding_hash_count": 320,
                "binding_destination_sha256": ["e" * 64] * 320,
                "phase_hash_count": 26,
                "phase_destination_sha256": {
                    f"phase_{index}": "f" * 64
                    for index in range(26)
                },
                "aggregate_destination_sha256": "d" * 64,
                "alias_groups": [],
            },
            payload_recorder=lambda **_kwargs: {},
            rank=1,
            status_reader=lambda: {
                "vmrss_kib": 1_200_000,
                "vmhwm_kib": 2_900_000,
            },
        )
        retained = module.prepare_retained_tp4_candidate(
            rank=1,
            state_factory=lambda: state,
        )
        state = None
        assert retained.ready_row()["all_private_objects_retained"] is True

        released = retained.release()

        assert released["all_private_objects_collected"] is True
        assert all(released["collected_private_objects"].values())
    finally:
        helpers._registered_tensors = original_helpers["registered"]
        helpers._snapshot_identity = original_helpers["identity"]
        helpers._require_identity_unchanged = original_helpers[
            "require_identity"
        ]
        pool_helpers._snapshot_pool = original_helpers["pool_snapshot"]
        pool_helpers._pool_unchanged = original_helpers["pool_unchanged"]
        if original_torch is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = original_torch


def test_real_retained_state_rejects_payload_drift_from_pristine_rank():
    module = _load_module()
    state, _, _, _ = _prepared_state(module, rank=2)
    expected = dict(state.ready_payload)
    expected["binding_destination_sha256"] = ["1" * 64] * 320
    expected["phase_destination_sha256"] = {
        f"phase_{index}": "2" * 64
        for index in range(26)
    }
    expected["alias_groups"] = []
    actual = dict(expected)
    actual["binding_destination_sha256"] = list(
        actual["binding_destination_sha256"]
    )
    actual["binding_destination_sha256"][17] = "3" * 64

    try:
        module.validate_payload_against_pristine_rank(
            actual,
            expected,
            rank=2,
        )
    except ValueError as error:
        assert "binding payload" in str(error)
    else:
        raise AssertionError("pristine rank payload drift must fail")


def _ready_message(module, rank):
    state, _, _, _ = _prepared_state(module, rank=rank)
    retained = module.prepare_retained_tp4_candidate(
        rank=rank,
        state_factory=lambda: state,
    )
    return retained.ready_row()


def test_staggered_coordinator_snapshots_four_live_workers_then_releases_reverse():
    module = _load_module()
    events = []
    alive = set()
    channels = {}

    class Process:
        def __init__(self, rank):
            self.rank = rank
            self.pid = 41_000_000 + rank
            self.exitcode = None

        def start(self):
            events.append(("spawn", self.rank))
            alive.add(self.pid)

        def join(self, timeout=None):
            events.append(("join", self.rank, timeout))
            alive.discard(self.pid)
            self.exitcode = 0

        def is_alive(self):
            return self.pid in alive

    class Channel:
        def __init__(self, rank):
            self.rank = rank
            self.responses = []

        def send(self, message):
            events.append(("send", self.rank, message))
            if message == {"command": "START", "rank": self.rank}:
                row = _ready_message(module, self.rank)
                row["memory"] = {
                    "before": {
                        "vmrss_kib": 300_000,
                        "vmhwm_kib": 300_000,
                    },
                    "ready": dict(row["ready_memory"]),
                }
                self.responses.append({
                    "event": "READY",
                    "rank": self.rank,
                    "row": row,
                })
            elif message == {
                "command": "RELEASE",
                "rank": self.rank,
            }:
                self.responses.append({
                    "event": "RELEASED",
                    "rank": self.rank,
                    "row": {
                        "schema_version": (
                            module.RELEASED_ROW_SCHEMA_VERSION
                        ),
                        "status": "RELEASED",
                        "provenance": module.PROVENANCE,
                        "claim_boundary": module.CLAIM_BOUNDARY,
                        "tp_size": 4,
                        "tp_rank": self.rank,
                        "all_selected_destinations_zero_after_clear": True,
                        "non_selected_tensors_unchanged": True,
                        "tensor_identity_preserved": True,
                        "pool_unchanged": True,
                        "collected_private_objects": {
                            name: True
                            for name in module.PRIVATE_OBJECT_NAMES
                        },
                        "all_private_objects_collected": True,
                    },
                })
            else:
                raise AssertionError(f"unexpected command: {message}")

        def recv(self):
            events.append(("recv", self.rank))
            return self.responses.pop(0)

        def poll(self):
            return bool(self.responses)

    def worker_factory(rank):
        channel = Channel(rank)
        channels[rank] = channel
        return Process(rank), channel

    snapshots = []

    def process_status_reader(pid):
        assert pid in alive
        return {
            "state": "S",
            "vmrss_kib": 1_100_000 + pid % 4,
            "vmhwm_kib": 2_900_000 + pid % 4,
        }

    result = module.run_staggered_tp4_candidate_residency(
        worker_factory=worker_factory,
        process_status_reader=process_status_reader,
        snapshot_writer=lambda snapshot: snapshots.append(snapshot),
        join_timeout_s=3.0,
    )

    assert result["start_order"] == [0, 1, 2, 3]
    assert result["ready_order"] == [0, 1, 2, 3]
    assert result["release_order"] == [3, 2, 1, 0]
    assert result["released_order"] == [3, 2, 1, 0]
    assert result["all_workers_exited"] is True
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["all_workers_live_concurrently"] is True
    assert (
        isinstance(snapshot["snapshot_unix_time_ns"], int)
        and snapshot["snapshot_unix_time_ns"] > 0
    )
    assert snapshot["live_process_ids"] == [
        41_000_000,
        41_000_001,
        41_000_002,
        41_000_003,
    ]
    assert snapshot["release_acknowledgement_count"] == 0
    assert [
        row["ready_memory"]["vmrss_kib"]
        for row in result["ready_rows"]
    ] == [1_100_000, 1_100_001, 1_100_002, 1_100_003]
    assert all(
        row["memory"]["ready"] == row["ready_memory"]
        for row in result["ready_rows"]
    )
    assert not alive
    assert [event for event in events if event[0] == "spawn"] == [
        ("spawn", 0),
        ("spawn", 1),
        ("spawn", 2),
        ("spawn", 3),
    ]
    assert [
        event[1]
        for event in events
        if event[:1] == ("send",)
        and event[2]["command"] == "START"
    ] == [0, 1, 2, 3]
    assert [
        event[1]
        for event in events
        if event[:1] == ("send",)
        and event[2]["command"] == "RELEASE"
    ] == [3, 2, 1, 0]


def test_staggered_coordinator_aborts_all_workers_on_premature_release():
    module = _load_module()
    events = []
    alive = set()

    class Process:
        def __init__(self, rank):
            self.rank = rank
            self.pid = 42_000_000 + rank
            self.exitcode = None

        def start(self):
            alive.add(self.pid)

        def join(self, timeout=None):
            events.append(("join", self.rank, timeout))
            alive.discard(self.pid)
            self.exitcode = 0

        def is_alive(self):
            return self.pid in alive

    class Channel:
        def __init__(self, rank):
            self.rank = rank
            self.responses = []
            self.premature = rank == 2

        def send(self, message):
            events.append(("send", self.rank, message["command"]))
            if message["command"] == "START":
                self.responses.append({
                    "event": "READY",
                    "rank": self.rank,
                    "row": _ready_message(module, self.rank),
                })
            elif message["command"] == "ABORT":
                self.premature = False
                self.responses = [{
                    "event": "RELEASED",
                    "rank": self.rank,
                    "row": {
                        "schema_version": (
                            module.RELEASED_ROW_SCHEMA_VERSION
                        ),
                        "status": "RELEASED",
                        "provenance": module.PROVENANCE,
                        "claim_boundary": module.CLAIM_BOUNDARY,
                        "tp_size": 4,
                        "tp_rank": self.rank,
                        "all_selected_destinations_zero_after_clear": True,
                        "non_selected_tensors_unchanged": True,
                        "tensor_identity_preserved": True,
                        "pool_unchanged": True,
                        "collected_private_objects": {
                            name: True
                            for name in module.PRIVATE_OBJECT_NAMES
                        },
                        "all_private_objects_collected": True,
                    },
                }]

        def recv(self):
            return self.responses.pop(0)

        def poll(self):
            return self.premature or bool(self.responses)

    def worker_factory(rank):
        return Process(rank), Channel(rank)

    try:
        module.run_staggered_tp4_candidate_residency(
            worker_factory=worker_factory,
            process_status_reader=lambda pid: {
                "state": "S",
                "vmrss_kib": 1,
                "vmhwm_kib": 1,
            },
            snapshot_writer=lambda snapshot: None,
            join_timeout_s=3.0,
        )
    except RuntimeError as error:
        assert "before concurrent snapshot" in str(error)
    else:
        raise AssertionError("premature release must fail")

    assert [
        event[1]
        for event in events
        if event[0] == "send" and event[2] == "ABORT"
    ] == [3, 2, 1, 0]
    assert len([event for event in events if event[0] == "join"]) == 4
    assert not alive


def test_retained_candidate_worker_waits_for_start_then_releases():
    module = _load_module()
    state, selected, _, _ = _prepared_state(module, rank=1)
    sent = []

    class Channel:
        def __init__(self):
            self.messages = [
                {"command": "START", "rank": 1},
                {"command": "RELEASE", "rank": 1},
            ]

        def recv(self):
            return self.messages.pop(0)

        def send(self, message):
            sent.append(message)

    result = module.run_retained_candidate_worker(
        rank=1,
        channel=Channel(),
        retained_factory=lambda: module.prepare_retained_tp4_candidate(
            rank=1,
            state_factory=lambda: state,
        ),
    )

    assert result == 0
    assert [message["event"] for message in sent] == [
        "READY",
        "RELEASED",
    ]
    assert sent[0]["rank"] == 1
    assert sent[1]["rank"] == 1
    assert all(tensor.value == 0 for tensor in selected)


def test_retained_candidate_worker_binds_before_and_ready_memory():
    module = _load_module()
    state, _, _, _ = _prepared_state(module, rank=1)
    retained = module.prepare_retained_tp4_candidate(
        rank=1,
        state_factory=lambda: state,
    )
    sent = []

    class Channel:
        def __init__(self):
            self.messages = [
                {"command": "START", "rank": 1},
                {"command": "RELEASE", "rank": 1},
            ]

        def recv(self):
            return self.messages.pop(0)

        def send(self, message):
            sent.append(message)

    result = module.run_retained_candidate_worker(
        rank=1,
        channel=Channel(),
        retained_factory=lambda: retained,
        status_reader=lambda: {
            "vmrss_kib": 300_000,
            "vmhwm_kib": 300_000,
        },
    )

    assert result == 0
    assert sent[0]["row"]["memory"] == {
        "before": {
            "vmrss_kib": 300_000,
            "vmhwm_kib": 300_000,
        },
        "ready": {
            "vmrss_kib": 1_200_000,
            "vmhwm_kib": 2_900_000,
        },
    }


def test_source_bound_cli_exposes_exact_four_modes():
    module = _load_module()
    parser = module._parser()
    action = next(
        action
        for action in parser._actions
        if action.dest == "command"
    )
    assert set(action.choices) == {
        "run",
        "internal-worker",
        "internal-finalize",
        "validate",
    }
    worker = parser.parse_args([
        "internal-worker",
        "--rank",
        "2",
        "--source-root",
        "/source",
        "--checkpoint-dir",
        "/checkpoint",
        "--channel-fd",
        "19",
        "--pristine-row",
        "/oracle-rank2.json",
    ])
    assert worker.channel_fd == 19
    assert worker.pristine_row == "/oracle-rank2.json"


def test_internal_worker_cli_uses_inherited_pipe_and_pristine_row():
    module = _load_module()
    parent, child = multiprocessing.Pipe(duplex=True)
    calls = []
    original = module.run_real_retained_candidate_worker

    def worker(**kwargs):
        calls.append(kwargs)
        kwargs["channel"].send({"event": "WORKER_TEST"})
        return 0

    module.run_real_retained_candidate_worker = worker
    try:
        inherited_fd = os.dup(child.fileno())
        with tempfile.TemporaryDirectory() as directory:
            pristine = Path(directory) / "rank2.json"
            pristine.write_text(
                json.dumps({"tp_rank": 2}),
                encoding="utf-8",
            )
            result = module.main([
                "internal-worker",
                "--rank",
                "2",
                "--source-root",
                "/source",
                "--checkpoint-dir",
                "/checkpoint",
                "--channel-fd",
                str(inherited_fd),
                "--pristine-row",
                str(pristine),
            ])
        assert result == 0
        assert parent.recv() == {"event": "WORKER_TEST"}
        assert calls[0]["rank"] == 2
        assert calls[0]["source_root"] == "/source"
        assert calls[0]["checkpoint_dir"] == "/checkpoint"
        assert calls[0]["pristine_row"] == {"tp_rank": 2}
    finally:
        module.run_real_retained_candidate_worker = original
        parent.close()
        child.close()


def test_retained_candidate_worker_rejects_release_before_start():
    module = _load_module()
    sent = []

    class Channel:
        def recv(self):
            return {"command": "RELEASE", "rank": 0}

        def send(self, message):
            sent.append(message)

    try:
        module.run_retained_candidate_worker(
            rank=0,
            channel=Channel(),
            retained_factory=lambda: None,
        )
    except RuntimeError as error:
        assert "START" in str(error)
    else:
        raise AssertionError("release before start must fail")
    assert not sent


def _memory_evidence(module):
    ready_rows = []
    process_status = []
    for rank in range(4):
        row = _ready_message(module, rank)
        row["memory"] = {
            "before": {
                "vmrss_kib": 300_000,
                "vmhwm_kib": 300_000,
            },
            "ready": {
                "vmrss_kib": 1_700_000,
                "vmhwm_kib": 2_900_000,
            },
        }
        ready_rows.append(row)
        process_status.append({
            "rank": rank,
            "process_id": row["process_id"],
            "state": "S",
            "vmrss_kib": 1_700_000,
            "vmhwm_kib": 2_900_000,
        })
    return {
        "host_memory_before": {
            "mem_available_kib": 20_000_000,
            "swap_total_kib": 0,
            "swap_free_kib": 0,
        },
        "host_memory_ready": {
            "mem_available_kib": 13_000_000,
            "swap_total_kib": 0,
            "swap_free_kib": 0,
        },
        "ready_rows": ready_rows,
        "concurrent_snapshot": {
            "process_status": process_status,
            "all_workers_live_concurrently": True,
            "release_acknowledgement_count": 0,
        },
    }


def _released_row(module, rank):
    return {
        "schema_version": module.RELEASED_ROW_SCHEMA_VERSION,
        "status": "RELEASED",
        "provenance": module.PROVENANCE,
        "claim_boundary": module.CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "collected_private_objects": {
            name: True for name in module.PRIVATE_OBJECT_NAMES
        },
        "all_private_objects_collected": True,
    }


def test_aggregate_memory_contract_recomputes_raw_observations():
    module = _load_module()
    evidence = _memory_evidence(module)

    summary = module.validate_aggregate_memory_contract(evidence)

    assert summary == {
        "per_worker_total_vmhwm_increment_kib": [
            2_600_000,
            2_600_000,
            2_600_000,
            2_600_000,
        ],
        "aggregate_worker_vmhwm_increment_kib": 10_400_000,
        "aggregate_ready_vmrss_kib": 6_800_000,
        "host_mem_available_decrease_kib": 7_000_000,
        "memory_contract_passed": True,
    }


def test_aggregate_memory_contract_rejects_worker_ceiling_breach():
    module = _load_module()
    evidence = _memory_evidence(module)
    evidence["ready_rows"][3]["memory"]["ready"]["vmhwm_kib"] = (
        3_600_000
    )

    try:
        module.validate_aggregate_memory_contract(evidence)
    except ValueError as error:
        assert "rank=3" in str(error)
        assert "3145728" in str(error)
    else:
        raise AssertionError("per-worker memory breach must fail")


def test_aggregate_memory_contract_rejects_insufficient_host_preflight():
    module = _load_module()
    evidence = _memory_evidence(module)
    evidence["host_memory_before"]["mem_available_kib"] = 16_000_000

    try:
        module.validate_aggregate_memory_contract(evidence)
    except ValueError as error:
        assert "16777216" in str(error)
    else:
        raise AssertionError("insufficient host memory must fail")


def test_ownership_artifact_binds_snapshot_memory_and_release_rows():
    module = _load_module()
    evidence = _memory_evidence(module)
    snapshot = {
        "schema_version": (
            "qwen35.tp4-live-concurrent-candidate-snapshot.v1"
        ),
        "coordinator_process_id": 50_000_000,
        "start_order": [0, 1, 2, 3],
        "ready_order": [0, 1, 2, 3],
        "live_process_ids": [
            row["process_id"] for row in evidence["ready_rows"]
        ],
        "ready_row_sha256": [
            module._sha256(module._canonical(row))
            for row in evidence["ready_rows"]
        ],
        "process_status": evidence[
            "concurrent_snapshot"
        ]["process_status"],
        "release_acknowledgement_count": 0,
        "all_workers_live_concurrently": True,
    }
    released = [_released_row(module, rank) for rank in (3, 2, 1, 0)]

    artifact = module.build_ownership_artifact(
        ready_rows=evidence["ready_rows"],
        concurrent_snapshot=snapshot,
        released_rows=released,
        host_memory_before=evidence["host_memory_before"],
        host_memory_ready=evidence["host_memory_ready"],
        source_file_sha256={"tools/example.py": "d" * 64},
        source_tree_sha256="e" * 64,
        prerequisite_oracle_sha256="f" * 64,
    )

    assert artifact["schema_version"] == (
        "qwen35.tp4-live-concurrent-candidate-ownership.v1"
    )
    assert artifact["status"] == "PASS"
    assert artifact["start_order"] == [0, 1, 2, 3]
    assert artifact["ready_order"] == [0, 1, 2, 3]
    assert artifact["release_order"] == [3, 2, 1, 0]
    assert artifact["all_workers_exited"] is True
    assert artifact["memory_summary"]["memory_contract_passed"] is True
    assert artifact["ready_rows_sha256"] == module._sha256(
        module._canonical(evidence["ready_rows"])
    )
    assert artifact["released_rows_sha256"] == module._sha256(
        module._canonical(released)
    )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 live concurrent retained-scope tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
