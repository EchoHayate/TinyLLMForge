from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.qwen35_mtp_graph import (
    Qwen35MTPExactGraphRunner,
    Qwen35MTPGraphEntry,
    Qwen35MTPGraphIdentity,
    Qwen35MTPGraphPreReplayError,
    Qwen35MTPGraphReplayError,
)


def _identity(**overrides):
    values = {
        "exact_q": 4,
        "exact_batch_size": 2,
        "device_index": 0,
        "compute_dtype": "torch.bfloat16",
        "hidden_size": 2048,
        "mtp_layer_count": 1,
        "block_table_width": 17,
    }
    values.update(overrides)
    return Qwen35MTPGraphIdentity(**values)


def test_identity_is_exact_for_every_static_dimension():
    identity = _identity()
    assert identity.sha256 == _identity().sha256
    for overrides in (
        {"exact_q": 3},
        {"exact_batch_size": 1},
        {"device_index": 1},
        {"compute_dtype": "torch.float16"},
        {"hidden_size": 1024},
        {"mtp_layer_count": 2},
        {"block_table_width": 18},
    ):
        assert _identity(**overrides).sha256 != identity.sha256


@pytest.mark.parametrize(
    "overrides",
    (
        {"exact_q": 1},
        {"exact_q": 0},
        {"exact_batch_size": 0},
        {"device_index": -1},
        {"compute_dtype": ""},
        {"hidden_size": 0},
        {"mtp_layer_count": 0},
        {"block_table_width": 0},
    ),
)
def test_identity_rejects_invalid_or_no_forward_family(overrides):
    with pytest.raises(ValueError):
        _identity(**overrides)


@dataclass(frozen=True)
class _Row:
    sequence_id: int
    live_slot_ids: tuple[int, ...]


@dataclass
class _ScratchLease:
    rows: tuple[_Row, ...]
    state: str = "active"


class _ScratchOwner:

    def __init__(self):
        self.acquired = []
        self.rolled_back = []
        self.fail_rollback = False

    def acquire(self, identity, rows):
        scratch_rows = tuple(
            replace(
                row,
                live_slot_ids=tuple(
                    slot_id + 10_000
                    for slot_id in row.live_slot_ids
                ),
            )
            for row in rows
        )
        lease = _ScratchLease(scratch_rows)
        self.acquired.append((identity, rows, lease))
        return lease

    def rollback(self, lease):
        if self.fail_rollback:
            raise RuntimeError("scratch rollback failed")
        assert lease.state == "active"
        lease.state = "rolled_back"
        self.rolled_back.append(lease)


class _CaptureBackend:

    def __init__(self):
        self.capture_calls = []
        self.replay_calls = []
        self.fail_capture = False
        self.fail_pre_replay = False
        self.fail_replay = False

    def estimate_static_bytes(self, identity, rows):
        return identity.exact_q * len(rows) * 64

    def capture(self, identity, rows, eager, scratch_lease):
        self.capture_calls.append((
            identity,
            rows,
            scratch_lease,
        ))
        if self.fail_capture:
            raise RuntimeError("capture failed")
        eager(identity.exact_q, rows)
        return Qwen35MTPGraphEntry(
            identity=identity,
            graph=object(),
            static_bytes=self.estimate_static_bytes(
                identity,
                rows,
            ),
            capture_duration_ns=100,
            reserved_delta_bytes=32,
        )

    def replay(self, entry, rows):
        self.replay_calls.append((entry, rows))
        if self.fail_pre_replay:
            raise Qwen35MTPGraphPreReplayError(
                "static input mismatch"
            )
        if self.fail_replay:
            raise RuntimeError("replay failed")
        return tuple(
            ("replay", entry.identity.exact_q, row.sequence_id)
            for row in rows
        )


def _runner(
    *,
    backend=None,
    scratch=None,
    min_observations=2,
    q_allowlist=(2, 3, 4),
    batch_allowlist=(1, 2, 4),
):
    return Qwen35MTPExactGraphRunner(
        enabled=True,
        q_allowlist=q_allowlist,
        batch_allowlist=batch_allowlist,
        min_observations=min_observations,
        max_entries=8,
        max_static_bytes=1 << 20,
        max_reserved_bytes=1 << 20,
        max_total_capture_ns=1_000_000,
        max_single_capture_ns=1_000_000,
        device_index=0,
        compute_dtype="torch.bfloat16",
        hidden_size=2048,
        mtp_layer_count=1,
        block_table_width=17,
        capture_backend=(
            _CaptureBackend() if backend is None else backend
        ),
        scratch_owner=(
            _ScratchOwner() if scratch is None else scratch
        ),
    )


def test_eager_then_private_capture_then_exact_replay():
    backend = _CaptureBackend()
    scratch = _ScratchOwner()
    runner = _runner(backend=backend, scratch=scratch)
    rows = (_Row(7, (1, 2)), _Row(9, (3, 4)))
    eager_calls = []

    def eager(exact_q, eager_rows):
        eager_calls.append((exact_q, eager_rows))
        return tuple(
            ("eager", exact_q, row.sequence_id)
            for row in eager_rows
        )

    first = runner.run(exact_q=4, rows=rows, eager=eager)
    second = runner.run(exact_q=4, rows=rows, eager=eager)
    third = runner.run(exact_q=4, rows=rows, eager=eager)

    assert first == (
        ("eager", 4, 7),
        ("eager", 4, 9),
    )
    assert second == first
    assert third == (
        ("replay", 4, 7),
        ("replay", 4, 9),
    )
    assert [call[0] for call in eager_calls] == [4, 4, 4]
    capture_rows = backend.capture_calls[0][1]
    assert capture_rows != rows
    assert rows == (_Row(7, (1, 2)), _Row(9, (3, 4)))
    assert all(
        slot_id >= 10_000
        for row in capture_rows
        for slot_id in row.live_slot_ids
    )
    assert len(scratch.rolled_back) == 1
    assert runner.summary()["captures"] == 1
    assert runner.summary()["replays"] == 1


def test_failed_eager_does_not_advance_capture_admission():
    runner = _runner(min_observations=1)
    rows = (_Row(7, (1,)),)

    def fail(*_):
        raise RuntimeError("eager failed")

    with pytest.raises(RuntimeError, match="eager failed"):
        runner.run(exact_q=2, rows=rows, eager=fail)
    assert runner.summary()["observation_counts"] == {}
    assert runner.summary()["capture_attempts"] == 0


def test_capture_failure_quarantines_only_exact_identity():
    backend = _CaptureBackend()
    backend.fail_capture = True
    runner = _runner(backend=backend, min_observations=1)
    eager_calls = []

    def eager(exact_q, rows):
        eager_calls.append((exact_q, rows))
        return tuple(row.sequence_id for row in rows)

    rows = (_Row(7, (1,)),)
    assert runner.run(exact_q=2, rows=rows, eager=eager) == (7,)
    first_identity = _identity(
        exact_q=2,
        exact_batch_size=1,
    )
    assert runner.quarantine_reason(first_identity) == "capture_failed"

    backend.fail_capture = False
    assert runner.run(exact_q=3, rows=rows, eager=eager) == (7,)
    second_identity = _identity(
        exact_q=3,
        exact_batch_size=1,
    )
    assert runner.quarantine_reason(second_identity) is None
    assert runner.ready_identity_sha256s() == (
        second_identity.sha256,
    )


def test_scratch_rollback_failure_prevents_publish():
    scratch = _ScratchOwner()
    scratch.fail_rollback = True
    runner = _runner(scratch=scratch, min_observations=1)
    rows = (_Row(7, (1,)),)

    with pytest.raises(RuntimeError, match="scratch rollback"):
        runner.run(
            exact_q=2,
            rows=rows,
            eager=lambda exact_q, rows: (),
        )
    identity = _identity(exact_q=2, exact_batch_size=1)
    assert (
        runner.quarantine_reason(identity)
        == "capture_rollback_failed"
    )
    assert runner.ready_identity_sha256s() == ()


def test_replay_failure_propagates_without_eager_retry():
    backend = _CaptureBackend()
    runner = _runner(
        backend=backend,
        min_observations=1,
    )
    rows = (_Row(7, (1,)),)
    eager_calls = []

    def eager(exact_q, eager_rows):
        eager_calls.append((exact_q, eager_rows))
        return (7,)

    assert runner.run(exact_q=2, rows=rows, eager=eager) == (7,)
    assert len(eager_calls) == 2
    backend.fail_replay = True
    with pytest.raises(Qwen35MTPGraphReplayError) as exc_info:
        runner.run(exact_q=2, rows=rows, eager=eager)
    assert isinstance(exc_info.value.cause, RuntimeError)
    assert len(eager_calls) == 2
    identity = _identity(exact_q=2, exact_batch_size=1)
    assert runner.quarantine_reason(identity) == "replay_failed"


def test_pre_replay_failure_falls_back_without_quarantine():
    backend = _CaptureBackend()
    runner = _runner(
        backend=backend,
        min_observations=1,
    )
    rows = (_Row(7, (1,)),)
    eager_calls = []

    def eager(exact_q, eager_rows):
        eager_calls.append((exact_q, eager_rows))
        return (("eager", exact_q, eager_rows[0].sequence_id),)

    expected = (("eager", 2, 7),)
    assert runner.run(exact_q=2, rows=rows, eager=eager) == expected
    assert len(eager_calls) == 2

    backend.fail_pre_replay = True
    assert runner.run(exact_q=2, rows=rows, eager=eager) == expected

    identity = _identity(exact_q=2, exact_batch_size=1)
    assert len(eager_calls) == 3
    assert runner.quarantine_reason(identity) is None
    assert runner.summary()["replays"] == 0
    assert runner.summary()["fallback_pre_replay"] == 1


def _load_config_class():
    config_path = ROOT / "tinyvllm/config.py"
    fake_transformers = types.ModuleType("transformers")
    module_name = "qwen35_mtp_graph_config_test"
    config_module = types.ModuleType(module_name)
    config_module.__file__ = os.fspath(config_path)

    class _AutoConfig:
        @staticmethod
        def from_pretrained(_):
            return types.SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = _AutoConfig
    original = sys.modules.get("transformers")
    original_config_module = sys.modules.get(module_name)
    sys.modules["transformers"] = fake_transformers
    sys.modules[module_name] = config_module
    try:
        exec(
            compile(
                config_path.read_text(),
                os.fspath(config_path),
                "exec",
            ),
            config_module.__dict__,
        )
        return config_module.Config
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original
        if original_config_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_config_module


def test_qwen35_mtp_graph_config_defaults_and_normalization():
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        config = Config(
            model=model,
            qwen35_mtp_cuda_graph_q_allowlist=[4, 2, 4],
            qwen35_mtp_cuda_graph_batch_allowlist=[4, 1, 4],
        )
    assert config.qwen35_mtp_cuda_graphs is False
    assert config.qwen35_mtp_cuda_graph_q_allowlist == (2, 4)
    assert config.qwen35_mtp_cuda_graph_batch_allowlist == (1, 4)
    assert config.qwen35_mtp_cuda_graph_min_observations == 2
    assert config.qwen35_mtp_cuda_graph_max_entries == 8


@pytest.mark.parametrize(
    "overrides",
    (
        {"qwen35_mtp_cuda_graphs": 1},
        {"qwen35_mtp_cuda_graph_q_allowlist": (1, 2)},
        {"qwen35_mtp_cuda_graph_batch_allowlist": ()},
        {"qwen35_mtp_cuda_graph_min_observations": 0},
        {"qwen35_mtp_cuda_graph_max_entries": True},
        {"qwen35_mtp_cuda_graph_max_static_bytes": 0},
        {"qwen35_mtp_cuda_graph_max_reserved_bytes": 0},
        {"qwen35_mtp_cuda_graph_max_total_capture_ns": 0},
        {"qwen35_mtp_cuda_graph_max_single_capture_ns": 0},
    ),
)
def test_qwen35_mtp_graph_config_rejects_invalid_controls(overrides):
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        with pytest.raises((AssertionError, TypeError, ValueError)):
            Config(model=model, **overrides)
