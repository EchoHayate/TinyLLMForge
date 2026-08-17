from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.autoregressive_draft_graph import (
    AutoregressiveDraftExactGraphRunner,
    AutoregressiveDraftGraphEntry,
    AutoregressiveDraftGraphIdentity,
    AutoregressiveDraftGraphPreReplayError,
    AutoregressiveDraftGraphReplayError,
)


def _identity(**overrides):
    values = {
        "exact_q": 4,
        "exact_batch_size": 4,
        "tensor_parallel_size": 4,
        "tensor_parallel_rank": 2,
        "device_index": 6,
        "compute_dtype": "torch.bfloat16",
        "backend_identity": "qwen3",
        "model_fingerprint": "model-sha256",
        "tokenizer_fingerprint": "tokenizer-sha256",
        "local_query_heads": 8,
        "local_kv_heads": 2,
        "kv_block_table_width": 512,
        "proposal_kv_capacity": 4096,
        "blockwise_offload": False,
    }
    values.update(overrides)
    return AutoregressiveDraftGraphIdentity(**values)


def test_identity_is_exact_for_every_static_dimension():
    identity = _identity()
    assert identity.sha256 == _identity().sha256
    for overrides in (
        {"exact_q": 3},
        {"exact_batch_size": 1},
        {"tensor_parallel_size": 3},
        {"tensor_parallel_rank": 1},
        {"device_index": 7},
        {"compute_dtype": "torch.float16"},
        {"backend_identity": "other-backend"},
        {"model_fingerprint": "other-model"},
        {"tokenizer_fingerprint": "other-tokenizer"},
        {"local_query_heads": 4},
        {"local_kv_heads": 1},
        {"kv_block_table_width": 513},
        {"proposal_kv_capacity": 4097},
        {"blockwise_offload": True},
    ):
        assert _identity(**overrides).sha256 != identity.sha256


@pytest.mark.parametrize(
    "overrides",
    (
        {"exact_q": 1},
        {"exact_batch_size": 0},
        {"tensor_parallel_size": 0},
        {"tensor_parallel_rank": -1},
        {"tensor_parallel_rank": 4},
        {"device_index": -1},
        {"compute_dtype": ""},
        {"backend_identity": ""},
        {"model_fingerprint": ""},
        {"tokenizer_fingerprint": ""},
        {"local_query_heads": 0},
        {"local_kv_heads": 0},
        {"kv_block_table_width": 0},
        {"proposal_kv_capacity": 0},
        {"blockwise_offload": 0},
    ),
)
def test_identity_rejects_invalid_fields(overrides):
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


@dataclass(frozen=True)
class _Transaction:
    transaction_id: str


@dataclass(frozen=True)
class _PreparedReplay:
    transactions: tuple[_Transaction, ...]


class _ScratchOwner:

    def __init__(self):
        self.acquired = []
        self.rolled_back = []
        self.fail_acquire = False
        self.fail_rollback = False

    def acquire(self, identity, rows):
        if self.fail_acquire:
            raise RuntimeError("scratch unavailable")
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
        self.aborted_prepared = []
        self.aborted_results = []

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
        return AutoregressiveDraftGraphEntry(
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
            raise AutoregressiveDraftGraphPreReplayError(
                "static input mismatch"
            )
        if self.fail_replay:
            raise RuntimeError("replay failed")
        return tuple(
            ("graph", entry.identity.exact_q, row.sequence_id)
            for row in rows
        )

    def prepare_replay(self, entry, rows):
        if self.fail_pre_replay:
            raise AutoregressiveDraftGraphPreReplayError(
                "static input mismatch"
            )
        return _PreparedReplay(tuple(
            _Transaction(f"transaction-{row.sequence_id}")
            for row in rows
        ))

    def replay_prepared(self, entry, rows, prepared):
        self.replay_calls.append((entry, rows, prepared))
        if self.fail_replay:
            raise RuntimeError("replay failed")
        return tuple(
            ("graph", entry.identity.exact_q, row.sequence_id)
            for row in rows
        )

    def abort_prepared(self, prepared):
        self.aborted_prepared.append(prepared)

    def abort_replay_result(self, result):
        self.aborted_results.append(result)


def _runner(
    *,
    backend=None,
    scratch=None,
    min_observations=2,
    q_allowlist=(4,),
    batch_allowlist=(4,),
):
    return AutoregressiveDraftExactGraphRunner(
        enabled=True,
        q_allowlist=q_allowlist,
        batch_allowlist=batch_allowlist,
        min_observations=min_observations,
        max_entries=4,
        max_static_bytes=1 << 20,
        max_reserved_bytes=1 << 20,
        max_total_capture_ns=1_000_000,
        max_single_capture_ns=1_000_000,
        tensor_parallel_size=4,
        tensor_parallel_rank=2,
        device_index=6,
        compute_dtype="torch.bfloat16",
        backend_identity="qwen3",
        model_fingerprint="model-sha256",
        tokenizer_fingerprint="tokenizer-sha256",
        local_query_heads=8,
        local_kv_heads=2,
        kv_block_table_width=512,
        proposal_kv_capacity=4096,
        blockwise_offload=False,
        capture_backend=(
            _CaptureBackend() if backend is None else backend
        ),
        scratch_owner=(
            _ScratchOwner() if scratch is None else scratch
        ),
    )


def _rows():
    return tuple(
        _Row(sequence_id, (sequence_id, sequence_id + 100))
        for sequence_id in range(1, 5)
    )


def _eager_recorder(calls):
    def eager(exact_q, rows):
        calls.append((exact_q, rows))
        return tuple(
            ("eager", exact_q, row.sequence_id)
            for row in rows
        )

    return eager


def test_successful_eager_observations_precede_private_capture_and_replay():
    backend = _CaptureBackend()
    scratch = _ScratchOwner()
    runner = _runner(backend=backend, scratch=scratch)
    rows = _rows()
    eager_calls = []
    eager = _eager_recorder(eager_calls)

    first = runner.run(exact_q=4, rows=rows, eager=eager)
    second = runner.run(exact_q=4, rows=rows, eager=eager)
    third = runner.run(exact_q=4, rows=rows, eager=eager)

    assert first == tuple(
        ("eager", 4, row.sequence_id) for row in rows
    )
    assert second == first
    assert third == tuple(
        ("graph", 4, row.sequence_id) for row in rows
    )
    assert [call[0] for call in eager_calls] == [4, 4, 4]
    assert len(backend.capture_calls) == 1
    capture_rows = backend.capture_calls[0][1]
    assert capture_rows != rows
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

    def fail(*_):
        raise RuntimeError("eager failed")

    with pytest.raises(RuntimeError, match="eager failed"):
        runner.run(exact_q=4, rows=_rows(), eager=fail)

    assert runner.summary()["observation_counts"] == {}
    assert runner.summary()["capture_attempts"] == 0


def test_unsupported_exact_family_stays_eager_without_observation():
    runner = _runner()
    eager_calls = []
    eager = _eager_recorder(eager_calls)
    rows = _rows()[:1]

    result = runner.run(exact_q=3, rows=rows, eager=eager)

    assert result == (("eager", 3, 1),)
    assert runner.summary()["observation_counts"] == {}
    assert runner.summary()["capture_attempts"] == 0


def test_capture_failure_rolls_back_and_quarantines_exact_identity():
    backend = _CaptureBackend()
    backend.fail_capture = True
    scratch = _ScratchOwner()
    runner = _runner(
        backend=backend,
        scratch=scratch,
        min_observations=1,
    )
    eager_calls = []

    result = runner.run(
        exact_q=4,
        rows=_rows(),
        eager=_eager_recorder(eager_calls),
    )

    assert result[0][:2] == ("eager", 4)
    assert len(scratch.rolled_back) == 1
    assert runner.summary()["captures"] == 0
    assert runner.summary()["quarantines"] == 1
    assert tuple(runner.summary()["quarantined"].values()) == (
        "capture_failed",
    )


def test_pre_replay_failure_falls_back_to_one_eager_attempt():
    backend = _CaptureBackend()
    runner = _runner(backend=backend, min_observations=1)
    eager_calls = []
    eager = _eager_recorder(eager_calls)
    rows = _rows()

    runner.run(exact_q=4, rows=rows, eager=eager)
    backend.fail_pre_replay = True
    result = runner.run(exact_q=4, rows=rows, eager=eager)

    assert result == tuple(
        ("eager", 4, row.sequence_id) for row in rows
    )
    assert len(eager_calls) == 3
    assert runner.summary()["fallback_pre_replay"] == 1
    assert runner.summary()["quarantines"] == 0


def test_replay_started_failure_quarantines_without_eager_retry():
    backend = _CaptureBackend()
    runner = _runner(backend=backend, min_observations=1)
    eager_calls = []
    eager = _eager_recorder(eager_calls)
    rows = _rows()

    runner.run(exact_q=4, rows=rows, eager=eager)
    backend.fail_replay = True
    with pytest.raises(
        AutoregressiveDraftGraphReplayError
    ) as caught:
        runner.run(exact_q=4, rows=rows, eager=eager)

    assert caught.value.identity == _identity()
    assert len(eager_calls) == 2
    assert runner.summary()["quarantines"] == 1
    assert tuple(runner.summary()["quarantined"].values()) == (
        "replay_failed",
    )


def test_pre_replay_convergence_failure_falls_back_on_every_rank():
    backend = _CaptureBackend()
    runner = _runner(backend=backend, min_observations=1)
    eager_calls = []
    eager = _eager_recorder(eager_calls)
    rows = _rows()
    stages = []

    def converge(*, stage, rows, local_error):
        stages.append((stage, rows, local_error))
        if stage == "graph_pre_replay":
            raise RuntimeError("peer preflight failed")

    runner.bind_convergence(converge)
    runner.run(exact_q=4, rows=rows, eager=eager)
    result = runner.run(exact_q=4, rows=rows, eager=eager)

    assert result == tuple(
        ("eager", 4, row.sequence_id) for row in rows
    )
    assert len(eager_calls) == 3
    assert backend.replay_calls == []
    assert stages[0][0] == "graph_pre_replay"
    assert stages[0][1] == {
        "exact_q": 4,
        "sequence_ids": (1, 2, 3, 4),
        "transaction_ids": (
            "transaction-1",
            "transaction-2",
            "transaction-3",
            "transaction-4",
        ),
    }
    assert runner.summary()["fallback_pre_replay"] == 1
    assert runner.summary()["quarantines"] == 0


def test_replay_complete_convergence_failure_quarantines_without_retry():
    backend = _CaptureBackend()
    runner = _runner(backend=backend, min_observations=1)
    eager_calls = []
    eager = _eager_recorder(eager_calls)
    rows = _rows()
    stages = []

    def converge(*, stage, rows, local_error):
        stages.append((stage, rows, local_error))
        if stage == "graph_replay_complete":
            raise RuntimeError("peer replay failed")

    runner.bind_convergence(converge)
    runner.run(exact_q=4, rows=rows, eager=eager)
    with pytest.raises(AutoregressiveDraftGraphReplayError):
        runner.run(exact_q=4, rows=rows, eager=eager)

    assert len(eager_calls) == 2
    assert [stage for stage, _, _ in stages] == [
        "graph_pre_replay",
        "graph_replay_complete",
    ]
    assert stages[1][1] == {
        "exact_q": 4,
        "sequence_ids": (1, 2, 3, 4),
        "transaction_ids": (
            "transaction-1",
            "transaction-2",
            "transaction-3",
            "transaction-4",
        ),
        "token_rows": (
            ("graph", 4, 1),
            ("graph", 4, 2),
            ("graph", 4, 3),
            ("graph", 4, 4),
        ),
    }
    assert runner.summary()["quarantines"] == 1


def test_scratch_rollback_failure_is_hard_and_quarantined():
    scratch = _ScratchOwner()
    scratch.fail_rollback = True
    runner = _runner(
        scratch=scratch,
        min_observations=1,
    )

    with pytest.raises(
        RuntimeError,
        match="scratch rollback failed",
    ):
        runner.run(
            exact_q=4,
            rows=_rows(),
            eager=_eager_recorder([]),
        )

    assert tuple(runner.summary()["quarantined"].values()) == (
        "capture_rollback_failed",
    )
