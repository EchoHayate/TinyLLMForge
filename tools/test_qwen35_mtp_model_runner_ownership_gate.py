from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
REMOTE_WRAPPER = Path(
    REPO_ROOT,
    "tools",
    "run_qwen35_mtp_model_runner_ownership_gate_remote.sh",
)
OWNERSHIP_GATE = Path(
    REPO_ROOT,
    "tools",
    "qwen35_mtp_model_runner_ownership_gate.py",
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.speculative",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        os.path.join(
            REPO_ROOT,
            package_name.replace(".", os.sep),
        )
    ]
    sys.modules.setdefault(package_name, package)

from tools.qwen35_mtp_model_runner_ownership_gate import (
    REQUIRED_BATCH_SIZES,
    REQUIRED_Q_VALUES,
    RealLoadedModelRunnerOwnershipBackend,
    SCHEMA_VERSION,
    _LoadedScenarioOwner,
    _build_fused_ownership_probe,
    _count_tensors,
    _observe_loaded_fused_call,
    _validate_public_result,
    run_gate,
    validate_ownership_gate_report,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalInput,
)
from tinyvllm.speculative.adapter import DraftProposal
from tinyvllm.speculative.batch_runtime import (
    FirstTargetProposalResult,
)


def _valid_report():
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": "/tmp/checkpoint",
        "checkpoint_manifest_sha256": "a" * 64,
        "device_name": "NVIDIA A100 80GB PCIe",
        "torch_version": "2.4.1+cu121",
        "cuda_version": "12.1",
        "q_values": list(REQUIRED_Q_VALUES),
        "batch_sizes": list(REQUIRED_BATCH_SIZES),
        "loader_passed": True,
        "fused_model_runner_path_exercised": True,
        "target_forward_real": True,
        "target_logits_cuda": True,
        "target_hidden_cuda": True,
        "target_hidden_consumed_by_real_executor": True,
        "target_logits_not_passed_to_mtp_executor": True,
        "public_result_tensor_count": 0,
        "public_result_pickle_roundtrip": True,
        "public_result_tensor_free": True,
        "executor_identity_preserved": True,
        "sequence_order_preserved": True,
        "graph_eager_first_target_tokens_equal": True,
        "graph_eager_proposal_tokens_equal": True,
        "graph_capture_count": 6,
        "graph_replay_count": 12,
        "cleanup_passed": True,
        "backend_failures": [],
        "status": "PASS",
        "promotion_classification": "NOT_PROMOTABLE",
        "coverage": {
            "tp1": True,
            "tp4": False,
            "kv_offload": False,
            "long_context": False,
            "second_model": False,
            "performance": False,
        },
        "limitations": [
            "TP1 only",
            "KV offload disabled",
            "no performance claim",
        ],
    }


def _public_row(
    sequence_id=3,
    *,
    target_token=11,
    proposal_tokens=(7, 8),
    metadata=None,
):
    return FirstTargetProposalResult(
        sequence_id=sequence_id,
        target_token=target_token,
        proposal=DraftProposal(
            sequence_id=sequence_id,
            token_ids=proposal_tokens,
            source_type="native_model_runner",
            metadata=metadata,
        ),
        first_target_metadata={"execution_mode": "decode"},
        proposal_metadata=metadata,
    )


class _FakeTensor:
    __module__ = "torch"

    def __init__(
        self,
        shape,
        *,
        device="cuda:0",
        dtype="torch.bfloat16",
    ):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self.is_cuda = str(device).startswith("cuda")


class _FakeKVSlice:

    def __init__(self, cache, block_id):
        self.cache = cache
        self.block_id = block_id

    def zero_(self):
        self.cache.nonzero_blocks.discard(self.block_id)
        self.cache.zero_events.append(self.block_id)
        return self


class _FakeKVCache:

    def __init__(self, block_count=16):
        self.shape = (2, 3, block_count, 256, 2, 8)
        self.nonzero_blocks = set()
        self.zero_events = []

    def __getitem__(self, key):
        block_id = key[2]
        return _FakeKVSlice(self, block_id)


class _FakeLease:

    def __init__(self, slot_id, generation, request_id):
        self.slot_id = slot_id
        self.generation = generation
        self.request_id = request_id


class _FakeAllocator:

    def __init__(self, capacity):
        self.capacity = capacity
        self.leases = {}

    def allocate(self, request_id):
        if len(self.leases) >= self.capacity:
            raise RuntimeError("slots exhausted")
        lease = _FakeLease(
            slot_id=len(self.leases),
            generation=1,
            request_id=request_id,
        )
        self.leases[request_id] = lease
        return lease

    def release(self, lease):
        if self.leases.get(lease.request_id) is not lease:
            raise RuntimeError("lease mismatch")
        del self.leases[lease.request_id]


class _FakePool:

    def __init__(self, capacity):
        self.capacity = capacity
        self.active = {}

    def validate(self, lease):
        if self.active.get(lease.slot_id) != (
            lease.request_id,
            lease.generation,
        ):
            raise RuntimeError("not active")
        return lease.slot_id


class _FakeBridge:

    def __init__(self, capacity):
        self.pool = _FakePool(capacity)
        self.release_events = []

    def release(self, leases):
        for lease in leases:
            self.pool.validate(lease)
            del self.pool.active[lease.slot_id]
            self.release_events.append(lease.request_id)


def _fake_sequence_factory(token_ids, sampling_params):
    sequence = SimpleNamespace(
        seq_id=-1,
        sequence_epoch=0,
        token_ids=list(token_ids),
        last_token=token_ids[-1],
        num_tokens=len(token_ids),
        num_prompt_tokens=len(token_ids),
        num_completion_tokens=0,
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        block_table=[],
        hybrid_state_slot_id=-1,
        hybrid_state_generation=0,
    )

    def append_token(token_id):
        sequence.token_ids.append(token_id)
        sequence.last_token = token_id
        sequence.num_tokens += 1
        sequence.num_completion_tokens += 1

    sequence.append_token = append_token
    return sequence


def _fake_sampling_params_factory(**kwargs):
    return SimpleNamespace(**kwargs)


def _proposal_input(
    sequence_id,
    *,
    target_hidden,
    target_logits=None,
):
    return ModelRunnerProposalInput(
        sequence_id=sequence_id,
        token_ids=(1, 2, sequence_id),
        remaining_output_tokens=4,
        max_proposal_tokens=4,
        first_target_token=10 + sequence_id,
        target_hidden=target_hidden,
        target_logits=target_logits,
    )


def test_valid_report_passes():
    validate_ownership_gate_report(
        _valid_report(),
        required_q_values=REQUIRED_Q_VALUES,
        required_batch_sizes=REQUIRED_BATCH_SIZES,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("loader_passed", False),
        ("fused_model_runner_path_exercised", False),
        ("target_forward_real", False),
        ("target_logits_cuda", False),
        ("target_hidden_cuda", False),
        ("target_hidden_consumed_by_real_executor", False),
        ("target_logits_not_passed_to_mtp_executor", False),
        ("public_result_tensor_count", 1),
        ("public_result_pickle_roundtrip", False),
        ("public_result_tensor_free", False),
        ("executor_identity_preserved", False),
        ("sequence_order_preserved", False),
        ("graph_eager_first_target_tokens_equal", False),
        ("graph_eager_proposal_tokens_equal", False),
        ("cleanup_passed", False),
        ("backend_failures", ["ownership"]),
        ("status", "FAIL"),
        ("promotion_classification", "PROMOTABLE"),
    ),
)
def test_critical_field_corruption_fails(field, value):
    report = _valid_report()
    report[field] = value

    with pytest.raises(ValueError):
        validate_ownership_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("q_values", [1, 2, 4]),
        ("batch_sizes", [1]),
        ("graph_capture_count", -1),
        ("graph_replay_count", -1),
        ("limitations", []),
    ),
)
def test_domain_and_count_corruption_fails(field, value):
    report = _valid_report()
    report[field] = value

    with pytest.raises(ValueError):
        validate_ownership_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


@pytest.mark.parametrize(
    "field",
    (
        "tp4",
        "kv_offload",
        "long_context",
        "second_model",
        "performance",
    ),
)
def test_unsupported_coverage_must_remain_false(field):
    report = _valid_report()
    report["coverage"][field] = True

    with pytest.raises(ValueError, match=field):
        validate_ownership_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


def test_missing_required_field_fails():
    report = deepcopy(_valid_report())
    del report["target_hidden_cuda"]

    with pytest.raises(ValueError, match="missing"):
        validate_ownership_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


def test_public_result_rejects_nested_tensor():
    Tensor = type("Tensor", (), {"__module__": "torch"})
    rows = (
        _public_row(
            1,
            metadata={"payload": Tensor()},
        ),
    )

    with pytest.raises(ValueError, match="tensor"):
        _validate_public_result(rows, (1,))


def test_public_result_pickle_roundtrip_is_canonical():
    rows = (_public_row(),)

    observation = _validate_public_result(rows, (3,))

    assert observation == {
        "tensor_count": 0,
        "tensor_free": True,
        "pickle_roundtrip": True,
        "sequence_order_preserved": True,
        "canonical_rows": (
            {
                "sequence_id": 3,
                "target_token": 11,
                "proposal_token_ids": (7, 8),
                "source_type": "native_model_runner",
            },
        ),
    }


def test_public_result_rejects_hidden_or_logits_attributes():
    for name in ("target_hidden", "target_logits"):
        row = SimpleNamespace(
            sequence_id=1,
            target_token=2,
            proposal=_public_row(1).proposal,
            **{name: object()},
        )
        with pytest.raises(ValueError, match=name):
            _validate_public_result((row,), (1,))


def test_public_result_rejects_sequence_order_drift():
    rows = (_public_row(2), _public_row(1))

    with pytest.raises(ValueError, match="order"):
        _validate_public_result(rows, (1, 2))


@pytest.mark.parametrize(
    "value",
    (
        lambda: None,
        SimpleNamespace(module_name="fixture"),
    ),
)
def test_public_result_rejects_callable_or_module_like_metadata(value):
    rows = (_public_row(1, metadata={"invalid": value}),)

    with pytest.raises(ValueError, match="public result"):
        _validate_public_result(rows, (1,))


def test_count_tensors_finds_nested_torch_class():
    Tensor = type("Tensor", (), {"__module__": "torch"})

    assert _count_tensors({"rows": ({"tensor": Tensor()},)}) == 1


def test_observer_delegates_and_restores_original_identities():
    logits = _FakeTensor((2, 128))
    hidden = _FakeTensor((2, 64))
    run_calls = []
    proposal_calls = []

    def run_model(*args, **kwargs):
        run_calls.append((args, kwargs))
        return logits, hidden

    def propose_batch(inputs):
        proposal_calls.append(inputs)
        return (
            _public_row(1).proposal,
            _public_row(2).proposal,
        )

    executor = SimpleNamespace(propose_batch=propose_batch)
    runner = SimpleNamespace(
        run_model=run_model,
        model=object(),
    )
    original_run_model = runner.run_model
    original_propose_batch = executor.propose_batch

    with _observe_loaded_fused_call(
        runner,
        executor,
    ) as observation:
        outputs = runner.run_model(
            "input_ids",
            "positions",
            False,
            return_hidden=True,
            execution_mode="decode",
        )
        proposals = executor.propose_batch((
            _proposal_input(1, target_hidden=hidden),
            _proposal_input(2, target_hidden=hidden),
        ))
        assert outputs == (logits, hidden)
        assert len(proposals) == 2

    assert runner.run_model is original_run_model
    assert executor.propose_batch is original_propose_batch
    assert observation.restored is True
    assert observation.executor_identity_preserved is True
    assert len(observation.forward_rows) == 1
    assert len(observation.executor_rows) == 2
    assert observation.forward_rows[0]["logits_device"] == "cuda:0"
    assert observation.forward_rows[0]["hidden_device"] == "cuda:0"
    assert observation.executor_rows[0]["target_logits_is_none"] is True
    assert run_calls
    assert proposal_calls
    assert _count_tensors(observation.__dict__) == 0


@pytest.mark.parametrize(
    ("logits_device", "hidden_device", "match"),
    (
        ("cpu", "cuda:0", "logits.*CUDA"),
        ("cuda:0", "cpu", "hidden.*CUDA"),
        ("cuda:0", "cuda:1", "device"),
    ),
)
def test_observer_rejects_non_cuda_or_mismatched_forward(
    logits_device,
    hidden_device,
    match,
):
    runner = SimpleNamespace(
        run_model=lambda *args, **kwargs: (
            _FakeTensor((1, 128), device=logits_device),
            _FakeTensor((1, 64), device=hidden_device),
        ),
        model=object(),
    )
    executor = SimpleNamespace(
        propose_batch=lambda inputs: (),
    )
    original_run_model = runner.run_model

    with pytest.raises(ValueError, match=match):
        with _observe_loaded_fused_call(runner, executor):
            runner.run_model(
                "input_ids",
                "positions",
                False,
                return_hidden=True,
                execution_mode="decode",
            )

    assert runner.run_model is original_run_model


def test_observer_rejects_target_logits_entering_mtp_executor():
    hidden = _FakeTensor((1, 64))
    logits = _FakeTensor((1, 128))
    runner = SimpleNamespace(
        run_model=lambda *args, **kwargs: (logits, hidden),
        model=object(),
    )
    executor = SimpleNamespace(
        propose_batch=lambda inputs: (),
    )
    original_propose_batch = executor.propose_batch

    with pytest.raises(ValueError, match="target logits"):
        with _observe_loaded_fused_call(runner, executor):
            runner.run_model(
                "input_ids",
                "positions",
                False,
                return_hidden=True,
                execution_mode="decode",
            )
            executor.propose_batch((
                _proposal_input(
                    1,
                    target_hidden=hidden,
                    target_logits=logits,
                ),
            ))

    assert executor.propose_batch is original_propose_batch


def test_observer_requires_real_decode_hidden_forward():
    logits = _FakeTensor((1, 128))
    hidden = _FakeTensor((1, 64))
    runner = SimpleNamespace(
        run_model=lambda *args, **kwargs: (logits, hidden),
        model=object(),
    )
    executor = SimpleNamespace(
        propose_batch=lambda inputs: (),
    )

    with pytest.raises(ValueError, match="decode.*return_hidden"):
        with _observe_loaded_fused_call(runner, executor):
            runner.run_model(
                "input_ids",
                "positions",
                False,
                return_hidden=False,
                execution_mode="prefill",
            )


def _scenario_owner(
    *,
    block_count=16,
    state_capacity=8,
    fail_release=False,
):
    kv_cache = _FakeKVCache(block_count)
    bridge = _FakeBridge(state_capacity)
    runner = SimpleNamespace(
        kv_cache=kv_cache,
        qwen35_hybrid_model_owner=SimpleNamespace(
            pool=SimpleNamespace(capacity=state_capacity),
        ),
        hybrid_state_runtime_bridge=bridge,
        bootstrap_events=[],
    )
    release_events = []

    def release_sequence(sequence_id, *, sequence_epoch):
        release_events.append((sequence_id, sequence_epoch))
        if fail_release:
            raise RuntimeError("release failed")

    executor = SimpleNamespace(
        release_sequence=release_sequence,
    )

    def bootstrap_callback(sequences):
        runner.bootstrap_events.append(tuple(
            int(sequence.seq_id)
            for sequence in sequences
        ))
        for sequence in sequences:
            bridge.pool.active[
                sequence.hybrid_state_slot_id
            ] = (
                int(sequence.seq_id),
                int(sequence.hybrid_state_generation),
            )
        return tuple(
            900 + index
            for index, _sequence in enumerate(sequences)
        )

    owner = _LoadedScenarioOwner(
        runner,
        executor,
        block_start=2,
        sequence_factory=_fake_sequence_factory,
        sampling_params_factory=_fake_sampling_params_factory,
        allocator_factory=_FakeAllocator,
        reset_context=lambda: None,
        bootstrap_callback=bootstrap_callback,
    )
    return owner, runner, executor, release_events


def test_scenario_owner_allocates_distinct_blocks_and_leases():
    owner, runner, _, _ = _scenario_owner()

    seqs = owner.build(4, 4, sequence_id_base=1000)

    assert tuple(seq.seq_id for seq in seqs) == (
        1000,
        1001,
        1002,
        1003,
    )
    assert tuple(seq.block_table for seq in seqs) == (
        [2],
        [3],
        [4],
        [5],
    )
    assert len({
        seq.hybrid_state_slot_id
        for seq in seqs
    }) == 4
    assert all(
        seq.max_tokens - seq.num_completion_tokens == 4
        for seq in seqs
    )
    assert tuple(seq.token_ids for seq in seqs) == (
        [128, 900],
        [129, 901],
        [130, 902],
        [131, 903],
    )
    assert runner.bootstrap_events == [(
        1000,
        1001,
        1002,
        1003,
    )]
    assert runner.kv_cache.zero_events == [2, 3, 4, 5]


def test_scenario_owner_cleanup_releases_and_zeros_everything():
    owner, runner, _, release_events = _scenario_owner()
    seqs = owner.build(2, 4, sequence_id_base=2000)
    for seq in seqs:
        lease = owner.leases_by_sequence_id[seq.seq_id]
        runner.hybrid_state_runtime_bridge.pool.active[
            lease.slot_id
        ] = (lease.request_id, lease.generation)
        runner.kv_cache.nonzero_blocks.add(seq.block_table[-1])

    result = owner.cleanup()

    assert result == {
        "cleanup_passed": True,
        "active_leases": 0,
        "nonzero_target_kv_rows": 0,
        "errors": [],
    }
    assert release_events == [
        (2003, 0),
        (2002, 0),
        (2001, 0),
        (2000, 0),
    ]
    assert runner.hybrid_state_runtime_bridge.release_events == [
        2003,
        2002,
        2001,
        2000,
    ]
    assert runner.kv_cache.nonzero_blocks == set()
    assert owner.cleanup() == result


def test_scenario_owner_rejects_capacity_before_mutation():
    owner, runner, _, _ = _scenario_owner(
        block_count=4,
        state_capacity=2,
    )

    with pytest.raises(ValueError, match="capacity"):
        owner.build(4, 4, sequence_id_base=3000)

    assert runner.kv_cache.zero_events == []


def test_scenario_owner_reports_cleanup_failure():
    owner, _, _, _ = _scenario_owner(fail_release=True)
    owner.build(1, 1, sequence_id_base=4000)

    result = owner.cleanup()

    assert result["cleanup_passed"] is False
    assert result["errors"] == [
        "RuntimeError: release failed",
    ]


class _FakeGraphRunner:

    def __init__(self):
        self.min_observations = 2
        self.observations = 0
        self.ready = False
        self.captures = 0
        self.replays = 0

    def summary(self):
        return {
            "captures": self.captures,
            "replays": self.replays,
        }


class _FakeProbeOwner:

    instances = []

    def __init__(
        self,
        runner,
        executor,
        *,
        block_start,
    ):
        self.runner = runner
        self.executor = executor
        self.block_start = block_start
        self.sequences = ()
        self.cleaned = False
        type(self).instances.append(self)

    def build(self, q, batch_size, *, sequence_id_base):
        self.sequences = tuple(
            SimpleNamespace(
                seq_id=sequence_id_base + index,
                sequence_epoch=0,
                token_ids=[128 + index],
                max_tokens=q,
                num_completion_tokens=0,
            )
            for index in range(batch_size)
        )
        return self.sequences

    def cleanup(self):
        self.cleaned = True
        return {
            "cleanup_passed": True,
            "active_leases": 0,
            "nonzero_target_kv_rows": 0,
            "errors": [],
        }


class _FailingCleanupProbeOwner(_FakeProbeOwner):

    def cleanup(self):
        self.cleaned = True
        return {
            "cleanup_passed": False,
            "active_leases": 1,
            "nonzero_target_kv_rows": 1,
            "errors": ["injected cleanup failure"],
        }


def _fake_fused_probe_runtime(*, fail_call=False):
    graph_runner = _FakeGraphRunner()
    hidden = _FakeTensor((4, 64))
    logits = _FakeTensor((4, 128))

    def propose_batch(inputs):
        return tuple(
            DraftProposal(
                sequence_id=input_row.sequence_id,
                token_ids=tuple(
                    20 + offset
                    for offset in range(
                        input_row.max_proposal_tokens
                    )
                ),
                source_type="native_model_runner",
                proposal_transaction_id=(
                    f"tx-{input_row.sequence_id}"
                ),
            )
            for input_row in inputs
        )

    finalize_events = []

    def prepare_finalize_batch(rows):
        finalize_events.append((
            "prepare",
            tuple(
                (
                    row.sequence_id,
                    row.proposal_transaction_id,
                    row.accepted_proposal_tokens,
                )
                for row in rows
            ),
        ))
        return f"ticket-{len(finalize_events)}"

    def rollback_finalize_batch(ticket_id):
        finalize_events.append(("rollback", ticket_id))

    executor = SimpleNamespace(
        graph_runner=graph_runner,
        propose_batch=propose_batch,
        prepare_finalize_batch=prepare_finalize_batch,
        rollback_finalize_batch=rollback_finalize_batch,
        finalize_events=finalize_events,
    )
    runner = SimpleNamespace(
        model=object(),
        calls=[],
    )
    runner.run_model = lambda *args, **kwargs: (
        logits,
        hidden,
    )

    def call(method_name, seqs, descriptor, identities):
        runner.calls.append((
            method_name,
            tuple(seq.seq_id for seq in seqs),
        ))
        if fail_call:
            raise RuntimeError("fused call failed")
        runner.run_model(
            "input_ids",
            "positions",
            False,
            return_hidden=True,
            execution_mode="decode",
        )
        inputs = tuple(
            _proposal_input(
                seq.seq_id,
                target_hidden=hidden,
            )
            for seq in seqs
        )
        proposals = executor.propose_batch(inputs)
        q = seqs[0].max_tokens
        if executor.graph_runner is not None and q > 1:
            if executor.graph_runner.ready:
                executor.graph_runner.replays += 1
            else:
                executor.graph_runner.observations += 1
                if (
                    executor.graph_runner.observations
                    >= executor.graph_runner.min_observations
                ):
                    executor.graph_runner.ready = True
                    executor.graph_runner.captures += 1
        return tuple(
            FirstTargetProposalResult(
                sequence_id=seq.seq_id,
                target_token=100 + index,
                proposal=proposals[index],
                first_target_metadata={
                    "execution_mode": "decode",
                },
                proposal_metadata=None,
            )
            for index, seq in enumerate(seqs)
        )

    runner.call = call
    descriptor = SimpleNamespace(executor_id="fixture")
    return runner, descriptor, executor


def test_probe_uses_model_runner_call_and_fresh_graph_eager_state():
    _FakeProbeOwner.instances = []
    runner, descriptor, executor = _fake_fused_probe_runtime()
    probe = _build_fused_ownership_probe(
        runner,
        descriptor,
        executor,
        scenario_owner_factory=_FakeProbeOwner,
    )

    result = probe(4, 4)

    assert tuple(call[0] for call in runner.calls) == (
        "run_spec_first_target_and_proposal_batch",
        "run_spec_first_target_and_proposal_batch",
        "run_spec_first_target_and_proposal_batch",
        "run_spec_first_target_and_proposal_batch",
    )
    graph_ids = runner.calls[2][1]
    eager_ids = runner.calls[3][1]
    assert set(graph_ids).isdisjoint(eager_ids)
    assert result["first_target_tokens_equal"] is True
    assert result["proposal_tokens_equal"] is True
    assert result["public_result_tensor_count"] == 0
    assert result["capture_count"] == 1
    assert result["replay_count"] == 1
    assert result["target_logits_cuda"] is True
    assert result["target_hidden_cuda"] is True
    assert result[
        "target_hidden_consumed_by_real_executor"
    ] is True
    assert result[
        "target_logits_not_passed_to_mtp_executor"
    ] is True
    assert result["cleanup_passed"] is True
    assert all(
        owner.cleaned
        for owner in _FakeProbeOwner.instances
    )
    assert len(executor.finalize_events) == 8
    assert all(
        event[0] == expected
        for event, expected in zip(
            executor.finalize_events,
            ("prepare", "rollback") * 4,
        )
    )
    assert executor.graph_runner is not None


def test_probe_q1_does_not_capture():
    _FakeProbeOwner.instances = []
    runner, descriptor, executor = _fake_fused_probe_runtime()
    probe = _build_fused_ownership_probe(
        runner,
        descriptor,
        executor,
        scenario_owner_factory=_FakeProbeOwner,
    )

    result = probe(1, 1)

    assert result["capture_count"] == 0
    assert result["replay_count"] == 0
    assert result["first_target_tokens_equal"] is True
    assert result["proposal_tokens_equal"] is True
    assert len(executor.finalize_events) == 4


def test_probe_cleans_up_and_restores_graph_runner_on_failure():
    _FakeProbeOwner.instances = []
    runner, descriptor, executor = _fake_fused_probe_runtime(
        fail_call=True,
    )
    graph_runner = executor.graph_runner
    probe = _build_fused_ownership_probe(
        runner,
        descriptor,
        executor,
        scenario_owner_factory=_FakeProbeOwner,
    )

    with pytest.raises(RuntimeError, match="fused call failed"):
        probe(2, 1)

    assert executor.graph_runner is graph_runner
    assert all(
        owner.cleaned
        for owner in _FakeProbeOwner.instances
    )


def test_probe_reports_owner_cleanup_failure():
    _FailingCleanupProbeOwner.instances = []
    runner, descriptor, executor = _fake_fused_probe_runtime()
    probe = _build_fused_ownership_probe(
        runner,
        descriptor,
        executor,
        scenario_owner_factory=_FailingCleanupProbeOwner,
    )

    result = probe(2, 1)

    assert result["cleanup_passed"] is False
    assert result["observer_restored"] is True


@pytest.mark.parametrize(
    ("q", "batch_size"),
    ((0, 1), (5, 1), (1, 2), (1, 8)),
)
def test_probe_rejects_out_of_domain(q, batch_size):
    runner, descriptor, executor = _fake_fused_probe_runtime()
    probe = _build_fused_ownership_probe(
        runner,
        descriptor,
        executor,
        scenario_owner_factory=_FakeProbeOwner,
    )

    with pytest.raises(ValueError, match="domain"):
        probe(q, batch_size)


class _RecordingOwnershipBackend:

    def __init__(
        self,
        *,
        fail_case=None,
        raise_case=None,
    ):
        self.fail_case = fail_case
        self.raise_case = raise_case
        self.calls = []
        self._failures = []

    def load(self, checkpoint_path):
        self.calls.append(("load", checkpoint_path))
        return {
            "checkpoint_manifest_sha256": "b" * 64,
            "device_name": "NVIDIA A100 80GB PCIe",
            "torch_version": "2.4.1+cu121",
            "cuda_version": "12.1",
            "loader_passed": True,
            "target_forward_real": True,
        }

    def compare_fused_graph_eager(self, q, batch_size):
        self.calls.append(("compare", q, batch_size))
        if self.raise_case == (q, batch_size):
            raise RuntimeError("injected ownership failure")
        result = {
            "q": q,
            "batch_size": batch_size,
            "capture_count": int(q > 1),
            "replay_count": int(q > 1),
            "first_target_tokens_equal": True,
            "proposal_tokens_equal": True,
            "public_result_tensor_count": 0,
            "public_result_tensor_free": True,
            "public_result_pickle_roundtrip": True,
            "sequence_order_preserved": True,
            "target_logits_cuda": True,
            "target_hidden_cuda": True,
            "target_hidden_consumed_by_real_executor": True,
            "target_logits_not_passed_to_mtp_executor": True,
            "executor_identity_preserved": True,
            "model_identity_preserved": True,
            "observer_restored": True,
            "cleanup_passed": True,
        }
        if self.fail_case == (q, batch_size):
            result["public_result_tensor_free"] = False
        return result

    def failures(self):
        return list(self._failures)


def test_run_gate_aggregates_all_required_cases():
    backend = _RecordingOwnershipBackend()

    report = run_gate("/readonly/model", backend=backend)

    assert report["status"] == "PASS"
    assert report["promotion_classification"] == "NOT_PROMOTABLE"
    assert report["checkpoint_manifest_sha256"] == "b" * 64
    assert report["backend_failures"] == []
    assert report["graph_capture_count"] == 6
    assert report["graph_replay_count"] == 6
    assert report["public_result_tensor_count"] == 0
    assert report["fused_model_runner_path_exercised"] is True
    assert report["target_forward_real"] is True
    assert len(report["cases"]) == 8
    assert backend.calls[1:] == [
        ("compare", q, batch_size)
        for batch_size in REQUIRED_BATCH_SIZES
        for q in REQUIRED_Q_VALUES
    ]
    validate_ownership_gate_report(
        report,
        required_q_values=REQUIRED_Q_VALUES,
        required_batch_sizes=REQUIRED_BATCH_SIZES,
    )


@pytest.mark.parametrize(
    "backend",
    (
        _RecordingOwnershipBackend(fail_case=(3, 4)),
        _RecordingOwnershipBackend(raise_case=(3, 4)),
    ),
)
def test_run_gate_fails_closed_for_case_failure(backend):
    report = run_gate("/readonly/model", backend=backend)

    assert report["status"] == "FAIL"
    assert report["promotion_classification"] == "NOT_PROMOTABLE"
    with pytest.raises(ValueError):
        validate_ownership_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )


def test_gate_report_is_json_serializable(tmp_path):
    report = run_gate(
        "/readonly/model",
        backend=_RecordingOwnershipBackend(),
    )
    output = tmp_path / "ownership.json"

    output.write_text(
        json.dumps(report, indent=2, sort_keys=True)
        + "\n"
    )
    loaded = json.loads(output.read_text())

    assert loaded["status"] == "PASS"
    assert Path(output).is_file()


def test_cli_main_guard_runs_after_report_verifier_definition():
    text = OWNERSHIP_GATE.read_text(encoding="utf-8")

    assert text.index(
        "def validate_ownership_gate_report("
    ) < text.index('if __name__ == "__main__":')


def _loaded_backend_runtime():
    descriptor = object()
    graph_runner = SimpleNamespace(
        max_reserved_bytes=512 * 1024 * 1024,
        summary=lambda: {
            "captures": 0,
            "replays": 0,
        },
    )
    executor = SimpleNamespace(graph_runner=graph_runner)
    runner = SimpleNamespace(
        model=object(),
        run_model=lambda *args, **kwargs: (),
        qwen35_hybrid_model_owner=SimpleNamespace(pool=object()),
        qwen35_mtp_executor_descriptor=descriptor,
        qwen35_mtp_executor=executor,
        qwen35_mtp_physical_store=object(),
        call=lambda *args: (),
    )
    return {
        "runner": runner,
        "loader_passed": True,
        "blockers": {},
    }


def test_real_backend_loads_one_runner_and_builds_fused_probe():
    runtime = _loaded_backend_runtime()
    runtime_loads = []
    probe_builds = []
    probe_calls = []

    def runtime_loader(checkpoint_path):
        runtime_loads.append(checkpoint_path)
        return runtime

    def probe_builder(runner, descriptor, executor):
        probe_builds.append((runner, descriptor, executor))

        def probe(q, batch_size):
            probe_calls.append((q, batch_size))
            return {"q": q, "batch_size": batch_size}

        return probe

    backend = RealLoadedModelRunnerOwnershipBackend(
        runtime_loader=runtime_loader,
        runtime_metadata_loader=lambda: {
            "device_name": "NVIDIA A100 80GB PCIe",
            "torch_version": "2.4.1+cu121",
            "cuda_version": "12.1",
        },
        manifest_loader=lambda path: "c" * 64,
        probe_builder=probe_builder,
    )

    metadata = backend.load("/readonly/model")
    result = backend.compare_fused_graph_eager(2, 4)

    assert runtime_loads == ["/readonly/model"]
    assert probe_builds == [(
        runtime["runner"],
        runtime["runner"].qwen35_mtp_executor_descriptor,
        runtime["runner"].qwen35_mtp_executor,
    )]
    assert probe_calls == [(2, 4)]
    assert result == {"q": 2, "batch_size": 4}
    assert metadata == {
        "checkpoint_manifest_sha256": "c" * 64,
        "device_name": "NVIDIA A100 80GB PCIe",
        "torch_version": "2.4.1+cu121",
        "cuda_version": "12.1",
        "loader_passed": True,
        "target_forward_real": True,
    }
    assert backend.failures() == []


def test_real_backend_raises_graph_budget_for_full_gate_domain():
    runtime = _loaded_backend_runtime()
    backend = RealLoadedModelRunnerOwnershipBackend(
        runtime_loader=lambda checkpoint_path: runtime,
        runtime_metadata_loader=lambda: {
            "device_name": "GPU",
            "torch_version": "torch",
            "cuda_version": "cuda",
        },
        manifest_loader=lambda path: "d" * 64,
        probe_builder=lambda *args: lambda q, batch_size: {
            "q": q,
            "batch_size": batch_size,
        },
    )

    metadata = backend.load("/readonly/model")

    assert metadata["loader_passed"] is True
    assert (
        runtime["runner"]
        .qwen35_mtp_executor
        .graph_runner
        .max_reserved_bytes
        == 3 * 1024 * 1024 * 1024
    )


@pytest.mark.parametrize(
    "missing_path",
    (
        "model",
        "qwen35_hybrid_model_owner",
        "qwen35_mtp_executor_descriptor",
        "qwen35_mtp_executor",
        "qwen35_mtp_physical_store",
        "qwen35_mtp_executor.graph_runner",
    ),
)
def test_real_backend_rejects_missing_loaded_identity(missing_path):
    runtime = _loaded_backend_runtime()
    owner, _, attribute = missing_path.rpartition(".")
    target = runtime["runner"]
    if owner:
        target = getattr(target, owner)
    setattr(target, attribute, None)
    backend = RealLoadedModelRunnerOwnershipBackend(
        runtime_loader=lambda checkpoint_path: runtime,
        runtime_metadata_loader=lambda: {
            "device_name": "GPU",
            "torch_version": "torch",
            "cuda_version": "cuda",
        },
        manifest_loader=lambda path: "d" * 64,
        probe_builder=lambda *args: pytest.fail(
            "probe must not be built for an incomplete runtime"
        ),
    )

    metadata = backend.load("/readonly/model")

    assert metadata["loader_passed"] is False
    assert metadata["target_forward_real"] is False
    assert backend.failures()
    with pytest.raises(RuntimeError, match="unavailable"):
        backend.compare_fused_graph_eager(1, 1)


def test_real_backend_rejects_checkpoint_manifest_drift():
    manifests = iter(("e" * 64, "f" * 64))
    backend = RealLoadedModelRunnerOwnershipBackend(
        runtime_loader=lambda checkpoint_path: _loaded_backend_runtime(),
        runtime_metadata_loader=lambda: {
            "device_name": "GPU",
            "torch_version": "torch",
            "cuda_version": "cuda",
        },
        manifest_loader=lambda path: next(manifests),
        probe_builder=lambda *args: lambda q, batch_size: {
            "q": q,
            "batch_size": batch_size,
        },
    )

    metadata = backend.load("/readonly/model")

    assert metadata["checkpoint_manifest_sha256"] == "e" * 64
    assert metadata["loader_passed"] is False
    assert metadata["target_forward_real"] is False
    assert any(
        failure.startswith("load:")
        and "manifest" in failure
        for failure in backend.failures()
    )


def test_real_backend_default_loader_reuses_existing_runtime_loader(
    monkeypatch,
):
    calls = []
    fake_module = SimpleNamespace(
        RealQwen35MTPGateBackend=SimpleNamespace(
            _load_real_runtime=lambda checkpoint_path: (
                calls.append(checkpoint_path)
                or _loaded_backend_runtime()
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.qwen35_mtp_real_checkpoint_gate",
        fake_module,
    )

    runtime = (
        RealLoadedModelRunnerOwnershipBackend._load_real_runtime(
            "/readonly/model"
        )
    )

    assert runtime["runner"].model is not None
    assert calls == ["/readonly/model"]


def test_remote_wrapper_is_serial_and_verifies_downloaded_artifact():
    text = REMOTE_WRAPPER.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert (
        'KRB5CCNAME="${KRB5CCNAME:-'
        'FILE:/Users/bytedance/krb5cc_sitian}"'
    ) in text
    assert (
        'REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"'
        in text
    )
    assert "ControlMaster=no" in text
    assert "ControlPath=none" in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"' in text
    assert "DIST_PORT=" in text
    assert "TINYVLLM_DIST_PORT='${DIST_PORT}'" in text
    assert "SOURCE_FILES=(" in text
    for source_file in (
        "tinyvllm/config.py",
        "tinyvllm/speculative/adapter.py",
        "tinyvllm/speculative/batch_runtime.py",
        "tinyvllm/engine/speculative_proposal_executor.py",
        "tinyvllm/engine/speculative_model_runner.py",
        "tinyvllm/engine/speculative_runtime.py",
        "tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py",
        "tinyvllm/engine/qwen35_mtp_registration.py",
        "tinyvllm/engine/model_runner.py",
        "tinyvllm/engine/proposal_kv_cache.py",
        "tinyvllm/engine/qwen35_mtp_executor.py",
        "tinyvllm/engine/qwen35_mtp_graph.py",
        "tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py",
        "tinyvllm/engine/qwen35_mtp_graph_scratch.py",
        "tinyvllm/utils/context.py",
        "tinyvllm/layers/qwen35_full_attention.py",
        "tinyvllm/models/qwen35_checkpoint.py",
        "tinyvllm/models/qwen35_components.py",
        "tinyvllm/models/qwen35_mtp_checkpoint.py",
        "tinyvllm/models/qwen35_mtp.py",
        "tools/qwen35_mtp_real_checkpoint_gate.py",
        "tools/qwen35_mtp_model_runner_ownership_gate.py",
    ):
        assert source_file in text
    assert (
        text.count(
            "'${REMOTE_PYTHON}' "
            "tools/qwen35_mtp_model_runner_ownership_gate.py"
        )
        == 1
    )
    assert (
        "qwen35_mtp_model_runner_ownership_gate.json"
        in text
    )
    assert (
        '"${REMOTE_HOST}:${REMOTE_ARTIFACT}" '
        '"${LOCAL_RUN_ROOT}/"'
    ) in text
    assert "validate_ownership_gate_report" in text
    assert "status=PASS" in text
    assert "xargs -P" not in text
    assert "\n&" not in text
    assert " &\n" not in text
