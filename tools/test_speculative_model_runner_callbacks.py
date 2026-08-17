from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm")
]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "engine")
]
speculative_package = types.ModuleType(
    "tinyvllm.speculative"
)
speculative_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "speculative")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)
sys.modules.setdefault(
    "tinyvllm.speculative",
    speculative_package,
)

import tinyvllm.engine.speculative_model_runner as speculative_model_runner_module
from tinyvllm.engine.speculative_model_runner import (
    FixedQTailBatch,
    build_model_runner_side_state_callbacks,
    build_model_runner_proposal_provider,
    build_fixed_q_tail_batches,
    run_host_first_targets_and_proposals,
    run_model_runner_first_targets,
    run_model_runner_first_targets_and_proposals,
    run_model_runner_tail_batch,
)
from tinyvllm.engine.speculative_side_state import (
    SpeculativeSideStateSelectionRow,
)
from tinyvllm.engine.spec_verify_exact_cuda_graph_cache import (
    SpecVerifyGraphReplayError,
)
from tinyvllm.engine.speculative_residency import (
    KVBlockIdentityRow,
)
from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftProposal,
)
from tinyvllm.speculative.batch_runtime import (
    FirstTargetProposalResult,
    FirstTargetResult,
    PreparedProposalFinalizeRow,
    TailBatchItem,
    TailBatchResult,
)
from tinyvllm.speculative.verifier import (
    SpecVerifyBatchResultRow,
    SpecVerifyPlan,
)


def _tail_item(sequence_id, query_len):
    start = sequence_id * 10
    return TailBatchItem(
        sequence_id=sequence_id,
        plan=SpecVerifyPlan(
            input_tokens=tuple(
                start + offset
                for offset in range(query_len)
            ),
            positions=tuple(
                start + 1 + offset
                for offset in range(query_len)
            ),
            logical_slots=tuple(
                start + offset
                for offset in range(query_len)
            ),
            context_len=start + query_len,
            visible_block_count=1,
        ),
        proxy_block_table=(sequence_id,),
    )


def _variable_q_items():
    return (
        _tail_item(8, 2),
        _tail_item(4, 1),
        _tail_item(2, 2),
        _tail_item(9, 3),
    )


def test_fixed_q_groups_are_stable_by_first_query_length():
    groups = build_fixed_q_tail_batches(
        _variable_q_items()
    )

    assert groups == (
        FixedQTailBatch(
            query_len=2,
            items=(
                _tail_item(8, 2),
                _tail_item(2, 2),
            ),
        ),
        FixedQTailBatch(
            query_len=1,
            items=(_tail_item(4, 1),),
        ),
        FixedQTailBatch(
            query_len=3,
            items=(_tail_item(9, 3),),
        ),
    )


def test_fixed_q_groups_preserve_transaction_authorization_by_identity():
    authorizations = {
        8: object(),
        4: object(),
        2: object(),
    }
    items = tuple(
        TailBatchItem(
            sequence_id=sequence_id,
            plan=_tail_item(sequence_id, query_len).plan,
            proxy_block_table=(sequence_id,),
            transaction_authorization=authorizations[sequence_id],
        )
        for sequence_id, query_len in (
            (8, 2),
            (4, 1),
            (2, 2),
        )
    )

    groups = build_fixed_q_tail_batches(items)

    assert groups[0].query_len == 2
    assert tuple(
        item.sequence_id for item in groups[0].items
    ) == (8, 2)
    assert groups[0].items[0].transaction_authorization is authorizations[8]
    assert groups[0].items[1].transaction_authorization is authorizations[2]
    assert groups[1].items[0].transaction_authorization is authorizations[4]


def test_tail_bridge_preserves_transaction_authorization_through_rpc():
    authorizations = {
        8: object(),
        4: object(),
        2: object(),
    }
    items = tuple(
        TailBatchItem(
            sequence_id=sequence_id,
            plan=_tail_item(sequence_id, query_len).plan,
            proxy_block_table=(sequence_id,),
            transaction_authorization=authorizations[sequence_id],
        )
        for sequence_id, query_len in (
            (8, 2),
            (4, 1),
            (2, 2),
        )
    )

    def callback(method_name, args):
        assert method_name == "run_spec_verify_batch"
        for item in args[0]:
            assert (
                item.transaction_authorization
                is authorizations[item.sequence_id]
            )
        return tuple(
            SpecVerifyBatchResultRow(
                sequence_id=item.sequence_id,
                target_tokens=(1,) * item.plan.query_len,
            )
            for item in args[0]
        )

    runner = _FakeModelRunner(callback)
    rows = run_model_runner_tail_batch(runner, items)

    assert tuple(row.sequence_id for row in rows) == (8, 4, 2)


def test_fixed_q_groups_reject_empty_duplicate_and_zero_q():
    with pytest.raises(ValueError, match="non-empty"):
        build_fixed_q_tail_batches(())
    duplicate = (
        _tail_item(8, 2),
        _tail_item(8, 2),
    )
    with pytest.raises(ValueError, match="unique"):
        build_fixed_q_tail_batches(duplicate)
    zero_q = TailBatchItem(
        sequence_id=1,
        plan=SpecVerifyPlan(
            input_tokens=(),
            positions=(),
            logical_slots=(),
            context_len=1,
            visible_block_count=1,
        ),
        proxy_block_table=(1,),
    )
    with pytest.raises(ValueError, match="query length"):
        build_fixed_q_tail_batches((zero_q,))


class _FakeModelRunner:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        return self.callback(method_name, args)


def test_side_state_callback_builder_disables_unavailable_runner():
    runner = _FakeModelRunner(
        lambda method_name, args: None
    )
    runner.speculative_side_state_available = lambda: False

    assert build_model_runner_side_state_callbacks(runner) is None
    assert runner.calls == []


def test_side_state_callback_builder_delegates_each_phase_once():
    calls = []
    active = {"phase": "idle"}

    def callback(method_name, args):
        calls.append((method_name, args))
        if method_name == "prepare_speculative_side_state_batch":
            assert active["phase"] == "idle"
            active["phase"] = "prepared"
            return {
                "operation": "prepare",
                "status": "prepared",
                "transaction_id": "tx-1",
                "sequence_ids": [8, 4],
            }
        expected = {
            "select_speculative_side_state_batch": (
                "prepared",
                "selected",
            ),
            "apply_speculative_side_state_batch": (
                "selected",
                "applied",
            ),
            "seal_speculative_side_state_batch": (
                "applied",
                "sealed",
            ),
        }
        before, after = expected[method_name]
        if active["phase"] != before:
            raise RuntimeError("duplicate transaction operation")
        active["phase"] = after
        return {
            "operation": method_name,
            "status": after,
            "transaction_id": "tx-1",
            "sequence_ids": [8, 4],
        }

    runner = _FakeModelRunner(callback)
    runner.speculative_side_state_available = lambda: True
    callbacks = build_model_runner_side_state_callbacks(runner)
    seqs = (
        SimpleNamespace(seq_id=8),
        SimpleNamespace(seq_id=4),
    )
    rows = (
        SpeculativeSideStateSelectionRow(
            sequence_id=8,
            proposal_token_count=4,
            accepted_draft_count=2,
            verify_input_count=3,
            committed_tail_input_count=2,
            committed_input_count=3,
        ),
        SpeculativeSideStateSelectionRow(
            sequence_id=4,
            proposal_token_count=2,
            accepted_draft_count=1,
            verify_input_count=1,
            committed_tail_input_count=1,
            committed_input_count=2,
        ),
    )

    handle = callbacks.prepare(seqs)
    callbacks.select(handle, rows)
    callbacks.apply(handle)
    callbacks.seal(handle)

    assert calls == [
        ("prepare_speculative_side_state_batch", (seqs,)),
        ("select_speculative_side_state_batch", (rows,)),
        ("apply_speculative_side_state_batch", ()),
        ("seal_speculative_side_state_batch", ()),
    ]
    assert isinstance(calls[1][1][0], tuple)
    with pytest.raises(
        RuntimeError,
        match="duplicate transaction operation",
    ):
        callbacks.seal(handle)


def test_side_state_callback_builder_uses_ordered_dispatcher():
    calls = []

    def dispatch(method_name, *args):
        calls.append((method_name, args))
        return {
            "operation": method_name,
            "status": "ok",
            "transaction_id": "tx-ordered",
            "sequence_ids": [8],
        }

    runner = _FakeModelRunner(
        lambda method_name, args: (
            (_ for _ in ()).throw(
                AssertionError(
                    "fire-and-forget runner call must not be used"
                )
            )
        )
    )
    runner.speculative_side_state_available = lambda: True
    callbacks = build_model_runner_side_state_callbacks(
        runner,
        dispatch=dispatch,
    )
    seqs = (SimpleNamespace(seq_id=8),)
    rows = (
        SpeculativeSideStateSelectionRow(
            sequence_id=8,
            proposal_token_count=1,
            accepted_draft_count=1,
            verify_input_count=2,
            committed_tail_input_count=1,
            committed_input_count=2,
        ),
    )

    handle = callbacks.prepare(seqs)
    callbacks.select(handle, rows)
    callbacks.apply(handle)
    callbacks.seal(handle)

    assert calls == [
        ("prepare_speculative_side_state_batch", (seqs,)),
        ("select_speculative_side_state_batch", (rows,)),
        ("apply_speculative_side_state_batch", ()),
        ("seal_speculative_side_state_batch", ()),
    ]
    assert runner.calls == []


def test_side_state_callback_builder_propagates_runner_error():
    runner = _FakeModelRunner(
        lambda method_name, args: (
            (_ for _ in ()).throw(RuntimeError("apply failed"))
            if method_name
            == "apply_speculative_side_state_batch"
            else {
                "operation": "prepare",
                "status": "prepared",
                "transaction_id": "tx-1",
                "sequence_ids": [8],
            }
        )
    )
    runner.speculative_side_state_available = lambda: True
    callbacks = build_model_runner_side_state_callbacks(runner)
    handle = callbacks.prepare((SimpleNamespace(seq_id=8),))

    with pytest.raises(RuntimeError, match="apply failed"):
        callbacks.apply(handle)


def _capabilities(
    *,
    requires_hidden=True,
    requires_logits=True,
    execution_domain="host",
    requires_lifecycle=False,
):
    return DraftCapabilities(
        source_type="fixture",
        supports_batch=True,
        requires_target_hidden=requires_hidden,
        requires_target_logits=requires_logits,
        max_proposal_tokens=4,
        execution_domain=execution_domain,
        requires_proposal_lifecycle=requires_lifecycle,
    )


def _lifecycle_descriptor():
    return SimpleNamespace(
        executor_id="fixture",
        capabilities=_capabilities(
            requires_hidden=False,
            requires_logits=False,
            execution_domain="model_runner",
            requires_lifecycle=True,
        ),
    )


def _prepared_finalize_row(**overrides):
    values = {
        "sequence_id": 1,
        "proposal_transaction_id": "tx-1",
        "accepted_proposal_tokens": 2,
    }
    values.update(overrides)
    return PreparedProposalFinalizeRow(**values)


class _Adapter:
    def __init__(self, capabilities, proposals):
        self.capabilities = capabilities
        self.proposals = proposals
        self.calls = []

    def propose_batch(self, contexts):
        self.calls.append(contexts)
        return self.proposals


def test_prepare_finalize_bridge_round_trips_ticket():
    expected_row_type = getattr(
        __import__(
            "tinyvllm.engine.speculative_proposal_executor",
            fromlist=["ProposalFinalizeRow"],
        ),
        "ProposalFinalizeRow",
    )

    def callback(method_name, args):
        assert (
            method_name
            == "prepare_speculative_proposal_finalize_batch"
        )
        assert args == (
            "fixture",
            (
                expected_row_type(
                    sequence_id=1,
                    proposal_transaction_id="tx-1",
                    accepted_proposal_tokens=2,
                ),
            ),
        )
        return "ticket-1"

    runner = _FakeModelRunner(callback)
    ticket = (
        speculative_model_runner_module
        .prepare_model_runner_proposal_finalize_batch(
            runner,
            _lifecycle_descriptor(),
            (_prepared_finalize_row(),),
        )
    )

    assert ticket == "ticket-1"


@pytest.mark.parametrize(
    "helper_name,method_name",
    [
        (
            "commit_model_runner_proposal_finalize_batch",
            "commit_speculative_proposal_finalize_batch",
        ),
        (
            "rollback_model_runner_proposal_finalize_batch",
            "rollback_speculative_proposal_finalize_batch",
        ),
    ],
)
def test_finalize_bridge_uses_exact_operation(
    helper_name,
    method_name,
):
    def callback(actual_method_name, args):
        assert actual_method_name == method_name
        assert args == ("fixture", "ticket-1")
        return None

    runner = _FakeModelRunner(callback)

    result = getattr(
        speculative_model_runner_module,
        helper_name,
    )(
        runner,
        _lifecycle_descriptor(),
        "ticket-1",
    )

    assert result is None


@pytest.mark.parametrize("ticket", ["", 1, None])
def test_prepare_finalize_bridge_rejects_invalid_ticket_ack(
    ticket,
):
    runner = _FakeModelRunner(
        lambda method_name, args: ticket
    )

    with pytest.raises(ValueError, match="ticket"):
        (
            speculative_model_runner_module
            .prepare_model_runner_proposal_finalize_batch(
                runner,
                _lifecycle_descriptor(),
                (_prepared_finalize_row(),),
            )
        )


def test_commit_finalize_bridge_rejects_malformed_ack():
    runner = _FakeModelRunner(
        lambda method_name, args: "unexpected"
    )

    with pytest.raises(ValueError, match="acknowledgement"):
        (
            speculative_model_runner_module
            .commit_model_runner_proposal_finalize_batch(
                runner,
                _lifecycle_descriptor(),
                "ticket-1",
            )
        )


def test_release_model_runner_proposal_sequence_uses_exact_rpc():
    def callback(method_name, args):
        assert (
            method_name
            == "release_speculative_proposal_sequence"
        )
        assert args == ("fixture", 7, 3)
        return None

    runner = _FakeModelRunner(callback)

    result = (
        speculative_model_runner_module
        .release_model_runner_proposal_sequence(
            runner,
            _lifecycle_descriptor(),
            7,
            3,
        )
    )

    assert result is None


@pytest.mark.parametrize(
    ("sequence_id", "sequence_epoch", "match"),
    (
        (True, 0, "sequence ID"),
        (-1, 0, "sequence ID"),
        (7, True, "sequence epoch"),
        (7, -1, "sequence epoch"),
    ),
)
def test_release_model_runner_proposal_sequence_rejects_invalid_identity(
    sequence_id,
    sequence_epoch,
    match,
):
    runner = _FakeModelRunner(
        lambda method_name, args: None
    )

    with pytest.raises(ValueError, match=match):
        (
            speculative_model_runner_module
            .release_model_runner_proposal_sequence(
                runner,
                _lifecycle_descriptor(),
                sequence_id,
                sequence_epoch,
            )
        )

    assert runner.calls == []


def test_release_model_runner_proposal_sequence_rejects_malformed_ack():
    runner = _FakeModelRunner(
        lambda method_name, args: "unexpected"
    )

    with pytest.raises(ValueError, match="acknowledgement"):
        (
            speculative_model_runner_module
            .release_model_runner_proposal_sequence(
                runner,
                _lifecycle_descriptor(),
                7,
                0,
            )
        )


def test_proposal_lifecycle_uses_explicit_dispatch_when_supplied():
    runner = _FakeModelRunner(
        lambda method_name, args: (_ for _ in ()).throw(
            AssertionError("model_runner.call must not be used")
        )
    )
    calls = []

    def dispatch(method_name, *args):
        calls.append((method_name, args))
        if method_name == (
            "prepare_speculative_proposal_finalize_batch"
        ):
            return "ticket-1"
        return None

    ticket = (
        speculative_model_runner_module
        .prepare_model_runner_proposal_finalize_batch(
            runner,
            _lifecycle_descriptor(),
            (
                _prepared_finalize_row(
                    proposal_transaction_id="tx-1",
                ),
            ),
            dispatch=dispatch,
        )
    )
    (
        speculative_model_runner_module
        .commit_model_runner_proposal_finalize_batch(
            runner,
            _lifecycle_descriptor(),
            ticket,
            dispatch=dispatch,
        )
    )
    (
        speculative_model_runner_module
        .release_model_runner_proposal_sequence(
            runner,
            _lifecycle_descriptor(),
            7,
            0,
            dispatch=dispatch,
        )
    )

    assert [name for name, _ in calls] == [
        "prepare_speculative_proposal_finalize_batch",
        "commit_speculative_proposal_finalize_batch",
        "release_speculative_proposal_sequence",
    ]
    assert runner.calls == []


def test_prepare_finalize_bridge_rejects_tensor_row():
    Tensor = type("Tensor", (), {"__module__": "torch"})
    runner = _FakeModelRunner(
        lambda method_name, args: "ticket-1"
    )

    with pytest.raises(ValueError, match="tensor"):
        (
            speculative_model_runner_module
            .prepare_model_runner_proposal_finalize_batch(
                runner,
                _lifecycle_descriptor(),
                (
                    _prepared_finalize_row(
                        proposal_transaction_id=Tensor(),
                    ),
                ),
            )
        )


def test_host_provider_combines_first_targets_and_proposals():
    seqs = (
        SimpleNamespace(
            seq_id=8,
            token_ids=[1, 8],
            max_tokens=6,
            num_completion_tokens=1,
        ),
        SimpleNamespace(
            seq_id=4,
            token_ids=[1, 4],
            max_tokens=5,
            num_completion_tokens=2,
        ),
    )
    first_targets = (
        FirstTargetResult(
            sequence_id=4,
            target_token=201,
            metadata={"batch_index": 1},
        ),
        FirstTargetResult(
            sequence_id=8,
            target_token=101,
            metadata={"batch_index": 0},
        ),
    )
    adapter = _Adapter(
        _capabilities(
            requires_hidden=False,
            requires_logits=False,
        ),
        (
            DraftProposal(4, (), "fixture"),
            DraftProposal(8, (11, 12), "fixture"),
        ),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: first_targets
    )

    rows = run_host_first_targets_and_proposals(
        runner,
        seqs,
        adapter,
    )

    assert rows == (
        FirstTargetProposalResult(
            sequence_id=8,
            target_token=101,
            proposal=DraftProposal(
                8,
                (11, 12),
                "fixture",
            ),
            first_target_metadata={
                "batch_index": 0,
            },
        ),
        FirstTargetProposalResult(
            sequence_id=4,
            target_token=201,
            proposal=DraftProposal(
                4,
                (),
                "fixture",
            ),
            first_target_metadata={
                "batch_index": 1,
            },
        ),
    )
    assert tuple(
        context.sequence_id
        for context in adapter.calls[0]
    ) == (8, 4)


def test_fused_provider_uses_one_rpc_and_restores_order():
    seqs = (
        SimpleNamespace(seq_id=8),
        SimpleNamespace(seq_id=4),
    )
    capabilities = _capabilities(
        requires_hidden=False,
        requires_logits=False,
        execution_domain="model_runner",
    )
    descriptor = SimpleNamespace(
        executor_id="fixture",
        capabilities=capabilities,
    )
    result = (
        FirstTargetProposalResult(
            sequence_id=4,
            target_token=201,
            proposal=DraftProposal(4, (), "fixture"),
        ),
        FirstTargetProposalResult(
            sequence_id=8,
            target_token=101,
            proposal=DraftProposal(
                8,
                (11, 12),
                "fixture",
            ),
        ),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: result
    )

    rows = run_model_runner_first_targets_and_proposals(
        runner,
        seqs,
        descriptor,
    )

    assert tuple(row.sequence_id for row in rows) == (8, 4)
    assert runner.calls == [
        (
            "run_spec_first_target_and_proposal_batch",
            (seqs, descriptor, ()),
        ),
    ]


def test_fused_provider_rejects_nested_tensor_result():
    Tensor = type("Tensor", (), {"__module__": "torch"})
    capabilities = _capabilities(
        requires_hidden=False,
        requires_logits=False,
        execution_domain="model_runner",
    )
    descriptor = SimpleNamespace(
        executor_id="fixture",
        capabilities=capabilities,
    )
    result = FirstTargetProposalResult(
        sequence_id=8,
        target_token=101,
        proposal=DraftProposal(
            sequence_id=8,
            token_ids=(11,),
            source_type="fixture",
            metadata={"tensor": Tensor()},
        ),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: (result,)
    )

    with pytest.raises(ValueError, match="tensor"):
        run_model_runner_first_targets_and_proposals(
            runner,
            (SimpleNamespace(seq_id=8),),
            descriptor,
        )


def test_fused_provider_rejects_nested_proposal_id_mismatch():
    capabilities = _capabilities(
        requires_hidden=False,
        requires_logits=False,
        execution_domain="model_runner",
    )
    descriptor = SimpleNamespace(
        executor_id="fixture",
        capabilities=capabilities,
    )
    result = FirstTargetProposalResult(
        sequence_id=8,
        target_token=101,
        proposal=DraftProposal(
            sequence_id=4,
            token_ids=(11,),
            source_type="fixture",
        ),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: (result,)
    )

    with pytest.raises(ValueError, match="proposal sequence ID"):
        run_model_runner_first_targets_and_proposals(
            runner,
            (SimpleNamespace(seq_id=8),),
            descriptor,
        )


def test_provider_builder_selects_host_domain():
    seq = SimpleNamespace(
        seq_id=8,
        token_ids=[1, 8],
        max_tokens=4,
        num_completion_tokens=0,
    )
    capabilities = _capabilities(
        requires_hidden=False,
        requires_logits=False,
    )
    adapter = _Adapter(
        capabilities,
        (DraftProposal(8, (11,), "fixture"),),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: (
            FirstTargetResult(8, 101),
        )
    )
    runtime = SimpleNamespace(
        capabilities=capabilities,
        draft_adapter=adapter,
        model_runner_executor=None,
    )
    provider = build_model_runner_proposal_provider(
        runner,
        runtime,
        lambda seqs: (),
    )

    rows = provider((seq,))

    assert rows[0].proposal.token_ids == (11,)
    assert runner.calls[0][0] == "run_spec_first_target_batch"


def test_provider_builder_selects_model_runner_domain():
    seq = SimpleNamespace(seq_id=8)
    capabilities = _capabilities(
        requires_hidden=False,
        requires_logits=False,
        execution_domain="model_runner",
    )
    descriptor = SimpleNamespace(
        executor_id="fixture",
        capabilities=capabilities,
    )
    result = FirstTargetProposalResult(
        sequence_id=8,
        target_token=101,
        proposal=DraftProposal(8, (11,), "fixture"),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: (result,)
    )
    runtime = SimpleNamespace(
        capabilities=capabilities,
        draft_adapter=None,
        model_runner_executor=descriptor,
    )
    provider = build_model_runner_proposal_provider(
        runner,
        runtime,
        lambda seqs: (),
    )

    rows = provider((seq,))

    assert rows == (result,)
    assert (
        runner.calls[0][0]
        == "run_spec_first_target_and_proposal_batch"
    )


def test_first_target_bridge_uses_one_rpc_and_restores_order():
    seqs = (
        SimpleNamespace(seq_id=8),
        SimpleNamespace(seq_id=4),
    )
    identity_rows = (
        KVBlockIdentityRow(8, ((1, 3),)),
        KVBlockIdentityRow(4, ((2, 5),)),
    )

    def callback(method_name, args):
        assert method_name == (
            "run_spec_first_target_batch"
        )
        assert args == (
            seqs,
            True,
            True,
            identity_rows,
        )
        return (
            FirstTargetResult(
                sequence_id=4,
                target_token=201,
            ),
            FirstTargetResult(
                sequence_id=8,
                target_token=101,
            ),
        )

    runner = _FakeModelRunner(callback)
    rows = run_model_runner_first_targets(
        runner,
        seqs,
        _capabilities(),
        identity_rows,
    )

    assert tuple(row.sequence_id for row in rows) == (8, 4)
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "result,match",
    [
        (None, "tuple"),
        ((object(),), "FirstTargetResult"),
        (
            (
                FirstTargetResult(8, 101),
                FirstTargetResult(8, 102),
            ),
            "unique",
        ),
        (
            (FirstTargetResult(8, 101),),
            "exactly match",
        ),
        (
            (
                FirstTargetResult(8, 101),
                FirstTargetResult(4, 201),
                FirstTargetResult(2, 301),
            ),
            "exactly match",
        ),
    ],
)
def test_first_target_bridge_rejects_invalid_results(
    result,
    match,
):
    runner = _FakeModelRunner(
        lambda method_name, args: result
    )
    seqs = (
        SimpleNamespace(seq_id=8),
        SimpleNamespace(seq_id=4),
    )

    with pytest.raises(ValueError, match=match):
        run_model_runner_first_targets(
            runner,
            seqs,
            _capabilities(
                requires_hidden=False,
                requires_logits=False,
            ),
        )


def test_tail_bridge_runs_once_per_distinct_q_and_merges_order():
    items = _variable_q_items()
    residency_ticket_id = 41

    def callback(method_name, args):
        assert method_name == "run_spec_verify_batch"
        group_items = args[0]
        assert args[1] == residency_ticket_id
        return tuple(
            SpecVerifyBatchResultRow(
                sequence_id=item.sequence_id,
                target_tokens=tuple(
                    item.sequence_id * 100 + offset
                    for offset in range(
                        item.plan.query_len
                    )
                ),
            )
            for item in reversed(group_items)
        )

    runner = _FakeModelRunner(callback)
    rows = run_model_runner_tail_batch(
        runner,
        items,
        residency_ticket_id=residency_ticket_id,
    )

    assert tuple(row.sequence_id for row in rows) == (
        8,
        4,
        2,
        9,
    )
    assert all(
        isinstance(row, TailBatchResult)
        for row in rows
    )
    assert tuple(
        call[1][0][0].plan.query_len
        for call in runner.calls
    ) == (2, 1, 3)
    assert len(runner.calls) == 3
    assert tuple(
        row.metadata["fixed_q_group_count"]
        for row in rows
    ) == (3, 3, 3, 3)
    assert tuple(
        row.metadata["fixed_q_group_index"]
        for row in rows
    ) == (0, 1, 0, 2)


def test_tail_bridge_propagates_replay_failure_to_batch_owner():
    replay_error = SpecVerifyGraphReplayError(
        "f" * 64,
        RuntimeError("replay failed"),
    )

    def callback(method_name, args):
        assert method_name == "run_spec_verify_batch"
        assert len(args[0]) == 1
        raise replay_error

    runner = _FakeModelRunner(callback)

    with pytest.raises(
        SpecVerifyGraphReplayError,
    ) as exc_info:
        run_model_runner_tail_batch(
            runner,
            (_tail_item(8, 2),),
        )

    assert exc_info.value is replay_error


@pytest.mark.parametrize(
    "result_factory,match",
    [
        (lambda items: None, "tuple"),
        (lambda items: (object(),), "batch result row"),
        (
            lambda items: (
                SpecVerifyBatchResultRow(
                    sequence_id=items[0].sequence_id,
                    target_tokens=(1,) * items[0].plan.query_len,
                ),
                SpecVerifyBatchResultRow(
                    sequence_id=items[0].sequence_id,
                    target_tokens=(2,) * items[0].plan.query_len,
                ),
            ),
            "unique",
        ),
        (
            lambda items: (
                SpecVerifyBatchResultRow(
                    sequence_id=999,
                    target_tokens=(1,) * items[0].plan.query_len,
                ),
            ),
            "exactly match",
        ),
        (
            lambda items: (
                SpecVerifyBatchResultRow(
                    sequence_id=items[0].sequence_id,
                    target_tokens=(1,),
                ),
            ),
            "target count",
        ),
    ],
)
def test_tail_bridge_rejects_invalid_group_results(
    result_factory,
    match,
):
    items = (_tail_item(8, 2),)
    runner = _FakeModelRunner(
        lambda method_name, args: result_factory(
            args[0]
        )
    )

    with pytest.raises(ValueError, match=match):
        run_model_runner_tail_batch(runner, items)
