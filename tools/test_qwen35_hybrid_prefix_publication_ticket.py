import importlib.util
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

hybrid = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
adapter_module = _load_module(
    "tinyvllm.engine.qwen35_layer_state",
    "tinyvllm/engine/qwen35_layer_state.py",
)
transaction_module = _load_module(
    "tinyvllm.engine.qwen35_state_transaction",
    "tinyvllm/engine/qwen35_state_transaction.py",
)
cache_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_cache",
    "tinyvllm/engine/qwen35_hybrid_prefix_cache.py",
)
ticket_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_ticket.py",
)

HybridStateComponentSpec = hybrid.HybridStateComponentSpec
HybridStateLayout = hybrid.HybridStateLayout
HybridStateLease = hybrid.HybridStateLease
HybridStateTensorPool = hybrid.HybridStateTensorPool
Qwen35LayerStateAdapter = adapter_module.Qwen35LayerStateAdapter
Qwen35CrossLayerStateTransaction = (
    transaction_module.Qwen35CrossLayerStateTransaction
)
Qwen35HybridPrefixKey = cache_module.Qwen35HybridPrefixKey
Qwen35HybridPrefixSnapshotCache = (
    cache_module.Qwen35HybridPrefixSnapshotCache
)
Qwen35HybridPrefixPublicationPayload = (
    ticket_module.Qwen35HybridPrefixPublicationPayload
)
Qwen35HybridPrefixPublicationParticipant = (
    ticket_module.Qwen35HybridPrefixPublicationParticipant
)


def _layout():
    return HybridStateLayout(tuple(
        component
        for layer_index in (0, 2)
        for component in (
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                (2, 3),
                torch.float32,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                (2, 2, 2),
                torch.float32,
            ),
        )
    ))


def _fixture(max_bytes=1 << 20):
    pool = HybridStateTensorPool(_layout(), capacity=3, device="cpu")
    leases = tuple(
        HybridStateLease(index, 1, 100 + index)
        for index in range(3)
    )
    for lease in leases:
        pool.activate(lease)
    adapters = (
        Qwen35LayerStateAdapter(pool, 0),
        Qwen35LayerStateAdapter(pool, 2),
    )
    transaction = Qwen35CrossLayerStateTransaction(adapters)
    for layer_offset, adapter in enumerate(adapters):
        for slot_id in range(3):
            adapter.convolution[slot_id].copy_(
                torch.arange(6).reshape(2, 3)
                + layer_offset * 1000
                + slot_id * 100
            )
            adapter.recurrent[slot_id].copy_(
                torch.arange(8).reshape(2, 2, 2)
                + layer_offset * 2000
                + slot_id * 200
            )
    cache = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=max_bytes,
    )
    participant = Qwen35HybridPrefixPublicationParticipant(
        0,
        pool,
        cache,
    )
    return pool, leases, adapters, cache, participant


def _key():
    return Qwen35HybridPrefixKey(
        token_hash=201,
        token_count=4,
        terminal_block_hash=201,
        block_size=4,
        model_fingerprint="model-a",
        layout_fingerprint="layout-a",
        tensor_parallel_size=1,
        dtype=torch.float32,
    )


def _payload(ticket_id=0, lease=None, participant_id=0):
    if lease is None:
        lease = HybridStateLease(0, 1, 100)
    return Qwen35HybridPrefixPublicationPayload(
        ticket_id=ticket_id,
        participant_id=participant_id,
        request_id=lease.request_id,
        key=_key(),
        token_ids=(1, 2, 3, 4),
        block_identities=((7, 3, 201),),
        lease=lease,
    )


def _expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_prepare_is_invisible_idempotent_and_payload_bound():
    _, leases, _, cache, participant = _fixture()
    payload = _payload(lease=leases[0])
    first = participant.prepare(payload)
    second = participant.prepare(payload)

    assert first == second
    assert first.status == "prepared"
    observation = cache.observation_snapshot()
    assert observation["current_entries"] == 0
    assert observation["current_prepared_publications"] == 1

    changed = _payload(ticket_id=payload.ticket_id, lease=leases[1])
    acknowledgement = participant.prepare(changed)
    assert acknowledgement.status == "error"
    assert "different payload" in acknowledgement.detail
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 1


def test_oversize_prepare_rejects_without_ticket_state():
    _, leases, _, cache, participant = _fixture(max_bytes=1)
    payload = _payload(lease=leases[0])
    acknowledgement = participant.prepare(payload)
    assert acknowledgement.status == "rejected"
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0
    assert participant.prepare(payload) == acknowledgement


def test_invalid_prepared_handle_reports_failed_cache_abort():
    pool, leases, _, cache, _ = _fixture()

    class InvalidHandleCache(Qwen35HybridPrefixSnapshotCache):
        def prepare_publication(self, *args, **kwargs):
            return object()

        def abort_current_publication(self):
            raise RuntimeError("injected cache abort failure")

    invalid_cache = InvalidHandleCache(
        cache.state_transaction,
        max_entries=4,
        max_bytes=1 << 20,
    )
    participant = Qwen35HybridPrefixPublicationParticipant(
        0,
        pool,
        invalid_cache,
    )
    acknowledgement = participant.prepare(_payload(lease=leases[0]))

    assert acknowledgement.status == "error"
    assert "rollback failed" in acknowledgement.detail


def test_rollback_aborts_rejected_and_unseen_tickets_idempotently():
    _, leases, _, cache, rejected_participant = _fixture(max_bytes=1)
    rejected_payload = _payload(lease=leases[0])
    assert (
        rejected_participant.prepare(rejected_payload).status
        == "rejected"
    )
    first = rejected_participant.rollback(rejected_payload)
    second = rejected_participant.rollback(rejected_payload)
    assert first == second
    assert first.status == "rolled_back"
    assert cache.observation_snapshot()["current_entries"] == 0
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0

    _, leases, _, cache, unseen_participant = _fixture()
    unseen_payload = _payload(ticket_id=9, lease=leases[1])
    first = unseen_participant.rollback(unseen_payload)
    second = unseen_participant.rollback(unseen_payload)
    assert first == second
    assert first.status == "rolled_back"
    assert cache.observation_snapshot()["current_entries"] == 0
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0


def test_commit_is_exact_idempotent_and_rejects_rollback_afterward():
    _, leases, adapters, cache, participant = _fixture()
    payload = _payload(lease=leases[1])
    staged = tuple(
        (
            adapter.convolution[1].clone(),
            adapter.recurrent[1].clone(),
        )
        for adapter in adapters
    )
    assert participant.prepare(payload).status == "prepared"
    assert participant.precommit(payload).status == "precommitted"
    assert participant.precommit(payload).status == "precommitted"
    for adapter in adapters:
        adapter.convolution[1].add_(9000)
        adapter.recurrent[1].add_(9000)

    first = participant.commit(payload)
    second = participant.commit(payload)
    assert first == second
    assert first.status == "finalized"
    sealed = participant.seal(payload)
    assert sealed.status == "committed"
    assert participant.seal(payload) == sealed
    assert cache.observation_snapshot()["current_entries"] == 1

    for adapter in adapters:
        adapter.convolution[2].zero_()
        adapter.recurrent[2].zero_()
    assert cache.acquire(
        payload.key,
        payload.token_ids,
        payload.block_identities,
        (leases[2],),
    ) is True
    for layer_index, adapter in enumerate(adapters):
        torch.testing.assert_close(
            adapter.convolution[2],
            staged[layer_index][0],
        )
        torch.testing.assert_close(
            adapter.recurrent[2],
            staged[layer_index][1],
        )
    rollback = participant.rollback(payload)
    assert rollback.status == "error"
    assert "committed" in rollback.detail


def test_rollback_is_idempotent_and_rejects_commit_afterward():
    _, leases, _, cache, participant = _fixture()
    payload = _payload(lease=leases[0])
    assert participant.prepare(payload).status == "prepared"
    first = participant.rollback(payload)
    second = participant.rollback(payload)
    assert first == second
    assert first.status == "rolled_back"
    assert cache.observation_snapshot()["current_entries"] == 0
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0
    commit = participant.commit(payload)
    assert commit.status == "error"
    assert "rolled_back" in commit.detail


def test_commit_failure_keeps_prepared_ticket_for_rollback():
    _, leases, _, cache, participant = _fixture()
    payload = _payload(lease=leases[0])
    assert participant.prepare(payload).status == "prepared"
    assert participant.precommit(payload).status == "precommitted"

    original_finalize = cache.finalize_publication

    def failing_finalize(_):
        raise RuntimeError("injected participant commit failure")

    cache.finalize_publication = failing_finalize
    acknowledgement = participant.commit(payload)
    cache.finalize_publication = original_finalize
    assert acknowledgement.status == "error"
    assert "participant commit failure" in acknowledgement.detail
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 1
    assert participant.rollback(payload).status == "rolled_back"
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0


def main():
    test_prepare_is_invisible_idempotent_and_payload_bound()
    test_oversize_prepare_rejects_without_ticket_state()
    test_invalid_prepared_handle_reports_failed_cache_abort()
    test_rollback_aborts_rejected_and_unseen_tickets_idempotently()
    test_commit_is_exact_idempotent_and_rejects_rollback_afterward()
    test_rollback_is_idempotent_and_rejects_commit_afterward()
    test_commit_failure_keeps_prepared_ticket_for_rollback()
    print("qwen35 hybrid prefix publication participant tests passed")


if __name__ == "__main__":
    main()
