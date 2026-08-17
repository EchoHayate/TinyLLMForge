from dataclasses import replace
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
coordinator_module = _load_module(
    "tinyvllm.engine.qwen35_hybrid_prefix_publication_coordinator",
    "tinyvllm/engine/qwen35_hybrid_prefix_publication_coordinator.py",
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
Qwen35HybridPrefixPublicationCoordinator = (
    coordinator_module.Qwen35HybridPrefixPublicationCoordinator
)


def _layout():
    return HybridStateLayout((
        HybridStateComponentSpec(
            0,
            "linear_convolution",
            (2, 3),
            torch.float32,
        ),
        HybridStateComponentSpec(
            0,
            "linear_recurrent",
            (2, 2, 2),
            torch.float32,
        ),
    ))


def _rank_fixture(rank, max_bytes=1 << 20):
    pool = HybridStateTensorPool(_layout(), capacity=2, device="cpu")
    leases = (
        HybridStateLease(0, 1, 100),
        HybridStateLease(1, 1, 101),
    )
    for lease in leases:
        pool.activate(lease)
    adapter = Qwen35LayerStateAdapter(pool, 0)
    for slot_id in range(2):
        adapter.convolution[slot_id].copy_(
            torch.arange(6).reshape(2, 3) + rank * 1000 + slot_id * 100
        )
        adapter.recurrent[slot_id].copy_(
            torch.arange(8).reshape(2, 2, 2)
            + rank * 2000
            + slot_id * 200
        )
    transaction = Qwen35CrossLayerStateTransaction((adapter,))
    cache = Qwen35HybridPrefixSnapshotCache(
        transaction,
        max_entries=4,
        max_bytes=max_bytes,
    )
    participant = Qwen35HybridPrefixPublicationParticipant(
        rank,
        pool,
        cache,
    )
    return leases, adapter, cache, participant


def _key():
    return Qwen35HybridPrefixKey(
        token_hash=201,
        token_count=4,
        terminal_block_hash=201,
        block_size=4,
        model_fingerprint="model-a",
        layout_fingerprint="layout-a",
        tensor_parallel_size=2,
        dtype=torch.float32,
    )


def _payload(rank, lease, ticket_id=7):
    return Qwen35HybridPrefixPublicationPayload(
        ticket_id=ticket_id,
        participant_id=rank,
        request_id=lease.request_id,
        key=_key(),
        token_ids=(1, 2, 3, 4),
        block_identities=((9, 4, 201),),
        lease=lease,
    )


def _fixture(max_bytes_by_rank=(1 << 20, 1 << 20)):
    ranks = tuple(
        _rank_fixture(rank, max_bytes_by_rank[rank])
        for rank in range(2)
    )
    coordinator = Qwen35HybridPrefixPublicationCoordinator(tuple(
        rank[3] for rank in ranks
    ))
    payloads = tuple(
        _payload(rank, ranks[rank][0][0])
        for rank in range(2)
    )
    return ranks, coordinator, payloads


def _expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_two_rank_publication_succeeds_after_precommit_barrier():
    ranks, coordinator, payloads = _fixture()
    assert coordinator.publish(payloads) is True
    for _, _, cache, _ in ranks:
        observation = cache.observation_snapshot()
        assert observation["current_entries"] == 1
        assert observation["current_prepared_publications"] == 0
        assert observation["publication_precommits"] == 1
        assert observation["publication_commits"] == 1


def test_matrix_mismatch_rejects_before_participant_mutation():
    ranks, coordinator, payloads = _fixture()
    bad_cases = (
        payloads[:1],
        (payloads[0], payloads[0]),
        (
            payloads[0],
            replace(
                payloads[1],
                token_ids=(5, 6, 7, 8),
            ),
        ),
        (
            payloads[0],
            replace(
                payloads[1],
                key=replace(payloads[1].key, tensor_parallel_size=3),
            ),
        ),
    )
    for payload_case in bad_cases:
        _expect_error(
            lambda payload_case=payload_case: coordinator.publish(
                payload_case
            ),
            "payload",
        )
        for _, _, cache, _ in ranks:
            assert cache.observation_snapshot()["current_entries"] == 0
            assert cache.observation_snapshot()[
                "current_prepared_publications"
            ] == 0


def test_precommit_failure_rolls_back_every_rank_before_visibility():
    ranks, coordinator, payloads = _fixture()
    failing_participant = ranks[1][3]
    original_precommit = failing_participant.precommit

    def failing_precommit(payload):
        acknowledgement = original_precommit(payload)
        if acknowledgement.status == "precommitted":
            return replace(
                acknowledgement,
                status="error",
                detail="injected rank precommit failure",
            )
        return acknowledgement

    failing_participant.precommit = failing_precommit
    _expect_error(
        lambda: coordinator.publish(payloads),
        "precommit failure",
    )
    for _, _, cache, _ in ranks:
        observation = cache.observation_snapshot()
        assert observation["current_entries"] == 0
        assert observation["current_prepared_publications"] == 0
        assert observation["current_bytes"] == 0


def test_prepare_rejection_rolls_back_earlier_rank_and_returns_false():
    ranks, coordinator, payloads = _fixture(
        max_bytes_by_rank=(1 << 20, 1),
    )
    assert coordinator.publish(payloads) is False
    for _, _, cache, _ in ranks:
        observation = cache.observation_snapshot()
        assert observation["current_entries"] == 0
        assert observation["current_prepared_publications"] == 0
        assert observation["current_bytes"] == 0


def test_finalize_failure_rolls_back_all_ranks_and_allows_reuse():
    ranks, coordinator, payloads = _fixture()
    failing_participant = ranks[1][3]
    original_commit = failing_participant.commit

    def failing_commit(payload):
        return ticket_module.Qwen35HybridPrefixPublicationAck(
            ticket_id=payload.ticket_id,
            participant_id=payload.participant_id,
            operation="commit",
            status="error",
            detail="injected rank finalize failure",
        )

    failing_participant.commit = failing_commit
    _expect_error(
        lambda: coordinator.publish(payloads),
        "finalize failure",
    )
    assert ranks[0][2].observation_snapshot()["current_entries"] == 0
    assert ranks[1][2].observation_snapshot()["current_entries"] == 0
    assert ranks[1][2].observation_snapshot()[
        "current_prepared_publications"
    ] == 0
    next_payloads = tuple(
        replace(payload, ticket_id=8)
        for payload in payloads
    )
    failing_participant.commit = original_commit
    assert coordinator.publish(next_payloads) is True


def test_seal_failure_poisons_and_blocks_reuse():
    _, coordinator, payloads = _fixture()
    failing_participant = coordinator.participants[1]
    original_seal = failing_participant.seal

    def failing_seal(payload):
        return ticket_module.Qwen35HybridPrefixPublicationAck(
            ticket_id=payload.ticket_id,
            participant_id=payload.participant_id,
            operation="seal",
            status="error",
            detail="injected rank seal failure",
        )

    failing_participant.seal = failing_seal
    _expect_error(
        lambda: coordinator.publish(payloads),
        "seal failure",
    )
    failing_participant.seal = original_seal
    _expect_error(
        lambda: coordinator.publish(tuple(
            replace(payload, ticket_id=9)
            for payload in payloads
        )),
        "poisoned",
    )


def main():
    test_two_rank_publication_succeeds_after_precommit_barrier()
    test_matrix_mismatch_rejects_before_participant_mutation()
    test_precommit_failure_rolls_back_every_rank_before_visibility()
    test_prepare_rejection_rolls_back_earlier_rank_and_returns_false()
    test_finalize_failure_rolls_back_all_ranks_and_allows_reuse()
    test_seal_failure_poisons_and_blocks_reuse()
    print("qwen35 hybrid prefix publication coordinator tests passed")


if __name__ == "__main__":
    main()
