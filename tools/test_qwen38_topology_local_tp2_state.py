from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip(
        "Qwen3.8 TP2 state tests require PyTorch",
        allow_module_level=True,
    )
if not hasattr(torch, "device"):
    pytest.skip(
        "Qwen3.8 TP2 state tests require a real PyTorch module",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


from tinyvllm.engine.hybrid_state import (
    HybridStateLease,
    HybridStateTensorPool,
)
from tinyvllm.engine.qwen35_hybrid_state import (
    build_qwen35_hybrid_state_layout,
)
from tinyvllm.engine.qwen35_layer_state import Qwen35LayerStateAdapter
from tinyvllm.engine.qwen35_state_transaction import (
    Qwen35CrossLayerStateTransaction,
)
from tinyvllm.engine.topology_local_tp2_island import (
    TopologyLocalTP2PairMap,
)


def frozen_qwen38_config(**overrides):
    values = {
        "num_hidden_layers": 64,
        "layer_types": tuple(
            "full_attention" if (index + 1) % 4 == 0
            else "linear_attention"
            for index in range(64)
        ),
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def build_source_transaction(
    *,
    capacity=2,
    config=None,
    tensor_parallel_size=4,
):
    config = config or frozen_qwen38_config()
    layout = build_qwen35_hybrid_state_layout(
        config,
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
        speculative_tokens=1,
    )
    pool = HybridStateTensorPool(layout, capacity, "cpu")
    adapters = tuple(
        Qwen35LayerStateAdapter(pool, layer_index)
        for layer_index in sorted({
            component.layer_index for component in layout.components
        })
    )
    return Qwen35CrossLayerStateTransaction(adapters)


def activate_source(
    transaction,
    *leases,
):
    for lease in leases:
        transaction.pool.activate(lease)
    for adapter in transaction.adapters:
        for lease in leases:
            convolution = adapter.convolution[lease.slot_id]
            convolution.copy_(torch.tensor(
                [0, 0, 1, 1, 2, 2],
                dtype=convolution.dtype,
            ).view(6, 1).expand_as(convolution))
            recurrent = adapter.recurrent[lease.slot_id]
            recurrent.fill_(float(adapter.layer_index))


class FakeWorldAllGather:
    def __init__(self, *, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def __call__(self, local):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected collective failure")
        return tuple(
            local.clone() + rank * 10
            for rank in range(4)
        )


def build_owner(
    *,
    logical_rank=0,
    source_transaction=None,
    all_gather=None,
    capacity=2,
    config=None,
):
    from tinyvllm.engine.qwen38_topology_local_tp2_state import (
        build_qwen38_topology_local_tp2_state_owner,
    )

    config = config or frozen_qwen38_config()
    source_transaction = (
        source_transaction
        or build_source_transaction(
            capacity=capacity,
            config=config,
        )
    )
    pair_identity = TopologyLocalTP2PairMap(
        ((0, 1), (2, 3))
    ).identity(logical_rank)
    return build_qwen38_topology_local_tp2_state_owner(
        hf_config=config,
        capacity=capacity,
        device="cpu",
        source_transaction=source_transaction,
        pair_identity=pair_identity,
        all_gather=all_gather or FakeWorldAllGather(),
    )


def test_builds_dual_layout_and_atomically_publishes_all_48_layers():
    source = build_source_transaction()
    lease = HybridStateLease(0, 1, 101)
    activate_source(source, lease)
    owner = build_owner(source_transaction=source)

    assert owner.destination_pool.layout.bytes_per_slot == (
        2 * owner.source_pool.layout.bytes_per_slot
    )

    rows = owner.migrate((lease,))

    assert len(rows) == 48
    assert all(row["published"] is True for row in rows)
    assert {row["layer_index"] for row in rows} == {
        index for index in range(64) if (index + 1) % 4
    }
    assert owner.phase_for(lease) == "tp2_decode"
    first = owner.gather((lease,))[0]
    assert first["layer_index"] == 0
    assert first["convolution_states"].shape == (1, 12, 3)
    assert first["recurrent_states"].shape == (1, 2, 2, 2)
    assert first["convolution_states"][0, :, 0].tolist() == [
        0,
        0,
        10,
        10,
        1,
        1,
        11,
        11,
        2,
        2,
        12,
        12,
    ]
    assert first["recurrent_states"][0, :, 0, 0].tolist() == [
        0.0,
        10.0,
    ]
    snapshot = owner.snapshot()
    assert snapshot["schema_version"] == (
        "qwen38.topology-local-tp2-state-snapshot.v1"
    )
    assert snapshot["publication_count"] == 1
    assert snapshot["rollback_count"] == 0
    assert snapshot["temporary_live_tensors"] == 0


def test_logical_rank_one_selects_world_ranks_two_and_three():
    source = build_source_transaction()
    lease = HybridStateLease(0, 1, 102)
    activate_source(source, lease)
    owner = build_owner(
        logical_rank=1,
        source_transaction=source,
    )

    owner.migrate((lease,))
    first = owner.gather((lease,))[0]

    assert first["convolution_states"][0, :, 0].tolist() == [
        20,
        20,
        30,
        30,
        21,
        21,
        31,
        31,
        22,
        22,
        32,
        32,
    ]
    assert first["recurrent_states"][0, :, 0, 0].tolist() == [
        20.0,
        30.0,
    ]


def test_candidate_canonical_state_components_match_tp4_source_quarters():
    from tinyvllm.engine.qwen38_topology_local_tp2_state import (
        build_qwen38_tp4_state_component_digests,
    )

    config = frozen_qwen38_config()
    lease = HybridStateLease(0, 1, 112)
    source_rank_zero = build_source_transaction(config=config)
    activate_source(source_rank_zero, lease)
    owner = build_owner(
        logical_rank=0,
        source_transaction=source_rank_zero,
        config=config,
    )
    owner.migrate((lease,))

    source_rank_one = build_source_transaction(config=config)
    activate_source(source_rank_one, lease)
    for adapter in source_rank_one.adapters:
        adapter.convolution[lease.slot_id].add_(10)
        adapter.recurrent[lease.slot_id].add_(10)

    baseline_rows = (
        build_qwen38_tp4_state_component_digests(
            source_transaction=source_rank_zero,
            leases=(lease,),
            global_rank=0,
        )
        + build_qwen38_tp4_state_component_digests(
            source_transaction=source_rank_one,
            leases=(lease,),
            global_rank=1,
        )
    )
    candidate_rows = owner.correctness_state_component_digests(
        (lease,)
    )

    def indexed(rows):
        return {
            (row["layer_index"], row["source_rank"]): {
                key: row[key]
                for key in (
                    "convolution_query_sha256",
                    "convolution_key_sha256",
                    "convolution_value_sha256",
                    "recurrent_sha256",
                )
            }
            for row in rows
        }

    assert len(baseline_rows) == 96
    assert len(candidate_rows) == 96
    assert indexed(candidate_rows) == indexed(baseline_rows)
    assert {
        row["logical_rank"] for row in candidate_rows
    } == {0}
    assert {
        row["source_rank"] for row in candidate_rows
    } == {0, 1}


@pytest.mark.parametrize(
    "source",
    (
        lambda: build_source_transaction(
            config=frozen_qwen38_config(
                layer_types=(
                    ("linear_attention",) * 47
                    + ("full_attention",) * 17
                ),
            ),
        ),
        lambda: build_source_transaction(tensor_parallel_size=2),
    ),
)
def test_builder_rejects_wrong_layer_inventory_or_source_shape(source):
    with pytest.raises(ValueError, match="source TP4 state layout"):
        build_owner(source_transaction=source())


def test_rejects_stale_generation_and_duplicate_migration():
    source = build_source_transaction()
    lease = HybridStateLease(0, 2, 103)
    activate_source(source, lease)
    owner = build_owner(source_transaction=source)

    with pytest.raises(RuntimeError, match="lease mismatch"):
        owner.migrate((HybridStateLease(0, 1, 103),))

    owner.migrate((lease,))
    with pytest.raises(RuntimeError, match="already migrated"):
        owner.migrate((lease,))


def test_generation_reuse_is_rejected_before_collective_or_decode_access():
    source = build_source_transaction()
    first = HybridStateLease(0, 1, 108)
    activate_source(source, first)
    collective = FakeWorldAllGather()
    owner = build_owner(
        source_transaction=source,
        all_gather=collective,
    )
    owner.migrate((first,))
    calls_after_first_migration = collective.calls

    source.pool.release(first)
    second = HybridStateLease(0, 2, 109)
    activate_source(source, second)

    with pytest.raises(RuntimeError, match="generation-sealed"):
        owner.migrate((second,))
    assert collective.calls == calls_after_first_migration
    with pytest.raises(RuntimeError, match="lease mismatch"):
        owner.gather((first,))


def test_partial_collective_failure_rolls_back_to_tp4_authority():
    source = build_source_transaction()
    lease = HybridStateLease(0, 1, 104)
    activate_source(source, lease)
    before = source.gather((lease,))
    owner = build_owner(
        source_transaction=source,
        all_gather=FakeWorldAllGather(fail_at=7),
    )

    with pytest.raises(RuntimeError, match="injected collective failure"):
        owner.migrate((lease,))

    after = source.gather((lease,))
    for before_pair, after_pair in zip(before, after):
        assert torch.equal(before_pair[0], after_pair[0])
        assert torch.equal(before_pair[1], after_pair[1])
    assert owner.phase_for(lease) == "tp4_prefill"
    with pytest.raises(RuntimeError, match="lease mismatch"):
        owner.destination_pool.validate(lease)
    snapshot = owner.snapshot()
    assert snapshot["publication_count"] == 0
    assert snapshot["rollback_count"] == 1
    assert snapshot["temporary_live_tensors"] == 0


def test_commit_requires_publication_then_release_seals_generation():
    source = build_source_transaction()
    lease = HybridStateLease(0, 1, 105)
    activate_source(source, lease)
    owner = build_owner(source_transaction=source)

    with pytest.raises(RuntimeError, match="not published"):
        owner.commit((lease,), ())

    owner.migrate((lease,))
    candidates = tuple({
        **row,
        "convolution_states": row["convolution_states"] + 1,
        "recurrent_states": row["recurrent_states"] + 1,
    } for row in owner.gather((lease,)))
    commit_rows = owner.commit((lease,), candidates)
    assert len(commit_rows) == 48
    assert all(row["committed"] is True for row in commit_rows)
    assert torch.equal(
        owner.gather((lease,))[0]["recurrent_states"],
        candidates[0]["recurrent_states"],
    )

    release_rows = owner.release((lease,))

    assert release_rows == ({
        "request_id": 105,
        "generation": 1,
        "slot_id": 0,
        "released": True,
    },)
    assert owner.phase_for(lease) == "released"
    with pytest.raises(RuntimeError, match="lease mismatch"):
        owner.destination_pool.validate(lease)
    with pytest.raises(RuntimeError, match="released"):
        owner.gather((lease,))


def test_multi_lease_publication_is_all_or_nothing():
    source = build_source_transaction()
    leases = (
        HybridStateLease(0, 1, 106),
        HybridStateLease(1, 1, 107),
    )
    activate_source(source, *leases)
    owner = build_owner(
        source_transaction=source,
        all_gather=FakeWorldAllGather(fail_at=100),
    )

    with pytest.raises(RuntimeError, match="injected collective failure"):
        owner.migrate(leases)

    assert all(owner.phase_for(lease) == "tp4_prefill" for lease in leases)
    assert owner.snapshot()["publication_count"] == 0
    assert owner.snapshot()["rollback_count"] == 1
