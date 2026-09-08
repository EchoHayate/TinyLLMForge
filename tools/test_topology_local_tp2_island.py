import dataclasses
from pathlib import Path
import sys
import types


import pytest


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


class FakeTensor:
    def __init__(self, rows, *, dtype="float32", device="cpu"):
        self.rows = [list(row) for row in rows]
        self.dtype = dtype
        self.device = device
        self.shape = (
            len(self.rows),
            0 if not self.rows else len(self.rows[0]),
        )

    def clone(self):
        return FakeTensor(
            self.rows,
            dtype=self.dtype,
            device=self.device,
        )

    def zero_(self):
        self.rows = [
            [0 for _ in row]
            for row in self.rows
        ]
        return self


torch = types.ModuleType("torch")
torch.Tensor = FakeTensor
torch.float32 = "float32"
torch.float64 = "float64"
torch.cat = lambda tensors, dim=0: FakeTensor(
    [
        row
        for tensor in tensors
        for row in tensor.rows
    ],
    dtype=tensors[0].dtype,
    device=tensors[0].device,
)
torch.equal = lambda left, right: (
    left.rows == right.rows
    and left.dtype == right.dtype
    and left.device == right.device
)
sys.modules.setdefault("torch", torch)


from tinyvllm.engine.topology_local_tp2_island import (
    TopologyLocalTP2PairMap,
    TopologyLocalTP2StateIdentity,
    assemble_logical_state_half,
    logical_half_bounds,
    select_best_pair_groups,
    validate_state_publication,
)


def topology_rows(overrides=None):
    links = {
        (0, 1): "PXB",
        (0, 2): "SYS",
        (0, 3): "SYS",
        (1, 2): "SYS",
        (1, 3): "SYS",
        (2, 3): "PIX",
    }
    links.update(overrides or {})
    rows = []
    for (left, right), link in links.items():
        rows.extend((
            {
                "left_rank": left,
                "right_rank": right,
                "link": link,
            },
            {
                "left_rank": right,
                "right_rank": left,
                "link": link,
            },
        ))
    return rows


def test_pair_map_assigns_two_replicas_and_logical_ranks():
    mapping = TopologyLocalTP2PairMap(((0, 1), (2, 3)))

    assert mapping.identity(0).pair_id == 0
    assert mapping.identity(0).logical_rank == 0
    assert mapping.identity(0).pair_ranks == (0, 1)
    assert mapping.identity(1).logical_rank == 1
    assert mapping.identity(2).pair_id == 1
    assert mapping.identity(2).logical_rank == 0
    assert mapping.identity(3).logical_rank == 1


@pytest.mark.parametrize(
    "pair_groups",
    (
        ((0, 1), (1, 3)),
        ((0, 1), (2, 4)),
        ((0, 1),),
        ((0, 1, 2), (3,)),
        ((False, 1), (2, 3)),
    ),
)
def test_pair_map_rejects_invalid_partitions(pair_groups):
    with pytest.raises(ValueError):
        TopologyLocalTP2PairMap(pair_groups)


def test_pair_map_rejects_unknown_identity_rank():
    mapping = TopologyLocalTP2PairMap(((0, 1), (2, 3)))

    with pytest.raises(ValueError, match="global_rank"):
        mapping.identity(4)


def test_logical_half_bounds_are_contiguous_and_complete():
    assert logical_half_bounds(6144, 0) == (0, 3072)
    assert logical_half_bounds(6144, 1) == (3072, 3072)


@pytest.mark.parametrize(
    ("total_width", "logical_rank"),
    (
        (0, 0),
        (-2, 0),
        (7, 0),
        (8, -1),
        (8, 2),
        (True, 0),
        (8, False),
    ),
)
def test_logical_half_bounds_reject_invalid_values(
    total_width,
    logical_rank,
):
    with pytest.raises(ValueError):
        logical_half_bounds(total_width, logical_rank)


def test_topology_selection_avoids_system_links():
    assert select_best_pair_groups(topology_rows()) == (
        (0, 1),
        (2, 3),
    )


def test_topology_selection_uses_deterministic_matching_tie_break():
    assert select_best_pair_groups(topology_rows({
        (0, 1): "PIX",
        (0, 2): "PIX",
        (0, 3): "PIX",
        (1, 2): "PIX",
        (1, 3): "PIX",
        (2, 3): "PIX",
    })) == ((0, 1), (2, 3))


def test_topology_selection_rejects_missing_reverse_row():
    rows = topology_rows()
    rows.remove({
        "left_rank": 1,
        "right_rank": 0,
        "link": "PXB",
    })

    with pytest.raises(ValueError, match="incomplete"):
        select_best_pair_groups(rows)


def test_topology_selection_rejects_asymmetric_link():
    rows = topology_rows()
    for row in rows:
        if row["left_rank"] == 1 and row["right_rank"] == 0:
            row["link"] = "SYS"

    with pytest.raises(ValueError, match="asymmetric"):
        select_best_pair_groups(rows)


def test_topology_selection_rejects_duplicate_and_unknown_links():
    with pytest.raises(ValueError, match="duplicated"):
        select_best_pair_groups(topology_rows() + [topology_rows()[0]])

    rows = topology_rows()
    rows[0]["link"] = "NV8"
    with pytest.raises(ValueError, match="invalid"):
        select_best_pair_groups(rows)


def test_assemble_logical_state_half_uses_global_head_order():
    quarters = tuple(
        FakeTensor([[rank, rank]], dtype=torch.float32)
        for rank in range(4)
    )

    assert torch.equal(
        assemble_logical_state_half(quarters, 0),
        FakeTensor([[0, 0], [1, 1]], dtype=torch.float32),
    )
    assert torch.equal(
        assemble_logical_state_half(quarters, 1),
        FakeTensor([[2, 2], [3, 3]], dtype=torch.float32),
    )


def test_state_assembly_does_not_mutate_source_quarters():
    quarters = tuple(
        FakeTensor(
            [[rank, rank + 1], [rank + 2, rank + 3]],
            dtype=torch.float32,
        )
        for rank in range(4)
    )
    snapshots = tuple(tensor.clone() for tensor in quarters)

    result = assemble_logical_state_half(quarters, 0)
    result.zero_()

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(quarters, snapshots)
    )


@pytest.mark.parametrize(
    "quarters",
    (
        tuple(FakeTensor([[0, 0]]) for _ in range(3)),
        (
            FakeTensor([[0, 0]]),
            FakeTensor([[0, 0], [0, 0]]),
            FakeTensor([[0, 0]]),
            FakeTensor([[0, 0]]),
        ),
        (
            FakeTensor([[0, 0]], dtype=torch.float32),
            FakeTensor([[0, 0]], dtype=torch.float64),
            FakeTensor([[0, 0]], dtype=torch.float32),
            FakeTensor([[0, 0]], dtype=torch.float32),
        ),
    ),
)
def test_state_assembly_rejects_incompatible_quarters(quarters):
    with pytest.raises(ValueError):
        assemble_logical_state_half(quarters, 0)


def test_state_publication_accepts_exact_identity():
    identity = TopologyLocalTP2StateIdentity(
        request_id=7,
        generation=3,
        slot_id=2,
        layer_index=0,
    )

    validate_state_publication(identity, identity)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("request_id", 8),
        ("generation", 4),
        ("slot_id", 3),
        ("layer_index", 1),
    ),
)
def test_state_publication_rejects_identity_drift(field, value):
    source = TopologyLocalTP2StateIdentity(
        request_id=7,
        generation=3,
        slot_id=2,
        layer_index=0,
    )

    with pytest.raises(RuntimeError, match="identity mismatch"):
        validate_state_publication(
            source,
            dataclasses.replace(source, **{field: value}),
        )


@pytest.mark.parametrize(
    "arguments",
    (
        {"request_id": True, "generation": 0, "slot_id": 0, "layer_index": 0},
        {"request_id": 1, "generation": -1, "slot_id": 0, "layer_index": 0},
        {"request_id": 1, "generation": 0, "slot_id": -1, "layer_index": 0},
        {"request_id": 1, "generation": 0, "slot_id": 0, "layer_index": -1},
    ),
)
def test_state_identity_rejects_invalid_fields(arguments):
    with pytest.raises(ValueError):
        TopologyLocalTP2StateIdentity(**arguments)
