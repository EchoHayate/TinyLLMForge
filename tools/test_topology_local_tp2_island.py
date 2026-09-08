from pathlib import Path
import sys
import types


import pytest


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


from tinyvllm.engine.topology_local_tp2_island import (
    TopologyLocalTP2PairMap,
    logical_half_bounds,
    select_best_pair_groups,
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
