from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_LINK_COST = {
    "PIX": 0,
    "PXB": 1,
    "PHB": 2,
    "NODE": 3,
    "SYS": 4,
}

_PERFECT_MATCHINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)


def _rank(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= 4
    ):
        raise ValueError(f"{name} must be an integer rank in [0, 3]")
    return value


@dataclass(frozen=True)
class TopologyLocalTP2RankIdentity:
    global_rank: int
    pair_id: int
    logical_rank: int
    pair_ranks: tuple[int, int]


@dataclass(frozen=True)
class TopologyLocalTP2PairMap:
    pair_groups: tuple[tuple[int, int], tuple[int, int]]

    def __post_init__(self) -> None:
        try:
            groups = tuple(tuple(group) for group in self.pair_groups)
        except TypeError as error:
            raise ValueError(
                "pair_groups must contain two rank pairs"
            ) from error
        if len(groups) != 2 or any(len(group) != 2 for group in groups):
            raise ValueError("pair_groups must contain two rank pairs")
        flattened = tuple(
            _rank(rank, "pair rank")
            for group in groups
            for rank in group
        )
        if sorted(flattened) != [0, 1, 2, 3]:
            raise ValueError("pair_groups must partition ranks 0..3")
        object.__setattr__(self, "pair_groups", groups)

    def identity(
        self,
        global_rank: int,
    ) -> TopologyLocalTP2RankIdentity:
        global_rank = _rank(global_rank, "global_rank")
        for pair_id, pair_ranks in enumerate(self.pair_groups):
            if global_rank in pair_ranks:
                return TopologyLocalTP2RankIdentity(
                    global_rank=global_rank,
                    pair_id=pair_id,
                    logical_rank=pair_ranks.index(global_rank),
                    pair_ranks=pair_ranks,
                )
        raise ValueError("global_rank is not present in pair_groups")


def logical_half_bounds(
    total_width: int,
    logical_rank: int,
) -> tuple[int, int]:
    if (
        isinstance(total_width, bool)
        or not isinstance(total_width, int)
        or total_width <= 0
        or total_width % 2 != 0
    ):
        raise ValueError(
            "total_width must be a positive even integer"
        )
    if (
        isinstance(logical_rank, bool)
        or not isinstance(logical_rank, int)
        or logical_rank not in (0, 1)
    ):
        raise ValueError("logical_rank must be zero or one")
    width = total_width // 2
    return logical_rank * width, width


def select_best_pair_groups(
    topology_rows: Iterable[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    directed: dict[tuple[int, int], int] = {}
    try:
        rows = tuple(topology_rows)
    except TypeError as error:
        raise ValueError("topology_rows must be iterable") from error
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("topology row must be a dictionary")
        left = _rank(row.get("left_rank"), "left_rank")
        right = _rank(row.get("right_rank"), "right_rank")
        link = row.get("link")
        if left == right or link not in _LINK_COST:
            raise ValueError("topology row is invalid")
        key = (left, right)
        if key in directed:
            raise ValueError("topology row is duplicated")
        directed[key] = _LINK_COST[link]

    undirected: dict[tuple[int, int], int] = {}
    for left in range(4):
        for right in range(left + 1, 4):
            forward = directed.get((left, right))
            reverse = directed.get((right, left))
            if forward is None or reverse is None:
                raise ValueError("topology rows are incomplete")
            if forward != reverse:
                raise ValueError("topology rows are asymmetric")
            undirected[(left, right)] = forward

    return min(
        _PERFECT_MATCHINGS,
        key=lambda matching: (
            tuple(sorted(
                undirected[tuple(sorted(pair))]
                for pair in matching
            )),
            matching,
        ),
    )
