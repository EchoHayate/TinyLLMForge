"""Dependency-light tests for exact CUDA Graph phase-reset wiring."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
LLM_ENGINE_PATH = ROOT / "tinyvllm" / "engine" / "llm_engine.py"


class _Ack:
    def __init__(self, rank, result):
        self.rank = rank
        self.result = result


def _zero_summary():
    return {
        "ready_entries": [],
        "rejected": {},
        "capturing": [],
        "observation_counts": {},
        "static_bytes": 0,
        "reserved_delta_bytes": 0,
        "total_capture_ns": 0,
        "cross_lease_replays": 0,
        "lease_manifest_rejections": 0,
        "unique_invocation_identities": 0,
        "unique_program_keys": 0,
        "hits": 0,
        "misses": 0,
        "capture_attempts": 0,
        "capture_successes": 0,
        "capture_failures": 0,
    }


def _receipt(rank, **overrides):
    row = {
        "rank": rank,
        "released_ready_entries": 1,
        "cleared_observations": 2,
        "cleared_rejections": 1,
        "summary": _zero_summary(),
    }
    row.update(overrides)
    return row


def _load_reset_method():
    tree = ast.parse(
        LLM_ENGINE_PATH.read_text(encoding="utf-8"),
        filename=str(LLM_ENGINE_PATH),
    )
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LLMEngine"
    )
    method = copy.deepcopy(next(
        node
        for node in engine_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "reset_exact_cuda_graph_cache"
    ))
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(
        compile(module, str(LLM_ENGINE_PATH), "exec"),
        namespace,
    )
    return namespace["reset_exact_cuda_graph_cache"]


class _Engine:
    def __init__(self, rows, *, world_size=None):
        self.rows = tuple(rows)
        self.model_runner = SimpleNamespace(
            world_size=(
                len(self.rows)
                if world_size is None
                else int(world_size)
            )
        )
        self.calls = []

    def call_model_runner_acknowledged(
        self,
        method_name,
        *args,
        timeout_s,
    ):
        self.calls.append((method_name, args, float(timeout_s)))
        return self.rows[0], tuple(
            _Ack(rank, self.rows[rank])
            for rank in range(1, len(self.rows))
        )


def _reset(engine, timeout_s=5.0):
    return _load_reset_method()(engine, timeout_s=timeout_s)


def _expect_error(callback, message):
    try:
        callback()
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(
            f"expected RuntimeError containing {message!r}"
        )


def test_engine_returns_ordered_agreeing_graph_reset_receipts():
    rows = tuple(_receipt(rank) for rank in range(4))
    engine = _Engine(rows)

    result = _reset(engine)

    assert result == rows
    assert engine.calls == [
        ("reset_exact_cuda_graph_cache", (), 5.0),
    ]


def test_engine_rejects_incomplete_or_misranked_graph_reset_receipts():
    incomplete = _Engine(
        tuple(_receipt(rank) for rank in range(3)),
        world_size=4,
    )
    _expect_error(
        lambda: _reset(incomplete),
        "ranks are incomplete",
    )

    misranked_rows = list(_receipt(rank) for rank in range(4))
    misranked_rows[2]["rank"] = 3
    _expect_error(
        lambda: _reset(_Engine(misranked_rows)),
        "acknowledgement is invalid",
    )


def test_engine_rejects_disagreeing_graph_reset_receipts():
    rows = list(_receipt(rank) for rank in range(4))
    rows[3]["cleared_observations"] = 3

    _expect_error(
        lambda: _reset(_Engine(rows)),
        "ranks disagree",
    )


def test_engine_rejects_nonzero_post_reset_graph_state():
    mutations = (
        {"ready_entries": ["a" * 64]},
        {"rejected": {"b" * 64: "capture_failed"}},
        {"capturing": ["c" * 64]},
        {"observation_counts": {"d" * 64: 1}},
        {"static_bytes": 1},
        {"reserved_delta_bytes": 1},
        {"total_capture_ns": 1},
        {"capture_attempts": 1},
    )
    for mutation in mutations:
        rows = list(_receipt(rank) for rank in range(4))
        for row in rows:
            summary = _zero_summary()
            summary.update(mutation)
            row["summary"] = summary
        _expect_error(
            lambda rows=rows: _reset(_Engine(rows)),
            "post-reset summary is not empty",
        )


def main():
    tests = (
        test_engine_returns_ordered_agreeing_graph_reset_receipts,
        test_engine_rejects_incomplete_or_misranked_graph_reset_receipts,
        test_engine_rejects_disagreeing_graph_reset_receipts,
        test_engine_rejects_nonzero_post_reset_graph_state,
    )
    for test in tests:
        test()
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
