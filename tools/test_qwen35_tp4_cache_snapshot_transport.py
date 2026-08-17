from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_method(relative_path, class_name, method_name):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == method_name
        ),
        None,
    )
    assert method_node is not None, (
        f"{class_name}.{method_name} is missing"
    )
    function = ast.FunctionDef(
        name=method_node.name,
        args=method_node.args,
        body=method_node.body,
        decorator_list=[],
        returns=method_node.returns,
        type_comment=method_node.type_comment,
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def test_model_runner_returns_rank_local_registered_fields_only():
    snapshot = _load_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "qwen35_hybrid_prefix_cache_snapshot",
    )

    class Cache:
        def observation_snapshot(self):
            return {
                "current_entries": 2,
                "current_bytes": 100,
                "current_logical_bytes": 150,
                "deduplicated_bytes": 50,
                "peak_entries": 3,
                "peak_bytes": 200,
                "publishes": 4,
                "hits": 5,
                "misses": 1,
                "evictions": 0,
                "validation_failures": 0,
                "failed_restores": 0,
                "current_interned_tensors": 4,
            }

    runner = SimpleNamespace(
        rank=2,
        qwen35_hybrid_prefix_restore_owner=SimpleNamespace(
            snapshot_cache=Cache(),
            representation="int8",
            representation_version="v1",
            codec="symmetric-per-token",
        ),
    )
    result = snapshot(runner)

    assert result == {
        "rank": 2,
        "representation": "int8",
        "representation_version": "v1",
        "codec": "symmetric-per-token",
        "current_entries": 2,
        "current_bytes": 100,
        "current_logical_bytes": 150,
        "deduplicated_bytes": 50,
        "peak_entries": 3,
        "peak_bytes": 200,
        "publishes": 4,
        "hits": 5,
        "misses": 1,
        "evictions": 0,
        "validation_failures": 0,
        "failed_restores": 0,
        "current_encoded_physical_bytes": 0,
        "current_encoded_logical_bytes": 0,
        "current_full_fidelity_logical_bytes": 0,
        "current_codec_metadata_bytes": 0,
        "current_reader_leases": 0,
        "current_temporary_encode_workspace_bytes": 0,
        "current_temporary_decode_workspace_bytes": 0,
        "current_temporary_decode_cuda_allocated_bytes": 0,
        "current_temporary_decode_cuda_reserved_bytes": 0,
        "peak_encoded_logical_bytes": 0,
        "peak_full_fidelity_logical_bytes": 0,
        "peak_codec_metadata_bytes": 0,
        "peak_reader_leases": 0,
        "peak_temporary_encode_workspace_bytes": 0,
        "peak_temporary_decode_workspace_bytes": 0,
        "peak_temporary_decode_cuda_allocated_bytes": 0,
        "peak_temporary_decode_cuda_reserved_bytes": 0,
        "deferred_snapshot_releases": 0,
        "quarantines": 0,
        "decode_failures": 0,
        "commit_failures": 0,
        "rollback_failures": 0,
        "fallbacks": 0,
        "partial_restore_attempts": 0,
        "mixed_representation_rejections": 0,
        "missing_layer_rejections": 0,
    }


def test_model_runner_rejects_missing_restore_owner():
    snapshot = _load_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "qwen35_hybrid_prefix_cache_snapshot",
    )
    try:
        snapshot(SimpleNamespace(
            rank=0,
            qwen35_hybrid_prefix_restore_owner=None,
        ))
    except RuntimeError as error:
        assert "restore owner" in str(error)
    else:
        raise AssertionError("missing restore owner was accepted")


def test_model_runner_authority_snapshot_includes_lifecycle_and_blocks():
    snapshot = _load_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "qwen35_hybrid_prefix_authority_snapshot",
    )
    identities = ((7, 2, 99),)

    class Cache:
        def observation_snapshot(self):
            return {
                "current_entries": 1,
                "hits": 2,
                "misses": 3,
                "publication_commits": 4,
                "invalidations": 5,
                "clears": 6,
            }

    runner = SimpleNamespace(
        rank=2,
        qwen35_hybrid_prefix_restore_owner=SimpleNamespace(
            snapshot_cache=Cache(),
            publication_participant=SimpleNamespace(
                _terminal_payloads={
                    3: SimpleNamespace(block_identities=identities)
                }
            ),
        ),
    )
    assert snapshot(runner) == {
        "rank": 2,
        "current_entries": 1,
        "hits": 2,
        "misses": 3,
        "publication_commits": 4,
        "invalidations": 5,
        "clears": 6,
        "last_publication_block_identities": [[7, 2, 99]],
    }


def test_engine_collects_exact_contiguous_rank_inventory():
    snapshot = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "qwen35_hybrid_prefix_cache_snapshots",
    )
    local = {"rank": 0}
    acknowledgements = tuple(
        SimpleNamespace(rank=rank, result={"rank": rank})
        for rank in (1, 2, 3)
    )
    engine = SimpleNamespace(
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            local,
            acknowledgements,
        ),
        model_runner=SimpleNamespace(world_size=4),
    )

    assert snapshot(engine, timeout_s=12.0) == (
        {"rank": 0},
        {"rank": 1},
        {"rank": 2},
        {"rank": 3},
    )


def test_engine_rejects_ack_rank_payload_mismatch():
    snapshot = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "qwen35_hybrid_prefix_cache_snapshots",
    )
    engine = SimpleNamespace(
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            {"rank": 0},
            (
                SimpleNamespace(rank=1, result={"rank": 2}),
                SimpleNamespace(rank=2, result={"rank": 1}),
                SimpleNamespace(rank=3, result={"rank": 3}),
            ),
        ),
        model_runner=SimpleNamespace(world_size=4),
    )
    try:
        snapshot(engine, timeout_s=12.0)
    except ValueError as error:
        assert "rank" in str(error)
    else:
        raise AssertionError("rank payload mismatch was accepted")


def test_engine_collects_authority_snapshot_with_rank_parity():
    snapshot = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "qwen35_hybrid_prefix_authority_snapshots",
    )
    identities = [[7, 2, 99]]
    local = {
        "rank": 0,
        "current_entries": 1,
        "hits": 2,
        "misses": 3,
        "publication_commits": 4,
        "invalidations": 5,
        "clears": 6,
        "last_publication_block_identities": identities,
    }
    acknowledgements = tuple(
        SimpleNamespace(rank=rank, result={**local, "rank": rank})
        for rank in (1, 2, 3)
    )
    engine = SimpleNamespace(
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            local,
            acknowledgements,
        ),
        model_runner=SimpleNamespace(world_size=4),
    )
    rows = snapshot(engine, timeout_s=12.0)
    assert [row["rank"] for row in rows] == [0, 1, 2, 3]
    assert all(
        row["last_publication_block_identities"] == identities
        for row in rows
    )


def test_engine_collects_exact_contiguous_memory_rank_inventory():
    snapshots = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "memory_snapshots",
    )
    local = {"rank": 0, "cuda_allocated_bytes": 1}
    acknowledgements = tuple(
        SimpleNamespace(
            rank=rank,
            result={"rank": rank, "cuda_allocated_bytes": rank + 1},
        )
        for rank in (1, 2, 3)
    )
    engine = SimpleNamespace(
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            local,
            acknowledgements,
        ),
        model_runner=SimpleNamespace(world_size=4),
    )

    assert snapshots(engine, timeout_s=12.0) == (
        local,
        acknowledgements[0].result,
        acknowledgements[1].result,
        acknowledgements[2].result,
    )


def test_model_runner_clears_rank_local_hybrid_prefix_cache():
    clear = _load_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "clear_qwen35_hybrid_prefix_cache",
    )

    class Cache:
        def clear(self):
            return 3

    runner = SimpleNamespace(
        rank=2,
        qwen35_hybrid_prefix_restore_owner=SimpleNamespace(
            snapshot_cache=Cache()
        ),
    )
    assert clear(runner) == {"rank": 2, "cleared_entries": 3}


def test_model_runner_invalidates_rank_local_hybrid_prefix_blocks():
    invalidate = _load_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        "invalidate_qwen35_hybrid_prefix_blocks",
    )
    identities = ((7, 2, 99),)

    class Cache:
        def invalidate_blocks(self, block_identities):
            assert block_identities == identities
            return 1

    runner = SimpleNamespace(
        rank=1,
        qwen35_hybrid_prefix_restore_owner=SimpleNamespace(
            snapshot_cache=Cache()
        ),
    )
    assert invalidate(runner, identities) == {
        "rank": 1,
        "invalidated_entries": 1,
    }


def test_engine_collects_all_rank_clear_with_count_parity():
    collect = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "_collect_qwen35_hybrid_prefix_cache_mutation",
    )
    clear = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "clear_qwen35_hybrid_prefix_caches",
    )
    acknowledgements = tuple(
        SimpleNamespace(
            rank=rank,
            result={"rank": rank, "cleared_entries": 2},
        )
        for rank in (1, 2, 3)
    )
    calls = []

    def acknowledged(*args, **kwargs):
        calls.append((args, kwargs))
        return {"rank": 0, "cleared_entries": 2}, acknowledgements

    engine = SimpleNamespace(
        call_model_runner_acknowledged=acknowledged,
        model_runner=SimpleNamespace(world_size=4),
    )
    engine._collect_qwen35_hybrid_prefix_cache_mutation = MethodType(
        collect,
        engine,
    )
    assert clear(engine, timeout_s=12.0) == (
        {"rank": 0, "cleared_entries": 2},
        {"rank": 1, "cleared_entries": 2},
        {"rank": 2, "cleared_entries": 2},
        {"rank": 3, "cleared_entries": 2},
    )
    assert calls == [
        (("clear_qwen35_hybrid_prefix_cache",), {"timeout_s": 12.0})
    ]


def test_engine_rejects_all_rank_invalidation_count_mismatch():
    collect = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "_collect_qwen35_hybrid_prefix_cache_mutation",
    )
    invalidate = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "invalidate_qwen35_hybrid_prefix_blocks",
    )
    identities = ((7, 2, 99),)
    engine = SimpleNamespace(
        call_model_runner_acknowledged=lambda *args, **kwargs: (
            {"rank": 0, "invalidated_entries": 1},
            (
                SimpleNamespace(
                    rank=1,
                    result={"rank": 1, "invalidated_entries": 1},
                ),
                SimpleNamespace(
                    rank=2,
                    result={"rank": 2, "invalidated_entries": 0},
                ),
                SimpleNamespace(
                    rank=3,
                    result={"rank": 3, "invalidated_entries": 1},
                ),
            ),
        ),
        model_runner=SimpleNamespace(world_size=4),
    )
    engine._collect_qwen35_hybrid_prefix_cache_mutation = MethodType(
        collect,
        engine,
    )
    try:
        invalidate(engine, identities, timeout_s=12.0)
    except ValueError as error:
        assert "parity" in str(error)
    else:
        raise AssertionError("rank-local invalidation mismatch accepted")


def test_engine_canonicalizes_authority_blocks_before_invalidation_dispatch():
    collect = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "_collect_qwen35_hybrid_prefix_cache_mutation",
    )
    invalidate = _load_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "invalidate_qwen35_hybrid_prefix_blocks",
    )
    calls = []

    def acknowledged(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            {"rank": 0, "invalidated_entries": 1},
            tuple(
                SimpleNamespace(
                    rank=rank,
                    result={
                        "rank": rank,
                        "invalidated_entries": 1,
                    },
                )
                for rank in (1, 2, 3)
            ),
        )

    engine = SimpleNamespace(
        call_model_runner_acknowledged=acknowledged,
        model_runner=SimpleNamespace(world_size=4),
    )
    engine._collect_qwen35_hybrid_prefix_cache_mutation = MethodType(
        collect,
        engine,
    )

    assert invalidate(
        engine,
        [[7, 2, 99]],
        timeout_s=12.0,
    ) == (
        {"rank": 0, "invalidated_entries": 1},
        {"rank": 1, "invalidated_entries": 1},
        {"rank": 2, "invalidated_entries": 1},
        {"rank": 3, "invalidated_entries": 1},
    )
    assert calls == [
        (
            (
                "invalidate_qwen35_hybrid_prefix_blocks",
                ((7, 2, 99),),
            ),
            {"timeout_s": 12.0},
        )
    ]


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cache snapshot transport tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
