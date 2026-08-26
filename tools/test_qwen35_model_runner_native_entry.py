import ast
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = REPO_ROOT / "tinyvllm" / "engine" / "model_runner.py"
LLM_ENGINE_PATH = REPO_ROOT / "tinyvllm" / "engine" / "llm_engine.py"


def _load_functions(*names, path=MODEL_RUNNER_PATH):
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert set(functions) == set(names)
    module = ast.Module(
        body=[functions[name] for name in names],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def test_qwen35_model_initialization_uses_native_loader_only():
    initialize_model, = _load_functions("_initialize_model_runner_model")
    calls = []
    native_model = object()
    native_owner = SimpleNamespace(model=native_model)

    model, owner, partition_identity = initialize_model(
        SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5"),
        ),
        rank=2,
        load_legacy_model=lambda _config: calls.append("legacy"),
        load_qwen35_model=lambda _config, rank: (
            calls.append(("qwen35", rank))
            or (native_model, native_owner, {"rank": rank})
        ),
    )

    assert model is native_model
    assert owner is native_owner
    assert partition_identity == {"rank": 2}
    assert calls == [("qwen35", 2)]


def test_legacy_model_initialization_preserves_existing_loader():
    initialize_model, = _load_functions("_initialize_model_runner_model")
    calls = []
    legacy_model = object()

    model, owner, partition_identity = initialize_model(
        SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3"),
        ),
        rank=0,
        load_legacy_model=lambda _config: (
            calls.append("legacy") or legacy_model
        ),
        load_qwen35_model=lambda _config, rank: calls.append(
            ("qwen35", rank)
        ),
    )

    assert model is legacy_model
    assert owner is None
    assert partition_identity is None
    assert calls == ["legacy"]


def test_packed_eager_dispatch_uses_active_leases_and_token_counts():
    run_eager, = _load_functions("_run_model_runner_eager")
    calls = []
    logits = object()
    hidden = object()

    class PackedModel:
        def run_step(
            self,
            leases,
            token_counts,
            input_ids,
            positions,
            input_embeds=None,
        ):
            calls.append(
                (
                    leases,
                    token_counts,
                    input_ids,
                    positions,
                    input_embeds,
                )
            )
            return hidden, logits

    result = run_eager(
        PackedModel(),
        input_ids="ids",
        positions="positions",
        input_embeds="embeds",
        active_leases=("lease-0", "lease-1"),
        token_counts=(3, 1),
        return_hidden=True,
    )

    assert result == (logits, hidden)
    assert calls == [
        (
            ("lease-0", "lease-1"),
            (3, 1),
            "ids",
            "positions",
            "embeds",
        )
    ]


def test_packed_eager_dispatch_can_return_uncommitted_prepared_step():
    run_eager, = _load_functions("_run_model_runner_eager")
    calls = []
    prepared = object()

    class PackedModel:
        def prepare_step(
            self,
            leases,
            token_counts,
            input_ids,
            positions,
            input_embeds=None,
            *,
            initial_candidates=None,
            capture_prefix_states=False,
        ):
            calls.append((
                leases,
                token_counts,
                input_ids,
                positions,
                input_embeds,
                initial_candidates,
                capture_prefix_states,
            ))
            return prepared

        def run_step(self, *_args, **_kwargs):
            raise AssertionError("prepared dispatch must not commit")

    result = run_eager(
        PackedModel(),
        input_ids="ids",
        positions="positions",
        input_embeds="embeds",
        active_leases=("lease-0",),
        token_counts=(3,),
        return_hidden=False,
        prepare_qwen35_state=True,
        initial_qwen35_candidates="candidate-state",
        capture_qwen35_prefix_states=True,
    )

    assert result is prepared
    assert calls == [(
        ("lease-0",),
        (3,),
        "ids",
        "positions",
        "embeds",
        "candidate-state",
        True,
    )]


def test_qwen35_step_token_counts_follow_prefill_decode_and_mixed_rows():
    token_counts, = _load_functions("_qwen35_step_token_counts")
    prefill = SimpleNamespace(
        prefill_chunk_start=2,
        prefill_chunk_end=7,
        num_cached_tokens=2,
        step_is_decode=False,
        __len__=lambda self: 7,
    )
    decode = SimpleNamespace(
        step_is_decode=True,
    )

    assert token_counts(
        [prefill],
        is_prefill=True,
        batch_kind=None,
    ) == (5,)
    assert token_counts(
        [decode, decode],
        is_prefill=False,
        batch_kind=None,
    ) == (1, 1)
    assert token_counts(
        [prefill, decode],
        is_prefill=True,
        batch_kind="mixed",
    ) == (5, 1)


def test_completed_prefill_and_decode_round_recurrent_pool_rows():
    round_final_prefill, = _load_functions(
        "_round_qwen35_final_prefill_recurrent_states"
    )
    convolution = torch.zeros(
        (3, 2, 2),
        dtype=torch.bfloat16,
    )
    recurrent = (
        torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        * 0.0011
    )
    original = recurrent.clone()
    pool = SimpleNamespace(
        layout=SimpleNamespace(
            components=(
                SimpleNamespace(
                    layer_index=0,
                    role="linear_convolution",
                    dtype=torch.bfloat16,
                ),
                SimpleNamespace(
                    layer_index=0,
                    role="linear_recurrent",
                    dtype=torch.float32,
                ),
            ),
        ),
        validate=lambda lease: lease.slot_id,
        component_tensor=lambda _layer, role: (
            convolution if role == "linear_convolution" else recurrent
        ),
    )
    runtime_bridge = SimpleNamespace(pool=pool)
    leases = (
        SimpleNamespace(slot_id=0, request_id=10),
        SimpleNamespace(slot_id=1, request_id=11),
        SimpleNamespace(slot_id=2, request_id=12),
    )
    final_prefill = SimpleNamespace(
        seq_id=10,
        prefill_chunk_start=4,
        prefill_chunk_end=8,
        step_is_decode=False,
        token_ids=tuple(range(8)),
    )
    partial_prefill = SimpleNamespace(
        seq_id=11,
        prefill_chunk_start=0,
        prefill_chunk_end=4,
        step_is_decode=False,
        token_ids=tuple(range(8)),
    )
    decode = SimpleNamespace(
        seq_id=12,
        step_is_decode=True,
        token_ids=tuple(range(8)),
    )

    round_final_prefill(
        runtime_bridge,
        (final_prefill, partial_prefill, decode),
        leases,
        is_prefill=True,
        batch_kind="mixed",
    )

    torch.testing.assert_close(
        recurrent[0],
        original[0].to(torch.bfloat16).to(torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        recurrent[1],
        original[1],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        recurrent[2],
        original[2].to(torch.bfloat16).to(torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        convolution,
        torch.zeros_like(convolution),
        rtol=0.0,
        atol=0.0,
    )

    recurrent[2].copy_(original[2] + 0.00037)
    before_decode = recurrent.clone()
    round_final_prefill(
        runtime_bridge,
        (decode,),
        (leases[2],),
        is_prefill=False,
        batch_kind=None,
    )
    torch.testing.assert_close(
        recurrent[:2],
        before_decode[:2],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        recurrent[2],
        before_decode[2].to(torch.bfloat16).to(torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_qwen35_local_kv_heads_replicate_when_tp_exceeds_heads():
    local_kv_heads, = _load_functions("_local_model_kv_heads")

    assert local_kv_heads(8, 4) == 2
    assert local_kv_heads(2, 4) == 1


def test_model_runner_shared_memory_name_is_dist_group_scoped():
    shared_memory_name, = _load_functions(
        "_model_runner_shared_memory_name"
    )

    assert shared_memory_name("61371") == "tinyvllm-61371"
    assert shared_memory_name(61371) == "tinyvllm-61371"
    assert shared_memory_name("61373") == "tinyvllm-61373"


def test_engine_prefix_restore_admission_builds_exact_key_and_marks_hit():
    restore, = _load_functions(
        "_try_qwen35_hybrid_prefix_restore",
        path=LLM_ENGINE_PATH,
    )
    calls = []

    class Key:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    block_manager = SimpleNamespace(
        block_size=4,
        max_reusable_tokens=lambda _seq: 8,
        compute_hash=lambda tokens, previous: (
            previous * 100 + sum(tokens)
        ),
    )
    identity = SimpleNamespace(
        model_fingerprint="a" * 64,
        layout_fingerprint="layout-a",
        dtype="bfloat16",
    )
    engine = SimpleNamespace(
        qwen35_hybrid_prefix_engine_restore_coordinator=SimpleNamespace(
            timeout_s=0.5,
        ),
        qwen35_hybrid_prefix_runtime_identity=identity,
        scheduler=SimpleNamespace(block_manager=block_manager),
        model_runner=SimpleNamespace(world_size=4),
        flush_pending_hybrid_state_releases=lambda **_kwargs: (),
        acquire_qwen35_hybrid_prefix=lambda seq, key, tokens: (
            calls.append((seq, key, tokens)) or True
        ),
    )
    sequence = SimpleNamespace(
        token_ids=list(range(1, 11)),
    )

    restored = restore(
        engine,
        sequence,
        key_type=Key,
    )

    assert restored is True
    assert sequence.hybrid_prefix_restore_attempted is True
    assert sequence.hybrid_prefix_restore_hit is True
    assert len(calls) == 1
    _, key, tokens = calls[0]
    assert tokens == tuple(range(1, 9))
    assert key.token_count == 8
    assert key.token_hash == key.terminal_block_hash
    assert key.block_size == 4
    assert key.model_fingerprint == "a" * 64
    assert key.layout_fingerprint == "layout-a"
    assert key.tensor_parallel_size == 4
    assert key.dtype == "bfloat16"


def test_engine_prefix_restore_admission_marks_miss_and_skips_unconfigured():
    restore, = _load_functions(
        "_try_qwen35_hybrid_prefix_restore",
        path=LLM_ENGINE_PATH,
    )
    sequence = SimpleNamespace(token_ids=[1, 2, 3])
    unconfigured = SimpleNamespace(
        qwen35_hybrid_prefix_engine_restore_coordinator=None,
        qwen35_hybrid_prefix_runtime_identity=None,
    )

    assert restore(
        unconfigured,
        sequence,
        key_type=object,
    ) is None
    assert not hasattr(
        sequence,
        "hybrid_prefix_restore_attempted",
    )

    block_manager = SimpleNamespace(
        block_size=2,
        max_reusable_tokens=lambda _seq: 2,
        compute_hash=lambda tokens, previous: sum(tokens),
    )
    configured = SimpleNamespace(
        qwen35_hybrid_prefix_engine_restore_coordinator=SimpleNamespace(
            timeout_s=0.5,
        ),
        qwen35_hybrid_prefix_runtime_identity=SimpleNamespace(
            model_fingerprint="a" * 64,
            layout_fingerprint="layout-a",
            dtype="bfloat16",
        ),
        scheduler=SimpleNamespace(block_manager=block_manager),
        model_runner=SimpleNamespace(world_size=4),
        flush_pending_hybrid_state_releases=lambda **_kwargs: (),
        acquire_qwen35_hybrid_prefix=lambda *_args: False,
    )

    assert restore(
        configured,
        sequence,
        key_type=lambda **kwargs: SimpleNamespace(**kwargs),
    ) is False
    assert sequence.hybrid_prefix_restore_attempted is True
    assert sequence.hybrid_prefix_restore_hit is False
