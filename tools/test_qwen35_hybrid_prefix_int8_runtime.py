from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[1]

for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(self._hash.digest(), "little")


xxhash_module = types.ModuleType("xxhash")
xxhash_module.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_module)

config_module = types.ModuleType("tinyvllm.config")
config_module.Config = object
sys.modules.setdefault("tinyvllm.config", config_module)


from tinyvllm.engine.hybrid_state import (
    HybridStateComponentSpec,
    HybridStateLayout,
    HybridStateLease,
    HybridStateRuntimeBridge,
    HybridStateSlotAllocator,
    HybridStateTensorPool,
)
from tinyvllm.engine.block_manager import BlockManager
from tinyvllm.engine.qwen35_hybrid_prefix_cache import (
    Qwen35HybridPrefixKey,
    Qwen35HybridPrefixSnapshotCache,
)
from tinyvllm.engine.qwen35_hybrid_prefix_engine_restore import (
    Qwen35HybridPrefixEngineRestoreCoordinator,
)
from tinyvllm.engine.qwen35_hybrid_prefix_int8_cache import (
    Qwen35HybridPrefixInt8SnapshotCache,
)
from tinyvllm.engine.qwen35_hybrid_prefix_owner import (
    Qwen35HybridPrefixRestoreOwner,
    build_qwen35_hybrid_prefix_restore_owner,
)
from tinyvllm.engine.qwen35_hybrid_prefix_publication_ticket import (
    Qwen35HybridPrefixPublicationPayload,
    Qwen35HybridPrefixPublicationParticipant,
)
from tinyvllm.engine.qwen35_hybrid_prefix_representation import (
    QWEN35_HYBRID_PREFIX_DEFAULT,
    QWEN35_HYBRID_PREFIX_EXACT,
    QWEN35_HYBRID_PREFIX_EXACT_VERSION,
    QWEN35_HYBRID_PREFIX_INT8_VERSION,
    QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    resolve_qwen35_hybrid_prefix_representation,
)
from tinyvllm.engine.qwen35_hybrid_prefix_restore_ticket import (
    Qwen35HybridPrefixRestoreParticipant,
)
from tinyvllm.engine.qwen35_recurrent_int8_codec import (
    QWEN35_RECURRENT_INT8_CODEC,
)
from tinyvllm.engine.scheduler import Scheduler
from tinyvllm.engine.sequence import Sequence
from tinyvllm.sampling_params import SamplingParams


P1_RESULT_FIELDS = {
    "participant_id",
    "capacity",
    "layout_fingerprint",
    "bytes_per_slot",
    "max_entries",
    "max_bytes",
}
P1_SNAPSHOT_FIELDS = (
    "rank",
    "current_entries",
    "current_bytes",
    "current_logical_bytes",
    "deduplicated_bytes",
    "peak_entries",
    "peak_bytes",
    "hits",
    "misses",
    "evictions",
    "validation_failures",
    "failed_restores",
)
P2_ZERO_SNAPSHOT_FIELDS = (
    "current_encoded_physical_bytes",
    "current_encoded_logical_bytes",
    "current_full_fidelity_logical_bytes",
    "current_codec_metadata_bytes",
    "current_reader_leases",
    "current_temporary_encode_workspace_bytes",
    "current_temporary_decode_workspace_bytes",
    "current_temporary_decode_cuda_allocated_bytes",
    "current_temporary_decode_cuda_reserved_bytes",
    "peak_encoded_logical_bytes",
    "peak_full_fidelity_logical_bytes",
    "peak_codec_metadata_bytes",
    "peak_reader_leases",
    "peak_temporary_decode_workspace_bytes",
    "peak_temporary_decode_cuda_allocated_bytes",
    "peak_temporary_decode_cuda_reserved_bytes",
    "deferred_snapshot_releases",
    "quarantines",
    "decode_failures",
    "commit_failures",
    "rollback_failures",
)
TASK5_OBSERVATION_FIELDS = {
    "representation",
    "representation_version",
    "codec",
    "publishes",
    "hits",
    "misses",
    "evictions",
    "validation_failures",
    "quarantines",
    "decode_failures",
    "commit_failures",
    "rollback_failures",
    "fallbacks",
    "partial_restore_attempts",
    "mixed_representation_rejections",
    "missing_layer_rejections",
    "current_full_fidelity_logical_bytes",
    "current_encoded_physical_bytes",
    "current_codec_metadata_bytes",
    "peak_temporary_encode_workspace_bytes",
    "peak_temporary_decode_workspace_bytes",
    "peak_temporary_decode_cuda_allocated_bytes",
    "peak_temporary_decode_cuda_reserved_bytes",
}


def _layout():
    return HybridStateLayout(tuple(
        component
        for layer_index in range(18)
        for component in (
            HybridStateComponentSpec(
                layer_index,
                "linear_convolution",
                (1, 2),
                torch.bfloat16,
            ),
            HybridStateComponentSpec(
                layer_index,
                "linear_recurrent",
                (1, 1, 2),
                torch.float32,
            ),
        )
    ))


def _pool(capacity=4):
    return HybridStateTensorPool(
        _layout(),
        capacity=capacity,
        device="cpu",
    )


def _publication_payload(lease, ticket_id=0):
    return Qwen35HybridPrefixPublicationPayload(
        ticket_id=ticket_id,
        participant_id=0,
        request_id=lease.request_id,
        key=Qwen35HybridPrefixKey(
            token_hash=201,
            token_count=4,
            terminal_block_hash=201,
            block_size=4,
            model_fingerprint="qwen35-int8-participant-model",
            layout_fingerprint="qwen35-int8-participant-layout",
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
        ),
        token_ids=(1, 2, 3, 4),
        block_identities=((7, 3, 201),),
        lease=lease,
    )


def _load_class_method(
    relative_path,
    class_name,
    method_name,
    namespace,
):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method_node.decorator_list = []
    module = ast.Module(body=[method_node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace[method_name]


def _load_function(relative_path, function_name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    function_node.decorator_list = []
    module = ast.Module(body=[function_node], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace[function_name]


def _runner_method(name):
    return _load_class_method(
        "tinyvllm/engine/model_runner.py",
        "ModelRunner",
        name,
        {
            "Qwen35HybridPrefixRestoreOwner": (
                Qwen35HybridPrefixRestoreOwner
            ),
            "Qwen35HybridPrefixRestoreParticipant": (
                Qwen35HybridPrefixRestoreParticipant
            ),
            "Qwen35HybridPrefixPublicationParticipant": (
                Qwen35HybridPrefixPublicationParticipant
            ),
            "build_qwen35_hybrid_prefix_restore_owner": (
                build_qwen35_hybrid_prefix_restore_owner
            ),
            "QWEN35_HYBRID_PREFIX_DEFAULT": (
                QWEN35_HYBRID_PREFIX_DEFAULT
            ),
        },
    )


def _runner(pool, rank):
    runner = types.SimpleNamespace(
        rank=rank,
        hybrid_state_runtime_bridge=HybridStateRuntimeBridge(pool),
        qwen35_hybrid_prefix_restore_participant=None,
        qwen35_hybrid_prefix_publication_participant=None,
        qwen35_hybrid_prefix_restore_owner=None,
    )
    install_restore = _runner_method(
        "install_qwen35_hybrid_prefix_restore_participant"
    )
    install_publication = _runner_method(
        "install_qwen35_hybrid_prefix_publication_participant"
    )
    runner.install_qwen35_hybrid_prefix_restore_participant = (
        lambda participant: install_restore(runner, participant)
    )
    runner.install_qwen35_hybrid_prefix_publication_participant = (
        lambda participant: install_publication(runner, participant)
    )
    return runner


class _WorkerAck:
    def __init__(self, rank, result):
        self.rank = rank
        self.result = result


class _Collector:
    def __init__(self):
        self.poison_reasons = []

    def poison(self, reason):
        self.poison_reasons.append(reason)


def _owner_row(
    rank,
    *,
    representation=QWEN35_HYBRID_PREFIX_EXACT,
    representation_version=QWEN35_HYBRID_PREFIX_EXACT_VERSION,
    codec=None,
):
    return {
        "participant_id": rank,
        "capacity": 4,
        "layout_fingerprint": "layout-a",
        "bytes_per_slot": 1024,
        "max_entries": 8,
        "max_bytes": 1 << 20,
        "representation": representation,
        "representation_version": representation_version,
        "codec": codec,
    }


def _engine(rows):
    allocator = HybridStateSlotAllocator(4)
    block_manager = BlockManager.__new__(BlockManager)
    engine = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(world_size=len(rows)),
        scheduler=types.SimpleNamespace(
            block_manager=block_manager,
            hybrid_state_allocator=allocator,
        ),
        model_runner_ack_collector=_Collector(),
        qwen35_hybrid_prefix_engine_restore_coordinator=None,
        qwen35_hybrid_prefix_restore_configuration=None,
    )
    calls = []

    def acknowledged(method_name, *args, timeout_s):
        calls.append((method_name, args, timeout_s))
        return (
            rows[0],
            tuple(
                _WorkerAck(rank, rows[rank])
                for rank in range(1, len(rows))
            ),
        )

    engine.call_model_runner_acknowledged = acknowledged
    poison = _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "_poison_model_runner_ack_collector",
        {},
    )
    engine._poison_model_runner_ack_collector = (
        lambda reason: poison(engine, reason)
    )
    engine.install_qwen35_hybrid_prefix_engine_restore_coordinator = (
        lambda coordinator: setattr(
            engine,
            "qwen35_hybrid_prefix_engine_restore_coordinator",
            coordinator,
        )
    )
    engine._calls = calls
    return engine


def _engine_configure_method():
    return _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        "configure_qwen35_hybrid_prefix_restore",
        {
            "Qwen35HybridPrefixEngineRestoreCoordinator": (
                Qwen35HybridPrefixEngineRestoreCoordinator
            ),
            "QWEN35_HYBRID_PREFIX_DEFAULT": (
                QWEN35_HYBRID_PREFIX_DEFAULT
            ),
            "resolve_qwen35_hybrid_prefix_representation": (
                resolve_qwen35_hybrid_prefix_representation
            ),
        },
    )


def _engine_method(name, namespace):
    return _load_class_method(
        "tinyvllm/engine/llm_engine.py",
        "LLMEngine",
        name,
        namespace,
    )


def _expect_error(callback, message):
    try:
        callback()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_omitted_representation_builds_unchanged_p1_owner_and_identity():
    pool = _pool()
    owner = build_qwen35_hybrid_prefix_restore_owner(
        pool,
        participant_id=0,
        max_entries=8,
        max_bytes=1 << 20,
    )

    assert type(owner.snapshot_cache) is Qwen35HybridPrefixSnapshotCache
    assert owner.pool is pool
    assert owner.participant.snapshot_cache is owner.snapshot_cache
    assert owner.publication_participant.snapshot_cache is (
        owner.snapshot_cache
    )
    assert owner.representation == QWEN35_HYBRID_PREFIX_EXACT
    assert owner.representation_version == QWEN35_HYBRID_PREFIX_EXACT_VERSION
    assert owner.codec is None


def test_explicit_int8_representation_builds_p2_owner_and_codec_identity():
    owner = build_qwen35_hybrid_prefix_restore_owner(
        _pool(),
        participant_id=0,
        max_entries=8,
        max_bytes=1 << 20,
        representation=QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    )

    assert type(owner.snapshot_cache) is Qwen35HybridPrefixInt8SnapshotCache
    assert owner.representation == QWEN35_HYBRID_PREFIX_RECURRENT_INT8
    assert owner.representation_version == QWEN35_HYBRID_PREFIX_INT8_VERSION
    assert owner.codec == QWEN35_RECURRENT_INT8_CODEC
    assert owner.participant.snapshot_cache is owner.snapshot_cache
    assert owner.publication_participant.snapshot_cache is (
        owner.snapshot_cache
    )


def test_int8_observation_snapshot_exposes_complete_task5_schema():
    owner = build_qwen35_hybrid_prefix_restore_owner(
        _pool(),
        participant_id=0,
        max_entries=8,
        max_bytes=1 << 20,
        representation=QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    )

    snapshot = owner.snapshot_cache.observation_snapshot()

    assert TASK5_OBSERVATION_FIELDS <= set(snapshot)


def test_explicit_int8_participants_publish_through_seal():
    pool = _pool()
    owner = build_qwen35_hybrid_prefix_restore_owner(
        pool,
        participant_id=0,
        max_entries=8,
        max_bytes=1 << 20,
        representation=QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    )
    restore_participant = Qwen35HybridPrefixRestoreParticipant(
        0,
        pool,
        owner.snapshot_cache,
    )
    publication_participant = Qwen35HybridPrefixPublicationParticipant(
        0,
        pool,
        owner.snapshot_cache,
    )
    lease = HybridStateLease(0, 1, 100)
    pool.activate(lease)
    payload = _publication_payload(lease)

    assert restore_participant.snapshot_cache is owner.snapshot_cache
    assert publication_participant.prepare(payload).status == "prepared"
    assert owner.snapshot_cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 1
    assert publication_participant.precommit(
        payload
    ).status == "precommitted"
    assert publication_participant.commit(payload).status == "finalized"
    assert publication_participant.seal(payload).status == "committed"
    observation = owner.snapshot_cache.observation_snapshot()
    assert observation["current_entries"] == 1
    assert observation["current_prepared_publications"] == 0


def test_explicit_int8_invalid_prepared_handle_is_rolled_back():
    pool = _pool()
    owner = build_qwen35_hybrid_prefix_restore_owner(
        pool,
        participant_id=0,
        max_entries=8,
        max_bytes=1 << 20,
        representation=QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    )

    class InvalidHandleInt8Cache(Qwen35HybridPrefixInt8SnapshotCache):
        def prepare_publication(self, *args, **kwargs):
            super().prepare_publication(
                *args,
                **kwargs,
            )
            return object()

    cache = InvalidHandleInt8Cache(
        owner.state_transaction,
        max_entries=8,
        max_bytes=1 << 20,
    )
    participant = Qwen35HybridPrefixPublicationParticipant(
        0,
        pool,
        cache,
    )
    lease = HybridStateLease(0, 1, 101)
    pool.activate(lease)
    payload = _publication_payload(lease, ticket_id=1)

    acknowledgement = participant.prepare(payload)

    assert acknowledgement.status == "error"
    assert "invalid prepared publication" in acknowledgement.detail
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0
    retry = participant.prepare(payload)
    assert retry.status == "error"
    assert "already prepared" not in retry.detail
    assert cache.observation_snapshot()[
        "current_prepared_publications"
    ] == 0


def test_model_runner_fake_ranks_report_one_representation_identity():
    configure = _runner_method(
        "configure_qwen35_hybrid_prefix_restore_owner"
    )
    exact = configure(_runner(_pool(), 0), 8, 1 << 20)
    int8 = configure(
        _runner(_pool(), 1),
        8,
        1 << 20,
        QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    )

    assert P1_RESULT_FIELDS <= set(exact)
    assert P1_RESULT_FIELDS <= set(int8)
    assert (
        exact["representation"],
        exact["representation_version"],
        exact["codec"],
    ) == (
        QWEN35_HYBRID_PREFIX_EXACT,
        QWEN35_HYBRID_PREFIX_EXACT_VERSION,
        None,
    )
    assert (
        int8["representation"],
        int8["representation_version"],
        int8["codec"],
    ) == (
        QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
        QWEN35_HYBRID_PREFIX_INT8_VERSION,
        QWEN35_RECURRENT_INT8_CODEC,
    )


def test_model_runner_rejects_cross_representation_reconfiguration():
    configure = _runner_method(
        "configure_qwen35_hybrid_prefix_restore_owner"
    )
    runner = _runner(_pool(), 0)
    configure(runner, 8, 1 << 20)

    _expect_error(
        lambda: configure(
            runner,
            8,
            1 << 20,
            QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
        ),
        "already configured",
    )


def test_engine_requires_all_fake_ranks_to_agree_on_runtime_identity():
    configure = _engine_configure_method()
    mismatches = (
        {
            "representation": QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
            "representation_version": QWEN35_HYBRID_PREFIX_EXACT_VERSION,
            "codec": None,
        },
        {
            "representation": QWEN35_HYBRID_PREFIX_EXACT,
            "representation_version": QWEN35_HYBRID_PREFIX_INT8_VERSION,
            "codec": None,
        },
        {
            "representation": QWEN35_HYBRID_PREFIX_EXACT,
            "representation_version": QWEN35_HYBRID_PREFIX_EXACT_VERSION,
            "codec": QWEN35_RECURRENT_INT8_CODEC,
        },
    )
    for mismatch in mismatches:
        engine = _engine((
            _owner_row(0),
            _owner_row(1, **mismatch),
        ))
        _expect_error(
            lambda engine=engine: configure(
                engine,
                max_entries=8,
                max_bytes=1 << 20,
                timeout_s=0.5,
            ),
            "mismatch",
        )
        assert engine.model_runner_ack_collector.poison_reasons
        assert engine.qwen35_hybrid_prefix_restore_configuration is None


def test_omitted_engine_args_preserve_p1_tuple_and_result_fields():
    configure = _engine_configure_method()
    engine = _engine((_owner_row(0), _owner_row(1)))

    coordinator = configure(
        engine,
        max_entries=8,
        max_bytes=1 << 20,
        timeout_s=0.5,
    )

    assert isinstance(
        coordinator,
        Qwen35HybridPrefixEngineRestoreCoordinator,
    )
    assert engine.qwen35_hybrid_prefix_restore_configuration == (
        8,
        1 << 20,
        QWEN35_HYBRID_PREFIX_EXACT,
        0.5,
    )
    assert engine._calls == [(
        "configure_qwen35_hybrid_prefix_restore_owner",
        (8, 1 << 20, QWEN35_HYBRID_PREFIX_EXACT),
        0.5,
    )]
    assert all(P1_RESULT_FIELDS <= set(row) for row in (
        _owner_row(0),
        _owner_row(1),
    ))


def test_engine_rejects_cross_representation_reconfiguration():
    configure = _engine_configure_method()
    engine = _engine((_owner_row(0), _owner_row(1)))
    configure(
        engine,
        max_entries=8,
        max_bytes=1 << 20,
        timeout_s=0.5,
    )

    _expect_error(
        lambda: configure(
            engine,
            max_entries=8,
            max_bytes=1 << 20,
            representation=QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
            timeout_s=0.5,
        ),
        "already configured",
    )


def test_p1_snapshot_retains_old_fields_and_zeros_every_p2_only_field():
    configure = _runner_method(
        "configure_qwen35_hybrid_prefix_restore_owner"
    )
    snapshot_method = _runner_method(
        "qwen35_hybrid_prefix_cache_snapshot"
    )
    runner = _runner(_pool(), 3)
    configure(runner, 8, 1 << 20)
    owner = runner.qwen35_hybrid_prefix_restore_owner
    lease = HybridStateLease(0, 1, 200)
    owner.pool.activate(lease)
    payload = _publication_payload(lease, ticket_id=2)
    assert owner.snapshot_cache.publish(
        payload.key,
        payload.token_ids,
        payload.block_identities,
        lease,
    )

    snapshot = snapshot_method(runner)

    assert tuple(
        field for field in P1_SNAPSHOT_FIELDS if field in snapshot
    ) == P1_SNAPSHOT_FIELDS
    assert snapshot["rank"] == 3
    assert snapshot["representation"] == QWEN35_HYBRID_PREFIX_EXACT
    assert (
        snapshot["representation_version"]
        == QWEN35_HYBRID_PREFIX_EXACT_VERSION
    )
    assert snapshot["codec"] is None
    assert snapshot["publishes"] == 1
    assert {
        field: snapshot[field]
        for field in P2_ZERO_SNAPSHOT_FIELDS
    } == {
        field: 0
        for field in P2_ZERO_SNAPSHOT_FIELDS
    }


def test_int8_model_runner_snapshot_exposes_complete_task5_schema():
    configure = _runner_method(
        "configure_qwen35_hybrid_prefix_restore_owner"
    )
    snapshot_method = _runner_method(
        "qwen35_hybrid_prefix_cache_snapshot"
    )
    runner = _runner(_pool(), 4)
    configure(
        runner,
        8,
        1 << 20,
        QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    )

    snapshot = snapshot_method(runner)

    assert TASK5_OBSERVATION_FIELDS <= set(snapshot)


def test_corrupted_int8_restore_misses_then_full_prefill_without_p1_lookup():
    pool = _pool()
    owner = build_qwen35_hybrid_prefix_restore_owner(
        pool,
        participant_id=0,
        max_entries=8,
        max_bytes=1 << 20,
        representation=QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
    )
    source_lease = HybridStateLease(0, 1, 100)
    pool.activate(source_lease)
    token_ids = (1, 2, 3, 4)
    exact_lookup_calls = 0
    exact_participant_calls = 0
    original_exact_lookup = Qwen35HybridPrefixSnapshotCache._lookup

    def fail_exact_lookup(*args, **kwargs):
        nonlocal exact_lookup_calls
        exact_lookup_calls += 1
        raise AssertionError("P1 exact lookup must not run for P2 restore")

    original_block_size = Sequence.block_size
    Sequence.block_size = 4
    scheduler = Scheduler(
        types.SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=16,
            max_model_len=16,
            max_num_prefill_tokens_per_step=0,
            chunked_prefill_decode_first=True,
            chunked_prefill_max_consecutive_chunks=0,
            chunked_prefill_mixed_batch=False,
            chunked_prefill_mixed_min_prompt_tokens=0,
            chunked_prefill_adaptive_mixed=False,
            chunked_prefill_adaptive_enter_waiting=8,
            chunked_prefill_adaptive_exit_waiting=2,
            chunked_prefill_adaptive_transition_steps=2,
            chunked_prefill_adaptive_max_mixed_steps=2,
            chunked_prefill_slo_mixed=False,
            chunked_prefill_slo_target_gap_ns=0,
            chunked_prefill_slo_reserve_ns=0,
            chunked_prefill_slo_cost_intercept_ns=0,
            chunked_prefill_slo_cost_per_prefill_token_ns=0,
            chunked_prefill_slo_min_chunk_tokens=1,
            eos=-1,
            num_kvcache_blocks=8,
            kvcache_block_size=4,
        ),
        hybrid_state_allocator=HybridStateSlotAllocator(4),
    )
    source = Sequence(
        list(token_ids),
        SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
        ),
    )
    scheduler.block_manager.allocate(
        source,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    scheduler.block_manager.commit_prefill(
        source,
        0,
        len(source),
    )
    source_block_id = source.block_table[0]
    source_block = scheduler.block_manager.blocks[source_block_id]
    block_identities = ((
        source_block_id,
        source_block.generation,
        source_block.hash,
    ),)
    scheduler.block_manager.deallocate(source)
    terminal_hash = block_identities[-1][2]
    key = Qwen35HybridPrefixKey(
        token_hash=terminal_hash,
        token_count=4,
        terminal_block_hash=terminal_hash,
        block_size=4,
        model_fingerprint="qwen35-int8-request-model",
        layout_fingerprint="qwen35-int8-request-layout",
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
    )
    assert owner.snapshot_cache.publish(
        key,
        token_ids,
        block_identities,
        source_lease,
    )
    pool.release(source_lease)

    resident = next(iter(owner.snapshot_cache._entries.values()))
    resident.layers[-1].recurrent_values.view(torch.uint8).fill_(0x80)
    int8_participant = Qwen35HybridPrefixRestoreParticipant(
        0,
        pool,
        owner.snapshot_cache,
    )

    class FailIfCalledExactParticipant:
        def prepare(self, payload):
            nonlocal exact_participant_calls
            exact_participant_calls += 1
            raise AssertionError(
                "P1 exact participant must not run for P2 restore"
            )

    exact_participant = FailIfCalledExactParticipant()
    engine = types.SimpleNamespace(
        tokenizer=None,
        scheduler=scheduler,
        model_runner=types.SimpleNamespace(world_size=1),
        qwen35_hybrid_prefix_engine_restore_coordinator=None,
        qwen35_hybrid_prefix_runtime_identity=types.SimpleNamespace(
            model_fingerprint=key.model_fingerprint,
            layout_fingerprint=key.layout_fingerprint,
            dtype=key.dtype,
        ),
    )

    def operation_row(payload, operation, status, detail=""):
        return ({
            "ticket_id": payload.ticket_id,
            "participant_id": 0,
            "operation": operation,
            "status": status,
            "detail": detail,
        },)

    def prepare_restore(payload, *, timeout_s):
        acknowledgement = int8_participant.prepare(payload)
        return operation_row(
            payload,
            "prepare",
            acknowledgement.status,
            acknowledgement.detail,
        )

    def validate_restore(payload, *, timeout_s):
        int8_participant.validate_prepared(payload)
        return operation_row(payload, "validate", "ok")

    def commit_restore(payload, *, timeout_s):
        int8_participant.commit(payload)
        return operation_row(payload, "commit", "ok")

    def rollback_restore(payload, *, timeout_s):
        int8_participant.rollback(payload)
        return operation_row(payload, "rollback", "ok")

    engine.prepare_model_runner_hybrid_prefix_restore = prepare_restore
    engine.validate_model_runner_hybrid_prefix_restore = validate_restore
    engine.commit_model_runner_hybrid_prefix_restore = commit_restore
    engine.rollback_model_runner_hybrid_prefix_restore = rollback_restore
    engine._poison_model_runner_ack_collector = lambda reason: None
    engine.exact_restore_participant = exact_participant
    coordinator = Qwen35HybridPrefixEngineRestoreCoordinator(
        engine,
        scheduler.block_manager,
        scheduler.hybrid_state_allocator,
        timeout_s=0.5,
    )
    engine.qwen35_hybrid_prefix_engine_restore_coordinator = coordinator
    acquire = _engine_method(
        "acquire_qwen35_hybrid_prefix",
        {},
    )
    engine.acquire_qwen35_hybrid_prefix = (
        lambda sequence, requested_key, requested_tokens: acquire(
            engine,
            sequence,
            requested_key,
            requested_tokens,
        )
    )
    engine.flush_pending_hybrid_state_releases = (
        lambda **_kwargs: ()
    )
    try_restore = _load_function(
        "tinyvllm/engine/llm_engine.py",
        "_try_qwen35_hybrid_prefix_restore",
        {
            "Qwen35HybridPrefixKey": Qwen35HybridPrefixKey,
        },
    )
    add_request = _engine_method(
        "add_request",
        {
            "Sequence": Sequence,
            "_try_qwen35_hybrid_prefix_restore": try_restore,
        },
    )

    Qwen35HybridPrefixSnapshotCache._lookup = fail_exact_lookup
    try:
        add_request(
            engine,
            [1, 2, 3, 4, 9],
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                ignore_eos=True,
            ),
        )
        scheduled, is_prefill, do_sample = scheduler.schedule()
    finally:
        Qwen35HybridPrefixSnapshotCache._lookup = original_exact_lookup
        Sequence.block_size = original_block_size

    assert scheduled
    admitted = scheduled[0]
    assert is_prefill is True
    assert do_sample is True
    assert admitted.hybrid_prefix_restore_attempted is True
    assert admitted.hybrid_prefix_restore_hit is False
    assert admitted.num_cached_tokens == 0
    assert admitted.prefill_chunk_start == 0
    assert admitted.prefill_chunk_end == len(admitted)
    assert exact_lookup_calls == 0
    assert exact_participant_calls == 0
    assert coordinator.last_ticket is not None
    assert coordinator.last_ticket.state == "rolled_back"
    assert tuple(
        row["status"]
        for row in coordinator.last_ticket.prepare_results
    ) == ("miss",)
    observation = owner.snapshot_cache.observation_snapshot()
    assert observation["fallbacks"] == 0, observation
    assert observation["quarantines"] == 1


def main():
    tests = (
        test_omitted_representation_builds_unchanged_p1_owner_and_identity,
        test_explicit_int8_representation_builds_p2_owner_and_codec_identity,
        test_int8_observation_snapshot_exposes_complete_task5_schema,
        test_explicit_int8_participants_publish_through_seal,
        test_explicit_int8_invalid_prepared_handle_is_rolled_back,
        test_model_runner_fake_ranks_report_one_representation_identity,
        test_model_runner_rejects_cross_representation_reconfiguration,
        test_engine_requires_all_fake_ranks_to_agree_on_runtime_identity,
        test_omitted_engine_args_preserve_p1_tuple_and_result_fields,
        test_engine_rejects_cross_representation_reconfiguration,
        test_p1_snapshot_retains_old_fields_and_zeros_every_p2_only_field,
        test_int8_model_runner_snapshot_exposes_complete_task5_schema,
        test_corrupted_int8_restore_misses_then_full_prefill_without_p1_lookup,
    )
    for test in tests:
        test()
    print(
        "qwen35 hybrid prefix INT8 runtime tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
