from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

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

config_module = types.ModuleType("tinyvllm.config")
config_module.Config = object
sys.modules["tinyvllm.config"] = config_module


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

sampling = _load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
sequence_module = _load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
_load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)
hybrid_module = _load_module(
    "tinyvllm.engine.hybrid_state",
    "tinyvllm/engine/hybrid_state.py",
)
scheduler_module = _load_module(
    "tinyvllm.engine.scheduler",
    "tinyvllm/engine/scheduler.py",
)

SamplingParams = sampling.SamplingParams
Sequence = sequence_module.Sequence
SequenceStatus = sequence_module.SequenceStatus
HybridStateLease = hybrid_module.HybridStateLease
HybridStateSlotAllocator = hybrid_module.HybridStateSlotAllocator
Scheduler = scheduler_module.Scheduler
Sequence.block_size = 2


def _config(**overrides):
    values = {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 64,
        "max_model_len": 64,
        "max_num_prefill_tokens_per_step": 0,
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_adaptive_enter_waiting": 8,
        "chunked_prefill_adaptive_exit_waiting": 2,
        "chunked_prefill_adaptive_transition_steps": 2,
        "chunked_prefill_adaptive_max_mixed_steps": 2,
        "chunked_prefill_slo_mixed": False,
        "chunked_prefill_slo_target_gap_ns": 0,
        "chunked_prefill_slo_reserve_ns": 0,
        "chunked_prefill_slo_cost_intercept_ns": 0,
        "chunked_prefill_slo_cost_per_prefill_token_ns": 0,
        "chunked_prefill_slo_min_chunk_tokens": 1,
        "eos": -1,
        "num_kvcache_blocks": 16,
        "kvcache_block_size": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sequence(tokens, max_tokens=4):
    return Sequence(
        list(tokens),
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )


def _lease(sequence):
    return HybridStateLease(
        sequence.hybrid_state_slot_id,
        sequence.hybrid_state_generation,
        sequence.seq_id,
    )


def _expect_error(function, message):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_hook_installation_is_default_off_and_idempotent():
    scheduler = Scheduler(_config())
    assert scheduler.prefill_commit_hook is None
    hook = lambda sequence: None

    scheduler.install_prefill_commit_hook(hook)
    scheduler.install_prefill_commit_hook(hook)
    assert scheduler.prefill_commit_hook is hook

    _expect_error(
        lambda: scheduler.install_prefill_commit_hook(object()),
        "callable",
    )
    _expect_error(
        lambda: scheduler.install_prefill_commit_hook(
            lambda sequence: None
        ),
        "already installed",
    )


def test_legacy_hook_runs_after_commit_before_append_or_release():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        _config(),
        hybrid_state_allocator=allocator,
    )
    sequence = _sequence([1, 2, 3, 4], max_tokens=1)
    events = []

    def hook(source):
        events.append(source.seq_id)
        assert source is sequence
        assert source.num_computed_tokens == source.num_prompt_tokens
        assert source.completion_token_ids == []
        assert source.block_table
        assert all(
            scheduler.block_manager.blocks[block_id].hash >= 0
            for block_id in source.block_table[:2]
        )
        allocator.validate(_lease(source))

    scheduler.install_prefill_commit_hook(hook)
    scheduler.add(sequence)
    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(
        seqs,
        [99],
        is_prefill,
        do_sample,
    )

    assert events == [sequence.seq_id]
    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.block_table == []
    assert sequence.hybrid_state_slot_id == -1


def test_legacy_hook_splits_unaligned_prompt_at_full_block_boundary():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        _config(),
        hybrid_state_allocator=allocator,
    )
    sequence = _sequence([1, 2, 3, 4, 5], max_tokens=1)
    events = []
    scheduler.install_prefill_commit_hook(
        lambda source: events.append((
            source.num_computed_tokens,
            tuple(source.completion_token_ids),
        ))
    )
    scheduler.add(sequence)

    first = scheduler.schedule()
    assert first[1:] == (True, False)
    assert first[0][0].prefill_chunk_start == 0
    assert first[0][0].prefill_chunk_end == 4
    scheduler.postprocess(
        first[0],
        None,
        first[1],
        first[2],
    )
    assert events == [(4, ())]
    assert sequence.status == SequenceStatus.PREFILLING

    second = scheduler.schedule()
    assert second[1:] == (True, True)
    assert second[0][0].prefill_chunk_start == 4
    assert second[0][0].prefill_chunk_end == 5
    scheduler.postprocess(
        second[0],
        [99],
        second[1],
        second[2],
    )

    assert events == [(4, ())]
    assert sequence.status == SequenceStatus.FINISHED


def test_hook_does_not_republish_restored_prefix_after_suffix_commit():
    scheduler = Scheduler(_config())
    sequence = _sequence([1, 2, 3, 4, 5], max_tokens=1)
    events = []
    scheduler.install_prefill_commit_hook(
        lambda source: events.append(source.num_computed_tokens)
    )
    sequence.num_cached_tokens = 4
    sequence.num_computed_tokens = 5
    sequence.prefill_chunk_start = 4
    sequence.prefill_chunk_end = 5
    sequence.prefill_chunk_final = True

    scheduler._notify_prefill_committed(sequence)

    assert events == []


def test_chunked_hook_splits_unaligned_prompt_at_publication_boundary():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        _config(max_num_prefill_tokens_per_step=8),
        hybrid_state_allocator=allocator,
    )
    sequence = _sequence([1, 2, 3, 4, 5], max_tokens=1)
    events = []
    scheduler.install_prefill_commit_hook(
        lambda source: events.append(source.num_computed_tokens)
    )
    scheduler.add(sequence)

    first = scheduler.schedule()

    assert first[1:] == (True, False)
    assert first[0][0].prefill_chunk_start == 0
    assert first[0][0].prefill_chunk_end == 4


def test_chunked_hook_runs_only_on_final_chunk_once():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        _config(max_num_prefill_tokens_per_step=2),
        hybrid_state_allocator=allocator,
    )
    sequence = _sequence([1, 2, 3, 4], max_tokens=2)
    events = []
    scheduler.install_prefill_commit_hook(
        lambda source: events.append((
            source.seq_id,
            source.num_computed_tokens,
            tuple(source.completion_token_ids),
        ))
    )
    scheduler.add(sequence)

    first = scheduler.schedule()
    scheduler.postprocess(
        first[0],
        None,
        first[1],
        first[2],
    )
    assert events == []

    second = scheduler.schedule()
    scheduler.postprocess(
        second[0],
        [90],
        second[1],
        second[2],
    )
    assert events == [(sequence.seq_id, 4, ())]

    scheduler.prefilling.append(sequence)
    sequence.status = SequenceStatus.PREFILLING
    sequence.prefill_chunk_start = 3
    sequence.prefill_chunk_end = 4
    sequence.prefill_chunk_final = True
    scheduler._postprocess_chunked_prefill(
        [sequence],
        [91],
        True,
    )
    assert events == [(sequence.seq_id, 4, ())]


def test_mixed_hook_runs_before_prefill_token_append():
    allocator = HybridStateSlotAllocator(3)
    scheduler = Scheduler(
        _config(
            max_num_prefill_tokens_per_step=4,
            chunked_prefill_mixed_batch=True,
            chunked_prefill_decode_first=False,
        ),
        hybrid_state_allocator=allocator,
    )
    prefill = _sequence([1, 2, 3, 4])
    decode = _sequence([7, 8])
    scheduler.add(decode)
    initial = scheduler.schedule()
    scheduler.postprocess(
        initial[0],
        [70],
        initial[1],
        initial[2],
    )
    scheduler.add(prefill)
    events = []
    scheduler.install_prefill_commit_hook(
        lambda source: events.append((
            source.seq_id,
            tuple(source.completion_token_ids),
            bool(source.block_table),
            allocator.validate(_lease(source)),
        ))
    )

    mixed = scheduler.schedule()
    assert len(mixed) == 4 and mixed[3] == "mixed"
    scheduler.postprocess(
        mixed[0],
        [80, 81],
        mixed[1],
        mixed[2],
        mixed[3],
    )

    assert events == [
        (
            prefill.seq_id,
            (),
            True,
            _lease(prefill),
        )
    ]


def test_hook_failure_poisons_before_append_and_release():
    allocator = HybridStateSlotAllocator(2)
    scheduler = Scheduler(
        _config(),
        hybrid_state_allocator=allocator,
    )
    sequence = _sequence([1, 2, 3, 4], max_tokens=1)
    scheduler.install_prefill_commit_hook(
        lambda source: (_ for _ in ()).throw(
            RuntimeError("injected hook failure")
        )
    )
    scheduler.add(sequence)
    scheduled = scheduler.schedule()

    _expect_error(
        lambda: scheduler.postprocess(
            scheduled[0],
            [99],
            scheduled[1],
            scheduled[2],
        ),
        "hook failure",
    )
    assert sequence.completion_token_ids == []
    assert sequence.block_table
    allocator.validate(_lease(sequence))
    _expect_error(
        scheduler.schedule,
        "poisoned",
    )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "scheduler prefill commit hook tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
