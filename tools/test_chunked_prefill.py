"""Chunked prefill scheduler/block-manager tests.

跑法：python3 tools/test_chunked_prefill.py
"""

import os
import sys
import types
import importlib.util
import hashlib
import pickle
from types import SimpleNamespace

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:
    torch = None
    dist = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "engine")]
utils_pkg = types.ModuleType("tinyvllm.utils")
utils_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "utils")]
layers_pkg = types.ModuleType("tinyvllm.layers")
layers_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "layers")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)
sys.modules.setdefault("tinyvllm.utils", utils_pkg)
sys.modules.setdefault("tinyvllm.layers", layers_pkg)


def load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_REPO_ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


config_mod = types.ModuleType("tinyvllm.config")
config_mod.Config = object
sys.modules["tinyvllm.config"] = config_mod
xxhash_mod = types.ModuleType("xxhash")


class _FakeXXH64:
    def __init__(self):
        self._h = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._h.update(data)

    def intdigest(self):
        return int.from_bytes(self._h.digest(), "little")


xxhash_mod.xxh64 = _FakeXXH64
sys.modules.setdefault("xxhash", xxhash_mod)
sampling_mod = load_module("tinyvllm.sampling_params", "tinyvllm/sampling_params.py")
context_mod = load_module("tinyvllm.utils.context", "tinyvllm/utils/context.py") if torch is not None else None
sequence_mod = load_module("tinyvllm.engine.sequence", "tinyvllm/engine/sequence.py")
block_manager_mod = load_module("tinyvllm.engine.block_manager", "tinyvllm/engine/block_manager.py")
scheduler_mod = load_module("tinyvllm.engine.scheduler", "tinyvllm/engine/scheduler.py")
sam_mod = load_module("sam_chunked_prefill_test", "tinyvllm/speculative/sam.py")
if dist is not None:
    dist.get_rank = lambda: 0
    dist.get_world_size = lambda: 1
    embed_head_mod = load_module("tinyvllm.layers.embed_head", "tinyvllm/layers/embed_head.py")
else:
    embed_head_mod = None

BlockManager = block_manager_mod.BlockManager
Sequence = sequence_mod.Sequence
SuffixAutomatonDraftIndex = sam_mod.SuffixAutomatonDraftIndex
SequenceStatus = sequence_mod.SequenceStatus
Scheduler = scheduler_mod.Scheduler
SamplingParams = sampling_mod.SamplingParams
ParallelLMHead = embed_head_mod.ParallelLMHead if embed_head_mod is not None else None
set_context = context_mod.set_context if context_mod is not None else None
reset_context_global = context_mod.reset_context if context_mod is not None else None


def make_config(**overrides):
    cfg = dict(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
        eos=-1,
        num_kvcache_blocks=32,
        kvcache_block_size=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_max_consecutive_chunks=0,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def reset_sequence_state(block_size: int = 4):
    Sequence.block_size = block_size


def make_seq(token_ids, max_tokens: int = 4):
    return Sequence(list(token_ids), SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=False))


def assert_queue_contains(queue, seq):
    assert list(queue) == [seq]


def test_intermediate_chunk_does_not_sample_or_append():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    seq = make_seq(range(10))
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq]
    assert is_prefill is True
    assert do_sample is False
    assert seq.prefill_chunk_start == 0
    assert seq.prefill_chunk_end == 4
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    assert len(seq) == 10
    assert seq.completion_token_ids == []
    assert seq.num_computed_tokens == 4
    assert seq.status == SequenceStatus.PREFILLING
    assert_queue_contains(scheduler.prefilling, seq)
    assert list(scheduler.running) == []


def test_final_chunk_samples_once_and_moves_to_running():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    seq = make_seq(range(10), max_tokens=3)
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, None, is_prefill, do_sample)
    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, None, is_prefill, do_sample)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq]
    assert is_prefill is True
    assert do_sample is True
    assert seq.prefill_chunk_start == 8
    assert seq.prefill_chunk_end == 10
    scheduler.postprocess(seqs, [99], is_prefill, do_sample)

    assert seq.completion_token_ids == [99]
    assert seq.num_computed_tokens == 10
    assert seq.status == SequenceStatus.RUNNING
    assert_queue_contains(scheduler.running, seq)
    assert list(scheduler.prefilling) == []


def test_chunked_prefill_batches_multiple_short_final_prompts():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_num_prefill_tokens_per_step=4,
    ))
    seq_a = make_seq([1, 2, 3, 4])
    seq_b = make_seq([5, 6, 7, 8])
    scheduler.add(seq_a)
    scheduler.add(seq_b)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq_a, seq_b]
    assert is_prefill is True
    assert do_sample is True
    assert seq_a.prefill_chunk_start == 0
    assert seq_a.prefill_chunk_end == 4
    assert seq_b.prefill_chunk_start == 0
    assert seq_b.prefill_chunk_end == 4
    scheduler.postprocess(seqs, [91, 92], is_prefill, do_sample)

    assert seq_a.completion_token_ids == [91]
    assert seq_b.completion_token_ids == [92]
    assert list(scheduler.running) == [seq_a, seq_b]


def test_chunked_prefill_batches_warm_prompt_by_uncached_tokens():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=2,
        max_num_batched_tokens=10,
        max_model_len=10,
        max_num_prefill_tokens_per_step=4,
    ))
    _publish_and_release(
        scheduler.block_manager,
        list(range(1, 9)),
    )
    cold = make_seq([21, 22, 23, 24], max_tokens=1)
    warm = make_seq(list(range(1, 10)), max_tokens=1)
    scheduler.add(cold)
    scheduler.add(warm)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [cold, warm]
    assert is_prefill is True
    assert do_sample is True
    assert cold.prefill_chunk_start == 0
    assert cold.prefill_chunk_end == 4
    assert warm.num_cached_tokens == 8
    assert warm.prefill_chunk_start == 8
    assert warm.prefill_chunk_end == 9


def test_decode_first_prioritizes_existing_running_sequence():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=True,
    ))
    running = make_seq([1, 2, 3])
    scheduler.block_manager.allocate(running)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    waiting = make_seq(range(12))
    scheduler.add(waiting)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    assert list(scheduler.waiting) == [waiting]


def test_add_rejects_request_beyond_max_model_len():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_model_len=8))
    seq = make_seq(range(8), max_tokens=1)

    try:
        scheduler.add(seq)
    except ValueError as exc:
        message = str(exc)
    else:
        assert False, "expected max_model_len admission failure"

    assert "max_model_len" in message
    assert "prompt_tokens=8" in message
    assert "max_tokens=1" in message
    assert list(scheduler.waiting) == []


def test_add_rejects_prompt_beyond_logical_kv_capacity():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_model_len=64,
        num_kvcache_blocks=2,
        kvcache_block_size=4,
    ))
    seq = make_seq(range(9), max_tokens=1)

    try:
        scheduler.add(seq)
    except ValueError as exc:
        message = str(exc)
    else:
        assert False, "expected KV capacity admission failure"

    assert "KV cache capacity" in message
    assert "required_blocks=3" in message
    assert "available_blocks=2" in message
    assert list(scheduler.waiting) == []


def test_add_accounts_for_decode_kv_capacity_boundary():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_model_len=64,
        num_kvcache_blocks=2,
        kvcache_block_size=4,
    ))
    fits = make_seq(range(7), max_tokens=2)
    scheduler.add(fits)
    assert list(scheduler.waiting) == [fits]

    too_many_decode_tokens = make_seq(range(7), max_tokens=3)
    try:
        scheduler.add(too_many_decode_tokens)
    except ValueError as exc:
        message = str(exc)
    else:
        assert False, "expected decode KV capacity admission failure"

    assert "KV cache capacity" in message
    assert "kv_tokens=9" in message
    assert "required_blocks=3" in message
    assert list(scheduler.waiting) == [fits]


def test_add_capacity_boundary_for_multiple_logical_block_counts():
    for num_blocks in (1, 2, 4):
        reset_sequence_state()
        scheduler = Scheduler(make_config(
            max_model_len=64,
            num_kvcache_blocks=num_blocks,
            kvcache_block_size=4,
        ))
        capacity_tokens = num_blocks * 4
        fits = make_seq(range(capacity_tokens), max_tokens=1)
        scheduler.add(fits)
        assert list(scheduler.waiting) == [fits]

        too_long = make_seq(range(capacity_tokens + 1), max_tokens=1)
        try:
            scheduler.add(too_long)
        except ValueError as exc:
            message = str(exc)
        else:
            assert False, f"expected KV capacity failure for num_blocks={num_blocks}"

        assert "KV cache capacity" in message
        assert f"available_blocks={num_blocks}" in message
        assert list(scheduler.waiting) == [fits]


def test_chunked_prefill_progresses_with_varied_chunk_sizes():
    for chunk_tokens in (1, 2, 4):
        reset_sequence_state()
        scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=chunk_tokens))
        prompt_tokens = 9
        seq = make_seq(range(prompt_tokens), max_tokens=2)
        scheduler.add(seq)

        expected_start = 0
        while expected_start < prompt_tokens:
            seqs, is_prefill, do_sample = scheduler.schedule()
            expected_end = min(expected_start + chunk_tokens, prompt_tokens)

            assert seqs == [seq]
            assert is_prefill is True
            assert seq.prefill_chunk_start == expected_start
            assert seq.prefill_chunk_end == expected_end
            assert do_sample is (expected_end == prompt_tokens)
            scheduler.postprocess(seqs, [700 + chunk_tokens] if do_sample else None, is_prefill, do_sample)
            expected_start = expected_end

        assert seq.completion_token_ids == [700 + chunk_tokens]
        assert seq.num_computed_tokens == prompt_tokens
        assert list(scheduler.prefilling) == []
        assert list(scheduler.running) == [seq]


def test_short_prefill_batch_respects_sequence_and_token_limits():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_num_prefill_tokens_per_step=4,
    ))
    seq_a = make_seq([1, 2, 3, 4], max_tokens=2)
    seq_b = make_seq([5, 6, 7, 8], max_tokens=2)
    seq_c = make_seq([9, 10, 11, 12], max_tokens=2)
    for seq in (seq_a, seq_b, seq_c):
        scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq_a, seq_b]
    assert is_prefill is True
    assert do_sample is True
    scheduler.postprocess(seqs, [101, 102], is_prefill, do_sample)
    assert seq_a.completion_token_ids == [101]
    assert seq_b.completion_token_ids == [102]
    assert list(scheduler.waiting) == [seq_c]
    assert list(scheduler.running) == [seq_a, seq_b]


def test_chunked_prefill_does_not_publish_future_block_hashes():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    seq = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()
    h0 = scheduler.block_manager.compute_hash(seq.block(0), -1)
    h1 = scheduler.block_manager.compute_hash(seq.block(1), h0)

    assert h0 not in scheduler.block_manager.hash_to_block_id
    assert h1 not in scheduler.block_manager.hash_to_block_id

    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    assert h0 in scheduler.block_manager.hash_to_block_id
    assert h1 not in scheduler.block_manager.hash_to_block_id

    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [42], is_prefill, do_sample)

    assert h1 in scheduler.block_manager.hash_to_block_id
    assert seq.completion_token_ids == [42]


def test_chunked_prefill_restores_reused_cached_block_metadata():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    cached = make_seq([1, 2, 3, 4])
    scheduler.block_manager.allocate(cached)
    h0 = scheduler.block_manager.compute_hash(cached.block(0), -1)
    block_id = cached.block_table[0]
    scheduler.block_manager.deallocate(cached)

    seq = make_seq([1, 2, 3, 4, 5, 6, 7, 8])
    scheduler.add(seq)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seq.num_cached_tokens == 4
    assert seq.num_computed_tokens == 4
    assert seq.block_table[0] == block_id
    assert scheduler.block_manager.blocks[block_id].hash == h0
    assert scheduler.block_manager.blocks[block_id].token_ids == [1, 2, 3, 4]

    scheduler.postprocess(seqs, [77], is_prefill, do_sample)
    h1 = scheduler.block_manager.compute_hash(seq.block(1), h0)

    assert h1 in scheduler.block_manager.hash_to_block_id
    assert seq.completion_token_ids == [77]


def _publish_and_release(block_manager, token_ids):
    seq = make_seq(token_ids, max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(seq, 0, len(seq))
    block_table = list(seq.block_table)
    block_manager.deallocate(seq)
    return block_table


def test_max_reusable_tokens_keeps_one_sampleable_token():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)

    expected = {
        3: 0,
        4: 0,
        5: 4,
        8: 4,
        9: 8,
    }
    for prompt_tokens, reusable_tokens in expected.items():
        seq = make_seq(range(prompt_tokens), max_tokens=1)
        assert block_manager.max_reusable_tokens(seq) == reusable_tokens


def test_allocate_caps_exact_block_aligned_cache_hit():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)
    cached_blocks = _publish_and_release(
        block_manager,
        [1, 2, 3, 4],
    )

    seq = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(seq),
    )

    assert seq.num_cached_tokens == 0
    assert seq.num_computed_tokens == 0
    assert seq.block_table[0] != cached_blocks[0]


def test_allocate_reuses_only_blocks_before_sampleable_suffix():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)
    cached_blocks = _publish_and_release(
        block_manager,
        list(range(1, 9)),
    )

    seq = make_seq(list(range(1, 9)), max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(seq),
    )

    assert seq.num_cached_tokens == 4
    assert seq.num_computed_tokens == 4
    assert seq.block_table[0] == cached_blocks[0]
    assert seq.block_table[1] != cached_blocks[1]


def test_can_allocate_excludes_live_prefix_hits_from_free_block_requirement():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    live = make_seq(list(range(1, 9)), max_tokens=2)
    block_manager.allocate(live, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(live, 0, len(live))

    warm = make_seq(list(range(1, 10)), max_tokens=1)
    assert len(block_manager.free_block_ids) == 1
    assert warm.num_blocks == 3
    assert block_manager.max_reusable_tokens(warm) == 8
    assert block_manager.can_allocate(warm) is True
    block_manager.allocate(
        warm,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(warm),
    )
    assert warm.num_cached_tokens == 8
    assert len(block_manager.free_block_ids) == 0


def test_can_allocate_counts_idle_prefix_hits_as_free_block_requirement():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    _publish_and_release(block_manager, list(range(1, 9)))

    unrelated_live = make_seq([21, 22, 23, 24], max_tokens=2)
    block_manager.allocate(
        unrelated_live,
        publish_hashes=False,
        max_cached_tokens=0,
    )

    warm = make_seq(list(range(1, 10)), max_tokens=1)
    assert len(block_manager.free_block_ids) == 2
    assert warm.num_blocks == 3
    assert block_manager.can_allocate(warm) is False


def test_estimate_admission_is_read_only_for_live_and_idle_hits():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=4, block_size=4)
    source = make_seq(list(range(1, 9)), max_tokens=2)
    block_manager.allocate(source, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(source, 0, len(source))
    source_blocks = list(source.block_table)

    live_sharer = make_seq(list(range(1, 6)), max_tokens=2)
    block_manager.allocate(
        live_sharer,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(live_sharer),
    )
    block_manager.deallocate(source)
    assert source_blocks[0] in block_manager.used_block_ids
    assert source_blocks[1] in block_manager.free_block_ids

    warm = make_seq(list(range(1, 10)), max_tokens=1)
    free_before = list(block_manager.free_block_ids)
    refs_before = [block.ref_count for block in block_manager.blocks]

    assert block_manager.estimate_admission(warm) == (8, 2)
    assert list(block_manager.free_block_ids) == free_before
    assert [block.ref_count for block in block_manager.blocks] == refs_before
    assert warm.block_table == []
    assert warm.num_cached_tokens == 0


def test_allocate_rejects_hash_collision_when_tokens_differ():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    cached = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        cached,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(cached, 0, len(cached))
    cached_block = cached.block_table[0]
    cached_hash = block_manager.blocks[cached_block].hash
    block_manager.deallocate(cached)

    original_compute_hash = block_manager.compute_hash
    block_manager.compute_hash = (
        lambda token_ids, prefix=-1: cached_hash
    )
    try:
        seq = make_seq([9, 8, 7, 6], max_tokens=1)
        assert block_manager.estimate_admission(seq)[0] == 0
        block_manager.allocate(
            seq,
            publish_hashes=False,
            max_cached_tokens=4,
        )
    finally:
        block_manager.compute_hash = original_compute_hash

    assert seq.num_cached_tokens == 0
    assert seq.block_table[0] != cached_block


def test_clear_reusable_cache_preserves_live_block_metadata():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    free_cached = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        free_cached,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(free_cached, 0, len(free_cached))
    free_block_id = free_cached.block_table[0]
    block_manager.deallocate(free_cached)

    live = make_seq([5, 6, 7, 8], max_tokens=2)
    block_manager.allocate(
        live,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(live, 0, len(live))
    live_block_id = live.block_table[0]
    live_hash = block_manager.blocks[live_block_id].hash
    live_tokens = list(block_manager.blocks[live_block_id].token_ids)

    cleared = block_manager.clear_reusable_cache()

    assert cleared == 1
    assert block_manager.blocks[free_block_id].hash == -1
    assert block_manager.blocks[free_block_id].token_ids == []
    assert block_manager.blocks[live_block_id].hash == live_hash
    assert block_manager.blocks[live_block_id].token_ids == live_tokens
    assert block_manager.blocks[live_block_id].ref_count == 1


def test_capacity_pressure_never_returns_live_shared_block():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    live = make_seq([1, 2, 3, 4], max_tokens=2)
    block_manager.allocate(
        live,
        publish_hashes=False,
        max_cached_tokens=0,
    )
    block_manager.commit_prefill(live, 0, len(live))
    live_block_id = live.block_table[0]

    shared = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    block_manager.allocate(
        shared,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(shared),
    )
    assert shared.block_table[0] == live_block_id
    assert block_manager.blocks[live_block_id].ref_count == 2

    other = make_seq([9, 8, 7, 6], max_tokens=1)
    block_manager.allocate(
        other,
        publish_hashes=False,
        max_cached_tokens=0,
    )

    assert other.block_table[0] != live_block_id
    assert live_block_id in block_manager.used_block_ids
    assert live_block_id not in block_manager.free_block_ids


def test_normal_prefill_publishes_only_after_postprocess():
    reset_sequence_state()
    scheduler = Scheduler(
        make_config(max_num_prefill_tokens_per_step=0)
    )
    seq = make_seq([1, 2, 3, 4, 5], max_tokens=2)
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()
    h0 = scheduler.block_manager.compute_hash([1, 2, 3, 4], -1)

    assert seqs == [seq]
    assert is_prefill is True
    assert do_sample is True
    assert seq.prefill_chunk_start == 0
    assert seq.prefill_chunk_end == 5
    assert h0 not in scheduler.block_manager.hash_to_block_id

    scheduler.postprocess(seqs, [99], is_prefill, do_sample)

    assert h0 in scheduler.block_manager.hash_to_block_id
    assert seq.num_computed_tokens == 5
    assert seq.completion_token_ids == [99]


def test_normal_prefill_does_not_reuse_prefix_created_in_same_batch():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_seqs=3,
        max_num_batched_tokens=32,
    ))
    seq_a = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    seq_b = make_seq([9, 8, 7, 6, 5], max_tokens=1)
    seq_c = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    for seq in (seq_a, seq_b, seq_c):
        scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq_a, seq_b, seq_c]
    assert is_prefill is True
    assert do_sample is True
    assert [seq.num_cached_tokens for seq in seqs] == [0, 0, 0]
    assert all(
        seq.prefill_chunk_end > seq.prefill_chunk_start
        for seq in seqs
    )
    assert seq_a.block_table[0] != seq_c.block_table[0]


def test_normal_prefill_exact_block_warm_hit_recomputes_final_block():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_batched_tokens=32,
    ))
    cold = make_seq([1, 2, 3, 4], max_tokens=1)
    scheduler.add(cold)
    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [70], is_prefill, do_sample)

    warm = make_seq([1, 2, 3, 4], max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [warm]
    assert warm.num_cached_tokens == 0
    assert warm.prefill_chunk_start == 0
    assert warm.prefill_chunk_end == 4


def _seed_scheduler_cache(scheduler, token_ids):
    seq = make_seq(token_ids, max_tokens=1)
    scheduler.add(seq)
    seqs, is_prefill, do_sample = scheduler.schedule()
    while not do_sample:
        scheduler.postprocess(seqs, None, is_prefill, do_sample)
        seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [71], is_prefill, do_sample)


def test_normal_prefill_warm_hit_reuses_only_complete_prefix_blocks():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_batched_tokens=32,
    ))
    _seed_scheduler_cache(scheduler, list(range(1, 9)))

    warm = make_seq(list(range(1, 9)), max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert warm.num_cached_tokens == 4
    assert warm.prefill_chunk_start == 4
    assert warm.prefill_chunk_end == 8
    assert do_sample is True


def test_normal_prefill_token_budget_counts_only_uncached_tokens():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_seqs=3,
        max_num_batched_tokens=10,
        max_model_len=10,
    ))
    _publish_and_release(
        scheduler.block_manager,
        list(range(1, 9)),
    )

    warm_seqs = [
        make_seq(list(range(1, 10)), max_tokens=1)
        for _ in range(3)
    ]
    for seq in warm_seqs:
        scheduler.add(seq)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == warm_seqs
    assert is_prefill is True
    assert do_sample is True
    assert [seq.num_cached_tokens for seq in seqs] == [8, 8, 8]
    assert [seq.prefill_chunk_start for seq in seqs] == [8, 8, 8]
    assert [seq.prefill_chunk_end for seq in seqs] == [9, 9, 9]


def test_normal_prefill_token_budget_still_limits_cold_prompts():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_seqs=3,
        max_num_batched_tokens=10,
        max_model_len=10,
    ))
    cold_seqs = [
        make_seq(range(offset, offset + 5), max_tokens=1)
        for offset in (0, 10, 20)
    ]
    for seq in cold_seqs:
        scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == cold_seqs[:2]
    assert is_prefill is True
    assert do_sample is True
    assert list(scheduler.waiting) == cold_seqs[2:]


def test_chunked_prefill_uses_same_sampleable_prefix_cap():
    reset_sequence_state()
    scheduler = Scheduler(
        make_config(max_num_prefill_tokens_per_step=4)
    )
    _seed_scheduler_cache(scheduler, list(range(1, 9)))

    warm = make_seq(list(range(1, 9)), max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert warm.num_cached_tokens == 4
    assert warm.prefill_chunk_start == 4
    assert warm.prefill_chunk_end == 8
    assert do_sample is True


def test_commit_accepted_tokens_appends_sequence_and_releases_unused_blocks():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)
    original_block = seq.block_table[0]

    reserved = block_manager.reserve_append_blocks(seq, 6)
    assert len(reserved) == 2

    block_manager.commit_accepted_tokens(seq, [4, 5], reserved)

    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert seq.last_token == 5
    assert seq.num_tokens == 5
    assert seq.block_table == [original_block]
    assert block_manager.blocks[reserved[0]].ref_count == 0
    assert reserved[0] in block_manager.free_block_ids
    assert block_manager.blocks[reserved[1]].ref_count == 0
    assert reserved[1] in block_manager.free_block_ids
    h0 = block_manager.compute_hash([1, 2, 3, 4], -1)
    assert block_manager.blocks[original_block].hash == h0
    assert block_manager.blocks[original_block].token_ids == [1, 2, 3, 4]
    assert block_manager.hash_to_block_id[h0] == original_block


def test_commit_accepted_tokens_zero_accept_releases_all_reserved_blocks():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3, 4])
    block_manager.allocate(seq)
    original_block_table = list(seq.block_table)
    original_token_ids = list(seq.token_ids)

    reserved = block_manager.reserve_append_blocks(seq, 5)
    assert len(reserved) == 2

    block_manager.commit_accepted_tokens(seq, [], reserved)

    assert seq.token_ids == original_token_ids
    assert seq.last_token == original_token_ids[-1]
    assert seq.num_tokens == len(original_token_ids)
    assert seq.block_table == original_block_table
    for block_id in reserved:
        assert block_manager.blocks[block_id].ref_count == 0
        assert block_id in block_manager.free_block_ids


def test_commit_accepted_tokens_publishes_multiple_full_block_hashes():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)

    reserved = block_manager.reserve_append_blocks(seq, 9)
    block_manager.commit_accepted_tokens(seq, [4, 5, 6, 7, 8, 9, 10, 11, 12], reserved)

    h0 = block_manager.compute_hash([1, 2, 3, 4], -1)
    h1 = block_manager.compute_hash([5, 6, 7, 8], h0)
    h2 = block_manager.compute_hash([9, 10, 11, 12], h1)
    assert seq.token_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert len(seq.block_table) == 3
    assert block_manager.hash_to_block_id[h0] == seq.block_table[0]
    assert block_manager.hash_to_block_id[h1] == seq.block_table[1]
    assert block_manager.blocks[seq.block_table[0]].token_ids == [1, 2, 3, 4]
    assert block_manager.blocks[seq.block_table[1]].token_ids == [5, 6, 7, 8]
    assert block_manager.blocks[seq.block_table[2]].hash == -1

    block_manager.may_append(seq)

    assert block_manager.hash_to_block_id[h2] == seq.block_table[2]
    assert block_manager.blocks[seq.block_table[2]].token_ids == [9, 10, 11, 12]


def test_commit_accepted_tokens_keeps_scheduler_hash_state_after_crossing_boundary():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)

    reserved = block_manager.reserve_append_blocks(seq, 2)
    block_manager.commit_accepted_tokens(seq, [4, 5], reserved)
    block_manager.may_append(seq)

    assert seq.token_ids == [1, 2, 3, 4, 5]
    assert len(seq.block_table) == 2
    assert block_manager.blocks[seq.block_table[0]].hash != -1
    assert block_manager.blocks[seq.block_table[1]].hash == -1


def test_sam_originated_acceptance_crosses_block_boundary():
    reset_sequence_state()
    block_size = 4
    block_manager = BlockManager(num_blocks=8, block_size=block_size)
    prompt_tokens = [1, 2, 3]
    repeated_verified_prefix = [4, 5, 6, 7, 1, 2]
    index = SuffixAutomatonDraftIndex(
        prompt_tokens + repeated_verified_prefix
    )
    draft = index.propose(max_draft_tokens=16)
    current_block_offset = len(prompt_tokens) % block_size
    assert len(draft.tokens) > block_size - current_block_offset

    seq = make_seq(prompt_tokens)
    block_manager.allocate(seq)
    reserved = block_manager.reserve_append_blocks(
        seq,
        len(draft.tokens) + block_size,
    )
    expected_target_prefix = list(draft.tokens)
    block_manager.commit_accepted_tokens(
        seq,
        expected_target_prefix,
        reserved,
    )
    committed_tokens = seq.token_ids[len(prompt_tokens):]

    assert committed_tokens == expected_target_prefix
    assert len(seq.block_table) > 1
    assert block_manager.blocks[seq.block_table[0]].token_ids == seq.block(0)
    adopted = set(seq.block_table) & set(reserved)
    unused = set(reserved) - adopted
    assert adopted
    assert unused
    assert all(block_manager.blocks[block_id].ref_count == 1 for block_id in adopted)
    assert all(block_manager.blocks[block_id].ref_count == 0 for block_id in unused)
    assert all(block_id in block_manager.free_block_ids for block_id in unused)
    assert len(committed_tokens) > block_size - current_block_offset


def test_commit_accepted_tokens_leaves_just_filled_last_block_for_scheduler_publish():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    seq = make_seq([1, 2, 3])
    block_manager.allocate(seq)

    reserved = block_manager.reserve_append_blocks(seq, 1)
    block_manager.commit_accepted_tokens(seq, [4], reserved)
    block_manager.may_append(seq)

    assert seq.token_ids == [1, 2, 3, 4]
    assert len(seq.block_table) == 1
    assert block_manager.blocks[seq.block_table[0]].hash != -1


def test_max_consecutive_prefill_chunks_yields_to_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_max_consecutive_chunks=2,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(16), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    scheduler.postprocess(seqs, [123], is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False


def test_mixed_prefill_decode_schedules_prefill_chunk_with_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(10), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert seqs == [long_prefill, running]
    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert long_prefill.step_is_decode is False
    assert running.step_is_decode is True
    assert long_prefill.prefill_chunk_start == 0
    assert long_prefill.prefill_chunk_end == 4
    assert long_prefill.prefill_chunk_final is False
    assert list(scheduler.running) == []


def test_mixed_short_prefill_batching_reserves_slot_for_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    short_prompts = [make_seq([i, i + 1, i + 2, i + 3], max_tokens=4) for i in range(0, 16, 4)]
    for seq in short_prompts:
        scheduler.add(seq)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert running in seqs
    assert seqs[-1] == running
    assert len([seq for seq in seqs if not seq.step_is_decode]) == 3
    assert list(scheduler.waiting) == [short_prompts[-1]]


def test_mixed_prefill_reserves_token_budget_for_decode_queries():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=12,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    short_prompts = [make_seq([i, i + 1, i + 2, i + 3], max_tokens=4) for i in range(0, 12, 4)]
    for seq in short_prompts:
        scheduler.add(seq)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert seqs == [short_prompts[0], short_prompts[1], running]
    assert sum(seq.prefill_chunk_end - seq.prefill_chunk_start for seq in seqs if not seq.step_is_decode) == 8
    assert len([seq for seq in seqs if seq.step_is_decode]) == 1
    assert list(scheduler.waiting) == [short_prompts[2]]


def test_mixed_decode_rows_respect_remaining_token_budget():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=9,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running_seqs = []
    for offset in (80, 90):
        running = make_seq([offset, offset + 1, offset + 2, offset + 3], max_tokens=4)
        scheduler.block_manager.allocate(running)
        running.append_token(offset + 4)
        running.status = SequenceStatus.RUNNING
        running.num_computed_tokens = len(running)
        scheduler.running.append(running)
        running_seqs.append(running)
    short_prompts = [make_seq([i, i + 1, i + 2, i + 3], max_tokens=4) for i in range(0, 8, 4)]
    for seq in short_prompts:
        scheduler.add(seq)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    prefill_tokens = sum(seq.prefill_chunk_end - seq.prefill_chunk_start for seq in seqs if not seq.step_is_decode)
    decode_rows = [seq for seq in seqs if seq.step_is_decode]
    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert prefill_tokens + len(decode_rows) <= scheduler.max_num_batched_tokens
    assert seqs == [short_prompts[0], short_prompts[1], running_seqs[0]]
    assert list(scheduler.running) == [running_seqs[1]]


def test_mixed_first_prefill_chunk_shrinks_to_leave_decode_budget():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_batched_tokens=5,
        max_num_prefill_tokens_per_step=8,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(12), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert seqs == [long_prefill, running]
    assert long_prefill.prefill_chunk_start == 0
    assert long_prefill.prefill_chunk_end == 4
    assert long_prefill.step_do_sample is False
    assert running.step_is_decode is True


def test_mixed_min_prompt_tokens_defers_short_waiting_prompt_to_decode():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
        chunked_prefill_mixed_min_prompt_tokens=8,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    short_prefill = make_seq([1, 2, 3, 4], max_tokens=4)
    scheduler.add(short_prefill)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True
    assert list(scheduler.waiting) == [short_prefill]


def test_mixed_min_prompt_tokens_still_admits_long_waiting_prompt():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
        chunked_prefill_mixed_min_prompt_tokens=8,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(10), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()

    assert seqs == [long_prefill, running]
    assert is_prefill is True
    assert do_sample is True
    assert batch_kind == "mixed"
    assert long_prefill.prefill_chunk_start == 0
    assert long_prefill.prefill_chunk_end == 4


def test_mixed_postprocess_commits_prefill_and_appends_decode_only_for_intermediate_chunk():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(10), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()
    # Intermediate prefill chunks do not need a sampled token; mixed sampling
    # should only return tokens for rows whose step_do_sample is True.
    scheduler.postprocess(seqs, [123], is_prefill, do_sample, batch_kind)

    assert long_prefill.completion_token_ids == []
    assert long_prefill.num_computed_tokens == 4
    assert long_prefill.status == SequenceStatus.PREFILLING
    assert running.completion_token_ids == [94, 123]
    assert running.status == SequenceStatus.RUNNING
    assert list(scheduler.prefilling) == [long_prefill]
    assert list(scheduler.running) == [running]


def test_mixed_final_prefill_chunk_and_decode_consume_tokens_in_sequence_order():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=4,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
    ))
    running_a = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running_a)
    running_a.append_token(94)
    running_a.status = SequenceStatus.RUNNING
    running_a.num_computed_tokens = len(running_a)
    scheduler.running.append(running_a)
    running_b = make_seq([80, 81, 82, 83], max_tokens=4)
    scheduler.block_manager.allocate(running_b)
    running_b.append_token(84)
    running_b.status = SequenceStatus.RUNNING
    running_b.num_computed_tokens = len(running_b)
    scheduler.running.append(running_b)
    final_prefill = make_seq([1, 2, 3, 4], max_tokens=4)
    scheduler.add(final_prefill)

    seqs, is_prefill, do_sample, batch_kind = scheduler.schedule()
    assert seqs == [final_prefill, running_a, running_b]
    scheduler.postprocess(seqs, [111, 222, 333], is_prefill, do_sample, batch_kind)

    assert final_prefill.completion_token_ids == [111]
    assert running_a.completion_token_ids == [94, 222]
    assert running_b.completion_token_ids == [84, 333]
    assert list(scheduler.running) == [final_prefill, running_a, running_b]


def test_mixed_prefill_fallback_counts_toward_consecutive_prefill_limit():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_seqs=1,
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=True,
        chunked_prefill_max_consecutive_chunks=1,
    ))
    running = make_seq([90, 91, 92, 93], max_tokens=4)
    scheduler.block_manager.allocate(running)
    running.append_token(94)
    running.status = SequenceStatus.RUNNING
    running.num_computed_tokens = len(running)
    scheduler.running.append(running)
    long_prefill = make_seq(range(12), max_tokens=4)
    scheduler.add(long_prefill)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [long_prefill]
    assert is_prefill is True
    assert do_sample is False
    scheduler.postprocess(seqs, None, is_prefill, do_sample)

    seqs, is_prefill, do_sample = scheduler.schedule()
    assert seqs == [running]
    assert is_prefill is False
    assert do_sample is True


def test_sequence_pickle_preserves_mixed_step_metadata_for_tp_workers():
    reset_sequence_state()
    seq = make_seq([1, 2, 3, 4], max_tokens=4)
    seq.step_is_decode = True
    seq.step_do_sample = False
    seq.num_computed_tokens = 4
    seq.prefill_chunk_start = 3
    seq.prefill_chunk_end = 4
    seq.prefill_chunk_final = True

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.step_is_decode is True
    assert restored.step_do_sample is False
    assert restored.num_computed_tokens == 4
    assert restored.prefill_chunk_start == 3
    assert restored.prefill_chunk_end == 4
    assert restored.prefill_chunk_final is True


def test_lm_head_prefill_uses_logits_indices_to_skip_unneeded_rows():
    if torch is None:
        return
    head = ParallelLMHead(4, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]))
    hidden = torch.tensor([
        [0.0, 0.0],
        [1.0, 2.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    set_context(True, cu_seqlens_q=torch.tensor([0, 2, 5, 6], dtype=torch.int32),
                logits_indices=torch.tensor([1, 5], dtype=torch.int64))

    logits = head(hidden)

    expected = torch.nn.functional.linear(hidden[[1, 5]], head.weight)
    assert logits.shape == (2, 4)
    assert torch.equal(logits, expected)
    reset_context_global()


def main():
    test_intermediate_chunk_does_not_sample_or_append()
    test_final_chunk_samples_once_and_moves_to_running()
    test_chunked_prefill_batches_multiple_short_final_prompts()
    test_chunked_prefill_batches_warm_prompt_by_uncached_tokens()
    test_decode_first_prioritizes_existing_running_sequence()
    test_add_rejects_request_beyond_max_model_len()
    test_add_rejects_prompt_beyond_logical_kv_capacity()
    test_add_accounts_for_decode_kv_capacity_boundary()
    test_add_capacity_boundary_for_multiple_logical_block_counts()
    test_chunked_prefill_progresses_with_varied_chunk_sizes()
    test_short_prefill_batch_respects_sequence_and_token_limits()
    test_chunked_prefill_does_not_publish_future_block_hashes()
    test_chunked_prefill_restores_reused_cached_block_metadata()
    test_max_reusable_tokens_keeps_one_sampleable_token()
    test_allocate_caps_exact_block_aligned_cache_hit()
    test_allocate_reuses_only_blocks_before_sampleable_suffix()
    test_can_allocate_excludes_live_prefix_hits_from_free_block_requirement()
    test_can_allocate_counts_idle_prefix_hits_as_free_block_requirement()
    test_estimate_admission_is_read_only_for_live_and_idle_hits()
    test_allocate_rejects_hash_collision_when_tokens_differ()
    test_clear_reusable_cache_preserves_live_block_metadata()
    test_capacity_pressure_never_returns_live_shared_block()
    test_normal_prefill_publishes_only_after_postprocess()
    test_normal_prefill_does_not_reuse_prefix_created_in_same_batch()
    test_normal_prefill_exact_block_warm_hit_recomputes_final_block()
    test_normal_prefill_warm_hit_reuses_only_complete_prefix_blocks()
    test_normal_prefill_token_budget_counts_only_uncached_tokens()
    test_normal_prefill_token_budget_still_limits_cold_prompts()
    test_chunked_prefill_uses_same_sampleable_prefix_cap()
    test_commit_accepted_tokens_appends_sequence_and_releases_unused_blocks()
    test_commit_accepted_tokens_zero_accept_releases_all_reserved_blocks()
    test_commit_accepted_tokens_publishes_multiple_full_block_hashes()
    test_commit_accepted_tokens_keeps_scheduler_hash_state_after_crossing_boundary()
    test_sam_originated_acceptance_crosses_block_boundary()
    test_commit_accepted_tokens_leaves_just_filled_last_block_for_scheduler_publish()
    test_max_consecutive_prefill_chunks_yields_to_decode()
    test_mixed_prefill_decode_schedules_prefill_chunk_with_decode()
    test_mixed_short_prefill_batching_reserves_slot_for_decode()
    test_mixed_prefill_reserves_token_budget_for_decode_queries()
    test_mixed_decode_rows_respect_remaining_token_budget()
    test_mixed_first_prefill_chunk_shrinks_to_leave_decode_budget()
    test_mixed_min_prompt_tokens_defers_short_waiting_prompt_to_decode()
    test_mixed_min_prompt_tokens_still_admits_long_waiting_prompt()
    test_mixed_postprocess_commits_prefill_and_appends_decode_only_for_intermediate_chunk()
    test_mixed_final_prefill_chunk_and_decode_consume_tokens_in_sequence_order()
    test_mixed_prefill_fallback_counts_toward_consecutive_prefill_limit()
    test_sequence_pickle_preserves_mixed_step_metadata_for_tp_workers()
    test_lm_head_prefill_uses_logits_indices_to_skip_unneeded_rows()
    print("chunked prefill tests passed")


if __name__ == "__main__":
    main()
