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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_pkg = types.ModuleType("tinyvllm")
tinyvllm_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_pkg = types.ModuleType("tinyvllm.engine")
engine_pkg.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm", "engine")]
sys.modules.setdefault("tinyvllm", tinyvllm_pkg)
sys.modules.setdefault("tinyvllm.engine", engine_pkg)


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
sequence_mod = load_module("tinyvllm.engine.sequence", "tinyvllm/engine/sequence.py")
load_module("tinyvllm.engine.block_manager", "tinyvllm/engine/block_manager.py")
scheduler_mod = load_module("tinyvllm.engine.scheduler", "tinyvllm/engine/scheduler.py")

Sequence = sequence_mod.Sequence
SequenceStatus = sequence_mod.SequenceStatus
Scheduler = scheduler_mod.Scheduler
SamplingParams = sampling_mod.SamplingParams


def make_config(**overrides):
    cfg = dict(
        max_num_seqs=4,
        max_num_batched_tokens=64,
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


def main():
    test_intermediate_chunk_does_not_sample_or_append()
    test_final_chunk_samples_once_and_moves_to_running()
    test_chunked_prefill_batches_multiple_short_final_prompts()
    test_decode_first_prioritizes_existing_running_sequence()
    test_chunked_prefill_does_not_publish_future_block_hashes()
    test_chunked_prefill_restores_reused_cached_block_metadata()
    test_max_consecutive_prefill_chunks_yields_to_decode()
    test_mixed_prefill_decode_schedules_prefill_chunk_with_decode()
    test_mixed_short_prefill_batching_reserves_slot_for_decode()
    test_mixed_postprocess_commits_prefill_and_appends_decode_only_for_intermediate_chunk()
    test_mixed_final_prefill_chunk_and_decode_consume_tokens_in_sequence_order()
    test_mixed_prefill_fallback_counts_toward_consecutive_prefill_limit()
    test_sequence_pickle_preserves_mixed_step_metadata_for_tp_workers()
    print("chunked prefill tests passed")


if __name__ == "__main__":
    main()
