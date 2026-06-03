"""Chunked prefill scheduler/block-manager tests.

跑法：python3 tools/test_chunked_prefill.py
"""

import os
import sys
import types
import importlib.util
import hashlib
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


def main():
    test_intermediate_chunk_does_not_sample_or_append()
    test_final_chunk_samples_once_and_moves_to_running()
    test_decode_first_prioritizes_existing_running_sequence()
    test_chunked_prefill_does_not_publish_future_block_hashes()
    test_chunked_prefill_restores_reused_cached_block_metadata()
    test_max_consecutive_prefill_chunks_yields_to_decode()
    print("chunked prefill tests passed")


if __name__ == "__main__":
    main()
