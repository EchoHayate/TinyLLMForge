from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


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
    package.__path__ = [
        str(ROOT / package_name.replace(".", "/"))
    ]
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

_load_module(
    "tinyvllm.sampling_params",
    "tinyvllm/sampling_params.py",
)
_load_module(
    "tinyvllm.engine.sequence",
    "tinyvllm/engine/sequence.py",
)
_load_module(
    "tinyvllm.engine.block_manager",
    "tinyvllm/engine/block_manager.py",
)
hybrid_state_module = types.ModuleType(
    "tinyvllm.engine.hybrid_state"
)


class _HybridStateLease:
    pass


class _HybridStateSlotAllocator:
    pass


hybrid_state_module.HybridStateLease = _HybridStateLease
hybrid_state_module.HybridStateSlotAllocator = (
    _HybridStateSlotAllocator
)
sys.modules["tinyvllm.engine.hybrid_state"] = hybrid_state_module
selection_module = _load_module(
    "tinyvllm.engine.speculative_selection",
    "tinyvllm/engine/speculative_selection.py",
)
scheduler_module = _load_module(
    "tinyvllm.engine.scheduler",
    "tinyvllm/engine/scheduler.py",
)

SpeculativeSelectionConfig = (
    selection_module.SpeculativeSelectionConfig
)
Scheduler = scheduler_module.Scheduler


class _Sequence:
    def __init__(
        self,
        sequence_id,
        *,
        completion_tokens=1,
        max_tokens=8,
        step_is_decode=False,
        step_do_sample=True,
        temperature=0.0,
    ):
        self.seq_id = sequence_id
        self.num_tokens = 8
        self.num_completion_tokens = completion_tokens
        self.max_tokens = max_tokens
        self.step_is_decode = step_is_decode
        self.step_do_sample = step_do_sample
        self.temperature = temperature


def _config():
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=64,
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
        num_kvcache_blocks=16,
        kvcache_block_size=2,
    )


def test_scheduler_selection_is_default_off_and_preserves_tuple():
    scheduler = Scheduler(_config())
    sequence = _Sequence(7)
    scheduled = ([sequence], False, True)

    returned = scheduler._return_schedule(
        scheduled,
        "decode_fixture",
    )

    assert returned is scheduled
    assert len(returned) == 3
    assert scheduler.schedule_generation == 1
    record = scheduler.last_speculative_selection
    assert record.schedule_generation == 1
    assert record.policy_branch == "decode_fixture"
    assert record.scheduled_sequence_ids == (7,)
    assert record.selected_rows == ()
    assert record.rows[0].suppression_reason == "disabled"


def test_scheduler_selection_installation_is_idempotent():
    scheduler = Scheduler(_config())
    config = SpeculativeSelectionConfig(
        enabled=True,
        max_proposal_tokens=4,
    )

    scheduler.install_speculative_selection(config)
    scheduler.install_speculative_selection(config)

    with pytest.raises(RuntimeError, match="already installed"):
        scheduler.install_speculative_selection(
            SpeculativeSelectionConfig(
                enabled=True,
                max_proposal_tokens=8,
            )
        )


def test_scheduler_publishes_enabled_decode_selection_once_per_return():
    scheduler = Scheduler(_config())
    scheduler.install_speculative_selection(
        SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=4,
        )
    )
    first_seq = _Sequence(8)
    first = ([first_seq], False, True)
    second_seq = _Sequence(4)
    second = ([second_seq], False, True, "mixed")
    second_seq.step_is_decode = True

    assert scheduler._return_schedule(first, "decode") is first
    first_record = scheduler.last_speculative_selection
    assert first_record.schedule_generation == 1
    assert first_record.selected_sequence_ids == (8,)

    assert scheduler._return_schedule(second, "mixed") is second
    second_record = scheduler.last_speculative_selection
    assert scheduler.schedule_generation == 2
    assert second_record.schedule_generation == 2
    assert second_record.batch_kind == "mixed"
    assert second_record.selected_sequence_ids == (4,)
    assert scheduler.last_policy_branch == "mixed"


def test_scheduler_mixed_record_preserves_all_rows_and_tuple_shape():
    scheduler = Scheduler(_config())
    scheduler.install_speculative_selection(
        SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=4,
        )
    )
    prefill = _Sequence(1, step_is_decode=False)
    decode = _Sequence(2, step_is_decode=True)
    scheduled = ([prefill, decode], True, True, "mixed")

    returned = scheduler._return_schedule(
        scheduled,
        "mixed_fixture",
    )

    assert returned is scheduled
    assert len(returned) == 4
    assert tuple(
        row.sequence_id
        for row in scheduler.last_speculative_selection.rows
    ) == (1, 2)
    assert (
        scheduler.last_speculative_selection
        .selected_sequence_ids
    ) == (2,)


def test_scheduler_suppresses_non_greedy_decode_before_runtime():
    scheduler = Scheduler(_config())
    scheduler.install_speculative_selection(
        SpeculativeSelectionConfig(
            enabled=True,
            max_proposal_tokens=4,
        )
    )
    sequence = _Sequence(7, temperature=0.7)
    scheduled = ([sequence], False, True)

    assert scheduler._return_schedule(
        scheduled,
        "decode_non_greedy",
    ) is scheduled
    record = scheduler.last_speculative_selection
    assert record.selected_rows == ()
    assert record.rows[0].temperature_snapshot == 0.7
    assert record.rows[0].suppression_reason == "non_greedy"
