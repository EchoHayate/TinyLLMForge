from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
speculative_package = types.ModuleType("tinyvllm.speculative")
speculative_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "speculative")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.speculative", speculative_package)

from tinyvllm.speculative.runtime import (
    NativeSpeculativeStepError,
    NativeTailResult,
    execute_native_speculative_step,
)


class _Sequence:
    block_size = 4

    def __init__(
        self,
        token_ids=(1, 2, 3),
        *,
        max_tokens=16,
        ignore_eos=False,
    ):
        self.seq_id = 7
        self.token_ids = list(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.last_token = self.token_ids[-1]
        self.block_table = [10]
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos

    def __len__(self):
        return self.num_tokens

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    def append_token(self, token_id):
        self.token_ids.append(int(token_id))
        self.num_tokens += 1
        self.last_token = int(token_id)


class _BlockManager:
    def __init__(
        self,
        *,
        fail_mark=False,
        fail_commit=False,
        fail_rollback=False,
    ):
        self.fail_mark = fail_mark
        self.fail_commit = fail_commit
        self.fail_rollback = fail_rollback
        self.calls = []

    def begin_speculative_kv_transaction(
        self,
        seq,
        proposed_token_count,
    ):
        self.calls.append(("begin", proposed_token_count))
        materialized_end = len(seq) + max(
            0,
            proposed_token_count - 1,
        )
        required_blocks = (
            materialized_end + seq.block_size - 1
        ) // seq.block_size
        missing_blocks = max(
            0,
            required_blocks - len(seq.block_table),
        )
        return SimpleNamespace(
            reserved_block_ids=tuple(
                range(11, 11 + missing_blocks)
            ),
            state="reserved",
        )

    def mark_speculative_kv_materialized(
        self,
        transaction,
        materialized_token_count,
    ):
        self.calls.append(("mark", materialized_token_count))
        if self.fail_mark:
            raise RuntimeError("mark failed")
        transaction.state = "materialized"

    def commit_speculative_kv_transaction(
        self,
        transaction,
        seq,
        accepted_tokens,
    ):
        self.calls.append(("commit", tuple(accepted_tokens)))
        if self.fail_commit:
            raise RuntimeError("commit failed")
        materialized_end = len(seq) + max(
            0,
            len(accepted_tokens) - 1,
        )
        required_blocks = (
            materialized_end + seq.block_size - 1
        ) // seq.block_size
        missing_blocks = max(
            0,
            required_blocks - len(seq.block_table),
        )
        seq.block_table.extend(
            transaction.reserved_block_ids[:missing_blocks]
        )
        for token_id in accepted_tokens:
            seq.append_token(token_id)
        transaction.state = "committed"

    def rollback_speculative_kv_transaction(
        self,
        transaction,
        seq,
    ):
        self.calls.append(("rollback", transaction.state))
        if self.fail_rollback:
            raise RuntimeError("rollback failed")
        transaction.state = "rolled_back"


def test_runtime_orders_callbacks_and_commits_full_acceptance():
    manager = _BlockManager()
    sequence = _Sequence()
    callback_calls = []
    metadata = object()
    auxiliary = {"oracle": True}

    def first_target():
        callback_calls.append("first")
        return 4

    def prepare_tail(plan, proxy_block_table):
        callback_calls.append(
            (
                "prepare",
                plan.input_tokens,
                tuple(proxy_block_table),
            )
        )
        return "prepared"

    def run_tail(prepared):
        callback_calls.append(("tail", prepared))
        return NativeTailResult(
            target_tokens=(5, 6),
            metadata=metadata,
            auxiliary=auxiliary,
        )

    result = execute_native_speculative_step(
        block_manager=manager,
        seq=sequence,
        draft_tokens=[4, 5, 6],
        eos_token=99,
        run_first_target=first_target,
        prepare_tail=prepare_tail,
        run_tail=run_tail,
    )

    assert callback_calls == [
        "first",
        ("prepare", (4, 5), (10, 11)),
        ("tail", "prepared"),
    ]
    assert manager.calls == [
        ("begin", 3),
        ("mark", 2),
        ("commit", (4, 5, 6)),
    ]
    assert result.target_tokens == (4, 5, 6)
    assert result.accepted_tokens == (4, 5, 6)
    assert result.greedy_accepted_count == 3
    assert result.reserved_blocks == (11,)
    assert result.proxy_block_table == (10, 11)
    assert result.committed_blocks == (11,)
    assert result.released_blocks == ()
    assert result.tail_metadata is metadata
    assert result.tail_auxiliary is auxiliary
    assert set(result.timing_ms) == {
        "reserve_blocks_ms",
        "decode_first_target_ms",
        "verify_prepare_ms",
        "target_forward_ms",
        "kv_materialize_ms",
        "accept_sample_ms",
        "commit_metadata_ms",
    }
    assert all(
        value >= 0.0
        for value in result.timing_ms.values()
    )


def test_runtime_k1_skips_tail_callbacks_and_marks_zero():
    manager = _BlockManager()
    sequence = _Sequence()

    result = execute_native_speculative_step(
        block_manager=manager,
        seq=sequence,
        draft_tokens=[4],
        eos_token=99,
        run_first_target=lambda: 4,
        prepare_tail=lambda *args: pytest.fail("prepare called"),
        run_tail=lambda *args: pytest.fail("tail called"),
    )

    assert result.accepted_tokens == (4,)
    assert result.plan.query_len == 0
    assert result.proxy_block_table == (10,)
    assert manager.calls == [
        ("begin", 1),
        ("mark", 0),
        ("commit", (4,)),
    ]


@pytest.mark.parametrize(
    (
        "draft_tokens",
        "first_target",
        "tail_targets",
        "max_tokens",
        "eos_token",
        "expected",
        "eos_truncated",
        "budget_truncated",
    ),
    (
        ([4, 5, 6], 9, (5, 6), 16, 99, (), False, False),
        ([4, 5, 6], 4, (9, 6), 16, 99, (4,), False, False),
        ([4, 5, 6], 4, (5, 9), 16, 99, (4, 5), False, False),
        ([4, 99, 6], 4, (99, 6), 16, 99, (4, 99), True, False),
        ([4, 5, 6], 4, (5, 6), 2, 99, (4, 5), False, True),
    ),
)
def test_runtime_acceptance_and_truncation(
    draft_tokens,
    first_target,
    tail_targets,
    max_tokens,
    eos_token,
    expected,
    eos_truncated,
    budget_truncated,
):
    manager = _BlockManager()
    sequence = _Sequence(max_tokens=max_tokens)

    result = execute_native_speculative_step(
        block_manager=manager,
        seq=sequence,
        draft_tokens=draft_tokens,
        eos_token=eos_token,
        run_first_target=lambda: first_target,
        prepare_tail=lambda plan, proxy: None,
        run_tail=lambda prepared: NativeTailResult(
            target_tokens=tail_targets,
        ),
    )

    assert result.accepted_tokens == expected
    assert result.eos_truncated is eos_truncated
    assert result.output_budget_truncated is budget_truncated
    assert sequence.token_ids == [1, 2, 3] + list(expected)


@pytest.mark.parametrize(
    ("failure_phase", "expected_phase", "expected_state"),
    (
        ("first", "first_target_decode", "reserved"),
        ("prepare", "verify_prepare", "reserved"),
        ("tail", "tail_forward", "reserved"),
        ("mark", "kv_materialize", "reserved"),
        ("commit", "metadata_commit", "materialized"),
    ),
)
def test_runtime_rolls_back_with_phase(
    failure_phase,
    expected_phase,
    expected_state,
):
    manager = _BlockManager(
        fail_mark=failure_phase == "mark",
        fail_commit=failure_phase == "commit",
    )
    sequence = _Sequence()

    def first_target():
        if failure_phase == "first":
            raise RuntimeError("first failed")
        return 4

    def prepare_tail(plan, proxy):
        if failure_phase == "prepare":
            raise RuntimeError("prepare failed")
        return None

    def run_tail(prepared):
        if failure_phase == "tail":
            raise RuntimeError("tail failed")
        return NativeTailResult(target_tokens=(5, 6))

    with pytest.raises(NativeSpeculativeStepError) as error:
        execute_native_speculative_step(
            block_manager=manager,
            seq=sequence,
            draft_tokens=[4, 5, 6],
            eos_token=99,
            run_first_target=first_target,
            prepare_tail=prepare_tail,
            run_tail=run_tail,
        )

    assert error.value.phase == expected_phase
    assert error.value.rollback_error is None
    assert manager.calls[-1] == ("rollback", expected_state)
    assert sequence.token_ids == [1, 2, 3]


def test_runtime_preserves_original_and_rollback_failures():
    manager = _BlockManager(fail_rollback=True)

    with pytest.raises(NativeSpeculativeStepError) as error:
        execute_native_speculative_step(
            block_manager=manager,
            seq=_Sequence(),
            draft_tokens=[4, 5],
            eos_token=99,
            run_first_target=lambda: 4,
            prepare_tail=lambda plan, proxy: None,
            run_tail=lambda prepared: (_ for _ in ()).throw(
                RuntimeError("tail failed")
            ),
        )

    assert "tail failed" in str(error.value.cause)
    assert "rollback failed" in str(error.value.rollback_error)


def test_runtime_rejects_invalid_tail_targets_and_rolls_back():
    manager = _BlockManager()

    with pytest.raises(NativeSpeculativeStepError) as error:
        execute_native_speculative_step(
            block_manager=manager,
            seq=_Sequence(),
            draft_tokens=[4, 5, 6],
            eos_token=99,
            run_first_target=lambda: 4,
            prepare_tail=lambda plan, proxy: None,
            run_tail=lambda prepared: NativeTailResult(
                target_tokens=(5,),
            ),
        )

    assert error.value.phase == "tail_forward"
    assert "target count" in str(error.value.cause)
    assert manager.calls[-1][0] == "rollback"
