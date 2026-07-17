"""n-gram speculative decoding helpers tests.

跑法：python tools/test_ngram_speculative.py
"""

import os
import sys
import importlib.util
import types
from enum import Enum

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_NGRAM_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "speculative", "ngram.py")
_SPEC = importlib.util.spec_from_file_location("ngram_under_test", _NGRAM_PATH)
ngram = importlib.util.module_from_spec(_SPEC)
sys.modules["ngram_under_test"] = ngram
_SPEC.loader.exec_module(ngram)

_DRAFT_SCHEMA_PATH = os.path.join(_REPO_ROOT, "tools", "draft_model_schema.py")
_DRAFT_SCHEMA_SPEC = importlib.util.spec_from_file_location("draft_model_schema", _DRAFT_SCHEMA_PATH)
draft_model_schema = importlib.util.module_from_spec(_DRAFT_SCHEMA_SPEC)
sys.modules["draft_model_schema"] = draft_model_schema
_DRAFT_SCHEMA_SPEC.loader.exec_module(draft_model_schema)

_PROFILE_PATH = os.path.join(_REPO_ROOT, "tools", "profile_ngram_commit.py")
_PROFILE_SPEC = importlib.util.spec_from_file_location("profile_ngram_under_test", _PROFILE_PATH)
profile_ngram = importlib.util.module_from_spec(_PROFILE_SPEC)
sys.modules["profile_ngram_under_test"] = profile_ngram
_PROFILE_SPEC.loader.exec_module(profile_ngram)

propose_ngram_draft = ngram.propose_ngram_draft
replay_ngram_acceptance = ngram.replay_ngram_acceptance
summarize_replay_stats = ngram.summarize_replay_stats
NGramOnlineDryRunState = ngram.NGramOnlineDryRunState
NGramOnlineDryRunTotals = ngram.NGramOnlineDryRunTotals
ngram_online_dry_run_step = ngram.ngram_online_dry_run_step
summarize_online_dry_run_totals = ngram.summarize_online_dry_run_totals
count_accepted_prefix = ngram.count_accepted_prefix
NGramTargetVerifyStats = ngram.NGramTargetVerifyStats
summarize_target_verify_stats = ngram.summarize_target_verify_stats
AdaptiveDraftState = ngram.AdaptiveDraftState
update_adaptive_draft_state = ngram.update_adaptive_draft_state
propose_draft = profile_ngram.propose_draft
validate_profile_args = profile_ngram.validate_profile_args
SuffixAutomatonDraftIndex = profile_ngram.SuffixAutomatonDraftIndex
sync_sam_index = profile_ngram.sync_sam_index
should_verify_draft = profile_ngram.should_verify_draft
rematerialize_accepted_kv = profile_ngram.rematerialize_accepted_kv
DraftModelInput = draft_model_schema.DraftModelInput
DraftModelContract = draft_model_schema.DraftModelContract
DraftModelResult = draft_model_schema.DraftModelResult
DraftModelStubConfig = draft_model_schema.DraftModelStubConfig
validate_draft_model_contract = draft_model_schema.validate_draft_model_contract
run_draft_model_stub = profile_ngram.run_draft_model_stub
summarize_hidden_to_draft_stub = profile_ngram.summarize_hidden_to_draft_stub
build_verify_tail_plan = profile_ngram._build_verify_tail_plan
verify_and_commit_block = profile_ngram.verify_and_commit_block


def _install_native_test_context():
    current = types.SimpleNamespace(mode="decode")

    def set_context(*args, mode=None, **kwargs):
        current.mode = mode or ("prefill" if args and args[0] else "decode")

    def reset_context():
        current.mode = "decode"

    context_module = types.ModuleType("tinyvllm.utils.context")
    context_module.get_context = lambda: current
    context_module.set_context = set_context
    context_module.reset_context = reset_context
    tinyvllm_module = sys.modules.setdefault(
        "tinyvllm",
        types.ModuleType("tinyvllm"),
    )
    tinyvllm_module.__path__ = []
    utils_module = sys.modules.setdefault(
        "tinyvllm.utils",
        types.ModuleType("tinyvllm.utils"),
    )
    utils_module.__path__ = []
    sys.modules["tinyvllm.utils.context"] = context_module
    return context_module


native_test_context = _install_native_test_context()


def _base_draft_model_metadata(metadata: dict) -> dict:
    return {
        key: value for key, value in metadata.items()
        if key not in ("input_schema", "contract")
    }


def test_verify_tail_plan_keeps_pending_token_on_decode_path():
    assert build_verify_tail_plan(history_len=52, draft_tokens=[10]) == {
        "input_tokens": [],
        "slot_positions": [],
        "positions": [],
        "kv_tokens": 52,
    }
    assert build_verify_tail_plan(history_len=52, draft_tokens=[10, 20, 30, 40]) == {
        "input_tokens": [10, 20, 30],
        "slot_positions": [52, 53, 54],
        "positions": [53, 54, 55],
        "kv_tokens": 55,
    }


def test_propose_ngram_draft_uses_latest_matching_suffix():
    # suffix [1, 2] appears twice before the end; use the latest previous match.
    draft = propose_ngram_draft([1, 2, 3, 1, 2, 4, 1, 2], ngram_size=2, max_draft_tokens=2)

    assert draft.tokens == [4, 1]
    assert draft.match_start == 3
    assert draft.ngram_size == 2


def test_propose_ngram_draft_respects_max_draft_tokens():
    draft = propose_ngram_draft([7, 8, 1, 2, 3, 4, 7, 8], ngram_size=2, max_draft_tokens=2)

    assert draft.tokens == [1, 2]


def test_propose_ngram_draft_returns_empty_without_match():
    draft = propose_ngram_draft([1, 2, 3, 4], ngram_size=2, max_draft_tokens=4)

    assert draft.tokens == []
    assert draft.match_start == -1


def test_replay_ngram_acceptance_counts_accepted_prefix_only():
    # At pos=6, suffix [1, 2] drafts [3, 4], future is [3, 9], so only one token is accepted.
    stats = replay_ngram_acceptance([1, 2, 3, 4, 1, 2, 3], prompt_len=6, ngram_size=2, max_draft_tokens=2)

    assert stats.positions == 1
    assert stats.draft_events == 1
    assert stats.drafted_tokens == 2
    assert stats.accepted_tokens == 1
    assert stats.acceptance_rate == 0.5
    assert stats.avg_draft_len == 2.0


def test_summarize_replay_stats_is_json_friendly():
    stats = replay_ngram_acceptance([1, 2, 3, 1, 2, 3], prompt_len=3, ngram_size=2, max_draft_tokens=1)

    summary = summarize_replay_stats(stats)


    assert summary == {
        "positions": 3,
        "draft_events": 1,
        "drafted_tokens": 1,
        "accepted_tokens": 1,
        "acceptance_rate": 1.0,
        "avg_draft_len": 1.0,
    }


def test_online_dry_run_accepts_pending_prefix_across_steps():
    state = NGramOnlineDryRunState(pending_tokens=[])
    totals = NGramOnlineDryRunTotals()

    first = ngram_online_dry_run_step([1, 2, 3, 4, 1, 2], 3, state, totals, ngram_size=2, max_draft_tokens=2)
    second = ngram_online_dry_run_step([1, 2, 3, 4, 1, 2, 3], 4, state, totals, ngram_size=2, max_draft_tokens=2)

    assert first["proposed"] is True
    assert first["draft_tokens"] == [3, 4]
    assert first["accepted"] is True
    assert first["pending_after"] == 1
    assert second["proposed"] is False
    assert second["accepted"] is True
    assert second["completed"] is True
    assert summarize_online_dry_run_totals(totals) == {
        "decode_positions": 2,
        "draft_events": 1,
        "drafted_tokens": 2,
        "accepted_tokens": 2,
        "rejected_events": 0,
        "completed_drafts": 1,
        "no_draft_positions": 0,
        "acceptance_rate": 1.0,
        "avg_draft_len": 2.0,
        "draft_coverage": 0.5,
        "theoretical_decode_step_reduction": 0.5,
    }


def test_online_dry_run_rejects_and_clears_pending_tokens():
    state = NGramOnlineDryRunState(pending_tokens=[])
    totals = NGramOnlineDryRunTotals()

    event = ngram_online_dry_run_step([1, 2, 3, 4, 1, 2], 9, state, totals, ngram_size=2, max_draft_tokens=2)

    assert event["proposed"] is True
    assert event["rejected"] is True
    assert event["expected_token"] == 3
    assert state.pending_tokens == []
    assert totals.decode_positions == 1
    assert totals.draft_events == 1
    assert totals.drafted_tokens == 2
    assert totals.accepted_tokens == 0
    assert totals.rejected_events == 1


def test_count_accepted_prefix_stops_at_first_mismatch():
    assert count_accepted_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert count_accepted_prefix([1, 2, 3], [9, 2, 3]) == 0
    assert count_accepted_prefix([1, 2, 3], [1]) == 1


def test_summarize_target_verify_stats_is_json_friendly():
    stats = NGramTargetVerifyStats(
        verify_events=2,
        verified_tokens=8,
        target_accepted_tokens=5,
        replay_accepted_tokens=5,
        mismatched_events=0,
        truncated_future_events=1,
    )

    assert summarize_target_verify_stats(stats) == {
        "verify_events": 2,
        "verified_tokens": 8,
        "target_accepted_tokens": 5,
        "replay_accepted_tokens": 5,
        "mismatched_events": 0,
        "truncated_future_events": 1,
        "target_acceptance_rate": 0.625,
        "replay_acceptance_rate": 0.625,
        "mismatch_rate": 0.0,
    }


def test_adaptive_draft_state_starts_at_k2():
    state = AdaptiveDraftState()

    assert state.levels == (1, 2, 4)
    assert state.selected_k == 2
    assert state.acceptance_ema == 0.5
    assert state.full_accept_streak == 0
    assert state.proposal_events == 0


def test_adaptive_two_strong_full_accepts_promote_and_saturate():
    state = AdaptiveDraftState()

    first = update_adaptive_draft_state(state, proposed=2, accepted=2)
    second = update_adaptive_draft_state(state, proposed=2, accepted=2)
    third = update_adaptive_draft_state(state, proposed=4, accepted=4)
    fourth = update_adaptive_draft_state(state, proposed=4, accepted=4)

    assert first["selected_k_before"] == 2
    assert first["selected_k_after"] == 2
    assert first["transition_reason"] == "full_accept_streak"
    assert second["selected_k_after"] == 4
    assert second["transition_reason"] == "promote"
    assert third["selected_k_after"] == 4
    assert fourth["selected_k_after"] == 4
    assert state.selected_k == 4


def test_adaptive_zero_accept_jumps_from_k4_to_k1():
    state = AdaptiveDraftState(level_index=2, acceptance_ema=0.9, full_accept_streak=1)

    event = update_adaptive_draft_state(state, proposed=4, accepted=0)

    assert event["selected_k_before"] == 4
    assert event["selected_k_after"] == 1
    assert event["transition_reason"] == "zero_accept"
    assert state.full_accept_streak == 0


def test_adaptive_weak_partial_accept_moves_down_one_level():
    state = AdaptiveDraftState(level_index=2, acceptance_ema=0.8)

    event = update_adaptive_draft_state(state, proposed=4, accepted=1)

    assert event["selected_k_after"] == 2
    assert event["transition_reason"] == "weak_acceptance"


def test_adaptive_weak_ema_demotes_k2_to_k1():
    state = AdaptiveDraftState(level_index=1, acceptance_ema=0.2)

    event = update_adaptive_draft_state(state, proposed=2, accepted=1)

    assert event["event_acceptance"] == 0.5
    assert event["acceptance_ema_after"] == 0.35
    assert event["selected_k_after"] == 1


def test_adaptive_partial_accept_resets_full_accept_streak():
    state = AdaptiveDraftState(full_accept_streak=1, acceptance_ema=0.9)

    event = update_adaptive_draft_state(state, proposed=2, accepted=1)

    assert event["selected_k_after"] == 2
    assert event["full_accept_streak_after"] == 0
    assert state.full_accept_streak == 0


def test_adaptive_rejects_invalid_counts_and_state():
    for proposed, accepted in ((0, 0), (-1, 0), (2, -1), (2, 3)):
        try:
            update_adaptive_draft_state(AdaptiveDraftState(), proposed, accepted)
        except ValueError:
            pass
        else:
            raise AssertionError((proposed, accepted))

    try:
        AdaptiveDraftState(levels=(1, 3, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid adaptive levels accepted")


def test_adaptive_transition_record_is_json_friendly_and_replayable():
    import json

    state = AdaptiveDraftState()
    event = update_adaptive_draft_state(state, proposed=1, accepted=1)

    assert json.loads(json.dumps(event)) == event
    assert event == {
        "levels": [1, 2, 4],
        "proposal_event": 1,
        "proposed_tokens": 1,
        "accepted_tokens": 1,
        "event_acceptance": 1.0,
        "acceptance_ema_before": 0.5,
        "acceptance_ema_after": 0.75,
        "full_accept_streak_before": 0,
        "full_accept_streak_after": 1,
        "selected_k_before": 2,
        "selected_k_after": 2,
        "transition_reason": "full_accept_streak",
        "promoted": False,
        "demoted": False,
    }


def test_propose_draft_dispatches_ngram_source():
    class Args:
        draft_source = "ngram"
        ngram_size = 2
        max_draft_tokens = 2

    draft = propose_draft([1, 2, 3, 1, 2, 4, 1, 2], Args())

    assert draft.source == "ngram"
    assert draft.tokens == [4, 1]
    assert draft.metadata == {"match_start": 3, "ngram_size": 2}


def test_propose_draft_accepts_per_event_cap_without_mutating_args():
    class Args:
        draft_source = "ngram"
        ngram_size = 2
        max_draft_tokens = 4

    draft = propose_draft(
        [1, 2, 3, 4, 1, 2],
        Args(),
        max_draft_tokens=1,
    )

    assert draft.tokens == [3]
    assert Args.max_draft_tokens == 4


def test_propose_draft_dispatches_fixed_sam_source():
    class Args:
        draft_source = "sam"
        draft_policy = "sam-fixed"
        ngram_size = 3
        max_draft_tokens = 16

    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = propose_draft(
        index.indexed_tokens,
        Args(),
        max_draft_tokens=16,
        sam_index=index,
    )
    assert draft.source == "sam"
    assert draft.tokens == [3, 1, 2]
    assert draft.metadata["selected_k"] == 16


def test_propose_draft_dispatches_match_aware_sam_bypass():
    class Args:
        draft_source = "sam"
        draft_policy = "sam-match-aware"
        ngram_size = 3
        max_draft_tokens = 16

    index = SuffixAutomatonDraftIndex([1, 9, 1])
    draft = propose_draft(index.indexed_tokens, Args(), sam_index=index)
    assert draft.tokens == []
    assert draft.metadata["selected_k"] == 0
    assert draft.metadata["bypass_reason"] == "no_usable_match"


def test_sam_profile_args_require_candidate_greedy_single_sequence():
    from types import SimpleNamespace

    valid = dict(
        model="/model",
        temperature=0.0,
        max_commit_events=0,
        warmup_output_len=0,
        simulate_kv_upload_mb=0.0,
        max_draft_tokens=16,
        draft_source="sam",
        draft_policy="sam-match-aware",
        mode="candidate-only",
        max_num_seqs=1,
    )
    validate_profile_args(SimpleNamespace(**valid))
    for override in (
        {"temperature": 0.7},
        {"mode": "paired"},
        {"max_num_seqs": 2},
        {"draft_source": "ngram"},
    ):
        args = SimpleNamespace(**{**valid, **override})
        try:
            validate_profile_args(args)
        except ValueError:
            pass
        else:
            raise AssertionError(override)


def test_sync_sam_index_extends_only_verified_history_tail():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    event = sync_sam_index(index, [1, 2, 3, 4, 5])
    assert index.indexed_tokens == [1, 2, 3, 4, 5]
    assert event["extended_tokens"] == [4, 5]
    assert event["runtime_mutation"] is False


def test_sync_sam_index_rejects_history_rewrite():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    try:
        sync_sam_index(index, [1, 9, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("history rewrite accepted")


def test_empty_sam_proposal_bypasses_verifier():
    class Args:
        draft_source = "sam"
        draft_policy = "sam-match-aware"
        ngram_size = 3
        max_draft_tokens = 16

    index = SuffixAutomatonDraftIndex([1, 9, 1])
    draft = propose_draft(index.indexed_tokens, Args(), sam_index=index)
    assert should_verify_draft(draft) is False


def test_sam_profiler_remains_profiler_owned():
    source = open(
        os.path.join(_REPO_ROOT, "tools", "profile_ngram_commit.py")
    ).read()
    assert "verify_and_commit_block(" in source
    assert '"runtime_mutation": False' in source
    assert "LLMEngine.step" not in source


def test_sam_verify_event_contract_is_profiler_owned():
    class Args:
        draft_source = "sam"
        draft_policy = "sam-fixed"
        ngram_size = 3
        max_draft_tokens = 16

    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = propose_draft(index.indexed_tokens, Args(), sam_index=index)
    event = {
        "draft_source": draft.source,
        "accepted_count": 2,
    }
    profile_ngram.attach_draft_policy_event(
        event,
        draft,
        selected_k=draft.selected_k
        if hasattr(draft, "selected_k")
        else draft.metadata["selected_k"],
        adaptive_state=None,
    )
    assert event["draft_source"] == "sam"
    assert event["draft_metadata"]["match_length"] >= 2
    assert event["runtime_mutation"] is False
    assert event["profiler_owned"] is True
    assert event["wasted_draft_tokens"] == (
        event["proposed_tokens"] - event["accepted_count"]
    )


def test_accepted_kv_rematerialization_uses_normal_decode_for_materialized_prefix():
    class FakeSequence:
        block_size = 4

        def __init__(self):
            self.token_ids = [1, 2, 3]
            self.block_table = [10]
            self.num_tokens = 3
            self.last_token = 3

        def append_token(self, token_id):
            self.token_ids.append(token_id)
            self.num_tokens += 1
            self.last_token = token_id

    class FakeModelRunner:
        def __init__(self):
            self.calls = []

        def run(self, seqs, is_prefill, do_sample):
            seq = seqs[0]
            self.calls.append({
                "token_ids": list(seq.token_ids),
                "block_table": list(seq.block_table),
                "is_prefill": is_prefill,
                "do_sample": do_sample,
            })

    class FakeLLM:
        def __init__(self):
            self.model_runner = FakeModelRunner()

    llm = FakeLLM()
    seq = FakeSequence()
    event = rematerialize_accepted_kv(
        llm,
        seq,
        accepted_tokens=[4, 5, 6],
        proxy_block_table=[10, 11],
    )

    assert [call["token_ids"] for call in llm.model_runner.calls] == [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    ]
    assert all(call["block_table"] == [10, 11] for call in llm.model_runner.calls)
    assert all(call["is_prefill"] is False for call in llm.model_runner.calls)
    assert all(call["do_sample"] is False for call in llm.model_runner.calls)
    assert seq.token_ids == [1, 2, 3]
    assert seq.block_table == [10]
    assert event["rematerialized_tokens"] == [4, 5]
    assert event["decode_calls"] == 2


def test_accepted_kv_rematerialization_skips_pending_only_token():
    class FailModelRunner:
        def run(self, *args, **kwargs):
            raise AssertionError("decode should not run")

    class FakeLLM:
        model_runner = FailModelRunner()

    class FakeSequence:
        token_ids = [1, 2, 3]
        block_table = [10]
        num_tokens = 3
        last_token = 3

    for accepted_tokens in ([], [4]):
        event = rematerialize_accepted_kv(
            FakeLLM(),
            FakeSequence(),
            accepted_tokens=accepted_tokens,
            proxy_block_table=[10],
        )
        assert event["rematerialized_tokens"] == []
        assert event["decode_calls"] == 0


class _FakeArgmax:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _FakeLogits:
    def __init__(self, target_tokens):
        self.target_tokens = list(target_tokens)

    def argmax(self, dim=-1):
        assert dim == -1
        return _FakeArgmax(self.target_tokens)

    def detach(self):
        return self

    def to(self, device=None, dtype=None):
        assert device == "cpu"
        return self

    def float(self):
        return self

    def tolist(self):
        return [
            [float(token_id), float(token_id) + 0.25]
            for token_id in self.target_tokens
        ]


class _NativeSequence:
    block_size = 4

    def __init__(self, token_ids, max_tokens=16, ignore_eos=False):
        self.token_ids = list(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.last_token = self.token_ids[-1]
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.block_table = [10]
        self.status = _NativeStatus.RUNNING

    def __len__(self):
        return self.num_tokens

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    def append_token(self, token_id):
        self.token_ids.append(int(token_id))
        self.num_tokens += 1
        self.last_token = int(token_id)


class _NativeBlockManager:
    def __init__(self, fail_commit=False):
        self.fail_commit = fail_commit
        self.reserve_calls = 0
        self.release_calls = []
        self.commit_calls = []
        self.deallocate_calls = []

    def reserve_append_blocks(self, seq, num_new_tokens):
        self.reserve_calls += 1
        final_len = len(seq) + num_new_tokens
        needed_blocks = (
            final_len + seq.block_size - 1
        ) // seq.block_size
        missing_blocks = max(0, needed_blocks - len(seq.block_table))
        return list(range(11, 11 + missing_blocks))

    def release_reserved_blocks(self, block_ids):
        self.release_calls.append(list(block_ids))

    def commit_accepted_tokens(self, seq, accepted_tokens, reserved_blocks):
        if self.fail_commit:
            raise RuntimeError("commit failure")
        self.commit_calls.append(
            (list(accepted_tokens), list(reserved_blocks))
        )
        if accepted_tokens:
            materialized_tokens = len(seq) + len(accepted_tokens) - 1
            needed_blocks = (
                materialized_tokens + seq.block_size - 1
            ) // seq.block_size
            missing = max(0, needed_blocks - len(seq.block_table))
            seq.block_table.extend(reserved_blocks[:missing])
            self.release_reserved_blocks(reserved_blocks[missing:])
            for token_id in accepted_tokens:
                seq.append_token(token_id)
        else:
            self.release_reserved_blocks(reserved_blocks)

    def deallocate(self, seq):
        self.deallocate_calls.append(seq)
        seq.block_table = []


class _NativeStatus(Enum):
    RUNNING = "running"
    FINISHED = "finished"


class _NativeModelRunner:
    def __init__(self, first_target, tail_targets, fail_tail=False):
        self.first_target = int(first_target)
        self.tail_targets = list(tail_targets)
        self.fail_tail = fail_tail
        self.kv_offload = None
        self.normal_decode_calls = 0
        self.spec_verify_calls = 0
        self.prepare_calls = []
        self.snapshot_calls = []

    def _validate_spec_verify_compatibility(self, **kwargs):
        return None

    def run(self, seqs, is_prefill):
        assert is_prefill is False
        self.normal_decode_calls += 1
        return [self.first_target]

    def prepare_spec_verify(
        self,
        seq,
        input_tokens,
        proxy_block_table,
        slot_positions,
    ):
        self.prepare_calls.append(
            {
                "input_tokens": list(input_tokens),
                "proxy_block_table": list(proxy_block_table),
                "slot_positions": list(slot_positions),
            }
        )
        metadata = type(
            "Metadata",
            (),
            {
                "query_len": len(input_tokens),
                "input_tokens": tuple(input_tokens),
                "positions": tuple(position + 1 for position in slot_positions),
                "logical_slots": tuple(slot_positions),
                "physical_slots": tuple(slot_positions),
                "context_len": slot_positions[-1] + 1,
                "block_table": tuple(proxy_block_table),
            },
        )()
        return object(), object(), metadata

    def run_model(
        self,
        input_ids,
        positions,
        is_prefill,
        execution_mode=None,
        return_hidden=False,
    ):
        assert is_prefill is False
        assert execution_mode == "spec_verify"
        assert return_hidden is False
        self.spec_verify_calls += 1
        if self.fail_tail:
            raise RuntimeError("tail failure")
        return _FakeLogits(self.tail_targets)

    def snapshot_kv_slots(self, physical_slots):
        self.snapshot_calls.append(list(physical_slots))
        return {
            "keys": _FakeLogits(physical_slots),
            "values": _FakeLogits(
                [slot + 100 for slot in physical_slots]
            ),
        }


def _native_verify_fixture(
    *,
    first_target,
    tail_targets,
    eos=99,
    max_tokens=16,
    fail_tail=False,
    fail_commit=False,
):
    sequence = _NativeSequence([1, 2, 3], max_tokens=max_tokens)
    block_manager = _NativeBlockManager(fail_commit=fail_commit)
    model_runner = _NativeModelRunner(
        first_target,
        tail_targets,
        fail_tail=fail_tail,
    )
    scheduler = type(
        "Scheduler",
        (),
        {
            "block_manager": block_manager,
            "eos": eos,
            "running": [sequence],
        },
    )()
    llm = type(
        "LLM",
        (),
        {
            "scheduler": scheduler,
            "model_runner": model_runner,
        },
    )()
    return llm, sequence, block_manager, model_runner


def test_native_verify_commits_without_decode_rematerialization():
    llm, seq, _, runner = _native_verify_fixture(
        first_target=4,
        tail_targets=[5, 6],
    )
    original = profile_ngram.rematerialize_accepted_kv
    profile_ngram.rematerialize_accepted_kv = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("native verifier must not rematerialize KV")
        )
    )
    try:
        event = verify_and_commit_block(
            llm,
            seq,
            [4, 5, 6],
            verifier_mode="native",
        )
    finally:
        profile_ngram.rematerialize_accepted_kv = original

    assert event["accepted_tokens"] == [4, 5, 6]
    assert event["verifier_mode"] == "native"
    assert event["query_len"] == 2
    assert event["accepted_kv_rematerialization"] == {
        "rematerialized_tokens": [],
        "decode_calls": 0,
        "elapsed_ms": 0.0,
    }
    assert event["timing_ms"]["accepted_kv_rematerialize_ms"] == 0.0
    assert event["accepted_kv_copy_calls"] == 0
    assert event["accepted_kv_replay_calls"] == 0
    assert runner.normal_decode_calls == 1
    assert runner.spec_verify_calls == 1
    assert runner.snapshot_calls == []


def test_native_oracle_evidence_captures_tail_logits_and_final_slot_kv():
    llm, seq, _, runner = _native_verify_fixture(
        first_target=4,
        tail_targets=[5, 6],
    )

    event = verify_and_commit_block(
        llm,
        seq,
        [4, 5, 6],
        verifier_mode="native",
        capture_oracle_evidence=True,
    )

    evidence = event["oracle_evidence"]
    assert evidence["logits"] == [[5.0, 5.25], [6.0, 6.25]]
    assert evidence["physical_slots"] == [3, 4]
    assert evidence["kv"] == {
        "keys": [[3.0, 3.25], [4.0, 4.25]],
        "values": [[103.0, 103.25], [104.0, 104.25]],
    }
    assert runner.snapshot_calls == [[3, 4]]


def test_native_k1_uses_first_target_without_tail_forward():
    llm, seq, _, runner = _native_verify_fixture(
        first_target=4,
        tail_targets=[],
    )

    event = verify_and_commit_block(
        llm,
        seq,
        [4],
        verifier_mode="native",
    )

    assert event["accepted_tokens"] == [4]
    assert event["query_len"] == 0
    assert runner.spec_verify_calls == 0
    assert runner.prepare_calls == []


def test_native_unsupported_mode_fails_before_reservation():
    llm, seq, block_manager, runner = _native_verify_fixture(
        first_target=4,
        tail_targets=[],
    )

    def fail_compatibility(**kwargs):
        raise RuntimeError("kv_offload_mvp0 is unsupported by spec_verify")

    runner._validate_spec_verify_compatibility = fail_compatibility
    try:
        verify_and_commit_block(
            llm,
            seq,
            [4],
            verifier_mode="native",
        )
    except RuntimeError as exc:
        assert "kv_offload_mvp0" in str(exc)
    else:
        raise AssertionError("unsupported native mode must fail")

    assert block_manager.reserve_calls == 0


def test_native_tail_failure_releases_reservation_and_resets_context():
    llm, seq, block_manager, _ = _native_verify_fixture(
        first_target=4,
        tail_targets=[5],
        fail_tail=True,
    )

    try:
        verify_and_commit_block(
            llm,
            seq,
            [4, 5],
            verifier_mode="native",
        )
    except RuntimeError as exc:
        assert "tail failure" in str(exc)
    else:
        raise AssertionError("tail failure must propagate")

    assert block_manager.release_calls == [[11]]
    assert native_test_context.get_context().mode == "decode"


def test_native_acceptance_matrix_preserves_pending_token_lifecycle():
    cases = (
        ("zero", 9, [5, 6], [], 0),
        ("one", 4, [9, 6], [4], 0),
        ("partial", 4, [5, 9], [4, 5], 0),
        ("full", 4, [5, 6], [4, 5, 6], 1),
    )

    for name, first_target, tail_targets, expected, committed_blocks in cases:
        llm, seq, _, _ = _native_verify_fixture(
            first_target=first_target,
            tail_targets=tail_targets,
        )
        event = verify_and_commit_block(
            llm,
            seq,
            [4, 5, 6],
            verifier_mode="native",
        )

        assert event["accepted_tokens"] == expected, name
        assert len(seq) == 3 + len(expected), name
        assert event["committed_blocks"] == (
            [11] if committed_blocks else []
        ), name
        assert event["released_blocks"] == (
            [] if committed_blocks else [11]
        ), name
        if expected:
            assert seq.last_token == expected[-1], name


def test_native_eos_and_output_budget_truncation_flags():
    llm, seq, _, _ = _native_verify_fixture(
        first_target=4,
        tail_targets=[99, 6],
        eos=99,
    )
    eos_event = verify_and_commit_block(
        llm,
        seq,
        [4, 99, 6],
        verifier_mode="native",
    )
    assert eos_event["accepted_tokens"] == [4, 99]
    assert eos_event["eos_truncated"] is True
    assert eos_event["output_budget_truncated"] is False
    assert eos_event["finished"] is True

    llm, seq, _, _ = _native_verify_fixture(
        first_target=4,
        tail_targets=[5, 6],
        max_tokens=2,
    )
    budget_event = verify_and_commit_block(
        llm,
        seq,
        [4, 5, 6],
        verifier_mode="native",
    )
    assert budget_event["accepted_tokens"] == [4, 5]
    assert budget_event["eos_truncated"] is False
    assert budget_event["output_budget_truncated"] is True
    assert budget_event["finished"] is True


def test_native_oracle_evidence_can_defer_terminal_deallocation():
    llm, seq, block_manager, _ = _native_verify_fixture(
        first_target=4,
        tail_targets=[99, 6],
        eos=99,
    )

    event = verify_and_commit_block(
        llm,
        seq,
        [4, 99, 6],
        verifier_mode="native",
        capture_oracle_evidence=True,
        defer_finish_for_oracle_evidence=True,
    )

    assert event["accepted_tokens"] == [4, 99]
    assert event["finished"] is False
    assert event["finish_would_trigger"] is True
    assert event["finish_deferred_for_oracle_evidence"] is True
    assert block_manager.deallocate_calls == []
    assert seq.block_table == [10]


def test_native_commit_failure_reports_phase_and_releases_reservation():
    llm, seq, block_manager, _ = _native_verify_fixture(
        first_target=4,
        tail_targets=[5],
        fail_commit=True,
    )

    try:
        verify_and_commit_block(
            llm,
            seq,
            [4, 5],
            verifier_mode="native",
        )
    except RuntimeError as exc:
        assert "metadata_commit" in str(exc)
        assert "commit failure" in str(exc)
    else:
        raise AssertionError("commit failure must propagate")

    assert seq.token_ids == [1, 2, 3]
    assert seq.block_table == [10]
    assert block_manager.release_calls == [[11]]
    assert native_test_context.get_context().mode == "decode"


def test_native_full_accept_commits_multiple_reserved_blocks():
    draft_tokens = list(range(4, 12))
    llm, seq, _, runner = _native_verify_fixture(
        first_target=4,
        tail_targets=list(range(5, 12)),
    )

    event = verify_and_commit_block(
        llm,
        seq,
        draft_tokens,
        verifier_mode="native",
    )

    assert event["query_len"] == 7
    assert event["accepted_tokens"] == draft_tokens
    assert event["reserved_blocks"] == [11, 12]
    assert event["committed_blocks"] == [11, 12]
    assert event["released_blocks"] == []
    assert seq.block_table == [10, 11, 12]
    assert seq.last_token == 11
    assert runner.spec_verify_calls == 1


def test_native_profile_args_require_supported_scope():
    base = {
        "model": "model",
        "temperature": 0.0,
        "max_commit_events": 1,
        "warmup_output_len": 0,
        "simulate_kv_upload_mb": 0.0,
        "max_draft_tokens": 4,
        "draft_policy": "fixed",
        "draft_source": "ngram",
        "mode": "candidate-only",
        "max_num_seqs": 1,
        "verifier_mode": "native",
        "enforce_eager": True,
        "kv_quant_bits": 0,
        "kv_offload_mvp0": False,
        "kv_offload_blockwise_decode": False,
        "kv_offload_blockwise_prefill": False,
        "quest_top_k_blocks": -1,
        "am_compact_blocks": 0,
        "kv_cartridge_blocks": 0,
    }
    validate_profile_args(type("Args", (), base)())

    invalid = (
        ("mode", "paired"),
        ("max_num_seqs", 2),
        ("enforce_eager", False),
        ("kv_quant_bits", 4),
        ("kv_offload_mvp0", True),
        ("kv_offload_blockwise_decode", True),
        ("kv_offload_blockwise_prefill", True),
        ("quest_top_k_blocks", 1),
        ("am_compact_blocks", 1),
        ("kv_cartridge_blocks", 1),
    )
    for name, value in invalid:
        values = dict(base)
        values[name] = value
        try:
            validate_profile_args(type("Args", (), values)())
        except ValueError as exc:
            assert "native verifier" in str(exc)
        else:
            raise AssertionError(name)


def test_candidate_profile_uses_routing_helper_and_paired_keeps_verifier_mode():
    source = open(
        os.path.join(_REPO_ROOT, "tools", "profile_ngram_commit.py")
    ).read()
    assert source.count("_run_draft_verification(") == 2
    assert source.count("verifier_mode=args.verifier_mode") == 1
    assert 'args.speculation_routing == "fixed-profitability"' in source


def test_profile_validation_rejects_adaptive_non_ngram_source():
    class Args:
        model = "model"
        temperature = 0.0
        max_commit_events = 0
        warmup_output_len = 1
        simulate_kv_upload_mb = 0.0
        draft_policy = "adaptive"
        draft_source = "dflash-toy"
        mode = "candidate-only"
        max_num_seqs = 1
        max_draft_tokens = 4

    try:
        validate_profile_args(Args())
    except ValueError as exc:
        assert "adaptive draft policy requires --draft-source ngram" in str(exc)
    else:
        raise AssertionError("adaptive non-ngram source accepted")


def test_profile_validation_requires_single_sequence_for_adaptive():
    class Args:
        model = "model"
        temperature = 0.0
        max_commit_events = 0
        warmup_output_len = 1
        simulate_kv_upload_mb = 0.0
        draft_policy = "adaptive"
        draft_source = "ngram"
        mode = "candidate-only"
        max_num_seqs = 2
        max_draft_tokens = 4

    try:
        validate_profile_args(Args())
    except ValueError as exc:
        assert "--max-num-seqs 1" in str(exc)
    else:
        raise AssertionError("batched adaptive profile accepted")


def test_attach_draft_policy_event_updates_adaptive_after_verification():
    state = AdaptiveDraftState()

    event = profile_ngram.attach_draft_policy_event(
        {"accepted_count": 0, "timing_ms": {"verify_commit_total_ms": 3.5}},
        profile_ngram.DraftProposal(tokens=[10, 11], source="ngram"),
        selected_k=2,
        adaptive_state=state,
    )

    assert event["selected_k"] == 2
    assert event["proposed_tokens"] == 2
    assert event["wasted_draft_tokens"] == 2
    assert event["adaptive_transition"]["selected_k_after"] == 1
    assert state.selected_k == 1


def test_propose_draft_dflash_toy_repeats_recent_window():
    class Args:
        draft_source = "dflash-toy"
        max_draft_tokens = 5
        dflash_toy_context_tokens = 2

    draft = propose_draft([10, 11, 12], Args())

    assert draft.source == "dflash-toy"
    assert draft.tokens == [10, 11, 12, 10, 11]
    assert draft.metadata["toy_strategy"] == "repeat_recent_tokens"
    assert draft.metadata["context_tokens"] == 2
    assert draft.metadata["window_tokens"] == 3


def test_propose_draft_dflash_toy_waits_for_context():
    class Args:
        draft_source = "dflash-toy"
        max_draft_tokens = 2
        dflash_toy_context_tokens = 4

    draft = propose_draft([10, 11], Args())

    assert draft.source == "dflash-toy"
    assert draft.tokens == []
    assert draft.metadata["reason"] == "insufficient_history"


def test_propose_draft_dflash_toy_ngram_or_repeat_prefers_ngram():
    class Args:
        draft_source = "dflash-toy-ngram-or-repeat"
        ngram_size = 2
        max_draft_tokens = 2
        dflash_toy_context_tokens = 1

    draft = propose_draft([1, 2, 3, 1, 2, 4, 1, 2], Args())

    assert draft.source == "dflash-toy-ngram-or-repeat"
    assert draft.tokens == [4, 1]
    assert draft.metadata["toy_strategy"] == "ngram_or_repeat"
    assert draft.metadata["selected_strategy"] == "ngram"


def test_propose_draft_dflash_toy_ngram_or_repeat_falls_back_to_repeat():
    class Args:
        draft_source = "dflash-toy-ngram-or-repeat"
        ngram_size = 3
        max_draft_tokens = 3
        dflash_toy_context_tokens = 1

    draft = propose_draft([5, 6], Args())

    assert draft.source == "dflash-toy-ngram-or-repeat"
    assert draft.tokens == [5, 6, 5]
    assert draft.metadata["toy_strategy"] == "ngram_or_repeat"
    assert draft.metadata["selected_strategy"] == "repeat_recent_tokens"


def test_summarize_hidden_to_draft_stub_returns_json_friendly_topk_preview():
    class FakeTensor:
        shape = (3, 1024)
        dtype = "torch.bfloat16"
        device = "cuda:0"

    logits = [
        [0.1, 2.0, 1.0, 0.0],
        [3.0, 0.5, 0.0, 2.0],
        [0.0, 0.1, 0.2, 0.3],
    ]

    summary = summarize_hidden_to_draft_stub(FakeTensor(), logits, top_k=2)

    assert summary["adapter"] == "target_hidden_topk_stub"
    assert summary["shape"] == [3, 1024]
    assert summary["dtype"] == "torch.bfloat16"
    assert summary["device"] == "cuda:0"
    assert summary["top_k"] == 2
    assert summary["rows"] == 3
    assert summary["preview"] == [
        {"row": 0, "token_ids": [1, 2], "scores": [2.0, 1.0]},
        {"row": 1, "token_ids": [0, 3], "scores": [3.0, 2.0]},
        {"row": 2, "token_ids": [3, 2], "scores": [0.3, 0.2]},
    ]


def test_summarize_hidden_to_draft_stub_defines_interface_schema_and_timing():
    class FakeTensor:
        shape = (2, 1024)
        dtype = "torch.bfloat16"
        device = "cuda:0"
        values = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

    summary = summarize_hidden_to_draft_stub(FakeTensor(), [[0.0, 1.0], [2.0, 0.0]], top_k=1)

    assert summary["interface_version"] == 1
    assert summary["runtime_mutation"] is False
    assert summary["input_schema"]["hidden_states"] == {
        "shape": [2, 1024],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
    }
    assert summary["input_schema"]["logits"] == {
        "shape": [2, 2],
        "dtype": "float32_preview",
        "device": "cpu_preview",
    }
    assert summary["input_schema"]["adapter"] == "topk-stub"
    assert summary["input_schema"]["top_k"] == 1
    assert summary["input_schema"]["hidden_rows"] == 2
    assert summary["input_schema"]["logit_rows"] == 2
    assert summary["input_schema"]["projected_rows"] == 2
    assert summary["output_schema"] == {
        "draft_token_ids": "list[int]",
        "draft_scores": "list[float]",
        "num_rows": "int",
        "projected_rows": "int",
        "source": "profiler_only_hidden_to_draft_adapter",
        "projection": "logits_topk",
    }
    assert summary["output"] == {
        "draft_token_ids": [1, 0],
        "draft_scores": [1.0, 2.0],
        "num_rows": 2,
        "projected_rows": 2,
        "source": "target_hidden_topk_stub",
    }
    assert set(summary["timing_ms"]) == {
        "adapter_total_ms",
        "candidate_select_ms",
        "draft_model_forward_ms",
        "logits_to_cpu_ms",
        "topk_ms",
    }
    assert all(isinstance(value, float) and value >= 0.0 for value in summary["timing_ms"].values())


def test_summarize_hidden_to_draft_stub_supports_linear_stub_interface():
    class FakeTensor:
        shape = (2, 1024)
        dtype = "torch.bfloat16"
        device = "cuda:0"
        values = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

    summary = summarize_hidden_to_draft_stub(
        FakeTensor(),
        [[0.0, 1.0, 3.0], [2.0, 0.0, 1.0]],
        top_k=1,
        adapter="linear-stub",
    )

    assert summary["adapter"] == "target_hidden_linear_stub"
    assert summary["runtime_mutation"] is False
    assert summary["input_schema"]["adapter"] == "linear-stub"
    assert summary["output"]["source"] == "target_hidden_linear_stub"
    assert summary["output"]["draft_token_ids"] == [1, 2]
    assert summary["output_schema"]["projection"] == "deterministic_hidden_linear_stub"
    assert set(summary["timing_ms"]) == {
        "adapter_total_ms",
        "candidate_select_ms",
        "draft_model_forward_ms",
        "hidden_to_cpu_ms",
        "logits_to_cpu_ms",
        "linear_projection_ms",
        "topk_ms",
    }
    assert summary["timing_ms"]["linear_projection_ms"] >= 0.0


def test_summarize_hidden_to_draft_stub_linear_stub_uses_hidden_projection_candidates():
    class FakeTensor:
        shape = (2, 4)
        dtype = "torch.float32"
        device = "cpu"

        def __init__(self):
            self.values = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

    logits = [
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
    ]

    summary = summarize_hidden_to_draft_stub(
        FakeTensor(),
        logits,
        top_k=1,
        adapter="linear-stub",
    )

    assert summary["output_schema"]["projection"] == "deterministic_hidden_linear_stub"
    assert summary["projection_metadata"] == {
        "seed": 17,
        "candidate_token_ids": [0, 1, 2, 3],
        "hidden_dim": 4,
        "candidate_count": 4,
    }
    assert summary["rows"] == 2
    assert summary["output"]["num_rows"] == 2
    assert len(summary["preview"]) == 2
    assert summary["output"]["draft_token_ids"] == [1, 3]
    assert summary["preview"] == [
        {"row": 0, "token_ids": [1], "scores": [1.25]},
        {"row": 1, "token_ids": [3], "scores": [0.75]},
    ]


def test_summarize_hidden_to_draft_stub_linear_stub_counts_hidden_rows_not_logits_rows():
    class FakeTensor:
        shape = (3, 4)
        dtype = "torch.float32"
        device = "cpu"
        values = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

    summary = summarize_hidden_to_draft_stub(
        FakeTensor(),
        [[0.0, 1.0, 3.0], [2.0, 0.0, 1.0]],
        top_k=1,
        adapter="linear-stub",
    )

    assert summary["rows"] == 3
    assert summary["output"]["num_rows"] == 3
    assert summary["output"]["projected_rows"] == 3
    assert summary["input_schema"]["hidden_rows"] == 3
    assert summary["input_schema"]["logit_rows"] == 2
    assert summary["input_schema"]["projected_rows"] == 3
    assert summary["input_schema"]["logits"]["shape"] == [2, 3]
    assert len(summary["preview"]) == 3
    assert len(summary["output"]["draft_token_ids"]) == 3


def test_summarize_hidden_to_draft_stub_draft_model_stub_reports_candidate_logits():
    class FakeTensor:
        shape = (2, 4)
        dtype = "torch.float32"
        device = "cpu"
        values = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    summary = summarize_hidden_to_draft_stub(
        FakeTensor(),
        [[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]],
        top_k=2,
        adapter="draft-model-stub",
    )

    assert summary["adapter"] == "target_hidden_draft_model_stub"
    assert summary["runtime_mutation"] is False
    assert summary["input_schema"]["adapter"] == "draft-model-stub"
    assert summary["input_schema"]["hidden_rows"] == 2
    assert summary["input_schema"]["logit_rows"] == 2
    assert summary["input_schema"]["projected_rows"] == 2
    assert summary["output_schema"]["projection"] == "deterministic_draft_model_stub"
    assert summary["output_schema"]["candidate_token_ids"] == "list[list[int]]"
    assert summary["output_schema"]["candidate_logits"] == "list[list[float]]"
    assert summary["output"]["source"] == "target_hidden_draft_model_stub"
    assert summary["output"]["projected_rows"] == 2
    assert summary["output"]["candidate_token_ids"] == [[2, 1], [3, 0]]
    assert summary["output"]["candidate_logits"] == [[1.6666666666666667, 1.0], [1.6666666666666667, 1.0]]
    assert summary["output"]["draft_token_ids"] == [2, 3]
    assert summary["draft_model_metadata"]["input_schema"] == {
        "hidden_rows": 2,
        "hidden_dim": 4,
        "candidate_count": 4,
        "top_k": 2,
        "source_shape": [2, 4],
        "source_dtype": "torch.float32",
        "source_device": "cpu",
    }
    assert summary["draft_model_metadata"]["contract"]["compatible"] is True
    assert _base_draft_model_metadata(summary["draft_model_metadata"]) == {
        "seed": 23,
        "candidate_token_ids": [0, 1, 2, 3],
        "hidden_dim": 4,
        "candidate_count": 4,
        "stub_version": 1,
    }
    assert set(summary["timing_ms"]) == {
        "adapter_total_ms",
        "candidate_select_ms",
        "draft_model_forward_ms",
        "hidden_to_cpu_ms",
        "logits_to_cpu_ms",
        "topk_ms",
    }
    assert summary["timing_ms"]["draft_model_forward_ms"] >= 0.0
    assert summary["timing_ms"]["candidate_select_ms"] >= 0.0


def test_run_draft_model_stub_exposes_replaceable_forward_boundary():
    result = run_draft_model_stub(
        hidden_rows=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        candidate_token_ids=[0, 1, 2, 3],
        top_k=2,
    )
    result_json = result.to_dict()

    assert isinstance(result, DraftModelResult)
    assert result_json["candidate_token_ids"] == [[2, 1], [3, 0]]
    assert result_json["candidate_logits"] == [[1.6666666666666667, 1.0], [1.6666666666666667, 1.0]]
    assert result_json["draft_token_ids"] == [2, 3]
    assert result_json["draft_scores"] == [1.6666666666666667, 1.6666666666666667]
    assert result_json["preview"] == [
        {"row": 0, "token_ids": [2, 1], "scores": [1.6666666666666667, 1.0]},
        {"row": 1, "token_ids": [3, 0], "scores": [1.6666666666666667, 1.0]},
    ]
    assert result_json["metadata"]["input_schema"] == {
        "hidden_rows": 2,
        "hidden_dim": 4,
        "candidate_count": 4,
        "top_k": 2,
        "source_shape": None,
        "source_dtype": None,
        "source_device": None,
    }
    assert result_json["metadata"]["contract"]["compatible"] is True
    assert _base_draft_model_metadata(result_json["metadata"]) == {
        "seed": 23,
        "candidate_token_ids": [0, 1, 2, 3],
        "hidden_dim": 4,
        "candidate_count": 4,
        "stub_version": 1,
    }
    assert set(result_json["timing_ms"]) == {"draft_model_forward_ms", "candidate_select_ms"}
    assert result_json["timing_ms"]["draft_model_forward_ms"] >= 0.0
    assert result_json["timing_ms"]["candidate_select_ms"] >= 0.0

    class FakeTensor:
        shape = (2, 4)
        dtype = "torch.float32"
        device = "cpu"
        values = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    summary = summarize_hidden_to_draft_stub(
        FakeTensor(),
        [[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]],
        top_k=2,
        adapter="draft-model-stub",
    )

    assert summary["output"]["candidate_token_ids"] == result_json["candidate_token_ids"]
    assert summary["output"]["candidate_logits"] == result_json["candidate_logits"]
    assert _base_draft_model_metadata(summary["draft_model_metadata"]) == _base_draft_model_metadata(
        result_json["metadata"]
    )


def test_run_draft_model_stub_accepts_config_and_validates_boundaries():
    config = DraftModelStubConfig(seed=23, stub_version=7)
    result = run_draft_model_stub(
        hidden_rows=[[1.0, 0.0, 0.0, 0.0]],
        candidate_token_ids=[0, 1, 2, 3],
        top_k=2,
        config=config,
    ).to_dict()

    assert result["metadata"]["seed"] == 23
    assert result["metadata"]["stub_version"] == 7
    assert result["metadata"]["candidate_count"] == 4
    assert result["draft_token_ids"] == [2]
    assert result["metadata"]["contract"]["compatible"] is True

    try:
        run_draft_model_stub([[1.0]], [], top_k=1)
    except ValueError as exc:
        assert "candidate_token_ids must not be empty" in str(exc)
    else:
        raise AssertionError("empty candidate set should fail")

    try:
        run_draft_model_stub([[1.0], [1.0, 2.0]], [0, 1], top_k=1)
    except ValueError as exc:
        assert "hidden_rows must have a consistent width" in str(exc)
    else:
        raise AssertionError("ragged hidden rows should fail")


def test_draft_model_input_makes_profiler_boundary_explicit():
    draft_input = DraftModelInput.from_rows(
        hidden_rows=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        candidate_token_ids=[0, 1, 2, 3],
        top_k=2,
        source_shape=[2, 4],
        source_dtype="torch.float32",
        source_device="cpu",
    )

    assert draft_input.to_dict() == {
        "hidden_rows": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "candidate_token_ids": [0, 1, 2, 3],
        "top_k": 2,
        "source_shape": [2, 4],
        "source_dtype": "torch.float32",
        "source_device": "cpu",
    }

    result = run_draft_model_stub(draft_input)

    assert result.draft_token_ids == [2, 3]
    assert result.metadata["input_schema"] == {
        "hidden_rows": 2,
        "hidden_dim": 4,
        "candidate_count": 4,
        "top_k": 2,
        "source_shape": [2, 4],
        "source_dtype": "torch.float32",
        "source_device": "cpu",
    }

    checked = run_draft_model_stub(
        draft_input,
        contract=DraftModelContract(
            expected_hidden_dim=4,
            target_vocab_size=8,
            draft_vocab_size=8,
            tokenizer_family="qwen3",
            draft_tokenizer_family="qwen3",
        ),
    )

    assert checked.metadata["contract"] == {
        "expected_hidden_dim": 4,
        "actual_hidden_dim": 4,
        "target_vocab_size": 8,
        "draft_vocab_size": 8,
        "tokenizer_family": "qwen3",
        "draft_tokenizer_family": "qwen3",
        "candidate_id_min": 0,
        "candidate_id_max": 3,
        "compatible": True,
    }


def test_draft_model_contract_validates_hidden_vocab_and_tokenizer_boundaries():
    draft_input = DraftModelInput.from_rows(
        hidden_rows=[[1.0, 0.0, 0.0, 0.0]],
        candidate_token_ids=[0, 3],
        top_k=2,
        source_shape=[1, 4],
        source_dtype="torch.float32",
        source_device="cpu",
    )
    contract = DraftModelContract(
        expected_hidden_dim=4,
        target_vocab_size=8,
        draft_vocab_size=8,
        tokenizer_family="qwen3",
        draft_tokenizer_family="qwen3",
    )

    metadata = validate_draft_model_contract(draft_input, contract)

    assert metadata == {
        "expected_hidden_dim": 4,
        "actual_hidden_dim": 4,
        "target_vocab_size": 8,
        "draft_vocab_size": 8,
        "tokenizer_family": "qwen3",
        "draft_tokenizer_family": "qwen3",
        "candidate_id_min": 0,
        "candidate_id_max": 3,
        "compatible": True,
    }

    try:
        validate_draft_model_contract(draft_input, DraftModelContract(expected_hidden_dim=5))
    except ValueError as exc:
        assert "hidden_dim mismatch" in str(exc)
    else:
        raise AssertionError("hidden_dim mismatch should fail")

    try:
        validate_draft_model_contract(draft_input, DraftModelContract(target_vocab_size=3))
    except ValueError as exc:
        assert "candidate token id out of target vocab" in str(exc)
    else:
        raise AssertionError("target vocab overflow should fail")

    try:
        validate_draft_model_contract(
            draft_input,
            DraftModelContract(tokenizer_family="qwen3", draft_tokenizer_family="llama"),
        )
    except ValueError as exc:
        assert "tokenizer family mismatch" in str(exc)
    else:
        raise AssertionError("tokenizer mismatch should fail")


def main():
    test_propose_ngram_draft_uses_latest_matching_suffix()
    test_propose_ngram_draft_respects_max_draft_tokens()
    test_propose_ngram_draft_returns_empty_without_match()
    test_replay_ngram_acceptance_counts_accepted_prefix_only()
    test_summarize_replay_stats_is_json_friendly()
    test_online_dry_run_accepts_pending_prefix_across_steps()
    test_online_dry_run_rejects_and_clears_pending_tokens()
    test_count_accepted_prefix_stops_at_first_mismatch()
    test_summarize_target_verify_stats_is_json_friendly()
    test_adaptive_draft_state_starts_at_k2()
    test_adaptive_two_strong_full_accepts_promote_and_saturate()
    test_adaptive_zero_accept_jumps_from_k4_to_k1()
    test_adaptive_weak_partial_accept_moves_down_one_level()
    test_adaptive_weak_ema_demotes_k2_to_k1()
    test_adaptive_partial_accept_resets_full_accept_streak()
    test_adaptive_rejects_invalid_counts_and_state()
    test_adaptive_transition_record_is_json_friendly_and_replayable()
    test_propose_draft_dispatches_ngram_source()
    test_propose_draft_accepts_per_event_cap_without_mutating_args()
    test_propose_draft_dispatches_fixed_sam_source()
    test_propose_draft_dispatches_match_aware_sam_bypass()
    test_sam_profile_args_require_candidate_greedy_single_sequence()
    test_sync_sam_index_extends_only_verified_history_tail()
    test_sync_sam_index_rejects_history_rewrite()
    test_empty_sam_proposal_bypasses_verifier()
    test_sam_profiler_remains_profiler_owned()
    test_sam_verify_event_contract_is_profiler_owned()
    test_accepted_kv_rematerialization_uses_normal_decode_for_materialized_prefix()
    test_accepted_kv_rematerialization_skips_pending_only_token()
    test_native_verify_commits_without_decode_rematerialization()
    test_native_oracle_evidence_captures_tail_logits_and_final_slot_kv()
    test_native_k1_uses_first_target_without_tail_forward()
    test_native_unsupported_mode_fails_before_reservation()
    test_native_tail_failure_releases_reservation_and_resets_context()
    test_native_acceptance_matrix_preserves_pending_token_lifecycle()
    test_native_eos_and_output_budget_truncation_flags()
    test_native_oracle_evidence_can_defer_terminal_deallocation()
    test_native_commit_failure_reports_phase_and_releases_reservation()
    test_native_full_accept_commits_multiple_reserved_blocks()
    test_native_profile_args_require_supported_scope()
    test_candidate_profile_uses_routing_helper_and_paired_keeps_verifier_mode()
    test_profile_validation_rejects_adaptive_non_ngram_source()
    test_profile_validation_requires_single_sequence_for_adaptive()
    test_attach_draft_policy_event_updates_adaptive_after_verification()
    test_propose_draft_dflash_toy_repeats_recent_window()
    test_propose_draft_dflash_toy_waits_for_context()
    test_propose_draft_dflash_toy_ngram_or_repeat_prefers_ngram()
    test_propose_draft_dflash_toy_ngram_or_repeat_falls_back_to_repeat()
    test_summarize_hidden_to_draft_stub_returns_json_friendly_topk_preview()
    test_summarize_hidden_to_draft_stub_defines_interface_schema_and_timing()
    test_summarize_hidden_to_draft_stub_supports_linear_stub_interface()
    test_summarize_hidden_to_draft_stub_linear_stub_uses_hidden_projection_candidates()
    test_summarize_hidden_to_draft_stub_linear_stub_counts_hidden_rows_not_logits_rows()
    test_summarize_hidden_to_draft_stub_draft_model_stub_reports_candidate_logits()
    test_run_draft_model_stub_exposes_replaceable_forward_boundary()
    test_run_draft_model_stub_accepts_config_and_validates_boundaries()
    test_draft_model_input_makes_profiler_boundary_explicit()
    test_draft_model_contract_validates_hidden_vocab_and_tokenizer_boundaries()
    print("ngram speculative tests passed")


if __name__ == "__main__":
    main()
