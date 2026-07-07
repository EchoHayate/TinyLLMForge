"""n-gram speculative decoding helpers tests.

跑法：python tools/test_ngram_speculative.py
"""

import os
import sys
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_NGRAM_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "speculative", "ngram.py")
_SPEC = importlib.util.spec_from_file_location("ngram_under_test", _NGRAM_PATH)
ngram = importlib.util.module_from_spec(_SPEC)
sys.modules["ngram_under_test"] = ngram
_SPEC.loader.exec_module(ngram)

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
propose_draft = profile_ngram.propose_draft
summarize_hidden_to_draft_stub = profile_ngram.summarize_hidden_to_draft_stub


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


def test_propose_draft_dispatches_ngram_source():
    class Args:
        draft_source = "ngram"
        ngram_size = 2
        max_draft_tokens = 2

    draft = propose_draft([1, 2, 3, 1, 2, 4, 1, 2], Args())

    assert draft.source == "ngram"
    assert draft.tokens == [4, 1]
    assert draft.metadata == {"match_start": 3, "ngram_size": 2}


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

    summary = summarize_hidden_to_draft_stub(FakeTensor(), [[0.0, 1.0], [2.0, 0.0]], top_k=1)

    assert summary["interface_version"] == 1
    assert summary["runtime_mutation"] is False
    assert summary["input_schema"] == {
        "hidden_states": {
            "shape": [2, 1024],
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
        },
        "logits": {
            "shape": [2, 2],
            "dtype": "float32_preview",
            "device": "cpu_preview",
        },
        "top_k": 1,
    }
    assert summary["output_schema"] == {
        "draft_token_ids": "list[int]",
        "draft_scores": "list[float]",
        "num_rows": "int",
        "source": "profiler_only_hidden_to_draft_adapter",
    }
    assert summary["output"] == {
        "draft_token_ids": [1, 0],
        "draft_scores": [1.0, 2.0],
        "num_rows": 2,
        "source": "target_hidden_topk_stub",
    }
    assert set(summary["timing_ms"]) == {"adapter_total_ms", "logits_to_cpu_ms", "topk_ms"}
    assert all(isinstance(value, float) and value >= 0.0 for value in summary["timing_ms"].values())


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
    test_propose_draft_dispatches_ngram_source()
    test_propose_draft_dflash_toy_repeats_recent_window()
    test_propose_draft_dflash_toy_waits_for_context()
    test_propose_draft_dflash_toy_ngram_or_repeat_prefers_ngram()
    test_propose_draft_dflash_toy_ngram_or_repeat_falls_back_to_repeat()
    test_summarize_hidden_to_draft_stub_returns_json_friendly_topk_preview()
    test_summarize_hidden_to_draft_stub_defines_interface_schema_and_timing()
    print("ngram speculative tests passed")


if __name__ == "__main__":
    main()
