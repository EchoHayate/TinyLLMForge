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

propose_ngram_draft = ngram.propose_ngram_draft
replay_ngram_acceptance = ngram.replay_ngram_acceptance
summarize_replay_stats = ngram.summarize_replay_stats


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


def main():
    test_propose_ngram_draft_uses_latest_matching_suffix()
    test_propose_ngram_draft_respects_max_draft_tokens()
    test_propose_ngram_draft_returns_empty_without_match()
    test_replay_ngram_acceptance_counts_accepted_prefix_only()
    test_summarize_replay_stats_is_json_friendly()
    print("ngram speculative tests passed")


if __name__ == "__main__":
    main()
