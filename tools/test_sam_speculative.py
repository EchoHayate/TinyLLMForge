from __future__ import annotations

import importlib.util
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SAM_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "speculative", "sam.py")
_SPEC = importlib.util.spec_from_file_location("sam_under_test", _SAM_PATH)
sam = importlib.util.module_from_spec(_SPEC)
sys.modules["sam_under_test"] = sam
_SPEC.loader.exec_module(sam)

SuffixAutomatonDraftIndex = sam.SuffixAutomatonDraftIndex
select_match_aware_k = sam.select_match_aware_k


def test_empty_and_one_token_histories_have_no_usable_match():
    assert SuffixAutomatonDraftIndex([]).longest_usable_suffix() is None
    assert SuffixAutomatonDraftIndex([7]).longest_usable_suffix() is None


def test_longest_usable_suffix_uses_earliest_representative():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 4, 1, 2])
    match = index.longest_usable_suffix()
    assert match is not None
    assert match.match_length == 2
    assert match.match_start == 0
    assert match.match_end == 1
    assert match.continuation_start == 2
    assert match.available_continuation_tokens == 4
    assert match.continuation_region == "prompt"


def test_terminal_only_occurrence_is_not_usable():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 4])
    assert index.longest_usable_suffix() is None


def test_suffix_link_fallback_finds_shorter_usable_suffix():
    index = SuffixAutomatonDraftIndex([9, 2, 3, 8, 2, 3])
    match = index.longest_usable_suffix()
    assert match is not None
    assert match.match_length == 2
    assert match.match_start == 1
    assert match.continuation_start == 3


def test_proposal_stops_at_observed_stream_boundary():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = index.propose(max_draft_tokens=16)
    assert draft.tokens == [3, 1, 2]
    assert draft.selected_k == 16
    assert draft.match is not None


def test_prompt_and_generated_continuation_metadata():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    index.extend_verified([8, 9, 8, 9])
    index.assert_history([1, 2, 3, 8, 9, 8, 9])
    match = index.longest_usable_suffix()
    assert match is not None
    assert match.continuation_region == "generated"
    assert match.continuation_start >= index.prompt_length


def test_history_invariant_rejects_missing_or_extra_tokens():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    for history in ([1, 2], [1, 2, 3, 4], [1, 9, 3]):
        try:
            index.assert_history(history)
        except ValueError:
            pass
        else:
            raise AssertionError(history)


def test_state_and_draft_metadata_are_json_friendly():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = index.propose(max_draft_tokens=4)
    assert json.loads(json.dumps(draft.metadata)) == draft.metadata
    assert draft.metadata["index_token_count"] == 5
    assert draft.metadata["index_state_count"] == len(index.states)


def _run_tests() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("sam speculative tests passed")


if __name__ == "__main__":
    _run_tests()
