from __future__ import annotations

import os
import sys
import types

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

from tinyvllm.speculative.adapter import (
    DraftContext,
    validate_draft_adapter_batch,
)
from tinyvllm.speculative.ngram_adapter import NGramDraftAdapter
from tinyvllm.speculative.sam_adapter import SAMDraftAdapter


def _context(
    sequence_id,
    token_ids,
    *,
    remaining_output_tokens=8,
    max_proposal_tokens=8,
):
    return DraftContext(
        sequence_id=sequence_id,
        token_ids=tuple(token_ids),
        remaining_output_tokens=remaining_output_tokens,
        max_proposal_tokens=max_proposal_tokens,
        first_target_token=99,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ngram_size": 0, "max_proposal_tokens": 4},
        {"ngram_size": True, "max_proposal_tokens": 4},
        {"ngram_size": 2, "max_proposal_tokens": 0},
        {"ngram_size": 2, "max_proposal_tokens": True},
    ],
)
def test_ngram_adapter_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        NGramDraftAdapter(**kwargs)


def test_ngram_adapter_proposes_mixed_rows_in_input_order():
    adapter = NGramDraftAdapter(
        ngram_size=2,
        max_proposal_tokens=4,
    )
    contexts = (
        _context(7, (1, 2, 3, 1, 2)),
        _context(3, (8, 9, 10)),
    )

    proposals = validate_draft_adapter_batch(adapter, contexts)

    assert tuple(row.sequence_id for row in proposals) == (7, 3)
    assert proposals[0].token_ids == (3, 1, 2)
    assert proposals[0].source_type == "ngram"
    assert proposals[0].metadata == {
        "ngram_size": 2,
        "match_start": 0,
        "selected_k": 4,
        "history_token_count": 5,
        "bypass_reason": None,
    }
    assert proposals[1].token_ids == ()
    assert proposals[1].metadata["bypass_reason"] == "no_match"


def test_ngram_adapter_caps_by_context_but_not_remaining_budget():
    adapter = NGramDraftAdapter(
        ngram_size=2,
        max_proposal_tokens=4,
    )
    context = _context(
        5,
        (1, 2, 3, 4, 1, 2),
        remaining_output_tokens=1,
        max_proposal_tokens=2,
    )

    proposal = validate_draft_adapter_batch(
        adapter,
        (context,),
    )[0]

    assert proposal.token_ids == (3, 4)
    assert proposal.metadata["selected_k"] == 2


def test_ngram_adapter_returns_empty_for_zero_context_limit():
    adapter = NGramDraftAdapter(
        ngram_size=2,
        max_proposal_tokens=4,
    )

    proposal = validate_draft_adapter_batch(
        adapter,
        (
            _context(
                5,
                (1, 2, 3, 1, 2),
                max_proposal_tokens=0,
            ),
        ),
    )[0]

    assert proposal.token_ids == ()
    assert proposal.metadata["selected_k"] == 0
    assert proposal.metadata["bypass_reason"] == "selected_k_zero"


def test_ngram_adapter_does_not_mutate_context_history():
    token_ids = (1, 2, 3, 1, 2)
    adapter = NGramDraftAdapter(
        ngram_size=2,
        max_proposal_tokens=4,
    )

    adapter.propose_batch((_context(1, token_ids),))

    assert token_ids == (1, 2, 3, 1, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_proposal_tokens": 0},
        {"max_proposal_tokens": True},
        {"max_proposal_tokens": 4, "match_aware": 1},
    ],
)
def test_sam_adapter_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SAMDraftAdapter(**kwargs)


@pytest.mark.parametrize(
    "sequence_id,token_ids",
    [
        (True, (1, 2)),
        (1, [1, 2]),
        (1, (1, True)),
    ],
)
def test_sam_adapter_rejects_invalid_registration(
    sequence_id,
    token_ids,
):
    adapter = SAMDraftAdapter(max_proposal_tokens=4)

    with pytest.raises(ValueError):
        adapter.register_sequence(sequence_id, token_ids)


def test_sam_adapter_registers_proposes_synchronizes_and_releases():
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    adapter.register_sequence(7, (1, 2, 3, 1, 2))

    proposal = validate_draft_adapter_batch(
        adapter,
        (_context(7, (1, 2, 3, 1, 2)),),
    )[0]
    appended = adapter.synchronize_verified_history(
        7,
        (1, 2, 3, 1, 2, 9),
    )

    assert proposal.token_ids == (3, 1, 2)
    assert proposal.source_type == "sam"
    assert proposal.metadata["policy"] == "fixed"
    assert proposal.metadata["policy_selected_k"] == 4
    assert proposal.metadata["selected_k"] == 4
    assert appended == 1
    adapter.release_sequence(7)
    with pytest.raises(ValueError, match="registered"):
        adapter.propose_batch(
            (_context(7, (1, 2, 3, 1, 2, 9)),)
        )


def test_sam_adapter_rejects_duplicate_and_unknown_lifecycle_calls():
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    adapter.register_sequence(1, (1, 2))

    with pytest.raises(ValueError, match="already registered"):
        adapter.register_sequence(1, (1, 2))
    with pytest.raises(ValueError, match="registered"):
        adapter.synchronize_verified_history(2, (1, 2))
    with pytest.raises(ValueError, match="registered"):
        adapter.release_sequence(2)


@pytest.mark.parametrize(
    "verified_history",
    [
        (1, 2),
        (1, 9, 3),
    ],
)
def test_sam_adapter_rejects_truncated_or_rewritten_history_without_mutation(
    verified_history,
):
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    adapter.register_sequence(1, (1, 2, 3))
    index = adapter._indexes[1]
    before = (
        tuple(index.indexed_tokens),
        len(index.states),
        index.last_state,
    )

    with pytest.raises(ValueError, match="prefix"):
        adapter.synchronize_verified_history(
            1,
            verified_history,
        )

    after = (
        tuple(index.indexed_tokens),
        len(index.states),
        index.last_state,
    )
    assert after == before


def test_sam_adapter_rejects_stale_context_history():
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    adapter.register_sequence(1, (1, 2, 3))

    with pytest.raises(ValueError, match="target-verified history"):
        adapter.propose_batch((_context(1, (1, 2)),))


def test_sam_adapter_proposal_is_read_only():
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    adapter.register_sequence(1, (1, 2, 3, 1, 2))
    index = adapter._indexes[1]
    before = (
        tuple(index.indexed_tokens),
        len(index.states),
        index.last_state,
        index.prompt_length,
    )

    adapter.propose_batch(
        (_context(1, (1, 2, 3, 1, 2)),)
    )

    after = (
        tuple(index.indexed_tokens),
        len(index.states),
        index.last_state,
        index.prompt_length,
    )
    assert after == before


def test_sam_adapter_never_indexes_rejected_draft_tokens():
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    history = (1, 2, 3, 1, 2)
    adapter.register_sequence(1, history)

    proposal = adapter.propose_batch((_context(1, history),))[0]
    adapter.synchronize_verified_history(1, history + (9,))

    assert proposal.token_ids == (3, 1, 2)
    assert tuple(adapter._indexes[1].indexed_tokens) == history + (9,)
    assert adapter.synchronize_verified_history(
        1,
        history + (9,),
    ) == 0


def test_sam_adapter_fixed_mode_caps_by_context_not_remaining_budget():
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    history = (1, 2, 3, 4, 1, 2)
    adapter.register_sequence(1, history)

    proposal = validate_draft_adapter_batch(
        adapter,
        (
            _context(
                1,
                history,
                remaining_output_tokens=1,
                max_proposal_tokens=2,
            ),
        ),
    )[0]

    assert proposal.token_ids == (3, 4)
    assert proposal.metadata["selected_k"] == 2
    assert proposal.metadata["adapter_limit"] == 2


def test_sam_adapter_match_aware_mode_records_policy_and_cap():
    adapter = SAMDraftAdapter(
        max_proposal_tokens=4,
        match_aware=True,
    )
    history = (1, 2, 3, 4, 5, 1, 2)
    adapter.register_sequence(1, history)

    proposal = validate_draft_adapter_batch(
        adapter,
        (
            _context(
                1,
                history,
                max_proposal_tokens=2,
            ),
        ),
    )[0]

    assert proposal.token_ids == (3, 4)
    assert proposal.metadata["policy"] == "match_aware"
    assert proposal.metadata["policy_selected_k"] == 4
    assert proposal.metadata["selected_k"] == 2
    assert proposal.metadata["adapter_limit"] == 2


def test_sam_adapter_supports_mixed_empty_and_nonempty_rows():
    adapter = SAMDraftAdapter(max_proposal_tokens=4)
    adapter.register_sequence(8, (1, 2, 3, 1, 2))
    adapter.register_sequence(4, (7, 8, 9))

    proposals = validate_draft_adapter_batch(
        adapter,
        (
            _context(8, (1, 2, 3, 1, 2)),
            _context(4, (7, 8, 9)),
        ),
    )

    assert tuple(row.sequence_id for row in proposals) == (8, 4)
    assert proposals[0].token_ids == (3, 1, 2)
    assert proposals[1].token_ids == ()
    assert proposals[1].metadata["bypass_reason"] == "no_usable_match"
