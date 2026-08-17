from __future__ import annotations

import math
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
    DraftCapabilities,
    DraftContext,
    DraftProposal,
    validate_draft_capabilities,
    validate_draft_adapter_batch,
)


class _Adapter:
    def __init__(self, capabilities, proposals):
        self._capabilities = capabilities
        self._proposals = tuple(proposals)
        self.calls = []

    @property
    def capabilities(self):
        return self._capabilities

    def propose_batch(self, contexts):
        self.calls.append(contexts)
        return self._proposals


def _capabilities(**overrides):
    values = {
        "source_type": "fixture",
        "supports_batch": True,
        "requires_target_hidden": False,
        "requires_target_logits": False,
        "max_proposal_tokens": 4,
    }
    values.update(overrides)
    return DraftCapabilities(**values)


def _context(sequence_id, **overrides):
    values = {
        "sequence_id": sequence_id,
        "token_ids": (1, 2, sequence_id),
        "remaining_output_tokens": 4,
        "max_proposal_tokens": 4,
        "first_target_token": 10 + sequence_id,
        "target_hidden": None,
        "target_logits": None,
    }
    values.update(overrides)
    return DraftContext(**values)


def _proposal(sequence_id, token_ids, **overrides):
    values = {
        "sequence_id": sequence_id,
        "token_ids": tuple(token_ids),
        "source_type": "fixture",
        "metadata": None,
        "timing_ms": {"draft_ms": 0.25},
    }
    values.update(overrides)
    return DraftProposal(**values)


def test_capabilities_default_to_host_execution():
    capabilities = _capabilities()

    assert capabilities.execution_domain == "host"
    assert capabilities.requires_proposal_lifecycle is False
    assert capabilities.requires_full_token_history is True
    assert validate_draft_capabilities(
        capabilities,
        expected_execution_domain="host",
    ) is capabilities


def test_capabilities_require_boolean_full_token_history_flag():
    capabilities = _capabilities(
        requires_full_token_history="no",
    )

    with pytest.raises(
        ValueError,
        match="full token history.*bool",
    ):
        validate_draft_capabilities(capabilities)


def test_proposal_defaults_to_no_transaction():
    proposal = _proposal(1, (10,))

    assert proposal.proposal_transaction_id is None


def test_host_capability_rejects_proposal_lifecycle():
    capabilities = _capabilities(
        requires_proposal_lifecycle=True,
    )

    with pytest.raises(ValueError, match="lifecycle.*model_runner"):
        validate_draft_capabilities(capabilities)


@pytest.mark.parametrize(
    "requires_proposal_lifecycle",
    [None, 0, 1, "yes"],
)
def test_rejects_non_boolean_lifecycle_capability(
    requires_proposal_lifecycle,
):
    capabilities = _capabilities(
        execution_domain="model_runner",
        requires_proposal_lifecycle=(
            requires_proposal_lifecycle
        ),
    )

    with pytest.raises(ValueError, match="lifecycle.*bool"):
        validate_draft_capabilities(capabilities)


@pytest.mark.parametrize(
    "execution_domain",
    ["", "gpu", "mtp", 1, None],
)
def test_rejects_invalid_execution_domain(execution_domain):
    adapter = _Adapter(
        _capabilities(execution_domain=execution_domain),
        (_proposal(1, (10,)),),
    )

    with pytest.raises(ValueError, match="execution domain"):
        validate_draft_adapter_batch(adapter, (_context(1),))


def test_host_adapter_rejects_model_runner_execution_domain():
    adapter = _Adapter(
        _capabilities(execution_domain="model_runner"),
        (_proposal(1, (10,)),),
    )

    with pytest.raises(ValueError, match="host"):
        validate_draft_adapter_batch(adapter, (_context(1),))


def test_validates_and_reorders_proposals_to_context_order():
    contexts = (_context(1), _context(2))
    adapter = _Adapter(
        _capabilities(),
        (
            _proposal(2, ()),
            _proposal(1, (10, 11)),
        ),
    )

    proposals = validate_draft_adapter_batch(adapter, contexts)

    assert tuple(item.sequence_id for item in proposals) == (1, 2)
    assert proposals[0].token_ids == (10, 11)
    assert proposals[1].token_ids == ()
    assert adapter.calls == [contexts]


@pytest.mark.parametrize(
    "contexts",
    [
        (_context(1), _context(1)),
        (_context(True),),
    ],
)
def test_rejects_invalid_context_sequence_ids(contexts):
    adapter = _Adapter(
        _capabilities(),
        (_proposal(1, (10,)),),
    )

    with pytest.raises(ValueError, match="sequence"):
        validate_draft_adapter_batch(adapter, contexts)


@pytest.mark.parametrize(
    "proposals,match",
    [
        ((_proposal(1, (10,)),), "exactly match"),
        (
            (
                _proposal(1, (10,)),
                _proposal(2, (20,)),
                _proposal(3, (30,)),
            ),
            "exactly match",
        ),
        (
            (
                _proposal(1, (10,)),
                _proposal(1, (11,)),
            ),
            "unique",
        ),
    ],
)
def test_rejects_missing_extra_or_duplicate_proposal_ids(
    proposals,
    match,
):
    adapter = _Adapter(_capabilities(), proposals)

    with pytest.raises(ValueError, match=match):
        validate_draft_adapter_batch(
            adapter,
            (_context(1), _context(2)),
        )


@pytest.mark.parametrize(
    "token_ids",
    [
        (True,),
        (1.5,),
        ("1",),
    ],
)
def test_rejects_non_integer_or_boolean_proposal_tokens(token_ids):
    adapter = _Adapter(
        _capabilities(),
        (_proposal(1, token_ids),),
    )

    with pytest.raises(ValueError, match="token"):
        validate_draft_adapter_batch(adapter, (_context(1),))


@pytest.mark.parametrize(
    "capability_limit,context_limit,proposal,match",
    [
        (1, 4, (10, 11), "capability"),
        (4, 1, (10, 11), "context"),
    ],
)
def test_rejects_proposal_length_over_any_limit(
    capability_limit,
    context_limit,
    proposal,
    match,
):
    adapter = _Adapter(
        _capabilities(max_proposal_tokens=capability_limit),
        (_proposal(1, proposal),),
    )
    context = _context(
        1,
        max_proposal_tokens=context_limit,
    )

    with pytest.raises(ValueError, match=match):
        validate_draft_adapter_batch(adapter, (context,))


def test_allows_runtime_to_truncate_proposal_over_remaining_budget():
    adapter = _Adapter(
        _capabilities(max_proposal_tokens=4),
        (_proposal(1, (10, 11, 12)),),
    )

    proposals = validate_draft_adapter_batch(
        adapter,
        (
            _context(
                1,
                max_proposal_tokens=4,
                remaining_output_tokens=1,
            ),
        ),
    )

    assert proposals[0].token_ids == (10, 11, 12)


@pytest.mark.parametrize(
    "capabilities,context,match",
    [
        (
            _capabilities(requires_target_hidden=True),
            _context(1, target_hidden=None),
            "hidden",
        ),
        (
            _capabilities(requires_target_logits=True),
            _context(1, target_logits=None),
            "logits",
        ),
    ],
)
def test_rejects_missing_required_target_payload(
    capabilities,
    context,
    match,
):
    adapter = _Adapter(
        capabilities,
        (_proposal(1, (10,)),),
    )

    with pytest.raises(ValueError, match=match):
        validate_draft_adapter_batch(adapter, (context,))


def test_accepts_required_target_payload_when_present():
    hidden = object()
    logits = object()
    adapter = _Adapter(
        _capabilities(
            requires_target_hidden=True,
            requires_target_logits=True,
        ),
        (_proposal(1, (10,)),),
    )

    proposals = validate_draft_adapter_batch(
        adapter,
        (
            _context(
                1,
                target_hidden=hidden,
                target_logits=logits,
            ),
        ),
    )

    assert proposals[0].token_ids == (10,)


@pytest.mark.parametrize(
    "timing",
    [
        {"draft_ms": -0.1},
        {"draft_ms": math.inf},
        {"draft_ms": math.nan},
        {"draft_ms": "slow"},
        {1: 0.1},
    ],
)
def test_rejects_invalid_timing_values(timing):
    adapter = _Adapter(
        _capabilities(),
        (_proposal(1, (10,), timing_ms=timing),),
    )

    with pytest.raises(ValueError, match="timing"):
        validate_draft_adapter_batch(adapter, (_context(1),))


@pytest.mark.parametrize(
    "capability_source,proposal_source",
    [
        ("", "fixture"),
        ("fixture", ""),
        ("fixture", "other"),
    ],
)
def test_rejects_invalid_or_mismatched_source_type(
    capability_source,
    proposal_source,
):
    adapter = _Adapter(
        _capabilities(source_type=capability_source),
        (
            _proposal(
                1,
                (10,),
                source_type=proposal_source,
            ),
        ),
    )

    with pytest.raises(ValueError, match="source_type"):
        validate_draft_adapter_batch(adapter, (_context(1),))


@pytest.mark.parametrize(
    "capabilities",
    [
        object(),
        _capabilities(supports_batch=False),
        _capabilities(max_proposal_tokens=0),
        _capabilities(max_proposal_tokens=True),
    ],
)
def test_rejects_invalid_adapter_capabilities(capabilities):
    adapter = _Adapter(
        capabilities,
        (_proposal(1, (10,)),),
    )

    with pytest.raises(ValueError, match="capabilit"):
        validate_draft_adapter_batch(adapter, (_context(1),))


def test_context_uses_immutable_history_snapshot():
    mutable_history = [1, 2, 3]
    context = _context(1, token_ids=tuple(mutable_history))
    mutable_history.append(4)
    adapter = _Adapter(
        _capabilities(),
        (_proposal(1, (10,)),),
    )

    validate_draft_adapter_batch(adapter, (context,))

    assert adapter.calls[0][0].token_ids == (1, 2, 3)
