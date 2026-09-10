"""Latent-state action drafter contract.

The published action-level speculation systems this design builds on
all draft with a smaller text model: the drafter generates the full
tool call as text and the actor's action is compared against it. Two
costs follow. First, the drafter pays for decoding every argument
token. Second, one text sample commits to one branch, so raising the
effective match probability requires sampling repeatedly.

This module defines the contract for a drafter that instead consumes
the actor's own latent state and emits a *set* of candidate action
signatures with calibrated confidences in a single pass. It mirrors
``tinyvllm.speculative.adapter`` deliberately: the token-level runtime
already accepts a drafter that declares ``requires_target_hidden``,
and reusing that shape keeps the two speculation levels reviewable
side by side.

This module is contract-only. It contains no model, performs no
inference, and asserts no accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Protocol

from tinyvllm.agentspec.action import (
    ActionSignature,
    ToolContract,
)


ActionRepresentation = Literal[
    "text",
    "continuous_latent",
    "discrete_code",
]

_CONFIDENCE_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class LatentActionDraftCapabilities:
    source_type: str
    representation: ActionRepresentation
    requires_target_hidden: bool
    requires_compressed_kv: bool
    max_candidate_actions: int
    max_horizon_actions: int
    execution_domain: str = "host"
    emits_match_confidence: bool = False


@dataclass(frozen=True)
class LatentActionDraftContext:
    trajectory_id: int
    step_index: int
    committed_action_digests: tuple
    tool_contracts: tuple
    target_hidden: object | None = None
    compressed_kv_handle: object | None = None


@dataclass(frozen=True)
class LatentActionCandidate:
    signature: ActionSignature
    confidence: float


@dataclass(frozen=True)
class LatentActionDraftProposal:
    trajectory_id: int
    step_index: int
    candidates: tuple
    source_type: str
    draft_gpu_seconds: float
    metadata: object | None = None
    timing_ms: object | None = None


class LatentActionDraftAdapter(Protocol):
    @property
    def capabilities(self) -> LatentActionDraftCapabilities:
        ...

    def propose(
        self,
        context: LatentActionDraftContext,
    ) -> LatentActionDraftProposal:
        ...


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return normalized


def build_capabilities(
    *,
    source_type: object,
    representation: object,
    requires_target_hidden: object,
    requires_compressed_kv: object,
    max_candidate_actions: object,
    max_horizon_actions: object,
    execution_domain: object = "host",
    emits_match_confidence: object = False,
) -> LatentActionDraftCapabilities:
    if representation not in (
        "text",
        "continuous_latent",
        "discrete_code",
    ):
        raise ValueError("representation is not a known kind")
    needs_hidden = _require_bool(
        requires_target_hidden,
        "requires_target_hidden",
    )
    if representation != "text" and not needs_hidden:
        raise ValueError(
            "latent representations must require target hidden state"
        )
    return LatentActionDraftCapabilities(
        _require_str(source_type, "source_type"),
        representation,
        needs_hidden,
        _require_bool(
            requires_compressed_kv,
            "requires_compressed_kv",
        ),
        _require_positive_int(
            max_candidate_actions,
            "max_candidate_actions",
        ),
        _require_positive_int(
            max_horizon_actions,
            "max_horizon_actions",
        ),
        _require_str(execution_domain, "execution_domain"),
        _require_bool(
            emits_match_confidence,
            "emits_match_confidence",
        ),
    )


def validate_context(
    capabilities: LatentActionDraftCapabilities,
    context: LatentActionDraftContext,
) -> LatentActionDraftContext:
    if not isinstance(context, LatentActionDraftContext):
        raise ValueError("context must be LatentActionDraftContext")
    if not isinstance(context.committed_action_digests, tuple):
        raise ValueError("committed_action_digests must be a tuple")
    for digest in context.committed_action_digests:
        _require_str(digest, "committed action digest")
    if not isinstance(context.tool_contracts, tuple):
        raise ValueError("tool_contracts must be a tuple")
    for contract in context.tool_contracts:
        if not isinstance(contract, ToolContract):
            raise ValueError("tool_contracts items must be contracts")
    if capabilities.requires_target_hidden:
        if context.target_hidden is None:
            raise ValueError(
                "drafter requires target hidden state but none given"
            )
    if capabilities.requires_compressed_kv:
        if context.compressed_kv_handle is None:
            raise ValueError(
                "drafter requires a compressed KV handle but none "
                "given"
            )
    return context


def validate_proposal(
    capabilities: LatentActionDraftCapabilities,
    proposal: LatentActionDraftProposal,
) -> LatentActionDraftProposal:
    """Validate one proposal against the declared capabilities.

    Confidences are treated as marginal probabilities over mutually
    exclusive next actions, so they must be within ``[0, 1]`` and must
    not sum above one. Candidates must be unique by digest and sorted
    by descending confidence so that branch selection is
    deterministic.
    """

    if not isinstance(proposal, LatentActionDraftProposal):
        raise ValueError("proposal must be LatentActionDraftProposal")
    if proposal.source_type != capabilities.source_type:
        raise ValueError("proposal source_type does not match drafter")
    _require_non_negative_float(
        proposal.draft_gpu_seconds,
        "draft_gpu_seconds",
    )
    candidates = proposal.candidates
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("candidates must be a non-empty tuple")
    if len(candidates) > capabilities.max_candidate_actions:
        raise ValueError(
            "candidate count exceeds max_candidate_actions"
        )
    seen = set()
    total = 0.0
    previous = None
    for candidate in candidates:
        if not isinstance(candidate, LatentActionCandidate):
            raise ValueError(
                "candidates items must be LatentActionCandidate"
            )
        if not isinstance(candidate.signature, ActionSignature):
            raise ValueError("candidate signature is not canonical")
        confidence = candidate.confidence
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
        ):
            raise ValueError("candidate confidence must be finite")
        confidence = float(confidence)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(
                "candidate confidence must be within [0, 1]"
            )
        if not capabilities.emits_match_confidence:
            if confidence != 1.0:
                raise ValueError(
                    "drafter without calibrated confidence must "
                    "report 1.0"
                )
        digest = candidate.signature.digest
        if digest in seen:
            raise ValueError("duplicate candidate action digest")
        seen.add(digest)
        if previous is not None and confidence > previous:
            raise ValueError(
                "candidates must be sorted by descending confidence"
            )
        previous = confidence
        total += confidence
    if capabilities.emits_match_confidence:
        if total > 1.0 + _CONFIDENCE_SUM_TOLERANCE:
            raise ValueError("candidate confidences sum above one")
    elif len(candidates) != 1:
        raise ValueError(
            "drafter without calibrated confidence must emit one "
            "candidate"
        )
    return proposal


def eligible_candidates(
    proposal: LatentActionDraftProposal,
    tool_contracts: tuple,
) -> tuple:
    """Drop candidates whose tool is not speculation eligible."""

    lookup = {}
    for contract in tool_contracts:
        if not isinstance(contract, ToolContract):
            raise ValueError("tool_contracts items must be contracts")
        lookup[contract.tool_name] = contract
    kept = []
    for candidate in proposal.candidates:
        contract = lookup.get(candidate.signature.tool_name)
        if contract is None:
            continue
        if not contract.speculation_eligible:
            continue
        kept.append(candidate)
    return tuple(kept)


def aggregate_match_probability(
    candidates: tuple,
    branch_count: int,
) -> float:
    """Sum the confidences of the branches actually executed.

    A latent drafter emits a candidate set in one pass, so widening
    from one branch to ``branch_count`` branches raises the match
    probability without raising drafter GPU cost. It does raise the
    number of speculative tool invocations, which the cost model
    accounts for separately.
    """

    if isinstance(branch_count, bool) or not isinstance(
        branch_count,
        int,
    ):
        raise ValueError("branch_count must be an integer")
    if branch_count <= 0:
        raise ValueError("branch_count must be > 0")
    total = 0.0
    for candidate in candidates[:branch_count]:
        total += float(candidate.confidence)
    return min(total, 1.0)


def capabilities_to_dict(
    capabilities: LatentActionDraftCapabilities,
) -> dict:
    return asdict(capabilities)
