from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class DraftCapabilities:
    source_type: str
    supports_batch: bool
    requires_target_hidden: bool
    requires_target_logits: bool
    max_proposal_tokens: int
    execution_domain: str = "host"
    requires_proposal_lifecycle: bool = False
    requires_full_token_history: bool = True


@dataclass(frozen=True)
class DraftContext:
    sequence_id: int
    token_ids: tuple[int, ...]
    remaining_output_tokens: int
    max_proposal_tokens: int
    first_target_token: int
    target_hidden: object | None = None
    target_logits: object | None = None


@dataclass(frozen=True)
class DraftProposal:
    sequence_id: int
    token_ids: tuple[int, ...]
    source_type: str
    metadata: object | None = None
    timing_ms: dict[str, float] | None = None
    proposal_transaction_id: str | None = None


class DraftAdapter(Protocol):
    @property
    def capabilities(self) -> DraftCapabilities:
        ...

    def propose_batch(
        self,
        contexts: tuple[DraftContext, ...],
    ) -> tuple[DraftProposal, ...]:
        ...


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_token_ids(
    token_ids: object,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(token_ids, tuple):
        raise ValueError(f"{name} token_ids must be a tuple")
    for token_id in token_ids:
        _validate_integer(token_id, f"{name} token")
    return token_ids


def validate_draft_capabilities(
    capabilities: object,
    *,
    expected_execution_domain: str | None = None,
) -> DraftCapabilities:
    if not isinstance(capabilities, DraftCapabilities):
        raise ValueError(
            "adapter capabilities must be DraftCapabilities"
        )
    if capabilities.execution_domain not in (
        "host",
        "model_runner",
    ):
        raise ValueError(
            "adapter capability execution domain is unsupported"
        )
    if (
        expected_execution_domain is not None
        and capabilities.execution_domain
        != expected_execution_domain
    ):
        raise ValueError(
            "adapter capability execution domain must be "
            f"{expected_execution_domain}"
        )
    if (
        not isinstance(capabilities.source_type, str)
        or not capabilities.source_type
    ):
        raise ValueError(
            "adapter capability source_type must be non-empty"
        )
    if capabilities.supports_batch is not True:
        raise ValueError(
            "adapter capabilities must support batch proposals"
        )
    if not isinstance(
        capabilities.requires_target_hidden,
        bool,
    ):
        raise ValueError(
            "adapter capability requires_target_hidden must be bool"
        )
    if not isinstance(
        capabilities.requires_target_logits,
        bool,
    ):
        raise ValueError(
            "adapter capability requires_target_logits must be bool"
        )
    if not isinstance(
        capabilities.requires_proposal_lifecycle,
        bool,
    ):
        raise ValueError(
            "adapter capability lifecycle requirement must be bool"
        )
    if not isinstance(
        capabilities.requires_full_token_history,
        bool,
    ):
        raise ValueError(
            "adapter capability full token history requirement "
            "must be bool"
        )
    if (
        capabilities.requires_proposal_lifecycle
        and capabilities.execution_domain != "model_runner"
    ):
        raise ValueError(
            "adapter proposal lifecycle requires model_runner execution"
        )
    _validate_integer(
        capabilities.max_proposal_tokens,
        "adapter capability max_proposal_tokens",
    )
    if capabilities.max_proposal_tokens <= 0:
        raise ValueError(
            "adapter capability max_proposal_tokens must be > 0"
        )
    return capabilities


def _validate_contexts(
    contexts: tuple[DraftContext, ...],
    capabilities: DraftCapabilities,
) -> tuple[int, ...]:
    if not isinstance(contexts, tuple) or not contexts:
        raise ValueError(
            "draft contexts must be a non-empty tuple"
        )
    sequence_ids = []
    for context in contexts:
        if not isinstance(context, DraftContext):
            raise ValueError(
                "draft context must be a DraftContext"
            )
        sequence_id = _validate_integer(
            context.sequence_id,
            "context sequence_id",
        )
        _validate_token_ids(
            context.token_ids,
            "context",
        )
        _validate_integer(
            context.remaining_output_tokens,
            "context remaining_output_tokens",
        )
        if context.remaining_output_tokens < 0:
            raise ValueError(
                "context remaining_output_tokens must be >= 0"
            )
        _validate_integer(
            context.max_proposal_tokens,
            "context max_proposal_tokens",
        )
        if context.max_proposal_tokens < 0:
            raise ValueError(
                "context max_proposal_tokens must be >= 0"
            )
        _validate_integer(
            context.first_target_token,
            "context first_target_token",
        )
        if (
            capabilities.requires_target_hidden
            and context.target_hidden is None
        ):
            raise ValueError(
                "adapter requires target hidden payload"
            )
        if (
            capabilities.requires_target_logits
            and context.target_logits is None
        ):
            raise ValueError(
                "adapter requires target logits payload"
            )
        sequence_ids.append(sequence_id)
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "context sequence IDs must be unique"
        )
    return tuple(sequence_ids)


def _validate_timing(
    timing_ms: object,
) -> None:
    if timing_ms is None:
        return
    if not isinstance(timing_ms, dict):
        raise ValueError(
            "proposal timing_ms must be a dictionary"
        )
    for name, value in timing_ms.items():
        if not isinstance(name, str):
            raise ValueError(
                "proposal timing names must be strings"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                "proposal timing values must be finite and non-negative"
            )


def validate_draft_adapter_batch(
    adapter: DraftAdapter,
    contexts: tuple[DraftContext, ...],
) -> tuple[DraftProposal, ...]:
    capabilities = validate_draft_capabilities(
        getattr(adapter, "capabilities", None),
        expected_execution_domain="host",
    )
    sequence_ids = _validate_contexts(
        contexts,
        capabilities,
    )
    propose_batch = getattr(adapter, "propose_batch", None)
    if not callable(propose_batch):
        raise ValueError(
            "draft adapter propose_batch must be callable"
        )
    proposals = propose_batch(contexts)
    if not isinstance(proposals, tuple):
        raise ValueError(
            "draft adapter must return a tuple of proposals"
        )

    proposals_by_id = {}
    for proposal in proposals:
        if not isinstance(proposal, DraftProposal):
            raise ValueError(
                "draft adapter result must be DraftProposal"
            )
        sequence_id = _validate_integer(
            proposal.sequence_id,
            "proposal sequence_id",
        )
        if sequence_id in proposals_by_id:
            raise ValueError(
                "proposal sequence IDs must be unique"
            )
        if (
            not isinstance(proposal.source_type, str)
            or not proposal.source_type
            or proposal.source_type
            != capabilities.source_type
        ):
            raise ValueError(
                "proposal source_type must match adapter source_type"
            )
        token_ids = _validate_token_ids(
            proposal.token_ids,
            "proposal",
        )
        _validate_timing(proposal.timing_ms)
        proposals_by_id[sequence_id] = proposal

        context = next(
            (
                item
                for item in contexts
                if item.sequence_id == sequence_id
            ),
            None,
        )
        if context is None:
            continue
        if (
            len(token_ids)
            > capabilities.max_proposal_tokens
        ):
            raise ValueError(
                "proposal length exceeds capability limit"
            )
        if len(token_ids) > context.max_proposal_tokens:
            raise ValueError(
                "proposal length exceeds context limit"
            )

    if set(proposals_by_id) != set(sequence_ids):
        raise ValueError(
            "proposal sequence IDs must exactly match context IDs"
        )
    return tuple(
        proposals_by_id[sequence_id]
        for sequence_id in sequence_ids
    )
