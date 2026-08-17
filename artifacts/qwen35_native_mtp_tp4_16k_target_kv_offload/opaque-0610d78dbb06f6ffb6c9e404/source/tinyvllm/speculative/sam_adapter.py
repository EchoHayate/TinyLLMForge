from __future__ import annotations

from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftContext,
    DraftProposal,
)
from tinyvllm.speculative.sam import SuffixAutomatonDraftIndex


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_positive_integer(value: object, name: str) -> int:
    normalized = _validate_integer(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be > 0")
    return normalized


def _validate_token_ids(
    token_ids: object,
    name: str,
) -> tuple[int, ...]:
    if not isinstance(token_ids, tuple):
        raise ValueError(f"{name} must be a tuple")
    normalized = []
    for token_id in token_ids:
        normalized.append(
            _validate_integer(token_id, f"{name} token")
        )
    return tuple(normalized)


class SAMDraftAdapter:
    def __init__(
        self,
        *,
        max_proposal_tokens: int,
        match_aware: bool = False,
    ):
        if not isinstance(match_aware, bool):
            raise ValueError("match_aware must be a bool")
        self._match_aware = match_aware
        self._capabilities = DraftCapabilities(
            source_type="sam",
            supports_batch=True,
            requires_target_hidden=False,
            requires_target_logits=False,
            max_proposal_tokens=_validate_positive_integer(
                max_proposal_tokens,
                "max_proposal_tokens",
            ),
        )
        self._indexes: dict[int, SuffixAutomatonDraftIndex] = {}

    @property
    def capabilities(self) -> DraftCapabilities:
        return self._capabilities

    def register_sequence(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> None:
        normalized_id = _validate_integer(
            sequence_id,
            "sequence_id",
        )
        normalized_tokens = _validate_token_ids(
            verified_token_ids,
            "verified_token_ids",
        )
        if normalized_id in self._indexes:
            raise ValueError(
                f"sequence {normalized_id} is already registered"
            )
        self._indexes[normalized_id] = (
            SuffixAutomatonDraftIndex(
                list(normalized_tokens)
            )
        )

    def synchronize_verified_history(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> int:
        normalized_id = _validate_integer(
            sequence_id,
            "sequence_id",
        )
        normalized_tokens = _validate_token_ids(
            verified_token_ids,
            "verified_token_ids",
        )
        index = self._require_index(normalized_id)
        indexed_tokens = tuple(index.indexed_tokens)
        if (
            len(normalized_tokens) < len(indexed_tokens)
            or normalized_tokens[:len(indexed_tokens)]
            != indexed_tokens
        ):
            raise ValueError(
                "target-verified history does not preserve "
                "the SAM indexed prefix"
            )
        extension = normalized_tokens[len(indexed_tokens):]
        index.extend_verified(list(extension))
        index.assert_history(list(normalized_tokens))
        return len(extension)

    def release_sequence(self, sequence_id: int) -> None:
        normalized_id = _validate_integer(
            sequence_id,
            "sequence_id",
        )
        self._require_index(normalized_id)
        del self._indexes[normalized_id]

    def propose_batch(
        self,
        contexts: tuple[DraftContext, ...],
    ) -> tuple[DraftProposal, ...]:
        proposals = []
        for context in contexts:
            index = self._require_index(context.sequence_id)
            index.assert_history(list(context.token_ids))
            adapter_limit = min(
                self._capabilities.max_proposal_tokens,
                context.max_proposal_tokens,
            )
            if self._match_aware:
                policy_draft = index.propose_match_aware()
                policy_selected_k = policy_draft.selected_k
                selected_k = min(
                    adapter_limit,
                    policy_selected_k,
                )
                draft = (
                    policy_draft
                    if selected_k == policy_selected_k
                    else index.propose(selected_k)
                )
                policy = "match_aware"
            else:
                policy_selected_k = adapter_limit
                selected_k = adapter_limit
                draft = index.propose(selected_k)
                policy = "fixed"
            metadata = dict(draft.metadata)
            metadata.update(
                {
                    "policy": policy,
                    "policy_selected_k": policy_selected_k,
                    "adapter_limit": adapter_limit,
                    "history_token_count": len(
                        context.token_ids
                    ),
                }
            )
            proposals.append(
                DraftProposal(
                    sequence_id=context.sequence_id,
                    token_ids=tuple(draft.tokens),
                    source_type="sam",
                    metadata=metadata,
                )
            )
        return tuple(proposals)

    def _require_index(
        self,
        sequence_id: int,
    ) -> SuffixAutomatonDraftIndex:
        normalized_id = _validate_integer(
            sequence_id,
            "sequence_id",
        )
        index = self._indexes.get(normalized_id)
        if index is None:
            raise ValueError(
                f"sequence {normalized_id} is not registered"
            )
        return index
