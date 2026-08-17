from __future__ import annotations

from tinyvllm.speculative.adapter import (
    DraftCapabilities,
    DraftContext,
    DraftProposal,
)
from tinyvllm.speculative.ngram import propose_ngram_draft


def _validate_positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


class NGramDraftAdapter:
    def __init__(
        self,
        *,
        ngram_size: int,
        max_proposal_tokens: int,
    ):
        self._ngram_size = _validate_positive_integer(
            ngram_size,
            "ngram_size",
        )
        self._capabilities = DraftCapabilities(
            source_type="ngram",
            supports_batch=True,
            requires_target_hidden=False,
            requires_target_logits=False,
            max_proposal_tokens=_validate_positive_integer(
                max_proposal_tokens,
                "max_proposal_tokens",
            ),
        )

    @property
    def capabilities(self) -> DraftCapabilities:
        return self._capabilities

    def propose_batch(
        self,
        contexts: tuple[DraftContext, ...],
    ) -> tuple[DraftProposal, ...]:
        proposals = []
        for context in contexts:
            selected_k = min(
                self._capabilities.max_proposal_tokens,
                context.max_proposal_tokens,
            )
            if selected_k == 0:
                token_ids = ()
                match_start = -1
                bypass_reason = "selected_k_zero"
            else:
                draft = propose_ngram_draft(
                    list(context.token_ids),
                    self._ngram_size,
                    selected_k,
                )
                token_ids = tuple(draft.tokens)
                match_start = draft.match_start
                bypass_reason = (
                    None if token_ids else "no_match"
                )
            proposals.append(
                DraftProposal(
                    sequence_id=context.sequence_id,
                    token_ids=token_ids,
                    source_type="ngram",
                    metadata={
                        "ngram_size": self._ngram_size,
                        "match_start": match_start,
                        "selected_k": selected_k,
                        "history_token_count": len(
                            context.token_ids
                        ),
                        "bypass_reason": bypass_reason,
                    },
                )
            )
        return tuple(proposals)
