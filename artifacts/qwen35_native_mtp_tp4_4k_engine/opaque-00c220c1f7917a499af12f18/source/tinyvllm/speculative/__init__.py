"""Speculative decoding helpers."""

from tinyvllm.speculative.adapter import (
    DraftAdapter,
    DraftCapabilities,
    DraftContext,
    DraftProposal,
    validate_draft_capabilities,
    validate_draft_adapter_batch,
)
from tinyvllm.speculative.batch_runtime import (
    FirstTargetProposalResult,
    FirstTargetResult,
    NativeSpeculativeBatchError,
    NativeSpeculativeBatchResult,
    NativeSpeculativeSequenceResult,
    PreparedNativeSpeculativeBatch,
    PreparedNativeSpeculativeSequence,
    TailBatchItem,
    TailBatchResult,
    commit_prepared_native_speculative_batch,
    execute_native_speculative_batch,
    prepare_native_speculative_batch,
    rollback_prepared_native_speculative_batch,
)
from tinyvllm.speculative.ngram import (
    NGramDraft,
    NGramReplayStats,
    propose_ngram_draft,
    replay_ngram_acceptance,
)
from tinyvllm.speculative.ngram_adapter import NGramDraftAdapter
from tinyvllm.speculative.sam_adapter import SAMDraftAdapter

__all__ = [
    "DraftAdapter",
    "DraftCapabilities",
    "DraftContext",
    "DraftProposal",
    "validate_draft_capabilities",
    "validate_draft_adapter_batch",
    "FirstTargetProposalResult",
    "FirstTargetResult",
    "TailBatchItem",
    "TailBatchResult",
    "NativeSpeculativeSequenceResult",
    "NativeSpeculativeBatchResult",
    "NativeSpeculativeBatchError",
    "PreparedNativeSpeculativeSequence",
    "PreparedNativeSpeculativeBatch",
    "prepare_native_speculative_batch",
    "commit_prepared_native_speculative_batch",
    "rollback_prepared_native_speculative_batch",
    "execute_native_speculative_batch",
    "NGramDraftAdapter",
    "SAMDraftAdapter",
    "NGramDraft",
    "NGramReplayStats",
    "propose_ngram_draft",
    "replay_ngram_acceptance",
]
