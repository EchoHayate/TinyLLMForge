"""Speculative decoding helpers."""

from tinyvllm.speculative.ngram import NGramDraft, NGramReplayStats, propose_ngram_draft, replay_ngram_acceptance

__all__ = [
    "NGramDraft",
    "NGramReplayStats",
    "propose_ngram_draft",
    "replay_ngram_acceptance",
]
