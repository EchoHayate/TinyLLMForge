from __future__ import annotations

"""N-gram draft helpers for speculative decoding research.

This module is intentionally CPU-only and side-effect free. It does not mutate KV
cache or engine state; it lets us estimate whether n-gram speculation is worth a
deeper target-verify integration.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NGramDraft:
    tokens: list[int]
    match_start: int
    ngram_size: int


@dataclass(frozen=True)
class NGramReplayStats:
    positions: int
    drafted_tokens: int
    accepted_tokens: int
    draft_events: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    @property
    def avg_draft_len(self) -> float:
        return self.drafted_tokens / self.draft_events if self.draft_events else 0.0


def propose_ngram_draft(history: list[int], ngram_size: int, max_draft_tokens: int) -> NGramDraft:
    """Draft continuation tokens by finding the latest previous suffix match.

    If the last ``ngram_size`` tokens appeared earlier in ``history``, reuse the
    tokens that followed that earlier occurrence as the draft continuation.
    """
    if ngram_size <= 0:
        raise ValueError("ngram_size must be > 0")
    if max_draft_tokens <= 0:
        raise ValueError("max_draft_tokens must be > 0")
    if len(history) < ngram_size:
        return NGramDraft(tokens=[], match_start=-1, ngram_size=ngram_size)

    suffix = history[-ngram_size:]
    match_start = -1
    for i in range(0, len(history) - ngram_size):
        if history[i:i + ngram_size] == suffix:
            match_start = i

    if match_start < 0:
        return NGramDraft(tokens=[], match_start=-1, ngram_size=ngram_size)

    draft_start = match_start + ngram_size
    draft_end = min(len(history), draft_start + max_draft_tokens)
    return NGramDraft(tokens=history[draft_start:draft_end], match_start=match_start, ngram_size=ngram_size)


def replay_ngram_acceptance(
    tokens: list[int],
    prompt_len: int,
    ngram_size: int,
    max_draft_tokens: int,
) -> NGramReplayStats:
    """Replay a known token stream and estimate n-gram draft acceptance.

    ``tokens`` should be ``prompt_token_ids + generated_token_ids``. At each
    generated position, the function drafts from prior history and compares that
    draft against the known future tokens.
    """
    if prompt_len < 0 or prompt_len > len(tokens):
        raise ValueError("prompt_len must be within the token stream")

    positions = 0
    drafted_tokens = 0
    accepted_tokens = 0
    draft_events = 0

    for pos in range(prompt_len, len(tokens)):
        draft = propose_ngram_draft(tokens[:pos], ngram_size, max_draft_tokens)
        positions += 1
        if not draft.tokens:
            continue

        draft_events += 1
        drafted_tokens += len(draft.tokens)
        future = tokens[pos:pos + len(draft.tokens)]
        for draft_token, future_token in zip(draft.tokens, future):
            if draft_token != future_token:
                break
            accepted_tokens += 1

    return NGramReplayStats(
        positions=positions,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        draft_events=draft_events,
    )


def summarize_replay_stats(stats: NGramReplayStats) -> dict[str, float | int]:
    return {
        "positions": stats.positions,
        "draft_events": stats.draft_events,
        "drafted_tokens": stats.drafted_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "acceptance_rate": stats.acceptance_rate,
        "avg_draft_len": stats.avg_draft_len,
    }
