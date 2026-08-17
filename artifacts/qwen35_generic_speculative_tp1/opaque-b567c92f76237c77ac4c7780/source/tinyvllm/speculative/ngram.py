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


@dataclass
class AdaptiveDraftState:
    levels: tuple[int, ...] = (1, 2, 4)
    level_index: int = 1
    acceptance_ema: float = 0.5
    full_accept_streak: int = 0
    proposal_events: int = 0

    def __post_init__(self):
        if self.levels != (1, 2, 4):
            raise ValueError("adaptive levels must be exactly (1, 2, 4)")
        if self.level_index < 0 or self.level_index >= len(self.levels):
            raise ValueError("level_index is outside adaptive levels")
        if not 0.0 <= self.acceptance_ema <= 1.0:
            raise ValueError("acceptance_ema must be within [0, 1]")
        if self.full_accept_streak < 0 or self.proposal_events < 0:
            raise ValueError("adaptive counters must be >= 0")

    @property
    def selected_k(self) -> int:
        return self.levels[self.level_index]


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


@dataclass
class NGramOnlineDryRunState:
    """Per-sequence online dry-run state for n-gram speculative opportunities."""

    pending_tokens: list[int]
    active_match_start: int = -1
    active_drafted_tokens: int = 0


@dataclass
class NGramOnlineDryRunTotals:
    decode_positions: int = 0
    draft_events: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    rejected_events: int = 0
    completed_drafts: int = 0
    no_draft_positions: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    @property
    def avg_draft_len(self) -> float:
        return self.drafted_tokens / self.draft_events if self.draft_events else 0.0

    @property
    def draft_coverage(self) -> float:
        return self.draft_events / self.decode_positions if self.decode_positions else 0.0

    @property
    def theoretical_decode_step_reduction(self) -> float:
        # If accepted tokens were verified in one target pass, each accepted token
        # beyond the first one in an event can remove one future autoregressive step.
        return self.accepted_tokens / (self.decode_positions + self.accepted_tokens) if (self.decode_positions + self.accepted_tokens) else 0.0


@dataclass
class NGramTargetVerifyStats:
    verify_events: int = 0
    verified_tokens: int = 0
    target_accepted_tokens: int = 0
    replay_accepted_tokens: int = 0
    mismatched_events: int = 0
    truncated_future_events: int = 0

    @property
    def target_acceptance_rate(self) -> float:
        return self.target_accepted_tokens / self.verified_tokens if self.verified_tokens else 0.0

    @property
    def replay_acceptance_rate(self) -> float:
        return self.replay_accepted_tokens / self.verified_tokens if self.verified_tokens else 0.0

    @property
    def mismatch_rate(self) -> float:
        return self.mismatched_events / self.verify_events if self.verify_events else 0.0


def update_adaptive_draft_state(
    state: AdaptiveDraftState,
    proposed: int,
    accepted: int,
) -> dict[str, int | float | str | bool | list[int]]:
    if proposed <= 0:
        raise ValueError("proposed must be > 0")
    if accepted < 0 or accepted > proposed:
        raise ValueError("accepted must be within [0, proposed]")

    selected_k_before = state.selected_k
    acceptance_ema_before = state.acceptance_ema
    full_accept_streak_before = state.full_accept_streak
    event_acceptance = accepted / proposed
    state.acceptance_ema = 0.5 * event_acceptance + 0.5 * state.acceptance_ema
    state.proposal_events += 1
    transition_reason = "hold"

    if accepted == 0:
        state.full_accept_streak = 0
        state.level_index = 0
        transition_reason = "zero_accept"
    elif event_acceptance < 0.5 or state.acceptance_ema < 0.5:
        state.full_accept_streak = 0
        state.level_index = max(0, state.level_index - 1)
        transition_reason = "weak_acceptance"
    elif accepted == proposed:
        state.full_accept_streak += 1
        transition_reason = "full_accept_streak"
        if state.acceptance_ema >= 0.75 and state.full_accept_streak >= 2:
            state.level_index = min(len(state.levels) - 1, state.level_index + 1)
            state.full_accept_streak = 0
            transition_reason = "promote"
    else:
        state.full_accept_streak = 0

    selected_k_after = state.selected_k
    return {
        "levels": list(state.levels),
        "proposal_event": state.proposal_events,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "event_acceptance": event_acceptance,
        "acceptance_ema_before": acceptance_ema_before,
        "acceptance_ema_after": state.acceptance_ema,
        "full_accept_streak_before": full_accept_streak_before,
        "full_accept_streak_after": state.full_accept_streak,
        "selected_k_before": selected_k_before,
        "selected_k_after": selected_k_after,
        "transition_reason": transition_reason,
        "promoted": selected_k_after > selected_k_before,
        "demoted": selected_k_after < selected_k_before,
    }


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


def count_accepted_prefix(draft_tokens: list[int], target_tokens: list[int]) -> int:
    """Count how many draft tokens match the target continuation prefix."""
    accepted = 0
    for draft_token, target_token in zip(draft_tokens, target_tokens):
        if int(draft_token) != int(target_token):
            break
        accepted += 1
    return accepted


def ngram_online_dry_run_step(
    history_before_decode: list[int],
    actual_next_token: int,
    state: NGramOnlineDryRunState,
    totals: NGramOnlineDryRunTotals,
    ngram_size: int,
    max_draft_tokens: int,
) -> dict[str, int | bool | list[int]]:
    """Advance online dry-run stats by one real decode token.

    This is side-effect free with respect to model/engine state. It proposes a
    draft when no draft is pending, then compares the next real token against the
    pending draft prefix. Multi-token acceptance is accumulated across future
    decode calls, which is enough to estimate online opportunity without KV
    mutation or target verification.
    """
    totals.decode_positions += 1
    proposed = False
    draft_tokens: list[int] = []
    if not state.pending_tokens:
        draft = propose_ngram_draft(history_before_decode, ngram_size, max_draft_tokens)
        if draft.tokens:
            state.pending_tokens = list(draft.tokens)
            state.active_match_start = draft.match_start
            state.active_drafted_tokens = len(draft.tokens)
            totals.draft_events += 1
            totals.drafted_tokens += len(draft.tokens)
            proposed = True
            draft_tokens = list(draft.tokens)
        else:
            totals.no_draft_positions += 1

    accepted = False
    rejected = False
    completed = False
    expected_token = state.pending_tokens[0] if state.pending_tokens else None
    if state.pending_tokens:
        if int(actual_next_token) == int(state.pending_tokens[0]):
            totals.accepted_tokens += 1
            accepted = True
            state.pending_tokens.pop(0)
            if not state.pending_tokens:
                totals.completed_drafts += 1
                completed = True
                state.active_match_start = -1
                state.active_drafted_tokens = 0
        else:
            totals.rejected_events += 1
            rejected = True
            state.pending_tokens = []
            state.active_match_start = -1
            state.active_drafted_tokens = 0

    return {
        "proposed": proposed,
        "draft_tokens": draft_tokens,
        "accepted": accepted,
        "rejected": rejected,
        "completed": completed,
        "expected_token": -1 if expected_token is None else int(expected_token),
        "actual_token": int(actual_next_token),
        "pending_after": len(state.pending_tokens),
    }


def summarize_replay_stats(stats: NGramReplayStats) -> dict[str, float | int]:
    return {
        "positions": stats.positions,
        "draft_events": stats.draft_events,
        "drafted_tokens": stats.drafted_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "acceptance_rate": stats.acceptance_rate,
        "avg_draft_len": stats.avg_draft_len,
    }


def summarize_online_dry_run_totals(stats: NGramOnlineDryRunTotals) -> dict[str, float | int]:
    return {
        "decode_positions": stats.decode_positions,
        "draft_events": stats.draft_events,
        "drafted_tokens": stats.drafted_tokens,
        "accepted_tokens": stats.accepted_tokens,
        "rejected_events": stats.rejected_events,
        "completed_drafts": stats.completed_drafts,
        "no_draft_positions": stats.no_draft_positions,
        "acceptance_rate": stats.acceptance_rate,
        "avg_draft_len": stats.avg_draft_len,
        "draft_coverage": stats.draft_coverage,
        "theoretical_decode_step_reduction": stats.theoretical_decode_step_reduction,
    }


def summarize_target_verify_stats(stats: NGramTargetVerifyStats) -> dict[str, float | int]:
    return {
        "verify_events": stats.verify_events,
        "verified_tokens": stats.verified_tokens,
        "target_accepted_tokens": stats.target_accepted_tokens,
        "replay_accepted_tokens": stats.replay_accepted_tokens,
        "mismatched_events": stats.mismatched_events,
        "truncated_future_events": stats.truncated_future_events,
        "target_acceptance_rate": stats.target_acceptance_rate,
        "replay_acceptance_rate": stats.replay_acceptance_rate,
        "mismatch_rate": stats.mismatch_rate,
    }
