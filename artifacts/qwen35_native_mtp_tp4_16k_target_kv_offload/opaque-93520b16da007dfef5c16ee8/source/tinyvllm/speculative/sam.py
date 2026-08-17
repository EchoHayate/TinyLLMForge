from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SAMState:
    max_length: int
    suffix_link: int
    transitions: dict[int, int] = field(default_factory=dict)
    first_end_position: int = -1


@dataclass(frozen=True)
class SAMMatch:
    match_length: int
    match_start: int
    match_end: int
    continuation_start: int
    available_continuation_tokens: int
    continuation_region: str


@dataclass(frozen=True)
class SAMDraft:
    tokens: list[int]
    selected_k: int
    match: SAMMatch | None
    metadata: dict[str, object]


class SuffixAutomatonDraftIndex:
    def __init__(self, prompt_tokens: list[int]):
        self.prompt_length = len(prompt_tokens)
        self.indexed_tokens: list[int] = []
        self.states = [SAMState(max_length=0, suffix_link=-1)]
        self.last_state = 0
        self.extend_verified(prompt_tokens)

    def _extend_one(self, token_id: int) -> None:
        position = len(self.indexed_tokens)
        current = len(self.states)
        self.states.append(
            SAMState(
                max_length=self.states[self.last_state].max_length + 1,
                suffix_link=0,
                first_end_position=position,
            )
        )
        previous = self.last_state
        while previous >= 0 and token_id not in self.states[previous].transitions:
            self.states[previous].transitions[token_id] = current
            previous = self.states[previous].suffix_link
        if previous < 0:
            self.states[current].suffix_link = 0
        else:
            target = self.states[previous].transitions[token_id]
            if (
                self.states[previous].max_length + 1
                == self.states[target].max_length
            ):
                self.states[current].suffix_link = target
            else:
                clone = len(self.states)
                self.states.append(
                    SAMState(
                        max_length=self.states[previous].max_length + 1,
                        suffix_link=self.states[target].suffix_link,
                        transitions=dict(self.states[target].transitions),
                        first_end_position=self.states[target].first_end_position,
                    )
                )
                while (
                    previous >= 0
                    and self.states[previous].transitions.get(token_id) == target
                ):
                    self.states[previous].transitions[token_id] = clone
                    previous = self.states[previous].suffix_link
                self.states[target].suffix_link = clone
                self.states[current].suffix_link = clone
        self.last_state = current
        self.indexed_tokens.append(int(token_id))

    def extend_verified(self, tokens: list[int]) -> None:
        for token_id in tokens:
            self._extend_one(int(token_id))

    def assert_history(self, history: list[int]) -> None:
        normalized = [int(token_id) for token_id in history]
        if normalized != self.indexed_tokens:
            raise ValueError(
                "SAM indexed stream does not match target-verified history"
            )

    def longest_usable_suffix(self) -> SAMMatch | None:
        state_id = self.last_state
        terminal_position = len(self.indexed_tokens) - 1
        while state_id > 0:
            state = self.states[state_id]
            match_length = state.max_length
            match_end = state.first_end_position
            match_start = match_end - match_length + 1
            continuation_start = match_end + 1
            if (
                match_length >= 2
                and match_start >= 0
                and match_end != terminal_position
                and continuation_start < len(self.indexed_tokens)
            ):
                return SAMMatch(
                    match_length=match_length,
                    match_start=match_start,
                    match_end=match_end,
                    continuation_start=continuation_start,
                    available_continuation_tokens=(
                        len(self.indexed_tokens) - continuation_start
                    ),
                    continuation_region=(
                        "prompt"
                        if continuation_start < self.prompt_length
                        else "generated"
                    ),
                )
            state_id = state.suffix_link
        return None

    def propose(self, max_draft_tokens: int) -> SAMDraft:
        if max_draft_tokens < 0:
            raise ValueError("max_draft_tokens must be >= 0")
        match = self.longest_usable_suffix()
        if max_draft_tokens == 0 or match is None:
            tokens = []
        else:
            draft_end = min(
                len(self.indexed_tokens),
                match.continuation_start + max_draft_tokens,
            )
            tokens = self.indexed_tokens[match.continuation_start:draft_end]
        metadata = {
            "match_length": 0 if match is None else match.match_length,
            "match_start": -1 if match is None else match.match_start,
            "match_end": -1 if match is None else match.match_end,
            "continuation_start": (
                -1 if match is None else match.continuation_start
            ),
            "available_continuation_tokens": (
                0 if match is None else match.available_continuation_tokens
            ),
            "continuation_region": (
                "none" if match is None else match.continuation_region
            ),
            "selected_k": int(max_draft_tokens),
            "index_token_count": len(self.indexed_tokens),
            "index_state_count": len(self.states),
        }
        copied_end = (
            -1 if match is None else match.continuation_start + len(tokens)
        )
        metadata["copied_span_crosses_prompt_boundary"] = (
            match is not None
            and match.continuation_start < self.prompt_length < copied_end
        )
        metadata["bypass_reason"] = (
            "selected_k_zero"
            if max_draft_tokens == 0
            else "no_usable_match"
            if match is None
            else None
        )
        return SAMDraft(
            tokens=list(tokens),
            selected_k=int(max_draft_tokens),
            match=match,
            metadata=metadata,
        )

    def propose_match_aware(self) -> SAMDraft:
        match = self.longest_usable_suffix()
        selected_k = select_match_aware_k(
            0 if match is None else match.match_length
        )
        draft = self.propose(selected_k)
        if selected_k == 0:
            draft.metadata["bypass_reason"] = "no_usable_match"
        return draft


def select_match_aware_k(match_length: int) -> int:
    if match_length < 0:
        raise ValueError("match_length must be >= 0")
    if match_length < 2:
        return 0
    if match_length < 4:
        return 4
    if match_length < 8:
        return 8
    return 16
