# Prompt+Dynamic SAM Drafter Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible profiler-only Qwen3-0.6B gate that compares a prompt+dynamic token suffix automaton with normal greedy, fixed n-gram `K=4`, adaptive n-gram, and fixed SAM `K=16`, then classifies the match-aware SAM as `GO`, `NO_GO`, or `INCOMPLETE`.

**Architecture:** Add a dependency-light `tinyvllm/speculative/sam.py` that owns token indexing, longest usable suffix lookup, continuation metadata, and match-aware `K in {0,4,8,16}` selection. Integrate one SAM index per candidate into `tools/profile_ngram_commit.py` without changing `verify_and_commit_block()`, then add a separate canonical gate/verifier and isolated remote runner that preregister five prompts, five policies, seven repetitions, paired statistics, strict evidence checks, and five canonical artifacts.

**Tech Stack:** Python 3 dataclasses and standard library (`argparse`, `hashlib`, `json`, `math`, `pathlib`, `random`, `socket`, `statistics`, `subprocess`, `time`), existing TinyLLMForge profiler and verify/commit code, Bash, SSH/SCP, Qwen3-0.6B on the existing remote CUDA host.

## Global Constraints

- The normative design is `docs/superpowers/specs/2026-07-15-sam-drafter-gate-design.md`.
- The implementation is strictly greedy, single-sequence, and profiler-owned; do not modify `LLMEngine.step()`, scheduler policy, `LLM.generate()`, persistent `Sequence` fields, target logits, accepted-prefix semantics, KV reservation/commit, EOS handling, or normal fallback decoding.
- The SAM indexes prompt tokens and only target-verified generated tokens; rejected draft tokens must never enter the index.
- SAM states contain `max_length`, `suffix_link`, `transitions`, and `first_end_position`; the v1 continuation is the earliest representative occurrence.
- Matches shorter than two tokens bypass speculation.
- Match-aware selection is fixed: length `<2 -> K=0`, `2..3 -> K=4`, `4..7 -> K=8`, `>=8 -> K=16`.
- `K=0` and empty proposals must not call `verify_and_commit_block()`.
- Every SAM event and process summary records `runtime_mutation=false` and `profiler_owned=true`.
- Canonical coverage is five committed prompts, five isolated policies, and seven repetitions: `5 * 5 * 7 = 175` unique process rows.
- Canonical policies are exactly `baseline`, `ngram_fixed_k4`, `ngram_adaptive`, `sam_fixed_k16`, and `sam_match_aware`; the completed n-gram settings are frozen.
- Pairing keys are `(repetition, prompt_name)`. Throughput comparisons use the median of 35 per-pair ratios, not ratios of independently aggregated medians.
- The primary gate requires SAM versus baseline `>= +10%`, plus either SAM versus fixed n-gram `K=4 >= +3%`, or `>= -1%` together with verify-attempt and drafted-waste reductions each `>=25%`.
- Natural, structured/code-like, transition-heavy, and prompt-copy/retrieval class medians must each remain within `-5%` of baseline.
- Missing/duplicate/failed rows, invalid timing, missing positive reduction references, output mismatch, trace mismatch, index invariant failure, missing required policy exercise, port failures after retries, or artifact/hash mismatch produce `INCOMPLETE`, never performance `NO_GO`.
- Remote execution uses `sitian@10.232.195.203`, `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`, and exact model path `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Upload into a new isolated remote directory; do not modify or trust a pre-existing remote checkout.
- Every model process receives distinct dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`; retry only narrow port-collision failures up to three times.
- Canonical evidence files are exactly `manifest.json`, `raw_rows.json`, `event_rows.json`, `summary.json`, and `report.md`.
- Do not relax thresholds after observing canonical measurements. A threshold change requires a new source commit, manifest, and run tag.
- Important commands, measured result, limitations, and next direction must be written to `README.md` and `AGENT_HANDOFF_STATE.md`.

## File Structure

- Create `tinyvllm/speculative/sam.py`: pure token SAM state, online extension, longest usable suffix query, proposal metadata, invariant checks, and cap selection.
- Create `tools/test_sam_speculative.py`: dependency-light SAM algorithm, policy, metadata, and invariant tests.
- Modify `tools/profile_ngram_commit.py`: SAM CLI choices, one-index-per-sequence lifecycle, bypass/verify events, exact history synchronization, and SAM summary fields.
- Modify `tools/test_ngram_speculative.py`: profiler dispatch, argument validation, bypass, verify/commit, event schema, and runtime-mutation regression tests.
- Modify `tools/test_chunked_prefill.py`: SAM-initiated accepted-token block-boundary regression through the unchanged verify/commit path.
- Create `tools/sam_drafter_gate.py`: prompt/policy manifest, isolated process driver, row normalization, trace reconciliation, paired metrics, fixed decision rule, artifact verifier, and report renderer.
- Create `tools/test_sam_drafter_gate.py`: synthetic 175-row `GO`/`NO_GO`/`INCOMPLETE`, branch exercise, pairing, resume, port, and five-file verifier tests.
- Create `tools/run_sam_drafter_gate_remote.sh`: isolated upload, exact remote preflight, dynamic process execution, resume, download, SHA-256 comparison, and local verification.
- Create `experiments/sam_drafter/qwen3-06b-sam-canonical-YYYYmmdd-HHMMSS/{manifest.json,raw_rows.json,event_rows.json,summary.json,report.md}`: canonical evidence after remote completion.
- Modify `README.md`: command, metrics, decision, claim boundaries, and next direction.
- Modify `AGENT_HANDOFF_STATE.md`: source SHA, remote paths, resume instructions, evidence audit, decision, and remaining work.

---

### Task 1: Pure Token Suffix Automaton

**Files:**
- Create: `tinyvllm/speculative/sam.py`
- Create: `tools/test_sam_speculative.py`

**Interfaces:**
- Produces: `SAMState(max_length: int, suffix_link: int, transitions: dict[int, int], first_end_position: int)`
- Produces: `SAMMatch(match_length: int, match_start: int, match_end: int, continuation_start: int, available_continuation_tokens: int, continuation_region: str)`
- Produces: `SAMDraft(tokens: list[int], selected_k: int, match: SAMMatch | None, metadata: dict[str, object])`
- Produces: `SuffixAutomatonDraftIndex(prompt_tokens: list[int])`
- Produces: `SuffixAutomatonDraftIndex.extend_verified(tokens: list[int]) -> None`
- Produces: `SuffixAutomatonDraftIndex.assert_history(history: list[int]) -> None`
- Produces: `SuffixAutomatonDraftIndex.longest_usable_suffix() -> SAMMatch | None`
- Produces: `SuffixAutomatonDraftIndex.propose(max_draft_tokens: int) -> SAMDraft`
- Produces: `select_match_aware_k(match_length: int) -> int`

- [ ] **Step 1: Write failing construction, lookup, and invariant tests**

Create `tools/test_sam_speculative.py` with direct import loading matching existing dependency-light tests:

```python
from __future__ import annotations

import importlib.util
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SAM_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "speculative", "sam.py")
_SPEC = importlib.util.spec_from_file_location("sam_under_test", _SAM_PATH)
sam = importlib.util.module_from_spec(_SPEC)
sys.modules["sam_under_test"] = sam
_SPEC.loader.exec_module(sam)

SuffixAutomatonDraftIndex = sam.SuffixAutomatonDraftIndex
select_match_aware_k = sam.select_match_aware_k
```

Add exact tests:

```python
def test_empty_and_one_token_histories_have_no_usable_match():
    assert SuffixAutomatonDraftIndex([]).longest_usable_suffix() is None
    assert SuffixAutomatonDraftIndex([7]).longest_usable_suffix() is None


def test_longest_usable_suffix_uses_earliest_representative():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 4, 1, 2])
    match = index.longest_usable_suffix()
    assert match is not None
    assert match.match_length == 2
    assert match.match_start == 0
    assert match.match_end == 1
    assert match.continuation_start == 2
    assert match.available_continuation_tokens == 4
    assert match.continuation_region == "prompt"


def test_terminal_only_occurrence_is_not_usable():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 4])
    assert index.longest_usable_suffix() is None


def test_suffix_link_fallback_finds_shorter_usable_suffix():
    index = SuffixAutomatonDraftIndex([9, 2, 3, 8, 2, 3])
    match = index.longest_usable_suffix()
    assert match is not None
    assert match.match_length == 2
    assert match.match_start == 1
    assert match.continuation_start == 3


def test_proposal_stops_at_observed_stream_boundary():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = index.propose(max_draft_tokens=16)
    assert draft.tokens == [3, 1, 2]
    assert draft.selected_k == 16
    assert draft.match is not None


def test_prompt_and_generated_continuation_metadata():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    index.extend_verified([8, 9, 8])
    index.assert_history([1, 2, 3, 8, 9, 8])
    match = index.longest_usable_suffix()
    assert match is not None
    assert match.continuation_region == "generated"
    assert match.continuation_start >= index.prompt_length


def test_history_invariant_rejects_missing_or_extra_tokens():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    for history in ([1, 2], [1, 2, 3, 4], [1, 9, 3]):
        try:
            index.assert_history(history)
        except ValueError:
            pass
        else:
            raise AssertionError(history)


def test_state_and_draft_metadata_are_json_friendly():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = index.propose(max_draft_tokens=4)
    assert json.loads(json.dumps(draft.metadata)) == draft.metadata
    assert draft.metadata["index_token_count"] == 5
    assert draft.metadata["index_state_count"] == len(index.states)
```

- [ ] **Step 2: Run the tests and confirm the missing module fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
```

Expected: failure loading `tinyvllm/speculative/sam.py`.

- [ ] **Step 3: Implement standard online SAM construction**

Create `tinyvllm/speculative/sam.py` with these concrete types and constructor:

```python
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
        self.states.append(SAMState(
            max_length=self.states[self.last_state].max_length + 1,
            suffix_link=0,
            first_end_position=position,
        ))
        previous = self.last_state
        while previous >= 0 and token_id not in self.states[previous].transitions:
            self.states[previous].transitions[token_id] = current
            previous = self.states[previous].suffix_link
        if previous < 0:
            self.states[current].suffix_link = 0
        else:
            target = self.states[previous].transitions[token_id]
            if self.states[previous].max_length + 1 == self.states[target].max_length:
                self.states[current].suffix_link = target
            else:
                clone = len(self.states)
                self.states.append(SAMState(
                    max_length=self.states[previous].max_length + 1,
                    suffix_link=self.states[target].suffix_link,
                    transitions=dict(self.states[target].transitions),
                    first_end_position=self.states[target].first_end_position,
                ))
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
            raise ValueError("SAM indexed stream does not match target-verified history")
```

- [ ] **Step 4: Implement suffix-link query and bounded proposal**

Add:

```python
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
            "continuation_start": -1 if match is None else match.continuation_start,
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
        return SAMDraft(
            tokens=list(tokens),
            selected_k=int(max_draft_tokens),
            match=match,
            metadata=metadata,
        )
```

- [ ] **Step 5: Run the pure SAM tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
```

Expected final line after adding a test runner: `sam speculative tests passed`.

- [ ] **Step 6: Commit the pure index**

```bash
git add tinyvllm/speculative/sam.py tools/test_sam_speculative.py
git commit -m "Add token suffix automaton draft index"
```

---

### Task 2: Match-Aware Policy and Source-Boundary Semantics

**Files:**
- Modify: `tinyvllm/speculative/sam.py`
- Modify: `tools/test_sam_speculative.py`

**Interfaces:**
- Consumes: `SAMMatch` and `SuffixAutomatonDraftIndex` from Task 1.
- Produces: `select_match_aware_k(match_length: int) -> int`
- Produces: `SuffixAutomatonDraftIndex.propose_match_aware() -> SAMDraft`
- Proposal metadata must distinguish prompt/generated start and exact copied-span boundary crossing.

- [ ] **Step 1: Add failing cap-boundary and metadata tests**

Add:

```python
def test_match_aware_k_boundaries():
    assert [select_match_aware_k(value) for value in (0, 1)] == [0, 0]
    assert [select_match_aware_k(value) for value in (2, 3)] == [4, 4]
    assert [select_match_aware_k(value) for value in (4, 7)] == [8, 8]
    assert [select_match_aware_k(value) for value in (8, 32)] == [16, 16]


def test_match_aware_bypass_for_short_match():
    index = SuffixAutomatonDraftIndex([1, 9, 1])
    draft = index.propose_match_aware()
    assert draft.selected_k == 0
    assert draft.tokens == []
    assert draft.metadata["bypass_reason"] == "no_usable_match"


def test_selected_cap_can_exceed_available_continuation():
    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = index.propose_match_aware()
    assert draft.selected_k == 4
    assert len(draft.tokens) == 3
    assert draft.metadata["available_continuation_tokens"] == 3


def test_copied_span_crossing_prompt_boundary_is_exact():
    index = SuffixAutomatonDraftIndex([1, 2, 1])
    index.extend_verified([2])
    draft = index.propose(max_draft_tokens=8)
    expected_end = draft.metadata["continuation_start"] + len(draft.tokens)
    assert draft.tokens == [1, 2]
    assert draft.metadata["copied_span_crosses_prompt_boundary"] == (
        draft.metadata["continuation_start"] < index.prompt_length < expected_end
    )
```

- [ ] **Step 2: Run and confirm missing policy interfaces**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
```

Expected: failure for undefined `select_match_aware_k` or `propose_match_aware`.

- [ ] **Step 3: Implement fixed policy mapping**

Add:

```python
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
```

Refine `propose()` metadata:

```python
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
```

Add:

```python
    def propose_match_aware(self) -> SAMDraft:
        match = self.longest_usable_suffix()
        selected_k = select_match_aware_k(
            0 if match is None else match.match_length
        )
        draft = self.propose(selected_k)
        if selected_k == 0:
            draft.metadata["bypass_reason"] = "no_usable_match"
        return draft
```

- [ ] **Step 4: Run pure tests and syntax validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
python3 -m py_compile tinyvllm/speculative/sam.py tools/test_sam_speculative.py
```

Expected: both commands exit 0.

- [ ] **Step 5: Commit policy semantics**

```bash
git add tinyvllm/speculative/sam.py tools/test_sam_speculative.py
git commit -m "Add match-aware SAM draft policy"
```

---

### Task 3: Profiler CLI and Pure Draft Dispatch

**Files:**
- Modify: `tools/profile_ngram_commit.py`
- Modify: `tools/test_ngram_speculative.py`

**Interfaces:**
- Consumes: `SuffixAutomatonDraftIndex`, `SAMDraft`, and `select_match_aware_k`.
- Produces CLI `--draft-source sam`.
- Produces CLI `--draft-policy {fixed,adaptive,sam-fixed,sam-match-aware}`.
- Produces `propose_draft(history, args, max_draft_tokens=None, sam_index=None) -> DraftProposal`.
- Produces `validate_profile_args(args)` rules for profiler-only SAM execution.

- [ ] **Step 1: Add failing import, dispatch, and validation tests**

Load the SAM module in `tools/test_ngram_speculative.py` and add:

```python
SuffixAutomatonDraftIndex = profile_ngram.SuffixAutomatonDraftIndex


def test_propose_draft_dispatches_fixed_sam_source():
    class Args:
        draft_source = "sam"
        draft_policy = "sam-fixed"
        ngram_size = 3
        max_draft_tokens = 16

    index = SuffixAutomatonDraftIndex([1, 2, 3, 1, 2])
    draft = propose_draft(
        index.indexed_tokens,
        Args(),
        max_draft_tokens=16,
        sam_index=index,
    )
    assert draft.source == "sam"
    assert draft.tokens == [3, 1, 2]
    assert draft.metadata["selected_k"] == 16


def test_propose_draft_dispatches_match_aware_sam_bypass():
    class Args:
        draft_source = "sam"
        draft_policy = "sam-match-aware"
        ngram_size = 3
        max_draft_tokens = 16

    index = SuffixAutomatonDraftIndex([1, 9, 1])
    draft = propose_draft(index.indexed_tokens, Args(), sam_index=index)
    assert draft.tokens == []
    assert draft.metadata["selected_k"] == 0
    assert draft.metadata["bypass_reason"] == "no_usable_match"


def test_sam_profile_args_require_candidate_greedy_single_sequence():
    from types import SimpleNamespace

    valid = dict(
        model="/model",
        temperature=0.0,
        max_commit_events=0,
        warmup_output_len=0,
        simulate_kv_upload_mb=0.0,
        max_draft_tokens=16,
        draft_source="sam",
        draft_policy="sam-match-aware",
        mode="candidate-only",
        max_num_seqs=1,
    )
    validate_profile_args(SimpleNamespace(**valid))
    for override in (
        {"temperature": 0.7},
        {"mode": "paired"},
        {"max_num_seqs": 2},
        {"draft_source": "ngram"},
    ):
        args = SimpleNamespace(**{**valid, **override})
        try:
            validate_profile_args(args)
        except ValueError:
            pass
        else:
            raise AssertionError(override)
```

- [ ] **Step 2: Run and confirm unsupported SAM dispatch**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
```

Expected: failure for missing SAM import, unsupported choice, or unsupported draft source.

- [ ] **Step 3: Import SAM and extend CLI choices**

Load `tinyvllm/speculative/sam.py` beside the n-gram module:

```python
_SAM_PATH = os.path.join(_REPO_ROOT, "tinyvllm", "speculative", "sam.py")
_SAM_SPEC = importlib.util.spec_from_file_location("sam_commit_profile", _SAM_PATH)
sam = importlib.util.module_from_spec(_SAM_SPEC)
sys.modules["sam_commit_profile"] = sam
_SAM_SPEC.loader.exec_module(sam)

SuffixAutomatonDraftIndex = sam.SuffixAutomatonDraftIndex
```

Change parser choices:

```python
p.add_argument(
    "--draft-source",
    type=str,
    default="ngram",
    choices=["ngram", "sam", "dflash-toy", "dflash-toy-ngram-or-repeat"],
)
p.add_argument(
    "--draft-policy",
    type=str,
    default="fixed",
    choices=["fixed", "adaptive", "sam-fixed", "sam-match-aware"],
)
```

- [ ] **Step 4: Extend pure proposal dispatch**

Change the signature:

```python
def propose_draft(
    history: list[int],
    args,
    max_draft_tokens: int | None = None,
    sam_index: SuffixAutomatonDraftIndex | None = None,
) -> DraftProposal:
```

Insert before toy sources:

```python
    if args.draft_source == "sam":
        if sam_index is None:
            raise ValueError("SAM draft source requires sam_index")
        sam_index.assert_history(history)
        if args.draft_policy == "sam-match-aware":
            draft = sam_index.propose_match_aware()
        else:
            draft = sam_index.propose(draft_cap)
        return DraftProposal(
            tokens=list(draft.tokens),
            source="sam",
            metadata=dict(draft.metadata),
        )
```

- [ ] **Step 5: Add explicit SAM argument validation**

Extend `validate_profile_args()`:

```python
    if args.draft_policy in ("sam-fixed", "sam-match-aware"):
        if args.draft_source != "sam":
            raise ValueError("SAM draft policy requires --draft-source sam")
        if args.mode != "candidate-only":
            raise ValueError("SAM draft policy requires --mode candidate-only")
        if args.max_num_seqs != 1:
            raise ValueError("SAM draft policy requires --max-num-seqs 1")
    if args.draft_source == "sam" and args.draft_policy not in (
        "sam-fixed",
        "sam-match-aware",
    ):
        raise ValueError("--draft-source sam requires a SAM draft policy")
```

- [ ] **Step 6: Run dependency-light profiler tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
python3 -m py_compile tools/profile_ngram_commit.py
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit profiler dispatch**

```bash
git add tools/profile_ngram_commit.py tools/test_ngram_speculative.py
git commit -m "Add SAM draft source to profiler"
```

---

### Task 4: Candidate Lifecycle, Bypass Events, and Index Synchronization

**Files:**
- Modify: `tools/profile_ngram_commit.py`
- Modify: `tools/test_ngram_speculative.py`

**Interfaces:**
- Consumes: profiler dispatch from Task 3.
- Produces one `sam_index` per candidate sequence initialized from the actual scheduler token history.
- Produces `sam_events` containing `proposal`, `bypass`, `verify`, and `index_integrity` records.
- Produces process/per-prompt summary fields: `sam_build_ms`, `sam_extension_ms`, `sam_lookup_ms`, `sam_state_count`, `sam_indexed_tokens`, `sam_bypass_count`, `sam_bypass_reasons`, `sam_match_length_counts`, `sam_continuation_region_counts`, `selected_k_counts` for `0/4/8/16`, `runtime_mutation=false`, and `profiler_owned=true`.

- [ ] **Step 1: Extract a dependency-light synchronization helper and write failing tests**

Add tests for a new helper:

```python
sync_sam_index = profile_ngram.sync_sam_index


def test_sync_sam_index_extends_only_verified_history_tail():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    event = sync_sam_index(index, [1, 2, 3, 4, 5])
    assert index.indexed_tokens == [1, 2, 3, 4, 5]
    assert event["extended_tokens"] == [4, 5]
    assert event["runtime_mutation"] is False


def test_sync_sam_index_rejects_history_rewrite():
    index = SuffixAutomatonDraftIndex([1, 2, 3])
    try:
        sync_sam_index(index, [1, 9, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("history rewrite accepted")
```

- [ ] **Step 2: Run and confirm helper is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
```

Expected: failure importing `sync_sam_index`.

- [ ] **Step 3: Implement synchronization without draft-token input**

Add:

```python
def sync_sam_index(
    sam_index: SuffixAutomatonDraftIndex,
    target_verified_history: list[int],
) -> dict:
    normalized = [int(token_id) for token_id in target_verified_history]
    indexed = sam_index.indexed_tokens
    if normalized[:len(indexed)] != indexed:
        raise ValueError("target history rewrote the SAM indexed prefix")
    extension = normalized[len(indexed):]
    t0 = time.perf_counter()
    sam_index.extend_verified(extension)
    extension_ms = (time.perf_counter() - t0) * 1000.0
    sam_index.assert_history(normalized)
    return {
        "event_type": "index_integrity",
        "extended_tokens": extension,
        "extension_ms": extension_ms,
        "index_token_count": len(sam_index.indexed_tokens),
        "index_state_count": len(sam_index.states),
        "history_match": True,
        "runtime_mutation": False,
        "profiler_owned": True,
    }
```

- [ ] **Step 4: Initialize SAM indexes from real candidate histories**

In `run_candidate_only_profile()`, after `candidate_id` is known, read the scheduler sequence and build:

```python
candidate = _find_running_seq(llm, candidate_id)
if candidate is None:
    raise RuntimeError("candidate sequence missing after add_request")
sam_build_t0 = time.perf_counter()
sam_index = (
    SuffixAutomatonDraftIndex(list(candidate.token_ids))
    if args.draft_source == "sam"
    else None
)
sam_build_ms = (time.perf_counter() - sam_build_t0) * 1000.0
```

Store:

```python
"sam_index": sam_index,
"sam_build_ms": sam_build_ms,
"sam_extension_ms": 0.0,
"sam_lookup_ms": 0.0,
"sam_events": [],
"sam_bypass_count": 0,
```

- [ ] **Step 5: Synchronize before each proposal and record lookup/bypass**

Immediately before proposing:

```python
sam_index = stats["sam_index"]
if sam_index is not None:
    integrity_event = sync_sam_index(sam_index, candidate.token_ids)
    stats["sam_extension_ms"] += integrity_event["extension_ms"]
    stats["sam_events"].append({
        "step": step_idx,
        "prompt_index": stats["prompt_index"],
        "candidate_seq_id": candidate_id,
        **integrity_event,
    })
lookup_t0 = time.perf_counter()
draft = propose_draft(
    candidate.token_ids,
    args,
    max_draft_tokens=selected_k,
    sam_index=sam_index,
)
lookup_ms = (time.perf_counter() - lookup_t0) * 1000.0
if sam_index is not None:
    stats["sam_lookup_ms"] += lookup_ms
    draft.metadata["lookup_time_ms"] = lookup_ms
```

For every SAM lookup, first append exactly one proposal event:

```python
proposal_event = {
    "event_type": "proposal",
    "step": step_idx,
    "prompt_index": stats["prompt_index"],
    "candidate_seq_id": candidate_id,
    "draft_source": "sam",
    "selected_k": int(draft.metadata["selected_k"]),
    "proposed_tokens": len(draft.tokens),
    "draft_metadata": draft.metadata,
    "runtime_mutation": False,
    "profiler_owned": True,
}
stats["sam_events"].append(proposal_event)
```

For empty SAM proposals, append a separate bypass event:

```python
bypass_event = {
    "event_type": "bypass",
    "step": step_idx,
    "prompt_index": stats["prompt_index"],
    "candidate_seq_id": candidate_id,
    "draft_source": "sam",
    "selected_k": int(draft.metadata["selected_k"]),
    "proposed_tokens": 0,
    "accepted_count": 0,
    "draft_metadata": draft.metadata,
    "runtime_mutation": False,
    "profiler_owned": True,
}
stats["sam_events"].append(bypass_event)
stats["sam_bypass_count"] += 1
```

Then continue to the existing normal decode path without calling
`verify_and_commit_block()`.

- [ ] **Step 6: Tag verify events and synchronize committed history**

For every non-empty SAM verify event, set:

```python
event["event_type"] = "verify"
event["runtime_mutation"] = False
event["profiler_owned"] = True
event["draft_metadata"] = draft.metadata
stats["sam_events"].append(event_record)
```

At the next loop iteration, `sync_sam_index()` extends from actual
`candidate.token_ids`; do not extend from `draft.tokens` or `accepted_count`
inside the verify helper.

- [ ] **Step 7: Add per-prompt and process SAM summaries**

Populate exact JSON-friendly fields:

```python
"sam_build_ms": stats["sam_build_ms"],
"sam_extension_ms": stats["sam_extension_ms"],
"sam_lookup_ms": stats["sam_lookup_ms"],
"sam_state_count": (
    len(stats["sam_index"].states) if stats["sam_index"] is not None else 0
),
"sam_indexed_tokens": (
    len(stats["sam_index"].indexed_tokens)
    if stats["sam_index"] is not None else 0
),
"sam_bypass_count": stats["sam_bypass_count"],
"sam_events": stats["sam_events"],
"runtime_mutation": False,
"profiler_owned": True,
```

For SAM policies, count selected caps with:

```python
{str(level): sum(
    1 for event in stats["sam_events"]
    if event.get("event_type") == "proposal"
    and int(event.get("selected_k", -1)) == level
) for level in (0, 4, 8, 16)}
```

Add process summary totals and expose top-level:

```python
"sam_events": [
    event for item in per_prompt for event in item.get("sam_events", [])
],
```

- [ ] **Step 8: Add source-scan and bypass regression assertions**

Add tests that:

```python
source = open(os.path.join(_REPO_ROOT, "tools", "profile_ngram_commit.py")).read()
assert "verify_and_commit_block(" in source
assert '"runtime_mutation": False' in source
assert "LLMEngine.step" not in source
```

Add a pure helper test ensuring an empty SAM proposal returns a bypass record
and no verifier callback is invoked by extracting:

```python
def should_verify_draft(draft: DraftProposal) -> bool:
    return bool(draft.tokens)
```

and testing `should_verify_draft(empty_draft) is False`.

- [ ] **Step 9: Run profiler regression suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
python3 -m py_compile tools/profile_ngram_commit.py
```

Expected: all commands exit 0.

- [ ] **Step 10: Commit candidate lifecycle**

```bash
git add tools/profile_ngram_commit.py tools/test_ngram_speculative.py
git commit -m "Track SAM lifecycle in speculative profiler"
```

---

### Task 5: Verify/Commit and Block-Boundary Regression

**Files:**
- Modify: `tools/test_ngram_speculative.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes unchanged `verify_and_commit_block()` and SAM event contract.
- Proves SAM draft source changes no target verification, accepted-prefix, EOS, output-budget, or block/hash lifecycle behavior.

- [ ] **Step 1: Add failing verify-event contract test**

Use the existing profiler helper test pattern to construct a SAM
`DraftProposal`, pass its tokens/source/metadata into the event attachment path,
and assert:

```python
assert event["draft_source"] == "sam"
assert event["draft_metadata"]["match_length"] >= 2
assert event["runtime_mutation"] is False
assert event["profiler_owned"] is True
assert event["wasted_draft_tokens"] == (
    event["proposed_tokens"] - event["accepted_count"]
)
```

- [ ] **Step 2: Add a SAM-originated accepted-token block regression**

In `tools/test_chunked_prefill.py`, reuse the existing speculative
block-boundary fixture but obtain draft tokens from:

```python
index = SuffixAutomatonDraftIndex(
    prompt_tokens + repeated_verified_prefix
)
draft = index.propose(max_draft_tokens=16)
assert len(draft.tokens) > block_size - current_block_offset
```

Feed `draft.tokens` through the same commit helper/fake target path used by the
existing accepted-token boundary test. Assert:

```python
assert committed_tokens == expected_target_prefix
assert block_manager_hashes_match_sequence_tokens()
assert unused_reserved_blocks_are_released()
assert accepted_tokens_crossed_block_boundary
```

- [ ] **Step 3: Run the tests and observe the missing SAM event fields/import**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: at least one new assertion/import fails before the test plumbing is completed.

- [ ] **Step 4: Make only test-facing event plumbing changes**

If necessary, update `attach_draft_policy_event()` so SAM events retain:

```python
event["runtime_mutation"] = False
event["profiler_owned"] = True
```

Do not edit `verify_and_commit_block()` acceptance or KV logic.

- [ ] **Step 5: Run both complete regression suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected final lines:

```text
ngram speculative tests passed
chunked prefill tests passed
```

- [ ] **Step 6: Commit correctness coverage**

```bash
git add tools/test_ngram_speculative.py tools/test_chunked_prefill.py tools/profile_ngram_commit.py
git commit -m "Test SAM speculative commit boundaries"
```

---

### Task 6: Canonical Manifest, Prompt Bank, and Process Matrix

**Files:**
- Create: `tools/sam_drafter_gate.py`
- Create: `tools/test_sam_drafter_gate.py`

**Interfaces:**
- Produces constants `POLICIES`, `THRESHOLDS`, `PROMPT_BANK`, `REQUIRED_UPLOAD_PATHS`, and `MAX_PORT_COLLISION_RETRIES`.
- Produces `build_run_specs(repetitions: int, base_seed: int) -> list[dict]`.
- Produces `build_manifest(repetitions: int, base_seed: int, source_commit: str, source_dirty: bool, model_path: str, model_identifier: str, host: str, python_bin: str, extra_environment: dict | None = None) -> dict`.
- Produces `_profiler_command(spec: dict, prompt: dict, python_bin: str, model_path: str, process_json: Path) -> list[str]`.
- Produces `_normalize_row(manifest: dict, spec: dict, profiler_result: dict | None, process: dict) -> tuple[dict, list[dict]]`.
- Produces `run_gate(out_dir: Path, python_bin: str, model_path: str, repetitions: int, base_seed: int, source_commit: str, source_dirty: bool, host: str, resume: bool, extra_environment: dict | None = None) -> dict`.
- `out_dir` contains only the five canonical evidence files. Process JSON and
  stdout/stderr live in a sibling `<out-dir-name>.runs/` directory.

- [ ] **Step 1: Write failing prompt/policy/matrix tests**

Create `tools/test_sam_drafter_gate.py` using the import style from
`tools/test_adaptive_ngram_gate.py`. Add:

```python
def test_prompt_bank_has_five_stable_classes():
    assert [item["name"] for item in gate.PROMPT_BANK] == [
        "natural_prose",
        "structured_code_like",
        "repeated_long_context",
        "transition_heavy",
        "prompt_copy_retrieval",
    ]
    assert {item["workload_class"] for item in gate.PROMPT_BANK} == {
        "natural",
        "structured",
        "high_repeat",
        "transition_heavy",
        "prompt_copy",
    }
    for prompt in gate.PROMPT_BANK:
        assert prompt["prompt_sha256"] == gate.sha256_text(prompt["prompt"])


def test_run_specs_are_175_unique_rows_for_canonical():
    specs = gate.build_run_specs(repetitions=7, base_seed=20260715)
    assert len(specs) == 175
    assert len({item["run_key"] for item in specs}) == 175
    assert {item["policy"] for item in specs} == {
        "baseline",
        "ngram_fixed_k4",
        "ngram_adaptive",
        "sam_fixed_k16",
        "sam_match_aware",
    }
    assert all(item["max_num_seqs"] == 1 for item in specs)


def test_required_upload_paths_cover_all_runtime_imports():
    assert gate.REQUIRED_UPLOAD_PATHS == (
        "tinyvllm",
        "tools/draft_model_schema.py",
        "tools/profile_ngram_commit.py",
        "tools/sam_drafter_gate.py",
    )
```

- [ ] **Step 2: Run and confirm the gate module is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
```

Expected: failure loading `tools/sam_drafter_gate.py`.

- [ ] **Step 3: Define exact policies and thresholds**

Create:

```python
POLICIES = {
    "baseline": {
        "mode": "baseline-only",
        "draft_source": "ngram",
        "draft_policy": "fixed",
        "max_draft_tokens": None,
    },
    "ngram_fixed_k4": {
        "mode": "candidate-only",
        "draft_source": "ngram",
        "draft_policy": "fixed",
        "max_draft_tokens": 4,
    },
    "ngram_adaptive": {
        "mode": "candidate-only",
        "draft_source": "ngram",
        "draft_policy": "adaptive",
        "max_draft_tokens": 4,
    },
    "sam_fixed_k16": {
        "mode": "candidate-only",
        "draft_source": "sam",
        "draft_policy": "sam-fixed",
        "max_draft_tokens": 16,
    },
    "sam_match_aware": {
        "mode": "candidate-only",
        "draft_source": "sam",
        "draft_policy": "sam-match-aware",
        "max_draft_tokens": 16,
    },
}

THRESHOLDS = {
    "sam_vs_baseline_min": 0.10,
    "sam_vs_ngram_k4_min": 0.03,
    "sam_near_ngram_k4_min": -0.01,
    "verify_attempt_reduction_min": 0.25,
    "draft_waste_reduction_min": 0.25,
    "critical_prompt_speedup_min": -0.05,
}
```

- [ ] **Step 4: Commit five concrete prompts**

Define literal prompts and fixed output lengths:

```python
PROMPT_BANK_BASE = (
    {
        "name": "natural_prose",
        "workload_class": "natural",
        "prompt": (
            "Explain why benchmark correctness must be established before "
            "performance tuning. Use two concrete engineering examples and "
            "finish with a concise recommendation."
        ),
        "max_output_len": 96,
    },
    {
        "name": "structured_code_like",
        "workload_class": "structured",
        "prompt": (
            "Continue this exact checklist format with twelve new lines:\n"
            "- validate input\n- run baseline\n- compare output\n- record timing\n"
        ),
        "max_output_len": 112,
    },
    {
        "name": "repeated_long_context",
        "workload_class": "high_repeat",
        "prompt": (
            "alpha beta gamma delta epsilon " * 128
            + "\nContinue the token pattern exactly:"
        ),
        "max_output_len": 128,
    },
    {
        "name": "transition_heavy",
        "workload_class": "transition_heavy",
        "prompt": (
            "A A A A B B B B C C C C. Now explain in natural language why "
            "a repeated pattern can stop abruptly, then emit the original "
            "A/B/C sequence one more time."
        ),
        "max_output_len": 112,
    },
    {
        "name": "prompt_copy_retrieval",
        "workload_class": "prompt_copy",
        "prompt": (
            "Reference block:\n"
            "BEGIN ALPHA\nid: 17\nstatus: verified\nowner: inference\nEND ALPHA\n"
            "Reference block:\n"
            "BEGIN BETA\nid: 29\nstatus: pending\nowner: runtime\nEND BETA\n"
            "Copy the complete ALPHA block exactly, then explain its status."
        ),
        "max_output_len": 112,
    },
)
```

Hash every prompt with `sha256_text`.

- [ ] **Step 5: Implement deterministic run specs and manifest**

Copy the atomic JSON/text writers, dynamic distinct-port allocation, narrow
port-collision classifier, and model identifier helper from
`tools/adaptive_ngram_gate.py`. Adapt run keys:

```python
def _run_key(repetition: int, prompt_name: str, policy: str) -> str:
    return f"r{repetition:02d}__{prompt_name}__{policy}"
```

The manifest claim scope must be:

```python
"claim_scope": {
    "single_sequence": True,
    "greedy_only": True,
    "profiler_owned": True,
    "ragged_batched_verify": False,
    "production_batch_throughput": False,
    "queue_tail_latency": False,
    "memory_reduction": False,
}
```

- [ ] **Step 6: Implement policy-specific profiler commands**

Every command includes:

```text
--prompt <literal>
--max-output-len <fixed>
--ignore-eos
--warmup-output-len min(8, max_output_len)
--temperature 0.0
--max-commit-events 0
--max-num-seqs 1
--max-model-len 4096
--gpu-memory-utilization 0.7
```

For candidates append exact `--draft-source`, `--draft-policy`,
`--max-draft-tokens`, and `--allow-zero-accept`. For n-gram candidates also
append `--ngram-size 5`.

- [ ] **Step 7: Normalize all profiler evidence**

Each row must include:

```python
"prompt_tokens"
"output_tokens"
"output_token_ids"
"output_token_sha256"
"elapsed_s"
"output_tokens_per_s"
"proposal_events"
"verify_attempts"
"no_draft_positions"
"drafted_tokens"
"accepted_tokens"
"wasted_draft_tokens"
"zero_accept_events"
"zero_accept_verify_ms"
"selected_k_counts"
"sam_build_ms"
"sam_extension_ms"
"sam_lookup_ms"
"sam_state_count"
"sam_indexed_tokens"
"sam_bypass_count"
"runtime_mutation"
"profiler_owned"
"process"
```

Normalize both `verify_events` and `sam_events` into `event_rows.json` with
`run_key`, `policy`, `prompt_name`, `prompt_class`, `repetition`, and
monotonic `event_index`. Avoid duplicating a SAM verify event that appears in
both profiler lists by assigning a stable event key:

```text
<run_key>:<step>:<event_type>:<candidate_seq_id>
```

and retaining one row per key.

- [ ] **Step 8: Implement atomic run/resume**

Reuse the adaptive driver structure but validate every resumable row against:

```text
source_commit
source_dirty
model_identifier
prompt_sha256
policy
repetition
returncode == 0
profiler_gate_pass == true
```

Invalid resumable rows are removed with their events and rerun. Write
`raw_rows.json` and `event_rows.json` atomically after each process.

Create transient paths outside the artifact root:

```python
run_data_dir = out_dir.parent / f"{out_dir.name}.runs"
logs_dir = run_data_dir / "logs"
process_dir = run_data_dir / "process_json"
```

Store paths relative to `run_data_dir.parent`; never create `logs/` or
`process_json/` beneath `out_dir`.

- [ ] **Step 9: Run manifest/matrix tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
python3 -m py_compile tools/sam_drafter_gate.py
```

Expected: prompt, policy, matrix, command, and upload-path tests pass.

- [ ] **Step 10: Commit gate orchestration**

```bash
git add tools/sam_drafter_gate.py tools/test_sam_drafter_gate.py
git commit -m "Add SAM drafter gate matrix"
```

---

### Task 7: Trace Reconciliation and Strict Decision Classifier

**Files:**
- Modify: `tools/sam_drafter_gate.py`
- Modify: `tools/test_sam_drafter_gate.py`

**Interfaces:**
- Produces `reconcile_run_trace(row: dict, events: list[dict]) -> dict`.
- Produces `summarize_rows(manifest: dict, raw_rows: list[dict], event_rows: list[dict]) -> dict`.
- Produces paired speedups/reductions exactly as preregistered.
- Produces strict `GO`, `NO_GO`, or `INCOMPLETE`.

- [ ] **Step 1: Build a synthetic complete 175-row fixture**

Add `_synthetic_complete_gate_rows()` that:

- uses `build_run_specs(7, 20260715)`;
- emits 100 output tokens with identical IDs for every paired policy;
- sets baseline `100 tok/s`, n-gram K4 `108 tok/s`, adaptive n-gram
  `106 tok/s`, fixed SAM `111 tok/s`, match-aware SAM `112 tok/s`;
- sets n-gram K4 verify attempts/waste to `20/40`;
- sets match-aware SAM verify attempts/waste to `14/28`;
- emits SAM events across the fixture covering `K=0/4/8/16`,
  prompt/generated continuations, zero accept, and full multi-token accept;
- includes one integrity event per SAM run;
- reconciles every process total.

- [ ] **Step 2: Add failing classification tests**

Add:

```python
def test_complete_175_row_fixture_is_go():
    manifest, rows, events = _synthetic_complete_gate_rows()
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "GO"
    assert summary["observed_rows"] == 175
    assert summary["correctness_pass"] is True
    assert summary["trace_reconciliation_pass"] is True
    assert summary["policy_exercise_pass"] is True


def test_missing_or_failed_evidence_is_incomplete_not_no_go():
    manifest, rows, events = _synthetic_complete_gate_rows()
    assert gate.summarize_rows(manifest, rows[:-1], events)["decision"] == "INCOMPLETE"
    rows[-1]["process"]["returncode"] = 1
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_output_mismatch_is_incomplete():
    manifest, rows, events = _synthetic_complete_gate_rows()
    candidate = next(row for row in rows if row["policy"] == "sam_match_aware")
    candidate["output_token_ids"] = [999]
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_performance_failure_is_no_go_only_after_evidence_passes():
    manifest, rows, events = _synthetic_complete_gate_rows()
    for row in rows:
        if row["policy"] == "sam_match_aware":
            row["output_tokens_per_s"] = 105.0
            row["elapsed_s"] = row["output_tokens"] / 105.0
    summary = gate.summarize_rows(manifest, rows, events)
    assert summary["decision"] == "NO_GO"
    assert summary["correctness_pass"] is True
```

- [ ] **Step 3: Add paired-statistic tests**

Construct asymmetric rows proving the implementation uses per-pair ratios:

```python
def test_speedup_is_median_of_paired_ratios():
    manifest, rows, events = _synthetic_complete_gate_rows()
    pairs = [
        row for row in rows
        if row["prompt_name"] == "natural_prose"
        and row["repetition"] in (0, 1)
        and row["policy"] in ("baseline", "sam_match_aware")
    ]
    # Set ratios to 1.20 and 1.00 while making independent medians misleading.
    # The literal assignments below prove the stored values derive per pair.
    for row in pairs:
        if row["repetition"] == 0 and row["policy"] == "baseline":
            row["output_tokens_per_s"] = 10.0
        elif row["repetition"] == 0:
            row["output_tokens_per_s"] = 12.0
        elif row["policy"] == "baseline":
            row["output_tokens_per_s"] = 100.0
        else:
            row["output_tokens_per_s"] = 100.0
        row["elapsed_s"] = row["output_tokens"] / row["output_tokens_per_s"]
    summary = gate.summarize_rows(manifest, rows, events)
    natural_pairs = summary["paired_speedups"]["sam_vs_baseline"]["natural_prose"]
    assert any(abs(value - 0.20) < 1e-12 for value in natural_pairs)
    assert 0.0 in natural_pairs
```

- [ ] **Step 4: Add reduction-reference and policy-exercise tests**

Test all required `INCOMPLETE` branches:

```python
def test_zero_positive_reference_for_required_reduction_is_incomplete():
    manifest, rows, events = _synthetic_complete_gate_rows()
    for row in rows:
        if row["policy"] == "ngram_fixed_k4":
            row["verify_attempts"] = 0
    assert gate.summarize_rows(manifest, rows, events)["decision"] == "INCOMPLETE"


def test_missing_each_required_policy_branch_is_incomplete():
    for field, value in (
        ("selected_k", 0),
        ("selected_k", 4),
        ("selected_k", 8),
        ("selected_k", 16),
        ("continuation_region", "prompt"),
        ("continuation_region", "generated"),
        ("accepted_count", 0),
    ):
        manifest, rows, events = _synthetic_complete_gate_rows()
        filtered = [event for event in events if event.get(field) != value]
        assert gate.summarize_rows(manifest, rows, filtered)["decision"] == "INCOMPLETE"
```

Add a separate assertion for a fully accepted multi-token proposal.

- [ ] **Step 5: Implement structural validation**

Reject as `INCOMPLETE`:

- row count/key mismatch;
- process failures or exhausted port collisions;
- invalid/missing prompt tokens;
- non-finite/non-positive elapsed or throughput;
- invalid negative counts;
- reused ports;
- source/model/prompt/policy mismatch;
- missing paired baseline/K4 row;
- profiler gate failure;
- `runtime_mutation != false` or `profiler_owned != true` for SAM rows/events.

- [ ] **Step 6: Implement trace reconciliation**

For each run:

```python
proposal_events = [e for e in events if e["event_type"] == "proposal"]
verify_events = [e for e in events if e["event_type"] == "verify"]
bypass_events = [e for e in events if e["event_type"] == "bypass"]
integrity_events = [e for e in events if e["event_type"] == "index_integrity"]
```

Reconcile:

```text
verify_attempts == len(verify_events)
drafted_tokens == sum(proposed_tokens for proposal events)
accepted_tokens == sum(accepted_count)
wasted_draft_tokens == drafted_tokens - accepted_tokens
zero_accept_events == count(accepted_count == 0)
sam_bypass_count == len(bypass_events)
selected_k_counts == selected-K counts from proposal events only
each non-empty proposal has one verify event at the same step/sequence
each empty proposal has one bypass event at the same step/sequence
last integrity index_token_count == prompt_tokens + output_tokens
all integrity history_match == true
```

Return field-specific failures; any failure is `INCOMPLETE`.

- [ ] **Step 7: Implement paired metrics**

For each of 35 `(repetition, prompt_name)` pairs:

```python
speedup = candidate_tps / reference_tps - 1.0
```

Store raw paired lists by prompt and overall. For verify attempts and waste:

```python
reduction = 1.0 - candidate_value / reference_value
```

Only include pairs with positive reference values. If none exist for a required
metric, append a structural failure.

- [ ] **Step 8: Implement exact classification order**

Classification order:

1. structural, pair, correctness, trace, runtime-mutation, artifact-input, or
   exercise failure -> `INCOMPLETE`;
2. otherwise performance/regression threshold failure -> `NO_GO`;
3. otherwise -> `GO`.

The direct win and efficient near-tie predicates are:

```python
direct_win = sam_vs_k4 >= 0.03
efficient_near_tie = (
    sam_vs_k4 >= -0.01
    and verify_attempt_reduction >= 0.25
    and draft_waste_reduction >= 0.25
)
```

- [ ] **Step 9: Run classifier tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
```

Expected final line: `sam drafter gate tests passed`.

- [ ] **Step 10: Commit strict decision logic**

```bash
git add tools/sam_drafter_gate.py tools/test_sam_drafter_gate.py
git commit -m "Verify SAM gate evidence and thresholds"
```

---

### Task 8: Artifact Regeneration, Five-File Hashes, and Resume Tests

**Files:**
- Modify: `tools/sam_drafter_gate.py`
- Modify: `tools/test_sam_drafter_gate.py`

**Interfaces:**
- Produces `render_report(manifest: dict, summary: dict) -> str`.
- Produces `verify_artifacts(out_dir: Path) -> dict`.
- Produces a canonical artifact SHA-256 manifest embedded in `summary.json` without adding a sixth canonical file.
- Produces strict resume-row validation.

- [ ] **Step 1: Add failing independent-regeneration tests**

Add a temporary-directory test that writes the five files, verifies them, then
mutates each derived file:

```python
def test_verify_artifacts_recomputes_summary_report_and_hashes(tmp_path):
    manifest, rows, events = _synthetic_complete_gate_rows()
    summary = gate.summarize_rows(manifest, rows, events)
    gate._write_canonical_artifacts(tmp_path, manifest, rows, events, summary)
    assert gate.verify_artifacts(tmp_path)["decision"] == "GO"
    (tmp_path / "report.md").write_text("tampered\n")
    try:
        gate.verify_artifacts(tmp_path)
    except ValueError as exc:
        assert "report.md" in str(exc)
    else:
        raise AssertionError("tampered report accepted")
```

- [ ] **Step 2: Add resume compatibility tests**

Test that `_row_is_resumable(manifest, spec, row)` returns `False` for every
one-field mismatch in source commit, dirty bit, model identifier, prompt hash,
policy, repetition, return code, profiler gate, or non-finite timing.

- [ ] **Step 3: Implement deterministic report rendering**

Report sections:

```text
Decision and reasons
Environment/source/model
Rows and completeness audits
Median throughput by policy
Paired speedups
Verify/waste reductions
Critical prompt regressions
Policy exercise
SAM CPU overhead
Fixed thresholds
Claim boundaries
Next direction by GO/NO_GO/INCOMPLETE
```

Render values from `summary.json`; do not recompute hidden values inside the
Markdown renderer.

- [ ] **Step 4: Embed hashes without recursive self-hashing**

Use this rule:

- `summary.json` stores SHA-256 for `manifest.json`, `raw_rows.json`, and
  `event_rows.json`;
- local verification independently regenerates and compares `summary.json` and
  `report.md`;
- the remote wrapper separately compares all five downloaded files byte for
  byte using transient shell output, not a sixth artifact.

Implement:

```python
summary["input_artifact_sha256"] = {
    "manifest.json": sha256_bytes(manifest_bytes),
    "raw_rows.json": sha256_bytes(raw_rows_bytes),
    "event_rows.json": sha256_bytes(event_rows_bytes),
}
```

- [ ] **Step 5: Implement verifier**

`verify_artifacts()` must:

1. require exactly the five canonical files at the artifact root;
2. verify the three stored input hashes;
3. recompute summary excluding only the stored hash field, then restore the
   independently calculated hash field;
4. require exact JSON object equality;
5. regenerate and require exact report text equality;
6. return the verified summary.

- [ ] **Step 6: Run artifact and resume tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
python3 -m py_compile tools/sam_drafter_gate.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit artifact verifier**

```bash
git add tools/sam_drafter_gate.py tools/test_sam_drafter_gate.py
git commit -m "Harden SAM gate artifacts and resume"
```

---

### Task 9: Isolated Remote Runner

**Files:**
- Create: `tools/run_sam_drafter_gate_remote.sh`
- Modify: `tools/test_sam_drafter_gate.py`

**Interfaces:**
- Produces modes `preflight`, `smoke`, and `canonical`.
- Uses exact defaults:
  - `REMOTE_HOST=sitian@10.232.195.203`
  - `REMOTE_PYTHON=/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
  - `MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Uploads only `REQUIRED_UPLOAD_PATHS`.
- Downloads and verifies exactly five canonical artifacts.

- [ ] **Step 1: Add runner contract tests**

Read the shell source and assert exact strings:

```python
def test_remote_runner_uses_exact_host_python_model_and_isolation():
    source = (
        gate.Path(_REPO_ROOT)
        / "tools"
        / "run_sam_drafter_gate_remote.sh"
    ).read_text()
    assert "sitian@10.232.195.203" in source
    assert "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python" in source
    assert "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B" in source
    assert "TINYVLLM_DIST_PORT" not in source  # allocated by Python per process
    assert "MASTER_PORT" not in source
    assert "sam-drafter-gates" in source
```

- [ ] **Step 2: Run and confirm missing runner**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
```

Expected: missing shell file failure.

- [ ] **Step 3: Create runner from the proven adaptive pattern**

Start with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/sam-drafter-gates}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-7}"
MODE="${1:-smoke}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)-$$}"
REMOTE_DIR="${REMOTE_BASE}/${RUN_TAG}"
LOCAL_OUT="${LOCAL_OUT:-${REPO_ROOT}/experiments/sam_drafter/${RUN_TAG}}"
BASE_SEED="${BASE_SEED:-20260715}"
```

Validate exact model directory and `config.json`; do not discover alternative
spellings.

- [ ] **Step 4: Upload isolated committed source**

Before upload:

```bash
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY=0
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  SOURCE_DIRTY=1
fi
```

Create a unique remote directory and tar:

```text
tinyvllm
tools/draft_model_schema.py
tools/profile_ngram_commit.py
tools/sam_drafter_gate.py
```

Run remote `py_compile` and both `--help` commands.

- [ ] **Step 5: Run gate and download only canonical evidence**

Modes:

```text
preflight -> compile/help only
smoke -> repetitions=1
canonical -> repetitions=7
```

After execution, SCP exactly:

```text
manifest.json
raw_rows.json
event_rows.json
summary.json
report.md
```

Leave remote stdout/stderr and process JSON in the sibling
`<artifact-dir-name>.runs/` directory. They are diagnostic data, not canonical
evidence, and are not downloaded into the local artifact root.

Capture remote `sha256sum` output transiently, compute local hashes, and fail
on any mismatch. Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/sam_drafter_gate.py verify \
  --out-dir "${LOCAL_OUT}"
```

- [ ] **Step 6: Validate shell and runner tests**

Run:

```bash
bash -n tools/run_sam_drafter_gate_remote.sh
chmod +x tools/run_sam_drafter_gate_remote.sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit remote runner**

```bash
git add tools/run_sam_drafter_gate_remote.sh tools/test_sam_drafter_gate.py
git commit -m "Add isolated SAM drafter remote gate"
```

---

### Task 10: Local Completion Matrix Before GPU Work

**Files:**
- Modify only if a failure reveals a SAM-specific defect in:
  - `tinyvllm/speculative/sam.py`
  - `tools/profile_ngram_commit.py`
  - `tools/sam_drafter_gate.py`
  - their focused tests

**Interfaces:**
- Produces a clean source commit suitable for isolated remote upload.

- [ ] **Step 1: Run all focused dependency-light tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
```

Expected: four passing final lines and exit 0.

- [ ] **Step 2: Run syntax and shell validation**

```bash
python3 -m py_compile \
  tinyvllm/speculative/sam.py \
  tools/profile_ngram_commit.py \
  tools/sam_drafter_gate.py \
  tools/test_sam_speculative.py \
  tools/test_sam_drafter_gate.py
bash -n tools/run_sam_drafter_gate_remote.sh
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Verify write-scope and runtime boundaries**

Run:

```bash
git diff 48ef363 --name-only
rg -n "runtime_mutation|profiler_owned|sam-match-aware|sam-fixed" \
  tinyvllm/speculative/sam.py \
  tools/profile_ngram_commit.py \
  tools/sam_drafter_gate.py
git diff 48ef363 -- \
  tinyvllm/engine \
  tinyvllm/scheduler.py \
  tinyvllm/sequence.py
```

Expected: no production runtime diff in the final command.

- [ ] **Step 4: Commit any validation-only fix**

If fixes were required:

```bash
git add tinyvllm/speculative/sam.py tools
git commit -m "Fix SAM gate validation gaps"
```

If no fixes were required, do not create an empty commit.

---

### Task 11: Remote Preflight and Branch-Coverage Smoke

**Files:**
- Remote outputs only until the smoke is valid.
- Modify source/tests only for demonstrated plumbing or evidence defects; never tune canonical thresholds.

**Interfaces:**
- Produces remote preflight evidence.
- Produces a one-repetition smoke with 25 process rows.
- Must cover `K=0`, non-empty SAM verify, prompt/generated continuation, zero accept, full multi-token accept, block boundary, exact output, and `runtime_mutation=false`.

- [ ] **Step 1: Confirm clean source and remote connectivity**

```bash
git status --short --branch
ssh -S /tmp/ssh-sitian-10.232.195.203 -o BatchMode=yes \
  sitian@10.232.195.203 true
```

Expected: clean branch and SSH exit 0. If the control socket is absent, use the
current Kerberos/API-cache route established for this host; do not switch user.

- [ ] **Step 2: Run exact remote preflight**

```bash
tools/run_sam_drafter_gate_remote.sh preflight
```

Expected: model config validation, remote compilation, and CLI help pass in a
unique isolated directory.

- [ ] **Step 3: Run one-repetition smoke**

```bash
RUN_TAG="qwen3-06b-sam-smoke-$(date +%Y%m%d-%H%M%S)" \
tools/run_sam_drafter_gate_remote.sh smoke
```

Expected: 25 unique process rows and five locally verified artifacts. The
decision may be `GO` or `NO_GO`; it must not be `INCOMPLETE`.

- [ ] **Step 4: Audit required smoke branches**

Run:

```bash
python3 - <<'PY' "$LOCAL_OUT/event_rows.json" "$LOCAL_OUT/raw_rows.json"
import json
import sys

events = json.load(open(sys.argv[1]))
rows = json.load(open(sys.argv[2]))
sam_events = [e for e in events if e.get("policy") == "sam_match_aware"]
assert len(rows) == 25
assert {e.get("selected_k") for e in sam_events} >= {0, 4, 8, 16}
assert {e.get("draft_metadata", {}).get("continuation_region") for e in sam_events} >= {
    "prompt", "generated"
}
assert any(e.get("accepted_count") == 0 for e in sam_events if e.get("event_type") == "verify")
assert any(
    e.get("event_type") == "verify"
    and e.get("accepted_count", 0) == e.get("proposed_tokens", -1)
    and e.get("accepted_count", 0) > 1
    for e in sam_events
)
assert all(e.get("runtime_mutation") is False for e in sam_events)
assert all(row.get("profiler_gate_pass") is True for row in rows)
print("SAM_SMOKE_BRANCH_AUDIT_OK")
PY
```

Use the actual smoke artifact path in place of `$LOCAL_OUT`.

- [ ] **Step 5: Run a dedicated block-boundary remote smoke if the matrix did not cross**

Invoke `tools/profile_ngram_commit.py` in the isolated remote source with a
repeated prompt, `--draft-source sam`, `--draft-policy sam-fixed`,
`--max-draft-tokens 16`, and an output budget that forces an accepted
speculative append across a KV block boundary. Assert exact token equality and
record the command/output in `AGENT_HANDOFF_STATE.md` later.

- [ ] **Step 6: Fix only evidence or plumbing defects and repeat**

If the smoke is `INCOMPLETE` or lacks required branches:

1. preserve failed artifacts;
2. identify whether the defect is prompt coverage, event schema, verifier,
   resume logic, or profiler lifecycle;
3. add a failing local test;
4. implement the minimal correction;
5. rerun Tasks 10 and 11.

Do not change performance thresholds.

---

### Task 12: Freeze Canonical Source and Run 175 Rows

**Files:**
- Create after completion:
  `experiments/sam_drafter/$CANONICAL_TAG/manifest.json`
- Create after completion:
  `experiments/sam_drafter/$CANONICAL_TAG/raw_rows.json`
- Create after completion:
  `experiments/sam_drafter/$CANONICAL_TAG/event_rows.json`
- Create after completion:
  `experiments/sam_drafter/$CANONICAL_TAG/summary.json`
- Create after completion:
  `experiments/sam_drafter/$CANONICAL_TAG/report.md`

**Interfaces:**
- Produces manifest-bound, resume-safe canonical evidence from one clean source commit.

- [ ] **Step 1: Commit all smoke-proven source before measuring**

```bash
git status --short
git add tinyvllm/speculative/sam.py tools
git commit -m "Prepare canonical SAM drafter gate"
git rev-parse HEAD
```

Skip the commit if already clean. Record the exact source SHA.

- [ ] **Step 2: Start canonical run**

```bash
CANONICAL_TAG="qwen3-06b-sam-canonical-$(date +%Y%m%d-%H%M%S)"
RUN_TAG="${CANONICAL_TAG}" \
tools/run_sam_drafter_gate_remote.sh canonical
```

Expected: 175 isolated process rows. Keep the same `RUN_TAG` for resume.

- [ ] **Step 3: Resume transient interruptions without changing manifest**

```bash
RESUME=1 RUN_TAG="${CANONICAL_TAG}" \
tools/run_sam_drafter_gate_remote.sh canonical
```

Expected: only invalid/missing runs execute. Source, prompt hashes, model,
policies, seed, repetitions, and thresholds must match the original manifest.

- [ ] **Step 4: Verify downloaded artifacts independently**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/sam_drafter_gate.py verify \
  --out-dir "experiments/sam_drafter/${CANONICAL_TAG}"
```

Expected: verified decision is one of `GO` or `NO_GO`, never `INCOMPLETE`.

- [ ] **Step 5: Run canonical prompt-to-artifact audit**

Create a temporary audit script or one-shot Python command that asserts:

```text
175 unique rows
35 rows per policy
7 rows per prompt/policy
all return codes zero
all ports globally unique
all paired outputs identical
all timings finite and positive
all SAM traces reconcile
all index histories reconcile
runtime_mutation=false
profiler_owned=true
K=0/4/8/16 exercised
prompt/generated continuation exercised
zero-accept and full multi-token accept exercised
summary/report independently regenerate
five local files match remote SHA-256 values
```

Expected final line: `SAM_CANONICAL_AUDIT_OK`.

- [ ] **Step 6: Commit only the canonical five-file evidence**

```bash
git add "experiments/sam_drafter/${CANONICAL_TAG}/"
git commit -m "Record canonical SAM drafter gate"
```

Verify:

```bash
git show --format= --name-only HEAD
```

Expected: exactly the five canonical evidence files plus no transient logs.

---

### Task 13: README, Handoff, and Final Completion Audit

**Files:**
- Modify: `README.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes canonical manifest, summary, report, source SHA, remote path, and audit output.
- Produces durable commands, result interpretation, limits, and next direction.

- [ ] **Step 1: Add README command and result section**

Document:

```bash
tools/run_sam_drafter_gate_remote.sh preflight
tools/run_sam_drafter_gate_remote.sh smoke
tools/run_sam_drafter_gate_remote.sh canonical
RESUME=1 RUN_TAG="${CANONICAL_TAG}" tools/run_sam_drafter_gate_remote.sh canonical
ARTIFACT_DIR="experiments/sam_drafter/${CANONICAL_TAG}"
python3 tools/sam_drafter_gate.py verify --out-dir "${ARTIFACT_DIR}"
```

Include measured policy tok/s, paired SAM speedups, verify/waste reductions,
critical prompt-class regressions, policy exercise, CPU overhead, final
decision, and exact claim boundaries.

- [ ] **Step 2: Update handoff with reproducible operational state**

Record:

- branch and source/canonical commits;
- exact local and remote canonical directories;
- host, Python, model, GPU, seed, repetitions, and row count;
- smoke and canonical commands;
- resume behavior;
- all five SHA-256 values;
- local test commands and canonical audit command;
- what the decision proves and does not prove;
- next action:
  - `GO`: native/index optimization plus a separately designed ragged-batch gate;
  - `NO_GO`: retain infrastructure and wait for a compatible learned drafter/checkpoint;
  - `INCOMPLETE`: repair evidence only and resume unchanged manifest.

- [ ] **Step 3: Run fresh final verification**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_ngram_speculative.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_sam_drafter_gate.py
python3 -m py_compile \
  tinyvllm/speculative/sam.py \
  tools/profile_ngram_commit.py \
  tools/sam_drafter_gate.py
bash -n tools/run_sam_drafter_gate_remote.sh
python3 tools/sam_drafter_gate.py verify \
  --out-dir "experiments/sam_drafter/${CANONICAL_TAG}"
git diff --check
```

Expected: all commands exit 0 and verifier returns the recorded canonical
decision.

- [ ] **Step 4: Build the final prompt-to-artifact checklist**

Map every explicit design requirement to evidence:

```text
CPU-only pure SAM -> sam.py plus unit tests
verified-token-only extension -> sync tests plus integrity events
longest usable suffix -> lookup tests and event metadata
K=0/4/8/16 -> policy tests and canonical exercise
unchanged verify/commit -> source diff plus block-boundary tests
five policies / five prompts / seven reps -> manifest and 175 rows
paired ratios -> synthetic statistic test and summary lists
strict thresholds -> manifest and classifier tests
INCOMPLETE separation -> synthetic failure tests
remote host/python/model/isolation -> runner and manifest
dynamic unique ports -> process rows and audit
five artifacts / hash equality -> verifier and remote SHA output
claim boundaries -> report, README, handoff
```

Treat any missing mapping as unfinished work.

- [ ] **Step 5: Commit documentation and any canonical audit fixes**

```bash
git add README.md AGENT_HANDOFF_STATE.md
git commit -m "Document SAM drafter gate result"
```

- [ ] **Step 6: Verify clean final state**

```bash
git status --short --branch
git log -5 --oneline
```

Expected: clean worktree and visible design, implementation, canonical
evidence, and documentation commits.

