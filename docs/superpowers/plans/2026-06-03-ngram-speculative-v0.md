# N-gram Speculative Decoding v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe v0 for n-gram speculative decoding research: a deterministic n-gram drafter plus an offline acceptance profiler that tells us whether full engine integration is worth doing.

**Architecture:** Start outside the hot decode loop. Add a small pure-Python n-gram draft module, test it thoroughly, then add a profiling tool that replays generated token streams and reports draft coverage / acceptance potential. This avoids corrupting KV cache while we learn whether n-gram speculation has enough acceptance rate to justify a deeper target-verify path.

**Tech Stack:** Python 3.10+, TinyLLMForge tokenizer/LLM APIs, no new dependencies.

---

## File Structure

- Create `tinyvllm/speculative/__init__.py`: package export for speculative helpers.
- Create `tinyvllm/speculative/ngram.py`: pure CPU n-gram draft functions and replay metrics.
- Create `tools/test_ngram_speculative.py`: dependency-light unit tests.
- Create `tools/profile_ngram_spec.py`: run prompts through the existing engine, replay generated token ids, and report potential acceptance metrics.
- Modify `docs/qwen3-8b-fixes.md`: document the research question and first results after profiling.

## Task 1: N-gram draft primitive

**Files:**
- Create: `tinyvllm/speculative/__init__.py`
- Create: `tinyvllm/speculative/ngram.py`
- Test: `tools/test_ngram_speculative.py`

- [ ] **Step 1: Write failing tests**

Create `tools/test_ngram_speculative.py` with tests for longest suffix match, max draft length, no-match behavior, and replay metrics.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tools/test_ngram_speculative.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'tinyvllm.speculative'`.

- [ ] **Step 3: Implement minimal module**

Create `tinyvllm/speculative/ngram.py` with:

```python
from dataclasses import dataclass


@dataclass
class NGramDraft:
    tokens: list[int]
    match_start: int
    ngram_size: int


@dataclass
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
    if ngram_size <= 0:
        raise ValueError("ngram_size must be > 0")
    if max_draft_tokens <= 0:
        raise ValueError("max_draft_tokens must be > 0")
    if len(history) < ngram_size:
        return NGramDraft(tokens=[], match_start=-1, ngram_size=ngram_size)
    suffix = history[-ngram_size:]
    best = -1
    for i in range(0, len(history) - ngram_size):
        if history[i:i + ngram_size] == suffix:
            best = i
    if best < 0:
        return NGramDraft(tokens=[], match_start=-1, ngram_size=ngram_size)
    start = best + ngram_size
    end = min(len(history), start + max_draft_tokens)
    return NGramDraft(tokens=history[start:end], match_start=best, ngram_size=ngram_size)


def replay_ngram_acceptance(tokens: list[int], prompt_len: int, ngram_size: int, max_draft_tokens: int) -> NGramReplayStats:
    drafted = accepted = events = positions = 0
    for pos in range(prompt_len, len(tokens)):
        history = tokens[:pos]
        draft = propose_ngram_draft(history, ngram_size, max_draft_tokens)
        if not draft.tokens:
            positions += 1
            continue
        events += 1
        drafted += len(draft.tokens)
        future = tokens[pos:pos + len(draft.tokens)]
        for a, b in zip(draft.tokens, future):
            if a != b:
                break
            accepted += 1
        positions += 1
    return NGramReplayStats(positions=positions, drafted_tokens=drafted, accepted_tokens=accepted, draft_events=events)
```

- [ ] **Step 4: Run tests to verify green**

Run: `python3 tools/test_ngram_speculative.py`

Expected: `ngram speculative tests passed`.

- [ ] **Step 5: Commit**

Commit message: `研究：新增 n-gram speculation 原语`

## Task 2: Offline profiling tool

**Files:**
- Create: `tools/profile_ngram_spec.py`
- Test: `tools/test_ngram_speculative.py`

- [ ] **Step 1: Add CLI smoke test via pure function**

Extend tests to call a helper that summarizes `NGramReplayStats` into a dict with `acceptance_rate`, `avg_draft_len`, and `accepted_tokens`.

- [ ] **Step 2: Implement profiler**

Create `tools/profile_ngram_spec.py` that:

1. accepts `--model`, `--prompts`, `--max-output-len`, `--ngram-size`, `--max-draft-tokens`;
2. runs normal `LLM.generate()`;
3. concatenates prompt token ids and output token ids;
4. calls `replay_ngram_acceptance()`;
5. prints JSON summary.

- [ ] **Step 3: Run syntax/test verification**

Run: `python3 -m py_compile tools/profile_ngram_spec.py tinyvllm/speculative/ngram.py tools/test_ngram_speculative.py && python3 tools/test_ngram_speculative.py`.

- [ ] **Step 4: Run a remote smoke profile**

Use `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` on Qwen3-8B with 3 short prompts and `--max-output-len 64`.

- [ ] **Step 5: Commit**

Commit message: `研究：添加 n-gram speculation 离线 profiler`

## Task 3: Research documentation

**Files:**
- Modify: `docs/qwen3-8b-fixes.md`

- [ ] **Step 1: Add §35**

Document the question: whether n-gram speculation has enough acceptance potential in TinyLLMForge workloads to justify KV-safe target verification integration.

- [ ] **Step 2: Record profiler results**

Include acceptance rate, average draft length, accepted token count, and conclusion.

- [ ] **Step 3: Commit**

Commit message: `记录：n-gram speculation v0 画像`

## Self-Review

- Spec coverage: covers draft primitive, profiling tool, tests, remote smoke, and documentation.
- Placeholder scan: no TBD/TODO placeholders remain.
- Scope check: intentionally does not implement full KV-mutating target verify in v0; that becomes v1 only if profiler shows useful acceptance.
