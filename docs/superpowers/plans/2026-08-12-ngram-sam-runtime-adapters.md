# N-Gram and SAM Runtime Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement concrete dependency-light n-gram and SAM adapters for the
generic speculative batch runtime, with explicit target-verified SAM state
lifecycle.

**Architecture:** `NGramDraftAdapter` is a stateless wrapper around
`propose_ngram_draft()`. `SAMDraftAdapter` owns one
`SuffixAutomatonDraftIndex` per sequence and separates read-only proposal from
explicit verified-history synchronization.

**Tech Stack:** Python 3.9+, dataclasses already defined in
`tinyvllm.speculative.adapter`, existing n-gram and SAM helpers, pytest.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, run `git clean`,
  or modify index state.
- No model-name behavior branches.
- Do not modify scheduler, engine, model-runner, KV transaction, offload,
  KV4/KV8, TP, CUDA Graph, or Qwen3.5-specific behavior.
- Proposal generation must not mutate target-verified history.
- Rejected draft tokens must never enter the SAM index.
- `remaining_output_tokens` remains advisory; runtime truncation remains
  authoritative.
- Keep tests dependency-light and CPU-only.
- No performance claim.

---

### Task 1: Dependency-Light RED Tests

**Files:**
- Create: `tools/test_speculative_source_adapters.py`

**Interfaces:**
- Consumes:
  - `DraftContext`
  - `validate_draft_adapter_batch()`
  - `propose_ngram_draft()`
  - `SuffixAutomatonDraftIndex`
- Produces executable behavior requirements for:
  - `NGramDraftAdapter`
  - `SAMDraftAdapter`

- [x] **Step 1: Add dependency-light package setup**

Follow `tools/test_speculative_adapter.py`: add the repository root to
`sys.path`, register lightweight `tinyvllm` and `tinyvllm.speculative`
packages, and import the future concrete adapter modules without importing
the top-level `tinyvllm` dependency surface.

- [x] **Step 2: Add n-gram behavior tests**

Use histories such as:

```python
(1, 2, 3, 1, 2)
```

Assert a configured `ngram_size=2` adapter proposes `(3, 1, 2)`, a no-match row
returns `()`, a context limit of zero returns `()`, and a smaller context
limit caps the proposal without using `remaining_output_tokens` as a hard
limit.

- [x] **Step 3: Add SAM lifecycle tests**

Assert:

```python
adapter.register_sequence(7, (1, 2, 3, 1, 2))
proposal = adapter.propose_batch((_context(7, ...),))[0]
assert proposal.token_ids == (3, 1, 2)
assert adapter.synchronize_verified_history(
    7,
    (1, 2, 3, 1, 2, 9),
) == 1
adapter.release_sequence(7)
```

Cover duplicate registration, unknown synchronization/release, stale context,
truncated history, and rewritten history.

- [x] **Step 4: Add proposal purity and rejected-token tests**

Snapshot `indexed_tokens`, state count, and `last_state` before proposal and
assert they are unchanged afterward. Propose a token, synchronize a different
verified token, then assert the rejected proposal token was not appended.

- [x] **Step 5: Add fixed and match-aware policy tests**

Assert fixed mode obeys adapter/context limits. Assert match-aware mode records
both the policy-selected K and the capped actual K, and supports mixed
empty/non-empty rows in stable input order.

- [x] **Step 6: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_speculative_source_adapters.py
```

Expected: collection fails because
`tinyvllm.speculative.ngram_adapter` and
`tinyvllm.speculative.sam_adapter` do not exist.

---

### Task 2: N-Gram Adapter

**Files:**
- Create: `tinyvllm/speculative/ngram_adapter.py`
- Test: `tools/test_speculative_source_adapters.py`

**Interfaces:**
- Consumes:
  - `DraftCapabilities`
  - `DraftContext`
  - `DraftProposal`
  - `propose_ngram_draft(history, ngram_size, max_draft_tokens)`
- Produces:
  - `NGramDraftAdapter`

- [x] **Step 1: Implement constructor and capabilities**

Validate positive integer `ngram_size` and `max_proposal_tokens`, rejecting
booleans. Publish source type `ngram`, batch support, no target payload
requirements, and the configured proposal maximum.

- [x] **Step 2: Implement ordered batch proposal**

For each context compute:

```python
effective_limit = min(
    self.capabilities.max_proposal_tokens,
    context.max_proposal_tokens,
)
```

Return an empty proposal for zero. Otherwise call
`propose_ngram_draft()` with a copied list of the immutable token tuple.
Return metadata defined by the design and preserve input order.

- [x] **Step 3: Run focused GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_speculative_source_adapters.py -k ngram
```

Expected: all n-gram adapter tests pass while SAM tests still fail because the
SAM adapter module is absent.

---

### Task 3: SAM Adapter

**Files:**
- Create: `tinyvllm/speculative/sam_adapter.py`
- Test: `tools/test_speculative_source_adapters.py`

**Interfaces:**
- Consumes:
  - `DraftCapabilities`
  - `DraftContext`
  - `DraftProposal`
  - `SuffixAutomatonDraftIndex`
- Produces:
  - `SAMDraftAdapter`
  - `register_sequence(sequence_id, verified_token_ids)`
  - `synchronize_verified_history(sequence_id, verified_token_ids)`
  - `release_sequence(sequence_id)`

- [x] **Step 1: Implement fail-closed lifecycle validation**

Validate integer non-boolean sequence IDs and tuple integer non-boolean token
histories. Reject duplicate registration and unknown synchronization or
release.

- [x] **Step 2: Implement verified-history synchronization**

Require the current indexed stream to be an exact prefix of the complete
verified history. Append only the verified suffix, assert final equality, and
return the appended count. Validate before mutation so truncated or rewritten
history leaves the index unchanged.

- [x] **Step 3: Implement fixed proposal mode**

Require exact equality between `DraftContext.token_ids` and indexed history,
then call `index.propose(effective_limit)`. Return existing SAM metadata plus
policy, policy-selected K, adapter limit, and history count.

- [x] **Step 4: Implement match-aware proposal mode**

Obtain the SAM policy-selected K, cap it by the effective limit, and regenerate
the draft with the capped K when necessary. Preserve both selected values in
metadata. Do not mutate index state.

- [x] **Step 5: Run complete adapter GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_speculative_source_adapters.py
```

Expected: all concrete adapter and lifecycle tests pass.

---

### Task 4: Public API, Documentation, and Regression

**Files:**
- Modify: `tinyvllm/speculative/__init__.py`
- Modify: `tools/test_speculative_public_api.py`
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: completed concrete adapters.
- Produces: stable package exports and exact current validation boundaries.

- [x] **Step 1: Add RED public-export assertions**

Assert `NGramDraftAdapter` and `SAMDraftAdapter` appear in package attributes
and `__all__`.

- [x] **Step 2: Run public API RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_speculative_public_api.py
```

Expected: failure because the new adapter names are not exported.

- [x] **Step 3: Export adapters and run public API GREEN**

Import both adapter classes in `tinyvllm/speculative/__init__.py`, add them to
`__all__`, and rerun the public API test.

- [x] **Step 4: Run focused speculative regression**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_speculative_source_adapters.py \
  tools/test_speculative_adapter.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_public_api.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_ngram_speculative.py
```

Expected: all tests pass.

- [x] **Step 5: Run compatibility and hygiene checks**

Run Python 3.9 and 3.12 `py_compile` for the new modules and tests, then:

```bash
rg -n \
  "Qwen|Llama|Mistral|rematerial|replay.*KV|copy.*KV" \
  tinyvllm/speculative/ngram_adapter.py \
  tinyvllm/speculative/sam_adapter.py

git diff --check
git diff --cached --name-only
```

Expected: source scan has no matches, diff check passes, and staged diff is
empty.

- [x] **Step 6: Update audit and handoff**

Record concrete adapter APIs, SAM lifecycle ordering, fresh test counts, and
the unchanged limitations:

```text
scheduler-visible speculative work:
  not implemented
real ModelRunner callbacks:
  not implemented
GPU/TP/long-context/exact-model parity:
  unproven
TPOT/TTFT/throughput/memory/real-KV-H2D improvement:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 7: Mark this plan complete**

Only after fresh verification, change every checkbox in this plan to `[x]`.
