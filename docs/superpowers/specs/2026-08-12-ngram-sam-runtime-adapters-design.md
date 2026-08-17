# N-Gram and SAM Runtime Adapters Design

**Date:** 2026-08-12  
**Status:** Approved by the existing generic speculative-runtime direction  
**Scope:** Concrete dependency-light proposal adapters only; no scheduler,
engine, model-runner, or performance integration

## Goal

Connect the existing n-gram and suffix-automaton proposal implementations to
the generic `DraftAdapter` protocol so
`execute_native_speculative_batch()` can consume either source without
source-specific branches.

The adapters must preserve the core speculative ownership rule:

- proposal generation never mutates target-verified sequence history;
- accepted or ordinary target tokens update SAM state only after the caller
  has committed those tokens;
- rejected draft tokens never enter the SAM index;
- the generic runtime remains authoritative for KV transactions, greedy
  acceptance, EOS handling, and output-budget truncation.

## Non-Goals

- Adding speculative selection policy to `Scheduler`.
- Connecting real `ModelRunner` first-target or tail-verifier batches.
- Automatically mutating adapters from batch-runtime results.
- Implementing a learned drafter or MTP adapter.
- Adding model-name checks or model-specific behavior.
- Changing n-gram or SAM matching algorithms.
- Claiming TPOT, TTFT, throughput, memory, or KV-movement improvement.

## Alternatives

### A. Put n-gram and SAM branches in the batch runtime

Rejected because the generic runtime would become coupled to concrete
proposal sources. Every future learned drafter or MTP implementation would
require another runtime branch.

### B. Let the SAM adapter update itself from proposals

Rejected because proposal tokens are unverified. Mutating the index during
proposal would allow rejected tokens to affect future drafts and violate the
target-verified-history invariant.

### C. Concrete protocol adapters with explicit SAM lifecycle

Selected. N-gram remains a stateless wrapper around
`propose_ngram_draft()`. SAM owns per-sequence indices and exposes explicit
register, synchronize-verified-history, and release operations. Proposal
generation is read-only.

## N-Gram Adapter

Create `tinyvllm/speculative/ngram_adapter.py`.

```python
class NGramDraftAdapter:
    def __init__(
        self,
        *,
        ngram_size: int,
        max_proposal_tokens: int,
    ): ...

    @property
    def capabilities(self) -> DraftCapabilities: ...

    def propose_batch(
        self,
        contexts: tuple[DraftContext, ...],
    ) -> tuple[DraftProposal, ...]: ...
```

Capabilities:

```text
source_type = "ngram"
supports_batch = true
requires_target_hidden = false
requires_target_logits = false
max_proposal_tokens = configured maximum
```

For each context, the effective proposal limit is:

```python
min(
    adapter.capabilities.max_proposal_tokens,
    context.max_proposal_tokens,
)
```

An effective limit of zero returns an empty proposal without calling
`propose_ngram_draft()`, whose standalone API requires a positive limit.
Otherwise the adapter calls:

```python
propose_ngram_draft(
    list(context.token_ids),
    ngram_size,
    effective_limit,
)
```

Proposal metadata records:

- `ngram_size`;
- `match_start`;
- `selected_k`;
- `history_token_count`;
- `bypass_reason`, either `selected_k_zero`, `no_match`, or `None`.

The adapter preserves input context order and sequence IDs. It ignores
`remaining_output_tokens` for hard truncation because that field is advisory
and the runtime owns final output-budget semantics.

## SAM Adapter

Create `tinyvllm/speculative/sam_adapter.py`.

```python
class SAMDraftAdapter:
    def __init__(
        self,
        *,
        max_proposal_tokens: int,
        match_aware: bool = False,
    ): ...

    def register_sequence(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> None: ...

    def synchronize_verified_history(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> int: ...

    def release_sequence(self, sequence_id: int) -> None: ...

    @property
    def capabilities(self) -> DraftCapabilities: ...

    def propose_batch(
        self,
        contexts: tuple[DraftContext, ...],
    ) -> tuple[DraftProposal, ...]: ...
```

Capabilities use `source_type = "sam"` and otherwise match the n-gram
adapter's dependency-free requirements.

### State ownership

The adapter exclusively owns:

```text
sequence_id -> SuffixAutomatonDraftIndex
```

`register_sequence()` initializes an index from a target-verified prompt or
history. Duplicate registration is an error because silently replacing an
index could hide a lifecycle bug.

`release_sequence()` removes the index. Releasing an unknown sequence is an
error for the same fail-closed reason.

### Verified synchronization

`synchronize_verified_history()` receives the complete target-verified token
history after ordinary or speculative metadata commit.

The method:

1. validates the sequence is registered;
2. verifies the existing indexed stream is an exact prefix of the supplied
   history;
3. appends only the new verified suffix through `extend_verified()`;
4. asserts the resulting index exactly matches the supplied history;
5. returns the number of appended tokens.

A shorter history or a rewritten prefix raises `ValueError` and leaves the
index unchanged.

The adapter does not accept draft tokens, accepted counts, or runtime result
objects directly. This keeps transaction commit and source-state commit as
separate explicit boundaries for the future engine integration.

### Read-only proposal

Before each proposal, the adapter calls:

```python
index.assert_history(list(context.token_ids))
```

This rejects stale or prematurely advanced source state.

In fixed mode, the adapter calls `index.propose(effective_limit)`.

In match-aware mode, it first obtains the SAM policy's selected K, caps it by
the effective adapter/context limit, and calls `index.propose(capped_k)` when
the cap changes the policy result. Metadata records both the policy-selected
K and the actual selected K.

Proposal metadata contains the existing SAM metadata plus:

- `policy`, either `fixed` or `match_aware`;
- `policy_selected_k`;
- `adapter_limit`;
- `history_token_count`.

Neither fixed nor match-aware proposal modifies indexed tokens, automaton
states, prompt length, or last-state ownership.

## Validation and Failure Semantics

Constructors reject:

- boolean or non-integer limits;
- non-positive adapter maximums;
- non-positive n-gram sizes;
- non-boolean `match_aware`.

Lifecycle methods reject:

- boolean or non-integer sequence IDs;
- non-tuple token histories;
- boolean or non-integer token IDs;
- duplicate registration;
- unknown synchronization or release;
- synchronization histories that truncate or rewrite indexed history.

`propose_batch()` returns one `DraftProposal` for every input context,
including empty proposals. It never silently skips unregistered SAM
sequences.

The existing `validate_draft_adapter_batch()` remains the authoritative
protocol-level validator for exact ID coverage, source identity, token type,
proposal length, payload capabilities, and timing shape.

## Testing

Create dependency-light tests in
`tools/test_speculative_source_adapters.py`.

Required n-gram coverage:

- constructor validation;
- matching and no-match rows in one batch, preserving the complete bounded
  continuation returned by the existing helper;
- context-level limit capping;
- zero context limit;
- stable IDs and order;
- immutable input history;
- protocol validation through `validate_draft_adapter_batch()`.

Required SAM coverage:

- register, duplicate-register, synchronize, and release lifecycle;
- stale, truncated, and rewritten history rejection;
- fixed and match-aware proposal limits;
- mixed empty/non-empty batch rows;
- no mutation during proposal;
- rejected draft tokens absent from later state;
- accepted/ordinary verified tokens appended only through explicit
  synchronization;
- protocol validation through `validate_draft_adapter_batch()`.

Public API coverage must prove both adapters are exported from
`tinyvllm.speculative`.

## Integration Boundary

This slice intentionally stops before scheduler integration.

The future engine owner must perform operations in this order:

1. select speculative work;
2. execute `execute_native_speculative_batch()`;
3. commit sequence output metadata;
4. synchronize the SAM adapter with the resulting complete target-verified
   history;
5. release adapter state when the sequence finishes or is aborted.

If KV commit or sequence metadata commit fails, SAM synchronization must not
run. If SAM synchronization fails after metadata commit, the engine must
treat it as a fatal source-state invariant failure and disable or rebuild that
sequence's SAM state rather than replay rejected draft tokens.

## Promotion Boundary

Passing these tests proves concrete source compatibility with the generic
adapter contract and explicit SAM state ownership. It does not prove:

- production scheduler selection;
- real multi-sequence GPU verification;
- exact model parity;
- TP1 or TP4 behavior;
- 4K, 16K, or 32K+ context behavior;
- end-to-end latency, throughput, memory, or real KV H2D improvement.

Overall goal classification remains `NOT_PROMOTABLE`.
