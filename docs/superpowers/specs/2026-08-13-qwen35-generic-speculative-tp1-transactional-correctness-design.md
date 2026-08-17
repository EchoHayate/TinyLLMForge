# Qwen3.5 Generic Speculative TP1 Transactional Correctness

## Status

The staged direction is approved:

1. establish a real Qwen3.5 TP1 generic n-gram speculative correctness
   authority;
2. make recurrent hybrid state participate in the same speculative
   transaction as full-attention KV; and
3. promote the same contract to TP4 only after TP1 is exact and fail-closed.

This specification has completed design review and is paired with the
implementation plan at
`docs/superpowers/plans/2026-08-13-qwen35-generic-speculative-tp1-transactional-correctness.md`.
Repository constraints forbid staging or committing either document.

## Goal

Establish source-bound evidence that the existing generic speculative runtime
can execute a materially different hybrid/recurrent model architecture without
changing exact greedy behavior or leaving rejected speculative state visible.

The first authority uses the real Qwen3.5-2B checkpoint and a real TP1
`LLMEngine.step()` loop. It must prove:

1. baseline and generic n-gram candidate runs produce exactly equal greedy
   output tokens;
2. the generic `spec_first_target` and `spec_verify` callbacks execute against
   the production Qwen3.5 model;
3. accepted full-attention KV is committed through the existing generic KV
   transaction rather than reconstructed token by token;
4. rejected full-attention KV suffixes are rolled back;
5. Qwen3.5 convolution and recurrent state commits exactly the state
   corresponding to the accepted consumed-input prefix, not the complete
   proposed suffix;
6. candidate runs include both accepted and rejected draft tokens;
7. batch-1 and batch-4 multi-sequence execution remain exact; and
8. runtime poisoning, leaked hybrid-state leases, or incomplete cleanup fail
   the gate.

Passing this gate establishes only second-model TP1 transactional correctness.
It does not establish TP4 second-model support, KV offload, 16K/32K context,
performance, learned-drafter support, or Phase 1 completion.

## Verified Starting Point

### Model identity

The approved remote model mirror is:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-hybrid-state-runs/
qwen35-2b-hybrid-acquire-20260723-222004/model
```

Its approved model manifest SHA-256 is:

```text
3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

The root configuration identifies:

- `model_type="qwen3_5"`;
- `architectures=["Qwen3_5ForConditionalGeneration"]`; and
- a `qwen3_5_text` text model with 24 layers.

The text layer pattern is six repetitions of:

```text
linear_attention
linear_attention
linear_attention
full_attention
```

This is materially different from the standard Qwen3 decoder already covered
by the generic speculative TP4 authority.

### Existing Engine authority

The repository already has a production Qwen3.5 model-loading path in
`ModelRunner`, a real Qwen3.5 `LLMEngine.step()` backend, and prior TP4 Engine
correctness authority. The latest inspected remote TP4 Engine authority is:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-tp4-engine-authority-runs/
qwen35-tp4-engine-correctness-20260804-161810-
attempt67-qwen35-dual-receipt-schema-r539
```

Its `authority_summary.json` classification is `PASS`, and it binds the same
model manifest. Therefore this design does not rebuild ordinary Qwen3.5 model
loading, TP partitioning, or non-speculative generation.

The default factories in some Qwen3.5 test helpers intentionally raise
`not implemented` when no real backend is injected. Those fail-closed defaults
are not evidence that the production Engine path is absent.

### Existing generic speculative authority

The standard Qwen3 generic runtime already proves:

- a host-side `NGramDraftAdapter`;
- scheduler-native speculative selection;
- batched first-target and verification callbacks;
- prepared KV commit and rollback;
- callback and collective profiling;
- speculative residency acknowledgement;
- exact greedy baseline/candidate parity; and
- Engine cleanup.

That authority is specific to a standard full-attention decoder. It does not
prove correctness for mutable recurrent side state.

## Missing Correctness Boundary

Qwen3.5 owns two kinds of request state:

1. paged KV for its full-attention layers; and
2. convolution and recurrent tensors for its linear-attention layers.

The existing generic KV transaction controls the first kind. Qwen3.5
`HybridStateLease` and `Qwen35CrossLayerStateTransaction` provide ownership,
gather, and atomic cross-layer commit for the second kind, but the generic
speculative runtime does not yet bind them together.

A multi-token target verification can mutate recurrent state through the full
proposal. If only a prefix is accepted, committing the final recurrent state
would make rejected tokens visible to later decode steps. Restoring the
pre-verification state would discard accepted progress. Replaying accepted
tokens one by one would preserve correctness but violate the Phase 1 goal of
batch-native verification without accepted-state rematerialization.

The required state is the checkpoint after the exact number of input tokens
consumed by the committed speculative result. This count is not inferred from
the output length. The runtime must derive and carry a canonical
`committed_input_count` because autoregressive state normally lags the newest
emitted output token.

## Considered Approaches

### 1. Token parity only

Run baseline and n-gram cells and compare final output tokens.

This may detect common recurrent-state corruption, but it does not prove that
rejected state was rolled back or that accepted state was committed directly.
A short output can pass before corrupted state changes the selected token.
This approach is rejected.

### 2. Restore the original state and replay accepted tokens

Snapshot the live recurrent state, perform the batch verification, restore the
snapshot, and execute the accepted prefix again.

This is a useful diagnostic fallback because it defines a correctness oracle.
It is rejected as the production solution because it repeats target work and
reintroduces accepted-state rematerialization.

### 3. Generic side-state transaction with prefix checkpoints

Add an optional model-runner side-state transaction provider. Qwen3 remains a
no-side-state implementation. Qwen3.5 implements the provider by running
verification against private candidate tensors, retaining state checkpoints
for the consumed-input prefixes produced by the same batch-native target
forward, and atomically publishing only the selected checkpoint.

This is the selected approach. It keeps draft-source logic generic, prevents
rejected writes from reaching live state, and extends the existing prepared
KV transaction instead of creating a separate Qwen3.5 speculative runtime.

## Architecture

### Generic side-state provider

The ModelRunner speculative callback boundary gains an optional provider with
five lifecycle phases:

1. `prepare`
2. `select`
3. `apply`
4. `seal`
5. `rollback`

The provider is model-state-specific but draft-source-neutral. It must not
inspect whether proposals came from n-gram, SAM, native MTP, or an independent
draft model.

`prepare` receives:

- canonical sequence IDs;
- active hybrid-state leases;
- proposal lengths;
- callback identity; and
- a unique transaction ID.

It returns private execution bindings and a prepared transaction. The
execution bindings may be candidate tensors or private slots; they are not
required to use physical slots. Live request state must remain unchanged
during target verification.

`select` receives a per-sequence `committed_input_count` computed by the
generic speculative runtime after verification. It selects exactly one
captured prefix checkpoint for each sequence and returns a fail-closed
selection receipt.

`apply` atomically copies the selected convolution/recurrent state into the
live leased slots while retaining the original live state. It uses the
existing cross-layer transaction so a failed copy restores every touched
layer.

`seal` discards the retained original state only after KV, Scheduler, and
proposal-finalization publication have succeeded.

`rollback` destroys private state and leaves the live leased slots unchanged.
Before `apply` it is idempotent. After `apply` and before `seal`, it restores
the retained original state atomically. It is forbidden after `seal`.

The provider contract contains no Qwen3.5 tensor names or shapes.

### Qwen3.5 candidate-state execution

Qwen3.5 supplies a provider backed by its existing hybrid model owner,
`HybridStateTensorPool`, layer adapters, and cross-layer transaction.

For each prepared speculative row:

1. gather the live convolution/recurrent state;
2. retain the gathered tensors as the private transaction base;
3. run the first-target callback with commit suppressed and retain its
   candidate state;
4. run the target verification callback from that candidate state with commit
   suppressed;
5. retain prefix state checkpoints indexed by consumed input count;
6. select the checkpoint requested by the generic runtime; and
7. atomically publish only that checkpoint to the live lease.

The existing Qwen3.5 model path already separates
`layer_stack.prepare()` from `layer_stack.commit()`. The implementation should
preserve that split through a prepared-step result rather than creating a
second model or duplicating the live state pool.

The implementation must not perform a second model forward for the accepted
prefix. If the current recurrent operator cannot expose prefix checkpoints
from one batched verification, the gate remains RED and implementation stops
at that explicit blocker. Per-token replay is not silently accepted as the
production path.

Checkpoint capture is required only while a speculative transaction is
active. Ordinary prefill and decode retain their existing execution path.

### Unified prepared publication

The Engine already prepares speculative runtime output, KV commit plans,
Scheduler publication, and optional proposal-executor finalization before
making results visible.

The side-state transaction joins this boundary:

1. target verification finishes against uncommitted KV and shadow side state;
2. verification computes accepted draft tokens and
   `committed_input_count`;
3. KV commit plans are prepared;
4. side-state checkpoints are selected;
5. Scheduler publication is prepared;
6. side state is reversibly applied;
7. KV and Scheduler state are committed;
8. proposal finalization is committed;
9. side state is sealed;
10. any failure before Scheduler visibility rolls back every prepared or
    applied participant; and
11. any failure after irreversible partial visibility poisons the runtime and
    fails closed.

The implementation must define one deterministic commit order and the reverse
rollback order. No participant may report success while another participant
remains merely prepared.

### Canonical consumed-input mapping

The transaction schema records, per sequence:

- proposal token count;
- accepted draft token count;
- emitted output token count;
- first-target input count;
- verification-tail input count;
- committed input count;
- selected state checkpoint index; and
- committed KV token count.

The runtime computes these values from the prepared speculative batch. The
Qwen3.5 provider validates the supplied count but does not recreate acceptance
logic.

For a row with `proposal_token_count` proposed tokens:

```text
verify_input_count = max(0, proposal_token_count - 1)
committed_tail_input_count =
  min(accepted_draft_count, verify_input_count)
committed_input_count = 1 + committed_tail_input_count
```

Checkpoint `1` is the candidate state after the first-target input. Tail
checkpoint `k` is indexed as committed input count `1 + k`. The newest emitted
output token remains unconsumed until the next decode iteration.

Tests must cover zero accepted drafts, a partially accepted proposal, a fully
accepted proposal, and termination by EOS or output budget. Off-by-one errors
between emitted tokens and consumed inputs must fail before publication.

## TP1 Authority Campaign

### Matrix

The initial remote matrix is:

- model: approved Qwen3.5-2B checkpoint;
- tensor parallel size: `1`;
- policies: `baseline`, `ngram`;
- batch sizes: `1`, `4`;
- context class: `4K`;
- sampling: greedy;
- n-gram size: `3`;
- maximum proposal tokens: `4`; and
- output budget: long enough to observe state after both accepted and rejected
  proposals.

The exact prompt token count must leave room for the complete output budget
within the configured model length. The artifact records both the requested
context class and the exact token count.

The workload contains:

- an acceptance-rich repeated-pattern prompt;
- a rejection-rich prompt whose pattern diverges after proposal;
- four distinct sequences in the batch-4 cell; and
- at least one sequence that continues for multiple decode iterations after a
  partial rejection.

Candidate evidence is invalid unless aggregate proposed, accepted, and
rejected draft token counts are all positive.

### Cell isolation

Every baseline or candidate cell runs in a fresh process with a fresh Engine.
Cells do not share:

- process groups;
- model state;
- hybrid-state pools;
- prefix-cache entries;
- speculative runtime lifecycle state; or
- profiler state.

This prevents a baseline cell from repairing or masking candidate corruption.

### Exact parity oracle

For each workload and batch size:

1. run a fresh baseline Engine;
2. run a fresh generic n-gram Engine;
3. compare every output token ID exactly;
4. require equal sequence termination reason and output length; and
5. continue generation beyond the first rejection so later tokens exercise
   the committed recurrent state.

An additional diagnostic oracle may restore the original state and replay the
committed prefix, but its output cannot substitute for the no-replay
production candidate result.

## Artifact Contract

The gate writes a new schema rather than extending the standard-Qwen TP4
artifact in place.

Top-level fields include:

- `schema_version`;
- `classification`;
- `claim_scope`;
- `limitations`;
- `source_tree_sha256`;
- `model_manifest_sha256`;
- `model_architecture`;
- `world_size`;
- `cells`;
- `parity`;
- `aggregate_runtime`;
- `side_state_authority`;
- `kv_transaction_authority`; and
- `cleanup_authority`.

Each candidate cell includes:

- prompt and output token rows;
- proposal, acceptance, and rejection counts;
- first-target and verify callback rows;
- transaction IDs and sequence IDs;
- consumed-input mapping rows;
- side-state prepare/select/apply/seal/rollback receipts;
- selected prefix checkpoint indices;
- KV prepare/commit/rollback receipts;
- proof that accepted KV did not use replay;
- runtime poison state;
- lease inventory before and after the cell; and
- Engine cleanup receipt.

The authoritative classification is:

```text
SECOND_MODEL_TP1_ESTABLISHED
```

It is not `PASS`, `PROMOTABLE`, or `PHASE1_COMPLETE`, so downstream tooling
cannot confuse the narrow gate with the full objective.

## Fail-Closed Invariants

### Model and execution identity

- The model manifest matches the approved Qwen3.5 manifest.
- The root model type is `qwen3_5`.
- The text layer inventory contains both linear-attention and full-attention
  layers.
- The Engine and ModelRunner classes are the production classes.
- Generation executes through `LLMEngine.step()`.

### Output and runtime

- Baseline and candidate output token rows are exactly equal.
- Candidate proposal, accepted, and rejected counts are positive.
- First-target and verify callbacks are both present.
- No baseline cell contains speculative callbacks.
- The candidate runtime is not poisoned.

### KV transaction

- Every speculative row has one prepared KV transaction.
- Committed KV length matches the canonical consumed-input mapping.
- Rejected KV suffixes are absent after commit.
- Accepted KV is committed without a second accepted-prefix model forward.
- Rollback leaves block ownership and refcounts equal to their pre-transaction
  values.

### Hybrid side-state transaction

- Every active Qwen3.5 sequence has exactly one live lease and one private
  speculative candidate-state binding.
- Live convolution/recurrent tensors are unchanged before commit.
- Every selected checkpoint exists and matches `committed_input_count`.
- No rejected-suffix checkpoint is published.
- Cross-layer apply and rollback are all-or-nothing.
- Applied state retains an exact original snapshot until seal.
- Rollback releases every private candidate-state binding.
- No private or live lease is leaked at cell cleanup.

### Failure handling

- Missing or duplicate transaction rows fail verification.
- Sequence, lease, transaction, or checkpoint identity mismatches fail before
  publication.
- A side-state failure rolls back the prepared KV and Scheduler participants.
- An incomplete rollback poisons the runtime.
- Failed remote runs preserve their partial artifacts and error receipts.

## Testing Strategy

Implementation follows strict TDD.

### Contract tests

First add failing tests for:

- side-state provider lifecycle validation;
- consumed-input mapping, including off-by-one cases;
- duplicate or missing sequence rows;
- partial acceptance checkpoint selection;
- rollback idempotence;
- apply-after-rollback, seal-after-rollback, and rollback-after-seal
  rejection; and
- artifact verifier failures.

### ModelRunner tests

Use small deterministic Qwen3.5 state fixtures to prove:

- verification writes only shadow state;
- partial acceptance selects the correct prefix state;
- rejected suffix state never reaches the live slot;
- cross-layer copy failure restores all original layers; and
- no accepted-prefix replay occurs.

### Engine integration tests

Construct an Engine with deterministic fake target callbacks and prove the
unified KV/side-state/Scheduler commit order and reverse rollback order.

Then run focused local tests for existing Qwen3 generic speculative behavior.
The optional provider must be a no-op for standard Qwen3, and existing generic
authority contracts must remain unchanged.

### Remote real-checkpoint gate

Only after local RED-to-GREEN completion:

1. synchronize the exact source closure to
   `sitian@10.232.195.203`;
2. use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
3. disable SSH ControlMaster and ControlPath;
4. run serial remote operations;
5. select a real GPU with recorded free-memory evidence;
6. execute all four TP1 cells;
7. independently verify the copied-back artifact; and
8. preserve failed artifacts.

## TP4 Promotion

TP4 work begins only after the TP1 artifact independently verifies.

The TP4 extension reuses the same side-state provider contract and adds:

- all-rank first-target and verification callback identity;
- all-rank TP collective identity;
- all-rank shadow-state transaction acknowledgements;
- matching consumed-input selections on ranks `0, 1, 2, 3`;
- all-rank lease cleanup; and
- exact baseline/candidate parity for batch 1 and 4.

The existing Qwen3.5 TP4 Engine authority supplies model loading and ordinary
generation evidence, but it cannot substitute for these speculative
transaction receipts.

## Explicit Non-Goals

This design does not:

- enable Qwen3.5 native checkpoint MTP;
- claim that Qwen3.5 native MTP is generic;
- enable KV offload for Qwen3.5;
- measure TPOT, TTFT, throughput, memory, or KV H2D bytes;
- cover 16K or 32K context;
- implement KV8 or KV4;
- add a learned draft model;
- fuse verifier, sampling, or commit kernels;
- add variable-length CUDA Graph authority;
- prove TP4 speculative correctness; or
- declare Phase 1 complete.

Those remain separate gates after second-model TP1 and TP4 correctness are
established.

## Completion Boundary

This design is implemented only when a source-bound, independently verified
real-checkpoint artifact proves all TP1 invariants above.

Local unit tests, synthetic state fixtures, the pre-existing Qwen3 authority,
or the pre-existing ordinary Qwen3.5 TP4 Engine authority are necessary
supporting evidence but are not sufficient completion evidence.
