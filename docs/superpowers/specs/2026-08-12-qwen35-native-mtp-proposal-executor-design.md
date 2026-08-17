# Qwen3.5 Native MTP Proposal Executor Design

**Date:** 2026-08-12

**Status:** Approved continuation design

## Objective

Implement the first real learned proposal source for the generic speculative
runtime by loading and executing the native Qwen3.5 MTP checkpoint head inside
`ModelRunner`.

The implementation must:

- use the checkpoint's real `mtp.*` parameters;
- share the target model embedding and LM head;
- keep target and MTP CUDA tensors inside `ModelRunner`;
- produce the existing tensor-free `DraftProposal` contract;
- preserve source-neutral Engine, Scheduler, verifier, residency, and
  transactional-KV code;
- maintain a real MTP full-attention KV cache with transactional
  commit/rollback;
- use distinct exact-Q execution families without Q padding, rounding, or
  merging.

This is a complete native-MTP executor slice, not another isolated Qwen3.5
projection optimization.

## Evidence and Checkpoint Contract

The read-only checkpoint selected for the first implementation is:

```text
/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/
qwen35-2b-hybrid-acquire-20260723-222004/model
```

Its model configuration reports:

- `hidden_size = 2048`;
- `vocab_size = 248320`;
- `num_hidden_layers = 24`;
- `mtp_num_hidden_layers = 1`;
- `mtp_use_dedicated_embeddings = false`;
- `tie_word_embeddings = true`.

The safetensors header contains these BF16 MTP tensors:

| Tensor | Shape |
| --- | --- |
| `mtp.fc.weight` | `(2048, 4096)` |
| `mtp.layers.0.input_layernorm.weight` | `(2048,)` |
| `mtp.layers.0.self_attn.q_proj.weight` | `(4096, 2048)` |
| `mtp.layers.0.self_attn.k_proj.weight` | `(512, 2048)` |
| `mtp.layers.0.self_attn.v_proj.weight` | `(512, 2048)` |
| `mtp.layers.0.self_attn.o_proj.weight` | `(2048, 2048)` |
| `mtp.layers.0.self_attn.q_norm.weight` | `(256,)` |
| `mtp.layers.0.self_attn.k_norm.weight` | `(256,)` |
| `mtp.layers.0.post_attention_layernorm.weight` | `(2048,)` |
| `mtp.layers.0.mlp.gate_proj.weight` | `(6144, 2048)` |
| `mtp.layers.0.mlp.up_proj.weight` | `(6144, 2048)` |
| `mtp.layers.0.mlp.down_proj.weight` | `(2048, 6144)` |
| `mtp.norm.weight` | `(2048,)` |
| `mtp.pre_fc_norm_embedding.weight` | `(2048,)` |
| `mtp.pre_fc_norm_hidden.weight` | `(2048,)` |

The loader must validate the configuration, complete tensor-name set, dtype,
and exact shapes before mutating the live MTP module. Missing, extra-required,
wrong-shape, or wrong-dtype tensors fail closed.

## First-Slice Scope

The first implementation is limited to:

- tensor parallel size 1;
- KV offload disabled;
- one native Qwen3.5 MTP layer;
- shared target embedding and LM head;
- greedy proposal sampling;
- batch-native execution;
- exact proposal length per sequence;
- distinct exact-Q execution families;
- target and MTP KV caches resident on the same CUDA worker;
- the existing target-verifier transactional KV path;
- an executor-local transactional MTP KV path.

The slice does not claim:

- TP4 support;
- KV-offload compatibility;
- multiple MTP layers;
- dedicated MTP embeddings;
- probabilistic proposal sampling;
- a second model architecture;
- 4K/16K/32K promotion coverage;
- performance improvement;
- H2D/D2H reduction;
- production promotion readiness.

Overall status remains `NOT_PROMOTABLE` until the promotion gates in this
document are satisfied.

## Considered Architectures

### 1. Native ModelRunner-Local MTP Executor

Load the Qwen3.5 MTP module beside the target model, share target embedding and
LM head, keep its hidden states and KV cache local, and expose only
`DraftProposal` plus a tensor-free proposal transaction ID.

This is the selected architecture. It matches the checkpoint, preserves the
source-neutral runtime boundary, and provides a real path to learned proposal
evidence.

### 2. Independent Draft ModelRunner

Run a separate draft-model worker and transfer target hidden states to it.

This is deferred because no compatible real learned-draft checkpoint has been
identified. It would also introduce hidden-state transport, duplicate model
ownership, and a second worker/KV lifecycle before those costs are justified.

### 3. Deterministic Stub or Stateless Full-Prefix Recompute

Register the profiler-only draft schema or recompute the MTP prefix on every
decode step.

This is rejected. The profiler schema is not a learned model, and stateless
recompute requires historical target hidden states while defeating the intended
runtime and performance properties.

## Model-Specific Components

### Qwen3.5 MTP Module

The module implements the checkpoint's native forward:

```text
shared embedding(input_token)
        │
pre_fc_norm_embedding
        │
        ├──────────────┐
        │              │
target_or_mtp_hidden   │
        │              │
pre_fc_norm_hidden     │
        │              │
        └─ concat ─────┘  [2 * hidden_size]
                │
             mtp.fc
                │
 Qwen3.5 full-attention decoder layer
                │
             mtp.norm
                │
      shared target LM head
```

The implementation reuses the existing Qwen3.5 RMSNorm, full-attention layer,
MLP, rotary embedding, and packed linear conventions. It does not add an
alternative projection layout.

### Checkpoint Binding

The existing target checkpoint loader currently skips `mtp.*` tensors. A
separate model-specific MTP binding maps only the validated `mtp.*` source
tensors to the MTP module destinations.

The target binding remains target-only. The MTP binding owns:

- complete source-name validation;
- packed Q/K/V destination mapping;
- packed gate/up destination mapping;
- assignment preflight;
- all-or-nothing mutation;
- loaded-source reporting.

Shared embedding and LM-head parameters are references to the loaded target
objects and are not loaded or copied a second time.

### Registration Factory

Qwen3.5 model construction may register the executor only after:

1. the target model is fully loaded;
2. checkpoint metadata confirms the supported MTP contract;
3. the MTP module is constructed on the target device;
4. every required MTP parameter is bound successfully;
5. shared embedding and LM-head identity checks pass.

All Qwen3.5 and checkpoint-specific dispatch remains in this factory and its
loader. Generic runtime files receive only an executor ID and capabilities.

## Generic Executor Lifecycle Extension

The current `ProposalExecutor.propose_batch()` contract is sufficient for a
stateless mock but not for a full-attention learned executor. The source-neutral
contract gains ModelRunner-local lifecycle operations:

```python
@dataclass(frozen=True)
class ProposalFinalizeRow:
    sequence_id: int
    proposal_transaction_id: str
    accepted_proposal_tokens: int


class ProposalExecutor(Protocol):
    @property
    def capabilities(self) -> DraftCapabilities: ...

    def observe_target_prefill(
        self,
        rows: tuple[TargetPrefillObservation, ...],
    ) -> None: ...

    def propose_batch(
        self,
        inputs: tuple[ModelRunnerProposalInput, ...],
    ) -> tuple[DraftProposal, ...]: ...

    def prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str: ...

    def commit_finalize_batch(self, ticket_id: str) -> None: ...

    def rollback_finalize_batch(self, ticket_id: str) -> None: ...
```

`TargetPrefillObservation` and `ModelRunnerProposalInput` are internal
ModelRunner values and may contain CUDA tensors. `DraftProposal`,
`ProposalFinalizeRow`, and transaction IDs are tensor-free.

The returned finalization ticket ID is tensor-free and identifies
ModelRunner-local prepared state. Preparing a ticket performs all sequence,
transaction, accepted-count, epoch, and slot-ownership validation without
publishing any cache mutation. Committing or rolling back a ticket is
single-use.

The registry validates lifecycle support from capabilities rather than
checking proposal-source or model names. Stateless host adapters remain
unchanged.

## Proposal Token Semantics

The current verifier compares proposal token zero with the target's first
token. Therefore an exact length `Q` proposal is:

```text
[
  first_target_token,
  mtp_token_1,
  ...,
  mtp_token_(Q - 1),
]
```

Consequences:

- `Q == 0` returns an empty proposal;
- `Q == 1` returns only `first_target_token` and performs no MTP forward;
- `Q > 1` performs exactly `Q - 1` autoregressive MTP forwards;
- proposal token zero is target-produced, not an MTP prediction;
- the first MTP forward consumes `first_target_token` plus the current target
  hidden state;
- every later MTP forward consumes the previous MTP token plus the previous
  MTP hidden state.

The proposal length is:

```text
min(
    configured_max_proposal_tokens,
    remaining_output_tokens,
)
```

No source-specific truncation occurs outside the executor.

## Prefill Bootstrap

A full-attention MTP layer needs prefix KV before the first speculative decode.
It cannot reconstruct this state from only the final target hidden row.

During target prefill, ModelRunner forwards the local target token IDs,
positions, hidden states, and sequence identity to each lifecycle-aware
executor. The Qwen3.5 executor retains the minimum pending bootstrap payload
needed until the first sampled target token becomes visible in the sequence.

At the first speculative decode, before proposal generation:

1. validate that the observed target prefix and current sequence identity
   match;
2. shift the prefix token embeddings by one position;
3. use the first already-sampled target token as the final shifted embedding;
4. pair those embeddings with the observed target hidden rows;
5. run the MTP layer over the exact prefix positions;
6. populate committed MTP KV through the final prompt position;
7. discard bootstrap-only logits and temporary hidden rows.

Bootstrap is idempotent only before publication. Once committed for a sequence,
a second bootstrap request is an invariant violation.

Pending bootstrap tensors never leave ModelRunner and are released on sequence
abort, executor disablement, or successful bootstrap.

## Executor-Local MTP KV

The MTP full-attention layer owns a separate paged KV store. Target KV pages
must not be aliased because the target and MTP layers have different weights,
layer counts, and sequence alignment.

The store maintains, per sequence:

- logical MTP block identities;
- committed length;
- physical CUDA slot bindings;
- at most one active proposal transaction;
- staged slot range;
- transaction generation;
- sequence epoch.

The initial implementation may reuse generic block-allocation primitives, but
MTP cache ownership and alignment remain encapsulated by the executor.

## MTP Proposal Transaction

For proposal length `Q`, the executor stages `Q - 1` MTP KV entries:

```text
forward 0: input first_target_token, writes transition 0
forward 1: input mtp_token_1,     writes transition 1
...
forward Q-2: input mtp_token_Q-2, writes transition Q-2
```

The proposal carries a tensor-free transaction ID in validated metadata. It
does not carry CUDA slots, tensors, cache objects, or allocator handles.

If the verifier accepts `k` proposal tokens, the executor commits:

```text
max(k - 1, 0)
```

staged MTP KV entries and rolls back the remainder.

The `k - 1` alignment is required because the last accepted token is paired
with the next target hidden state on the following decode step. Committing `k`
entries would advance MTP KV one position too far.

Finalize validation requires:

- exact sequence ID;
- exact active transaction ID;
- matching sequence epoch;
- `0 <= k <= proposal_length`;
- no duplicate finalize;
- no finalize after abort;
- no overlapping transaction for the sequence.

Accepted entries remain in their existing physical slots. Rejected suffix
slots are returned directly to the MTP allocator. There is no accepted-entry
replay, copy, or per-token rematerialization.

## Engine and Scheduler Finalization Bridge

The generic batch runtime already computes the accepted proposal prefix. It
adds tensor-free `ProposalFinalizeRow` records to the prepared speculative
result.

`LLMEngine` uses common ModelRunner prepare, commit, and rollback callbacks.
The callbacks route by registered executor ID, not by model or source type.

Ordering is:

1. target verifier forward completes;
2. target KV commit plans are prepared;
3. MTP finalization is prepared and all contract/ownership checks pass;
4. target transactional KV and sequence metadata commit;
5. the prepared MTP finalization ticket commits the same accepted counts;
6. only then may the next proposal batch start for those sequences.

If target or Scheduler commit fails before step 5, Engine rolls back the
prepared MTP finalization ticket, which aborts all still-staged proposal
transactions. A failure while committing the already-prevalidated MTP ticket
after target publication poisons the speculative runtime; it cannot retry or
pretend that target and MTP cache state are aligned.

If failure occurs before target replay starts, both target and MTP transactions
abort and the existing fallback policy may run. Once target replay starts, a
CUDA failure must propagate; eager retry is forbidden.

## Exact-Q Execution Families

Each sequence's effective Q is exact. The executor partitions a mixed batch by
Q and executes a separate family for every present value.

An execution-family key contains at least:

```text
(
  exact_q,
  exact_batch_size,
  device,
  compute_dtype,
  hidden_size,
  mtp_layer_count,
)
```

For family `Q`, the captured or eager body contains exactly `Q - 1` MTP
forwards. A `Q=1` family contains no MTP layer execution.

The implementation must not:

- pad a shorter Q to a larger graph;
- round Q to a bucket;
- merge distinct Q values;
- generate tokens and discard padded suffixes;
- reuse a graph whose captured Q differs from the request Q.

Eager execution may be used as a correctness oracle while graph capture is
being established, but graph support is part of the first-slice completion
gate.

## Error Handling

### Startup Failures

Unsupported config, incomplete MTP tensors, incorrect shapes/dtypes, failed
shared-parameter identity, or unsupported runtime mode prevent executor
registration. The target model may remain usable without speculative MTP.

### Bootstrap Failures

Missing observations, prefix identity mismatch, stale sequence epochs, or
bootstrap CUDA errors abort unpublished MTP state. When this occurs before
target replay, the existing non-speculative fallback remains allowed.

### Proposal Failures

An exception before a proposal transaction is published aborts all staged MTP
slots. A published transaction must be finalized or explicitly aborted before
another transaction can start.

### Finalization Failures

Unknown, stale, duplicated, mismatched, or out-of-range finalization fails
during prepare, before target publication. The executor must not guess an
accepted count or silently reset its cache. Reusing a prepared ticket, or
failing its metadata-only commit after target publication, is a hard poisoned
runtime error.

### Post-Replay CUDA Failures

After target replay begins, CUDA errors propagate without eager retry. This
preserves the existing no-double-mutation boundary.

## Source-Neutrality Rules

These generic files must not contain Qwen3.5, MTP-source, learned-drafter, or
checkpoint-name branches:

- `tinyvllm/speculative/batch_runtime.py`;
- `tinyvllm/speculative/verifier.py`;
- `tinyvllm/engine/speculative_runtime.py`;
- `tinyvllm/engine/speculative_model_runner.py`;
- `tinyvllm/engine/scheduler.py`;
- generic target KV transaction and residency modules.

Allowed generic decisions include:

- `execution_domain`;
- lifecycle capability presence;
- executor ID lookup;
- exact Q;
- accepted token count;
- transaction state;
- TP and KV-offload capability gates.

All source/model-specific construction and loading remains under Qwen3.5 model
modules and the ModelRunner registration factory.

## Testing Strategy

### Checkpoint and Binding Tests

- exact 15-tensor source-name contract;
- exact BF16 shape contract;
- packed Q/K/V and gate/up mapping;
- preflight before mutation;
- all-or-nothing assignment;
- shared embedding and LM-head object identity;
- wrong config, shape, dtype, missing source, and duplicate source rejection.

### MTP Math Tests

On deterministic small tensors:

- offset RMSNorm parity;
- embedding/hidden normalization;
- concatenation order;
- `fc` projection orientation;
- full-attention layer invocation;
- final norm;
- shared LM-head logits;
- greedy token selection.

The oracle must be independent of the production module and match the official
Qwen3.5 MTP equations.

### Proposal Semantics Tests

- `Q=0`, `Q=1`, and multiple `Q>1` cases;
- exact `Q - 1` forward count;
- proposal starts with `first_target_token`;
- subsequent token/hidden feedback alignment;
- remaining-budget truncation;
- batch result order and sequence identity;
- no tensor in returned proposal rows or metadata.

### Bootstrap Tests

- shifted prompt-token alignment;
- final sampled token placement;
- exact target-hidden pairing;
- committed prefix length;
- idempotence rejection;
- sequence epoch mismatch;
- pending-tensor cleanup.

### Transaction Tests

For every `Q` and every `k` in `[0, Q]`:

- stage count is `Q - 1`;
- commit count is `max(k - 1, 0)`;
- rejected suffix slots are released;
- accepted slots preserve physical identity;
- duplicate/stale finalize fails;
- finalization prepare performs no visible cache mutation;
- rollback of a prepared finalization releases staged suffix state;
- commit/rollback tickets are single-use;
- abort leaves committed prefix unchanged;
- no replay/copy/rematerialization path executes.

### Exact-Q Tests

- mixed-Q batch partitions into distinct families;
- family key contains exact Q;
- no padding or rounded family lookup;
- eager and captured family token parity;
- captured family uses exactly `Q - 1` MTP forwards.

### Integration Tests

- constructed ModelRunner with real Qwen3.5 component shapes;
- target prefill observation through first speculative decode;
- fused first-target plus native-MTP proposal;
- target and MTP transaction finalization with one accepted prefix;
- fallback before replay;
- no retry after replay starts;
- generic static source-neutrality scans;
- existing host n-gram/SAM regressions.

### Remote Real-Checkpoint Gates

Using the selected checkpoint:

- loader and shared-weight identity;
- eager MTP hidden/logit parity against an independent reference;
- greedy proposal parity;
- batch 1 and batch 4;
- multiple exact Q values;
- partial-accept rollback followed by deterministic continuation;
- exact-Q CUDA graph versus eager parity;
- peak memory and cache-slot accounting.

Remote evidence must distinguish real MTP KV operations from simulated cache
copies.

## Completion and Promotion Boundaries

The implementation slice is complete only when:

1. the real checkpoint loads through the native MTP binding;
2. prefill bootstrap creates aligned MTP KV;
3. proposals execute through exact-Q eager and CUDA-graph families;
4. accepted MTP KV commits in place;
5. rejected MTP KV suffix rolls back without replay or copy;
6. target and MTP transaction counts remain aligned;
7. exact greedy parity passes on the real checkpoint;
8. existing source-neutral runtime and host-adapter tests remain green.

Even after this slice completes, the generic optimization remains
`NOT_PROMOTABLE` until broader project gates cover:

- at least two model structures;
- TP1 and TP4;
- 4K, 16K, and 32K or longer contexts;
- batch 1, batch 4, and multi-sequence workloads;
- exact greedy parity;
- TPOT, TTFT, throughput, peak memory, real KV H2D bytes, and acceptance;
- real KV offload rather than simulated copy accounting.

No result from this first slice may be represented as satisfying those broader
promotion gates.
