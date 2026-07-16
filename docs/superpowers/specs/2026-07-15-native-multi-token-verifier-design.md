# Native Multi-Token Decode Verifier Design

Date: 2026-07-15

## Objective

Replace the current speculative verifier's prefill-based tail forward and
accepted-KV decode rematerialization with a native, decode-equivalent
multi-token target forward.

The first phase must:

1. Preserve token-identical greedy output relative to stable normal decode.
2. Verify one linear draft of `K` tokens with at most one tail verifier
   forward after the existing first-target decode; `K=1` requires no tail
   forward.
3. Write the KV for every verifier query token directly into its final
   current/reserved cache slot using decode-equivalent attention semantics.
4. Commit an accepted prefix by updating sequence and block metadata only;
   accepted KV must not be replayed, copied, or recomputed.
5. Keep the final accepted token pending, matching the existing
   `Sequence`/`BlockManager` lifecycle.
6. Establish an independent row-expanded decode oracle before using the
   native path for performance claims.
7. Cover `K in {1,4,8,16}`, zero/partial/full acceptance, EOS, output-budget
   truncation, and cache-block boundaries.
8. Produce exactness and performance artifacts from an isolated remote
   Qwen3-0.6B smoke before any broader integration claim.

The first phase is limited to profiler-owned, greedy, single-sequence, linear
draft verification with FP KV and eager execution. It does not claim
production batching, ragged verification, non-greedy distribution
equivalence, CUDA graph support, KV-offload support, quantized-KV support, or
memory reduction.

## Motivation

The completed Prompt+Dynamic SAM gate reached exact greedy output only after
`rematerialize_accepted_kv()` replayed `accepted_tokens[:-1]` through normal
single-token decode. The resulting canonical gate was strictly `NO_GO`.

The replay was necessary because the current verifier tail is represented as
varlen prefill:

- `_build_verify_tail_plan()` supplies `draft_tokens[:-1]` as target inputs;
- `verify_and_commit_block()` sets `Context.is_prefill=True`;
- `Attention.forward()` therefore uses `flash_attn_varlen_func`;
- the tail KV is written before attention, but the produced states do not
  match the normal `flash_attn_with_kvcache` decode trajectory closely enough
  for exact continuation;
- accepted KV is consequently rebuilt one token at a time.

This removes the main benefit of accepting multiple tokens. A useful
speculative path must make the verifier itself produce the same KV trajectory
that later normal decode expects.

Primary-source comparisons support this direction:

- vLLM represents speculative target work as a multi-token decode query for
  one request, with consecutive positions and cache slots.
- SGLang allocates target KV slots before verification, lets the target
  verifier write candidate KV, and retains or moves the accepted path instead
  of replaying accepted tokens.
- FlashAttention's KV-cache interface supports `seqlen_q > 1` with causal
  masking aligned to the lower-right of the combined key sequence.

TinyLLMForge already has the other required lifecycle pieces:

- `reserve_append_blocks()` reserves capacity without publishing it;
- `commit_accepted_tokens()` assumes accepted KV already exists in the
  current/reserved slots;
- `materialized_tokens = final_len - 1` deliberately leaves the final token
  pending for the next decode input.

The missing component is a dedicated multi-query decode mode with an explicit
metadata contract.

## Alternatives Considered

### 1. Recommended: Native Multi-Query Decode Verifier

Add a dedicated `spec_verify` context mode. For one sequence, pass
`draft_tokens[:-1]` as a contiguous query of length `Q = K - 1`, use the
normal paged KV cache and final slot mapping, and invoke
`flash_attn_with_kvcache` once with `seqlen_q=Q`.

Advantages:

- matches the architecture used by mature speculative engines;
- preserves the existing final-slot and metadata-only commit lifecycle;
- eliminates accepted-token decode replay;
- adds no scratch KV allocation or accepted-KV copy;
- has a narrow first-phase implementation and test surface.

Risks:

- TinyLLMForge currently treats decode tensors as one query per batch row;
- context-length and position conventions must be made explicit for `Q > 1`;
- FlashAttention version behavior must be checked on the remote environment;
- optional decode features currently assume `q_len=1` and must fail closed.

### 2. Correctness Oracle: Row-Expanded Single-Token Decode

Represent each verifier query as an independent decode row. Every row shares
the same logical block table but uses the context length appropriate to that
query position.

This reuses the established single-token decode kernel and provides an
independent semantic oracle for logits and KV. It is not the product path:

- it duplicates block-table rows;
- it performs `Q` independent attention rows rather than one causal query;
- it increases metadata and launch work;
- safe KV comparison requires isolated oracle slots or serialized execution.

The oracle is mandatory for development and exactness tests, but is excluded
from performance conclusions.

### 3. Deferred: Scratch Verify Plus Accepted-KV Copy

Write all candidate KV to scratch slots, then copy only accepted KV to final
slots.

This isolates rejected candidates and generalizes to tree verification, but
adds scratch allocation, slot translation, device-to-device copies, and
failure cleanup. For the current linear draft, direct writes into reserved
final slots are already safe because rejected or unused blocks are not
published. This option is deferred until tree-shaped candidates or concurrent
mutation require stronger isolation.

## Decision

Implement Alternative 1 as the only performance candidate.

Implement Alternative 2 as a test-only correctness oracle.

Do not implement Alternative 3 in the first phase.

The native verifier remains profiler-owned initially. It does not enter the
production scheduler until the exactness matrix and remote performance smoke
both pass.

## Execution Mode

### Explicit Mode, Not Another Boolean Combination

`Context` will gain an explicit attention mode with these values:

- `prefill`
- `decode`
- `spec_verify`

Existing callers may continue to derive legacy `is_prefill` behavior during
the transition, but attention dispatch must use the explicit mode. The design
must not encode `spec_verify` as a fragile combination such as
`is_prefill=False` plus a non-null auxiliary tensor.

The mode contract is:

| Mode | Query shape per request | Attention path | May publish metadata |
| --- | ---: | --- | --- |
| `prefill` | `Q >= 1` | varlen prefill | prefill lifecycle only |
| `decode` | `Q = 1` | KV-cache decode | normal decode lifecycle |
| `spec_verify` | `Q >= 0`, normally `K-1` | KV-cache multi-query decode | no |

`Q=0` is valid when `K=1`. In that case the existing first-target decode is
the complete verifier and no tail forward occurs.

### Fail-Closed Compatibility

The first phase supports `spec_verify` only when all of the following hold:

- exactly one sequence;
- linear draft;
- greedy target acceptance;
- eager execution;
- FP16 or BF16 KV cache;
- no KV offload;
- no blockwise attention;
- no Quest;
- no Attention Matching compaction;
- no KV cartridge;
- no C4 or C8 KV quantization;
- no mixed prefill/decode batch.

Unsupported combinations must raise a descriptive error before KV mutation.
They must not silently fall back to prefill or to decode rematerialization.

## Verifier Tensor Contract

Let:

- `H = len(seq)` before verification;
- draft tokens be `d[0:K]`;
- `Q = max(0, K - 1)`;
- `proxy_block_table = seq.block_table + reserved_block_ids`.

The existing first-target decode produces `t[0]`, the target token against
which `d[0]` is checked.

The tail verifier inputs are:

- `input_ids = d[0:Q]`
- `positions = [H + 1, H + 2, ..., H + Q]`
- logical slot positions = `[H, H + 1, ..., H + Q - 1]`
- physical `slot_mapping` is derived from `proxy_block_table`
- one block-table row containing every block visible through position
  `H + Q - 1`
- `query_lens = [Q]`
- `context_lens = [H + Q]`

The apparent one-position offset is intentional:

- the pending input token at sequence index `H - 1` was processed by the
  first-target decode at model position `H`;
- `d[0]`, if accepted, becomes the next pending input and is processed at
  model position `H + 1`;
- its KV is stored at the logical cache slot beginning at `H`;
- its output logits predict `d[1]`.

For the current example with `H=52` and
`draft=[10,20,30,40]`:

- `input_ids=[10,20,30]`
- logical slot positions `[52,53,54]`
- model positions `[53,54,55]`
- `context_lens=[55]`
- verifier logits predict target tokens for `[20,30,40]`.

The output logits shape is `[Q, vocab_size]`. No logits row may be discarded
inside `ModelRunner`; acceptance needs every row.

## Attention Semantics

### Required Causal View

For a verifier query of length `Q`, `flash_attn_with_kvcache` must see:

- the existing prefix KV;
- the `Q` new K/V rows written to their final slots;
- `cache_seqlens = H + Q`;
- `q` shaped `[1, Q, num_heads, head_dim]`;
- one paged block-table row;
- `causal=True`.

For query row `j`, zero-indexed within the verifier, attention may read:

- every existing prefix token;
- verifier K/V rows `0..j`;
- no verifier K/V row greater than `j`.

This is the decode-equivalent causal invariant. It must be tested directly
against the row-expanded oracle rather than inferred only from final tokens.

### KV Write Ordering

`Attention.forward()` currently writes K/V before dispatching the attention
kernel. The native path retains that ordering:

1. project and apply RoPE using the verifier positions;
2. write all `Q` K/V rows to `slot_mapping`;
3. execute one causal multi-query KV-cache attention;
4. return all `Q` hidden/logit rows.

Rejected candidate KV may remain in unpublished slots or unused portions of
the current block. Correctness depends on metadata visibility, not on clearing
those bytes. A later accepted or normal decode write must overwrite any slot
before it becomes visible.

### Kernel Capability Check

The implementation must include a focused runtime capability smoke on the
remote FlashAttention build:

- compare `Q in {1,3,7,15}` multi-query outputs with serialized single-query
  decode;
- use the same prefix KV, positions, slots, and dtype;
- compare logits and written K/V with explicit tolerances;
- classify unsupported or divergent `seqlen_q > 1` behavior as
  `INCOMPLETE`, not as a verifier failure to optimize around.

No custom Triton kernel is authorized in the first implementation plan.

## ModelRunner Design

### Preparation Boundary

Add a focused preparation function for one sequence, conceptually:

`prepare_spec_verify(seq, input_tokens, proxy_block_table, slot_positions)`

It owns:

- argument validation;
- consecutive model positions;
- final-slot mapping;
- one-row block-table construction;
- query/context length tensors;
- `spec_verify` context installation.

The profiler must stop constructing raw context tensors itself once this API
exists. That centralizes the position and slot contract next to
`prepare_decode()`.

### Forward Boundary

`run_model()` will accept or derive the explicit execution mode.
`spec_verify` always uses eager execution in phase one, independent of
ordinary decode CUDA graph availability.

The model and logits head remain unchanged. `Attention.forward()` is the only
model-layer component that needs a new dispatch branch.

### Result Boundary

The verifier preparation/forward API returns:

- all tail logits in query order;
- optional hidden states only for existing profiler debug features;
- a structured metadata record with `query_len`, positions, logical slots,
  physical slots, context length, and block table.

It does not sample, mutate `Sequence`, or publish reserved blocks.

## Verify and Commit Flow

For a draft of length `K`:

1. Validate supported mode and draft length.
2. Reserve append blocks for `K` tokens.
3. Run the existing normal decode once to obtain `first_target`.
4. Build `proxy_block_table`.
5. If `K > 1`, run the native tail verifier for `draft[:-1]`.
6. Form `target_tokens = [first_target] + tail_argmax_tokens`.
7. Count the accepted prefix.
8. Truncate at EOS and remaining output budget.
9. Commit accepted tokens with `commit_accepted_tokens()`.
10. Release all unused reserved blocks.
11. Run the existing finish/deallocation check.
12. Reset context in a `finally` block.

The following old action is removed from the native path:

- `rematerialize_accepted_kv()`

The profiler may retain the function temporarily for legacy artifact replay,
but native-verifier events must report:

- `accepted_kv_rematerialization.decode_calls = 0`
- `accepted_kv_rematerialization.rematerialized_tokens = []`
- `timing_ms.accepted_kv_rematerialize_ms = 0`

These are acceptance invariants, not merely expected performance.

## Acceptance and Pending-Token Semantics

If `A` draft tokens are accepted:

- sequence length becomes `H + A`;
- KV is materialized only through sequence index `H + A - 2`;
- the token at index `H + A - 1` remains pending;
- `BlockManager.commit_accepted_tokens()` publishes only full blocks covered
  by `materialized_tokens = H + A - 1`;
- the next normal decode consumes the final accepted token and writes its KV.

Consequences:

- zero acceptance publishes nothing and releases every reserved block;
- one accepted token requires no tail KV and is valid with `Q=0`;
- partial acceptance reuses only the accepted prefix of already written tail
  KV;
- full acceptance still leaves the last draft token pending;
- EOS may be pending because no later token needs to attend to it;
- output-budget truncation uses the same rule as EOS.

No new sequence metadata field is required in phase one.

## Row-Expanded Decode Oracle

### Purpose

The oracle answers whether the native verifier is equivalent to established
single-token decode, independently of final output-token agreement.

It is test-only and must not share mutated candidate slots with the native
path.

### Construction

For each tail query `j in [0,Q)`:

1. Clone the pre-verification sequence metadata.
2. Use an isolated block allocation or a fresh model/cache fixture.
3. Append verifier inputs through `d[j]` in order.
4. Run normal `prepare_decode()`/decode attention for each token.
5. Capture the logit row and the K/V written for the corresponding position.

The oracle produces:

- `Q` target logit rows;
- K/V tensors for every layer and verifier slot;
- the accepted prefix under the same greedy acceptance function;
- at least 16 subsequent normal-decode tokens and their logits/KV.

For the real engine block size of `256` and phase-one `K <= 16`, one verifier
event can cross into at most one new block. Remote block-boundary evidence
therefore covers current-block, one-new-block, and multi-block visible-context
cases. Multiple newly reserved blocks are a lifecycle-only test dimension
using a smaller dependency-light block size; remote artifacts must not claim
that physically unreachable case.

Native and oracle runs must start from identical prefix state.

### Comparison

The exactness gate requires:

- identical argmax target tokens for every verifier row;
- identical accepted prefix;
- identical committed sequence metadata and block-table visibility;
- identical subsequent greedy tokens for at least 16 decode steps;
- finite logits and KV everywhere;
- numerical comparison of logits and K/V using fixed dtype-specific
  tolerances recorded in the test artifact.

Token equality is the hard correctness gate. Numerical tolerances diagnose
kernel drift but cannot excuse a token mismatch.

## Failure Handling and Rollback

Before the tail forward, the verifier owns `reserved_block_ids`.

On any preparation, forward, acceptance, or commit exception:

- release every still-owned reserved block;
- do not append tokens to `Sequence`;
- do not publish prefix hashes;
- reset global context;
- preserve the original block table;
- report the phase and exception.

The first-target decode writes only the normal pending-token slot and follows
the established decode lifecycle. It is not rolled back.

After tail KV has been written but before commit, failure may leave stale
bytes in unpublished slots. This is allowed only if:

- metadata remains unchanged;
- all reserved blocks are released;
- future allocation overwrites slots before publication.

Tests must force failures:

- before the tail forward;
- after tail KV write and before acceptance;
- during metadata commit;
- at a boundary requiring a newly reserved block.

## Instrumentation

Each verifier event records:

- `verifier_mode`
- `query_len`
- `history_len`
- `draft_len`
- input tokens
- model positions
- logical and physical slot positions
- context length
- proxy block table
- reserved, committed, and released block IDs
- target tokens
- accepted tokens and count
- EOS and output-budget truncation flags
- `decode_calls`
- `accepted_kv_rematerialize_ms`
- prepare, target-forward, acceptance, metadata-commit, and total timing
- native-versus-oracle comparison status when enabled

The remote smoke summary also reports:

- baseline and native token streams;
- per-case exactness status;
- native verifier latency by `K`;
- old rematerializing verifier latency by `K`;
- end-to-end decode time and tokens per second;
- accepted tokens per verifier event;
- number of target forwards;
- maximum allocated GPU memory.

Memory is diagnostic only. The phase does not claim memory reduction.

## Testing Strategy

### Pure Contract Tests

Extend `tools/test_ngram_speculative.py` to cover:

- tensor-plan construction for `K in {1,4,8,16}`;
- the `H=52`, `K=4` reference contract;
- consecutive positions and slots;
- `Q=0` behavior;
- context length and visible block count;
- invalid non-consecutive or out-of-range slots;
- unsupported feature combinations failing before mutation.

### Context and Dispatch Tests

Add focused tests for:

- explicit `prefill`, `decode`, and `spec_verify` modes;
- legacy callers retaining current behavior;
- `spec_verify` selecting KV-cache attention, never varlen prefill;
- eager execution regardless of ordinary CUDA graph availability;
- every optional incompatible decode feature failing closed.

### Synthetic Attention Tests

With small deterministic tensors:

- compare native multi-query attention to serialized single-query decode;
- cover `Q in {1,3,7,15}`;
- cover one-block and cross-block queries;
- cover GQA;
- cover FP16 and BF16 when supported;
- inspect both outputs and written K/V;
- verify each query cannot attend to future verifier rows.

### Verifier Lifecycle Tests

Using fake runner/block-manager fixtures, cover:

- zero acceptance;
- acceptance of one token;
- partial acceptance;
- full acceptance;
- EOS in the accepted prefix;
- output-budget truncation;
- current-block-only writes;
- one and multiple new reserved blocks;
- exact full-block hash publication boundaries;
- unused-block release;
- exception rollback at each phase;
- no rematerialization calls.

### Model Exactness Tests

On the remote Qwen3-0.6B environment:

- compare native verifier with the row-expanded oracle;
- compare both with stable normal greedy decode;
- cover `K in {1,4,8,16}`;
- cover zero, partial, and full acceptance using deterministic draft cases;
- include EOS and block-boundary cases;
- continue normal decode for at least 16 tokens after each commit;
- require token-identical output in every case.

## Remote Smoke

GPU/model validation runs on:

- host: `sitian@10.232.195.203`
- Python:
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- model:
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`

The runner must:

- upload a source snapshot to a unique isolated remote directory;
- use a fresh dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT` for each model
  process;
- run capability and exactness cases before performance cases;
- download all artifacts;
- run a local artifact verifier;
- leave the original remote checkout untouched.

No Qwen3-8B run is required in phase one.

## Acceptance Gates

### Gate 1: Static and Unit Correctness

Required:

- all new focused tests pass;
- existing speculative, SAM, and chunked-prefill focused suites pass;
- unsupported modes fail before KV mutation;
- no native event invokes accepted-KV rematerialization.

Any failure is `INCOMPLETE`.

### Gate 2: Native Versus Oracle Exactness

For every required matrix case:

- verifier argmax tokens match;
- accepted prefixes match;
- committed metadata matches;
- subsequent 16-token greedy continuation matches;
- no NaN or infinity occurs;
- numerical comparisons remain within preregistered tolerances.

Any token, acceptance, lifecycle, or continuation mismatch is `NO_GO` for the
native path. Infrastructure or unavailable-kernel evidence is `INCOMPLETE`.

### Gate 3: Stable Baseline Exactness

Native final output must be token-identical to stable normal greedy decode for
every remote smoke case.

Any mismatch is `NO_GO`.

### Gate 4: Rematerialization Elimination

Every native event must satisfy:

- `decode_calls == 0`
- `accepted_kv_rematerialize_ms == 0`
- no accepted-token KV copy
- no accepted-token replay

Any violation is `NO_GO`.

### Gate 5: Performance Qualification

Performance is measured only after Gates 1–4 pass.

For accepted events with `K > 1`, the native path must:

- reduce median verifier-plus-commit time relative to the current exact
  rematerializing verifier;
- reduce target model forward count by exactly the removed rematerialization
  decode calls;
- show no regression greater than 1% for the `K=1` control;
- report end-to-end throughput without excluding zero-accept cases.

The first smoke is diagnostic rather than a product `GO` gate. A subsequent
written performance-gate spec must preregister prompt bank, repetitions, and
minimum speedup before any production recommendation.

## Classification

The design phase leads to one of:

- `READY_FOR_PERFORMANCE_GATE`: all exactness and elimination gates pass, and
  smoke timing shows the expected direction;
- `NO_GO`: a native semantic mismatch, lifecycle error, or inability to
  eliminate replay is demonstrated;
- `INCOMPLETE`: environment, kernel capability, artifact, or coverage is
  insufficient to decide.

`READY_FOR_PERFORMANCE_GATE` is deliberately not called `GO`. It authorizes a
separate preregistered performance experiment, not production integration.

## Scope Boundaries

### Included

- dedicated `spec_verify` context mode;
- one-sequence linear multi-query decode;
- direct writes to final current/reserved KV slots;
- metadata-only accepted-prefix commit;
- row-expanded single-token decode oracle;
- focused CPU/synthetic/GPU exactness tests;
- isolated Qwen3-0.6B remote smoke;
- reproducible artifacts and artifact verification.

### Excluded

- scheduler-owned speculative decoding;
- multiple active sequences in one verifier forward;
- ragged query lengths;
- tree candidates;
- speculative sampling or non-greedy equivalence;
- CUDA graph capture/replay;
- KV offload;
- blockwise attention;
- Quest or Attention Matching;
- KV cartridge;
- C4/C8 or other quantized KV;
- scratch KV and accepted-KV copies;
- Qwen3-8B or multi-model generalization;
- production throughput or tail-latency claims;
- claimed memory reduction.

## Claim Boundaries

A successful first phase shows only that, on the fixed remote Qwen3-0.6B
single-sequence greedy cases, TinyLLMForge can use one native multi-query
KV-cache decode forward to generate accepted-prefix KV that is equivalent to
the established single-token decode trajectory, without replaying accepted
tokens.

It does not show:

- benefit under production batching or queue pressure;
- correctness for ragged batches or tree verification;
- non-greedy distribution preservation;
- compatibility with KV offload, sparse attention, compressed attention, or
  quantized KV;
- CUDA graph compatibility;
- reduced peak or resident memory;
- generalization to other models, FlashAttention builds, or GPUs;
- a production-worthy end-to-end speedup.

## Deliverables After Written-Spec Approval

The implementation plan will map this design to:

- explicit context-mode and attention-dispatch changes;
- `ModelRunner.prepare_spec_verify()` and eager forward integration;
- profiler migration away from raw prefill context construction;
- row-expanded oracle helpers;
- contract, dispatch, attention, lifecycle, and rollback tests;
- isolated remote capability/exactness/performance-smoke runner;
- artifact verifier and structured summaries;
- README and `AGENT_HANDOFF_STATE.md` updates;
- a prompt-to-artifact completion audit before any claim.

No implementation begins until this written specification is reviewed and
approved.
