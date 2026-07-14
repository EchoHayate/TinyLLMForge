# Prefix Cache Correctness and Performance Gate Design

Date: 2026-07-14

## Objective

Turn TinyLLMForge's existing hash-based prefix cache into a correctness-safe,
measurable cross-request optimization before considering a radix tree,
cache-aware scheduling, or more aggressive same-batch reuse.

The first phase must:

1. Prevent reuse of KV blocks whose prefill forward has not completed.
2. Guarantee that every sampled prefill row has a valid query token and logits.
3. Preserve exact greedy output and bounded logits agreement across cold, warm,
   and cache-cleared execution.
4. Measure real prefill-token and TTFT reduction for shared prefixes.
5. Produce an explicit go/no-go decision for later APC architecture work.

This phase optimizes prompt prefill and TTFT. It does not claim decode
acceleration or physical KV-memory reduction.

## Motivation

TinyLLMForge already hashes full KV blocks and reuses matching blocks across
requests. Non-chunked scheduling currently publishes a newly allocated full
block to `hash_to_block_id` before the model forward computes its KV. A later
request admitted into the same batch can therefore treat that block as cached.

Industry engines may support same-batch prefix sharing, but they also have
execution data flow that preserves producer/consumer ordering and produces one
valid logits row per request. TinyLLMForge does not currently provide those
semantics.

Remote Qwen3-0.6B validation on 2026-07-14 established two correctness
failures with the default 256-token block size:

### Same-Batch Full-Hit Logits Corruption

For one batch containing prompts `[P, Q, P]`, where both `P` and `Q` are
exactly 256 tokens:

- row 0: `P`, `num_cached_tokens=0`, `query_tokens=256`;
- row 1: `Q`, `num_cached_tokens=0`, `query_tokens=256`;
- row 2: `P`, `num_cached_tokens=256`, `query_tokens=0`.

`ParallelLMHead` derives default row indices as
`cu_seqlens_q[1:] - 1`. The zero-query third row therefore indexes `-1`,
which selects the final hidden state from row 1. Captured full-vocabulary
logits showed:

- row 2 versus batch row 1: `max_abs=0.0`, `mean_abs=0.0`;
- row 2 versus the independent `P` baseline:
  `max_abs=7.626953125`, `mean_abs=1.1561013460159302`.

The third request silently receives `Q` logits instead of `P` logits.

### Cross-Batch Exact-Block Warm-Hit Crash

After a cold 256-token `P` request completed, submitting the same prompt again
produced:

- `num_cached_tokens=256`;
- `prefill_chunk_start=256`;
- `prefill_chunk_end=256`;
- `query_tokens=0`.

The model received an empty prefill input and failed in rotary embedding while
reshaping a zero-element query tensor. Therefore exact-block warm reuse is
also unsafe even without same-batch publication.

These results make prefix-cache correctness a P0 prerequisite for any
performance claim.

## Decision

Use conservative compute-complete publication and require one uncached query
token per sampled prefill request.

### Publication Rule

New full blocks are not inserted into the reusable hash index during
allocation. A block becomes reusable only after its KV slots have been
successfully computed by a completed prefill forward.

Non-chunked prefill will follow the same publication lifecycle already used by
chunked prefill:

1. Allocate blocks without publishing newly allocated hashes.
2. Execute the prefill forward.
3. Publish every newly completed full block during scheduler postprocessing.

Existing valid cache hits may still restore their hash and token metadata
during allocation. Only uncomputed new blocks are withheld.

### Sampleable-Suffix Rule

For a prefill step that must sample the next token, prefix lookup may consume
at most:

```text
floor((prompt_tokens - 1) / block_size) * block_size
```

tokens.

This leaves at least one real query token in the forward and guarantees that
the LM head receives a valid per-request hidden state.

Examples for a 256-token block:

| Prompt tokens | Maximum reusable tokens | Query tokens |
|---:|---:|---:|
| 255 | 0 | 255 |
| 256 | 0 | 256 |
| 257 | 256 | 1 |
| 512 | 256 | 256 |
| 513 | 512 | 1 |

The initial implementation will not cache the final hidden state or logits.
Consequently, an exactly block-aligned prompt recomputes its final 256-token
block. This is an intentional correctness-first trade-off.

### Same-Batch Rule

Requests in one scheduler batch may reuse blocks that were computed before the
batch began. They may not reuse blocks first produced by another request in
that same forward.

Same-batch producer/consumer execution waves are excluded from this phase.
They may be considered later only if cross-request APC passes the gate and
same-batch reuse represents a material workload opportunity.

## Alternatives Considered

### 1. Same-Batch Dependency Waves

Split one logical batch into producer and consumer waves. Prefix producers
compute and publish KV first; dependent requests execute afterward.

This preserves same-batch sharing but adds dependency analysis, multiple model
forwards, and scheduling complexity. It is not justified before basic
cross-request APC is proven useful.

### 2. Cache Final Hidden State or Logits

Store enough terminal state to sample a fully cached exact-block prompt
without executing a query token.

This can recover zero-prefill exact hits but introduces another cache with
model, adapter, dtype, sampling, lifecycle, and eviction validity concerns.
It is excluded from the initial gate.

### 3. Keep Immediate Publication and Patch Only LM-Head Indexing

Explicit logits indices could prevent one zero-query row from selecting the
previous row, but a zero-query request still has no newly computed hidden
state. It also does not establish safe producer/consumer KV ordering within
one forward. This treats a symptom rather than the execution invariant.

## Scope

### Included

- Delayed publication for non-chunked prefill.
- A prefix-match cap that leaves one sampleable query token.
- Correct post-forward publication of newly computed full blocks.
- CPU scheduler/block-manager regression tests.
- Remote Qwen3-0.6B logits and output correctness tests.
- A reusable APC benchmark supporting cold, warm, and cache-cleared runs.
- Metrics for cached tokens, executed prefill tokens, prefill latency, and
  TTFT.
- Cache lifecycle tests covering reuse, deallocation, collision defense, and
  eviction/reclamation behavior already present in the block manager.
- README and handoff updates after the gate result is known.

### Excluded

- Radix-tree replacement.
- Cache-aware request routing or scheduling.
- Same-batch producer/consumer execution waves.
- Final-hidden-state or logits caching.
- Partial-block prefix reuse.
- Decode throughput claims.
- Logical accounting presented as physical GPU-memory savings.
- Distributed cache sharing across workers or hosts.

## Components

### 1. Block-Manager Prefix Lookup and Publication

`tinyvllm/engine/block_manager.py` remains responsible for:

- chained full-block hashing;
- hash plus exact-token collision defense;
- reference counting;
- allocation and deallocation;
- publishing full blocks whose KV has completed.

Prefix matching must accept or derive a maximum reusable token count. It must
stop before the final prompt block when consuming it would leave no query
token.

`allocate(seq, publish_hashes=False)` remains the allocation primitive for
prefill. `commit_prefill(seq, old_end, new_end)` is the publication primitive
after successful computation.

The implementation must not publish a block if the corresponding prefill
range did not complete.

### 2. Scheduler Lifecycle

`tinyvllm/engine/scheduler.py` applies one lifecycle for both non-chunked and
chunked prefill:

1. allocate without new-hash publication;
2. record `prefill_chunk_start` and `prefill_chunk_end`;
3. run the model;
4. commit completed full blocks in postprocessing;
5. append the sampled token or return the sequence to prefill scheduling.

For non-chunked prefill, postprocessing must commit the prompt range before
normal token append/deallocation handling.

The scheduler must ensure every sequence in a sampled prefill batch has:

```text
prefill_chunk_end > prefill_chunk_start
```

### 3. Model-Runner Invariant

`tinyvllm/engine/model_runner.py` is not the primary repair location. The
scheduler and block manager must prevent sampled zero-query rows from reaching
`prepare_prefill`.

The gate adds an explicit invariant check close to prefill preparation or
scheduling so future regressions fail with a clear error rather than silently
indexing another row's hidden state.

The pre-existing unrelated modifications in this file must remain isolated
from APC commits.

### 4. Correctness Probe

A focused remote probe will:

- use Qwen3-0.6B;
- use greedy sampling;
- capture full-vocabulary logits before sampling;
- record `num_cached_tokens`, chunk boundaries, query tokens, and block tables;
- compare independent cold, warm, same-batch, and cache-cleared cases.

Required cases:

1. 255-token prompt repeated.
2. 256-token prompt repeated.
3. 257-token prompt repeated.
4. 512-token prompt repeated.
5. 513-token prompt repeated.
6. Same-batch `[P, Q, P]`.
7. Two prompts sharing one or more full blocks with different suffixes.
8. Hash collision simulation where hash matches but token IDs differ.
9. Cache-cleared rerun matching cold output.

### 5. Performance Benchmark

A reusable benchmark under `tools/` will run shared-prefix prompt pairs or
small batches with:

- 256-token shared prefix;
- 1024-token shared prefix;
- 2048-token or longer shared prefix;
- different suffixes so every request has a valid query path.

Each case runs in three states:

1. cold: no reusable prefix;
2. warm: prefix produced by a completed earlier request;
3. cache-cleared: reusable metadata cleared while preserving engine state.

The benchmark uses CUDA synchronization around measured model steps and emits
machine-readable JSON plus a human-readable summary.

## Correctness Requirements

All required cases must satisfy:

- no exception;
- no sampled zero-query row;
- greedy token IDs exactly match the corresponding cold baseline;
- decoded text exactly matches the corresponding cold baseline;
- warm and cache-cleared logits retain the same argmax as cold;
- full-vocabulary logits meet both:
  - maximum absolute difference no greater than `0.25`;
  - mean absolute difference no greater than `0.05`.

The numeric tolerances allow normal batching and kernel-order differences
observed in the remote baseline while remaining far below the reproduced
wrong-row error.

For `[P, Q, P]`, the third row must match the independent `P` baseline and
must not be identical to the `Q` row unless the independent `P` and `Q`
baselines are themselves identical within the same tolerance. Test prompts
must be chosen so their reference logits are distinguishable.

## Cache Lifecycle Requirements

The gate must verify:

- only complete blocks are reusable;
- uncomputed blocks never enter `hash_to_block_id`;
- exact-token comparison rejects synthetic hash collisions;
- live shared blocks maintain positive reference counts;
- deallocation preserves reusable metadata only for blocks no longer live;
- reallocation of an idle cached block restores it safely;
- cache clearing does not mutate live blocks;
- capacity pressure does not return a block still referenced by a live
  sequence;
- publication order preserves chained parent hashes.

This phase does not require a new LRU policy. If the existing free-block queue
cannot satisfy the lifecycle requirements under pressure, the gate result is
no-go and an eviction-policy design becomes a prerequisite.

## Performance Metrics

Every measured row records:

- model and source revision;
- block size;
- prompt length, shared-prefix length, and suffix length;
- cold, warm, or cache-cleared state;
- scheduler-reported cached tokens;
- actual prefill query tokens;
- prefill model-forward latency;
- request TTFT;
- first generated token;
- correctness status;
- median, minimum, maximum, and sample count across repetitions.

Cached-token accounting is not accepted as performance evidence by itself.
The benchmark must show a corresponding reduction in actual query tokens and
wall-clock prefill/TTFT.

## Go/No-Go Decision

The APC gate is **GO** only if all correctness and lifecycle cases pass and:

- for 1024-token and 2048-token-or-longer shared prefixes, warm median TTFT is
  at least 20% lower than the corresponding cold median;
- actual warm prefill query-token count decreases by the expected number of
  reusable full blocks;
- no measured case has a warm median TTFT regression greater than 5%;
- outputs remain correct for every performance sample.

If correctness passes but performance does not, retain the safety fixes but
do not begin radix-tree or cache-aware scheduling work.

If correctness fails, mark the APC gate **NO_GO** and do not publish any
performance claim.

If the gate is GO, the next design compares:

1. the current chained-hash lookup;
2. radix-tree longest-prefix matching;
3. cache-aware scheduling or request routing;
4. optional same-batch dependency waves.

## Validation Environment

Primary remote validation:

- host: `sitian@10.232.195.203`;
- repository:
  `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`;
- Python:
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`;
- model:
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`;
- block size: 256 tokens;
- greedy sampling with `temperature=0.0`;
- dynamic, matching `TINYVLLM_DIST_PORT` and `MASTER_PORT`;
- CUDA synchronization around timing boundaries.

The validation report must record hashes of the relevant remote source files
because the remote checkout may not be a Git worktree and may differ from the
local working tree.

## Deliverables

Expected implementation-phase artifacts:

- focused CPU tests for allocation, publication, query-token capping, and
  lifecycle behavior;
- a remote correctness probe with JSON output;
- an APC performance benchmark with JSON output;
- one canonical experiment directory containing raw results and summary;
- an explicit `GO` or `NO_GO` decision;
- README usage updates;
- `AGENT_HANDOFF_STATE.md` results, limitations, and next steps.

Implementation commits must not include the pre-existing unrelated changes in
`tinyvllm/engine/model_runner.py` or `AGENT_HANDOFF_STATE.md`. Any required
edits to those files must be isolated and staged by exact hunk or exact file
after verifying the diff.

## Claim Boundaries

A passing gate proves that TinyLLMForge can safely skip previously computed
full-block prompt KV across requests and reduce prefill work and TTFT on the
measured workload.

It does not prove:

- decode-token acceleration;
- reduced physical KV allocation;
- reduced model-weight memory;
- optimal cache eviction;
- optimal scheduling under multi-tenant load;
- benefit for prompts without a full-block shared prefix;
- production-scale distributed APC behavior.
