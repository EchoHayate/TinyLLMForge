# Decode Residency-Aware Read-Window Planner Design

Date: 2026-07-22

## Status

This document designs a decode-only, cross-layer residency-aware read-window
planner for the existing blockwise KV-offload attention path.

The current implementation and its existing alternating layer order remain the
authoritative baseline until the correctness and source-bound performance gates
in this document both return `GO`.

This design does not authorize an implementation performance claim, README
update, or default enablement by itself.

## Objective

Reduce repeated KV H2D reloads and clean evictions in low-capacity blockwise
decode by making one cached read-window plan express both:

1. the current layer's remaining window traversal; and
2. the next layer's likely reuse order.

The first implementation must:

1. apply only to blockwise KV-offload decode;
2. preserve the existing even-layer forward and odd-layer reverse traversal;
3. preserve exact online-softmax attention order within each selected window;
4. keep the current window's required blocks at the highest priority;
5. represent current-layer future blocks separately from cross-layer reuse
   blocks;
6. use only spare staging capacity for cross-layer reuse hints;
7. pass cross-layer reuse only as a soft eviction hint;
8. avoid proactive H2D, wider read windows, and extra waits;
9. rebuild the cached plan when its complete logical identity changes;
10. preserve current behavior when no spare staging capacity exists;
11. expose enough planning evidence to distinguish real data-movement savings
    from host-side planner-call reductions;
12. earn `GO` only when exact correctness and the strict source-bound movement
    and latency gates all pass.

## Non-Goals

This stage does not:

- change prefill planning or prefill attention;
- combine this work with Light Doc Cache, Gist KV layer sharing, token
  sparsity, low rank, Attention Matching, KV quantization, or speculative
  decoding;
- change CUDA Graph capture or replay;
- change the online-softmax equations, attention masks, GQA math, token order,
  or output sampling;
- add proactive next-window or next-layer H2D prefetch;
- increase `kv_offload_blockwise_blocks`;
- increase `kv_offload_gpu_blocks`;
- change dirty-writeback correctness or copy-stream ownership;
- add a second KV residency manager or duplicate the existing H2D/D2H copy
  coalescer;
- promote KV offload or blockwise decode from their current default-off
  experimental status;
- claim stable throughput or latency improvement from a single smoke.

## Current Evidence and Problem

The current code already has the following optimizations:

- even layers scan decode windows forward;
- odd layers scan decode windows in reverse;
- stale H2D waits are skipped;
- already-resident, non-pending windows skip redundant staging calls while
  refreshing LRU recency;
- one `ensure_resident()` call coalesces contiguous H2D and D2H copy pairs;
- decode window plans, position templates, and masks are cached per forward
  context.

The alternating traversal produced a real movement reduction at
`gpu_blocks=2`, `blockwise_blocks=1`:

```text
h2d_copies   811 -> 728
evictions    815 -> 732
copy_waits  1626 -> 1543
```

The stale-wait fix then reduced:

```text
copy_waits  1543 -> 1460
```

The resident-window fast path reduced:

```text
prefetch_plans         933 -> 737
prefetch_read_blocks   924 -> 728
```

without changing H2D copies or evictions.

The remaining issue is that `_build_blockwise_decode_window_plan()` expresses
only each layer's local forward and reverse lookahead. It does not explicitly
tell `KVOffloadMVP0` which currently resident blocks are most useful immediately
after the next layer changes traversal direction. With staging capacity larger
than the required window, the eviction scorer can therefore discard a block
that the next layer will soon revisit.

A rejected negative branch must not be repeated: immediately prefetching the
next read window after the current window preserved correctness but did not
reduce H2D copies, evictions, or waits, and increased `prefetch_plans` from
`933` to `1017`. This design adds no extra staging call and performs no
proactive load.

## Alternatives Considered

### 1. Recommended: Unified Cross-Layer Residency Hints

Build one immutable decode plan containing required blocks, same-layer future
hints, and next-layer reuse hints for both traversal directions. Feed the
bounded union of the two hint classes to the existing eviction scorer.

Advantages:

- targets the remaining reload/eviction mechanism directly;
- reuses the existing manager and copy batching;
- does not expand the correctness-critical read set;
- becomes a no-op when no spare slot exists;
- can be tested at CPU/planning level before remote execution.

Risks:

- a poor hint can preserve the wrong resident block and increase reloads;
- benefits depend on staging capacity and multi-layer access shape;
- soft hints can have no effect if the hinted blocks are not resident.

### 2. Cross-Window Batch Staging

Load multiple adjacent windows in one staging call to create larger contiguous
copy batches.

This is deferred because it changes the active resident set, can exceed tight
staging capacity, and risks repeating the rejected proactive-prefetch branch.
The current manager already coalesces copy pairs within one call.

### 3. Planner-Only Structural Refactor

Unify forward and reverse plan construction without adding cross-layer reuse
semantics.

This is insufficient as the selected optimization. It may reduce Python work,
but it cannot by itself satisfy the required H2D or eviction improvement gate.

## Decision

Implement Alternative 1 after this written design and a separate
implementation plan are approved.

No new public configuration switch is required. The behavior is contained
inside the already default-off `kv_offload_blockwise_decode` path. Canonical
baseline and candidate processes use identical runtime configuration but
different immutable source snapshots.

## Planner Contract

### Cache Identity

The per-`Context` cache stores:

```text
identity
forward plans
reverse plans
```

The identity includes all values that can change window membership, validity,
directional hints, or staging capacity:

```text
normalized logical block rows
context lengths
maximum logical row width
KV block size
configured read-window block count
current write-block set
manager GPU staging-block count
```

The identity uses immutable tuples and sorted write-block IDs. Reusing the same
`Context` with any different identity rebuilds the complete plan. A mismatch
must never silently reuse stale plans. The cache remains bounded to the current
`Context` lifecycle and does not become a process-global cache.

### Per-Window Record

Every directional window record contains:

```text
window_rows
window_lens
required_blocks
intra_layer_future_blocks
cross_layer_reuse_blocks
max_window_tokens
```

`required_blocks` is the stable first-seen unique order of readable logical
blocks in the current window. Existing row shape and `window_lens` continue to
drive physical-slot mapping and masks.

`intra_layer_future_blocks` is the existing direction-aware lookahead used by
the current layer. Forward traversal looks toward larger window indices;
reverse traversal looks toward smaller window indices.

`cross_layer_reuse_blocks` is ordered from the next layer's traversal frontier.
For a forward current layer, the next layer is reverse; for a reverse current
layer, the next layer is forward. The current required blocks and write blocks
are excluded from this list.

At a layer boundary, this definition naturally favors the window that follows
the shared boundary window in the next layer. For example, while staging the
last forward window, the cross-layer candidates begin with the preceding
window because the odd layer will consume the last window first and then move
backward.

### Spare-Capacity Rule

For each window:

```text
spare_capacity =
    gpu_blocks
    - unique(required_blocks union write_blocks)
```

If `spare_capacity <= 0`, `cross_layer_reuse_blocks` is empty and behavior
reduces to the current alternating planner.

Otherwise, at most `spare_capacity` unique cross-layer candidates are retained
in next-layer traversal order. Candidates already in required blocks or write
blocks do not consume the budget. Duplicate candidates across batch rows are
deduplicated in stable first-seen order.

The budget is deliberately local to the current staging call. It does not
promise that every hinted block is resident and does not reserve a physical
slot.

## Staging and Eviction Semantics

`_stage_blockwise_read_window()` continues to load and wait for only:

```text
required_blocks
```

The manager call receives:

```text
future_logical_blocks =
    required_blocks
    union intra_layer_future_blocks
    union cross_layer_reuse_blocks

protected_logical_blocks =
    current write blocks only
```

Cross-layer reuse hints:

- never enter the `logical_blocks` load list;
- never enter `protected_logical_blocks`;
- never enter `pending_wait_blocks`;
- never trigger `wait_for_blocks()`;
- never increase `prefetch_plans`;
- never increase `prefetch_read_blocks`;
- never make an unreadable CPU block readable;
- never suppress required dirty writeback;
- affect only `_victim_score()` through the existing future-reuse penalty.

The required read window remains the hard correctness set. The hint is soft:
if every candidate slot has a future penalty, the existing score and LRU
ordering still select a victim. Capacity errors remain based on required and
protected blocks, not hint count.

The resident-window fast path remains valid. It touches required resident slots
and performs no manager staging or wait call. It does not load or touch a
cross-layer hint merely because that hint appears in the plan.

## Decode Data Flow

For each blockwise decode forward:

1. normalize logical block rows and compute the complete cache identity;
2. reuse the cached plan only on exact identity equality;
3. otherwise construct forward and reverse records together and replace the
   context-local cache;
4. choose forward records for even layers and reverse records for odd layers;
5. mark current write blocks dirty using existing semantics;
6. stage and wait only for the current record's required blocks;
7. map required logical blocks to physical staging slots;
8. gather K/V for the current window;
9. compute the existing exact score, mask, and online-softmax merge;
10. continue until all visible windows have contributed.

No attention window is skipped, duplicated, or reordered relative to the
already established even/odd traversal. The only new behavior is the
eviction-score hint attached to an existing staging call.

## Error Handling and Fail-Closed Behavior

The implementation must:

- preserve the existing error when one required window exceeds GPU staging
  capacity;
- rebuild rather than reuse on cache identity mismatch;
- reject malformed row/context-length combinations through existing runtime
  checks;
- not catch or downgrade unreadable logical-block, capacity, copy, CUDA, or
  attention correctness errors;
- keep `layer_idx < 0` on the current forward compatibility path;
- treat empty cross-layer capacity as a normal no-op;
- preserve current default behavior when blockwise KV-offload decode is
  disabled.

No fallback may silently switch to approximate attention or omit KV blocks.

## Telemetry

Existing manager counters remain authoritative:

```text
h2d_copies
h2d_bytes
h2d_batches
h2d_batch_spans
d2h_copies
d2h_bytes
d2h_batches
d2h_batch_spans
evictions
evict_clean
evict_dirty
copy_waits
prefetch_plans
prefetch_read_blocks
prefetch_write_blocks
resident_blocks
```

The validation harness additionally records planner evidence sufficient to
audit:

```text
cache builds
cache hits
identity invalidations
windows with positive spare capacity
cross-layer hint blocks emitted
cross-layer hint blocks already resident at staging time
cross-layer hinted blocks retained across a layer boundary
```

These planner counters are diagnostic only. A reduction in planner calls or an
increase in hint hits cannot produce `GO` unless the required movement gate
also passes.

## Local Test Strategy

### Planning Tests

Extend `tools/test_blockwise_attention_planning.py` to cover:

1. forward records emit reverse-frontier cross-layer candidates;
2. reverse records emit forward-frontier cross-layer candidates;
3. stable deduplication across multiple block-table rows;
4. cross-layer hints exclude required and write blocks;
5. cross-layer hint count is bounded by exact spare capacity;
6. zero spare capacity emits no cross-layer hints;
7. zero spare capacity preserves current alternating staging behavior;
8. cross-layer hints enter `future_logical_blocks` only;
9. cross-layer hints never enter load, protected, or wait arguments;
10. resident-window fast path does not touch or load hinted-only blocks;
11. changing any cache identity component rebuilds the plan;
12. exact identity reuse does not rebuild the plan;
13. `layer_idx < 0` retains forward traversal;
14. a multi-layer manager simulation reduces or preserves H2D and eviction
    counters without changing the required window sequence.

### Manager Regression Tests

Run and, where needed, extend `tools/test_kv_offload.py` to prove:

- future-only blocks are not loaded;
- future-only blocks are not protected from all eviction;
- pending-wait membership is unchanged for hinted-only blocks;
- dirty evictions and deferred D2H waits retain current behavior;
- copy-pair coalescing retains current behavior;
- stale and shared-event wait fixes remain covered.

### Existing Regressions

The focused local/remote regression set is:

```text
tools/test_blockwise_attention_planning.py
tools/test_kv_offload.py
tools/test_chunked_prefill.py
tools/test_ngram_speculative.py
```

Static validation includes Python compilation for changed Python files and
`git diff --check`.

## Remote Validation Contract

All GPU/model execution occurs on:

```text
host:         sitian@10.232.195.203
control path: /tmp/ssh-sitian-10.232.195.203
python:       /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model:        /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
GPU:          CUDA_VISIBLE_DEVICES=0
```

Every model process uses fresh, mutually distinct dynamic values for
`TINYVLLM_DIST_PORT` and `MASTER_PORT`. Only `EADDRINUSE` may be retried.
Validation must not kill unrelated processes, switch GPUs, modify the remote
checkout, use `rsync`, or clean shared `/tmp`.

Source is staged as an immutable snapshot. Every result records local commit,
tracked-tree hash, staged-source hash, dirty status, command, environment, GPU,
model path, Python path, and output file hashes.

## Correctness Gate

The candidate must pass all of:

1. the existing blockwise online-softmax mathematical smoke;
2. exact generated token equality against the paired baseline;
3. existing strict logits tolerance for every compared decode step;
4. single-prompt long-context decode;
5. multi-prompt/request thrash decode;
6. all four staging shapes:

```text
gpu_blocks=2, blockwise_blocks=1
gpu_blocks=2, blockwise_blocks=2
gpu_blocks=4, blockwise_blocks=1
gpu_blocks=4, blockwise_blocks=2
```

An invalid shape or inability to execute one frozen case is a gate failure,
not permission to drop the case after seeing results.

## Performance Gate

For each frozen workload and staging shape, baseline and candidate use the same
model, prompts, decode length, GPU, process settings, and runtime
configuration. Warmup is excluded identically. Each measured case runs at
least five stable repetitions per source snapshot.

The top-level movement gate requires:

```text
H2D copies improve by at least 5%
OR
evictions improve by at least 5%
```

and simultaneously:

```text
the other H2D/eviction metric worsens by no more than 1%
copy_waits do not worsen
prefetch_plans do not worsen
D2H copies and bytes do not worsen
dirty evictions/writeback do not worsen
peak staging blocks and peak CUDA memory do not worsen
median decode latency worsens by no more than 2%
```

Ratios are computed from independently parsed raw artifacts, not copied summary
labels. Counter comparisons use exact integer totals over the same decoded
token count. Latency uses the median of measured repetitions after verifying
that no repetition is missing or failed.

At least one low-capacity case and the multi-prompt thrash workload must satisfy
the `>=5%` movement improvement. Aggregate improvement cannot hide regression
outside the stated tolerance in another frozen staging shape.

If only planner counters improve, or if latency improves without the movement
gate, classification is `NO_GO`.

## Classification

The final classification is exactly one of:

```text
GO
NO_GO
INVALID
```

- `GO`: source identity, correctness, coverage, movement, regression, memory,
  and latency gates all pass.
- `NO_GO`: evidence is valid and correctness passes, but any required
  performance threshold fails.
- `INVALID`: source identity, required case coverage, raw evidence, process
  isolation, correctness, or verifier reconstruction is incomplete or
  inconsistent.

The implementation may remain as a tested internal optimization after
`NO_GO` only if it is behaviorally neutral and the user explicitly approves
keeping it. Otherwise it is reverted. `INVALID` evidence cannot support any
performance conclusion.

## Evidence Layout

The implementation plan will define one canonical artifact directory under:

```text
experiments/kv_offload/
```

It must contain:

```text
spec and implementation-plan references
frozen workload/config manifest
baseline and candidate source identities
staged source snapshots or complete source manifests
all commands and environment fields
raw stdout/stderr logs
raw per-run JSON
token and logits correctness records
per-run KV and memory counters
per-run decode latency records
independently recomputed comparison report
top-level classification
file hashes
```

The verifier must fail closed on missing cases, duplicate case identities,
unexpected extra cases, inconsistent decoded-token counts, source mismatch,
non-finite metrics, or summary/raw disagreement.

## Documentation and Claim Policy

After execution:

- record implementation, tests, commands, raw artifact path, source identity,
  negative branches, limitations, and classification in
  `AGENT_HANDOFF_STATE.md`;
- add the experiment to the repository test/evidence registry used by the
  implementation plan;
- update README or user-facing performance documentation only after `GO`;
- if classification is `NO_GO` or `INVALID`, state that plainly and do not
  claim that the inference engine became faster or more memory-efficient.

## Implementation Boundaries

Expected implementation files are:

```text
tinyvllm/layers/attention.py
tools/test_blockwise_attention_planning.py
tools/test_kv_offload.py
```

The implementation plan may add focused KV-offload experiment runners,
verifiers, registry entries, and evidence files. Changes to prefill semantics,
the scheduler, speculative decoding, CUDA Graph logic, quantization, Light Doc
Cache, or public defaults are out of scope and require a new design review.

## Acceptance Summary

This design is accepted for implementation planning only when:

1. cross-layer information remains a bounded soft eviction hint;
2. required block loading, waits, attention math, and traversal remain exact;
3. zero spare capacity is behaviorally equivalent to the current planner;
4. cache identity invalidation is explicit and complete;
5. remote validation uses only the approved host, user, GPU, Python, model,
   dynamic ports, and immutable staged source;
6. real H2D or eviction reduction, not planner-call reduction, determines
   `GO`;
7. all evidence is source-bound, independently reconstructable, and landed
   before any performance claim.
