# Phase 1 Promotion Checklist

**Date:** 2026-08-14

**Repository:** `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`

**Decision:** `PHASE_1=NOT_ACHIEVED`
**Promotion:** `NOT_PROMOTABLE`

## Purpose

This is the compact, artifact-first Phase 1 promotion checklist. It
supersedes stale intermediate conclusions in longer audits, but it does not
replace the retained raw artifacts or their independent verifiers.

Evidence is classified with these rules:

1. a completed, source-bound loaded-model artifact is authority only for its
   recorded model, TP, context, batch, runtime, and feature configuration;
2. local code and tests may establish an implementation contract, but they do
   not establish real-checkpoint correctness or performance;
3. a failed authority remains failed even if adjacent cells pass;
4. simulated KV movement is never accepted as real KV-offload evidence; and
5. no partial row may be used to claim Phase 1 completion or production
   readiness.

## Frozen Exit Criteria

Phase 1 requires one generic speculative runtime with transactional KV
ownership that:

1. serves at least two materially different model structures;
2. covers TP1 and TP4;
3. covers 4K, 16K, and 32K contexts;
4. covers batch 1, batch 4, and real multi-sequence execution;
5. preserves exact greedy output parity;
6. reports TTFT, TPOT, throughput, peak memory, acceptance, and real KV
   H2D/D2H movement;
7. proves accepted proposal KV is committed in place and rejected suffix KV
   is released without accepted-prefix target replay;
8. includes a real learned proposal source beyond model-free n-gram/SAM; and
9. keeps all unsupported offload, precision, graph, and sampling combinations
   fail closed.

The generic n-gram matrix now closes many of the original infrastructure and
coverage gates. It does not automatically promote native MTP, an independent
learned drafter, proposal-KV offload, KV4/KV8 plus offload, or the whole
Phase 1 objective.

## Executive Status

| Evidence family | Established | Still missing or failed |
| --- | --- | --- |
| Generic source-neutral runtime | Batch-native first-target and fixed-Q verification, per-sequence transactional KV ownership, accepted-prefix direct commit, rejected-suffix rollback, Scheduler metadata commit, source-neutral proposal executor registration | Full promotion remains blocked by learned-source, precision-tier, and broader controlled-performance gaps |
| Generic n-gram authorities | Two target model structures; TP1 and TP4; 4K, 16K, and 32K; batch 1 and 4; exact greedy parity; transactional cleanup; real KV movement in recorded scopes | No universal 32K performance authority, no KV4/KV8 plus offload authority, no production-readiness claim |
| Native Qwen3.5 MTP | TP1/4K, TP4/4K, and TP4/16K target-KV-offload exact-parity authorities | TP4/32K batch-1 parity failed; controlled native-MTP performance, proposal-KV offload, and 32K authority are missing |
| Independent Qwen3 drafter | Concrete Qwen3 backend, physical proposal-KV ownership, TP1 gate harness, and locally verified TP4 sharded executor contract | No real Qwen3-draft/Qwen3.5-target TP1 or TP4 loaded-checkpoint authority; no long-context, offload, or performance artifact |
| CUDA Graph | Legacy Qwen3.5 TP1/no-offload artifact records `Q=(1,2,3,4)`, batch `(1,4)`, graph/eager parity booleans, 6 captures, 12 replays, and a complete transaction-case domain; a separate exact `(B,Q,W)` gate has a strong local semantic verifier contract | The retained legacy artifact remains non-reconstructable beyond transaction sets; the exact-family gate has no PASS artifact or archived-verifier receipt; no TP4, long-context, offload, or performance claim |
| KV4/KV8 | Independent storage/configuration paths exist for `kv_quant_bits in (0,4,8)`; fresh CPU reference KV4 and CPU-reference-plus-actual-dequant KV8 round-trips cover group sizes 32/64/128; local KV8 cached-prefill routing and verifier fail-closed tests pass | No authorized Triton store-kernel round-trip, loaded parity, memory, performance, retained execution receipt, or unified precision/residency authority |

## 1. Generic N-Gram Promotion Evidence

The following generic n-gram evidence is established within the exact scopes
of the retained artifacts.

| Gate | Authority | Classification |
| --- | --- | --- |
| Qwen3 TP4/4K exact parity | `artifacts/generic_speculative_tp4/tp4-opaque-48d18e4aba16756d/authority/result.json` | `b1=true`, `b4=true`; artifact remains `NOT_PROMOTABLE` because its scope is incomplete |
| Qwen3.5 TP1 nominal-4K exact parity | `artifacts/qwen35_generic_speculative_tp1/opaque-d4e74cb46fccbc57319c3c4f/artifacts/authority/result.json` | `SECOND_MODEL_TP1_ESTABLISHED`; raw prompt rows contain 4048 tokens, so this cell must not be described as an exact 4096-token prompt |
| Qwen3.5 TP4/4K exact parity | `artifacts/qwen35_generic_speculative_tp4/opaque-24f8ae471a2ba439ecb5a3b1/artifacts/authority/result.json` | `SECOND_MODEL_TP4_4K_ESTABLISHED` |
| Qwen3.5 TP4/16K exact parity | `artifacts/qwen35_generic_speculative_tp4_16k/opaque-3b8050a916f037bc92412ea5/artifacts/authority/result.json` | `SECOND_MODEL_TP4_16K_ESTABLISHED` |
| Qwen3.5 TP4/32K exact parity | `artifacts/qwen35_generic_speculative_tp4_32k/opaque-03a0a96654a14441b314800f/artifacts/authority/result.json` | `SECOND_MODEL_TP4_32K_ESTABLISHED` |
| Qwen3 TP1/4K controlled performance | `artifacts/speculative_runtime_performance/20260812T085852Z/result.json` | Exact batch-1/4 parity, positive TPOT/throughput direction, real MVP-0 movement; still `NOT_PROMOTABLE` |
| Qwen3.5 TP4/16K controlled performance | `artifacts/qwen35_generic_speculative_tp4_16k_performance/opaque-c9807d19e6402acc22d4a615/artifacts/authority/result.json` | `SECOND_MODEL_TP4_16K_PERFORMANCE_MEASURED`, `campaign_direction=POSITIVE` |

Fresh payload-level recomputation from raw prompt/output rows, rather than the
top-level parity booleans, confirms:

```text
Qwen3 generic TP4 nominal 4K:
  prompt tokens: 4096
  batch 1 / batch 4 raw output parity: true / true

Qwen3.5 generic TP1 nominal 4K:
  prompt tokens: 4048
  batch 1 / batch 4 raw output parity: true / true

Qwen3.5 generic TP4 4K / 16K / 32K:
  prompt tokens: 4096 / 16384 / 32768
  batch 1 / batch 4 raw output parity: true for every cell

Qwen3.5 native MTP TP1 4K, TP4 4K, TP4 16K:
  prompt tokens: 4096 / 4096 / 16384
  batch 1 / batch 4 raw output parity: true for every retained passing cell
```

The Qwen3 TP1/4K performance artifact independently recomputes to:

```text
batch 1:
  TPOT ratio:        0.777298
  throughput ratio:  1.250800
  H2D byte ratio:    1.000000
  TTFT ratio:        0.932292

batch 4:
  TPOT ratio:        0.519766
  throughput ratio:  1.668044
  H2D byte ratio:    1.000000
  TTFT ratio:        0.662030
```

Both policies record positive real H2D and D2H counters. This artifact proves
real movement and positive TPOT/throughput direction in its scope, but it does
**not** prove an H2D-byte reduction; the raw H2D ratio is exactly `1.0` for
both batch sizes.

The Qwen3.5 TP4/16K performance artifact records:

```text
batch 1:
  TPOT ratio:        0.5677486644
  throughput ratio:  1.4279094053
  H2D byte ratio:    0.5408970976
  TTFT ratio:        1.1405970670

batch 4:
  TPOT ratio:        0.5443779098
  throughput ratio:  1.7823893148
  H2D byte ratio:    0.5396825397
  TTFT ratio:        1.0609167401
```

This is positive TPOT, throughput, and movement direction for one controlled
TP4/16K configuration, with explicit TTFT regressions. Five measured runs do
not establish statistical significance, 32K performance, or production
readiness.

### Generic n-gram conclusion

```text
two target model structures:
  ESTABLISHED for generic n-gram authorities

TP1 / TP4:
  ESTABLISHED within recorded generic n-gram scopes

4K / 16K / 32K and batch 1 / batch 4:
  ESTABLISHED within recorded generic n-gram scopes

exact greedy parity:
  ESTABLISHED within recorded generic n-gram scopes

real KV movement:
  ESTABLISHED within recorded offload scopes

complete cross-matrix performance and production promotion:
  NOT ESTABLISHED
```

## 2. Learned-Source and Native-MTP Evidence

### Established native-MTP cells

| Gate | Authority | Classification |
| --- | --- | --- |
| Qwen3.5 native MTP TP1/4K | `artifacts/qwen35_native_mtp_tp1_4k_engine/opaque-57a3a62810d43636b96295da/local-authority/result.json` | `QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED`; batch-1/4 parity true |
| Qwen3.5 native MTP TP4/4K | `artifacts/qwen35_native_mtp_tp4_4k_engine/opaque-95aa0889f8365beac8be2b6f/artifacts/authority/result.json` | `QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED`; baseline/native and TP1/TP4 parity true |
| Qwen3.5 native MTP TP4/16K target-KV offload | `artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/lifecycle-release-fix-20260814-2/artifacts/authority/result.json` | `QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED`; batch-1/4 parity true |

These authorities establish production `LLMEngine.step()` execution,
batch-native first-target and verifier callbacks, exact greedy parity,
transactional proposal-slot commit/reject/release, zero accepted-prefix target
replay, and zero terminal proposal/slot leaks within their recorded scopes.

The TP4/16K authority establishes **target-KV** offload. It explicitly does
not establish proposal-KV offload.

### Failed TP4/32K cell

The retained TP4/32K authority attempt is:

```text
artifacts/qwen35_native_mtp_tp4_32k_target_kv_offload/
  native-mtp-tp4-32k-20260814-2/
```

It failed closed with:

```text
ValueError: baseline/native output parity mismatch for batch 1
```

Observed batch-1 outputs:

```text
baseline:
  [220, 15, 15, 15, 15, 15, 15, 15]

native MTP:
  [220, 15, 15, 220, 15, 15, 220, 15]

differing output indices:
  3, 6
```

Batch 4 passed exact parity, but it cannot override the failed batch-1 cell.
Therefore:

```text
TP4_32K_ENGINE_PARITY=NOT_ESTABLISHED
QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD=NOT_ESTABLISHED
```

### Artifact-only TP4/32K diagnostic refinement

The retained four-cell artifact narrows the failure beyond the final
baseline/native token mismatch.

The batch-1 and batch-4 baseline cells use the exact same prompt-0 token row
and differ in engine configuration only at `max_num_seqs` (`1` versus `4`).
Nevertheless, ordinary baseline execution is not batch-shape invariant at
32K:

```text
baseline batch 1 prompt 0:
  [220, 15, 15, 15, 15, 15, 15, 15]

baseline batch 4 prompt 0:
  [220, 15, 15, 220, 15, 15, 220, 15]

native MTP batch 1 prompt 0:
  [220, 15, 15, 220, 15, 15, 220, 15]
```

The existing `AUTHORITY_TARGET_LOGITS` rows show that the two ordinary
baseline executions have identical first-token compact logits. With the same
prompt and the same emitted first token `220`, their next-decode compact
top-k rows already differ at prediction index 1, while the argmax remains
`15`. The first baseline batch-shape argmax and output divergence is
prediction/output index 3:

```text
32K baseline batch 1, prediction 3:
  top tokens: [15, 220, 1970, 12102, 18169]
  top logits: [18.875, 18.625, 14.6875, 14.6875, 14.5625]

32K baseline batch 4 prompt 0, prediction 3:
  top tokens: [220, 8381, 5979, 1970, 15]
  top logits: [19.25, 17.875, 17.625, 17.25, 17.0]
```

This is 32K-specific in the retained authorities. At 16K, prompt-0 ordinary
baseline batch-1 and batch-4 compact target-logit rows are exactly equal for
all eight prediction indices, and their outputs are equal.

Static control-flow inspection narrows the first shared boundary without
establishing a root cause:

1. both retained cells have `enforce_eager=true`, so batch-1 and batch-4
   ordinary decode both use eager execution; CUDA Graph dispatch is not the
   authority difference;
2. with `chunked_prefill_decode_first=false`, batch-4 prompt 0 completes
   prefill and emits its first token, then remains in `running` while the
   other three 32K prompts finish their 1024-token prefill chunks; only after
   `waiting` and `prefilling` drain does `_schedule_decode()` select all four
   rows;
3. Qwen3.5 recurrent state is held in distinct lease-backed slots, and the
   packed heterogeneous layer stack executes full-attention and linear-state
   layers through request-local row contexts; no static cross-request
   recurrent-slot overwrite was found;
4. the request-local decode context slices `slot_mapping`, `context_lens`,
   `block_tables`, `kv_offload_logical_block_tables`, and
   `kv_offload_context_lens`, but deliberately retains the full batch's
   `kv_offload_write_blocks`; all rows therefore share one
   `KVOffloadMVP0` residency manager and protect/mark the batch-wide write
   block set while each request-local full-attention row is evaluated; and
5. a 16K row occupies 64 logical blocks at block size 256 and fits inside the
   68 GPU staging blocks. A 32K row occupies 128 logical blocks and cannot
   fit, so even batch 1 must repeatedly evict and reload windows.

The movement counters match that threshold. Per rank, 16K baseline batch 1
records zero H2D copies and zero evictions, while 32K baseline batch 1 records
2623 H2D copies and 2684 evictions. Batch 4 changes the residency history
further: prompt 0 waits through three additional long-prefill streams and
each request-local decode row carries the batch-wide write-block protection
set. This makes the first evidence-backed shared boundary the
longer-than-staging-capacity blockwise residency path, not speculative
verification. It does not prove that residency history is the defect.

Existing local tests cover request-local metadata slicing, eager
multi-sequence dispatch, blockwise plan construction, staging order, and
bounded future hints. They do not cover exact logits/output invariance for
the same longer-than-capacity row under batch-1 versus batch-4 residency
history, nor a 128-block Qwen3.5 ordinary decode row with 68 staging slots.

A dependency-light CPU experiment extracted the real blockwise planning and
online-softmax functions from `tinyvllm/layers/attention.py`. The same
logical row remained bit-exact under forward/reverse layer order, batch-1-like,
other-row, and mixed-tail initial residency, and both row-only and full-batch
write-block sets:

```text
BLOCKWISE_CPU_RESIDENCY_HISTORY_EXACT_INVARIANCE=PASS
```

This narrows the remaining defect surface from blockwise planning/arithmetic
to copy or state management, assuming staged KV bytes are correct.

Static copy-control-flow inspection then found a more specific synchronization
risk. With the authority defaults
`kv_offload_async_copy=true`,
`kv_offload_batch_copy=true`, and
`kv_offload_writeback_on_evict=false`:

1. `_kv_offload_after_forward()` calls `writeback_dirty()`;
2. `_enqueue_d2h_pairs()` orders the copy stream after the current stream with
   `copy_stream.wait_stream(current_stream)`, then records `d2h_done`;
3. it does not add the reverse dependency from a future current stream to
   that D2H event;
4. on the next step, `_stage_kv_offload_write_blocks()` can request the same
   block while it remains resident;
5. `ensure_resident()` returns before its eviction-only D2H wait path when no
   block is missing; and
6. `_kv_offload_before_forward()` calls `wait_for_pending()`, whose pending
   set tracks H2D completion rather than resident D2H readers.

A one-off AST plus dependency-light mock executed the real
`writeback_dirty()`, `ensure_resident()`,
`_stage_kv_offload_write_blocks()`, `wait_for_pending()`, and
`_kv_offload_before_forward()` control flow. It retained the resident block's
`d2h_done` event, selected the same physical slot for the next forward, and
observed zero current-stream event waits:

```text
ASYNC_D2H_RESIDENT_REWRITE_DEPENDENCY=MISSING_IN_STATIC_CONTROL_FLOW
RESIDENT_BLOCK=0 SLOT=0 D2H_EVENT_RETAINED=1 CURRENT_STREAM_WAIT_EVENT_CALLS=0
TP4_32K_EXACT_ROOT_CAUSE=NOT_ESTABLISHED
```

This establishes a missing reverse stream dependency in the static control
flow. It does not prove that the race materialized in the retained TP4/32K
authority or caused its batch-shape divergence. A focused GPU trace or
controlled synchronization experiment requires separate approval.

Prediction-index alignment further limits this D2H candidate. The 32K prompt
is exactly `128 * 256` tokens. Prediction index 0 is sampled by the final
prefill. After that token is appended, the first decode sees sequence length
32769, allocates a fresh block, and writes offset 0. Rewriting the same
resident decode block after its previous D2H can first occur on the following
decode at offset 1, which produces prediction index 2. The retained compact
logits already differ at prediction index 1:

```text
PREDICTION_0_EXECUTION=FINAL_PREFILL_SAMPLE
PREDICTION_1_EXECUTION=FIRST_DECODE_FRESH_BLOCK_OFFSET_0
PREDICTION_2_EXECUTION=SECOND_DECODE_RESIDENT_BLOCK_OFFSET_1
FIRST_COMPACT_LOGIT_DRIFT_PREDICTION_INDEX=1
ASYNC_D2H_RESIDENT_REWRITE_CANNOT_EXPLAIN_FIRST_COMPACT_LOGIT_DRIFT=true
```

The resident-D2H rewrite dependency remains a real static defect candidate,
but it cannot be the complete explanation for the first observed divergence.

The same investigation found an earlier, capacity-aligned H2D dependency
candidate inside the first blockwise decode. `_enqueue_h2d_pairs()` makes the
future current stream wait for the H2D completion event, but the copy stream
does not first wait for the current stream to finish reading the physical
slot that is about to be reused. The dependency graph can therefore be:

```text
current stream: read old KV slot -> wait for new H2D event
copy stream:    overwrite slot by H2D -> record new H2D event

missing edge:   read old KV slot -> overwrite slot by H2D
```

The blockwise decode loop reads each staged slot into `k_dense`/`v_dense` on
the current stream, then immediately stages the next window. There is no
explicit current-to-copy stream dependency between those operations.

A dependency-light static simulation used the production decode-window plan,
LRU-cost eviction, `ensure_resident()`, and
`_stage_blockwise_read_window()` control flow. It modeled pending
current-stream reads to expose destination-slot reuse, not CUDA completion
timing:

```text
16K / 64 prompt blocks + 1 write block:
  windows: 9
  H2D writes: 0
  evictions: 0
  H2D writes to slots with prior pending reads: 0

32K / 128 prompt blocks + 1 write block:
  windows: 17
  H2D writes: 69
  evictions: 70
  H2D writes to slots with prior pending reads: 61
  first such reuse: logical block 64 -> physical slot 3
```

This matches the retained capacity boundary: 65 total blocks fit in 68 slots,
while 129 do not. Unlike the resident-D2H rewrite candidate, H2D slot reuse is
reachable during prediction index 1, so it remains compatible with the first
compact-logit drift. The simulation proves a missing dependency and reachable
slot reuse, not that CUDA overlap occurred in the authority. Existing KV
offload tests cover H2D completion waits and D2H-before-reload ordering but
not current-stream read completion before H2D overwrites a reused slot.

An AST source-order assertion confirms that this is not limited to ordinary
decode. All three production blockwise paths have the same window structure:

```text
_blockwise_online_decode_attention
_blockwise_online_spec_verify_attention
_blockwise_online_prefill_attention

per window:
  _stage_blockwise_read_window()
  current-stream k_cache/v_cache reads
  next loop iteration may stage H2D into reused slots
```

None of those loops contains a `wait_stream`, `wait_event`, `synchronize`, or
`record_stream` boundary between the current-stream cache reads and the next
window's staging call. `_enqueue_h2d_pairs()` also has no current-to-copy
stream wait. A focused search of the KV-offload, blockwise-planning, and
blockwise-speculative tests found no regression that covers read completion
before H2D slot reuse:

```text
BLOCKWISE_STAGE_READ_NEXT_STAGE_DEPENDENCY_GAP=PASS path=decode
BLOCKWISE_STAGE_READ_NEXT_STAGE_DEPENDENCY_GAP=PASS path=spec_verify
BLOCKWISE_STAGE_READ_NEXT_STAGE_DEPENDENCY_GAP=PASS path=prefill
ASYNC_H2D_CURRENT_TO_COPY_WAIT=ABSENT
H2D_SLOT_REUSE_IMPACT_SCOPE=DECODE_SPEC_VERIFY_PREFILL
CROSS_STREAM_SLOT_REUSE_REGRESSION=ABSENT
```

The official PyTorch `torch.cuda.Stream` API semantics confirm that the
existing wait direction is insufficient for this reverse dependency.
`waiting_stream.wait_event(event)` constrains only future work submitted to
the waiting stream, and `waiting_stream.wait_stream(other_stream)` constrains
only future work on the waiting stream against work already submitted to the
other stream. Therefore the current stream waiting for an H2D completion
event establishes:

```text
H2D overwrite -> later current-stream work
```

It does not establish the required reverse edge:

```text
prior current-stream slot read -> later copy-stream H2D overwrite
```

This API-level directionality supports the static dependency-gap
classification, but it still does not prove that unsafe overlap occurred in
the retained authority execution.

The retained authority cells, campaign log, and source manifest do not record
the PyTorch version, CUDA runtime version, or NVIDIA driver version. The
official API contract supports the dependency analysis, but the missing
runtime identity prevents binding the retained failure to one exact software
and driver stack:

```text
AUTHORITY_TORCH_VERSION=NOT_RECORDED
AUTHORITY_CUDA_RUNTIME_VERSION=NOT_RECORDED
AUTHORITY_NVIDIA_DRIVER_VERSION=NOT_RECORDED
```

Git history shows that this is not a reverse dependency removed by a recent
wait-coalescing optimization. Commit `2234631a` introduced the async KV H2D
copy stream and the production blockwise decode loop together on 2026-06-30.
The initial H2D path already lacked a
`copy_stream.wait_stream(current_stream)` edge, while the initial blockwise
loop only called `wait_for_pending()` before reading the staged slots. That
wait ordered H2D completion before the current-stream reads, but did not order
those reads before the next window's copy-stream overwrite. Commit `6ddd636a`
later added blockwise prefill with the same direction. Commits `e7b7256e` and
`8672e64c` narrowed or skipped redundant H2D completion waits; neither removed
a current-to-copy edge because no such edge existed in the initial blockwise
implementation.

The repository contains an internal directionality reference in
`tinyvllm/utils/cpu_offload.py`: its H2D prefetch path explicitly calls
`prefetch_stream.wait_stream(torch.cuda.current_stream())` before submitting
new H2D work. That separate weight-offload path is not proof of the required
KV correction, but it confirms that the repository already uses a
current-to-copy dependency when a copy stream may otherwise race work already
submitted to the compute stream.

The original green KV-offload smokes do not close this coverage gap:

1. commit `2234631a` did not contain `tools/test_kv_offload.py`; that test file
   first appeared in commit `3924881e`;
2. the initial synthetic thrash smoke used
   `ensure_resident(..., wait=True)`, `synchronize_copies()`, and a host-visible
   `.cpu()` content comparison for every window before advancing, so it did
   not reproduce an asynchronous current-stream read followed immediately by
   copy-stream overwrite of the same physical slot;
3. the initial real single-request blockwise smoke recorded
   `h2d_copies=0`;
4. the real two-request blockwise smoke recorded H2D movement, but its
   `baseline-only` `gate_pass` checked only that every request produced the
   requested output length. It did not compare exact greedy tokens or logits
   against a non-offloaded or serialized reference.

Therefore the historical green results establish execution and page movement,
not exact correctness under cross-stream physical-slot reuse. They do not
contradict the current H2D slot-reuse hypothesis:

```text
ASYNC_H2D_SLOT_REUSE_DEPENDENCY_GAP_ORIGIN=PRESENT_SINCE_INITIAL_BLOCKWISE_COMMIT
INITIAL_SYNTHETIC_THRASH_CROSS_STREAM_OVERWRITE_COVERAGE=ABSENT
INITIAL_BLOCKWISE_H2D_REAL_SMOKE_EXACT_PARITY=NOT_ESTABLISHED
IN_REPO_H2D_CURRENT_TO_COPY_ORDERING_REFERENCE=ESTABLISHED
HISTORICAL_GREEN_CONTRADICTS_H2D_HYPOTHESIS=NO
TP4_32K_EXACT_ROOT_CAUSE=NOT_ESTABLISHED
```

A dependency-light first-decode simulation then compared the ordinary
`baseline:b1` and `baseline:b4` production layouts rather than only checking
that the race is reachable in one row shape. The simulation reproduced the
production decode-window construction, cross-layer hints, protected write
blocks, LRU-cost victim selection, multi-block slot reassignment, and
forward/reverse layer alternation without importing Torch or executing CUDA.

The logical layout assumptions are source- and artifact-bound:

- `BlockManager` allocates cache misses from the head of its free-block deque;
- each 32K prompt occupies exactly 128 full 256-token blocks;
- the first decode token allocates one fresh write block per row;
- all four batch-4 prompt hashes are distinct, and rows 1-3 first differ from
  row 0 at token index 5, inside the first block. Therefore the four rows do
  not share a reusable full prefix block.

The resulting first-decode rows are:

```text
baseline:b1:
  prompt blocks: 0..127
  fresh write block: 128

baseline:b4:
  row 0 prompt blocks: 0..127, fresh write block: 512
  row 1 prompt blocks: 128..255, fresh write block: 513
  row 2 prompt blocks: 256..383, fresh write block: 514
  row 3 prompt blocks: 384..511, fresh write block: 515
```

Five 68-slot starting layouts and both recency orders were used for each batch
shape. Each case ran one forward and one reverse blockwise layer while
retaining submitted current-stream slot reads as potentially incomplete:

```text
baseline:b1:
  windows: 17
  required blocks per full window: 8
  protected write blocks: 1
  first-layer H2D writes: 61 to 122
  first-layer evictions: 61 to 122
  H2D overwrites of slots with prior submitted reads: 61 to 65
  first hazardous window start block: 56 or 64
  unique hazardous physical slots after two layers: 61 to 64

baseline:b4:
  windows: 17
  required blocks per full window: 32
  protected write blocks: 4
  first-layer H2D writes: 496 to 512
  first-layer evictions: 496 to 512
  H2D overwrites of slots with prior submitted reads: 448 in every case
  first hazardous window start block: 16
  unique hazardous physical slots after two layers: 64 in every case
```

The invariant batch-4 count is its prompt-block capacity deficit after
reserving four writes: `512 - (68 - 4) = 448`. The minimum batch-1 count is
the corresponding one-row deficit: `128 - (68 - 1) = 61`; some histories
reload additional previously evicted blocks. This demonstrates that the
candidate dependency gap has a materially different batch-1 versus batch-4
overwrite topology and becomes reachable much earlier in the batch-4 window
scan. It strengthens compatibility with the retained batch-shape divergence,
but it is not CUDA timing evidence and does not identify which retained output
is correct:

```text
TP4_32K_B1_B4_H2D_SLOT_REUSE_TOPOLOGY=DIFFERENT_AND_REACHABLE
TP4_32K_B1_FIRST_LAYER_HAZARD_OVERWRITES=61_TO_65
TP4_32K_B4_FIRST_LAYER_HAZARD_OVERWRITES=448
TP4_32K_B1_FIRST_HAZARD_WINDOW_START_BLOCK=56_OR_64
TP4_32K_B4_FIRST_HAZARD_WINDOW_START_BLOCK=16
BATCH_SHAPE_H2D_HAZARD_DIFFERENCE_COMPATIBILITY=SUPPORTED_NOT_CAUSAL
TP4_32K_EXACT_ROOT_CAUSE=NOT_ESTABLISHED
```

The retained first divergence is still ordinary decode, so this wider impact
scope must not be used to claim that prefill or spec-verify corruption
occurred. It does mean that any eventual evidence-grounded correction and
regression matrix must cover all three shared blockwise consumers rather than
patching only the native-MTP verifier.

The static reachability result is also robust to the unknown retained
residency history. The production plan/eviction simulation was repeated for
five 68-slot layouts (`low`, `high`, even/mixed, modular-stride, and
head/tail), both forward and reversed recency ordering, and both forward and
reverse layer window order. All 20 cases produced dependency-free H2D reuse
of slots with prior current-stream reads:

```text
H2D writes by case: 61 to 128
evictions by case: 62 to 129
dependency-free reused slots in every case: 61

ASYNC_H2D_SLOT_REUSE_STATIC_REACHABILITY_ACROSS_HISTORIES=PASS
```

The stable count is the capacity deficit after reserving one write block:
`128 prompt blocks - (68 slots - 1 write slot) = 61`. This is still a static
legal-schedule/read-dependency result, not evidence that the copy engine won
the race on the retained GPUs.

The 16K and 32K campaigns captured byte-identical frozen implementations for
the Engine, ModelRunner, transactional runtime, Qwen3.5 speculative state,
state transaction, attention implementations, and inherited 16K
worker/gate. The 32K overlay changes the prompt length and authority
requirements rather than introducing another runtime implementation.

Fresh local artifact assertions:

```text
TP4_32K_ARTIFACT_DIAGNOSTIC_ASSERTIONS=PASS
FIRST_BASELINE_BATCH_SHAPE_TOPK_DIFF_PREDICTION_INDEX=1
FIRST_BASELINE_BATCH_SHAPE_ARGMAX_DIFF_PREDICTION_INDEX=3
FIRST_BASELINE_BATCH_SHAPE_OUTPUT_DIFF_INDEX=3
TP4_16K_BASELINE_BATCH_SHAPE_LOGITS=EXACT_MATCH
TP4_32K_NATIVE_B1_EQUALS_BASELINE_B4_OUTPUT=true
```

This evidence makes a purely MTP-exclusive explanation insufficient. It does
not establish that the batch-1 baseline is wrong, that the batch-4/native
result is correct, or that there is only one defect. The legacy compact log
records ordinary logits and native first-target logits, but it does not
record the native verify-tail logits or aligned target-KV and side-state
lineage. Those missing boundaries remain the purpose of the approved local
paired-trace implementation.

Exact diagnostic boundary:

```text
TP4_32K_ORDINARY_BASELINE_BATCH_SHAPE_INVARIANCE=FAILED
TP4_16K_ORDINARY_BASELINE_BATCH_SHAPE_INVARIANCE=ESTABLISHED
TP4_32K_FIRST_COMPACT_LOGIT_DRIFT_PREDICTION_INDEX=1
TP4_32K_FIRST_BASELINE_BATCH_SHAPE_ARGMAX_DIVERGENCE_INDEX=3
PURE_MTP_EXCLUSIVE_ROOT_CAUSE=NOT_SUPPORTED
CORRECT_32K_REFERENCE_PATH=NOT_ESTABLISHED
TP4_32K_FIRST_STATIC_SHARED_BOUNDARY=BLOCKWISE_RESIDENCY_BEYOND_SINGLE_ROW_CAPACITY
TP4_32K_CUDA_GRAPH_DISPATCH_DIFFERENCE=NOT_SUPPORTED
TP4_32K_CROSS_REQUEST_RECURRENT_SLOT_OVERWRITE=NOT_SUPPORTED_BY_STATIC_TRACE
TP4_32K_BLOCKWISE_RESIDENCY_HISTORY_DEFECT=PLAUSIBLE_NOT_ESTABLISHED
BLOCKWISE_CPU_RESIDENCY_HISTORY_EXACT_INVARIANCE=PASS
ASYNC_D2H_RESIDENT_REWRITE_DEPENDENCY=MISSING_IN_STATIC_CONTROL_FLOW
ASYNC_D2H_RESIDENT_REWRITE_FIRST_DRIFT_COMPATIBILITY=NOT_SUPPORTED
ASYNC_H2D_SLOT_REUSE_REVERSE_DEPENDENCY=MISSING_IN_STATIC_CONTROL_FLOW
TP4_32K_FIRST_DRIFT_H2D_SLOT_REUSE_COMPATIBILITY=PLAUSIBLE_NOT_ESTABLISHED
H2D_SLOT_REUSE_IMPACT_SCOPE=DECODE_SPEC_VERIFY_PREFILL
CROSS_STREAM_SLOT_REUSE_REGRESSION=ABSENT
ASYNC_H2D_SLOT_REUSE_STATIC_REACHABILITY_ACROSS_HISTORIES=PASS
TP4_32K_BATCH_SHAPE_EXACT_INVARIANCE_REGRESSION=ABSENT
TP4_32K_EXACT_ROOT_CAUSE=NOT_ESTABLISHED
```

### Paired diagnostic status

The default-off paired verify trace is locally implemented and verified:

```text
TP4_32K_PAIRED_VERIFY_TRACE_LOCAL_IMPLEMENTATION=ESTABLISHED
TP4_32K_PAIRED_VERIFY_TRACE_DEFAULT_OFF_CONTRACT=ESTABLISHED
TP4_32K_PAIRED_VERIFY_TRACE_SOURCE_BOUNDARY=ESTABLISHED
```

Fresh recorded local evidence:

```text
focused trace/state/worker matrix:
  50 passed in 6.59s

complete related matrix:
  294 passed in 9.53s

py_compile:
  PASS

remote-runner bash syntax:
  PASS

scoped git diff --check:
  PASS
```

No real paired trace artifact has been captured:

```text
TP4_32K_FIRST_DIVERGENCE_ARTIFACT=NOT_ESTABLISHED
TP4_32K_ROOT_CAUSE=NOT_ESTABLISHED
```

The current worker exposes `run_paired_trace_diagnostic()` only as an
internal function. Its ordinary `__main__` still resolves to the single-cell
authority CLI, and the existing remote shell invokes that ordinary authority
path. The trace cannot be captured through the current remote entry point.

Fresh invocation-chain verification:

```text
uv run --offline --python 3.12 --with torch python \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  --help
```

The exposed CLI accepts only:

```text
--model
--gpu-indices
--policy {baseline,native_mtp}
--batch-size {1,4}
--dist-port
--master-port
--out
```

It has no paired-diagnostic subcommand or output contract. Source inspection
also confirms:

```text
run_paired_trace_diagnostic():
  defined in the 32K worker

worker __main__:
  sys.exit(main())

main():
  inherited from the frozen ordinary 16K single-cell worker

current 32K remote shell:
  derives and executes the ordinary 32K authority runner from the frozen
  16K authority runner
```

Therefore a remote command using the current worker or shell can only execute
ordinary single-cell/authority flows. This is a demonstrated invocation gap,
not an inference from missing artifacts.

### Independent learned drafter

The Qwen3 autoregressive draft path has:

```text
concrete dense Qwen3 draft backend
batch-native exact-Q proposal execution
multi-layer physical proposal KV
checkpoint and tokenizer compatibility contracts
source-neutral ModelRunner registration
TP1 real-checkpoint gate harness and preflight mode
TP4 local sharded executor contract
```

The local TP4 contract is not a loaded-checkpoint authority. There is no
completed real Qwen3-draft/Qwen3.5-target artifact for TP1 or TP4.

The local TP4 gate now also has a schema-v2 source-bound bundle contract.
Each acceptance event binds its prompt, generation step, output boundary,
proposal tokens, and exact accepted-prefix token IDs. The bundle freezes a
30-file producer/runtime/verifier inventory, writes a deterministic
regular-file-only `source.tar`, binds `result.json` and `source.tar` in
`source_manifest.json`, and publishes only when current-source and
archived-source verifier receipts match. This establishes artifact machinery,
not a retained loaded-checkpoint authority.

The independent drafter does reuse the shared transactional Proposal-KV
lifecycle locally. `AutoregressiveDraftExecutor` registers proposal
transactions with `ProposalKVLifecycle`, prepares one finalize ticket for the
accepted prefix, commits exactly `max(accepted - 1, 0)` staged entries, and
rolls back the unused suffix. Repeated accepted rounds append without prompt
bootstrap replay. ModelRunner exposes the same prepare/commit/rollback
lifecycle methods used by the source-neutral runtime.

Fresh local contract verification:

```text
uv run --offline --python 3.12 --with pytest --with torch pytest -q \
  tools/test_autoregressive_draft_tp.py \
  tools/test_autoregressive_draft_tp4_local_gate.py \
  tools/test_qwen3_draft_backend.py \
  tools/test_autoregressive_draft_executor.py \
  tools/test_autoregressive_draft_registration.py \
  tools/test_autoregressive_draft_model_runner_integration.py

219 passed in 15.30s

focused py_compile:
  PASS
```

Fresh direct-commit re-audit:

```text
autoregressive executor, registration, ModelRunner integration,
TP4 direct validator, and TP4 snapshot transport:
  176 passed in 15.29s

production learned-drafter rematerialization symbol scan:
  PASS

schema-v2 gate and source-bound bundle/verifier:
  41 passed in 0.74s

gate, archived verifier, TP4 validator, snapshot transport,
executor, registration, and ModelRunner integration:
  217 passed in 6.86s

source inventory:
  SOURCE_INVENTORY_CHECK=PASS files=30

unchecked extractall scan:
  PASS
```

The production autoregressive executor, ModelRunner, `LLMEngine`, proposal
cache, and proposal lifecycle contain no reference to
`rematerialize_accepted_kv`, `legacy_rematerialize`, or
`commit_accepted_tokens`.

This verifies the dependency-light TP topology, sharded backend, local TP4
evidence schema, executor, registration, and ModelRunner integration
contracts. It does not load either checkpoint, execute CUDA/NCCL, establish
exact output parity, or provide performance evidence.

```text
AUTOREGRESSIVE_DRAFT_ACCEPTED_PREFIX_DIRECT_COMMIT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_REJECTED_SUFFIX_ROLLBACK=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_PRODUCTION_REMATERIALIZATION_SYMBOLS=ABSENT
TP4_RESULT_CONTENT_FOR_INDEPENDENT_VERIFICATION=ESTABLISHED_LOCAL_CONTRACT
TP4_ACCEPTED_PREFIX_IDENTITY=RECONSTRUCTABLE_LOCAL_CONTRACT
TP4_SOURCE_BOUND_BUNDLE_CONTRACT=ESTABLISHED_LOCAL
TP4_ARCHIVED_VERIFIER=ESTABLISHED_LOCAL
TP4_ATOMIC_ARTIFACT_BUNDLE=ESTABLISHED_LOCAL
TP4_RETAINED_SOURCE_BOUND_EXECUTION_ARTIFACT=ABSENT
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
SECOND_LEARNED_STRUCTURE_TP1_CORRECTNESS=NOT_ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_TP4_REAL_CHECKPOINT_AUTHORITY=NOT_ESTABLISHED
```

## 3. Requirement-to-Evidence Checklist

| Phase 1 requirement | Current authority | Verdict |
| --- | --- | --- |
| Source-neutral speculative runtime | Generic adapter/runtime, ModelRunner executor registry, batch-native verifier, Scheduler and transactional KV contracts | `ESTABLISHED` as architecture and local/loaded-runtime contract |
| Transactional accepted-prefix commit | Generic and native authorities require direct accepted KV ownership transfer and zero accepted-prefix target replay | `ESTABLISHED` within cited scopes |
| Rejected-suffix rollback/release | Generic and native authorities plus lifecycle cleanup checks | `ESTABLISHED` within cited scopes |
| Two target model structures | Qwen3 and Qwen3.5 generic n-gram authorities | `ESTABLISHED` for generic n-gram |
| Real learned proposal source | Qwen3.5 native MTP authorities through TP4/16K | `PARTIAL`; TP4/32K failed and controlled performance is missing |
| Second learned proposal structure | Qwen3 independent drafter local contracts only | `NOT_ESTABLISHED` |
| TP1 and TP4 | Generic n-gram matrix is established; native MTP has TP1/4K and TP4 through 16K | `PARTIAL` for learned-source promotion |
| 4K, 16K, and 32K | Generic n-gram matrix is established; native MTP TP4/32K failed | `PARTIAL` for learned-source promotion |
| Batch 1 and batch 4 | Generic matrix and native MTP through TP4/16K pass | `PARTIAL`; native TP4/32K batch 1 failed |
| Multi-sequence production execution | Batch-4 authorities exercise multiple simultaneous sequences and per-sequence transactions | `ESTABLISHED` within cited greedy scopes; broader cancellation/mixed-state coverage remains incomplete |
| Exact greedy parity | Generic matrix established; native MTP established through TP4/16K | `FAILED` at native MTP TP4/32K batch 1 |
| Controlled TTFT/TPOT/throughput | Qwen3 TP1/4K and Qwen3.5 TP4/16K n-gram artifacts | `PARTIAL`; no controlled native-MTP performance authority |
| Peak memory | Measured in generic performance artifacts | `PARTIAL`; no general reduction and no learned-source matrix |
| Acceptance | Recorded for generic n-gram and native proposal execution scopes | `PARTIAL`; no independent-drafter or complete learned-source performance matrix |
| Real KV H2D/D2H | Real `KVOffloadMVP0` counters in generic and target-KV-offload scopes | `PARTIAL`; proposal-KV offload and unified learned-source movement are missing |
| Proposal-KV offload | Proposal K/V remains GPU-resident in current learned-source authorities | `NOT_ESTABLISHED` |
| KV8/KV4 plus offload | `kv_quant_bits` supports 0/4/8, but offload asserts `kv_quant_bits == 0` | `NOT_ESTABLISHED` |
| Variable-Q CUDA Graph | Legacy TP1/no-offload Qwen3.5 artifact plus 28 raw transaction rows | `PARTIAL`; transaction slot-set equations are independently recomputable, but graph/eager token parity and per-family capture/replay remain producer-asserted; no TP4, long-context, offload, or performance authority |
| Source/config/checkpoint integrity | Nine primary correctness/performance authorities retain matching source manifests, model/checkpoint identities, and verifier receipts; two legacy artifacts have weaker provenance | `PARTIAL`; the Qwen3 TP1 performance artifact has embedded source-file hashes but no standalone manifest/tree/archive/checkpoint binding, the CUDA Graph artifact is checkpoint-bound but not source-bound, and no cited result binds a `source.tar` SHA-256 |

### Retained artifact integrity re-audit

A fresh 2026-08-15 filesystem and JSON audit checked every artifact path
explicitly cited by this checklist:

```text
referenced JSON artifacts:                         11
present and JSON-decodable:                       11
result/source-manifest pairs:                      9
matching source_tree_sha256:                       9 / 9
matching model/checkpoint manifest hashes:         9 / 9
authority verify.json receipts present/decodable:  9 / 9
source.tar retained:                               8 / 9
result payloads binding source_archive_sha256:      0 / 11
```

The manifest-backed authorities remain source-bound to the exact file hashes
recorded in their manifests. Eight also retain a source archive containing an
archived verifier, but their result schemas do not bind the archive bytes
with a `source_archive_sha256`. The older generic Qwen3 TP4 authority has a
matching source manifest and PASS verifier receipt but no retained
`source.tar`.

The two non-manifest-backed legacy artifacts are weaker:

- `artifacts/speculative_runtime_performance/20260812T085852Z/result.json`
  embeds nine source-file hashes and has local/remote PASS verifier receipts,
  but has no standalone source manifest, source-tree hash, source archive, or
  model-manifest hash;
- `artifacts/qwen35-mtp-runs/qwen35-mtp-graph-gate-opaque-7/
  qwen35_mtp_real_checkpoint_gate.json` retains a checkpoint-manifest hash and
  explicit limitations, but has no source-file/tree binding, source archive,
  or independent verifier receipt.

These provenance gaps do not reverse the recorded within-scope correctness or
performance observations. They prevent treating artifact integrity as a
uniformly established Phase 1 property and are another reason promotion
remains fail closed.

### Transactional KV raw-receipt re-audit

A fresh 2026-08-15 read-only assertion pass recomputed transactional KV
invariants from the retained payloads rather than trusting producer
classification strings:

```text
TRANSACTIONAL_KV_RAW_RECEIPT_ASSERTIONS=PASS

Qwen3.5 generic correctness:
  passing cells:                    8
  accepted draft tokens:         125
  rejected draft tokens:          55
  TP4 per-rank transaction rows: 140
  explicit accepted-prefix replay: 0 in every cell

Qwen3.5 generic TP4/16K performance:
  passing cells:                    2
  warmup/parity/measured runs:     14
  accepted draft tokens:         2170
  explicit replay/rejection receipt fields: absent

Qwen3.5 native MTP TP1:
  passing cells:                    2
  proposal-KV receipts:            40
  accepted proposal tokens:       150
  rejected proposal tokens:        10

Qwen3.5 native MTP TP4:
  passing cells:                    4
  rank snapshots:                  16
  per-rank proposal transactions: 212
  canonical accepted tokens:      174
  canonical rejected tokens:       32
```

For all eight Qwen3.5 generic correctness cells, the raw
`consumed_input_mappings` recompute exactly to the recorded accepted and
rejected totals, both totals are non-zero, and
`accepted_prefix_replays == 0`. The TP4 artifacts additionally retain all
four ranks' acceptance masks and `kv_decision` rows; each mask population
equals `accepted_draft_count`, and each decision records committed-prefix
plus rolled-back-suffix handling. Their terminal receipts have zero prepared
transactions/live leases and no owned child processes. The TP1 artifact has
the aggregate mappings, explicit zero replay, leaked-sequence inventory, and
terminal cleanup, but not the TP4 rank-level `kv_decision` rows.

The generic TP4/16K performance artifact is intentionally weaker for this
specific question. Its fourteen raw runs establish non-zero acceptance and
the cell cleanup receipts are terminal, but the run schema does not retain
`accepted_prefix_replays`, `rejected_draft_tokens`, or transaction masks.
That performance payload cannot independently strengthen the transactional
zero-replay or rejected-suffix claims; the matching correctness authority is
the relevant source for those claims.

For native MTP TP1, all forty `proposal_kv_receipts` report
`accepted_slot_identity_preserved=true` and
`rejected_slots_released=true`; their accepted/rejected counts exactly
recompute the runtime totals. Both cells have
`accepted_prefix_target_replays == 0`, zero proposal-KV slots in use, no open
transactions/finalize tickets/sequences, and an all-zero native terminal
snapshot.

For native MTP TP4, every one of the sixteen rank snapshots reports zero
accepted-prefix target replay. Per-rank proposal-transaction sums exactly
match the rank-level accepted/rejected totals, and every passing cell contains
rejected proposal tokens. The snapshots do not expose the TP1
`accepted_slot_identity_preserved` boolean, so that exact receipt claim is
limited to TP1. TP4 instead retains committed transaction rows followed by
zero active transactions, zero allocated physical slots, zero prepared
tickets, zero cache-owned slots, and zero active cache sequences; this is the
raw terminal release evidence for the rejected suffixes.

The older Qwen3 generic TP4 artifact has non-zero acceptance and terminal
cleanup but predates the explicit replay and rejection fields:

```text
QWEN3_GENERIC_TP4_TRANSACTIONAL_CLASSIFICATION=
  ACCEPTANCE_ESTABLISHED_BUT_ZERO_REPLAY_FIELD_MISSING
```

It must not be used by itself as zero-replay or rejected-suffix proof. The
same fail-closed rule applies to any future artifact whose producer emits a
green classification without the raw counters and receipts needed to
recompute it.

### Archived transactional-verifier coverage audit

The presence of nine source-bound `verify.json` receipts does not mean every
transactional invariant above was independently recomputed by the archived
verifiers. A fresh 2026-08-15 audit loaded the gate implementations directly
from each retained `source.tar`, first validated the original result, and then
ran controlled in-memory mutation probes:

```text
ARCHIVED_GATE_ORIGINAL_RESULTS=PASS

generic Qwen3.5 TP1:
  accepted aggregate +1:                         ACCEPT
  rejected aggregate +1:                         ACCEPT

generic Qwen3.5 TP4:
  rejected aggregate +1:                         ACCEPT
  replace every rank kv_decision with opaque text: ACCEPT

native MTP TP1:
  accepted aggregate +1:                         ACCEPT
  rejected aggregate +1:                         ACCEPT
  accepted_slot_identity_preserved=false:         REJECT

native MTP TP4:
  all-rank accepted/rejected aggregates +1:       ACCEPT
  proposal-KV release ticket count +1:            REJECT

ARCHIVED_GATE_MUTATION_EXPECTATIONS=PASS
```

The accepted mutations expose specific verifier-coverage gaps rather than
failures in the retained original payloads:

- the generic TP1 gate requires positive accepted/rejected counters and
  validates each committed-input mapping, but does not equate either aggregate
  counter to the mapping totals;
- the generic TP4 gate equates proposed and accepted totals to mappings and
  checks cross-rank transaction digests, but does not equate
  `rejected_draft_tokens` to `proposal_token_count - accepted_draft_count`;
- generic TP4 requires a non-empty `kv_decision` string, but does not validate
  the `commit_prefix_*_rollback_suffix` decision vocabulary or semantics;
- native MTP TP1 enforces both identity/release booleans, but only requires
  receipt accepted/rejected sums to be less than or equal to the runtime
  totals, not equal;
- native MTP TP4 deeply validates proposal transactions, commit/release ticket
  counts, all-rank transaction parity, and terminal zero ownership, but does
  not equate rank-level accepted/rejected aggregates to transaction sums.

The long-context generic gates reuse the generic TP4 validator, so the
rejected-total and decision-string gaps apply to their archived verifier
coverage as well. The performance verifier intentionally validates acceptance
and cleanup but has no transaction-level replay/rejection payload to
recompute.

This does not invalidate the original retained values: the separate raw
payload re-audit above recomputed the missing equalities and decision
invariants and passed. It does mean that those stronger assertions currently
live only in the 2026-08-15 completion audit; they are not protected by the
retained archived verifier receipts.

```text
AUTHORITY_VERIFIER_RECEIPT_PRESENCE=ESTABLISHED
ARCHIVED_TRANSACTIONAL_VERIFIER_SEMANTIC_COVERAGE=PARTIAL
CURRENT_RAW_PAYLOAD_TRANSACTIONAL_REAUDIT=PASS
TAMPER_RESISTANT_TRANSACTIONAL_ARTIFACT_AUTHORITY=NOT_ESTABLISHED_UNIFORMLY
```

Future authority bundles must make aggregate-to-row equality, decision
semantics, accepted identity, rejected release, replay zero, and terminal
ownership checks part of the archived verifier itself. A current-checkout
ad-hoc assertion is useful audit evidence but is not a substitute for a
result-bound frozen verifier.

### Performance and movement verifier coverage audit

The two cited controlled-performance artifacts were also mutation-audited
against their retained verifier logic:

- the Qwen3 TP1/4K artifact has no `source.tar`; all nine embedded
  `source_files` SHA-256 values were checked, with five current matches and
  four expected mismatches in actively modified runtime files. The three
  verifier-side files
  (`speculative_runtime_performance_gate.py`, worker, and verifier) match
  exactly, so only that verifier logic is byte-identical frozen-equivalent;
  the current execution tree is not;
- the Qwen3.5 TP4/16K gate was loaded only from the regular files named by its
  retained source manifest inside `source.tar`.

Both original artifacts passed validation. Controlled mutations produced:

```text
Qwen3 TP1/4K:
  aggregate TPOT median +1:                    REJECT
  aggregate TTFT/throughput/peak memory +1:    REJECT
  aggregate accepted-token total +1:           REJECT
  run H2D total no longer matching rank rows:  REJECT
  parity output token changed:                 REJECT
  top-level direction changed:                 REJECT
  per-request TTFT/TPOT/completion changed:     REJECT
  stored throughput or request rate changed:   REJECT
  raw batch_elapsed_s +1 only:                  ACCEPT
  run-level simulated_upload_mb added:          ACCEPT
  top-level simulate_kv_upload_mb added:        ACCEPT

Qwen3.5 TP4/16K:
  aggregate TPOT median +1:                    REJECT
  aggregate TTFT/throughput/peak memory +1:    REJECT
  aggregate accepted-token total +1:           REJECT
  run H2D total no longer matching rank rows:  REJECT
  parity output token changed:                 REJECT
  comparison TPOT ratio changed:               REJECT
  raw batch_elapsed_s +1 only:                  REJECT
  run-level simulated_upload_mb added:          REJECT
  top-level simulate_kv_upload_mb added:        ACCEPT
```

The strong coverage is material:

- aggregate TPOT, throughput, memory, movement, and runtime totals are
  reconstructed from measured runs;
- movement totals must match the per-rank deltas;
- raw parity runs are compared token-for-token;
- Qwen3 direction and Qwen3.5 per-batch comparison ratios are recomputed;
- Qwen3.5 cross-checks elapsed time against stored token throughput.

Two boundaries remain:

1. the Qwen3 verifier validates stored throughput/request-rate and their
   aggregate values, but does not cross-check `batch_elapsed_s` itself against
   those rates;
2. both top-level schemas ignore an unknown simulation marker, and the older
   Qwen3 run-level movement schema also ignores one. Qwen3.5 rejects the same
   run-level marker through canonical aggregate mismatch, but not a top-level
   marker.

The retained original artifacts still contain no simulation fields, their
movement totals match all rank rows, and source tracing binds those counters
to `LLMEngine.kv_offload_summaries()` and production `KVOffloadMVP0` copy
accounting. Therefore the recorded real-movement observations remain valid
within scope. The mutation result means the verifier does not uniformly
enforce the no-simulation rule against every possible unknown-field
injection.

```text
PERFORMANCE_AGGREGATE_RECOMPUTATION=ESTABLISHED
MOVEMENT_RANK_SUM_VALIDATION=ESTABLISHED
EXACT_PARITY_MUTATION_REJECTION=ESTABLISHED
QWEN3_BATCH_ELAPSED_RATE_CONSISTENCY=NOT_ENFORCED
NO_SIMULATION_VERIFIER_ENFORCEMENT=PARTIAL
PERFORMANCE_ARTIFACT_TAMPER_RESISTANCE=PARTIAL
```

Future promotion verifiers must use exact schemas and reject all simulation
markers at every nesting level, then derive throughput/request rate directly
from elapsed time and token/request counts rather than accepting them as
independently mutable measurements.

### Long-context blockwise and residency verifier coverage audit

The retained Qwen3.5 generic TP4/16K, generic TP4/32K, and native-MTP
TP4/16K target-KV-offload artifacts were revalidated with their frozen
verifiers, then tested with in-memory mutations:

```text
LONG_CONTEXT_ORIGINAL_RESULTS=PASS

generic TP4/16K and TP4/32K:
  context length +1:                      REJECT
  profiling disabled:                    REJECT
  movement provenance changed:           REJECT
  all batch-4 movement counters zeroed:  REJECT
  residency phases emptied:              REJECT

native MTP TP4/16K:
  blockwise prefill disabled:             REJECT
  blockwise decode disabled:              REJECT
  prefill chunk 1024 changed:             REJECT
  peak residency above GPU capacity:      REJECT
  movement provenance changed:            REJECT
  all native batch-4 movement zeroed:     REJECT
  residency phases emptied:               REJECT
```

This establishes that the frozen gates bind prompt length, required
configuration, movement provenance, non-zero movement, transaction/residency
lifecycle, and—within the native TP4/16K schema—bounded GPU capacity. The
native artifact records peak/resident block counts of `65` against `68` GPU
blocks for batch 1 and `68` against `68` for batch 4.

It does not directly establish which attention implementation executed.
None of the three result payloads records `_blockwise_online_*`,
`blockwise_online`, or an `attention_path` observation. Their source binding
also differs:

```text
generic TP4/16K source manifest: 16 files
generic TP4/32K source manifest: 16 files
  both include the campaign workers
  neither includes tinyvllm/layers/attention.py
  neither includes tinyvllm/layers/qwen35_full_attention.py

native MTP TP4/16K source manifest: 112 files
  includes tinyvllm/config.py
  includes tinyvllm/engine/scheduler.py
  includes tinyvllm/layers/attention.py
  includes tinyvllm/layers/qwen35_full_attention.py
```

The archived native `attention.py` dispatches to blockwise online prefill,
decode, and speculative-verify functions when the frozen blockwise flags are
enabled. This is source-bound dispatch/configuration evidence, not a direct
runtime path or kernel observation. The generic 16K/32K artifacts are strong
long-prompt parity, real-movement, and lifecycle authorities, but their
manifests do not bind the blockwise attention implementation.

Exact boundary:

```text
LONG_CONTEXT_PROMPT_AND_EXACT_PARITY=ESTABLISHED_WITHIN_RETAINED_SCOPES
LONG_CONTEXT_REAL_TARGET_KV_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPES
LONG_CONTEXT_RESIDENCY_LIFECYCLE=ESTABLISHED_WITHIN_RETAINED_SCOPES
NATIVE_TP4_16K_BOUNDED_GPU_RESIDENCY=ESTABLISHED
NATIVE_TP4_16K_BLOCKWISE_SOURCE_AND_CONFIG_BINDING=ESTABLISHED
GENERIC_16K_32K_BLOCKWISE_IMPLEMENTATION_SOURCE_BINDING=MISSING
DIRECT_BLOCKWISE_RUNTIME_PATH_OBSERVATION=MISSING
END_TO_END_BLOCKWISE_ONLINE_SOFTMAX_AUTHORITY=PARTIAL
```

### Async/batched migration and dirty-writeback evidence audit

The cited generic TP4/16K, generic TP4/32K, native-MTP TP4/16K, and
Qwen3.5 TP4/16K performance artifacts retain positive production
`h2d_copies`, `h2d_bytes`, `d2h_copies`, and `d2h_bytes`. Their source
manifests bind `llm_engine.py`, `model_runner.py`, the campaign worker, and
the gate. This establishes real device/host KV movement through
`KVOffloadMVP0` in the retained scopes.

The movement schemas do not preserve all properties of that movement.
`MOVEMENT_KEYS` includes:

```text
h2d_copies
h2d_bytes
d2h_copies
d2h_bytes
copy_waits
evictions
evict_clean
speculative-residency counters
```

It omits:

```text
evict_dirty
dirty_blocks
async_copy
batch_copy
h2d_batches / d2h_batches
h2d_batch_spans / d2h_batch_spans
writeback_on_evict
```

Positive D2H counters are still meaningful. In the source-bound manager,
`_enqueue_d2h_pairs()` is reached by dirty eviction or
`writeback_dirty()`, so the artifacts establish real KV writeback. They do
not independently reconstruct how many writebacks were eviction-triggered,
how many were explicit, whether every transfer used the async copy stream,
or how many requests were coalesced into each batch/span.

Frozen-verifier unknown-field probing shows:

```text
generic TP4/16K added evict_dirty field: PASS
generic TP4/32K added evict_dirty field: PASS
native MTP TP4/16K added evict_dirty field: FAIL
```

The generic verifier normalizes only its selected movement keys and does not
fail closed on the additional rank-row field. The native verifier rejects
the non-canonical field, but its canonical schema still does not require an
`evict_dirty` receipt.

Fresh dependency-light checks:

```text
generation/diagnostic dirty-D2H selection:
  2 passed, 32 deselected in 0.17s

direct AST-loaded writeback_dirty batch contract:
  DEPENDENCY_LIGHT_WRITEBACK_BATCH_CONTRACT=PASS
```

The production CUDA test
`test_dirty_evictions_are_batched_when_loading_multiple_blocks` was not
collected on this host. Importing `tinyvllm.engine.model_runner` fails first
with `ModuleNotFoundError: No module named 'flash_attn'`; therefore its
internal no-CUDA early return was never reached. This is neither a PASS nor a
new implementation failure, and it cannot serve as fresh CUDA batch-copy
evidence.

Exact boundary:

```text
REAL_TARGET_KV_H2D_D2H_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPES
DIRTY_WRITEBACK_IMPLEMENTATION=ESTABLISHED
REAL_TARGET_KV_D2H_WRITEBACK=ESTABLISHED_WITHIN_RETAINED_SCOPES
DEPENDENCY_LIGHT_WRITEBACK_BATCH_CONTRACT=ESTABLISHED
DIRTY_EVICTION_EXACT_ARTIFACT_RECEIPT=MISSING
EXPLICIT_VS_EVICTION_WRITEBACK_ARTIFACT_CLASSIFICATION=MISSING
ASYNC_COPY_RUNTIME_RECEIPT=MISSING
BATCHED_COPY_RUNTIME_RECEIPT=MISSING
CURRENT_HOST_CUDA_DIRTY_BATCH_TEST=NOT_COLLECTED_MISSING_FLASH_ATTN
GENERIC_MOVEMENT_SCHEMA_UNKNOWN_FIELD_REJECTION=NOT_ENFORCED
```

### Prefix sharing, deduplication, and CPU-backing evidence audit

The Prefix-KV objective has three distinct evidence layers that must not be
collapsed into one loaded authority:

1. ordinary full-attention KV block reuse through hash/token identity,
   reservations, attachment, generations, and `Block.ref_count`;
2. Qwen3.5-specific convolution/recurrent-state tensor interning and
   reference counting; and
3. composition of a reusable ordinary KV block identity with CPU-valid
   offload backing and an H2D residency request.

Fresh dependency-light verification:

```text
uv run --offline --python 3.12 --with pytest --with torch pytest -q \
  tools/test_prefix_kv_offload_integration.py \
  tools/test_qwen35_hybrid_prefix_cache.py \
  tools/test_chunked_prefill.py \
  -k 'prefix or reusable or hash_collision or ref_count or deduplic or intern'

45 passed, 84 deselected in 9.59s
```

The ordinary-prefix tests cover exact token matching behind hash collisions,
live and idle reuse, multi-owner refcounts, reservation rollback,
attachment/reference transfer, generation lifetime, capacity pressure, and
sampleable-token boundaries. The Qwen3.5 hybrid cache tests cover exact and
partial tensor interning, physical versus logical byte accounting,
last-reference release, replacement, collision handling, and failure
rollback.

The CPU-backing composition test is intentionally dependency-light. It
constructs `KVOffloadMVP0` through `__new__`, supplies list-backed state, and
replaces `_enqueue_h2d_pairs()` and `_enqueue_d2h_pairs()` with Python
recorders. It proves that a same-generation idle prefix with `cpu_valid=True`
schedules `(logical_block, slot)` H2D residency, that stale generations
invalidate CPU backing, and that cached-prefix reads require valid backing.
It does not execute a CUDA copy, pinned-memory transfer, asynchronous stream,
or loaded model.

A repository artifact-JSON scan found no result field named
`num_cached_tokens`, `prefix_cache_hits`, `prefix_hit`,
`prefix_block_count`, `deduplicated_bytes`, `current_intern_references`,
`qwen35_hybrid_prefix`, or `reused_prompt_tokens`. Some native source
manifests include `block_manager.py` and `qwen35_hybrid_prefix_cache.py`, but
source inclusion does not prove that a workload produced a cache hit,
shared-reference lifetime, deduplication, or CPU-backed prefix restore. The
artifact fields named `accepted_prefix_*` refer to speculative accepted-token
transactions and are unrelated to cross-request prefix caching.

Exact boundary:

```text
ORDINARY_PREFIX_HASH_TOKEN_REUSE_LOCAL_CONTRACT=ESTABLISHED
ORDINARY_PREFIX_MULTI_OWNER_REFCOUNT_LOCAL_CONTRACT=ESTABLISHED
QWEN35_HYBRID_TENSOR_DEDUP_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_RESIDENCY_SCHEDULING_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_REAL_CUDA_COPY=NOT_ESTABLISHED
LOADED_CROSS_REQUEST_PREFIX_HIT_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_REFCOUNT_LIFETIME_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_DEDUP_BYTE_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_CPU_RESTORE_AUTHORITY=NOT_ESTABLISHED
GENERIC_HYBRID_STATE_PREFIX_DEDUP=NOT_ESTABLISHED
```

### Legacy CUDA Graph artifact semantic coverage audit

The retained Qwen3.5 CUDA Graph artifact is a single legacy JSON with a
checkpoint-manifest hash but no source manifest, source archive, frozen
verifier, or independent verification receipt. Its raw payload records:

```text
q_values=[1,2,3,4]
batch_sizes=[1,4]
graph_capture_count=6
graph_replay_count=12
graph_eager_argmax_equal=true
graph_eager_proposal_tokens_equal=true
transaction_cases=28
```

The 28 transaction rows cover every canonical
`(batch_size, q, accepted)` tuple for batch `(1,4)`, `Q=(1,2,3,4)`, and
`accepted=0..q`. Independent raw-set recomputation passes:

```text
staged slots = batch_size * max(q - 1, 0)
committed slots = batch_size * max(accepted - 1, 0)
released slots = staged - committed
committed and released are disjoint
committed union released equals staged

GRAPH_TRANSACTION_RAW_SET_ASSERTIONS=PASS cases=28
GRAPH_TOP_LEVEL_COUNTS_AND_DOMAIN_ASSERTIONS=PASS
```

The artifact does not retain per-`(q,batch)` graph/eager token arrays,
logits, capture counts, replay counts, or backend receipts. Therefore the
top-level equality booleans and aggregate `6/12` counts cannot be
independently reconstructed from the JSON.

The current, non-source-bound
`tools/qwen35_mtp_real_checkpoint_gate.py` validator was mutation-probed:

```text
original:                       ACCEPT
graph_capture_count=1:          ACCEPT
graph_replay_count=1:           ACCEPT
eager max_abs_diff=999999:      ACCEPT
arbitrary transaction slot sets: ACCEPT
graph_eager_argmax_equal=false: REJECT
accepted identity=false:        REJECT
q_values missing Q=4:           REJECT
```

It enforces the domain and required booleans, but only requires positive
aggregate capture/replay counts, accepts any finite nonnegative max
difference, and validates slot-list types rather than their set equations.
Fresh current-worktree unit verification is green:

```text
tools/test_qwen35_mtp_real_checkpoint_gate.py:
  53 passed in 0.26s
```

Those unit tests establish the current gate contract; they do not convert
the legacy JSON into a frozen or independently verified artifact.

Exact boundary:

```text
CUDA_GRAPH_Q1_Q4_BATCH1_BATCH4_RECORDED=ESTABLISHED
CUDA_GRAPH_TRANSACTION_RAW_SET_EQUATIONS=ESTABLISHED
CUDA_GRAPH_TOP_LEVEL_CAPTURE_REPLAY_COUNTS=PRODUCER_ASSERTED
CUDA_GRAPH_EAGER_TOKEN_PARITY=NOT_INDEPENDENTLY_RECOMPUTABLE
CUDA_GRAPH_PER_FAMILY_CAPTURE_REPLAY=NOT_INDEPENDENTLY_RECOMPUTABLE
CUDA_GRAPH_SOURCE_BOUND_PROVENANCE=MISSING
CUDA_GRAPH_FROZEN_INDEPENDENT_VERIFIER=MISSING
CUDA_GRAPH_CURRENT_VALIDATOR_SEMANTIC_COVERAGE=PARTIAL
VARIABLE_Q_CUDA_GRAPH_AUTHORITY=PARTIAL
```

### KV4/KV8 Precision-Path Audit

The independent quantized-KV paths are real implementation paths, but they
are not retained loaded-model authorities.

Source-bound contracts:

```text
configuration:
  kv_quant_bits in {0, 4, 8}
  KV4 requires an even group size and symmetric quantization
  KV8 requires symmetric quantization
  AM compact rejects KV4 but permits FP KV or KV8
  kv_offload_mvp0 requires kv_quant_bits == 0
  blockwise prefill/decode require kv_offload_mvp0

storage:
  KV4 cache dtype=int8, final dimension=head_dim/2
  KV8 cache dtype=int8, final dimension=head_dim
  both allocate per-group scale tensors
  scale tensors are attached to each attention module

attention:
  KV4 and KV8 have distinct cache-write and dequantization paths
  cached prefill and decode dequantize selected blocks
  spec_verify rejects every quantized-KV mode
  blockwise offload prefill/decode assert kv_quant_bits == 0
```

Fresh dependency-light evidence:

```text
focused fail-closed/transaction tests:
  3 passed

covered:
  quantized snapshot rejection
  spec-verify unsupported-feature rejection
  unsupported native mode fails before reservation

KV4 NumPy reference round-trip:
  group_size=32  max_err=2.9071  bound=3.9056
  group_size=64  max_err=3.1450  bound=3.9056
  group_size=128 max_err=3.1450  bound=3.9056

KV8 CPU reference quantization plus actual dequant_kv_blocks_q8():
  group_size=32  max_err=0.15293193  bound=0.16280018
  group_size=64  max_err=0.15620777  bound=0.16280018
  group_size=128 max_err=0.16151990  bound=0.16280018
  exact dequant parity and padded block-table handling passed

KV8 cached-prefill routing:
  tools/test_qwen35_full_attention_shell.py::
    test_cached_prefill_quantized_kv_uses_original_backend
  1 passed in 1.38s
```

The KV8 numerical check used an isolated CPU Torch environment. It AST-extracted
and executed the current pure-Torch `dequant_kv_blocks_q8()` implementation
from `tinyvllm/layers/attention.py`; inputs came from a CPU reference quantizer
matching the current `store_kvcache_q8_kernel` formula. This establishes the
CPU reference plus actual-dequant numerical contract, not execution of the
Triton store kernel. The optional real KV4 and KV8 Triton round-trips were not
run because GPU workloads require separate authorization.

A recursive structured-key scan over retained JSON/JSONL artifacts found zero
fields named for `kv_quant_bits`, quantized cache dtype, KV scale, quantization
group size, or quantized-KV execution. Existing generic and blockwise results
instead explicitly list `no KV8/KV4 evidence` in their limitations. Therefore
configuration support, source routing, and local reference arithmetic must not
be promoted into loaded parity, memory reduction, or performance authority.

Exact boundary:

```text
KV4_CPU_REFERENCE_ROUNDTRIP=ESTABLISHED
KV4_KV8_STORAGE_AND_ROUTING_CONTRACT=PARTIAL
KV4_KV8_SPEC_VERIFY_FAIL_CLOSED=ESTABLISHED_LOCALLY
KV4_KV8_OFFLOAD_BLOCKWISE_COMPOSITION=INTENTIONALLY_REJECTED
KV4_REAL_GPU_TRITON_ROUNDTRIP=NOT_RUN
KV8_CPU_REFERENCE_ACTUAL_DEQUANT_ROUNDTRIP=ESTABLISHED
KV8_CACHED_PREFILL_ROUTING=ESTABLISHED_LOCAL
KV8_TRITON_STORE_KERNEL_ROUNDTRIP=NOT_RUN
KV4_KV8_LOADED_PARITY=NOT_ESTABLISHED
KV4_KV8_MEMORY_REDUCTION=NOT_ESTABLISHED
KV4_KV8_PERFORMANCE=NOT_ESTABLISHED
KV4_KV8_RETAINED_EXECUTION_ARTIFACT=ABSENT
```

### Per-Layer/Per-Token Heat-Tiering Audit

The current target-KV offload runtime has recency- and future-aware eviction,
but it does not implement the requested generic heat-tier state machine.

What exists:

```text
KVOffloadMVP0._touch():
  one monotonically increasing last-used clock per physical GPU slot

KVOffloadMVP0._victim_score():
  lru:
    slot_last_used

  lru_cost:
    slot_last_used
    + fixed dirty-block penalty
    + fixed future-window penalty
    + fixed pending-H2D penalty

blockwise staging:
  bounded future block sets are passed into eviction scoring

configuration:
  kv_offload_evict_policy in {"lru", "lru_cost"}
```

These mechanisms choose an eviction victim among one homogeneous FP
GPU-staging population. They do not provide:

```text
per-layer heat
per-token heat
access-frequency accumulation or decay
hot/warm/cold identities
precision-tier ownership
residency-tier transition records
promotion/demotion thresholds
FP <-> KV8 <-> KV4 conversion
tier-aware transactional commit/rollback
```

Fresh local evidence:

```text
source-bound _victim_score probe:
  PASS

scores for one slot with block_nbytes=100:
  base LRU-cost:                 5.0
  future-window penalty:      805.0
  dirty plus future:         1205.0
  pending-H2D plus dirty:    1805.0

prefix CPU-backing identity/generation tests:
  3 passed
```

The selected `tools/test_kv_offload.py` pytest nodes did not collect because
the available interpreter lacks `torch`. The exact production method was
therefore AST-extracted for the source-bound probe; this does not establish
loaded runtime behavior or a heat-tier implementation.

A strict structured-key scan over retained JSON/JSONL artifacts found zero
KV heat/tier fields such as heat score, hot/warm/cold blocks, precision tier,
residency tier, promotion count, demotion count, or access frequency. Generic
artifact fields such as `promotion_classification`, `warmup_runs`, and
`rank_snapshots` were explicitly excluded as unrelated false positives.

Exact boundary:

```text
KV_LRU_RECENCY_POLICY=ESTABLISHED_IN_SOURCE
KV_FUTURE_DIRTY_PENDING_EVICTION_BIAS=ESTABLISHED_IN_SOURCE
PREFIX_CPU_BACKING_LOCAL_CONTRACT=ESTABLISHED
PER_LAYER_KV_HEAT=NOT_IMPLEMENTED
PER_TOKEN_KV_HEAT=NOT_IMPLEMENTED
HOT_WARM_COLD_KV_STATE_MACHINE=NOT_IMPLEMENTED
PRECISION_TIER_TRANSITIONS=NOT_IMPLEMENTED
HEAT_TIER_TRANSACTIONAL_KV_COMPOSITION=NOT_IMPLEMENTED
HEAT_TIER_RETAINED_EXECUTION_ARTIFACT=ABSENT
HEAT_TIER_LOADED_PARITY=NOT_ESTABLISHED
HEAT_TIER_MEMORY_REDUCTION=NOT_ESTABLISHED
HEAT_TIER_PERFORMANCE=NOT_ESTABLISHED
```

### Verifier/Sampling/KV-Commit Fusion Audit

The production speculative path has a correct batch-native transactional
publication sequence, but it is not one fused kernel or fused runtime phase.

Observed production sequence:

```text
target first-token and tail verification forwards
  -> host-visible target tokens / logits
  -> acceptance and greedy selection
  -> PreparedNativeSpeculativeBatch
  -> Scheduler.prepare_postprocess()
  -> BlockManager.prepare_speculative_kv_commit()
  -> optional residency precommit
  -> apply prepared side state
  -> BlockManager.commit_speculative_kv_commit_batch()
  -> Scheduler.commit_prepared_postprocess()
  -> optional residency seal
```

The runtime and profiler expose separate timing and state boundaries:

```text
target_forward_ms
accept_sample_ms
commit_metadata_ms
prepared side-state state
prepared KV commit plans
prepared Scheduler publication
```

Functions named `verify_and_commit_block()` aggregate the workflow under one
Python API, but internally still execute target forward, host argmax/acceptance,
KV materialization, metadata commit, and finish checks as distinct phases.
Likewise, token-free batch KV commit avoids accepted-token replay but does not
fuse verification or sampling into that commit operation.

Fresh dependency-light evidence:

```text
phase-order / token-free / exactly-once / Scheduler publication tests:
  5 passed
```

Those tests establish:

```text
side-state preparation precedes target execution
acceptance precedes prepared publication
KV batch commit is token-free and exactly once
Scheduler output commit appends selected tokens exactly once
runtime callback ordering is explicit and rollback-capable
```

A strict retained JSON/JSONL scan found zero fields for fused
verify/sample/commit execution, kernel launch count, fused graph nodes, or a
fusion-specific performance attribution. Existing end-to-end TPOT results
therefore cannot be used to claim that fusion exists.

Exact boundary:

```text
BATCH_NATIVE_MULTI_TOKEN_VERIFICATION=ESTABLISHED_WITHIN_CITED_SCOPES
TRANSACTIONAL_TOKEN_FREE_KV_COMMIT=ESTABLISHED
PREPARED_SCHEDULER_PUBLICATION=ESTABLISHED
VERIFY_SAMPLE_COMMIT_PHASE_ORDER=ESTABLISHED_LOCALLY
VERIFY_SAMPLE_KV_COMMIT_KERNEL_FUSION=NOT_IMPLEMENTED
VERIFY_SAMPLE_KV_COMMIT_RUNTIME_FUSION=NOT_IMPLEMENTED
FUSION_KERNEL_LAUNCH_REDUCTION=NOT_ESTABLISHED
FUSION_RETAINED_EXECUTION_ARTIFACT=ABSENT
FUSION_PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
```

### TP Collective Overlap and Reduction-Fusion Audit

The retained TP4 authorities prove that every rank participates in the
expected collectives. They do not prove communication/compute overlap,
collective fusion, ReduceScatter, or persistent hidden-state sharding.

Production collective call sites:

```text
VocabParallelEmbedding:
  synchronous dist.all_reduce

RowParallelLinear decode:
  one synchronous dist.all_reduce per output/chunk

RowParallelLinear dense prefill preservation:
  synchronous dist.all_gather followed by torch.cat

DecodeInternalProfiler.profile_collective():
  wraps and times the complete blocking collective call
```

The calls do not pass `async_op=True`, return a pending work handle, use a
dedicated communication stream, or defer a wait until after independent
compute. `RowParallelLinear` all-reduces its local partial output into a full
replicated output at each boundary, so sharded weights/model state do not
imply persistent hidden-state sharding.

Fresh retained-artifact inventory:

```text
Qwen3 generic TP4 authority:
  collective rows: 10032
  row_parallel_all_reduce: 9856
  vocab_parallel_embedding_all_reduce: 176
  async_op fields: 0

Qwen3.5 generic TP4/4K authority:
  collective rows: 5856
  row_parallel_all_reduce: 5760
  vocab_parallel_embedding_all_reduce: 96
  async_op fields: 0
```

No retained JSON/JSONL contains a structured field for collective overlap,
overlap ratio/time, AllReduce fusion, ReduceScatter, or persistent hidden
sharding. No production `tinyvllm` call site invokes `reduce_scatter`.

Fresh dependency-light evidence:

```text
collective profiler and TP4 inventory/mutation tests:
  6 passed
```

These tests establish that collective rows are recorded and that missing or
cross-rank-mismatched collective evidence is rejected. They do not establish
overlap or fusion.

Exact boundary:

```text
TP4_ALL_RANK_COLLECTIVE_PARTICIPATION=ESTABLISHED_WITHIN_CITED_SCOPES
TP4_COLLECTIVE_IDENTITY_VALIDATION=ESTABLISHED
TP_COLLECTIVE_CALLS_SYNCHRONOUS=ESTABLISHED_IN_SOURCE
TP_COLLECTIVE_COMPUTE_OVERLAP=NOT_IMPLEMENTED
SPECULATIVE_ALLREDUCE_FUSION=NOT_IMPLEMENTED
SPECULATIVE_REDUCESCATTER=NOT_IMPLEMENTED
PERSISTENT_HIDDEN_STATE_SHARDING=NOT_IMPLEMENTED
TP_OVERLAP_FUSION_RETAINED_ARTIFACT=ABSENT
TP_OVERLAP_FUSION_PERFORMANCE=NOT_ESTABLISHED
```

## 4. Objective-Wide Three-Direction Checklist

This section maps every explicit item in the three-direction objective. An
implementation primitive is not promoted to end-to-end authority unless a
loaded-model artifact exercises the required composition.

### Direction 1: KV Cache Utilization

| Explicit objective | Concrete source/artifact evidence | Status and boundary |
| --- | --- | --- |
| Decouple logical KV pages from physical GPU slots | `KVOffloadMVP0.logical_to_slot` and `slot_to_logical` in `tinyvllm/engine/model_runner.py`; block-generation identities in `tinyvllm/engine/speculative_residency.py`; real offload artifacts cited above | `ESTABLISHED` for the current offload runtime |
| GPU/CPU tiered cache | Pinned CPU backing, GPU staging slots, generation binding, residency and eviction in `KVOffloadMVP0` | `ESTABLISHED` as implementation and in recorded real-movement scopes; no generic hot/warm/cold precision tier |
| Asynchronous prefetch | Dedicated copy stream/events, pending waits, `ensure_resident()`, `_stage_blockwise_read_window()` | `ESTABLISHED` as implementation; retained results do not record an async-copy/stream/event receipt, and overlap benefit is not promoted |
| Batched H2D/D2H | `_enqueue_h2d_pairs()` and `_enqueue_d2h_pairs()` plus production copy/byte counters | Real H2D/D2H movement is `ESTABLISHED` within recorded scopes; batch count/span execution is `PARTIAL` because retained schemas omit batch receipts |
| Dirty writeback | `mark_dirty()`, `writeback_dirty()`, dirty eviction and D2H accounting in source-bound `KVOffloadMVP0` | Implementation and real D2H writeback are `ESTABLISHED` within cited scopes; exact dirty-eviction count and explicit-versus-eviction classification are `MISSING` from retained schemas |
| Prefix KV sharing across requests | Hash-chained full blocks, `hash_to_block_ids`, reusable-prefix reservations and attach/release in `tinyvllm/engine/block_manager.py`; focused scheduler/allocator tests | Implementation and local contract are `ESTABLISHED`; retained loaded artifacts do not record an explicit cross-request prefix-cache hit/reuse receipt |
| Prefix deduplication and reference counting | `Block.ref_count`, reusable hash buckets, block generations; exact tensor interning/refcounts in `tinyvllm/engine/qwen35_hybrid_prefix_cache.py` | Ordinary and Qwen3.5-specific local contracts are `ESTABLISHED`; generic loaded dedup/refcount authority is `NOT_ESTABLISHED` |
| Prefix KV plus CPU-resident backing | `tools/test_prefix_kv_offload_integration.py` verifies idle shared-prefix identity reuse, scheduled H2D restore, stale-generation invalidation, and required CPU validity | `ESTABLISHED` as a dependency-light composition contract; real CUDA copy and loaded-model reuse authority are `NOT_ESTABLISHED` |
| KV8/KV4 storage | `kv_quant_bits in (0,4,8)`, packed INT4/INT8 allocation, scale tensors, quantization and dequantization in `model_runner.py` and `attention.py`; fresh KV4 CPU reference and KV8 CPU-reference-plus-actual-dequant round-trips; fresh KV8 cached-prefill routing test | `PARTIAL`; local storage/routing and CPU numerical contracts exist, but Triton store-kernel execution, loaded parity, memory, performance, and retained execution authority do not |
| KV8/KV4 with offload | `tinyvllm/config.py` requires `kv_quant_bits == 0` for `kv_offload_mvp0`; blockwise offload asserts unquantized KV; focused local verifier tests reject quantized KV before transaction work | `NOT_ESTABLISHED` and deliberately fail closed |
| Per-layer/per-token heat grading | `_touch()` and `lru_cost` provide physical-slot recency plus fixed dirty/future/pending penalties; strict artifact scan found no heat/tier fields | `MISSING`; no layer/token heat, hot/warm/cold identity, precision transition, or promotion/demotion state machine |
| Accepted speculative/MTP KV commits directly | `begin_speculative_kv_transaction()`, `prepare_speculative_kv_commit()`, token-free batch commit, native proposal receipts and zero accepted-prefix replay | `ESTABLISHED` within cited generic/native scopes |
| Rejected suffix rollback | `rollback_speculative_kv_transaction()` and speculative-residency prepare/precommit/seal/rollback; tests require rejected blocks to be discarded without D2H | `ESTABLISHED` within cited scopes |
| Avoid per-token accepted-KV rematerialization | Generic/native authorities require zero accepted-prefix target replay; direct KV ownership is transferred at commit | `ESTABLISHED` within cited scopes; must remain true after future fusion/tiering changes |

#### Accepted-KV fallback reachability audit

The repository still contains an explicit legacy comparison implementation:

```text
tools/profile_ngram_commit.py:
  verifier_mode="legacy_rematerialize"
  rematerialize_accepted_kv()
  BlockManager.commit_accepted_tokens()

tools/native_verifier_oracle.py:
  legacy_rematerialize comparison policy
```

That path performs the historical per-token accepted-KV decode
rematerialization and remains useful as a negative/control policy. It is not
the production `LLMEngine.step()` speculative commit path.

Current production runtime call-chain inspection shows:

```text
LLMEngine.step()
  -> BlockManager.prepare_speculative_kv_commit()
  -> BlockManager.commit_speculative_kv_commit_batch()
  -> Scheduler prepared metadata commit

generic single-batch runtime
  -> BlockManager.commit_speculative_kv_transaction()
```

No production runtime file calls `BlockManager.commit_accepted_tokens()`.
Outside tests, its callers are the explicit legacy profiler/oracle paths.
Native and generic authority gates independently require accepted-prefix
replay/rematerialization counters to remain zero.

Fresh focused verification:

```text
uv run --offline --python 3.12 --with pytest --with torch pytest -q \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_ngram_speculative.py \
  -k 'commit or rollback or rematerial or replay or native_verify'

44 passed, 151 deselected in 1.89s
```

Exact boundary:

```text
PRODUCTION_RUNTIME_ACCEPTED_KV_REMATERIALIZATION_FALLBACK=NOT_REACHABLE
PRODUCTION_RUNTIME_ACCEPTED_PREFIX_TARGET_REPLAY=ZERO_IN_CITED_AUTHORITIES
LEGACY_PROFILER_REMATERIALIZATION_CONTROL=RETAINED
REPOSITORY_WIDE_REMATERIALIZATION_CODE_REMOVAL=NOT_CLAIMED
```

### Direction 2: Longer Context

| Explicit objective | Concrete source/artifact evidence | Status and boundary |
| --- | --- | --- |
| Blockwise/chunked prefill | Chunk scheduling and postprocess in `tinyvllm/engine/scheduler.py`; `_blockwise_online_prefill_attention()`; Qwen3/Qwen3.5 16K/32K authorities | Long-prompt execution and exact parity are `ESTABLISHED` within cited scopes; direct runtime observation of the blockwise attention path is `MISSING` |
| Full visible KV need not reside simultaneously on GPU | Blockwise read windows call `ensure_resident()` against bounded GPU staging while logical block tables remain complete; native TP4/16K records peaks of 65/68 blocks for batch 1 and 68/68 for batch 4 | `ESTABLISHED` for the native TP4/16K bounded-capacity scope; generic 16K/32K artifacts establish movement and lifecycle but do not independently bind the blockwise implementation |
| Blockwise online-softmax attention | `_blockwise_online_decode_attention()`, `_blockwise_online_prefill_attention()`, and `_blockwise_online_spec_verify_attention()` in `tinyvllm/layers/attention.py`; focused dense-oracle tests; native TP4/16K source archive | Implementation/test evidence is `ESTABLISHED`; end-to-end loaded authority is `PARTIAL` because no retained result records the selected runtime attention path, and the generic 16K/32K manifests omit the attention implementation |
| KV eviction and prefetch based on future windows | Forward/reverse future-hint builders, cross-layer reuse hints, `_stage_blockwise_read_window()`, LRU/cost eviction; retained real movement and eviction counters | Implementation and real movement are `ESTABLISHED`; execution of the specific future-window staging path is `PARTIAL` without a direct runtime path observation |
| Prefix cache with CPU-resident KV | Local integration test plus generic long-context offload runtime | Composition contract `ESTABLISHED`; broad cross-model performance promotion remains incomplete |
| Sliding-window attention | Explicitly deferred by the objective | `DEFERRED`; not a Phase 1 completion gate |
| Sparse attention | Explicitly deferred by the objective | `DEFERRED`; not a Phase 1 completion gate |
| Context parallel | Explicitly deferred by the objective | `DEFERRED`; not a Phase 1 completion gate |

### Direction 3: Lower TPOT

| Explicit objective | Concrete source/artifact evidence | Status and boundary |
| --- | --- | --- |
| One unified speculative runtime | `tinyvllm/engine/speculative_runtime.py`, batch runtime, ModelRunner proposal executor registry, Scheduler/Engine transactional flow | `ESTABLISHED` as source-neutral architecture |
| MTP-head proposal source | Qwen3.5 native MTP registration/executor and TP1/4K, TP4/4K, TP4/16K authorities | `PARTIAL`; TP4/32K failed and controlled native-MTP performance is missing |
| Independent draft-model source | Qwen3 backend, physical proposal KV, `autoregressive-draft` registration, TP1 gate and TP4 local contract | `PARTIAL`; no real loaded-checkpoint correctness artifact |
| Model-free n-gram source | Generic n-gram adapter and two-model TP1/TP4, 4K/16K/32K authorities | `ESTABLISHED` within recorded scopes |
| Model-free SAM source | SAM adapter and lifecycle/local tests | `PARTIAL`; no current loaded-model performance/promotion artifact |
| Batch-native multi-token verifier | Fixed-Q grouping and one target forward per group through `run_spec_verify_batch`; generated target KV is committed transactionally | `ESTABLISHED` within cited greedy scopes |
| Variable proposal lengths without padding | Distinct exact-Q groups and exact `(B,Q,W)` graph identities | `ESTABLISHED` as runtime contract |
| CUDA Graph for variable proposal length | Legacy artifact: `artifacts/qwen35-mtp-runs/qwen35-mtp-graph-gate-opaque-7/qwen35_mtp_real_checkpoint_gate.json`; newer contract: `tools/spec_verify_cuda_graph_smoke.py`, `tools/verify_spec_verify_cuda_graph_gate.py`, and `tools/run_spec_verify_cuda_graph_gate_remote.py` | `PARTIAL`; the legacy artifact exposes only producer booleans/aggregate counters plus independently recomputable transaction sets. The newer exact-family verifier recomputes per-family graph/eager outputs, KV digest, replay, failure, and transaction semantics with 11 current-source hashes, but no PASS artifact, deterministic source archive, archived verifier, TP4, offload, long-context, or performance authority exists |
| Fuse verifier, sampling, and KV commit | Runtime records separate target verification, host acceptance/greedy selection, prepared KV commit, Scheduler publication, and side-state/residency phases; focused phase-order tests pass | `MISSING`; the current path is transactional and batch-native but not kernel/runtime fused, and `verify_and_commit` function names are not fusion evidence |
| TP collective overlap | TP4 authorities record 15,888 synchronous AllReduce rows across the cited Qwen3 and Qwen3.5 cells; source calls do not use async work handles or communication streams | `MISSING`; all-rank participation is established, overlap is not implemented |
| AllReduce fusion | Row-parallel outputs invoke one blocking AllReduce per output/chunk; no fused operation or launch-reduction receipt exists | `MISSING` |
| ReduceScatter | No production `tinyvllm` call site invokes ReduceScatter and no retained artifact records it | `MISSING` |
| Persistent hidden-state sharding | Row-parallel partials are AllReduced into replicated outputs at each boundary; sharded weights/model state are not persistent hidden-state sharding | `MISSING` as a TPOT optimization authority |

### Promotion-Gate Coverage

| Required promotion dimension | Current evidence | Verdict |
| --- | --- | --- |
| Two model structures | Qwen3 and Qwen3.5 generic n-gram authorities | Generic scope `ESTABLISHED`; second learned proposal structure is not established |
| TP1 and TP4 | Generic n-gram matrix; native MTP TP1/4K and TP4 through 16K | Learned-source matrix `PARTIAL` |
| 4K/16K/32K or longer | Generic n-gram matrix covers all; native MTP 32K failed | Learned-source matrix `PARTIAL` |
| Batch 1/4/multi-sequence | Generic and native authorities through 16K; batch 4 is real multi-sequence execution | `PARTIAL` because native TP4/32K batch 1 failed and broader cancellation/mixed-state gates remain incomplete |
| Exact greedy parity | Generic matrix and native MTP through TP4/16K | `FAILED` at native MTP TP4/32K batch 1 |
| TPOT, TTFT and throughput | Qwen3 TP1/4K and Qwen3.5 TP4/16K generic n-gram performance artifacts | `PARTIAL`; no native-MTP or independent-drafter controlled performance authority |
| Peak GPU memory | Recorded by generic performance artifacts | `PARTIAL`; no general reduction or learned-source matrix |
| Real KV H2D bytes | Real `KVOffloadMVP0` counters in generic and target-KV-offload artifacts | `PARTIAL`; proposal-KV movement and unified learned-source matrix are missing |
| Acceptance | Generic and native authorities record proposal/acceptance rows | `PARTIAL`; independent drafter and complete learned-source performance matrix are missing |
| No simulated KV-copy claim | Authoritative movement rows use real MVP-0 counters; simulated upload options remain profiler-only | `ESTABLISHED` for cited authorities |

### Real KV Movement Provenance Audit

The cited performance and target-KV-offload authorities do not contain
`simulated_upload_*`, `simulate_kv_upload_mb`, or other simulation fields:

```text
artifacts/speculative_runtime_performance/20260812T085852Z/result.json
artifacts/qwen35_generic_speculative_tp4_16k_performance/opaque-c9807d19e6402acc22d4a615/artifacts/authority/result.json
artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/lifecycle-release-fix-20260814-2/artifacts/authority/result.json
```

Their movement provenance is the production offload implementation:

1. the generic performance worker takes before/after snapshots through
   `LLMEngine.kv_offload_summaries()` and validates each four-rank delta;
2. the native target-KV worker also takes before/after
   `engine.kv_offload_summaries()` snapshots and records every delta row with
   `provenance="engine.kv_offload_summaries"`;
3. `LLMEngine.kv_offload_summaries()` calls each rank's
   `ModelRunner.kv_offload_summary()`;
4. `ModelRunner.kv_offload_summary()` returns `KVOffloadMVP0.summary()`; and
5. `KVOffloadMVP0` increments `h2d_copies`/`h2d_bytes` in
   `_enqueue_h2d_pairs()` and `d2h_copies`/`d2h_bytes` in
   `_enqueue_d2h_pairs()` around the actual KV-cache/backing-store copy path.

The gates reject absent or fabricated movement rather than accepting a single
aggregate number:

- the generic TP4/16K performance gate requires exactly four rank rows,
  recomputes totals from those rows, rejects a rank-sum mismatch, and requires
  positive H2D and D2H copies and bytes for both batch-4 baseline and n-gram;
- the native TP4/16K target-KV gate requires positive native batch-4 target-KV
  H2D and D2H copies and bytes.

The native authority's batch-4 per-rank deltas are concrete and symmetric:

```text
baseline:  h2d=10752 copies / 67645734912 bytes
           d2h=  284 copies /  1786773504 bytes
native_mtp:h2d= 6762 copies / 42542825472 bytes
           d2h=  273 copies /  1717567488 bytes
```

These values are per rank across ranks 0-3. Batch-1 H2D is zero in that native
authority and is not misrepresented as bidirectional movement; the promotion
gate is deliberately attached to batch 4.

Exact boundary:

```text
CITED_AUTHORITY_SIMULATED_KV_UPLOAD_FIELDS=ABSENT
CITED_AUTHORITY_MOVEMENT_SOURCE=KVOffloadMVP0_REAL_COPY_COUNTERS
GENERIC_TP4_16K_BATCH4_BIDIRECTIONAL_MOVEMENT=GATE_ENFORCED
NATIVE_TP4_16K_BATCH4_TARGET_KV_BIDIRECTIONAL_MOVEMENT=GATE_ENFORCED
PROPOSAL_KV_OFFLOAD=NOT_ESTABLISHED
ISOLATED_KV_OFFLOAD_PERFORMANCE_CAUSALITY=NOT_CLAIMED
```

## 5. Deferred Broader Optimizations

The following are useful follow-on directions, but they must not distract
from the current correctness blocker:

```text
proposal-KV offload and unified target/proposal residency
KV8/KV4 plus offload and blockwise execution
generic hot/warm/cold precision and residency tiers
fused verify, sampling, and KV commit
TP collective/compute overlap
AllReduce fusion
ReduceScatter and persistent hidden-state sharding
TP4/long-context CUDA Graph
broader cancellation, non-greedy, and mixed-state serving matrices
```

None of these should be used to bypass the failed TP4/32K exact-parity gate.

## 5.1 Focused H2D Causal-Diagnostic Local Readiness

The focused diagnostic now has a local producer, tensor-free gate, independent
artifact verifier, and dependency-light lifecycle tests. Static inspection
confirms the producer is restricted to exactly:

```text
observe:b1
observe:b4
control:b1
control:b4
policy=baseline
prediction indices=0,1
```

Non-baseline policy is rejected before the frozen 32K worker is loaded. The
worker has no executable CLI, subprocess launcher, remote route, `torchrun`
entry point, or NCCL launch command, so this local audit cannot accidentally
start the campaign by invoking the focused files directly. The only executable
CLI in this focused set is the offline verifier.

Fresh dependency-light validation:

```text
tools/test_h2d_slot_reuse_attention_markers.py
tools/test_h2d_slot_reuse_diagnostic.py
tools/test_h2d_slot_reuse_manager.py
tools/test_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py

result:
  99 passed in 0.45s

adjacent dependency-light regressions:
  169 passed in 3.98s

py_compile:
  diagnostic core
  model runner
  attention
  LLM engine
  gate
  worker
  verifier
  PASS
```

The plan-to-code audit found and corrected one local execution blocker before
any GPU run: the worker classifies the first ordinary scheduling step as
`prefill`, but the real `ModelRunner` wrapper accepted only `decode`. A focused
RED test reproduced the rejection. The wrapper now accepts exactly
`{"prefill", "decode"}` and still rejects `spec_verify`, preserving the
baseline-only boundary. The focused and adjacent suites above are the fresh
GREEN evidence.

The same audit also found two gate-level false-positive paths. The evaluator
could return `SUPPORTED` after both observe/control batch-4 fixtures were
mutated together because it compared logits without proving that all four
cells represented the same workload. Two RED fixtures demonstrated false
`SUPPORTED` results for:

1. a changed batch-4 prompt-0 token sequence; and
2. changed batch-4 prediction-index-1 `position/context_length`.

The gate now fails closed unless prompt-0 tokens are identical across all four
cells and prediction indices 0 and 1 share identical `input_token_id`,
`position`, and `context_length`, with
`context_length == position + 1`. Both regressions are GREEN in the focused
suite above.

The exact plan-listed focused suite was also attempted. The default system
Python has no `torch`, but an existing Homebrew Python 3.12 environment
provides CPU-only `torch 2.12.0`. Using an isolated `/tmp` bridge containing
only the pure-Python pytest dependencies produced:

```text
tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py:
  35 passed in 3.57s

tools/test_kv_offload.py:
tools/test_blockwise_attention_planning.py:
  collection blocked by ModuleNotFoundError: flash_attn
  repository headers require a Torch/CUDA environment
```

The remaining two suites also import `triton` through the production attention
module. This macOS CPU environment has neither CUDA extension, and the
repository contains no dependency-light flash-attn/triton stub harness.
Faking those modules would not establish the advertised CUDA-side contract, so
the two collection failures remain environment limitations rather than passing
or failing test results.

Two readiness boundaries remain and must not be hidden:

1. human authorization is not represented by a required runtime token,
   approval file, or explicit `authorized` argument. The campaign API is
   callable by an importer, so authorization is an operational/documented
   boundary rather than a code-enforced fail-closed gate;
2. `source_tree_sha256` exactly follows the seven-file manifest declared by
   the implementation plan, but the worker dynamically imports
   `qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py`. That imported
   producer dependency and its execution closure are not included in the
   focused source digest. The verifier therefore detects drift in the focused
   gate/worker/verifier and instrumented runtime files, but does not establish
   complete producer-source provenance.

The missing closure is already enumerable rather than unknown. The inherited
32K authority gate exposes a 126-file `DEFAULT_SOURCE_FILES` inventory:

```text
tinyvllm Python files: 115
authority tool files: 11
total: 126
```

That inventory includes the 32K, 16K, and 4K frozen workers/gates plus
`model_runner.py`, `llm_engine.py`, and `attention.py`. The focused artifact
currently records only its separate seven-file digest and discards the
inherited authority source digest; availability of the 126-file inventory is
therefore a closure candidate, not retained provenance.

An import-only audit under CPU Torch, without calling `run_policy_cell`, also
showed that the inherited inventory is not itself the exact focused producer
closure. The focused producer dynamically loads eleven tool modules. Compared
with the eleven tool files in inherited `DEFAULT_SOURCE_FILES`, the producer
requires the focused gate and focused worker, while the inherited inventory
instead contains the 16K and 32K independent verifiers, which are not loaded
by the producer path. Consequently:

```text
inherited source inventory:
  115 tinyvllm + 11 tools = 126 files

conservative focused producer union:
  115 tinyvllm + 13 tools = 128 files

focused authority union including independent verifier:
  115 tinyvllm + 14 tools = 129 files
```

All 129 files exist locally. This refines the closure candidate but does not
change the approved seven-file artifact contract. Expanding the digest would
require an explicit authority-contract design decision because the written
plan requires the seven listed files and keeps the focused gate independent
from the paired-trace worker.

These limitations do not invalidate the local state-machine tests, but they
prevent the current tooling from being described as a fully source-closed,
code-authorized GPU authority.

```text
FOCUSED_H2D_DIAGNOSTIC_LOCAL_CONTRACT=ESTABLISHED
FOCUSED_H2D_DIAGNOSTIC_BASELINE_ONLY_GATE=ESTABLISHED
FOCUSED_H2D_DIAGNOSTIC_FOUR_CELL_INVENTORY=ESTABLISHED
FOCUSED_H2D_DIAGNOSTIC_PREDICTION_INDEX_0_1_COVERAGE=ESTABLISHED_LOCALLY
FOCUSED_H2D_CROSS_BATCH_PROMPT_IDENTITY=ENFORCED
FOCUSED_H2D_CROSS_BATCH_PREDICTION_IDENTITY=ENFORCED
FOCUSED_H2D_DIAGNOSTIC_DEPENDENCY_LIGHT_TESTS=99_PASSED
FOCUSED_H2D_DIAGNOSTIC_ADJACENT_REGRESSIONS=169_PASSED
FOCUSED_H2D_PREFILL_CONTEXT_BLOCKER=FIXED_LOCALLY
FOCUSED_H2D_32K_AUTHORITY_GATE_TESTS=35_PASSED_CPU_TORCH
FOCUSED_H2D_PLAN_LISTED_CUDA_EXTENSION_TESTS=2_COLLECTION_BLOCKED
FOCUSED_H2D_DIAGNOSTIC_GPU_LAUNCH_SURFACE=ABSENT
FOCUSED_H2D_DIAGNOSTIC_CODE_ENFORCED_AUTHORIZATION=ABSENT
FOCUSED_H2D_DIAGNOSTIC_AUTHORIZATION_BOUNDARY=OPERATIONAL_ONLY
FOCUSED_H2D_SOURCE_MANIFEST_PLAN_CONFORMANCE=ESTABLISHED
FOCUSED_H2D_FROZEN_AUTHORITY_SOURCE_CLOSURE_CANDIDATE=126_FILES
FOCUSED_H2D_CONSERVATIVE_PRODUCER_SOURCE_CLOSURE_CANDIDATE=128_FILES
FOCUSED_H2D_CONSERVATIVE_AUTHORITY_SOURCE_CLOSURE_CANDIDATE=129_FILES
FOCUSED_H2D_FROZEN_32K_WORKER_SOURCE_BINDING=MISSING
FOCUSED_H2D_COMPLETE_PRODUCER_PROVENANCE=NOT_ESTABLISHED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=NOT_ESTABLISHED
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
```

## 6. Single Next Critical Path

The next critical path is diagnostic, not optimization:

1. retain the existing local **focused GPU synchronization diagnostic** for
   the ordinary TP4/32K baseline without executing it. Before an authorized
   run, explicitly resolve or accept the operational-only authorization and
   incomplete frozen-worker source-binding boundaries above. The frozen
   authority CLI, schema, prompts, proposal length, and parity rules must
   remain unchanged;
2. run `baseline:b1` and `baseline:b4` first, because the first retained
   divergence is ordinary decode rather than native-MTP verification;
3. capture source-bound evidence for the first decode window, including:

   ```text
   logical block -> physical slot assignment
   H2D destination slot and copy-stream event
   prior current-stream read completion for that slot
   whether H2D begins before the prior read is ordered complete
   batch-dependent first hazardous window and overwrite count
   compact target logits at prediction indices 0 and 1
   PyTorch, CUDA runtime, and NVIDIA driver versions
   ```

4. use one diagnostic-only synchronization control to test causality. A
   positive result requires both observed unsafe overlap in the original path
   and removal of the prediction-index-1 batch-shape drift when that overlap
   is prevented; a serialized control alone is not sufficient proof, and the
   historical output-length-only `gate_pass` must not be used as an exact
   parity signal;
5. if H2D slot reuse is confirmed, design the smallest shared correction and
   regression matrix for blockwise decode, spec verify, and prefill. The
   resident-D2H rewrite dependency remains a secondary correctness item;
6. if the H2D hypothesis is rejected or does not fully explain the divergence,
   then, after separate approval, use the independent paired-trace driver and
   runner for:

   ```text
   baseline:b1
   native_mtp:b1
   baseline:b4
   native_mtp:b4
   ```

7. capture the first aligned divergence across target logits, logical
   target-KV coverage, and Qwen3.5 side-state lineage;
8. rerun the frozen TP4/32K authority unchanged after any evidence-grounded
   correction; and
9. proceed to controlled native-MTP performance and the independent learned
   drafter only after TP4/32K exact parity is green.

Neither the focused GPU synchronization diagnostic nor its diagnostic-only
control is approved. The paired diagnostic driver/runner also remains
separately unapproved. No remote, GPU, or NCCL run should be started, no
runtime fix should be implemented, and the ordinary authority CLI/schema
must remain unchanged until approval.

```text
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=NOT_ESTABLISHED
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
PAIRED_TRACE_REMOTE_DIAGNOSTIC=NOT_APPROVED
```

## Final Classification

```text
GENERIC_NGRAM_CORE_AND_COVERAGE=ESTABLISHED_WITHIN_RECORDED_SCOPES
NATIVE_MTP_TP1_4K=ESTABLISHED
NATIVE_MTP_TP4_4K=ESTABLISHED
NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD=ESTABLISHED
NATIVE_MTP_TP4_32K=NOT_ESTABLISHED
TP4_32K_FIRST_DIVERGENCE_ARTIFACT=NOT_ESTABLISHED
TP4_32K_ROOT_CAUSE=NOT_ESTABLISHED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=NOT_ESTABLISHED
CONTROLLED_NATIVE_MTP_PERFORMANCE=NOT_ESTABLISHED
SECOND_LEARNED_STRUCTURE_TP1_CORRECTNESS=NOT_ESTABLISHED
PROPOSAL_KV_OFFLOAD=NOT_ESTABLISHED
LEARNED_DRAFTER_TP4_LOADED_EXECUTION=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
KV8_KV4_PROMOTION=NOT_ESTABLISHED
KV4_KV8_RETAINED_EXECUTION_ARTIFACT=ABSENT
HEAT_TIER_RETAINED_EXECUTION_ARTIFACT=ABSENT
FUSION_RETAINED_EXECUTION_ARTIFACT=ABSENT
TP_OVERLAP_FUSION_RETAINED_ARTIFACT=ABSENT
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
