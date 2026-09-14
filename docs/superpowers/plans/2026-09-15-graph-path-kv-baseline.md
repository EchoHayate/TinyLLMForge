# The graph-path KV baseline, and what it does to the capacity thesis

Date: 2026-09-15
Branch: `feat/kv-sparse-attention`
Model: Qwen3-8B, A100 80GB, bf16 weights, TP1, torch 2.4.1+cu121, flash_attn 2.6.3

## Why this run exists

GATE A was rerun on a real multi-sequence CUDA graph path and put the decode
constant back at ~11.7 ms instead of the ~40 ms the eager fallback had been
charging (see `2026-09-14-gatea-rerun-multi-sequence-graph.md`). That voided the
earlier "capacity ceiling ~3.59x" number, because that ceiling had been computed
against a 40 ms constant. The wall sweep had to be redone on the graph path
before any KV-compression variant could be priced against it.

This document is that baseline, plus the first compression arm measured against
it, plus three harness defects the run exposed.

## What "baseline" means here

- **Graph path.** `multi_sequence_cuda_graphs=True`, and every measured cell
  reports `dispatch_measured.graph_share = 1.0`. Per-step dispatch labels exist
  precisely so an eager fallback cannot be read as a graph measurement.
- **bf16 KV, no compression.** The thing to beat.
- **Pinned KV pool: 640 blocks = 163,840 tokens.** Not derived from free memory.
  See "The pool had to be pinned" below.
- 24 warmup steps, 24 measured steps, median step time, batch verified stable at
  the target for every measured step.

Artifact: `experiments/kvcapacity_step_scaling/step-scaling-baseline-pool640-20260914-233549`

```
L=2048                        L=8192
  B   step_ms     seq/s         B   step_ms     seq/s
  8    15.995    500.16         4    17.893    223.55
 16    18.711    855.10         8    23.554    339.64
 32    25.873   1236.83        12    28.563    420.13
 48    30.250   1586.77        16    34.225    467.49
 64    37.138   1723.29        19    39.358    482.74
 70    40.148   1743.54

step_ms = 12.839 + 0.383*B  R^2=0.9959    step_ms = 12.113 + 1.407*B  R^2=0.9977
peak 1743.54 seq/s at B=70 (wall)         peak 482.74 seq/s at B=19 (wall)
```

Tool reading: **CAPACITY WEAK** — "throughput still rises but the last sequences
added far less than the first". At L=2048 the last sequences added 7.6% of what
the first ones added; at L=8192, 17.5%. Scaling efficiency across the sweep is
0.40 and 0.45 of proportional.

## The prize on the capacity axis is now small

Capacity compression buys batch, and batch is priced by the fit above. Holding
the fit fixed and asking what more batch is worth:

| | L=2048 | L=8192 |
|---|---|---|
| at the wall | B=70, 1744 seq/s | B=19, 483 seq/s |
| 2x KV capacity | B=140, 2107 seq/s = **1.21x** | B=38, 579 seq/s = **1.20x** |
| 4x KV capacity | B=280, 2332 seq/s = 1.34x | B=76, 638 seq/s = 1.32x |
| 8x KV capacity | B=560, 2463 seq/s = 1.41x | B=152, 673 seq/s = 1.39x |
| **infinite** compression | 2611 seq/s = **1.50x** | 711 seq/s = **1.47x** |

The asymptote is `1000/slope`, and it does not depend on the pool: the pool only
decides where on the curve you currently sit. So **the entire remaining prize on
the capacity axis is about 1.5x, and a realistic 2x KV compression buys ~1.20x.**

Note the direction of the correction. Fixing the engine overhead (the right thing
to do for production realism) *shrank* this prize rather than growing it. A large
constant is what makes added concurrency look valuable, because concurrency
amortises it. With the constant at 12 ms instead of 40 ms, the per-sequence slope
dominates almost immediately, so the capacity axis saturates earlier. The voided
3.59x was partly an artifact of measuring against a slow engine.

### Cross-validation against an independent run

The 640-block fit was checked against `step-scaling-sweep-wall-msgraph2-20260914-231013`,
a separate run on a different device with an auto-sized 1193-block pool, i.e. 1.9x
more capacity and different cells:

| cell | predicted | measured | error |
|---|---|---|---|
| L=2048 B=32 | 25.09 ms | 26.06 ms | -3.7% |
| L=2048 B=64 | 37.35 ms | 36.65 ms | +1.9% |
| L=2048 B=96 | 49.61 ms | 51.27 ms | -3.3% |
| L=2048 B=128 | 61.86 ms | 62.89 ms | -1.6% |
| L=8192 B=16 | 34.62 ms | 34.63 ms | -0.0% |
| L=8192 B=32 | 57.14 ms | 62.26 ms | -8.2% |

Five of six within 4% across an independent run with a different pool size. The
constant and slope are properties of the model and hardware, not of the pool.

### Where the slope comes from

Fitting `slope(L) = a + c1*L` across the two contexts:

```
slope(L) = 0.0417 ms/seq + 0.1667 us/token * L
  L=2048: per-token term is 89% of the slope
  L=8192: per-token term is 97% of the slope
```

`c1 = 0.167 us/token` agrees with GATE A's M2 fit (0.161 us/token) from a
different grid. So the per-sequence cost is almost entirely per-token KV work.
That is the interesting reading: **the money is in the bytes-per-token term, not
in holding more sequences.** Compression that removes bytes per token attacks
0.167 us/token directly; compression that only frees capacity gets the <=1.5x
above.

## The first compression arm: int8 KV, falsified hard

Pre-registered before looking: at a pinned pool, int8 holds the token count
constant and removes only bytes, so if the slope is KV-memory-traffic bound then
halving KV bytes should roughly halve the per-token term, predicting **-30% at
L=2048 B=70 and -33% at L=8192 B=19.**

Measured (`step-scaling-kv8-pool640b-20260914-235737`, same pinned 640-block
pool, same grid, `graph_share=1.0` on every measured cell):

| | constant | slope | at the wall | throughput |
|---|---|---|---|---|
| L=2048 bf16 | 12.84 ms | 0.383 ms/seq | 40.1 ms | 1744 seq/s |
| L=2048 int8 | 14.48 ms (+13%) | 3.364 ms/seq (**8.8x worse**) | 250.3 ms (6.2x) | 280 seq/s (**0.16x**) |
| L=8192 bf16 | 12.11 ms | 1.407 ms/seq | 39.4 ms | 483 seq/s |
| L=8192 int8 | 14.36 ms (+19%) | 12.327 ms/seq (**8.8x worse**) | 249.3 ms (6.3x) | 76 seq/s (**0.16x**) |

The prediction was not merely wrong in magnitude, it was wrong in sign. And the
damage is almost entirely in the per-sequence slope while the constant barely
moves, which is the signature of a per-sequence transient buffer rather than of
saved bandwidth.

### Root cause, already documented in the codebase

`tinyvllm/layers/attention.py` dequantises hit blocks into a full fp16 buffer and
then calls flash-attn on it. The repo's own comment says so, and reports the same
failure from an earlier evaluation of C4:

```
# 单独 C4：要把命中块全部 dequant 成 fp16；瞬态 buffer ~ B*max_blocks*block_size*..
#         A-3 评测显示长 ctx 下这层瞬态 buffer 把"省下的 KV 带宽"全吃了，反慢 5.4x
```

So this arm reads int8 (half the bytes), writes a bf16 copy, then reads that
copy: strictly more traffic than never quantising. My 8.8x is the KV8 analogue of
their 5.4x for KV4, measured on the graph path at a pinned pool.

**Consequence for the research line:** this engine currently has no usable
KV-byte mechanism. Any low-rank / MLA-style / latent KV variant implemented as
"materialise the fp16 KV, then attend" will hit exactly the same wall, and will
look catastrophic for reasons that have nothing to do with the idea being tested.
A fused dequant-inside-attention decode kernel (or native low-rank attention) is
a **prerequisite** for the KV-byte line, not an optimisation to add later.

Getting this now is cheap news: it would have been very expensive to discover
after training a latent KV projection.

## Three harness defects this run exposed

### The pool had to be pinned

The first graph wall sweep reported an implausibly small capacity and was refused
almost everywhere. Cause: KV capacity is derived from free memory at construction
time, and this A100 is shared. Observed on the *same* device within one hour:
foreign memory 0 -> 38 GiB, and ctx8192 capacity 343,808 -> 89,856 tokens.

Fixes landed:
- `device_memory_before_load()` records free/total/foreign share in every artifact.
- The analysis emits **CONTAMINATED** as its own reading instead of laundering the
  problem into INCONCLUSIVE ("no context length carried enough batches to judge"),
  which reads like the grid was too small. Applied retroactively it condemns both
  contaminated sweeps and leaves the clean one's reading intact. Unknown foreign
  share is treated as contaminated, not as clean.
- `--kv-blocks` pins the pool. A pinned pool either is served in full or fails at
  construction, which is the outcome worth having; it took three attempts to find
  a size the shared card would serve (1100 and 850 blocks were refused, 640 held).
- The comparison design follows from this: **all arms share one pinned pool**, so
  compression shows up as more sequences in a fixed budget rather than as each arm
  quietly getting a different budget.

### The feasibility guard was wrong in both directions

The old skip message printed naive `L*B*bytes_per_token` against raw available
bytes while the guard actually compared against `available*0.95`, producing skips
that looked self-contradictory:

```
L=2048 B=144: required 43486543872, available 45034242048   -> skipped
```

Residency is now charged the way the allocator charges it — whole KV blocks per
sequence, including generated tokens — and the cushion is a fixed 8-block
scheduler reserve rather than a percentage. The percentage was wrong twice over:
on a 1193-block pool it refused L=2048 B=128, which had already measured cleanly
with 41 blocks to spare, and it refused L=8192 B=40, which is the eager run's
wall point and the one cell the capacity ratio needs. The guard is now pinned in
tests to what the engine actually did at the wall: B=128 fits, B=140 (could not
form the batch) and B=144 (pins more blocks than exist) do not.

### A blocked capture path

`snapshot_kv_slots` refused quantised KV outright (`KV snapshot requires FP KV`).
Capture borrows scratch KV slots and must return them byte-identical, so this
silently disabled the entire multi-sequence graph path whenever `kv_quant_bits`
was set — meaning a KV-compression measurement would have become an eager
measurement. Quantised KV needs nothing conceptually new: the payload is integer,
so copying it is exact, and the only extra state is the scale tensor at the same
(block, offset) coordinates. Both are now snapshotted and restored together, and
restoring a payload-only snapshot into a quantised cache is refused rather than
silently pairing integers with foreign scales. Five round-trip tests
(`tools/test_kv_slot_snapshot_quantised.py`) pass against real tensors on the GPU
host.

## Honest caveats

- The int8 arm's **token equivalence was not rechecked**. bf16 eager-vs-graph
  equivalence passed earlier (0.6B B=4 and 8B B=8, identical token ids), and the
  new snapshot path is unit-tested for byte-exact round trips, but int8
  eager-vs-graph token equality has not been measured. The int8 conclusion here is
  about speed, and the speed result is bad enough that correctness is moot for the
  decision — but it is not evidence that int8 graph decode is correct.
- `tinyvllm/engine/model_runner.py` carried unrelated in-flight edits to the
  cohort-burst path during these runs, so `source_provenance.json` will show a
  dirty tracked path. That code is a different dispatch route and every measured
  cell reports `graph`, but the runs are not from a clean tree.
- The 640-block pool is smaller than the card can serve when idle. It was chosen
  to survive ~20 GiB of neighbours so all future arms can reuse the identical
  pool. Absolute wall positions are therefore lower than a dedicated card would
  give; the fit and the ratios are what transfer, and the cross-validation above
  shows they do.
- "CAPACITY WEAK" and the 1.5x asymptote are specific to Qwen3-8B GQA on one
  A100. A model with a fatter KV footprint per token, or a card with less
  bandwidth headroom, would sit differently on the same curve.

## What this changes

1. **The capacity axis is not worth a research project on this setup.** <=1.5x
   with infinite compression, ~1.2x for a realistic 2x. Any latent-KV proposal
   justified by "hold more sequences" should be repriced against 1.2x, not 3.59x.
2. **The bytes-per-token axis holds 89-97% of the per-sequence cost** and is where
   a real win would come from — but it is currently unreachable in this engine,
   because the only implemented KV-byte mechanism makes things 8.8x worse by
   materialising fp16.
3. So the next decision is not "which compression idea" but **"is a fused
   dequant-in-attention decode kernel worth building?"** Until that exists, every
   KV-byte arm measured here will be measuring the transient buffer, not the idea.

## Suggested next steps, cheapest first

1. **KV8 + Quest** (`quest_top_k_blocks > 0`), same pinned pool. The engine's own
   intended mitigation is to select top-k blocks and dequantise only those. If
   selective dequant recovers most of the 8.8x, the KV-byte line is open at modest
   cost; if not, a fused kernel is unavoidable. One sweep, ~10 minutes.
2. If Quest does not recover it, scope a fused int8-KV decode attention kernel and
   re-measure the same grid. That is the gate for the whole KV-byte line.
3. Only after a KV-byte mechanism that is not self-defeating exists, compare
   low-rank / MLA-style / latent KV against bf16 at the shared pinned pool.
4. Independently: the <=1.5x capacity result argues for re-examining the
   token-count axis (Cartridges-style context compression), where the win is fewer
   tokens attended rather than fewer bytes per token — that attacks `c1*L` through
   `L` instead of through `c1`, and does not need a new kernel.
