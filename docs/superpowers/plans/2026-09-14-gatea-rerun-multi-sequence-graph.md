# GATE A, rerun on the multi-sequence graph path

2026-09-14

## One-line result

The 39.8 ms constant that killed the latent KV line was mostly this engine's
uncaptured decode path, not hardware. On a path where decode batches above one
actually replay a captured graph, the constant is **11.74 ms** against Stage 0's
assumed 13.05 ms, and the token term is **0.161 us/token** against 0.151 assumed.
Stage 0's cost model was approximately right. The gate that appeared to refute it
was measuring the harness.

## What was wrong

`tinyvllm/engine/model_runner.py` refused to replay a captured graph for any
decode batch above one:

```python
# FlashAttention decode replay is only correctness-validated for one
# sequence. Multi-sequence captured graphs can corrupt rows after the
# first one, so keep the batch-1 graph fast path and fail closed to
# eager execution for larger decode batches.
multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
```

The opt-in replacement, `multi_sequence_cuda_graphs`, exists and was off. So
every GATE A cell with B >= 2 — that is, every cell the capacity argument depends
on — ran uncaptured and paid a per-step Python and launch cost that a production
engine does not. The tell was in the first artifact and was not read as one: a
weight-bandwidth roofline for Qwen3-8B bf16 on an A100 sits near 8.5-11 ms, the
batch-1 graph measured 12.98 ms, and the fitted constant was 39.8 ms. Roughly
27 ms was unexplained, and the explanation was the harness.

Turning the flag on was not sufficient. Two further faults surfaced, both found
by instrumenting dispatch rather than by reading step times:

1. **Every capture failed.** `_capture_exact_multi_sequence_graph()` recorded a
   receipt named `hot_path_eager_prerequisite` without performing it. The freshly
   allocated static buffers miss torch.compile's guards, so the first forward
   through them recompiles, and dynamo reads the CUDA RNG state while compiling,
   which is illegal mid-capture: *"Cannot call
   CUDAGeneratorImpl::current_seed during CUDA graph capture"*. The legacy
   `capture_cudagraph()` runs exactly this warmup forward for this reason. Fixed
   in `3c3b6b6a` by performing the warmup forward before entering the capture
   region; the scratch KV slots it writes are already snapshotted and restored.

2. **The capture budget silently decided the measurement.** With captures
   working, batch 2 was rejected post-capture with `single_capture_budget`,
   because the first capture in a process also pays compilation and overran the
   2 s default. Batch 4 and 8 captured fine. That left batch 2 at 31.4 ms next to
   batch 4 at 4.8 ms on the 0.6B smoke, which reads as a batch-scaling cliff and
   is in fact a policy. One-time capture cost is not what GATE A measures, so the
   worker lifts the budgets and records that it did.

Neither fault announces itself in a step time. Both were caught because the
worker now labels every decode step with how the engine dispatched it, using the
event's `step_id` so that a stale event is reported as `unpublished` rather than
inherited from a prefill step.

## Measurement

`experiments/kvcapacity_step_scaling/step-scaling-measure-msgraph-warm-20260914-200930`,
Qwen3-8B on one A100 80GB, 24 warmup and 24 measured steps per cell, independent
random prompts, only steps whose observed running batch equalled the target.

Dispatch audit: **every B >= 2 cell replayed a captured graph for 24/24 measured
steps.** B = 1 cells report `unpublished`, which is the legacy batch-1 graph fast
path that does not emit an event.

| L | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| 8192 | 13.97 | 16.51 | 17.81 | 23.29 | 34.30 | 63.64 |
| 16384 | 15.18 | 17.84 | 23.53 | 34.38 | 55.62 | |
| 32768 | 17.69 | 22.99 | 33.89 | 56.99 | | |
| 40448 | 18.62 | 25.24 | 39.11 | | | |

Fits on B >= 2:

```text
M1  step_ms = c0 + c1*L*B                 c0 = 11.741 ms  c1 = 0.1768 us/tok  R^2 = 0.9854
M2  step_ms = c0 + a*B + c1*L*B           c0 = 11.792 ms  a = 0.2383 ms/seq   c1 = 0.1607 us/tok  R^2 = 0.9937
M3  step_ms = c0 + c1*L*B + c2*(L*B)^2    c0 = 13.529 ms  c1 = 0.1394 us/tok  c2 = 1.27e-10       R^2 = 0.9878
```

Against the frozen Stage 0 inputs: `c0` ratio **0.90**, `c1` ratio **1.17** (M1)
or **1.06** (M2). For comparison, the same session's eager path reproduced the
old artifact at `c0 = 40.44 ms`, confirming the earlier number was real and
path-specific rather than a mistake in the harness arithmetic.

## Verdict: still FAIL, for a different and much smaller reason

```text
[PASS] m1_r_squared          0.9854 against 0.98
[FAIL] collision_consistency 3/4 equal-product groups agree within 10%
[PASS] batch_term_share      12.4% against 20%
[FAIL] curvature_share       14.8% against 10%
[PASS] window_stability      14/14
[PASS] sample_dispersion     18/18
[PASS] grid_coverage         18/18
[PASS] batch_coverage        5 distinct batches
```

Both failures trace to one cell. At `L*B = 262144` the three equal-product cells
are 63.64 (8192,32), 55.62 (16384,16) and 56.99 (32768,8) ms, a 13.6% spread,
while the three smaller product groups agree within 0.1%, 2.3% and 1.4%. M2
identifies the mechanism: at fixed resident tokens, sequences cost about
0.24 ms each, and M2 removes 57% of M1's residual. M3's "curvature" is the same
effect mis-specified — batch correlates with `L*B` inside a fixed-L series, so an
omitted `a*B` term reappears as a bend.

So the honest statement is: **`step = c0 + a*B + c1*L*B` describes this engine
well; Stage 0's `c0 + c1*L*B` is a good approximation that breaks at 13% by the
32-sequence corner.** Stage 0 is not refuted in the way the first run claimed. It
is refuted in the way a first-order model usually is.

## Correctness: the graph path is not fast-and-wrong

A faster path that returns different tokens would void every number above, and
the comment that installed the eager fallback predicted exactly that failure
("can corrupt rows after the first one"). So the same prompts were decoded
greedily on both paths and the token ids diffed:

- `msgraph-equivalence-20260914-202340`: Qwen3-0.6B, L=1024, B=4, 32 tokens.
  28/31 decode steps replayed a graph. **All 4 sequences identical.**
- `msgraph-equivalence-8b-20260914-202615`: Qwen3-8B, L=8192, B=8, 16 tokens.
  12/15 decode steps replayed a graph. **All 8 sequences identical.**

This does not prove the path is correct for every shape. It does mean the
regime GATE A measured produces the same tokens as eager.

## Repricing: what this does to the latent directions

At the largest measured operating point the fixed cost is no longer the story:

| operating point | step | KV-proportional term (M2) | share |
|---|---|---|---|
| L=8192, B=32 | 63.6 ms | 42.1 ms | 66% |
| L=32768, B=8 | 57.0 ms | 42.1 ms | 74% |
| L=8192, B=8 | 23.3 ms | 10.5 ms | 45% |
| L=8192, B=2 | 16.5 ms | 2.6 ms | 16% |

Consequences:

1. **The "latency axis is dead" conclusion is withdrawn.** It was derived from a
   step whose 40 ms floor left KV bytes looking like a rounding error. With an
   11.8 ms floor, KV-resident-token work is two thirds to three quarters of a
   decode step at a realistic long-context, high-concurrency operating point. A
   4x KV reduction removes about 32 ms of a 64 ms step — roughly 1.7x decode
   throughput at the same batch, *plus* 4x the resident capacity.

2. **The 3.59x capacity ceiling is void.** Both wall sweeps
   (`step-scaling-sweep-20260913-202001`,
   `step-scaling-sweep-wall-20260913-225202`) ran on the eager path, where a
   ~28 ms per-step overhead flattens throughput and manufactures an early
   plateau. The capacity axis has to be re-priced on the graph path before any
   number from those runs is quoted again. The sweep tooling needs one change
   first: the allowlist and the static/reserved capture budgets have to cover
   batches like 128 and 144, and it is not yet known whether captures at that
   width fit.

3. **The baseline latent must beat gets stronger, not weaker.** If KV bytes are
   now worth 42 ms at the corner, then `kv_quant_bits=8` and `=4` — already in
   this repository — collect much of that prize for a fraction of the work of a
   low-rank latent retrofit. Any MLA/CARE-style proposal must be priced against
   quantised KV at the same operating point, not against bf16 KV.

4. **The token-count directions are unaffected in their own right but change
   rank.** Cartridges-style context compression still removes prefill *and* KV
   residency, which is strictly more than KV-byte compression removes. But its
   GATE 0 (does the workload reuse a long context enough times to amortise
   offline self-study?) is a workload question, not an engine question, and the
   engine question just moved a long way in favour of the KV-byte line.

## What could still be wrong with this result

- The equivalence check covers two shapes, not the whole grid. A corrupted row at
  B=32 or at a page-table width not exercised here would not have been caught,
  and B=32 is exactly the cell that carries both remaining check failures.
- The lifted capture budgets are a measurement decision. They are legitimate for
  pricing steady-state serving and illegitimate for claiming this configuration
  is production-ready; a real deployment pays capture cost on every new identity,
  and identity includes page-table width, which grows as sequences decode.
- `c1` still disagrees with Stage 0 by 6-17% depending on the model form, and the
  0.24 ms/seq term is real. Any capacity arithmetic should be refit on M2 rather
  than reusing Stage 0's two constants.
- One box, one model, one tensor-parallel degree.

## Next step, in order

1. Re-run the concurrency wall sweep on the graph path, after extending the
   allowlist and capture budgets to the batches it needs. Until that exists,
   there is no defensible number for what KV bytes buy in throughput.
2. Refit the Stage 0 capacity arithmetic on M2 constants
   (`c0 = 11.79 ms`, `a = 0.238 ms/seq`, `c1 = 0.161 us/token`).
3. Then, and only then, choose between the KV-byte line (now with a real prize,
   and a strong cheap baseline in KV quantisation) and the token-count line
   (Cartridges-style, gated on workload reuse).
