# Mechanism gates: the handshake, the selector, and the pipeline

Date: 2026-09-17
Host: `n232-195-203`, 8x A100 80GB PCIe, Xeon Platinum 8336C (64 physical cores / 128 threads),
2015 GB RAM. Load average during all runs: **103-111, 25-26 other users**. Every CPU number
below is therefore pessimistic and every GPU-under-load number is realistic.
Raw data: `experiments/mechanism_gates/mechanism-20260917-221357/`.

Later additions to this document, on a quiet box (load 3-14): the re-baseline (2026-09-18,
`experiments/mechanism_gates/mechanism-clean-20260918-170003/`), Gate 7 correctness
(`experiments/cpu_offload_correctness/cpuoffload-*`) and Gate 8, which makes the benchmarked
AVX-512 kernel the one that passes the correctness gate
(`experiments/cpu_offload_correctness/cpuoffload-avx512-*`, 2026-09-20).

The three gates before this one said: the answer survives at 2-3% of the context (gran=32,
shared-head Quest), the CPU can compute that sparse attention in 2.05 ms, and a CUDA-graph
engine only inflates 1.523 ms under a saturated CPU. What they could not say is whether the
remaining 7.18 ms of nominal headroom is collectable. Three things stood between the numbers
and a mechanism, and this document measures all three.

## Gate 4 - can the per-layer handshake be made async?

The overlap gate charged 1.355 ms/step for a synchronous per-layer Q down / output up
handshake and noted that one fused transfer of the same bytes costs 0.068 ms. A 20x gap is an
invitation, so: CUDA events, a dedicated stream, and sending Q one layer ahead.

| variant | per step | per layer | host syncs | vs device_sync |
|---|---|---|---|---|
| `device_sync` (baseline) | 1.3161 ms | 36.56 us | 72 | 1.00x |
| `event_sync` | 1.9819 ms | 55.05 us | 72 | **0.66x** |
| `stream_sync` | 1.9492 ms | 54.14 us | 72 | **0.68x** |
| `lookahead` (Q one layer early, device-side wait) | 1.9643 ms | 54.56 us | 36 | **0.67x** |
| `fused` (all 36 layers in one transfer) | 0.0673 ms | 1.87 us | 2 | 19.56x |

**Every async variant is slower than the naive one.** Halving the host syncs (lookahead: 36
instead of 72) bought nothing, which means the cost is not the number of host syncs - it is
per-transfer launch and completion overhead, and PyTorch's event/stream objects add to it
rather than hide it. Direction was also corrected here: Q is D2H (GPU produces, CPU consumes)
and the attention output is H2D; the earlier note had them swapped, though the total is
unaffected since both directions are present either way.

The `fused` number is a bound, not a plan: layer *i*'s Q does not exist until layer *i-1* has
run, so 36 layers of Q cannot be shipped in one transfer within a single microbatch. The
achievable version of `fused` is a *batched* transfer across microbatches or across kv heads,
which is a scheduling change, not an event-API change. That is the actionable reading of this
gate: **1.316 ms (10.9% of the 12.113 ms window) is the current price of coordination, and it
will not come down by writing better PyTorch. It needs either a C++/CUDA handshake or a
schedule that batches transfers.**

## Gate 5 - what does the selector itself cost?

Never measured before. gran=32 over 8192 tokens is 256 units per kv head per layer, x8 heads
x36 layers, and the shared-head Quest bound has to touch every one of them each step.
Measured separately: summary maintenance (fold the new key into unit min/max) and
scoring + top-k.

| placement | gran | dtype | scoring | maintenance | **total** | summary size |
|---|---|---|---|---|---|---|
| CUDA | 32 | fp32 | 0.2825 ms | 0.0452 ms | **0.3277 ms** | 72 MiB |
| CUDA | 32 | bf16 | 0.2128 ms | 0.0450 ms | **0.2578 ms** | 36 MiB |
| CPU x1 | 32 | fp32 | - | - | **62.85 ms** | 72 MiB |
| CPU x8 | 32 | fp32 | - | - | **18.33 ms** | 72 MiB |
| CPU x64 | 32 | fp32 | - | - | **17.46 ms** | 72 MiB |
| CPU x8 | 32 | bf16 | - | - | **10.59 ms** | 36 MiB |
| CPU x64 | 32 | bf16 | - | - | **7.86 ms** | 36 MiB |
| CUDA | 256 | bf16 | - | - | **0.1683 ms** | 4.5 MiB |
| CPU x64 | 256 | bf16 | - | - | **0.2715 ms** | 4.5 MiB |

Two conclusions, one of which reverses a design assumption:

1. **The selector cannot live on the CPU at gran=32.** 7.86 ms best case is 65% of the whole
   window and **4x the cost of the sparse attention it is supposed to enable** (2.05 ms). The
   naive mental model - "CPU owns the KV, so CPU owns the selection" - is dead in this
   implementation. Note the caveat: this is PyTorch CPU, not the AVX-512 kernel that took CPU
   attention from dumb to 2.05 ms, so a hand-written selector could plausibly close much of
   this. But nothing entitles us to assume it will; on today's evidence CPU selection is out.
2. **The GPU selector is nearly free (0.258 ms, bf16)** and the summaries it needs are only
   36 MiB for the entire model - 0.04% of an 80 GB card. Scoring runs at 177-267 GB/s, i.e.
   memory-bound as expected, which is why bf16 summaries beat fp32 by 1.27x on GPU and 2.2x
   on CPU.
   Maintenance is cheap everywhere (0.045 ms); **scoring + top-k is the whole cost.**

Architecturally this splits the mechanism cleanly: **summaries and selection on the GPU, K/V
bytes and attention on the CPU.** The GPU sends indices down instead of Q, which also shrinks
the handshake payload.

gran=256 is cheap on either side, but the fidelity gate already priced it: 25.2% of the
context to keep the answer, versus 2-3% at gran=32. Cheap selection over the wrong units is
not a saving.

## Gate 6 - does cross-microbatch pipelining collect the headroom?

The claim under test: `pipelined ~= max(gpu_only, cpu_only)` rather than `gpu + cpu`. GPU arm
is a 16 GB weight-read GEMV pass (the decode window's shape); CPU arm is the *same* C kernel
that produced 2.05 ms, linked as a shared library so the two gates cannot drift apart.

### First result was a false negative

Orchestrated from a Python thread, pipelining looked *worse* than serial (15.565 ms vs
13.423 ms, "hides -105% of the CPU time"). Two reruns with different core counts and
`--gpu-mode graph` reproduced it. The effect being measured is ~2 ms; Python thread
scheduling and GIL handoffs are the same order of magnitude, so the instrument was the
finding. Moving the async submit/wait into C (`csa_submit` / `csa_wait`, pthread + condvar,
no Python involved in the critical path) changed the answer completely.

### Then a second, sharper trap: how the host waits for the GPU

| GPU mode | host wait | gpu_only | cpu_only | serial | **pipelined** | efficiency | CPU hidden |
|---|---|---|---|---|---|---|---|
| graph | `cuda.synchronize()` (spin) | 11.300 | 2.047 | 13.351 | **13.398 ms** | 0.843 | **-2.3%** |
| graph | blocking event (sleep) | 11.306 | 2.072 | 13.417 | **11.396 ms** | 0.992 | **97.5%** |
| eager | spin | 11.323 | 2.070 | 13.396 | **11.407 ms** | 0.993 | 96.1% |
| eager | blocking event | 11.353 | 2.088 | 13.482 | **11.476 ms** | 0.989 | 96.1% |
| graph | spin, CPU capped at 63 threads | 11.289 | 2.046 | 13.350 | **13.367 ms** | 0.845 | -0.8% |

Read the first two rows together: **same graph replay, same C kernel, same thread count - the
only change is that the host sleeps instead of polls, and 0% hiding becomes 97.5%.** A
CUDA-graph engine that waits with the default spinning synchronize gets exactly serial
behaviour and would have "proved" this mechanism impossible.

Why: `torch.cuda.synchronize()` polls, so in graph mode the host thread does nothing for
~11 ms except burn a core at 100% without a single preemption point, right next to a
64-thread OpenMP team whose parallel-for ends in a barrier - the slowest thread sets the step
time. In eager mode the host is interleaving 36 kernel launches, which gives the scheduler
natural yield points, so spinning there is harmless. Capping the CPU team to 63 threads did
**not** fix the graph case - but *pinning* the host thread did:

| GPU mode | host wait | CPU team | host thread | **pipelined** | efficiency |
|---|---|---|---|---|---|
| graph | spin | 63 threads on cores 0-62 | unpinned | 13.020 ms | 0.863 |
| graph | spin | 63 threads on cores 0-62 | **pinned to cpu 127** | **11.275 ms** | **0.999** |
| graph | blocking event | 63 threads on cores 0-62 | pinned to cpu 127 | 11.324 ms | 0.997 |

cpu 127 is the SMT sibling of cpu 63, so a host thread pinned there shares no physical core
with a team bound to 0-62. That closes the causal story: **the failure is an unpinned,
never-yielding host poll loop landing on a core the OpenMP team needs**, and the barrier at the
end of the parallel-for makes the whole team pay for the one thread that got descheduled.
Merely lowering the thread count does not help because on a machine at load ~105 there is no
idle core to fall into - the spinner has to be *told* where to go, or told to sleep.

Two independent fixes therefore exist, and the engine should prefer the second:
pin the host thread off the CPU team's cores, or wait with a blocking event. Pinning depends on
knowing what else is on the machine; sleeping does not.

Two side observations from the pinned arms, both worth remembering:

- Hard-binding the CPU team (`GOMP_CPU_AFFINITY=0-62`) made the attention itself *faster*
  (1.51-1.89 ms versus 2.05 ms unpinned) but inflated the `serial` and Python-thread arms to
  18-22 ms and 32-52 ms. The leading explanation is thread wake-up: between iterations the team
  sleeps, and on hard-pinned cores under foreign load it cannot be woken promptly, while the
  back-to-back `cpu_only` loop never sleeps in the first place. A persistent, warm CPU team is
  therefore part of the mechanism, not an optimisation.
- Pushing that further with `OMP_WAIT_POLICY=active` and `GOMP_SPINCOUNT=3e7` was a disaster:
  `cpu_only` went from 2.05 ms to 33-47 ms. 63 hard-spinning threads pinned to cores that
  already carry foreign load is the worst of both worlds. Re-running the plain baseline
  immediately afterwards reproduced 2.047 ms exactly, so this was self-inflicted by the
  environment variables and not a change in the machine.

The Python-thread control arm in the same runs tracks this too: 23.5 ms with graph+spin vs
12.3 ms with graph+blocking - the GIL holder and the spinner compound each other.

### The ceiling nobody can schedule away

CPU attention cost is per sequence, while the GPU's weight read is amortised over the batch -
and raising the batch is the entire reason to offload KV:

| concurrent sequences | CPU attention |
|---|---|
| 1 | 2.161 ms |
| 2 | 4.093 ms |
| 4 | 8.187 ms |
| 8 | 16.330 ms |

Linear at ~2.04 ms/sequence, no shared work to exploit.


## Gate 7 - is it the same mechanism? (correctness)

Everything above is performance, and none of it had produced a single token through the offload
path. That is a gap worth naming: four gates of stand-ins can agree with each other and still be
measuring something that does not compute attention correctly.

`tools/cpu_offload_correctness.py` closes it. The KV cache lives in pinned CPU DRAM and is
written one token per step; the min/max summaries live on the GPU and are maintained
*incrementally*; the selector ranks units on the GPU (Gate 5) and ships **indices** down; the
CPU gathers the selected units out of its own mirror and computes the attention in fp32; the
output goes back up. The selection math is imported from `tools/e2e_sparse_attention.py` rather
than reimplemented, because a correctness harness that quietly disagrees with the fidelity
harness proves nothing.

Three failures are possible and they are reported separately, because only one of them is a bug:
a different selection (incremental-maintenance bug), different arithmetic (gather/attention bug),
or the same selection and arithmetic with a different continuation (that is the fidelity story,
already measured).

Run: Qwen3-8B's *per-layer shape* (36 layers, 32 q heads, 8 kv heads, head_dim 128), 8192
context, gran=32, k_frac=0.051, 16 decode steps, every sparse call verified.

| check | result | bound |
|---|---|---|
| incremental summaries vs from-scratch recompute | **0.0** drift, 560 checks | must be exactly 0 |
| selection from maintained vs recomputed summaries | **0 mismatches / 560** | must be 0 |
| CPU fp32 output vs GPU **fp32** with identical indices | **max 1.209e-06** (mean 7.3e-07) | < 1e-4 |
| CPU fp32 output vs GPU **bf16** with identical indices | max 3.924e-03 (mean 2.4e-03) | context: this is bf16 |
| tokens vs the GPU sparse arm | **1.0000 agreement**, no divergence | must be 1.0 |
| context touched | 5.18% | k_frac was 0.051 |
| traffic | D2H 1154 MiB (1.125 GiB prefill KV + 1 token/step), H2D 4.38 MiB | |
| **verdict** | **PASS** | |

So the offload path computes the same attention the GPU computes, selects the same units the
selector would select from a recomputed summary, and produces the same tokens as the GPU sparse
arm - **a token has now come out of the CPU.** The bf16 column is the useful context: the CPU's
fp32 output differs from the GPU's bf16 output by 3.9e-03, i.e. by exactly the noise the model
already runs on, and it differs from the GPU's fp32 output by 1.2e-06.

Two honest limits on this run:

- **Weights are random.** The checkpoint is no longer on the box, so the harness builds a Qwen3
  with the right per-layer shape and meaningless weights. That is legitimate here and only here:
  every check compares two paths through the *same* weights. It is not legitimate for fidelity,
  and the "tokens vs dense" number this run prints (0.0588) is therefore meaningless - with
  random weights there is no answer to preserve. Answer-level fidelity was measured separately
  with real weights.
  (This bullet originally blamed `/data00` being 96% full. That was the wrong device - see the
  correction dated 2026-09-20 below. The checkpoint's absence is real; the explanation was not.)
- **This is a correctness harness, not a fast one.** It computes fp32 references and recomputes
  summaries on every sparse call. The 3.8 s for 16 steps says nothing about the mechanism's
  speed; the gates above are where speed lives.

Test coverage: `tools/test_cpu_offload_correctness.py`, 14 tests, model-free and CPU-only. The
weight of them sits on the incremental min/max maintenance - a unit that is starting must not
inherit the +-inf sentinel, and a half-filled unit must not be widened by tokens it does not yet
hold. That bug does not crash and does not produce NaNs; it produces a slightly different
selection, which would then be misread as "sparse attention costs a little quality" - the most
expensive kind of bug in this project, since the entire fidelity gate is denominated in exactly
those units.


## Re-baseline on a quiet machine (2026-09-18)

Every number above was measured at load 103-111. The box went quiet on 2026-09-18 (load 8-14),
so all three gates were re-run with the engine-relevant configuration (`--gpu-mode graph`,
blocking wait). This was the first open question from the day before, and the answer is mostly
boring, which is the useful kind of boring:

| measurement | at load ~105 | at load ~10 | moved? |
|---|---|---|---|
| handshake `device_sync` | 1.3161 ms | 1.3074 ms | no |
| handshake `fused` bound | 0.0673 ms | 0.0672 ms | no |
| handshake `event/stream/lookahead` | 1.95-1.98 ms | 1.92-1.94 ms | no, still worse than naive |
| selector, GPU gran=32 bf16 | 0.2578 ms | 0.2597 ms | no |
| selector, CPU gran=32 bf16 x64 | 7.858 ms | 8.073 ms | no |
| **CPU sparse attention** | 2.05 ms | **1.676 ms** | **yes, -18%** |
| pipeline efficiency (graph + blocking) | 0.992 | 0.99 | no |
| CPU time hidden | 97.5% | 96.9% | no |
| per-sequence CPU cost | 2.04 ms | 1.86-1.91 ms | tracks the attention number |

Two things changed and neither changes a conclusion:

1. **CPU attention is 18% faster on a quiet machine** (2.05 -> 1.676 ms). Every CPU-side budget
   number in this document was therefore pessimistic, not optimistic - the direction that does
   not invalidate anything.
2. **The Python-thread arm stopped being a false negative** (11.614 ms, 87.3% hidden, versus
   17-23 ms under load). This retroactively confirms the earlier diagnosis: that arm was
   measuring contention, not orchestration. The C-level submit is still better and is still the
   right design, but the reason the Python version looked catastrophic was the machine.

Everything structural survived a 10x change in background load: the handshake is 1.31 ms, the
CPU selector in torch is unusable at gran=32, and the per-sequence CPU cost is linear in batch.

### One finding the clean run sharpened: the CPU selector is implementation-bound, not hardware-bound

The selector gate now reports achieved bandwidth, and the gap is the whole story:

| placement | gran=32 bf16 | achieved bandwidth |
|---|---|---|
| CUDA | 0.2597 ms | **176 GB/s** |
| CPU x64 | 8.073 ms | **4.71 GB/s** |
| CPU x8 | 10.736 ms | 3.53 GB/s |
| CPU x1 | 33.570 ms | 1.13 GB/s |

4.7 GB/s on a dual-socket Xeon is roughly **2% of what the machine's DRAM can do**, and the
scoring is a pure streaming min/max-bound reduction - the most bandwidth-friendly shape there is.
So "the CPU selector costs 8 ms" is a statement about PyTorch's CPU reduction path, not about the
CPU. The earlier verdict needs softening in exactly one word: CPU-side selection is dead **in
torch**, and a hand-written AVX-512 selector - the same move that took CPU attention from dumb to
1.68 ms - has one to two orders of magnitude of headroom to reclaim. That is now a worthwhile
experiment rather than a consolation prize, because keeping selection on the CPU would remove
both the GPU-side summaries and the index transfer.

### Defaults changed, because a default is a finding

`tools/microbatch_pipeline_prototype.py` now defaults to `--gpu-wait blocking_event`. It used to
default to `spin`, which meant running the prototype with no arguments reproduced the broken
result (-2.3% of the CPU time hidden) - a trap to hand to the next reader, including a future me.
`--pin-host-core` stays opt-in at `-1`, because pinning requires knowing what else runs on the
machine while sleeping does not. Both defaults are now pinned by tests in
`tools/test_new_gates.py`.

### Correctness is reproducible

Gate 7 was re-run twice with different seeds *and* a different granularity, since repeating one
configuration would only prove determinism:

| run | seed | gran | k_frac | summary drift | selection | rel err vs fp32 | tokens vs GPU sparse |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 32 | 0.051 | 0.0 | 0/560 | 1.209e-06 | 1.0000 |
| 2 | 1 | 32 | 0.051 | 0.0 | 0/560 | 1.202e-06 | 1.0000 |
| 3 | 2 | **256** | 0.11 | 0.0 | 0/560 | 1.546e-06 | 1.0000 |

All PASS. The gran=256 run matters more than the reseeded one: it exercises the unit arithmetic
with 32 units instead of 256, including a differently shaped partial last unit.

### Correction to yesterday's note about background load

Yesterday this document blamed part of the load on two of the user's own 8-day-old processes
(about 10.5 cores). They are **still running** (now 9 days) and the load still fell from ~105 to
~9, so they were never the main contaminant - other tenants' jobs were. The earlier note
overstated their role.

## Gate 8 - the verified path and the fast path are now one path (2026-09-20)

Gate 7 proved the mechanism computes the right thing. It did so with `torch.nn.functional` on the
CPU, while the 2.051 ms/step that makes this route worth considering at all was measured by the
AVX-512 C kernel. Read carefully, that left two claims about two different programs: *a* CPU
implementation is correct, and *another* CPU implementation is fast. Nothing had ever been both.

The kernel is now callable on the harness's own buffers (`csa_attend_external` in
`tools/cpu_sparse_attention_lib.c`) and `tools/cpu_offload_correctness.py --cpu-impl` chooses
which implementation produces the tokens. Two deliberate decisions:

- **One inner loop, not two.** `attend_head` in the benchmark was refactored into
  `attend_head_strided`, which takes the KV layout as strides. The benchmark models the cache as
  `[S, KVH, D]`; the harness mirrors the engine's real cache as `[1, KVH, S, D]`. A second copy of
  the loop for the second layout is exactly how a measured kernel and a verified kernel drift
  apart, so the layout is a parameter and the arithmetic is shared. The benchmark's own self-test
  still passes (max abs err 3.5e-08), which is what says the refactor changed nothing.
- **No staging gather on the C path.** The torch path calls `mirror.gather()` and materialises a
  `[1, KVH, T, D]` tensor; the C path gets base pointers, strides and absolute token indices and
  gathers while the line is hot - which is what the 2.05 ms was measured doing. Handing the
  kernel a pre-gathered buffer would have verified a kernel nobody benchmarked.

The gate gained a fourth check that only exists when the C kernel ran: the fast kernel against
the torch path that was accepted first. Check (c) alone is not enough - a wrong kernel can pass a
comparison against a wrong reference by being wrong in the same direction - and a clause that is
vacuously true when the kernel did not run would read like evidence, so it is absent instead.
`--cpu-impl both` additionally runs the decode loop twice and compares the *token streams*, since
a discrepancy that only appears between verification steps would still change the continuation.

Run on the quiet box (load 3.5), `gcc -O3 -march=native -fopenmp`, `__AVX512F__` confirmed
defined in the build - so the vector path, not the scalar fallback, is the one that ran:

| config | fast kernel vs torch CPU | vs GPU fp32 | tokens avx512 vs torch | tokens vs GPU sparse | verdict |
|---|---|---|---|---|---|
| seed 0, gran=32, k=0.051 | **max 1.810e-06** | max 1.880e-06 | **1.0000** | 1.0000 | PASS |
| seed 3, gran=32, k=0.051 | **max 1.853e-06** | max 1.822e-06 | **1.0000** | 1.0000 | PASS |
| seed 5, gran=256, k=0.11 | **max 2.431e-06** | max 2.380e-06 | **1.0000** | 1.0000 | PASS |

Summary drift 0.0 and 0 selection mismatches out of 560 in all three, as before. The six-clause
verdict now reads: summaries exact, selection identical, arithmetic matches fp32, tokens match the
GPU sparse arm, **fast kernel matches the torch CPU path**, **both CPU kernels produce the same
tokens**.

So the sentence that was not previously available is available now: the kernel that was timed at
2.051 ms/step is the kernel that produces correct tokens through a CPU-resident KV cache.

Three things this run does **not** say, stated because the numbers invite the opposite reading:

- **It is not a speed measurement.** The harness calls the kernel one layer at a time and
  synchronously, so the only parallelism available is over 8 kv heads; the 2.051 ms figure came
  from `collapse(2)` over 36 layers x 8 heads. The AVX arm's shorter wall time (2.85 s vs 3.68 s
  for the torch arm, while doing *more* work - it also computes the torch reference for the new
  check) is an observation, not a result.
- **Weights are still random.** The Qwen3-8B checkpoint is still gone from the box, so the same
  reasoning as Gate 7 applies: every check compares two paths through the same weights, which is
  legitimate for correctness and illegitimate for fidelity. The "tokens vs dense 0.0588" line
  remains meaningless.
- **fp32 accumulation is what was verified.** The kernel reads bf16 storage and accumulates in
  fp32, and it refuses non-bf16 buffers rather than reinterpreting them (there is a test for
  that, because the failure would be silent garbage). An int8/VNNI variant - the obvious next
  speed step - is a different kernel and would need this gate re-run.

Test coverage went from 14 to 22. The eight new ones pin the handover rather than the arithmetic,
because that is where a wrong answer would come from: capacity-based strides (not live-length),
the q-head-to-kv-head mapping, bf16 refusal, a missing `.so` failing before the prefill instead of
40 layers into it. They are real: with correct strides the kernel agrees to 1.3e-07, and the three
plausible bugs - live-length head stride, swapped K/V, q heads rolled by one - all come out at
O(1) relative error.


## Correction: the 97% disk was never our disk, and never the blocker (2026-09-20)

Two earlier notes in this document explained the missing Qwen3-8B checkpoint by pointing at
`/data00` being 96-97% full. That reasoning used the wrong device, and it mattered, because it
turned a 10-minute download into something that looked like it had to wait for someone to clean
a shared volume.

`/data00/home/sitian` is **not** part of `/data00`. It is a separate block device mounted over
that path:

```
/dev/nvme0n1p3  1.7T  1.5T   62G  97% /data00              <- the host disk, shared, nearly full
/dev/nbd16      2.9T  2.6T  207G  93% /data00/home/sitian  <- our home, its own device
```

So the 97% figure describes a volume this work does not write to. Everything the gates produce -
`/data00/home/sitian/tllm/...`, the staged sources, the JSON - lands on nbd16, which has **207 GB
free**. A Qwen3-8B bf16 checkpoint is ~16 GB. **Disk space was never what stopped the real-weights
run**; the checkpoint is simply absent (the whole `.ms_cache` directory is gone, not just
`Qwen/`), and re-fetching it is unblocked right now.

Three things found while checking, worth writing down because each one fails in a confusing way:

- **Our own volume is at 93%, 207 GB left, and there is no quota** (`quota -s` reports none). So
  nothing stops a run from filling it, and the failure mode is a write error mid-experiment
  rather than a refusal up front. Current top consumers: `pypilot_workspace` 928G, `RL` 601G,
  `tllm` 189G, `tinyllmforge-workspaces` 169G, `models` 141G. The per-run staging directories
  under `tllm/` and `tinyllmforge-workspaces/` are the cheapest 350 GB to reclaim, since each is
  a disposable copy of a source tree.
- **Two filesystems are stacked on the home path**: `/dev/nbd2` mounted **ro**, with `/dev/nbd16`
  mounted **rw** on top. The rw one wins today, so this is invisible. If nbd16 ever fails to
  mount, the path still exists and is still readable - it just silently becomes read-only, and
  the failure will read like a permissions bug rather than a mount problem. Worth recognising
  the shape before losing an hour to it.
- **`df -h /data00` is the wrong command for this question** and answered it wrongly for three
  days. `findmnt -T <path>` is the right one: it names the device actually serving that path.


## Revised budget, one sequence, 8192 context, gran=32

| item | cost | note |
|---|---|---|
| GPU weight-read window | 12.113 ms | what we have to hide inside |
| engine inflation under saturated CPU | 1.523 ms | measured, CUDA-graph engine |
| per-layer handshake | 1.307 ms | Gate 4; async did not help, load-independent |
| GPU selector (bf16 summaries) | 0.260 ms | Gate 5 |
| CPU sparse attention | 1.676 ms | **~97% hideable** if the host sleeps (Gate 6) |
| **visible overhead** | **about 3.09 ms (25.5% of window)** | |
| **CPU budget left inside the window** | **about 9.0 ms** | about **4-5 sequences** at 1.9 ms each |

(Quiet-machine numbers. CPU attention and the per-sequence cost improved by 18% versus the
loaded runs; the coordination costs did not move at all.)

## Where this leaves the route

Alive, with a much more specific shape than "put the KV on the CPU":

- gran=32, shared-head Quest, layer 0 dense (fidelity gate)
- **selector + min/max summaries on the GPU**, K/V bytes and attention on the CPU (Gate 5,
  reverses the earlier assumption)
- cross-microbatch pipelining orchestrated **in C**, not Python (Gate 6)
- the engine's GPU wait **must be a blocking/sleeping wait**, not a spin, or the host thread
  must be pinned off the CPU team's cores (Gate 6)
- the CPU attention team must be kept **warm**; letting it sleep between steps costs more than
  the attention (Gate 6, side observation)
- the path is **verified correct** end to end: same selection, same arithmetic to 1.2e-06, same
  tokens as the GPU sparse arm (Gate 7), and the **AVX-512 kernel that was timed is the kernel
  that passes those checks** - 1.8e-06 against the torch path, identical token streams (Gate 8)
- coordination costs ~1.3 ms/step until the handshake moves out of PyTorch or the transfers
  are batched across microbatches (Gate 4)

Dead: naive per-layer synchronous offload; gran=256; CPU-side selection **in PyTorch** (the achieved 4.7 GB/s says the CPU itself was never the problem); any hope
that async event plumbing in Python fixes the handshake; and the throughput framing - at
~2.04 ms/sequence the CPU becomes the critical path around **4 concurrent sequences at 8k**.
This is a *capacity* mechanism (longer context, more sequences resident in DRAM), not a
throughput mechanism, and 128k context or large batch needs the selected-token count to grow
sublinearly, an int8/VNNI kernel, or more cores.

## Open questions, in the order they should be answered

1. ~~Clean-machine rerun~~ **done 2026-09-18**: nothing structural moved and CPU attention
   improved 18% to 1.676 ms. See "Re-baseline on a quiet machine".
2. A C++/CUDA handshake microbenchmark: can per-layer transfer without per-layer host sync
   approach the 0.067 ms bound, or is 1.3 ms structural?
3. A hand-written AVX-512 selector, now the most promising open item: torch achieves only
   4.7 GB/s on a pure streaming reduction, so there is 1-2 orders of magnitude to reclaim. If it
   lands near DRAM bandwidth, the GPU-side summaries and the index transfer both disappear.
4. Re-run Gate 7 with a real checkpoint, to add the answer-level arm to a path that is already
   numerically verified. **No longer waiting on anything**: the home volume has 207 GB free and a
   bf16 8B checkpoint is ~16 GB (see the 2026-09-20 correction). This is a download, not a
   dependency.
5. ~~Replace the harness's PyTorch CPU attention with the AVX-512 kernel from the compute gate~~
   **done 2026-09-20**: the benchmarked kernel now produces the tokens and passes the same checks,
   agreeing with the torch path to 1.8e-06 with identical token streams. See Gate 8. What remains
   open from this item is the *scheduling* half: the harness calls the kernel one layer at a time
   and synchronously, so nothing here re-confirms 2.05 ms/step. Re-confirming it means driving the
   verified kernel from the Gate 6 pipeline (C-level submit/wait, blocking GPU wait) instead of
   from the harness's per-layer call, i.e. one program that is correct *and* overlapped.
6. An int8/VNNI CPU attention kernel is the obvious next speed step and would be a different
   kernel: Gate 8 verified fp32 accumulation over bf16 storage, so that variant needs its own
   correctness run rather than inheriting this one.
