# Mechanism gates: the handshake, the selector, and the pipeline

Date: 2026-09-17
Host: `n232-195-203`, 8x A100 80GB PCIe, Xeon Platinum 8336C (64 physical cores / 128 threads),
2015 GB RAM. Load average during all runs: **103-111, 25-26 other users**. Every CPU number
below is therefore pessimistic and every GPU-under-load number is realistic.
Raw data: `experiments/mechanism_gates/mechanism-20260917-221357/`.

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
**not** fix the graph case, which rules out the simple "reserve a core" story: this machine is
at load ~105, there is no idle core to reserve, and the spinner is unpinned so it lands
wherever it wants. I can prove the fix (blocking wait) and the necessary conditions; the exact
scheduler mechanics deserve a clean-machine rerun with a pinned host thread before I claim
more.

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

## Revised budget, one sequence, 8192 context, gran=32

| item | cost | note |
|---|---|---|
| GPU weight-read window | 12.113 ms | what we have to hide inside |
| engine inflation under saturated CPU | 1.523 ms | measured, CUDA-graph engine |
| per-layer handshake | 1.316 ms | Gate 4; async did not help |
| GPU selector (bf16 summaries) | 0.258 ms | Gate 5 |
| CPU sparse attention | 2.051 ms | **~97% hideable** if the host sleeps (Gate 6) |
| **visible overhead** | **~3.10 ms (25.6% of window)** | |
| **CPU budget left inside the window** | **~9.0 ms** | ~**4 sequences** at 2.04 ms each |

## Where this leaves the route

Alive, with a much more specific shape than "put the KV on the CPU":

- gran=32, shared-head Quest, layer 0 dense (fidelity gate)
- **selector + min/max summaries on the GPU**, K/V bytes and attention on the CPU (Gate 5,
  reverses the earlier assumption)
- cross-microbatch pipelining orchestrated **in C**, not Python (Gate 6)
- the engine's GPU wait **must be a blocking/sleeping wait**, not a spin (Gate 6)
- coordination costs ~1.3 ms/step until the handshake moves out of PyTorch or the transfers
  are batched across microbatches (Gate 4)

Dead: naive per-layer synchronous offload; gran=256; CPU-side selection in PyTorch; any hope
that async event plumbing in Python fixes the handshake; and the throughput framing - at
~2.04 ms/sequence the CPU becomes the critical path around **4 concurrent sequences at 8k**.
This is a *capacity* mechanism (longer context, more sequences resident in DRAM), not a
throughput mechanism, and 128k context or large batch needs the selected-token count to grow
sublinearly, an int8/VNNI kernel, or more cores.

## Open questions, in the order they should be answered

1. Clean-machine rerun with a pinned host thread: confirm the spin/blocking result is not an
   artifact of load ~105, and get uncontaminated CPU numbers.
2. A C++/CUDA handshake microbenchmark: can per-layer transfer without per-layer host sync
   approach the 0.067 ms bound, or is 1.3 ms structural?
3. A hand-written AVX-512 selector: does CPU selection become viable (which would remove the
   36 MiB of GPU summaries and the index transfer), or is the GPU split final?
4. Correctness before more performance: a CPU full-attention prototype that matches the GPU
   path numerically, then selector + CPU attention + output return, then scheduling.
