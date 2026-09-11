# Stage 1b erratum: the surviving cell was a cold-prefill artifact

Status: analysis over committed artifacts, no new GPU time.
Verdict: **the "compressed drafter context is nearly free at long
context" result does not survive prefix caching, and neither does the
one cell step 1 said survived.** Both conclusions were computed against
an actor demand `D` that was deliberately measured with a cold prefill.
An agent loop with append-only context does not pay that prefill. Once
`D` is priced warm, the drafter tax rises 4x, the profitability floor
rises 3x, and the training-free n-gram fails everywhere too.
Line: latent action speculation (`tinyvllm/agentspec/`).
Corrects: `2026-09-10-latent-action-speculation-stage1a-bis.md`,
`2026-09-11-latent-action-speculation-stage1b-step1.md`.

## The measurement that was right, used the wrong way

`tools/agentspec_engine_demand_worker.py` says so itself:

> A fresh random prompt is used for every repetition because the engine
> carries prefix caching, and reusing a prompt would make prefill look
> free.

That is the correct choice for measuring the *cost of a prefill*. It is
the wrong choice for measuring *the demand of an agent turn*, and I
used it for the second thing without noticing the substitution.

`tinyvllm/engine/block_manager.py` implements block-hash prefix caching
and, on deallocate, keeps the `hash_to_block_id` mapping and the block's
`token_ids` alive until the block is physically recycled. So an agent
whose context grew append-only re-enters the engine and re-prefills only
its newest observation, not its history.

Warm demand, using the same measured decode steps, with a 512-token
fresh observation per turn:

```text
context      D cold      D warm    actor prefill share
   1024      0.4907      0.4544    0.081 -> 0.045
   4096      0.7487      0.4741    0.320 -> 0.045   (-86%)
  16384      2.0385      0.5262    1.557 -> 0.045   (-97%)
```

## What that does to both conclusions

Stage 1a-bis concluded that `tau` for `code_drafter_ckv` falls with
context (0.0691 -> 0.0453 -> 0.0166). It falls because the numerator is
a fixed 512-token prefill while the denominator inflates with a cold
prefill. Remove the inflation and the trend reverses:

```text
context   tau_ckv cold   tau_ckv warm
   1024        0.0691         0.0746
   4096        0.0453         0.0715
  16384        0.0166         0.0644
```

The compressed drafter context was never buying a shrinking tax. It was
being divided by a number that a real agent loop does not pay.

Step 1 closed by noting one surviving cell: at 16384 context and
5-second tools, a free drafter needs only `p >= 0.0909`, which the
training-free trigram clears (0.2078 APIGen, 0.1129 SWE). Repricing on
warm demand:

```text
tool=5s, rho=0.6, rollback=0.5s, drafter tax = 0
scenario                        D        min_p     trigram apigen / swe
16384 cold (as published)   2.0385      0.0909      PASS / PASS
16384 warm, 512-tok delta   0.5262      0.2754      fail / fail
16384 warm, 2048-tok delta  0.6412      0.2378      fail / fail
```

The surviving cell does not survive. For the trigram to clear the floor
again the actor would have to re-prefill roughly 4000 fresh observation
tokens *every turn*, which is a measurable property of a workload, not
an assumption anyone has checked. That is the one open question the
speculation line still has, and it is a trace-statistics question, not a
GPU question.

## Then how much is context compression itself worth?

Separate the compression idea from the speculation idea and price it
directly. The measured decode steps give the KV-attention share of a
step, because everything else in a decode step is weight traffic and
does not move with context:

```text
step(L) = 13.05 ms + 0.151 us/token * L        (fit over the three measured points)

context     step      KV-attn share of step
   1024   13.20 ms                     1.2%
   4096   13.67 ms                     4.5%
  16384   15.52 ms                    15.9%
  65536   22.94 ms                    43.1%
 131072   32.84 ms                    60.3%
```

An oracle compressor that made KV attention completely free would save,
per agent turn with 5-second tools:

```text
context   D warm    KV-attn in D    max wall-clock saving
  16384   0.526 s        0.077 s                   1.39%
  65536   0.756 s        0.307 s                   5.33%
 131072   1.063 s        0.613 s                  10.12%
```

At the context lengths this line has been measuring, a *perfect*
compressor is worth 1.4% of wall clock. The prefill side is already
handled losslessly by the prefix cache, which removes 97% of it at
16384. Compression only becomes interesting past 64k.

## And the capacity argument does not rescue it either

The natural fallback is that compression buys concurrency rather than
latency, since an agent waiting 5 seconds on a tool holds its KV
resident while computing nothing. Qwen3-8B in bf16 costs 0.141 MiB per
token of KV (36 layers, 8 KV heads, head_dim 128). On one A100 80GB at
`gpu_memory_utilization=0.85`, after weights and workspace, roughly
47.6 GiB is available for KV:

```text
context   GiB/agent   memory cap   compute cap @rho=0.6   binding
  16384        2.25    21.2 agents            6.3 agents   compute
  65536        9.00     5.3 agents            4.6 agents   compute
 131072       18.00     2.6 agents            3.4 agents   memory
```

Memory does not bind until about 128k. Below that the box runs out of
compute first, and compute spent on the actor's own 31 output tokens is
not something a context compressor can remove.

Even in the eviction case there is a lossless competitor that has to be
beaten first. Spilling an idle agent's KV to host memory and restoring
it over PCIe 4 at roughly 25 GB/s:

```text
context    KV size   H2D restore   cold re-prefill   ratio
  16384    2.25 GiB         97 ms           1557 ms   16.1x
```

Offload is 16x cheaper than re-prefilling and it is exact. A lossy
compressor has to beat 97 ms, not 1557 ms.

## What is actually left

- **Dead**: compressed drafter context as a cost argument. It was an
  artifact of the denominator.
- **Dead**: the last surviving speculation cell, unless a trace study
  shows agents append about 4000 fresh tokens per turn.
- **Not worth measuring**: KV compression for latency below 64k
  context. The ceiling is 1.4% of wall clock with a perfect compressor.
- **Possibly alive, narrow**: KV compression at 64k-128k+ agent
  contexts, where KV attention is 43-60% of a decode step and memory
  starts to bind. `tinyvllm/engine/kv_cartridge.py` already has the
  training-free read-side primitive for this. The honest framing is
  that this is ordinary long-context KV compression, competing with a
  crowded literature and with lossless offload, and it has nothing to
  do with latent action representations.
- **Cheapest next measurement, if the line continues**: re-run the
  engine demand worker with prefix caching allowed to hit, so the repo
  carries a warm `D` next to the cold one. Everything above is
  arithmetic over the existing artifact and should be replaced by a
  measurement before anyone builds on it.

## Provenance

All numbers derive from
`experiments/agentspec_engine_demand/engine-demand-measure-a100-20260910-2258/engine_demand.json`
(payload sha256 `1a9bb5c6...fb32ab`, cuda_graph mode) and from
`tinyvllm/agentspec/cost_model.py`. Model shapes read from the Qwen3-8B
and Qwen3-0.6B configs on the measurement host. The warm-prefill figure
of 45 ms for a 512-token delta is interpolated from the measured actor
prefill at 1024 tokens and is the only unmeasured input; the
conclusions hold for any value below roughly 280 ms, which is where the
trigram would return to break-even.
