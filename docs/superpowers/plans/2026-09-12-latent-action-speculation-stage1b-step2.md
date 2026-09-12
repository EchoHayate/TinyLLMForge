# Stage 1b step 2: the reversal condition is not met

Status: measured, CPU only, no GPU time, no training.
Verdict: **the erratum stands and the speculation line is closed.** A
real agent turn re-prefills a median of 239 to 295 tokens and a p99 of
1494 to 2162. The reversal condition was about 4000 tokens *per turn*.
Warm demand therefore stays near 0.45 to 0.51 seconds, the free-drafter
floor stays at 0.28 to 0.31, and the best training-free predictor
(0.1129 to 0.2078) clears it on **0.01% to 0.03% of turns**. For the
line to pay, 79% to 89% of agent turns would have to miss the prefix
cache entirely.
Line: latent action speculation (`tinyvllm/agentspec/`).
Predecessor: `2026-09-11-latent-action-speculation-stage1b-erratum.md`.

## The question this closes

The erratum showed that both surviving conclusions of the line rested on
an actor demand measured with prefix caching deliberately defeated. It
also identified the single number that could reverse it: if a real agent
appends enough fresh context per turn, the warm demand stays large and
speculation can still pay. The estimate was that roughly 4000 fresh
tokens per turn were needed. That number had never been measured.

## Three quantities that are easy to confuse

`tools/agentspec_context_growth.py` replays each trace through the same
KV accounting the engine performs, and it separates three things that
earlier reasoning in this line had merged:

- **`assistant`** — tokens the actor generated itself. These grow the
  context but are **already resident**, because decoding writes KV. They
  are never re-prefilled.
- **`observation`** — tokens appended by the environment since the actor
  last ran. Genuinely new text.
- **`prefill`** — what the engine must actually recompute. Neither of the
  above, because `block_manager.py` caches at `block_size` granularity
  and the previous turn's trailing partial block is not reusable:

```text
resident = floor((ctx_before + assistant) / block) * block
prefill  = observation + wrapper + ((ctx_before + assistant) mod block)
```

`prefill` is the quantity that enters `D`. Conflating it with
`observation + assistant` overstates the work; conflating it with
`observation` alone understates it.

## Measured, tokens per steady-state turn

Corpora are byte-identical to Stage 1b step 0 (`apigen-mt_5k.json`
sha256 `5225b541...b5841`, `train-00000-of-00012.parquet` sha256
`5a395e8c...067a13`), Qwen3 tokenizer, `block_size=256` matching
`Config.kvcache_block_size`. The opening turn of each trace is excluded:
it is a genuine cold prefill of the task statement, but it happens once
per trace and is not what the loop pays.

```text
SWE-agent, 2000 traces, 53440 turns
  observation   mean  432.7   p50  136   p75  682   p90 1246   p99 2084   max 22931
  prefill       mean  560.3   p50  295   p75  832   p90 1378   p99 2162   max 23023
  assistant     mean  111.1   p50   83   p75  125   p90  190   p99  532   max  2925
  ctx_before    mean 12674.7  p50 10853  p75 19329 p90 25547  p99 30295  max 37125

APIGen-MT, 5000 traces (full), 41127 turns
  observation   mean  216.6   p50   46   p75  344   p90  583   p99 1344   max 10173
  prefill       mean  346.2   p50  239   p75  481   p90  718   p99 1494   max 10308
  assistant     mean   92.7   p50   65   p75  121   p90  205   p99  368   max  2053
  ctx_before    mean  5183.9  p50 4799   p75 6005  p90 6833   p99 8740   max 18187
```

Agents accumulate long contexts — SWE-agent sits at a median of 10.9k
tokens and reaches 37k — but they accumulate them in small increments.
The p99 turn re-prefills 2162 tokens. The reversal condition needed
about 4000 at the *mean*.

## Demand, and the floor it produces

Prefill is fitted as `a*L + b*L^2` and the decode step as
`base + slope*L` over the three measured points in the Stage 1a-bis
artifact, then applied per turn at that turn's real context length and
real prefill size. The prefill fit under-predicts the 1024-token point
by 5.4 ms, which biases warm demand slightly *low* and therefore the
floor slightly *high* — the fit error works against the conclusion
below rather than for it, and it is far too small to matter at a gap of
this size.

```text
                D warm                          D cold
SWE-agent   mean 0.5055  p50 0.5019  p90 0.5801  |  mean 1.6988  p90 3.2687
APIGen      mean 0.4543  p50 0.4451  p90 0.4865  |  mean 0.8449  p90 0.9977
```

With the drafter priced at **zero** — the friendliest possible
assumption, unreachable by any real drafter — the cost model still
demands:

```text
corpus      pre_p50  pre_p99   D_warm   min_p@warm   best training-free p   gate
swe_agent       295     2162   0.5055       0.2835                 0.1129   FAIL
apigen          239     1494   0.4543       0.3057                 0.2078   FAIL

turns clearing the floor:
  swe_agent   warm 0.0001   cold 0.4472
  apigen      warm 0.0003   cold 0.6823
```

Four turns out of 53440, and twelve out of 41127. The cold columns are
what the line has been quoting all along, and they are the artifact.

## The number a serving team can actually check

Real serving is neither fully warm nor fully cold: an agent idle for
five seconds on a tool call holds KV that a loaded scheduler may recycle.
Expected demand is a mixture, so the honest question is not "warm or
cold" but **how often the prefix cache must miss** before speculation
pays. Solving for the demand at which the floor drops to the measured
training-free `p`:

```text
corpus      D threshold   required prefix-cache miss rate
swe_agent      1.5715 s                               89%
apigen         0.7625 s                               79%
```

Action speculation on these corpora pays only on a server whose prefix
cache is missing four turns out of five. That is not a speculation
opportunity, it is a thrashing cache — and the erratum already showed
the correct fix for that is spilling idle KV to host memory and
restoring it at 97 ms instead of re-prefilling at 1557 ms, losslessly.
**The regime in which action speculation pays is precisely the regime
that should be fixed by other means.**

## Robustness

Block size is the one free parameter in the accounting, and it moves the
result in the direction that would help speculation, so it was swept.
300 traces per cell:

```text
block   corpus       prefill mean   partial-block waste   turns clearing warm
   16   swe_agent           460.9                  1.6%                0.0000
   16   apigen              186.3                  4.0%                0.0005
   64   swe_agent           484.2                  6.4%                0.0000
   64   apigen              210.8                 15.2%                0.0005
  256   swe_agent           579.8                 21.8%                0.0000
  256   apigen              310.5                 42.4%                0.0010
 1024   swe_agent           957.6                 52.7%                0.0001
 1024   apigen              697.0                 74.3%                0.0010
```

Even at `block_size=1024`, four times the engine's actual setting, the
fraction of turns clearing the floor stays at or below 0.001. The
conclusion does not depend on this choice.

## An unrelated finding worth recording honestly

The partial-block term is not a rounding detail. At the engine's actual
`block_size=256`, it is **22.8% of everything SWE-agent re-prefills and
37.4% of everything APIGen re-prefills**. APIGen's median observation is
46 tokens while its median prefill is 239: the block granularity, not
the new text, dominates the work.

Dropping to `block_size=16` cuts APIGen's mean prefill from 310 to 186
tokens per turn. It is lossless and it is a config change. But it should
be reported at its real size rather than dressed up: prefill is a small
part of warm demand, so this is worth roughly 2% of `D` and about 0.2%
of agent wall-clock at five-second tools. Real, free, and minor.

## Status of the line

- **Closed**: action-level speculation with any training-free drafter,
  on these two corpora, at these tool latencies. The reversal condition
  was pre-stated in the erratum and measured here; it failed by roughly
  2x at p99 and by a factor of 10 at the median.
- **Closed**: the argument that a compressed drafter context is nearly
  free at long context. It divided by the cold prefill.
- **Not reopened by training**: the gap is between 0.11 and 0.28, and
  the 8B diagnostic in step 1 showed the shortfall is not model
  capacity. Nothing measured here argues a learned head would cross it.
- **Still open, and narrow**: KV compression at 64k to 128k agent
  contexts, on the capacity axis rather than the latency axis, competing
  against lossless host offload. `tinyvllm/engine/kv_cartridge.py`
  already holds the primitive. This has nothing to do with latent action
  representations.
- **Retained as an asset**: `tinyvllm/agentspec/` is a working
  analytic cost gate with 48 tests. It has now falsified three
  successive proposals on paper before any of them consumed a training
  run, which is what it was built to do.

## Artifacts

```text
tools/agentspec_context_growth.py
tools/agentspec_context_growth_verdict.py
experiments/agentspec_context_growth/2026-09-12/context_growth_swe_agent.json
experiments/agentspec_context_growth/2026-09-12/context_growth_apigen.json
experiments/agentspec_context_growth/2026-09-12/verdict.json
```

Growth payload digests: SWE-agent
`a956b6a5317b012fcdd8712fde225ffb9431ef462388b2f457ab3bf2f729d20a`,
APIGen
`9707d15d58d8dad8f4e8c838404240947cb3e2f34d0b5659c9b60e4f6272d14b`.
Cost inputs throughout: `tool_seconds=5.0`, `rho=0.6`,
`rollback=0.5s`, drafter tax `0`. Corpora are not committed; only token
counts and derived costs are.
