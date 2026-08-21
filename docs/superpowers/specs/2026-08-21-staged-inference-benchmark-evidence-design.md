# Staged Inference Benchmark Evidence Design

Date: 2026-08-21

## Objective

Build resume-grade, source-bound performance evidence in two stages:

1. use Qwen3-0.6B to complete low-cost controlled gates for the existing
   cross-request prefix cache and chunked-prefill fairness policy; then
2. promote exactly one winner to Qwen3-8B using the same metric definitions,
   correctness rules, and frozen thresholds.

Every published result must report both benefit and cost. A feature that has
only local correctness coverage, or that misses its controlled performance
gate, must be described as implemented and understood rather than as an
optimization or speedup.

This design does not change the active Qwen3-8B plus Qwen3-0.6B TP4
autoregressive-draft tail-latency campaign. That campaign remains an
independent prerequisite for any TP4 n-gram tail-latency claim.

## Current Evidence Boundary

### Chunked Prefill

The current tree contains scheduler contracts, a step-latency profiler, and a
production-style arrival-load harness. Historical arrival-load artifacts are
not current-source promotion evidence:

- an earlier bounded-prefill policy had exact-output failures;
- later mixed-prefill policies preserved correctness but produced no
  throughput or memory benefit and violated ITL, decode-gap, or service-class
  tail guards; and
- all historical runs predate substantial current-source changes.

The new gate therefore compares current-source disabled chunking directly
against one frozen fairness policy. It does not tune among many policies after
seeing results.

### Prefix Cache

The implemented ordinary cross-request cache is a hash-based full-block prefix
cache with token-identity collision checks and compute-complete publication.
It is not a radix tree. The existing profiler and report logic cover cold,
warm, and cache-cleared execution, but no completed current-source remote
canonical is retained.

The new gate may claim full-block prefix reuse, executed-prefill reduction,
and TTFT improvement if it passes. It may not claim RadixAttention, partial
block reuse, cache-aware routing, decode acceleration, physical memory
savings, or increased context capacity.

### N-gram Speculation

Existing controlled evidence already records acceptance, TTFT, TPOT,
throughput, peak memory, and real KV movement. The active TP4 schedstat
campaign is intended to establish or reject a host-scheduling explanation for
the remaining decode-tail instability. No new n-gram headline is created by
the Prefix or Chunked gates.

## Global Execution Constraints

- The authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`.
- The source branch is `feat/kv-sparse-attention`, and every remote source
  snapshot must be bound to the exact pushed
  `origin/feat/kv-sparse-attention` commit.
- Remote execution uses `sitian@10.232.195.203`.
- Remote Python is `/data00/home/sitian/tllm/env/bin/python`.
- Qwen3-0.6B is
  `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`.
- Qwen3-8B is `/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B`.
- Every remote run, source snapshot, temporary directory, cache, log,
  manifest, and verification artifact must be below:

  ```text
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
  ```

- Remote execution must not write under `/`, `/tmp`, `/private/tmp`, or the
  remote checkout `/data00/home/sitian/tllm/TinyLLMForge`.
- The local SSH ControlMaster path may remain
  `/tmp/ssh-sitian-10.232.195.203`.
- Do not refresh Kerberos automatically.
- Do not terminate or interfere with unrelated GPU processes.
- Admission requires a selected GPU to have memory used `<=1024 MiB`,
  utilization `<=5%`, and no compute process.
- Each run tag is immutable. A partial or failed tag is preserved and never
  resumed or overwritten.
- Measured paths must not gain synchronization, `.item()`, logging,
  profiling, acknowledgements, fences, or GC controls solely for the gate.
- Every canonical result is reconstructed by an independent verifier from raw
  evidence and protected by a manifest.

## Stage 1A: Qwen3-0.6B Prefix Cache Gate

### Fixed Comparison

Run one loaded engine in isolated processes for each case:

- `cold`: reusable cache cleared before the measured request or batch;
- `warm`: a producer request materializes the shared prefix before the
  measured consumer request or batch; and
- `cache_cleared`: warm setup followed by explicit reusable-cache clearing
  before the measured request or batch.

Use greedy fixed-length output and exact token comparison. Retain the existing
full-vocabulary logit checks for correctness, while keeping profiler-only
device-to-host work outside the timed interval.

### Fixed Shapes

- shared-prefix lengths: `256`, `1024`, and `2048` tokens;
- suffix length: `64` tokens;
- single-request cases for every prefix length;
- one eight-request shared-prefix batch for `1024` and `2048`;
- two warmup repetitions and seven measured repetitions;
- default full-block size and sampleable-suffix rule;
- eager execution for the first canonical so graph capture state cannot
  contaminate cold/warm comparison.

### Benefit Metrics

- reusable-token ratio:
  `cached_prompt_tokens / reusable_prompt_tokens`;
- executed-prefill reduction:
  `cold_query_tokens - warm_query_tokens`;
- single-request median and p95 TTFT;
- eight-request median and p95 full-batch elapsed time;
- warm model-batch count.

### Cost Metrics

- retained reusable block count and retained logical KV bytes;
- peak CUDA allocated and reserved bytes;
- cache-clear host time, reported separately and excluded from model timing;
- cache-cleared TTFT relative to the adjacent cold state;
- correctness and cache-isolation failures.

Logical retained KV bytes must not be described as physical memory saved or
consumed unless CUDA memory observations independently support that statement.

### Frozen Decision

`PREFIX_CACHE_GO` requires all of:

1. exact greedy output IDs and decoded text for every case;
2. argmax equality with full-logit `max_abs <= 0.25` and
   `mean_abs <= 0.05`;
3. exact cached-token and executed-query-token accounting;
4. warm reusable-token ratio `1.0` for every expected full block;
5. at least `20%` warm median TTFT improvement at both `1024` and `2048`;
6. at least `15%` warm median full-batch elapsed improvement at both batch
   shapes;
7. all eight warm consumers admitted in one model batch;
8. no cache-cleared median TTFT regression above `5%` against the adjacent
   cold state;
9. no unexplained peak CUDA reserved regression above `5%`; and
10. complete raw evidence, manifest, and independent-verifier agreement.

Any correctness, accounting, source, environment, isolation, or artifact
failure is `PREFIX_CACHE_INCOMPLETE_OR_INCORRECT`. A complete correct run that
misses a performance threshold is `PREFIX_CACHE_NO_GO`.

## Stage 1B: Qwen3-0.6B Chunked-Prefill Fairness Gate

### Fixed Policies

Compare exactly two policies:

- `OFF`:
  `max_num_prefill_tokens_per_step=0`;
- `FAIR_CHUNKED`:
  `max_num_prefill_tokens_per_step=128`,
  `chunked_prefill_decode_first=False`,
  `chunked_prefill_max_consecutive_chunks=2`,
  `chunked_prefill_mixed_batch=False`,
  `chunked_prefill_adaptive_mixed=False`, and
  `chunked_prefill_slo_mixed=False`.

`FAIR_CHUNKED` intentionally gives prefill bounded progress but forces a
decode-only yield after at most two consecutive prefill chunks when runnable
decode exists. The canonical gate does not include always-on mixed batching,
adaptive mixed batching, or SLO-based mixed admission.

### Fixed Workload

Use one source-bound arrival manifest for both policies:

- eight warmup requests excluded from metrics;
- 96 measured requests;
- 58 short prompts of `64` tokens;
- 24 medium prompts of `512` tokens;
- 14 long prompts of `4096` tokens;
- short and long outputs of `16` and `64` tokens, balanced within each prompt
  class;
- `max_model_len=4352`, `max_num_batched_tokens=16384`, and
  `max_num_seqs=512` for both policies;
- deterministic arrivals containing steady, burst, and long-prompt-injection
  phases;
- one workload calibration before canonical execution;
- five measured repetitions per policy;
- distinct dynamic distributed ports and isolated model processes.

The same request IDs, prompt tokens, output lengths, arrival offsets, and
sampling parameters are used for both policies.

### Benefit Metrics

- short-request p50, p95, and p99 TTFT;
- short-request p50, p95, and p99 ITL;
- p95 and p99 completion latency by prompt/output service class;
- maximum decode gap;
- completed-request throughput and generated-token throughput;
- queue depth and starvation count.

### Cost Metrics

- long-request p50, p95, and p99 TTFT and completion latency;
- total prefill model steps and total model steps;
- peak logical KV bytes;
- peak CUDA allocated and reserved bytes;
- throughput regression;
- unfinished, rejected, or starved requests;
- exact-output or lifecycle mismatch.

### Frozen Decision

`FAIR_CHUNKED_GO` requires:

1. exact token-for-token output equality and complete request lifecycle;
2. zero dropped, rejected, truncated, unfinished, or starved requests;
3. short-request p99 TTFT improvement of at least `10%`;
4. short-request p99 ITL does not regress by more than `5%`;
5. maximum decode gap does not regress by more than `10%`;
6. every service class p95 completion latency does not regress by more than
   `10%`;
7. long-request p95 completion latency does not regress by more than `10%`;
8. completed-request and token throughput each do not regress by more than
   `3%`;
9. peak CUDA reserved bytes do not regress by more than `5%`;
10. the benefit direction is present in at least four of five paired
    repetitions; and
11. complete raw evidence, manifest, and independent-verifier agreement.

A complete and correct run that misses any performance guard is
`FAIR_CHUNKED_NO_GO`. Missing or invalid evidence is
`FAIR_CHUNKED_INCOMPLETE`.

## Stage 2: Promote One Winner to Qwen3-8B

### Eligibility

Only a Stage 1 feature with a verified `GO` is eligible. If both features
reach `GO`, select one using this deterministic order:

1. larger normalized primary benefit:
   - Prefix Cache: minimum of the `1024` and `2048` warm median TTFT
     improvements;
   - Chunked Prefill: short-request p99 TTFT improvement;
2. smaller worst protected-metric regression;
3. lower peak CUDA reserved regression; and
4. Prefix Cache wins an exact tie because its Qwen3-8B gate requires fewer
   concurrent requests and less remote occupancy.

If neither Stage 1 feature reaches `GO`, do not promote either one merely to
obtain a larger-model number.

### Qwen3-8B Revalidation

- preserve the winning policy, workload proportions, metric definitions,
  correctness requirements, and decision thresholds;
- scale only model-capacity-dependent prompt or batch limits when a frozen
  preflight proves the exact Stage 1 shape cannot execute;
- record any shape scaling before model results are visible;
- use at least one warmup and five measured repetitions;
- retain a new immutable source/environment/workload manifest and independent
  verification.

Qwen3-0.6B gains do not transfer to Qwen3-8B. Only the completed 8B gate may
support an 8B performance statement.

## Artifacts

Each gate produces an immutable primary bundle and an independently rebuilt
controller bundle containing:

- `run_manifest.json`;
- frozen resolved configuration and workload;
- raw request, step, scheduler, cache, and memory rows;
- per-repetition status and logs;
- aggregate summary;
- human-readable report;
- remote verification receipt;
- local independent verification receipt; and
- `manifest.sha256`.

The report must include a two-column conclusion:

| Benefit | Cost |
|---|---|
| primary latency/throughput/cache-reuse improvement | protected latency, throughput, memory, retained-KV, or fairness penalty |

## Claim Language

Allowed after a verified `GO`:

- “On the frozen Qwen3-0.6B workload, hash-based full-block prefix reuse
  reduced executed prefill tokens by X and median TTFT by Y, while cold-path
  TTFT changed by Z and retained K logical KV bytes.”
- “On the frozen Qwen3-0.6B mixed-arrival workload, bounded fair chunking
  changed short-request p99 TTFT by X, with throughput change Y and
  long-request p95 completion change Z.”
- The equivalent statement for Qwen3-8B only after Stage 2 passes.

Required language after correctness-only or `NO_GO` evidence:

- “Implemented and validated the mechanism and its correctness boundaries;
  the controlled benchmark did not establish a performance improvement.”

Forbidden:

- “Radix Tree Prefix Cache” for the current hash-based implementation;
- model-size-generalized gains from the 0.6B gate;
- “optimized” or “improved” when the applicable controlled gate is absent,
  incomplete, or `NO_GO`;
- memory-saving or longer-context claims inferred only from logical cache
  accounting; and
- selecting only favorable repetitions or changing thresholds after results.
