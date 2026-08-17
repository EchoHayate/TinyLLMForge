# Qwen3.5 TP4 Hybrid-Prefix Performance and Cache Authority Design

Date: 2026-07-29

## Status

This document defines the first source-bound Qwen3.5 TP4 authority for
end-to-end hybrid-prefix performance and cache cost.

The authority is intentionally not runnable yet. It remains hard-blocked until
all of the following correctness prerequisites independently pass:

1. TP4 real root-logit correctness;
2. cached-continuation correctness;
3. constructed `ModelRunner` and `LLMEngine` correctness;
4. exact greedy output equality for the benchmark workload.

The active TP4 root-logit source remains:

```text
tag:
  qwen35-tp4-source-prep-20260729-010400
source tree:
  b2d0b77de953e273dbf62f0e7b2bbe689ef33c183edf65830940e43123bb485f
model manifest:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

That source identity is a correctness prerequisite, not a performance result.
The current server has no eligible four-GPU set, so this design authorizes only
documentation and dependency-light contract tests. It authorizes no GPU
benchmark, performance claim, cache claim, README update, or default
enablement.

## Objective

Create a reproducible TP4 benchmark authority that can answer, after
correctness is complete:

1. Does exact full-fidelity Qwen3.5 hybrid-prefix restore reduce repeated-prefix
   prefill work and time to first token?
2. Does it improve repeated-prefix request throughput without harming decode
   latency?
3. What additional physical cache bytes, logical snapshot bytes, and CUDA
   memory does the optimization consume?
4. Does exact tensor interning reduce physical snapshot storage relative to
   logical snapshot storage?
5. Can a later compressed candidate such as int4 state storage, token sparsity,
   or low rank be compared against the same full-fidelity authority without
   changing workloads, thresholds, or correctness rules?

The first canonical pair establishes an exact full-fidelity baseline. It does
not claim that the current cache is compressed or smaller than having no cache.

## Non-Goals

This authority does not:

- modify the frozen TP4 root-logit correctness source;
- bypass or weaken any correctness prerequisite;
- run on fewer than four truly idle GPUs;
- exempt known daemons or existing compute processes from the GPU guard;
- kill or signal any pre-existing remote process;
- compare TinyLLMForge with Hugging Face or another engine;
- mix scheduler, CUDA Graph, speculative decoding, KV offload, sparse
  attention, token sparsity, low rank, int4 quantization, and hybrid-prefix
  restore in one first result;
- treat logical bytes as physical-memory savings;
- treat a single warm run as a performance conclusion;
- tune thresholds after observing canonical results;
- change schema-v2 canonical `NO_GO`;
- update README performance guidance before an independently verified `GO`.

## Alternatives Considered

### 1. Recommended: Paired Full-Engine Recompute Versus Exact Restore

Run the same Qwen3.5 TP4 engine source, model, workload, capacity, and sampling
configuration in isolated processes:

- `recompute`: hybrid-prefix publication and restore are disabled;
- `exact_restore`: full-fidelity KV-prefix reuse plus convolution/recurrent
  state publication and restore are enabled.

This directly measures the currently implemented cache path and provides the
full-fidelity reference required by later compression candidates.

Advantages:

- isolates the real end-to-end optimization;
- preserves exact states and a strict quality boundary;
- records both saved execution and added cache cost;
- can later accept a compressed candidate without changing the authority.

Limitations:

- `exact_restore` may use more memory than `recompute`;
- a `GO` means the exact cache is worthwhile under the registered workloads,
  not that cache bytes are reduced relative to no cache;
- compression remains a separate candidate stage.

### 2. Compare TinyLLMForge with the Official Transformers Model

This is rejected for performance classification. Different kernels, batching,
memory allocation, and serving paths would make attribution impossible. The
official model remains a correctness oracle only.

### 3. Benchmark int4, Sparse, and Low-Rank State Compression Immediately

This is deferred. Combining a new numerical representation with the first
full-engine benchmark would make a failure ambiguous between engine
integration, cache lifecycle, compression error, and performance measurement.
Every lossy candidate must first pass the exact authority's workload and
artifact contract.

## Policy Profiles

### P0: `recompute`

The baseline:

- uses the same native Qwen3.5 TP4 model and engine source as the candidate;
- disables Qwen3.5 hybrid-prefix publication and restore;
- clears ordinary reusable prefix metadata between measured requests;
- executes the complete prompt for every request;
- uses greedy decoding and the same output length as P1;
- records zero hybrid-prefix entries and zero hybrid-prefix snapshot bytes.

P0 is not allowed to reduce model capacity, batch size, maximum context, or KV
capacity to make its memory footprint look better.

### P1: `exact_restore`

The first candidate:

- enables the existing exact Qwen3.5 hybrid-prefix publication runtime;
- uses full-fidelity convolution and recurrent state tensors;
- uses the existing exact tensor interning implementation;
- publishes only after a completed source prefill;
- restores only when token, block, model, layout, TP, and dtype identities all
  match;
- executes the same continuation and decode request as P0;
- records cache hits, misses, physical bytes, logical bytes, deduplicated
  bytes, entries, evictions, and validation failures.

P1 must not quantize, sparsify, truncate, approximate, or project state.

### Future P2 Profiles

A later design may add one profile at a time:

```text
int4_state
token_sparse_state
low_rank_state
gist_layer_share
```

Each P2 profile must:

- retain P0 and P1 in the matrix;
- use P1 as its numerical and cache-cost reference;
- declare its representation and metadata bytes explicitly;
- pass a candidate-specific quality gate before performance is considered;
- preserve the same workload manifest and paired execution rules;
- receive a new schema version and source identity.

## Correctness Prerequisite Manifest

The benchmark consumes one immutable
`correctness_prerequisites.json`. It contains exactly:

```text
schema_version
model_manifest_sha256
tp4_root_logit:
  run_tag
  source_tree_sha256
  artifact_sha256
  independent_verification_sha256
  classification
cached_continuation:
  run_tag
  source_tree_sha256
  artifact_sha256
  independent_verification_sha256
  classification
engine_correctness:
  run_tag
  source_tree_sha256
  artifact_sha256
  independent_verification_sha256
  classification
```

Every classification must equal `PASS`. Every referenced file must exist,
match its SHA-256, bind the approved model manifest, and have an independent
verification result. A producer summary, unit-test result, or hand-written
status is not a substitute.

Until the cached-continuation and engine-correctness authorities exist, the
contract classifier returns `BLOCKED_CORRECTNESS`.

The TP4 root-logit source tree above is a prerequisite identity only. The
benchmark runtime source is a separate deterministic staged bundle containing
the benchmark contract, worker, verifier, runner, required TinyLLMForge
runtime, and tests. `source_manifest.json` records that bundle's own SHA-256,
and every raw benchmark row must bind to it. A benchmark bundle must never
claim the older root-logit source-tree SHA after adding benchmark files.

## Workload Matrix

The canonical workload uses deterministic token-ID manifests rather than
free-form text. Every prompt leaves at least one uncached token for a valid
sample path.

### W0: Short Control

```text
shared prefix:
  256 tokens
suffix:
  32 tokens
requests per repetition:
  1 source + 1 continuation
generated tokens:
  32
```

This detects fixed setup overhead. It is not required to show a speedup.

### W1: Medium Reuse

```text
shared prefix:
  1024 tokens
suffix:
  64 tokens
requests per repetition:
  1 source + 4 continuations
generated tokens:
  64
```

### W2: Long Reuse

```text
shared prefix:
  4096 tokens
suffix:
  64 tokens
requests per repetition:
  1 source + 4 continuations
generated tokens:
  64
```

### W3: Batched Fan-Out

```text
shared prefix:
  2048 tokens
distinct suffixes:
  8 x 64 tokens
requests:
  1 source + one batch of 8 continuations
generated tokens:
  32 per continuation
```

### W4: Miss and Invalidation Control

W4 uses:

- one token mismatch inside the reusable prefix;
- one stale block-generation identity;
- one cache clear between source and continuation.

Every case must miss or invalidate exactly as registered. W4 is correctness
and accounting evidence, not a speed target.

## Execution Design

Each `(workload, repetition)` pair runs P0 and P1 in separate fresh processes.
The order alternates deterministically:

```text
even repetition:
  recompute, exact_restore
odd repetition:
  exact_restore, recompute
```

Each process:

1. passes the unchanged four-GPU resource preflight;
2. records GPU indices, UUIDs, free bytes, and empty compute-process lists;
3. verifies source tree, model manifest, correctness prerequisite manifest,
   workload manifest, and configuration hash before model construction;
4. constructs the TP4 `LLMEngine`;
5. performs one excluded warmup workload;
6. resets CUDA peak-memory counters;
7. runs one correctness phase;
8. runs five measured repetitions;
9. synchronizes CUDA at registered timing boundaries;
10. drains and exits all four model ranks;
11. preserves all logs and artifacts even on failure.

The runner must assign unique dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`
pairs per process. It never reuses a remote run tag or output directory.

## Raw Measurements

Every request row records:

```text
row_id
policy
workload
phase
repetition
request_id
source_tree_sha256
model_manifest_sha256
workload_manifest_sha256
correctness_prerequisites_sha256
prompt_tokens
reused_kv_tokens
restored_hybrid_state
executed_prefill_tokens
generated_tokens
ttft_ns
e2e_ns
decode_step_ns
output_token_ids
output_token_ids_sha256
final_logits_path
final_logits_sha256
```

Every process row records:

```text
initialization_ns
cuda_allocated_bytes
cuda_reserved_bytes
cuda_peak_allocated_bytes
cuda_peak_reserved_bytes
kv_capacity_bytes
hybrid_cache_current_entries
hybrid_cache_current_bytes
hybrid_cache_current_logical_bytes
hybrid_cache_deduplicated_bytes
hybrid_cache_peak_entries
hybrid_cache_peak_bytes
hybrid_cache_hits
hybrid_cache_misses
hybrid_cache_evictions
hybrid_cache_validation_failures
hybrid_cache_failed_restores
```

All timing values are raw integer nanoseconds. Derived milliseconds, ratios,
medians, and percentiles are computed offline.

## Correctness Rules

Correctness is conjunctive and precedes performance:

1. Every prerequisite authority is `PASS`.
2. P0 and P1 produce identical greedy output token IDs for every request.
3. Correctness-phase final logits satisfy the approved Qwen3.5 decision
   preservation rule and registered tolerance:
   `atol=2e-5`, `rtol=1e-5`.
4. P1 records a restore hit in W1, W2, and W3.
5. P0 records no hybrid restore and zero hybrid cache bytes.
6. W4 mismatch, stale-generation, and clear cases do not restore.
7. No cache validation failure, failed restore, non-finite value, missing rank,
   worker timeout, or surviving gate-owned child occurs.
8. Every rank exits and destroys its process group.

Any correctness failure makes the result `INVALID`, not a performance
`NO_GO`.

## Performance and Cache Classification

The independent verifier computes medians from five measured repetitions.

### Required Performance Conditions

- W1 P1 median continuation TTFT is at least `15%` lower than P0.
- W2 P1 median continuation TTFT is at least `25%` lower than P0.
- W3 P1 median completed-request throughput is at least `1.15x` P0.
- Every W1/W2/W3 repetition has TTFT ratio `<= 1.05`.
- Median decode-step latency ratio is `<= 1.02` for every workload.
- Engine initialization ratio is `<= 1.10`.

W0 and W4 have no positive speed threshold, but neither may exceed a `5%`
median end-to-end regression.

### Required Cache-Cost Conditions

- P1 physical hybrid-cache bytes never exceed the configured `max_bytes`.
- P1 peak entries never exceed the configured `max_entries`.
- `current_bytes <= current_logical_bytes`.
- `deduplicated_bytes == current_logical_bytes - current_bytes`.
- P1 peak CUDA reserved-memory ratio is `<= 1.10`.
- P1 scheduler-visible KV capacity equals P0 exactly.
- P1 `kv_capacity_bytes` equals P0 exactly.
- No eviction or oversize rejection occurs in W1/W2/W3.

The verifier reports, but does not preregister a positive claim for:

```text
logical_to_physical_snapshot_ratio
physical_snapshot_bytes_per_reused_token
added_cuda_bytes_per_reused_token
saved_prefill_tokens_per_physical_snapshot_byte
```

These metrics establish the denominator for future compression work.

They are independently reconstructed over measured W1/W2/W3 cases only:

```text
logical_to_physical_snapshot_ratio =
  sum(P1 current_logical_bytes per fresh process)
  / sum(P1 current_bytes per fresh process)

physical_snapshot_bytes_per_reused_token =
  sum(P1 current_bytes per fresh process)
  / sum(P1 reused_kv_tokens across continuation requests)

added_cuda_bytes_per_reused_token =
  sum(max(0, P1 peak_reserved_bytes - paired P0 peak_reserved_bytes))
  / sum(P1 reused_kv_tokens across continuation requests)

saved_prefill_tokens_per_physical_snapshot_byte =
  sum(P1 prompt_tokens - P1 executed_prefill_tokens)
  / sum(P1 current_bytes per fresh process)
```

Each canonical case runs in a fresh process, so one process snapshot is
counted once rather than once per continuation row. Negative CUDA deltas are
reported as zero added bytes; this denominator measures overhead, not memory
savings. These four values are reporting-only and do not change schema-v1
`GO | NO_GO` thresholds.

### Classifications

- `BLOCKED_CORRECTNESS`: one or more prerequisite authorities are absent or
  not `PASS`; no GPU workload is allowed.
- `BLOCKED_RESOURCES`: prerequisites pass but four eligible GPUs are absent;
  no model process is allowed.
- `INVALID`: a run started but evidence, provenance, process safety, or
  correctness is incomplete or invalid.
- `NO_GO`: evidence and correctness are valid, but one or more performance or
  cache-cost conditions fail.
- `GO`: every correctness, performance, cache-cost, provenance, and process
  condition passes.

A valid `NO_GO` exits successfully from verification because the experiment is
complete. `BLOCKED_*` and `INVALID` exit nonzero.

## Artifact Contract

One canonical run publishes exactly:

```text
correctness_prerequisites.json
workload_manifest.json
source_manifest.json
environment.json
gpu_assignments.json
commands.json
case_rows.jsonl
process_rows.jsonl
logits_manifest.json
worker_logs_manifest.json
summary.json
artifact_manifest.json
independent_verification.json
report.md
```

Tensor and log files named by the two manifests live under closed
`logits/` and `logs/` directories. Authority artifacts referenced by
`correctness_prerequisites.json` live under the closed `prerequisites/`
directory. `artifact_manifest.json` hashes every producer/raw top-level input
plus every nested prerequisite/logits/log file. It deliberately does not hash
itself, `independent_verification.json`, or `report.md`; those files are
written only after the verifier validates the manifest domain. The verifier
separately requires the exact final inventory and rejects links, sockets,
extra files, duplicate row IDs, non-finite numbers, reordered command
identities, mixed source trees, or mixed model manifests.

`summary.json` is producer output only. The final classification is copied
from `independent_verification.json`.

## Independent Verification

The verifier:

- does not import the runner's aggregation or classification functions;
- reads raw rows and tensors directly;
- recomputes every SHA-256, median, percentile, ratio, and classification;
- validates the exact workload and repetition matrix;
- reconstructs P0/P1 pairs by workload and repetition;
- validates output IDs and logits independently;
- validates cache accounting equations and capacity parity;
- checks every worker log for declared completion and forbidden tracebacks;
- writes its result atomically;
- rejects tampered prerequisites, source identity, commands, rows, tensors,
  cache counters, memory values, or thresholds.

## Remote Safety

The authority uses only:

```text
sitian@10.232.195.203
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
```

It reuses the current strict TP4 GPU selector:

- minimum free bytes per GPU: `24 * 1024**3`;
- exactly four unique GPUs;
- no active compute process on any selected GPU;
- identity and free-memory recheck immediately before launch.

Cleanup is limited to child processes created by the current run and follows:

```text
terminate
bounded join
kill survivors
bounded join
fail closed if any child survives
```

The four-rank runtime uses one monotonic group deadline. No command may use
`pkill`, kill an arbitrary PID, remove another run, mutate another checkout,
or weaken the resource guard.

## Claim Boundary

Before a canonical independent `GO`, the only allowed claims are:

- the benchmark contract exists;
- dependency-light contract and tamper tests pass;
- the run is blocked by correctness or resources;
- no performance/cache result has been measured.

After P0/P1 `GO`, the allowed claim is:

> Exact full-fidelity Qwen3.5 TP4 hybrid-prefix restore improved the registered
> repeated-prefix workloads within the measured cache and memory bounds.

It is still forbidden to claim:

- int4, token-sparse, low-rank, or Gist compression benefit;
- general quality preservation outside registered workloads;
- universal serving throughput improvement;
- reduced cache bytes relative to no cache;
- schema-v2 production authorization.

Every compressed P2 candidate requires its own approved design, correctness
gate, source identity, and canonical result.

## Success Criteria for This Design Stage

1. The design and implementation plan are written without changing the frozen
   TP4 correctness source.
2. A dependency-light contract can classify missing prerequisites as
   `BLOCKED_CORRECTNESS` without importing Torch or launching SSH.
3. Contract tests freeze policy names, workloads, repetitions, thresholds,
   artifact names, pairing order, and claim boundaries.
4. No remote GPU workload is launched while prerequisites or resources remain
   blocked.
5. The handoff records that this stage establishes benchmark authority only
   and proves no speed or cache benefit.
