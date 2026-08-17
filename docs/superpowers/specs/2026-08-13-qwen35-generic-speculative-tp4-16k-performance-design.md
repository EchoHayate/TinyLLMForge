# Qwen3.5 Generic Speculative TP4/16K Performance Gate Design

**Date:** 2026-08-13

**Status:** Approved for autonomous execution by the existing no-confirmation
instruction

**Phase classification before and after this gate:** `NOT_PROMOTABLE`

## Goal

Create an independent, source-bound performance authority for the real
Qwen3.5 hybrid model running the generic n-gram speculative runtime at TP4,
16,384 input tokens, and batch sizes 1 and 4.

The gate must report synchronized TTFT, TPOT, completion latency, token and
request throughput, peak GPU memory, real KV H2D/D2H movement, speculative
acceptance, target callback counts, and exact greedy parity. It must not infer
performance from the existing correctness authority, synthetic tensor copies,
or helper-only microbenchmarks.

This is the first of two independent long-context performance gates. The 32K
gate follows only after the 16K methodology and artifact verifier pass.

## Existing Evidence and Gap

The following correctness authority already exists:

```text
artifacts/qwen35_generic_speculative_tp4_16k/
  opaque-3b8050a916f037bc92412ea5/
    artifacts/authority/
```

It proves exact greedy parity, transactional commit/rollback, zero
accepted-prefix replay, real KV movement, all-rank callbacks, and clean
shutdown. It does not establish performance because each policy/batch cell
contains one short eight-token generation and no repeated synchronized timing
distribution.

The existing TP1 performance gate under
`tools/speculative_runtime_performance_gate.py` provides reusable pure metric
and validation logic, but its loaded worker is not valid for TP4:

- it fixes `tensor_parallel_size=1`;
- it manually evicts active history through the rank-0 local KV manager;
- its campaign and environment schema claim TP1 and 4K;
- it does not carry Qwen3.5 distributed cleanup evidence.

The new gate may reuse pure timing, aggregation, counter-delta, and memory
validation helpers, but it must own an independent TP4 worker, gate,
verifier, runner, schema, and artifact directory.

## Alternatives

### Derive performance from the 16K correctness artifact

Rejected. The artifact has one run per cell, only eight output tokens, no
warmup, no measured distribution, and profiling enabled for diagnostic
correctness rather than controlled end-to-end comparison.

### Generalize the frozen TP1 performance gate in place

Rejected. Changing its constants, worker lifecycle, movement behavior, and
schema would invalidate the source-bound TP1 authority and blur checkout
history.

### Independent TP4/16K gate reusing only pure helpers

Selected. Create Qwen3.5-specific TP4 orchestration and loaded execution while
reusing well-tested pure functions such as synchronized run-metric
calculation, counter subtraction, measurement aggregation, and direction
classification.

This preserves existing authorities and keeps the new claim boundary explicit.

## Fixed Campaign

```text
schema:
  qwen35.generic-speculative-tp4-16k-performance.v1

classification:
  SECOND_MODEL_TP4_16K_PERFORMANCE_MEASURED

model:
  approved Qwen3.5 hybrid checkpoint

tensor_parallel_size:
  4

prompt_tokens:
  16,384 per request

max_output_tokens:
  64

batch_sizes:
  (1, 4)

policies:
  (baseline, ngram)

temperature:
  0.0

ignore_eos:
  true

warmup_runs:
  1

parity_runs:
  1

measured_runs:
  5

ngram_size:
  3

max_proposal_tokens:
  4
```

The five measured runs provide a raw distribution and median. They do not
establish statistical significance.

## Engine Configuration

Every policy/batch cell constructs one fresh Engine process group with:

```text
tensor_parallel_size=4
enforce_eager=True
max_model_len=33024
max_num_batched_tokens=132096
max_num_seqs=batch_size
max_num_prefill_tokens_per_step=1024
chunked_prefill_decode_first=False
chunked_prefill_mixed_batch=False
kv_offload_mvp0=True
kv_offload_gpu_blocks=48
kv_offload_logical_blocks=640
kv_offload_blockwise_decode=True
kv_offload_blockwise_prefill=True
kv_offload_blockwise_blocks=8
```

The physical GPU staging budget is deliberately reduced from the correctness
gate's 68 blocks to 48. A 16K request exceeds 48 blocks, so both baseline and
candidate must exercise natural production KV eviction and restoration. The
worker must not call a benchmark-only upload, tensor copy, forced rank-0
eviction, or accepted-KV rematerialization helper.

## Workload

The worker reuses the approved deterministic prompt builder from
`tools/qwen35_generic_speculative_tp4_16k_gate.py`. Every prompt row contains
exactly 16,384 token IDs and records its seed, token digest, tokenizer
identity, and prompt index.

Before every parity or measured run:

1. require the scheduler to be idle;
2. clear the reusable prefix cache;
3. collect rank-wise cumulative KV-offload summaries;
4. reset rank-wise CUDA peak-memory statistics;
5. synchronize the rank-0 CUDA stream immediately before the timer.

The same Engine remains alive for the seven request batches inside one cell,
but baseline and candidate always use separate fresh worker processes.
Sequence IDs may increase; prompt token IDs and output expectations may not
change.

## Timing Semantics

Use `time.perf_counter_ns()` and `torch.cuda.synchronize()` on the rank-0
stream immediately before the start timestamp and after every
`engine.step()`.

TP collectives couple the rank streams. Synchronizing the rank-0 stream after
the collective-bearing step waits for the target work without adding a new
all-rank command-channel round trip to every decode step.

For each request:

```text
TTFT =
  synchronized end of the first step that emits completion tokens
  - batch start

completion latency =
  synchronized end of the finishing step
  - batch start

TPOT =
  (completion latency - TTFT) / 63
```

For each request batch:

```text
batch token throughput =
  total completion tokens / synchronized batch elapsed time

request throughput =
  completed requests / synchronized batch elapsed time
```

Model loading, worker startup, warmup, counter collection, peak reset,
artifact serialization, and worker cleanup remain outside measured request
latency.

## Runtime and Acceptance Evidence

Each run aggregates the public `last_step_observation` fields:

- engine step count;
- prefill and decode step counts;
- selected and proposal rows;
- proposed and accepted draft tokens;
- first-target callback count;
- fixed-Q verifier callback count;
- speculative output tokens;
- generic runtime timing fields.

Every candidate measured campaign must have positive proposal rows, proposed
tokens, accepted draft tokens, first-target callbacks, and verifier callbacks.
Acceptance rate is reported from real accepted/proposed token counts and is
not prescribed.

The separate 16K correctness authority remains the source of the explicit
zero accepted-prefix replay proof. This performance gate must source-bind the
same runtime files and must not contain a rematerialization call, but it does
not relabel source inspection as a new replay counter.

## Real KV Movement

Every parity and measured run records before/after deltas from
`engine.kv_offload_summaries()` for all four ranks:

```text
h2d_copies
h2d_bytes
d2h_copies
d2h_bytes
copy_waits
evictions
evict_clean
speculative_residency_committed_blocks
speculative_residency_rejected_blocks
speculative_residency_rejected_d2h_copies
```

All deltas must be non-negative integers and equal their rank-wise sums.
Both batch-4 policy cells must have positive aggregate H2D and D2H copies and
bytes. Candidate rejected speculative D2H copies must remain zero.

Batch-1 movement may be zero only if the raw per-rank evidence is complete;
the fixed 48-block budget is expected to make it positive.

## Peak GPU Memory

For every measured run and every rank, store:

- allocated and reserved bytes immediately after peak reset;
- peak allocated and peak reserved bytes after the run;
- KV capacity bytes;
- peak-minus-reset allocated and reserved deltas.

The cell aggregate reports the five-run median, minimum, maximum, and
population standard deviation for the maximum peak across the four ranks.
Memory direction is reported as a trade-off and does not decide the primary
performance direction.

## Exact Parity

For each batch size:

- all baseline parity and measured runs must produce the same 64 token IDs per
  prompt;
- all candidate parity and measured runs must produce the same token IDs;
- baseline and candidate prompt rows and outputs must match exactly.

Any mismatch invalidates the artifact before latency or throughput is
interpreted.

## Aggregation and Direction

Store all raw runs. Per-request metrics are reduced to one run-level median,
then the five run-level values are summarized with:

```text
count
median
min
max
population standard deviation
```

For each batch size:

```text
IMPROVED:
  candidate median TPOT < baseline median TPOT
  and candidate median batch token throughput > baseline median throughput

REGRESSED:
  candidate median TPOT > baseline median TPOT
  and candidate median batch token throughput < baseline median throughput

MIXED:
  every other combination
```

The campaign direction is `POSITIVE` only if both batch sizes are
`IMPROVED`, `NEGATIVE` if either batch is `REGRESSED`, and `MIXED` otherwise.

The artifact also reports candidate/baseline ratios and percentage deltas for
TTFT, TPOT, throughput, peak allocated memory, H2D bytes, and D2H bytes.
No minimum effect-size or statistical-significance claim is made.

`SECOND_MODEL_TP4_16K_PERFORMANCE_MEASURED` means the measurement is
authoritative, not that the optimization improved. A speedup claim is allowed
only when the independently recomputed direction is `POSITIVE`.

## GPU Selection and Stability

The bounded remote runner:

- uses only `sitian@10.232.195.203`;
- uses `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- disables SSH ControlMaster and ControlPath;
- selects one fixed four-GPU set before the campaign;
- records the full `nvidia-smi` inventory and selected physical indices;
- checks selected-GPU free memory and utilization before every cell;
- records post-cell inventory after the worker exits;
- fails closed if any selected GPU has less than 48 GiB free or more than 10%
  utilization before a cell;
- fails closed if post-cell free memory differs from the pre-cell value by
  more than 4 GiB after a bounded settle interval.

The runner uses serial SSH calls, finite retries, an opaque run ID, bounded
polling, and retains partial worker logs/results under `authority.failed`.

## Components

Create:

```text
tools/qwen35_generic_speculative_tp4_16k_performance_gate.py
tools/qwen35_generic_speculative_tp4_16k_performance_worker.py
tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py
tools/run_qwen35_generic_speculative_tp4_16k_performance_gate_remote.sh
tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py
```

Reuse without modifying:

```text
tools/speculative_runtime_performance_gate.py
tools/qwen35_generic_speculative_tp4_16k_gate.py
tools/qwen35_generic_speculative_tp4_worker.py
tools/run_qwen35_generic_speculative_tp4_16k_gate_remote.sh
```

The new gate owns pure schema validation, worker orchestration, source
hashing, derived ratios, failure artifacts, and final artifact assembly. The
worker owns loaded TP4 execution and cleanup. The verifier imports only the
new gate's pure validation and source-hash functions.

## Failure Boundary

Fail closed and retain evidence if:

- a worker exits nonzero or omits its JSON result;
- the selected GPU set changes;
- pre-cell GPU capacity/utilization checks fail;
- prompt count, length, or digest differs;
- a run emits anything other than 64 output tokens per request;
- exact token parity fails;
- any rank inventory is incomplete;
- a movement delta is malformed, decreasing, or not rank-derived;
- batch-4 real H2D/D2H evidence is absent;
- candidate proposals, accepted tokens, or callbacks are absent;
- rejected speculative D2H is nonzero;
- peak-memory reset or final snapshots are incomplete;
- warmup/parity/measured run counts are not exactly 1/1/5;
- cleanup leaves a process group, child, lease, prepared transaction, or
  runtime poison;
- source or model digests fail independent verification;
- a positive-direction claim disagrees with raw medians.

## Claim Boundary

This gate may establish only:

```text
Qwen3.5
generic n-gram speculative runtime
TP4
16K
batch 1/4
controlled repeated performance measurement
```

It does not establish 32K performance, learned-drafter performance, native
MTP performance, KV4/KV8, statistical significance, production readiness, or
Phase 1 completion.
