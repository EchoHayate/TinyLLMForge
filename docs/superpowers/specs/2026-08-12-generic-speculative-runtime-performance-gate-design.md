# Generic Speculative Runtime Performance Gate Design

**Date:** 2026-08-12

**Status:** Executed; authoritative TP1 4K campaign passed

**Classification before and after this gate:** `NOT_PROMOTABLE`

## Goal

Create the first authoritative end-to-end performance gate for the current
generic speculative runtime and transactional KV residency path. Compare an
ordinary greedy baseline with
`EngineSpeculativeRuntime(NGramDraftAdapter)` under the same loaded target
model, prompts, MVP-0 configuration, and output budget.

The first campaign answers only whether the current implementation shows a
repeatable positive direction at TP1, 4K input context, and batch sizes 1 and
4. It does not establish promotion readiness or generalize to TP4, longer
contexts, another model, a learned drafter, or MTP.

## Current Evidence Gap

The schema-v2 TP1 parity artifact proves exact greedy output parity through the
generic runtime. The residency-boundary artifact proves accepted in-place
commit, rejected rollback without D2H, and real MVP-0 H2D movement. Neither
artifact is a controlled performance comparison:

- each boundary case loads a separate engine and reports one uninterpreted
  elapsed duration;
- the old `tools/profile_ngram_commit.py` manually drives speculative
  transactions and calls `rematerialize_accepted_kv()`;
- old profiler upload/rematerialization counters are not the current
  `LLMEngine.activate_speculative_runtime()` execution path;
- no current artifact reports repeated TTFT, TPOT, throughput, peak GPU
  memory, or per-run real MVP-0 movement.

Therefore end-to-end optimization is not established before this gate runs.

## Alternatives

### Reuse `profile_ngram_commit.py`

Rejected. It measures an older manually orchestrated path, contains accepted
KV rematerialization, and cannot be used as evidence for the current generic
runtime.

### Benchmark baseline and candidate inside one long-lived engine

Rejected. A speculative runtime cannot be uninstalled, and switching policy
inside one engine would mix allocator, prefix-cache, movement-counter, and
peak-memory state.

### Separate policy workers with a pure artifact assembler

Selected. Run one fresh worker process for every `(policy, batch_size)` pair.
Each worker loads exactly one engine, performs one warmup run, one parity run,
and five measured runs. A parent gate assembles the four worker results and an
independent verifier recomputes hashes and validates all claims.

This keeps model-load time outside request latency while preventing baseline
and candidate runtime state from contaminating each other.

## Campaign Matrix

The first campaign is fixed:

```text
model=Qwen3-0.6B
tensor_parallel_size=1
input_prompt_tokens=4096
max_output_tokens=64
batch_sizes=(1, 4)
policies=(baseline, ngram)
temperature=0.0
ignore_eos=True
warmup_runs=1
parity_runs=1
measured_runs=5
ngram_size=3
max_proposal_tokens=4
classification=NOT_PROMOTABLE
```

The engine configuration is also fixed:

```text
enforce_eager=True
max_model_len=4352
max_num_batched_tokens=16384
max_num_prefill_tokens_per_step=1024
chunked_prefill_mixed_batch=False
kv_offload_mvp0=True
kv_offload_gpu_blocks=68
kv_offload_logical_blocks=128
kv_offload_blockwise_decode=False
kv_offload_blockwise_prefill=False
kv_offload_blockwise_blocks=1
```

An input contains exactly 4096 prompt tokens. The 64-token output budget fits
within `max_model_len=4352`. Chunked prefill prevents a batch-4 prompt set from
requiring one monolithic 16384-token forward.

The generic speculative verifier rejects both blockwise decode and blockwise
prefill, so this gate uses the full-attention-compatible configuration above.
After the first completion token is emitted, the worker writes back all active
sequence blocks, synchronizes the real MVP-0 copy path, and calls
`evict_clean_resident_blocks()` with the exact `(logical_block, generation)`
identities. The next decode restores those blocks through real H2D. Baseline
and candidate use the same configuration and eviction timing.

## Deterministic Workload

The worker tokenizes a fixed, versioned repetitive seed once, repeats the
resulting token pattern, and truncates it to exactly 4096 token IDs. Batch
members use fixed seed variants recorded in the artifact. The token IDs, seed
texts, tokenizer identity, and SHA-256 digest of every prompt are stored.

The workload is deliberately repetitive so the real `NGramDraftAdapter` can
find proposals. It is not a fixture adapter and does not force target
acceptance. A candidate result is invalid unless the measured campaign
observes:

```text
selected_rows > 0
proposal_rows > 0
proposed_tokens > 0
accepted_draft_tokens > 0
first_target_callbacks > 0
tail_callbacks > 0
```

The exact acceptance rate is reported rather than prescribed.

## Worker Lifecycle

Each worker owns one policy and one batch size:

1. construct one engine with the fixed configuration;
2. install `EngineSpeculativeRuntime(NGramDraftAdapter(...))` only for the
   candidate policy;
3. execute one unrecorded warmup request batch;
4. execute one recorded parity request batch;
5. execute five recorded measured request batches;
6. exit the engine in `finally`.

Within every request batch, after the first completion token appears, the
worker performs the fixed clean-writeback/eviction sequence described above
exactly once. This creates positive real H2D evidence without enabling a
blockwise mode that is incompatible with speculative verification.

Before every parity or measured run, the worker:

- requires the scheduler to be idle;
- calls `clear_reusable_prefix_cache()`;
- synchronizes CUDA;
- records rank-wise MVP-0 summaries;
- resets rank-wise CUDA peak-memory statistics through a generic acknowledged
  engine API;
- synchronizes CUDA again immediately before starting the timer.

After the run, the worker synchronizes CUDA before the finish timestamp,
collects rank-wise memory snapshots, and computes movement as the after-minus-
before delta of cumulative `KVOffloadMVP0` summaries. MVP-0 statistics are
never reset or synthesized.

## Peak-Memory Reset API

Add one generic rank-aware method:

```python
ModelRunner.reset_peak_memory_stats() -> dict
LLMEngine.reset_peak_memory_stats(*, timeout_s: float) -> tuple[dict, ...]
```

`ModelRunner.reset_peak_memory_stats()` synchronizes its current CUDA device,
calls `torch.cuda.reset_peak_memory_stats()`, and returns
`memory_snapshot()`. `LLMEngine.reset_peak_memory_stats()` dispatches the
operation through the existing acknowledged command channel and validates one
rank-tagged result per TP rank.

The API contains no model, n-gram, or benchmark behavior. It is needed because
constructor-time cumulative peaks cannot identify one measured request batch.

## Timing Semantics

Every timed boundary uses `time.perf_counter_ns()` with CUDA synchronization
immediately before the start and after every `engine.step()`.

All requests in a batch share the same arrival timestamp. For each sequence:

```text
TTFT = synchronized end of the first step that emits a completion token
       - batch start

completion latency = synchronized end of the step that finishes the sequence
                     - batch start

TPOT = (completion latency - TTFT) / (output token count - 1)
```

`max_output_tokens=64` and `ignore_eos=True`, so TPOT always has a nonzero
denominator. When one speculative step emits multiple tokens, those tokens
share that step's synchronized end timestamp. TPOT is therefore an observed
end-to-end average, not a kernel-level per-token duration.

Per-run batch metrics are:

```text
batch token throughput = total completion tokens / batch elapsed seconds
request throughput = completed requests / batch elapsed seconds
```

Model loading, engine construction, warmup, cache clearing, counter snapshots,
and artifact I/O are excluded.

## Runtime and Acceptance Metrics

The worker aggregates `last_step_observation` without adding benchmark-only
branches to the engine:

- engine step count;
- scheduled prefill and decode step counts;
- selected rows and proposal rows;
- proposed and accepted draft tokens;
- acceptance rate;
- first-target callback count;
- fixed-Q tail callback count;
- emitted speculative token count;
- accumulated generic runtime timing fields.

`first_target_callbacks + tail_callbacks` is reported as the observed target
callback count for the generic speculative path. Baseline target work is
reported as engine step count. These are diagnostic counts with different
semantics and are not presented as a cross-policy one-for-one forward count.

## Real KV Movement

For every parity and measured run, collect rank-wise summary deltas for the
existing real counters, including:

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

Every delta must be an integer and non-negative. Each policy/batch campaign
must have positive aggregate `h2d_copies` and `h2d_bytes`; otherwise the
MVP-0 performance comparison fails closed. No simulated copy, manual upload,
or accepted-KV rematerialization is allowed.

## Peak GPU Memory

For each measured run and rank, store:

- allocated and reserved bytes immediately after peak reset;
- peak allocated and peak reserved bytes after completion;
- KV capacity bytes;
- peak-minus-reset allocated and reserved deltas.

The batch-level value is the maximum over TP ranks. TP1 is the only current
matrix member, but the schema remains rank-aware.

## Exact Parity

The parity run and every measured run store output token IDs. For each batch
size:

- every baseline run must be internally identical;
- every candidate run must be internally identical;
- baseline and candidate outputs must match exactly by prompt digest and
  sequence position.

Any mismatch writes a FAIL artifact and prohibits performance interpretation.

## Aggregation and Direction

The artifact stores all five raw measured runs. For TTFT, TPOT, completion
latency, token throughput, request throughput, movement, acceptance, and peak
memory, it reports median, minimum, maximum, and population standard
deviation. Per-request latency is first reduced to one batch-run median before
the five-run campaign aggregate is calculated.

For each batch size:

```text
IMPROVED:
  candidate median TPOT < baseline median TPOT
  and candidate median token throughput > baseline median token throughput

REGRESSED:
  candidate median TPOT > baseline median TPOT
  and candidate median token throughput < baseline median token throughput

MIXED:
  all other combinations, including equality
```

The campaign direction is `POSITIVE` only when both batch sizes are
`IMPROVED`; `NEGATIVE` when either is `REGRESSED`; otherwise it is `MIXED`.
No minimum percentage threshold or significance claim is made with five runs.
TTFT and peak memory are reported as trade-offs and do not change this first
direction label.

Even a `POSITIVE` direction remains `NOT_PROMOTABLE`.

## Components

Create:

```text
tools/speculative_runtime_performance_worker.py
tools/speculative_runtime_performance_gate.py
tools/verify_speculative_runtime_performance_gate.py
tools/run_speculative_runtime_performance_gate_remote.sh
tools/test_speculative_runtime_performance_gate.py
```

The worker owns loaded-engine execution. The gate owns subprocess orchestration
and artifact assembly. The verifier imports only pure schema-validation and
hashing helpers; it never imports CUDA/model runtime code.

## Artifact

Schema version `1` stores:

- status, classification, claim scope, direction, and limitations;
- exact command, environment, device, model, tokenizer, dtype, and source
  hashes;
- fixed campaign and engine configuration;
- prompt seeds, token IDs, token counts, and digests;
- raw warmup diagnostics, parity outputs, and five measured runs per case;
- rank-wise counter deltas and memory snapshots;
- aggregated latency, throughput, acceptance, invocation, movement, and
  memory metrics;
- per-batch and campaign direction labels.

The independent verifier recomputes source hashes and all derived aggregates
and labels from the raw runs.

## Failure Boundary

The gate fails closed if:

- a worker exits nonzero or does not produce exactly one policy/batch result;
- any prompt is not exactly 4096 tokens or any request does not emit 64 tokens;
- exact greedy parity fails;
- the candidate does not execute proposals, callbacks, and at least one
  accepted draft token;
- MVP-0 H2D evidence is absent for any policy/batch campaign;
- a movement delta is negative, malformed, or not derived from before/after
  summaries;
- peak reset or rank acknowledgement is incomplete;
- any measured run count differs from five;
- source hashes fail independent verification;
- the artifact contains a promotion or generalized speedup claim.

A diagnostic FAIL artifact is written whenever the parent has enough worker
state to do so.

## Remote Execution

The authoritative runner uses:

```text
REMOTE_HOST=sitian@10.232.195.203
CONTROL_SOCKET=/tmp/ssh-sitian-10.232.195.203
REMOTE_PYTHON=/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
MODEL_PATH=/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
REMOTE_REPO=/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
GPU_ID=0
```

It synchronizes the complete current `tinyvllm/` tree and the gate, worker,
verifier, and test files; runs a fixed-venv `py_compile` preflight remotely
before the campaign because the remote model environment has no `pytest`;
always downloads logs and partial worker results; and performs both remote
and local independent verification. Local dependency-light pytest remains the
test authority.

## Non-Goals

- no modification or reuse of `profile_ngram_commit.py`;
- no model-name or proposal-source branch in runtime, Scheduler, allocator, or
  verifier;
- no TP4, second model, 16K/32K context, nonzero temperature, or EOS-sensitive
  generation;
- no learned drafter or MTP claim;
- no statistical significance, production SLO, or promotion threshold;
- no simulated movement or accepted-KV replay/rematerialization;
- no comparison against another inference system.

## Expansion Boundary

Only after this TP1 4K batch-1/batch-4 campaign has exact parity, real
movement, real acceptance, and a recorded direction should the matrix expand
in this order:

1. 16K and 32K contexts;
2. TP4;
3. a second supported model;
4. learned drafter or MTP adapters;
5. promotion thresholds with more repetitions and variance controls.

Until those stages are complete:

```text
end-to-end optimization:
  determined only for the fixed first-campaign cells
general performance claim:
  not established
overall classification:
  NOT_PROMOTABLE
```

## Executed Campaign Result

The final source-hash-aligned campaign is:

```text
artifact:
  artifacts/speculative_runtime_performance/20260812T085852Z/result.json
remote verifier:
  artifacts/speculative_runtime_performance/20260812T085852Z/verify.remote.json
local verifier:
  artifacts/speculative_runtime_performance/20260812T085852Z/verify.json
artifact SHA-256:
  d987b288176beec3ab841a7f640bb1d68cabf86445cadad50aec6337bcc4fb9f
schema/status:
  1 / PASS
batch directions:
  batch 1 = IMPROVED
  batch 4 = IMPROVED
campaign direction:
  POSITIVE
classification:
  NOT_PROMOTABLE
```

Exact baseline/candidate output-token parity passed for the one batch-1 row
and all four batch-4 rows, each with 64 output tokens. Both policies recorded
the same real H2D totals over five measured runs:

```text
h2d_copies=80
h2d_bytes=2348810240
```

Candidate acceptance and median end-to-end metrics were:

```text
batch 1:
  proposed/accepted=300/300
  acceptance_rate=1.0
  median TTFT=0.483959581 s
  median TPOT=0.038769407 s
  median token throughput=21.869633566 tok/s

batch 4:
  proposed/accepted=1240/1225
  acceptance_rate=0.9879032258
  median TTFT=4.091824249 s
  median TPOT=0.026363790 s
  median token throughput=28.774541098 tok/s
```

Relative to baseline medians, TPOT decreased by 22.27% at batch 1 and 48.02%
at batch 4, while token throughput increased by 25.08% and 66.80%,
respectively. Peak allocated and reserved bytes were unchanged within each
batch-size comparison. These are fixed-cell measurements from five runs, not
a significance result or a generalized promotion claim.
