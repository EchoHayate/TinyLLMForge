# Production Arrival-Load Continuous-Batching Gate Design

Date: 2026-07-18

## Objective

Build a source-auditable, fail-closed canonical gate that measures how the
existing TinyLLMForge scheduler behaves under request arrivals instead of
preloading a fixed batch.

The first phase must:

1. Drive real request arrival times through `LLMEngine.add_request()` while
   repeatedly advancing the synchronous engine with `LLMEngine.step()`.
2. Compare the repository-default scheduler with explicit decode-first,
   bounded-consecutive-prefill, and mixed prefill-plus-decode policies.
3. Measure request-level injection lag, queue delay, time to first token
   (TTFT), inter-token latency (ITL), end-to-end latency, maximum decode gap,
   starvation, throughput, fairness, and peak CUDA/KV usage.
4. Preserve exact greedy output and complete every admitted request without
   dropping, truncating, or silently changing its sampling contract.
5. Calibrate one default-policy saturation reference, freeze a workload
   manifest, and reuse that exact manifest for every policy and repetition.
6. Require at least three measured repetitions for every non-duplicate case
   and report both median and worst-repetition results.
7. Save raw arrival, lifecycle, scheduler, output, memory, environment, and
   source evidence that an independent verifier can recompute without
   trusting the harness summary.
8. Produce `GO`, `PROMISING_NOT_PROVEN`, `NO_GO`, or `INCOMPLETE` without
   changing thresholds after results are visible.

This gate evaluates scheduler policies that already exist in the repository.
It does not implement a new scheduler in the first phase and does not claim a
complete network serving stack.

## Motivation

The completed speculative-decoding investigations established useful
infrastructure and several local fast regions, but neither the K1/SAM gate nor
the profitability-router canonical gate produced a deployable performance
`GO`. The router canonical result remains `NO_GO`; a controlled K16 full-accept
case was locally faster, but the complete workload failed correctness and
performance-direction gates. Those results must not be reinterpreted as an
overall engine speedup.

The repository already contains materially different scheduling behavior:

- waiting, prefilling, and running queues;
- continuous decode batching;
- chunked prefill;
- decode-first admission;
- bounded consecutive prefill chunks;
- mixed prefill-plus-decode batches;
- prefix-aware admission.

Existing profilers mainly submit a known set of requests before stepping the
engine. They therefore cannot answer the production-relevant questions:

- What happens when requests arrive while decode work is active?
- Where does the default policy saturate?
- Does a policy improve throughput without damaging tail TTFT or ITL?
- Does long-prompt pressure starve decoding?
- Does a favorable mean hide a bad repetition, request bucket, or maximum
  decode gap?
- Does a policy reduce peak CUDA or KV pressure under the same arrivals?

The next useful gate is therefore arrival-load measurement around the current
engine, not another speculative microbenchmark.

## Alternatives Considered

### 1. Recommended: External Inline Arrival Driver

Use an external, deterministic driver that owns the arrival clock, calls
`LLMEngine.add_request()` when requests become due, and calls
`LLMEngine.step()` inline whenever the engine has work.

Advantages:

- evaluates the real scheduler and model path with minimal behavioral change;
- keeps workload generation, event collection, aggregation, and verification
  separable;
- can measure arrival pressure before a server or async API exists;
- makes every request and scheduler step attributable in raw artifacts;
- supports dependency-light synthetic tests and a small remote lifecycle
  smoke before expensive canonical runs.

Limitations:

- the synchronous driver is not a full production serving stack;
- tokenizer, HTTP, RPC, network, and client scheduling latency are excluded;
- one Python process controls both injection and stepping, so injection lag
  must be measured and guarded rather than assumed negligible.

### 2. Build an Async Serving Front End First

Add an HTTP or RPC server, async admission tasks, and client-side load
generation before measuring scheduler policy.

This is deferred. It would mix transport overhead, concurrency implementation,
request parsing, and scheduler behavior in one first result. A failed run would
not identify which layer caused the failure.

### 3. Modify the Scheduler Before Establishing a Baseline

Introduce a new fairness or preemption algorithm and compare it with the
current default.

This is rejected for the first phase. The current scheduler already exposes
multiple policies, but their arrival-load behavior is not measured. Changing
the scheduler before establishing a trustworthy gate would optimize without a
production-shaped baseline.

## Decision

Implement Alternative 1 as an independent arrival-load canonical gate.

The gate owns workload creation, load driving, raw event capture, aggregation,
artifact verification, and remote orchestration. It may add observation-only
engine instrumentation where an event is otherwise not observable, but it
must not alter scheduling decisions, queue ordering, token generation, block
allocation, or policy defaults.

All GPU/model execution occurs remotely on:

- host `sitian@10.232.195.203`;
- Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`;
- model
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.

Local execution is limited to source preparation, dependency-light tests,
artifact inspection, aggregation tests, and independent verification.

## Architecture

The gate has six bounded components.

### 1. Arrival Workload Builder

The builder creates immutable calibration and canonical workload manifests.
It owns:

- request identity;
- prompt text or prompt-bank reference and hash;
- scheduled arrival offset;
- sampling parameters;
- prompt-length and requested-output-length metadata;
- scenario, seed, and arrival-rate identity;
- service-time bucket identity.

It does not initialize the model or run the engine.

### 2. Inline Load Driver

The driver reads one frozen workload manifest, initializes one engine policy,
and executes an event loop:

1. derive `scheduled_arrival_ns` from a monotonic run epoch and the request's
   frozen arrival offset;
2. inject every request whose arrival is due;
3. record `actual_arrival_ns` immediately before `add_request()`;
4. recover the new `seq_id` from the newly appended waiting sequence and bind
   it to the immutable request ID;
5. call `step()` when the engine has pending work;
6. record scheduler and token-progress events after the synchronous step
   completes;
7. continue until every manifest request finishes or a preregistered watchdog
   fails the run.

The driver captures raw facts only. It must not compute percentiles, choose a
winner, or classify a gate.

### 3. Minimal Engine Instrumentation

Observation-only instrumentation exposes lifecycle data that cannot be
reliably reconstructed outside the engine:

- scheduled sequence IDs and phase for each step;
- queue sizes before and after scheduling;
- prefill and decode token counts;
- policy branch selected by the scheduler;
- allocation/high-water information needed for KV accounting.

Instrumentation must not:

- reorder queues;
- introduce synchronization solely to improve reported timings;
- change scheduler policy selection;
- mutate sequence state outside the existing lifecycle;
- make event collection a prerequisite for normal engine execution.

### 4. Offline Aggregator

The aggregator consumes raw manifests and event streams and produces
request-, repetition-, case-, scenario-, and policy-level metrics. It owns all
percentile and gate computations used by the harness summary.

### 5. Independent Verifier

The verifier parses raw files independently. It must not import the driver's
aggregator, percentile helpers, or classification implementation. It
reconstructs request metrics and the final classification, validates source
and process identity, and writes its own result under
`independent-verify/`.

### 6. Remote Runner

The runner packages the approved source snapshot, uploads it to a run-local
directory, validates it before model initialization, executes calibration and
canonical cases serially, downloads artifacts, and supports verification-only
or download-only recovery without relaunching model work.

Every model process receives a unique dynamic pair:

- `TINYVLLM_DIST_PORT`;
- `MASTER_PORT`.

The runner must not kill unrelated GPU processes, clean system `/tmp`, modify
another checkout, or reuse another run's source or artifact directory.

## Policy Matrix

The gate compares four named policies while preserving all non-policy engine
settings.

Common configuration:

- `max_num_batched_tokens=16384`;
- `max_num_seqs=512`;
- `max_num_prefill_tokens_per_step=128`;
- `enforce_eager=False`;
- identical model, tokenizer, dtype, sampling parameters, block size, and
  memory settings;
- greedy decoding for exact output comparison.

The fixed nonzero chunk size is part of the preregistered gate rather than a
candidate policy. It enables the existing chunked-prefill scheduler for every
policy and is not tuned after calibration or policy results. The value `128`
matches the repository's existing remote chunked-prefill validation setting.

### P0: Repository Default

Construct the engine with the common fixed chunk size but without overriding
the four chunked-prefill policy fields. Record the fully resolved
configuration after initialization.

At design time, the expected defaults are:

- `chunked_prefill_decode_first=True`;
- `chunked_prefill_max_consecutive_chunks=0`;
- `chunked_prefill_mixed_batch=False`;
- `chunked_prefill_mixed_min_prompt_tokens=0`.

The recorded runtime configuration, not this expectation, is authoritative.
A default drift is evidence and must not be silently normalized.

### P1: Explicit Decode-First Chunked Prefill

Set:

- `chunked_prefill_decode_first=True`;
- `chunked_prefill_max_consecutive_chunks=0`;
- `chunked_prefill_mixed_batch=False`;
- `chunked_prefill_mixed_min_prompt_tokens=0`.

P1 is an explicit control for default drift. If its fully resolved
configuration equals P0, it is marked `duplicate_configuration` and is not
counted as an independent performance policy. Its alias relationship remains
in the artifact.

### P2: Bounded-Consecutive Chunked Prefill

Set:

- `chunked_prefill_decode_first=False`;
- `chunked_prefill_max_consecutive_chunks=2`;
- `chunked_prefill_mixed_batch=False`;
- `chunked_prefill_mixed_min_prompt_tokens=0`.

This permits at most two consecutive prefill chunks before yielding to
already-running decode work.

### P3: Mixed Prefill Plus Decode

Set:

- `chunked_prefill_decode_first=False`;
- `chunked_prefill_max_consecutive_chunks=0`;
- `chunked_prefill_mixed_batch=True`;
- `chunked_prefill_mixed_min_prompt_tokens=0`.

This evaluates the existing mixed varlen-forward path without adding a new
admission threshold.

### Policy Identity and Deduplication

Policy identity is the canonical JSON serialization and SHA-256 of every
resolved scheduler-affecting configuration field, not merely the policy name.

If two policies resolve to the same identity:

- both names remain in the manifest;
- one canonical execution supplies the shared raw evidence;
- aliases cannot increase repetition count or vote independently;
- comparisons between duplicate identities are invalid;
- the verifier rejects a summary that treats aliases as independent evidence.

An unexpected collision between non-control policies is `INCOMPLETE` until
the configuration or manifest is corrected.

## Workload Design

### Prompt and Output Classes

The canonical prompt bank contains fixed short, medium, and long prompts and
fixed short and long requested-output budgets. Token counts are measured once
with the canonical tokenizer and frozen in the workload manifest.

Every request records:

- prompt token count;
- requested maximum output tokens;
- prompt class;
- output class;
- service-time bucket.

Service-time buckets are the fixed cross-product of prompt class and output
class. They are not recomputed from observed latency.

### Arrival Generation

Steady-load scenarios use seeded exponential inter-arrival times. The builder
stores the realized arrival offset for every request, so no policy or
repetition resamples arrivals.

Burst scenarios store explicit burst boundaries and per-request offsets. A
burst is represented by many requests becoming due in a bounded interval,
followed by a fixed recovery interval.

The manifest records the generator version, seed, requested rate, realized
duration, request count, prompt-bank hash, and exact ordered request list.

### Saturation Calibration

Calibration runs only P0 and uses a separate calibration manifest. It searches
increasing offered arrival rates until the default policy reaches a stable
throughput ceiling or shows sustained backlog growth.

For each candidate rate, calibration records:

- offered requests and output tokens per second;
- completed requests and output tokens per second;
- end-of-window unfinished request count;
- queue/backlog trajectory;
- TTFT and E2E tail latency;
- GPU and KV high-water marks.

`lambda_ref` is the highest offered request rate whose measured window:

- completes all requests within the bounded drain period;
- has no sustained positive backlog slope in the final measurement segment;
- has finite request metrics and exact outputs;
- reaches at least 95% of the maximum stable completed-request throughput
  observed during the search.

If no stable point or no clear ceiling is observed within the preregistered
search bounds, calibration is `INCOMPLETE`; the harness cannot guess
`lambda_ref`.

After calibration, `lambda_ref`, all generator parameters, and every realized
arrival offset are frozen before any policy comparison.

### Canonical Scenarios

The canonical matrix contains:

1. **Steady moderate:** `0.6 * lambda_ref`, balanced prompt/output classes.
2. **Near saturation:** `0.9 * lambda_ref`, the same class mixture.
3. **Overload:** `1.2 * lambda_ref`, followed by a bounded drain interval.
4. **Burst:** frozen request bursts separated by recovery windows.
5. **Long-prompt pressure:** elevated long-prompt share while decode requests
   are active.
6. **Mixed service time/fairness:** all fixed service-time buckets with enough
   requests per bucket to report tails and starvation.

Scenario duration, warmup requests, measured requests, drain timeout, seeds,
and prompt-bank membership are fixed in the workload manifest generated after
calibration. They cannot vary by policy.

Each non-duplicate policy/scenario case has at least three measured
repetitions. Execution order is deterministically interleaved by repetition
and scenario so one policy is not systematically measured only first or last.

## Timeline and Raw Event Contract

All timestamps use `time.monotonic_ns()` from one process and one run-local
clock domain.

For every request, the raw timeline contains:

- `request_id`;
- `seq_id`;
- `scheduled_arrival_ns`;
- `actual_arrival_ns`;
- `first_scheduled_ns`;
- `first_token_ns`;
- `token_timestamps_ns`;
- `completion_ns`;
- `output_token_ids`;
- prompt and requested-output token counts;
- finish reason and error state.

Definitions:

- `scheduled_arrival_ns`: run epoch plus the frozen arrival offset;
- `actual_arrival_ns`: timestamp immediately before `add_request()`;
- `first_scheduled_ns`: first scheduler event containing the sequence;
- `first_token_ns`: timestamp after the step that first appends a completion
  token;
- `token_timestamps_ns[i]`: timestamp after the step that makes completion
  token `i` observable;
- `completion_ns`: timestamp after the engine marks the sequence finished.

When one synchronous step produces multiple completion tokens for a request,
those tokens share the step-completion timestamp. The event records the token
count delta so this is distinguishable from missing timestamps.

The scheduler trace contains one row per engine step with:

- step index and start/end timestamps;
- queue sizes before and after scheduling;
- scheduled sequence IDs and execution phase;
- prompt, decode, and total scheduled token counts;
- chunk and mixed-batch metadata;
- scheduler policy branch;
- newly produced token counts by sequence;
- finished sequence IDs;
- CUDA allocated/reserved memory observations;
- KV block usage and high-water observations when available.

Raw streams are append-only. An interrupted or malformed final record makes
the affected repetition `INCOMPLETE`.

## Metric Definitions

Primary request metrics use the scheduled arrival, not actual injection, so a
lagging driver cannot hide offered-load failure.

- `injection_lag = actual_arrival_ns - scheduled_arrival_ns`;
- `queue_delay = first_scheduled_ns - actual_arrival_ns`;
- `TTFT = first_token_ns - scheduled_arrival_ns`;
- `E2E = completion_ns - scheduled_arrival_ns`;
- `ITL[i] = token_timestamps_ns[i] - token_timestamps_ns[i-1]` for `i > 0`;
- `maximum_decode_gap = max(ITL)` for multi-token outputs.

Requests with one output token have no ITL sample. They remain in TTFT, E2E,
throughput, completion, and correctness metrics.

The gate reports:

- request throughput and output-token throughput over the measured interval;
- p50, p95, and p99 TTFT, ITL, E2E, queue delay, and injection lag;
- maximum decode gap and maximum injection lag;
- unfinished, dropped, truncated, and starved request counts;
- per-service-time-bucket p50/p95/p99 and worst request;
- peak CUDA allocated and reserved bytes;
- peak used KV blocks and peak KV bytes when derivable;
- repetition median and worst repetition for every gated metric.

A request is starved if either:

- it remains admitted but is not first-scheduled before the scenario drain
  deadline; or
- after its first output token, one decode gap exceeds the fixed starvation
  deadline stored in the workload manifest.

The gate also reports diagnostic fairness:

- each service-time bucket's completed-request share;
- each bucket's throughput and p95 E2E relative to P0 for the same manifest;
- Jain's index over the per-bucket normalized service rates, where each
  policy's bucket service rate is divided by P0's rate for that bucket.

Fairness cannot be improved by omitting an unfinished bucket: a missing or
zero-completion required bucket is a correctness failure, not an excluded
sample.

Percentiles use one documented nearest-rank implementation over sorted finite
samples. The harness and independent verifier implement it separately.

## Correctness and Completeness

P0 establishes the canonical greedy output for every request ID in a frozen
workload. Every other non-duplicate policy and every measured repetition must:

- admit the exact same request set;
- preserve prompt and sampling parameters;
- produce token-identical output;
- produce the expected output-token count and finish reason;
- complete all requests within the bounded drain period;
- emit monotonic and internally consistent timestamps;
- emit exactly one lifecycle record per request;
- contain no unaccounted sequence or duplicate request binding.

P0 itself must be internally repeatable. Any P0 output disagreement across
measured repetitions is `NO_GO`, because no alternate policy can be evaluated
against an unstable baseline.

No result may improve by:

- dropping or rejecting requests;
- shortening output budgets;
- changing prompt mix or arrival offsets;
- starving long requests;
- excluding a service-time bucket;
- subtracting warm execution or synchronization costs from only one policy;
- selecting only the best repetition.

## Preregistered Classification

All comparisons are paired by scenario, frozen workload manifest, and
repetition index. P0 is the baseline identity. P1 aliases P0 when their
resolved configurations match.

### Structural `INCOMPLETE`

The result is `INCOMPLETE` if any required evidence cannot support a complete
comparison, including:

- failed or ambiguous `lambda_ref` calibration;
- missing/truncated artifacts or non-finite metrics;
- fewer than three complete measured repetitions;
- duplicate or reused process-port pairs;
- source, environment, prompt-bank, or workload hash disagreement;
- invalid policy identity or improper duplicate-policy counting;
- missing request, scheduler, output, or memory events;
- independent verifier failure to complete.

`verify.exitcode=0` means only that the verifier completed. It does not mean
the gate passed.

### Correctness `NO_GO`

A structurally complete result is `NO_GO` if any policy/repetition has:

- output-token, finish-state, or request-set mismatch;
- dropped, rejected, truncated, unfinished, or starved requests;
- timestamp or lifecycle inconsistency;
- an unexplained sequence or token-progress event.

Correctness failures cannot be overridden by performance.

### Performance `GO`

A correct candidate policy produces `GO` only if at least one canonical
scenario satisfies one of these benefit paths against P0:

1. median throughput improves by at least 5%, while median p95 TTFT and p95
   ITL each regress by no more than 5%; or
2. median p95 TTFT or p95 ITL improves by at least 10%, while median
   throughput regresses by no more than 2% and the other p95 latency metric
   regresses by no more than 5%; or
3. median peak KV bytes or peak CUDA reserved bytes improves by at least 5%,
   while median throughput, p95 TTFT, and p95 ITL each regress by no more than
   2%.

For the same candidate, all canonical scenarios must satisfy the regression
guards:

- p99 TTFT, p99 ITL, and p99 E2E regress by no more than 10%;
- maximum decode gap regresses by no more than 10%;
- every required service-time bucket's p95 E2E regresses by no more than 10%;
- the worst measured repetition preserves the same correctness and
  starvation requirements;
- no gated median is produced from fewer finite samples than P0.

The benefit path must hold both for the median across repetitions and in a
directionally consistent worst repetition. A single best repetition cannot
produce `GO`.

### `PROMISING_NOT_PROVEN`

A correct, complete candidate is `PROMISING_NOT_PROVEN` when:

- its median direction is favorable in a preregistered benefit path;
- it misses the required improvement magnitude or worst-repetition
  confirmation;
- it does not violate any 10% tail, decode-gap, or service-bucket guard.

This classification is diagnostic and cannot be described as a performance
win.

### Performance `NO_GO`

A correct, complete result is `NO_GO` when no candidate reaches `GO` or
`PROMISING_NOT_PROVEN`, or when a candidate's apparent benefit violates a
tail, maximum-gap, service-bucket, starvation, or worst-repetition guard.

The gate-level classification is the best valid candidate classification,
except that any global structural incompleteness yields `INCOMPLETE` and any
cross-policy correctness instability yields `NO_GO`.

## Artifact Contract

Each run has a unique directory:

```text
experiments/arrival_load/<run_tag>/
```

The canonical artifact contains:

- `run_manifest.json`;
- `calibration_manifest.jsonl`;
- `calibration_rows.jsonl`;
- `workload_manifest.jsonl`;
- `request_timeline.jsonl`;
- `scheduler_trace.jsonl`;
- `memory_trace.jsonl`;
- `case_rows.jsonl`;
- `summary.json`;
- `report.md`;
- `source_evidence.json`;
- `source.patch`;
- `source_snapshot.tar.gz`;
- `artifact_hashes.json`;
- per-process stdout, stderr, and exit-code files;
- environment and package identity;
- model/tokenizer identity;
- prompt-bank content and hash;
- process-port allocation records;
- `independent-verify/`.

`run_manifest.json` records:

- base commit and source tree hash;
- command lines and resolved configurations;
- remote host, Python, model, tokenizer, dtype, and GPU identity;
- calibration result and frozen `lambda_ref`;
- scenario and repetition matrix;
- policy identities and aliases;
- every process's unique `TINYVLLM_DIST_PORT` and `MASTER_PORT`;
- artifact schema version.

The artifact retains raw records even when a run fails. `summary.json` and
`report.md` are derived views and are never authoritative.

## Independent Verification

The independent verifier must:

1. verify artifact hashes and source identity;
2. verify model, prompt-bank, workload, policy, process, and port identity;
3. reconstruct the expected case matrix and repetition count;
4. validate request-set equality and request-to-sequence bindings;
5. validate timestamp ordering and token timestamp counts;
6. validate exact outputs and completion states;
7. recompute injection lag, queue delay, TTFT, ITL, E2E, decode gap,
   throughput, memory peaks, percentiles, service buckets, and fairness;
8. reject duplicate policy identities counted as independent evidence;
9. recompute every regression guard and final classification;
10. compare its recomputation with `summary.json` and report any disagreement.

The verifier writes:

- `independent-verify/summary.json`;
- `independent-verify/report.md`;
- `independent-verify/verify.stdout`;
- `independent-verify/verify.stderr`;
- `independent-verify/verify.exitcode`.

Artifact absence, truncation, malformed JSON, hash disagreement, impossible
timestamp ordering, sample-count disagreement, or classification disagreement
fails closed.

## Error Handling and Resume

Each process writes its exit code even when the model command fails. A
watchdog failure records the last known queue state and preserves partial raw
events.

Resume rules:

- successful, hash-valid measured repetitions are immutable;
- only failed or structurally incomplete repetitions may be rerun;
- replacing a repetition replaces all of its stale request, scheduler,
  memory, stdout, stderr, and exit-code records;
- calibration cannot be silently rerun after a canonical policy result
  exists; a new calibration requires a new run tag and workload manifest;
- changed source, prompt bank, workload, policy configuration, threshold, or
  environment requires a new run tag.

Changing the common chunk size also requires a new run tag and a new
calibration. The canonical gate never selects a chunk size from candidate
policy results.

The runner supports download-only and verify-only recovery. Neither mode may
launch model work.

## Testing Strategy

### Dependency-Light Tests

Local tests must cover:

- seeded steady and burst workload determinism;
- exact manifest ordering and hashing;
- scheduled versus actual arrival accounting;
- request-to-sequence binding;
- lifecycle reconstruction with multiple tokens from one step;
- one-token output with no ITL sample;
- queue delay, TTFT, ITL, E2E, and maximum decode-gap calculations;
- nearest-rank p50/p95/p99 boundaries;
- throughput measurement windows;
- service-time buckets and fairness;
- peak CUDA/KV aggregation;
- starvation and drain-timeout classification;
- policy identity and duplicate-configuration rejection;
- three-repetition completeness and worst-repetition selection;
- every `GO`, `PROMISING_NOT_PROVEN`, `NO_GO`, and `INCOMPLETE` boundary;
- output, request-set, timestamp, port, source, hash, and artifact failures;
- resume immutability and failed-row replacement;
- independence of verifier aggregation and classification code.

Existing scheduler and chunked-prefill tests remain required regression
coverage.

### Remote Lifecycle Smoke

Before calibration, run a two-to-four-request smoke that includes:

- one request arriving while another is decoding;
- one long prompt requiring chunked prefill;
- one multi-token output with observable ITL;
- one case that exercises the selected policy branch;
- exact output comparison against P0;
- raw timeline, scheduler trace, memory trace, and unique port evidence;
- local independent verification after download.

The smoke is an instrumentation and lifecycle check. It cannot produce a
performance classification.

### Calibration and Canonical Execution

After the smoke verifies:

1. calibrate P0 and freeze `lambda_ref`;
2. generate and hash the canonical workload manifest;
3. execute the policy/scenario/repetition matrix serially;
4. preserve each process artifact before starting the next process;
5. download the complete run directory;
6. run the local independent verifier;
7. publish only the verifier-confirmed classification.

## Implementation Phases

Implementation follows these checkpoints:

1. dependency-light workload, timeline, aggregation, and gate tests;
2. source-auditable artifact and independent-verifier tests;
3. inline driver and observation-only instrumentation;
4. remote two-to-four-request lifecycle smoke;
5. P0 saturation calibration;
6. frozen canonical workload generation;
7. serial policy/repetition execution;
8. artifact download and local independent verification;
9. documentation of what the result proves, does not prove, and the next
   decision.

No canonical experiment starts before the lifecycle smoke artifact verifies.
No performance claim is made before independent verification.

## Claim Boundaries

A verified `GO` proves only that one existing scheduler policy improves the
preregistered throughput, latency, or memory objective on the frozen
Qwen3-0.6B arrival-load matrix while preserving exact greedy output and the
tail/fairness guards.

It does not prove:

- production readiness;
- HTTP, RPC, tokenizer, network, or multi-client latency;
- behavior on another model, tokenizer, GPU, prompt distribution, or output
  distribution;
- stochastic-sampling equivalence;
- multi-node or tensor-parallel scaling;
- compatibility with every speculative, KV-offload, Quest, Attention
  Matching, quantized-KV, or prefix-cache mode;
- that the policy should become the repository default without a separate
  integration decision.

`PROMISING_NOT_PROVEN` is not a win. `NO_GO` preserves a useful negative
result and redirects optimization rather than inviting threshold tuning on
the same workload.

## Non-Goals

The first phase does not:

- modify scheduler behavior or introduce a new scheduling algorithm;
- implement a complete serving API or async request stack;
- include tokenizer, HTTP, RPC, network, or client-side latency;
- parallelize policy processes on one GPU;
- clean unrelated remote processes or system temporary files;
- reduce quality, alter outputs, drop requests, or starve long requests to
  improve a metric;
- use averages or a single best repetition in place of raw evidence, medians,
  tails, and worst repetitions;
- promote the failed K1, SAM, or speculation-router paths;
- generalize one model/workload result to all models or production traffic.
