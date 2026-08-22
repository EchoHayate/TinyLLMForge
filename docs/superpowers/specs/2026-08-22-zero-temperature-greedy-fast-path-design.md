# Zero-Temperature Greedy Fast Path Design

Date: 2026-08-22

## Objective

Reduce Qwen3-0.6B batch-1 generation TPOT by bypassing stochastic-sampling
work when the request is already semantically greedy
(`temperature == 0.0`), while preserving the existing float32-argmax result,
exact output tokens, decoded text, logits, fallback behavior, and protected
latency, throughput, and memory metrics.

The candidate is motivated by the current TinyLLMForge implementation rather
than by a paper transplant. `Sampler.forward()` currently performs all of the
following even when every row is greedy:

1. cast logits to float32;
2. compute the greedy argmax;
3. upload a temperature tensor;
4. divide the complete vocabulary logits by temperature;
5. compute a complete-vocabulary softmax;
6. fill a complete-vocabulary Gumbel buffer with exponential RNG;
7. divide probabilities by the Gumbel values;
8. compute a second argmax;
9. select the first argmax with `torch.where`.

For a zero-temperature request, steps 3 through 9 do not contribute to the
selected token. The proposed path keeps steps 1 and 2 exactly and removes the
semantically dead work.

## Scope

Stage 1 is intentionally narrow:

- model: Qwen3-0.6B;
- tensor parallel size: one;
- batch size: one;
- ordinary generation only;
- exact `temperature == 0.0`;
- no mixed batch;
- feature flag default: disabled;
- benchmark prompt lengths: 256, 2048, and 8192 tokens;
- generated tokens: 128;
- two warmups and five measured repetitions;
- alternating OFF/ON construction order.

The implementation uses the same helper for the final prefill sample and all
subsequent decode samples because both consume the same logits and sampling
policy. A 128-token Stage-1 ON row must therefore record exactly 128 optimized
sampling steps. The performance gate reports TTFT and decode TPOT separately
so a TTFT gain cannot mask a decode regression.

No claim may be extended to:

- nonzero-temperature sampling;
- mixed greedy/stochastic batches;
- batch sizes above one;
- tensor parallel execution;
- speculative verification;
- Qwen3-8B;
- another model family.

## Alternatives

### A. Host-policy greedy short circuit

Before uploading temperatures, inspect the request-owned Python sampling
policy. If the exact Stage-1 eligibility contract is satisfied, compute:

```python
logits.to(torch.float32).argmax(dim=-1)
```

and return its token IDs. Otherwise call the existing
`prepare_sample()` plus `Sampler.forward()` path unchanged.

This is the selected approach. It removes the maximum known dead work while
preserving the existing greedy precision and tie behavior.

### B. Branch inside `Sampler.forward()`

Upload temperatures first, reduce them on GPU, then branch inside the sampler.
This preserves a smaller public surface but retains the temperature H2D and
requires a device-to-host decision or graph-specialized control flow. It is
strictly weaker for the Stage-1 batch-1 case.

### C. GPU-resident token handoff

Avoid `.tolist()` and keep the sampled token on device for the next decode
step. This has larger theoretical upside, but it crosses scheduler,
`Sequence`, stop-condition, tokenizer, and KV transaction ownership
boundaries. It is a separate optimization and is excluded here.

## Architecture

Create a dependency-light policy module:

```text
tinyvllm/engine/greedy_sampling_fast_path.py
```

It owns immutable eligibility and accounting records:

```python
GreedySamplingFastPathDecision
GreedySamplingFastPathStats
decide_greedy_sampling_fast_path(...)
```

`ModelRunner` owns one stats object and a private sampling router:

```python
_sample_tokens_with_optional_greedy_fast_path(
    logits,
    sample_seqs,
    *,
    is_prefill,
    batch_kind,
) -> list[int]
```

The router:

1. evaluates all eligibility conditions from host-owned request metadata;
2. records one explicit fallback reason when ineligible;
3. on the eligible path, executes float32 argmax and `.tolist()`;
4. on fallback, executes the current temperature upload and sampler exactly;
5. exposes a JSON-serializable cumulative summary.

The feature flag is:

```python
zero_temperature_greedy_fast_path: bool = False
```

Non-boolean values are rejected by `Config.__post_init__()`.

## Eligibility

Stage-1 optimization requires all of:

- feature flag is `True`;
- rank is zero;
- exactly one sampled sequence;
- `batch_kind != "mixed"`;
- request temperature is exactly numeric `0.0`;
- logits have two dimensions and one row.

Every rejected condition has a stable fallback reason. Eligibility must be
resolved before calling `prepare_sample()` so the optimized path performs no
temperature tensor H2D.

## Semantic preservation

The old greedy token is computed before stochastic sampling as:

```python
greedy_tokens = logits.to(torch.float32).argmax(dim=-1)
```

The new path must execute the same expression. It must not:

- argmax bf16/fp16 logits directly;
- change tie-breaking;
- mutate logits;
- consume random numbers;
- alter the stochastic fallback;
- inspect a CUDA scalar to decide eligibility.

Because the branch is driven entirely by host-owned sampling parameters, it
adds no synchronization before argmax.

## Accounting

The cumulative summary records:

- `eligible_steps`;
- `optimized_steps`;
- `fallback_counts`;
- `avoided_temperature_h2d_bytes`;
- `avoided_softmax_calls`;
- `avoided_gumbel_rng_calls`;
- `avoided_stochastic_divisions`;
- `avoided_stochastic_argmax_calls`;
- `avoided_where_calls`.

For Stage-1 batch size one, each optimized step avoids four temperature bytes.
Counters describe eliminated operations; they are not a substitute for
measured performance.

## Benchmark evidence

The source-bound worker produces:

```text
case_rows.jsonl
correctness_rows.jsonl
workload_manifest.json
source_manifest.json
summary.json
comparison.json
gate.json
manifest.sha256
independent-verification.json
```

Each performance row contains:

- immutable run and source identity;
- context bucket and repetition;
- OFF/ON policy;
- exact output token IDs;
- decoded-text SHA256;
- TTFT, E2E, and throughput;
- all per-token TPOT samples;
- decode host and CUDA timings;
- CUDA allocated and reserved peaks;
- fast-path accounting summary.

The correctness probe records paired pre-sampling logits for exactly three
sampling points in every context bucket: the final prefill sample, the first
true decode sample, and the final true decode sample. Each selected logits
row is serialized as little-endian float32 bytes with an adjacent shape and
SHA256 record. The gate computes:

- maximum absolute logit difference;
- mean absolute logit difference;
- argmax equality;
- exact generated-token equality;
- exact decoded-text hash equality.

The bounded float32 sidecars are retained and bound by the manifest. The
producer and independent verifier reconstruct the metrics from those bytes
rather than trusting worker booleans.

## Stage-1 gate

Correctness is mandatory:

- output token IDs exactly equal for every pair;
- decoded-text hashes exactly equal for every pair;
- logit `max_abs <= 0.25`;
- logit `mean_abs <= 0.05`;
- logit argmax equal;
- every ON row records exactly 128 optimized generation steps;
- every OFF generation step records zero optimized steps.

Performance requires:

- median TPOT improvement `>= 5%` in at least two of three buckets;
- aggregate nearest-rank P95 TPOT improvement `>= 5%`;
- no bucket median or P95 TPOT regression above `3%`;
- no TTFT or E2E regression above `3%`;
- no throughput regression above `2%`;
- no CUDA reserved-memory regression above `1%`.

Cost reporting is mandatory:

- avoided operation counts;
- temperature H2D bytes avoided;
- persistent CUDA-memory delta;
- peak allocated and reserved CUDA-memory deltas;
- any host-side state added by the implementation.

The terminal classifications are:

```text
GO_ZERO_TEMPERATURE_GREEDY_FAST_PATH
NO_GO_CORRECTNESS
NO_GO_OPTIMIZED_PATH_INCOMPLETE
NO_GO_TPOT_MEDIAN
NO_GO_TPOT_P95
NO_GO_PROTECTED_REGRESSION
NO_GO_EVIDENCE_INCOMPLETE
```

## Remote execution and safety

The local controller must:

- require local HEAD to equal `origin/feat/kv-sparse-attention`;
- require at least 5,400 seconds of Kerberos lifetime without refreshing it;
- use the existing SSH ControlMaster transport;
- admit only a GPU with memory used `<=1024 MiB`, utilization `<=5%`, and no
  compute process;
- never terminate unrelated processes;
- write all remote task state below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`;
- isolate `TMPDIR`, Python bytecode, Hugging Face, XDG, and Torch extension
  caches below the approved run staging directory;
- reject a reused run tag;
- preserve failed and partial evidence;
- download and verify all terminal artifacts locally.

## Promotion boundary

`zero_temperature_greedy_fast_path` stays default-disabled unless the
Qwen3-0.6B Stage-1 producer and independent verifier both return
`GO_ZERO_TEMPERATURE_GREEDY_FAST_PATH` with matching comparison and manifest
digests.

A Stage-1 GO authorizes a separate Qwen3-8B confirmation run; it does not
itself establish an 8B claim. Any NO-GO keeps the flag default-disabled,
prevents the 8B run, and is recorded as a complete negative result.
