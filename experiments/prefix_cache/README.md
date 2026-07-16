# Prefix Cache Gate

## Purpose

Validate correctness-safe cross-request full-block KV reuse and measure prompt
prefill and time-to-first-token effects. This gate does not measure decode
acceleration, physical KV allocation reduction, or increased KV-cache capacity.

## Implemented Safety Rules

- Reusable hashes are published only after the corresponding prefill forward
  completes.
- A sampled prefill keeps at least one query token:
  `floor((prompt_tokens - 1) / block_size) * block_size`.
- Requests in the same scheduler batch cannot consume blocks first computed by
  another request in that batch.
- Hash hits are checked against the original block token IDs.
- Clearing reusable metadata preserves live referenced blocks.

## Local Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_prefix_cache.py
python3 -m py_compile \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tools/profile_prefix_cache.py
bash -n tools/run_prefix_cache_gate_remote.sh
```

These tests cover lifecycle and reporting logic without loading a model. They do
not substitute for the remote Qwen3-0.6B correctness and TTFT gate.

## Remote Gate

GPU/model experiments run only on `sitian@10.232.195.203`, from an isolated
uploaded source tree. The runner creates a run-local temporary directory and
uses independent dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT` values.

One-repetition smoke:

```bash
TAG=20260716-smoke REPETITIONS=1 WARMUP_REPETITIONS=1 \
  tools/run_prefix_cache_gate_remote.sh
```

Canonical gate:

```bash
TAG=20260716 REPETITIONS=7 WARMUP_REPETITIONS=2 \
  tools/run_prefix_cache_gate_remote.sh
```

The canonical output directory will be:

```text
experiments/prefix_cache/qwen3_0_6b_gate_20260716/
```

Expected files:

- `manifest.json`
- `correctness_rows.json`
- `performance_rows.json`
- `summary.json`
- `report.md`

## Decision Rules

The generated decision is `GO` only when all of the following hold:

- every correctness and lifecycle row passes;
- greedy output token IDs and decoded text exactly match cold baselines;
- full-vocabulary logits preserve argmax with `max_abs <= 0.25` and
  `mean_abs <= 0.05`;
- warm cached-token counts equal the expected reusable full-block prefix;
- cold minus warm executed query tokens equals the expected reusable prefix;
- warm median TTFT improves by at least 20% for shared prefixes of 1024 and
  2048 tokens;
- no warm median TTFT regresses by more than 5%.

The 256-token case is retained as a small-prefix regression detector, but it is
not required to reach the 20% improvement threshold.

The profiler captures full logits for correctness. It keeps the GPU-to-CPU
transfer outside the timed region and uses CUDA events to measure the remaining
GPU clone instrumentation. Performance rows store `raw_ttft_ms`,
`capture_overhead_ms`, and adjusted `ttft_ms`; the gate compares the adjusted
value so profiler-only full-logit capture does not bias cold/warm TTFT.

## Current Status

The safety implementation, CPU regressions, report helpers, executable GPU
profiler, isolated remote runner, and source-hash manifest support are complete.
No canonical `GO` or `NO_GO` has been recorded yet.

Remote execution is currently blocked by expired Kerberos credentials for
`sitian@BYTEDANCE.COM`; the latest SSH probe returned
`Connection closed by UNKNOWN port 65535`. Do not update the root README or
handoff with a final APC decision until the canonical `summary.json` has been
mirrored locally and audited.

## Prompt-to-Artifact Audit

Current evidence is intentionally split between completed local checks and
remote-only checks that remain blocked:

- [x] Compute-complete publication:
  `tools/test_chunked_prefill.py` and `tinyvllm/engine/scheduler.py`.
- [x] Sampleable suffix leaves a positive query row:
  255/256/257/512/513 CPU boundary tests.
- [x] Same-batch publication isolation:
  normal/chunked scheduler CPU regressions.
- [x] Hash-collision token validation:
  `test_allocate_rejects_hash_collision_when_tokens_differ`.
- [x] Cache clearing and live-block safety:
  `test_clear_reusable_cache_preserves_live_block_metadata` and capacity
  pressure coverage.
- [x] Gate threshold logic:
  `tools/test_profile_prefix_cache.py`.
- [x] Cold/warm/cache-cleared profiler paths and cached/query-token reporting:
  `tools/profile_prefix_cache.py`.
- [x] Source hashes and isolated upload:
  profiler manifest plus `tools/run_prefix_cache_gate_remote.sh`.
- [x] Claim boundaries:
  generated `report.md` template and this README.
- [ ] Remote 255/256/257/512/513 exact-token and full-logit rows:
  requires canonical `correctness_rows.json`.
- [ ] Remote `[P,Q,P]` wrong-row isolation:
  requires canonical `correctness_rows.json`.
- [ ] Remote shared-prefix/different-suffix and cache-cleared cases:
  requires canonical `correctness_rows.json`.
- [ ] Seven-sample cold/warm/cache-cleared TTFT for 256/1024/2048:
  requires canonical `performance_rows.json` and `summary.json`.
- [ ] Final threshold decision and rejection reasons:
  requires canonical `summary.json`.
- [ ] Root `README.md` and `AGENT_HANDOFF_STATE.md` final result:
  must be selected from the audited canonical decision, not inferred.

Any unchecked item means the APC performance gate is incomplete.

## Claim Boundaries

- A passing gate supports correctness-safe full-block cross-request prefix
  reuse and measured prefill/TTFT improvement for the tested Qwen3-0.6B setup.
- It does not support claims about decode speed.
- It does not support claims about lower physical KV allocation or greater
  model/context capacity.
- It does not validate radix trees, partial-block reuse, cache-aware scheduling,
  same-batch dependency waves, or final-hidden-state/logits caching.
