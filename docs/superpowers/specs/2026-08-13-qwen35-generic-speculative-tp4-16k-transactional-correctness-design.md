# Qwen3.5 Generic Speculative TP4 16K Transactional Correctness

## Status

Approved for implementation and execution on August 13, 2026.

This is a new, independent 16K-only authority. It must not modify, replace,
parameterize, or reinterpret the established Qwen3.5 TP4/4K authority.

Repository constraints forbid staging, committing, pushing, switching
branches, creating a worktree, stashing, resetting, or cleaning the checkout.

## Goal

Establish source-bound evidence that the production generic speculative
runtime executes the approved Qwen3.5 hybrid checkpoint at tensor-parallel
world size four and 16,384 prompt tokens while preserving exact greedy output,
transactional full-attention KV semantics, and transactional recurrent state
semantics on every rank.

The 16K authority must prove:

1. all four ranks load the approved Qwen3.5 checkpoint;
2. baseline and n-gram candidate output token IDs are exactly equal;
3. batch-1 and batch-4 are real Engine runs, not replayed or synthetic rows;
4. every rank executes `spec_first_target` and `spec_verify`;
5. proposal, acceptance, rejection, committed-input, KV, and recurrent-state
   decisions agree across ranks for every sequence;
6. every candidate cell contains accepted and rejected draft tokens;
7. accepted full-attention KV comes from the batch-native verification result
   without a second accepted-prefix full-model forward;
8. rejected KV suffixes and rejected recurrent side state remain unpublished;
9. recurrent state publishes the canonical consumed-input checkpoint;
10. speculative residency prepare, precommit, seal, and cleanup are observed;
11. the batch-4 candidate cell records real positive H2D copies and bytes from
    the production KV-offload path;
12. all process groups, workers, leases, prepared transactions, and Engines
    are cleaned up; and
13. an independent verifier fails closed for missing, inconsistent, tampered,
    or source-unbound evidence.

Passing establishes only:

```text
SECOND_MODEL_TP4_16K_ESTABLISHED
```

The overall Phase 1 status remains:

```text
NOT_PROMOTABLE
```

## Fixed Inputs

### Workspace and remote

All local changes are confined to:

```text
/Users/bytedance/dev/TinyLLMForge-adaptive-ngram
```

The only authorized remote target is:

```text
sitian@10.232.195.203
```

Remote commands use:

```text
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
```

SSH, rsync, launch, and polling are serial, bounded, and use finite retries.

### Model

Checkpoint:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-hybrid-state-runs/
qwen35-2b-hybrid-acquire-20260723-222004/model
```

Remote Python:

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
```

Approved model-manifest SHA-256:

```text
3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

Required identity:

- `model_type=qwen3_5`;
- architecture `Qwen3_5ForConditionalGeneration`;
- 24 text layers;
- 18 linear-attention layers; and
- 6 full-attention layers.

### Frozen gate matrix

- schema:
  `qwen35.generic-speculative-tp4-16k-transactional-correctness.v1`;
- classification: `SECOND_MODEL_TP4_16K_ESTABLISHED`;
- claim scope: `second_model_tp4_16k_only`;
- world size: `4`;
- batch sizes: `1` and `4`;
- policies: `baseline` and `ngram`;
- context tokens: `16384`;
- output tokens: `8`;
- n-gram size: `3`;
- maximum proposal tokens: `4`;
- decoding: greedy;
- `max_model_len=33024`;
- `max_num_batched_tokens=132096`;
- `max_num_prefill_tokens_per_step=1024`;
- `chunked_prefill_decode_first=False`;
- `chunked_prefill_mixed_batch=False`;
- `kv_offload_gpu_blocks=68`;
- `kv_offload_logical_blocks=640`;
- `kv_offload_blockwise_blocks=8`;
- blockwise prefill: enabled; and
- blockwise decode: enabled.

The capacity values intentionally reuse the already exercised long-context
authority envelope. At 16K and batch 4, the prompt occupies 256 logical
16-token blocks, exceeding the 68 GPU-resident block budget. Therefore the
candidate batch-4 cell must observe production H2D movement. A zero-H2D
candidate batch-4 result is a gate failure, not an optimization result.

## Authority Boundary

The implementation creates:

```text
tools/qwen35_generic_speculative_tp4_16k_gate.py
tools/qwen35_generic_speculative_tp4_16k_worker.py
tools/verify_qwen35_generic_speculative_tp4_16k_gate.py
tools/run_qwen35_generic_speculative_tp4_16k_gate_remote.sh
tools/test_qwen35_generic_speculative_tp4_16k_gate.py
```

It may reuse side-effect-free orchestration and validators from the frozen 4K
authority, but the 16K module must override the schema, classification, claim
scope, context length, source inventory, and long-context movement rule before
any campaign executes. The 4K files and their established artifacts remain
byte-for-byte untouched.

The production runtime remains:

- `tinyvllm/engine/llm_engine.py`;
- `tinyvllm/engine/model_runner.py`;
- `tinyvllm/engine/speculative_model_runner.py`;
- `tinyvllm/engine/qwen35_speculative_state.py`;
- `tinyvllm/engine/speculative_side_state.py`;
- `tinyvllm/speculative/batch_runtime.py`; and
- the existing Qwen3.5 packed/hybrid model stack.

The authority code must not implement a second speculative runtime, synthetic
KV movement, or accepted-prefix replay.

## Result Contract

The result contains the same rank-local and cross-rank transactional evidence
as the frozen 4K authority, with the independent 16K constants above.

Additionally:

- `context_tokens` must equal `16384`;
- every cell must report Engine configuration matching the frozen long-context
  values;
- the candidate batch-4 aggregate must have `h2d_copies > 0`;
- the candidate batch-4 aggregate must have `h2d_bytes > 0`;
- H2D evidence must be derived from rank-local production KV-offload summaries;
- baseline/candidate token parity remains exact for every request;
- `accepted_prefix_model_forward_count` must be zero; and
- source binding must include both the frozen reusable 4K authority modules
  and all new 16K authority modules.

No H2D count or byte threshold beyond positivity is a performance claim.

## Failure Semantics

The gate fails closed for:

- wrong schema, classification, scope, matrix, model, or source identity;
- missing rank or cell;
- baseline/candidate token mismatch;
- missing acceptance or rejection;
- rank disagreement;
- invalid consumed-input mapping;
- missing KV prepare/commit/rollback evidence;
- missing recurrent prepare/select/apply/seal/rollback evidence;
- accepted-prefix replay;
- missing callback, collective, residency, cleanup, or poison evidence;
- batch-4 candidate H2D copies or bytes equal to zero;
- malformed or tampered source/result manifests; or
- replay of an existing run directory.

Failure writes `authority.failed`. Only a successful campaign followed by a
successful independent source-bound verification may write `authority`.

## Validation

Validation proceeds in this order:

1. focused local contract tests;
2. focused RED/GREEN tests for the long-context overlay and worker config;
3. all 16K authority tests;
4. frozen 4K authority and existing Qwen3.5 regression tests;
5. Python compilation and runner `bash -n`;
6. remote source-copy direct tests where remote `pytest` is unavailable;
7. one fresh, non-replayable real TP4 GPU campaign;
8. independent verification against the copied source tree;
9. artifact audit, `git diff --check`, and handoff/audit documentation update.

## Explicit Non-Claims

This gate does not establish:

- 32K context;
- TPOT, TTFT, throughput, memory, or traffic improvement;
- lower H2D traffic than baseline;
- learned drafter or native MTP;
- KV8 or KV4;
- production readiness; or
- Phase 1 promotion.
