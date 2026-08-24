# Packed-QK Single-Pass RMSNorm Design

**Date:** 2026-08-24

**Status:** Approved under the standing autonomous-optimization authorization

**Stage-1 model:** Qwen3-0.6B

**Primary target:** reduce decode kernel-launch and memory-traffic overhead by
normalizing the contiguous packed Q/K projection region once per layer

## Objective

Replace the two independent Q and K RMSNorm dispatches in each Qwen3
attention layer with one packed-QK RMSNorm dispatch while preserving the
existing per-head reduction, Q-specific weight, K-specific weight, BF16
rounding boundary, RoPE implementation, attention inputs, and model outputs.

This is a runtime-data-flow-specific original engineering design. It does not
claim academic novelty. The implementation alone does not establish a
performance improvement; a source-bound paired GPU gate must report both
benefit and cost.

## Evidence and Opportunity Boundary

A source-bound Qwen3-0.6B K8 exact-burst profile at commit
`c685b9b626ceac6ddef169a1f6084cbde7c9d76c`, with a 2,048-token prompt and
32 generated tokens, observed:

```text
31 measured decode tokens
8,096 Triton kernel launches
about 261 Triton launches per decode token
```

The same profile confirmed that deleting the standalone KV-cache store has
only a low-single-digit CUDA-time ceiling, so the next opportunity is reducing
the large inventory of small compiled kernels rather than changing KV
ownership.

An isolated 28-layer BF16 CUDA Graph probe compared:

```text
baseline:  separate Q RMSNorm + K RMSNorm   143.315 us
candidate: packed QK single-pass RMSNorm     48.651 us
isolated improvement                          66.05%
maximum absolute output difference              0.0
bitwise equality                               true
```

The probe used distinct Q and K weights and retained a reduction of exactly
128 elements per head. It is mechanism evidence only. It does not prove an
end-to-end gain.

## Current Data Flow

For Qwen3-0.6B TP1, `qkv_proj` returns one contiguous row:

```text
[Q: 16 heads x 128] [K: 8 heads x 128] [V: 8 heads x 128]
```

The current path creates views for Q and K and invokes the same compiled
RMSNorm algorithm twice:

```text
Q view -> Q RMSNorm(weight=q_norm.weight)
K view -> K RMSNorm(weight=k_norm.weight)
```

Both reductions have the same head dimension, dtype, epsilon, token count,
device, and execution point. The only semantic difference is which learned
128-element weight applies to each head.

The selected path treats the contiguous Q+K prefix as:

```text
[tokens, 24 heads, 128]
```

and performs one reduction kernel. Head indices `0..15` use the Q weight and
head indices `16..23` use the K weight. The normalized packed result is then
split back into the original Q and K views before the unchanged RoPE call.

## Fixed Scope

Stage 1 is limited to the existing Qwen3 attention implementation when:

- Q and K are adjacent in the packed QKV projection output;
- Q and K use the same `head_dim`;
- Q and K RMSNorm use the same epsilon;
- both weights are one-dimensional tensors of length `head_dim`;
- the packed prefix is contiguous and viewable as
  `[tokens, q_heads + kv_heads, head_dim]`; and
- the feature is explicitly enabled through a default-disabled config flag.

The path must preserve:

- a separate learned Q weight and K weight;
- a separate RMS reduction for every individual head;
- float32 conversion before the square/mean/rsqrt sequence;
- the existing BF16 conversion before multiplication by the learned weight;
- exact Q and K shapes passed to RoPE;
- all RoPE, KV-store, attention, CUDA Graph, and scheduler behavior; and
- bitwise-identical output tokens and sampled logits in the canonical gate.

The path must not:

- normalize across multiple heads;
- average Q and K statistics together;
- share or average Q/K learned weights;
- fuse RMSNorm with RoPE in Stage 1;
- change QKV packing, checkpoint names, or weight loading;
- add a materialized per-token Q/K concatenation;
- change prefill/decode policy selection;
- become enabled by default; or
- claim benefit from the isolated probe alone.

## Considered Approaches

### A. Packed-QK single-pass RMSNorm

Normalize the already-contiguous Q+K prefix in one compiled function, selecting
the Q or K learned weight by head index.

Advantages:

- removes one small reduction dispatch per attention layer;
- does not copy Q or K into a new activation buffer before normalization;
- preserves the original per-head reduction dimension and arithmetic order;
- leaves RoPE and attention unchanged; and
- has already demonstrated bitwise equality in an isolated BF16 probe.

Costs and risks:

- introduces a Qwen3-specific packed normalization helper;
- the compiler could materialize the logical Q/K weight expansion;
- prefill shapes may choose a less favorable kernel schedule;
- graph capture memory or compile time may increase; and
- isolated savings may be hidden by GEMM and FlashAttention time.

This is the selected approach.

### B. Joint Q/K RMSNorm plus RoPE fusion

Compile normalization and rotation as one larger function.

An isolated probe reduced subpath time by about 25.8%, but changed BF16
results with maximum absolute difference `0.015625`. It violates the current
exact-logit contract and is rejected.

### C. Model-level RoPE materialization hoist

Read the common position row once per model step and reuse it across all
layers.

An isolated 28-layer probe improved the subpath by only about 5.36% and did
not preserve bitwise output in the tested implementation. The measured
standalone index-select contribution was also only about `94 us` over the
complete 31-token profile. It is rejected for Stage 1.

## Architecture

### 1. Default-off configuration

Add:

```text
Config.packed_qk_single_pass_rmsnorm: bool = False
```

Validation requires a boolean. The flag is passed from `LLM` construction to
the Qwen3 model. Unsupported model families or incompatible Q/K topology must
fail closed during model construction; there is no silent numerical fallback
after graph capture.

### 2. Packed normalization helper

Add a narrow compiled helper owned by the Qwen3 attention layer. Its logical
contract is:

```text
input:
  packed_qk [tokens, (q_heads + kv_heads) * head_dim]
  q_weight [head_dim]
  k_weight [head_dim]

output:
  normalized_qk with the same shape, dtype, and device
```

The helper:

1. views the packed prefix as
   `[tokens, q_heads + kv_heads, head_dim]`;
2. converts activations to float32;
3. computes `pow(2).mean(-1, keepdim=True)` independently for every head;
4. multiplies by `rsqrt(variance + epsilon)`;
5. converts back to the input dtype;
6. applies `q_weight` to Q heads and `k_weight` to K heads; and
7. returns a view matching the packed prefix.

The logical weight expansion must be compiler-fused. Tests and profiling must
reject an implementation that adds a persistent activation-sized copy.

### 3. Attention integration

`QWen3Attention.forward()` retains the packed QKV projection. When the feature
is disabled, it executes the current two-call path unchanged.

When enabled:

1. take a view of the contiguous Q+K prefix;
2. invoke the packed normalization helper once;
3. split the normalized prefix into Q and K views;
4. retain the original V view;
5. call the existing RoPE implementation; and
6. call the existing attention and output projection.

No scheduler, KV-cache, sampling, or graph-replay interface changes.

### 4. Observability

Expose a per-model immutable mode receipt containing:

```text
packed_qk_single_pass_rmsnorm_enabled
q_heads
kv_heads
head_dim
```

The benchmark records the mode in every row and rejects mixed-mode evidence.
No per-token Python counter is added to the hot path.

## Correctness and Test Strategy

### Unit tests

- configuration is default-disabled and strictly boolean;
- enabled construction rejects incompatible Q/K dimensions or epsilon;
- disabled attention follows the existing two-RMSNorm path;
- enabled attention invokes the packed helper once;
- Q heads receive only Q weights and K heads receive only K weights;
- FP32 normalization and BF16 cast boundaries remain explicit;
- randomized BF16 Q/K inputs are bitwise equal to the existing path;
- edge shapes cover one token, multiple tokens, GQA, and equal Q/K head counts;
- gradients are out of scope because TinyLLMForge is an inference runtime.

### Model-level correctness

For fixed prompts and generated lengths, compare disabled versus enabled:

- output token IDs;
- sampled logits at prefill-final, decode-first, decode-middle, and
  decode-final;
- target-model forward count;
- graph replay count;
- D2H calls and bytes; and
- KV-cache block and reserved-memory inventory.

All sampled logits require maximum absolute difference exactly `0.0`.

## Hardware Gate

Use Qwen3-0.6B on one strictly clean A100 with:

- TP1, batch size 1;
- prompt lengths `256`, `2048`, and `8192`;
- generated length `128`;
- disabled/enabled interleaved order;
- 10 paired repetitions per context;
- 60 performance rows;
- 24 correctness rows; and
- source-bound remote and local independent verification.

GO requires:

- complete `60/60 + 24/24` evidence;
- exact output tokens and sampled logits;
- equal forward, replay, and D2H inventories;
- aggregate TPOT median improvement at least `2%`;
- aggregate TPOT P95 not regressing by more than `1%`;
- no context bucket regressing E2E or throughput by more than `3%`;
- aggregate TTFT regression no more than `2%`;
- reserved-memory regression no more than `1%`; and
- no persistent activation-sized Q/K weight expansion.

If the candidate is exact but misses the TPOT threshold, classify it
`NO_GO_PACKED_QK_RMSNORM_PERFORMANCE` and leave the feature disabled. If any
exactness or inventory check fails, classify it
`NO_GO_PACKED_QK_RMSNORM_CORRECTNESS`.

## Benefit and Cost Reporting

The final report must include:

- paired TPOT median and P95 by context and aggregate;
- TTFT, E2E, and throughput deltas;
- CUDA peak allocated and reserved memory;
- compiled-kernel count or an equivalent launch-inventory probe;
- compilation/capture cost if it changes materially;
- exactness and execution-inventory results; and
- whether the feature remains disabled or is eligible to become the default.

The isolated `66.05%` subpath result is reported only as mechanism evidence.
No end-to-end claim is allowed until the canonical hardware gate completes.
