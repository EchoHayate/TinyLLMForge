# Qwen3.5 Long-Prefill Blockwise Online-Softmax

## Status

Approved as the root-cause fix required by the independent TP4/32K authority.

The first real 32K campaign failed in `baseline:b1` before result validation:
`qwen35_prefill_eager_attention` attempted to allocate an 8 GiB dense
attention probability tensor, then other ranks timed out in NCCL. Raising the
GPU free-memory threshold would only wait for a larger O(n²) allocation and
would not advance the long-context objective.

## Selected Design

Keep the existing dense path unchanged through 16,384 reference tokens. For a
longer pure-prefill row, compute exact causal attention with bounded query and
key windows and merge key-window contributions with FP32 online softmax.

The implementation:

- preserves current KV cache writes;
- operates only on real query/key/value tokens and does not materialize padded
  future tokens used only for dense shape normalization;
- repeats KV heads exactly as the existing GQA path;
- tiles queries and keys with a default 512-token tile;
- keeps score memory bounded by
  `heads * query_tile * key_tile`, independent of total context squared;
- accumulates running maximum, exponential sum, and weighted value sum in
  FP32;
- applies exact absolute-position causal masking per tile;
- returns the original dtype and shape; and
- leaves cached-prefill, decode, spec-verify, quantized-KV, and <=16K behavior
  unchanged.

Tests may force the blockwise path on small tensors through optional context
attributes:

```text
qwen35_prefill_blockwise_threshold
qwen35_prefill_block_tokens
```

Production defaults are:

```text
threshold = 16384
block tokens = 512
```

## Rejected Alternatives

1. **Wait for four GPUs with more free memory.** Rejected because it preserves
   the O(n²) temporary and cannot guarantee 32K batch execution.
2. **Use PyTorch SDPA/FlashAttention only for 32K.** Rejected for this gate
   because backend selection and numerical behavior are less explicit and the
   current Qwen3.5 path is intentionally dependency-light and auditable.
3. **Reuse the offload-manager blockwise cached-prefill helper directly.**
   Rejected because the failing path is pure prefill and lacks historical
   prefix windows; importing manager-specific planning would add unnecessary
   state and coupling.

## Correctness Requirements

- small forced-blockwise output matches the dense causal oracle within a
  strict FP32 tolerance;
- multi-request boundaries remain isolated;
- GQA head replication matches the dense path;
- score matmul shapes never exceed the configured query/key tile sizes;
- reference-length padding does not allocate or write fake KV tokens;
- output contains no NaN or Inf;
- existing dense bit-exact tests remain unchanged and pass;
- successful real TP4/32K baseline/candidate parity remains exact; and
- the change makes no performance claim.

## Failure Boundary

The fix is not established by a synthetic tensor benchmark. It is established
only when focused numerical tests pass and the source-bound real TP4/32K
campaign completes with exact parity, transactional evidence, positive real
KV movement, and independent verifier PASS.
