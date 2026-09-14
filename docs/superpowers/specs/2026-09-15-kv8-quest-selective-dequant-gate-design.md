# KV8 + Quest Selective-Dequant Gate Design

Date: 2026-09-15

## Objective

Determine whether Quest block selection removes enough of TinyLLMForge's
existing KV8 full-dequant overhead to justify further kernel work, while
reporting the associated long-context retrieval-quality cost.

This is a diagnostic gate, not a production qualification. It does not change
the attention implementation or enable KV8/Quest by default.

## Why a separate gate is required

The graph-path KV baseline established two facts:

1. Qwen3-8B bf16 decode on the multi-sequence CUDA graph path has a roughly
   12 ms constant and a per-sequence slope dominated by KV bytes.
2. The current KV8 path is about 8.8 times worse in per-sequence slope because
   it materializes the complete selected KV set as bf16 before FlashAttention.

Quest can reduce the number of blocks materialized before attention. However,
the current runtime deliberately sends active Quest decode through eager
execution, so comparing a graph-path KV8 arm directly with KV8+Quest would
confound selective dequantization with dispatch-path overhead.

## Experimental design

### Performance diagnostic

Run three arms sequentially on the same A100:

- bf16 full attention, eager;
- KV8 full attention, eager;
- KV8 + Quest top-16 blocks, eager.

All arms use:

- Qwen3-8B;
- TP1;
- a pinned 640-block KV pool;
- context length 8192;
- batches 4, 8, 12, 16, and 19;
- identical seed, warmup count, measured-step count, and prompt construction;
- a fresh immutable tag per arm.

Context 8192 contains 32 full 256-token blocks before decode. Top-16 therefore
tests an approximately 50% block budget. Context 2048 is excluded because the
runtime's short-sequence protection would disable top-16 Quest there, making
the label misleading.

Every artifact must record both requested configuration and resolved engine
identity:

- `kv_quant_bits`;
- `quest_top_k_blocks`;
- `quest_min_seq_len`;
- `enforce_eager`;
- `num_kvcache_blocks`;
- model path, seed, grid, and dispatch observations.

### Quality diagnostic

Use `tools/eval_needle.py` with fixed prompts and greedy decoding:

- context length 8192;
- depths 0.0, 0.25, 0.5, 0.75, and 1.0;
- five trials per depth;
- separate bf16 baseline run;
- one KV8 engine run containing full-attention (`top_k=-1`) and Quest
  (`top_k=16`) settings;
- prefix-cache metadata cleared between fixed-prompt settings.

The primary quality comparison is KV8+Quest versus KV8 full attention on the
same prompts. The bf16 arm records the total cost of KV quantization plus
sparsity.

## Gate

The diagnostic is `GO_TO_FUSED_KERNEL_SCOPE` only if all of the following hold:

1. all performance cells are measured at the requested batch;
2. every performance arm is confirmed eager, so no arm receives a graph-path
   advantage;
3. model, pool, grid, seed, warmup, and measured-step identities match;
4. KV8+Quest reduces median step time relative to KV8 full attention in every
   measured cell;
5. at the B=19 wall cell, KV8+Quest removes at least 50% of KV8's excess
   latency over bf16 eager;
6. KV8+Quest needle accuracy is no more than five percentage points below KV8
   full attention overall;
7. no individual depth loses more than twenty percentage points.

Any identity mismatch or incomplete arm yields `INCONCLUSIVE`. A speed or
quality threshold failure yields `NO_GO_KV8_QUEST`.

Passing does not authorize production use. It only justifies scoping a fused
dequant-inside-attention or graph-compatible selective-dequant kernel.

## Harness changes

Extend the existing step-scaling worker and remote runner with:

- worker CLI `--quest-top-k-blocks`;
- worker CLI `--quest-min-seq-len`;
- runner environment variables `QUEST_TOP_K_BLOCKS` and
  `QUEST_MIN_SEQ_LEN`;
- passthrough to `tinyvllm.LLM`;
- requested configuration fields in the payload;
- resolved Quest fields in engine identity.

The default values keep Quest disabled and preserve all existing runs.

## Evidence and storage

Remote runs must write below:

`/data00/home/sitian/tllm/kvcapacity-runs/`

Repository evidence is limited to compact JSON/text summaries, the design,
the implementation plan, and the final audit/handoff. Large traces or model
files are not committed.

## Claim boundary

The result is limited to Qwen3-8B, A100 80GB PCIe, TP1, eager decode,
8192-token synthetic contexts, the pinned 640-block pool, and the fixed needle
workload. It is not evidence for CUDA-graph compatibility, production QPS,
real-request quality, TP2/TP4, other models, or a fused kernel that has not
been built.
