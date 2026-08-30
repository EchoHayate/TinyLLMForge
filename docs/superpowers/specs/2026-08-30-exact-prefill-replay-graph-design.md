# Exact Prefill Replay Graph Design

Date: 2026-08-30

## Objective

Reduce first-token latency for repeated, exact-shape dense prefill workloads
without changing generated token IDs, decode behavior, or unsupported request
semantics.

The initial authority is Qwen3-0.6B, BF16, TP1, batch one, greedy sampling,
with prompt lengths 256 and 2048 on A100 80GB PCIe.

## Measured problem

A clean-cache first-token diagnostic attributed median wall time as follows:

| Prompt tokens | External TTFT | Model dispatch | Scheduler |
| ---: | ---: | ---: | ---: |
| 256 | 33.38 ms | 31.91 ms | 0.085 ms |
| 2048 | 36.68 ms | 34.82 ms | 0.180 ms |
| 8192 | 107.61 ms | 105.24 ms | 0.449 ms |

For the 256-token case, `torch.profiler` observed approximately 31.1 ms of
self CPU time but only 4.0 ms of summed self CUDA kernel time. The prefill
issued 113 matrix multiplications, 253 Triton kernels, and 28 FlashAttention
calls. The current short-prompt path is therefore host-launch-bound rather
than scheduler-bound or attention-compute-bound.

The diagnostic artifacts remain remote under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  first-token-diagnostic-20260830-r2/
```

## Selected design

Introduce an opt-in, model-agnostic `ExactPrefillCudaGraphCache`.

At engine initialization, after KV-cache allocation and ordinary decode graph
capture, the runner captures one complete model-forward graph for each
explicitly allowlisted prefill token count. The graph includes embedding,
all transformer layers, KV-cache writes, and final hidden-state production.
The existing LM head and greedy sampler remain outside the graph in the first
version so their established correctness and accounting paths are unchanged.

For each allowlisted token count, capture owns:

- static input IDs;
- static positions;
- static slot mapping;
- static `cu_seqlens_q` and `cu_seqlens_k`;
- static hidden-state output;
- one CUDA graph and its private memory pool.

Replay copies live input IDs, positions, and slot mapping into the static
buffers, installs the captured exact prefill context, replays the graph, and
returns the static hidden-state output. The normal LM-head and sampler then
produce the first token.

## Eligibility and fallback

Version one is deliberately narrow. Replay is eligible only when all of the
following are true:

- the feature is explicitly enabled;
- execution is prefill;
- tensor parallel size and runtime world size are both one;
- there is exactly one sequence;
- query and key lengths are equal to the input token count;
- there is no prefix-cache block table;
- the exact token count is in the configured allowlist;
- input embeddings and hidden-state return are not requested;
- CPU offload, KV offload, KV quantization, compact attention, and other
  alternate model execution paths are inactive;
- the model uses the plain `forward` interface rather than a stateful
  `run_step` interface.

Every ineligible identity and every failure detected before replay starts uses
the existing eager prefill path. Capture failure for one token count
quarantines only that token count. A failure after `graph.replay()` starts
quarantines the identity and propagates a typed error; it must not retry eager
prefill after the graph may have mutated live KV slots.

## Correctness contract

- Eager and graph arms must emit identical token IDs for every retained case.
- Replay must use the live request's input IDs, positions, and KV slot mapping.
- Graph capture and replay must never alias a live request's host metadata.
- Unsupported modes must fail closed to eager execution.
- A post-replay failure must not issue a second transformer forward.
- Decode graph behavior and Exact Greedy K8 behavior must remain unchanged.
- The optimization is not enabled by default.

## Cost accounting

The runtime exposes, per token count:

- capture duration;
- retained static tensor bytes;
- allocated-memory delta;
- reserved-memory delta;
- replay count;
- rejection reason, if any.

The benchmark reports both steady-state TTFT benefit and startup/capture plus
memory cost. A result is not a GO if it omits either side.

## Initial gate

The candidate must satisfy:

- exact token equality against eager prefill;
- at least 25% median TTFT reduction at 256 tokens;
- no more than 2% median TTFT regression at 2048 tokens;
- no more than 2% median TPOT or end-to-end regression;
- no capture or replay failure;
- measured capture duration and GPU-memory cost are present;
- all focused unit and integration tests pass.

The 8192-token path remains eager in version one because the previous
cross-engine evidence already showed a competitive long-prefill slope and
because a large prefill graph has a materially larger memory pool.

## Deferred extension

`Sentinel-Filled Prefill Graph Buckets` may later map a range of prompt
lengths to one fixed-shape graph by adding an isolated synthetic sequence.
It is not part of this implementation and must not be implied by version-one
results.
