# Exact Prefill CUDA Graph Completion Audit

Date: 2026-08-30

## Terminal conclusion

The opt-in exact-shape prefill CUDA Graph path completed its frozen
Qwen3-0.6B BF16 TP1 batch-one paired gate with:

```text
GO_EXACT_PREFILL_GRAPH
```

The candidate emitted exactly the same generated token IDs as eager prefill
for every retained sample. It replayed the captured prefill graph for every
measured candidate request, with zero capture failures, replay failures, or
quarantines.

This is a positive within-TinyLLMForge mechanism result for the exact tested
shapes. It is not a cross-engine result, a multi-request throughput result,
or evidence for arbitrary prompt lengths, TP greater than one, prefix-cache
hits, offload, KV quantization, compact attention, or stateful model-forward
paths.

## Prompt-to-artifact completion checklist

- [x] Used the authoritative checkout
      `/Users/bytedance/Desktop/TinyLLMForge`, which resolves to
      `/Users/bytedance/dev/TinyLLMForge`.
- [x] Kept the feature opt-in and disabled by default.
- [x] Preserved exact token equality against eager execution.
- [x] Restricted replay to TP1, runtime world size one, batch one, dense
      prefill, exact allowlisted shapes, and the plain model `forward` path.
- [x] Failed closed to eager execution for unsupported identities and
      pre-replay failures.
- [x] Quarantined and propagated post-replay failures without an unsafe eager
      retry after possible live KV mutation.
- [x] Copied each live request's input IDs, positions, slot mapping, and
      sequence metadata into graph-owned static tensors before replay.
- [x] Kept decode execution, the LM head, and sampling outside the new graph.
- [x] Recorded capture duration, retained static bytes, allocated-memory
      delta, reserved-memory delta, replay count, and quarantine state.
- [x] Used one clean NVIDIA A100 80GB PCIe selected as physical GPU 1; no
      foreign process was terminated or adopted.
- [x] Reused the existing remote Qwen3-0.6B model and Python environment.
- [x] Kept all remote task state under
      `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- [x] Ran the frozen two-arm, two-shape, two-round AB/BA gate with two warmups
      and five measured repetitions per case.
- [x] Retained all eight raw result files and their zero exit receipts.
- [x] Ran the producer, remote independent verifier, and local independent
      verifier successfully.
- [x] Bound the result to the source base commit, seven source-file hashes,
      benchmark-contract hash, model path, Python path, GPU identity, and case
      order.
- [x] Retained only the compact 136 KiB evidence bundle locally; full runtime
      logs remain remote.
- [x] Reported benefit and startup/memory cost together.
- [x] Explicitly excluded sentinel-filled graph buckets from this version.

## Source and run identity

```text
branch:
  feat/kv-sparse-attention
source base commit:
  62c417782d08c88198c8e912b204ae8cd861ce50
measurement run:
  exact-prefill-graph-paired-20260830-r3
finalized postprocessing bundle:
  exact-prefill-graph-paired-20260830-r5
contract SHA-256:
  49f065418a2f17018857970232c77b7055145c4e0e891f60b966fc740f0a82a8
```

The r5 bundle reuses the r3 raw measurements unchanged. It overlays the
current runtime, gate, verifier, and source inventory, then reruns producer
and independent verification after correcting the `summary.json` schema
field and hardening failure handling. The successful replay path, worker,
benchmark contract, and raw case measurements did not change after r3. The
runtime hardening affects only replay failures: after replay starts, a
failure now quarantines and propagates instead of attempting an unsafe eager
retry. A second GPU measurement campaign was therefore not required for
these postprocessing and failure-path-only corrections.

The seven source hashes recorded in `run_manifest.json` match the current
working tree:

```text
tinyvllm/config.py
  27f74c27d1453076da28e161679bcd8ebeeb7a74cb4a0c328acabcf3e3dcd21f
tinyvllm/engine/exact_prefill_cuda_graph.py
  f3e6b5907ad126371f4641e73d9fee7bebbe1c96a2256fb93e5ac2646c66675c
tinyvllm/engine/model_runner.py
  1ef07e5c2ac4766dd48e062415262a6a0784e889dda04e38c6740c25559de33f
tools/exact_prefill_cuda_graph_benchmark_contract.py
  41e5e1a5e22e7eb6d6443e17a6f211c7d92ddc50611690f3a32456d038b526f7
tools/exact_prefill_cuda_graph_benchmark_worker.py
  f3de2f2d17bc28e7e08dfadb2acdfd76d86086c0ebf22eb8ae4eab223801fc22
tools/exact_prefill_cuda_graph_gate.py
  ece5d54d40fea4e3af96270687ab739e166f12799ea521947c3242e47a72bc1e
tools/exact_prefill_cuda_graph_verify.py
  74eb7891d9bc61c2d61700337ea94cbf0d68ca83e9c91516fb9c7fda47595534
```

## Environment and frozen workload

```text
model:
  /data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B
Python:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    cross-engine-k8-qwen3-06b/envs/tinyllmforge-c6e73cab00b6/bin/python
GPU:
  NVIDIA A100 80GB PCIe
physical GPU index:
  1
GPU UUID:
  GPU-7dc22583-df04-6c76-4ba5-ea32c428c130
clean admission:
  true
tensor parallel:
  1
batch size:
  1
prompt tokens:
  256, 2048
generated tokens:
  16
temperature:
  0.0
ignore EOS:
  true
rounds:
  2, AB/BA
warmups:
  2 per case
measured repetitions:
  5 per case
samples:
  10 per arm and prompt shape
```

## Benefit and cost

Lower is better for all latency metrics.

| Prompt | Metric | Eager median | Graph median | Benefit |
| ---: | --- | ---: | ---: | ---: |
| 256 | TTFT | 32.4475 ms | 5.2900 ms | 83.70% lower |
| 256 | TPOT | 3.3879 ms | 3.3413 ms | 1.38% lower |
| 256 | E2E | 84.3913 ms | 56.2728 ms | 33.32% lower |
| 2048 | TTFT | 34.9052 ms | 21.7850 ms | 37.59% lower |
| 2048 | TPOT | 3.5442 ms | 3.5205 ms | 0.67% lower |
| 2048 | E2E | 89.0353 ms | 75.4464 ms | 15.26% lower |

The startup and retained-memory costs were:

| Cost | Result |
| --- | ---: |
| Median capture duration | 727.917 ms |
| Maximum capture duration | 736.033 ms |
| Retained static tensors | 4,764,704 bytes |
| Allocated-memory delta | 0 bytes |
| Reserved-memory delta | 41,943,040 bytes (40 MiB) |

The approximately 0.73-second startup capture is amortized only when the same
allowlisted shape is reused. The result does not claim a benefit for
one-shot processes that terminate before amortizing capture.

## Gate evaluation

| Requirement | Threshold | Observed | Verdict |
| --- | ---: | ---: | --- |
| 256-token median TTFT improvement | at least 25% | 83.70% | PASS |
| 2048-token median TTFT regression | at most 2% | -37.59% | PASS |
| Median TPOT regression | at most 2% | -1.38%, -0.67% | PASS |
| Median E2E regression | at most 2% | -33.32%, -15.26% | PASS |
| Exact generated-token equality | required | all exact | PASS |
| Candidate replay per measured sample | required | all replayed | PASS |
| Capture/replay failure or quarantine | none allowed | all zero | PASS |
| Capture and memory cost present | required | present | PASS |

Negative regression values indicate improvements.

## Independent verification and retained artifacts

The producer manifest covers the eight raw case results plus
`run_manifest.json`, `comparison.json`, `summary.json`, and `report.md`.
Remote and local independent-verifier receipts are byte-identical, with
SHA-256:

```text
80a24ce3658d97f17d9731affdf1cce3d500dba0192b4a88730a9b7c4c5687de
```

Both receipts contain:

```json
{
  "classification": "GO_EXACT_PREFILL_GRAPH",
  "manifest_verified": true,
  "raw_metrics_reconstructed": true,
  "verified": true
}
```

All eight worker exit receipts, the producer receipt, the verifier receipt,
and the terminal receipt are zero. The compact local evidence is:

```text
artifacts/exact_prefill_cuda_graph/20260830-qwen3-06b-paired-r5/
```

The measurement and finalized remote roots are:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  exact-prefill-graph-paired-20260830-r3/

/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  exact-prefill-graph-paired-20260830-r5/
```

Full remote stderr contains only the existing Transformers `torch_dtype`
deprecation warning and Torch symbolic-shape warnings. These warnings were
not copied into the compact local bundle and did not produce nonzero exits.

## Implementation boundary

The implementation adds:

- normalized `prefill_cuda_graphs` and exact token allowlist configuration;
- a model-agnostic graph identity and cache;
- startup capture after KV allocation;
- live-input copy followed by exact-shape replay;
- per-identity quarantine and eager fallback;
- typed post-replay failure propagation with no second transformer forward;
- runtime capture/replay/cost counters;
- a frozen benchmark contract and isolated worker;
- a producer gate and separately implemented independent verifier.

The feature remains off by default. Version one does not implement:

- graph buckets for non-allowlisted lengths;
- sentinel-filled or padded synthetic sequences;
- TP2/TP4 capture;
- multi-sequence prefill capture;
- prefix-cache-aware graph replay;
- graph replay with CPU/KV offload or KV quantization;
- graph replay with Quest, compact attention, or chunked prefill;
- stateful `run_step` models;
- LM-head or sampling capture;
- a cross-engine performance claim.

## Environment limitation

The adjacent Mac regression command imports GPU-dependent modules and reports
33 `ModuleNotFoundError: No module named 'torch'` collection failures because
the local Mac environment has no Torch installation. The remaining 118 tests
in that command passed, and no assertion regression was observed. This is an
environment limitation, not a claim that the full GPU-dependent suite passed.

## Final classification

```text
EXACT_PREFILL_CUDA_GRAPH_IMPLEMENTATION=COMPLETE
PRODUCER_CLASSIFICATION=GO_EXACT_PREFILL_GRAPH
REMOTE_INDEPENDENT_VERIFIER=PASS
LOCAL_INDEPENDENT_VERIFIER=PASS
EXACT_OUTPUT_PARITY=PASS
SUPPORTED_SCOPE=QWEN3_0_6B_BF16_TP1_BATCH1_EXACT_256_OR_2048
FEATURE_DEFAULT=OFF
CROSS_ENGINE_ADVANTAGE_CLAIM=false
SENTINEL_BUCKET_CLAIM=false
NEXT_ACTION=commit and push the source-bound implementation and compact evidence
```
