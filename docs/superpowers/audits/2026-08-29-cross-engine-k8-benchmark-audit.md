# Exact Greedy K8 cross-engine benchmark audit

Date: 2026-08-30

## Terminal conclusion

Attempt `20260829-cross-engine-k8-qwen3-06b-r7` completed the frozen
Qwen3-0.6B canonical matrix and both independent verification paths. Its
terminal classification is:

```text
INCOMPLETE
```

The only formal gate reason is:

```text
metric_unavailable:peak_gpu_memory_ratio
```

Public vLLM 0.11.2 did not expose a per-process peak-GPU-memory value through
the frozen measurement interface. The value is retained as `NOT_EXPOSED`;
it was not replaced with zero or another estimate. Producer, repaired remote
verifier, and local streaming verifier all recomputed `INCOMPLETE`.

This is not `GO_CROSS_ENGINE_ADVANTAGE`. It must not be presented as proof
that TinyLLMForge is horizontally faster than vLLM.

## Prompt-to-artifact completion checklist

- [x] Used the authoritative checkout:
      `/Users/bytedance/Desktop/TinyLLMForge`, which resolves to
      `/Users/bytedance/dev/TinyLLMForge`.
- [x] Continued the existing immutable tag
      `20260829-cross-engine-k8-qwen3-06b-r7`; no replacement tag was made.
- [x] Kept all remote task state under
      `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/cross-engine-k8-qwen3-06b`.
- [x] Reused the existing model at
      `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`; no model copy or
      redownload was used by the campaign.
- [x] Used one admitted NVIDIA A100 80GB PCIe with two clean samples, zero
      foreign compute processes, zero utilization at admission, and no
      unauthorized cleanup.
- [x] Preserved exact-output equality; all 63 correctness rows have
      `matches_reference=true`.
- [x] Completed 7 repetitions x 3 arms x 3 contexts: 21 fresh workers,
      63 performance rows, and 63 unique correctness keys.
- [x] Measured both the within-TinyLLMForge mechanism result and the absolute
      TinyLLMForge-versus-vLLM result.
- [x] Reported benefit and cost together for latency, throughput, GPU memory,
      and RSS where exposed.
- [x] Ran an independent remote verifier and an independent local streaming
      verifier; both are `valid=true`.
- [x] Retained exactly the 11-file local evidence allowlist, totaling
      582,150 bytes, below the 50 MiB cap.
- [x] Preserved unavailable metrics as `NOT_EXPOSED` and failed the strict
      gate closed as `INCOMPLETE`.
- [x] Recorded interrupted and rejected paths without rewriting their
      evidence into successful runs.

## Source and tooling identity

The measured runtime was frozen at:

```text
runtime source:
  c6e73cab00b61101ec8a0bfd3ac8a3c3cd1c8bbb
branch:
  feat/kv-sparse-attention
run tag:
  20260829-cross-engine-k8-qwen3-06b-r7
```

Post-run evidence handling required three later tooling-only commits:

```text
42ae5d2dbf0217c1386aab20fe0b68f570087aa8
  fail closed when a protected public-engine metric is NOT_EXPOSED

fd5d20a7662e052827e81053ecd404564968f320
  stream large remote JSONL payloads through SSH stdin
  remote repaired verifier source actually used

ec8e2bb4e464283ff542ba5c98822a702d16b872
  support direct local streaming-verifier entrypoint
```

These commits did not change the measured case rows. They repaired
finalization, transport, and independent-verification behavior around the
already completed canonical workers.

## Frozen environment and workload

```text
model:
  Qwen3-0.6B
model inventory SHA-256:
  412b44d3a047326a8fb0e2f50289f252bdc81609fa7ddbf990954284b178923d
workload manifest SHA-256:
  9300de7c1093a992a2782375c6cf1e420982def53acbfc0d13a00359126a8fdb
precision:
  BF16
tensor parallel:
  1
batch size:
  1
temperature:
  0
ignore EOS:
  true
prompt lengths:
  256, 2048, 8192
output tokens:
  128
warmups:
  2 per worker
measured repetitions:
  7
GPU UUID used by all retained rows:
  GPU-57be086f-e967-c022-3832-93df4fc77bd0
```

TinyLLMForge used Python 3.11.15, PyTorch 2.4.1+cu121, CUDA 12.1,
Triton 3.0.0, and FlashAttention 2.6.3. The isolated vLLM environment used
vLLM 0.11.2, Python 3.11.15, PyTorch 2.9.0+cu128, CUDA 12.8, and Triton
3.5.0.

The eligible arms were:

```text
tinyllmforge_host_greedy
tinyllmforge_exact_k8
vllm_default_greedy
```

The selected compatible public vLLM version did not expose a public
multi-step greedy arm, so no within-vLLM multi-step comparison was eligible.
The strongest eligible vLLM arm was `vllm_default_greedy`.

## Matrix and correctness

The canonical campaign produced:

```text
fresh workers:                 21
performance rows:             63
correctness rows:             63
unique repetition/context/arm keys:
                              63
exact reference matches:      63
correctness failures:         0
terminal receipts valid:      true
storage valid:                true
```

Every arm emitted exactly 128 tokens for every context and repetition. The
cross-engine token hashes match the frozen host-greedy references.

## Within-TinyLLMForge result

The Exact K8 aggregate is compared with TinyLLMForge host greedy below.
Lower is better for latency and memory; higher is better for throughput.

| Metric | Host greedy | Exact K8 | Benefit or cost |
|---|---:|---:|---:|
| Median TPOT | 3,630,347 ns | 2,667,398 ns | 26.52% lower |
| Throughput | 250.7661 tok/s | 337.1744 tok/s | 34.46% higher |
| E2E | 510,435,845 ns | 379,625,552 ns | 25.63% lower |
| TTFT | 39,395,344 ns | 39,254,364 ns | 0.36% lower |
| P95 TPOT | 4,000,960 ns | 2,721,040 ns | 31.99% lower |
| P99 TPOT | 4,138,765 ns | 2,721,040 ns | 34.26% lower |
| Peak GPU memory | 66,221,768,704 B | 66,253,225,984 B | 0.05% higher |
| Peak RSS | 1,427,021,824 B | 1,431,662,592 B | 0.33% higher |

This is a valid positive within-engine mechanism result: Exact K8 removed a
substantial fraction of TinyLLMForge host-side greedy overhead while paying
small measured GPU-memory and RSS costs.

## Absolute TinyLLMForge K8 versus public vLLM

Ratios below are `TinyLLMForge Exact K8 / vLLM default greedy`.

| Metric | TinyLLMForge K8 | vLLM default | Ratio | Interpretation |
|---|---:|---:|---:|---|
| Median TPOT | 2,667,398 ns | 2,755,463 ns | 0.9680 | Tiny 3.20% lower |
| Throughput | 337.1744 tok/s | 346.4533 tok/s | 0.9732 | Tiny 2.68% lower |
| TTFT | 39,254,364 ns | 19,065,626 ns | 2.0589 | Tiny 105.89% higher |
| E2E | 379,625,552 ns | 369,458,215 ns | 1.0275 | Tiny 2.75% higher |
| P95 TPOT | 2,721,040 ns | 2,783,949 ns | 0.9774 | Tiny 2.26% lower |
| P99 TPOT | 2,721,040 ns | 2,862,085 ns | 0.9507 | Tiny 4.93% lower |
| Peak GPU memory | 66,253,225,984 B | `NOT_EXPOSED` | `NOT_EXPOSED` | Strict gate incomplete |
| Peak RSS | 1,431,662,592 B | 1,052,233,728 B | 1.3606 | Tiny 36.06% higher |

Context-specific median-TPOT ratios were:

```text
short:   0.9161311471
medium:  0.9680398539
long:    0.9363524829
```

TinyLLMForge K8 therefore had lower median TPOT in all three context buckets
and lower P95/P99 TPOT in the aggregate. It did not satisfy the frozen
horizontal GO rule: aggregate throughput was lower, TTFT/E2E/RSS exceeded
the protected regression limits, and the required peak-GPU-memory ratio was
unavailable.

## Storage and retained evidence

```text
remote allocated bytes:
  17,059,774,464
remote hard stop:
  21,474,836,480
local retained bytes:
  582,150
local retention cap:
  52,428,800
```

The local allowlist is:

```text
controller_manifest.json
environment_manifest.json
workload_manifest.json
case_rows.jsonl
correctness_rows.jsonl
comparison.json
summary.json
gate.json
remote_verification.json
local_verification.json
manifest.sha256
```

The manifest covers the first eight producer artifacts. The remote and local
verifier files are byte-identical, each with SHA-256:

```text
f8538a51ccfc9f5fae20066a5df08fb74057c3c5a24e68b9385672a2aa07f9a6
```

## Recovery and evidence boundaries

The following failures were recovered without changing the frozen measured
workload or silently promoting rejected evidence:

1. The first canonical host worker passed admission, then an external
   `allm-native-server` appeared. That worker was excluded and archived under
   `canonical/failed-workers/`; no external PID was terminated or adopted.
2. The first aggregate transfer failed with
   `mux_client_request_session: write packet: Broken pipe`. Its empty partial
   aggregate was archived under `recovery/`.
3. Worker correctness rows lacked the repetition field required by the
   verifier's `(repetition, context, arm)` uniqueness key. Repetition was
   reconstructed only from verified worker receipts and immutable worker
   paths, then the aggregate was rebuilt atomically. Performance rows and
   token outputs were not modified.
4. Initial finalization attempted to convert vLLM's
   `peak_gpu_memory_bytes="NOT_EXPOSED"` to a float. Commit `42ae5d2` made the
   producer and verifier preserve the sentinel and classify the gate as
   `INCOMPLETE`.
5. A second SSH `Broken pipe` occurred when large JSONL payloads were passed
   through command-line arguments. Commit `fd5d20a` changed this transport to
   SSH stdin streaming; the partial `remote-final` was archived.
6. The verifier frozen with the runtime source predated `NOT_EXPOSED`
   handling. Its invalid result is preserved as
   `recovery/remote-verification-runtime-source-invalid.json`. The repaired
   remote verifier used source `fd5d20a`; its tooling identity is separately
   recorded and it agrees with the local independent recomputation.
7. Direct local invocation initially failed with
   `ModuleNotFoundError: No module named 'tools'`. Commit `ec8e2bb` repaired
   only the script entrypoint; the retained local verification result is
   valid.

These are evidence-pipeline repairs, not permission to rerun selected cases,
retune thresholds, replace unavailable metrics, or claim a horizontal win.

## Verifier reconciliation

```text
producer classification:
  INCOMPLETE
producer gate reasons:
  ["metric_unavailable:peak_gpu_memory_ratio"]
remote verifier:
  valid=true
  producer_agrees=true
  recomputed_classification=INCOMPLETE
local verifier:
  valid=true
  producer_agrees=true
  recomputed_classification=INCOMPLETE
comparison.verifiers_agree:
  true
```

## Fresh terminal verification

The prescribed `/opt/homebrew/bin/python3.12` executable was present but did
not have pytest installed, so it stopped before collection with
`No module named pytest`. The same complete ten-file suite was then run with
the available system `python3` and pytest 8.4.2:

```text
149 passed in 2.77s
```

Additional fresh terminal checks:

```text
all 11 retained files present:                    PASS
all retained JSON documents parse:               PASS
manifest SHA-256 verification (8 producer files): PASS
case rows:                                        63
correctness rows:                                 63
matches_reference=true:                           63
remote/local verifier byte identity:              PASS
local retained bytes:                             582,150
focused compileall:                               PASS
git diff --check:                                 PASS
```

## Final classification and permitted claim

```text
CROSS_ENGINE_K8_CANONICAL=COMPLETE
CORRECTNESS=63_OF_63_EXACT
REMOTE_VERIFIER=PASS
LOCAL_VERIFIER=PASS
PRODUCER_CLASSIFICATION=INCOMPLETE
FORMAL_GATE_REASON=metric_unavailable:peak_gpu_memory_ratio
GO_CROSS_ENGINE_ADVANTAGE=false
```

Permitted statement:

> On the frozen Qwen3-0.6B BF16 TP1 batch-1 workload, TinyLLMForge Exact K8
> improved its own host-greedy median TPOT by 26.52% and throughput by 34.46%.
> Against pinned vLLM 0.11.2 default greedy, TinyLLMForge K8 showed 3.20%
> lower median TPOT but 2.68% lower throughput, 105.89% higher TTFT, 2.75%
> higher E2E, and 36.06% higher RSS. The strict horizontal gate is
> `INCOMPLETE` because vLLM peak GPU memory was not exposed.

This result does not establish global engine superiority, online-serving or
batching superiority, larger-model behavior, tensor-parallel behavior,
production-default readiness, or academic novelty.
