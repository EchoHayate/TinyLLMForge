# Quantized Draft INT4 Stage-0 Audit

## Executive Verdict

The real single-A100 Stage-0 gate completed on August 31, 2026 with the
terminal classification:

```text
NO_GO_PERFORMANCE
```

The repository-owned fused W4A16 kernel is correct, consumes packed INT4
weights without materializing a full dequantized weight, reduces persistent
drafter weight storage to 26.56 percent of BF16, supports CUDA Graph capture
and replay, and cleans up its owned process and GPU state. It is nevertheless
far slower than the BF16 reference on every measured decode shape.

The frozen stop rule therefore applies:

```text
stop_before_distillation = true
```

No quantization-aware distillation training, teacher-data generation, runtime
dispatch integration, or end-to-end speedup claim is authorized by this
result.

## Frozen Scope

- Terminal run:
  `20260831-quantized-draft-int4-stage0-r5`
- Source revision:
  `c1fadc1e8dab70bd930861bc293cfd20a64faa0c`
- Branch:
  `feat/kv-sparse-attention`
- Remote root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`
- Shape source:
  `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`
- Checkpoint composite SHA-256:
  `aff3539a458ed237dcb00853de3d0e289200f8120375a8b4b440fce9bdc24ed5`
- Device:
  NVIDIA A100 80GB PCIe, CUDA capability 8.0
- Software:
  Python 3.11.15, PyTorch 2.4.1+cu121, CUDA 12.1
- Quantization:
  packed asymmetric-free INT4 values with FP32 per-group scales,
  group size 128
- Warmup pairs per shape:
  2
- Measured pairs per shape:
  200
- Total measured rows:
  800
- Arm order:
  400 `bf16/dequant/fused_int4` and
  400 `fused_int4/dequant/bf16`

The model is used only to extract real draft-linear shapes and execution
counts. This is a kernel qualification result, not a Qwen model-quality or
end-to-end serving result.

## Admission and Execution Identity

The controller selected GPU 1 immediately before launch:

- UUID: `GPU-7dc22583-df04-6c76-4ba5-ea32c428c130`
- admission memory: 3 MiB
- admission utilization: 0 percent
- compute processes: 0

The Kerberos preflight used principal `sitian@BYTEDANCE.COM`, required at
least 5,400 seconds of remaining lifetime, and recorded 28,637 seconds.

The uploaded source was generated from committed Git objects. The source
archive included the complete tracked `tinyvllm/` runtime tree plus the four
gate producer/verifier tools. Untracked and ignored local artifacts were not
uploaded.

## Benefit and Cost

CUDA timing uses 200 paired measurements per shape. P99 is recomputed using
the frozen nearest-rank rule.

| Real decode shape | BF16 median | Fused INT4 median | Median ratio | BF16 P99 | Fused INT4 P99 | P99 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 K1024 N4096 | 30.160 us | 145.408 us | 4.821x | 39.072 us | 167.488 us | 4.287x |
| M1 K1024 N6144 | 26.384 us | 126.928 us | 4.811x | 43.360 us | 147.616 us | 3.404x |
| M1 K2048 N1024 | 18.432 us | 242.720 us | 13.168x | 38.624 us | 250.880 us | 6.495x |
| M1 K3072 N1024 | 19.968 us | 304.032 us | 15.226x | 40.736 us | 321.536 us | 7.893x |

Execution-count-weighted ratios:

- fused INT4 / BF16 median: `9.506594724062909`
- fused INT4 / BF16 P99: `5.519921832212185`

The gate required every shape to achieve:

- median ratio at most `0.75`; and
- P99 ratio at most `0.95`.

All four shapes fail both performance requirements. Although fused INT4 is
faster than the existing dequantize-then-GEMM arm on three shapes, it is not
competitive with the BF16 reference and therefore cannot improve the
speculative runtime.

The likely mechanism is the decode geometry rather than weight storage:
these are M=1 GEMMs, while the Triton dot path must execute a 16-row tile and
mask 15 rows. Nibble unpacking and per-group scale application also occur in
the kernel. The audit does not claim a measured attribution between those
costs; it records them as implementation-level explanations consistent with
the observed result.

## Correctness, Memory, Graph, and Cleanup

- Maximum absolute output error: `0.001953125`
- Maximum relative output error: `0.006500244140625`
- Unexpected fallbacks: 0
- Rows observing full dequantized-weight allocation: 0
- Observed BF16 weight bytes: 880,803,840
- Observed packed candidate weight bytes: 233,963,520
- Packed/BF16 weight-byte ratio: `0.265625`
- Maximum candidate allocated delta: 0 bytes
- CUDA Graph capture: passed for all four shapes
- CUDA Graph replay count: 2 per shape
- Static pointers stable: yes for all four shapes
- Worker cleanup classification: `CLEAN`
- Post-run exact-tag process scans: three empty scans

The candidate therefore passes the correctness, memory, graph, and cleanup
parts of Stage 0. The terminal rejection is exclusively due to performance.

## Producer and Independent Verification

The producer, remote independent verifier, and freshly rerun local independent
verifier agree:

```text
producer:        NO_GO_PERFORMANCE
remote verifier: NO_GO_PERFORMANCE
local verifier:  NO_GO_PERFORMANCE
manifest:        PASS
```

The independent verifier reconstructed 800 unique
`(shape_id, pair_index)` rows, all four per-shape medians and P99 values, the
execution-count-weighted summary, numerical-error maxima, weight-byte ratio,
graph status, cleanup status, run identity, and source identity. The SHA-256
manifest verifies all ten producer artifacts.

The final committed source was also exercised in the remote CUDA environment
against the 16-file Stage-0 and adjacent quantization/linear suite:

```text
145 passed in 7.58s
```

## Non-Terminal Attempts

Earlier immutable tags were retained as diagnostics and are not used as
performance evidence:

| Tag | Failure before terminal classification | Resolution |
| --- | --- | --- |
| `r1` | source archive omitted the runtime engine tree | archive the complete tracked `tinyvllm/` tree |
| `r2` | lazy Triton `tl` symbol was absent from JIT globals | expose the lazily imported language module to the kernel global namespace |
| `r3` | decode tile M was below Triton's dot-product minimum | pad the physical M tile to 16 and mask inactive rows |
| `r4` | local Python 3.11 rejected `extractall(filter="data")` | explicitly validate regular files/directories and use compatible extraction |

Only `r5` is the terminal Stage-0 result.

## Prompt-to-Artifact Completion Checklist

| Requirement | Concrete evidence | Result |
| --- | --- | --- |
| Work only in the authoritative checkout | controller source identity and Git revision `c1fadc1...` | PASS |
| Use one clean A100 | `controller/gpu_admission.json` selects process-free GPU 1 | PASS |
| Require valid Kerberos without renewal | `controller/kerberos_preflight.json`; no `kinit` or `krenew` | PASS |
| Keep all remote task data under the approved mount | `controller/plan.json` paths all descend from the approved `/data00/home/sitian/...` root | PASS |
| Use committed, source-bound code | tracked-only `git archive` and matching source revision | PASS |
| Consume packed INT4 directly | fused-kernel source plus zero full-dequant flags in all 800 rows | PASS |
| Use real drafter shapes | `shape_manifest.json` is derived from the fingerprinted Qwen3-0.6B checkpoint | PASS |
| Measure 200 unique pairs per shape | 800 unique rows across four shapes | PASS |
| Balance arm order | 400 rows in each order | PASS |
| Enforce numerical tolerances | maximum absolute and relative errors remain below frozen limits | PASS |
| Measure persistent weight bytes | `memory.json` records ratio `0.265625` | PASS |
| Verify CUDA Graph behavior | `graph.json` records capture plus two replays for every shape | PASS |
| Verify clean teardown | worker cleanup and three controller scans are clean | PASS |
| Independently recompute the classification | remote and local verifier receipts both pass and agree | PASS |
| Verify complete compact inventory | all ten manifest entries verify by SHA-256 | PASS |
| Report both benefit and cost | latency, memory, correctness, graph, and cleanup results are recorded above | PASS |
| Enter distillation only after kernel GO | classification is `NO_GO_PERFORMANCE` | STOP RULE ENFORCED |

## Final Classification

```text
NO_GO_PERFORMANCE
```

The Stage-0 implementation is a valid negative result: low-bit storage is
substantially smaller and numerically acceptable, but this custom Triton
W4A16 kernel is 4.81x to 15.23x slower than BF16 for the measured M=1 draft
GEMMs on A100. TinyLLMForge must not proceed to distillation through this
kernel.

The next compression experiment, if pursued, should start from a production
A100 low-bit GEMM backend or from structural drafter compression. It must
first prove real M=1 kernel speed on the same frozen shapes before any
teacher-data or training expense.
