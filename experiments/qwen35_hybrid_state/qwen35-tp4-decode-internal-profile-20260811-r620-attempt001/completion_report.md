# Qwen3.5 TP4 Decode-Internal Profile r620 Completion Report

Date: 2026-08-11

## Status

The final attempt is complete and passes the prompt-to-artifact audit.

```text
run tag:                     qwen35-tp4-decode-internal-profile-20260811-r620-attempt001
attempt classification:      COMPLETE
completion audit:            PASS
resource policy:             shared-low-utilization
exclusive:                   false
structured workers:          12 / 12, all return code 0
cleanup:                     CLEAN
```

This diagnostic is supporting evidence only. It does not replace or modify
the canonical benchmark manifest, case matrix, case-row schema, existing
`profile.json` schema, or the r607-r611 canonical artifacts.

## Structured Result

The frozen classifier returns:

```text
mixed_or_inconclusive
```

Policy medians:

```text
                                      recompute       exact restore
first-step CUDA                       96.752560 ms     91.351761 ms
first-step wall                       96.752904 ms     91.351780 ms
steady-state CUDA                     83.515106 ms     77.073265 ms
steady-state wall                     83.515896 ms     77.073588 ms
collective CUDA                       50.221536 ms     53.263872 ms
non-CUDA upper bound                   0.001196 ms      0.001279 ms
```

Ratios are exact restore divided by recompute:

```text
                                      paired median   ratio of medians
first step                            0.944433x        0.944176x
steady state                          1.010220x        0.922861x
collective                            1.165687x        1.060578x
non-CUDA upper bound                  1.069398x        1.069398x
```

The five paired steady-state ratios are:

```text
0.825997x, 1.030626x, 1.027751x, 0.893131x, 1.010220x
```

The five paired collective ratios are:

```text
0.740892x, 1.175624x, 1.563902x, 0.867090x, 1.165687x
```

The directions do not agree in at least four of five repetitions. The paired
steady-state median says exact restore is about 1.02 percent slower, while
the ratio of policy medians says it is about 7.71 percent faster. Collective
ratios vary from 0.74x to 1.56x. Under the frozen rules and shared GPU
environment, this is not a stable decode regression or speedup claim.

`step_wall_ns - step_cuda_ns` remains only an upper bound combining host
orchestration, launch gaps, and possible synchronization waiting. The values
are around one microsecond and do not identify host or synchronization
waiting as the cause of a material slowdown.

## Profile Inventory

All 12 downloaded `decode_profile.json` files validate. Across all 48 rank
snapshots, every rank has:

```text
total steps:                  32
prefill steps:                 4
decode steps:                 28
request digest groups:         4
collectives:                 700
  embedding all-reduce:       28
  replicated-weight AllGather:672
```

The AllGather source is
`ReplicatedWeightRowParallelLinear.forward()`. It is now recorded as
`replicated_weight_row_parallel_all_gather`, alongside
`vocab_parallel_embedding_all_reduce`.

Output parity passes for all 24 request comparisons: four warmup requests and
twenty measured requests. Recompute and exact restore produced identical
token IDs and token-ID SHA256 values.

## Nsight Evidence

Nsight Systems:

```text
version:                     2024.7.1.84-247135125610v0
requested trace:             cuda,nvtx,osrt,nccl
effective trace:             cuda,nvtx,osrt
cuda_gpu_kern_sum:           available
nvtx_pushpop_sum:            available
nvtx_kern_sum:               available
nccl_sum:                    unavailable
```

Both representative replays have the expected NVTX inventory:

```text
prefill ranges:                               16
first-decode ranges:                          16
steady-decode ranges:                         96
embedding all-reduce ranges:                 112
replicated-weight AllGather ranges:         2688
```

The representative r4 replay is strongly perturbed:

```text
                                      exact / recompute
first-decode total kernel time             4.141x
first-decode AllGather kernel time          4.836x
first-decode GEMM/GEMV kernel time          0.998x
steady-decode total kernel time             3.455x
steady-decode AllGather kernel time          4.061x
steady-decode GEMM/GEMV kernel time          0.999x
```

The extra replay kernel time is overwhelmingly associated with NCCL
AllGather, while GEMM/GEMV and other non-collective kernels are essentially
unchanged. However, the r4 Nsight replay disagrees sharply with the
five-repetition structured result and was collected on shared, non-exclusive
GPUs. It is evidence that AllGather is the sensitivity point when contention
occurs, not evidence of a stable exact-restore regression.

`nccl_sum` is unavailable because the installed Nsight version does not
provide that report. NCCL evidence is limited to NVTX-associated CUDA kernel
names and timings.

## Guards and Overhead

All 17 entry/worker/overhead/Nsight guards ended `READY` on GPUs `2,4,5,6`.
The minimum admitted free memory was 52.964 GiB per selected GPU and the
maximum admitted utilization was 0 percent. No guard query error occurred.

The fresh-process overhead smoke measured:

```text
baseline:                     33.973174 s
decode-internal profile:      71.284163 s
ratio:                         2.098249x
```

This ratio includes model initialization and the cost of recording 700
collective events per rank. It is not per-token profiler overhead and must not
be used as an inference throughput result.

## Conclusion

The current evidence does not support a stable first-step, steady-state,
collective, CUDA-compute, or host/synchronization regression caused by exact
prefix restore. The five paired structured measurements are directionally
inconsistent, so the correct result is `mixed_or_inconclusive`.

The useful implementation conclusion is narrower: TP AllGather is the only
component that expands dramatically in the perturbed representative Nsight
replay; GEMM/GEMV and other kernels stay flat. If performance work continues,
the next experiment should isolate AllGather under an exclusive or more
controlled GPU allocation, or run a longer paired decode campaign. Further
restore micro-subdivision is not the next priority.

## Fresh Local Validation

Validation run after the completion audit and report were written:

```text
focused pytest suite:                102 passed in 0.83s
python3 -m py_compile:               PASS
completion audit JSON parse:         PASS
completion report marker check:      PASS
scoped git diff --check:             PASS
canonical tracked-diff name guard:   PASS
```

The dedicated
`tools/test_replicated_weight_row_parallel_linear.py` test could not collect
under the local Command Line Tools Python because that interpreter does not
have `torch`:

```text
ModuleNotFoundError: No module named 'torch'
```

This is an environment limitation, not a passing or failing model-path test.
The actual TP4 path, including the replicated-weight AllGather, executed in
all 12 remote workers and both Nsight replays.

The r607 artifact directory is already untracked in this checkout, so Git
cannot provide a historical byte-diff baseline for it. This r620 workflow
used a fresh attempt-specific remote and local output path and did not write
to r607-r611.

## Artifact Digests

```text
attempt_receipt.json
7fe69e66ffee3fb6e72eb20ae7975e1dd29aae438cfb5ac9a6a3766eb847d337

decode_summary.json
095690c90bf489de3e30406b2d6491ed3ba050dbf071780efbf894caa880080b

nsys_receipt.json
b2d99d583d32fb8fd7c09b7150e550de67d4e313bbb787775a166883348508fa

completion_audit.json
fbeaf70c05b11ec298a2facc120ec78e1374435ed28b58906860a5634817efed
```
