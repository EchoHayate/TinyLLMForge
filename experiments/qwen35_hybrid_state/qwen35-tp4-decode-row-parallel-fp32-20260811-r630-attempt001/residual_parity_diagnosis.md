# r630 residual parity diagnosis

Status: `NO_GO`

## Deterministic mismatch

All 12 workers exited with return code 0 and all case rows use source tree
SHA256:

```text
7a80d62c2c9e71f7899dc397f810427286b330d95da2b63f67839aa98d47b3b3
```

The downloaded artifacts contain 24 recompute/exact-restore request pairs:
four requests in one warmup pair and five measured repetition pairs.

Exactly six pairs mismatch. Every mismatch is `request-1`, and every one
diverges at generated token index 0:

```text
recompute:     [68, 197, 197, 92, 197, 197, 92, 197]
exact_restore: [197, 197, 13, 197, 197, 13, 197, 197]
```

The other 18 pairs are exact token matches. The same mismatch repeats in
warmup r0 and measured r0-r4, so this is not classified as intermittent
NCCL/process nondeterminism.

## Cross-run localization

For measured `request-1`, all five repetitions are identical within each
run:

```text
r620 recompute:     [197, 197, 13, 197, 197, 13, 197, 197]
r620 exact_restore: [197, 197, 13, 197, 197, 13, 197, 197]

r621 recompute:     [68, 197, 197, 13, 197, 197, 13, 197]
r621 exact_restore: [68, 197, 197, 13, 197, 197, 13, 197]

r630 recompute:     [68, 197, 197, 92, 197, 197, 92, 197]
r630 exact_restore: [197, 197, 13, 197, 197, 13, 197, 197]
```

FP32 local partial accumulation therefore repairs the exact-restore path,
including its decode and 64-token suffix prefill, but does not preserve the
legacy result for the benchmark's source-then-long-recompute path.

The earlier r629 standalone request-1 probe produced token `197` for both
recompute and restore. Its recompute request ran before the source
publication request, unlike the benchmark worker, so it did not reproduce
the full benchmark history.

## Performance boundary

The normal aggregator rejected r630 before writing `decode_summary.json`.
Reading the validated per-case decode profiles without bypassing the parity
gate gives these medians:

```text
                                      recompute       exact restore
steady-state wall                    73.896042 ms     72.226914 ms
steady collective CUDA              42.740864 ms     42.990192 ms
first-step wall                      79.065518 ms     73.270706 ms
```

Relative to r620, the raw r630 policy medians are:

```text
recompute steady wall:               11.5186% lower
exact-restore steady wall:            6.2884% lower
recompute collective CUDA:           14.8953% lower
exact-restore collective CUDA:       19.2883% lower
```

These are diagnostic performance numbers only. r630 remains `NO_GO`
because output parity failed, and its five repetitions contain substantial
shared-GPU outliers.

## Cleanup boundary

`run_real_attempt()` executes `_cleanup_attempt_processes(run_tag)` in a
`finally` block, so cleanup was attempted after aggregation raised:

```text
ValueError: output parity mismatch: repetition=0, request=1
```

The exception is re-raised before the runner attaches the cleanup result to
`attempt_receipt.json`, so no structured cleanup receipt was preserved.
Current remote verification is blocked because the `sitian` Kerberos
credentials for `jump-proxy-hl` and `10.232.195.203` expired on
2026-07-29. Direct port 22 access times out. Do not upgrade cleanup to
confirmed `CLEAN` until a fresh remote process check succeeds.

## Implemented next candidate

The next local candidate preserves the original dense prefill math while
retaining true row-parallel decode:

- Qwen3.5 output projections cache the full BF16 checkpoint weight only
  when `preserve_dense_prefill=True`.
- Prefill uses the legacy input AllGather plus one full BF16 `F.linear`.
- Decode uses the local axis-1 shard, FP32 local `F.linear`, FP32 AllReduce,
  then casts back to the activation dtype.
- Both linear-attention and full-attention shells dispatch
  `forward_prefill()` only when `context.is_prefill`.

This intentionally optimizes decode rather than changing long-prefill
numerics. It increases projection-weight residency and requires a fresh GPU
resource guard before execution.

## Local validation

TDD RED:

```text
3 failed
- missing preserve_dense_prefill constructor argument
- full-attention prefill did not call forward_prefill
- linear-attention prefill did not call forward_prefill
```

GREEN and focused validation:

```text
output projection row-parallel:       3 passed
full-attention shell:                 12 passed
linear-attention shell:               15 passed
concrete component factory:            4 passed
checkpoint target binding:             4 passed
checkpoint output-projection slice:     1 passed, 8 deselected
decode profile wiring:                  4 passed
row-parallel comparison:                4 passed
python3 -m py_compile:                 PASS
scoped git diff --check:               PASS
```

The complete checkpoint-assignment file still has three pre-existing
segmented-QKV/packed-oracle failures; its output-projection-specific test
passes.

## Next action

After refreshing Kerberos:

1. Read-only check that no process command line contains the r630 tag.
2. Run the resource guard on GPUs `2,4,5,6`.
3. Package the current source under a fresh tag; do not reuse r630.
4. Run the full paired benchmark.
5. Require all 24 token pairs to match before generating the comparison,
   Nsight evidence, completion audit, or any performance-pass claim.
