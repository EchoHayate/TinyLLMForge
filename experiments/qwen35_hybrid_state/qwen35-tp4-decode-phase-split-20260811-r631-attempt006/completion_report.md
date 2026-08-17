# Qwen3.5 TP4 Decode Phase-Split r631 Completion Report

Date: 2026-08-11

## Status

The decode projection optimization is complete for the controlled
`w2_long_reuse` TP4 campaign.

```text
run tag:                     qwen35-tp4-decode-phase-split-20260811-r631-attempt006
attempt receipt:             DOWNLOADED
comparison classification:  PERFORMANCE_PASS
completion audit:            PASS
resource policy:             shared-low-utilization
exclusive:                   false
workers:                     12 / 12 complete
token parity:                24 / 24 pairs
cleanup:                     CLEAN
```

The conclusion is specifically about decode tensor-parallel projection
communication and weight layout. No further restore micro-subdivision was
performed.

## What Changed

The two Qwen3.5 attention output projections now use a phase split:

```text
prefill:
  legacy input AllGather
  + full BF16 checkpoint weight
  + dense F.linear

decode:
  local axis-1 FP32 weight shard
  + local FP32 F.linear
  + FP32 AllReduce
  + cast to activation dtype
```

This preserves the accepted long-prefill numerical path while replacing the
decode-only replicated-weight projection AllGather.

The number of projection collectives did not decrease. Across the ten
measured profiles:

```text
                                      r620            r631
legacy projection AllGather           26,880               0
row-parallel AllReduce                      0          26,880
embedding AllReduce                    1,120           1,120
```

The valid claim is therefore a collective-type, payload/layout, and latency
improvement, not a reduction in collective call count.

## Correctness

The complete token matrix contains six recompute/exact-restore case pairs:
one warmup pair and five measured pairs. Each pair contains four requests.

```text
candidate recompute vs exact restore: 24 / 24 match
candidate recompute vs r620:          24 / 24 match
candidate exact restore vs r620:      24 / 24 match
```

The normal aggregate ran twice and produced byte-identical
`decode_summary.json` files. The comparison authority also reports:

```text
output_key_sets_match: true
output_parity:         true
```

This closes the r630 failure. r630 remains `NO_GO`: six of its 24 pairs
mismatch, all on `request-1` at generated-token index zero. Its timing values
remain diagnostic-only.

## Performance

Policy medians:

```text
                         r620 recompute  r630 recompute  r631 recompute
steady wall                  83.515896 ms    73.896042 ms    72.100637 ms
steady CUDA                  83.515106 ms    73.895904 ms    72.100288 ms
collective CUDA              50.221536 ms    42.740864 ms    35.860224 ms

                         r620 exact      r630 exact      r631 exact
steady wall                  77.073588 ms    72.226914 ms    71.883384 ms
steady CUDA                  77.073265 ms    72.225792 ms    71.883282 ms
collective CUDA              53.263872 ms    42.990192 ms    35.310368 ms
```

r631 improvement relative to the valid r620 baseline:

```text
                                      recompute       exact restore
steady wall                            13.6684%          6.7341%
steady CUDA                            13.6680%          6.7338%
collective CUDA                        28.5959%         33.7067%
steady wall p90                        31.3464%         31.8676%
steady CUDA p90                        31.3464%         31.8687%
```

Both policies exceed the frozen 5 percent steady-wall speedup gate, improve
steady CUDA, and do not regress the p90 gates. The comparison classification
is therefore `PERFORMANCE_PASS`.

r631 is also modestly faster than the parity-failed r630 diagnostic:

```text
                                      recompute       exact restore
steady wall                             2.4296%          0.4756%
steady CUDA                             2.4299%          0.4742%
collective CUDA                        16.0985%         17.8641%
```

Because the GPUs are shared and non-exclusive, these numbers establish the
result for this controlled five-pair campaign, not a universal hardware
throughput guarantee.

`step_wall_ns - step_cuda_ns` remains only an upper bound combining host
orchestration, launch gaps, and possible synchronization waiting. It is not
used to attribute a specific host-side bottleneck.

## Resource and Cleanup Evidence

All 12 worker guards admitted exactly GPUs `2,4,5,6`.

```text
minimum observed free memory: 56,982,765,568 bytes
maximum observed utilization: 0 percent
minimum required free memory:  26,843,545,600 bytes
maximum allowed utilization:   10 percent
```

The guards permit unrelated low-utilization processes and label the run
shared/non-exclusive. No dummy reservation was created and no unrelated
process was killed.

The runner receipt records exact-attempt cleanup as `CLEAN`. A separate
read-only `/proc` check after download found zero command lines containing
the exact attempt tag in three consecutive samples:

```text
CHECK 1 MATCHES 0
CHECK 2 MATCHES 0
CHECK 3 MATCHES 0
```

## Validation

Fresh isolated-process CPU tests:

```text
output projection row-parallel:        3 passed
full-attention shell:                 12 passed
linear-attention shell:               15 passed
concrete component factory:            4 passed
checkpoint target binding:             4 passed
decode profile wiring:                 4 passed
row-parallel comparison:               4 passed
checkpoint output-projection slice:     1 passed, 8 deselected
total:                                47 passed, 8 deselected
python3.12 -m py_compile:              PASS
scoped git diff --check:               PASS
```

The tests are run as separate pytest processes because the shell tests
intentionally install lightweight module stubs during collection. Combining
all files in one pytest process pollutes `sys.modules`; the isolated-process
results above are the valid repository test mode.

The complete checkpoint-assignment file still has three pre-existing
segmented-QKV/packed-oracle failures. Its output-projection-specific test
passes.

## Artifacts

```text
attempt_receipt.json
831c149ebb0f738320a4e9d28a3f598326cb1de6017592b8cfb81c8249f3871c

decode_summary.json
dd06ac6b52faba6674171ecc81c7c7b2d407ea7e0d41b3c949b3413b8c071888

row_parallel_comparison.json
460e57ac8faa02a9c6c24d1063222e95ff271fe72141be3bbb3c0292ec227b77

evidence_inventory.json
3cad0e7694fbf5ade4c0c139775e0b0e027ef08d1c9d26a1a31dd7bda190d234

three_way_metrics.json
7db5e27f5cf6c27ffc7365967774f6e244387400e190ee5d5f4a29ad6257ac98

download/result.tar
f4151e149c20783e0e5c3f520be8c47a99688b03770b36553728e73292e79269
```

## Conclusion

The next worthwhile optimization target was correctly moved from restore
micro-timing to decode TP projection communication. The phase-split
implementation preserves prefill correctness, removes the decode legacy
AllGather, executes true row-parallel AllReduce, and passes the controlled
performance gate.

The natural next optimization, if more work is desired, is to reduce or fuse
the remaining decode collective calls or improve the AllReduce schedule.
That is a new optimization phase; it is not required to validate this
AllGather-to-row-parallel migration.
