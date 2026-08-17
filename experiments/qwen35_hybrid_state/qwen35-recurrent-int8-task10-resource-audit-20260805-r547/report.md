# Qwen3.5 Recurrent INT8 Task 10 Resource Audit

## Decision

`BLOCKED_RESOURCES`

The historical r542 correctness prerequisite now validates under the current
schema-v2 contract, but the fresh strict-P1 read-only preflight cannot
authorize execution because the complete fixed GPU set `2,4,5,6` is not
eligible.

## Superseded Finding

The r546 completion audit recorded r542 as `BLOCKED_CORRECTNESS`. That finding
was caused by validator drift:

- historical source tags were rejected too narrowly;
- controlled-shared resource guards were treated as strict-exclusive guards;
- receipt and artifact verifier check counts were required to be identical.

The compatibility fix preserves the security bindings while accepting the
real historical authority semantics. Fresh validation now returns:

```text
classification: PASS
authorized:     true
reasons:        []
```

The r542 document itself is unchanged:

```text
experiments/qwen35_hybrid_state/
qwen35-tp4-performance-correctness-prerequisites-20260804-attempt67-r542/
correctness_prerequisites.json
SHA256 35b4bf092d5c4c84746b88ecd88b32bf14357a21d2923336d62653186cf352f8
```

The r546 findings that calibration `PASS`, strict-P1 `GO`, and Gate-1 audit
`PASS` documents are absent remain unresolved.

## Strict-P1 Preflight

Command mode: schema-v1 `preflight`

```text
run tag:
qwen35-tp4-strict-p1-preflight-20260805-224025-task10-r547

classification:       BLOCKED_RESOURCES
authorized:           false
remote query:         executed
remote path created:  false
```

Fixed GPU observations:

```text
GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,963,939,328 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 63,191,384,064 free bytes, 3 compute processes
```

The free-byte threshold alone was satisfied, but the strict policy requires
the complete fixed set `2,4,5,6` to be eligible. Existing compute processes
on GPUs 2, 4, and 6 therefore blocked authorization. No process was killed or
modified.

Evidence:

```text
benchmark_preflight.json
SHA256 3532919a71e82bff48620bf11457e4b134af4b55388e1434462bfadb460e0c14

benchmark_source.tar
SHA256 043c5639535722dca5cd7e69426e77f7efb0d17bd8dd402021fdb9eeef18e138

source tree
SHA256 74eab06baa781abb997f8e3421768527ad9a0392f7070704c9977d3dac1e54a0
```

The run directory contains only the preflight JSON and local source tar. It
contains no execution plan, authorization, execution receipt, downloaded
artifact, or verifier result.

## Safety Boundary

The read-only SSH/GPU inventory query ran. The resource-blocked path created
no remote run directory, staged no source, launched no worker, created or
consumed no authorization, and executed no benchmark authority.

Strict-P1 smoke, recurrent-INT8 calibration, v2 preflight, and P2 canonical
authority are prohibited while this resource block remains.

No correctness, accuracy, cache, capacity, memory, compression, latency,
throughput, or speed benefit is established.

## Exact Next Action

Wait for all fixed GPUs `2,4,5,6` to satisfy the strict resource policy, then
rerun a fresh strict-P1 read-only preflight with a new unique tag. If it is
`READY`, inspect and authorize the deterministic strict-P1 smoke plan. Only a
real independently verified strict-P1 `GO` may feed full-fidelity snapshots
into recurrent-INT8 calibration.
