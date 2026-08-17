# Qwen3.5 Recurrent INT8 Gate-1 Audit

## Decision

`PASS`

Gate 1 proves the local, default-off runtime integration of
`recurrent_int8_per_row` against source tree:

```text
e265b3ead9d9717d92d8bc0507ac051d93ec22f8403b7929c3625ee4153ccfd7
```

The default remains `exact_restore`. Schema-v1 remains restricted to
`recompute` and `exact_restore`; it rejects the P2 representation.

## Verification

The frozen 12-entrypoint Gate-1 suite ran in isolated Python 3.12 processes:

```text
180 passed
0 failed
```

The isolation is intentional. Several dependency-light test files install
module stubs, so combining all files in one pytest process creates cross-file
collection pollution. Every file was therefore collected and executed in its
own process.

Additional gates:

```text
key module py_compile: PASS
scoped git diff --check: PASS
owned source files: 89
```

The default `python3` on this host does not contain Torch. Runtime tests used
`/opt/homebrew/bin/python3.12`. Its missing pytest dependency was supplied only
through the existing pure-Python `/tmp/tinyllmforge-pytest312-shim`.

## Coverage

The detailed checklist in `checklist.json` maps all 15 Gate-1 acceptance
requirements to concrete test functions. It covers:

- closed representation/config selection and default-off behavior;
- codec range, scale, decode, and tamper validation;
- exact 18-layer P2 snapshot inventory;
- atomic publication and rollback;
- collision-safe tensor interning;
- immutable reader leases;
- private all-layer FP32 decode staging;
- quarantine, accounted miss, recompute, and no P1 fallback;
- cross-layer commit and rollback failure handling;
- resident/workspace/CUDA accounting separation;
- owner/ModelRunner/Engine distributed identity;
- P1 exact regressions;
- schema-v1 P0/P1 closure;
- compilation and scoped whitespace hygiene.

## Claim Boundary

This is a local integration gate. It does not establish that P2 preserves
canonical model outputs, reduces physical cache bytes, increases same-budget
capacity, lowers CUDA memory, improves TTFT, improves throughput, or improves
decode latency.

Those claims remain prohibited until real calibration passes, strict-P1 is
independently classified `GO`, resources are `READY`, and a complete P2
canonical artifact is independently classified `GO`.

## Next Action

Produce a fresh strict-P1 authority and independently verify it as `GO` when
the fixed GPUs `2,4,5,6` are all eligible. Its source tree must match this
Gate-1 audit or a fresh Gate-1 audit must be generated for the newer source
identity. Only then may full-fidelity snapshots feed recurrent-INT8
calibration.
