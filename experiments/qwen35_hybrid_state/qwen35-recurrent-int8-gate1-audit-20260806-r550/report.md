# Qwen3.5 Recurrent INT8 Gate-1 Audit

## Decision

`PASS`

Gate 1 proves the local, default-off runtime integration of
`recurrent_int8_per_row` against the current deterministic benchmark-owned
source tree:

```text
88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c
```

The source bundle contains 91 owned files. It was generated in a temporary
local directory and deleted after hashing.

## Verification

The frozen 12-entrypoint Gate-1 suite ran in isolated Python 3.12 processes:

```text
185 passed
0 failed
```

Additional gates:

```text
key module py_compile: PASS
scoped git diff --check: PASS
checklist requirements: 15 PASS
```

The increase from the historical 180 tests to 185 comes from five additional
engine-adapter tests added by the separately approved full-fidelity capture
work.

## Test Maintenance

The first current-source run found one stale test assertion:

```text
tools/test_qwen35_hybrid_prefix_publication_candidate.py
```

Its intent is to prove that
`capture_qwen35_hybrid_prefix_publication_candidate` is not wired into
`LLMEngine`, `Scheduler`, or `ModelRunner`. The assertion instead forbade the
broad substring `capture_qwen35`, which falsely matched the independent
default-off method `_capture_qwen35_recurrent_source_state`.

The test was narrowed to the exact publication-candidate symbol. No production
file changed. The focused file then passed 6 tests, and the complete Gate-1
suite was rerun from the beginning.

## Coverage

The 15-item checklist covers:

- closed representation/config selection and default-off behavior;
- recurrent INT8 codec validation and private FP32 decode;
- exact 18-layer P2 snapshot inventory;
- atomic publication, rollback, collision-safe interning, and immutable
  leases;
- private all-layer staging, quarantine, accounted miss, and no P1 fallback;
- workspace, CUDA allocated, CUDA reserved, logical, and physical accounting;
- owner, ModelRunner, Engine, and distributed rank identity;
- P1 transaction regressions; and
- schema-v1 P0/P1 closure.

## Claim Boundary

Gate 1 is a local integration and regression authority only. It does not prove
canonical model-output parity, physical cache reduction, capacity gain, CUDA
memory reduction, latency improvement, throughput improvement, or decode
improvement.

No SSH, remote query, remote directory, CUDA import, GPU operation, process
mutation, authorization, or execution receipt occurred during this refresh.

## Next Action

The current source now has a Gate-1 `PASS`, but it still lacks:

```text
current-source strict-P1 authority: missing GO
real full-fidelity capture:         not run
real recurrent-INT8 calibration:    missing PASS
canonical current-source P2:        not run
```

The next action requires separate approval for a fresh read-only SSH resource
preflight. Only if fixed GPUs `2,4,5,6` are `READY` may the strict-P1 chain
continue. If resources are blocked, create no remote path and launch no
worker.
