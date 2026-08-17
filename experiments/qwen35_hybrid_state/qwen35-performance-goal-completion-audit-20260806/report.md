# Qwen3.5 High-Performance Inference Goal Completion Audit

## Decision

`NOT_ACHIEVED`

The implementation has substantial local coverage, but the user-level goal
requires three real, source-bound outcomes at the same time:

1. canonical real-model accuracy does not regress;
2. end-to-end inference latency or throughput improves; and
3. physical cache or CUDA-resident memory decreases.

No current-source artifact proves all three outcomes. Passing local tests,
synthetic calibration, a historical correctness prerequisite, or a
current-source Gate-1 audit is not a substitute for a real P1/P2 authority
comparison.

## Current Source Identity

The deterministic schema-v1 benchmark source-bundle builder was invoked
locally without its preflight or remote query:

```text
owned files:        91
source tree SHA256: 88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c
```

The temporary tar was deleted after hashing. No SSH, remote query, remote
directory, CUDA import, GPU operation, process mutation, authorization, or
execution receipt was created.

The old strict-P1 resource preflight is bound to:

```text
e265b3ead9d9717d92d8bc0507ac051d93ec22f8403b7929c3625ee4153ccfd7
```

It is historical evidence and cannot authorize the current benchmark-owned
source. Its resource result was also `BLOCKED_RESOURCES`, not `READY`.

A current-source Gate-1 audit has now passed:

```text
artifact:    qwen35-recurrent-int8-gate1-audit-20260806-r550
source tree: 88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c
tests:       185 passed
```

This closes the local integration gate only.

A current-source strict-P1 local readiness package is also complete:

```text
artifact:       qwen35-tp4-strict-p1-readiness-20260806-r551
classification: READY_FOR_RESOURCE_PREFLIGHT
source tree:    88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c
source files:   91
cases:          70
evidence files: 18
local audit:    PASS
```

The production prerequisite validator returns `PASS`, `authorized=true`, and
`reasons=[]` from the copied package location. All execution, SSH, GPU, and
remote-path authorization fields remain `false`. This closes local input
portability and integrity only; it is not a resource preflight or a P1 run.

## Prompt-to-Artifact Checklist

| Requirement | Required authority | Current evidence | Decision |
|---|---|---|---|
| Do not reduce accuracy | Current-source real P1 plus P2 quality comparison under canonical rules | Historical correctness prerequisite is `PASS`; current P2 authority is absent | Missing |
| Make inference faster | Current-source canonical P1/P2 latency or throughput artifact and independent `GO` | No canonical P2 artifact classified `GO` exists | Missing |
| Use less cache | Current-source canonical P1/P2 physical cache/CUDA-byte comparison and independent `GO` | Gate-1 tests cover accounting behavior only; no real physical reduction result exists | Missing |
| Preserve default behavior | Default-off regression and schema closure | Current-source Gate-1: 185 passed; capture suite: 234 passed | Current-source local gate passed |
| Capture real calibration data | Real TP4 full-fidelity capture, four-rank closure, immutable bundle | CPU synthetic pipeline passes; real capture not run | Missing |
| Calibrate recurrent INT8 | Independent real calibration classification `PASS` | Producer/verifier exist; no real bundle result | Missing |
| Portable strict-P1 inputs | Current source tar, canonical workload, complete correctness bundle, Gate-1 binding, and local independent audit | r551 readiness is `READY_FOR_RESOURCE_PREFLIGHT`; 91 source files, 70 cases, 18 evidence files, local audit `PASS` | Current-source local gate passed |
| Complete P2 decision gates | Independent verifier consumes all frozen quality, cache, capacity, performance, decode, and CUDA-memory thresholds | r553 audit: 12/12 thresholds consumed; 3 omitted-gate regressions added; 19 verifier tests pass | Current-source local gate passed |
| Auditable P2 report | Human report binds source/model/workload/configuration/artifact hashes, prints every measured gate and frozen threshold, identifies failures, and bounds W3 claims | r553 report authority: canonical `GO` completeness plus CUDA-only `NO_GO_PERFORMANCE` failed-gate regression; 19 verifier tests pass | Current-source local gate passed |
| Safe execution | Fresh preflight `READY`, then explicit SSH/GPU authorization | Fresh current-source r554 preflight is `BLOCKED_RESOURCES`; remote query ran, but no remote path was created and no worker launched | Blocked |
| Same frozen identity | Correctness, P1, calibration, Gate-1, and P2 bind accepted identities | Gate-1 and readiness bind current source; real P1, calibration, and P2 do not yet exist | Partial |

The machine-readable mapping is in `checklist.json`.

## What Existing Green Signals Prove

- The historical real correctness prerequisite bundle is accepted as `PASS`
  by the current contract.
- The current Gate-1 audit ran 185 local tests and classified the default-off
  recurrent-INT8 integration `PASS`.
- The full-fidelity capture project completed a 234-test CPU suite, including
  synthetic capture, close, assemble, calibrate, and verify paths.
- The runtime and authority tooling exist for exact-restore P1 and compressed
  P2 comparisons.
- The strict-P1 readiness package is self-contained: its sibling
  `prerequisites/` closure contains all 18 evidence files, and its
  independent local audit validates all 91 owned source files and 70 cases.
- The P2 verifier now consumes all 12 approved frozen thresholds. Before the
  r553 audit, it omitted W1 recompute-relative median TTFT and both W2
  exact-relative TTFT gates; three negative regressions now prevent those
  artifacts from being incorrectly classified `GO`.
- The verifier-generated human report now exposes all measured cache,
  capacity, TTFT, W3 concurrent-E2E proxy, decode, and CUDA-memory ratios,
  prints all 12 frozen threshold gates with per-gate `PASS` or `FAIL`, binds
  the source/model/workload/configuration and producer artifact hashes, and
  explicitly forbids interpreting W3 as sustained serving QPS or tokens/s.

These signals prove implementation readiness and regression coverage. They do
not prove the requested real accuracy, speed, cache, or VRAM outcomes.

## Missing Critical Path

The fresh current-source r556 preflight observed that GPU `5` was idle, while
fixed GPUs `2`, `4`, and `6` had active compute processes. It therefore
classified `BLOCKED_RESOURCES`, with `authorized=false` and
`remote_path_created=false`. All four GPUs had more than the 25 GiB free-byte
minimum, but the strict-P1 policy also requires zero active compute processes.
The observed processes pre-existed the preflight; this task did not allocate
GPU memory or mutate them. The current source tree and source tar remain
identical to r551 and r554.

The next valid chain is:

1. after GPUs `2`, `4`, and `6` are idle, obtain separate approval for another
   fresh read-only SSH resource preflight using the verified r551 package;
2. require fixed GPUs `2,4,5,6` to return `READY`;
3. execute and independently verify current-source strict-P1 as `GO`;
4. run the five canonical real full-fidelity capture workers;
5. close all four rank staging directories and assemble the immutable bundle;
6. run calibration and require independent classification `PASS`;
7. execute canonical P2 with capture disabled;
8. require the independent P2 verifier to classify `GO`; and
9. report accuracy, latency/throughput, physical cache bytes, CUDA peak
    memory, and capacity deltas only from that bound P1/P2 pair.

The canonical W3 `throughput_ratio` is a frozen eight-request concurrent E2E
proxy, not a sustained online requests/s or tokens/s measurement. A later
serving stress authority may add arrival-rate, queueing, makespan,
requests/s, tokens/s, TTFT/ITL percentiles, cache-hit rate, and device-memory
telemetry, but it must not replace or weaken canonical schema-v2.

If the fresh resource preflight is `BLOCKED_RESOURCES`, the chain must stop
without creating a remote path or launching a worker.

## Claim Boundary

As of this audit, the project must not claim that Qwen3.5 inference is faster,
uses less physical cache or VRAM, supports more capacity, or preserves
canonical quality under recurrent INT8. Those remain hypotheses awaiting the
real current-source authority chain.
