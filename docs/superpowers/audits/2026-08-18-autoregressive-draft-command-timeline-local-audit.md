# Autoregressive Draft Command Timeline Local Completion Audit

**Date:** 2026-08-18

**Authoritative checkout:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

**Pre-audit local HEAD:** `5503f4ed33d23601f0b8eb1480e55615487cbc81`

**Pre-audit origin HEAD:** `8cf39121ffbe357812941e2e05628ed8ab1153ac`

## Claim Boundary

This audit establishes only the local, CPU-testable implementation and
source-bound runner contract for the command-timeline diagnostic.

It does not establish:

- a completed source-bound TP4/B4/Q4 command-timeline bundle;
- a localized runtime boundary;
- a runtime optimization;
- a performance improvement;
- Phase 1 completion; or
- promotion readiness.

The real GPU campaign was not run during Task 7 or Task 8.

## Storage Boundary

All generated validation environments, caches, pytest basetemps, pycache,
review reports, and receipts used by this audit are under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

No Task 7/8 archive, cache, bundle, telemetry file, log, receipt, manifest,
validation root, review artifact, pycache, or pytest temporary directory was
created under local or remote `/`, `/tmp`, `/private/tmp`, or the repository
source tree.

The runner exports:

```text
TMPDIR
TMP
TEMP
PYTHONPYCACHEPREFIX
XDG_CACHE_HOME
```

to paths beneath the Sitian task root for every SSH payload. It bootstraps
only:

```text
runtime/ssh/scratch
runtime/ssh/pycache
runtime/ssh/xdg
```

before the payload executes, preventing a missing-directory fallback to
remote `/tmp`.

## Prompt-to-Artifact Checklist

| Requirement | Source implementation | Focused test | Expanded test | Artifact/verifier coverage | Status | Remaining remote evidence |
|---|---|---|---|---|---|---|
| Reuse paired-stability admission | `autoregressive_draft_command_timeline_diagnostic.py` reuses fixed block admission and stability thresholds | Command-timeline diagnostic suite | 533- and 687-test suites | Canonical artifact recomputes admission | ESTABLISHED_LOCALLY | Real eight-epoch samples |
| Reuse host/GPU telemetry | Frozen host sampler plus full GPU sampler schema in the remote runner | Runner telemetry tests | 533- and 687-test suites | Per-repeat telemetry sidecars are source/raw-input bound | ESTABLISHED_LOCALLY | Real sampler output |
| Dual verifier and manifest | Primary frozen source and controller-copy frozen source are independently verified | Verifier and runner tests | 533- and 687-test suites | Manifest and normalized receipt comparison | ESTABLISHED_LOCALLY | Real primary/controller receipts |
| Exact graph/eager identity | Fixed `eager_graph, graph_eager, graph_eager, eager_graph` schedule and TP4/B4/Q4 worker command | Schedule/worker command tests | 533- and 687-test suites | Configuration and epoch identity are canonical fields | ESTABLISHED_LOCALLY | Real epoch payloads |
| Default-off command timeline | Config, model-runner recorder, engine-step recorder, diagnostic-only worker enablement | Timeline/config/wiring tests | 533- and 687-test suites | Timeline snapshots are embedded and recomputed | ESTABLISHED_LOCALLY | Real all-rank snapshots |
| Queue debt, CUDA, acknowledgement, scheduler/postprocess decomposition | Command recorder, deferred CUDA rows, ack timing, engine-step phase spans | Recorder, ack, profiler, diagnostic tests | 533- and 687-test suites | Boundary effects and conservation are canonical | ESTABLISHED_LOCALLY | Real cross-rank timing rows |
| No new measured-path fence or synchronization | Observation remains default-off; deferred CUDA events resolve after existing synchronization | Wiring and forbidden-pattern tests | 533- and 687-test suites | Audit forbidden-pattern scan had zero matches | ESTABLISHED_LOCALLY | Runtime trace confirmation |
| Exact token and Proposal-KV transaction parity | Existing exact-greedy and transaction invariants remain required by worker admission | Performance, graph-gate, and diagnostic tests | 533- and 687-test suites | Epoch admission validates parity and zero active transactions | ESTABLISHED_LOCALLY | Real worker outputs |
| Timing conservation | Step and command decomposition enforce integer conservation tolerance | Engine-step and diagnostic tests | 533- and 687-test suites | Artifact recomputation rejects nonconserving rows | ESTABLISHED_LOCALLY | Real measured spans |
| Stationarity and localization thresholds | MAD/median, half drift, 60% explanation, sign, and 10% residual rules | Diagnostic threshold tests | 533- and 687-test suites | Classification is fully recomputed | ESTABLISHED_LOCALLY | Real paired effects |
| Immutable schema-v2 `r3` | New runner uses a distinct command-timeline family and does not rewrite `r3` | Immutable/tag/path tests | 533- and 687-test suites | No `r3` artifact is consumed as mutable output | PRESERVED | None |
| Source-bound closure | Working-tree source archive is built in memory, streamed, persisted only remotely, and used by workload and both verifiers | Archive and frozen-source tests | 533- and 687-test suites | Source manifest/tree digest and source patch are bound | ESTABLISHED_LOCALLY | Real bundle source digests |
| Runner process ownership | Dedicated worker session/process group, TP4 UUID-to-owned-PID binding, bounded timeout, group-only cleanup | Ownership, unowned-process, timeout, and cleanup tests | 533- and 687-test suites | Owned PID and GPU binding files are authoritative inputs | ESTABLISHED_LOCALLY | Real PID/GPU binding |
| Kerberos TTL fail-fast | Every remote command passes the centralized Kerberos lifetime guard | Kerberos/wrapper tests | 533- and 687-test suites | Preflight records the accepted local status | ESTABLISHED_LOCALLY | Fresh TTL at campaign time |
| Remote execution not run | Runner contract only; no `execute` invocation | N/A | N/A | No new run tag/bundle was created | CONFIRMED | Entire real bundle |
| Runtime optimization remains unauthorized | Diagnostic, result summary, verifier receipt, and runner force the field false | Claim-tamper tests | 533- and 687-test suites | Canonical equality rejects any true value | CONFIRMED_FALSE | Separate approved optimization design after localization |

## Task 7 Re-review

Review scope:

```text
7 files
approximately 1031 changed lines
```

The review initially found two Important defects:

1. the top-level worker could exit while TP children remained in the
   dedicated process group, and cleanup returned without signalling them;
2. SSH cache variables could name directories that did not yet exist,
   allowing library fallback to remote `/tmp`.

Both defects were fixed with regression tests.

Final re-review:

```text
Critical: 0
Important: 0
unresolved P0-P2: 0
```

Remote-only review report:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  reviews/task7-rereview-20260818/report.md
```

## Exact Verification

### Task 7 affected suite

Environment:

```text
pytest 8.4.2
remote Sitian validation root
no GPU campaign
```

Files:

```text
tools/test_autoregressive_draft_command_timeline_diagnostic.py
tools/test_autoregressive_draft_cuda_graph_gate.py
tools/test_autoregressive_draft_performance_gate.py
```

Result:

```text
292 passed in 41.80s
```

The three runner/diagnostic/verifier entry points also passed `py_compile`.

### Task 8 dependency-light suite

The exact eleven files from Task 8 Step 1 ran with pinned pytest 8.4.2.

Result:

```text
533 passed in 51.40s
```

### Task 8 Torch-backed expanded suite

The remote package index did not provide `uv`. A new isolated Python 3.11
venv was therefore created under the Sitian task root with the same pinned
package versions:

```text
pytest 8.4.2
torch 2.7.1+cu126
transformers 4.57.6
```

`CUDA_VISIBLE_DEVICES` was empty, so this validation did not allocate a GPU.

Result:

```text
687 passed, 1 warning in 79.15s
```

The warning is the Transformers 4.57.6 deprecation notice for
`TRANSFORMERS_CACHE`; `HF_HOME` was also set beneath the Sitian task root.

### Source and forbidden-pattern checks

The ten files named by Task 8 Step 3 passed `compileall`.

```text
git diff --check: PASS
forbidden-pattern scan: 0 matches
```

The scan covered:

```text
torch.cuda.synchronize
requires_ack=True
pkill
killall
fuser -k
rm -rf
```

in the two timeline profiler modules and the command-timeline remote runner.

## Live Preflight Follow-up

The first live read-only preflight used the never-before-used tag:

```text
20260818-command-timeline-tp4-b4-q4-r1
```

It passed the source-commit and Kerberos gates and confirmed that neither the
primary nor controller destination existed. It stopped before execution
because only three GPUs were fully idle and process-free.

The occupied GPUs belonged to existing external `server.py`, VLLM, xLLM,
and Manhattan services. No process was paused, signalled, or terminated.

The live attempt also exposed that insufficient idle GPUs produced a Python
traceback instead of a structured environment result. The orchestration
layer now converts strict GPU-classification failures into:

```text
status=INCONCLUSIVE_ENVIRONMENT
gpu_indices=[]
gpu_uuids=[]
available_idle_gpu_count=<observed count>
```

The strict four-idle-GPU classifier and all ownership thresholds remain
unchanged. A new regression test covers this result, and the affected suite
now passes:

```text
292 passed in 41.80s
```

The campaign remains unstarted and the tag remains reusable because no
primary or controller run directory was created.

Post-push live verification confirmed that the CLI now emits the structured
result and exits with status `2`, without a traceback. The result included
the pushed source commit, a READY Kerberos receipt above the 5400-second
minimum, the untouched primary/controller paths, and an empty GPU selection.

## Final Classification

```text
COMMAND_TIMELINE_LOCAL_IMPLEMENTATION=ESTABLISHED
COMMAND_TIMELINE_REMOTE_BUNDLE=NOT_RUN
BOUNDARY_LOCALIZED=NOT_ESTABLISHED
RUNTIME_OPTIMIZATION=NOT_AUTHORIZED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

The next allowed action after commit and push is a separately authorized,
never-before-used source-bound command-timeline bundle. It is not a runtime
optimization and must not reuse or rewrite immutable schema-v2 `r3`.
