# TP4 Collective-Stable Decode Replay Stage-0 Audit

**Date:** 2026-08-31

**Attempt:** `20260831-qwen38-tp4-decode-replay-r1`

**Source revision:** `c66f2dbfe12ba31ed010c6d733b569ae83fc7aa1`

**Source tree SHA-256:**
`e3c212edd89328b923680530156981f20032da6c68d34aa4a4aa6bde0e33a6a8`

**Attempt classification:** `INCOMPLETE`

**Performance classification:** unavailable

**Stage-1 integration:** prohibited

## 1. Executive conclusion

The first Qwen3.8-27B TP4 decode-replay attempt stopped before SSH, GPU
admission, or worker launch because the local Kerberos ticket did not cover
the frozen six-hour remote command window plus its 15-minute guard margin.

The controller persisted:

```text
artifacts/tp4_decode_replay/
  20260831-qwen38-tp4-decode-replay-r1/
    controller/
      source_identity.json
      ssh_storage_preflight.json
```

The preflight receipt records:

- `classification = INCOMPLETE`;
- `reason = Kerberos TTL preflight failed`;
- `minimum_required_lifetime_seconds = 22500`;
- `remaining_lifetime_seconds = -231288`;
- `expires_at = 2026-08-29T01:07:31+08:00`; and
- `attempt_exists = false`.

No remote attempt, GPU worker, raw measurement, immutable bundle, producer
classification, remote verifier, local frozen-source verifier, or performance
result exists for this tag. Therefore no benefit, cost, correctness, replay,
or collective-stability claim is supported.

This result does prove that the controller fails before remote mutation when
the credential window is insufficient and leaves compact local evidence
instead of silently disappearing.

The tag `20260831-qwen38-tp4-decode-replay-r1` is terminally reserved for this
prelaunch `INCOMPLETE` attempt. After credentials are restored, the real gate
must use a fresh tag such as
`20260831-qwen38-tp4-decode-replay-r2`.

## 2. Benefit and cost table

No measured row exists. `Unavailable` means the metric was not observed; it
does not mean zero change.

| Scope | Output throughput ratio | Median TPOT ratio | P99 TPOT ratio | P99 E2E ratio | TTFT ratio | Replay coverage | Capture duration | Amortization tokens | Allocated delta/rank | Reserved delta/rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q0 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| Q1 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| Q2 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| Aggregate | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

The only observed cost is local preflight execution. It is not a model-runtime
cost and must not be included in a future benefit/cost comparison.

## 3. Exact evidence boundary

The current evidence supports only these statements:

1. The source identity was frozen at revision
   `c66f2dbfe12ba31ed010c6d733b569ae83fc7aa1`.
2. The frozen model identity is `Qwen/Qwen3.8-27B` revision
   `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
3. The configured remote root is below the approved mounted
   `/data00/home/sitian/tinyllmforge-workspaces/` boundary.
4. The controller required 22,500 seconds of credential lifetime.
5. The observed credential was expired, so the controller did not query GPU
   readiness or create a remote attempt.
6. The local failure receipt is compact and parseable JSON.

The current evidence does not support:

- exact greedy output parity;
- Q0/Q1/Q2 completion;
- any throughput or latency ratio;
- replay coverage;
- all-rank graph dispatch agreement;
- collective-order agreement;
- capture duration or amortization;
- allocated or reserved memory cost;
- four-GPU strict-clean admission;
- worker or communicator cleanup;
- manifest completeness or hash validity;
- producer/remote/local verifier agreement; or
- Stage-1 justification.

## 4. Local implementation and regression evidence

Fresh local verification on 2026-08-31 produced:

| Command | Result | Coverage boundary |
|---|---|---|
| `python3 tools/test_tp4_decode_replay_contract.py` | `12 passed` | Frozen profile, validation, classification, and thresholds |
| `python3 tools/test_tp4_decode_replay_worker.py` | `7 passed` | Dependency-injected worker behavior; no real GPU/model |
| `python3 tools/test_assemble_tp4_decode_replay.py` | `6 passed` | Synthetic bundle completeness and tamper handling |
| `python3 tools/test_verify_tp4_decode_replay.py` | `6 passed` | Synthetic independent reconstruction and mutations |
| `python3 tools/test_run_tp4_decode_replay.py` | `16 passed` | Controller ordering, paths, GPU readmission, ports, frozen verifier, TTL, and failure receipt |
| `python3 tools/test_model_runner_spec_verify.py` | passed | Existing model-runner source/spec checks after correcting one stale test-list reference |
| Three adjacent TP4 controller tests with `PYTHONPATH=.` | exit zero | Existing controller/supervisor compatibility |
| Gate-file `py_compile` | exit zero | Python syntax/import compilation only |
| Gate-file `git diff --check` | exit zero | Whitespace integrity only |

One planned adjacent regression is not green locally:

| Command | Actual blocker | Interpretation |
|---|---|---|
| `python3 tools/test_multi_sequence_cuda_graph_gate.py` | `ModuleNotFoundError: No module named 'torch'` | Local environment lacks PyTorch; no runtime verdict |

Passing dependency-injected tests do not replace the missing real TP4 run.
The remaining environment failure prevents claiming the complete Task-5
suite is green.

## 5. Source and push evidence

The Stage-0 tooling is split across these pushed commits:

| Commit | Purpose |
|---|---|
| `b49f0d4` | Freeze qualification contract and classifier |
| `e133d2e` | Add canonical TP4 worker |
| `0049f6f` | Add assembler and independent verifier |
| `b965d28` | Add controller and remote orchestration |
| `c66f2db` | Persist failed credential preflight as `INCOMPLETE` |

After the last push, local `HEAD`, the local tracking ref, and the GitHub
branch ref all resolved to:

```text
c66f2dbfe12ba31ed010c6d733b569ae83fc7aa1
```

The latest commit contains exactly one:

```text
Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

## 6. Prompt-to-artifact checklist

| Requirement | Exact artifact / field | Verifier or check | Result | Limitation |
|---|---|---|---|---|
| Work only in authoritative checkout | `/Users/bytedance/dev/TinyLLMForge`; Desktop is a symlink | Repository path inspection | complete | Does not prove remote execution |
| Source-bound run | `controller/source_identity.json`: `source_revision`, `source_tree_sha256` | Source freeze before preflight | complete | No remote source archive exists |
| Frozen model revision | `source_identity.json` and `ssh_storage_preflight.json`: `model_revision` | Exact string comparison | complete | Model directory was not reached over SSH |
| TP4 topology | Contract `RANKS=(0,1,2,3)` and worker/controller tests | Unit tests | implementation complete | No four-rank process receipt |
| Four strict-clean GPUs | Planned controller admission and readmission checks | Controller tests | not observed | Credential failure occurred before GPU query |
| Approved remote root | `ssh_storage_preflight.json`: `remote_root` | Prefix and path-safety tests | configured | Remote path was not created or inspected |
| No remote `/` or `/tmp` task writes | Controller environment/path tests | Unit tests | implementation complete | No real remote command ran |
| No `kinit` or `krenew` | Controller has query-only credential guard | Source inspection and real receipt | complete for r1 | User must restore credentials externally |
| No external GPU process termination | Controller has selection-only monitor | Controller tests | no action occurred | No real GPU inventory |
| Q0/Q1/Q2 frozen matrix | `tp4_decode_replay_contract.py`: `WORKLOADS`; 30-case matrix | Contract tests | implementation complete | Zero measured rows |
| Exact greedy output parity | Planned `correctness_rows.jsonl` | Producer plus both verifiers | missing | No worker launch |
| All-rank dispatch agreement | Planned `rank_dispatch_events.jsonl` | Contract and verifier logic | missing | No rank rows |
| Collective order agreement | Planned `rank_collective_events.jsonl` | Contract and verifier logic | missing | No collective rows |
| Replay coverage at least 0.80 | Planned summary metrics | Producer plus both verifiers | missing | No candidate decode |
| Throughput benefit | Planned `performance_rows.jsonl` and `summary.json` | Frozen threshold reconstruction | missing | No performance claim |
| TPOT/E2E/TTFT cost | Planned `performance_rows.jsonl` | Frozen threshold reconstruction | missing | No latency claim |
| Capture and amortization cost | Planned `capture_cost_rows.jsonl` | Producer plus both verifiers | missing | No capture |
| Allocated/reserved memory cost | Planned `memory_rows.jsonl` | Per-rank threshold checks | missing | No CUDA process |
| Worker and communicator cleanup | Planned process/lifecycle/cleanup receipts | Assembler and both verifiers | missing | No owned remote child existed |
| Unique port per 30 arms | Planned `process_receipts.json` | Assembler and independent verifier | implementation complete | No arm launched |
| Immutable bundle | Planned `final_bundle/manifest.json` | Hash inventory verification | missing | No bundle |
| Producer classification | Planned `producer_classification.json` | Independent reconstruction | missing | Preflight receipt is not a producer result |
| Remote independent verifier | Planned controller artifact | Remote frozen-source verifier | missing | SSH was not entered |
| Local frozen-source verifier | Planned controller artifact | Frozen revision extraction and execution | missing | No bundle to verify |
| Tests | Section 4 | Fresh commands | partial | Two adjacent tests are not green |
| Exact commit discipline | Commits listed in Section 5 | Git metadata | complete for implementation commits | Terminal evidence commit remains pending |
| Push and remote SHA | Section 5 | Local/tracking/remote equality | complete at `c66f2db...` | Future evidence commit will need a new equality check |

## 7. Missing terminal artifacts

All of these are absent for r1:

```text
final_bundle/manifest.json
final_bundle/source_manifest.json
final_bundle/source.patch
final_bundle/environment.json
final_bundle/gpu_inventory.json
final_bundle/workload_profile.json
final_bundle/process_receipts.json
final_bundle/rank_environment.jsonl
final_bundle/rank_dispatch_events.jsonl
final_bundle/rank_collective_events.jsonl
final_bundle/rank_lifecycle_rows.jsonl
final_bundle/request_rows.jsonl
final_bundle/performance_rows.jsonl
final_bundle/memory_rows.jsonl
final_bundle/correctness_rows.jsonl
final_bundle/capture_cost_rows.jsonl
final_bundle/summary.json
final_bundle/producer_classification.json
controller/remote_independent_verification.json
controller/remote_post_verification_manifest.json
controller/local_frozen_source_verification.json
final_bundle/report.md
```

Their absence is expected after preflight rejection, but it means the
optimization remains unqualified.

## 8. Next executable checkpoint

Do not retry r1. After an externally restored credential satisfies the same
22,500-second guard:

1. choose a fresh run tag, beginning with
   `20260831-qwen38-tp4-decode-replay-r2`;
2. freeze the then-current committed source revision;
3. run the local `monitor-and-run` controller;
4. wait for four strict-clean GPUs and launch immediately;
5. require all 30 arm receipts and all four rank evidence streams;
6. run producer, remote independent, and local frozen-source verification;
7. write the measured benefit/cost table; and
8. reconcile the design and plan only from the immutable verified bundle.

Until those steps succeed, the only defensible terminal statement is:

> Attempt r1 is prelaunch `INCOMPLETE` because the credential lifetime was
> insufficient. It contains no performance evidence, and Stage 1 is not
> authorized.

## 9. 2026-09-01 r14 prelaunch resource-window reconciliation

Attempt `20260901-qwen38-tp4-decode-replay-r14` froze the corrected paged
Qwen decode-attention source at:

```text
source revision:
  1f866e3cd736b64377c533ea643f7d0db60c39be
source tree SHA-256:
  32132598b557850fb52415e68967af558d2c25f9b0f3bc112a3d53d60e2135b8
model revision:
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

The local controller completed SSH/storage preflight with:

```text
classification:
  PASS
remote attempt existed:
  false
Kerberos expiration:
  2026-09-01T19:11:44+08:00
required remaining lifetime:
  22,500 seconds
observed remaining lifetime at preflight:
  29,376 seconds
```

The controller then remained in the local strict-clean GPU monitor. It never
created the remote attempt and never launched a worker. Read-only snapshots
showed fewer than four qualifying GPUs throughout the usable credential
window. Representative observations were:

| Time, Asia/Shanghai | Strict-clean GPUs | Relevant occupied GPUs |
|---|---|---|
| `2026-09-01 11:43:18` | `4, 7` | GPU 6: 6,878 MiB |
| `2026-09-01 11:55:06` | `7` | GPU 4: 3,837 MiB; GPU 6: 8,522 MiB |
| `2026-09-01 12:22:17` | none | all eight GPUs above the 1,024 MiB limit |
| `2026-09-01 12:52:27` | `7` | all other GPUs above the limit |
| `2026-09-01 12:57:28` | `7` | all other GPUs above the limit |

With a ticket expiration of `19:11:44` and a frozen six-hour command timeout
plus 15-minute guard margin, the latest valid admission time was
`12:56:44`. The local r14 monitor was interrupted after that deadline because
no future admission under the existing ticket could satisfy the frozen
22,500-second lifetime requirement.

A direct read-only SSH check after interruption confirmed that the planned
remote r14 path was absent:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  tp4-collective-stable-decode-replay/
  20260901-qwen38-tp4-decode-replay-r14
```

Therefore r14 is prelaunch `INCOMPLETE`. It contains only:

```text
controller/source_identity.json
controller/ssh_storage_preflight.json
```

It contains no launch admission, remote attempt, worker output, measured
case, pair, rank, correctness, replay, collective, latency, throughput,
memory, capture-cost, cleanup, bundle, manifest, or independent-verifier
evidence. No performance or Stage-1 claim is permitted, and the r14 tag must
not be reused.

### 9.1 Monitor TTL fail-fast correction

The r14 observation exposed an orchestration inefficiency: initial preflight
and final admission both checked the Kerberos window, but the local GPU
monitor did not recheck it on every poll. It could therefore continue waiting
after a valid launch had become impossible.

Commit `71dd7767a66f184ceb174f99609218bee7c79f69` corrects this by reusing the
same production adapter and invoking its existing full-command-window
Kerberos guard before every remote GPU inventory query.

TDD evidence:

```text
RED:
  tools/test_run_tp4_decode_replay.py failed because
  adapter.kerberos_checks was 0 instead of 2
GREEN:
  tools/test_run_tp4_decode_replay.py: 19 passed
```

Fresh adjacent verification after the correction:

| Command | Result |
|---|---|
| `PYTHONPATH=. python3 tools/test_model_runner_spec_verify.py` | passed |
| `PYTHONPATH=. python3 tools/test_tp4_decode_replay_worker.py` | `8 passed` |
| `python3 tools/test_run_tp4_decode_replay.py` | `19 passed` |
| `PYTHONPATH=. python3 tools/test_tp4_decode_replay_contract.py` | `12 passed` |
| `PYTHONPATH=. python3 tools/test_assemble_tp4_decode_replay.py` | `6 passed` |
| `PYTHONPATH=. python3 tools/test_verify_tp4_decode_replay.py` | `6 passed` |
| `PYTHONPATH=. python3 -m pytest -q tools/test_qwen35_mtp_cuda_graph_backend.py` | `21 passed` |
| `python3 -m py_compile tools/run_tp4_decode_replay.py tools/test_run_tp4_decode_replay.py` | passed |
| `git diff --check -- tools/run_tp4_decode_replay.py tools/test_run_tp4_decode_replay.py` | passed |

Local `HEAD`, the local tracking ref, and the GitHub branch ref were all
verified at:

```text
71dd7767a66f184ceb174f99609218bee7c79f69
```

### 9.2 Updated prompt-to-artifact checklist

| Requirement | r14 evidence | Result | Remaining action |
|---|---|---|---|
| Correct source frozen | `controller/source_identity.json` | complete | Future run must freeze `71dd776...` or a later committed source |
| Approved remote storage | `controller/ssh_storage_preflight.json` | complete | None |
| Sufficient ticket at initial preflight | same receipt: 29,376 s remaining | complete | Ticket no longer covers a new six-hour run |
| Four strict-clean GPUs | repeated local monitor snapshots | missing | Wait for four GPUs at or below 1,024 MiB, at or below 5%, with no compute process |
| Remote attempt creation | direct SSH path check: absent | not performed | Use a fresh tag after credential restoration |
| 30 cases / 15 pairs | no raw rows | missing | Run full frozen matrix |
| Four-rank agreement | no rank rows | missing | Run full frozen matrix |
| Exact output parity | no correctness rows | missing | Run full frozen matrix |
| Replay coverage | no dispatch rows | missing | Run full frozen matrix |
| Throughput and latency benefit | no performance rows | missing | Run full frozen matrix |
| Memory and capture cost | no memory/capture rows | missing | Run full frozen matrix |
| Clean teardown | no worker existed; no terminal cleanup receipt | unverified by terminal receipt | Future run must produce `cleanup.json` |
| Immutable manifest | no bundle | missing | Assemble only after a complete worker run |
| Remote verifier | no bundle | missing | Run from future frozen source |
| Local frozen-source verifier | no bundle | missing | Run from future frozen source |
| Monitor does not outlive usable ticket | commit `71dd776...`; 19 controller tests | complete in code | Exercise in next production monitor |

The next executable checkpoint is:

1. restore a Kerberos ticket externally; do not run `kinit` or `krenew` from
   the agent;
2. require at least 22,500 seconds of remaining lifetime;
3. start a new controller with a fresh tag, beginning with r15;
4. let the TTL-aware local monitor wait for four strict-clean GPUs and launch
   immediately; and
5. complete the unchanged 30-case gate and both independent verifiers.

## 10. r15-r20 diagnostic reconciliation and shutdown root cause

Attempts r15-r20 are consumed and must not be reused. They narrowed the
remaining failure from a generic post-capture hang to one exact lifecycle
boundary.

| Attempt | Frozen source | Outcome | Evidence boundary |
|---|---|---|---|
| `20260901-qwen38-tp4-decode-replay-r15` | `c774dbb3e2a4628960d6ae1fff49b2632fa5dc22` | worker exit `250`; cleanup `DIRTY` | no complete 30-case bundle |
| `20260901-qwen38-tp4-decode-replay-r16-capture-diagnostic` | `0c4ee73365b49cde667754ee06d8a9101f0ccbbf` | capture-phase diagnostic; worker exit `250`; cleanup `DIRTY` | no terminal performance classification |
| `20260901-qwen38-tp4-decode-replay-r17-replay-diagnostic` | `18646960b6f424d77166ef3390e287e37e61bb43` | transport failed with exit `255`; cleanup found no exact-tag child | pre-terminal `INCOMPLETE` |
| `20260901-qwen38-tp4-decode-replay-r18-replay-diagnostic` | `18646960b6f424d77166ef3390e287e37e61bb43` | worker SSH returned `255`; no replay receipt | pre-worker `INCOMPLETE` |
| `20260901-qwen38-tp4-decode-replay-r19-replay-diagnostic` | `18646960b6f424d77166ef3390e287e37e61bb43` | strict-clean admission SSH returned `255` | pre-worker `INCOMPLETE`; cleanup `CLEAN` |
| `20260901-qwen38-tp4-decode-replay-r20-replay-diagnostic` | `18646960b6f424d77166ef3390e287e37e61bb43` | reproduced the post-inference shutdown hang | diagnostic evidence only; no performance verdict |

The stable retry path used an LF jump-host ControlMaster at:

```text
/tmp/ssh-sitian-10.232.195.203-lf
```

r20 admitted strict-clean GPUs `0,3,6,7`. Its Q0 eager case completed and
was atomically written. The graph arm then produced complete capture and
replay phase receipts on all four ranks. Every rank reached:

```text
capture:
  scratch_restore_completed
replay:
  entered_replay
  static_inputs_copied
  context_set
  graph_replay_returned
  logits_compute_returned
  context_reset_completed
last replay ordinal:
  124
```

This evidence rules out a first-replay hang and rules out the sampler
`.tolist()` boundary. Inference completed, GPU utilization fell to zero, and
the graph case still did not reach its atomic case write.

At `2026-09-01 16:24:32 +08:00`, all four ranks reported:

```text
Future for ProcessGroup abort timed out after 600000 ms
```

A rank-0 debugger backtrace placed the blocked thread in:

```text
c10d::ProcessGroupNCCL::waitForFutureOrTimeout
  -> c10d::ProcessGroupNCCL::shutdown
```

At `2026-09-01 16:32:59 +08:00`, all ranks reported a missing NCCL watchdog
heartbeat. At `2026-09-01 16:42:59 +08:00`, all ranks terminated through the
NCCL fatal watchdog path. The remote wrapper returned `-6`; the controller
recorded worker return code `250`.

The local r20 terminal receipts are:

```text
artifacts/tp4_decode_replay/
  20260901-qwen38-tp4-decode-replay-r20-replay-diagnostic/
    controller/worker_wait.json
    controller/cleanup.json
```

`cleanup.json` is `DIRTY` because no rank completed
`dist.destroy_process_group()`, but all three exact-tag scans are empty and
`owned_children_remaining` is empty. A direct read-only process scan after
the fatal exit also found no r20 tag process.

### 10.1 Single root-cause hypothesis

The exact CUDA Graph cache retained graph objects containing NCCL collective
work until after `ModelRunner.exit()` entered
`dist.destroy_process_group()`. The process group therefore tried to shut
down while graph-owned collective resources were still live.

This ordering differs from the repository's successful TP4 NCCL diagnostic,
which explicitly performs:

```text
graph.reset()
torch.cuda.synchronize()
dist.destroy_process_group()
```

The bounded correction is therefore to release all ready exact graphs before
the shared-memory barrier and process-group destruction. It does not change
the frozen workload, graph eligibility, output path, thresholds, or default
feature state.

### 10.2 TDD correction and pushed source

The regression tests were written first:

```text
test_exact_graph_cache_release_resets_graphs_before_synchronize
test_model_runner_exit_releases_exact_graphs_before_process_group_shutdown
```

RED:

```text
2 failed
```

The minimal implementation adds
`ExactCudaGraphCache.release_ready_graphs(*, synchronize)` and invokes it
from `ModelRunner.exit()` before the shared-memory barrier and
`dist.destroy_process_group()`. The method:

1. requires a callable synchronizer;
2. calls `reset()` on every ready graph;
3. clears each entry's graph reference;
4. clears the ready cache;
5. zeroes tracked static and retained-reserved bytes; and
6. synchronizes once after releasing non-empty graph inventory.

GREEN and adjacent checks:

| Command | Result |
|---|---|
| Two new targeted regression tests | `2 passed` |
| `PYTHONPATH=. python3 tools/test_model_runner_spec_verify.py` | passed |
| `PYTHONPATH=. python3 tools/test_tp4_decode_replay_contract.py` | `12 passed` |
| `PYTHONPATH=. python3 tools/test_tp4_decode_replay_worker.py` | `8 passed` |
| `python3 tools/test_run_tp4_decode_replay.py` | `19 passed` |
| `PYTHONPATH=. python3 tools/test_assemble_tp4_decode_replay.py` | `6 passed` |
| `PYTHONPATH=. python3 tools/test_verify_tp4_decode_replay.py` | `6 passed` |
| focused `python3 -m py_compile` | passed |
| focused `git diff --check` | passed |

The correction is committed and pushed:

```text
commit:
  1aa6afdbbc98084148a868adb9ce1cf44e9a39ab
subject:
  fix(tp4): release exact graphs before NCCL shutdown
local/tracking/GitHub SHA:
  1aa6afdbbc98084148a868adb9ce1cf44e9a39ab
```

The commit contains exactly one:

```text
Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

### 10.3 Remaining proof obligation

The unit-level ordering correction and the r20 failure localization do not
prove the real TP4 lifecycle is fixed. A fresh source-bound run must still:

1. execute all 30 cases / 15 pairs at `1aa6afd...`;
2. atomically write every eager and graph case;
3. show all four ranks completing process-group destruction;
4. pass exact output and collective-order checks;
5. quantify throughput, latency, capture, memory, startup, and teardown;
6. assemble the immutable manifest-bound bundle;
7. pass the remote independent verifier;
8. pass the local frozen-source verifier; and
9. produce a clean post-verification cleanup receipt.

Until that fresh run finishes, the correction is implementation-complete but
hardware-GREEN remains unproven. Stage 1 remains prohibited.

## 11. r21 terminal reconciliation

Attempt
`20260901-qwen38-tp4-decode-replay-r21-shutdown-release` is consumed and is
classified `INCOMPLETE`. It used:

```text
source revision:
  1aa6afdbbc98084148a868adb9ce1cf44e9a39ab
source tree SHA-256:
  599d3b86eebba1d8e88615c215edf6deb59e9227861171fcdf09cfe1e9095994
model revision:
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
selected GPUs:
  0,3,4,6
```

The run atomically wrote all ten Q0 cases and the Q1 repetition-zero eager
case. It did not write the Q1 repetition-zero graph case or any later case, so
the required inventory is `11/30` rather than `30/30`.

### 11.1 Shutdown correction hardware result

Every completed Q0 eager and graph arm reported all four rank exit codes as
zero and `process_group_destroyed=true`. This is real TP4 evidence that
releasing ready exact graphs before NCCL shutdown corrected the r20 teardown
hang for those ten completed arms.

It is not terminal lifecycle proof for the whole matrix. The Q1 graph arm
failed before normal cleanup, and the controller therefore emitted:

```text
worker return code:
  1
cleanup classification:
  DIRTY
exact-tag scans:
  empty on all three scans
owned children remaining:
  none
```

### 11.2 Stable Q0 correctness failure

All five completed Q0 pairs reproduced the same one-request divergence:

```text
request index:
  1
first differing output token index:
  37
eager token:
  198
graph token:
  317
other requests:
  exact
```

The graph/eager output-throughput ratios were:

```text
r0  3.652732497742768
r1  3.7181949316837013
r2  3.7454432451739432
r3  3.8083740795583356
r4  3.6887218373133366
```

These ratios are diagnostic only. Exact-token correctness failed in every
completed pair, so none of them is a publishable performance win.

GPU-level attention isolation found:

1. the old eager true-context-width path and graph-compatible fixed-page-width
   path differed in `2/6144` BF16 output elements, with maximum absolute
   difference `0.00048828125`;
2. direct execution and CUDA Graph replay of the same graph-compatible
   function were bit exact; and
3. FP32 value accumulation was exact in only `15/40` sampled
   seed/context combinations and was rejected.

This localizes the Q0 token divergence to different attention reduction shapes
and orders, not to CUDA Graph replay itself or transactional KV/state restore.

### 11.3 Q1 allocator failure

After the five Q0 pairs, the Q1 repetition-zero eager arm completed. During the
following Q1 graph arm, multiple ranks failed at `torch.cuda.graph(...).__enter__`
with:

```text
RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at
../c10/cuda/CUDACachingAllocator.cpp:1839
```

Capture rejection then differed across ranks, and the worker terminated on:

```text
RuntimeError: graph observations disagree across ranks
```

The current single hypothesis is cross-arm CUDA allocator/graph-pool lifetime
contamination: the remote driver creates and destroys successive engines in
one long-lived rank-zero CUDA process. Five Q0 graph arms completed before the
first different batch shape attempted Q1 capture. This hypothesis still
requires a fresh-process Q1 graph diagnostic before any controller change is
accepted.

### 11.4 Attention parity repair checkpoint

A regression fixture using the real Qwen3.8 TP4 local attention dimensions
was added to:

```text
tools/test_qwen35_cached_prefill_eager_attention.py::
  test_cached_decode_graph_captures_paged_cache_on_cuda
```

Its hardware RED was:

```text
mismatched elements:
  1 / 6144
maximum absolute difference:
  0.00048828125
mismatch index:
  (1, 862)
```

The bounded candidate makes eager and graph decode call one shared,
graph-capturable fixed-page-width per-request attention helper. It does not
read `context_lens` through `.item()` and therefore keeps dynamic validity in
GPU tensor operations.

Fresh A100 evidence on strict-clean GPU 4:

```text
targeted exact parity:
  1 passed in 2.49s
complete attention test file:
  15 passed in 2.31s
```

The complete file includes both existing official CUDA reduction fixtures.
This is attention-layer GREEN only. Before this candidate can enter a complete
gate, a source-isolated short TP4 diagnostic must prove:

1. patched eager equals patched graph;
2. patched eager preserves the frozen r21 eager token stream; and
3. a fresh-process Q1 graph arm does not reproduce the allocator assertion.

## 12. r22 external-preemption reconciliation

Attempt
`20260901-qwen38-tp4-decode-replay-r22-attention-short` is consumed and is
classified `INCOMPLETE`. It copied the patched attention source into an
isolated remote source directory and admitted strict-clean GPUs `0,3,6,7`.
The first arm, `Q0__r0__eager`, began normally.

While that arm was still loading the model, unrelated processes appeared on
physical GPUs 3 and 7. Two of the four task ranks then failed in
`ModelRunner.allocate_kv_cache()` at:

```text
assert auto_num_blocks > 0
```

The other two ranks remained blocked in distributed work. No case JSON was
written, so r22 proves neither attention parity nor performance. In
particular, it is not evidence that the attention repair failed and it is not
a reproduction of r21's CUDA allocator assertion.

Ownership was established before cleanup:

```text
tag:
  20260901-qwen38-tp4-decode-replay-r22-attention-short
selected GPUs:
  0,3,6,7
remaining owned rank:
  PID 2376943
ownership evidence:
  PYTHONPATH contained the exact r22 source path
  TINYVLLM_DIST_PORT=37491
```

Only exact r22-owned processes were terminated. A final GPU check showed that
the task's allocations on GPUs 0 and 6 were released; unrelated processes on
the other GPUs were not touched.

The replacement short diagnostic adds two operational safeguards without
changing the model, workload, candidate, or correctness criteria:

1. require the same four GPUs to remain strict-clean for four consecutive
   admission samples; and
2. while an arm runs, classify every compute PID on the selected GPUs by the
   exact attempt tag. If an unrelated PID appears, terminate only tagged task
   processes and restart the complete three-arm diagnostic under a fresh
   attempt tag.

This runtime watchdog is an experiment-orchestration guard. It is not part of
the candidate optimization and must not be counted as performance benefit.

## 13. r24 allocator-lifetime and attention-semantics reconciliation

The r24 diagnostics used isolated Python processes per arm and an exact-tag
watchdog. They separate the allocator-lifetime correction from the rejected
attention-semantic change.

### 13.1 Fresh-process Q1 allocator result

Attempt
`20260901-qwen38-tp4-decode-replay-r24-pool-lifetime-q1-a001` ran the Q1
graph arm on GPUs `0,3,4,6`. All four ranks exited zero, all four cleanup
receipts reported `process_group_destroyed=true`, and no owned child remained.
The r21 allocator assertion did not recur.

Each rank completed a capture attempt:

```text
rank 0: 3,366,468,004 ns; reserved delta           0 bytes
rank 1: 3,385,160,884 ns; reserved delta 469,762,048 bytes
rank 2: 3,361,706,084 ns; reserved delta 469,762,048 bytes
rank 3: 3,382,762,410 ns; reserved delta 469,762,048 bytes
```

All four attempts exceeded the frozen two-second single-capture ceiling.
Consequently the cache rejected the entry and all `1,016` measured decode
rank-steps dispatched eager. This proves that the stale-pool allocator failure
was fixed, but Q1 still has zero replay coverage under the unchanged contract.

The accepted implementation gives exact multi-sequence graphs their own pool,
publishes a pool handle only after a ready graph owns it, and clears the handle
after releasing ready graphs during engine exit. It also isolates every gate
arm in a fresh Python process, freezes `PYTHONPATH`, and scans both
`/proc/<pid>/cmdline` and `/proc/<pid>/environ` for exact-tag ownership.

The correction is committed and pushed:

```text
commit:
  1e18c30e5cf134943b39f984100583b2b1a3f55d
subject:
  fix(tp4): isolate decode replay graph lifetimes
local/tracking/GitHub SHA:
  1e18c30e5cf134943b39f984100583b2b1a3f55d
```

### 13.2 Fixed-width eager candidate rejection

Attempt
`20260901-qwen38-tp4-decode-replay-r24-pool-lifetime-q1-a002` completed Q0
eager and graph arms on GPUs `0,3,6,7`. Both arms had four zero rank exits and
four successful process-group destruction receipts. The graph arm recorded
`992` graph dispatch rows and `32` eager dispatch rows, and its outputs exactly
matched the same-source eager arm.

That apparent correctness was invalid as a preservation result: the candidate
had changed eager attention to the same fixed-page-width reduction used by the
graph. Its Q0 eager output differed from the frozen r21 eager authority in
request zero beginning at output index three. Attempt
`20260901-qwen38-tp4-decode-replay-r24-pool-lifetime-q1-a003` repeated the
candidate eager arm on the original r21 GPU topology `0,3,4,6` and produced the
same changed output. The drift therefore came from the eager attention change,
not GPU topology.

The candidate was reverted. Frozen r24 sources remain immutable and are not
used as the source of any later gate.

### 13.3 Graph-only per-request reduction rejection

A follow-up candidate preserved the old eager implementation and changed only
the graph path from one vectorized fixed-width BF16 value matmul to one
fixed-width BF16 matmul per request.

The first exploratory sample had zero mismatches, but a deterministic
Qwen3.8 TP4 CUDA Graph replay fixture with:

```text
batch size:       4
local query heads: 6
local KV heads:    1
head dimension:  256
context length:  295 after replay update
fixed graph width: 512
```

failed bit-exact comparison against the unchanged eager path:

```text
mismatched elements:          2 / 6144
maximum absolute difference:  0.0001220703125
```

The graph-only candidate was therefore also reverted before any model-level
gate. Per-request grouping is not sufficient: fixed-width graph execution
still reduces over a different K dimension from true-context-width eager
execution. No tolerance was relaxed, and no favorable random sample is treated
as correctness proof.

## 14. r25 full-gate launch checkpoint

The complete replacement gate is:

```text
tag:
  20260901-qwen38-tp4-decode-replay-r25-full
source revision:
  1e18c30e5cf134943b39f984100583b2b1a3f55d
source tree SHA-256:
  78ed59e9782c2a6e22d643faa6475730758daf3bf7a69e94c0c89be09fc8e020
model revision:
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
selected GPUs:
  0,3,4,6
```

The local controller admitted four strict-clean GPUs and launched from the
frozen pushed revision. The full `30`-case / `15`-pair matrix remains required
even if early pairs reproduce the known exact-token failure. Final
classification, compact bundle, both independent verifier receipts, manifest,
and terminal cleanup evidence remain pending while the worker runs.

## 15. r25 external-preemption terminal reconciliation

Attempt
`20260901-qwen38-tp4-decode-replay-r25-full` is consumed and is terminally
`INCOMPLETE`. It used the pushed source and admitted topology recorded in
Section 14.

At `2026-09-01T22:13:05.133965+08:00`, the local GPU guard observed external
compute PID `3877390` on one of the selected GPUs while the current r25 arm
owned PIDs `3874042`, `3874509`, `3874510`, and `3874511`. The guard followed
the frozen ownership boundary:

- it did not signal PID `3877390`;
- it terminated only processes whose environment contained the exact r25
  tag; and
- its cleanup command returned zero with an empty final owned-process set at
  `2026-09-01T22:13:11.808382+08:00`.

The worker wrapper consequently returned `-15`; the controller recorded
return code `241`. The durable local receipt is:

```text
artifacts/tp4_decode_replay/
  20260901-qwen38-tp4-decode-replay-r25-full/
    controller/
      external_preemption.json
      worker_wait.json
      cleanup.json
```

The remote attempt contains `13/30` atomic case files:

```text
Q0: all 10 cases / all 5 pairs
Q1: r0 eager and graph; r1 graph only
Q2: no cases
```

This is only `6/15` complete pairs. No final bundle, producer
classification, remote independent verification, post-verification manifest,
local frozen-source verification, or report exists. The missing matrix takes
precedence over every diagnostic result.

### 15.1 Partial diagnostic evidence

All 13 completed arms have four zero rank exits and four
`process_group_destroyed=true` receipts. The five complete Q0 pairs and the
complete Q1 r0 pair show:

| Pair | Exact tokens | Graph/eager output throughput | Measured replay rank-steps | Eligible rank-steps | Diagnostic |
|---|---|---:|---:|---:|---|
| Q0 r0 | no | `3.9194675320` | 496 | 508 | request 1 diverges at output index 37: `198 -> 317` |
| Q0 r1 | no | `3.8722946982` | 496 | 508 | same divergence |
| Q0 r2 | no | `3.7891169171` | 496 | 508 | same divergence |
| Q0 r3 | yes | `1.0109628012` | 0 | 508 | measured capture rejected by the 2 s ceiling |
| Q0 r4 | no | `3.7165901203` | 496 | 508 | same divergence |
| Q1 r0 | yes | `0.9923496872` | 0 | 508 | measured capture rejected by the 2 s ceiling |

The large Q0 ratios are diagnostic only because exact-token correctness
failed. The Q0 r3 measured capture durations were approximately
`3.09–3.25 s` across ranks. The Q1 r0 measured capture durations were
approximately `3.07–3.16 s` across ranks. Both exceed the frozen
`2,000,000,000 ns` ceiling and therefore correctly fall back to eager.

The Q0 partial replay coverage is `1984 / 2540 = 0.781102...`, already below
the frozen `0.80` mechanism threshold. This cannot be promoted to a terminal
mechanism verdict because Q1 and Q2 are incomplete.

### 15.2 Cleanup receipt false-positive and correction

The controller's original `cleanup.json` says `DIRTY`, but its reported
remaining PIDs are the concurrent cleanup scanner's own shell and Python
processes. Their command lines contained the raw run tag; neither represented
a surviving worker rank. The guard's independent environment-tag cleanup
reported an empty final set, and a later read-only scan using the corrected
ownership rule also returned `[]`.

The root cause was that `_scan_exact_tag()` accepted any command line
containing the short run tag. A concurrent guard cleanup command therefore
looked owned. The corrected rule is:

```text
owned when:
  full attempt_root appears in cmdline
  OR exact run_tag appears in process environment
```

TDD evidence:

```text
RED:
  test_exact_tag_scan_uses_attempt_root_for_cmdline_ownership failed because
  the generated scanner still used `tag in command`
GREEN:
  focused pytest: 1 passed
  complete controller script suite: 20 passed
  post-fix live r25 exact-tag scan: []
```

This correction changes only ownership detection. It does not relabel r25 as
cleanly completed; the rank processes were intentionally interrupted by the
external-preemption guard, so r25 remains `INCOMPLETE`.

### 15.3 Measured capture-cost reporting correction

r25 also exposed that `run_arm()` populated `capture_cost_rows` from warmup
dispatch rows. Lease-sealed execution may capture a different identity in the
measured phase, so the old row could under-report the actual measured capture
and conceal a `single_capture_budget` rejection.

The minimal correction now derives each case's capture-cost rows from measured
dispatch evidence. TDD evidence:

```text
RED:
  expected measured capture_duration_ns=20,000,000
  observed warmup capture_duration_ns=10,000,000
GREEN:
  focused pytest: 1 passed
  complete worker script suite: 8 passed
```

The raw r25 dispatch rows retain the true measured capture durations quoted
above, but its generated `capture_cost_rows` were produced by the old frozen
source. They must not be treated as complete measured-cost evidence.

### 15.4 Updated prompt-to-artifact checklist

| Requirement | r25 evidence | Result | Remaining action |
|---|---|---|---|
| Frozen pushed source | `controller/source_identity.json`; `1e18c30...` | complete | Next run must freeze the new correction commit |
| Approved remote storage | `controller/ssh_storage_preflight.json` | complete | None |
| Four strict-clean GPUs at admission | `controller/strict_clean_admission.json`; GPUs `0,3,4,6` | complete | Next run must re-admit |
| Continuous local GPU ownership guard | `controller/external_preemption.json` | complete and exercised | Reuse with a fresh tag |
| Do not terminate external work | foreign PID excluded from guard-owned cleanup | complete | None |
| Complete 30-case / 15-pair matrix | remote `raw/cases/`: 13 cases / 6 pairs | missing | Fresh full run |
| Exact-token correctness | five Q0 mismatches; Q1 r0 exact | failed diagnostically | Full matrix still required |
| Replay coverage | Q0 partial `0.781102...`; Q1 r0 zero | below gate diagnostically | Full matrix still required |
| Benefit plus cost | partial ratios and raw measured capture rows | incomplete | Fresh run with corrected capture-cost source |
| Four-rank lifecycle | complete for 13 written cases | partial | Full matrix still required |
| Final bundle and manifest | absent | missing | Fresh full run |
| Remote verifier | absent | missing | Fresh full run |
| Local frozen-source verifier | absent | missing | Fresh full run |
| No owned process remains | guard final set `[]`; corrected live scan `[]` | complete | None |
| Stage-1 authorization | no terminal verified classification | prohibited | Only a future verified `GO` may authorize |

### 15.5 Next executable checkpoint

Do not reuse r25. Commit and push the ownership-scan and measured-capture-cost
corrections, then use fresh tag
`20260901-qwen38-tp4-decode-replay-r26-full` with the default six-hour worker
window. Before launch, the query-only Kerberos preflight must observe at least
`22,500` seconds of remaining ticket lifetime. The agent must not run
`kinit` or `krenew`.

## 16. r26-r29 shared-host interruption reconciliation

The corrected source was committed and pushed as
`80b2c008de118bb4645d720197edf0e3f4c2546a`, with frozen source-tree SHA-256
`7bcc22a482869a0423f8e2fd9686ce4d50d3b7a7f275d0a76618f1b508ba325f`.
Three fresh attempts then passed storage, source-identity, Kerberos, and
strict-clean TP4 admission on GPUs `0,3,4,6`, but each was preempted by a new
external compute process on an admitted GPU:

| Attempt | External PID | Completed cases | Completed pairs | Final bundle | Terminal classification |
|---|---:|---:|---:|---|---|
| r26 | `4051684` | 0/30 | 0/15 | absent | `INCOMPLETE_EXTERNAL_PREEMPTION` |
| r27 | `4146190` | 4/30 | 2/15 | absent | `INCOMPLETE_EXTERNAL_PREEMPTION` |
| r28 | `4161055` | 0/30 | 0/15 | absent | `INCOMPLETE_EXTERNAL_PREEMPTION` |

r27 wrote exactly `Q0__r0__eager`, `Q0__r0__graph`,
`Q0__r1__eager`, and `Q0__r1__graph` before interruption. The large raw case
files remain on the approved remote `/data00/home/sitian/...` volume; only
compact controller receipts are retained locally.

For every attempt, the local guard excluded the foreign PID from cleanup and
terminated only processes carrying the exact task tag. The guard reported
`returncode=0`, `remote_worker_returncode=-15`, controller worker-wait
`returncode=241`, and an empty final owned-environment set. A fresh read-only
scan on 2026-09-02 found no exact-tag-owned PID for r26, r27, or r28. r28's
controller cleanup receipt briefly observed its own tagged Python resource
tracker on the first scan; the next two scans and the later live scan were
empty.

The compact evidence is:

```text
artifacts/tp4_decode_replay/<r26|r27|r28>/controller/
  source_identity.json
  ssh_storage_preflight.json
  strict_clean_admission.json
  worker_wait.json
  cleanup.json
  external_preemption.json
```

### 16.1 r29 stable-window monitor

r29 did not create a local or remote experiment attempt. Its local supervisor
required 80 consecutive 15-second samples, or 20 minutes, with the same four
strict-clean GPUs before launch. Candidate sets repeatedly reached 40/80 and
twice reached 70/80 before shared-host activity reset the counter.

The monitor then exited because one `ssh ... nvidia-smi` sample exceeded its
30-second timeout. This was an orchestration robustness defect, not an
experiment result. The local supervisor now treats a telemetry exception as a
recoverable failed sample: it emits `prelaunch_gpu_query_error`, clears the
stable GPU identity and count, sleeps for one poll interval, and continues.
TDD evidence:

```text
RED:
  TimeoutExpired escaped wait_for_stable_gpu_window
GREEN:
  4 supervisor tests passed
  r29_supervisor.py and test_r29_supervisor.py py_compile passed
```

At the latest 2026-09-02 check, r29 remained absent locally and remotely.
The local supervisor was restarted, but the ticket had only `18,719` seconds
remaining versus the frozen `22,500`-second requirement, so it correctly
remained in `BLOCKED_KERBEROS_TTL` without creating an attempt.

### 16.2 Current prompt-to-artifact checklist

| Requirement | Current evidence | Result | Remaining action |
|---|---|---|---|
| Frozen pushed source | r26-r28 `source_identity.json`; commit `80b2c008...` | complete | Fresh attempt must use the same clean source |
| Approved remote storage | r26-r28 `ssh_storage_preflight.json` | complete | Recheck for fresh attempt |
| Four strict-clean GPUs at admission | r26-r28 `strict_clean_admission.json` | complete at admission only | Require 20-minute stable prelaunch window |
| Continuous local GPU ownership guard | r26-r28 `external_preemption.json` | complete and exercised | Keep enabled |
| Do not terminate external work | all three foreign PIDs excluded from owned cleanup | complete | Keep invariant |
| Complete 30-case / 15-pair matrix | best corrected-source attempt r27: 4 cases / 2 pairs | missing | Fresh full run |
| Exact-token correctness | no complete corrected-source matrix | missing | Fresh full run |
| Replay coverage | no complete corrected-source matrix | missing | Fresh full run |
| Benefit plus cost | no complete corrected-source matrix | missing | Fresh full run |
| Four-rank lifecycle | interrupted in r26-r28 | incomplete | Fresh full run |
| Final bundle and manifest | absent in r26-r28 | missing | Fresh full run |
| Remote verifier | absent in r26-r28 | missing | Fresh full run |
| Local frozen-source verifier | absent in r26-r28 | missing | Fresh full run |
| No owned process remains | guard final sets and fresh live scans are empty | complete | Recheck after fresh attempt |
| Stage-1 authorization | no terminal verified Stage-0 result | prohibited | Only verified `GO_STAGE1_JUSTIFIED` may authorize |

The next valid launch requires an externally refreshed Kerberos ticket with at
least `22,500` seconds remaining and one uninterrupted 20-minute strict-clean
four-GPU window. The supervisor must continue waiting locally and must not run
`kinit`, `krenew`, or terminate external GPU work.
