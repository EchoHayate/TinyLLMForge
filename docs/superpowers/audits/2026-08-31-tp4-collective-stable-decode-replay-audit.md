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
