# Qwen3.8-27B TP4 synchronous collective-reduction audit

Date: 2026-08-29

## Objective and claim boundary

This audit qualifies a default-disabled, synchronous-only collective
reduction candidate for the Qwen3.8-27B text stage at tensor parallel size
four. It does not authorize asynchronous collectives, a communication stream,
event-based overlap, chunked ReduceScatter/AllGather, or a production default.

The gate must establish all of the following from one immutable attempt:

- exact source, model, GPU, workload, and 130-site collective identities;
- matched calibration at event budgets 0, 8, 16, and 32;
- exact output equality and four-rank census agreement;
- a producer classification reconstructed by independent remote and local
  verifiers;
- a complete post-verification manifest;
- process-group destruction, no owned children, and three empty exact-tag
  scans; and
- both candidate benefit and measured cost, without converting a
  qualification result into a speedup claim.

## Immutable identities

```text
authoritative checkout:
  /Users/bytedance/Desktop/TinyLLMForge
branch:
  feat/kv-sparse-attention
remote:
  https://github.com/EchoHayate/TinyLLMForge.git
model:
  Qwen/Qwen3.8-27B
model revision:
  1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
approved remote root:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
overlap design authorized:
  false
async collectives authorized:
  false
```

## Attempt lineage

### r6: rejected producer evidence

Attempt `20260828-qwen38-tp4-collective-reduction-r6` used source
`fc01b2d68e91797178439d743b0a3df868b20fe3`. Its worker and supervisor
completed naturally:

```text
worker return code:              0
case artifacts:                  119
process group destroyed:         true
owned children remaining:        []
exact-tag scans:                 [[], [], []]
supervisor stderr bytes:         0
```

The frozen assembler rejected the case inventory with
`ValueError: case identity is invalid`. The worker emitted
`workload_family` but omitted the top-level `prompt_tokens`,
`output_tokens`, and `concurrency` fields required by the assembler's strict
case-identity contract. No bundle or verifier result was produced. The r6
case files were not rewritten or reused.

The defect was reproduced by a focused RED test and fixed by commit
`c4dc02adc27f43af8b27a648b0d89a77af5c0343`, which makes every worker case
self-describing while retaining the assembler's strict checks. Verification
for that repair was:

```text
focused RED:                     KeyError: prompt_tokens
focused GREEN:                   1 passed
worker plus assembler suites:    25 passed
collective-reduction suite:      112 passed
py_compile:                      PASS
git diff --check:                PASS
```

Commit `bd7a064` separately prevents a missing local cleanup file from
masking an earlier postprocess exception. That controller-only repair passed
113 collective-reduction tests and does not change the frozen runtime source
of the next attempt.

### r7: rejected profiler-overhead terminal handling

```text
attempt:
  20260828-qwen38-tp4-collective-reduction-r7
frozen runtime source:
  c4dc02adc27f43af8b27a648b0d89a77af5c0343
frozen source-tree SHA-256:
  648084abbe5c72a43b3e636d17eae379829e8142181d8dd0350656b13b7a147d
controller source at launch:
  c4dc02adc27f43af8b27a648b0d89a77af5c0343
later local harness-only repair:
  bd7a064
selected GPUs:
  rank 0 -> physical 1 -> GPU-7dc22583-df04-6c76-4ba5-ea32c428c130
  rank 1 -> physical 2 -> GPU-63c05907-407b-8240-07a0-f38872840867
  rank 2 -> physical 3 -> GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d
  rank 3 -> physical 4 -> GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1
launch admission:
  3 MiB, 0% utilization, no compute process on every selected GPU
```

The worker completed all 84 calibration cases, then raised:

```text
ValueError: count-only profiler overhead exceeds limits
```

This was a controller/worker contract defect. A budget-zero overhead failure
is a valid `INCONCLUSIVE_PROFILER_OVERHEAD` outcome, not an exceptional
worker failure. The supervisor also recorded two generic
`foreign GPU process detected` violations, but the frozen receipt did not
identify the GPU UUID, PID, or process name, so that evidence cannot
distinguish a real external process from an ownership-observation gap.

The attempt remained immutable and failed closed:

```text
case artifacts:                  84
worker return code:              1
supervisor classification:       FAIL
resource snapshots:              8,022
process group destroyed:         true
owned children remaining:        []
exact-tag scans:                 [[], [], []]
```

Commit `db96454` changed the pure selector to return `selected_budget=null`
when the budget-zero control itself exceeds the frozen overhead limits. The
full worker now emits a successful worker receipt and skips the unnecessary
35 terminal cases for that valid terminal outcome. Commit `c7a552e` also
made the supervisor retain every PID previously confirmed in the worker
process group for the full attempt lifetime.

### r8: worker terminal fixed; resource identity still unproven

```text
attempt:
  20260828-qwen38-tp4-collective-reduction-r8
frozen runtime source:
  c7a552effda97481c71684878ff38d34d1d03da7
frozen source-tree SHA-256:
  c8a2f993021d9031e2dd5fd831a42721be3e7e57ad494177ff8f7948be930b0b
case artifacts:
  84
worker classification:
  PASS
selected event budget:
  null
worker return code:
  0
terminal cases:
  0, correctly skipped
```

The worker-side overhead terminal defect was therefore resolved. The
supervisor still failed closed after recording 8,009 successful resource
snapshots and 84 resource samples because it observed two generic
`foreign GPU process detected` violations. Cleanup completed:

```text
process group destroyed:         true
owned children remaining:        []
exact-tag scans:                 [[], [], []]
```

Because the r8 schema did not retain the offending process identity, r8
cannot be postprocessed into a formal gate. Commit `6b7deaf` preserves the
strict rejection while adding the GPU UUID, PID, and process name to every
future violation.

### r9: rejected cleanup evidence

```text
attempt:
  20260828-qwen38-tp4-collective-reduction-r9
frozen runtime source:
  6b7deaf5445879d7cf2626878f82a15626c19f77
frozen source-tree SHA-256:
  bf3f61af819aeb2a0d1c41a11df9e9e46c279af53c20fd12ed919eaa7a314e7f
selected GPUs:
  rank 0 -> physical 1 -> GPU-7dc22583-df04-6c76-4ba5-ea32c428c130
  rank 1 -> physical 2 -> GPU-63c05907-407b-8240-07a0-f38872840867
  rank 2 -> physical 3 -> GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d
  rank 3 -> physical 4 -> GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1
launch admission:
  3 MiB, 0% utilization, no compute process on every selected GPU
```

The single r9 supervisor and worker were launched from the frozen source.
All observed target-GPU processes were either the worker or its three
same-PGID rank children. No second worker was launched. The original local
controller later exhausted its bounded SSH retries while the remote worker
continued. A same-tag resume was rejected before any worker start by the
Kerberos TTL fail-fast guard.

The remote worker then completed all 84 calibration cases and exited zero:

```text
worker classification:           PASS
selected event budget:           null
case artifacts:                  84
terminal cases:                  0, correctly skipped
worker return code:              0
resource snapshots:              8,141
resource samples:                84
resource violations:             []
process group destroyed:         true
owned children remaining:        []
```

The supervisor nevertheless failed closed because the first post-worker
exact-tag scan reported two transient PIDs while the next two scans were
empty:

```text
supervisor classification:       FAIL
exact-tag scans:                 [[2837114, 2837115], [], []]
cleanup complete:                false
```

Both transient PIDs had exited before inspection, and all r9 supervisor,
worker, and rank processes were absent. The defect was reproduced locally:
`exact_tag_processes()` flattened `/proc/<pid>/cmdline` and used substring
search, so a read-only controller probe whose `python -c` source text
contained both the attempt tag and worker filename was misclassified as a
worker. The argv-aware remote-state query did not have this defect.

Commit `f4f6ee7` adds a focused RED/GREEN regression and changes the
supervisor scanner to require the attempt as an exact argv element and the
worker filename as an argv basename. Verification was:

```text
focused RED:                     expected [], observed [123]
supervisor suite:                8 passed
production-controller suite:     23 passed
py_compile:                      PASS
git diff --check:                PASS
local/remote commit SHA:         f4f6ee7a9182f47d5e4f6577c217db1aa9793391
```

The repair does not alter r9 evidence. r9 remains an immutable failed
attempt, has no producer classification, and must not be assembled or
verified as a terminal gate.

That rejected resume exposed a local evidence-lifecycle defect: the
fail-fast branch replaced the successful canonical `dry_run.json` and
`ssh_storage_preflight.json` even though it did not query or mutate the
remote attempt. The original `DRY_RUN_READY` payload was recovered from the
contemporaneous controller output, including the 5,627-second TGT lifetime;
the successful preflight was reconstructed from that payload and the frozen
path/model response. The rejected resume remains separately preserved as
`controller/resume_blocked.json`. Commit `0ce14ff` adds a RED/GREEN
regression test and makes future blocked resumes preserve existing canonical
receipts while writing only the separate resume-blocked observation.

### r10: post-fix qualification attempt

```text
attempt:
  20260828-qwen38-tp4-collective-reduction-r10
frozen runtime source:
  f4f6ee7a9182f47d5e4f6577c217db1aa9793391
frozen source-tree SHA-256:
  b478a9b3c59d11f7b1fb94f2d2530110d40d3f44cc77b4341b6a5c85eed4f83a
selected GPUs:
  rank 0 -> physical 1 -> GPU-7dc22583-df04-6c76-4ba5-ea32c428c130
  rank 1 -> physical 2 -> GPU-63c05907-407b-8240-07a0-f38872840867
  rank 2 -> physical 3 -> GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d
  rank 3 -> physical 4 -> GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1
launch admission:
  3 MiB, 0% utilization, no compute process on every selected GPU
Kerberos expiry:
  2026-08-29T16:33:34+08:00
Kerberos remaining lifetime at launch guard:
  27,832 seconds
```

r10 was created only after the r9 cleanup defect had a focused regression
test, an argv-aware repair, passing supervisor/controller suites, a pushed
source commit, and exact local/remote SHA agreement. It is a new immutable
attempt rather than a reinterpretation or relaunch of r9.

### r10 terminal execution

The one r10 worker completed naturally on 2026-08-29. No replacement worker
or alternate source was launched.

```text
supervisor PID:                   2850309
worker PID / PGID:               2850310 / 2850310
worker return code:              0
worker classification:           PASS
calibration cases:               84 / 84
resource snapshots:              8,106
resource samples:                84
resource violations:             []
process group destroyed:         true
owned children remaining:        []
exact-tag scans:                 [[], [], []]
supervisor classification:       PASS
```

All 60 measured calibration pairs were valid, but no nonzero event budget
satisfied both frozen overhead limits:

| Event budget | Pairs | Median overhead | Maximum overhead | Frozen verdict |
| ---: | ---: | ---: | ---: | --- |
| 0 | 15 | 2.773087% | 6.320970% | control maximum exceeds 5% |
| 8 | 15 | 3.042056% | 4.561507% | median exceeds 3% |
| 16 | 15 | 2.579423% | 5.042663% | maximum exceeds 5% |
| 32 | 15 | 3.099692% | 8.660734% | median and maximum exceed limits |

The selected event budget is therefore `null`. Per the frozen plan, the
35 conditional terminal cases were not launched. Consequently there are no
terminal census or timing rows and no measured TPOT, latency, or throughput
benefit to report.

The static inventory is nevertheless complete:

```text
expected collective sites:       130
observed catalog rows:            130
consumer dependency proofs:      130
immediate-consumer sites:         129
static removable sites:           1
removable operation:              embedding.input all-reduce
calls removed per decode step:    1
additional persistent bytes/rank: 1,907,097,600
additional peak bytes/rank:       1,907,097,600
correctness rows:                 560
```

This is a costed static opportunity, not a performance win. Replicating the
full embedding could remove one collective per decode step, but costs about
1.776 GiB of persistent and peak device memory per rank, and the frozen
measurement method could not qualify any nonzero event budget.

### Postprocess timeout and same-attempt recovery

The first local controller invocation used a 120-second SSH command timeout.
The remote assembler required about 153 seconds, continued after the local
timeout, and atomically completed the producer bundle. The worker,
supervisor, cleanup evidence, cases, and producer were not modified.

A read-only state query then proved:

```text
attempt action:                  POSTPROCESS
live exact-tag PIDs:             []
supervisor classification:       PASS
producer classification:         INCONCLUSIVE_PROFILER_OVERHEAD
independent verifier present:    false
```

The same r10 attempt and frozen source were resumed with a 600-second command
timeout. The adapter reused the existing producer, did not stage or launch a
worker, and completed the remote verifier, bounded download, local verifier,
and post-verification manifest. Commit
`d8d85da9479a06f122a93987b1d87b9a5f8e0cd0` subsequently gives all
postprocess commands a minimum 600-second timeout. That controller-only
hardening is not part of r10's frozen runtime source.

### Independent verification and manifest

```text
producer classification:         INCONCLUSIVE_PROFILER_OVERHEAD
remote verifier status:          PASS
remote reconstructed result:     INCONCLUSIVE_PROFILER_OVERHEAD
local verifier status:           PASS
local reconstructed result:      INCONCLUSIVE_PROFILER_OVERHEAD
remote/local verifier SHA-256:
  c115dee20a4700ae71d316df45878b7fae2973aa3a28058c1e0b720e7dc8223d
remote/local verifier bytes:      identical
manifest hashed artifacts:       16
post-verification manifest SHA-256:
  11c09aeea95a3a722b692b9f31ffdec9ea4367561f2e997a1649d52b0d020383
```

The verifier independently confirmed the source/model/GPU identities,
60 calibration pairs, 560 correctness rows, 84 resource samples, exact
conditional zero-row terminal inventory, artifact hashes, and cleanup.

## Prompt-to-artifact checklist

| Requirement | Evidence | Verdict |
| --- | --- | --- |
| Frozen source identity | r10 source `f4f6ee7...`, tree SHA-256 `b478a9b3...` | `PASS` |
| Frozen model identity | Qwen3.8-27B revision `1d4bf0f...` | `PASS` |
| Four strict-clean GPUs | physical GPUs 1-4, 3 MiB, 0%, no compute process at launch | `PASS` |
| Approved storage | all remote task data below `/data00/home/sitian/.../command-timeline-20260818` | `PASS` |
| Exactly one worker | one supervisor/worker launch; same-attempt postprocess-only recovery | `PASS` |
| Calibration coverage | 84 cases and 60 measured pairs | `PASS` |
| Event-budget selection | no nonzero budget satisfies both overhead ceilings | `NO_SELECTION` |
| Terminal coverage | correctly skipped under the frozen conditional contract | `PASS_CONDITIONAL_SKIP` |
| Static inventory | 130 catalog rows and 130 dependency proofs | `PASS` |
| Correctness and resources | 560 valid correctness rows, 84 resource samples, zero violations | `PASS` |
| Cleanup | destroyed process group, no children, three empty exact-tag scans | `PASS` |
| Producer / remote / local agreement | all reconstruct `INCONCLUSIVE_PROFILER_OVERHEAD` | `PASS` |
| Immutable manifest | 16 artifact hashes verified locally | `PASS` |
| Forbidden mechanisms | no async collectives, communication stream, overlap, or chunked collective implementation | `PASS` |

## Final classification

```text
QWEN38_TP4_COLLECTIVE_REDUCTION_QUALIFICATION=COMPLETE
PRODUCER_CLASSIFICATION=INCONCLUSIVE_PROFILER_OVERHEAD
REMOTE_VERIFIER=PASS
LOCAL_VERIFIER=PASS
OVERLAP_DESIGN_AUTHORIZED=false
ASYNC_COLLECTIVES_AUTHORIZED=false
SYNC_COLLECTIVE_CANDIDATE_DESIGN_AUTHORIZED=false
NEXT_COMMAND=select another optimization
```

The qualification is complete, but it does not establish a speedup and does
not authorize a synchronous collective-reduction implementation. The
measurement cost itself exceeded the frozen gate, so the safe next action is
to select a different optimization direction or first design a lower-overhead
measurement method as a separate, newly frozen experiment.
