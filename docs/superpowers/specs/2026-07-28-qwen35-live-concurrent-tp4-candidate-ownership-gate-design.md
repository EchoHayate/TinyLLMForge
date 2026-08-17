# Qwen3.5 Live Concurrent TP4 Candidate Ownership Gate Design

## Objective

Prove that four independently loaded Qwen3.5 TP4 checkpoint candidates can
remain live at the same time, with one rank-local candidate per process,
before all four are released and collected:

```text
rank0 load -> retain -> ready
rank1 load -> retain -> ready
rank2 load -> retain -> ready
rank3 load -> retain -> ready
all four candidates simultaneously live
  -> validate rank identities and payload provenance
  -> release all ranks
  -> clear selected tensors
  -> collect all private objects
  -> prove every worker exited
```

This is the next ownership boundary after the completed serial provenance
gate. It remains CPU-only and construction-free with respect to production
`LLMEngine` and `ModelRunner`.

## Claim

Successful evidence may use:

```text
provenance:
  real-checkpoint-derived-live-concurrent-tp4-ownership
claim boundary:
  not-constructed-engine-runtime-binding
```

It may state that four real TP4 rank-local candidates were live
simultaneously in four fresh processes. It must not state that a constructed
production Engine or ModelRunner owned or bound them.

## Prerequisite

Freeze the pristine serial gate:

```text
run:
  qwen35-tp4-real-candidate-replay-20260728-145713
source tree:
  42dddc0eac0a6db6041d5abb71df34db4d5e7c99d3b74d69f94598a2f24eb137
oracle:
  d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef
result:
  dc25e1cc72701d994745210022ddbd6bc603054a5acf842400aa48a3159e88e4
manifest:
  f6c1c8846fc478bd341c720665b1230abe896db672434aee8c314764def22ead
```

Reuse the approved checkpoint, authorization, 58-file source closure, frozen
production method hashes, TP4 KV-head replication policy, and exact producer
payload validation. Do not regenerate expected payload hashes from a
different model or synthetic row.

## Approaches

### Four Simultaneous Loads

Rejected. Starting all four streamed loads together can materialize four
1,017,118,720-byte source tensors at once, adding avoidable peak pressure and
making failure diagnosis dependent on I/O timing.

### Sequential Production Followed by Replay

Already completed, but insufficient for this gate. It proves four real rows,
not simultaneous candidate ownership.

### Staggered Loads with Concurrent Residency

Selected. Start rank0 and wait for its retained-candidate ready signal, then
rank1, rank2, and rank3. Earlier candidates remain live while later ranks
load. Once all four workers report ready, the coordinator proves all PIDs are
alive and all release acknowledgements are pending. It then broadcasts
release and joins every worker.

This provides simultaneous real ownership while limiting active streamed
source materialization to one worker at a time.

## Worker Lifecycle

Each rank worker must:

1. validate the approved checkpoint, authorization, source tree, and method
   hashes;
2. construct one fresh TP4 rank-local CPU target;
3. invoke the frozen production load-and-publish and published-candidate
   binding methods exactly once;
4. validate 320 binding hashes, 26 phase hashes, aggregate hash, aliases,
   loader statistics, model/layout/dtype, and participant rank against the
   pristine serial oracle;
5. retain the runner, publication slot, candidate, owner, model, pool, target,
   runtime bridge, and runtime identity;
6. report a canonical ready row and wait without clearing or dropping any
   retained object;
7. receive exactly one release command;
8. clear every selected destination in reverse unique-object order;
9. prove non-selected values, tensor identities, and pool state unchanged;
10. drop the retained graph, run garbage collection, and prove every private
    object collected;
11. report a final released row and exit zero.

Workers must not start checkpoint loading until the coordinator explicitly
issues their rank-specific start command. This makes source materialization
strictly staggered even though prior candidates remain resident.

## Coordinator Protocol

Use four fresh worker processes and four independent duplex control channels.
The coordinator state machine is:

```text
spawn ranks 0..3
for rank in 0..3:
  send START(rank)
  receive READY(rank)
  validate row
  prove all previously ready PIDs remain alive
after READY(3):
  sample all four /proc/<pid>/status records
  prove all four PIDs alive concurrently
  prove no RELEASED row exists
  write atomic concurrent-residency snapshot
send RELEASE to ranks 3,2,1,0
receive RELEASED rows
join all workers
prove all PIDs absent
finalize authoritative artifact
```

Reverse release order exercises ownership independence and avoids coupling
cleanup to load order.

## Concurrent Residency Evidence

The atomic snapshot must contain:

- four unique live PIDs and ranks `0..3`;
- coordinator PID and timestamp;
- exact ready-row SHA256 per rank;
- exact candidate/model/layout/dtype identities;
- 320/26/aggregate payload identities per rank;
- each worker's ready `VmRSS` and `VmHWM`;
- coordinator-observed `/proc/<pid>/status` `VmRSS`, `VmHWM`, and process
  state for all four PIDs;
- proof that every worker had acknowledged ready and none had acknowledged
  release;
- exact start and ready ordering `(0, 1, 2, 3)`;
- exact later release order `(3, 2, 1, 0)`.

The coordinator must reject a PID that exits, changes rank identity, reports a
second ready row, accepts release before the snapshot, or overlaps another
rank.

## Memory Contract

The pristine serial rows observed per-rank total `VmHWM` increments:

```text
rank0: 2589652 KiB
rank1: 2596372 KiB
rank2: 2593224 KiB
rank3: 2593456 KiB
sum:  10372704 KiB
```

The exact sum is 10,372,704 KiB. Freeze conservative, 256-MiB-aligned
correctness ceilings:

```text
per-worker total VmHWM increment:
  3145728 KiB
aggregate worker VmHWM increment sum:
  12582912 KiB
aggregate ready VmRSS:
  8388608 KiB
host MemAvailable decrease from pre-spawn baseline:
  12582912 KiB
```

The remote host had 1,659,713,452 KiB available before design finalization,
so these ceilings are well below current capacity. The gate must still sample
the live host before execution and refuse to start unless:

```text
MemAvailable >= 16777216 KiB
swap is not required
all four worker slots can be created
```

Memory ceilings are safety limits, not performance or cache-saving claims.

## Failure Modes

Directed tests must cover:

- rank1 exits after rank0 ready;
- rank2 ready row has a participant mismatch;
- rank3 exceeds its per-worker memory ceiling;
- aggregate ready `VmRSS` exceeds the ceiling;
- a worker reports released before the concurrent snapshot;
- one worker ignores release and times out;
- cleanup leaves one private object reachable;
- source loading begins for rank N+1 before rank N ready.

Every failure must send release/abort to every live worker, clear all selected
destinations, join or terminate only after the graceful timeout, preserve the
failed run directory, and prevent authoritative publication.

## Static Safety

Require:

- no import or construction of production `LLMEngine` or `ModelRunner`;
- no scheduler, `LLMEngine.step()`, CUDA operation, forward, or inference;
- only read-only `torch.cuda.is_initialized()` observations;
- one authorized loader-builder site;
- exact frozen production method extraction;
- exact real worker hard rejection unchanged;
- schema-v2 canonical `NO_GO` unchanged;
- no fixed shared-memory or IPC name;
- no signal that can release a worker before its ready row is validated.

## Artifacts

Publish only after all workers release and exit:

```text
tp4_live_concurrent_candidate_ownership.json
source_manifest.json
```

The main artifact contains ready rows, the atomic concurrent-residency
snapshot, released rows, ordering, memory observations, cleanup evidence, and
source/prerequisite identities.

A standard-library-only verifier must independently validate the full
artifact and reject re-signed production imports, missing concurrent
residency, PID reuse/overlap, reordered starts/releases, payload drift,
premature release, memory drift, and incomplete collection.

## Allowed Conclusion

Passing proves four real Qwen3.5 TP4 rank-local checkpoint candidates were
live simultaneously, source-bound, payload-verified, and fully released under
the frozen aggregate CPU-memory contract.

It does not prove:

- ownership by constructed production Engine/ModelRunner instances;
- scheduler or `LLMEngine.step()` integration;
- CUDA, forward, inference, model-output correctness, or quality;
- latency, throughput, cache savings, GPU-memory savings, or compression.

The next safe boundary is constructed Engine/ModelRunner ownership and
all-rank binding without scheduler or forward execution.

## Authoritative Evidence

The final authoritative remote run completed on the approved host:

```text
run:
  qwen35-tp4-live-concurrent-ownership-20260728-163700
source tree:
  0a4ae63468b7f0bdccc0c41d4803e36d418e9966b5d66525ea7690f8203bfeb3
result:
  f2d38ca089a53a413236fbf18c057fb10df04b84a338248b2004d77f5060c280
manifest:
  d9d0166214d4e78d756f6c2a20306a0e537f5fc5f138adde155cd6c9f6b1b236
ready rows:
  522fd45a3ef2852b50255a754f267aa513a123b63ca1898c668e34e54a7400d7
released rows:
  b2f850bf9f6df4bd76b1162dec735397b66e15fa899aa439ec3e380225179d60
pristine serial oracle:
  d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef
```

The four concurrently live worker PIDs were:

```text
rank0: 344450
rank1: 344451
rank2: 344452
rank3: 344453
```

The atomic snapshot timestamp was
`1785227853574306874` Unix nanoseconds. Start and ready order were
`(0, 1, 2, 3)`; release and released order were `(3, 2, 1, 0)`.
No release acknowledgement existed at snapshot time, and all four PIDs were
absent after cleanup.

Every ready row carried and matched the external pristine rank row for:

```text
320 binding destination SHA256 values
26 phase destination SHA256 values
24 alias groups
aggregate destination SHA256
loader statistics
model/layout/dtype identity
participant rank
```

Observed memory remained within the frozen contract:

```text
per-rank total VmHWM increment:
  2930424, 2939212, 2939224, 2935008 KiB
aggregate worker VmHWM increment:
  11743868 KiB <= 12582912 KiB
aggregate ready VmRSS:
  5801008 KiB <= 8388608 KiB
host MemAvailable decrease:
  7767172 KiB <= 12582912 KiB
```

Both local and run-external remote standard-library verifiers passed
`500` checks with the same result, manifest, and source-tree hashes. The
remote verifier was outside the authoritative run:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-live-concurrent-verifier-20260728-163913
```

The first published live run
`qwen35-tp4-live-concurrent-ownership-20260728-162329` and the later
`...-163033` run remain preserved as superseded evidence. The first lacked
full 320/26/alias payload fields; the second lacked the required snapshot
timestamp and executable `internal-worker` evidence. Neither is the final
authoritative artifact.

## Requirement Audit

| Requirement | Evidence |
|---|---|
| Four workers exist before loading | coordinator test plus final unique PIDs |
| Exact staggered starts | artifact `start_order=[0,1,2,3]` |
| Four candidates live simultaneously | atomic snapshot, four live PIDs, zero release acknowledgements |
| Full pristine payload provenance | 320 binding, 26 phase, 24 alias, aggregate and identity equality per rank |
| Reverse release | artifact `release_order=[3,2,1,0]` |
| Complete cleanup | four released rows, all selected tensors zero, all private objects collected |
| No residual processes | artifact empty residual PID list and remote `/proc` absence check |
| Aggregate memory contract | raw host/worker observations and recomputed memory summary |
| Source-bound execution | 59-file source closure and matching source tree |
| Static safety | no production Engine/ModelRunner import or construction; no scheduler/forward/inference |
| Independent verification | local and remote `500`-check PASS |
| Exact worker rejection and schema-v2 `NO_GO` preserved | adjacent regression suites and AST constant check |

This closes only the live-concurrent candidate ownership gate. It does not
establish output accuracy, quality retention, latency, throughput, cache
savings, GPU-memory savings, or compression. The long-term performance goal
therefore remains active.
