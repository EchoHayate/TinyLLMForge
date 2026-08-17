# Qwen3.5 TP4 Real-Candidate Provenance Replay Gate Design

## Objective

Produce four independently loaded, published, and bound TP4 checkpoint
candidate rows from the approved real Qwen3.5 checkpoint, release every
candidate and private CPU object, and only then replay those immutable rows
through the production TP4 shared-memory acknowledgement transport and
production Engine all-rank binder:

```text
approved real checkpoint + manifest authorization
  -> rank0 fresh process: load -> publish -> bind -> release
  -> rank1 fresh process: load -> publish -> bind -> release
  -> rank2 fresh process: load -> publish -> bind -> release
  -> rank3 fresh process: load -> publish -> bind -> release
  -> immutable four-row real provenance artifact
  -> fresh production TP4 shared-memory transport
  -> production Engine all-rank binder
  -> homogeneous commit or directed provenance rejection
```

This is a CPU-only provenance and transaction gate. It does not keep four
real candidates alive concurrently. It does not construct `LLMEngine` or
`ModelRunner`, enter the scheduler, call `LLMEngine.step()`, initialize CUDA,
execute forward, or run inference.

## Claim Name

Every row and artifact must use:

```text
provenance:
  real-checkpoint-derived-serial-rank-replay
claim boundary:
  not-live-concurrent-tp4-candidate-binding
```

The result may be described as real-checkpoint-derived TP4 provenance replay.
It must not be described as four live rank candidates bound concurrently.

## Why Serial Rank Production

The proven TP2 real-candidate gate measured approximately 4.57 GiB total
`VmHWM` increment per rank. Holding four candidates simultaneously would
multiply retained CPU storage and turn a correctness gate into an
uncontrolled memory-pressure experiment.

Serial fresh processes provide the required rank-specific evidence while
bounding peak live ownership to one candidate. Each process must prove full
release before the next rank starts. The later TP4 transport phase consumes
only immutable JSON rows and therefore cannot retain checkpoint tensors.

## Approaches Considered

### Four Concurrent Real Candidates

Rejected for this gate. It would most closely resemble live TP4 ownership but
would retain four complete rank-local model targets simultaneously, require a
new aggregate memory budget, and conflate provenance validation with
concurrency and memory-pressure behavior.

### Duplicate TP2 Rows as TP4

Rejected. Changing participant IDs on TP2 rows would not prove TP4 sharding,
rank2/rank3 assignment, TP4 layout identity, or TP4 payload hashes.

### Serial TP4 Real Candidate Production and Row Replay

Selected. It proves that every participant row comes from a complete
rank-specific real checkpoint load and production publication/binding method,
while preserving one-candidate peak memory and independently testing the
already proven production TP4 transport and binder.

## Immutable Prerequisites

Use the approved checkpoint:

```text
checkpoint directory:
  /data00/home/sitian/sitian-workspace01/tllm/
  qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model
model manifest:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
config:
  ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4
index:
  aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9
config/index header:
  27da983f5ef3e38480d8b5d5976e5c434fc4b5d0c70d09511c35154beecd8db9
shard:
  model.safetensors-00001-of-00001.safetensors
shard bytes:
  4548221488
shard SHA256:
  aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1
authorization:
  10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4
maximum source tensor bytes:
  1017118720
```

Freeze the completed production method prerequisites:

```text
ModelRunner load-and-publish result:
  d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18
source tree:
  a1bf0161eeedf3c73fb176a0f26ab2156bb3d944096db187a9c83eeb98ae5cc8

ModelRunner published-candidate binding result:
  79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a
source tree:
  0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785

TP4 synthetic transport/binder result:
  803c8fac331eeee82b90013e0b0872de8f079661b6dd1ba43225fb446006cce4
source tree:
  e88236ebe4f97ddecf55004e4bbcdb46a677462f183b6724031d85d8648a6de0
```

The new gate inherits the exact 57-file TP4 source closure and adds one gate
file, for 58 source files total.

## Frozen Production Methods

Freeze these file and AST segment identities:

```text
tinyvllm/engine/model_runner.py:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849
load_and_publish_qwen35_checkpoint_candidate:
  9134c5bad8c4127714e07ffd8af56209c247a746e9f0d0ceceb60227c1358612
bind_published_qwen35_loaded_checkpoint_candidate:
  aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd
write_shm:
  f9a377bf748d5be91a3c3722850e5e486f8e7dd8157e87d3dc6d692a60be6d76
read_shm:
  1266b5d20b2978b655716f9ec8b58ce0a5644b9709164a23c18b85346170054a
loop:
  342bac6d01606e4834e7ed77ef3e76d59b2fc3ea617afebe2c195912159dd2bb
dispatch_command:
  9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342

tinyvllm/engine/llm_engine.py:
  6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae
call_model_runner_acknowledged:
  6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d
bind_qwen35_loaded_checkpoint_candidates:
  82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c
```

The gate extracts and compiles these methods without importing production
`model_runner.py` or `llm_engine.py`.

## Phase A: TP4 Real Rank Production

Run four fresh worker processes sequentially:

```text
(tensor_parallel_size=4, tensor_parallel_rank=0)
(tensor_parallel_size=4, tensor_parallel_rank=1)
(tensor_parallel_size=4, tensor_parallel_rank=2)
(tensor_parallel_size=4, tensor_parallel_rank=3)
```

Each worker:

1. validates the approved manifest, shard, authorization, and source tree;
2. prepares a fresh TP4 rank-local CPU target;
3. installs the existing manifest-bound authorized loader on a private runner
   shell;
4. invokes the exact production load-and-publish method;
5. invokes the exact production published-candidate binding method;
6. records all 320 binding hashes, 26 phase hashes, aggregate hash, aliases,
   loader statistics, layout fingerprint, dtype, and returned bound row;
7. clears all selected destinations;
8. proves non-selected values, tensor identities, and pool state are unchanged;
9. proves the runner, slot, request, target, candidate, owner, model, pool,
   runtime bridge, and runtime identity are collected;
10. proves CUDA remained uninitialized and forward counters remained zero.

The rank row is accepted only when:

```text
participant_id == tensor_parallel_rank
operation == bind_loaded_checkpoint_candidate
status == bound
model_fingerprint == approved manifest SHA256
dtype == bfloat16
detail == empty
```

All four rows must independently report the same TP4 layout fingerprint. Rank
payload hashes may differ and must be retained as rank-specific provenance.

## TP4 Memory Budget

Because only one worker is alive at a time, the gate uses a per-process
ceiling rather than a four-rank aggregate ceiling.

Freeze conservative TP4 ceilings:

```text
total VmHWM increment:
  6291456 KiB
post-Torch VmHWM increment:
  6029312 KiB
post-metadata VmHWM increment:
  5767168 KiB
```

These equal the previously frozen complete-transaction TP2 ceilings and
remain above the observed TP2 success increments. Any TP4 worker exceeding a
ceiling fails the run before row publication. The result must record actual
measurements; the ceiling is not a performance claim.

## Phase B: Immutable Provenance Oracle

After all four workers exit and all private objects are collected, atomically
write:

```text
tp4_real_candidate_provenance_oracle.json
```

The oracle contains:

- exact approved checkpoint and authorization identities;
- exact source tree and frozen method hashes;
- four unique producer PIDs;
- rank-specific 320 binding hashes, 26 phase hashes, aggregate hash, loader
  statistics, memory observations, layout fingerprint, dtype, and bound row;
- explicit serial-production provenance and non-live claim boundary;
- proof that every producer exited before replay starts.

The oracle SHA256 becomes an input to Phase C.

## Phase C: Production TP4 Replay

Use the completed TP4 shared-memory transport:

- one fresh uniquely named 1 MiB shared-memory segment;
- three real Events;
- three real acknowledgement pipes;
- three fresh worker-loop children;
- production dispatch/write/read/loop/ack collector;
- production `LLMEngine.call_model_runner_acknowledged`;
- production `LLMEngine.bind_qwen35_loaded_checkpoint_candidates`.

Rank shells return only the immutable real-derived rows. They do not retain or
reconstruct checkpoint candidates.

Success must:

- send acknowledgements in deliberate order `(3, 2, 1)`;
- collect ranked results `(1, 2, 3)`;
- commit the exact model/layout/`bfloat16` tuple;
- return the same row tuple on exact repeat with zero new binding dispatch;
- send one fire-and-forget exit envelope;
- join all children and unlink the segment exactly once.

## Directed Failure Modes

Run three fresh replay attempts derived from the immutable real oracle:

```text
tp4_real_replay_success
tp4_real_replay_rank2_model_mismatch
tp4_real_replay_rank2_layout_mismatch
tp4_real_replay_rank2_dtype_mismatch
```

Each mismatch changes only one rank2 identity field after oracle validation
and marks the row as a directed negative replay. It must not alter rank2
binding hashes, phase hashes, aggregate hash, loader statistics, or producer
evidence.

All acknowledgements remain `ok` and the collector remains healthy. The
production binder rejects the exact field and leaves completion unset.

## Orchestration

Use one unique remote run directory and deterministic staging:

```text
58 source files
3 immutable prerequisite artifacts
approved checkpoint path validated in place
```

Execution order:

1. stage and remotely rehash every source and prerequisite;
2. run four serial real-rank producer workers;
3. run a separate provenance-oracle finalizer;
4. prove every producer PID is absent;
5. run four fresh TP4 replay attempts;
6. run a separate result finalizer;
7. atomically publish result and source manifest locally and remotely.

Any partial producer set, producer cleanup failure, replay failure, or
finalizer failure prevents authoritative publication. Preserve every failed
run directory.

## Artifacts

Publish:

```text
tp4_real_candidate_provenance_oracle.json
tp4_real_candidate_provenance_replay_preflight.json
source_manifest.json
```

The run directory therefore contains three immutable inputs, one intermediate
oracle, two final result files, and the exact 58-file source closure.

## Independent Verification

A standard-library-only verifier imports neither TinyLLMForge nor any gate.
It independently validates:

- all three prerequisite SHA256 values and schemas;
- approved checkpoint, manifest, config, index, shard, and authorization;
- exact 58-file source closure and source tree;
- frozen file and method hashes/signatures;
- four unique serial producer PIDs and producer-before-replay ordering;
- exact TP4 rank set and participant IDs;
- 320 binding hashes, 26 phase hashes, aggregate hash, aliases, loader stats,
  memory ceilings, collection, CUDA false, and forward zero for every rank;
- homogeneous model/layout/dtype across the real oracle;
- provenance and non-live claim boundary;
- four unique replay outer PIDs, twelve child PIDs, and four segment names;
- transport envelopes, payload bytes, Event/read/executor counts, ordering,
  acknowledgements, completion, exact repeat, directed mismatch scope, and
  cleanup;
- local/remote result SHA equality and exact inventory.

Tamper tests must reject:

1. a real producer row whose participant ID does not match its TP rank;
2. a re-signed source tree that imports production Engine or ModelRunner;
3. a replay mismatch that changes a second unauthorized field;
4. a producer PID that overlaps or occurs after a replay outer PID.

## Static Safety

Require:

- no production Engine or ModelRunner import/construction;
- no scheduler or `LLMEngine.step()` call;
- no CUDA operation, forward, or inference call;
- exactly one authorized loader-builder call site;
- exact extracted production method invocation sites;
- no fixed `tinyvllm` shared-memory name;
- no concurrent producer workers;
- exact real worker hard rejection unchanged;
- schema-v2 canonical `NO_GO` unchanged.

Read-only `torch.cuda.is_initialized()` observations remain allowed inside
producer workers.

## Allowed Conclusion

Passing proves:

- four real TP4 rank-local checkpoint candidates were independently loaded,
  published, and bound through the frozen production methods;
- each rank produced a complete, source-bound, value-hash-backed bound row;
- every candidate and private CPU object was released before replay;
- the production TP4 transport and production all-rank binder accept the four
  homogeneous real-derived rows;
- the binder fails closed for directed rank2 model/layout/dtype mismatches.

It does not prove:

- four candidates were live concurrently;
- a constructed `LLMEngine` or `ModelRunner` performed the transaction;
- scheduler or `LLMEngine.step()` integration;
- CUDA, forward, inference, model correctness, or quality;
- latency, throughput, cache savings, GPU-memory savings, or compression.

The next safe boundary after this gate is live concurrent TP4 ownership under
an explicit aggregate CPU-memory budget, followed only later by constructed
Engine/ModelRunner runtime integration.

## Authoritative Evidence

The pristine authoritative run completed on the approved remote host:

```text
run tag:
  qwen35-tp4-real-candidate-replay-20260728-145713
remote target:
  sitian@10.232.195.203
source tree:
  42dddc0eac0a6db6041d5abb71df34db4d5e7c99d3b74d69f94598a2f24eb137
oracle SHA256:
  d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef
result SHA256:
  dc25e1cc72701d994745210022ddbd6bc603054a5acf842400aa48a3159e88e4
manifest SHA256:
  f6c1c8846fc478bd341c720665b1230abe896db672434aee8c314764def22ead
```

Producer PIDs, in rank order:

```text
2790251, 2792474, 2794572, 2797068
```

Replay outer PIDs:

```text
2799604, 2799841, 2800095, 2800421
```

Replay child PIDs:

```text
2799761, 2799763, 2799764
2799965, 2799966, 2799967
2800201, 2800204, 2800206
2800547, 2800548, 2800550
```

All twenty PIDs were unique. Every producer exited before replay, all private
objects were collected, CUDA remained uninitialized, and model/attention
forward counters remained zero.

Per-rank aggregate destination hashes:

```text
rank0:
  1ebc443caba87a0962b158733781ced7c7ee40b546759218c8676f1dc1b5ded4
rank1:
  fdb72b9c44174b141510fa8c7858a193e59d28d2366f382abd15d945a3cec633
rank2:
  7e80aa2e34984292ebcbee06376510b8a0b5f82a0b1ae51ebe6ae261f1b487c7
rank3:
  6e97fcf213db2116bcfd778ae67efac62d48fda98715cefdb905422690193dca
```

Every rank reported 320 binding hashes, 26 phase hashes, 3,763,655,360
loaded bytes, and a 1,017,118,720-byte maximum source tensor. Total observed
`VmHWM` increments were 2,589,652, 2,596,372, 2,593,224, and 2,593,456 KiB,
all below the frozen 6,291,456 KiB ceiling.

The successful replay committed the exact homogeneous
model/layout/`bfloat16` tuple. The three negative replays changed only rank2
`model_fingerprint`, `layout_fingerprint`, or `dtype`; all acknowledgements
remained `ok`, the collector remained healthy, the exact field was rejected,
and completion remained unset.

Independent verification:

```text
local three-file publication:
  PASS, 2206 checks
full remote inventory:
  PASS, 2211 checks
focused verifier tests:
  PASS, 5 tests
```

The five verifier tests cover the pristine artifact, participant mismatch,
unauthorized second replay-field mutation, producer/replay PID overlap, and a
fully re-signed source tree importing production `LLMEngine`. Static safety
is evaluated before the frozen-tree identity check so the re-signed attack is
rejected for the prohibited import itself.

The verifier was staged outside the authoritative run at:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-tp4-real-verifier-20260728-151500/
```

The remote run inventory remained unchanged and its `source/` directory did
not receive the verifier. The superseded
`qwen35-tp4-real-candidate-replay-20260728-143815` run remains preserved but
is not authoritative because its remote source directory was later polluted
by an uploaded verifier.

## Requirement Audit

| Requirement | Evidence |
| --- | --- |
| Four real TP4 rank rows | Four unique producer PIDs and ranks `0..3` |
| Strict serial production | Every prior producer exited before the next and before replay |
| Frozen production methods | AST signatures and method hashes independently verified |
| Complete payload provenance | 320 binding hashes, 26 phase hashes, aggregate hash per rank |
| Immutable oracle | Canonical oracle SHA bound into every replay |
| Production TP4 transport/binder | Success replay plus exact-repeat zero-dispatch |
| Directed fail-closed behavior | Rank2 model/layout/dtype mismatch rows |
| Cleanup | All private objects collected; all 20 PIDs unique and exited |
| Static safety | No Engine/ModelRunner construction, scheduler, step, CUDA, forward, or inference |
| Independent tamper resistance | Five focused tests including fully re-signed import attack |
| Pristine source-bound evidence | 58-file source tree and isolated remote verifier |

The gate is complete at its stated boundary. It still does not prove live
concurrent TP4 candidate ownership, constructed Engine/ModelRunner runtime
integration, model correctness, quality, latency, throughput, cache savings,
GPU-memory savings, or compression.
