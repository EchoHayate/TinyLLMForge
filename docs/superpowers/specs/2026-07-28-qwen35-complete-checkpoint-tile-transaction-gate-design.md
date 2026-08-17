# Qwen3.5 Complete Checkpoint Tile Transaction Gate Design

## Status

Approved for inline implementation and one source-bound live run on
`sitian@10.232.195.203`.

This gate extends the completed four-layer cadence-block transaction to the
entire real 320-binding Qwen3.5 checkpoint target. It remains a dedicated,
reversible CPU preflight. The production checkpoint loader, assignment,
`target.take()`, candidate installation, CUDA, Engine, publication, forward,
and inference remain forbidden.

## Considered Approaches

### 1. Full 320-binding single transaction

Stream embedding, all 24 decoder layers, and final norm through one
source-bound transaction. Snapshot and restore all 296 unique destinations.

Advantages:

- proves the complete checkpoint target rather than only decoder layers;
- preserves one global transaction hash and one rollback boundary;
- closes the two remaining root bindings instead of creating another gate.

Cost:

- embedding adds 1,017,118,720 bytes at TP=1 and 508,559,360 bytes per TP=2
  rank;
- the full selected snapshot is approximately 3.51 GiB at TP=1 and 1.75 GiB
  at TP=2.

### 2. Decoder-only 318-binding transaction

Exclude embedding binding `0` and final-norm binding `319`.

This reduces bytes and memory but does not prove a complete checkpoint
transaction. Only two bindings remain outside the gate, so a second root-only
gate would add process and evidence complexity without improving isolation.

### 3. Six cadence-block transactions

Run six independent four-layer transactions.

This minimizes peak snapshot memory but cannot prove one global transaction,
global rollback, or root binding participation. It repeats a boundary already
proven by the four-layer gate.

## Decision

Use approach 1: one complete 320-binding transaction.

## Frozen Binding and Phase Contract

The exact binding indices are:

```text
0..319
```

Totals:

```text
320 bindings
296 unique destination tensor objects
24 intentional gate/up alias groups
```

The real binding plan is not numerical layer order. The exact 26 phase runs
are:

```text
embed_tokens: binding 0
layer 0:      bindings 1..14
layer 1:      bindings 15..28
layer 10:     bindings 29..42
layer 11:     bindings 43..53
layer 12:     bindings 54..67
layer 13:     bindings 68..81
layer 14:     bindings 82..95
layer 15:     bindings 96..106
layer 16:     bindings 107..120
layer 17:     bindings 121..134
layer 18:     bindings 135..148
layer 19:     bindings 149..159
layer 2:      bindings 160..173
layer 20:     bindings 174..187
layer 21:     bindings 188..201
layer 22:     bindings 202..215
layer 23:     bindings 216..226
layer 3:      bindings 227..237
layer 4:      bindings 238..251
layer 5:      bindings 252..265
layer 6:      bindings 266..279
layer 7:      bindings 280..290
layer 8:      bindings 291..304
layer 9:      bindings 305..318
final_norm:   binding 319
```

The state machine must preserve this exact binding-plan phase order. It must
not sort by numerical layer index.

Intentional alias groups:

```text
[12,13]    [26,27]    [40,41]    [45,46]
[65,66]    [79,80]    [93,94]    [98,99]
[118,119]  [132,133]  [146,147]  [151,152]
[171,172]  [185,186]  [199,200]  [213,214]
[218,219]  [229,230]  [249,250]  [263,264]
[277,278]  [282,283]  [302,303]  [316,317]
```

Every pair must be an ordered, disjoint, gap-free partition of exactly one
layer-local `gate_up_proj.weight`. No root binding or non-paired binding may
alias another selected destination.

## Frozen Tile Contract

All production tiles use:

```text
max_tile_bytes = 65536
```

TP=1 rank0:

```text
bindings 320
unique destinations 296
tiles 58169
kind counts:
  replicated 79
  axis0 38248
  segmented_axis0 6912
  squeeze_axis0 18
  axis1 12912
ranges per pass 58169
bytes per pass 3763655360
two-pass pread count 116338
logical payload bytes 7527310720
```

TP=2 rank0/rank1:

```text
bindings 320
unique destinations 296
tiles 29169
kind counts:
  replicated 79
  axis0 19160
  segmented_axis0 3456
  squeeze_axis0 18
  axis1 6456
ranges per pass 121017
bytes per pass 1881935712
two-pass pread count 242034
logical payload bytes 3763871424
```

Root binding contracts:

```text
binding 0:
  target embed_tokens.weight
  loader custom_parameter_loader
  TP=1 local shape [248320,2048]
  TP=2 local shape [124160,2048]

binding 319:
  target final_norm.weight
  loader default_parameter_copy
  local shape [2048]
```

Per decoder layer, retain the already proven linear/full layer contracts from
the four-layer gate.

## Complete Transaction

Each fresh rank process:

1. constructs the real CPU target without taking or installing it;
2. validates the complete 24-layer architecture schedule;
3. validates all 320 binding indices and the exact 26 phase runs;
4. builds the production 65,536-byte tile plan;
5. snapshots all 296 unique destination objects;
6. opens the approved shard exactly twice;
7. streams each tile through independent production/verifier `pread`s;
8. copies only through `_copy_qwen35_checkpoint_tile`;
9. accumulates per-binding, per-phase, per-layer, root, and transaction hashes;
10. before every next phase, proves every completed phase changed and every
    future phase remains zero;
11. proves all non-selected registered tensors remain zero;
12. validates all 24 alias partitions;
13. restores all 296 objects in reverse first-seen order;
14. proves every registered tensor is zero after rollback.

Returning to an earlier phase, skipping a phase, mutating any future phase,
incomplete coverage, short reads, payload mismatch, alias gap/overlap, root
binding drift, memory drift, or rollback failure must fail closed.

## Streaming and Memory

Live tile payload remains bounded by:

```text
one production tile
one verifier tile
one decoded tile
3 * 65536 bytes total
```

The dominant additional allocation is the complete rollback snapshot:

```text
TP=1 snapshot bytes 3763655360
TP=2 snapshot bytes 1881935712
```

Conservative VmHWM increment ceilings:

```text
TP=1 total/post-Torch/post-metadata:
  10485760 / 9961472 / 9699328 KiB

TP=2 total/post-Torch/post-metadata:
  6291456 / 6029312 / 5767168 KiB
```

## Independent Verification

The independent verifier must not import the gate or TinyLLMForge modules and
must not reuse the range helper. It reparses the safetensors header and uses
direct `os.pread` to reproduce:

```text
960 TP-local binding hashes
78 phase hashes
3 transaction hashes
320 TP reconstructions
72 rank-local alias partition checks
```

Binding-level TP reconstruction counts:

```text
replicated equality 79
axis0 concatenation 157
squeeze_axis0 concatenation 18
segmented Q/K/V reconstruction 18
axis1 row-interleave reconstruction 48
```

Total independent checks:

```text
1433
```

## Source Closure and Publication

The source closure is the authoritative 42-file four-layer closure plus:

```text
tools/qwen35_real_checkpoint_complete_transaction_preflight.py
```

Total:

```text
43 files
```

Authoritative outputs:

```text
complete_checkpoint_transaction_preflight.json
source_manifest.json
```

Three fresh rank workers and a separate finalizer are required. Failed or
superseded remote runs remain preserved. Partial local publication is
forbidden.

## Allowed Conclusion

Passing proves the entire real Qwen3.5 checkpoint target—embedding, all 24
decoder layers, and final norm—can participate in one deterministic, bounded,
source-bound, reversible tile transaction at TP=1 and TP=2.

It does not prove the production loader loop, assignment, `target.take()`,
candidate installation, inference correctness, production speed, cache
savings, GPU-memory savings, compression, or model quality.

## Completed Evidence

The source-bound live gate passed on `sitian@10.232.195.203`:

```text
run tag:
  qwen35-complete-checkpoint-20260728-065128

local evidence:
  experiments/qwen35_hybrid_state/
  qwen35-complete-checkpoint-20260728-065128/
```

Three fresh workers and the separate finalizer passed:

```text
TP=1 rank0 PID 3946836
TP=2 rank0 PID 3960911
TP=2 rank1 PID 3966499
```

Per-rank aggregate hashes:

```text
TP=1 rank0:
  4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963
TP=2 rank0:
  4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041
TP=2 rank1:
  5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca
```

The root endpoints were included. The TP=1 embedding hash was:

```text
b222b11204158144e369ae8fca02cab9cb63b0a8cde1dd59dd4d0c60690824ed
```

The final norm was replicated identically on all ranks:

```text
0df050ad8e61ea06a06b50c0eb0ae6975d97fb5712dfb0dec2d0b154c0676dd5
```

Observed total/post-Torch/post-metadata VmHWM increments remained below all
frozen ceilings:

```text
TP=1 rank0:
  8881980 / 8538600 / 8408632 KiB
TP=2 rank0:
  4681648 / 4338280 / 4208140 KiB
TP=2 rank1:
  4682516 / 4338204 / 4208668 KiB
```

Every row preserved the exact 26-phase binding-plan order, passed all 25
transition isolation checks, covered all 320 bindings and 296 unique
destinations, validated all 24 alias groups, left non-selected tensors zero,
restored every selected snapshot in reverse first-seen order, and ended with
all registered tensors zero. Loader, assignment, `target.take()`, model
forward, and attention forward counters were zero. CUDA remained false.

An independent verifier used only Python standard-library imports. It did not
import the gate or TinyLLMForge modules, reparsed the safetensors header, and
used direct `os.pread` to pass exactly:

```text
960 TP-local binding hashes
78 phase hashes
3 transaction hashes
320 TP reconstructions:
  replicated equality 79
  axis0 concatenation 157
  squeeze-axis0 concatenation 18
  segmented Q/K/V reconstruction 18
  axis1 row-interleave reconstruction 48
72 rank-local alias partition checks

1433 independent checks total
```

The remote inventory was exactly 43 staged source files plus the two
authoritative artifacts. CLI validation and the remote source-file hashes
passed after round trip.

Authoritative hashes:

```text
source tree:
  da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042
complete_checkpoint_transaction_preflight.json:
  7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176
source_manifest.json:
  9513c3d329b4bd310158416194673d5b60faa118983c88c674c984a9b9d6bd9e
```

Fresh local verification passed the complete gate plus the four-layer,
two-layer, layer0, five-transform bundle, one-tile, CPU materialization, meta
target, loader construction, metadata, reader, worker, candidate factory,
real binding, loader configuration, candidate loader, authorization, and
safety suites. Static closure, two-descriptor, forbidden-call, worker hard
rejection, `git diff --check`, and staged-file audits also passed.

## Next Safe Boundary

The next gate may exercise the production loader loop only as an explicitly
authorized, CPU-only, fail-closed transaction over a freshly constructed
candidate. It must retain the same source identity, tile schedule, coverage,
memory, and rollback evidence; still forbid `target.take()`, candidate
installation, CUDA, Engine wiring, publication, forward, and inference; and
must not alter schema-v2 canonical `NO_GO`.
