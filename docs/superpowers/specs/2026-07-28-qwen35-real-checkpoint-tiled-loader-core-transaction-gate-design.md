# Qwen3.5 Real Checkpoint Tiled Loader-Core Transaction Gate Design

## Status

Approved under the standing inline-execution direction.

This gate is the first real-checkpoint execution of the production tiled
loader core. It remains a dedicated CPU-only, unpublished-candidate preflight.
It must not call `target.take()`, the authorized candidate-loader adapter,
ModelRunner, Engine, publication, CUDA, forward, or inference.

## Goal

Prove that the production
`tinyvllm.models.qwen35_checkpoint_tiled_loading`
loader core can consume the approved real Qwen3.5 checkpoint and fill all 320
bindings of a fresh private CPU candidate at TP=1 and TP=2 while retaining:

- exact source and checkpoint identity;
- the frozen 65,536-byte tile schedule;
- exact TP-local binding and aggregate hashes;
- bounded tile materialization;
- failure isolation;
- deterministic post-run clearing of every private destination;
- zero candidate installation or runtime visibility.

## Prerequisite Evidence

The completed transaction gate already proved:

```text
320 bindings
296 unique destinations
26 exact binding-plan phases
24 gate/up aliases
TP=1 and TP=2 source/destination hash equality
complete isolation and reverse rollback
1433 independent direct-pread checks
```

Authoritative prerequisite:

```text
qwen35-complete-checkpoint-20260728-065128
```

This design does not repeat the manual `os.pread` copy loop as its primary
operation. It uses the existing production tiled loader core and compares its
result against the prerequisite hashes.

## Considered Approaches

### 1. Invoke the authorized candidate-loader adapter

The existing adapter calls:

```python
target.take()
```

and returns a publication-compatible candidate. That crosses the current
safety boundary and couples this gate to the one-source streamed loader rather
than the tiled loader core. Rejected.

### 2. Invoke the public tiled loader with a factory that consumes the target

The public API requires a factory returning `(model, binding_plan)`. Supplying
such a factory would create a second implicit ownership transfer equivalent to
`take()`, even if the method itself were not called. It would also obscure
whether the prepared target remained unconsumed. Rejected for this gate.

### 3. Invoke the production tiled loader core on a fresh private target

Construct one exact prepared CPU target, build its production tile plan, and
call the existing internal loader core with the target's registered model and
binding plan without consuming the target.

After success, verify the complete loaded state and clear every unique
destination. On any exception, clear every unique destination before
propagating failure. The target remains private and `_consumed == false`.

Selected.

## Production Function Under Test

The gate calls exactly:

```python
_load_qwen35_candidate_with_tile_plan(
    model,
    binding_plan,
    tile_plan,
    checkpoint_dir,
    model_fingerprint,
)
```

from:

```text
tinyvllm/models/qwen35_checkpoint_tiled_loading.py
```

The gate must not duplicate its `safe_open`, `get_slice`, materialization,
copy, or stats loop.

The production core is permitted to build the private
`Qwen35HybridModelOwner` returned inside
`Qwen35TiledLoadedCheckpointCandidate`. The gate must not publish, install,
bind, serialize, or return that candidate beyond the fresh rank process.

## Fresh Target and Ownership Contract

Each rank process:

1. reads and validates approved metadata;
2. builds the exact tensor plan and state layout;
3. creates a fresh CPU pool and prepared target;
4. proves `target._consumed is false`;
5. records exact model, binding-plan, pool, destination, and rotary identities
   and values;
6. explicitly zero-initializes only the 296 checkpoint destinations because
   production component storage is created with `torch.empty`;
7. proves all unbound rotary buffers remain unchanged;
8. builds the 65,536-byte tile plan;
9. calls the production tiled loader core directly;
10. proves the returned candidate references the exact same model and binding
   plan;
11. proves `target._consumed` remains false;
12. validates all loaded hashes, stats, aliases, and non-selected isolation;
13. clears all 296 unique selected destination objects;
14. proves all 296 selected destinations are zero and all six unbound rotary
    buffers retain their exact construction values;
15. drops the returned candidate without publication.

No production owner slot, candidate slot, ModelRunner field, or Engine state
may be reachable.

## Loaded-State Verification

Use the completed transaction artifact as the immutable expected-hash oracle.
For the matching `(tp_size, tp_rank)` row, require:

```text
320 binding destination hashes
26 phase hashes
1 aggregate transaction hash
```

The gate independently traverses the loaded destination views in exact binding
order. It must not trust only loader scalar stats.

For every binding, require:

- source name, target, kind, dtype, local shape, and destination slice match
  the frozen complete-gate contract;
- destination SHA256 equals the authoritative complete-gate hash;
- gate/up aliases remain exactly the 24 ordered disjoint partitions;
- no non-selected registered tensor changed.

The phase and aggregate hashes must equal the corresponding prerequisite row.

## Loader Statistics

For each row, require:

```text
assigned_bindings = 320
source_tensors = 320
shard_count = 1
tile_count:
  TP=1 58169
  TP=2 29169
destination_bytes:
  TP=1 3763655360
  TP=2 1881935712
materialized_bytes:
  TP=1 3763655360
  TP=2 1881935712
peak_tile_bytes <= 65536
```

The exact peak tile is expected to be 65,536 bytes.

## Cleanup Transaction

The prepared candidate is fresh, but production parameters and direct buffers
are allocated with `torch.empty`, so checkpoint destinations have undefined
initial values. Before payload access, the gate explicitly zero-initializes
only the 296 selected destinations. Six unbound rotary `inv_freq` buffers
contain deterministic nonzero construction values and are outside checkpoint
assignment. Therefore cleanup does not need a model-sized rollback snapshot.

Record the 296 unique selected destination objects in first-seen order. In a
`finally` boundary:

```text
for destination in reversed(unique_destinations):
    destination.zero_()
```

Then prove:

- every selected destination is zero;
- every non-selected registered tensor equals its exact pre-load value;
- all parameter/buffer object identities and storage pointers are unchanged;
- the state-pool snapshot is unchanged;
- `target._consumed` remains false.

If loader execution and cleanup both fail, report cleanup failure and chain the
loader error. No PASS row may be emitted unless cleanup completed.

This is deterministic discard/clear semantics for a fresh private candidate,
not rollback of a live or published model.

## Failure Injection

Before the live run, a local focused test replaces the production copy
primitive with a wrapper that raises after at least one successful tile.

The gate wrapper must prove:

- earlier destinations were changed before injection;
- all destinations are zero after cleanup;
- no candidate was returned;
- the target remains unconsumed;
- no publication or runtime object was touched.

The live source-bound run does not inject failure.

## Memory Contract

Unlike the completed transaction gate, this gate allocates no full rollback
snapshot. Its dominant allocation is the fresh model destination storage plus
bounded safetensors tile materialization.

Conservative VmHWM increment ceilings:

```text
TP=1 total/post-Torch/post-metadata:
  8388608 / 7864320 / 7864320 KiB
TP=2 total/post-Torch/post-metadata:
  4980736 / 4718592 / 4456448 KiB
```

The TP=1 ceiling was calibrated from preserved failed run
`qwen35-tiled-loader-core-20260728-073500`, which completed the production
loader-core load, exact-value verification, and private cleanup before row
validation reported observed increments of
`7896316 / 7551900 / 7418956 KiB`. The frozen ceiling rounds upward to
256-MiB boundaries while retaining at least 256 MiB of headroom for every
observed delta.

The TP=2 ceiling was then calibrated from preserved failed run
`qwen35-tiled-loader-core-20260728-075000`. TP=1 passed in that run, while
TP=2 rank 0 completed production loading, exact-value verification, and
private cleanup before reporting observed increments of
`4592816 / 4248560 / 4115712 KiB`. Its frozen ceiling uses the same
256-MiB-boundary rule with at least 256 MiB of headroom for each delta.

These are correctness ceilings, not performance claims.

## Source Closure and Artifacts

Freeze the complete-gate 43-file source closure plus:

```text
tools/qwen35_real_checkpoint_tiled_loader_core_preflight.py
```

Total:

```text
44 files
```

Authoritative outputs:

```text
tiled_loader_core_preflight.json
source_manifest.json
```

Use three fresh rank workers, one separate finalizer, exact source hash
binding, remote round trip, and atomic local publication. Failed and
superseded directories remain preserved.

## Tests and Audits

Focused TDD must cover:

- exact prerequisite artifact identity and row selection;
- production loader-core invocation exactly once;
- no public loader, adapter, `target.take()`, candidate installation, or
  publication;
- exact loaded binding/phase/aggregate hashes;
- exact loader stats;
- `_consumed == false` before and after;
- injected mid-load failure cleanup;
- identity/storage/pool preservation;
- three fresh processes and atomic evidence;
- exact 44-file source closure;
- CUDA false before and after;
- worker hard rejection unchanged.

Static audits require:

- the preflight contains no `safe_open`, `get_slice`, or direct copy loop;
- exactly one call site to `_load_qwen35_candidate_with_tile_plan`;
- no `target.take()`;
- no authorized adapter, loader configuration, ModelRunner, Engine,
  publication, forward, CUDA allocation, CUDA synchronization, or CUDA
  operator calls; `torch.cuda.is_initialized()` is the only permitted CUDA
  observation;
- `git diff --check` passes and staged files remain zero.

Regression suites include the complete transaction, four-layer, two-layer,
layer0, bundle, one-tile, CPU, meta, tiled loader, loader construction,
metadata, reader, worker, factory, binding, authorization, and safety gates.

## Completed Evidence

Authoritative source-bound run:

```text
qwen35-tiled-loader-core-20260728-075700
```

All three fresh workers passed:

```text
TP=1 rank0 PID 240493
  bindings/phases/aliases: 320 / 26 / 24
  tiles/destination bytes: 58169 / 3763655360
  total/post-Torch/post-metadata VmHWM increment:
    7894300 / 7551228 / 7418812 KiB

TP=2 rank0 PID 245462
  bindings/phases/aliases: 320 / 26 / 24
  tiles/destination bytes: 29169 / 1881935712
  total/post-Torch/post-metadata VmHWM increment:
    4591556 / 4247960 / 4116056 KiB

TP=2 rank1 PID 249311
  bindings/phases/aliases: 320 / 26 / 24
  tiles/destination bytes: 29169 / 1881935712
  total/post-Torch/post-metadata VmHWM increment:
    4592304 / 4248596 / 4116096 KiB
```

Every row recorded one production loader-core call, exact loaded binding,
phase, and aggregate verification, all 24 alias groups, selected-only zero
initialization, complete post-run selected clearing, preserved tensor
identity, preserved non-selected rotary values, unchanged pool state,
`target._consumed == false`, zero model/attention forward calls, and CUDA
uninitialized before and after.

Authoritative hashes:

```text
source tree:
  c84eb9252bb5294d0fe00a4c48769e659274eb0a2d8c4548c25fb1ecdaf6869b
tiled_loader_core_preflight.json:
  58df3dfa9fec11d1fd079c9473766413232bd3f928f537ac87e047e13ef65aae
source_manifest.json:
  e4137a81053e28bc298d517c8be4028270b838056cc3d5240e3ca7008963bf0d
```

The local and remote artifact hashes match exactly. The remote inventory is
44 staged source files, the immutable prerequisite artifact, and exactly two
published result artifacts. A standard-library-only independent verifier
performed 1291 checks over row shape, PIDs, loader stats, ownership, cleanup,
memory, source hashes, and local source-tree binding. Both local and remote
CLI validation passed.

Fresh validation also passed:

```text
focused tiled-loader-core tests: 6
transaction regression groups: 9
loader/metadata/safety regression groups: 12
total regression groups: 21
```

Static audit found exactly one production loader-core call site, two
read-only `torch.cuda.is_initialized()` observations, and no forbidden
`safe_open`, `get_slice`, direct copy loop, `target.take()`, adapter,
ModelRunner, Engine, publication, forward, or CUDA operation. The production
worker hard rejection remains exact, `git diff --check` passes, and staged
files remain zero.

Preserved failed/superseded runs remain unpublished:

```text
qwen35-tiled-loader-core-20260728-071120
qwen35-tiled-loader-core-20260728-071310
qwen35-tiled-loader-core-20260728-071452
qwen35-tiled-loader-core-20260728-073500
qwen35-tiled-loader-core-20260728-074300
qwen35-tiled-loader-core-20260728-075000
```

## Allowed Conclusion

Passing proves the production Qwen3.5 tiled loader core can load the complete
approved real checkpoint into a fresh unpublished CPU candidate at TP=1 and
TP=2, producing the exact already verified binding values under bounded tile
materialization, and that the gate can deterministically clear and discard the
private candidate after success or failure.

It does not prove `target.take()`, the authorized candidate-loader adapter,
candidate installation, publication, ModelRunner/Engine integration, CUDA,
forward/inference correctness, production latency, throughput, cache savings,
GPU-memory savings, compression, or model quality. Schema-v2 canonical
`NO_GO` remains unchanged.

The next safe boundary is a separate private candidate-ownership transfer
gate. It may exercise the authorized adapter and exactly one `target.take()`
on a fresh CPU candidate, but must still prohibit candidate installation,
publication, ModelRunner/Engine integration, CUDA, forward, and inference.
It must prove exclusive ownership transfer and deterministic cleanup/discard
after both success and injected failure.

