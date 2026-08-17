# Qwen3.5 Recurrent INT8 Prefix Snapshot Codec Design

## Objective

Define the first P2 physical-cache compression candidate for the existing
Qwen3.5 hybrid-prefix snapshot cache.

This design is architecture-first. It does not modify the runtime, authorize a
GPU run, or claim compression, memory, speed, or quality benefit. Its purpose
is to freeze the exact production tensor inventory, select one representation,
define its runtime integration, and preregister the correctness, cache,
performance, memory, failure-atomicity, and independent-authority gates that
must pass before P2 may be enabled or described as beneficial.

The selected first candidate is:

```text
exact BF16 convolution state
+ per-row symmetric INT8 recurrent state
+ FP32 row scales
```

It deliberately does not reuse the failed Light Doc Cache selector and does
not combine quantization with low rank, sparsity, layer sharing, or int4.

## Prerequisite Boundary

Runtime integration remains blocked until all of the following are complete:

1. a fresh source-bound TP4 correctness campaign passes in strict order:
   `tp4_root_logit -> cached_continuation -> engine_correctness`;
2. the strict-exclusive P0 recompute versus P1 exact-restore benchmark runs;
3. P1 establishes the full-fidelity numerical and cache-cost reference;
4. a separately source-bound codec calibration gate passes;
5. a new P2 benchmark schema and authorization are approved.

Every runtime or authority run must bind the current source tree and its own
fresh prerequisite receipt. Source changes invalidate prior authorization,
ports, nonces, tags, artifacts, and benchmark results. A resource-blocked
preflight is not a failed benchmark and does not authorize a later run.

Passing a CPU codec test would prove only codec behavior. It would not prove
production physical bytes, CUDA allocator savings, task quality, latency, or
throughput.

## Current Snapshot Inventory

The production ModelRunner constructs the Qwen3.5 hybrid state layout with:

```text
tensor parallel size: 4
linear-attention layers: 18 of 24
linear key heads: 16 global, 4 local
linear value heads: 16 global, 4 local
key head dimension: 128
value head dimension: 128
convolution kernel dimension: 4
speculative tokens: 1
convolution dtype: BF16
recurrent dtype: FP32
```

For every rank-local prefix snapshot, the cache owns one convolution tensor
and one recurrent tensor for each linear-attention layer.

| Field | Count | Per-tensor shape | Dtype | Bytes/tensor | Bytes/rank |
|---|---:|---:|---:|---:|---:|
| `convolution_states` | 18 | `[1536, 4]` | BF16 | 12,288 | 221,184 |
| `recurrent_states` | 18 | `[4, 128, 128]` | FP32 | 262,144 | 4,718,592 |
| total tensor payload | 36 | - | mixed | - | 4,939,776 |

Therefore:

```text
full-fidelity bytes per rank:  4,939,776 = 4.7109375 MiB
full-fidelity bytes across TP4: 19,759,104 = 18.84375 MiB
convolution share:              4.4776119403%
recurrent share:               95.5223880597%
```

These are deterministic layout bytes for one snapshot. They do not include
Python object metadata, token tuples, block identities, allocator overhead, or
temporary publication/restore workspaces.

### Identity Metadata

Every snapshot also binds:

- exact prefix token IDs;
- exact block ID, generation, and hash tuples;
- model and layout fingerprints;
- tensor-parallel size;
- source dtype;
- block size and token boundary.

The P2 representation must not weaken or reinterpret any of these identity
checks.

### Current Physical Ownership

`Qwen35HybridPrefixSnapshotCache` currently:

1. gathers one live source row from every linear layer;
2. creates detached contiguous full-fidelity clones;
3. content-addresses each complete tensor by dtype, shape, device, and SHA-256;
4. confirms byte equality before sharing a canonical tensor;
5. reports logical referenced bytes separately from unique physical tensor
   bytes.

Interning is rank-local and exact. It can share byte-identical tensors between
entries, but it cannot reduce the first unique snapshot and does not share
storage across TP ranks.

### Restore Consumer

On a hit, the current cache:

1. validates the exact key, tokens, blocks, destination leases, shapes, dtypes,
   and devices;
2. expands each stored single-row tensor across the destination batch;
3. calls `Qwen35CrossLayerStateTransaction.commit()`;
4. writes the original BF16 convolution and FP32 recurrent tensors into the
   live `HybridStateTensorPool`;
5. rolls every destination layer back if a copy fails.

A compressed representation therefore cannot be passed directly to the live
pool. It must decode to the original shape, dtype, and device before the
transaction validates or writes any destination.

## Candidate Comparison

### A. Recurrent-Only Per-Row INT8

Selected.

Keep convolution tensors unchanged. For each recurrent tensor
`R[head, value_row, key_dim]`, quantize every length-128
`R[head, value_row, :]` row independently with one FP32 scale.

Advantages:

- attacks the component responsible for 95.5% of snapshot bytes;
- preserves convolution history exactly;
- has bounded, local error and simple deterministic metadata;
- encoding and decoding are linear-time elementwise operations;
- avoids SVD and rank-selection cost during publication;
- naturally supports independent calibration by layer, head, and row;
- retains the current layer ordering and restore transaction boundary.

Theoretical rank-local encoded bytes:

```text
unchanged convolution:
  18 * 12,288 = 221,184

INT8 recurrent payload:
  18 * 4 * 128 * 128 = 1,179,648

FP32 row scales:
  18 * 4 * 128 * 4 = 36,864

total:
  1,437,696 bytes = 1.37109375 MiB per rank
  5.484375 MiB across TP4

full-fidelity / encoded:
  3.4358974359x

theoretical tensor-payload saving:
  70.8955223881%
```

This is a static representation estimate, not a measured production result.

### B. Recurrent-Only Low Rank

Deferred.

Each `[128, 128]` matrix per value head could be represented by BF16 factors.
For example, rank 32 has approximately the same static footprint as the
selected INT8 representation, while lower ranks have a smaller footprint.

It is not selected first because:

- publication requires decomposition rather than a simple reduction and pack;
- rank must be calibrated against prompt, layer, head, and state spectrum;
- approximation error is globally coupled across each matrix;
- factor reconstruction adds matrix multiplication to the restore path;
- the existing external rank-24/rank-32 process observations have no owned
  source, artifacts, verifier, or TinyLLMForge quality result.

Those process observations are resource history only and cannot be reused as
evidence for the reported 2.57x compression result.

### C. Recurrent INT4

Deferred.

INT4 offers a larger static ratio but introduces packing, narrower dynamic
range, more sensitive scale grouping, and a higher quality risk. The existing
ordinary KV int4 path and weight int4 helpers use different tensor semantics
and are not a recurrent-state codec.

INT4 may be evaluated only after the INT8 authority establishes the error and
performance budget.

### D. Light Doc Cache or Gist Selector

Rejected for this candidate.

The canonical Light Doc Cache multi-target result is `NO_GO`:

```text
holdout wins:                  0 / 8
aggregate relative change:   -105.13%
worst relative regression:    322.98%
logical saving:               approximately 17.5%
```

Its selector removes or reconstructs token-derived KV content. The Qwen3.5
hybrid snapshot contains fixed convolution history and recurrent matrices with
different restore consumers. Reusing or retuning that selector would neither
match this representation nor address its prior quality failure.

## Selected Codec Contract

### Encoded Recurrent Tensor

For one full-fidelity recurrent tensor:

```python
R.shape == (local_value_heads, value_head_dim, key_head_dim)
R.dtype == torch.float32
```

The encoded representation contains:

```text
values:
  shape [local_value_heads, value_head_dim, key_head_dim]
  dtype int8

scales:
  shape [local_value_heads, value_head_dim]
  dtype float32

source_shape:
  exact original shape

source_dtype:
  torch.float32

codec:
  qwen35_recurrent_symmetric_int8_per_row_v1
```

For every row `x`:

```text
amax = max(abs(x))

if amax == 0:
  scale = 1.0
  q = 0
else:
  scale = amax / 127.0
  q = clamp(round(x / scale), -127, 127)

decoded = float32(q) * scale
```

The value `-128` is intentionally unused so the positive and negative finite
range is symmetric.

The encoder must reject:

- non-FP32 recurrent input;
- an unexpected shape;
- NaN or infinity in the source;
- a non-finite or non-positive scale;
- any output outside the declared int8 range.

The decoder must reject malformed metadata, shape mismatch, dtype mismatch,
non-finite scales, and non-finite decoded values.

### Convolution Representation

Convolution tensors remain detached, contiguous, exact BF16 clones. No
quantization, casting, sparsity, truncation, or projection is permitted in the
first P2.

### Snapshot Representation

The P2 snapshot must distinguish:

```text
full_fidelity_logical_bytes
encoded_physical_bytes
codec_metadata_bytes
temporary_encode_workspace_bytes
temporary_decode_workspace_bytes
```

The existing `storage_bytes` meaning must not be silently changed. A new
snapshot type or explicit versioned fields are required so P1 accounting
remains stable.

The cache byte limit for P2 must charge actual owned encoded payload and scale
storage. It must not charge the hypothetical FP32 reconstructed tensor as
resident cache storage.

### Encoded Interning

Interning remains exact over the encoded immutable representation:

- convolution tensors use the current exact tensor interning;
- INT8 value tensors may share only after exact metadata and byte equality;
- FP32 scale tensors may share only after exact metadata and byte equality;
- a digest match alone is never sufficient;
- decoded floating-point equality is not an interning criterion.

Codec identity must be part of the intern key so future grouping or int4
profiles cannot alias this representation.

## Publication Boundary

P1 `exact_restore` remains the default representation and its behavior,
accounting, namespace, and authority contract remain unchanged. P2 adds a
separate, explicit, default-off representation:

```text
exact BF16 convolution
+ INT8 recurrent values
+ FP32 per-row scales
```

P1 and P2 entries must occupy representation-specific cache namespaces. An
exact entry must never be interpreted as an INT8 entry, and an INT8 entry must
never satisfy an exact lookup. Codec identity, representation version, model
identity, layout identity, prefix identity, and TP identity are all part of the
cache key and validation boundary.

The P2 publication transaction must remain fail-closed:

1. gather and validate all full-fidelity source tensors;
2. clone exact convolution tensors;
3. encode the recurrent state for all 18 linear-attention layers into private
   candidates;
4. validate every encoded payload, scale, layer identity, shape, dtype,
   device, codec identity, and accounting record;
5. compute full-fidelity, encoded, metadata, and workspace byte counters;
6. reject an oversize candidate before visible cache mutation;
7. acquire exact intern references for all encoded components;
8. create the complete private snapshot;
9. precommit, finalize, seal, or roll back through the existing distributed
   publication phases.

The complete 18-layer candidate becomes visible atomically only after every
layer and every identity/accounting check succeeds. Failure must release all
private candidates and newly acquired intern references without overwriting a
previous valid entry. No visible entry may contain a mixture of old and new
codec components, exact and INT8 recurrent layers, or a partial layer set.

Encoding must not mutate the live source pool or replace the P1 publication
path. The first implementation remains default-off and independently
selectable.

## Restore Boundary

The P2 restore transaction must acquire an immutable reader lease and decode
before any live state mutation:

1. validate snapshot identity and codec version;
2. validate all encoded convolution/recurrent components;
3. allocate private FP32 recurrent decode candidates;
4. decode all 18 recurrent layers;
5. validate every decoded shape, dtype, device, and finite-value invariant;
6. assemble the unchanged BF16 convolution plus decoded FP32 recurrent
   candidate tuple;
7. call the existing cross-layer transaction once;
8. release temporary decode candidates after success or failure.

The first implementation must decode all layers before calling
`Qwen35CrossLayerStateTransaction.commit()`. Streaming decode directly into
live pool rows is rejected because a late malformed layer or decode failure
could expose a partial restore.

The existing transaction may still clone destination rows for rollback.
Temporary restore memory must therefore be measured rather than hidden.

If corruption or decode failure occurs before the cross-layer commit, the
entry is quarantined, the reader lease is released, and the request follows a
normal cache miss and recomputes the prefix. This is a correctness-preserving
miss, not an INT8 success and not a permitted benchmark fallback. If the
cross-layer commit itself fails, the existing transaction must restore every
destination layer to its pre-commit value and fail closed.

The runtime must never:

- expose a partial restore;
- mix exact and INT8 recurrent layers;
- commit any layer before all 18 layers have decoded and validated;
- silently lower precision;
- silently retry through `exact_restore` and count the request as P2;
- hide a quarantine, corruption, rollback, miss, or fallback from accounting.

The first implementation deliberately does not add a fused decode/restore
kernel and does not stream decoded rows directly into the live pool. Private
FP32 staging is required and its allocated/reserved CUDA workspace is part of
the measured result.

## Calibration Gate

Before runtime integration, a source-bound offline calibration gate must use
real P1 full-fidelity snapshots captured from the approved Qwen3.5 model and
frozen workloads.

The calibration artifact must report, by layer and aggregate:

- source and encoded tensor hashes;
- exact shapes and dtypes;
- full-fidelity bytes;
- INT8 payload bytes;
- FP32 scale bytes;
- compression ratio including metadata;
- zero-row count;
- saturation count;
- maximum absolute error;
- mean absolute error;
- RMSE;
- relative L2 error;
- cosine similarity;
- finite-value checks;
- encode and decode time, separately;
- peak temporary allocated and reserved CUDA bytes when run on CUDA.

Calibration is `NO_GO` on any malformed value, NaN, infinity, shape drift,
undeclared saturation, accounting mismatch, or missing workload/layer.

Numerical thresholds must be preregistered from a read-only pilot and frozen
before the canonical calibration run. The canonical result must not tune
grouping or thresholds against its own outputs.

## P2 Runtime Authority

After calibration passes and P1 is authoritative, a new schema-v2 benchmark
may add exactly one profile:

```text
recurrent_int8_per_row
```

The matrix retains:

```text
P0 recompute
P1 exact_restore
P2 recurrent_int8_per_row
```

P2 uses P1 as both its numerical and cache-cost reference.

The three profiles must use the same model, source SHA, prompt corpus, sampling
configuration, concurrency, GPU set, repetition schedule, and workload
identity. Correctness runs are strictly serial. P2 must use fresh
prerequisites, fresh preflight, one-time authorization, fresh ports, fresh
nonce, and fresh artifact paths.

Required producer evidence includes:

- all P1 prerequisite and workload bindings;
- candidate-specific codec identity;
- actual resident encoded bytes including scale metadata;
- unique physical bytes separately from logical referenced bytes;
- temporary encode/decode CUDA peaks;
- per-rank and TP4 aggregate bytes;
- same-budget entry capacity;
- cache hits, misses, evictions, and validation failures;
- quarantines, rollbacks, corruption events, partial-restore attempts, and
  fallbacks;
- continuation token outputs;
- final logits;
- TTFT, decode latency, throughput, initialization, allocated memory, and
  reserved memory.

Performance, memory, and cache claims are considered only after strict
correctness passes.

The physical compression decision must use measured P2 bytes divided by
measured P1 bytes. The static 3.4359x estimate is not an acceptance result.

### P2 GO Gates

P2 is `GO` only when every gate below passes:

#### Correctness

```text
continuation tokens:
  exactly identical to exact_restore

final logits:
  atol = 2e-5
  rtol = 0
```

The threshold is fixed and must not be widened after observing a canonical
run. Any token or logit failure is `NO_GO_CORRECTNESS`.

#### Physical cache and capacity

```text
measured unique physical snapshot bytes:
  int8_restore / exact_restore <= 0.40

same configured cache-byte budget:
  int8_restore entry capacity / exact_restore entry capacity >= 2.5
```

The numerator includes resident INT8 values, exact BF16 convolution tensors,
FP32 scales, and representation-owned metadata. It excludes temporary decode
workspace from resident cache bytes but reports that workspace separately.
Allocator overhead and interning effects must be reported rather than inferred
from static tensor shapes. Failure is `NO_GO_CACHE`.

#### Performance and memory relative to exact restore

```text
W1 median TTFT ratio <= 1.03
W1 every-repetition TTFT ratio <= 1.05
W2 median TTFT ratio <= 1.03
W2 every-repetition TTFT ratio <= 1.05
W3 throughput ratio >= 0.98
steady-state peak CUDA reserved ratio <= 1.05
```

These gates prevent cache savings from being purchased through a material
regression against the existing exact path.

#### Performance relative to recompute

```text
W1 median TTFT ratio <= 0.85
W2 median TTFT ratio <= 0.75
W3 throughput ratio >= 1.15
decode-latency ratio <= 1.02
```

These retain the P1 speed-versus-recompute requirements. Failure of either
performance group is `NO_GO_PERFORMANCE`.

#### Runtime safety

The canonical matrix must contain no OOM, undeclared eviction, corruption,
fallback, partial restore, mixed representation, missing layer, or failed
rollback. Any occurrence is `NO_GO_RUNTIME_SAFETY`.

### Result Classification

The closed result vocabulary is:

- `GO`;
- `NO_GO_CORRECTNESS`;
- `NO_GO_RUNTIME_SAFETY`;
- `NO_GO_CACHE`;
- `NO_GO_PERFORMANCE`;
- `BLOCKED_RESOURCES`;
- `INVALID_ARTIFACT`.

All observed failure reasons remain in the artifact. When one primary
classification is required, precedence is correctness, runtime safety, cache,
then performance. `BLOCKED_RESOURCES` means no canonical P2 conclusion was
formed. Missing, malformed, unbound, tampered, or incomplete evidence is
`INVALID_ARTIFACT`.

### Independent Authority

The producer records raw per-repetition results, token and logit evidence,
snapshot inventories, cache events, memory observations, process identity, and
run receipts. It does not decide `GO`.

A separate verifier reads the artifact from disk and independently recomputes:

- token equality and logit differences;
- unique physical byte totals and ratios;
- same-budget capacity and ratio;
- TTFT, throughput, decode, and CUDA-memory statistics;
- failure-event counts;
- every threshold and final classification.

The verifier must not trust producer summaries. The artifact uses a closed
schema and binds at least:

- source SHA and dirty-tree identity policy;
- model and tokenizer identity;
- codec ID `qwen35_recurrent_symmetric_int8_per_row_v1`;
- exact representation version;
- GPU inventory and TP topology;
- workload and prompt hashes;
- sampling parameters and random seeds;
- ports, nonce, run tag, and artifact path;
- prerequisite, preflight, authorization, and execution receipt hashes;
- producer and verifier versions.

Unknown fields, missing raw evidence, identity mismatch, hash mismatch,
inventory mismatch, threshold mismatch, or an unrecognized version must fail
closed. Tampering with a manifest, receipt, inventory, tensor hash, raw value,
summary, or threshold must make the verifier reject the artifact.

INT8 remains default-off even after an experimental `GO`; enabling it requires
an explicit CLI or configuration choice. The runtime must not automatically
switch from exact restore to INT8, or from a failed INT8 restore to exact
restore, without exposing and accounting for that choice.

## Test Contract for a Future Implementation

The implementation plan must include RED/GREEN coverage at the following
levels.

### Codec tests

1. deterministic per-row encode/decode;
2. all-zero rows;
3. positive/negative extrema and symmetric range;
4. source clone isolation;
5. NaN and infinity rejection;
6. malformed scale, payload, shape, dtype, and codec rejection;
7. exact convolution preservation;
8. decoded recurrent FP32 shape and device restoration;
9. CPU reference codec versus CUDA codec agreement.

### Cache and transaction tests

1. all 18 layers publish atomically;
2. no old entry replacement before candidate completion;
3. publication rollback after partial encoded interning;
4. immutable reader lease across restore and concurrent eviction;
5. no live-pool mutation before all layers decode;
6. late decode failure preserving all destination rows;
7. commit failure rolling back every destination layer;
8. exact encoded-content interning and digest collision safety;
9. exact and INT8 namespace isolation;
10. rejection of exact/INT8 mixed layers and partial snapshots;
11. byte limits based on encoded owned storage;
12. separate logical, encoded, metadata, and workspace accounting;
13. cache clear, invalidation, replacement, LRU, quarantine, and distributed
    rollback;
14. P1 behavior and counters unchanged when the codec is disabled.

### Runtime integration tests

1. real publish, hit, miss, replacement, eviction, and quarantine paths;
2. corruption before commit becoming an accounted cache miss and recompute;
3. no silent fallback to exact restore or lower precision;
4. all runtime events reporting representation and codec identity;
5. FP32 staging allocation and release on success and failure;
6. source-bound real-snapshot calibration.

### Canonical authority tests

1. P0/P1/P2 W1/W2/W3 matrix with matched configuration;
2. strict token and logit gates;
3. measured physical bytes and same-budget capacity gates;
4. exact-relative and recompute-relative performance gates;
5. CUDA allocated, reserved, and workspace accounting;
6. closed-schema producer receipts;
7. independent verifier recomputation;
8. tamper rejection for manifests, receipts, tensor inventory, raw values,
   summaries, and thresholds;
9. resource-blocked and invalid-artifact classifications that form no benefit
   claim.

## Claim Boundary

This design establishes:

- the exact current TP4 snapshot tensor inventory;
- that recurrent FP32 tensors account for 95.5% of the tensor payload;
- that recurrent-only per-row INT8 is the lowest-complexity first P2;
- a static representation estimate of 3.4359x including FP32 row scales;
- a default-off dual-representation runtime architecture;
- atomic publication and restore failure semantics;
- fixed correctness, cache, capacity, performance, memory, and safety gates;
- the required accounting, calibration, producer, verifier, and tamper
  boundaries.

It does not establish:

- runtime integration of the existing codec;
- production cache reduction;
- GPU allocator savings;
- task-quality retention;
- accuracy preservation;
- latency or throughput improvement;
- the reported 2.57x external result;
- any int4, low-rank, sparse, Gist, or layer-sharing benefit.

Only a canonical artifact accepted as `GO` by the independent verifier may
support the bounded statement that, for the bound model, hardware, workloads,
source, configuration, and thresholds, P2 preserved the observed token/logit
contract, reduced measured unique physical snapshot bytes, increased
same-budget capacity, and met the preregistered performance and memory gates.

Even a canonical `GO` does not establish mathematical losslessness, universal
prompt equivalence, generalization to another model or topology, production
fleet stability, CPU-offload benefit, fused-kernel benefit, or any unmeasured
quality property. Offline calibration, unit tests, the theoretical 3.4359x
ratio, a smoke run, or a producer-authored summary cannot independently support
a production benefit claim.

