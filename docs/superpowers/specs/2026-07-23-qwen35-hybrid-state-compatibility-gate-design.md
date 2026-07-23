# Qwen3.5 Hybrid-State Compatibility Gate Design

Date: 2026-07-23

## Status

This document designs an architecture-first compatibility gate for adding
Qwen3.5-style hybrid recurrent state to TinyLLMForge.

The gate is intentionally narrower than native model integration. It must first
establish a source-bound reference execution, an independently reconstructable
state-memory ledger, and exact state-lifecycle semantics. A compatibility
`GO` means that a later native TinyLLMForge implementation has a complete and
testable contract. It does not mean that TinyLLMForge already runs Qwen3.5, that
recurrent-state compression is safe, or that any kernel is faster.

The user approved compatibility and physical state-memory accounting as the
first-stage `GO` criterion. This stage does not require a latency or throughput
improvement.

## Objective

Determine whether Qwen3.5-2B's hybrid attention state can be represented safely
by a future TinyLLMForge cache abstraction without forcing recurrent state into
the current token-indexed Qwen3 KV-block layout.

The gate must:

1. bind every result to one immutable Qwen3.5-2B model snapshot, tokenizer
   snapshot, probe source snapshot, Python environment, and GPU environment;
2. verify the model's actual layer schedule rather than assuming that every
   layer uses the same attention mechanism;
3. identify every persistent tensor needed to continue generation after
   prefill and after each decode step;
4. distinguish fixed-size linear-attention recurrent and convolution state from
   sequence-growing full-attention KV state;
5. record tensor shape, dtype, logical bytes, unique physical storage bytes,
   device, layer, request, state role, and lifetime;
6. prove single-request continuation equivalence across one-shot, chunked
   prefill, and token-by-token decode paths;
7. prove multi-request state isolation under interleaved decode, request
   completion, slot reuse, and explicit reset;
8. define an engine-facing state contract that is independent of Transformers
   object identity and Python container layout;
9. report real CUDA allocated and reserved memory in addition to tensor byte
   accounting;
10. classify unavailable model artifacts, insufficient disk, missing required
    runtime support, or unsupported reference semantics as `INCOMPLETE`, not as
    a failed optimization;
11. earn compatibility `GO` only when the complete frozen correctness and
    state-accounting domain passes independent verification;
12. keep recurrent-state compression and sparse-kernel performance as separate
    later gates.

## Non-Goals

This stage does not:

- add Qwen3.5 to `tinyvllm/models/`;
- modify `ModelRunner`, the scheduler, block tables, or production cache
  allocation;
- download model weights or install dependencies before the implementation
  plan authorizes an exact, source-bound procedure;
- delete or clean any remote files to create disk space;
- use another user's model cache or modify the remote checkout;
- implement FP16, INT8, INT4, low-rank, token-sparse, or layer-shared state
  compression;
- combine Qwen3.5 with the previous Light Doc Cache selector, Attention
  Matching, KV quantization, KV offload, speculative decoding, or CUDA Graphs;
- replace a model-native linear-attention layer with an unrelated sparse
  attention approximation;
- treat an isolated Triton or CUDA kernel benchmark as an engine speedup;
- claim quality retention from logit similarity alone;
- claim physical GPU-memory reduction from logical tensor byte calculations;
- update README performance claims or enable a new default path.

## Current Evidence and Constraints

TinyLLMForge currently has one production model path:

```text
ModelRunner
  -> Qwen3ForCausalLM
  -> Qwen3DecoderLayer
  -> QWen3Attention
  -> token-indexed K/V cache
```

`ModelRunner.allocate_kv_cache()` allocates one homogeneous cache using:

```text
[K or V, layer, block, token offset, KV head, head dimension]
```

It derives capacity from `num_hidden_layers`, `num_key_value_heads`,
`head_dim`, block size, and a uniform KV dtype. Quantization, KV offload,
Quest summaries, block tables, and attention modules all assume this
token-indexed layout.

The Qwen3.5-2B configuration previously inspected for this design describes a
24-layer hybrid schedule:

```text
18 linear-attention layers
6 full-attention layers
full_attention_interval = 4
```

The linear layers use recurrent Gated DeltaNet state plus convolution state.
The relevant configuration fields observed during design exploration were:

```text
linear_num_key_heads = 16
linear_num_value_heads = 16
linear_key_head_dim = 128
linear_value_head_dim = 128
linear_conv_kernel_dim = 4
mamba_ssm_dtype = float32
```

These values are expectations only. The gate must read and freeze them from
the acquired model snapshot and reject a snapshot that does not match the
declared canonical domain.

The source-bound remote execution environment is:

```text
host: sitian@10.232.195.203
GPU: CUDA_VISIBLE_DEVICES=0 only
Python: /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
SSH ControlMaster: /tmp/ssh-sitian-10.232.195.203
```

The latest read-only preflight observed:

```text
torch: 2.4.1+cu121
transformers: 5.8.1
fla: absent
causal_conv1d: absent
Qwen3.5-2B weights: absent from the checked cache path
free space under /data00/home/sitian: approximately 9.41 GiB
```

The earlier estimate of approximately 295 GiB is stale and must not be used.
Disk availability must be measured again immediately before any authorized
artifact acquisition.

The previous Light Doc Cache multi-target route and the decode cross-layer
residency planner are already canonical `NO_GO` results. This gate must not
repeat either method or mix their claims into Qwen3.5 evidence.

## Alternatives Considered

### 1. Recommended: Reference-First Hybrid-State Contract

Run the official reference model in an isolated probe, instrument every
persistent generation state tensor, prove continuation and request isolation,
then export a framework-neutral state schema and memory ledger.

Advantages:

- establishes the semantics TinyLLMForge must preserve before engine edits;
- separates model compatibility from kernel availability and speed;
- exposes whether fixed-size recurrent state actually creates a worthwhile
  memory opportunity;
- provides exact fixtures for later native implementation and compression;
- avoids forcing heterogeneous state into the existing homogeneous KV cache.

Risks:

- the official fallback may be slow without optimized linear-attention
  dependencies;
- Python hooks can miss hidden or aliased storage unless the independent
  verifier reconstructs the ledger;
- model acquisition may be blocked by current disk availability.

### 2. Direct Native TinyLLMForge Integration

Add Qwen3.5 model classes, a new cache type, scheduler support, and kernels in
one implementation.

This is rejected for the first stage. A mismatch could come from model
translation, cache lifecycle, kernel math, batching, or weight loading, making
failures difficult to attribute. It would also create substantial production
surface before the required state contract exists.

### 3. Kernel-First Sparse Linear-Attention Benchmark

Benchmark a Gated DeltaNet or sparse linear-attention kernel independently,
then infer likely engine speedup.

This is rejected as the compatibility gate. Kernel speed does not prove model
semantic compatibility, request-state isolation, scheduler integration, or
end-to-end memory savings. It becomes a valid separate gate only after the
state contract is `GO`.

## Decision

Implement Alternative 1.

The first implementation is an isolated reference probe, frozen contract,
remote runner, and independent verifier under `tools/` and `experiments/`.
Production engine files remain unchanged.

## Canonical Model and Environment Binding

### Model Identity

The canonical run must record:

- model repository or artifact source;
- resolved immutable revision;
- local canonical path;
- SHA256 for `config.json`, tokenizer metadata, generation metadata, index
  files, and every weight shard;
- total weight bytes;
- Transformers class names selected by `AutoConfig` and `AutoModel`;
- `trust_remote_code` value;
- tokenizer class and vocabulary size;
- model dtype requested and actual parameter dtypes;
- complete layer-type schedule.

Mutable aliases such as `main`, `latest`, or an unpinned cache directory are not
enough. If the acquisition mechanism cannot provide an immutable revision and
file hashes, the run is `INCOMPLETE`.

### Probe Source Identity

The remote runner must create a source snapshot from the local approved commit
and record:

- local branch and commit;
- dirty-state rejection;
- SHA256 of every staged probe, contract, runner, and verifier file;
- remote staged-file SHA256;
- exact command lines;
- stdout and stderr paths;
- process exit codes.

The runner must not use untracked remote leftovers as dependencies.

### Runtime Identity

Each process must record:

- host and user;
- GPU name and UUID;
- driver and CUDA runtime versions;
- Python executable and version;
- PyTorch and Transformers versions;
- availability and versions of `fla`, `causal_conv1d`, Triton, and
  FlashAttention;
- environment variables affecting model execution;
- `CUDA_VISIBLE_DEVICES`;
- fresh `TINYVLLM_DIST_PORT` and `MASTER_PORT`, even when the reference process
  does not initialize TinyLLMForge distributed execution.

No retry is allowed except when stderr contains the exact string
`EADDRINUSE`, with at most three attempts and fresh ports for every attempt.

## Artifact and Disk Preflight

The runner first performs a read-only preflight:

1. inspect the approved model cache roots for a complete immutable snapshot;
2. verify shard/index completeness and hashes if a snapshot exists;
3. measure free bytes on the exact target filesystem;
4. estimate acquisition peak bytes as:

```text
weight shards
+ tokenizer/config files
+ temporary download overhead
+ generated probe artifacts
+ fixed safety reserve
```

5. report candidate sources without downloading or deleting anything.

The implementation plan must freeze the safety reserve and exact acquisition
method. If no complete snapshot exists and free space is below the frozen
requirement, the result is `INCOMPLETE_RESOURCE_BLOCKED`. The runner stops
before model acquisition and GPU execution.

The gate never runs `rm`, cache cleanup, package-cache cleanup, or shared
`/tmp` cleanup. Resolving insufficient disk requires a separate explicit user
decision.

## Reference Execution Modes

The canonical reference must support these modes using the same model snapshot
and dtype:

### One-Shot Oracle

Run the complete prompt and continuation in one causal forward pass without
reusing an externally supplied cache. This produces oracle logits for every
continuation position.

### Cached Prefill and Decode

Run one prefill call that returns the reference cache/state object, followed by
one-token decode calls. Record state before prefill, after prefill, and after
every decode step.

### Chunked Prefill and Decode

Split the same prompt at frozen chunk boundaries, carry state across chunks,
then run the same one-token continuation. Chunking must include boundaries
that are not multiples of four and that cross multiple hybrid-layer cycles.

### Interleaved Multi-Request Decode

Prefill at least three distinct requests, then decode them in a frozen
interleaving order. One request finishes early; its slot is reset and reused
for a new request. A separate serial execution of each request is the oracle.

The canonical reference path may use the official PyTorch fallback. Optimized
linear-attention packages are not required for compatibility `GO`. Their
absence must be recorded and prevents any performance interpretation.

## State Discovery and Normalization

### State Categories

Every persistent tensor must be assigned exactly one normalized role:

```text
full_attention_key
full_attention_value
linear_recurrent_state
linear_convolution_state
position_or_sequence_metadata
other_persistent_state
```

`other_persistent_state` is not silently accepted. Its producer, consumer,
shape evolution, and necessity for continuation must be documented. An
unexplained persistent tensor makes the gate `INCOMPLETE_SCHEMA`.

### Normalized State Record

Each tensor record contains:

```text
request_id
request_generation
layer_index
declared_layer_type
state_role
tensor_path
shape
stride
dtype
device
requires_grad
logical_numel
logical_bytes
storage_data_ptr
storage_offset
storage_nbytes
storage_identity
lifetime_epoch
sequence_length
update_kind
```

`request_generation` increments whenever a scheduler slot is reused. This
prevents a recycled slot from being mistaken for the same logical request.

`update_kind` is one of:

```text
created
unchanged
replaced
grown
mutated_in_place
released
```

The probe must compare consecutive snapshots using both tensor metadata and
content hashes. Object identity alone is not evidence of state stability.

### Physical Storage Accounting

Logical bytes are:

```text
logical_numel * element_size
```

Unique physical tensor-storage bytes are reconstructed by deduplicating aliased
views by device and storage identity, then accounting for the complete
underlying storage only once.

The ledger reports separately:

- logical bytes by state role and layer type;
- unique physical storage bytes by state role and layer type;
- parameter bytes;
- non-state temporary CUDA peak;
- CUDA allocated and reserved bytes before load, after load, after prefill,
  after each decode step, after release, and after slot reuse.

The report must never subtract two unrelated CUDA snapshots and label the
difference as exact state bytes. Tensor storage accounting and allocator
snapshots are separate evidence.

## State-Lifetime Contract

The compatibility gate must prove these invariants from observed reference
behavior:

1. full-attention K/V state grows with accepted sequence length;
2. recurrent and convolution state have a bounded shape independent of total
   context length after initialization;
3. every state tensor belongs to exactly one logical request generation;
4. decoding one request cannot change another request's state content;
5. resetting a completed request releases all references associated with its
   old generation before slot reuse;
6. a reused slot starts from the documented initial state and cannot read
   content from the prior request generation;
7. chunked prefill and unchunked prefill produce equivalent continuation state
   and logits within the frozen tolerance;
8. state ordering is determined by explicit request and layer identity, not by
   incidental Python list position;
9. state export and import preserve the next-step logits;
10. unknown state roles or unsupported mutation semantics fail closed.

## Engine-Facing Compatibility Contract

The gate exports a framework-neutral manifest describing each layer:

```text
layer_index
layer_type
state_components
component_shapes
component_dtypes
growth_axis or fixed_shape
initialization rule
prefill update rule
decode update rule
reset rule
serialization order
```

It also defines the conceptual future interface:

```text
HybridRequestState
  - request_id
  - request_generation
  - sequence_length
  - per_layer_state

HybridLayerState
  - layer_index
  - layer_type
  - full_attention_kv or linear_recurrent_state
  - linear_convolution_state when required
```

This is a semantic contract, not production code. A later native design may
change storage layout, batching, paging, or kernel ownership only if it
preserves the contract and passes the exported fixtures.

The existing homogeneous `ModelRunner.kv_cache` cannot satisfy this contract
unchanged because:

- only full-attention layers require token-growing K/V;
- linear-attention layers require fixed-size recurrent and convolution state;
- the state components have different shapes, dtypes, update rules, and
  lifetimes;
- recurrent state is request-indexed rather than KV-block-indexed.

The later native design must therefore introduce an explicit heterogeneous
request-state abstraction rather than allocating dummy K/V blocks for linear
layers.

## Frozen Correctness Domain

### Prompt Shapes

Use deterministic token-ID fixtures rather than natural-language-only
prompts. The frozen domain includes:

- short prompt: 17 tokens;
- hybrid-cycle boundary prompt: 65 tokens;
- medium prompt: 257 tokens;
- long compatibility prompt: 1025 tokens;
- three-request mixed batch with distinct lengths;
- one slot-reuse request introduced after an earlier request completes.

Each single-request case decodes at least eight continuation tokens. The
multi-request case decodes enough steps to exercise early completion, continued
peers, and the reused slot.

If the acquired model's minimum supported input or reference API makes one
shape invalid, the implementation plan must revise and freeze the domain before
the canonical run. The runner may not silently skip a case.

### Chunk Boundaries

For the 65-, 257-, and 1025-token prompts, include at least:

```text
[1, remainder]
[3, 5, remainder]
[31, 34, remainder]
[64, remainder]
```

Invalid zero-length remainder chunks are omitted by the contract generator, not
by the runtime worker.

### Equivalence Evidence

For every comparison, record:

- decoded token IDs;
- per-step full-logit SHA256 after canonical CPU serialization;
- top-k token IDs and logits;
- maximum and mean absolute logit difference;
- maximum and mean relative logit difference;
- normalized state-component content hashes;
- sequence length and position metadata.

The canonical tolerance is dtype-aware and must be frozen in the implementation
plan after a same-path repeatability probe. Exact token equality is always
required. A tolerance may account for numerically equivalent chunking but may
not hide a wrong layer schedule, missing state component, cross-request
mutation, or stale slot content.

## Canonical Gate Matrix

The canonical manifest enumerates every required row before execution:

```text
phase
case_id
execution_mode
prompt_length
chunk_schedule
request_count
decode_steps
repeat_index
expected_state_snapshots
```

Required phases:

1. environment and artifact preflight;
2. model/config architecture verification;
3. same-path deterministic repeatability;
4. one-shot versus cached decode;
5. one-shot versus chunked prefill plus cached decode;
6. state export/import continuation;
7. interleaved multi-request isolation;
8. completion, release, and slot reuse;
9. state-memory ledger;
10. post-run process and artifact audit.

The exact row count is generated from the frozen contract and stored in the
manifest. The independent verifier reconstructs the expected domain from the
contract rather than trusting the manifest's declared count.

## Result Classification

### `GO`

Compatibility is `GO` only when:

- all source, model, tokenizer, and environment identities are complete;
- the observed architecture matches the frozen Qwen3.5-2B hybrid schedule;
- every canonical row exists exactly once;
- every continuation comparison satisfies token and frozen logit tolerances;
- state export/import preserves continuation;
- multi-request isolation and slot reuse pass;
- every persistent tensor has an explained normalized role;
- logical and unique physical storage ledgers reconstruct independently;
- recurrent and convolution state are proven bounded with context length;
- full-attention K/V growth is measured rather than assumed;
- no process, artifact, or case-domain contamination is found.

`GO` authorizes a separate native TinyLLMForge hybrid-state integration design.
It does not authorize compression or performance claims.

### `NO_GO`

Compatibility is `NO_GO` when the complete canonical domain runs but the
reference semantics cannot be represented safely under the required contract,
for example:

- continuation cannot be reproduced from explicit exported state;
- persistent state cannot be assigned to request generations;
- interleaved requests mutate each other's state;
- reset and slot reuse cannot be made stale-state-free;
- required persistent state has unbounded or opaque semantics incompatible with
  the proposed engine boundary.

### `INCOMPLETE`

The result is `INCOMPLETE` when evidence is unavailable or invalid, including:

- model snapshot unavailable;
- insufficient disk under the frozen safety policy;
- acquisition or dependency failure;
- unsupported model/runtime version;
- GPU unavailable or occupied beyond the frozen safety threshold;
- missing or duplicate rows;
- worker crash or timeout;
- source, artifact, or environment binding failure;
- unexplained persistent tensor;
- verifier or ledger reconstruction failure.

`INCOMPLETE` must never be converted to `NO_GO` or used as evidence against the
architecture.

## Artifact Layout

Canonical results are written under:

```text
experiments/qwen35_hybrid_state/<run-id>/
```

Required files:

```text
manifest.json
source_manifest.json
model_manifest.json
environment.json
case_rows.jsonl
state_snapshots.jsonl
state_components.jsonl
memory_snapshots.jsonl
processes.json
ports.json
stdout/
stderr/
summary.json
independent_verification.json
report.md
```

Large logits and tensor contents are not copied into the repository. The
artifacts contain deterministic hashes, scalar comparisons, tensor metadata,
and only the minimal bounded fixtures needed for independent verification.

Every artifact listed by a manifest includes size and SHA256. The verifier
rejects unlisted files that could alter classification inputs.

## Independent Verification

The verifier must:

1. reconstruct the required case domain from the frozen contract;
2. verify source, model, tokenizer, environment, process, and port bindings;
3. check artifact size and SHA256;
4. reject missing, duplicate, unknown, or contaminated rows;
5. reconstruct layer schedule and state-role coverage;
6. reconstruct logical bytes from shape and dtype;
7. reconstruct unique physical storage bytes from storage identities;
8. verify state-lifetime transitions;
9. verify no cross-request state hash changes outside the active request;
10. verify release and request-generation changes during slot reuse;
11. recompute every correctness comparison and classification guard;
12. emit one authoritative `GO`, `NO_GO`, or `INCOMPLETE` result.

The verifier may trust cryptographic hashes of bounded tensors emitted by the
worker, but it may not trust worker-computed aggregate bytes, pass booleans,
case counts, or final classification.

## Failure Handling and Remote Safety

- Use only `sitian@10.232.195.203`.
- Use only `CUDA_VISIBLE_DEVICES=0`.
- Do not use `rsync`.
- Do not modify the remote checkout.
- Do not run `kill`, `pkill`, or shared cleanup.
- Do not switch GPUs.
- Do not delete model caches or temporary files owned by other tasks.
- Use a unique remote run directory.
- Use fresh distinct ports for every model process.
- Retry only exact `EADDRINUSE`, at most three attempts.
- Preserve partial artifacts and classify the run `INCOMPLETE`.
- Record unrelated GPU processes before and after execution.
- Stop before GPU work when source, model, disk, or dependency preflight fails.

## Promotion Sequence After Compatibility `GO`

Compatibility `GO` unlocks, in order:

1. **Native hybrid-state integration design**
   - model translation;
   - heterogeneous request-state allocation;
   - scheduler lifecycle;
   - reference-fixture correctness.
2. **Recurrent-state precision gate**
   - FP32 reference versus FP16/BF16;
   - physical state bytes;
   - task-quality and continuation correctness.
3. **Recurrent-state INT8 or low-rank gate**
   - isolated compression method;
   - no mixing with sparse kernels;
   - quality degradation threshold at or below the separately frozen 0.1%
     task metric;
   - physical GPU bytes, not logical compression alone.
4. **Model-native linear-attention kernel gate**
   - official/reference semantics;
   - prefill and decode measured separately;
   - single-request and multi-request workloads;
   - throughput, latency, allocated memory, and reserved memory;
   - end-to-end TinyLLMForge evidence rather than kernel-only timing.

Any later technique receives its own design, implementation plan, canonical
domain, independent verifier, and `GO` decision.

## Claim Discipline

Before compatibility `GO`, the only allowed claim is that Qwen3.5 is a
promising architecture candidate whose state contract is under evaluation.

After compatibility `GO`, allowed claims are limited to:

- the reference hybrid state is fully characterized;
- the state lifecycle is representable by the exported engine-facing contract;
- recurrent and full-attention state bytes are measured for the canonical
  model and workloads;
- native integration work is justified.

The following remain prohibited until their separate canonical gates pass:

- “TinyLLMForge supports Qwen3.5”;
- “recurrent state is compressed by 2.57x”;
- “quality loss is below 0.1%”;
- “linear attention is 2.3x to 2.5x faster”;
- “single-request inference is 1.25x to 1.3x faster”;
- any README or public performance claim.

## Completion Criteria

This design is ready for implementation planning when:

- the compatibility-only `GO` meaning is accepted;
- the reference-first approach is accepted;
- production engine files remain out of scope for the first implementation;
- the disk and artifact preflight is fail-closed and non-destructive;
- correctness, state schema, memory accounting, request isolation, and
  independent verification requirements are unambiguous;
- compression and sparse-kernel work remain separate later gates.
