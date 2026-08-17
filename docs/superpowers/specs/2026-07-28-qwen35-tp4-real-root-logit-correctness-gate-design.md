# Qwen3.5 TP4 Real Root-Logit Correctness Gate Design

## Status

Approved for inline execution under the standing instruction to continue
without per-step confirmation. This design follows the completed TP1
real-root-logit gate and the completed TP4 real-candidate provenance,
live-concurrent ownership, KV-head replication, and constructed
Engine/ModelRunner ownership gates.

The first TP4 numerical gate remains below `ModelRunner` and `LLMEngine`.
It proves the distributed model root before scheduler, sampling, generation,
or production cache behavior is added to the failure surface.

## Objective

Load the approved real Qwen3.5-2B checkpoint as four rank-local
`Qwen35PackedForCausalLM` candidates, initialize one real four-rank NCCL
process group across four distinct GPUs, execute the same three frozen
one-shot prompts on all ranks, gather the complete vocabulary logits on
rank 0, and compare rank-0 final-token logits with the immutable official
Transformers reference used by the TP1 gate.

The distributed native path is:

```text
four fresh rank processes
  -> one distinct GPU per rank
  -> one real NCCL process group
  -> one real TP4 checkpoint candidate per rank
  -> gate-owned rank-local causal attention
  -> production embedding all_reduce
  -> production row-parallel all_reduce
  -> production lm-head gather
  -> rank0 full-vocabulary logits
  -> rank1..3 lm-head result is None
  -> per-rank state commit and release
  -> barrier and process-group destruction
  -> all four processes exit
```

The gate must produce independently verifiable evidence for:

- exact frozen prompt tokens and official reference rows;
- four distinct process, GPU, and tensor-parallel rank identities;
- one shared process-group identity with exact world size four;
- complete rank-local checkpoint loading and source binding;
- rank-local attention topology and replicated KV-head ownership;
- production embedding, row-parallel, and lm-head collective semantics;
- rank-0 complete-vocabulary logits and rank1-3 `None` logits;
- per-rank recurrent-state mutation, release, and pool zeroing;
- successful barrier, process-group destruction, CUDA cleanup, and process exit;
- the same BF16 decision-preserving comparison policy as TP1.

## Approaches Considered

### Reuse Constructed Engine/ModelRunner Ownership as the Numerical Gate

Rejected for the first TP4 numerical boundary. The constructed ownership gate
uses an inert dependency capsule and explicitly does not execute CUDA forward.
Extending it immediately to real `ModelRunner` processes would mix model math,
distributed collectives, KV allocation, warmup, shared memory, scheduler
configuration, and runtime command transport.

### Compare Four Rank-Local Partial Logit Tensors Independently

Rejected. A rank-local vocabulary shard is not a language-model output and
cannot be compared directly with the official full-vocabulary reference.
This would also bypass the production `ParallelLMHead.gather` behavior that
the gate needs to prove.

### Four Real Distributed Model Roots with Rank-0 Logit Authority

Selected. Four fresh processes initialize a real NCCL group, construct and
load rank-local real candidates, and call the native model root directly.
Production collectives assemble the hidden states and logits. Rank 0 is the
only numerical-output authority; ranks 1-3 provide mandatory participation,
state, collective, and cleanup evidence.

This keeps the gate focused on:

```text
TP4 checkpoint mapping + TP4 model math + TP4 collectives
```

while leaving:

```text
ModelRunner + Engine + Scheduler + cached continuation
```

for later gates.

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
shard:
  aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1
model vocabulary:
  248320
tokenizer vocabulary:
  248044
```

Freeze these completed prerequisite conclusions:

```text
TP1 real root-logit authority:
  qwen35-tp1-authority-20260728-195153-r2
  PASS, 179 checks

TP4 serial real-candidate provenance:
  qwen35-tp4-real-candidate-replay-20260728-145713

TP4 live concurrent candidate ownership:
  qwen35-tp4-live-concurrent-ownership-20260728-163700

TP4 constructed Engine/ModelRunner ownership:
  independently verified against its immutable source snapshot
  PASS, 281 checks

global attention topology:
  8 query heads / 2 KV heads
TP4 local topology:
  2 query heads / 1 replicated KV head per rank
```

The new run must rehash its current source closure and checkpoint inventory.
It may reference prerequisite artifacts, but must not accept their hashes as a
substitute for current-source verification.

## Frozen Prompt and Comparison Contract

Reuse the exact TP1 contract module without copying or regenerating prompt
tokens:

```text
tools/qwen35_tp1_real_root_logit_correctness_contract.py
```

The cases remain:

```text
p17
p65
synthetic
```

The official reference tensor artifact is not copied blindly into the TP4
result. A fresh official reference worker must run from the approved
checkpoint and produce the three complete FP32 rows. Its rows must match the
frozen prompt identities and use:

```text
comparison policy:
  bf16_decision_preserving
atol:
  2e-5
rtol:
  1e-5
```

The comparison records full-row SHA256, top-20, winner, runner-up, margins,
maximum and mean absolute differences, percentiles, cosine similarity,
allclose violation count, and maximum scaled error. `PASS` requires the same
decision-preservation guards as TP1. It does not require elementwise allclose.

## Process Architecture

The source-bound coordinator executes two phases in strict order.

### Phase A: Official Reference

Run one fresh isolated official Transformers process on one selected GPU:

1. verify the checkpoint and source manifest;
2. recheck the GPU UUID and free-memory floor;
3. load the official model locally with BF16 and eager attention;
4. execute the three frozen prompts with `use_cache=False`;
5. atomically write the official tensor map and process row;
6. release the model, empty the allocator, synchronize, and exit;
7. prove the PID is absent before Phase B starts.

The reference GPU may be one of the later TP4 GPUs, because the reference
process must be completely gone before any native rank starts.

### Phase B: Distributed Native Group

Select four distinct GPUs. Spawn four fresh rank processes concurrently with:

```text
WORLD_SIZE=4
RANK=0..3
LOCAL_RANK=0..3
CUDA_VISIBLE_DEVICES=<four selected physical GPU indices>
TINYVLLM_DIST_PORT=<fresh port>
MASTER_PORT=<different fresh port>
```

The native process group uses exactly one of the two fresh ports as its
audited rendezvous endpoint. The unused port remains reserved to prevent
collision with other TinyLLMForge tasks. Fixed ports are forbidden.

Each process:

1. binds its logical local rank to exactly one visible GPU;
2. verifies the physical GPU index and UUID selected by the coordinator;
3. initializes `torch.distributed` with backend `nccl`, rank `0..3`, and
   world size `4`;
4. records a common process-group nonce and rendezvous identity;
5. constructs and loads one real TP4 rank-local checkpoint candidate;
6. migrates the candidate and rank-local hybrid-state pool to its GPU;
7. validates the local 2Q/1KV attention topology and replicated KV policy;
8. executes all three cases in identical order;
9. releases every lease and proves rank-local state zeroing;
10. enters a final all-rank barrier;
11. destroys the process group in `finally`;
12. synchronizes, empties CUDA cache, records cleanup, and exits.

No rank may publish an authoritative row until all four rank rows have been
collected and all four PIDs are absent.

## Gate-Owned TP4 Causal Attention

Add a correctness-only backend with the same public contract as the TP1
backend, but configured from each rank's local head counts:

```python
class Qwen35TP4CausalAttentionBackend(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

For the approved topology, each rank receives:

```text
query:
  [tokens, 2 local query heads, head_dim]
key/value:
  [tokens, 1 replicated local KV head, head_dim]
```

The backend:

- repeats the single local KV head to the two local query heads;
- computes scaled dot-product attention in FP32;
- applies a strict causal lower-triangular mask;
- applies FP32 softmax and value multiplication;
- casts the flattened local result back to BF16.

It performs no distributed collective. Distribution remains owned by the
production projections, embedding, and lm-head. FlashAttention, paged KV,
CUDA graphs, sparse attention, cache compaction, and optimized kernels are
forbidden in this correctness gate.

## Rank-Local Candidate Construction

Generalize the proven TP1 candidate path rather than introducing a second
loader:

```text
read bounded checkpoint metadata once per rank process
build complete tensor plan
build TP4 rank-local hybrid-state layout
create a capacity-one rank-local state pool
prepare rank-specific Qwen35PackedForCausalLM
stream and transactionally assign all rank-selected checkpoint bindings
validate candidate provenance and payload
move rank-local model and pool to the assigned GPU
```

Required parameters:

```text
tensor_parallel_size:
  4
tensor_parallel_rank:
  current rank
compute dtype:
  bfloat16
recurrent state dtype:
  float32
convolution state dtype:
  bfloat16
capacity:
  1
```

Each rank row must retain the real-candidate provenance fields needed to tie
the numerical run back to the completed TP4 candidate gates. Rank-specific
payload hashes are expected to differ and must not be normalized away.

## Collective Semantics

The gate validates the existing production behavior rather than replacing it.

### Embedding

`VocabParallelEmbedding.forward()` masks tokens outside the local vocabulary
partition, performs local lookup, and calls `dist.all_reduce(y)`. Every rank
must therefore observe equivalent full hidden embeddings after the collective.

### Row-Parallel Layers

Existing row-parallel output projections and MLP down projections perform
their production reductions. The gate records collective participation
counters through narrowly scoped wrappers that delegate to the real
`torch.distributed` functions. Wrappers may count and validate calls, but must
not alter tensor values, order, group, source, destination, or async behavior.

### LM Head

`ParallelLMHead.forward()` computes one local vocabulary partition and calls:

```text
dist.gather(local_logits, gather_list, dst=0)
```

Required postconditions:

```text
rank0:
  logits is a finite tensor of shape [1, 248320]
rank1..3:
  logits is exactly None
```

The concatenation order must be rank order along vocabulary dimension. Any
non-root tensor result, missing root result, partial vocabulary width, or
gather call drift is a hard failure.

Because `Qwen35PackedForCausalLM` currently validates that lm-head output is a
tensor on every rank, the implementation must make the root contract
distributed-aware: rank 0 validates the tensor, while non-root ranks accept
only `None` when an initialized process group reports world size greater than
one. TP1 behavior must remain unchanged and receive explicit regression tests.

## Native Case Transaction

For each prompt, all ranks execute the same transaction:

1. allocate one rank-local lease with the same request ID;
2. activate the lease;
3. create identical token IDs, positions, cumulative lengths, and final-token
   `logits_indices`;
4. set the production prefill context;
5. call `Qwen35PackedForCausalLM.run_step()` once;
6. validate finite normalized hidden states on every rank;
7. validate root/non-root lm-head semantics;
8. snapshot changed convolution and recurrent state components;
9. release the pool lease and allocator lease;
10. prove all rank-local state tensors are zero and bindings are absent;
11. enter a per-case barrier before the next case.

All 18 linear-attention layers must change both state components on every
rank. Full-attention layers must not add hybrid-state components. Commit must
occur exactly once per rank and case.

If one rank fails, the coordinator must terminate the group as a failed,
preserved attempt. It must never synthesize missing rank evidence.

## GPU and Resource Contract

Before reference execution, and again immediately before native spawning, the
coordinator queries all GPUs with `nvidia-smi` and records:

```text
physical index
UUID
name
total memory
free memory
active compute processes
```

Native selection requires four distinct GPUs, each with at least:

```text
24 * 1024**3 bytes free
```

The coordinator must recheck all four UUIDs and free-memory values after the
reference process exits. A missing GPU, UUID change, insufficient memory,
foreign process appearance, duplicate mapping, or rank/GPU mismatch yields
`INCOMPLETE_RESOURCE`.

Memory values are safety observations, not performance or GPU-memory-saving
claims.

## Cleanup and Failure Handling

Every native rank uses `try/finally` around candidate lifetime and the process
group. Cleanup evidence must distinguish:

```text
lease released
pool zeroed
pool binding absent
model reference dropped
candidate reference dropped
CUDA synchronized
CUDA cache emptied
final barrier completed
process group destroyed
worker exited
```

On a rank-local exception:

- preserve the first exception classification and traceback;
- attempt lease and candidate cleanup;
- attempt process-group abort/destruction without waiting forever at a
  success-only barrier;
- let the coordinator stop the remaining ranks after a bounded grace period;
- preserve the failed run directory under its unique tag;
- do not publish the authoritative exact-five artifact.

The coordinator must detect timeout, early PID disappearance, duplicate PID,
rank mismatch, GPU mismatch, rendezvous failure, collective failure, and
cleanup failure separately.

## Artifact Contract

Publish exactly five files only after the reference and all four native ranks
exit successfully:

```text
tp4_real_root_logit_correctness.json
reference_logits.pt
native_rank0_logits.pt
rank_evidence.json
source_manifest.json
```

`native_rank0_logits.pt` contains only the three contiguous CPU FP32
full-vocabulary rows from rank 0. `rank_evidence.json` contains all four
process rows, topology, collective counts, state evidence, cleanup evidence,
and root/non-root output semantics. It contains no full logits.

The source manifest binds:

- exact current source closure and tree hash;
- checkpoint inventory and hashes;
- prerequisite artifact identities;
- prompt arrays and token hashes;
- reference and four native process identities;
- four physical GPU indices and UUIDs;
- rendezvous ports and process-group nonce;
- result and artifact hashes.

Partial files use a `.partial` suffix and are removed on finalization failure.
Failed or superseded run directories are never deleted, overwritten, or
reused.

## Independent Verifier

Create a standard-library-plus-PyTorch verifier that imports neither
TinyLLMForge nor the producer. It must:

- require the exact five-file inventory;
- rehash all files, sources, checkpoint entries, and prerequisite identities;
- load reference and rank-0 tensor maps on CPU;
- recompute all comparison metrics and classification;
- validate exact prompts, case order, vocabulary width, dtype, and finiteness;
- require four unique native PIDs, ranks, physical GPUs, and GPU UUIDs;
- require one common world size, rendezvous identity, and process-group nonce;
- require rank0 tensor logits and rank1-3 exact `None` output evidence;
- validate collective call counts and ordering constraints;
- validate local 2Q/1KV topology and replicated KV policy;
- validate per-rank state mutation, release, zeroing, and commit counts;
- validate final barrier, process-group destruction, PID absence, and source
  binding;
- reject metric re-signing, source re-signing, missing ranks, swapped ranks,
  duplicated GPUs, forged non-root evidence, partial vocabulary, relaxed
  tolerance, extra files, or incomplete cleanup.

It prints:

```text
PASS, <N> checks
```

only if every guard passes.

## Failure Classification

Use exact classifications:

```text
PASS
NO_GO_LOGIT
NO_GO_STATE
NO_GO_TOPOLOGY
NO_GO_COLLECTIVE
INCOMPLETE_RESOURCE
INCOMPLETE_REFERENCE
INCOMPLETE_NATIVE_GROUP
INCOMPLETE_CLEANUP
INCOMPLETE_ARTIFACT
```

`NO_GO_LOGIT` requires completed reference and native execution with a failed
decision guard. `NO_GO_STATE`, `NO_GO_TOPOLOGY`, and `NO_GO_COLLECTIVE`
require completed rank evidence that violates the named hard contract.
Incomplete outcomes must not publish an authoritative PASS artifact.

## Static Safety

The source audit requires:

- no construction of `LLMEngine`, `ModelRunner`, Scheduler, sampler, or
  tokenizer in the native group;
- no call to `LLMEngine.step()`, generation, or production worker loop;
- no FlashAttention, paged-KV, sparse attention, cache compaction, or CUDA
  graph path;
- no fixed process-group port or shared-memory name;
- no replacement of production embedding, row-parallel, or lm-head math;
- collective wrappers, if present, must delegate exactly once to the original;
- the immutable schema-v2 canonical `NO_GO` remains unchanged.

## Test Strategy

### Contract Tests

Cover:

- TP1 prompt and comparison reuse;
- rank0 tensor and rank1-3 `None` validation;
- rejection of partial vocabulary and non-root tensors;
- topology and GPU uniqueness validation;
- classification behavior.

### Attention Tests

Cover:

- manual FP32 local-head oracle;
- 2Q/1KV replication;
- causal masking and future-token poisoning;
- BF16 output dtype;
- malformed shape, head count, dtype, and device rejection.

### Root Tests

Extend the transactional root tests to prove:

- TP1 still requires a tensor;
- TP4 rank0 requires a tensor;
- TP4 non-root accepts only `None`;
- non-root state commit still occurs after successful gather participation;
- invalid root/non-root outputs do not commit.

### Coordinator Tests

Use fake processes and GPU rows to cover:

- reference-before-native ordering;
- four-rank concurrent launch;
- dynamic distinct ports;
- duplicate or missing GPU rejection;
- rank/PID/GPU mismatch;
- one-rank early exit or timeout;
- failed process-group initialization;
- cleanup and exact-five publication rules.

### Verifier Tamper Tests

Reject:

- changed rank, PID, GPU, UUID, port, nonce, or world size;
- changed topology or replicated KV evidence;
- rank1-3 tensor claims;
- missing gather or altered collective count;
- changed state/cleanup rows;
- replaced logits or recomputed metrics;
- relaxed tolerance, extra files, or source drift.

### Remote Validation

Run in this order:

1. focused local unit tests;
2. local `py_compile` and `git diff --check`;
3. remote resource preflight;
4. one unique-tag native-only distributed smoke;
5. one new unique-tag full source-bound authority run;
6. independent verifier over the downloaded immutable artifact;
7. focused completion matrix against TP1 and prerequisite TP4 gates.

Every failed or superseded remote run remains preserved.

## Forbidden Conclusions

Passing this gate proves only TP4 distributed one-shot final-token
decision preservation for the three frozen prompts under the audited
checkpoint, source, topology, and BF16 policy.

It does not prove:

- elementwise BF16 allclose;
- cached decode or chunked-prefill equivalence;
- `ModelRunner.run()` or `LLMEngine.step()` numerical correctness;
- Scheduler, sampler, generation, or request interleaving correctness;
- latency, throughput, cache savings, GPU-memory savings, compression, or
  production quality gains.

## Next Gates

If TP4 passes:

1. define and prove a full-attention cache contract;
2. prove cached continuation against one-shot execution;
3. prove production `ModelRunner.run()` numerical correctness;
4. prove bounded `LLMEngine.step()` integration;
5. only then benchmark latency, throughput, cache, and GPU memory.

If TP4 fails, preserve all evidence and localize the first divergent rank,
collective, layer, or component before any performance work.

