# Heuristic-Equivalent Exact-Width CUDA Graph Recovery Design

Date: 2026-07-22

## Status

This design supersedes the fixed-16 production-admission path in:

```text
docs/superpowers/specs/2026-07-21-fixed-split-multi-sequence-cuda-graph-recovery-design.md
```

The fixed-16 canonical artifact remains authoritative:

```text
Gate A exact replay: EXACT_REPLAY_CORRECT
rounded replay:      ROUNDED_REPLAY_CORRUPT
Gate B compatibility: LEGACY_INCOMPATIBLE
```

The production batch-greater-than-one eager guard must remain unchanged until
this design's diagnostic gates pass and a later production performance gate
returns `GO`.

## Objective

Recover an exact-batch multi-sequence CUDA Graph candidate for FlashAttention
2.6.3 without changing legacy auto-split eager semantics.

The work must:

1. reproduce the FlashAttention 2.6.3 split heuristic exactly from immutable
   runtime inputs;
2. bind CUDA Graph identity to both the effective split and exact page-table
   width;
3. allow a diagnostic process to use more than one graph identity when decode
   crosses a 256-token KV page boundary;
4. prove exact replay against an eager comparator using the same explicit
   per-step split;
5. separately prove that the explicit heuristic-equivalent eager path remains
   compatible with legacy `num_splits=0` eager;
6. preserve the full frozen batch, trajectory, repetition, step, tensor, KV,
   token, tolerance, and independent-verifier coverage;
7. keep rounded replay classified separately and disabled in production;
8. fail closed on unsupported FlashAttention versions, unsupported devices,
   heuristic drift, graph-key drift, missing per-step policy evidence, source
   drift, or any correctness failure;
9. defer production dispatch and performance claims until diagnostic gates
   pass.

## Root-Cause Evidence

### FlashAttention 2.6.3 Heuristic

The authoritative tag is:

```text
v2.6.3
418d677192b483dfc1decfdf9aadca40b402485d
```

For paged-KV decode, `set_params_splitkv()` derives:

```text
block_n = 128                         # head_dim = 128
num_n_blocks = ceil(seqlen_k / 128)
seqlen_k = block_table_width * 256
batch_nheads_mblocks = batch * 8 * 1 # Qwen3 GQA swap: 16 Q heads -> 8 KV heads
num_SMs = A100_SM_count * 2 = 216
```

It then selects the smallest eligible split whose occupancy efficiency reaches
at least 85% of the best eligible efficiency.

For the canonical ragged batches:

```text
batch 5,  width 1 -> auto split 2
batch 8,  width 2 -> auto split 2
batch 9,  width 2 -> auto split 2
batch 16, width 3 -> auto split 3
```

### Why Fixed 16 Failed Gate B

With one 256-token page there are two 128-token KV tiles. Auto split 2 and
fixed split 16 both reduce one effective tile per non-empty split, so they can
be bit-exact.

With two or more pages:

```text
auto split 2/3 -> multiple tiles can be accumulated inside one split
fixed split 16 -> one tile per effective split, then a different combine tree
```

Both paths are legal, but floating-point reduction is non-associative. BF16
ULP-level differences appear in the first transformer layer and can amplify
through 28 layers into logits and greedy-token changes.

### Exact Width Is Part of Semantics

An A100 attention-level CUDA Graph probe established:

```text
capture width == runtime width
and explicit split == FA2.6.3 auto heuristic
    -> graph replay is bit-exact with auto eager

capture width padded to 4 pages
while runtime width is 1, 2, or 3 pages
    -> BF16 differences reappear
```

Therefore a graph key containing only batch size and split is insufficient.
The page-table width changes `params.seqlen_k`, tile partitioning, and reduction
order even when every real `context_len` is unchanged.

## Industry Reference Patterns

### vLLM

Current vLLM distinguishes normal eager scheduling from CUDA Graph scheduling.
For FlashAttention 3 it passes a configured maximum split bound during full
CUDA Graph execution so scheduler metadata and intermediate buffers can be
preallocated. Batch-invariant mode forces one split.

This is not directly portable to FlashAttention 2.6.3 because FA2 interprets
positive `num_splits` as an exact split count, not a scheduler upper bound.

### FlashInfer and SGLang

FlashInfer CUDA Graph wrappers separate planning from replay and require
persistent, preallocated workspace and metadata buffers. SGLang creates
per-capture-size wrappers and updates their persistent metadata before replay.
For deterministic inference SGLang disables CUDA Graph KV splitting rather than
silently accepting a changing reduction schedule.

The reusable principle is:

```text
capture-stable kernel plan + persistent metadata identity + fail-closed fallback
```

### Decision for TinyLLMForge

TinyLLMForge remains on FlashAttention 2.6.3 for this recovery. The smallest
attributable design is exact-shape specialization:

```text
graph identity =
    graph batch size
    + active batch size
    + exact page-table width
    + effective explicit split
    + FlashAttention version
    + device SM count
    + Q/KV head counts
    + head dimension
    + KV page size
```

No dependency upgrade or attention-backend migration is included.

## Alternatives

### 1. Recommended: FA2.6.3 Heuristic-Equivalent Exact-Width Graph Cache

Mirror the upstream heuristic in a dependency-light pure function. For each
decode step, compute the effective split from the active batch and exact
`block_tables.size(1)`. Capture or reuse a graph keyed by exact graph identity.

Advantages:

- preserves legacy auto-eager reduction semantics;
- fixes both sources of graph identity drift: split and page width;
- keeps the dependency and kernel build unchanged;
- supports page-boundary transitions by switching graph identities;
- provides direct attribution if the gates pass.

Costs:

- more graph identities and higher capture/startup memory;
- graph creation must snapshot and restore KV write slots;
- production admission needs a graph-count and memory budget.

### 2. Piecewise CUDA Graph with Attention Outside the Graph

Capture graph-safe model regions and run FlashAttention eagerly each layer.

Advantages:

- preserves native auto heuristic;
- removes attention metadata and split identity from the captured graph.

Costs:

- requires model partitioning and many graph boundaries;
- likely retains substantial Python/kernel-launch overhead;
- is much larger than the current manual graph implementation.

This remains a fallback if exact-width graph specialization is not profitable.

### 3. Upgrade to FA3 or Move Decode to FlashInfer

Adopt AOT scheduling or plan/replay wrappers designed for dynamic paged decode.

Advantages:

- follows current high-performance engine architecture;
- can support dynamic scheduling with preallocated metadata.

Costs:

- changes dependency, kernels, numerics, packaging, and performance together;
- makes root-cause attribution impossible in this recovery;
- requires a separate compatibility and backend-migration program.

This is deferred.

## Architecture

### Pure Heuristic Module

Create:

```text
tinyvllm/engine/flash_attn_split_policy.py
```

It owns:

```python
@dataclass(frozen=True)
class FlashAttentionSplitInputs:
    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    page_table_width: int
    max_seqlen_q: int
    multi_processor_count: int


def flash_attn_263_decode_num_splits(
    inputs: FlashAttentionSplitInputs,
) -> int:
    ...
```

The function must mirror upstream v2.6.3:

- apply the Qwen GQA query-group swap when its upstream predicate is true;
- derive `block_n`, `num_n_blocks`, and `num_m_blocks`;
- multiply SM count by two;
- preserve split eligibility and the 85% threshold;
- reject unsupported head dimensions, page sizes, query lengths, or versions
  rather than approximating.

It must not import Torch.

### Per-Step Policy Evidence

Every measured row must record:

```text
split_policy_name = "fa2_263_heuristic_exact_width"
flash_attn_version = "2.6.3"
page_table_width
effective_num_splits
heuristic_batch_size
heuristic_num_heads
heuristic_head_dim
heuristic_page_block_size
heuristic_max_seqlen_q
heuristic_multi_processor_count
graph_identity_sha256
```

Case-level artifacts may summarize the ordered unique graph identities, but
must not replace per-step evidence with one split value.

### Diagnostic Graph Cache

Replace the diagnostic's one-graph-per-process assumption with a private cache:

```python
dict[DiagnosticGraphIdentity, CapturedDecodeGraph]
```

For every step:

1. prepare current dynamic decode inputs;
2. read the exact `block_tables.size(1)`;
3. compute effective split from immutable heuristic inputs;
4. construct the graph identity;
5. capture the graph on first use, snapshotting and restoring every capture
   write slot;
6. replay only a graph whose identity exactly matches the step;
7. otherwise fail closed.

Static `block_tables` for a captured graph must have exactly the identity's
page-table width. It must not use `max_model_len` width or a larger zero-padded
width.

### Same-Policy and Legacy Comparisons

Gate A remains:

```text
candidate eager explicit heuristic
vs exact graph explicit heuristic
vs rounded graph explicit heuristic
```

Gate B becomes:

```text
legacy eager auto
vs candidate eager explicit heuristic
```

Both policies must use the same prompts, reference-token trajectory, initial
KV state, measured steps, and tensor/KV observations.

The frozen logical coverage remains:

```text
Gate A:
7 batches × 3 trajectories × 3 modes × 3 repetitions = 189 processes

Gate B:
7 batches × 3 trajectories × 2 policies × 3 repetitions = 126 processes

Total:
315 isolated processes
```

No batch, trajectory, repetition, warmup step, measured step, tensor, KV
observation, token comparison, tolerance, or hash requirement may be removed.

### Classification

The independent verifier must report separately:

```text
exact_classification:
    EXACT_REPLAY_CORRECT | EXACT_REPLAY_CORRUPT | INCOMPLETE

rounded_classification:
    ROUNDED_REPLAY_CORRECT | ROUNDED_REPLAY_CORRUPT | INCOMPLETE

legacy_compatibility:
    LEGACY_COMPATIBLE | LEGACY_INCOMPATIBLE | INCOMPLETE

policy_integrity:
    POLICY_EXACT | POLICY_DRIFT | INCOMPLETE
```

Diagnostic `GO` requires:

```text
exact_classification == EXACT_REPLAY_CORRECT
legacy_compatibility == LEGACY_COMPATIBLE
policy_integrity == POLICY_EXACT
structural_failures == 0
```

Rounded classification does not authorize rounded production replay.

## Production Boundary

This design authorizes only diagnostic and verifier changes.

The following remain unchanged:

```python
multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
```

Production graph dispatch, startup capture, configuration, README claims, and
performance gates require a follow-up implementation stage after diagnostic
`GO`.

The production design must include:

- an allowlist of exact batch sizes;
- an allowlist or bounded range of page-table widths;
- a maximum graph count;
- startup time and peak reserved-memory limits;
- exact-identity hit/fallback telemetry;
- eager fallback on any unsupported identity;
- source-bound throughput, ITL, request-rate, graph-hit, initialization, and
  memory gates.

## Verification

### Dependency-Light Tests

Tests must cover:

- exact upstream heuristic outputs for known inputs;
- split eligibility edge cases;
- GQA swap behavior;
- rejection of unsupported inputs;
- graph identity changes when width or split changes;
- graph identity stability for identical inputs;
- matrix size and uniqueness remain 189/126/315;
- per-step policy evidence is mandatory;
- verifier rejects missing, mixed, or recomputation-inconsistent policy data;
- production guard remains fail closed.

### Remote A100 Tests

Before canonical execution:

1. direct FlashAttention probe:
   `auto == explicit heuristic` for every batch/width combination;
2. attention-level graph probe:
   exact-width graph replay equals auto eager;
3. negative control:
   padded-width capture is detected as incompatible;
4. model smoke:
   batches `5`, `8`, and `16`, including a page-boundary transition;
5. fresh source-bound 315-process canonical and independent verification.

Every remote process must use:

```text
host: sitian@10.232.195.203
CUDA_VISIBLE_DEVICES=0
unique TINYVLLM_DIST_PORT
unique MASTER_PORT
```

The remote checkout must not be modified or synchronized.

## Success Criteria

This diagnostic stage succeeds only when a fresh independent artifact proves:

```text
315 unique case IDs
630 unique ports
189 Gate A processes
126 Gate B processes
all required raw/layer/KV tensor shards present
all artifact hashes recompute
POLICY_EXACT
EXACT_REPLAY_CORRECT
LEGACY_COMPATIBLE
zero structural failures
```

Anything else is `NO_GO`, and the production guard remains.

