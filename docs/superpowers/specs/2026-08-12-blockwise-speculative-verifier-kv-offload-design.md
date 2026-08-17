# Blockwise Speculative Verifier KV-Offload Design

**Date:** 2026-08-12

**Status:** Approved by the existing long-context and transactional-speculation objective

**Classification before and after this gate:** `NOT_PROMOTABLE`

## Goal

Allow the generic batch-native speculative verifier to run when the visible
logical KV history is larger than the available GPU staging-slot inventory.
Verification must use exact blockwise online-softmax attention over CPU-backed
logical KV pages while preserving:

- one target forward per fixed-Q verifier group;
- accepted KV in-place commit;
- rejected reserved suffix discard without accepted-KV replay;
- exact greedy token parity;
- real `KVOffloadMVP0` H2D/D2H counters.

This is the prerequisite for controlled TP1 16K/32K, batch-1/batch-4
performance campaigns with a fixed 68-slot GPU KV budget.

## Current Blocking Geometry

The existing 4K campaign uses 256-token blocks and 68 GPU staging slots:

```text
4K batch 1:
  16 visible blocks
4K batch 4:
  64 visible blocks
16K batch 4:
  256 visible blocks
32K batch 4:
  512 visible blocks
```

The current full-attention verifier requires every visible block to be
resident before the target forward. It therefore works for the 4K matrix but
cannot run 16K/32K batch 4 under the same 68-slot budget.

Increasing GPU slots to 260/516 would make the workload execute, but would not
establish longer-context execution under a fixed GPU KV budget. Splitting the
matrix to single-sequence-only cells would leave the explicit batch-4 gate
uncovered. Both alternatives are rejected.

## Existing Reusable Foundation

The repository already contains:

- logical KV block IDs decoupled from physical GPU staging slots;
- CPU backing, async H2D/D2H, dirty tracking, clean eviction, and generation
  validation in `KVOffloadMVP0`;
- exact blockwise online-softmax decode and prefill attention;
- batch-native fixed-Q speculative verification;
- transactional speculative residency prepare/precommit/seal/rollback;
- real loaded-model exact-parity and movement gates.

The implementation should extend these generic paths rather than introduce a
model-specific verifier or a benchmark-only copy path.

## Selected Architecture

### Streaming residency preparation

`SpeculativeResidencyParticipant.prepare_batch()` gains an explicit
keyword-only policy:

```python
prepare_batch(
    ticket_id,
    rows,
    *,
    stage_all_original_blocks: bool = True,
)
```

The default preserves current behavior.

When `stage_all_original_blocks=False`, prepare:

1. binds every original and reserved block generation;
2. derives the blocks touched by verifier logical write slots;
3. stages only:
   - original blocks containing verifier writes, with `require_valid=True`;
   - reserved blocks containing verifier writes, with `require_valid=False`;
4. protects those write blocks from eviction;
5. leaves untouched original history CPU-resident for attention-layer window
   staging.

`ModelRunner.prepare_speculative_residency_batch()` chooses the streaming
policy only when both `kv_offload_mvp0` and
`kv_offload_blockwise_decode` are enabled. No proposal-source or model-name
branch is added.

### Logical verifier metadata

For blockwise speculative verification,
`ModelRunner.prepare_spec_verify_batch()`:

- keeps full visible block-table rows as logical IDs;
- maps only verifier logical write positions to physical slots;
- records the logical rows, context lengths, query lengths, and write blocks
  in the generic `Context`;
- does not call `map_block_rows()` for the full visible history;
- installs `mode="spec_verify"` with
  `kv_offload_blockwise_decode=True`.

The non-blockwise path remains unchanged and continues to use physical block
tables plus `flash_attn_with_kvcache`.

`kv_offload_blockwise_prefill=True` is no longer rejected merely because the
later verifier runs in spec-verify mode. Quantized KV, Quest, AM compaction,
mixed prefill/decode, non-greedy acceptance, and non-transactional hybrid
state remain fail-closed.

### Blockwise multi-query verifier attention

Add a dedicated attention helper:

```python
_blockwise_online_spec_verify_attention(
    q,
    k_cache,
    v_cache,
    context,
    num_heads,
    head_dim,
    scale,
    *,
    layer_idx=-1,
) -> torch.Tensor
```

Inputs remain the existing fixed-Q batch shape:

```text
q:
  [batch * query_len, num_heads, head_dim]
logical block rows:
  [batch, visible blocks]
context lengths:
  one final verifier context length per row
query lengths:
  homogeneous fixed Q within the group
```

The helper reshapes queries to `[B, Q, H, D]` and maintains FP32 online-softmax
state:

```text
running maximum:
  [B, Q, H]
running exponential sum:
  [B, Q, H]
running weighted value:
  [B, Q, H, D]
```

Each logical block window is staged through the existing
`_stage_blockwise_read_window()` path. The window causal mask uses absolute
positions:

```text
query_start = context_len - query_len
query_position[j] = query_start + j
key_position = window_start_token + local_key_offset
visible iff key_position <= query_position[j]
```

This matches the lower-right causal alignment used by
`flash_attn_with_kvcache` for multi-query verification. The merge uses the
existing numerically stable online-softmax recurrence and returns the original
flattened query shape.

Layers alternate forward/reverse window order exactly like blockwise decode
to preserve bounded cross-layer reuse. Current verifier write blocks remain
protected in every window. The window plan is cacheable by:

```text
logical block rows
context lengths
query lengths
block size
window blocks
write blocks
GPU slot count
```

### Capacity rule

For every window:

```text
unique required read blocks
+ protected verifier write blocks
<= GPU staging slots
```

The first long-context gate uses:

```text
gpu staging slots=68
window blocks per sequence=8
batch sizes=1 and 4
```

At batch 4 this bounds one window to at most 32 read blocks plus verifier write
blocks, leaving staging capacity for protected writes and prefetch hints.

## Data Flow

For one fixed-Q verifier group:

1. Scheduler reserves speculative tail blocks and builds proxy block tables.
2. Residency prepare binds all generations but stages only verifier write
   blocks in blockwise mode.
3. `prepare_spec_verify_batch()` maps write positions to physical slots while
   retaining logical history rows.
4. Every attention layer stores the query K/V directly into the protected
   physical write slots.
5. The blockwise verifier streams historical windows from CPU backing,
   computes exact causal multi-query attention, and evicts/reuses staging
   slots between windows.
6. The target logits produce the greedy accepted prefix.
7. Precommit partitions reserved blocks into committed and rejected sets.
8. Seal keeps accepted materialized KV in place, marks committed writes dirty,
   and discards rejected reserved blocks without D2H.

No accepted token is replayed and no accepted KV is copied into a second
logical location.

## Correctness Tests

### Residency participant

Dependency-light tests must prove:

- default prepare still stages all original blocks;
- streaming prepare stages only materialized original write blocks and
  reserved write blocks;
- untouched original history remains non-resident;
- generation binding covers every original and reserved block;
- prepare failure discards only resident reserved blocks;
- precommit/seal/rollback semantics are unchanged.

### ModelRunner preparation

AST/dependency-light tests must prove:

- blockwise mode keeps logical block tables;
- only logical write slots are mapped to physical slots;
- context records logical rows, context lengths, query lengths, write blocks,
  and blockwise mode;
- full-attention mode remains byte-for-byte schema compatible;
- blockwise prefill is allowed with blockwise verifier;
- quantized KV, mixed batch, hybrid state without transactions, and missing
  residency tickets still fail closed.

### Attention math

Small real-tensor tests compare blockwise verifier output against a dense
causal reference for:

- batch 1 and batch 4;
- query lengths 2 and 4;
- context lengths crossing multiple windows;
- uneven per-row context lengths;
- GQA where `num_heads != num_kv_heads`;
- forward and reverse layer window order;
- partial final windows;
- write blocks that overlap the final visible history block.

Tolerance must follow the existing blockwise FP32 online-softmax tests and
must be tight enough to detect causal-mask or row-alignment errors.

### Loaded-model gate

The first remote loaded-model gate uses the current fixed A100 route and:

```text
model=Qwen3-0.6B
tensor_parallel_size=1
GPU staging slots=68
logical blocks sufficient for 32K batch 4
blockwise prefill=True
blockwise decode=True
blockwise window blocks=8
temperature=0.0
repetitive deterministic prompts
contexts=(16384, 32768)
batch sizes=(1, 4)
output tokens=8
```

For each context/batch cell, baseline and
`EngineSpeculativeRuntime(NGramDraftAdapter)` must have exact output-token
parity. Candidate execution must observe proposals, accepted draft tokens,
first-target callbacks, tail callbacks, positive real H2D bytes/copies, and
`speculative_residency_rejected_d2h_copies=0`.

This loaded gate is correctness and movement evidence only. Its elapsed times
must not be reported as a performance result.

## Failure Boundaries

Fail closed when:

- a fixed-Q group is empty or has heterogeneous query lengths;
- query rows do not end at their recorded verifier context lengths;
- logical block rows do not cover every visible key position;
- protected write blocks plus one read window exceed GPU slots;
- a required historical block lacks valid CPU backing;
- a verifier write block is evicted during the forward;
- any residency generation changes between prepare and materialization;
- exact dense/blockwise attention parity exceeds tolerance;
- loaded-model token parity fails;
- H2D evidence is absent from a cell that exceeds GPU capacity;
- rejected reserved blocks produce D2H copies.

## Artifact

Create a schema-v1 blockwise verifier gate artifact containing:

- fixed configuration and cell matrix;
- exact prompt token IDs and digests;
- baseline/candidate output token IDs;
- proposed and accepted token counts;
- callback counts;
- residency prepare/precommit/seal/rollback counters;
- committed/rejected block counts;
- real H2D/D2H/copy-wait/eviction counters;
- source file SHA-256 values;
- remote and local independent verification receipts;
- explicit `NOT_PROMOTABLE` classification.

## Non-Goals

- no TP4;
- no second model;
- no learned drafter or MTP adapter;
- no CUDA Graph work;
- no KV8/KV4 support in speculative verification;
- no Quest, AM compaction, sliding window, sparse attention, or context
  parallel;
- no performance claim from the correctness gate;
- no change to variable-Q grouping: distinct fixed Q values remain separate
  verifier groups without padding.

## Expansion Boundary

Only after the blockwise loaded-model correctness gate passes should the
existing performance-gate framework be generalized to:

```text
contexts=(16384, 32768)
batch sizes=(1, 4)
output tokens=64
one warmup
one parity run
five measured runs
```

That later campaign must report TTFT, TPOT, throughput, peak GPU memory, real
H2D/D2H, acceptance, and exact parity. It remains `NOT_PROMOTABLE` until TP4
and a second model structure are independently covered.

## Execution Record — 2026-08-13

The approved TP1 loaded-model campaign completed under the opaque run ID
`blockwise-tp1-opaque-17786-19070`. The opaque identifier is not date or
ordering evidence.

Authoritative local evidence:

```text
artifact directory:
  artifacts/blockwise_speculative_verifier/
    blockwise-tp1-opaque-17786-19070/

result artifact:
  result.json

result SHA-256:
  2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600

remote verifier:
  verify.remote.json
  PASS / NOT_PROMOTABLE

local verifier:
  verify.json
  PASS / NOT_PROMOTABLE

remote and local verified artifact SHA-256:
  2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600

remote direct KV regression:
  tools/test_kv_offload.py
  kv offload tests passed
```

The repository base commit at evidence review time was
`3217895019a26154270db40c432495c7657abcb1`. The worktree was dirty, so the
base commit alone is not source authority. `result.json` records SHA-256
digests for every gate-relevant source file, and both independent verifiers
matched those digests against the downloaded artifact/current source tree.

The four exact-greedy parity cells all passed:

| Cell | Proposed | Accepted | First-target callbacks | Tail callbacks | Candidate H2D copies | Candidate H2D bytes | Visible logical blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16K / batch 1 | 8 | 5 | 2 | 2 | 0 | 0 | 64 |
| 16K / batch 4 | 32 | 26 | 2 | 2 | 21,608 | 634,413,645,824 | 256 |
| 32K / batch 1 | 8 | 3 | 5 | 2 | 38,865 | 1,141,081,374,720 | 128 |
| 32K / batch 4 | 32 | 24 | 5 | 3 | 160,937 | 4,725,130,919,936 | 512 |

The 16K/batch-1 cell has 64 visible logical blocks, which fits within the
68-slot GPU staging budget, so positive H2D is not required there. Every cell
whose visible history exceeds the GPU staging budget has positive real
`KVOffloadMVP0` H2D copies and bytes.

Candidate totals across the four cells:

```text
proposed tokens:                         80
accepted draft tokens:                   58
first-target callbacks:                  14
tail callbacks:                           9
H2D copies:                          221,410
H2D bytes:                     6,500,625,940,480
D2H copies:                            1,004
D2H bytes:                        29,477,568,512
copy waits:                            80,993
evictions:                            222,377
speculative committed blocks:               0
speculative rejected blocks:                0
rejected speculative D2H copies:             0
```

The zero committed/rejected block counters are recorded, not hidden. This
eight-token campaign did not produce a positive block-count transition, so
the loaded artifact establishes accepted-token parity, positive proposal
execution, callback execution, real long-context movement, and the invariant
that rejected speculative residency produced no D2H copy. Positive
block-level commit/reject-count coverage remains supplied by the focused
transaction/residency tests rather than this artifact.

This gate is correctness and movement authority only. It does not establish
16K/32K TPOT, TTFT, throughput, peak-memory improvement, or any performance
direction. It also does not establish TP4, a second model structure, a learned
draft model, native MTP with KV offload, or KV8/KV4 speculative verification.
The result therefore remains `NOT_PROMOTABLE`.

Fresh post-document validation:

```text
strict artifact verifier:                  PASS
dependency-light isolated regression:      124 passed
remote direct KV regression:               PASS
runner lifecycle repetition:               10 / 10 passed
Python compile / shell syntax:              PASS / PASS
artifact-condition assertion:              PASS
scoped diff check for gate files:           PASS
staged diff:                                empty
```

The repository-wide `git diff --check` remains blocked by trailing whitespace
in unrelated `tinyvllm/engine/model_runner.py` edits around
`warmup_model()`. This is recorded as a worktree hygiene boundary and is not
converted into a blockwise correctness failure.
