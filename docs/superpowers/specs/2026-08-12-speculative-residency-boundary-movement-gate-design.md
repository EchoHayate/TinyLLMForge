# Speculative Residency Boundary and Movement Gate Design

**Date:** 2026-08-12

**Status:** Approved continuation of the transactional residency gate

**Classification before and after this gate:** `NOT_PROMOTABLE`

## Goal

Add a deterministic loaded-model TP1 correctness gate that closes the three
coverage holes left by the schema-v2 parity artifact:

- accepted speculative KV crosses a 256-token block boundary and publishes a
  real reserved block in place;
- rejected speculative KV crosses the same boundary and discards a real
  reserved block without D2H;
- an ordinary historical block is cleanly evicted after writeback and is
  reloaded through a real `KVOffloadMVP0` H2D copy before speculative
  verification.

The gate remains correctness-only. It must not claim TPOT, TTFT, throughput,
memory, long-context, TP4, learned-drafter, or MTP improvement.

## Current Evidence Gap

The passing artifact
`artifacts/speculative_tp1_parity/20260812T062046Z/result.json` proves exact
greedy parity and fourteen prepare/precommit/seal cycles with real MVP-0
counters. Its boundary counters are all zero:

```text
h2d_copies=0
committed_blocks=0
rejected_blocks=0
```

Therefore it does not yet prove loaded-model accepted reserved-block
retention, rejected reserved-block discard, or host-to-device reload.

## Alternatives

### Prompt-only natural n-gram behavior

Use long repeated text and hope one proposal is accepted and another is
rejected at a block boundary.

Rejected as the authoritative gate. Model/tokenizer changes can move the
boundary or alter acceptance, making the required counters flaky.

### Gate-specific deterministic adapter plus real residency operations

Use pretokenized 254-token prompts. Prefill appends one token, leaving the
sequence at length 255. The baseline run supplies the next three deterministic
greedy target tokens to a source-agnostic fixture adapter, which then proposes
either:

- the exact three-token baseline suffix, forcing three accepted draft tokens;
  or
- the same suffix with the first token changed, forcing immediate rejection
  and target fallback.

After prefill, write back the ordinary resident block, evict the clean
CPU-valid mapping through a validated manager API, and let the next real
ModelRunner decode reload it through H2D.

Selected. Acceptance/rejection is deterministic, all KV bytes are produced
and consumed by the loaded target model, and movement counters remain real
`KVOffloadMVP0` counters.

### Runtime debug mode or model-name-specific hook

Add a runtime/config flag that forces eviction or branches on Qwen.

Rejected. Test orchestration does not belong in the generic runtime, and the
core must remain model-name-free.

## Clean Eviction Primitive

Add one narrow manager operation:

```python
def evict_clean_resident_blocks(
    self,
    block_identities: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    ...
```

The caller must first make each block clean and CPU-valid through existing
writeback and copy synchronization. The method:

- validates the complete batch before mutation;
- requires exact bound generations;
- requires every block to be resident, clean, CPU-valid, and free of pending
  H2D waits;
- removes only logical-to-physical mappings and completed copy-event
  metadata;
- preserves pinned-CPU validity and allocator generation;
- increments `evictions` and `evict_clean` exactly once per block;
- returns the normalized identities in input order.

It never performs a synthetic copy and never discards CPU backing.

Dirty blocks, stale generations, duplicate IDs, missing residency, unreadable
CPU backing, and pending H2D fail before any mapping changes.

## Deterministic Boundary Workloads

Each case uses:

```text
prompt token count=254
Sequence.block_size=256
max_tokens=4
temperature=0.0
ignore_eos=True
tensor_parallel_size=1
kv_offload_mvp0=True
kv_offload_gpu_blocks=2
kv_offload_logical_blocks=64
```

The first engine step performs prefill and appends one baseline target token,
so the live sequence length becomes 255. The gate then:

1. records the resident block identity from rank-0 MVP-0 generation state;
2. calls existing `writeback_dirty()`;
3. calls existing `synchronize_copies()`;
4. calls `evict_clean_resident_blocks()`;
5. runs the next decode step.

The decode step reloads the original block through H2D. A speculative
transaction reserves for `proposal_count - 1` materialized tail tokens, so a
one-token proposal cannot reserve a block at live length 255. The corrected
three-token proposal materializes positions 255 and 256 and forces one
reserved second block.

### Accepted boundary case

The adapter returns the three baseline continuation tokens after the token
already appended by prefill:

```python
DraftProposal(
    sequence_id=context.sequence_id,
    token_ids=baseline_completion_token_ids[1:4],
    source_type="boundary_fixture",
)
```

The adapter verifies that the configured first token still equals
`context.first_target_token`; baseline/model drift fails closed.

Required observations:

```text
accepted_draft_tokens > 0
speculative_residency_committed_blocks > 0
speculative_residency_rejected_blocks == 0
h2d_copies > 0
h2d_bytes > 0
exact baseline token parity
```

### Rejected boundary case

The adapter changes only the first configured token to `0`, unless the target
token is `0`, in which case it uses `1`. The remaining two configured tokens
are retained but cannot be accepted because verification rejects at the first
position.

Required observations:

```text
accepted_draft_tokens == 0
speculative_residency_committed_blocks == 0
speculative_residency_rejected_blocks > 0
speculative_residency_rejected_d2h_copies == 0
h2d_copies > 0
h2d_bytes > 0
exact baseline token parity
```

## Artifact

Create a separate artifact and verifier rather than changing the existing
schema-v2 parity contract:

```text
tools/speculative_residency_boundary_gate.py
tools/verify_speculative_residency_boundary_gate.py
tools/run_speculative_residency_boundary_gate_remote.sh
```

Schema version `1` stores:

- environment and exact source hashes;
- baseline, accepted-boundary, and rejected-boundary output token IDs;
- real movement and residency counters for every case;
- proposal/acceptance observations;
- exact eviction identities;
- elapsed times as uninterpreted diagnostics;
- `NOT_PROMOTABLE` claim boundaries.

The independent verifier recomputes source hashes and rejects:

- output divergence;
- missing or synthesized movement keys;
- non-positive H2D evidence in either speculative case;
- absent accepted committed-block evidence;
- absent rejected-block evidence;
- any rejected speculative D2H copy;
- malformed or negative counters.

## Failure Boundary

The gate aborts before claiming PASS if:

- prefill does not leave one live sequence at length 255;
- the selected ordinary block is not clean and CPU-valid after writeback;
- clean eviction does not remove its GPU mapping;
- the next decode does not reload through H2D;
- either deterministic adapter does not create its required acceptance class;
- exact token parity fails.

A FAIL artifact is still written when enough state exists to diagnose the
failure.

## Non-Goals

- no performance comparison or slowdown ratio;
- no natural n-gram acceptance claim;
- no learned drafter or MTP evidence;
- no TP4 or second model;
- no context longer than the single 256-token boundary;
- no blockwise decode or sparse attention;
- no runtime/config switch for forced eviction.

## Promotion Boundary

Passing this gate upgrades only loaded-model residency correctness evidence:

```text
accepted reserved-block retention:
  covered
rejected reserved-block discard without D2H:
  covered
real ordinary-block H2D reload:
  covered
end-to-end performance optimization:
  still not established
overall classification:
  NOT_PROMOTABLE
```

