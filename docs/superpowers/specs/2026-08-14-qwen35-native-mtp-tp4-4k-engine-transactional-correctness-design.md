# Qwen3.5 Native MTP TP4/4K Engine Transactional Correctness Design

**Date:** 2026-08-14

**Status:** Approved continuation design

## Objective

Establish the first real-checkpoint, end-to-end `LLMEngine.step()` authority
for the Qwen3.5 native MTP proposal source at tensor parallel size four.

The authority must prove that the learned MTP decoder is genuinely sharded
across ranks, that every rank executes the same proposal lifecycle against
rank-local proposal KV, and that rank 0 remains the sole host token authority
without replaying accepted target prefixes.

The production path under authority is:

```text
all-rank target prefill observation
-> all-rank MTP bootstrap into rank-local proposal KV
-> all-rank sharded MTP proposal step
-> rank-0 greedy token selection
-> token-ID broadcast to ranks 1/2/3
-> deterministic all-rank proposal transaction registration
-> all-rank target first-token and fixed-Q verification
-> all-rank target and side-state transaction publication
-> all-rank MTP proposal-KV finalize commit/rollback
-> rank-0 Scheduler publication
-> all-rank proposal sequence release and cleanup
```

This gate follows the established Qwen3.5 native MTP TP1/4K Engine authority.
It adds the distributed proposal-execution dimension only. It does not add
KV offload, MTP CUDA Graphs, longer contexts, performance thresholds, or a
second learned model structure.

## Claim Scope

The authority is deliberately bounded:

- model: the existing real Qwen3.5-2B checkpoint;
- tensor parallel size: 4;
- ranks: exactly `0, 1, 2, 3`;
- context: exactly 4,096 prompt tokens per request;
- output: exactly 32 greedy tokens per request;
- batch sizes: 1 and 4;
- policies: baseline and native MTP;
- native MTP layer count: 1;
- maximum MTP proposal length: 4;
- target execution: eager;
- MTP execution: eager;
- KV offload: disabled;
- MTP CUDA Graphs: disabled;
- exact baseline/native-MTP greedy output parity;
- exact TP1/native-MTP greedy output parity for the same prompt corpus;
- real target KV and rank-local MTP proposal-KV transactions;
- real recurrent side-state transactions;
- no accepted-prefix target replay; and
- complete four-rank shutdown and GPU-process cleanup evidence.

Passing establishes:

```text
QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED
```

The overall roadmap classification remains `NOT_PROMOTABLE`.

Passing does not establish:

- native MTP with target or proposal KV offload;
- 16K or 32K native-MTP correctness;
- native-MTP latency, throughput, or memory improvement;
- MTP CUDA Graph correctness under TP4;
- a separate learned draft model;
- a second learned-MTP architecture;
- KV8 or KV4;
- production readiness; or
- Phase 1 completion.

## Verified Starting Point

The TP1/4K production Engine authority already proves:

- real checkpoint and MTP checkpoint binding;
- batch 1 and 4 with 4,096-token prompts and 32-token outputs;
- accepted and rejected native MTP proposal tokens;
- real target KV and proposal KV commit/rollback;
- recurrent side-state publication;
- zero accepted-prefix replay;
- finished-sequence release;
- zero pending/bootstrap/transaction/ticket/physical-slot leaks; and
- complete local and remote independent verification.

The existing Qwen3.5 target model factory already supports TP-aware embedding,
full-attention, MLP, normalization, and LM-head construction. Its checkpoint
binder invokes destination parameter `weight_loader` functions, so sharded
parameters receive the correct rank-local checkpoint slice.

The remaining TP1-only boundaries are:

1. `build_qwen35_native_mtp()` rejects every rank other than TP1 rank 0 and
   constructs under `_distributed_construction_context(1, 0)`.
2. `_maybe_register_qwen35_mtp_executor()` only registers at world size one
   and passes hard-coded TP1 construction arguments.
3. `Qwen35MTPProposalExecutor` assumes `forward_step()` returns full logits
   on every rank and performs local `argmax`.
4. `run_spec_first_target_and_proposal_batch()` rejects world size greater
   than one and returns from nonzero ranks before proposal execution.
5. the model-runner proposal runtime activation contract still declares
   ModelRunner proposal execution TP1-only.

## Considered Approaches

### 1. Replicate the full MTP model on rank 0

Keep target execution TP4 but construct and execute the complete MTP decoder
only on rank 0.

This avoids distributed proposal coordination, but duplicates target-owned
embedding and LM-head storage, bypasses the existing TP checkpoint layout,
and cannot naturally maintain rank-local proposal KV that matches the target
attention partition. It also fails to prove that learned proposal execution
is compatible with the production TP model topology.

This approach is rejected.

### 2. Broadcast full MTP logits from rank 0

Execute the sharded MTP decoder on all ranks, gather full logits to rank 0,
then broadcast the full vocabulary row back to all ranks so every executor
can perform the same local `argmax`.

This preserves symmetric executor code but introduces an unnecessary
vocabulary-sized collective for every proposal token. The only distributed
decision required by later MTP steps is the selected token ID.

This approach is rejected.

### 3. All-rank sharded MTP with rank-0 token-ID authority

Construct the native MTP decoder shard and proposal KV store on every rank.
`ParallelLMHead` gathers the vocabulary logits and returns the full row on
rank 0 only. Rank 0 selects the greedy token and broadcasts one `int64` token
per active proposal row. Every rank then continues with the same token while
retaining its local hidden state and rank-local proposal KV.

All ranks deterministically create and finalize matching logical proposal
transactions. Rank 0 returns the host-side `DraftProposal` rows to the generic
runtime. Nonzero ranks return a tensor-free acknowledgement after completing
the same local proposal work.

This is the selected approach because it preserves the existing TP model and
checkpoint topology, minimizes collective payload, and gives every rank a
real proposal-KV lifecycle to validate.

## Architecture

### TP-aware MTP construction

`build_qwen35_native_mtp()` accepts any valid
`tensor_parallel_size/tensor_parallel_rank` pair supported by the target
component factory.

Construction occurs under:

```python
_distributed_construction_context(
    tensor_parallel_size,
    tensor_parallel_rank,
)
```

The MTP decoder layer receives the same rank-local attention geometry as the
target model. Its attention backend builder continues to use MTP layer index
zero, but receives local query and KV head counts from the TP-aware component
factory.

The shared target `embed_tokens` and `lm_head` objects remain the MTP module's
embedding and output projection. Their object identity must hold on every
rank. `mtp.fc.weight` remains replicated because its current checkpoint and
runtime contract are replicated; changing that projection is outside this
gate.

Construction fails closed when:

- TP size or rank is invalid;
- target embedding and LM head are not the same shared-weight pair;
- decoder geometry is not divisible under the existing component rules; or
- the target and MTP rank topology disagree.

### Distributed greedy token selector

The executor receives an explicit token selector with this logical contract:

```python
def select_next_tokens(
    logits: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return identical int64 token IDs of shape [batch_size] on all ranks."""
```

At TP1, the default selector performs local greedy selection and no
collective. Existing TP1 behavior remains unchanged.

At TP4:

1. rank 0 requires one full floating-point logit row per active proposal row;
2. rank 0 computes `argmax` in the existing model output precision;
3. ranks 1, 2, and 3 require `logits is None`, matching
   `ParallelLMHead.forward()`;
4. rank 0 broadcasts only an `int64[batch_size]` token tensor;
5. every rank validates exact shape, dtype, device, and token range; and
6. every rank uses the returned token for the next MTP step.

The selector never broadcasts logits, hidden states, Python objects, or
proposal metadata.

The first implementation may execute one proposal row at a time because the
current executor is row-oriented. The selector still exposes a batch-shaped
contract so a later batched MTP decoder does not require a new distributed
authority boundary.

### All-rank proposal execution

`run_spec_first_target_and_proposal_batch()` keeps the target first-token
forward on all ranks. After the target forward:

- rank 0 computes the first target token rows;
- rank 0 broadcasts the first target token tensor to ranks 1/2/3;
- every rank constructs identical `ModelRunnerProposalInput` rows using its
  rank-local target hidden tensors and the shared token IDs;
- every rank calls the registered proposal executor;
- every rank completes bootstrap, proposal KV reservation, MTP forwards,
  token broadcasts, materialization, and transaction registration; and
- only rank 0 returns the `DraftProposal` tuple to the Engine.

Nonzero ranks return `None` only after local proposal execution succeeds.
Moving the rank gate after proposal execution is mandatory.

The executor is registered on every rank with:

```text
tensor_parallel_size = world_size = 4
tensor_parallel_rank = local rank
```

The existing target-prefill observation command already executes through the
ModelRunner command path on every rank. Finalize and release commands continue
to dispatch to every rank and reuse the existing tensor-free command
acknowledgement surface.

### Deterministic logical transaction identity

Proposal KV is physically rank-local. Physical slot numbers are not required
to match across ranks because each rank owns a different KV head shard.

Logical transaction identity must match across ranks. For each proposal row,
the authority records:

- sequence ID and sequence epoch;
- exact Q;
- proposal transaction ID;
- ordered proposed token IDs;
- staged entry count;
- accepted proposal token count;
- transaction state transitions;
- finalize ticket identity;
- committed logical token count;
- rejected suffix logical token count; and
- post-release local allocation count.

The proposal cache's deterministic allocation and call order are expected to
produce identical transaction and ticket IDs. The gate treats any mismatch as
a distributed correctness failure.

Physical evidence is validated per rank:

- accepted local slots remain committed until sequence release;
- rejected local suffix slots are released during finalize;
- rollback releases every staged local slot;
- release clears all committed local slots; and
- no local slot is shared by two live logical owners.

### Finalize, release, and acknowledgement behavior

Rank 0 remains the generic runtime coordinator. Proposal finalization and
finished-sequence release use the existing acknowledged ModelRunner command
path:

```text
rank-0 command ID
-> worker ranks execute local executor operation
-> workers return tensor-free command acknowledgements
-> rank 0 executes the same operation locally
-> rank 0 validates the complete rank inventory
```

For each command, the gate requires:

- local rank result is `None`;
- worker acknowledgement ranks are exactly `[1, 2, 3]`;
- all workers report success for the same command ID;
- logical transaction or sequence identity matches the rank-0 request;
- the local executor state after the command agrees across ranks; and
- no command is silently retried after a partial failure.

Any worker error poisons the speculative runtime and fails the cell. A
successful rank-0 local finalize cannot mask a worker-rank failure.

### Failure handling

The distributed path fails closed:

- malformed rank-0 logits abort the current local transaction;
- unexpected nonzero-rank logits abort the current local transaction;
- token broadcast failure aborts the current local transaction;
- token shape, dtype, range, or device mismatch aborts the transaction;
- proposal row-count or sequence-order mismatch aborts the proposal group;
- finalize acknowledgement failure poisons the speculative runtime;
- release acknowledgement failure poisons the speculative runtime; and
- cleanup mismatch fails the authority even when generated tokens match.

The gate does not attempt distributed recovery after a failed proposal
collective. The Engine cell exits and the independent verifier requires a
clean failure receipt.

## End-to-End Authority

### Cells

The campaign contains four independent Engine cells:

```text
baseline:b1
native_mtp:b1
baseline:b4
native_mtp:b4
```

Every cell creates and destroys its own Engine and process group. No Engine,
proposal cache, shared-memory channel, CUDA allocator state, or distributed
process group is reused across cells.

### Engine configuration

```text
tensor_parallel_size = 4
max_model_len = 8192
max_num_batched_tokens = 16384
max_num_seqs = batch size
max_num_prefill_tokens_per_step = 1024
enforce_eager = true
kv_offload_mvp0 = false
qwen35_mtp_enabled = native-MTP cell only
qwen35_mtp_cuda_graphs = false
qwen35_mtp_max_proposal_tokens = 4
```

The existing `RuntimeError("Qwen3.5 MTP CUDA graphs require TP1")` remains
valid for this authority. TP4 construction does not relax graph execution.

### Prompt and generation contract

- Every request has exactly 4,096 prompt tokens.
- Prompts are deterministic and distinct by request index.
- Prompt token rows and SHA-256 digests are stored in the artifact.
- Sampling uses `temperature=0.0`.
- `ignore_eos=true`.
- Every request emits exactly 32 output tokens.
- Baseline and native-MTP rows match exactly within each batch size.
- TP4 native-MTP rows match the frozen TP1 authority corpus exactly.

### Required runtime evidence

Native-MTP cells record:

- executor registration and capabilities on ranks `0, 1, 2, 3`;
- native MTP module and proposal physical-store presence on every rank;
- shared embedding and LM-head identity on every rank;
- TP size, TP rank, local query-head count, and local KV-head count;
- target-prefill observation counts per rank;
- bootstrap counts and bootstrapped prefix lengths per rank;
- proposal row, proposal token, accepted token, and rejected token counts;
- first-target and fixed-Q verifier callback counts per rank;
- token broadcast count, shape, dtype, source rank, and ordered token digest;
- proposal transaction and finalize ticket identity per rank;
- proposal-finalization prepare, commit, and rollback receipts per rank;
- recurrent side-state `prepare -> select -> apply -> seal` receipts;
- zero accepted-prefix replay;
- zero runtime poison;
- zero active proposal transactions;
- zero prepared batch-finalization tickets;
- zero pending or bootstrapped executor sequences after release;
- zero allocated rank-local MTP physical slots after release; and
- complete Engine and process-group cleanup.

Across the native-MTP campaign, both accepted and rejected learned proposal
tokens are mandatory. A run with only accepted tokens or only rejected tokens
does not establish transactional suffix behavior.

Baseline cells record zero speculative activity and no native-MTP executor
registration.

## Authority Schema

The artifact schema is:

```text
qwen35.native-mtp-tp4-4k-engine-transactional-correctness.v1
```

Top-level fields:

- `schema_version`;
- `classification`;
- `promotion_status`;
- `claim_scope`;
- `limitations`;
- `source_tree_sha256`;
- `target_model_manifest_sha256`;
- `mtp_model_manifest_sha256`;
- `tp1_authority_sha256`;
- `world_size`;
- `rank_inventory`;
- `gpu_indices`;
- `gpu_process_inventory_before`;
- `gpu_process_inventory_after`;
- `cells`;
- `parity`;
- `rank_authority`;
- `transaction_authority`;
- `cleanup_authority`; and
- `verifier`.

Each cell contains:

- policy, batch size, context length, output length, dtype, tokenizer, and
  prompt digests;
- exact output token rows;
- per-rank module and executor registration evidence;
- per-rank callback counters;
- per-rank token-broadcast evidence;
- per-rank proposal transaction and finalize evidence;
- per-rank proposal KV logical and physical summaries;
- side-state lifecycle receipts;
- target KV transaction receipts;
- rank and worker acknowledgement inventories;
- accepted-prefix replay counters;
- runtime poison state; and
- Engine, process-group, shared-memory, and GPU cleanup receipts.

The artifact classification is
`QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED`. Its promotion status is
`NOT_PROMOTABLE`.

## Fail-Closed Invariants

### Output parity

- Every cell returns exactly one output row per request.
- Every output row has exactly 32 tokens.
- Baseline and native-MTP rows are exactly equal for each batch size.
- TP4 native-MTP rows equal the frozen TP1 authority rows for the same prompt.
- Candidate cells have positive proposal, accepted, first-target, and verify
  counts.
- The native-MTP campaign has positive rejected proposal count.

### Rank participation

- World size is exactly four.
- Rank inventory is exactly `[0, 1, 2, 3]`.
- Worker acknowledgement inventory is exactly `[1, 2, 3]`.
- Every native-MTP rank constructs one executor and one physical store.
- Every native-MTP rank observes the same ordered sequence IDs.
- Every native-MTP rank executes the same number of bootstrap, proposal,
  finalize, and release operations.

### Token authority

- Rank 0 alone receives full LM-head logits.
- Ranks 1, 2, and 3 receive `None` from the LM head.
- Every proposal step has exactly one rank-0 token broadcast.
- Broadcast tensors use `torch.int64` and exact shape `[active_rows]`.
- Ordered selected token IDs and digests match across all ranks.
- No full-logit, hidden-state, Python-object, or metadata broadcast occurs.

### Transaction parity

- Ordered logical transaction IDs match across all ranks.
- Ordered finalize ticket IDs match across all ranks.
- Sequence IDs, epochs, exact Q values, staged counts, accepted counts, and
  rejected counts match across all ranks.
- State transition sequences match across all ranks.
- Rank-local physical slot counts match the logical staged/committed counts.
- Physical slot values may differ across ranks and are never compared as
  globally identical storage addresses.

### No replay

- Accepted-prefix target replay count is exactly zero.
- Target first-token forward count equals the generic runtime's expected
  first-target callback count.
- Target verification forward count equals the fixed-Q callback count.
- No accepted token is materialized by a second full target-forward path.

### Cleanup

- Active proposal transactions are zero on every rank.
- Prepared proposal finalize tickets are zero on every rank.
- Pending and bootstrapped sequence counts are zero on every rank.
- Allocated proposal physical slots are zero on every rank.
- Speculative runtime poison is absent in successful cells.
- ModelRunner command acknowledgements are complete and successful.
- Engine workers, process groups, shared memory, and child processes exit.
- GPU process inventory after the campaign equals the inventory before it.

## Independent Verification

The remote runner writes the authority artifact atomically only after all
cells complete. It then invokes an independent verifier that:

1. loads JSON without importing worker implementation objects;
2. validates the exact schema and classification;
3. recomputes source-tree and checkpoint-manifest bindings;
4. verifies the frozen TP1 authority binding;
5. recomputes prompt and output digests;
6. checks baseline/native-MTP and TP1/TP4 exact token parity;
7. validates complete rank, callback, broadcast, transaction, finalize,
   release, and cleanup inventories;
8. rejects missing, duplicate, malformed, or inconsistent rank evidence;
9. requires positive accepted and rejected learned proposal evidence;
10. requires zero accepted-prefix replay and zero rank-local leaks; and
11. verifies unchanged GPU process inventory.

Local and remote verifier runs must both return:

```json
{"classification":"PASS","failures":[]}
```

## Implementation Boundaries

The implementation must not:

- enable target or proposal KV offload;
- enable TP4 MTP CUDA Graphs;
- broadcast full logits or hidden states;
- replicate the full MTP model on rank 0;
- compare rank-local physical slot IDs as global addresses;
- weaken TP1 authority behavior;
- change baseline generation behavior;
- add performance claims;
- stage, commit, push, stash, reset, clean, or switch worktrees; or
- modify files outside
  `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.

All behavior changes follow strict RED -> GREEN TDD. Real GPU authority is
run only after focused CPU/source-contract tests and the existing native MTP
suite pass.
