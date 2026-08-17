# Qwen3.5 Native MTP TP4 16K Target-KV Offload

## Status

Approved for specification on August 14, 2026 under the user's standing
authorization to continue correctness gates without repeated confirmation.

This design defines a new, independent TP4/16K authority. It must not modify,
replace, parameterize, or reinterpret the established Qwen3.5 native-MTP
TP4/4K authority.

Repository constraints forbid staging, committing, pushing, switching
branches or worktrees, stashing, resetting, or cleaning the checkout.

## Goal

Establish source-bound evidence that the production Qwen3.5 native-MTP runtime
can execute at tensor-parallel world size four with 16,384-token prompts while
the target model uses the production transactional KV-offload and blockwise
attention paths.

The authority must prove:

1. all four ranks load the approved target and native-MTP checkpoints;
2. baseline and native-MTP output token IDs are exactly equal under greedy
   decoding for batch 1 and batch 4;
3. native MTP performs real proposal, acceptance, and rejection work;
4. accepted target KV is committed directly from batch-native verification,
   without an accepted-prefix full-model replay;
5. rejected target KV reservations are rolled back and remain unpublished;
6. target recurrent side state follows prepare, select, apply, and seal or
   rollback semantics on every rank;
7. target speculative residency follows prepare, precommit, seal, and cleanup
   semantics on every rank;
8. 16K batch 4 exceeds the fixed target-KV GPU staging budget;
9. the production target-KV offload path records real positive D2H and H2D
   movement for the native-MTP batch-4 cell;
10. target prefill, ordinary decode, and speculative verification use the
    configured blockwise paths instead of requiring the complete target KV
    history to be simultaneously resident on GPU;
11. proposal MTP KV remains transactionally correct in its existing
    GPU-resident physical store; and
12. all ranks finish with no active residency ticket, KV transaction,
    proposal transaction, sequence, slot, lease, process group, or owned
    child process.

Passing establishes only:

```text
QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED
```

The overall promotion status remains:

```text
NOT_PROMOTABLE
```

## Scope Decision

The first 16K native-MTP gate offloads only target-model KV.

The native-MTP proposal executor continues to use its existing
`ProposalKVCache` and GPU-resident physical store. Its bootstrap and proposal
transactions remain part of the correctness evidence, but its bytes and
entries are excluded from target-KV movement claims.

This split is intentional:

- target speculative residency, blockwise attention, dirty writeback, and
  real movement counters already exist in the production runtime;
- proposal KV offload would require a separate physical store, residency
  protocol, generation binding, eviction policy, and rollback authority;
- mixing both changes into the first 16K gate would make failures ambiguous;
  and
- increasing target GPU slots until the full 16K history fits would not prove
  fixed-budget long-context execution.

This authority therefore proves fixed-budget target-KV execution, not a fixed
total KV-memory bound. Proposal KV still grows with context length and remains
a separately tracked limitation.

## Fixed Inputs

### Workspace and remote

All local changes are confined to:

```text
/Users/bytedance/dev/TinyLLMForge-adaptive-ngram
```

The only authorized remote target is:

```text
sitian@10.232.195.203
```

Remote commands use:

```text
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
```

SSH, source copy, launch, and polling are serial, bounded, and use finite
retries. The gate must never terminate unrelated GPU processes.

### Model identity

Target checkpoint:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-hybrid-state-runs/
qwen35-2b-hybrid-acquire-20260723-222004/model
```

Remote Python:

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
```

Target-model manifest SHA-256:

```text
3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

MTP-checkpoint manifest SHA-256:

```text
9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b
```

Required target identity includes:

- `model_type=qwen3_5`;
- architecture `Qwen3_5ForConditionalGeneration`;
- 24 text layers;
- 18 linear-attention layers; and
- 6 full-attention layers.

### Frozen gate matrix

- schema:
  `qwen35.native-mtp-tp4-16k-target-kv-offload.v1`;
- classification:
  `QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED`;
- claim scope: `qwen35_native_mtp_tp4_16k_target_kv_only`;
- world size: `4`;
- batch sizes: `1` and `4`;
- policies: `baseline` and `native_mtp`;
- prompt tokens: `16384`;
- output tokens: `8`;
- maximum native-MTP proposal tokens: `4`;
- decoding: greedy;
- eager execution: enabled;
- target speculative CUDA Graphs: disabled;
- native-MTP CUDA Graphs: disabled;
- `max_model_len=33024`;
- `max_num_batched_tokens=132096`;
- `max_num_prefill_tokens_per_step=1024`;
- `chunked_prefill_decode_first=False`;
- `chunked_prefill_mixed_batch=False`;
- `kv_offload_mvp0=True`;
- `kv_offload_gpu_blocks=68`;
- `kv_offload_logical_blocks=640`;
- `kv_offload_blockwise_prefill=True`;
- `kv_offload_blockwise_decode=True`;
- `kv_offload_blockwise_blocks=8`; and
- `kvcache_block_size=256`.

At 16K, each request has 64 visible target-KV blocks. Batch 4 therefore has
256 visible logical blocks, which is greater than the fixed 68-slot GPU
staging budget. The authority must not increase that budget to make the run
pass.

## Architecture

### Independent authority overlay

The implementation creates:

```text
tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py
tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
tools/run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh
tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
```

The overlay may reuse side-effect-free validators, prompt construction,
checkpoint identity checks, receipt capture, and orchestration from the
frozen TP4/4K native-MTP authority. It must override the schema,
classification, prompt length, engine configuration, source inventory,
movement contract, and cleanup contract before any campaign starts.

The existing TP4/4K authority files and artifacts remain untouched.

### Production runtime boundary

The authority exercises the existing production components:

- `tinyvllm/engine/llm_engine.py` for speculative selection and residency
  coordination;
- `tinyvllm/engine/model_runner.py` for target KV offload, blockwise
  prefill/decode/spec-verify, and movement counters;
- `tinyvllm/engine/speculative_residency.py` for generation-aware
  prepare/precommit/seal/rollback;
- `tinyvllm/engine/speculative_execution.py` for residency row construction;
- `tinyvllm/engine/qwen35_mtp_executor.py` for native-MTP bootstrap and
  proposal execution;
- `tinyvllm/engine/proposal_kv_cache.py` for proposal-KV transactions; and
- the existing Qwen3.5 packed hybrid model stack.

The authority must not implement a second offload manager, a synthetic copy
path, a replayed transaction, or a second speculative runtime.

If focused RED tests expose a missing production connection, the smallest
runtime change is permitted only after the authority contract test fails for
that exact reason. Runtime changes remain limited to target-KV
residency/blockwise wiring; proposal-KV offload is out of scope.

## Data Flow

### Chunked target prefill

Each 16K prompt is processed in 1,024-token chunks. Target full-attention KV is
written through the production `KVOffloadMVP0` manager.

As logical target blocks outgrow the 68 physical GPU slots:

1. dirty target blocks are written back through the real D2H path;
2. logical block identity remains stable while physical GPU-slot ownership
   changes;
3. blockwise prefill reads only the required prefix windows; and
4. production counters record copies, bytes, block identities, and peak slot
   use.

The proposal MTP bootstrap observes target hidden states as before and writes
proposal KV into its own transaction. It receives no target offload manager.

### Native-MTP speculative round

For each selected sequence:

1. the engine builds target logical block-identity rows;
2. speculative residency prepares a ticket with
   `stage_all_original_blocks=False`;
3. proposal MTP uses committed proposal KV plus its staged suffix to produce
   up to four draft tokens;
4. target batch-native verification maps verifier writes to protected target
   slots;
5. blockwise verifier attention scans logical target history in eight-block
   windows, using production H2D prefetch and eviction;
6. greedy selection identifies the accepted prefix and rejected suffix;
7. target KV precommit publishes only accepted block identities;
8. target recurrent side state publishes the canonical accepted state;
9. target residency and target KV transactions seal;
10. proposal KV commits the accepted proposal prefix and discards its rejected
    suffix; and
11. sequence release removes all remaining transaction and residency state.

No accepted token may be regenerated by a second accepted-prefix target-model
forward.

### Movement accounting

Movement evidence comes only from rank-local production
`KVOffloadMVP0` summaries.

The independent verifier checks:

- D2H copies and bytes;
- H2D copies and bytes;
- moved logical block identities;
- peak resident GPU slots;
- configured GPU and logical block capacities;
- blockwise window size; and
- per-rank agreement on the engine configuration.

Proposal-KV allocation, CUDA tensor copies inside the proposal executor, and
synthetic helper copies are excluded.

## Result Contract

Every result cell contains:

- model and MTP manifest identity;
- source-tree and authority-source identity;
- policy, batch size, rank inventory, and selected GPU inventory;
- complete engine configuration;
- prompt-token digest and output-token rows;
- target prefill and native-MTP runtime observations;
- native proposal, acceptance, rejection, and selected-token counts;
- target KV prepare, precommit, commit, rollback, seal, and release evidence;
- target recurrent-state lifecycle evidence;
- proposal-KV bootstrap and proposal transaction history;
- speculative residency receipts;
- per-rank production movement summaries;
- peak target-KV GPU-slot use;
- poison state; and
- cleanup state.

The following invariants are mandatory:

- baseline/native output rows are byte-for-byte equal in each batch cell;
- every native cell has positive proposal, accepted-draft, and rejected-draft
  counts;
- `accepted_prefix_target_replays == 0`;
- all ranks agree on selected tokens and transaction decisions;
- proposal bootstrap commits exactly 16,384 entries per sequence before
  proposal rounds begin;
- proposal committed length evolves only through accepted-prefix commits;
- target rejected identities never appear in committed identities;
- native batch 4 has production `d2h_copies > 0` and `d2h_bytes > 0`;
- native batch 4 has production `h2d_copies > 0` and `h2d_bytes > 0`;
- native batch-4 peak target slots are no greater than 68;
- no cell reports a synthetic movement source; and
- cleanup is complete on every rank.

Positive movement proves that the production path executed. It is not, by
itself, a performance or traffic-reduction claim.

## Failure Semantics

The gate fails closed for:

- wrong schema, classification, scope, model, MTP, or source identity;
- missing or duplicated rank, policy, batch, or sequence;
- incorrect prompt length or engine configuration;
- baseline/native token mismatch;
- missing proposal, acceptance, or rejection;
- rank disagreement;
- accepted-prefix target replay;
- malformed target KV, recurrent-state, residency, or proposal-KV lifecycle;
- rejected target or proposal state becoming visible;
- missing real D2H or H2D movement in the native batch-4 cell;
- movement evidence not originating from `KVOffloadMVP0`;
- target peak GPU slots exceeding 68;
- runtime poison;
- changed selected-GPU process inventory caused by the campaign;
- incomplete cleanup;
- malformed or tampered result/source manifests; or
- replay of an existing run directory.

Failure writes `authority.failed`. Only a fresh successful campaign followed
by successful independent source-bound verification may write `authority`.

## Test Strategy

Validation proceeds in this order:

1. focused RED contract tests for the 16K overlay constants and worker
   configuration;
2. RED tests proving the verifier rejects zero, synthetic, malformed, or
   inconsistent movement evidence;
3. RED tests proving the verifier rejects target peak-slot overflow;
4. RED tests proving bootstrap and proposal transaction histories are
   distinguished correctly at 16K;
5. minimal implementation followed by focused GREEN tests;
6. all new 16K authority tests;
7. frozen TP4/4K gate, verifier, and executor regressions;
8. Python compilation, runner `bash -n`, and scoped `git diff --check`;
9. remote direct tests using the remote Torch environment;
10. one fresh real-checkpoint TP4 GPU campaign covering baseline/native and
    batch 1/4;
11. independent verification against the copied source tree; and
12. artifact and cleanup audit followed by a handoff update.

The campaign is not allowed to pass by weakening acceptance/rejection,
movement, slot-budget, exact-parity, or cleanup requirements.

## Promotion Boundary

Passing this authority advances native MTP from TP4/4K target-KV-resident
correctness to TP4/16K production target-KV-offload correctness.

It does not establish:

- proposal MTP KV offload;
- a fixed total target-plus-proposal KV-memory bound;
- TP1/16K native-MTP parity;
- 32K context;
- a second learned model structure;
- independent draft-model execution;
- variable-length speculative CUDA Graphs;
- KV8 or KV4;
- prefix sharing, deduplication, or reference counting;
- TPOT, TTFT, throughput, memory, or H2D-byte improvement;
- production readiness; or
- Phase 1 promotion.

Those remain explicit later gates rather than inferred consequences of this
authority.
