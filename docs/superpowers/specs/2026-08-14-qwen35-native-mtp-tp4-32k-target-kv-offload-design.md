# Qwen3.5 Native MTP TP4 32K Target-KV Offload Design

## Status

Approved under the user's standing authorization to continue the ordered
correctness-gate program without repeated confirmation.

This design creates an independent TP4/32K authority overlay. It must not
modify, replace, parameterize, or reinterpret the established native-MTP
TP4/16K target-KV-offload authority.

Repository constraints forbid staging, committing, pushing, switching
branches or worktrees, stashing, resetting, cleaning, or terminating
unrelated GPU processes.

## Goal

Establish source-bound evidence that production `LLMEngine.step()` executes
Qwen3.5 native MTP at tensor-parallel world size four with 32,768-token
prompts while the target model uses the production transactional KV-offload,
chunked-prefill, blockwise-attention, and speculative-residency paths.

The authority must prove:

1. all four ranks load the approved target and native-MTP checkpoints;
2. baseline and native-MTP greedy output token IDs are exactly equal for
   batch 1 and batch 4;
3. native MTP performs positive proposal, acceptance, and rejection work;
4. accepted target KV commits directly from batch-native verification;
5. accepted-prefix target replay remains zero;
6. rejected target KV reservations remain unpublished and are rolled back;
7. target recurrent side state follows prepare/select/apply/seal semantics;
8. speculative target residency follows prepare/precommit/seal semantics;
9. target KV uses real production H2D and D2H movement for both native cells;
10. peak target-KV GPU residency never exceeds the fixed 68-slot budget;
11. proposal KV preserves its existing transaction, commit, reject, and
    sequence-release semantics; and
12. every rank exits with zero active ticket, transaction, sequence, physical
    slot, process group, shared-memory allocation, and owned child process.

Passing establishes only:

```text
QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED
```

Promotion remains:

```text
NOT_PROMOTABLE
```

## Scope

This authority offloads only target-model KV.

The native-MTP proposal executor retains its existing GPU-resident
`ProposalKVCache` and physical-slot store. Proposal-KV transactions and final
release inventory remain correctness evidence, but proposal-KV bytes are not
included in target-KV movement claims.

If the 32K campaign exposes a proposal-KV capacity or memory limit, retain
that failure as architecture evidence. Do not hide it by increasing the
target-KV GPU-slot budget, reducing the prompt length, simulating movement,
or weakening exact parity.

Out of scope:

- proposal-KV offload;
- TP1/32K;
- native-MTP performance improvement;
- KV8/KV4;
- a second learned draft-model structure;
- production-readiness or Phase 1 promotion.

## Frozen Inputs

Workspace:

```text
/Users/bytedance/dev/TinyLLMForge-adaptive-ngram
```

Remote:

```text
sitian@10.232.195.203
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
```

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

Checkpoint identities:

```text
target:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0

native MTP:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b
```

Frozen matrix:

```text
schema:
  qwen35.native-mtp-tp4-32k-target-kv-offload.v1

classification:
  QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED

policies:
  baseline
  native_mtp

world size:                         4
batch sizes:                        1 and 4
prompt tokens:                      32768
output tokens:                      8
maximum proposal tokens:            4
decoding:                           greedy
enforce eager:                      true
max_model_len:                      33024
max_num_batched_tokens:            132096
max_num_prefill_tokens_per_step:     1024
chunked_prefill_decode_first:       false
chunked_prefill_mixed_batch:        false
kv_offload_mvp0:                    true
kv_offload_gpu_blocks:              68
kv_offload_logical_blocks:         640
kv_offload_blockwise_prefill:       true
kv_offload_blockwise_decode:        true
kv_offload_blockwise_blocks:         8
kvcache_block_size:                256
target speculative CUDA graphs:    false
native-MTP CUDA graphs:             false
```

Each 32K request exposes 128 target-KV blocks. Batch 4 exposes 512 logical
blocks, which must execute through the fixed 68-slot target staging budget.

## Architecture

Create an isolated overlay:

```text
tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py
tools/verify_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh
tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
```

The 32K gate loads the frozen 16K native authority as a private module and
overrides only:

- schema and classification;
- prompt length;
- required limitations;
- source inventory;
- default worker and verifier paths; and
- the stronger 32K movement requirement.

The 32K worker loads the frozen 16K worker, replaces its `gate` module with
the 32K overlay, and delegates production execution. It must not duplicate or
fork runtime logic.

The independent verifier loads the 32K gate and reuses the frozen verifier's
canonical artifact, source-manifest, source-tree, result-digest, and cleanup
checks.

The derived remote runner must preserve:

- approved checkout validation;
- Kerberos and SSH preflight;
- fresh non-ephemeral port selection;
- four-idle-GPU selection without process termination;
- complete source-tree binding;
- bounded retries and polling;
- failed-artifact retention;
- remote independent verification;
- copied-source local independent verification; and
- one terminal campaign marker.

## Result Contract

The canonical result contains exactly:

```text
baseline:b1
native_mtp:b1
baseline:b4
native_mtp:b4
```

Validation must require:

- exact baseline/native output equality for each batch size;
- four rank snapshots per cell;
- native proposal, accepted-token, and rejected-token totals greater than
  zero;
- first-target target forwards equal first-target callbacks;
- verify target forwards greater than or equal to verify callbacks;
- zero accepted-prefix target replay;
- release rows equal sequence IDs `0..batch_size-1` on every rank;
- target-KV receipts `prepare -> commit` for every sequence;
- side-state receipts `prepare -> select -> apply -> seal` for every sequence;
- residency prepare/precommit/seal participation from all four ranks;
- `gpu_blocks == 68`, `logical_blocks == 640`, and
  `peak_resident_blocks <= 68` on every rank;
- positive H2D copies and bytes for native batch 1 and native batch 4;
- positive D2H copies and bytes for native batch 1 and native batch 4;
- movement provenance `engine.kv_offload_summaries`;
- no runtime poison;
- zero active proposal transactions, tickets, sequences, and physical slots;
- rank exit codes `[0, 0, 0, 0]`;
- destroyed process groups, released shared memory, and no owned children;
- unchanged selected-GPU process inventory; and
- exact source and checkpoint identity.

The required limitations are:

```text
phase1_not_promotable
proposal_kv_offload_not_established
tp1_32k_not_established
performance_not_established
kv_quantization_not_established
second_learned_structure_not_established
```

## Failure Semantics

- A source, schema, manifest, or canonical-result mismatch is a hard failure.
- A worker failure retains `authority.failed` and cannot publish authority.
- A GPU idle-gate failure exits without starting a workload.
- A shared-GPU process appearing after preflight may cause a retained resource
  failure; the runner must not kill it.
- OOM in target attention, target KV, proposal KV, or model initialization is
  a real failure, not permission to weaken the contract.
- Any runtime poison, incomplete release inventory, live transaction, or
  cleanup mismatch is a hard correctness failure.

## Test Strategy

Local RED/GREEN tests cover:

1. frozen constants and source inventory;
2. isolation from the 16K authority;
3. default 32K worker and verifier dispatch;
4. worker prompt length and engine configuration;
5. b1 and b4 real H2D/D2H requirements;
6. release-row inventory on all ranks;
7. exact parity and accepted-prefix replay rejection;
8. canonical manifest, result, and artifact inventory;
9. runner host, Kerberos, SSH, source archive, GPU gate, polling, and verifier
   contracts; and
10. failed-artifact retention.

The final authority requires a fresh remote campaign and a fresh explicit
local verifier run against the downloaded source tree.

## Claim Boundary

Success proves one narrow point:

```text
Qwen3.5 native MTP
TP4
32K prompt
batch 1 and 4
production Engine execution
exact greedy parity
transactional target-KV offload and bounded residency
transactional proposal-KV correctness
source-bound independent verification
```

It does not prove proposal-KV offload, performance improvement, KV8/KV4, a
second learned structure, production readiness, or Phase 1 completion.
