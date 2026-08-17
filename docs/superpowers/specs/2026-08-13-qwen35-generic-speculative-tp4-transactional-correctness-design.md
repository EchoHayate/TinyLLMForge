# Qwen3.5 Generic Speculative TP4 Transactional Correctness

## Status

The design is approved for implementation planning.

This specification promotes the established Qwen3.5 TP1 generic speculative
transaction contract to a real four-rank Engine authority. It deliberately
uses a new, narrow Qwen3.5-specific gate while reusing the production generic
speculative runtime and existing TP4 Engine construction.

Repository constraints forbid staging, committing, pushing, switching
branches, creating a worktree, stashing, resetting, or cleaning the checkout.

## Goal

Establish source-bound evidence that the generic speculative runtime executes
the real Qwen3.5 hybrid architecture under tensor parallel world size four
without changing exact greedy output or publishing rejected speculative state
on any rank.

The authority must prove:

1. the approved Qwen3.5 checkpoint is loaded by all four ranks;
2. baseline and generic n-gram candidate runs produce exactly equal greedy
   output tokens;
3. batch-1 and batch-4 cells execute as real multi-sequence Engine runs;
4. every rank executes the generic `spec_first_target` and `spec_verify`
   callback path;
5. all ranks agree on proposal, acceptance, rejection, and committed-input
   decisions for every sequence;
6. accepted full-attention KV is committed from the batch-native target
   verification result without a second accepted-prefix model forward;
7. rejected full-attention KV suffixes are rolled back;
8. Qwen3.5 convolution and recurrent state publishes exactly the selected
   consumed-input checkpoint on every rank;
9. rejected recurrent side state remains invisible;
10. callback collectives, speculative residency phases, and real KV movement
    are observed on all ranks;
11. every process group, worker process, state lease, prepared transaction,
    and Engine instance is cleaned up; and
12. the independent verifier fails closed when any required rank-local or
    cross-rank evidence is missing or inconsistent.

Passing this gate establishes only:

```text
SECOND_MODEL_TP4_4K_ESTABLISHED
```

It does not establish:

- 16K or 32K context;
- TPOT, TTFT, throughput, or memory improvement;
- KV offload performance or reduced H2D traffic;
- learned-drafter or native-MTP support;
- KV8 or KV4 support;
- production readiness; or
- Phase 1 completion.

The overall Phase 1 status remains `NOT_PROMOTABLE`.

## Fixed Inputs

### Workspace

All local changes are confined to:

```text
/Users/bytedance/dev/TinyLLMForge-adaptive-ngram
```

### Remote execution

The only authorized remote target is:

```text
sitian@10.232.195.203
```

The runner must use:

```text
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
```

SSH, rsync, and remote status polling must be serial, bounded, and use finite
retries. The runner must not create a persistent local unified-exec process.

### Model

The approved checkpoint is:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-hybrid-state-runs/
qwen35-2b-hybrid-acquire-20260723-222004/model
```

The approved manifest is:

```text
/data00/home/sitian/sitian-workspace01/tllm/
qwen35-hybrid-state-runs/
qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json
```

Its SHA-256 is:

```text
3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

The model identity must be bound as:

- `model_type=qwen3_5`;
- architecture `Qwen3_5ForConditionalGeneration`;
- 24 text layers;
- 18 linear-attention layers; and
- 6 full-attention layers.

The remote Python executable is:

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
```

### Gate matrix

The first TP4 authority fixes:

- world size: `4`;
- batch sizes: `1` and `4`;
- context tokens: `4096`;
- maximum output tokens: `8`;
- draft source: host-side n-gram;
- n-gram size: `3`;
- maximum proposal tokens: `4`;
- decoding: greedy; and
- baseline/candidate prompts: identical within each cell.

The prompts must produce at least one accepted and one rejected draft token in
each batch-size cell. A candidate cell with no rejection cannot establish
rollback correctness.

## Verified Starting Point

### Qwen3.5 TP1 authority

The existing TP1 authority establishes:

- real Qwen3.5 Engine execution;
- exact greedy baseline/candidate parity;
- batch-1 and batch-4 execution;
- accepted and rejected draft tokens;
- prepared full-attention KV commit and rollback;
- recurrent side-state prepare, selection, apply, seal, and rollback;
- canonical consumed-input mapping;
- no accepted-prefix model replay;
- lease and poison cleanup; and
- independent source-bound verification.

Its classification is:

```text
SECOND_MODEL_TP1_ESTABLISHED
```

The TP4 design must preserve its state semantics rather than introducing a
second speculative implementation.

### Standard Qwen3 generic TP4 authority

The existing standard-decoder TP4 gate establishes the distributed evidence
shape for:

- four-rank Engine execution;
- callback and collective profiling;
- speculative residency acknowledgement;
- real KV movement accounting;
- rank-local result collection;
- batch-1 and batch-4 cells;
- bounded remote execution; and
- independent verification.

That authority does not contain Qwen3.5 recurrent side-state lifecycle or
consumed-input checkpoint evidence.

### Qwen3.5 ordinary TP4 Engine authority

The existing Qwen3.5 TP4 Engine backend establishes:

- real checkpoint loading on four ranks;
- TP4 process-group construction;
- ordinary Engine stepping;
- rank-local process evidence; and
- bounded cleanup.

It does not exercise the generic speculative transaction.

## Considered Approaches

### 1. Parameterize the existing standard Qwen3 TP4 gate

This would add a model profile and optional recurrent-state evidence to the
existing gate.

It reduces file count, but it makes one authority schema describe two
materially different state models. A verifier branch could accidentally make
Qwen3.5 side-state evidence optional or alter the already-established Qwen3
authority. This approach is rejected.

### 2. Extend the Qwen3.5 TP1 gate in place

This would add distributed launch, rank evidence, collective profiling, and
TP4 cleanup to the existing TP1 gate.

It reuses more gate code, but it turns a frozen TP1 authority into a
multi-world-size schema and risks invalidating prior evidence. It also makes
TP1-only and TP4-only requirements harder to audit independently. This
approach is rejected.

### 3. Add a narrow Qwen3.5 TP4 gate and share runtime helpers

This creates a new Qwen3.5 TP4 gate, worker, independent verifier, remote
runner, and focused contract tests. It reuses production Engine/runtime code
and may extract only side-effect-free source helpers when duplication would
otherwise cause drift.

This is the selected approach because it:

- preserves existing Qwen3 TP4 and Qwen3.5 TP1 schemas;
- makes all-rank hybrid-state evidence mandatory;
- gives the independent verifier a single, unambiguous contract;
- limits regression risk; and
- keeps the authority classification narrow.

## Architecture

### New authority surface

The implementation introduces:

```text
tools/qwen35_generic_speculative_tp4_gate.py
tools/qwen35_generic_speculative_tp4_worker.py
tools/verify_qwen35_generic_speculative_tp4_gate.py
tools/run_qwen35_generic_speculative_tp4_gate_remote.sh
tools/test_qwen35_generic_speculative_tp4_gate.py
```

These files are authority and orchestration code. They must not contain a
parallel speculative runtime, a model-specific KV implementation, or a
fallback accepted-prefix replay path.

### Reuse boundaries

The new gate reuses:

1. the production generic speculative runtime and transactional KV path;
2. the Qwen3.5 side-state provider already established by TP1;
3. Qwen3.5 TP4 Engine configuration and rank construction patterns;
4. standard generic TP4 callback, collective, residency, and movement
   profiling sources; and
5. existing source-tree and model-manifest hashing conventions.

Shared helpers may be extracted only when they are:

- deterministic;
- side-effect-free;
- independently testable; and
- unable to weaken either existing authority schema.

The existing Qwen3 TP4 gate and Qwen3.5 TP1 gate remain behaviorally frozen.

### Rank roles

All four ranks load the model and execute target callbacks.

Rank 0 additionally:

- owns the user-visible Engine step loop;
- records baseline and candidate output tokens;
- assembles scheduler and transaction summaries;
- gathers rank-local evidence; and
- writes the final raw result.

Ranks 1 through 3:

- execute model callbacks and collectives;
- record rank-local callback and side-state receipts;
- record rank-local residency, movement, and cleanup evidence; and
- return bounded evidence to rank 0.

Rank 0 is not allowed to synthesize missing rank evidence. A missing,
duplicate, stale, or malformed rank record fails the run.

## Transaction Semantics

### Canonical committed-input mapping

For each sequence:

```text
verify_input_count = max(0, proposal_token_count - 1)
committed_tail_input_count = min(
    accepted_draft_count,
    verify_input_count,
)
committed_input_count = 1 + committed_tail_input_count
```

The mapping is computed once by the generic runtime and carried into both the
full-attention KV decision and Qwen3.5 side-state selection.

Every rank must report the same:

- sequence ID;
- proposal token count;
- accepted draft count;
- rejected draft count;
- verify input count; and
- committed input count.

A rank disagreement is a transaction-integrity failure even when final output
tokens happen to match.

### Full-attention KV

The gate requires evidence that:

- speculative target KV is written into prepared/private slots;
- accepted slots are published through the prepared KV transaction;
- rejected suffix slots are released or restored;
- accepted KV is not reconstructed by a second model forward; and
- the recorded movement bytes originate from the production block manager or
  KV movement profiler, not from a synthetic tensor copy.

### Recurrent side state

For each `(rank, handle_id, sequence_id)` lifecycle, the worker records:

1. `prepare`;
2. target callback observations;
3. `select(committed_input_count)`;
4. `apply`;
5. `seal`; or
6. `rollback`.

The successful path requires exactly one selected checkpoint and exactly one
sealed publication per active sequence on every rank.

The failed or rejected path must prove that private rejected state is
discarded and no rejected checkpoint becomes the live lease state.

Lifecycle aggregation is keyed by:

```text
(rank, handle_id, sequence_id)
```

It must never aggregate only by batch, handle, callback invocation, or rank.

### Cross-rank agreement

The worker derives a deterministic digest for each sequence transaction from:

- cell identity;
- sequence identity;
- proposal tokens;
- acceptance mask;
- committed-input count;
- KV publication decision; and
- side-state selected-checkpoint identity.

The verifier requires identical semantic digests across all ranks. Rank-local
physical slot numbers and TP-sharded tensor metadata are excluded because
they may legitimately differ.

## Execution Data Flow

For each batch-size cell:

1. construct a baseline TP4 Engine;
2. load and verify the approved checkpoint on all ranks;
3. run the baseline prompts greedily;
4. close the baseline Engine and prove four-rank cleanup;
5. construct a fresh candidate TP4 Engine;
6. enable the host n-gram draft adapter;
7. run identical prompts through the generic speculative path;
8. collect rank-local callback, collective, transaction, residency, movement,
   and cleanup evidence;
9. compare exact baseline and candidate output token IDs;
10. verify proposal acceptance includes both accepted and rejected tokens;
11. close the candidate Engine and prove four-rank cleanup; and
12. write one cell result only after all required evidence is complete.

The two cells run serially. Engines, rendezvous ports, run directories, and
rank processes are not reused across cells.

## Frozen Result Schema

The authority schema is:

```text
qwen35.generic-speculative-tp4-transactional-correctness.v1
```

The top-level result includes:

- schema and classification;
- source manifest and source-tree hashes;
- checkpoint path and manifest hash;
- model identity;
- world size and TP configuration;
- fixed gate parameters;
- runner state-machine evidence;
- one batch-1 cell;
- one batch-4 cell;
- aggregate cleanup evidence; and
- explicit unsupported/not-established claims.

Each cell includes:

- baseline output token IDs per sequence;
- candidate output token IDs per sequence;
- proposal, acceptance, and rejection counts;
- per-sequence committed-input records;
- four rank-local evidence records;
- cross-rank semantic digests;
- callback and collective profiles;
- speculative residency phases;
- real KV movement counters and provenance;
- no-replay evidence;
- side-state lifecycle receipts;
- lease and poison status; and
- process/Engine cleanup status.

Each rank record includes:

- rank and world size;
- process identity;
- checkpoint-loaded receipt;
- callback counts and callback identities;
- collective counts by operation;
- per-sequence side-state lifecycle receipts;
- transaction semantic digests;
- residency observations;
- movement observations;
- process-group cleanup; and
- worker exit status.

## Independent Verification

The verifier operates in a fresh Python process and reads only the emitted
authority bundle and source tree.

It must reject the bundle unless all of the following hold:

1. schema equals the frozen v1 schema;
2. source manifest and source tree match the executed sources;
3. the checkpoint manifest hash matches the approved value;
4. model identity matches the Qwen3.5 hybrid architecture;
5. world size equals four;
6. rank set is exactly `{0, 1, 2, 3}` in every cell and phase;
7. batch-size cells are exactly `{1, 4}`;
8. context length equals 4096;
9. baseline and candidate token IDs are exactly equal for every sequence;
10. both accepted and rejected draft tokens exist in each cell;
11. committed-input counts satisfy the canonical mapping;
12. every sequence has complete KV and side-state transaction evidence on
    every rank;
13. cross-rank semantic digests agree;
14. callback and collective evidence is present on every rank;
15. speculative residency phases are complete;
16. KV movement counters have production provenance;
17. no accepted-prefix model replay occurred;
18. no rank reports a live lease, poisoned runtime, prepared transaction, or
    unclosed Engine after cleanup;
19. the remote runner completed through its non-replayable status machine;
    and
20. unsupported claims remain explicitly false.

Only then may it emit:

```text
classification=SECOND_MODEL_TP4_4K_ESTABLISHED
```

The verifier must not infer success from a worker exit code, raw-result
presence, token parity alone, aggregate rank-0 summaries, or a prior verifier
output.

## Remote Runner

The shell runner follows the existing non-replayable campaign state machine:

```text
campaign.status
campaign.pid
campaign.exit_code
```

It must:

- validate the Kerberos cache before transfer;
- use serial bounded SSH and rsync;
- disable SSH connection sharing;
- transfer only the required source snapshot;
- allocate fresh non-ephemeral rendezvous ports;
- launch one bounded remote campaign;
- prevent replay of a completed or failed campaign;
- poll using short-lived SSH commands;
- retrieve raw artifacts before local verification;
- preserve failed evidence;
- run the independent verifier locally in a fresh process; and
- never convert a timeout, transport failure, missing status, or partial
  artifact into success.

The runner writes `authority.failed` when execution or verification fails and
`authority` only after independent verification passes.

## Error Handling

The gate fails closed on:

- checkpoint or manifest mismatch;
- any rank failing to load the checkpoint;
- missing, duplicated, or unexpected rank records;
- rank disagreement on proposal or committed-input semantics;
- absent accepted or rejected drafts;
- token parity mismatch;
- incomplete side-state lifecycle;
- side-state checkpoint mismatch;
- rejected state becoming live;
- missing KV commit or rollback evidence;
- accepted-prefix model replay;
- callback or collective absence on any rank;
- synthetic or unbound KV movement evidence;
- missing residency phases;
- leaked leases or prepared transactions;
- runtime poison;
- worker, process-group, or Engine cleanup failure;
- malformed or stale runner status;
- source hash mismatch; or
- independent-verifier failure.

Failure evidence is retained. The runner must not automatically retry a
semantic failure under the same campaign identity.

## Testing Strategy

Implementation follows strict RED, minimal implementation, GREEN sequencing.

### Contract RED tests

Focused tests first establish that no existing tool can satisfy the new
schema. They cover:

- frozen schema and classification;
- fixed model/checkpoint identity;
- exact four-rank set;
- batch-1 and batch-4 matrix;
- canonical committed-input mapping;
- rank-local side-state aggregation key;
- cross-rank semantic agreement;
- accepted and rejected draft requirements;
- no-replay requirement;
- movement provenance;
- cleanup completeness;
- runner SSH/Kerberos/status-machine constraints; and
- explicit `NOT_PROMOTABLE` claims.

### Worker RED tests

Worker tests use controlled fakes only to validate orchestration and evidence
assembly. They must not be accepted as GPU or checkpoint authority.

They cover:

- four-rank evidence collection;
- missing-rank rejection;
- semantic-digest mismatch;
- duplicate lifecycle receipt rejection;
- incomplete callback/collective rejection;
- rank-local cleanup failure;
- baseline/candidate token mismatch; and
- failed-artifact preservation.

### Runtime regressions

Any production behavior change requires a focused failing runtime regression
before implementation. The existing Qwen3 TP4 and Qwen3.5 TP1 gate tests must
remain green and their schemas must not change.

### Real authority

The final gate requires:

- real remote GPUs;
- the approved Qwen3.5 checkpoint;
- four live ranks;
- batch-1 and batch-4;
- 4K context;
- exact greedy parity;
- accepted and rejected drafts;
- complete all-rank side-state transactions;
- callback/collective/residency/movement evidence;
- no replay;
- cleanup; and
- fresh independent verification.

Local fakes, CPU-only runs, ordinary non-speculative TP4 runs, or inherited TP1
evidence cannot substitute for this authority.

## Validation Commands

The implementation plan must include focused commands for:

```bash
python -m pytest -q tools/test_qwen35_generic_speculative_tp4_gate.py
```

It must also include the existing focused authority regressions:

```bash
python -m pytest -q \
  tools/test_generic_speculative_tp4_gate.py \
  tools/test_qwen35_generic_speculative_tp1_gate.py
```

Before remote execution:

```bash
python -m py_compile \
  tools/qwen35_generic_speculative_tp4_gate.py \
  tools/qwen35_generic_speculative_tp4_worker.py \
  tools/verify_qwen35_generic_speculative_tp4_gate.py

bash -n tools/run_qwen35_generic_speculative_tp4_gate_remote.sh
git diff --check
```

The remote authority command is:

```bash
bash tools/run_qwen35_generic_speculative_tp4_gate_remote.sh
```

The exact final regression set may expand when focused RED tests expose
production runtime defects.

## Promotion Boundary

This gate answers one question:

> Does the real Qwen3.5 hybrid model preserve exact generic speculative
> transaction semantics across four tensor-parallel ranks at 4K context?

A passing result adds only the second-model TP4/4K row to the Phase 1 evidence
matrix.

The next independent gates remain:

1. 16K and 32K transactional correctness;
2. controlled TPOT, TTFT, throughput, memory, KV H2D byte, and acceptance
   measurement;
3. a learned drafter or native-MTP source through the same runtime;
4. real KV offload and migration interaction; and
5. broader failure and recovery campaigns.

No result from this gate may be reported as Phase 1 completion.
