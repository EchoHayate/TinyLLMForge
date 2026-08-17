# Autoregressive Draft TP4 Loaded Direct Gate Design

Date: 2026-08-15

Repository: `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`

## Objective

Create a runnable TP4 loaded-checkpoint gate for a Qwen3 independent draft
model and Qwen3.5 target model. The gate must compare target-only and learned
speculative execution for batch 1 and batch 4, preserve exact greedy output
parity, and publish the four-rank learned-drafter authority snapshots already
supported by the engine transport and production TP4 validator.

The implementation is prepared and dependency-light tested locally. This
design does not authorize or claim a GPU execution.

## Current Gap

The repository now has:

- a loaded TP1 target-versus-learned gate;
- TP4 learned-drafter registration and executor contracts;
- `LLMEngine.autoregressive_draft_authority_snapshots()`;
- a production direct-allocator TP4 snapshot validator; and
- mature native-MTP and generic TP4 worker patterns.

It does not have one runnable learned-drafter TP4 gate that combines those
pieces into an exact-parity artifact.

## Considered Approaches

### 1. Independent learned-drafter TP4 gate

Create `tools/autoregressive_draft_tp4_engine_gate.py`, reuse the TP1 prompt,
workload, checkpoint, and tokenizer helpers, and use the existing TP4 engine
snapshot transport and validator.

This is selected because it keeps the TP1 schema stable and avoids coupling
the learned-drafter artifact to native-MTP-specific receipts.

### 2. Generalize the TP1 gate to TP1 and TP4

This would overload the TP1 schema with distributed rank snapshots and change
an already validated authority surface. It is rejected because TP1 and TP4
have materially different evidence contracts.

### 3. Fork the native-MTP TP4 campaign

This would copy model-manifest freezing, TP1 authority dependencies, remote
publication, and MTP-specific receipt normalization. It is rejected because
the learned drafter has separate checkpoint/tokenizer identity and does not
need MTP checkpoint-specific evidence.

## Gate Scope

The gate covers:

```text
target structure: Qwen3.5
proposal source: independent Qwen3 draft model
tensor parallel size: 4
allocator mode: direct
sampling: greedy, temperature=0
max proposal tokens: 4
batch sizes: 1 and 4
```

It does not claim:

- Proposal-KV offload movement;
- 16K or 32K coverage;
- controlled TPOT, TTFT, throughput, or memory improvement;
- TP1/TP4 cross-parity against a frozen TP1 artifact;
- CUDA Graph support; or
- Phase 1 promotion.

## Runtime Structure

The gate uses one target-only engine followed by one learned engine. Each
engine runs batch 1 and batch 4 in that order. The target engine is closed
before the learned engine is created.

The production adapter constructs:

```text
LLM(
  target_model,
  tensor_parallel_size=4,
  enforce_eager=True,
  autoregressive_draft_enabled=<learned>,
  autoregressive_draft_model=<draft model in learned mode>,
  autoregressive_draft_backend="qwen3",
  autoregressive_draft_max_proposal_tokens=4,
  autoregressive_draft_gpu_slot_capacity=<workload capacity>,
  autoregressive_draft_proposal_kv_offload_enabled=False,
)
```

The learned adapter validates registration, activates
`EngineSpeculativeRuntime(model_runner_executor=descriptor)`, and runs normal
`LLMEngine.step()` generation.

After each learned case it:

1. flushes pending hybrid/proposal releases;
2. calls `autoregressive_draft_authority_snapshots(timeout_s=60)`;
3. validates the snapshots with
   `validate_autoregressive_draft_tp4_local_evidence()`; and
4. stores both raw rank snapshots and the normalized summary.

## Distributed Environment

The CLI requires exactly four distinct nonnegative GPU indices and positive
distributed/master ports. During execution it sets and later restores:

```text
CUDA_VISIBLE_DEVICES
TINYVLLM_DIST_PORT
MASTER_PORT
```

No environment mutation occurs during preflight or dependency-light tests.

## Artifact Contract

The schema-v1 payload contains:

```text
schema_version
gate
configuration
checkpoint_identity
tokenizer_contract
workload
cases.batch_1
cases.batch_4
performance_pass_criterion
real_proposal_kv_movement
gate_pass
```

Each case contains:

```text
prompts
target_output_token_ids
learned_output_token_ids
exact_output_parity
acceptance_rows
rank_snapshots
rank_summary
```

`rank_summary.classification` must remain `NOT_PROMOTABLE`, and its promotion
boundary must retain:

```text
real_checkpoint_tp4 = NOT_ESTABLISHED
performance = NOT_ESTABLISHED
real_kv_movement = NOT_ESTABLISHED
phase_1 = NOT_ACHIEVED
```

The local gate payload sets:

```text
performance_pass_criterion = false
real_proposal_kv_movement = false
```

A real loaded execution may establish TP4 parity, but this direct gate cannot
establish Proposal-KV offload movement or performance.

## Failure Behavior

The gate fails closed for:

- invalid GPU inventory or ports;
- checkpoint or tokenizer incompatibility;
- learned registration failure;
- target/learned output mismatch;
- missing acceptance rows;
- malformed, incomplete, or inconsistent rank snapshots;
- nonzero accepted Proposal-KV copy/replay/rematerialization;
- terminal transaction, logical-entry, or physical-slot leaks;
- allocator mode other than direct;
- performance or movement promotion claims; or
- a pre-existing output file.

## Testing Boundary

Dependency-light tests inject fake engines and snapshots to prove:

1. target engine closes before learned engine creation;
2. both engines receive TP4/direct configuration;
3. batch 1 and batch 4 exact output parity is required;
4. learned acceptance rows are retained;
5. raw rank snapshots and normalized summaries are published;
6. malformed rank evidence fails closed;
7. no movement or performance claim is emitted; and
8. CLI/environment validation does not require CUDA.

Passing these tests establishes only the local gate contract.
