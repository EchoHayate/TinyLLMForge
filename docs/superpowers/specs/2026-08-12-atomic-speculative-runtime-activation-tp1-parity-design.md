# Atomic Speculative Runtime Activation and TP1 Parity Design

**Date:** 2026-08-12  
**Repository:** `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`  
**Status:** approved for implementation planning  
**Scope:** source-agnostic runtime activation and the first real loaded-model
TP1 exact-greedy parity gate

## Problem

The generic speculative control-plane is locally connected through
`LLMEngine.step()`, but activation currently requires two independent calls:

1. install an enabled `SpeculativeSelectionConfig` into the Scheduler;
2. install a matching `EngineSpeculativeRuntime` into the engine.

`validate_engine_speculative_runtime()` requires both proposal limits to match,
so callers can leave the engine in a partially configured state if the first
call succeeds and the second fails. There is also no real loaded-model artifact
showing that the connected engine path produces the same greedy tokens as
ordinary decoding.

The next gate must solve both problems before speculative transactions are
connected to KV offload residency.

## Goals

1. Provide one source-agnostic engine API that activates Scheduler selection
   and the runtime as one logical operation.
2. Validate every dependency before publishing either active state.
3. Preserve default-off ordinary execution and existing fail-closed behavior.
4. Keep proposal-source behavior behind the existing adapter and lifecycle
   contracts.
5. Produce a reproducible TP1 artifact comparing ordinary and speculative
   greedy decoding on a real loaded target model.
6. Record token parity, proposal and acceptance counts, target invocations,
   source identity, model identity, configuration, and failure boundaries.

## Non-Goals

- learned-drafter or MTP adapter implementation;
- speculative ownership integration with `KVOffloadMVP0`;
- TP4, multi-model, long-context, or performance promotion;
- stochastic speculative decoding;
- transactional recurrent or convolution state;
- CUDA Graphs, verifier/sampling fusion, or collective optimization.

Those remain required later promotion gates, but combining them with the first
activation/parity gate would make failures harder to attribute.

## Public Boundary

`LLMEngine` will expose one new method:

```python
def activate_speculative_runtime(
    self,
    runtime: EngineSpeculativeRuntime,
) -> None:
    ...
```

The proposal limit is derived from
`runtime.draft_adapter.capabilities.max_proposal_tokens`; callers do not pass a
second independently configurable value.

The method constructs:

```python
SpeculativeSelectionConfig(
    enabled=True,
    max_proposal_tokens=proposal_limit,
)
```

and activates that selection together with the supplied runtime.

The API remains source-agnostic. It may inspect capability fields and lifecycle
method availability, but it must not branch on adapter class, model name, or
proposal-source name.

## Validation and Publication

Activation has two phases.

### Phase 1: Prepare

Before mutating the Scheduler or engine:

1. validate `EngineSpeculativeRuntime`;
2. validate adapter capabilities, batch support, positive proposal limit, and
   callable `propose_batch`;
3. validate the ModelRunner callback bridge;
4. validate the optional lifecycle contract;
5. construct and validate the exact enabled selection config;
6. reject activation when another different runtime or proposal limit is
   already active.

Runtime validation must support a candidate selection config without requiring
that config to have already been published into the Scheduler. This may be
implemented by passing the candidate config to the validator or by separating
runtime-only validation from Scheduler/runtime compatibility validation.

### Phase 2: Publish

After preparation succeeds:

1. snapshot the current Scheduler speculative selection config and current
   engine runtime;
2. install the candidate Scheduler selection;
3. publish the candidate engine runtime;
4. if either publication step raises, restore both snapshots before returning
   the error.

No request may observe a state where selection is enabled without the matching
runtime.

## Existing Low-Level APIs

`Scheduler.install_speculative_selection()` remains the Scheduler's validation
and publication primitive.

`LLMEngine.install_speculative_runtime()` remains available to focused tests
and compatibility callers during this gate, but normal serving activation uses
`activate_speculative_runtime()`. It must not become a second configuration
source.

Repeated activation with the same runtime object and exact proposal limit is
idempotent. Activation with a different active runtime or limit fails before
state mutation. A separate disable/reconfigure API is not introduced in this
gate.

## Request and Failure Semantics

- With no activation, scheduling and execution remain ordinary and existing
  observations retain their default speculative fields.
- Enabled selection and installed runtime always have the same proposal limit.
- Non-greedy rows remain suppressed and stale stochastic selected rows remain
  rejected before ModelRunner execution.
- Stateful non-KV rows remain fail closed.
- Activation failure does not poison the runtime because no speculative request
  has executed.
- Post-commit lifecycle failure keeps committed target state authoritative and
  continues to use the existing poisoned-runtime behavior.
- KV or Scheduler commit failures continue to use the existing transaction and
  snapshot rollback path.

## TP1 Exact-Greedy Parity Gate

The first real artifact uses an existing batch-capable, stateless n-gram
adapter to minimize unrelated lifecycle and model-loading variables.

The gate runs two engines or two isolated engine executions against the same
loaded target model:

1. ordinary baseline with no speculative activation;
2. speculative execution activated through the new atomic API.

Both executions use:

- the same model checkpoint and tokenizer;
- TP1;
- identical prompts and output-token limits;
- `temperature=0`;
- identical EOS and sampling configuration;
- fresh request and cache state unless the artifact explicitly records a
  controlled shared-prefix case.

The artifact must include:

- exact baseline and speculative output token IDs;
- a hard equality assertion over the complete generated token sequence;
- prompt token IDs and output-token budget;
- model/checkpoint identity and tokenizer identity;
- adapter identity and proposal limit;
- proposal count, proposed tokens, accepted tokens, acceptance rate, and
  accepted tokens per target invocation;
- target first-token and verifier callback counts;
- zero/partial/full acceptance coverage when naturally observed or through
  separate deterministic prompts;
- command, environment, device, dtype, TP size, and raw result path;
- explicit notice that TPOT or throughput improvement is unproven unless a
  separate controlled measurement is included.

The harness must fail rather than silently falling back if the speculative path
was never selected, no proposal was generated, or no verifier callback ran.

## Test Strategy

### Dependency-Light Unit Tests

Add focused tests covering:

1. successful activation derives and installs the exact proposal limit;
2. invalid adapter/runtime leaves Scheduler and engine unchanged;
3. Scheduler publication failure restores both snapshots;
4. engine publication failure restores both snapshots;
5. repeated identical activation is idempotent;
6. conflicting runtime or proposal limit fails before mutation;
7. ordinary default-off execution remains unchanged;
8. activated selected execution reaches the existing callback bridge;
9. generic activation modules contain no model or source-name branches.

### Existing Regression Matrix

Re-run the focused Scheduler/engine/runtime matrix, the first-target/fixed-Q
matrix, sequence serialization tests, chunked prefill, hybrid state Scheduler,
Scheduler prefill hooks, Python 3.9/3.12 compilation, source scans, and
`git diff --check`.

### Real Loaded-Model Gate

Create a dedicated runner and result schema for the TP1 baseline/speculative
comparison. Local CPU/unit tests do not satisfy this gate. If the local machine
cannot load the target checkpoint, run the gate on
`sitian@10.232.195.203` and preserve the command and raw artifact in the
repository's existing artifact convention.

## Promotion Boundary

Passing this design's tests proves only:

- activation cannot leave mismatched Scheduler/runtime state;
- the existing generic engine path is reachable through one serving boundary;
- one real loaded target model on TP1 has exact greedy token parity with an
  n-gram proposal source.

It does not prove:

- lower TPOT, higher throughput, or lower memory;
- TP4 correctness;
- two-model structural coverage;
- 4K/16K/32K+ context coverage;
- real KV H2D reduction;
- speculative/offload residency correctness;
- learned-drafter or MTP support.

The repository therefore remains `NOT_PROMOTABLE` after this gate unless the
full promotion matrix is independently completed.

## Follow-Up Order

1. atomic activation and TP1 parity from this design;
2. speculative KV transaction to `KVOffloadMVP0` residency integration with
   real H2D/D2H counters;
3. learned-drafter adapter and loaded-model gate;
4. capability-based MTP adapter;
5. TP4, second model structure, long-context, batch, and performance campaign.
