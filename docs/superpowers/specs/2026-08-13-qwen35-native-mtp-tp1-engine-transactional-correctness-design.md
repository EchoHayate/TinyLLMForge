# Qwen3.5 Native MTP TP1 Engine Transactional Correctness Design

**Date:** 2026-08-13

**Status:** Approved continuation design

## Objective

Establish the first real-checkpoint, end-to-end `LLMEngine.step()` authority
for a learned proposal source using the existing generic speculative runtime.

The authority must prove that the already validated Qwen3.5 native MTP
checkpoint executor is not only correct in isolation, but is also used by the
production Engine pipeline:

```text
target prefill observation
-> native MTP proposal
-> generic first-target callback
-> batch-native target verifier
-> target KV prepare/commit/rollback
-> Qwen3.5 side-state prepare/select/apply/seal
-> MTP proposal-KV finalize commit/rollback
-> Scheduler publication
-> sequence release and complete cleanup
```

This is the next ordered gate after:

- generic Qwen3.5 TP1/TP4 transactional correctness;
- Qwen3.5 4K/16K/32K correctness;
- controlled Qwen3.5 TP4/16K n-gram performance; and
- isolated real-checkpoint native-MTP loader, eager/reference, exact-Q CUDA
  Graph, physical proposal-KV, and loaded-GPU ownership authorities.

## Claim Scope

The first authority is deliberately narrow:

- model: the existing real Qwen3.5-2B checkpoint;
- tensor parallel size: 1;
- context: 4,096 prompt tokens per request;
- output: 32 greedy tokens per request;
- batch sizes: 1 and 4;
- policies: baseline and native MTP;
- native MTP layer count: 1;
- MTP maximum proposal length: 4;
- target and MTP execution: eager;
- KV offload: disabled;
- exact baseline/native-MTP output parity;
- real target KV and real MTP proposal-KV transactions;
- real recurrent side-state transactions;
- no accepted-prefix target replay.

Passing establishes:

```text
Qwen3.5 native MTP TP1/4K Engine transactional correctness:
  ESTABLISHED
```

It does not establish:

- TP4 native MTP;
- native MTP with target KV offload;
- 16K/32K native-MTP correctness;
- native-MTP performance improvement;
- a separate learned draft model;
- a second learned-MTP architecture;
- KV8/KV4;
- production readiness; or
- Phase 1 promotion.

The overall roadmap remains `NOT_PROMOTABLE`.

## Verified Starting Point

The real-checkpoint native-MTP authority already proves:

- exact loading of all 15 `mtp.*` tensors;
- shared target embedding and LM-head object identity;
- Q values 1 through 4 and batch sizes 1 and 4;
- independent eager/reference greedy parity;
- six exact-Q CUDA Graph families for Q2/Q3/Q4 across batch 1/4;
- physical accepted MTP K/V slot identity preservation;
- rejected MTP K/V suffix release;
- rollback-safe continuation equality;
- replay failure quarantine with no post-replay eager retry; and
- tensor-free public ownership boundaries.

Its current authoritative artifact is:

```text
artifacts/qwen35-mtp-runs/
qwen35-mtp-graph-gate-opaque-7/
qwen35_mtp_real_checkpoint_gate.json
```

That artifact has `status=PASS` and no backend failures, but explicitly does
not cover end-to-end `LLMEngine.step()` activation.

The generic Engine already supports a
`ModelRunnerProposalExecutorDescriptor`. The Qwen3.5 ModelRunner registers the
native executor under:

```text
executor_id = native_checkpoint_proposal
source_type = native_model_runner
```

The missing authority is an actual request campaign that activates this
descriptor and observes all publication participants together.

## Missing Lifecycle Boundary

The native MTP executor retains per-sequence state:

- pending prefill observations;
- a bootstrapped sequence record;
- committed MTP proposal-KV slots;
- active proposal transactions; and
- prepared batch-finalization tickets.

Isolated gates explicitly call:

```python
executor.release_sequence(
    sequence_id,
    sequence_epoch=sequence_epoch,
)
```

The current Engine completion path releases an optional host-side
`runtime.lifecycle`, but it does not release a ModelRunner-local proposal
executor sequence. A successful learned-MTP request can therefore finish with
committed proposal-KV slots and executor sequence state still owned by the
finished request.

This is a generic ModelRunner-executor lifecycle gap, not a Qwen3.5-specific
generation algorithm gap.

## Considered Approaches

### 1. Gate the isolated executor again

Run more direct `Qwen35MTPProposalExecutor` probes with longer synthetic
prefixes.

Rejected because existing authorities already cover the isolated executor.
It would not prove production Engine activation, Scheduler publication, target
KV transactions, recurrent side state, or request cleanup.

### 2. Automatically enable native MTP whenever the config flag is set

Have `LLMEngine.__init__()` immediately install the ModelRunner descriptor.

Deferred because this makes model construction implicitly change generation
behavior and removes the current explicit activation boundary. Automatic
policy selection is a separate product/configuration decision and is not
required for correctness.

### 3. Explicit Engine activation plus generic sequence release

Construct the Engine with native MTP registration enabled, retrieve the
registered descriptor, explicitly activate:

```python
EngineSpeculativeRuntime(
    model_runner_executor=descriptor,
)
```

and add one source-neutral release operation for finished
ModelRunner-executor sequences.

Selected because it exercises the real generic runtime with the smallest
production change and keeps policy activation explicit.

### 4. Jump directly to TP4 plus KV offload

Rejected for this gate. The first native-MTP implementation is TP1-only and
the generic ModelRunner-executor runtime currently fails closed when target KV
offload is enabled. Combining TP distribution, offload, and first Engine
activation would make failures difficult to localize.

## Generic Sequence Release Contract

The source-neutral proposal executor protocol gains:

```python
def release_sequence(
    self,
    sequence_id: int,
    *,
    sequence_epoch: int,
) -> None:
    ...
```

The registry gains:

```python
def release_sequence(
    self,
    executor_id: str,
    sequence_id: int,
    sequence_epoch: int,
    capabilities: DraftCapabilities,
) -> None:
    ...
```

The ModelRunner command boundary gains:

```python
def release_speculative_proposal_sequence(
    self,
    executor_id: str,
    sequence_id: int,
    sequence_epoch: int,
) -> None:
    ...
```

The tensor-free Engine bridge validates the descriptor, sequence ID, epoch,
and `None` acknowledgement before returning.

After prepared publication succeeds, the Engine releases each finished
sequence from the ModelRunner executor. The epoch is the same canonical
`Sequence.sequence_epoch` value used by target-prefill observations; absent
explicit epochs remain zero for the current Engine.

Release ordering is:

```text
proposal finalization commit
-> side-state seal
-> finished-sequence ModelRunner proposal release
-> ordinary Engine result return
```

Release failure poisons the speculative runtime and propagates. It must never
be converted into a successful request with leaked proposal state.

No release occurs before proposal finalization. No release operation may
abort an active proposal transaction or silently discard a prepared ticket.

## End-to-End Gate

### Cells

The campaign contains four independent Engine cells:

```text
baseline:b1
native_mtp:b1
baseline:b4
native_mtp:b4
```

Each cell constructs and destroys its own Engine. This prevents baseline/MTP
state contamination and makes cleanup receipts authoritative per cell.

### Engine configuration

```text
tensor_parallel_size = 1
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

The first Engine authority uses eager MTP deliberately. Exact-Q graph
correctness is already covered independently; adding graph capture to this
first full-Engine gate would mix two failure domains.

### Prompt and generation contract

- Every request has exactly 4,096 prompt tokens.
- Prompt rows are deterministic, distinct by request index, and stored in the
  artifact with SHA-256 digests.
- Sampling is greedy with `temperature=0.0`.
- `ignore_eos=true`.
- Every request emits exactly 32 output tokens.
- Baseline and native-MTP output token rows must match exactly.

### Required runtime evidence

Native-MTP cells must record:

- the registered executor ID and capabilities;
- real MTP module and physical-store presence;
- shared embedding and LM-head identity;
- target-prefill observation calls;
- proposal row and token counts;
- accepted and rejected proposal tokens;
- first-target and fixed-Q verifier callbacks;
- proposal-finalization prepare and commit receipts;
- no proposal-finalization rollback on successful cells;
- Qwen3.5 side-state `prepare -> select -> apply -> seal` receipts;
- zero accepted-prefix replay;
- no runtime poison;
- zero active proposal transactions;
- zero prepared batch-finalization tickets;
- zero pending or bootstrapped executor sequences;
- zero allocated MTP physical slots after finished-sequence release; and
- complete Engine cleanup.

The gate requires both accepted and rejected learned proposal tokens across
the native-MTP campaign. A run with proposals but no accepted tokens or no
rejected tokens is insufficient evidence and remains RED.

Baseline cells must record zero speculative activity and no native-MTP
executor registration.

## Authority and Verification

The gate uses a new schema:

```text
qwen35.native-mtp-tp1-4k-engine-transactional-correctness.v1
```

Classification:

```text
QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED
```

Claim scope:

```text
qwen35_native_mtp_tp1_4k_engine_only
```

Authority publication must bind:

- exact source-file inventory and tree SHA-256;
- real checkpoint manifest SHA-256;
- model type and architecture;
- all four validated cells;
- exact parity;
- runtime and cleanup receipts;
- successful campaign exit;
- limitations; and
- result JSON SHA-256.

The verifier independently recomputes:

- schema validation;
- source-tree digest;
- checkpoint/model digest binding;
- cell keys and policy/batch identity;
- baseline/native output parity;
- aggregate accepted/rejected learned tokens;
- lifecycle ordering;
- zero-leak cleanup state; and
- authority result digest.

The remote wrapper must:

- use `sitian@10.232.195.203`;
- export `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- use `ControlMaster=no` and `ControlPath=none`;
- use finite serial retries;
- stage into a unique isolated run root;
- never edit the canonical remote checkout;
- perform pre/post GPU admission checks;
- never kill unrelated processes;
- download the complete authority; and
- run the verifier remotely and locally.

## Failure Boundaries

The gate fails closed when:

- native MTP registration is missing or reports an error;
- descriptor identity or capabilities differ from the expected learned
  ModelRunner source;
- shared parameter identity is broken;
- output parity fails;
- proposals, accepted tokens, or rejected tokens are absent;
- target callbacks are absent;
- proposal finalization ordering is incomplete;
- side-state ordering is incomplete;
- accepted-prefix replay is observed;
- sequence release happens before finalize commit;
- any executor transaction, ticket, sequence, or physical slot remains live;
- runtime poison is set;
- rank/process-group cleanup is incomplete;
- source/model binding differs; or
- verifier recomputation differs.

No proxy or synthetic KV-copy result may satisfy this gate.

## Next Step After PASS

A PASS closes only the learned-MTP Engine-activation blocker. The next ordered
work is:

1. allow the same ModelRunner learned executor to coexist with real target KV
   offload at TP1;
2. establish native-MTP TP1/16K transactional correctness with real H2D/D2H
   movement;
3. establish TP1/32K;
4. implement TP4 native-MTP ownership and collectives;
5. run controlled native-MTP TPOT/TTFT/throughput/memory/KV-traffic evidence;
6. add an independent learned draft-model architecture.

No staging, commit, push, branch switch, stash, reset, or clean is part of
this design.
