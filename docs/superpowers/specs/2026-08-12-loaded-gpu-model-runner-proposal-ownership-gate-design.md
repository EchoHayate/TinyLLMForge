# Loaded-GPU ModelRunner Proposal Ownership Gate Design

**Date:** 2026-08-12

**Status:** Selected continuation design

## Objective

Add an independent, fail-closed real-checkpoint gate proving that the
source-neutral fused proposal path keeps target hidden states and logits
inside `ModelRunner` while returning only tensor-free speculative results to
the Engine-side runtime.

The first authority uses the already loaded Qwen3.5 native-MTP executor, but
the property under test is the generic
`ModelRunner.run_spec_first_target_and_proposal_batch(...)` ownership
boundary rather than another Qwen3.5 projection primitive.

## Approved First-Version Domain

The gate is deliberately limited to:

- tensor parallel size 1;
- KV offload disabled;
- greedy decoding;
- one Qwen3.5 MTP layer;
- shared target/MTP embedding and LM head;
- exact proposal lengths `Q=(1,2,3,4)`;
- batch sizes `(1,4)`;
- Q1 eager passthrough;
- Q2/Q3/Q4 exact CUDA Graph families;
- one real target checkpoint and its real native-MTP checkpoint tensors;
- GPU 7 on `sitian@10.232.195.203`;
- serial SSH using `ControlMaster=no` and `ControlPath=none`.

The gate does not claim TP4, KV offload, arbitrary Q or batch sizes,
multiple MTP layers, non-greedy sampling, long-context behavior, a second
model structure, end-to-end performance improvement, or promotion
readiness.

## Existing Evidence and Remaining Gap

The current runtime already establishes:

- a source-neutral ModelRunner-local proposal executor registry;
- a fused first-target/proposal method;
- `assert_tensor_free(...)` on its public result;
- real Qwen3.5 target and MTP checkpoint loading;
- real CUDA MTP eager/reference parity;
- exact-Q graph/eager proposal parity;
- physical proposal-KV commit/rollback without accepted-KV replay, copy, or
  rematerialization.

The authoritative native-MTP graph gate invokes the executor directly. It
does not execute the production fused ModelRunner method and therefore does
not prove that:

1. the target hidden rows supplied to the executor are CUDA tensors produced
   by the loaded target forward;
2. target logits remain inside ModelRunner and are not returned through the
   command boundary;
3. the returned first-target/proposal rows are tensor-free after a real
   loaded-GPU execution;
4. graph and eager fused executions preserve the same greedy target and
   proposal tokens on fresh state.

This design closes only that gap.

## Considered Approaches

### 1. Extend the Existing Native-MTP Graph Artifact

Add ownership fields to
`tools/qwen35_mtp_real_checkpoint_gate.py` and make the existing schema
responsible for both direct executor correctness and ModelRunner command
ownership.

This is rejected. The graph artifact is already authoritative and stable.
Changing its required schema would couple two independently useful gates,
make historical artifacts unverifiable, and make a ModelRunner-boundary
failure look like a graph-backend regression.

### 2. Static Source and Fake-Tensor Ownership Checks

Rely on source assertions, `assert_tensor_free(...)`, and the existing
dependency-light fake-tensor test.

This is rejected as the authority. Those checks are valuable local
regressions, but they do not show that a real loaded target produces CUDA
hidden/logit tensors or that the real MTP executor consumes them in place.

### 3. Independent Loaded-GPU Fused-Path Gate

Create a separate artifact that loads the production ModelRunner, invokes
the production fused command path, observes tensor ownership inside that
call, validates its returned value, and compares fresh graph/eager fused
executions.

This is selected. It preserves the existing graph artifact, exercises the
missing boundary directly, and keeps ownership evidence distinct from
performance and promotion claims.

## Production Path Under Test

The authoritative call is:

```text
ModelRunner.call(
  "run_spec_first_target_and_proposal_batch",
  sequences,
  qwen35_mtp_executor_descriptor,
  (),
)
  -> prepare_decode(...)
  -> real target run_model(..., return_hidden=True)
  -> target logits argmax inside ModelRunner
  -> ModelRunnerProposalInput with local target_hidden
  -> ModelRunnerProposalExecutorRegistry.execute_batch(...)
  -> real Qwen35MTPProposalExecutor.propose_batch(...)
  -> FirstTargetProposalResult rows
  -> assert_tensor_free(...)
  -> tensor-free return
```

`ModelRunner.call(...)` is used rather than invoking the method attribute
directly. For TP1 the command executes in the rank-zero process, so this is
not evidence for an operating-system cross-process transport. The result
must additionally survive `pickle.dumps(...)`/`pickle.loads(...)` and remain
tensor-free, proving that it is valid for the existing command transport
contract without claiming TP4 transport execution.

## Real Sequence and State Ownership

Every scenario uses fresh sequence IDs and fresh target-side state:

1. construct deterministic token histories inside the tokenizer vocabulary;
2. reserve distinct target KV blocks from a gate-local bounded block range;
3. acquire distinct Qwen3.5 hybrid-state leases for every active sequence;
4. bind each lease to the sequence's
   `hybrid_state_slot_id`/`hybrid_state_generation`;
5. run exactly one real target decode forward through the fused method;
6. finalize or roll back any MTP proposal transaction produced by the
   scenario;
7. release MTP sequence state, target hybrid-state leases, and gate-local
   target KV ownership in `finally`;
8. reset the global attention context and restore all temporary observers.

Graph and eager comparisons must never reuse target KV, hybrid state,
proposal KV, sequence IDs, or proposal transaction IDs.

If the production runner does not expose a safe bounded state-acquisition
path, the gate must fail closed. It must not patch `run_model`, substitute a
synthetic target forward, or write arbitrary state slots merely to obtain a
PASS.

## Gate-Local Ownership Observation

The gate installs two temporary, identity-preserving wrappers and restores
them in `finally`.

### Target Forward Observer

The wrapper delegates to the original `runner.run_model(...)` and records
only scalar metadata:

- whether logits are a `torch.Tensor`;
- logits device type and device index;
- logits dtype and shape;
- whether returned hidden states are a `torch.Tensor`;
- hidden device type and device index;
- hidden dtype and shape;
- whether hidden and logits share the active CUDA device;
- whether `return_hidden=True` and `execution_mode="decode"` were requested.

It must not detach, clone, move, serialize, hash, or retain either tensor.

### Proposal Executor Observer

The wrapper delegates to the already registered real MTP executor and
records, for every `ModelRunnerProposalInput`:

- target hidden is a `torch.Tensor`;
- target hidden is CUDA-resident;
- target hidden device matches the target forward device;
- target hidden dtype and final dimension match the target forward row;
- `target_logits is None` for the native-MTP capability;
- input sequence order exactly matches the fused batch;
- the executor returns only validated `DraftProposal` rows.

The observer must not retain tensor references after the delegated call
returns. Its durable observation is scalar/list metadata only.

The gate must verify that the real executor object identity before, during,
and after observation is unchanged.

## Public Boundary Contract

For every scenario the returned value must satisfy all of the following:

- it is a tuple of `FirstTargetProposalResult`;
- sequence IDs exactly match the input batch and preserve input order;
- target tokens are plain Python integers;
- proposal token IDs are tuples of plain Python integers;
- metadata contains no tensor, storage, CUDA event, stream, graph, module, or
  callable;
- no result row has `target_hidden` or `target_logits` attributes;
- `assert_tensor_free(...)` accepts the complete nested value;
- a pickle round-trip preserves the canonical tensor-free payload;
- recursively scanning both the original and round-tripped value finds zero
  torch tensors.

It is not sufficient for pickling to succeed if a tensor is copied to CPU
and included in the payload. The recursive tensor count must be exactly
zero.

## Graph/Eager Fused Correctness

For every `(Q,batch)` pair in the approved domain:

1. run a fresh fused scenario with the production graph configuration;
2. run another fresh fused scenario with the executor graph runner
   temporarily disabled;
3. compare the complete ordered first-target token tuple;
4. compare the complete ordered proposal-token tuples;
5. require both comparisons to be exactly equal;
6. require every public result from both sides to satisfy the ownership
   contract.

Q1 must report eager passthrough and must not increase the graph capture
count. Q2/Q3/Q4 crossed with batch 1/4 must map to the six exact graph
families already authorized by the graph gate.

This comparison establishes exact greedy token parity for the fused
ModelRunner boundary. It does not establish bitwise hidden/logit equality.
The existing eager/reference and graph/eager artifacts remain the numerical
authority for full-logit and MTP graph behavior.

## Artifact Contract

Create a separate schema-v1 JSON artifact with at least:

```text
schema_version
checkpoint_path
checkpoint_manifest_sha256
device_name
torch_version
cuda_version
q_values
batch_sizes
loader_passed
fused_model_runner_path_exercised
target_forward_real
target_logits_cuda
target_hidden_cuda
target_hidden_consumed_by_real_executor
target_logits_not_passed_to_mtp_executor
public_result_tensor_count
public_result_pickle_roundtrip
public_result_tensor_free
executor_identity_preserved
sequence_order_preserved
graph_eager_first_target_tokens_equal
graph_eager_proposal_tokens_equal
graph_capture_count
graph_replay_count
cleanup_passed
backend_failures
status
promotion_classification
limitations
```

Required successful values include:

```text
fused_model_runner_path_exercised=true
target_forward_real=true
target_logits_cuda=true
target_hidden_cuda=true
target_hidden_consumed_by_real_executor=true
target_logits_not_passed_to_mtp_executor=true
public_result_tensor_count=0
public_result_pickle_roundtrip=true
public_result_tensor_free=true
executor_identity_preserved=true
sequence_order_preserved=true
graph_eager_first_target_tokens_equal=true
graph_eager_proposal_tokens_equal=true
cleanup_passed=true
backend_failures=[]
status=PASS
promotion_classification=NOT_PROMOTABLE
```

The verifier must corrupt every critical field independently and require
`FAIL / NOT_PROMOTABLE`. A missing observation is a failure; it must never
default to a successful value.

## Failure and Cleanup Semantics

The gate fails closed for:

- unavailable real target or MTP executor;
- synthetic or replaced target forward;
- non-CUDA hidden/logit tensors;
- device mismatch;
- target logits entering the MTP proposal input;
- any tensor in the public result;
- executor identity drift;
- sequence-order drift;
- graph/eager target or proposal token mismatch;
- leaked proposal transactions;
- unreleased hybrid-state leases;
- uncleared target or proposal KV ownership;
- observer restoration failure;
- checkpoint manifest drift.

Cleanup executes in reverse ownership order. A failure after graph replay
must preserve the existing no-eager-retry/quarantine contract; this gate
must not add a second fallback path.

## Local TDD and Remote Authority

Local dependency-light tests cover:

- schema validation and field corruption;
- source/wrapper contracts;
- observer restoration;
- recursive tensor-free scanning;
- pickle canonicalization;
- sequence-order checks;
- graph/eager result comparison;
- fail-closed cleanup reporting;
- remote wrapper source synchronization.

Torch/CUDA execution is authoritative only through the serial GPU7 run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh -o ControlMaster=no -o ControlPath=none \
  sitian@10.232.195.203 \
  'CUDA_VISIBLE_DEVICES=7 ...'
```

The wrapper must use a fresh opaque run ID, a fresh high distributed port,
and no parallel SSH sessions. The run ID is not date evidence.

## Promotion Boundary

A PASS closes only the loaded-GPU ModelRunner hidden/logit ownership gate for
the approved TP1 Qwen3.5 native-MTP domain.

The first-phase objective remains `NOT_PROMOTABLE` until the broader matrix
includes at least:

- a second materially different model structure;
- TP4;
- 4K/16K/32K or longer contexts;
- batch 1/4/multiple sequences;
- exact greedy parity;
- TPOT, TTFT, throughput, peak memory, real KV H2D/D2H bytes, and
  acceptance;
- evidence that does not treat simulated KV copies as offload gains.
