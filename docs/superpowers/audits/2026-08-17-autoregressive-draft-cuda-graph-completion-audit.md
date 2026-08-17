# Autoregressive Draft Exact-Shape CUDA Graph Completion Audit

**Date:** 2026-08-17

**Repository:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

**Classification:** `INCONCLUSIVE_ENVIRONMENT`

**Promotion:** `NOT_PROMOTABLE`

## Executive Decision

The current tree implements the approved default-off exact-shape CUDA Graph
path for the independent Qwen3 learned drafter:

```text
tensor parallel size: 4
batch size:           4
proposal length:      4
sampling:             greedy
Proposal-KV:          dense direct allocation
Proposal-KV offload:  disabled
shape policy:         exact only; no padding or rounding
```

Local source-contract and lifecycle verification is green. The implementation
preserves exact proposal tokens, shared proposal registration, accepted-prefix
commit, rejected-suffix abort, zero live transactions after finalization, and
TP-wide failure convergence in the tested dependency-light paths.

No controlled real-GPU before/after result was produced. The remote host had
no four clean GPUs, and the previously used Python/model environment was no
longer present. The preflight stopped before source upload, model loading,
CUDA Graph capture, correctness comparison, or performance measurement. It
did not terminate, reassign, or modify any existing remote process.

Therefore:

```text
LOCAL_EXACT_GRAPH_IMPLEMENTATION=ESTABLISHED
LOCAL_GRAPH_LIFECYCLE_CONTRACT=ESTABLISHED
REAL_TP4_CUDA_GRAPH_EXECUTION=NOT_ESTABLISHED
REAL_EAGER_GRAPH_CORRECTNESS_PARITY=NOT_ESTABLISHED
CONTROLLED_CUDA_GRAPH_PERFORMANCE=NOT_ESTABLISHED
FINAL_CLASSIFICATION=INCONCLUSIVE_ENVIRONMENT
```

This is not a `GO`, `NO_GO_PERFORMANCE`, or `NO_GO_CORRECTNESS` result.

## Implemented Runtime Boundary

### Admission and budgets

`tinyvllm/config.py` adds nine default-off configuration fields. Enabled
admission is restricted to TP4, B4, Q4, greedy direct Proposal-KV with
offload disabled. The defaults are:

```text
autoregressive_draft_cuda_graphs=false
Q allowlist=(4,)
batch allowlist=(4,)
minimum successful eager observations=2
maximum graph entries=4
maximum static bytes=67108864
maximum reserved bytes=536870912
maximum total capture time ns=5000000000
maximum single capture time ns=2000000000
```

Unsupported topology, batch, proposal length, sampling, and offload
combinations fail closed or remain on the existing eager path. Exact
identities are not rounded or padded into an admitted family.

### Capture and replay

`tinyvllm/engine/autoregressive_draft_graph.py` owns exact identities,
successful-eager observation counts, capture budgets, replay admission, and
permanent quarantine after replay-started failures.

The policy is:

1. observe two successful eager executions for the exact identity;
2. capture using private scratch Proposal-KV transactions;
3. prepare live replay metadata outside the graph;
4. converge all TP ranks before replay;
5. replay the three-step proposal graph;
6. perform one final host token readback;
7. converge all TP ranks after replay; and
8. register proposals through the same executor lifecycle used by eager mode.

A pre-replay failure cleans up and converges all ranks onto one eager
fallback. Once replay has started, a failure aborts live transactions,
quarantines the identity, and does not retry eagerly.

### Qwen3 backend and Proposal-KV ownership

`tinyvllm/engine/qwen3_draft_cuda_graph_backend.py` captures three fixed B4
decode steps, root-rank argmax, and TP broadcast while chaining selected token
tensors on GPU. It performs one final `.tolist()` after replay.

`tinyvllm/engine/qwen3_draft_graph_scratch.py` allocates private scratch
transactions for capture. Scratch and live transaction namespaces do not
share executor transaction maps. Live committed Proposal-KV is exposed only
through read leases, and scratch rollback is performed in reverse order.

`tinyvllm/engine/autoregressive_draft_executor.py` remains authoritative for
proposal registration and finalization. Eager and graph modes share the same
logical authority representation; `execution_mode` remains telemetry rather
than part of the logical proposal digest.

## Source-Bound Gate Contract

The gate implementation is split into:

```text
tools/autoregressive_draft_cuda_graph_gate.py
tools/autoregressive_draft_cuda_graph_contract.py
tools/verify_autoregressive_draft_cuda_graph_gate.py
tools/run_autoregressive_draft_cuda_graph_gate_remote.py
```

The intended campaign uses two warmup pairs and eight measured
position-balanced eager/graph pairs for TP4/B4/Q4, prompt length 256, and
output length 16. The verifier requires:

- exact target-token equality;
- exact proposal-token equality;
- exact accepted-prefix equality;
- exact transaction-authority equality;
- zero active transactions after every run;
- graph replay on every rank;
- no measured fallback or quarantine;
- source patch, source tree, and focused-file hash binding;
- recomputation of all raw-row aggregates;
- positive median throughput improvement;
- no median TPOT regression; and
- paired bootstrap throughput-confidence lower bound greater than zero.

The result schema enforces exact target shape `B4 x output16` and exact
proposal shape `calls x B4 x Q4`.

## Prompt-to-Artifact Checklist

| Requirement | Implementation / evidence | Verdict |
| --- | --- | --- |
| Default-off graph mode | `tinyvllm/config.py`; config tests | `ACHIEVED_LOCALLY` |
| Only TP4/B4/Q4/greedy/dense-direct/no-offload | config validation, `model_runner.py`, registration tests | `ACHIEVED_LOCALLY` |
| Exact identities without padding or rounding | `autoregressive_draft_graph.py`; identity tests | `ACHIEVED_LOCALLY` |
| Capture only after successful eager observations | graph state-machine tests | `ACHIEVED_LOCALLY` |
| Failed eager calls do not advance capture admission | graph state-machine tests | `ACHIEVED_LOCALLY` |
| Three draft forwards plus argmax and TP broadcast captured | `qwen3_draft_cuda_graph_backend.py`; fake-torch backend tests | `ACHIEVED_LOCALLY` |
| Selected tokens remain on GPU until one final readback | backend implementation and one-`.tolist()` tests | `ACHIEVED_LOCALLY` |
| Private capture scratch does not enter live executor maps | `qwen3_draft_graph_scratch.py`; ownership tests | `ACHIEVED_LOCALLY` |
| Eager and graph proposals share registration/finalization | executor refactor and dispatch/finalize tests | `ACHIEVED_LOCALLY` |
| Accepted prefix committed; rejected suffix aborted | executor, Proposal-KV, and transaction tests | `ACHIEVED_LOCALLY` |
| Pre-replay failure converges to one eager fallback | graph/executor TP failure-injection tests | `ACHIEVED_LOCALLY` |
| Replay-started failure aborts, quarantines, and does not retry | graph/executor failure-injection tests | `ACHIEVED_LOCALLY` |
| Tensor-free authority snapshot | executor graph summary tests | `ACHIEVED_LOCALLY` |
| Source-bound raw-row verifier rejects tampering | contract, verifier, and tamper tests | `ACHIEVED_LOCALLY` |
| Exact B4/output16 and calls/B4/Q4 gate shapes | contract validation tests | `ACHIEVED_LOCALLY` |
| Real TP4 CUDA Graph capture/replay | remote execution artifact | `NOT_ACHIEVED_ENVIRONMENT` |
| Real eager/graph token and transaction parity | remote correctness campaign | `NOT_ACHIEVED_ENVIRONMENT` |
| Controlled before/after throughput, TPOT, memory, acceptance | remote paired campaign | `NOT_ACHIEVED_ENVIRONMENT` |
| Archived verifier, local verifier, and manifest checksum pass | completed source-bound result bundle | `NOT_ACHIEVED_ENVIRONMENT` |

## Verification

The final pre-commit verification used an isolated uv-managed Python 3.11
environment with PyTorch 2.7.1, pytest 8.4.2, NumPy, and Transformers 4.57.6.
The system Python 3.9 did not contain PyTorch, so its collection error was
classified as an interpreter-environment failure and was not treated as a
test result.

Fresh final results after the audit and handoff updates:

```text
21-file exact graph/runtime/Proposal-KV suite:
  491 passed in 12.28s

focused Python 3.11 compileall:
  PASS

focused git diff --check:
  PASS
```

The 491-test set includes exact graph config/policy/gate/backend/scratch,
executor, registration, ModelRunner integration, TP1/TP4 local contracts,
Proposal-KV allocator/cache/lifecycle, Qwen3 backend/storage, performance
gate, snapshot transport, and Qwen3.5 config compatibility coverage. These
results establish dependency-light source behavior and regression
compatibility. They do not execute a real Qwen3 checkpoint, CUDA Graph,
NCCL/TP4 model path, or controlled end-to-end performance campaign.

## Remote Preflight

The latest read-only preflight recorded:

```text
host:                 n232-195-203
GPU indices checked:  0..7
clean GPUs available: 0
required clean GPUs:  4
observed memory use:  approximately 31580-62184 MiB per GPU
classification:       INCONCLUSIVE_ENVIRONMENT
reason:               fewer than four clean GPUs are available
```

The previously used remote Python path and target/draft model paths were also
absent. The runner now performs a shell/`nvidia-smi` preflight before source
upload or model fingerprinting so an unavailable environment cannot create a
misleading partial performance bundle.

The local unversioned preflight receipt is:

```text
/tmp/autoregressive-draft-cuda-graph-preflight-20260817.json
```

It is operational evidence only and is not a source-bound correctness or
performance artifact.

## What This Proves

- The current source has a narrowly admitted, default-off graph architecture.
- Exact-family policy, capture scratch ownership, shared proposal lifecycle,
  and TP failure convergence are covered by focused local tests.
- The gate can reject malformed, tampered, non-exact, or source-drifted
  evidence.
- Existing eager semantics remain the authority outside the admitted family.

## What This Does Not Prove

- that PyTorch can capture this exact loaded Qwen3 TP4 path on the remote
  software stack;
- that eager and graph execution produce identical tokens on real GPUs;
- that replay occurs successfully on all four ranks;
- that graph mode improves throughput or TPOT;
- that memory, acceptance, TTFT, or transaction behavior improves;
- that the path generalizes beyond TP4/B4/Q4 short-context greedy direct
  Proposal-KV; or
- that this work changes the Phase 1 promotion decision.

## Next Action

When four clean GPUs and a valid Python/model environment are available:

1. run the source-bound remote gate without changing other processes;
2. require both archived and current verifiers plus manifest checksum to pass;
3. classify the result as `GO`, `NO_GO_CORRECTNESS`, or
   `NO_GO_PERFORMANCE`; and
4. only if `GO`, consider broader exact families or default-policy changes.

Until then:

```text
AUTOREGRESSIVE_DRAFT_EXACT_CUDA_GRAPH=INCONCLUSIVE_ENVIRONMENT
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
