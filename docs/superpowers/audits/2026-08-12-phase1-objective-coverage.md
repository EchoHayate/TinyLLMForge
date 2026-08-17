# Phase 1 Generic Speculative Runtime Objective Coverage Audit

Date: 2026-08-12

Latest evidence refresh: 2026-08-14

Classification: `NOT_PROMOTABLE`

## Objective Restatement

The first-phase deliverable is a source-agnostic speculative runtime and
transactional KV foundation that:

1. improves KV-cache utilization through logical/physical decoupling,
   GPU/CPU movement, prefix reuse, quantization, and direct speculative KV
   commit/rollback;
2. supports long contexts through chunked prefill and exact blockwise
   attention without requiring all visible KV to reside on GPU;
3. reduces TPOT through batch-native multi-token verification shared by MTP,
   independent draft models, and model-free drafters;
4. is eligible for promotion only after two model structures, TP1 and TP4,
   4K/16K/32K, batch 1/4/multi-sequence, exact greedy parity, performance,
   memory, real KV movement, and acceptance evidence are all present.

Passing unit tests or one TP1 Qwen3 artifact is not sufficient to satisfy this
objective.

## Status Definitions

- `ESTABLISHED`: implemented and covered by direct tests or authoritative
  loaded-model artifacts matching the stated scope.
- `PARTIAL`: implemented for only part of the requested scope, model-specific,
  incompatible with another required feature, or lacking authoritative
  evidence.
- `MISSING`: no implementation or no evidence covering the requirement.
- `BLOCKED`: implementation exists, but the required authoritative run cannot
  currently execute because of an external dependency.

## Prompt-to-Artifact Checklist

### 1. KV Cache Utilization

| Requirement | Status | Concrete evidence | Missing boundary |
|---|---|---|---|
| Logical KV pages decoupled from physical GPU slots | `ESTABLISHED` | `tinyvllm/engine/model_runner.py` implements `KVOffloadMVP0` with `logical_to_slot` and `slot_to_logical`; `tools/test_kv_offload.py` covers assignment, eviction, generation binding, and contiguous staging | Current loaded-model authority is TP1 only |
| GPU/CPU hierarchy | `ESTABLISHED` | `KVOffloadMVP0` owns GPU staging slots and CPU backing storage; `artifacts/speculative_residency_boundary/20260812T065636Z/result.json` is a loaded-model TP1 PASS | No TP4 authority |
| Asynchronous prefetch | `ESTABLISHED` | KV-offload events and wait paths in `tinyvllm/engine/model_runner.py`; future-window hints and staging in `tinyvllm/layers/attention.py`; direct tests in `tools/test_kv_offload.py` | No end-to-end overlap measurement |
| Batched/coalesced H2D and D2H | `ESTABLISHED` | `_enqueue_h2d_pairs`, `_enqueue_d2h_pairs`, copy coalescing, and batch counters in `tinyvllm/engine/model_runner.py`; batching tests in `tools/test_kv_offload.py`; real movement counters in the residency artifact and `artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/result.json` | No TP4 authority |
| Dirty writeback | `ESTABLISHED` | `mark_dirty`, `writeback_dirty`, dirty eviction, deferred-event handling, and real D2H counters; tests in `tools/test_kv_offload.py`; TP1 16K/32K blockwise campaign records real D2H movement | No TP4 authority or performance-direction evidence |
| Prefix KV reuse across requests | `ESTABLISHED` for ordinary KV | Hash-chained reusable blocks, `hash_to_block_ids`, prefix reservations, attach/release, and block refcounts in `tinyvllm/engine/block_manager.py`; profiling and validation in `tools/profile_prefix_cache.py` | No inspected promotion-grade artifact in the current worktree |
| Prefix deduplication and reference counting | `ESTABLISHED` for ordinary KV; `PARTIAL` for hybrid state | Generic KV block reuse is reference counted in `tinyvllm/engine/block_manager.py`; exact tensor interning/refcounts exist in `tinyvllm/engine/qwen35_hybrid_prefix_cache.py` and its tests | Hybrid-state deduplication is Qwen3.5-specific, not a generic runtime facility |
| KV8/KV4 quantization | `PARTIAL` | `kv_quant_bits` supports 0/4/8 in `tinyvllm/config.py`; quantized cache storage and dequantization exist in `tinyvllm/engine/model_runner.py` and `tinyvllm/layers/attention.py` | `tinyvllm/config.py` explicitly rejects KV offload with KV4/KV8; blockwise prefill/decode also require unquantized KV |
| Per-layer/per-token heat grading | `MISSING` | No generic heat-tier policy or promotion artifact found | Requires a separate design and gate |
| Speculative accepted KV commits in place | `ESTABLISHED` | `begin_speculative_kv_transaction`, `prepare_speculative_kv_commit`, and `commit_speculative_kv_commit_batch` in `tinyvllm/engine/block_manager.py`; token-free batch commit tests in `tools/test_speculative_kv_transaction.py` | Promotion matrix remains incomplete |
| Rejected suffix rollback without replay/copy/rematerialization | `ESTABLISHED` | Transaction rollback and reserved-block release in `tinyvllm/engine/block_manager.py`; residency precommit/seal/rollback in `tinyvllm/engine/speculative_residency.py`; rejected speculative D2H is gated to zero | No TP4/second-model evidence |

### 2. Longer Context

| Requirement | Status | Concrete evidence | Missing boundary |
|---|---|---|---|
| Chunked prefill | `ESTABLISHED` | Chunked scheduling and postprocess paths in `tinyvllm/engine/scheduler.py`; chunked prefill tooling/tests under `tools/test_chunked_prefill.py` and `tools/profile_chunked_prefill.py`; Qwen3 and Qwen3.5 loaded-model authorities now cover 16K/32K speculative execution | No controlled long-context performance direction |
| Blockwise prefill without full GPU residency | `ESTABLISHED` | `_blockwise_online_prefill_attention` and logical-window staging in `tinyvllm/layers/attention.py`; bounded Qwen3.5 pure-prefill online softmax in `tinyvllm/layers/qwen35_full_attention.py`; successful Qwen3.5 TP4/32K authority under `artifacts/qwen35_generic_speculative_tp4_32k/opaque-03a0a96654a14441b314800f/artifacts/authority` | No controlled performance claim |
| Blockwise online-softmax decode | `ESTABLISHED` in implementation and focused tests | `_blockwise_online_decode_attention` in `tinyvllm/layers/attention.py`; `tools/test_blockwise_attention_planning.py`; loaded 16K/32K authorities exercise the long-context decode path | No controlled performance direction |
| Blockwise multi-query speculative verification | `ESTABLISHED` | `_blockwise_online_spec_verify_attention`; dense causal oracle coverage for batch 1/4, Q=2/4, GQA, and window order in `tools/test_blockwise_attention_planning.py`; loaded Qwen3 and Qwen3.5 16K/32K authorities prove exact greedy parity and real movement | No controlled performance direction |
| Future-window prefetch and eviction | `ESTABLISHED` in implementation | Forward/reverse future-hint builders and `_stage_blockwise_read_window` in `tinyvllm/layers/attention.py`; real movement diagnostics recorded in `AGENT_HANDOFF_STATE.md` | Diagnostic movement is not an authoritative parity artifact |
| Prefix cache combined with CPU-resident KV | `ESTABLISHED` for the local composition contract | `tools/test_prefix_kv_offload_integration.py` composes real `BlockManager`, generation identities, `KVOffloadMVP0`, and `ModelRunner.prepare_prefill`: an idle shared prefix retains same-generation CPU backing, reuse schedules H2D, block-ID recycling invalidates stale backing, and cached-prefix prefill requires valid backing | CPU-only scheduling evidence is not a loaded-model parity or real-copy performance artifact |
| Sliding-window attention | Deferred by objective | Not required for the first blockwise gate | No generic promotion evidence |
| Sparse attention | Deferred by objective | Not required for the first blockwise gate | No generic promotion evidence |
| Context parallel | Deferred by objective | Not required for the first blockwise gate | No generic promotion evidence |

### 3. Lower TPOT and Generic Speculative Runtime

| Requirement | Status | Concrete evidence | Missing boundary |
|---|---|---|---|
| Source-agnostic speculative runtime | `ESTABLISHED` | `tinyvllm/engine/speculative_runtime.py` and `tinyvllm/speculative/batch_runtime.py` define the source-neutral adapter/lifecycle/verification contract; `tinyvllm/engine/speculative_proposal_executor.py` adds a ModelRunner-local executor registry; `LLMEngine.step()` consumes one tensor-free first-target/proposal provider without model-name or proposal-source branches | No concrete learned/MTP executor or promotion matrix |
| Model-free n-gram drafter | `ESTABLISHED` | `tinyvllm/speculative/ngram_adapter.py`; source-adapter and engine runtime tests; Qwen3-0.6B and Qwen3.5 TP4 authorities prove batch-1/4 exact output parity, all-rank callbacks, transactional KV, and 4K/16K/32K coverage | TP4 performance and learned-drafter evidence remain open |
| Model-free SAM drafter | `ESTABLISHED` in runtime adapter | `tinyvllm/speculative/sam_adapter.py` and lifecycle tests | No current end-to-end runtime performance artifact |
| MTP head | `ESTABLISHED` for Qwen3.5 TP1/4K production Engine correctness | The native Qwen3.5 MTP module, exact 15-tensor checkpoint plan/binder, ModelRunner registration, proposal executor, transactional proposal-slot lifecycle, and real-checkpoint loader/shared-weight gate are implemented. `artifacts/qwen35_native_mtp_tp1_4k_engine/opaque-57a3a62810d43636b96295da/local-authority/result.json` proves real `LLMEngine.step()` execution at batch 1/4, exact greedy parity, first-target and verifier target-forward accounting, direct accepted proposal-KV commit, rejected-suffix release, zero accepted-prefix replay, and zero final proposal/slot leaks. Both independent verifiers report `PASS` | TP4 native MTP, 16K/32K native MTP with real KV offload, controlled native-MTP performance, and a second learned model structure remain unproved |
| Independent learned draft model | `PARTIAL` | `tinyvllm/engine/autoregressive_draft_executor.py`, `tinyvllm/engine/qwen3_draft_backend.py`, and `tinyvllm/engine/autoregressive_draft_registration.py` implement a concrete in-process Qwen3 dense drafter, real multi-layer proposal K/V ownership, checkpoint/tokenizer contracts, batch-native learned decode, and source-neutral ModelRunner registration. `tools/autoregressive_draft_tp1_engine_gate.py` provides a fail-closed real-checkpoint TP1 gate and preflight mode | No completed real Qwen3-draft/Qwen3.5-target GPU parity artifact. TP4 remains intentionally rejected pending design approval |
| Batch-native multi-token verifier | `ESTABLISHED` | Fixed-Q grouping in `tinyvllm/engine/speculative_model_runner.py`; one target forward per fixed-Q group via `run_spec_verify_batch`; batch runtime directly commits generated KV; Qwen3.5 TP4 4K/16K/32K authorities record matching `spec_first_target` and `spec_verify` callbacks plus collective coverage on ranks 0-3 | No TP4 performance direction |
| Variable proposal lengths without padding | `ESTABLISHED` locally | Distinct fixed-Q groups are formed without padding or rounding; exact `(B,Q,W)` identities are implemented by `tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py` and exercised by `tools/test_model_runner_spec_verify_cuda_graph.py` | Real CUDA evidence is blocked by the absence of an idle GPU |
| CUDA Graph for variable proposal length | `PARTIAL` and externally `BLOCKED` | `tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`, `tinyvllm/engine/model_runner.py`, `tools/spec_verify_cuda_graph_smoke.py`, `tools/verify_spec_verify_cuda_graph_gate.py`, and `tools/run_spec_verify_cuda_graph_gate_remote.py` implement exact TP1/no-offload graph families, private capture scratch, pre-replay fallback, post-replay no-retry, quarantine, correctness evidence, and controlled performance schema | `experiments/spec_verify_cuda_graph/preflight-20260812-task9-refresh-idle-gpu-blocked.json` proves the runner did not upload source or start CUDA because all eight GPUs were occupied; no correctness or performance PASS artifact exists |
| Fused verifier, sampling, and KV commit | `MISSING` | Current runtime records separate tail forward, acceptance, KV materialization, and metadata commit phases | Requires kernel/runtime fusion design |
| TP collective overlap | `MISSING` for this runtime | The TP4 authority proves speculative callbacks execute collectives on all four ranks, but it does not measure or demonstrate collective/compute overlap | Requires a separate controlled overlap/performance gate |
| AllReduce fusion | `MISSING` for this runtime | No speculative-path implementation or artifact identified | Requires separate TP optimization |
| ReduceScatter and persistent hidden-state sharding | `MISSING` for this runtime | No learned/MTP executor and no TP4 evidence | Must be designed with the learned executor |

## Promotion-Gate Matrix

| Gate dimension | Current authority | Status |
|---|---|---|
| Model structure 1 | Qwen3-0.6B | `ESTABLISHED` for TP1 4K correctness/performance and TP1 16K/32K correctness/movement |
| Model structure 2 | Qwen3.5 hybrid recurrent/full-attention model | `ESTABLISHED` for generic n-gram transactional correctness at TP1 and TP4 4K/16K/32K; no performance claim |
| TP1 | Qwen3-0.6B and Qwen3.5 artifacts | `ESTABLISHED` within recorded correctness scopes |
| TP4 | Qwen3-0.6B 4K plus Qwen3.5 4K/16K/32K authorities | `ESTABLISHED` for batch-1/4 exact correctness, all-rank callback/collective identity, residency acknowledgements, cleanup, and real movement; no performance claim |
| 4K | Qwen3 performance artifact plus Qwen3/Qwen3.5 TP4 correctness authorities | PASS within recorded scopes; `NOT_PROMOTABLE` |
| 16K | Qwen3 TP1 plus Qwen3.5 TP4 authority `opaque-3b8050a916f037bc92412ea5` | PASS correctness/movement for both model structures; no performance direction |
| 32K | Qwen3 TP1 plus Qwen3.5 TP4 authority `opaque-03a0a96654a14441b314800f` | PASS correctness/movement for both model structures; no performance direction |
| Batch 1 | Qwen3 and Qwen3.5 4K/16K/32K artifacts | `ESTABLISHED` within recorded TP1/TP4 correctness scopes |
| Batch 4 / multiple sequences | Qwen3 and Qwen3.5 4K/16K/32K artifacts | `ESTABLISHED` within recorded TP1/TP4 correctness scopes |
| Exact greedy parity | Qwen3 and Qwen3.5 TP1/TP4 authorities | `ESTABLISHED` for both model structures within recorded 4K/16K/32K scopes |
| TPOT / TTFT / throughput | 4K performance artifact | `ESTABLISHED` only for Qwen3 TP1 |
| Peak memory | Recorded by 4K artifact | No reduction established |
| Real KV H2D/D2H | Residency, 4K performance, and 16K/32K blockwise artifacts | `ESTABLISHED` only for recorded TP1 scopes |
| Acceptance | 4K performance and 16K/32K blockwise artifacts | `ESTABLISHED` for n-gram TP1 |
| No simulated KV-copy claim | Real `KVOffloadMVP0` counters are used by authoritative artifacts | `ESTABLISHED` for those artifacts |
| Variable-Q CUDA Graph correctness | Exact TP1/no-offload `(B,Q,W)` gate implementation | `BLOCKED`: source/schema gates pass, but no idle GPU exists |
| Variable-Q CUDA Graph performance | Controlled 2-warmup/5-measurement-per-family schema | `BLOCKED`: no real warmed latency, mixed TPOT/TTFT/throughput, memory, or capture artifact |
| Native Qwen3.5 MTP real checkpoint | TP1, 4K, production `LLMEngine.step()`, no offload, max proposal 4, batch `(1,4)` | `ESTABLISHED` for transactional correctness: classification `QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED`, b1/b4 parity true, remote/local independent verifier PASS, accepted-prefix target replay zero, and all proposal lifecycle/physical-slot cleanup counters zero. Still `NOT_PROMOTABLE` outside this scope |

## Fresh Verification

Executed locally on 2026-08-12:

```text
Each test file was run in a separate Python process to avoid module-stub
pollution from tools/test_model_runner_spec_verify.py.

  tools/test_speculative_kv_transaction.py             40 passed
  tools/test_speculative_batch_runtime.py
                                                        21 passed
  tools/test_engine_speculative_runtime.py              40 passed
  tools/test_model_runner_spec_verify.py                80 passed
  tools/test_model_runner_spec_verify_cuda_graph.py     41 passed
  tools/test_spec_verify_cuda_graph_config.py           20 passed
  tools/test_spec_verify_cuda_graph_smoke.py            14 passed
  tools/test_spec_verify_cuda_graph_gate.py             57 passed
  tools/test_run_spec_verify_cuda_graph_gate_remote.py  22 passed
  tools/test_prefix_kv_offload_integration.py            3 passed
  tools/test_speculative_residency.py                   10 passed

Total: 348 passed
```

The tests establish local contracts. They do not replace TP4, second-model,
or long-context loaded-model evidence.

### Source-neutral ModelRunner proposal executor slice

The first learned/MTP-ready execution boundary was validated with each file
in a separate Python process:

```text
  tools/test_speculative_adapter.py                    35 passed
  tools/test_speculative_source_adapters.py            24 passed
  tools/test_model_runner_proposal_executor.py         25 passed
  tools/test_speculative_batch_runtime.py              23 passed
  tools/test_engine_speculative_runtime.py             48 passed
  tools/test_speculative_model_runner_callbacks.py     23 passed
  tools/test_speculative_kv_transaction.py             40 passed
  tools/test_speculative_residency_boundary_gate.py    26 passed
  tools/test_speculative_public_api.py                  1 passed
  tools/test_model_runner_spec_verify.py               81 passed
  tools/test_model_runner_spec_verify_cuda_graph.py    41 passed

  Total:                                              367 passed
  py_compile:                                         PASS
  placeholder scan:                                   PASS
  scoped git diff --check:                            PASS
```

The evidence proves that target hidden/logits remain ModelRunner-local,
recursive tensor leakage is rejected, provider failure precedes transaction
creation, empty proposals reserve no KV, non-empty proposals preserve exact-Q
verification, and the existing in-place commit/rejected-suffix rollback path
is unchanged. Scope is TP1, KV offload disabled, exact-Q graph families only.
It does not prove a real learned/MTP checkpoint, CUDA correctness, TP4,
offload compatibility, or performance improvement.

### Native Qwen3.5 MTP real-checkpoint gate

The real-checkpoint gate was rerun on GPU 7 after installing both the physical
MTP K/V transaction path and an independent eager/reference comparison:

```text
remote run root:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-mtp-runs/
    qwen35-mtp-20260813-044311-35753

local artifact:
  artifacts/qwen35-mtp-runs/
    qwen35-mtp-20260813-044311-35753/
      qwen35_mtp_real_checkpoint_gate.json

checkpoint manifest SHA-256:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b
```

The run-directory token contains `20260813` even though the current session
date is 2026-08-12. Treat that token as an opaque remote-generated identifier;
the artifact contents and checkpoint hash are the evidence, not timestamp
ordering inferred from the directory name.

Established by the artifact:

```text
device:                         NVIDIA A100 80GB PCIe
PyTorch / CUDA:                 2.4.1+cu121 / 12.1
checkpoint unchanged:           true
target plus MTP loader:         true
config/tensor contract:         true
shared embedding identity:      true
shared lm_head identity:        true
Q domain:                       (1, 2, 3, 4)
batch domain:                   (1, 4)
acceptance transaction records: 28
eager/reference greedy argmax:    true
eager/reference max logits diff:  0.171875
accepted slot identity:          true
rejected slots released:         true
rollback continuation equal:     true
status:                         FAIL
promotion classification:       NOT_PROMOTABLE
```

The eager/reference domain now runs the same loaded checkpoint twice with
fresh sequences and identical deterministic inputs. The production side uses
the existing Qwen3.5 eager attention equations; the reference side replaces
the prefill/decode entry points with independent PyTorch SDPA equations. Every
required Q/batch case preserves the greedy argmax. The maximum full-logits
absolute difference is `0.171875` in bfloat16.

The transaction domain also continues to pass with real loaded-checkpoint
CUDA attention writes and device-resident K/V tensors. Accepted rows retain
their physical slot IDs, data pointers, and values through commit; rejected
suffix rows are released and zeroed; repeated identical continuations produce
equal token outputs while leaving committed K/V unchanged.

The gate now fails closed for one implementation gap:

```text
graph/eager:
  Qwen3.5 MTP CUDA graph capture backend is not installed
```

The physical transaction implementation uses a Qwen3.5-local block-size-one
store. Each logical proposal token maps directly to a physical K/V slot, so
arbitrary accepted prefixes remain in place without accepted-KV replay, copy,
or rematerialization. Bootstrap uses an exact prefill slot mapping; proposal
steps use decode contexts containing committed slots plus the staged prefix
visible at that step.

```text
physical store and ProposalKVCache focused tests:          17 passed
physical store/executor/integration focused tests:         43 passed
real runtime transaction-probe installation contract:      1 passed
real transaction-probe algorithm cases:                     3 passed
real eager/reference algorithm and corruption cases:         5 passed

isolated local regression process 1:                        51 passed
isolated local regression process 2:                        48 passed
isolated local regression process 3:                        27 passed
total isolated local regression:                           126 passed
```

This proves exact greedy eager/reference parity and physical transaction
properties for TP1, KV offload disabled, one Qwen3.5 MTP layer, Q values 1
through 4, and batch sizes 1 and 4. Synthetic deterministic target hidden rows
feed the real loaded MTP checkpoint module; projections, attention writes,
full logits, and K/V tensors are real CUDA operations. The nonzero bfloat16
maximum difference must remain visible; this is argmax parity, not bitwise
logits identity. It does not prove MTP CUDA Graph correctness, TP4,
KV-offload compatibility, long-context behavior, a second architecture, or
performance improvement.

`tools/test_blockwise_attention_planning.py` was not counted in the fresh
total. In a combined pytest process, the preceding ModelRunner test installs
an intentionally minimal `tinyvllm.layers.attention` stub and pollutes later
collection. In a clean local process, importing the real attention module is
blocked because the local Python environment lacks `flash_attn`. The source
does define `BlockwiseDecodePlan`; this is an environment/test-isolation
limitation, not fresh numerical blockwise evidence.

## Current Native MTP Boundary

The production Engine TP1/4K native-MTP transactional correctness gate is no
longer blocked. The authoritative artifact is:

```text
artifacts/qwen35_native_mtp_tp1_4k_engine/
  opaque-57a3a62810d43636b96295da/
    local-authority/
```

Its classification is
`QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED`; promotion remains
`NOT_PROMOTABLE`.

The remaining native-MTP critical path is:

1. TP4/4K correctness with all-rank learned-MTP execution;
2. TP4/16K and TP4/32K with real KV offload and movement accounting;
3. controlled native-MTP TPOT, TTFT, throughput, peak-memory, H2D/D2H, and
   acceptance evidence;
4. a second learned-drafter/model structure;
5. variable-Q CUDA Graph and fusion/collective optimizations after the
   correctness matrix exists.

## 2026-08-14 Qwen3.5 Native MTP TP1/4K Engine Authority

The frozen four-cell campaign completed in this order:

```text
baseline:b1
native_mtp:b1
baseline:b4
native_mtp:b4
```

Authority:

```text
artifacts/qwen35_native_mtp_tp1_4k_engine/
  opaque-57a3a62810d43636b96295da/
    local-authority/result.json
```

Independent verification:

```text
verify.remote.json: PASS, failures=[]
verify.local.json:  PASS, failures=[]
```

Established evidence:

```text
schema:
  qwen35.native-mtp-tp1-4k-engine-transactional-correctness.v1

classification:
  QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED

promotion:
  NOT_PROMOTABLE

exact greedy parity:
  batch 1: true
  batch 4: true

native batch 1:
  proposal rows:                   8
  proposed tokens:                32
  accepted draft tokens:          30
  rejected draft tokens:          2
  first-target target forwards:   8
  verifier target forwards:       8
  accepted-prefix target replays: 0

native batch 4:
  proposal rows:                   32
  proposed tokens:                128
  accepted draft tokens:          120
  rejected draft tokens:          8
  first-target target forwards:   32
  verifier target forwards:       32
  accepted-prefix target replays: 0
```

Both native cells end with:

```text
pending_prefix_count:             0
bootstrapped_sequence_count:      0
proposal_transaction_count:       0
batch_ticket_count:               0
batch_ticket_transaction_count:   0
allocated_physical_slot_count:    0
```

The source-bound run uses target manifest
`3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0`,
MTP manifest
`9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b`,
and source-tree digest
`4b962e5ba84b6594682fb5fd4cf72144d2c7d5faff982f0107fed7edae81fb4d`.
The before/after GPU process inventories are identical.

Two production defects were removed before the passing campaign:

1. bootstrap previously called `lm_head` for all batch-by-4096 hidden rows,
   materializing unnecessary full-vocabulary logits and OOMing batch 4;
   `Qwen35NativeMTP.forward_hidden()` now runs the bootstrap hidden/KV path
   without logits, while proposal steps retain full logits;
2. lifecycle evidence retained stale active sequence IDs after side-state
   seal, causing ordinary scheduler commits after sequence release to pollute
   batch-4 speculative ordering; the capture scope now ends at seal.

Fresh related local verification after both fixes:

```text
200 passed in 6.54s
py_compile: PASS
scoped git diff --check: PASS
```

This authority does not establish TP4 native MTP, native MTP with KV offload,
16K/32K learned-MTP correctness, native-MTP performance direction, KV8/KV4,
or Phase 1 production readiness.

## Ordered Next Actions

1. Run a controlled 16K/32K performance campaign covering TTFT, TPOT,
   throughput, peak memory, real movement, and acceptance.
2. Add TP4 performance profiling that measures collective/compute overlap
   rather than only proving collective execution.
3. Add one real learned-draft or end-to-end native-MTP executor path through
   `LLMEngine.step()` with KV offload enabled.
4. Design the remaining combinations: KV4/KV8 with offload, heat-tiered KV,
   verifier/sampling/commit fusion, and TP collective overlap.

## Completion Audit Verdict

The first-phase objective is **not achieved**.

Established local deliverables:

```text
source-neutral speculative adapter/runtime contract
batch-native fixed-Q multi-token verification
token-free transactional KV commit and rejected-suffix rollback
speculative residency prepare/precommit/seal/rollback contract
ordinary prefix plus CPU-resident KV composition contract
exact TP1/no-offload variable-Q CUDA Graph implementation and verifier schema
real Qwen3.5 target plus native-MTP checkpoint loading and shared-weight identity
real Qwen3.5 MTP device-resident proposal K/V transaction
accepted physical-slot in-place commit and rejected-suffix release
rollback-safe cached continuation across all 28 Q/batch/acceptance cases
independent real-checkpoint eager/SDPA-reference exact-greedy parity for
Q=(1,2,3,4) and batch=(1,4)
Qwen3-0.6B generic n-gram TP4 batch-1/4 exact output parity
Qwen3.5 hybrid generic n-gram TP4 4K/16K/32K batch-1/4 exact parity
Qwen3.5 TP4/32K bounded pure-prefill online softmax and real KV movement
all-rank speculative callback and collective identity
all-rank residency prepare/precommit/seal acknowledgement and clean shutdown
```

Missing or insufficiently verified deliverables:

```text
real learned draft-model executor in LLMEngine.step()
variable-Q CUDA performance PASS artifacts
16K/32K TPOT, TTFT, throughput, and peak-memory direction
complete promotion matrix for parity, TPOT, TTFT, throughput, memory,
real KV movement, and acceptance
TP4 performance and collective-overlap direction
KV4/KV8 plus offload
generic per-layer/per-token heat tiers
verifier/sampling/KV-commit fusion
TP collective overlap, AllReduce fusion, ReduceScatter, and persistent
hidden-state sharding
```

The objective is not complete, and no promotion claim is permitted.

## Exact-Q Native MTP CUDA Graph Gate Continuation

This section supersedes the earlier statement that the Qwen3.5 native-MTP
CUDA Graph backend is not installed. The earlier transaction and
eager/reference evidence remains valid; the authoritative graph continuation
artifact is:

```text
opaque run ID:
  qwen35-mtp-graph-gate-opaque-7

local artifact:
  artifacts/qwen35-mtp-runs/
    qwen35-mtp-graph-gate-opaque-7/
      qwen35_mtp_real_checkpoint_gate.json

checkpoint manifest SHA-256:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b

device:                          NVIDIA A100 80GB PCIe
PyTorch / CUDA:                  2.4.1+cu121 / 12.1
status:                          PASS
promotion classification:       NOT_PROMOTABLE
backend failures:                []
```

The run ID is opaque and is not used as date or ordering evidence. The
artifact covers TP1, KV offload disabled, greedy sampling, one MTP layer,
Q values `(1,2,3,4)`, and batch sizes `(1,4)`. Q1 remains eager passthrough;
the six captured exact families are Q2/Q3/Q4 crossed with batch 1/4.

```text
graph backend installed:                 true
graph capture count:                     6
graph replay count:                     12
graph/eager greedy argmax equal:         true
graph/eager proposal tokens equal:       true
graph transaction commit:               true
graph transaction rollback:             true
post-replay failure quarantined:         true
post-replay eager retry count:           0

eager/reference max logits difference:   0.171875
eager/reference greedy argmax equal:     true
accepted slot identity preserved:        true
rejected slots released:                 true
post-rollback continuation equal:        true
```

The graph implementation uses distinct exact-Q/batch families with static
buffers, a shared CUDA graph memory pool and capture stream, and a private
scratch cache/sequence namespace. The block-size-one proposal store uses a
capture-safe GPU attention path rather than the generic FlashAttention paged
KV kernel. Before replay starts, a graph failure may use eager fallback.
After `graph.replay()` starts, an exception quarantines that family and must
not trigger eager retry. The injected Q4/batch-4 replay failure establishes
the latter boundary with a retry count of zero.

Accepted proposal K/V remains in its original physical slots through commit.
Rejected suffix slots are rolled back and released. The graph transaction
probe therefore does not establish correctness by replaying, copying, or
rematerializing accepted K/V.

Fresh local evidence associated with this continuation:

```text
tools/test_qwen35_mtp_real_checkpoint_gate.py:       53 passed
tools/test_qwen35_mtp_cuda_graph_backend.py:         21 passed
dependency-light combined graph/generic regression: 215 passed
post-document eight-file dependency-light rerun:     211 passed
artifact validate_gate_report(...):                 PASS
Python syntax compilation:                          PASS
remote wrapper bash syntax:                         PASS
scoped git diff check:                              PASS
```

The complete 13-file Task 8 command was also attempted after the document
update. It stopped during collection because this local Python installation
does not provide `torch`; the five direct-`torch` test files each raised
`ModuleNotFoundError: No module named 'torch'`. The remaining eight
dependency-light files were rerun together and produced the `211 passed`
result above. This is a local dependency boundary, not a passing result for
the five omitted files. Their loaded Torch/CUDA behavior is represented by
the real GPU7 artifact, which passed the gate schema verifier.

The planned standalone synthetic
`tools/qwen35_mtp_cuda_graph_smoke.py` was not created. The real-checkpoint
gate supplies the stronger loaded-GPU evidence and is the authority for this
continuation; no claim depends on a nonexistent synthetic smoke file.

This evidence does not establish TP4, KV-offload compatibility, arbitrary
Q or batch sizes, multiple MTP layers, non-greedy sampling, a second model
structure, long-context behavior, end-to-end speculative activation in
`LLMEngine.step()`, or any latency, throughput, memory, or other performance
gain.

The exact-Q native-MTP CUDA Graph blocker is closed for the narrow domain
above, but the first-phase objective remains **not achieved** and the result
remains **NOT_PROMOTABLE** because the independent promotion matrix and the
other missing deliverables listed above remain open.

## Loaded-GPU ModelRunner Proposal Ownership Gate Continuation

This continuation closes the previously ordered loaded-GPU ownership gate
for the same narrow TP1 Qwen3.5 domain. It does not change the phase-level
verdict.

Authoritative artifact:

```text
opaque run ID:
  qwen35-mtp-ownership-26023-99638

local artifact:
  artifacts/qwen35-mtp-runs/
    qwen35-mtp-ownership-26023-99638/
      qwen35_mtp_model_runner_ownership_gate.json

checkpoint manifest SHA-256:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b

device:                          NVIDIA A100 80GB PCIe
PyTorch / CUDA:                  2.4.1+cu121 / 12.1
status:                          PASS
promotion classification:       NOT_PROMOTABLE
backend failures:                []
graph capture / replay count:    6 / 6
public result tensor count:      0
cleanup passed:                  true
```

The run ID is opaque and is not date or ordering evidence. The gate executes
the loaded target ModelRunner forward for Q `(1,2,3,4)` and batch `(1,4)`.
Q1 is eager-only. Q2/Q3/Q4 each capture and replay one exact graph for batch
1 and batch 4.

```text
fused ModelRunner path exercised:              true
real target forward:                           true
target logits on CUDA:                         true
target hidden on CUDA:                         true
target hidden consumed by real executor:       true
target logits not passed to MTP executor:      true
public result pickle roundtrip:                true
public result tensor-free:                     true
executor identity preserved:                   true
sequence order preserved:                      true
graph/eager first-target tokens equal:         true
graph/eager proposal tokens equal:             true
all eight case cleanups passed:                true
```

The observer records only scalar/list metadata and does not retain, clone,
detach, hash, serialize, or copy hidden/logits tensors to CPU. Public results
contain token and transaction metadata only.

Each graph/eager side owns fresh sequences, blocks, and hybrid-state leases.
Proposal transactions are finalized with zero accepted proposal tokens and
rolled back before sequence release. Cleanup releases executor sequence
state with the keyword-only sequence epoch, releases runtime bindings and
allocator leases, zeros reserved target K/V blocks, and restores the model
context. Every artifact case reports no cleanup errors.

The six-family loaded gate requires more reserved CUDA allocator headroom
than the production default graph budget. The ownership tool therefore raises
only its loaded gate instance to a 3 GiB reserved-byte ceiling. The production
default in `tinyvllm/config.py` remains 512 MiB. This is validation capacity,
not a runtime promotion or memory-efficiency claim.

Fresh local evidence before this document update:

```text
ownership focused suite:                         71 passed
proposal/integration/graph/checkpoint regression: 126 passed
ModelRunner first-target/spec-verify regression:  83 passed
runtime regression total:                         209 passed
downloaded artifact strict verifier:              PASS
```

The document-after combined relevant pytest matrix passed `280` tests.
Python compilation, remote-wrapper shell syntax, the downloaded artifact
verifier, and the scoped diff check also passed.

This result establishes only loaded-GPU proposal ownership, graph/eager token
parity, and cleanup for TP1, KV offload disabled, greedy sampling, one MTP
layer, Q `(1,2,3,4)`, and batch `(1,4)`. It does not establish TP4, KV
offload, a second model structure, 16K/32K context, multiple MTP layers,
non-greedy sampling, end-to-end activation in `LLMEngine.step()`, or any
TPOT, TTFT, throughput, memory, or KV movement improvement.

The loaded-GPU ownership blocker is closed for this narrow domain. The first
phase objective remains **not achieved** and **NOT_PROMOTABLE**. At that
checkpoint the next action was the TP1 16K/32K blockwise campaign; the
continuation below records its completed result and supersedes that pending
status.

## TP1 16K/32K Blockwise Speculative Verifier Continuation

The pending TP1 long-context correctness/movement campaign is now closed for
Qwen3-0.6B, batch 1/4, baseline versus the generic n-gram runtime.

Authoritative artifact:

```text
opaque run ID:
  blockwise-tp1-opaque-17786-19070

local artifact:
  artifacts/blockwise_speculative_verifier/
    blockwise-tp1-opaque-17786-19070/result.json

artifact SHA-256:
  2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600

device:                       NVIDIA A100 80GB PCIe
PyTorch / CUDA:               2.4.1+cu121 / 12.1
status / classification:      PASS / NOT_PROMOTABLE
remote verifier:              PASS
local verifier:               PASS
remote direct KV regression:  kv offload tests passed
```

The run ID is opaque and is not date evidence. Both verifiers independently
matched the same artifact SHA-256 and the artifact-recorded source hashes.
All eight worker JSON files are present.

```text
16K / batch 1 parity: PASS
16K / batch 4 parity: PASS
32K / batch 1 parity: PASS
32K / batch 4 parity: PASS

candidate proposed tokens:       80
candidate accepted tokens:       58
first-target callbacks:          14
tail callbacks:                   9
candidate real H2D copies:   221,410
candidate real H2D bytes: 6,500,625,940,480
candidate real D2H copies:     1,004
candidate real D2H bytes: 29,477,568,512
rejected speculative D2H:          0
```

The 16K/batch-4, 32K/batch-1, and 32K/batch-4 cells expose 256, 128, and 512
logical blocks respectively, all greater than the 68 GPU slots, and each has
positive real H2D copies/bytes. The 16K/batch-1 cell exposes 64 blocks and
fits in the GPU staging budget, so its zero H2D delta is expected and is not
used as movement evidence.

The loaded campaign records zero speculative committed/rejected block-count
deltas. It therefore proves exact accepted-token behavior and zero rejected
speculative D2H, but positive block-level commit/reject-count transitions
remain focused-test authority rather than a loaded-campaign claim.

This closes only the TP1 Qwen3-0.6B 16K/32K correctness/movement gap. It does
not establish a 16K/32K performance direction, TP4, a second model structure,
a learned draft model, native MTP plus KV offload, KV8/KV4 speculative
verification, or promotion. The Phase 1 objective remains **not achieved**
and **NOT_PROMOTABLE**.

Post-document verification produced `124` dependency-light passes in isolated
Python processes, a strict artifact-verifier PASS, a remote direct
`tools/test_kv_offload.py` PASS, ten consecutive runner lifecycle passes,
Python compilation PASS, shell syntax PASS, artifact-condition PASS, and an
empty staged diff. The gate-file scoped diff check passes. The repository-wide
diff check is still blocked by unrelated trailing whitespace in existing
`tinyvllm/engine/model_runner.py` changes; that hygiene issue is not evidence
against the downloaded long-context artifact.

## Generic Speculative Runtime TP4 Correctness Authority

The generic Qwen3-0.6B host n-gram runtime now has a real TP4 correctness and
collective authority for batch sizes 1 and 4.

Authoritative local directory:

```text
artifacts/generic_speculative_tp4/
  tp4-opaque-48d18e4aba16756d/
    authority/
```

The run ID is opaque and is not date or ordering evidence.

Authority identity:

```text
classification:              NOT_PROMOTABLE
world size:                  4
physical GPUs:               7,5,2,0
distributed/master bases:    3623 / 3723
remote ephemeral range:      10000-65535
result.json SHA-256:
  4e504110074eb8c6a5d449d381d599d5e4303ac05371ad8c40cf9cea50955e9b
source tree SHA-256:
  88d30b69246ac9c15caab5ce3c7f5f82fad00d7ec24e4c005e8ab31beed97546
model manifest SHA-256:
  6bb7f90f4ad46c059c9e3df600532147ecc00683e58e96ce9dd6bc5084f2c90e
campaign verifier:            PASS
remote verifier:              PASS
runner local verifier:        PASS
fresh local verifier:         PASS
```

The remote runner selects four GPUs from a real `nvidia-smi` inventory and
uses four distinct rendezvous pairs below the host's ephemeral port range.
This was required because the host uses `10000-65535` for ephemeral source
ports; earlier high-port attempts reproducibly failed the second cell with
`EADDRINUSE`.

Exact baseline/candidate output parity:

```text
batch 1: PASS
batch 4: PASS
```

Candidate runtime evidence:

```text
batch 1:
  proposal rows / proposed / accepted: 2 / 8 / 7
  first-target / tail callbacks:       2 / 2

batch 4:
  proposal rows / proposed / accepted: 8 / 32 / 28
  first-target / tail callbacks:       2 / 2
```

Each candidate cell records eight speculative callback profile steps on each
rank, covering both `spec_first_target` and `spec_verify`. Each rank records
456 collectives whose step indices cover every speculative callback. The
callback and collective identities match across ranks 0, 1, 2, and 3.

Each candidate cell also records two complete
`prepare -> precommit -> seal` residency transactions. Every phase contains
successful acknowledgements from participants 0, 1, 2, and 3. Cleanup records
rank exit codes `[0,0,0,0]`, no owned children remaining, and destroyed
process groups on all four ranks.

The TP4 KV movement rows are real `KVOffloadMVP0` counters. Batch 1 records
20 D2H copies / 146,800,640 bytes per rank; batch 4 records 80 D2H copies /
587,202,560 bytes, 68 evictions, and 17 copy waits per rank. Both candidate
cells record zero rejected speculative D2H copies. The loaded run records zero
speculative committed/rejected block-count deltas, so positive transactional
block-count transitions remain focused-test evidence rather than a TP4
loaded-run claim.

Fresh local validation after the final authority:

```text
TP4 gate plus ModelRunner callback regression: 116 passed
Python compilation:                            PASS
remote wrapper shell syntax:                   PASS
four independent artifact verifier outputs:   PASS
```

This closes the generic Qwen3-0.6B TP4 correctness, all-rank callback,
collective-presence, residency-acknowledgement, output-parity, and cleanup
gate for 4K batch 1/4. It does not establish TP4 performance, collective
overlap, 16K/32K TP4 behavior, a second model structure, a learned drafter,
native MTP plus KV offload, KV8/KV4, or Phase 1 completion. The objective
remains **not achieved** and **NOT_PROMOTABLE**.

## Qwen3.5 Second-Model TP1 Transactional Correctness Authority

The full Qwen3.5-2B hybrid/recurrent Engine path now has a real TP1 generic
n-gram speculative correctness authority for batch sizes 1 and 4.

Authoritative local directory:

```text
artifacts/qwen35_generic_speculative_tp1/
  opaque-d4e74cb46fccbc57319c3c4f/
    artifacts/authority/
```

The opaque run ID is not date or ordering evidence.

Authority identity:

```text
classification:              SECOND_MODEL_TP1_ESTABLISHED
selected physical GPU:       0
GPU free / total MiB:        46,465 / 81,920
result.json SHA-256:
  4113627fe0efc4bdb767d8f6098f4264f6439c2bc3634aecff9be540c6be6e99
source_manifest.json SHA-256:
  6ca74660189b0ee21b3cc0d1c0d90771410ff152fbf5f49b16c3cf46acd45d88
source tree SHA-256:
  43babb8801bdfdd903aad1240e017b96d4347174223d056dd73cdab63f4e0f2f
model manifest SHA-256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
remote verifier:              PASS
runner local verifier:        PASS
fresh local verifier:         PASS
```

The bound checkpoint identifies `Qwen3_5ForConditionalGeneration`,
`model_type=qwen3_5`, 24 text layers, 18 linear-attention layers, and six
full-attention layers.

Design-to-artifact checklist:

```text
real Qwen3.5 architecture identity:          ESTABLISHED
TP1 baseline and generic n-gram cells:       ESTABLISHED
batch 1 and batch 4:                         ESTABLISHED
4K-class prompts:                            4,048 tokens each
eight-token exact greedy output parity:      ESTABLISHED
positive proposal / accept / reject:         ESTABLISHED
first-target and verify callbacks:           ESTABLISHED
accepted full-attention KV publication:      ESTABLISHED in the bound path
rejected full-attention KV exclusion:        ESTABLISHED in the bound path
recurrent consumed-input checkpoint select:  ESTABLISHED
reversible apply and seal lifecycle:         ESTABLISHED
accepted-prefix target replay:               0
hybrid-state lease cleanup:                  PASS
Engine and process-group cleanup:            PASS
source and model identity binding:            PASS
independent verification:                    PASS
```

The loaded artifact exercises mixed acceptance and rejection through the
source-bound speculative KV transaction and side-state transaction. It does
not expose separate block-delta counters, so the checklist does not claim a
loaded block-count transition receipt.

Exact baseline/candidate output parity:

```text
batch 1: PASS
batch 4: PASS
```

Candidate runtime evidence:

```text
batch 1:
  proposal rows / proposed / accepted / rejected: 2 / 8 / 7 / 1
  first-target / verify callbacks:                2 / 2
  consumed-input checkpoint selections:           2
  side-state lifecycle receipts:                  8

batch 4:
  proposal rows / proposed / accepted / rejected: 8 / 32 / 28 / 4
  first-target / verify callbacks:                2 / 2
  consumed-input checkpoint selections:           8
  side-state lifecycle receipts:                 32
```

Every successful sequence lifecycle contains
`prepare -> select -> apply -> seal`. Both candidate cells record zero
accepted-prefix replay, zero remaining hybrid-state leases, an unpoisoned
runtime, worker/rank exit code zero, no owned child process, and a destroyed
process group.

The real campaign exposed and retained two useful failed authorities before
the final pass:

- `opaque-68c4fa3269d1952a1bf9e134` found that batched `spec_verify`
  request-local Qwen3.5 full-attention execution received the global
  `B * Q` slot mapping;
- `opaque-b567c92f76237c77ac4c7780` found that the authority validator
  grouped shared batch transaction receipts only by handle rather than by
  `(handle, sequence)`.

Both defects received focused RED/GREEN tests. A prior retained failure,
`opaque-b9b5605018e102e53c98080f`, established that side-state sealing must
remain legal after the prepared batch container enters the committed state.

Final validation:

```text
focused Task 11 regression:       347 passed
changed-source Python compilation: PASS
remote runner shell syntax:        PASS
independent artifact verifier:     SECOND_MODEL_TP1_ESTABLISHED
```

Only the narrow second-model TP1 correctness row is established:

```text
second-model TP1 transactional correctness: ESTABLISHED
second-model TP4:                            MISSING
second-model 16K/32K:                        MISSING
second-model performance:                    MISSING
Phase 1:                                     NOT_PROMOTABLE
```

The next ordered gate is Qwen3.5 TP4 generic speculative transaction
authority. This result does not establish TP4, 16K/32K behavior, performance,
a learned drafter, native MTP plus KV offload, KV8/KV4, or Phase 1 completion.

## Qwen3.5 Second-Model TP4/4K Transactional Correctness Authority

The real four-GPU Qwen3.5 hybrid/recurrent Engine path now has a source-bound
generic n-gram speculative transactional correctness authority at 4K context
for batch sizes 1 and 4.

Objective coverage row:

```text
Qwen3.5 | generic n-gram speculative | TP4 | 4K | batch 1/4 |
SECOND_MODEL_TP4_4K_ESTABLISHED
```

Authoritative local directory:

```text
artifacts/qwen35_generic_speculative_tp4/
  opaque-24f8ae471a2ba439ecb5a3b1/
    artifacts/authority/
```

Identity and verification:

```text
classification:
  SECOND_MODEL_TP4_4K_ESTABLISHED

selected physical GPUs:
  7,5,3,2

result.json SHA-256:
  1d99a3b0c40d94ab57f6f67e5e3d1b3a6431395aaeca57333a7249e0388b3ea6

source_manifest.json SHA-256:
  138f338408d7cc4ee0b1821613233115bc60a4dcc8593d456bc23679f1bf7755

source tree SHA-256:
  c119fbc63d15a7f615aa8436777582d9a8b131f43b91fe8047c12bfee61ec06c

model manifest SHA-256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0

campaign verifier:       PASS
remote fresh verifier:   PASS
runner local verifier:   PASS
fresh repository verifier: PASS
raw evidence audit:      PASS
```

Exact baseline versus generic n-gram greedy parity:

```text
batch 1: PASS
batch 4: PASS
context tokens per request: 4,096
output tokens per request:  8
```

Candidate transactional evidence:

```text
batch 1:
  proposal rows / proposed / accepted / rejected: 2 / 8 / 5 / 3
  first-target / verify callbacks:                2 / 2
  transactions per rank:                          2
  side-state receipts per rank:                   8
  profile steps / collectives per rank:            9 / 125
  residency phases:                               6

batch 4:
  proposal rows / proposed / accepted / rejected: 8 / 32 / 20 / 12
  first-target / verify callbacks:                2 / 2
  transactions per rank:                          8
  side-state receipts per rank:                  32
  profile steps / collectives per rank:           21 / 485
  residency phases:                               6
```

Every candidate sequence transaction uses the canonical consumed-input
mapping:

```text
verify_input_count = max(0, proposal_token_count - 1)
committed_tail_input_count = min(
  accepted_draft_count,
  verify_input_count,
)
committed_input_count = 1 + committed_tail_input_count
```

All four ranks record the same semantic digest for each
`(sequence_id, transaction_ordinal)`. Every successful side-state lifecycle
is `prepare -> select -> apply -> seal`. Both candidate cells record zero
accepted-prefix target replay, complete all-rank callback/collective evidence,
complete `prepare -> precommit -> seal` residency acknowledgements, no live
leases or prepared transactions after execution, no runtime poison, no owned
children, rank exit codes `[0,0,0,0]`, and destroyed process groups.

The loaded KV movement evidence comes from
`engine.kv_offload_summaries`, not a synthetic tensor copy:

```text
batch 1 per rank:
  D2H copies / bytes: 20 / 125,829,120

batch 4 per rank:
  D2H copies / bytes: 80 / 503,316,480
```

The loaded cells record zero speculative residency committed/rejected
block-count deltas and zero rejected D2H copies. Therefore this authority
establishes the source-bound transactional execution and real KV movement
provenance, but does not claim positive loaded-run speculative block-delta
counts or a performance benefit.

Real-run defects found and fixed with focused RED/GREEN:

1. production residency acknowledgements use tuple-valued sequence and block
   identities before JSON serialization; the gate now accepts list/tuple
   inputs and normalizes them to lists;
2. request-local Qwen3.5 full-attention execution now slices
   `kv_offload_logical_block_tables` and `kv_offload_context_lens` for decode
   and `spec_verify`, preventing a single-request query from broadcasting
   against all batch KV rows.

Retained failed authorities:

```text
opaque-eb570c58546e0cb8956887db
  ngram:b1 production tuple receipt boundary

opaque-7c230a52740e853def8fd546
  baseline:b4 request-local KV-offload row mismatch
```

Final validation:

```text
new TP4 gate module:                  36 passed
existing TP4 plus Qwen3.5 TP1 gates: 59 passed
remote focused context tests:         2 passed
changed-source Python compilation:    PASS
remote runner shell syntax:           PASS
git diff --check:                     PASS
fresh independent verifier:
  {"classification":"PASS","failures":[]}
```

Updated boundary:

```text
Qwen3.5 second-model TP1 transactional correctness: ESTABLISHED
Qwen3.5 second-model TP4/4K correctness:             ESTABLISHED
Qwen3.5 second-model TP4 16K/32K:                    MISSING
Qwen3.5 second-model TP4 performance:                MISSING
learned drafter / native MTP plus KV offload:        MISSING
KV8/KV4 authority:                                   MISSING
Phase 1:                                             NOT_PROMOTABLE
```

This closes only the narrow second-model TP4/4K correctness row. It must not
be extended to 16K/32K, TPOT/TTFT/throughput or memory improvement, a learned
drafter, native MTP plus KV offload, KV8/KV4, or Phase 1 completion.

## Qwen3.5 Second-Model TP4/16K Transactional Correctness Authority

The real four-GPU Qwen3.5 hybrid/recurrent Engine path now has a separate,
source-bound generic n-gram speculative transactional correctness authority
at 16,384 context tokens for batch sizes 1 and 4.

Objective coverage row:

```text
Qwen3.5 | generic n-gram speculative | TP4 | 16K | batch 1/4 |
SECOND_MODEL_TP4_16K_ESTABLISHED
```

Authoritative local directory:

```text
artifacts/qwen35_generic_speculative_tp4_16k/
  opaque-3b8050a916f037bc92412ea5/
    artifacts/authority/
```

Identity and verification:

```text
classification:
  SECOND_MODEL_TP4_16K_ESTABLISHED

selected physical GPUs:
  2,7,6,5

result.json SHA-256:
  71f379500c82f155af2f0181d85cdf370849b30ca7c7ea016164885fa1c9a86e

source_manifest.json SHA-256:
  cc9add07d2e924b09a568b34124be59dae732e78ec62a4c44a55694be2cd7210

source tree SHA-256:
  07e945dd62b0d4afa08d7c28ffced87f4688a9719d43cbd8b54427017509ad5f

model manifest SHA-256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0

runner local verifier:      PASS
fresh repository verifier: PASS
raw evidence audit:         PASS
```

Exact baseline versus generic n-gram greedy parity:

```text
batch 1: PASS
batch 4: PASS
context tokens per request: 16,384
output tokens per request:  8
accepted-prefix replay:     0 in every cell
```

Candidate transactional evidence:

```text
batch 1:
  proposal rows / proposed / accepted / rejected: 3 / 12 / 6 / 6
  first-target / verify callbacks:                3 / 3
  transactions per rank:                          3
  side-state receipts per rank:                  12
  profile steps / collectives per rank:           22 / 150
  residency phases:                               9

batch 4:
  proposal rows / proposed / accepted / rejected: 12 / 48 / 24 / 24
  first-target / verify callbacks:                 3 / 3
  transactions per rank:                         12
  side-state receipts per rank:                  48
  profile steps / collectives per rank:           70 / 582
  residency phases:                               9
```

All four ranks agree on every sequence transaction semantic digest and record
the complete recurrent lifecycle `prepare -> select -> apply -> seal`.
Full-attention KV residency records complete `prepare -> precommit -> seal`
phases, rejected suffix rollback, and zero rejected speculative D2H copies.

The 16K batch-4 candidate cell exceeds the fixed 68-block GPU staging budget
and records real production KV-offload movement:

```text
provenance:
  engine.kv_offload_summaries

ngram batch 4 aggregate:
  H2D copies:  36,864
  H2D bytes:   231,928,233,984
  D2H copies:  1,104
  D2H bytes:   6,945,767,424
  copy waits:  14,016
  evictions:   37,632
  rejected speculative D2H copies: 0
```

These counters prove real loaded-model movement and the long-context
residency boundary. They do not prove reduced traffic or any performance
improvement.

Cleanup is complete in every cell:

```text
rank exit codes:                 [0,0,0,0]
process group destroyed:         true
owned children remaining:        []
live leases per rank:             0
prepared transactions per rank:   0
runtime poisoned per rank:        false
```

The first retained 16K campaign failed before authority creation:

```text
artifacts/qwen35_generic_speculative_tp4_16k/
  opaque-693783011850036d504d862a/
    artifacts/authority.failed/
```

Root cause: the reused frozen `run_campaign()` had definition-time defaults
bound to the 4K worker, 4K source tuple, and 4K verifier. It therefore
dispatched a valid 4K cell into the 16K validator and failed with
`cell schema version mismatch`. A focused RED/GREEN test now requires the
16K campaign adapter to inject the 16K worker, extended source inventory, and
independent 16K verifier.

Final validation:

```text
new 16K authority tests:        11 passed
frozen 4K authority tests:      36 passed
changed-source Python compile:  PASS
remote runner shell syntax:     PASS
git diff --check:               PASS
fresh independent verifier:
  {"classification":"PASS","failures":[]}
```

The local `tools/test_qwen35_packed_layer_stack.py` collection remains
unavailable in the host Python because `torch` is not installed. This is an
environment limitation, not a passing or failing result; the production
packed Qwen3.5 path was exercised by the successful real GPU campaign.

Updated boundary:

```text
Qwen3.5 second-model TP1 transactional correctness: ESTABLISHED
Qwen3.5 second-model TP4/4K correctness:             ESTABLISHED
Qwen3.5 second-model TP4/16K correctness:            ESTABLISHED
Qwen3.5 second-model TP4/32K correctness:            MISSING
Qwen3.5 second-model TP4 performance:                MISSING
learned drafter / native MTP plus KV offload:        MISSING
KV8/KV4 authority:                                   MISSING
Phase 1:                                             NOT_PROMOTABLE
```

This closes only the independent second-model TP4/16K correctness row. It
must not be extended to 32K, TPOT/TTFT/throughput or memory improvement,
learned drafting, native MTP plus KV offload, KV8/KV4, production readiness,
or Phase 1 completion.

## Qwen3.5 Second-Model TP4/32K Transactional Correctness Authority

The independent Qwen3.5 TP4/32K gate is complete and independently verified.
It closes the second-model 32K correctness row without making a performance
or promotion claim.

Authoritative local directory:

```text
artifacts/qwen35_generic_speculative_tp4_32k/
  opaque-03a0a96654a14441b314800f/
    artifacts/authority/
```

Identity:

```text
schema:
  qwen35.generic-speculative-tp4-32k-transactional-correctness.v1

classification:
  SECOND_MODEL_TP4_32K_ESTABLISHED

scope:
  second_model_tp4_32k_only

selected physical GPUs:
  7,5,3,2

result.json SHA-256:
  2b81b1e9379c454a89e062ab5ec6cf0fb93860b3c11273f927dc9fcb0fe15c28

source_manifest.json SHA-256:
  9fffe8d88b22eb3f7903dd799fb154c3a12227d15843331d62bd805ea0b70ac6

source tree SHA-256:
  f4d4c684a39bd404fc32821a6d4a8997c4ca36adbe28526ef2fbfab8d5cd54da

model manifest SHA-256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

Exact parity and candidate evidence:

```text
context tokens per request: 32,768
output tokens per request:  8
batch 1 parity:             PASS
batch 4 parity:             PASS
accepted-prefix replay:     0 in every cell

ngram batch 1:
  proposal rows / proposed / accepted / rejected: 2 / 8 / 7 / 1
  first-target / verify callbacks:                2 / 2
  transactions per rank:                          2
  side-state receipts per rank:                   8

ngram batch 4:
  proposal rows / proposed / accepted / rejected: 8 / 32 / 28 / 4
  first-target / verify callbacks:                2 / 2
  transactions per rank:                          8
  side-state receipts per rank:                  32
```

Real production KV-offload movement:

```text
ngram batch 1 aggregate:
  H2D copies / bytes: 6,100 / 38,377,881,600
  D2H copies / bytes:   528 /  3,321,888,768

ngram batch 4 aggregate:
  H2D copies / bytes: 49,152 / 309,237,645,312
  D2H copies / bytes:  2,112 /  13,287,555,072
  copy waits:          18,880
  evictions:           50,944

rejected speculative D2H copies:
  0 in both candidate cells
```

All four ranks record equal output tokens, matching transactional semantic
digests, complete side-state and residency lifecycles, zero live leases,
zero prepared transactions, no runtime poison, rank exits `[0,0,0,0]`, no
owned children, and destroyed process groups. The independent verifier
returned:

```json
{"classification":"PASS","failures":[]}
```

The first real 32K campaign retained at
`opaque-470c642662df1144a3198663/artifacts/authority.failed` exposed an
8 GiB dense pure-prefill attention allocation. The root fix adds bounded
query/key tiled causal online softmax in
`tinyvllm/layers/qwen35_full_attention.py`: reference lengths up to 16,384
keep the frozen dense path, while longer pure-prefill segments use 512-token
tiles with FP32 online-softmax state and never materialize a full Q-by-K
matrix. KV cache writes remain limited to real tokens; reference padding is
not materialized as fake KV.

A second retained campaign,
`opaque-758ce5286988bca040f8c65c/artifacts/authority.failed`, completed
`baseline:b1` and thereby proved the dense OOM was removed. Its subsequent
`ngram:b1` Engine initialization encountered transient GPU capacity pressure
before speculative runtime activation. A fresh run on physical GPUs
`7,5,3,2` completed all four cells and the verifier.

Fresh validation:

```text
blockwise attention RED:       3 expected failures
focused blockwise GREEN:       3 passed
complete attention regression: 13 passed
4K/16K/32K authority tests:    62 passed
changed-source Python compile: PASS
git diff --check:              PASS
remote campaign:               COMPLETE / exit 0
independent verifier:          PASS
```

Updated boundary:

```text
Qwen3.5 second-model TP1 transactional correctness: ESTABLISHED
Qwen3.5 second-model TP4/4K correctness:             ESTABLISHED
Qwen3.5 second-model TP4/16K correctness:            ESTABLISHED
Qwen3.5 second-model TP4/32K correctness:            ESTABLISHED
Qwen3.5 second-model TP4 performance:                MISSING
learned drafter / native MTP plus KV offload:        MISSING
KV8/KV4 authority:                                   MISSING
Phase 1:                                             NOT_PROMOTABLE
```

This closes only the second-model TP4 4K/16K/32K correctness matrix. It must
not be extended to TPOT, TTFT, throughput, peak-memory improvement, learned
drafting, native MTP plus KV offload, KV8/KV4, production readiness, or
Phase 1 completion.

## 2026-08-13 Artifact-Backed Promotion Re-Audit and Native-MTP Readiness

This update re-read the actual authority and verifier files. It does not rely
on the handoff summary as proof. The Phase 1 objective remains incomplete.

### Prompt-to-artifact checklist

| Objective requirement | Current evidence | Status and boundary |
| --- | --- | --- |
| One source-neutral speculative runtime | `tinyvllm/engine/speculative_runtime.py` accepts exactly one host `draft_adapter` or one `model_runner_executor`; `tinyvllm/engine/speculative_model_runner.py` routes both through the same Engine lifecycle | `ESTABLISHED` as a code and test contract |
| Model-free n-gram/SAM sources | `tinyvllm/speculative/ngram_adapter.py`, SAM adapter coverage in `tools/test_speculative_source_adapters.py`, and real n-gram authorities listed below | n-gram has real authority; SAM remains code/test evidence only |
| Independent learned draft model | Concrete Qwen3 dense backend, loader/registration contract, physical proposal-KV store, batch-native autoregressive executor, TP1 Engine gate, and preflight-only negative-evidence mode now exist | `PARTIAL`; no completed real checkpoint TP1 authority, TP4 remains fail-closed, and no performance matrix exists |
| Native MTP head | Qwen3.5 native checkpoint executor and production `LLMEngine.step()` gates now have completed TP1/4K, TP4/4K, and TP4/16K target-KV-offload authorities | `PARTIAL`; TP4/32K failed exact batch-1 parity, controlled performance is missing, and proposal-KV offload is not established |
| Batch-native multi-token target verifier | `run_spec_first_target_and_proposal_batch` and `run_spec_verify_batch` are counted by native gates; TP1/4K, TP4/4K, and TP4/16K authorities require matching callback/forward counts and exact baseline parity | Established within those cells; native TP4/32K batch-1 parity remains failed |
| Accepted KV commit without per-token target replay | Generic authorities require zero accepted-prefix replay; the native gate separately requires zero ordinary accepted-prefix target replay and exact first-target/verify forward counts | Established for cited n-gram authorities; native MTP only locally verified |
| Rejected suffix rollback and exactly-once ownership | `tools/test_speculative_kv_transaction.py`, proposal finalize prepare/commit/rollback, side-state and residency rollback, physical proposal-slot release | Strong code/test coverage; native real-GPU evidence is pending |
| Two model structures | Qwen3 Transformer and Qwen3.5 hybrid recurrent/full-attention authorities | `ESTABLISHED` for generic n-gram, not for learned drafting |
| TP1 and TP4 | Qwen3.5 TP1 plus Qwen3/Qwen3.5 TP4 authorities | `ESTABLISHED` for generic n-gram |
| 4K, 16K, and 32K | Artifact matrix below | `ESTABLISHED` for generic n-gram correctness |
| Batch 1 and batch 4 | Every cited correctness authority contains `baseline:b1`, `baseline:b4`, `ngram:b1`, and `ngram:b4` | `ESTABLISHED` for generic n-gram |
| Exact greedy parity | Every cited correctness result records `parity={"b1": true, "b4": true}` | `ESTABLISHED` within cited cells |
| TPOT, TTFT, throughput, memory, KV bytes, acceptance | Qwen3.5 TP4/16K controlled performance authority records all required metrics and real Engine movement counters | `MEASURED / POSITIVE` for n-gram only |
| No simulated KV-copy promotion claim | 16K/32K authorities use production `engine.kv_offload_summaries` with non-zero H2D/D2H/copy-wait/eviction evidence | Satisfied for cited generic long-context authorities |
| KV8/KV4 and heat-tier policy | Separate primitive and research gates exist | `MISSING` unified promotion authority |
| CUDA Graph variable proposal length | Local graph/backend tests exist | `MISSING` promotion-grade cross-model matrix |
| Verifier/sampling/KV-commit fusion and TP collective optimization | No complete controlled authority covering these items | `MISSING` |

### Re-inspected generic authority matrix

The first-model Qwen3 TP4 authority is at the following corrected path:

```text
artifacts/generic_speculative_tp4/
  tp4-opaque-48d18e4aba16756d/
    authority/result.json
```

An attempted re-audit lookup containing an additional
`artifacts/authority/` component did not exist. The authoritative result
above records:

```text
schema version:  1
classification:  NOT_PROMOTABLE
world size:      4
cells:           baseline:b1, baseline:b4, ngram:b1, ngram:b4
parity:          b1=true, b4=true
```

All inspected Qwen3 verifier files report `PASS` with no failures:

```text
authority/verify.json
authority/verify.local.json
authority/verify.local.final.json
authority/verify.local.fresh.json
verify.remote.json
```

The second-model Qwen3.5 authorities were present at:

```text
TP1:
  artifacts/qwen35_generic_speculative_tp1/
    opaque-d4e74cb46fccbc57319c3c4f/
    artifacts/authority/result.json
  classification: SECOND_MODEL_TP1_ESTABLISHED
  verifier:       no failures

TP4 / 4K:
  artifacts/qwen35_generic_speculative_tp4/
    opaque-24f8ae471a2ba439ecb5a3b1/
    artifacts/authority/result.json
  classification: SECOND_MODEL_TP4_4K_ESTABLISHED
  verifier:       PASS

TP4 / 16K:
  artifacts/qwen35_generic_speculative_tp4_16k/
    opaque-3b8050a916f037bc92412ea5/
    artifacts/authority/result.json
  classification: SECOND_MODEL_TP4_16K_ESTABLISHED
  verifier:       PASS

TP4 / 32K:
  artifacts/qwen35_generic_speculative_tp4_32k/
    opaque-03a0a96654a14441b314800f/
    artifacts/authority/result.json
  classification: SECOND_MODEL_TP4_32K_ESTABLISHED
  verifier:       PASS
```

Each correctness result contains the four batch/policy cells and exact
greedy parity for batch 1 and batch 4.

The controlled Qwen3.5 TP4/16K performance authority was also re-read:

```text
artifacts/qwen35_generic_speculative_tp4_16k_performance/
  opaque-c9807d19e6402acc22d4a615/
  artifacts/authority/result.json

classification:
  SECOND_MODEL_TP4_16K_PERFORMANCE_MEASURED

campaign direction:
  POSITIVE

independent verifier:
  PASS
```

The recorded candidate/baseline ratios are:

```text
batch 1:
  TPOT:               0.5677486644
  request throughput: 1.4279094053
  TTFT:               1.1405970670
  H2D bytes:          0.5408970976

batch 4:
  TPOT:               0.5443779098
  request throughput: 1.7823893148
  TTFT:               1.0609167401
  H2D bytes:          0.5396825397
```

This is positive n-gram evidence. It is not native-MTP or independent
learned-draft-model performance evidence.

### Native-MTP authority status

The frozen native authority scope is:

```text
model structure:       Qwen3.5 hybrid
proposal source:       native learned MTP checkpoint
tensor parallel size:  1
prompt tokens:         4096
batch sizes:           1 and 4
output tokens:         32
proposal limit:        4
execution surface:     production LLMEngine.step()
required parity:       exact greedy
promotion result:      always NOT_PROMOTABLE
```

Local gate hardening now verifies:

```text
first_target_target_forwards == first_target_callbacks
verify_target_forwards       == verify_callbacks
accepted_prefix_target_replays == 0

proposal finalize:
  prepare -> commit

side state:
  prepare -> select -> apply -> seal

proposal KV:
  accepted slots committed
  rejected slots released

finished sequence:
  proposal sequence released after all seals

cleanup:
  zero proposal transactions
  zero finalize tickets
  zero proposal sequence IDs
  zero physical proposal slots
  no poisoned runtime
```

The focused and related regression suite records:

```text
208 passed
Python compilation: PASS
remote runner syntax: PASS
scoped diff validation: PASS
```

Those are readiness signals, not authority evidence. The only file under the
native authority artifact parent is:

```text
artifacts/qwen35_native_mtp_tp1_4k_engine/
  opaque-d74aec38372fbd8ac7ce5354/
    cell-order.txt
```

There is no `result.json`, source manifest, completed cell output, or
independent verifier result. This directory must not be treated as an
authority.

The remote runner now fails before creating another local run directory when
either prerequisite is unavailable:

```text
Kerberos ticket preflight:
  klist -t -c FILE:/Users/bytedance/krb5cc_sitian
  expired cache -> exit 2, no artifact

SSH route preflight:
  retry_remote_command "true"
  unreachable route -> exit 255, no artifact
```

Authority source integrity is also tightened. The manifest no longer hashes
only 10 selected files while executing the complete package. Its default
inventory is the deterministic union of:

```text
every tinyvllm/**/*.py file
the native-MTP gate
the native-MTP worker
the independent verifier

current inventory:
  106 / 106 files present and hashed
```

An isolated reconstruction of the exact runner tar verified all three CLIs
and recomputed all 106 hashes successfully:

```text
ISOLATED_FULL_SOURCE_BUNDLE_OK
```

Therefore changes to scheduler, block manager, sequence ownership, model
layers, KV paths, and other executed Python modules are now bound into the
authority source-tree digest and independently rechecked.

The verifier also rejects any manifest whose source-file key set differs
from the complete default inventory. A deliberately reduced one-file
manifest is now rejected with:

```text
classification: FAIL
failure:        source file inventory mismatch
```

This closes the remaining loophole where a producer could hash a valid but
selectively incomplete inventory.

The campaign also binds the source snapshot temporally:

```text
before worker execution:
  hash all 106 source files once

after all cells and checkpoint identity reads:
  recompute all 106 hashes

required:
  final hash mapping == pre-execution hash mapping

result digest:
  derived from the pre-execution snapshot
```

A focused RED/GREEN test changes one bound file after the first worker cell.
The campaign now rejects it with `source changed during campaign`. It can no
longer publish a digest for post-run code that was not consistently executed
by all cells.

Offline independent verification no longer trusts the aggregate tree digest
without recomputation. It always derives a tree hash from the 106 manifest
digest entries and compares it with `result.source_tree_sha256`. A
valid-format per-file digest mutation is rejected without requiring access
to the source directory:

```text
classification: FAIL
failure:        source tree digest mismatch
```

Supplying `source_root` adds per-file content verification on top of this
offline internal-consistency check.

Manifest semantics are now strict rather than permissive:

```text
schema_version:
  must equal the frozen authority schema

artifact inventory:
  must contain exactly result.json and status.json
```

A schema mutation and an undeclared extra artifact entry both previously
verified as `PASS`; focused RED/GREEN tests now require `FAIL`.

The raw `result.json` must also equal the canonical value returned by
`validate_result()`. An appended undeclared field such as
`undeclared_claim="production_ready"` previously survived artifact-digest
refresh and verified as `PASS`; it is now rejected with
`result is not canonical`.

The source manifest top-level key set is also canonical. An added
`undeclared_claim="production_ready"` now fails with
`source manifest is not canonical` rather than being ignored.

The current fixed Kerberos cache is expired and cannot be renewed with
`kinit -R`. SSH to `sitian@10.232.195.203` closes before the target command.
Therefore the real checkpoint/GPU campaign has not started.

### Completion verdict

```text
generic n-gram runtime:
  two model structures: ESTABLISHED
  TP1 and TP4:          ESTABLISHED
  4K/16K/32K:          ESTABLISHED
  batch 1/4:           ESTABLISHED
  exact greedy parity: ESTABLISHED
  real KV movement:    ESTABLISHED
  controlled metrics:  POSITIVE for Qwen3.5 TP4/16K

native learned MTP:
  local Engine gate:   READY
  real TP1/4K result:  MISSING
  TP4/long-context:    MISSING
  performance:         MISSING

independent learned draft model:
  real authority:      MISSING

KV8/KV4 promotion:
  MISSING

Phase 1:
  NOT ACHIEVED
  NOT_PROMOTABLE
```

The next valid evidence-producing action remains the real Qwen3.5 native-MTP
TP1/4K campaign after obtaining a fresh interactive Kerberos TGT. No broader
native-MTP, KV-offload, KV8/KV4, production-readiness, or Phase 1 completion
claim is permitted before that authority independently verifies.

## 2026-08-14 Superseding Native-MTP TP4/16K Target-KV-Offload Update

The earlier native-MTP status above is stale. Qwen3.5 native MTP now has a
completed production-Engine TP4/16K target-KV-offload authority:

```text
run:
  lifecycle-release-fix-20260814-2

authority:
  artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/
    lifecycle-release-fix-20260814-2/artifacts/authority

schema:
  qwen35.native-mtp-tp4-16k-target-kv-offload.v1

classification:
  QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED

promotion:
  NOT_PROMOTABLE

remote independent verifier:
  PASS, failures=[]

runner local verifier:
  PASS, failures=[]

fresh independent local verifier:
  PASS, failures=[]
```

Exact greedy parity is established for both frozen batch sizes:

```text
baseline_native.b1: true
baseline_native.b4: true
```

The native batch-4 lifecycle inventory is complete on every TP rank:

```text
release rows:
  sequence 0, epoch 0
  sequence 1, epoch 0
  sequence 2, epoch 0
  sequence 3, epoch 0

active proposal transactions after completion: 0
allocated physical proposal slots:             0
prepared proposal tickets:                     0
bootstrapped sequences:                        0
runtime poisoned:                              false
```

The authority records real target-KV movement rather than simulated copies.
For every TP rank in native batch 4:

```text
logical blocks:          640
GPU blocks:               68
peak resident blocks:     68
D2H copies:              273
D2H bytes:        1717567488
H2D copies:             6762
H2D bytes:       42542825472
evictions:              6954
copy waits:             2841
movement provenance:
  engine.kv_offload_summaries
```

Native batch-4 speculative execution records:

```text
proposal rows:                    10
proposed tokens:                  36
accepted draft tokens:            19
rejected draft tokens:            17
first-target callbacks:            3
first-target target forwards:      3
verify callbacks:                  3
verify target forwards:            5
accepted-prefix target replays:    0
```

All four target-KV receipts contain `prepare -> commit`; all four side-state
receipts contain `prepare -> select -> apply -> seal`. Engine exit,
process-group destruction, shared-memory release, rank exit codes, and owned
child cleanup all pass.

The source and checkpoint binding is:

```text
target model manifest:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0

MTP checkpoint manifest:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b

source tree:
  32c08ac67057cbad6317cc837da489323be4043a12ccd1efeedffbcf4b11c804
```

The final defect was an Engine lifecycle asymmetry. A sequence that finished
in an ordinary-only speculative-suppressed step released the host draft
lifecycle but did not dispatch
`release_speculative_proposal_sequence` to the ModelRunner proposal
executor. The batch-4 authority therefore observed an incomplete release-row
inventory. A focused RED test reproduced the missing dispatch; the
ordinary-only finish path now mirrors the already-correct speculative finish
path and preserves the same poison-on-release-failure semantics.

One earlier retry,
`lifecycle-release-fix-20260814-1`, is not authority. A new unrelated process
occupied a selected shared GPU after preflight, leaving only 1.05 GiB free
and causing a 2 GiB bootstrap allocation to fail. No process was terminated;
the second run re-executed the idle-GPU gate and completed on a fresh
selection.

Fresh local regression evidence after the lifecycle fix:

```text
tools/test_engine_speculative_runtime.py:
  60 passed

tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
tools/test_model_runner_spec_verify.py:
  160 passed

py_compile:
  PASS

scoped git diff --check:
  PASS
```

### Updated Phase 1 Verdict

```text
generic n-gram runtime:
  broad two-model correctness matrix: ESTABLISHED

Qwen3.5 native learned MTP:
  TP1/4K production Engine:            ESTABLISHED
  TP4/4K production Engine:            ESTABLISHED
  TP4/16K target-KV offload:           ESTABLISHED
  TP4/32K target-KV offload:           MISSING
  controlled native-MTP performance:  MISSING

second learned draft structure:
  LOCAL IMPLEMENTATION PRESENT
  TP1 REAL-CHECKPOINT CORRECTNESS NOT ESTABLISHED
  TP4 INTENTIONALLY FAIL-CLOSED

proposal-KV offload:
  NOT ESTABLISHED

KV8/KV4 promotion:
  MISSING

Phase 1:
  NOT ACHIEVED
  NOT_PROMOTABLE
```

This authority permits the narrow TP4/16K native-MTP target-KV-offload
correctness claim only. It does not establish proposal-KV offload, TP1/16K,
32K, performance improvement, a second learned structure, KV8/KV4,
production readiness, or Phase 1 completion.

## Qwen3.5 Native MTP TP4/32K Target-KV-Offload Attempt

The independent TP4/32K overlay and bounded remote runner are complete, but
the production-Engine correctness authority is not established. The
classification
`QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED` must not be emitted.

Frozen contract:

```text
schema:
  qwen35.native-mtp-tp4-32k-target-kv-offload.v1

prompt/output tokens:             32768 / 8
max proposal tokens:                     4
world size:                              4
batch sizes:                          1, 4
max model length:                    33024
max batched tokens:                132096
max prefill tokens per step:          1024
target-KV GPU/logical blocks:        68/640
target-KV block size:                   256
blockwise blocks:                         8
```

Two attempts are retained:

```text
native-mtp-tp4-32k-20260814-1:
  non-authority infrastructure failure
  missing CLI main dispatch
  verifier classification: FAIL
  verifier failure: result is missing

native-mtp-tp4-32k-20260814-2:
  complete real four-cell execution
  retained under artifacts/authority.failed
  terminal failure:
    baseline/native output parity mismatch for batch 1
```

The CLI-dispatch defect from the first attempt is fixed and covered by a
subprocess `--help` regression. It does not alter the correctness result of
the second attempt.

Exact output evidence from the second attempt:

```text
batch 1 baseline:
  [220, 15, 15, 15, 15, 15, 15, 15]

batch 1 native MTP:
  [220, 15, 15, 220, 15, 15, 220, 15]

different indices:
  3, 6

batch 4:
  exact baseline/native parity for all four prompts
```

The first prompt's batch-4 baseline output is
`[220, 15, 15, 220, 15, 15, 220, 15]`, matching native batch 1 rather than
baseline batch 1. This is evidence of a batch/query-shape-sensitive boundary,
not a proven root cause.

The first native batch-1 proposal is:

```text
proposal:
  [15, 15, 2658, 8381]

accepted proposal tokens:
  2

fallback emitted:
  220

serial baseline expected:
  15
```

Every TP rank reports the same native batch-1 speculative counters:

```text
proposal rows:                    3
proposed tokens:                 10
accepted draft tokens:           5
rejected draft tokens:           5
accepted-prefix target replays:   0
```

Real target-KV movement is present in every cell, with provenance
`engine.kv_offload_summaries`. Aggregate four-rank movement is:

```text
baseline batch 1:
  H2D 10492 /  66009956352 bytes
  D2H   540 /   3397386240 bytes

native batch 1:
  H2D  9028 /  56799264768 bytes
  D2H   532 /   3347054592 bytes

baseline batch 4:
  H2D 86016 / 541165879296 bytes
  D2H  2160 /  13589544960 bytes

native batch 4:
  H2D 60636 / 381488726016 bytes
  D2H  2120 /  13337886720 bytes
```

The movement proves that the production offload path operated under the
frozen 68/640 block budget. It cannot be used to promote a result that fails
exact greedy parity.

All four cells completed process-group destruction, shared-memory release,
owned-child cleanup, and zero rank exit codes. Native receipts include
target-KV `prepare -> commit` and side-state
`prepare -> select -> apply -> seal`. Proposal transactions, physical slots,
prepared tickets, and bootstrapped sequence counts return to zero; release
rows are complete and the runtime is not poisoned.

Checkpoint binding:

```text
target model manifest:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0

MTP checkpoint manifest:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b
```

The current evidence is consistent with a single-sequence multi-token
speculative-verify target/state boundary after an accepted prefix. It does
not prove the exact root cause. No production verify behavior was changed.
A follow-up requires a separately reviewed design, with internal batch-1
verify-tail serialization as the leading investigation while retaining the
frozen proposal limit of 4, exact parity, the same prompt, and zero
accepted-prefix replay.

### Updated Phase 1 Verdict After TP4/32K Attempt

```text
generic n-gram runtime:
  broad two-model correctness matrix: ESTABLISHED

Qwen3.5 native learned MTP:
  TP1/4K production Engine:            ESTABLISHED
  TP4/4K production Engine:            ESTABLISHED
  TP4/16K target-KV offload:           ESTABLISHED
  TP4/32K target-KV offload:           NOT ESTABLISHED
  TP4/32K batch-4 parity:              PASS
  TP4/32K batch-1 parity:              FAIL
  controlled native-MTP performance:  MISSING

second learned draft structure:
  MISSING

proposal-KV offload:
  NOT ESTABLISHED

KV8/KV4 promotion:
  MISSING

Phase 1:
  NOT ACHIEVED
  NOT_PROMOTABLE
```

No claim is made for TP4/32K correctness authority, performance improvement,
proposal-KV offload, KV8/KV4, a second learned structure, production
readiness, Phase 1 completion, or promotion.

## 2026-08-14 Independent Qwen3 Draft and TP4 Design Re-Audit

The independent learned-draft row is no longer accurately described as a
schema-only or ABI-only placeholder. The current worktree contains a concrete
Qwen3 dense autoregressive drafter:

```text
tinyvllm/engine/autoregressive_draft_executor.py
tinyvllm/engine/qwen3_draft_backend.py
tinyvllm/engine/autoregressive_draft_registration.py
tools/autoregressive_draft_tp1_engine_gate.py
```

The implementation provides:

```text
batch-native prompt bootstrap and exact-Q proposal execution
real Qwen3 model prefill/decode forward calls
multi-layer block-size-one physical proposal K/V
accepted-prefix commit and rejected-suffix rollback/release
checkpoint composite fingerprints
ordered tokenizer and special-token compatibility
source-neutral ModelRunner executor registration
real-forward and proposal-KV authority counters
preflight-only checkpoint/tokenizer/prompt/capacity validation
```

The preflight path explicitly reports
`correctness_established=false`; it is not correctness or performance
evidence.

The latest recorded dependency-light validation in
`AGENT_HANDOFF_STATE.md` is:

```text
TP1 gate suite:                  15 passed
full dependency-light matrix:  296 passed
py_compile:                     PASS
source-neutral static search:  PASS
git diff --check:              PASS
```

A fresh local focused rerun used an offline temporary uv environment backed
only by packages already present in the local uv cache:

```text
uv run --offline --python 3.12 --with pytest --with torch pytest -q \
  tools/test_autoregressive_draft_executor.py \
  tools/test_qwen3_draft_backend.py \
  tools/test_autoregressive_draft_registration.py \
  tools/test_autoregressive_draft_model_runner_integration.py \
  tools/test_autoregressive_draft_tp1_engine_gate.py

88 passed in 14.71s
```

The bare shell still has no `pytest` executable, and `/usr/bin/python3` has
no `torch`; the offline uv command is the fresh authority for this focused
local matrix. These tests establish dependency-light contracts only. They do
not replace a real checkpoint GPU gate.

The real TP1 learned-draft gate has not run against an attributable
Qwen3-draft/Qwen3.5-target checkpoint pair. No remote diagnostic or GPU
command was launched during this audit. Therefore:

```text
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
SECOND_LEARNED_STRUCTURE_TP1_CORRECTNESS=NOT_ESTABLISHED
```

TP4 remains deliberately rejected by:

```text
autoregressive draft executor currently requires TP1
autoregressive draft currently requires TP1
```

The reviewed TP4 design candidate keeps Qwen3 weights and proposal K/V
TP-sharded and rank-local, materializes full-vocabulary logits only on rank
zero, broadcasts selected token IDs, synchronizes logical transaction
authority rather than physical slot IDs, and publishes the executor only
after all-rank private construction succeeds. The design has been presented
for approval but has not been written as an approved spec or implemented.

### Current promotion checklist

| Requirement | Current evidence | Verdict |
| --- | --- | --- |
| Generic source-neutral runtime | n-gram, SAM contract, native MTP, and independent-draft executor boundary | `ESTABLISHED` as code/test architecture |
| Native learned MTP TP1/4K | Completed production Engine authority | `ESTABLISHED` |
| Native learned MTP TP4/4K | `artifacts/qwen35_native_mtp_tp4_4k_engine/opaque-95aa0889f8365beac8be2b6f/artifacts/authority/result.json` records `QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED`; local and remote verifier PASS | `ESTABLISHED` within 4K batch 1/4 eager scope |
| Native learned MTP TP4/16K target-KV offload | `artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/lifecycle-release-fix-20260814-2/artifacts/authority/result.json` records `QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED`; local and remote verifier PASS | `ESTABLISHED` for target-KV offload and exact batch 1/4 parity |
| Native learned MTP TP4/32K target-KV offload | `native-mtp-tp4-32k-20260814-2` failed authority assembly on baseline/native batch-1 output parity | `NOT ESTABLISHED` |
| Independent Qwen3 draft TP1 | Concrete implementation and fail-closed gate harness; no real checkpoint result | `NOT ESTABLISHED` |
| Independent Qwen3 draft TP4 | Design candidate only; runtime still rejects TP4 before dependency calls | `NOT ESTABLISHED` |
| Controlled native learned-source performance | No complete TPOT/TTFT/throughput/memory comparison | `MISSING` |
| Proposal-KV offload | Proposal K/V remains GPU local in current learned-source implementations | `NOT ESTABLISHED` |
| KV8/KV4 promotion matrix | No unified learned-source/offload authority | `MISSING` |

The Phase 1 verdict remains:

```text
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## 2026-08-14 Superseding TP4 Independent-Draft and 32K Diagnostic Audit

The preceding tail section is stale in two respects:

1. the independent Qwen3 draft runtime now has a completed local TP4 sharded
   executor contract rather than a TP1-only design candidate; and
2. the TP4/32K paired verify trace is implemented and locally verified rather
   than design-only.

Fresh evidence for the paired trace:

```text
focused trace/state/worker matrix:
  50 passed in 6.59s

complete local regression matrix:
  294 passed in 9.53s

py_compile:
  PASS

remote-runner bash syntax:
  PASS

scoped git diff --check:
  PASS
```

The complete matrix covered:

```text
generic immutable trace rows and deterministic top-five compaction
Qwen3.5 checkpoint fingerprint stability and non-mutation
32K semantic pairing, lineage validation, and first-divergence selection
16K authority compatibility
generic 32K authority compatibility
Engine speculative-runtime behavior
ModelRunner first-target, verify-tail, ordinary-decode, and lifecycle wiring
```

Source binding is not inferred from test names. The 16K frozen gate builds its
default manifest by recursively including every `tinyvllm/**/*.py` file, so
both trace helpers are bound; the 32K overlay adds its own gate, worker, and
verifier. The ordinary authority schema remains
`qwen35.native-mtp-tp4-32k-target-kv-offload.v1`.

Updated local classification:

```text
independent Qwen3 draft TP4 local sharded contract:
  ESTABLISHED

independent Qwen3 draft TP4 real-checkpoint authority:
  NOT ESTABLISHED

TP4/32K paired verify trace local implementation:
  ESTABLISHED

TP4/32K paired verify trace default-off and source-bound contracts:
  ESTABLISHED

TP4/32K first-divergence artifact:
  NOT ESTABLISHED

TP4/32K exact root cause:
  NOT ESTABLISHED

TP4/32K exact greedy Engine parity:
  NOT ESTABLISHED

Phase 1:
  NOT ACHIEVED
  NOT_PROMOTABLE
```

The next evidence-producing action requires separate remote/GPU
authorization: run the diagnostic-only four-cell paired trace and retain its
source-bound first-divergence artifact. It must not be treated as a 32K
authority rerun or as a fix.
