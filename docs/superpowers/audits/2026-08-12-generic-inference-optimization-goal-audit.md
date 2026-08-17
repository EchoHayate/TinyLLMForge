# Generic Inference Optimization Goal Audit

> Current canonical phase-1 coverage and blocker status is maintained in
> `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`. This
> longer audit contains historical intermediate conclusions; where the two
> documents differ, use the phase-1 coverage audit.

**Date:** 2026-08-12  
**Repository:** `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`  
**Overall classification:** `NOT_PROMOTABLE`

## Scope

This audit maps the three model-independent optimization directions to the
current production code, tests, experiment surfaces, and promotion gates:

1. generic MTP/speculative runtime with transactional KV ownership;
2. residency-aware tiered KV cache with real offload and KV4/KV8 storage;
3. chunked/blockwise execution with production batch scheduling.

The audit intentionally excludes Qwen3.5-specific projection/layout work from
the forward plan. Existing model-specific artifacts remain useful evidence,
but they are not substitutes for model-independent runtime integration.

## Status Legend

- `VERIFIED`: implemented and covered by fresh focused validation in the
  current worktree.
- `PARTIAL`: implementation exists, but it is profiler-only, single-sequence,
  default-off, historically validated, or missing a required integration.
- `MISSING`: the required production behavior or promotion evidence does not
  exist.

## Direction 1: Generic MTP/Speculative Runtime

| Requirement | Status | Current evidence | Remaining gap |
| --- | --- | --- | --- |
| Transactional speculative KV ownership | `VERIFIED` | `tinyvllm/engine/block_manager.py`; `tinyvllm/engine/llm_engine.py`; `tools/test_speculative_kv_transaction.py`; `tools/test_engine_speculative_runtime.py`. Prepared execution keeps reservations private, token-free KV commit plans validate the whole batch, and `LLMEngine.step()` commits KV ownership before one rollback-safe Scheduler metadata commit. | GPU parity and speculative-residency integration remain unverified. |
| Accepted KV direct commit | `VERIFIED` | `commit_speculative_kv_transaction()` attaches only accepted materialized blocks; native profiler assertions keep accepted-KV rematerialization/copy/replay at zero. | Must preserve this invariant after scheduler batching and real draft adapters are connected. |
| Rejected suffix rollback | `VERIFIED` | `rollback_speculative_kv_transaction()` plus zero/partial/full acceptance and phase-failure tests. | Batch cancellation and mixed per-sequence success/failure are not covered. |
| Model-independent verify/accept/commit orchestration | `VERIFIED` locally and on one TP1 loaded-model gate | `tinyvllm/speculative/runtime.py`; `tinyvllm/speculative/batch_runtime.py`; `tinyvllm/engine/speculative_runtime.py`; `tinyvllm/engine/llm_engine.py`; focused engine/runtime tests; `artifacts/speculative_tp1_parity/20260812T051123Z/result.json`. `LLMEngine.step()` executes selected rows through prepare, prevalidated KV commit, prepared Scheduler commit, rollback, and optional post-commit lifecycle synchronization. | TP4, a second model structure, stochastic sampling, cancellation, and long-context execution remain unverified. |
| Runtime-facing draft adapter contract | `PARTIAL` | `tinyvllm/speculative/adapter.py`; `tinyvllm/speculative/ngram_adapter.py`; `tinyvllm/speculative/sam_adapter.py`; `tinyvllm/engine/speculative_runtime.py`; adapter and engine/runtime tests. `LLMEngine.activate_speculative_runtime()` atomically derives and installs the exact Scheduler selection config plus runtime, with idempotence, conflict rejection, and rollback tests. The n-gram adapter passed one real loaded-model TP1 gate. | Learned-model and MTP adapters are absent; SAM has no loaded-model gate; TP4 and multi-model coverage are missing. |
| Batch-native callback orchestration | `VERIFIED` locally and for one TP1 loaded-model case | `tinyvllm/speculative/batch_runtime.py`; `tinyvllm/engine/speculative_model_runner.py`; `tinyvllm/engine/llm_engine.py`; focused first-target/fixed-Q and engine/runtime tests; TP1 parity artifact. `LLMEngine` invokes one batched first-target callback and stable distinct-fixed-Q verifier groups without padding. | Stateful non-KV models fail closed pending transactional recurrent/convolution state; TP4 and broader GPU shapes are unverified. |
| Native profiler integration | `VERIFIED` | `tools/profile_ngram_commit.py`; `tools/test_ngram_speculative.py` | This remains a single-sequence experimental surface, not serving runtime behavior. |
| N-gram and SAM proposal sources | `VERIFIED` locally through the generic runtime contract; n-gram also verified on one loaded TP1 case | `tinyvllm/speculative/ngram_adapter.py`, `tinyvllm/speculative/sam_adapter.py`, source-adapter tests, engine/runtime lifecycle tests, atomic activation tests, and `artifacts/speculative_tp1_parity/20260812T051123Z/result.json`. | SAM loaded-model execution, learned drafter/MTP sources, TP4, and broader model coverage are absent. |
| Learned drafter adapter | `PARTIAL` | `tools/draft_model_schema.py` defines a profiler-only schema and deterministic stub | No loaded learned draft model, lifecycle, device execution, or scheduler adapter exists. |
| MTP adapter | `MISSING` | No production MTP proposal implementation found | Add a capability-based adapter without model-name branches. |
| Batched speculative scheduling | `VERIFIED` locally for greedy KV-only rows and one loaded TP1 case | `tinyvllm/engine/speculative_selection.py`, `tinyvllm/engine/speculative_execution.py`, `tinyvllm/engine/scheduler.py`, `tinyvllm/engine/llm_engine.py`, atomic activation tests, and the TP1 parity artifact cover immutable selection snapshots, selected/suppressed partitioning, stale-state rejection, non-greedy suppression, variable proposal lengths, per-sequence transactions, prepared metadata commit, mixed ordinary/speculative execution, release transport, and rollback. | TP4, cancellation, multi-model coverage, and production observability remain missing. |
| Engine output commit planning | `VERIFIED` locally and on one loaded TP1 exact-token gate | `tinyvllm/engine/speculative_execution.py`, `tinyvllm/engine/scheduler.py`, `tinyvllm/engine/llm_engine.py`, focused tests, and the TP1 parity artifact validate exact ordered IDs and proposal prefixes, then atomically apply empty, zero, partial, and full acceptance rows with EOS, output-budget handling, mixed-row accounting, and snapshot rollback. | Cancellation and broader serving/model/TP matrices remain unverified. |
| Real GPU/TP validation | `PARTIAL` | `artifacts/speculative_tp1_parity/20260812T051123Z/result.json` and `verify.remote.json` prove exact greedy output parity for Qwen3-0.6B BF16 TP1 on two prompts after independent source-hash verification. | Validate a second materially different model, TP4, longer contexts, larger batches, cancellation, and repeated controlled runs. |

### Direction 1 conclusion

The local source-agnostic control-plane is complete for greedy KV-only rows:
selection, selected/suppressed execution, first-target/fixed-Q verification,
transactional KV ownership, atomic Scheduler metadata commit, rollback, and
optional SAM lifecycle synchronization are connected through
`LLMEngine.step()`.
The feature is still not promotable because activation is a manual two-call
protocol, no real loaded-model TP1 parity artifact exists, learned-drafter and
MTP adapters are absent, and stateful non-KV models remain excluded until
their mutable state has commit/rollback semantics.

## Direction 2: Tiered KV Residency, Offload, and Quantized Storage

| Requirement | Status | Current evidence | Remaining gap |
| --- | --- | --- | --- |
| Logical KV block to GPU slot mapping | `PARTIAL` | `KVOffloadMVP0.logical_to_slot` in `tinyvllm/engine/model_runner.py` | Default-off research path; no promotion matrix in the current worktree state. |
| Real pinned-CPU backing and H2D/D2H movement | `PARTIAL` | `KVOffloadMVP0`, async/batched copies, events, dirty tracking, writeback, and `h2d_bytes` statistics | Historical smokes are not a current two-model promotion campaign. |
| Read/write residency planning | `PARTIAL` | `ensure_resident()`, prefetch counters, protected write blocks, decode window plans | Speculative reserved/accepted/rejected blocks are not yet coordinated with residency ownership. |
| Context larger than GPU staging slots | `PARTIAL` | Exact blockwise online decode/prefill attention in `tinyvllm/layers/attention.py` | Needs current long-context, multi-sequence, TP1/TP4 performance and parity evidence. |
| KV4 storage and dequantization | `PARTIAL` | Packed INT4 cache allocation and attention dequantization paths | KV offload blockwise paths explicitly require unquantized KV. |
| KV8 storage and dequantization | `PARTIAL` | INT8 cache allocation and attention dequantization paths | No unified offload-plus-KV8 tiering policy or promotion evidence. |
| Hot/warm/cold KV tier policy | `MISSING` | LRU/cost and future-hint mechanisms exist, but not a general precision/residency tier state machine | Define ownership, transitions, precision, eviction, and recovery semantics. |
| Speculative KV residency integration | `MISSING` | Transaction API tracks allocator ownership only | Reserve/materialize/commit/rollback must update or invalidate residency state exactly once. |
| Real KV H2D reduction gate | `MISSING` | Runtime exposes real offload counters, while profiler also contains a separate simulated upload option | Promotion must use `KVOffloadMVP0` bytes/events only; simulated copies are non-evidence. |

### Direction 2 conclusion

The repository contains real KV offload and independent KV4/KV8 mechanisms,
but they are separate feature families. It does not yet contain a unified
tiered KV runtime that combines speculative ownership, residency, precision
transitions, and batch scheduling.

## Direction 3: Chunked/Blockwise Batch Execution

| Requirement | Status | Current evidence | Remaining gap |
| --- | --- | --- | --- |
| Chunked prefill scheduling | `VERIFIED` at CPU contract level | `tinyvllm/engine/scheduler.py`; `tools/test_chunked_prefill.py` | Must be exercised together with speculative batches and tiered KV. |
| Mixed prefill/decode scheduling | `PARTIAL` | Standard, SLO, adaptive, and mixed scheduling paths exist | No explicit speculative work class, proposal budget, or transaction lifecycle. |
| Exact blockwise decode attention | `PARTIAL` | `_blockwise_online_decode_attention()` with online-softmax merge | Current promotion matrix is absent. |
| Exact blockwise prefill attention | `PARTIAL` | `_blockwise_online_prefill_attention()` streams historical KV and merges local causal attention | Current promotion matrix is absent. |
| Residency-aware window planning | `PARTIAL` | Directional windows, cache identity, future hints, cross-layer reuse, and staging helper | Evidence is feature-specific rather than end-to-end across required models/shapes. |
| Multi-sequence speculative batch | `VERIFIED` locally for greedy KV-only rows | Scheduler publishes a source-agnostic selected/suppressed record; `LLMEngine` validates and partitions it, executes selected rows through generic batch callbacks, preserves ordinary suppressed execution, commits KV before one prepared metadata commit, and rolls back injected KV/Scheduler failures. The first-target path performs one multi-row decode forward, and proposal tails use one verifier RPC per distinct fixed Q. | Non-greedy selection and stateful non-KV transactions remain fail closed; real loaded-model, GPU, and TP evidence is missing. |
| Unified batch observability | `MISSING` | Individual profiler/offload/scheduler counters exist | Add per-request acceptance, transaction, TTFT, TPOT, throughput, memory, and real KV movement accounting. |

### Direction 3 conclusion

Chunked prefill and blockwise attention provide the execution substrate for
long contexts. The local engine now has a production-shaped speculative
control-plane for selected and suppressed rows, including callback execution
and atomic metadata commit. The immediate critical path has moved to atomic
activation and real loaded-model parity, followed by residency integration
and end-to-end observability.

## Prompt-to-Artifact Checklist

| Requested outcome | Current artifact | Status | Promotion boundary |
| --- | --- | --- | --- |
| Generic speculative/MTP runtime | Source-agnostic selection, atomic activation, runtime, callback bridge, transactional engine commit, rollback, and lifecycle contracts | `PARTIAL` | N-gram control-plane has one loaded TP1 parity gate; learned drafter, MTP adapter, TP4, and multi-model coverage are missing. |
| Accepted KV reuse without replay/copy | Token-free speculative KV batch commit with accepted-prefix ownership transfer | `VERIFIED` locally | Must remain zero-rematerialization under real GPU/TP execution. |
| Rejected suffix rollback | Per-sequence reservation/materialization rollback plus injected phase-failure tests | `VERIFIED` locally | Real serving cancellation and residency invalidation remain unverified. |
| Longer context through tiered KV | Real pinned-host offload and independent KV4/KV8/blockwise paths | `PARTIAL` | No unified speculative ownership/residency/precision state machine or long-context promotion matrix. |
| Lower TPOT | Fixed-Q grouped verification and accepted-prefix reuse remove redundant target work in principle | `MISSING` as a performance claim | No controlled loaded-model TTFT/TPOT/throughput measurements. |
| Production batch integration | Mixed selected/suppressed `LLMEngine.step()` execution, atomic runtime activation, and rollback-safe Scheduler postprocess | `VERIFIED` locally; one TP1 loaded-model case | No TP4, second-model, long-context, or production observability campaign. |
| Promotion evidence | Unit/regression matrices and source-integrity checks | `MISSING` | Two model structures, TP1/TP4, 4K/16K/32K+, batch 1/4/mixed, exact parity, latency, throughput, memory, real KV movement, and acceptance artifacts are mandatory. |

## Promotion Matrix

All rows below are mandatory. A row is `MISSING` unless a reproducible artifact
contains the complete comparison and independent correctness result.

| Promotion gate | Required matrix | Current status |
| --- | --- | --- |
| Model coverage | At least two materially different model structures | `MISSING` |
| Tensor parallelism | TP1 and TP4 | `MISSING` |
| Context length | 4K, 16K, and 32K+ | `MISSING` |
| Batch shape | Batch 1, batch 4, and multi-sequence mixed execution | `MISSING` |
| Correctness | Exact greedy token parity against non-speculative baseline | `PARTIAL`: one Qwen3-0.6B BF16 TP1 two-prompt artifact is green |
| Speculative semantics | zero/partial/full acceptance, EOS, output budget, cancellation, and failure rollback | `PARTIAL`: core unit coverage exists; production batch coverage does not |
| Latency | TTFT and TPOT, with warmup and repeated controlled runs | `MISSING` |
| Throughput | Request and output-token throughput | `MISSING` |
| Memory | Peak device memory and host KV footprint | `MISSING` |
| Real KV movement | H2D/D2H bytes, copy batches, waits, evictions, and writebacks | `MISSING` as a complete comparison |
| Draft quality | Acceptance rate and accepted tokens per target invocation | `MISSING` as a complete comparison |
| Source integrity | Exact source identity, config, model, hardware, and raw artifacts | `PARTIAL`: the TP1 artifact hashes 15 source files and records command/model/hardware/runtime identity |

No isolated CPU test, callback microbenchmark, model-specific projection
profile, or simulated KV upload may satisfy these promotion gates.

## Immediate Engineering Gate

The atomic activation and first loaded-model TP1 exact-greedy parity gate are
complete. The next implementation gate is:

1. connect speculative reserve/materialize/commit/rollback transitions to
   `KVOffloadMVP0` residency ownership without KV replay or copy;
2. define exactly-once residency invalidation for rejected suffix blocks and
   preserve accepted materialized blocks in place;
3. expose real H2D/D2H bytes, copy batches, waits, evictions, and writebacks
   per speculative request;
4. add failure-injection tests spanning allocator ownership plus residency
   ownership before running new GPU experiments;
5. rerun loaded-model parity with offload enabled before making any movement,
   memory, TPOT, or throughput claim;
6. keep learned-drafter/MTP, TP4, and second-model gates separate rather than
   weakening the residency transaction proof.

The core must remain model-name-free. Model differences belong behind
capability and adapter boundaries.

## Current Decision

`NOT_PROMOTABLE`

The repository now has a reusable transactional KV contract, generic native
batch orchestration, an explicitly installed source-agnostic engine runtime,
Scheduler-visible speculative selection, rollback-safe multi-token Scheduler
postprocessing, and production `LLMEngine.step()` selected/suppressed control
flow. Selected rows prepare private KV transactions through the fixed-Q
ModelRunner callback bridge; suppressed rows retain ordinary execution and
release-event transport. KV ownership commits before one prepared Scheduler
metadata commit, and injected KV or Scheduler failures restore reservations
and host metadata. Optional draft lifecycle state synchronizes only after
target commit; synchronization failure keeps target state authoritative,
poisons the runtime, and blocks later selected work before ModelRunner
execution.

## Fresh Local Verification

```text
Scheduler prepared postprocess plus engine/runtime focused regression:
  116 passed
first-target/fixed-Q verifier plus speculative/engine/serialization regression:
  322 passed
sequence temperature serialization focused regression:
  10 passed
prepared runtime/KV/public API focused regression:
  47 passed
ModelRunner callback bridge focused regression:
  18 passed
greedy selection contract/scheduler focused regression:
  33 passed
first-target/fixed-Q ModelRunner focused regression:
  69 passed
native verifier attention:
  passed; CUDA numerical cases deferred to remote gate
chunked prefill:
  passed
hybrid state Scheduler:
  passed
Scheduler prefill commit hook:
  passed (8 tests)
Python 3.9 and 3.12 py_compile:
  passed
generic source scan and git diff hygiene:
  passed; staged diff empty
```

Strict boundary:

```text
real LLMEngine speculative execution:
  implemented for greedy selected rows with ordinary suppressed rows
greedy KV-only first-target ModelRunner batch:
  implemented
fixed-Q batch-native ModelRunner tail verification:
  implemented
non-KV/hybrid speculative state:
  fail closed pending transaction design
non-greedy speculative decoding:
  scheduler suppresses and stale-checks; ModelRunner also fails closed
TP worker temperature transport:
  schema 15 implemented; old schemas default to greedy 0.0
prepared speculative runtime:
  implemented; reservations/materialization remain private until commit
token-free atomic KV batch commit:
  implemented; no Sequence token append inside BlockManager
variable-Q verifier:
  no padded kernel; variable proposals grouped into distinct fixed-Q RPCs
LLMEngine callback wiring:
  implemented through generic first-target and fixed-Q tail callbacks
multi-token scheduler postprocess:
  implemented with prepared commit and snapshot rollback
SAM post-commit synchronization:
  implemented through the generic optional DraftLifecycle contract
GPU exact-token parity:
  established for one Qwen3-0.6B BF16 TP1 greedy two-prompt gate
GPU performance improvement:
  not established; the single diagnostic run was slower with speculation
overall classification:
  NOT_PROMOTABLE
```

## Atomic Activation and TP1 Parity Result

The atomic activation boundary is now implemented:

```text
LLMEngine.activate_speculative_runtime(runtime)
```

It derives one exact `SpeculativeSelectionConfig` from the runtime and
ModelRunner, installs Scheduler selection and engine runtime as one logical
operation, is idempotent for the installed runtime, rejects conflicts before
mutation, and restores selection/runtime/poison snapshots if either
publication phase fails.

The first real loaded-model run initially failed exact token parity and wrote
the diagnostic artifact:

```text
artifacts/speculative_tp1_parity/20260812T050504Z/result.json
```

The first prompt diverged at generated token index 4. The root cause was an
off-by-one verifier position contract: the first draft input was written to
logical KV slot `L` but evaluated with rotary position `L + 1`. Standard
decode semantics require the input token's position and destination logical
slot to both be `L`. The corrected invariant is:

```text
SpecVerifyPlan.positions == SpecVerifyPlan.logical_slots
```

This is enforced in the plan builder and both single-row and batch ModelRunner
preparation paths.

The corrected real gate passed:

```text
artifact:
  artifacts/speculative_tp1_parity/20260812T051123Z/result.json
independent remote verification:
  artifacts/speculative_tp1_parity/20260812T051123Z/verify.remote.json
model:
  Qwen3-0.6B
hardware:
  NVIDIA A100 80GB PCIe
mode:
  BF16, TP1, temperature 0.0, 32 output tokens, two prompts
exact output parity:
  PASS
proposed tokens:
  58
accepted draft tokens:
  50
acceptance rate:
  0.8620689655
first-target callbacks:
  14
fixed-Q tail callbacks:
  14
accepted tokens per target invocation:
  1.7857142857
source hashes:
  15 files independently revalidated
```

The elapsed times in this one diagnostic run were:

```text
baseline:
  28.5332622305 s
speculative:
  45.8338325806 s
```

These are not a controlled benchmark and do not establish a stable
slowdown ratio, but they definitively do not support a TPOT, latency, or
throughput improvement claim. Correctness is green for this one TP1 gate;
performance promotion remains closed.

Fresh focused verification after the position fix:

```text
parity artifact contract:
  27 passed
native verifier contract:
  6 passed
ModelRunner/verifier/batch/engine direct matrix:
  114 passed
combined speculative/runtime/verifier/parity process:
  231 passed before an unrelated test-module stub collision
isolated KV transaction matrix:
  28 passed
real loaded-model TP1 parity:
  PASS
independent local artifact verification:
  PASS
```

The combined-process `tools/test_speculative_kv_transaction.py` errors were
test isolation contamination: a prior dependency-light test had installed a
stub `Sequence` module without `block_size`. Running that file in its normal
isolated process passed all 28 tests.

Promotion remains:

```text
NOT_PROMOTABLE
```

Missing gates include a second model structure, TP4, long contexts, batch
shape coverage, cancellation, offload residency integration, repeated
controlled TTFT/TPOT/throughput measurements, peak memory, and real KV
movement evidence.

## Transactional Speculative Residency TP1 Correctness Gate

This section is the authoritative update for the generation-aware
`KVOffloadMVP0` residency transaction work completed on August 12, 2026. It
supersedes the older same-day statements above that describe offload
residency integration as entirely missing.

Classification remains:

```text
NOT_PROMOTABLE
```

### Implemented transaction boundary

The generic speculative runtime now propagates allocator generations through
ordinary and speculative KV access and coordinates rank-local residency
participants with this publication order:

```text
residency prepare
fixed-Q tail verifier forward(s)
residency precommit
atomic allocator KV commit
prepared Scheduler metadata commit
residency seal
optional draft lifecycle synchronization
```

The participant accepts distinct fixed-Q verifier groups as disjoint subsets
of one prepared ticket and accumulates materialization until the full ticket
is ready for precommit. The first-target decode remains ordinary KV work and
therefore does not require a speculative residency ticket. Unknown,
duplicated, stale-generation, malformed-acknowledgement, and incomplete
materialization cases fail closed.

Before allocator publication, failures roll back residency first and
allocator reservations second. Residency rollback failure poisons the
runtime. Seal failure occurs after allocator and Scheduler publication, keeps
target tokens authoritative, and poisons later speculative execution rather
than attempting an invalid host-state rollback.

### Fresh local and remote evidence

Dependency-light suites were run in isolated processes:

```text
tools/test_speculative_kv_transaction.py:
  31 passed
tools/test_speculative_residency.py:
  8 passed
tools/test_engine_speculative_execution.py:
  21 passed
tools/test_engine_speculative_runtime.py:
  35 passed
tools/test_speculative_model_runner_callbacks.py:
  14 passed
tools/test_model_runner_batch_spec_verify_source.py:
  3 passed
tools/test_model_runner_spec_verify.py:
  64 passed
tools/test_speculative_tp1_parity_gate.py:
  32 passed
```

The plan referenced
`tools/test_model_runner_speculative_residency.py`, but that standalone file
does not exist in this worktree; its residency coverage is in
`tools/test_model_runner_spec_verify.py` and
`tools/test_speculative_residency.py`.

Broad speculative regression:

```text
220 passed
```

The prescribed local `/opt/homebrew/bin/python3.12 tools/test_kv_offload.py`
could not import `flash_attn`, so it was not counted as a local pass. After
explicitly synchronizing the current test file, the same direct KVOffload
suite passed in the remote A100 environment:

```text
kv offload tests passed
```

Static validation:

```text
Python py_compile:
  passed for all changed runtime, residency, gate, and verifier files
git diff --check:
  passed
staged diff:
  empty
```

Loaded-model artifact:

```text
artifact:
  artifacts/speculative_tp1_parity/20260812T062046Z/result.json
independent remote verifier:
  artifacts/speculative_tp1_parity/20260812T062046Z/verify.remote.json
independent local verifier:
  artifacts/speculative_tp1_parity/20260812T062046Z/verify.json
schema:
  2
status:
  PASS
model and mode:
  Qwen3-0.6B BF16, TP1, temperature 0.0, two prompts, 32 output tokens
exact baseline/speculative token parity:
  PASS
source hashes:
  16 files independently revalidated
proposed tokens:
  58
accepted draft tokens:
  50
proposal rows:
  17
selected rows:
  24
target invocations:
  28
```

Real rank-0 `KVOffloadMVP0` movement counters:

```text
baseline:
  h2d_copies=0
  h2d_bytes=0
  d2h_copies=64
  d2h_bytes=1879048192
  copy_waits=0
  evictions=0
  evict_clean=0
  evict_dirty=0
speculative:
  h2d_copies=0
  h2d_bytes=0
  d2h_copies=43
  d2h_bytes=1262485504
  copy_waits=0
  evictions=0
  evict_clean=0
  evict_dirty=0
```

Real speculative residency counters:

```text
prepares=14
precommits=14
seals=14
rollbacks=0
committed_blocks=0
rejected_blocks=0
rejected_d2h_copies=0
```

Observed elapsed times, recorded only as diagnostics:

```text
baseline=33.1633478850 s
speculative=51.3615803123 s
```

### What this proves

- The loaded TP1 engine can run baseline and speculative decoding with real
  MVP-0 enabled and preserve exact greedy output tokens.
- Generation-aware prepare, grouped verifier materialization, precommit, and
  seal executed fourteen times with independently verified source identity.
- Rejected speculative residency did not cause a D2H copy in this workload.
- The two remote failures found during bring-up are now regression-covered:
  first-target decode no longer requires a not-yet-created ticket, and
  distinct fixed-Q groups materialize disjoint subsets of one ticket.

### What this does not prove

- `h2d_copies=0` means the gate did not exercise a real host-to-device reload.
- `committed_blocks=0` and `rejected_blocks=0` mean the prompts did not force
  reserved speculative blocks across a block boundary. The gate therefore
  does not yet prove accepted reserved-block retention or rejected
  reserved-block discard on a loaded model.
- The single speculative observation was slower than baseline and is not a
  controlled benchmark. It supports no TPOT, TTFT, throughput, latency,
  memory, or long-context improvement claim.
- TP4, a second model structure, stochastic decoding, cancellation,
  recurrent/convolution state transactions, learned drafter/MTP execution,
  and production serving remain unverified.

### Next promotion gates

1. Create a deterministic boundary-crossing workload with positive
   `committed_blocks` and `rejected_blocks`, while preserving
   `rejected_d2h_copies=0`.
2. Force real eviction and H2D reload so movement, waits, and generation
   safety are exercised rather than merely reported as zero.
3. Repeat controlled baseline/speculative runs and report TTFT, TPOT,
   throughput, peak GPU memory, and movement distributions without using
   this correctness run as performance evidence.
4. Only after those gates remain green, expand independently to TP4 and a
   materially different model structure.

## Deterministic Residency Boundary and Movement Gate

This section is the latest authoritative update and supersedes the boundary
limitations immediately above.

Classification remains:

```text
NOT_PROMOTABLE
```

### Corrected boundary geometry

The first remote attempt used the original one-token fixture design and
failed before speculative execution:

```text
ValueError: enabled selection requires max_proposal_tokens >= 2
```

Investigation showed a deeper geometry issue rather than a threshold-only
problem. A speculative transaction reserves capacity for
`proposal_count - 1` materialized tail tokens. At live sequence length 255, a
one-token proposal cannot reserve a second 256-token block.

The authoritative gate therefore keeps the 254-token prompt and length-255
post-prefill boundary, but runs `max_tokens=4` and derives a three-token greedy
suffix from the baseline output:

```text
accept fixture:
  proposes the exact three-token baseline suffix
reject fixture:
  changes the first suffix token and retains the remaining two
fixture drift check:
  configured first token must equal the live first_target_token
```

The three-token proposal materializes positions 255 and 256 and forces one
reserved second block. This change is gate-local and source-agnostic; no
model-name or fixture behavior was added to generic runtime, Scheduler,
verifier, allocator, or residency code.

### Authoritative loaded-model artifact

```text
artifact:
  artifacts/speculative_residency_boundary/20260812T065636Z/result.json
remote verifier:
  artifacts/speculative_residency_boundary/20260812T065636Z/verify.remote.json
local verifier:
  artifacts/speculative_residency_boundary/20260812T065636Z/verify.json
schema/status:
  1 / PASS
artifact SHA-256:
  fb40e71362f3cbaa9ff17ed0bf7233296073e9736145c64107278df4ba42fd25
model/mode:
  Qwen3-0.6B BF16, TP1, temperature 0.0
prompt/output:
  254 prompt tokens, 4 output tokens
exact baseline/accept/reject token parity:
  [215, 215, 215, 215] / PASS
evicted block identity in every case:
  (logical_block=0, generation=1)
independently revalidated source files:
  13
classification:
  NOT_PROMOTABLE
```

Real rank-0 `KVOffloadMVP0` movement was identical in all three cases:

```text
h2d_copies=1
h2d_bytes=29360128
d2h_copies=5
d2h_bytes=146800640
copy_waits=1
evictions=1
evict_clean=1
evict_dirty=0
```

Accepted-boundary evidence:

```text
proposed_tokens=3
accepted_draft_tokens=3
prepares=1
precommits=1
seals=1
rollbacks=0
committed_blocks=1
rejected_blocks=0
rejected_d2h_copies=0
elapsed_s=47.8559054136
```

Rejected-boundary evidence:

```text
proposed_tokens=6
accepted_draft_tokens=0
prepares=2
precommits=2
seals=2
rollbacks=0
committed_blocks=0
rejected_blocks=2
rejected_d2h_copies=0
elapsed_s=46.3958967924
```

Baseline elapsed observation:

```text
elapsed_s=50.0561158024
```

Elapsed values are recorded only as diagnostics. The cases load separate
engines, run once, and are not a controlled performance benchmark.

### Fresh validation

```text
dependency-light boundary/schema/runner suite:
  26 passed
focused plus broad speculative/offload regression:
  169 passed
remote A100 direct KVOffload suite after source sync:
  kv offload tests passed
remote independent verifier:
  PASS
local independent verifier:
  PASS
```

### What this proves

- A loaded TP1 target model preserved exact greedy output tokens across
  baseline, fully accepted, and immediately rejected boundary workloads.
- Accepted speculative KV retained one real reserved block in place through
  prepare, precommit, allocator/Scheduler publication, and seal.
- Rejected speculative KV discarded real reserved blocks with
  `speculative_residency_rejected_d2h_copies=0`.
- Clean writeback plus generation-validated eviction caused a real H2D reload,
  a real copy wait, and preserved the exact block generation identity.
- Remote and local independent verifiers recomputed all recorded source
  hashes and agreed on the same artifact digest.

### What this still does not prove

- This is an end-to-end correctness and movement gate, not evidence of an
  end-to-end performance optimization.
- The single elapsed observations do not establish TPOT, TTFT, throughput,
  latency, peak-memory, or long-context improvement.
- The prompt crosses one 256-token boundary only; long-context behavior and
  repeated eviction pressure remain unmeasured.
- TP4, a second model structure, stochastic decoding, cancellation,
  recurrent/convolution state transactions, learned drafter/MTP quality, and
  production serving remain unverified.

### Next promotion gates

1. Add repeated warm controlled baseline/speculative measurements with
   separated model-load time and report TTFT, TPOT, throughput, peak GPU
   memory, and movement distributions.
2. Exercise multiple boundaries and sustained GPU-block pressure before
   making any long-context or memory-capacity claim.
3. Independently validate TP4 and a materially different model structure.

## Controlled TP1 4K Speculative Runtime Performance Gate

This is the latest authoritative performance update and supersedes the
performance evidence gap above.

Classification remains:

```text
NOT_PROMOTABLE
```

### Authoritative artifact and environment

```text
artifact:
  artifacts/speculative_runtime_performance/20260812T085852Z/result.json
remote verifier:
  artifacts/speculative_runtime_performance/20260812T085852Z/verify.remote.json
local verifier:
  artifacts/speculative_runtime_performance/20260812T085852Z/verify.json
schema/status:
  1 / PASS
artifact SHA-256:
  d987b288176beec3ab841a7f640bb1d68cabf86445cadad50aec6337bcc4fb9f
source HEAD:
  3217895019a26154270db40c432495c7657abcb1
source tree:
  dirty; authority is the artifact's independently verified per-file hashes
device:
  NVIDIA A100 80GB PCIe
runtime:
  Python 3.11.15, torch 2.4.1+cu121
model/mode:
  Qwen3-0.6B, TP1, greedy temperature 0.0
prompt/output:
  4096 prompt tokens, 64 output tokens, ignore_eos=True
batch sizes:
  1 and 4
runs per cell:
  1 warmup, 1 parity, 5 measured
```

Final engine and offload configuration:

```text
enforce_eager=True
max_model_len=4352
max_num_batched_tokens=16384
max_num_prefill_tokens_per_step=1024
chunked_prefill_mixed_batch=False
kv_offload_mvp0=True
kv_offload_gpu_blocks=68
kv_offload_logical_blocks=128
kv_offload_blockwise_decode=False
kv_offload_blockwise_prefill=False
```

The initial blockwise design could not execute with the generic speculative
verifier. The final gate uses full-attention-compatible execution and, after
the first completion token, performs the same real clean writeback,
synchronization, and generation-validated eviction for baseline and n-gram.
The next decode restores the evicted blocks through real H2D.

### Exact parity and runtime activity

Baseline and candidate output token IDs matched exactly for the batch-1 row
and all four batch-4 rows; every row emitted exactly 64 tokens.

```text
ngram batch 1:
  proposed=300
  accepted=300
  acceptance_rate=1.0
  proposal_rows=75
  target_callbacks=165
  tail_callbacks=75

ngram batch 4:
  proposed=1240
  accepted=1225
  acceptance_rate=0.9879032258
  proposal_rows=310
  target_callbacks=650
  tail_callbacks=310
```

### Median performance

```text
batch 1 baseline:
  TTFT=0.519107182 s
  TPOT=0.049877169 s
  completion latency=3.660381415 s
  request throughput=0.273195574 req/s
  token throughput=17.484516706 tok/s

batch 1 ngram:
  TTFT=0.483959581 s
  TPOT=0.038769407 s
  completion latency=2.926432206 s
  request throughput=0.341713024 req/s
  token throughput=21.869633566 tok/s

batch 1 relative:
  TTFT=-6.77%
  TPOT=-22.27%
  completion latency=-20.05%
  token throughput=+25.08%

batch 4 baseline:
  TTFT=6.180727196 s
  TPOT=0.050722383 s
  completion latency=9.376237303 s
  request throughput=0.269538505 req/s
  token throughput=17.250464337 tok/s

batch 4 ngram:
  TTFT=4.091824249 s
  TPOT=0.026363790 s
  completion latency=5.776165820 s
  request throughput=0.449602205 req/s
  token throughput=28.774541098 tok/s

batch 4 relative:
  TTFT=-33.80%
  TPOT=-48.02%
  completion latency=-38.40%
  token throughput=+66.80%
```

All raw five-run values plus min/max/population standard deviation remain in
the artifact. Five runs do not establish statistical significance.

### Peak GPU memory and real MVP-0 movement

Peak memory was unchanged between policies within each batch:

```text
batch 1:
  peak allocated=3688463872 bytes
  peak reserved=4213178368 bytes

batch 4:
  peak allocated=3532291072 bytes
  peak reserved=4213178368 bytes
```

Five-run real movement totals:

```text
batch 1 baseline:
  H2D=80 copies / 2348810240 bytes
  D2H=475 copies / 13946060800 bytes
  clean evictions=131

batch 1 ngram:
  H2D=80 copies / 2348810240 bytes
  D2H=325 copies / 9542041600 bytes
  clean evictions=131

batch 4 baseline:
  H2D=80 copies / 2348810240 bytes
  D2H=1660 copies / 48737812480 bytes
  clean evictions=420

batch 4 ngram:
  H2D=80 copies / 2348810240 bytes
  D2H=1050 copies / 30828134400 bytes
  clean evictions=420
```

The candidate reduced D2H copies by 31.58% at batch 1 and 36.75% at batch 4
while preserving identical forced H2D evidence and clean-eviction counts.

### Direction and evidence boundary

```text
batch 1 direction:
  IMPROVED
batch 4 direction:
  IMPROVED
campaign direction:
  POSITIVE
classification:
  NOT_PROMOTABLE
```

What this proves:

- for these fixed TP1 4K batch-1/batch-4 cells, the real generic n-gram
  runtime preserved exact greedy tokens and improved both median TPOT and
  median token throughput;
- the comparison includes real MVP-0 H2D/D2H counters, synchronized timing,
  repeated warm measurements, and rank-aware peak memory;
- accepted speculative KV remained on the current transactional path; the
  legacy accepted-KV rematerialization profiler was not used.

What this does not prove:

- no TP4, 16K/32K, second-model, stochastic-decoding, cancellation,
  recurrent/convolution state, learned-drafter, or MTP evidence exists;
- the five-run direction is not a statistical significance or production-SLO
  result;
- unchanged peak allocation does not establish higher maximum context
  capacity;
- a fixed repetitive n-gram-friendly workload does not generalize to natural
  prompt distributions.

### Next controlled expansion

The next optimization gate should design 16K and 32K TP1 campaigns with
sustained multi-boundary pressure, while preserving exact parity, the same
real-counter authority, and `NOT_PROMOTABLE`. TP4, a second model, and learned
drafter/MTP remain separate later gates.

### Fresh final validation

```text
independent local artifact verifier:
  PASS
focused performance/parity/residency regression:
  145 passed
remote A100 direct tools/test_kv_offload.py:
  kv offload tests passed
py_compile:
  passed
remote runner bash syntax:
  passed
git diff --check:
  passed
staged diff:
  empty
```
