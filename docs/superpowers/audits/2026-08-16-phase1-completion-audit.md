# Phase 1 Completion Audit

**Date:** 2026-08-16

**Repository:** `/Users/bytedance/Desktop/TinyLLMForge`

**Checkout authority note:** on 2026-08-17 the complete
`TinyLLMForge-adaptive-ngram` working state was migrated into the Desktop
checkout. Historical artifact paths still record the checkout used when each
campaign ran, but all current and future audit updates must use
`/Users/bytedance/Desktop/TinyLLMForge`.

**Decision:** `PHASE_1=NOT_ACHIEVED`

**Promotion:** `NOT_PROMOTABLE`

## Audit Rule

This audit uses the current filesystem artifacts as authority. It supersedes
stale completion statements in earlier audits, especially statements written
before the 2026-08-15 independent Qwen3 learned-drafter authorities existed.

Evidence is not composable across unrelated scopes for promotion. In
particular, the following pieces cannot be added together and called a
completed learned speculative system:

- generic n-gram long-context coverage;
- TP1 learned-drafter Proposal-KV offload;
- TP4 learned-drafter direct-allocation correctness;
- Qwen3.5 native-MTP target-KV offload; and
- a separate generic n-gram performance result.

A promotion claim requires one coherent source-bound runtime path to satisfy
the required correctness, topology, context, batch, lifecycle, movement, and
performance gates. A failed cell remains failed, and uncertainty is classified
as not established.

## Fresh Current-Tree Re-Audit

A fresh read-only re-audit on 2026-08-16 checked the retained evidence against
the current working tree rather than relying only on stored classifications:

```text
referenced key JSON artifacts:
  13 present
  0 missing

Qwen3.5 generic TP4/32K:
  frozen archived source verifier: PASS
  current working-tree verifier:   FAIL
  reason: current source file identity mismatch

Qwen3 independent draft TP4:
  retained run-time current/archive receipts: PASS and matching at capture time
  fresh current working-tree verifier:        FAIL
  reason: source file identity mismatch

Qwen3 independent draft TP4 performance:
  retained artifact schema: 2
  current verifier schema:   3
  fresh current verifier:    FAIL before semantic validation
  reason: artifact schema version mismatch

Qwen3.5 native-MTP TP4/16K:
  retained verifier receipt: PASS
  fresh current verifier against archived source: FAIL
  fresh current verifier against current source:  FAIL
  reason: current verifier source-file inventory no longer matches the
          historical manifest
```

Therefore, retained GPU authorities remain valid evidence for their exact
captured source/manifests and recorded scopes, but they are **not current-head
certification**. A stored `PASS` receipt proves that the frozen verifier passed
at capture time; it does not prove that the current working tree or current
verifier still reproduces that result. The final promotion campaign must bind
and verify the then-current source closure, not inherit historical green
statuses.

The source drift is concentrated in runtime-critical code rather than only
documentation or test harnesses:

```text
Qwen3.5 generic TP4/32K manifest:
  16 files bound
  12 unchanged
  4 changed
    tinyvllm/engine/llm_engine.py
    tinyvllm/engine/model_runner.py
    tinyvllm/engine/qwen35_speculative_state.py
    tinyvllm/engine/speculative_model_runner.py

Qwen3 independent-draft TP4 manifest:
  30 files bound
  28 unchanged
  2 changed
    tinyvllm/engine/autoregressive_draft_executor.py
    tinyvllm/engine/llm_engine.py

Qwen3.5 native-MTP TP4/16K manifest:
  112 files bound
  102 unchanged
  10 changed
    tinyvllm/config.py
    tinyvllm/engine/llm_engine.py
    tinyvllm/engine/model_runner.py
    tinyvllm/engine/proposal_kv_cache.py
    tinyvllm/engine/qwen35_mtp_executor.py
    tinyvllm/engine/qwen35_mtp_registration.py
    tinyvllm/engine/qwen35_speculative_state.py
    tinyvllm/engine/speculative_proposal_executor.py
    tinyvllm/engine/speculative_selection.py
    tinyvllm/layers/attention.py
```

Function-level inspection confirms that the changed regions include
Proposal-KV allocator semantics, MTP proposal storage and executor lifecycle,
autoregressive-draft finalize batching, speculative side-state release,
H2D-slot diagnostics, and attention behavior. This is material runtime drift.
It cannot be dismissed as verifier-only or telemetry-only evolution.

## Concrete Phase 1 Deliverables

Phase 1 is the proposed **Generic MTP/Speculative Runtime + Transactional KV
Cache** milestone. Its completion criteria are:

1. one source-neutral runtime that can host MTP, an independent learned draft
   model, and model-free n-gram/SAM proposal sources;
2. at least two materially different target model structures;
3. TP1 and TP4 execution;
4. 4K, 16K, and 32K contexts;
5. batch 1, batch 4, and real multi-sequence execution;
6. exact greedy output parity;
7. TTFT, TPOT, throughput, peak memory, acceptance, and real KV H2D/D2H
   measurements;
8. accepted proposal KV committed in place, rejected suffix KV released, and
   zero accepted-prefix target replay;
9. at least one real learned proposal source beyond n-gram/SAM;
10. real, workload-derived Proposal-KV capacity and real KV movement when
    offload is claimed; and
11. unsupported offload, precision, CUDA Graph, and sampling combinations
    failing closed.

The wider optimization objective also includes prefix sharing/refcount,
KV4/KV8, heat tiering, blockwise long-context attention, variable-Q CUDA
Graphs, verifier/sampling/commit fusion, and TP collective overlap. These are
audited below as adjacent deliverables, but their partial implementation does
not compensate for a missing Phase 1 promotion gate.

## Executive Completion Matrix

| Requirement | Verdict | Current evidence | Completion boundary |
| --- | --- | --- | --- |
| Source-neutral speculative runtime | `PARTIAL_WITH_DUAL_ABI_AND_ASYMMETRIC_RETAINED_PROVENANCE` | The current tree accepts a host `DraftAdapter` domain and a model-runner `ProposalExecutor` domain, then normalizes both into the same batch verifier, target-KV transaction, Scheduler publication, and commit/rollback path; native MTP and independent draft share the model-runner registry/lifecycle ABI | There is no single proposal-generation ABI, and no one source-closed retained authority binds an identical shared-runtime file set and source identity across n-gram, native MTP, and independent draft |
| Batch-native multi-token verifier | `PARTIAL_WITH_FIXED_Q_GROUPING_AND_ASYMMETRIC_RETAINED_RECEIPTS` | The current tree groups sequences by tail query length, performs one `run_model(..., execution_mode="spec_verify")` per homogeneous fixed-Q group, splits the flat logits back into sequence rows, materializes verifier KV in transaction-authorized slots, and commits only the accepted materialized prefix | A speculative step still uses a separate first-target forward plus the tail verifier forward; heterogeneous proposal lengths require multiple fixed-Q forwards, and retained query-length/forward receipts are not uniform across n-gram, native MTP, and independent draft |
| MTP source | `PARTIAL` | Qwen3.5 native MTP passes TP1/4K, TP4/4K, and TP4/16K target-KV offload | TP4/32K batch-1 parity failed; no controlled native-MTP performance authority |
| MTP target-forward reduction and TPOT benefit | `NOT_ESTABLISHED_AS_A_CONTROLLED_BENEFIT` | Native MTP authorities retain speculative-side first-target and verifier call counts, high acceptance in TP4/4K, exact parity in passing cells, and no accepted-prefix replay | Baseline native-MTP cells retain normalized zero placeholders rather than measured ordinary-decode forward counts; no native-MTP authority records controlled TTFT/TPOT/throughput timing, so generic n-gram gains cannot be transferred to MTP |
| Independent learned draft source | `PARTIAL` | Qwen3 target/draft TP1 Proposal-KV offload and TP4 direct correctness authorities exist | TP4 has no offload, long-context, or positive performance authority; only one learned target/draft structure |
| Independent learned draft exact-shape CUDA Graph | `INCONCLUSIVE_ENVIRONMENT` | The current tree contains a default-off exact TP4/B4/Q4 greedy dense-direct graph path, private capture scratch, shared eager/graph registration, TP failure convergence, source-bound gate/verifier tooling, and focused local regression coverage | No four clean remote GPUs or valid retained Python/model environment were available, so no loaded capture/replay, real eager/graph parity, or controlled performance result exists |
| Model-free source | `ACHIEVED` within recorded scopes | Generic n-gram authorities cover Qwen3 and Qwen3.5 | SAM remains local/limited and is not needed to rescue the n-gram evidence |
| Two materially different target structures | `PARTIAL_WITH_ASYMMETRIC_PROVENANCE` | Retained generic n-gram authorities identify Qwen3-0.6B and Qwen3.5 as separate checkpoint families; the Qwen3.5 authority self-contains `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`, and the 18-linear/6-full-attention hybrid layout | The Qwen3 authority retains only the Qwen3-0.6B path/family label and model-manifest digest, without config, architecture/model-type receipt, model implementation, or per-file checkpoint manifest; its pure-Transformer structure cannot be independently verified from the bundle |
| TP1 and TP4 | `ACHIEVED` for generic n-gram; `PARTIAL` for learned sources | Generic matrix includes both; native MTP and independent draft each have TP1/TP4 evidence in narrower scopes | No one learned path closes the full topology/context/performance matrix |
| 4K, 16K, and 32K | `ACHIEVED` for generic n-gram; `FAILED` for native-MTP completion | Qwen3.5 generic TP4 4K/16K/32K passes | Native MTP TP4/32K batch 1 failed; independent draft uses short prompts |
| Batch 1, batch 4, multi-sequence | `ACHIEVED` within cited greedy scopes | Eight passing correctness authorities each retain two batch-4 cells with four distinct prompts, four outputs, and 4/4 per-sequence parity; execution evidence records either active sequence count 4 or sequence IDs 0-3 | Mixed cancellation/heterogeneous sequence state and a full learned long-context matrix remain incomplete |
| Exact greedy parity | `PARTIAL`, with one explicit failed promotion cell | Generic matrix and learned scopes through native TP4/16K pass | Native TP4/32K batch 1 differs at output indices 3 and 6 |
| Transactional direct commit/release | `PARTIAL_WITH_SOURCE_ASYMMETRY` | Native MTP directly records commit/release and zero accepted-prefix target replay; independent draft records zero accepted-entry copy/replay/rematerialization and TP4 transaction-ticket commit/release pairing | Generic Qwen3 retains no equivalent transaction/replay receipt; generic Qwen3.5 lacks slot-identity and accepted-entry-copy receipts; independent TP4 lacks an accepted-prefix target-replay counter |
| TTFT, TPOT, throughput | `PARTIAL` | Only the generic Qwen3 TP1/4K and Qwen3.5 TP4/16K authorities provide five-run batch-1/4 controlled comparisons with all three metrics | Independent Qwen3 TP4 is a negative three-run 256-token pilot; native MTP has no controlled performance authority |
| Peak memory | `PARTIAL` | The two generic authorities and the independent-draft pilot record peak allocated and reserved bytes | Generic results show no material allocated-memory reduction; no learned-source long-context matrix exists |
| Acceptance | `PARTIAL` | Generic authorities and the short-context independent-draft pilot record proposed/accepted tokens and acceptance rate | Native MTP has no controlled performance authority; no learned-source 4K/16K/32K metric matrix exists |
| Real target-KV H2D/D2H | `ACHIEVED` within recorded offload scopes | Engine `kv_offload_summaries` counters and real movement provenance are retained | Not every passing cell requires H2D; no universal reduction claim |
| Real Proposal-KV H2D/D2H | `PARTIAL` | Independent Qwen3 TP1 offload records real bidirectional movement | TP4 uses direct allocation; no TP4/long-context Proposal-KV offload authority |
| Fail-closed unsupported combinations | `PARTIAL` | Quantized-KV plus offload/spec-verify is rejected locally; schemas and verifiers reject several malformed configurations | No unified loaded authority spanning offload, precision, graph, and sampling combinations |
| Artifact provenance and independent verification | `PARTIAL_AND_DRIFTED` | Retained authorities have manifests and capture-time receipts; Qwen3.5 generic TP4/32K independently passes against its frozen archived source | Fresh current-tree checks fail for generic TP4/32K and independent-draft TP4 due source drift; native-MTP TP4/16K and learned-performance verification are also blocked by verifier/schema drift |

### Source-neutral runtime boundary

The current working tree has one shared speculative orchestration and
publication path, but it does not expose one proposal-source ABI.

`EngineSpeculativeRuntime` requires exactly one of two source domains:

```text
host:
  DraftAdapter.propose_batch(...)

model_runner:
  ModelRunnerProposalExecutorDescriptor
  -> ProposalExecutor.propose_batch(...)
  -> prepare/commit/rollback_finalize_batch(...)
  -> release_sequence(...)
```

`build_model_runner_proposal_provider(...)` normalizes both domains into
first-target-plus-proposal rows. `LLMEngine.step()` then sends either source
through:

```text
prepare_native_speculative_batch(...)
  -> shared tail verification and target-KV transaction preparation
  -> build_engine_prepared_speculative_commit_rows(...)
  -> Scheduler prepared publication
  -> _commit_prepared_speculative_publication(...)
       -> target-KV commit
       -> Scheduler publication
       -> side-state commit/rollback
       -> model-runner Proposal-KV finalize when applicable
```

Qwen3.5 native MTP and the independent Qwen3 draft both register against
`ModelRunnerProposalExecutorRegistry` and the same `ProposalExecutor`
lifecycle protocol, under executor IDs `native_checkpoint_proposal` and
`autoregressive-draft`. N-gram instead uses the host `DraftAdapter` domain.
The implementation is therefore source-neutral at the orchestration,
verification, target-KV transaction, and publication layers, not at the
proposal-generation ABI boundary.

The retained source provenance does not close this as one cross-source
historical authority:

| Representative authority | Manifest-bound shared-runtime coverage | Boundary |
| --- | --- | --- |
| Qwen3 generic n-gram TP4/4K | 11 files; binds `llm_engine.py`, `model_runner.py`, `speculative_runtime.py`, and `speculative_model_runner.py` | Does not bind `batch_runtime.py` or `speculative_proposal_executor.py` |
| Qwen3.5 generic n-gram TP4/16K performance | 15 files; binds `llm_engine.py`, `model_runner.py`, `speculative_runtime.py`, `batch_runtime.py`, and `ngram_adapter.py` | Does not bind `speculative_model_runner.py` or `speculative_proposal_executor.py` |
| Qwen3.5 native MTP TP4/16K | 112 files; binds all six core files: engine, model runner, runtime, batch runtime, proposal executor, and model-runner bridge | Strongest source-closed shared-runtime authority, but only for native MTP |
| Qwen3 independent draft TP4 direct | 30 files; binds engine, model runner, runtime, batch runtime, proposal executor, adapter, and independent-draft registration | Does not bind `speculative_model_runner.py`; the bound proposal-executor hash also differs from native MTP |
| Qwen3 independent draft TP1 offload | Broad `source_sha256.txt` inventory binds all six core files and both source domains | The compact bundle explicitly retains no frozen source tarball or independent current/archive verifier |

Even where filenames overlap, the generic, native-MTP, and independent-draft
authorities were captured from different source identities. For example,
native MTP TP4/16K and independent draft TP4 bind the same
`speculative_runtime.py` and `batch_runtime.py` hashes, but bind different
`llm_engine.py`, `model_runner.py`, and `speculative_proposal_executor.py`
hashes. Generic authorities predate or omit parts of the learned-source
executor closure.

Consequently, current-tree implementation evidence establishes a shared
source-neutral back half, and retained executions separately establish that
all three source families have used versions of that architecture. The
artifacts do not establish one identical, source-closed, cross-source runtime
authority, so `ACHIEVED` was too strong.

### Batch-native multi-token verifier boundary

The current verifier is batch-native within a homogeneous tail query length.
It is not one unconditional target forward for an entire heterogeneous
speculative batch.

For a proposal of length `N`, `build_spec_verify_plan(...)` constructs:

```text
first proposal token:
  checked by the separately computed first-target token

remaining proposal tokens:
  verifier input tokens = proposal[:-1]
  tail query length      = N - 1
```

`build_fixed_q_tail_batches(...)` groups all selected sequences by that
`query_len`. `run_model_runner_tail_batch(...)` issues one
`run_spec_verify_batch` call per fixed-Q group. Within each group,
`ModelRunner._run_spec_verify_batch(...)`:

```text
prepare_spec_verify_batch(all rows in the group)
run_model(..., execution_mode="spec_verify") exactly once
argmax the flattened logits once
split the flattened target-token rows back by sequence
```

This proves one target tail forward can verify multiple proposed tokens for
multiple sequences of the same verifier width. It does **not** mean one total
target forward per speculative step: the first-target result is produced
separately, followed by the tail verifier forward when `N > 1`.

The target KV produced by the tail forward is transaction-authorized and
directly reused. The runtime marks exactly `N - 1` proposal positions as
materialized. Commit preparation then requires:

```text
accepted_materialized_tokens = max(accepted_count - 1, 0)
accepted_materialized_tokens <= materialized_token_count
```

Only the required accepted-prefix blocks are committed; unused reserved blocks
are released. The subtraction by one is the normal autoregressive cache
boundary: the last accepted output token becomes the next decode input and
does not yet have target KV. This is direct verifier-KV commit, not accepted
token replay or per-token rematerialization.

The strongest retained runtime counters are native MTP TP4/4K:

```text
batch 1:
  proposal rows / proposed tokens:  8 / 32
  first-target target forwards:     8
  verify target forwards:           8
  proposed tokens per verify call:  4.0
  accepted-prefix target replays:   0

batch 4:
  proposal rows / proposed tokens: 32 / 128
  first-target target forwards:    32
  verify target forwards:          32
  proposed tokens per verify call:  4.0
  accepted-prefix target replays:   0
```

These counters establish one first-target plus one multi-token tail verifier
call for each width-4 proposal row, rather than four verifier forwards per
proposal. Exact output parity and Proposal-KV/target-KV transaction receipts
establish the result and direct-commit behavior in that retained scope.

The TP4/16K batch-4 authority also exposes the fixed-Q limitation:

```text
engine speculative callbacks: 3
proposal rows:                10
proposed tokens:              36
verify target forwards:        5
```

Ten rows are batch-native, but their heterogeneous terminal proposal lengths
produce five fixed-Q verifier calls across three engine callbacks. Therefore
the implementation is batch-native by homogeneous query-length group, not a
single ragged-Q verifier forward.

Retained evidence is asymmetric across sources:

- independent-draft TP1 records `first_target_forward_count=8`,
  `tail_verification_forward_count=8`,
  `extra_target_forward_count=0`, and zero accepted-entry
  copy/replay/rematerialization;
- independent-draft TP4 retains proposal rows with widths `4/3/2/1` and
  transaction snapshots, but no dedicated target-verifier forward counter;
- generic Qwen3.5 retains width-4 proposal and zero replay receipts but no
  equivalent complete first-target/tail-forward receipt set;
- generic Qwen3 retains no direct transaction, replay, or verifier
  query-length receipt.

The implementation contract is established, and native MTP directly
demonstrates it. Uniform cross-source retained execution proof, a ragged-Q
single-forward path, and the failed native-MTP TP4/32K parity cell remain
open. The correct Phase 1 classification is therefore partial.

### MTP target-forward reduction and TPOT boundary

The current implementation provides the mechanism needed to reduce target
decode work: one first-target forward plus one multi-token tail verifier call
can accept several proposal tokens, and accepted verifier KV is committed
without replay. That mechanism is not equivalent to a controlled performance
benefit.

The passing native-MTP authorities directly retain speculative-side call
counts:

```text
TP4/4K batch 1:
  output tokens:                 32
  proposal rows / tokens:         8 / 32
  accepted / rejected:           30 / 2
  first-target / verify calls:    8 / 8
  speculative target calls:      16

TP4/4K batch 4:
  output tokens:                128
  proposal rows / tokens:        32 / 128
  accepted / rejected:          120 / 8
  first-target / verify calls:   32 / 32
  speculative target calls:      64

TP4/16K batch 1:
  output tokens:                  8
  proposal rows / tokens:         3 / 10
  accepted / rejected:            5 / 5
  first-target / verify calls:    3 / 3
  speculative target calls:       6

TP4/16K batch 4:
  output tokens:                 32
  proposal rows / tokens:        10 / 36
  accepted / rejected:           19 / 17
  first-target / verify calls:    3 / 5
  speculative target calls:       8
```

These are real MTP-side method-call counters and prove that each speculative
step does not replay every accepted token through a separate target forward.
They do not provide a controlled baseline comparison.

The baseline rank snapshots in these authorities contain zeros for
`first_target_target_forwards` and `verify_target_forwards` because the worker
constructs baseline snapshots with a fixed non-speculative zero schema. Those
zeros do **not** count ordinary autoregressive target forwards. The bundles
retain no separate ordinary-decode forward counter, profiler call inventory,
or baseline-vs-MTP launch count.

It is possible to derive a mechanism-level call density from the speculative
side, and current-tree ordinary decode semantics imply one target invocation
per decode iteration. That inference is insufficient for a source-bound
promotion result because batch scheduling, fixed-Q grouping, TP dispatch,
proposal overhead, and kernel duration determine whether fewer logical decode
iterations actually reduce TPOT.

No retained native-MTP authority contains TTFT, TPOT, throughput, peak-memory,
or repeated controlled timing fields. The two positive controlled performance
authorities are generic n-gram campaigns. The independent learned-draft TP4
pilot has complete timing fields but regresses TPOT and throughput.

Therefore:

```text
MTP mechanism for multi-token target verification: established
native-MTP speculative-side target call counters: established
native-MTP measured baseline target call counters: absent
controlled native-MTP target-forward reduction: not established
controlled native-MTP TPOT benefit: not established
```

The design rationale remains plausible and partially evidenced, but Phase 1
cannot claim that MTP has already reduced target decode work or improved TPOT
under a controlled source-bound comparison.

### Two-target structure provenance

The retained generic n-gram matrix records two different checkpoint families,
but the architectural provenance is asymmetric.

The Qwen3.5 TP4/32K authority is self-contained enough to verify the target
structure independently:

```text
architectures:               [Qwen3_5ForConditionalGeneration]
model_type:                  qwen3_5
text layers:                 24
linear/recurrent layers:     18
full-attention layers:        6
```

Its source manifest also binds the architecture-specific implementation:

```text
tinyvllm/models/qwen35_packed.py
tinyvllm/layers/qwen35_linear_attention.py
tinyvllm/layers/qwen35_packed_layer_stack.py
```

The Qwen3 TP4/4K authority records the Qwen3-0.6B tokenizer/checkpoint path,
the `Qwen3-0.6B` claim scope, and model-manifest digest
`6bb7f90f4ad46c059c9e3df600532147ecc00683e58e96ce9dd6bc5084f2c90e`.
However, the bundle retains no `architectures` or `model_type` receipt, no
model config, no Qwen3 model implementation in its 11-file source manifest,
and no canonical per-file checkpoint-manifest rows. The original checkpoint
path is not locally available, and no archived source copy matching the
manifest-bound historical `model_runner.py` was found. Independent-draft
authorities add Qwen3 checkpoint identity but do not add architecture/config
receipts.

It is historically plausible that these were the intended pure-Transformer
Qwen3 and hybrid Qwen3.5 targets, but path semantics and family labels are not
self-contained architectural proof. Under this audit's fail-closed rule,
uncertainty is incomplete evidence. The two-target-structure gate therefore
cannot remain unconditionally achieved.

### Batch-4 and multi-sequence evidence

A fail-closed retained-artifact recomputation covered the eight passing
correctness authorities cited in the matrix:

```text
passing authorities:                    8
batch-4 cells:                         16
prompt rows:                           64
distinct prompt rows within each cell:  4 / 4
output rows:                           64
per-sequence baseline/alternative
  parity comparisons:                  32 / 32

authorities retaining active_sequence_count >= 4:
  4
authorities retaining sequence IDs 0,1,2,3 in runtime/transaction evidence:
  7
```

Every authority records `batch_size=4`, prompt indices `[0,1,2,3]`, four
different prompt hashes/token rows, and four positional or prompt-index-bound
outputs in both the baseline and speculative/native cell. Recomputing parity
row by row gives `4/4` exact output equality for every authority rather than
relying on the aggregate `parity.b4=true` field.

The execution evidence is also stronger than payload cardinality alone. The
Qwen3 TP4 and Qwen3.5 generic TP4 profiles reach
`active_sequence_count == 4`; the Qwen3.5 TP1, generic TP4, and native-MTP
transaction/runtime receipts cover sequence IDs `0,1,2,3`. Every cited
authority has at least one of these direct four-sequence signals.

This establishes real four-sequence greedy execution in the cited scopes. It
does not establish heterogeneous prompt lengths, asynchronous arrival,
mid-flight cancellation, mixed finished/active states, or a complete learned
4K/16K/32K performance matrix.

## Prompt-to-Artifact Checklist

### Independent Qwen3 draft exact-shape CUDA Graph

The current tree adds a second CUDA Graph family distinct from the
speculative-verifier Variable-Q graph below. This path captures independent
Qwen3 draft proposal generation, not target-side verification.

| Explicit gate | Artifact | Raw evidence | Verdict |
| --- | --- | --- | --- |
| Default-off TP4/B4/Q4 greedy dense-direct admission | `tinyvllm/config.py`, `tinyvllm/engine/model_runner.py`, config/integration tests | Exact allowlists and fail-closed topology/offload checks | `ACHIEVED_LOCALLY` |
| Two successful eager observations before capture | `tinyvllm/engine/autoregressive_draft_graph.py` | State-machine tests distinguish failed and successful eager observations | `ACHIEVED_LOCALLY` |
| Private scratch capture and reverse rollback | `tinyvllm/engine/qwen3_draft_graph_scratch.py` | Ownership tests preserve live committed state and clear scratch transactions | `ACHIEVED_LOCALLY` |
| Three-step GPU token chain with one final readback | `tinyvllm/engine/qwen3_draft_cuda_graph_backend.py` | Fake-torch tests cover three forwards, argmax/broadcast, and one `.tolist()` | `ACHIEVED_LOCALLY` |
| Shared eager/graph proposal authority and finalization | `tinyvllm/engine/autoregressive_draft_executor.py` | Dispatch, logical digest, commit/abort, and TP convergence tests | `ACHIEVED_LOCALLY` |
| Tamper-resistant source-bound controlled gate | `tools/autoregressive_draft_cuda_graph_contract.py`, gate/runner/verifier tools | Contract tests reject token, transaction, counter, source, order, memory, and aggregate tampering | `ACHIEVED_LOCALLY` |
| Real TP4 eager/graph correctness and performance | `/tmp/autoregressive-draft-cuda-graph-preflight-20260817.json` | Read-only preflight found fewer than four clean GPUs and stopped before source upload | `INCONCLUSIVE_ENVIRONMENT` |

Canonical focused audit:

```text
docs/superpowers/audits/
  2026-08-17-autoregressive-draft-cuda-graph-completion-audit.md
```

### Generic n-gram coverage

| Explicit gate | Artifact | Raw evidence | Verdict |
| --- | --- | --- | --- |
| Qwen3 TP4 nominal 4K, batch 1/4 | `artifacts/generic_speculative_tp4/tp4-opaque-48d18e4aba16756d/authority/result.json` | Four cells, prompt token count 4096, raw baseline/ngram outputs equal for batch 1 and four batch-4 sequences | `ACHIEVED` |
| Qwen3.5 TP1 nominal 4K, batch 1/4 | `artifacts/qwen35_generic_speculative_tp1/opaque-d4e74cb46fccbc57319c3c4f/artifacts/authority/result.json` | Raw prompts contain 4048 tokens, not exactly 4096; parity booleans and payload rows pass | `ACHIEVED` only as nominal-4K |
| Qwen3.5 TP4 4K, batch 1/4 | `artifacts/qwen35_generic_speculative_tp4/opaque-24f8ae471a2ba439ecb5a3b1/artifacts/authority/result.json` | 4096-token raw rows and exact parity | `ACHIEVED` |
| Qwen3.5 TP4 16K, batch 1/4 | `artifacts/qwen35_generic_speculative_tp4_16k/opaque-3b8050a916f037bc92412ea5/artifacts/authority/result.json` | 16384-token raw rows and exact parity | `ACHIEVED` |
| Qwen3.5 TP4 32K, batch 1/4 | `artifacts/qwen35_generic_speculative_tp4_32k/opaque-03a0a96654a14441b314800f/artifacts/authority/result.json` | 32768-token raw rows and exact parity | `ACHIEVED` |
| Qwen3 TP1/4K controlled performance | `artifacts/speculative_runtime_performance/20260812T085852Z/result.json` | Five measured runs, batch 1/4 exact parity, TTFT/TPOT/throughput/peak memory/acceptance/real movement fields | `PASS_NOT_PROMOTABLE` |
| Qwen3.5 TP4/16K controlled performance | `artifacts/qwen35_generic_speculative_tp4_16k_performance/opaque-c9807d19e6402acc22d4a615/artifacts/authority/result.json` | Five measured runs and complete metric fields; campaign direction positive | `PASS_NOT_PROMOTABLE` |

The Qwen3 TP1/4K controlled comparison recomputes to:

```text
batch 1:
  TPOT ratio:        0.777298
  throughput ratio:  1.250800
  H2D byte ratio:    1.000000
  TTFT ratio:        0.932292

batch 4:
  TPOT ratio:        0.519766
  throughput ratio:  1.668044
  H2D byte ratio:    1.000000
  TTFT ratio:        0.662030
```

This proves real H2D/D2H movement and positive TPOT/throughput direction, but
not H2D-byte reduction.

The Qwen3.5 TP4/16K controlled comparison recomputes to:

```text
batch 1:
  TPOT ratio:        0.5677486644
  throughput ratio:  1.4279094053
  H2D byte ratio:    0.5408970976
  TTFT ratio:        1.1405970670

batch 4:
  TPOT ratio:        0.5443779098
  throughput ratio:  1.7823893148
  H2D byte ratio:    0.5396825397
  TTFT ratio:        1.0609167401
```

This is positive for TPOT, throughput, and H2D bytes, but TTFT regresses in
both batches. Five measured runs do not establish statistical significance or
production readiness.

### Promotion metric coverage

A fresh retained-artifact recomputation distinguishes metric-field presence
from a complete promotion matrix.

| Authority | Scope | Repetitions | TTFT / TPOT / throughput | Peak memory | Acceptance | Real KV movement | Audit classification |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Qwen3 generic n-gram | TP1, 4K, batch 1/4 | 1 warmup + 5 measured | Present | Allocated and reserved present | Present | Target-KV H2D and D2H are positive in all 10 speculative measured runs | Complete fields in one generic scope |
| Qwen3.5 generic n-gram | TP4, 16K, batch 1/4 | 1 warmup + 5 measured | Present | Allocated and reserved present | Present | Target-KV H2D and D2H are positive in all 10 speculative measured runs | Complete fields in one generic scope |
| Qwen3 independent learned draft | TP4, 256-token prompt, batch 1/4 | 1 warmup + 3 measured | Present, but TPOT/throughput direction is negative | Allocated and reserved present | Present | Proposal-KV H2D/D2H are zero; no target-KV movement comparison is retained | Pilot only, not a promotion authority |
| Qwen3.5 native MTP | Correctness scopes through TP4/16K | No controlled campaign | Missing | Missing controlled comparison | Correctness counters only | TP4/16K correctness authority proves real target-KV movement, not controlled performance | Missing |

The two complete-field authorities are both model-free n-gram campaigns. They
do not establish that MTP or an independent learned draft improves a
production-relevant workload. The independent-draft pilot uses only 256
prompt tokens and three measured repetitions. Its learned path recomputes to:

```text
batch 1:
  TTFT ratio:          0.874739331
  TPOT ratio:          1.458980053
  throughput ratio:    0.704521491
  peak allocated ratio: 1.003988411
  acceptance:          1.000000000
  Proposal-KV H2D/D2H: 0 / 0 bytes

batch 4:
  TTFT ratio:          1.094911932
  TPOT ratio:          1.900349735
  throughput ratio:    0.483019445
  peak allocated ratio: 1.004294478
  acceptance:          0.736111111
  Proposal-KV H2D/D2H: 0 / 0 bytes
```

Consequently, the required metric names are represented somewhere in retained
artifacts, but the promotion requirement is not closed across learned/native
proposal sources, long contexts, TP1/TP4, and batch 1/4. Zero movement in the
direct-allocation learned pilot is not offload evidence and is not counted as
a movement benefit.

### Blockwise/chunked long-context authority binding

| Explicit gate | Artifact or fresh command | Raw evidence | Verdict |
| --- | --- | --- | --- |
| Current blockwise online-softmax implementation | `tinyvllm/layers/attention.py`, `tinyvllm/engine/model_runner.py`, `tinyvllm/engine/scheduler.py` | Current dispatch includes blockwise prefill, decode, and speculative-verify paths; ModelRunner activates blockwise prefill and Scheduler implements chunked-prefill admission/lifecycle | `ESTABLISHED_IN_CURRENT_SOURCE` |
| Qwen3.5 tiled blockwise prefill numerical oracle | `tools/test_qwen35_cached_prefill_eager_attention.py::test_prefill_blockwise_matches_dense_gqa_and_bounds_tiles` | Fresh isolated CPU-Torch execution: `1 passed in 5.65s` | `ESTABLISHED_LOCAL` |
| Chunked-prefill scheduler lifecycle | Five focused nodes in `tools/test_chunked_prefill.py` | Fresh dependency-light execution: `5 passed in 1.63s` | `ESTABLISHED_LOCAL` |
| General current blockwise attention suite | `python -m pytest -q tools/test_blockwise_attention_planning.py` | Collection stops at `ModuleNotFoundError: No module named 'flash_attn'` | `ENVIRONMENT_BLOCKED`, neither PASS nor semantic FAIL |
| TP1 Qwen3 16K/32K, batch 1/4 | `artifacts/blockwise_speculative_verifier/blockwise-tp1-opaque-17786-19070/result.json` | Four retained cells (`16384:b1`, `16384:b4`, `32768:b1`, `32768:b4`) pass exact parity; result SHA256 is `2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600` | `HISTORICAL_SCOPE_ESTABLISHED` |
| TP1 current-source verification | `tools/verify_blockwise_speculative_verifier_gate.py` plus seven-file source manifest | Fresh verifier fails closed on source drift; only 4/7 source files match, with drift in `model_runner.py`, `speculative_residency.py`, and `attention.py` | `BLOCKED_SOURCE_DRIFT` |
| Generic TP4 16K/32K long-context binding | Generic TP4 16K and 32K authority manifests | Both 16-file manifests omit `config.py`, `scheduler.py`, `attention.py`, and `layers/qwen35_full_attention.py`; results contain no selected attention-path field | `IMPLEMENTATION_SOURCE_BINDING_MISSING` |
| Native-MTP TP4/16K blockwise binding | `artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/lifecycle-release-fix-20260814-2/artifacts/authority/` | The 112-file source manifest includes config, scheduler, generic attention, and Qwen3.5 attention; config enables blockwise prefill/decode with eight staging blocks and 1024 prefill tokens per step; retained residency is bounded at 65/68 blocks for batch 1 and 68/68 for batch 4 | `SOURCE_AND_CONFIG_BINDING_ESTABLISHED` |
| Direct selected runtime path observation | TP1, generic TP4, and native TP4 result schemas | No result records `attention_path`, `blockwise_online`, `attention_backend`, `kernel_name`, or an equivalent selected-path observation | `MISSING` |

The retained long-context authorities still establish prompt length, exact
greedy parity, real target-KV movement, residency lifecycle, and native
TP4/16K bounded GPU residency within their recorded scopes. They do not
provide a current-head, source-closed, end-to-end observation that the
selected loaded runtime actually executed the blockwise online-softmax path.

### Qwen3.5 native MTP

| Explicit gate | Artifact | Raw evidence | Verdict |
| --- | --- | --- | --- |
| TP1/4K engine, batch 1/4 | `artifacts/qwen35_native_mtp_tp1_4k_engine/opaque-57a3a62810d43636b96295da/local-authority/result.json` | Exact parity, accepted/rejected receipts, zero accepted-prefix replay, cleanup to zero | `ACHIEVED` |
| TP4/4K engine, batch 1/4 | `artifacts/qwen35_native_mtp_tp4_4k_engine/opaque-95aa0889f8365beac8be2b6f/artifacts/authority/result.json` | Baseline/native and TP1/TP4 parity, transactional receipts | `ACHIEVED` |
| TP4/16K target-KV offload, batch 1/4 | `artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/lifecycle-release-fix-20260814-2/artifacts/authority/result.json` | Exact parity, real per-rank engine movement, lifecycle cleanup | `ACHIEVED` |
| TP4/32K target-KV offload, batch 1/4 | `artifacts/qwen35_native_mtp_tp4_32k_target_kv_offload/native-mtp-tp4-32k-20260814-2/` | Batch 4 passes, batch 1 fails exact parity | `FAILED` |
| Controlled native-MTP performance | No retained authority | No TTFT/TPOT/throughput/peak-memory promotion result | `MISSING` |

The TP4/16K batch-4 per-rank movement is real engine movement:

```text
baseline H2D:
  10752 copies
  67645734912 bytes

native MTP H2D:
  6762 copies
  42542825472 bytes

native MTP D2H:
  273 copies
  1717567488 bytes
```

This authority proves **target-KV** offload. It does not prove Proposal-KV
offload.

The TP4/32K failure is retained and cannot be overridden by the passing
batch-4 cell:

```text
baseline batch 1:
  [220, 15, 15, 15, 15, 15, 15, 15]

native MTP batch 1:
  [220, 15, 15, 220, 15, 15, 220, 15]

differing indices:
  3, 6
```

The retained payload also shows ordinary baseline batch-shape drift at 32K:
the same prompt produces different later decode behavior when
`max_num_seqs` changes from 1 to 4. Offline token comparison establishes the
first output-level divergence at zero-based generated-token index `3`, with a
second divergence at index `6`. It does not establish the first internal
logit/state/KV divergence or the root cause.

The failed cell JSON files do not retain the step observations or full logits,
but the corresponding `.log` files do retain compact rank-0 top-5
`AUTHORITY_TARGET_LOGITS` rows. These rows must be aligned by engine
observation semantics rather than by global log-line number:

- the worker reads `last_step_logits()` after `engine.step()`;
- for ordinary baseline execution it associates those rows with sequences
  whose `new_completion_tokens_by_seq` entry is non-empty;
- an ordinary sampled step appends one completion token per sampled sequence;
- batch 4 therefore emits four separate final-prefill observations for
  sequence IDs `0`, `1`, `2`, and `3` before joint decode observations begin.

Aligning sequence 0 by its own non-empty completion observation order gives:

```text
generated-token prediction index 0:
  baseline:b1 log line 49
  baseline:b4 log line 49
  compact top-5 rows exactly equal
  top-1 token 220 in both cells

generated-token prediction index 1:
  baseline:b1 log line 50
  baseline:b4 log line 53
  compact top-5 rows differ
  top-1 token 15 in both cells

generated-token prediction index 2:
  baseline:b1 log line 51
  baseline:b4 log line 54
  compact top-5 rows differ
  top-1 token 15 in both cells

generated-token prediction index 3:
  baseline:b1 log line 52
  baseline:b4 log line 55
  compact top-5 rows differ
  top-1 tokens diverge: 15 versus 220
```

This establishes the first **retained compact target-logit** drift at
generated-token prediction index `1` and the first retained compact top-1
and output divergence at index `3`. It still does not establish whether an
earlier hidden-state, recurrent-state, KV-state, movement, or synchronization
divergence occurred before the LM-head observation at index `1`.

The native-MTP compact rows have a stricter limitation. When speculative
sequences are selected, the worker labels `last_step_logits()` with
`speculative_selected_seq_ids` instead of token deltas. In the archived
ModelRunner, ordinary sampled execution and the speculative first-target
callback write `_last_step_logits_cpu`; `run_spec_verify_batch()` does not.
Consequently, the retained native rows are engine-step-level rank-0 snapshots,
not retained verify-tail logits. A speculative engine step may commit multiple
completion tokens, so native log occurrence index is not generated-token
prediction index. The native batch-4 log also has four post-prefill
sequence-set rows while its rank snapshot records only three first-target
callbacks; without the omitted step observations, the compact rows alone
cannot fully attribute every post-prefill row to ordinary versus speculative
execution. The native logs therefore cannot localize the first per-token
verify-tail divergence.

A local cross-bundle search does not provide an independent oracle for the
failed prompt. Among `112` filesystem-visible JSON files that mention the
retained target-model
manifest digest, only four cell payloads contain the exact 32K prompt digest

```text
81db4f7843d9d33bf990270ddf165172b1b31c1196f02fb39771bbc067c63493
```

and all four are the cells in this same failed bundle. Searching the prompt
digest directly across all local artifact JSON produces the same four files.
Therefore neither the batch-1 baseline output nor the batch-4/native output
has an independent same-input local reproduction.

The batch-4/native output digest

```text
7b4af449f7b0aa8710b6047cd9862e666314fe9a6c9212733ba9461a3abed276
```

is not an input-bound authority identity. It occurs in `36` local cell output
rows: `33` use a different 16K prompt, and the only three 32K occurrences are
the current failed bundle's `baseline:b4`, `native_mtp:b1`, and
`native_mtp:b4` rows. Reusing that output digest at 16K therefore does not
independently support the 32K branch.

There is a separate passing Qwen3.5 generic-speculative TP4/32K authority with
the same target-model manifest digest:

```text
artifacts/qwen35_generic_speculative_tp4_32k/
  opaque-03a0a96654a14441b314800f/
```

Its verifier is `PASS`, and its baseline batch-1/batch-4 prompt-0 outputs are
equal. However, it uses prompt digest

```text
c9f2329c2b269097b20bb6d7adf53223d69b2888fc665cda0387752755924516
```

and output digest

```text
a637cd093389926d8402852226a2f183df26da5685b6ef260acc2497e25cf287
```

so it proves that this checkpoint can preserve TP4/32K batch parity on a
different workload; it does not select the correct output for the failed
prompt. The earlier native-MTP 32K attempt
`native-mtp-tp4-32k-20260814-1` retains no result or cells and verifies
`FAIL` with `result is missing`, so it also provides no oracle.

Comparing the two retained campaign identities narrows the transfer strength
further. Both launch scripts use the same Python executable and target-model
path, and both bind the same target-model manifest digest. The generic PASS
worker's manifest-bound 32K overlay reconstructs the following effective
ordinary-baseline configuration:

```text
tensor_parallel_size:               4
enforce_eager:                      true
max_model_len:                      33024
max_num_batched_tokens:             132096
max_num_prefill_tokens_per_step:    1024
max_num_seqs:                       1 or 4
kvcache_block_size:                 256
chunked_prefill_decode_first:       false
chunked_prefill_mixed_batch:        false
kv_offload_mvp0:                    true
kv_offload_gpu_blocks:              68
kv_offload_logical_blocks:          640
kv_offload_blockwise_prefill:       true
kv_offload_blockwise_decode:        true
kv_offload_blockwise_blocks:        8
qwen35_mtp_enabled:                 false
qwen35_mtp_cuda_graphs:             false
qwen35_mtp_max_proposal_tokens:     4
```

This matches the failed native bundle's recorded baseline `engine_config`
apart from the evidence form: the failed bundle records the complete mapping
inside each cell, while the generic cells omit it and require reconstruction
from the source-manifest-bound overlay, base worker, and byte-identical
`tinyvllm/config.py` defaults. Both bundles also record positive real H2D and
D2H movement. Thus an obvious 32K capacity, chunking, eager-mode, or target-KV
offload parameter mismatch is not the differentiator between these two
historical baselines.

The comparison is still not controlled:

```text
generic PASS prompt SHA:
  c9f2329c2b269097b20bb6d7adf53223d69b2888fc665cda0387752755924516
failed native prompt SHA:
  81db4f7843d9d33bf990270ddf165172b1b31c1196f02fb39771bbc067c63493

generic PASS source tree:
  f4d4c684a39bd404fc32821a6d4a8997c4ca36adbe28526ef2fbfab8d5cd54da
failed native source tree:
  d722bc58f309c21695ea406035d69638d87337e7bac6ce0779e8848eb92fa6b8

generic manifest files:              16
failed native manifest files:       115
generic paths present in native:     11 / 16
matching hashes among generic files:  8 / 16

generic GPU order:  [7, 5, 3, 2]
failed GPU order:   [3, 2, 1, 0]
```

The retained hardware metadata narrows, but does not close, that GPU/rank
dimension. A fail-closed comparison of both `gpu_inventory.csv` files,
`selected_gpu_indices.txt` files, campaign scripts, and the four ordinary
baseline logs establishes:

```text
inventory schema:
  index,memory.free,memory.total,utilization.gpu
inventory index set in both bundles:
  [0, 1, 2, 3, 4, 5, 6, 7]
memory.total for every indexed device in both bundles:
  81920 MiB
utilization.gpu for every indexed device in both snapshots:
  0
exact inventory-row overlap:
  indices [0, 1, 2, 4, 6]
rows differing only in the retained free-memory snapshot:
  indices [3, 5, 7]
selected-device overlap:
  indices [2, 3]
campaign Python executable:
  identical
campaign target-model path:
  identical
unique libibverbs missing-driver warning signature:
  identical, 11 lines
```

The inventory records no GPU UUID, GPU name, PCI bus ID, driver version,
CUDA runtime version, PyTorch version, NCCL version, or hostname. It therefore
supports the same eight-index topology and the same 80-GiB capacity class, but
does not prove exact physical-device identity, exact host identity, or exact
software-stack identity. The different selected orders remain a real rank-to-
device mapping difference, with only two selected indices shared, so a
hardware-by-rank or prompt-by-hardware interaction remains unexcluded.

In particular, `tinyvllm/engine/llm_engine.py`,
`tinyvllm/engine/model_runner.py`, and
`tinyvllm/engine/speculative_model_runner.py` have different manifest hashes.
Their archived diffs are substantial by line count (`218` insertions and `1`
deletion in `llm_engine.py`; `227` insertions and `25` deletions in
`model_runner.py`), but a function-level reachability audit materially narrows
their causal strength for the retained ordinary baseline:

- every changed `speculative_model_runner.py` function is proposal,
  side-state, or proposal-lifecycle plumbing and is unreachable when the
  speculative runtime is absent;
- the changed `llm_engine.py` branches are likewise speculative publication,
  side-state, proposal release, or runtime-lifecycle handling. The ordinary
  `model_runner.call("run", ...)` and `scheduler.postprocess(...)` path is
  unchanged; the added postprocess lifecycle checks reduce to no-ops when
  `runtime is None`;
- archived `ModelRunner.run`, `_run_model_step`, `prepare_prefill`,
  `prepare_decode`, `prepare_mixed`, `run_model`,
  `_kv_offload_before_forward`, and `_kv_offload_after_forward` function
  bodies are byte-identical between the two source snapshots;
- the tensor-parallel greedy selector change is confined to
  `run_spec_first_target_and_proposal_batch`, not ordinary `ModelRunner.run`,
  and is therefore unreachable for the recorded baseline cells with
  `qwen35_mtp_enabled=false`;
- the only manifest-bound ordinary-baseline-reachable `model_runner.py`
  change is a call to `_record_peak_resident_blocks()` after slot assignment.
  That helper reads `logical_to_slot`, raises only if resident blocks exceed
  capacity, and otherwise updates a statistic. Both retained baseline cells
  completed all four ranks with `runtime_poisoned=false` and recorded
  `peak_resident_blocks == gpu_blocks == 68`, so this successful telemetry
  path does not mutate logits, sampling, or token IDs.

The broader archived source tree also shows a `Sequence.__getstate__` /
`__setstate__` change that transports `max_tokens` to worker ranks. This file
is not included in the generic bundle's 16-file source manifest, so it is not
manifest-bound evidence for that authority. In the archived implementation it
is ordinary transport-reachable, but worker-side ordinary model execution
does not read `max_tokens`; rank 0 keeps the original sequences and owns
scheduler termination. It therefore does not identify a mechanism for the
observed first-eight-token logit/output drift, while still preventing a claim
of complete source identity.

The prompt, source provenance, rank-to-device mapping, and exact retained
hardware/software identity therefore remain simultaneously uncontrolled.
The capacity class and eight-index topology are aligned, while the source
difference is now classified as a residual provenance confound rather than an
identified ordinary token-affecting path difference. The PASS bundle is useful
only as a weak counterexample to the checkpoint contents being a universally
sufficient cause of every TP4/32K batch divergence. It cannot authenticate
either failed-prompt output, exclude a prompt-by-source or prompt-by-hardware
interaction, identify the first internal divergence, or establish root cause.

The failed bundle's source provenance is internally self-consistent within
its recorded manifest boundary:

```text
source manifest files:                    115
missing source files:                       0
per-file SHA-256 mismatches:                0
recorded source_tree_sha256:
  d722bc58f309c21695ea406035d69638d87337e7bac6ce0779e8848eb92fa6b8
recomputed source_tree_sha256:
  d722bc58f309c21695ea406035d69638d87337e7bac6ce0779e8848eb92fa6b8
aggregate match:                         true
source.tar regular files:                 258
manifest files absent from source.tar:      0
unmanifested source.tar files:             143
unmanifested non-pyc files:                  0
```

The aggregate was recomputed with the archived gate's exact algorithm:
sorted UTF-8 file name, 8-byte big-endian name length, file-name bytes, then
the raw 32-byte file digest. All `143` extra tar members are
`__pycache__/*.pyc`. Therefore the manifest closure is complete and its
aggregate digest is valid, but `source.tar` is a superset rather than an
exact manifest-only archive. The failed campaign manifest is also a
campaign-start partial manifest without the successful authority schema's
`artifacts` field, so it cannot be treated as a complete PASS authority
bundle.

The four retained cell payloads are internally paired as intended:

```text
cells present:
  baseline:b1
  native_mtp:b1
  baseline:b4
  native_mtp:b4

shared across all four:
  schema
  model identity
  world size 4
  GPU indices [3, 2, 1, 0]
  rank inventory
  prompt token count 32768
  max output tokens 8

same-batch baseline/native prompt rows:
  exact equality

batch-1 prompt vs batch-4 row 0:
  exact equality

batch-4 prompt rows:
  four unique rows

same-batch engine-config difference:
  qwen35_mtp_enabled only

same-policy batch-1/batch-4 engine-config difference:
  max_num_seqs only
```

This rules out a mismatched prompt, model identity, rank assignment, or
unrelated engine-config drift as an explanation for the retained output
comparison. It does not prove which output is correct. Because the partial
failed manifest has no `artifacts` map, the four cell JSON/log files are also
not SHA-bound by `source_manifest.json`; their present local hashes are
observable retained-file identities, not a complete authority publication
contract.

The target-model and MTP-checkpoint digest references are consistent across
all four cells, the failed source manifest, and the frozen gate constants:

```text
target model manifest SHA-256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0

MTP checkpoint manifest SHA-256:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b
```

However, neither original manifest payload is retained in the failed bundle
or `source.tar`; the only retained manifest-like file is
`artifacts/authority.failed/source_manifest.json`. The digests therefore
provide internally consistent opaque identities, but this bundle alone
cannot independently re-open either manifest, inspect its model/checkpoint
inventory, or recompute those two digests from the original payloads.

Cross-bundle local recovery is asymmetric. The target identity is the
SHA-256 of an actual `model_manifest.json` payload. A filename-targeted scan
found `21` byte-identical copies elsewhere in the current checkout,
including:

```text
experiments/qwen35_hybrid_state/
  qwen35-tp4-strict-p1-readiness-20260806-r551/
  model_manifest.json
```

Each copy is `1258` bytes and hashes to the recorded target digest.

The MTP identity uses a different contract and does **not** correspond to a
standalone manifest file. `checkpoint_manifest_sha256()` recursively sorts
all regular checkpoint files, hashes each payload, serializes one canonical
JSON row per file:

```text
{"path": relative_path, "sha256": payload_sha256, "size": byte_count}
```

and hashes the newline-delimited rows. The original real-checkpoint gate,
current native-MTP worker, and archived 32K worker produce the same digest
when executed over the same read-only probe directory. Current and archived
worker implementations are also AST-identical.

All `25` retained MTP-run JSON files expose only the final aggregate:

```text
checkpoint_manifest_sha256:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b
```

They identify one historical checkpoint directory:

```text
/data00/home/sitian/sitian-workspace01/tllm/
  qwen35-hybrid-state-runs/
  qwen35-2b-hybrid-acquire-20260723-222004/model
```

That directory is not present on the current local machine. Therefore the
algorithm and historical path are established, but the specific
`9a975b...` aggregate cannot be freshly recomputed without access to the
original checkpoint contents. The absence of an MTP manifest JSON is not
itself a defect; the actual self-containment gap is that the failed bundle
retains neither the checkpoint files nor the canonical per-file
path/hash/size rows needed to independently reconstruct the aggregate.

### Independent Qwen3 learned drafter

| Explicit gate | Artifact | Raw evidence | Verdict |
| --- | --- | --- | --- |
| TP1 learned source with Proposal-KV offload | `experiments/autoregressive_draft/tp1-qwen3-loaded-offload-gpu4-20260815/authority_summary.json` | Batch 1/4 exact parity; real Qwen3 draft forwards; zero accepted-entry copy/replay/rematerialization; workload-derived logical capacity 90; real H2D/D2H | `ACHIEVED` within short-context TP1 scope |
| TP4 learned source direct correctness | `experiments/autoregressive_draft/tp4-qwen3-loaded-direct-gpu3467-authority-r2-20260815/authority_summary.json` | Batch 1/4 exact parity on all ranks; direct Proposal-KV ownership; current/archive verifier receipts match | `ACHIEVED` within short-context direct-allocation scope |
| TP4 controlled performance | `experiments/autoregressive_draft/tp4-qwen3-controlled-performance-timing-gpu3467-r4b-20260815/result.json` | Source-bound three-run pilot; complete timing and memory fields | `PASS_PILOT_ONLY_NEGATIVE` |
| TP4 Proposal-KV offload | No retained authority | TP4 authority records direct allocation and zero H2D/D2H | `MISSING` |
| 4K/16K/32K learned-draft contexts | No retained authority | Correctness prompts are 9-13 tokens; performance prompts are 256 tokens | `MISSING` |
| Second independent learned structure | No retained authority | Only the Qwen3 target/draft pairing is established | `MISSING` |

The TP1 Proposal-KV offload authority records:

```text
logical entry capacity:       90
GPU slot capacity:             8
CPU backing capacity:         90
live slots after release:      0
H2D operations:             9838
H2D bytes:             2094891008
D2H operations:               58
D2H bytes:               9519104
real bidirectional movement: true
```

The controlled TP4 timing pilot remains negative:

```text
batch 1:
  TPOT:       +45.90%
  throughput: -29.55%
  acceptance: 100%

batch 4:
  TPOT:       +90.03%
  throughput: -51.70%
  acceptance: 73.61%
```

It is valid evidence that the current learned TP4 path is slower in this
scope. It is not a promotion result.

## Transactional KV Lifecycle

The transactional implementation is real, but retained evidence is not
uniform across proposal sources.

| Source and scope | Direct commit evidence | Rejected suffix release | Accepted-prefix target replay | Cleanup |
| --- | --- | --- | --- | --- |
| Qwen3.5 native MTP TP1/4K | 40/40 receipts preserve accepted slot identity | 40/40 receipts report rejected slots released | Four cell counters are zero | Proposal transactions and slots return to zero |
| Qwen3.5 native MTP TP4/4K | 160 proposal transactions pair with tickets; 440 proposal entries commit | 20 rejecting transactions release 40 entries | 16 rank/cell counters are zero | Active transactions, owned slots, allocated slots, and prepared tickets are zero |
| Qwen3.5 native MTP TP4/16K offload | 52 proposal transactions pair with tickets; 44 proposal entries commit | 36 rejecting transactions release 88 entries | 16 rank/cell counters are zero | Active transactions, owned slots, allocated slots, and prepared tickets are zero |
| Qwen3 independent draft TP4 direct | 88 proposal transactions pair with tickets; accepted-entry copy/replay/rematerialization counters are zero on every retained rank snapshot | 64 rejecting transactions release 160 entries | No dedicated `accepted_prefix_target_replays` or `extra_target_forward_count` receipt | Active transactions, owned entries, and prepared tickets are zero |
| Qwen3 independent draft TP1 offload | Aggregate accepted-entry copy/replay/rematerialization counters are zero | Terminal live slots fall from 46 to zero; no per-transaction release inventory is retained | `extra_target_forward_count=0` | Live slots after release are zero |
| Qwen3.5 generic n-gram TP4/32K | Semantic transactions record `commit_prefix_*_rollback_suffix`; four accepted-prefix replay counters are zero | Rollback decision is recorded | Four cell counters are zero | Cleanup receipts are present |
| Qwen3 generic n-gram TP4/4K | No equivalent transaction, slot-identity, accepted-entry-copy, or accepted-prefix-replay receipts | Not directly retained | Not directly retained | Aggregate cleanup receipts only |

For native-MTP TP4 and independent-draft TP4, the artifact-only verifier
paired every proposal transaction with its Proposal-KV ticket and checked:

```text
commit_entry_count  == max(accepted_proposal_tokens - 1, 0)
release_entry_count == rejected_proposal_tokens
proposal transaction state == committed
KV ticket state             == committed
KV cache transaction state  == committed
```

The subtraction accounts for the verifier-selected target token, which is not
a stored draft Proposal-KV entry. These pairings establish direct commit and
suffix release in the cited native and independent TP4 paths.

This closes the original accepted-KV per-token rematerialization problem in
the strongest native-MTP and independent-draft scopes. It does not establish
the same complete receipt set for every source, topology, graph, precision,
sampling, cancellation, or offload composition. In particular, path-level
success and terminal cleanup are not substitutes for a missing accepted-prefix
target-replay counter or a missing accepted-slot identity receipt.

## GPU/CPU Tiering, Async Prefetch, Batched Copies, and Dirty Writeback

The current tree contains real GPU/CPU tiering and copy mechanisms, and
retained loaded authorities contain real target-KV H2D/D2H bytes. The stronger
claims that those runs exercised safe asynchronous prefetch, copy batching, or
dirty eviction are not retained.

`KVOffloadMVP0` owns:

```text
logical KV blocks
  -> a pinned CPU backing store
  -> a bounded physical GPU-slot map
  -> optional CUDA copy stream and completion events
  -> contiguous H2D/D2H span coalescing
  -> explicit dirty writeback
  -> clean/dirty eviction accounting
  -> prefetch-plan and read/write-block counters
```

The current source creates a dedicated CUDA stream when `async_copy=true`.
D2H makes that stream wait for the current producer stream. H2D waits for an
existing D2H completion event for the same logical block. With
`batch_copy=true`, consecutive logical blocks mapped to consecutive physical
slots are coalesced into one span. `writeback_dirty()` collects all selected
resident dirty blocks and submits one `_enqueue_d2h_pairs()` call.

Fresh dependency-light verification covered the extracted current class and
diagnostic ownership:

```text
tools/test_kv_offload_generation_metadata.py
tools/test_h2d_slot_reuse_manager.py

34 passed in 0.15s
```

A separate AST-loaded probe executed the exact current
`_coalesce_copy_pairs()` and `writeback_dirty()` method bodies:

```text
LOCAL_BATCH_AND_WRITEBACK_METHOD_PROBE=PASS
CURRENT_SOURCE_ASYNC_BATCH_WRITEBACK_MECHANISMS=PASS
```

This establishes current source and dependency-light contracts. It does not
execute production CUDA copies.

Three known loaded authorities were recomputed directly:

```text
artifacts/speculative_residency_boundary/20260812T065636Z/result.json
artifacts/speculative_tp1_parity/20260812T062046Z/result.json
artifacts/speculative_runtime_performance/20260812T085852Z/result.json
```

Across 65 movement nodes, the retained rows contain real positive H2D/D2H
copy counts and bytes. The residency-boundary and TP1-parity rows also retain
`evict_dirty=0`. The performance authority omits that field. None of the
65 movement nodes retain:

```text
async_copy
batch_copy
h2d_batches / d2h_batches
h2d_batch_spans / d2h_batch_spans
writeback_on_evict
```

No retained row has `evict_dirty > 0`. Therefore positive D2H bytes establish
real writeback movement, but do not establish dirty eviction, distinguish
explicit writeback from eviction-triggered writeback in every authority, or
prove that asynchronous or batched execution produced those bytes.

The two historical dirty-eviction filenames remain absent from the checkout:

```text
profile_out/kv_offload_batched_dirty_evict_migration_20260708_r2.json
profile_out/kv_offload_batched_dirty_evict_thrash_20260708_r2.json
```

Their handoff summaries cannot substitute for the missing JSON payload,
source manifest, and verifier receipt.

The production test module could not collect in the current local Python:

```text
python3 -m pytest --collect-only -q tools/test_kv_offload.py

ModuleNotFoundError: No module named 'torch'
```

This is an environment blocker, not a production semantic failure or a pass.

Finally, current static control flow has two unresolved reverse-dependency
risks already identified by the TP4/32K forensic audit:

- a resident block can retain a D2H completion event while the next current
  stream targets the same slot without an explicit wait on that D2H event;
- an H2D can reuse a destination slot without first making the copy stream
  wait for prior current-stream reads of the old slot contents.

These are static defect candidates, not established causes of the retained
TP4/32K divergence. They are sufficient to prevent promotion of the current
asynchronous copy path as fully safe.

The strict boundary is:

```text
real target-KV H2D/D2H movement: established in retained scopes
current async stream/event mechanism: established in source
current contiguous copy coalescing: established in source/local contract
current dirty writeback batching: established in source/local contract
loaded async execution receipt: missing
loaded batched-copy receipt: missing
positive dirty-eviction receipt: missing
fully classified explicit-versus-eviction writeback: missing
safe reverse stream dependencies: not established
```

## Future-Window Eviction and Prefetch

The current tree implements future-aware eviction hints. It does not implement
future-window H2D prefetch in the literal sense of staging a future block
before that block becomes part of the required read window.

For blockwise attention, `_blockwise_read_window_future_hint_blocks()` computes
a capacity-bounded set of later blocks. `_stage_blockwise_read_window()` then:

```text
passes current-window blocks to ensure_resident()
passes current plus future-hint blocks as future_logical_blocks
waits only for current-window blocks
```

`KVOffloadMVP0._victim_score()` uses `future_logical_blocks` only to add a
reuse penalty during victim selection. The future-only blocks are not included
in the `ensure_resident()` block list and therefore do not trigger early H2D.
The full-decode staging helper has the same distinction: visible/read and
write blocks are staged, while the broader future set affects eviction
preference.

A fresh AST-loaded execution of the current helper bodies confirmed:

```text
current future-hint helper probe: PASS
```

The focused current-tree Torch suites could not collect in the local Python:

```text
tools/test_blockwise_attention_planning.py
tools/test_kv_decode_residency_planner_gate.py

ModuleNotFoundError: No module named 'torch'
```

This is an environment blocker rather than a semantic failure or pass.

The strongest retained authority is the source-bound canonical bundle:

```text
experiments/kv_offload/kv-residency-canonical-20260723-9705e7b/
```

Direct recomputation establishes:

```text
complete rows:                         112 / 112
independently verified classification: NO_GO
baseline H2D copies:                   22130
candidate H2D copies:                  22130
baseline evictions:                    22220
candidate evictions:                   22220
H2D improvement:                       0.0000%
eviction improvement:                  0.0000%
multi-prompt movement improvement:     0.0000%
```

All eight correctness pairs retained identical decoded token IDs and decode
logit hashes. The candidate did execute its planner:

```text
decode_cross_layer_hint_blocks:        40950
decode_cross_layer_hint_resident:      40950
decode_cross_layer_hint_retained:      40950
decode_plan_builds:                      900
decode_plan_cache_hits:                24300
decode_plan_identity_invalidations:        0
```

The result is a valid negative result: every hinted cross-layer block was
already resident and retained, but the hints changed neither required
residency lifetime nor movement.

This canonical bundle is historical, not a current-head certification.
The current planner contract and frozen verifier are byte-identical to the
archived candidate versions, but current `tinyvllm/layers/attention.py` and
`tinyvllm/engine/model_runner.py` are not byte-identical; the attention file
has substantial later changes. The historical `NO_GO` remains authoritative
for its frozen source and domain, while current-head correctness and
performance remain uncertified.

The strict boundary is:

```text
capacity-bounded future eviction hint: established
cross-layer hint execution: established in historical source-bound authority
literal early H2D of future-only blocks: not implemented
historical movement benefit: zero
current-head loaded planner authority: not established
future-window prefetch benefit: not established
```

## Prefix Cache Plus CPU-Resident KV

The current tree establishes a dependency-light local composition contract
between ordinary cross-request prefix reuse and CPU-backed KV residency. It
does not retain loaded execution authority for that composition.

The ordinary prefix-cache lifecycle is implemented by `BlockManager`:

```text
hash plus token identity
  -> reusable block lookup
  -> refcount acquisition by another sequence
  -> release to idle reusable state without destroying identity
  -> generation change on physical-slot recycle
```

`KVOffloadMVP0` binds CPU validity to the same block generation. A matching
generation with `cpu_valid=true` can schedule an H2D restore, while a recycled
block generation invalidates stale CPU backing and fails closed. On a cached
prefix prefill, `ModelRunner.prepare_prefill` first stages the old prefix with
`require_valid=true` and only then translates the block table used by the
model.

Fresh dependency-light execution covered these contracts:

```text
tools/test_prefix_kv_offload_integration.py
tools/test_profile_prefix_cache.py

23 passed in 0.47s
```

The focused composition tests establish:

- idle shared-prefix hash/token identity and generation survive refcount
  release;
- same-generation CPU-valid backing schedules an H2D pair;
- physical-slot recycle invalidates stale CPU backing;
- cached-prefix prefill requires valid old-prefix residency before block-table
  translation;
- the prefix profiler and gate schema recompute their synthetic raw rows
  consistently.

This is not real CUDA movement authority. The tests construct
`KVOffloadMVP0` without its loaded runtime initializer, use list-backed
residency state, and record H2D/D2H pairs in Python. The profiler test uses a
fake engine. These paths do not prove pinned CPU memory, asynchronous copy
streams/events, loaded checkpoint parity, physical CUDA H2D/D2H bytes, or a
performance benefit.

The two Torch-dependent hybrid-prefix suites could not even collect in the
current local Python environment:

```text
tools/test_qwen35_hybrid_prefix_cache.py
tools/test_qwen35_hybrid_prefix_acquisition.py

ModuleNotFoundError: No module named 'torch'
```

That is an environment blocker, not a semantic failure or a passing result.

A schema-aware retained-JSON scan found zero occurrences for every direct
loaded-composition receipt requested by this audit:

```text
warm_median_cached_tokens=0
prefix_cache_hits=0
reused_prompt_tokens=0
loaded_prefix=0
cpu_restore=0
prefix_h2d=0
prefix_d2h=0
current_intern_references=0
deduplicated_bytes=0
```

Archived prefix/hybrid-cache source files are source provenance, not proof
that a loaded run exercised cross-request prefix reuse. Likewise,
speculative `accepted_prefix_*` receipts describe accepted tokens inside one
speculative transaction; they are not ordinary cross-request prefix-cache
hits.

The strict boundary is therefore:

```text
local ordinary-prefix identity/refcount contract: established
local generation-bound CPU-backing scheduling contract: established
loaded cross-request prefix hit: not established
loaded multi-owner refcount lifetime: not established
loaded deduplicated-byte accounting: not established
loaded CPU restore: not established
real prefix-specific CUDA H2D/D2H: not established
composition performance benefit: not established
```

## KV4/KV8 Precision Paths and Composition Boundary

The current tree still contains independent KV4 and KV8 storage, cache-write,
and dequantization paths, but the quantized paths remain deliberately excluded
from target-KV offload, blockwise offload attention, and speculative
verification.

Fresh current-head source inspection establishes:

```text
configuration:
  kv_quant_bits is restricted to 0 / 4 / 8
  KV4 requires an even group size and symmetric quantization
  KV8 requires symmetric quantization
  AM compact rejects KV4 and permits FP KV or KV8

storage:
  KV4 uses packed int8 storage with final dimension head_dim / 2
  KV8 uses int8 storage with final dimension head_dim
  both allocate per-group scale tensors

routing:
  KV4 and KV8 use distinct Triton cache-write kernels
  cached prefill and decode select the corresponding dequantizer

fail closed:
  kv_offload_mvp0 requires kv_quant_bits == 0
  blockwise prefill/decode require kv_offload_mvp0
  blockwise offload attention therefore remains FP-only
  spec_verify rejects kv_quant_bits != 0
```

The observed current source hashes for this reconciliation are:

```text
tinyvllm/config.py:
  ed4c900b0000802e25deedba6149ba2d1615c8a9e247a852494c8889d1601874
tinyvllm/engine/model_runner.py:
  9f98d2912e6bb99d3a519f2bc2a83da7ff7ca99df6089d8d66f943abbc69ec28
tinyvllm/layers/attention.py:
  6948b9ed1abd1a127c6dcbb9966c7f03952fd6057c2c8a40bac30b89fe09cf6b
```

Fresh local CPU/focused evidence:

```text
quantized snapshot rejection
spec-verify unsupported-feature rejection
unsupported native mode rejection before reservation
  3 passed in 0.63s

KV4 NumPy reference round-trip:
  group_size=32   max_err=2.9071 <= bound=3.9056
  group_size=64   max_err=3.1450 <= bound=3.9056
  group_size=128  max_err=3.1450 <= bound=3.9056

KV8 CPU reference quantization plus current production
dequant_kv_blocks_q8():
  group_size=32   max_err=0.06396008 <= bound=0.12858975
  group_size=64   max_err=0.06427240 <= bound=0.12858975
  group_size=128  max_err=0.06429148 <= bound=0.12858975
  exact production-dequant parity: PASS
  padded block-table handling: PASS

KV8 cached-prefill quantized routing:
  1 passed in 7.48s
```

The KV8 numerical probe AST-extracted and executed the exact current
`dequant_kv_blocks_q8()` body in an isolated CPU Torch process. Its inputs came
from a CPU reference quantizer matching the current Q8 kernel formula. It did
not execute `store_kvcache_q8_kernel`, Triton, CUDA, a loaded checkpoint, or a
generation workload. The KV4 test likewise executed only the existing NumPy
reference branch; the optional real GPU branch was not run.

The retained-artifact conclusion requires one correction to the older audit.
A strict recursive JSON/JSONL structured-key scan covered:

```text
candidate files:             22526
parsed documents:            2384456
malformed candidate files:   9
heat-tier matching files:    0
quant-config matching files: 296
```

All 296 quant-config matches contain:

```text
kv_quant_bits:       0
kv_quant_group_size: 128
```

There are zero nonzero `kv_quant_bits` occurrences and zero structured
execution fields for quantized cache dtype, KV scale, or quantized-KV
execution. The nine malformed files were checked by raw target-key search and
contain none of the quant/heat keys. Therefore the retained records prove only
that the default unquantized configuration was serialized; they do not prove
KV4/KV8 execution.

Exact boundary:

```text
KV4_CPU_REFERENCE_ROUNDTRIP=ESTABLISHED
KV8_CPU_REFERENCE_ACTUAL_DEQUANT_ROUNDTRIP=ESTABLISHED
KV8_CACHED_PREFILL_ROUTING=ESTABLISHED_LOCAL
KV4_KV8_STORAGE_AND_ROUTING_CONTRACT=PARTIAL
KV4_KV8_SPEC_VERIFY_FAIL_CLOSED=ESTABLISHED_LOCALLY
KV4_KV8_OFFLOAD_BLOCKWISE_COMPOSITION=INTENTIONALLY_REJECTED
KV4_REAL_GPU_TRITON_ROUNDTRIP=NOT_RUN
KV8_TRITON_STORE_KERNEL_ROUNDTRIP=NOT_RUN
RETAINED_NONZERO_KV_QUANT_EXECUTION_ROWS=ZERO
KV4_KV8_LOADED_PARITY=NOT_ESTABLISHED
KV4_KV8_MEMORY_REDUCTION=NOT_ESTABLISHED
KV4_KV8_PERFORMANCE=NOT_ESTABLISHED
KV4_KV8_RETAINED_EXECUTION_ARTIFACT=ABSENT
```

## Per-Layer/Per-Token Heat Tiering

The current target-KV residency manager implements recency- and
future-aware victim selection, not the requested heat-driven precision and
residency state machine.

Current behavior:

```text
KVOffloadMVP0._touch():
  one monotonically increasing last-used clock per physical GPU slot

KVOffloadMVP0._victim_score():
  LRU score
  fixed dirty-block penalty
  fixed future-window penalty
  fixed pending-H2D penalty

population:
  one homogeneous FP KV GPU-staging population
```

A fresh AST execution of the exact current `_victim_score()` method produced:

```text
block_nbytes=100
base:                         5.0
future:                     805.0
dirty plus future:         1205.0
pending plus dirty/future: 1805.0
CURRENT_KV_VICTIM_SCORE_PROBE=PASS
```

This establishes the current fixed-cost eviction bias only. It does not
provide:

```text
layer heat or token heat
access-frequency accumulation or decay
hot / warm / cold block identity
residency-tier or precision-tier ownership
promotion / demotion thresholds
FP <-> KV8 <-> KV4 transitions
tier-aware speculative commit / rollback
```

The current Python source tree contains none of the corresponding structured
state names. The 2,384,456-document retained JSON/JSONL scan found zero
heat-tier structured-key matches after excluding unrelated fields such as
promotion classification, warmup counts, and rank snapshots.

Exact boundary:

```text
KV_LRU_RECENCY_POLICY=ESTABLISHED_IN_SOURCE
KV_FUTURE_DIRTY_PENDING_EVICTION_BIAS=ESTABLISHED_IN_SOURCE
PER_LAYER_KV_HEAT=NOT_IMPLEMENTED
PER_TOKEN_KV_HEAT=NOT_IMPLEMENTED
HOT_WARM_COLD_KV_STATE_MACHINE=NOT_IMPLEMENTED
PRECISION_TIER_TRANSITIONS=NOT_IMPLEMENTED
HEAT_TIER_TRANSACTIONAL_KV_COMPOSITION=NOT_IMPLEMENTED
HEAT_TIER_RETAINED_EXECUTION_ARTIFACT=ABSENT
HEAT_TIER_LOADED_PARITY=NOT_ESTABLISHED
HEAT_TIER_MEMORY_REDUCTION=NOT_ESTABLISHED
HEAT_TIER_PERFORMANCE=NOT_ESTABLISHED
```

## Blockwise/Chunked Prefill and Online-Softmax Attention

The current source contains explicit blockwise online-softmax execution paths:

```text
tinyvllm/layers/attention.py:
  _blockwise_online_prefill_attention()
  _blockwise_online_decode_attention()
  _blockwise_online_spec_verify_attention()

tinyvllm/engine/model_runner.py:
  blockwise-prefill activation and bounded residency staging

tinyvllm/engine/scheduler.py:
  chunked-prefill scheduling and lifecycle
```

Fresh local execution establishes two narrow current-head contracts:

```text
tools/test_qwen35_cached_prefill_eager_attention.py::
  test_prefill_blockwise_matches_dense_gqa_and_bounds_tiles

1 passed in 5.65s

five focused chunked-prefill scheduler/lifecycle nodes:

5 passed in 1.63s
```

The Qwen3.5 test numerically matches tiled GQA blockwise prefill against its
dense oracle while bounding tile sizes. The scheduler tests cover chunk
selection, per-step token limits, decode-first behavior, and chunk lifecycle.
Neither result is a loaded long-context end-to-end observation of the selected
attention backend.

The general current blockwise attention module could not be freshly executed:

```text
python -m pytest -q tools/test_blockwise_attention_planning.py

ModuleNotFoundError: No module named 'flash_attn'
```

This is an environment collection limitation, not a semantic failure and not
a pass.

The retained TP1 blockwise authority is:

```text
artifacts/blockwise_speculative_verifier/
  blockwise-tp1-opaque-17786-19070/
```

Independent recomputation confirms:

```text
result SHA256:
  2ecde42b605f6147a36a424ba12f71cff8a5714ef858bd0a28011706ea1dc600

16384:b1 = PASS
16384:b4 = PASS
32768:b1 = PASS
32768:b4 = PASS

retained classification = PASS / NOT_PROMOTABLE
```

This is historical authority for TP1 Qwen3-0.6B 16K/32K, batch 1/4,
blockwise target-KV offload correctness against the generic n-gram runtime.
The local and remote verification receipts are byte/semantic-equivalent, and
the recorded artifact digest is valid.

It is not current-head certification. The fresh verifier fails closed on
source drift. Recomputing the seven-file source binding gives:

```text
matching source files: 4 / 7

drifted:
  tinyvllm/engine/model_runner.py
  tinyvllm/engine/speculative_residency.py
  tinyvllm/layers/attention.py
```

The retained generic TP4 16K and 32K authorities establish long-context exact
parity, real target-KV movement, and lifecycle within their recorded scopes,
but their 16-file manifests omit:

```text
tinyvllm/config.py
tinyvllm/engine/scheduler.py
tinyvllm/layers/attention.py
tinyvllm/layers/qwen35_full_attention.py
```

They therefore do not bind their passing result to the blockwise attention
implementation.

The native-MTP TP4/16K authority is stronger. Its 112-file manifest includes
the configuration, scheduler, generic attention, and Qwen3.5 attention
implementations. Its retained configuration enables:

```text
kv_offload_blockwise_prefill=true
kv_offload_blockwise_decode=true
kv_offload_blockwise_blocks=8
max_num_prefill_tokens_per_step=1024
gpu_blocks=68
```

Its retained peak/resident block counts are bounded at `65/68` for batch 1
and `68/68` for batch 4. This establishes source/configuration dispatch
binding and bounded residency in that historical TP4/16K scope.

None of the TP1, generic TP4, or native TP4 result schemas directly records a
selected `attention_path`, `blockwise_online` flag, attention backend, kernel
name, or equivalent runtime-path observation. Source/config binding is not a
substitute for direct path telemetry.

The canonical final classification therefore keeps the retained
long-context correctness/movement/lifecycle claims, but limits end-to-end
blockwise online-softmax authority to `PARTIAL`.

## Variable-Q Spec-Verify CUDA Graph

The current tree contains a real exact-family speculative-verifier CUDA Graph
implementation, but its retained execution authority is substantially weaker
than its local contract.

The current implementation is disabled by default. Its default batch allowlist
is `(1, 4)`, while its default query-length allowlist is empty and must be
configured explicitly. The dependency-light tests configure query lengths
`(1, 3)` and exercise exact `(batch size, verifier query length, page-table
width)` identities, capture, warmed replay, eager fallback, replay-failure
quarantine, and transaction-safe private capture scratch.

The current remote producer and independent standard-library verifier define
an eight-family MVP:

```text
tensor parallel size: 1
context length:       4096
KV offload:           false
batch sizes:          1, 4
query lengths:        1, 3
page-table widths:    1, 2
identity:             exact B/Q/W
```

The verifier recomputes eager/graph logits hashes, target tokens,
accepted lengths, final tokens, accepted-prefix KV hashes, one-replay/no-eager-
retry behavior, transaction slot-set equations, replay-failure propagation,
and quarantine. It also supports a controlled performance section with five
warmed measurements per exact family and excludes capture latency from warmed
hit latency.

No retained PASS artifact from that producer exists in the checkout. A
schema-aware search found zero exact-family artifacts. The only two records
under `experiments/spec_verify_cuda_graph/` are `BLOCKED` preflights; both
record `source_upload_started=false` and `cuda_gate_started=false` because no
idle GPU was available. Consequently, neither correctness nor optional
performance execution occurred.

The retained legacy Qwen3.5 artifact is:

```text
artifacts/qwen35-mtp-runs/qwen35-mtp-graph-gate-opaque-7/
  qwen35_mtp_real_checkpoint_gate.json
```

It records TP1, no offload, no long-context coverage, no performance claim,
batch sizes `(1, 4)`, legacy Q values `(1, 2, 3, 4)`, six captures, twelve
replays, graph/eager equality booleans, and 28 transaction rows. The complete
`(batch, Q, accepted=0..Q)` transaction domain independently satisfies:

```text
staged    = batch * max(Q - 1, 0)
committed = batch * max(accepted - 1, 0)
released  = staged - committed
committed and released are disjoint
committed union released equals staged
```

However, the directory contains only that JSON. It has no source manifest,
aggregate source hash, archived source, frozen verifier, performance section,
or per-family eager/graph token, logit, capture, replay, and backend rows.
Therefore its transaction equations are independently recomputable, while its
graph/eager parity and six/twelve capture/replay distribution remain
producer-asserted.

Current admission makes the unsupported topology combinations explicit:
`world_size != 1`, `kv_offload_mvp0`, and blockwise prefill/decode all select
eager fallback instead of the spec-verify graph. Long-context execution is not
covered by the exact-family gate contract or by any retained graph PASS
artifact.

## Wider Objective Coverage

| Objective item | Verdict | Evidence boundary |
| --- | --- | --- |
| Logical KV page / physical GPU slot decoupling | `ACHIEVED` as core architecture | Physical-slot allocators and residency managers are exercised by loaded authorities |
| GPU/CPU tiering and real migration | `PARTIAL_WITH_REAL_MOVEMENT_AND_LOCAL_MECHANISM_ONLY` | Real target-KV H2D/D2H bytes are retained and current async/batch/writeback mechanisms have local contracts; loaded async/batch receipts, positive dirty-eviction authority, writeback classification, and safe reverse stream dependencies are missing |
| Prefix sharing, dedup, refcount, CPU backing | `PARTIAL_WITH_LOCAL_CONTRACT_ONLY` | Ordinary hash/token reuse, refcount release, generation binding, and CPU-residency scheduling have local focused coverage; loaded cross-request hits, multi-owner lifetime, deduplicated bytes, CPU restore, and real prefix-specific CUDA movement are not retained |
| KV8/KV4 | `PARTIAL` | Storage/routing and CPU numerical tests exist; no loaded parity, offload, memory, or performance authority |
| Per-layer/per-token heat tiering | `MISSING` | No retained implementation/authority for hot/warm/cold precision-residency transitions |
| Blockwise/chunked prefill and online-softmax attention | `PARTIAL` | Current source contains blockwise prefill/decode/spec-verify online-softmax dispatch; the Qwen3.5 tiled-prefill dense oracle and five chunked-scheduler lifecycle tests pass fresh, while retained TP1 and TP4 authorities establish long-context parity, real target-KV movement, lifecycle, and bounded residency in narrower scopes | The general blockwise attention suite is environment-blocked by missing `flash_attn`; the retained TP1 source binding now matches only 4/7 files; generic TP4 manifests omit the attention implementation; and no retained result directly records the selected blockwise attention path or kernel |
| Future-window eviction/prefetch | `PARTIAL_WITH_EVICTION_HINT_ONLY` | Capacity-bounded future/cross-layer hints influence victim selection; current helpers do not stage future-only blocks early, and the historical 112-row canonical gate was source-bound `NO_GO` with exactly zero H2D/eviction improvement |
| Prefix cache plus CPU-resident KV | `PARTIAL_WITH_LOCAL_CONTRACT_ONLY` | The local composition contract is established, but no loaded authority records ordinary prefix reuse plus CPU restore, real CUDA bytes, parity, or performance |
| Variable-Q CUDA Graph | `PARTIAL` | Local exact-family contract plus legacy TP1/no-offload single-JSON evidence; no current exact-family PASS, source-closed authority, long-context run, or performance run, while TP4/offload/blockwise explicitly fall back to eager |
| Independent Qwen3 draft exact-shape CUDA Graph | `INCONCLUSIVE_ENVIRONMENT` | Local exact TP4/B4/Q4 graph policy, backend, scratch ownership, executor integration, and source-bound gate contract are established; the remote preflight stopped before source upload because four clean GPUs and the prior Python/model environment were unavailable |
| Verifier/sampling/KV-commit fusion | `MISSING` | Ordered phases exist, but no fused runtime/kernel or launch-count attribution |
| TP collective overlap/fusion/ReduceScatter | `MISSING` | Current authorities use synchronous collectives; no optimization-specific artifact |

## Superseded Earlier Conclusions

The following older statements are no longer current:

```text
INDEPENDENT_QWEN3_DRAFT_TP4_REAL_CHECKPOINT_AUTHORITY=NOT_ESTABLISHED
LEARNED_DRAFTER_TP4_LOADED_EXECUTION=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PROPOSAL_KV_OFFLOAD=NOT_ESTABLISHED
```

They are replaced by:

```text
INDEPENDENT_QWEN3_DRAFT_TP1_PROPOSAL_KV_OFFLOAD=ESTABLISHED_IN_RECORDED_SCOPE
INDEPENDENT_QWEN3_DRAFT_TP4_DIRECT_CORRECTNESS=ESTABLISHED_IN_RECORDED_SCOPE
INDEPENDENT_QWEN3_DRAFT_TP4_PERFORMANCE=PILOT_ONLY_NEGATIVE
PROPOSAL_KV_OFFLOAD=PARTIAL_TP1_SHORT_CONTEXT_ONLY
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
LEARNED_LONG_CONTEXT_MATRIX=NOT_ESTABLISHED
```

These updates improve Phase 1 completeness materially, but they do not change
the final promotion decision.

## Why Phase 1 Is Not Achieved

Phase 1 remains incomplete for four independent reasons:

1. **A required correctness cell is failed.** Native Qwen3.5 MTP TP4/32K
   batch 1 does not preserve exact greedy parity, and ordinary 32K baseline
   behavior is not batch-shape invariant in the retained artifact.
2. **No learned proposal path closes the promotion matrix.** Native MTP
   reaches 16K but fails at 32K and lacks controlled performance. The
   independent Qwen3 draft is real, but its correctness runs are short-context,
   TP4 has no offload, and its controlled performance is negative.
3. **The metrics are not complete for a learned promotable configuration.**
   Generic n-gram has positive controlled results, but learned/native routes do
   not jointly establish TTFT, TPOT, throughput, peak memory, acceptance, and
   real movement over the required matrix.
4. **Composition and provenance remain partial.** KV4/KV8 plus offload,
   variable-Q graph plus long-context/offload, all unsupported combinations,
   and complete producer/archive binding are not uniformly established.
   Historical source-bound authorities also do not certify the current
   working tree: fresh verification found source identity, source-inventory,
   and schema-version drift across key retained results.

Therefore, passing generic coverage cannot be used as a proxy for learned
runtime completion, and passing local contracts cannot be used as a proxy for
loaded CUDA authority.

## Single Next Critical Path

The next correctness path is to establish and fix the first cause of the
Qwen3.5 native-MTP TP4/32K batch-shape parity failure before adding more
optimization work:

1. do not change the frozen TP4/32K authority workload or claim a speculative
   verifier root cause yet;
2. retain the established local focused-H2D diagnostic contract implemented
   from:
   `docs/superpowers/plans/2026-08-14-qwen35-tp4-32k-h2d-slot-reuse-causal-diagnostic.md`;
3. treat the dependency-light local suite as contract evidence only
   (`121 passed in 4.35s` on 2026-08-16, including the source-bound campaign
   contract); the plan's original CUDA-extension suite remains
   environment-blocked and is neither a pass nor a failure;
4. retain the established local source-bound contract from
   `docs/superpowers/specs/2026-08-16-qwen35-tp4-32k-h2d-source-bound-campaign-design.md`
   and
   `docs/superpowers/plans/2026-08-16-qwen35-tp4-32k-h2d-source-bound-campaign.md`;
   it deterministically binds the complete `tinyvllm` Python tree and the
   recursively discovered focused producer/verifier helper closure, which was
   137 files in fresh local verification;
5. do not confuse that contract with launch readiness: the authorization-first
   executor invokes only an explicitly injected callback and intentionally has
   no built-in SSH/subprocess/GPU transport; no real checkpoint-bound campaign
   plan or active authorization has been created;
6. obtain the separate exact written execution authorization below, then bind
   a real checkpoint manifest, repetitions, the frozen GPU inventory
   `[0,1,2,3]`, ports, and `/dev/shm` output path into one prepared plan and
   provide the reviewed command-runner callback without changing the plan:

```text
允许只运行一个 source-bound focused-H2D four-cell campaign
```

7. run only the authorized frozen Qwen3.5 diagnostic cells, starting with
   ordinary `baseline:b1` and `baseline:b4`, and retain prediction-index 0/1
   compact logits, physical-slot occupancy generations, H2D predecessor-read
   timing, movement inventories, and software/driver identity;
8. if that focused hypothesis is rejected or incomplete, separately approve
   the broader Qwen3.5 paired verify trace covering
   `baseline:b1`, `native_mtp:b1`, `baseline:b4`, and `native_mtp:b4`;
9. implement only the smallest evidence-grounded correction;
10. rerun the frozen TP4/32K authority unchanged; and
11. create fresh current-source manifests and independent verifier receipts
   for the corrected runtime rather than reusing historical green receipts;
12. only after exact parity and current-source verification are green, run
   controlled native-MTP performance and extend the independent learned
   drafter to TP4 Proposal-KV offload and long contexts.

The autoregressive-draft paired-stability analyzer, verifier, tests, and
remote wrapper are a separate Qwen3 independent-drafter performance-stability
protocol. Its `AB, BA, BA, AB` labels both run the same learned policy at batch
4; it does not execute the Qwen3.5 TP4/32K four-cell correctness matrix and
cannot establish that parity failure's first divergence or root cause. That
remote bundle has not run, and no replication bundle is authorized.

## Focused-H2D Source-Bound Local Contract

The approved local-only source-bound work is now implemented in:

```text
tools/qwen35_tp4_32k_h2d_slot_reuse_source_bundle.py
tools/qwen35_tp4_32k_h2d_slot_reuse_campaign.py
tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_authorization.py
tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_executor.py
tools/test_qwen35_tp4_32k_h2d_slot_reuse_campaign.py
```

Prompt-to-artifact checklist:

| Approved local requirement | Concrete evidence | Verdict |
| --- | --- | --- |
| Complete producer/verifier closure | Full `tinyvllm/**/*.py` plus recursively discovered local tool imports and literal `.py` dynamic loads; fresh closure count 137 | `ESTABLISHED_AS_LOCAL_CONTRACT` |
| Deterministic inventory/tree/tar | Sorted exact members, per-file SHA-256, length-delimited tree SHA-256, fixed tar metadata; two independently built bundles compare equal | `ESTABLISHED` |
| Dynamic frozen 32K worker binding | Tests require both TP4/32K and dynamically loaded TP4/16K workers plus focused gate/worker/verifier | `ESTABLISHED` |
| Four exact cells | Plan freezes `observe:b1`, `observe:b4`, `control:b1`, `control:b4` | `ESTABLISHED` |
| Repetition/GPU/port/path/model binding | Canonical plan and authorization bind repetitions, `[0,1,2,3]`, distinct ports, model/checkpoint manifest, and output under the approved `/dev/shm` root | `ESTABLISHED_AS_CONTRACT` |
| `/data00` write prohibition | Plan accepts `/data00` only for existing Python/model inputs; remote run output is derived only under `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815` | `ESTABLISHED` |
| Exact separate execution authorization | Exact text is frozen in plan and authorization modules; wrong or extended text fails | `ESTABLISHED` |
| Plan-bound, nonce-bound, single use | Canonical plan SHA and all source/checkpoint/workload identities are bound; unsafe nonce, tamper, reuse, and existing output fail closed | `ESTABLISHED` |
| Local prepare/validate CLI | `prepare` freezes GPU `0,1,2,3` and writes only bundle/plan output; `validate` reads and verifies an existing plan; neither creates authorization nor invokes transport | `ESTABLISHED` |
| Authorization before any execution callback | Executor consumes by rename-first and creates the consumed record before invoking the injected callback | `ESTABLISHED` |
| Partial-attempt replay prevention | Callback failure leaves authorization consumed | `ESTABLISHED` |
| No local implementation-time remote/GPU action | Executor contains no built-in subprocess/socket/SSH/torch transport; tests use local callbacks only | `ESTABLISHED` |
| Real campaign launch | No concrete real-checkpoint plan, command-runner transport, active authorization, SSH, GPU, CUDA, or NCCL execution exists | `NOT_APPROVED_NOT_RUN` |

Fresh verification:

```text
focused diagnostic + source-bound campaign contract:
  121 passed in 4.35s

new source-bound modules and test py_compile:
  PASS

fresh source bundle:
  SOURCE_FILE_COUNT=137
  SOURCE_TREE_SHA256=1e60c48ca519e87af9c1d14c6c270c14186030566a0f7300bf1db334ce30f5ac
  SOURCE_TAR_SHA256=9dd2c75ed262f6a762839d58c5c6d8bd32977da45541a046f0592f8607aa5f73
  SOURCE_CLOSURE_CHECK=PASS
```

This replaces the earlier launch-readiness statement that code-enforced
authorization and producer provenance were wholly absent. The narrower current
boundary is:

```text
FOCUSED_H2D_SOURCE_BOUND_LOCAL_RUNNER=ESTABLISHED
FOCUSED_H2D_CODE_ENFORCED_AUTHORIZATION=ESTABLISHED_AT_CALLBACK_BOUNDARY
FOCUSED_H2D_COMPLETE_PRODUCER_PROVENANCE_CONTRACT=ESTABLISHED_137_FILES
FOCUSED_H2D_BUILTIN_REMOTE_TRANSPORT=ABSENT
FOCUSED_H2D_REAL_CHECKPOINT_PLAN=NOT_CREATED
FOCUSED_H2D_ACTIVE_EXECUTION_AUTHORIZATION=NOT_CREATED
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
```

## Current-Head Dependency-Light Runtime Regression

The current-tree drift audit was followed by a focused local regression over
Proposal-KV, speculative selection/publication, Scheduler postprocess,
learned-draft allocator contracts, and native-MTP integration:

```text
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_proposal_kv_residency_local_gate.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_selection_record.py \
  tools/test_scheduler_speculative_selection.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_autoregressive_draft_proposal_kv_allocator_contract.py \
  tools/test_qwen35_mtp_executor_leases.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_physical_kv.py

result:
  264 passed in 1.78s
```

The first run was `263 passed, 1 failed`. The failure was not a runtime
behavior regression. The test
`test_step_merges_selected_and_suppressed_rows_in_schedule_order` still
required the complete public timing mapping to equal only:

```text
{"draft_proposal_ms": 1.0}
```

Current transactional publication intentionally adds the non-negative
`commit_metadata_ms` phase to the prepared runtime timing mapping before
`LLMEngine.step()` copies it into
`last_step_observation["speculative_runtime_timing_ms"]`. This is required by
the runtime timing design, separately tested by the publication helper, and
consumed by performance gates. The stale exact-dictionary assertion was
therefore tightened to require both timing keys, preserve the exact draft
proposal value, and validate a non-negative metadata-commit duration. The
single reproduction then passed, followed by the full 264-test pass above.

The local interpreter boundary remains:

```text
/usr/bin/python3:
  ModuleNotFoundError: No module named 'torch'
```

Therefore, this result establishes dependency-light current-head contracts
only. Torch/CUDA-dependent collection and loaded model execution remain
environment-blocked locally and are neither passes nor failures. It does not
repair historical source drift, certify current-head GPU behavior, or change
the Phase 1 promotion decision.

An extended drift-linked pass then added engine publication, immutable
selection-record, source-contract, proposal-prefill observation, MTP ownership
gate, attention-marker, and context-mode coverage. It exposed a second real
current-head contract mismatch:

```text
test_partition_preserves_selected_and_suppressed_order:
  expected selected sequence IDs: (8, 2)
  actual selected sequence IDs:   (8, 4, 2)
```

Sequence `4` had exactly one remaining output token. The authoritative
selection design and handoff require at least two remaining output tokens
before selecting a row, because the ordinary first-target decode already
satisfies a one-token remainder and there is no multi-token speculative
opportunity. `build_speculative_selection_record()` incorrectly suppressed
only a zero-token remainder. The production condition was corrected from
`remaining_output_tokens == 0` to `remaining_output_tokens < 2`.

Two stale selection-record assertions that had drifted to the opposite
single-token behavior were aligned with the authoritative contract:

```text
single-token remainder:
  selected=False
  max_proposal_tokens=0
  suppression_reason=insufficient_output_budget

selected sequence validation:
  returns only selected rows in stable scheduled order
```

Fresh focused and merged verification:

```text
selection + engine publication focused suite:
  121 passed in 0.90s

extended dependency-light drift-linked suite:
  121 passed in 0.52s

merged unique current-head dependency-light suite:
  385 passed in 2.28s
```

`tools/test_qwen35_mtp_executor_graph_registration.py` is not part of the
dependency-light pass: despite having no visible top-level `import torch`, its
executor import reaches `tinyvllm/engine/tensor_parallel_greedy.py` and fails
collection on unavailable `torch.distributed`. It remains correctly
classified as environment-blocked rather than failed.

The next engine/ModelRunner source-contract pass initially produced:

```text
66 passed, 8 failed
```

Seven failures came from AST tests that still inspected the public profiled
wrappers as though they directly contained preparation, model forward, and
context cleanup. Current ModelRunner structure intentionally separates:

```text
run_spec_first_target_batch
  -> run_profiled_step
  -> _run_spec_first_target_batch

run_spec_verify_batch
  -> run_profiled_step
  -> _run_spec_verify_batch
```

The private helpers still prepare once and execute exactly one runtime model
forward. Their source contains two static `run_model` call sites because
Qwen3.5 prepared-state and ordinary execution are mutually exclusive
`if/else` branches. The tests now verify that mutual exclusion, verify that
neither call is nested in per-item iteration, locate the unique `finally`
containing `reset_context`, and separately preserve the public positional RPC
argument order including `kv_block_identity_rows`.

The eighth failure was collection-order pollution in
`tools/test_model_runner_command_ack.py`. That test unconditionally replaced
the canonical `tinyvllm.engine.model_runner_command_ack` module in
`sys.modules`. A previously imported `ModelRunnerCommandEnvelope` then no
longer matched the class registered under its pickle module path, causing:

```text
PicklingError:
  it is not the same object as
  tinyvllm.engine.model_runner_command_ack.ModelRunnerCommandEnvelope
```

The dependency-light loader now creates package stubs only when absent and
reuses the canonical ack module whenever it is already loaded. It no longer
invalidates dataclass identity across test modules.

Fresh verification:

```text
focused AST and ack regressions:
  35 passed in 1.10s

complete engine/ModelRunner source and ack group:
  74 passed in 3.40s

combined current-head dependency-light and source-contract suite:
  459 passed in 6.29s
```

No ModelRunner runtime behavior was changed for these eight failures. They
were stale source-test observation boundaries and test module-isolation
defects.

The final material-drift pass added dependency-light direct coverage for the
batch verifier, native attention contract, variable-proposal CUDA Graph
backend, retained/current Qwen3.5 native-MTP gate contracts, learned-drafter
telemetry, TP4 source-bound verifier/producer contracts, and the generic TP4
speculative gate.

The first 11-file run exposed collection-order pollution:

```text
tools/test_native_verifier_attention.py followed by
tools/test_qwen35_mtp_cuda_graph_backend.py:
  No module named 'tinyvllm.engine.qwen35_mtp_graph';
  'tinyvllm.engine' is not a package
```

`tools/test_native_verifier_attention.py` installed dependency-light package
stubs while dynamically loading `tinyvllm/layers/attention.py`, but retained
those stubs after collection. It now snapshots every touched `sys.modules`
entry, installs the stubs only inside the dynamic-load boundary, and restores
the exact prior module objects in `finally`.

Focused verification:

```text
native verifier attention:
  6 passed in 0.05s

native verifier attention followed by variable-Q CUDA Graph backend:
  27 passed in 0.44s

11-file direct-light material-drift group before the next isolation fix:
  465 passed in 10.61s
```

Combining that group with the prior 459-test suite then exposed a separate
canonical dataclass identity leak:

```text
initial 40-file combined run:
  919 passed, 5 failed in 13.66s

failure:
  PicklingError:
    ModelRunnerCommandEnvelope is not the same object as
    tinyvllm.engine.model_runner_command_ack.ModelRunnerCommandEnvelope
```

The direct-light `tools/test_model_runner_spec_verify.py` loader
unconditionally reloaded `tinyvllm.engine.model_runner_command_ack` during
collection. Ack tests collected earlier retained the original dataclass,
while pickle resolved the replacement module and rejected the identity
mismatch. A focused two-file reproduction returned `114 passed, 4 failed`.
An explicit identity-preservation regression and the pickle round-trip both
failed before the fix.

The loader now imports the command-ack source only when the canonical module
is absent and otherwise reuses the exact preexisting module object. The new
regression asserts that behavior whenever another test collected the module
first. No production runtime code changed.

Fresh current-tree verification:

```text
focused identity regression plus pickle round-trip:
  2 passed in 0.20s

complete command-ack plus ModelRunner verifier pair:
  119 passed in 0.58s

standalone 11-file direct-light material-drift group:
  465 passed, 1 skipped in 8.60s

  The skip is the identity-preservation regression when no command-ack
  module existed before this standalone group. The full combined suite
  exercises and passes that branch.

unique 40-file dependency-light/source-contract/material-drift suite:
  925 passed in 13.87s
```

The 925-test result is the strongest current-tree local host/source-contract
regression in this audit. It proves the tested dependency-light contracts can
coexist in one pytest collection without replacing the native-attention
package boundary or canonical command-ack dataclass identity. It still does
not execute real torch/CUDA kernels, loaded checkpoints, NCCL, TP4/32K,
source-bound focused-H2D movement, or controlled performance measurements.

## 2026-08-17 Independent Qwen3 Draft Exact-Shape CUDA Graph Reconciliation

The current tree now contains a separate default-off CUDA Graph path for
independent Qwen3 learned-draft proposal generation. It is intentionally
narrower than the target-side Variable-Q verifier graph:

```text
TP=4
batch=4
Q=4
sampling=greedy
Proposal-KV=dense direct allocation
Proposal-KV offload=false
padding/rounding=forbidden
```

The implementation covers exact identity and capture budgets, two successful
eager observations before capture, private scratch Proposal-KV transactions,
three draft forward/argmax/broadcast steps with GPU-resident token chaining,
one final host readback, shared eager/graph proposal registration, and TP-wide
pre/post-replay convergence.

Local focused verification establishes the dependency-light architecture and
lifecycle contract. The source-bound gate requires exact eager/graph target,
proposal, accepted-prefix, and transaction equality; zero live transactions;
all-rank replay; no measured fallback/quarantine; positive median throughput;
no median TPOT regression; and a positive paired-bootstrap lower bound.

The read-only remote preflight found all eight GPUs occupied and fewer than
four clean GPUs. The prior Python and model environment was also absent. It
stopped before source upload, model loading, capture, correctness comparison,
or performance measurement and did not alter any remote process.

Strict classification:

```text
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_SCOPE=TP4_B4_Q4_GREEDY_DENSE_DIRECT
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_DEFAULT=OFF
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_LOCAL_CONTRACT=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_REAL_CAPTURE_REPLAY=NOT_ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_REAL_CORRECTNESS_PARITY=NOT_ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_CONTROLLED_PERFORMANCE=NOT_ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_CLASSIFICATION=INCONCLUSIVE_ENVIRONMENT
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Final Classification

```text
GENERIC_RUNTIME_ARCHITECTURE=ESTABLISHED
SOURCE_NEUTRAL_RUNTIME=PARTIAL_WITH_DUAL_ABI_AND_ASYMMETRIC_RETAINED_PROVENANCE
SOURCE_PROPOSAL_ABI=SPLIT_HOST_DRAFT_ADAPTER_AND_MODEL_RUNNER_PROPOSAL_EXECUTOR
SHARED_SPECULATIVE_ORCHESTRATION_VERIFIER_TARGET_KV_COMMIT=CURRENT_TREE_ESTABLISHED
NATIVE_MTP_AND_INDEPENDENT_DRAFT_SHARED_MODEL_RUNNER_EXECUTOR_ABI=ESTABLISHED
CROSS_SOURCE_IDENTICAL_SOURCE_CLOSED_RUNTIME_AUTHORITY=NOT_ESTABLISHED
BATCH_NATIVE_MULTI_TOKEN_VERIFIER=PARTIAL_WITH_FIXED_Q_GROUPING_AND_ASYMMETRIC_RETAINED_RECEIPTS
VERIFY_TAIL_TARGET_FORWARD=ONE_PER_HOMOGENEOUS_QUERY_LENGTH_GROUP
SPECULATIVE_STEP_TARGET_FORWARD_COUNT=SEPARATE_FIRST_TARGET_PLUS_VERIFY_TAIL
HETEROGENEOUS_QUERY_LENGTH_SINGLE_FORWARD=NOT_ESTABLISHED_BY_DESIGN
TARGET_KV_MATERIALIZED_TOKEN_COUNT=PROPOSAL_LENGTH_MINUS_ONE
TARGET_KV_ACCEPTED_MATERIALIZED_COMMIT_COUNT=MAX_ACCEPTED_COUNT_MINUS_ONE
CROSS_SOURCE_RETAINED_QUERY_LENGTH_FORWARD_RECEIPTS=NOT_ESTABLISHED
MTP_TARGET_FORWARD_REDUCTION=MECHANISM_AND_SPECULATIVE_SIDE_COUNTERS_ONLY
NATIVE_MTP_BASELINE_TARGET_FORWARD_COUNTERS=NOT_RETAINED_ZERO_FIELDS_ARE_PLACEHOLDERS
CONTROLLED_NATIVE_MTP_TARGET_FORWARD_REDUCTION=NOT_ESTABLISHED
CONTROLLED_NATIVE_MTP_TPOT_BENEFIT=NOT_ESTABLISHED
POSITIVE_CONTROLLED_TPOT_AUTHORITIES=GENERIC_NGRAM_ONLY
INDEPENDENT_LEARNED_TP4_TPOT_DIRECTION=NEGATIVE_PILOT
GENERIC_NGRAM_TWO_MODEL_TP1_TP4_4K_16K_32K=ESTABLISHED_IN_RECORDED_SCOPES
REAL_MULTI_SEQUENCE_GREEDY_EXECUTION=ESTABLISHED_IN_RECORDED_SCOPES
TRANSACTIONAL_ACCEPT_COMMIT_REJECT_RELEASE=ESTABLISHED_IN_RECORDED_SCOPES
ZERO_ACCEPTED_PREFIX_TARGET_REPLAY=ESTABLISHED_IN_RECORDED_SCOPES

QWEN35_NATIVE_MTP_TP1_4K=ESTABLISHED
QWEN35_NATIVE_MTP_TP4_4K=ESTABLISHED
QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD=ESTABLISHED
QWEN35_NATIVE_MTP_TP4_32K=FAILED_OR_NOT_ESTABLISHED
CONTROLLED_NATIVE_MTP_PERFORMANCE=NOT_ESTABLISHED

INDEPENDENT_QWEN3_DRAFT_TP1_PROPOSAL_KV_OFFLOAD=ESTABLISHED_IN_RECORDED_SCOPE
INDEPENDENT_QWEN3_DRAFT_TP4_DIRECT_CORRECTNESS=ESTABLISHED_IN_RECORDED_SCOPE
INDEPENDENT_QWEN3_DRAFT_TP4_PERFORMANCE=PILOT_ONLY_NEGATIVE
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_SCOPE=TP4_B4_Q4_GREEDY_DENSE_DIRECT
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_DEFAULT=OFF
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_LOCAL_CONTRACT=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_REAL_CAPTURE_REPLAY=NOT_ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_REAL_CORRECTNESS_PARITY=NOT_ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_CONTROLLED_PERFORMANCE=NOT_ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_CLASSIFICATION=INCONCLUSIVE_ENVIRONMENT
INDEPENDENT_DRAFT_LONG_CONTEXT=NOT_ESTABLISHED
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
TWO_TARGET_CHECKPOINT_FAMILIES=RECORDED
QWEN35_TARGET_ARCHITECTURE_PROVENANCE=SELF_CONTAINED
QWEN3_TARGET_ARCHITECTURE_PROVENANCE=PATH_AND_DIGEST_ONLY_NOT_SELF_CONTAINED
TWO_MATERIALLY_DIFFERENT_TARGET_STRUCTURES=PARTIAL_WITH_ASYMMETRIC_PROVENANCE
TRANSACTIONAL_KV_NATIVE_MTP=DIRECT_COMMIT_RELEASE_AND_ZERO_TARGET_REPLAY_ESTABLISHED
TRANSACTIONAL_KV_INDEPENDENT_TP4=COMMIT_RELEASE_ZERO_COPY_REPLAY_REMAT_TARGET_REPLAY_RECEIPT_ABSENT
TRANSACTIONAL_KV_INDEPENDENT_TP1=AGGREGATE_ZERO_COPY_REPLAY_REMAT_AND_ZERO_EXTRA_TARGET_FORWARD
TRANSACTIONAL_KV_GENERIC_QWEN35=SEMANTIC_TRANSACTION_WITH_ZERO_TARGET_REPLAY_NO_SLOT_IDENTITY
TRANSACTIONAL_KV_GENERIC_QWEN3=DIRECT_TRANSACTIONAL_RECEIPTS_ABSENT
TRANSACTIONAL_DIRECT_COMMIT_RELEASE=PARTIAL_WITH_SOURCE_ASYMMETRY
PROMOTION_METRIC_COMPLETE_AUTHORITIES=2_GENERIC_ONLY
PROMOTION_METRIC_QWEN3_GENERIC_TP1_4K=5_RUNS_BATCH1_BATCH4
PROMOTION_METRIC_QWEN35_GENERIC_TP4_16K=5_RUNS_BATCH1_BATCH4
LEARNED_TP4_PERFORMANCE=3_RUN_256_TOKEN_PILOT_NEGATIVE
LEARNED_TP4_PROPOSAL_KV_MOVEMENT=ZERO_DIRECT_ALLOCATION_NOT_OFFLOAD_EVIDENCE
NATIVE_MTP_CONTROLLED_PERFORMANCE=ABSENT
FULL_PROMOTION_METRIC_MATRIX=NOT_ESTABLISHED

GPU_CPU_TIERING=PARTIAL_WITH_REAL_MOVEMENT_AND_LOCAL_MECHANISM_ONLY
REAL_TARGET_KV_H2D_D2H_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPES
ASYNC_COPY_MECHANISM=CURRENT_SOURCE_ESTABLISHED
BATCHED_COPY_MECHANISM=CURRENT_SOURCE_ESTABLISHED
DIRTY_WRITEBACK_MECHANISM=CURRENT_SOURCE_ESTABLISHED
DEPENDENCY_LIGHT_ASYNC_BATCH_WRITEBACK_CONTRACT=34_PASSED_PLUS_METHOD_PROBE
DIRTY_EVICTION_EXACT_ARTIFACT_RECEIPT=MISSING
ASYNC_COPY_RUNTIME_RECEIPT=MISSING
BATCHED_COPY_RUNTIME_RECEIPT=MISSING
EXPLICIT_VS_EVICTION_WRITEBACK_ARTIFACT_CLASSIFICATION=MISSING
ASYNC_D2H_RESIDENT_REWRITE_DEPENDENCY=MISSING_IN_STATIC_CONTROL_FLOW
ASYNC_H2D_SLOT_REUSE_REVERSE_DEPENDENCY=MISSING_IN_STATIC_CONTROL_FLOW
CURRENT_PRODUCTION_KV_OFFLOAD_TEST_COLLECTION=ENVIRONMENT_BLOCKED_NO_TORCH

FUTURE_WINDOW_EVICTION_PREFETCH=PARTIAL_WITH_EVICTION_HINT_ONLY
CAPACITY_BOUNDED_FUTURE_EVICTION_HINT=CURRENT_SOURCE_ESTABLISHED
CURRENT_FUTURE_HINT_ONLY_NO_EARLY_STAGE_PROBE=PASS
LITERAL_FUTURE_ONLY_BLOCK_EARLY_H2D=NOT_IMPLEMENTED
CROSS_LAYER_RESIDENCY_PLANNER_HISTORICAL_CANONICAL=NO_GO_112_OF_112
CROSS_LAYER_RESIDENCY_PLANNER_HISTORICAL_H2D_IMPROVEMENT=ZERO
CROSS_LAYER_RESIDENCY_PLANNER_HISTORICAL_EVICTION_IMPROVEMENT=ZERO
CURRENT_HEAD_CROSS_LAYER_PLANNER_CERTIFICATION=NOT_ESTABLISHED_SOURCE_DRIFT
CURRENT_TORCH_DEPENDENT_PLANNER_REGRESSION=ENVIRONMENT_BLOCKED_NO_TORCH
FUTURE_WINDOW_PREFETCH_PERFORMANCE=NOT_ESTABLISHED

PREFIX_CPU_COMPOSITION=PARTIAL_WITH_LOCAL_CONTRACT_ONLY
ORDINARY_PREFIX_HASH_TOKEN_REUSE_LOCAL_CONTRACT=ESTABLISHED
ORDINARY_PREFIX_MULTI_OWNER_REFCOUNT_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_RESIDENCY_SCHEDULING_LOCAL_CONTRACT=ESTABLISHED
PREFIX_CPU_BACKING_REAL_CUDA_COPY=NOT_ESTABLISHED
LOADED_CROSS_REQUEST_PREFIX_HIT_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_REFCOUNT_LIFETIME_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_DEDUP_BYTE_AUTHORITY=NOT_ESTABLISHED
LOADED_PREFIX_CPU_RESTORE_AUTHORITY=NOT_ESTABLISHED
PREFIX_CPU_COMPOSITION_PERFORMANCE=NOT_ESTABLISHED
CURRENT_TORCH_DEPENDENT_PREFIX_REGRESSION=ENVIRONMENT_BLOCKED_NO_TORCH

KV4_KV8_PLUS_OFFLOAD=NOT_ESTABLISHED
LONG_CONTEXT_PROMPT_AND_EXACT_PARITY=ESTABLISHED_WITHIN_RETAINED_SCOPES
LONG_CONTEXT_REAL_TARGET_KV_MOVEMENT=ESTABLISHED_WITHIN_RETAINED_SCOPES
LONG_CONTEXT_RESIDENCY_LIFECYCLE=ESTABLISHED_WITHIN_RETAINED_SCOPES
NATIVE_TP4_16K_BOUNDED_GPU_RESIDENCY=ESTABLISHED
NATIVE_TP4_16K_BLOCKWISE_SOURCE_AND_CONFIG_BINDING=ESTABLISHED
BLOCKWISE_TP1_16K_32K_HISTORICAL_ARTIFACT=ESTABLISHED_WITHIN_RETAINED_SCOPE
BLOCKWISE_TP1_CURRENT_SOURCE_MATCH=4_OF_7
BLOCKWISE_TP1_FRESH_CURRENT_SOURCE_VERIFICATION=BLOCKED_SOURCE_DRIFT
BLOCKWISE_CURRENT_ATTENTION_TEST_COLLECTION=ENVIRONMENT_BLOCKED_NO_FLASH_ATTN
GENERIC_16K_32K_BLOCKWISE_IMPLEMENTATION_SOURCE_BINDING=MISSING
DIRECT_BLOCKWISE_RUNTIME_PATH_OBSERVATION=MISSING
END_TO_END_BLOCKWISE_ONLINE_SOFTMAX_AUTHORITY=PARTIAL
VARIABLE_Q_CUDA_GRAPH=PARTIAL_WITH_LOCAL_EXACT_FAMILY_CONTRACT_AND_LEGACY_TP1_ONLY
CURRENT_EXACT_FAMILY_GRAPH_PASS_ARTIFACT=ABSENT
CURRENT_EXACT_FAMILY_GRAPH_PREFLIGHTS=2_BLOCKED_BEFORE_SOURCE_UPLOAD_AND_CUDA_GATE
LEGACY_VARIABLE_Q_GRAPH_SCOPE=Q1_Q4_BATCH1_BATCH4_TP1_NO_OFFLOAD_SHORT_CONTEXT
LEGACY_VARIABLE_Q_GRAPH_TRANSACTION_EQUATIONS=28_OF_28_RECOMPUTED_PASS
LEGACY_VARIABLE_Q_GRAPH_PARITY=PRODUCER_ASSERTED_NOT_INDEPENDENTLY_RECOMPUTABLE
VARIABLE_Q_GRAPH_SOURCE_CLOSED_AUTHORITY=ABSENT
VARIABLE_Q_GRAPH_TP4=EXPLICIT_EAGER_FALLBACK
VARIABLE_Q_GRAPH_OFFLOAD=EXPLICIT_EAGER_FALLBACK
VARIABLE_Q_GRAPH_BLOCKWISE=EXPLICIT_EAGER_FALLBACK
VARIABLE_Q_GRAPH_LONG_CONTEXT=NOT_ESTABLISHED
VARIABLE_Q_GRAPH_PERFORMANCE=NOT_ESTABLISHED
VARIABLE_Q_GRAPH_PROMOTION=NOT_ESTABLISHED
VERIFIER_SAMPLING_COMMIT_FUSION=NOT_ESTABLISHED
TP_COLLECTIVE_OVERLAP_FUSION=NOT_ESTABLISHED

TP4_32K_FIRST_OUTPUT_DIVERGENCE_GENERATED_TOKEN_INDEX=3
TP4_32K_FIRST_RETAINED_COMPACT_TARGET_LOGIT_DIVERGENCE_GENERATED_TOKEN_INDEX=1
TP4_32K_FIRST_RETAINED_COMPACT_TARGET_ARGMAX_DIVERGENCE_GENERATED_TOKEN_INDEX=3
TP4_32K_FIRST_INTERNAL_DIVERGENCE=NOT_ESTABLISHED
TP4_32K_NATIVE_COMPACT_LOGITS=STEP_LEVEL_NOT_VERIFY_TAIL_NOT_TOKEN_ALIGNED
TP4_32K_INDEPENDENT_SAME_PROMPT_LOCAL_REPRODUCTION=ABSENT
TP4_32K_DIFFERENT_PROMPT_SAME_CHECKPOINT_PARITY=ESTABLISHED_NOT_AN_ORACLE
TP4_32K_CROSS_BUNDLE_CORE_ENGINE_CONFIG=ALIGNED_BY_BOUND_SOURCE_RECONSTRUCTION
TP4_32K_CROSS_BUNDLE_GENERIC_CELL_ENGINE_CONFIG_RECEIPT=ABSENT
TP4_32K_CROSS_BUNDLE_SOURCE_LINEAGE=DIFFERENT
TP4_32K_CROSS_BUNDLE_GENERIC_MANIFEST_HASH_OVERLAP=8_OF_16
TP4_32K_CROSS_BUNDLE_MANIFEST_DIFF_BASELINE_REACHABILITY=SPECULATIVE_ONLY_EXCEPT_KV_TELEMETRY_ASSERT
TP4_32K_ORDINARY_MODEL_RUN_FUNCTIONS=BYTE_IDENTICAL_IN_ARCHIVED_SOURCES
TP4_32K_TP_GREEDY_DIFF=SPECULATIVE_ONLY_UNDER_RUNTIME_ABSENT
TP4_32K_SEQUENCE_MAX_TOKENS_DIFF=ORDINARY_TRANSPORT_REACHABLE_NOT_MODEL_TOKEN_PATH_AND_NOT_GENERIC_MANIFEST_BOUND
TP4_32K_SOURCE_CONFOUND_STRENGTH=RESIDUAL_PROVENANCE_NO_IDENTIFIED_ORDINARY_TOKEN_AFFECTING_DIFF
TP4_32K_CROSS_BUNDLE_GPU_INDEX_TOPOLOGY=SAME_0_TO_7
TP4_32K_CROSS_BUNDLE_GPU_CAPACITY_CLASS=SAME_81920_MIB
TP4_32K_CROSS_BUNDLE_SELECTED_GPU_ORDER=DIFFERENT_WITH_2_OF_4_INDEX_OVERLAP
TP4_32K_CROSS_BUNDLE_EXACT_PHYSICAL_GPU_IDENTITY=NOT_RETAINED
TP4_32K_CROSS_BUNDLE_EXACT_SOFTWARE_STACK_IDENTITY=NOT_RETAINED
TP4_32K_HARDWARE_RANK_INTERACTION=NOT_EXCLUDED
BATCH4_MULTI_SEQUENCE_PASSING_AUTHORITIES=8
BATCH4_MULTI_SEQUENCE_CELLS=16
BATCH4_PROMPT_ROWS=64_WITH_4_UNIQUE_PER_CELL
BATCH4_OUTPUT_ROWS=64
BATCH4_PER_SEQUENCE_PARITY_COMPARISONS=32_ALL_PASS
BATCH4_EXECUTION_EVIDENCE=ACTIVE_SEQUENCE_COUNT_4_OR_SEQUENCE_IDS_0_TO_3
MIXED_CANCELLATION_HETEROGENEOUS_SEQUENCE_STATE=NOT_ESTABLISHED
TP4_32K_DIFFERENT_PROMPT_PASS_TRANSFER_STRENGTH=WEAK_COUNTEREXAMPLE_NOT_ORACLE
TP4_32K_CHECKPOINT_ALONE_AS_SUFFICIENT_CAUSE=NOT_SUPPORTED
TP4_32K_OUTPUT_SHA_AS_INPUT_BOUND_ORACLE=INVALID
TP4_32K_ROOT_CAUSE=NOT_ESTABLISHED
TP4_32K_FAILED_BUNDLE_SOURCE_MANIFEST_FILES=115_ALL_MATCH
TP4_32K_FAILED_BUNDLE_SOURCE_AGGREGATE=RECOMPUTED_MATCH
TP4_32K_FAILED_BUNDLE_SOURCE_TAR=MANIFEST_COMPLETE_WITH_143_UNMANIFESTED_PYC
TP4_32K_FAILED_BUNDLE_PASS_AUTHORITY_MANIFEST=ABSENT
TP4_32K_FAILED_BUNDLE_FOUR_CELL_INPUT_PAIRING=ESTABLISHED
TP4_32K_FAILED_BUNDLE_CELL_ARTIFACT_HASH_BINDING=ABSENT
TP4_32K_FAILED_BUNDLE_MODEL_MANIFEST_DIGEST_CONSISTENCY=ESTABLISHED
TP4_32K_FAILED_BUNDLE_ORIGINAL_MODEL_MANIFESTS=NOT_RETAINED_IN_BUNDLE
TP4_32K_TARGET_MODEL_MANIFEST_CROSS_BUNDLE_PAYLOAD=21_MATCHING_COPIES
TP4_32K_MTP_CHECKPOINT_CONTENT_DIGEST_ALGORITHM=ESTABLISHED
TP4_32K_MTP_CHECKPOINT_DIGEST_IMPLEMENTATIONS=THREE_WAY_EQUIVALENT
TP4_32K_MTP_CHECKPOINT_CONTENT_RECOMPUTATION=BLOCKED_CHECKPOINT_PATH_ABSENT
TP4_32K_FAILED_BUNDLE_MTP_CHECKPOINT_FILE_ROWS=NOT_RETAINED
FOCUSED_H2D_DIAGNOSTIC_LOCAL_CONTRACT=ESTABLISHED
FOCUSED_H2D_DEPENDENCY_LIGHT_AND_CAMPAIGN_TESTS=121_PASSED
FOCUSED_H2D_FULL_PLAN_TEST_MATRIX=PARTIAL_ENVIRONMENT_BLOCKED
FOCUSED_H2D_SOURCE_BOUND_LOCAL_RUNNER=ESTABLISHED
FOCUSED_H2D_DIAGNOSTIC_CODE_ENFORCED_AUTHORIZATION=ESTABLISHED_AT_CALLBACK_BOUNDARY
FOCUSED_H2D_COMPLETE_PRODUCER_PROVENANCE_CONTRACT=ESTABLISHED_137_FILES
FOCUSED_H2D_BUILTIN_REMOTE_TRANSPORT=ABSENT
FOCUSED_H2D_REAL_CHECKPOINT_PLAN=NOT_CREATED
FOCUSED_H2D_ACTIVE_EXECUTION_AUTHORIZATION=NOT_CREATED
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
PAIRED_STABILITY_REMOTE_BUNDLE=NOT_RUN_NOT_AUTHORIZED
HISTORICAL_GPU_AUTHORITIES_CURRENT_HEAD_CERTIFICATION=NOT_ESTABLISHED
FRESH_CURRENT_TREE_REVERIFICATION=SOURCE_AND_SCHEMA_DRIFT_DETECTED
CURRENT_HEAD_DEPENDENCY_LIGHT_RUNTIME_REGRESSION=264_PASSED
CURRENT_HEAD_EXTENDED_DEPENDENCY_LIGHT_REGRESSION=385_PASSED
CURRENT_HEAD_ENGINE_MODEL_RUNNER_SOURCE_REGRESSION=74_PASSED
CURRENT_HEAD_DIRECT_LIGHT_MATERIAL_DRIFT_REGRESSION=465_PASSED_1_SKIPPED_STANDALONE
CURRENT_HEAD_CROSS_GROUP_MODULE_ISOLATION_REGRESSION=925_PASSED
CURRENT_HEAD_COMBINED_DEPENDENCY_LIGHT_REGRESSION=925_PASSED
SINGLE_TOKEN_SPECULATIVE_SELECTION=SUPPRESSED_AS_DESIGNED
CURRENT_HEAD_TORCH_DEPENDENT_LOCAL_REGRESSION=ENVIRONMENT_BLOCKED

PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## 2026-08-17 Exact-Shape Draft CUDA Graph Runtime Reconciliation

This EOF reconciliation supersedes the earlier same-day statement that real
TP4 capture/replay and eager/graph correctness parity were not established.
The historical entry remains above as an audit trail of the earlier
preflight-only state.

### Executive matrix update

| Dimension | Current evidence | Classification |
| --- | --- | --- |
| Exact graph scope | TP4/B4/Q4, greedy, dense direct Proposal-KV, no offload | `ESTABLISHED` |
| Default policy | disabled unless explicitly enabled | `ESTABLISHED` |
| Real all-rank capture/replay | v5 diagnostic, v6 production-default worker, four measured graph workers | `ESTABLISHED` |
| Production-default capture budget | four-second single-capture ceiling; observed real maximum approximately 3.101 s | `ESTABLISHED` |
| Eager/graph target-token parity | production-default pilot plus four measured pairs | `ESTABLISHED` |
| Proposal/accepted-prefix parity | production-default pilot plus four measured pairs | `ESTABLISHED` |
| Transaction digest and zero-leak parity | digest `d102ac0a...4942989`; zero active transactions | `ESTABLISHED` |
| Graph/process-group teardown | reset, CUDA synchronize, then process-group destruction | `ESTABLISHED` |
| Two-warmup/eight-measured performance gate | external root process took GPU 3 after four measured pairs | `INCONCLUSIVE_ENVIRONMENT_PARTIAL_4_OF_8` |
| Positive paired bootstrap | eight-pair payload absent | `NOT_ESTABLISHED` |
| Broad graph generalization | only exact TP4/B4/Q4 recorded | `NOT_ESTABLISHED` |

### Real execution evidence

The pure TP4 collective diagnostic captured and replayed all-reduce and
broadcast on all ranks, reset its graph, synchronized CUDA, and exited all
process groups cleanly.

The high-budget real-checkpoint diagnostic measured capture times of
approximately `2.741-2.750 s`, retained `8,520,704 bytes` reserved and
`53,408 bytes` static per rank, and identified the original two-second
single-capture ceiling as the quarantine cause.

The production-default worker used no budget override and recorded:

```text
capture_attempts=1 per rank
captures=1 per rank
replays=1 per rank
quarantines=0 per rank
fallback_pre_replay=0 per rank
capture time approximately 3.094-3.098 s
accepted/proposed=51/70
acceptance rate=0.7285714285714285
active transactions=0
```

Its same-source eager control had exact target outputs, proposal rows,
accepted-prefix counts, transaction digest, and acceptance values.

### Partial controlled performance evidence

The source-bound paired gate completed both warmup pairs and four measured
pairs with balanced order counts:

```text
eager_graph=2
graph_eager=2
```

All four measured pairs had exact correctness and all-rank capture/replay.
The diagnostic aggregate was:

```text
median eager throughput:       0.989071982502008 tok/s
median graph throughput:       0.9749348881219722 tok/s
mean paired throughput delta: -0.002993426456363135 tok/s
median eager TPOT:             1.9743923907000003 s
median graph TPOT:             2.0375849077333337 s
```

During pair 4, an unrelated root-owned `VLLM::EngineCore` appeared on
physical GPU 3 and consumed approximately `73.3 GiB`. Rank 3 exited and the
remaining ranks stopped progressing. Only the gate's own process group was
terminated. The external process was not modified.

Four measured pairs are insufficient for the required eight-pair bootstrap,
so the numbers above are diagnostic and cannot be promoted to
`NO_GO_PERFORMANCE` or `GO`.

Current source-bound partial evidence:

```text
artifacts/autoregressive_draft_cuda_graph/
  20260817-production-default-paired-gate-tp4-b4-q4/
```

### Reconciled final classification

```text
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_SCOPE=TP4_B4_Q4_GREEDY_DENSE_DIRECT
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_DEFAULT=OFF
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_LOCAL_CONTRACT=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_REAL_CAPTURE_REPLAY=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_REAL_CORRECTNESS_PARITY=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_TRANSACTION_PARITY=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_TEARDOWN_LIFECYCLE=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_PRODUCTION_DEFAULT_BUDGET=ESTABLISHED
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_CONTROLLED_PERFORMANCE=INCONCLUSIVE_ENVIRONMENT_PARTIAL_4_OF_8
INDEPENDENT_QWEN3_DRAFT_EXACT_GRAPH_CLASSIFICATION=RUNTIME_CORRECTNESS_ESTABLISHED_PERFORMANCE_INCONCLUSIVE
CURRENT_EXACT_FAMILY_GRAPH_PASS_ARTIFACT=REAL_TP4_RUNTIME_AND_CORRECTNESS_PASS
CURRENT_EXACT_FAMILY_GRAPH_COMPLETE_PERFORMANCE_ARTIFACT=ABSENT_EXTERNAL_GPU_INTERFERENCE

PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

This materially advances the exact graph line from local-contract-only to
real runtime and correctness authority. It does not close the broader Phase 1
promotion gaps: the full learned 4K/16K/32K matrix, second learned structure,
native-MTP controlled performance, broad transactional-KV symmetry, and
complete eight-pair graph performance authority remain absent.

## 2026-08-20 Bounded Rollback Journal Reconciliation

### Prompt-to-artifact checklist

| Requirement | Concrete artifact/evidence | Status |
| --- | --- | --- |
| Remove Proposal-KV full-capacity rollback snapshot | `tinyvllm/engine/block_manager.py`; commit `8506673` | `ESTABLISHED` |
| Remove scheduler full-capacity rollback snapshot | `tinyvllm/engine/scheduler.py`; commit `4463800` | `ESTABLISHED` |
| Journal size scales with touched state | non-iterable 4096-block guards and touched-block-count assertions | `ESTABLISHED_LOCALLY` |
| Avoid hidden O(capacity) free-deque membership | `test_prepare_postprocess_does_not_scan_free_blocks`; direct `used_block_ids` lookup | `ESTABLISHED_LOCALLY` |
| Extend scheduler journal before Proposal-KV commit | engine event-order tests in both speculative engine files | `ESTABLISHED_LOCALLY` |
| Preserve Proposal-KV atomic rollback | `tools/test_speculative_kv_transaction.py`; `44 passed` | `ESTABLISHED_LOCALLY` |
| Preserve scheduler decode/prefill/completion rollback | `tools/test_scheduler_prepared_postprocess.py`; `22 passed` | `ESTABLISHED_LOCALLY` |
| Preserve exact multi-lease release order | real `HybridStateSlotAllocator` fixture and two-lease failure injection | `ESTABLISHED_LOCALLY` |
| Poison runtime on journal rollback failure | typed rollback errors and engine-level poisoning test | `ESTABLISHED_LOCALLY` |
| Preserve engine publication/finalization ordering | `tools/test_engine_speculative_execution.py` and `tools/test_engine_speculative_runtime.py`; `91 passed` | `ESTABLISHED_LOCALLY` |
| Broad dependency-light regression | 19 affected test files; `834 passed in 37.46s` | `ESTABLISHED_LOCALLY` |
| Torch-dependent local regression | four files fail collection because Mac system Python has no `torch` | `ENVIRONMENT_BLOCKED` |
| Syntax and whitespace validation | task-local `PYTHONPYCACHEPREFIX` py_compile; `git diff --check` | `PASS` |
| No new measured-path sync/GC/logging/profiling | focused static scan and diff review | `PASS` |
| Candidate TPOT p95 `<=105.87 ms` | fresh paired four-GPU candidate campaign | `NOT_RUN` |
| Candidate TPOT median `<=85.66 ms` | fresh paired four-GPU candidate campaign | `NOT_RUN` |
| TTFT p95 regression `<=3%` | fresh paired four-GPU candidate campaign | `NOT_RUN` |
| Throughput regression `<=3%` | fresh paired four-GPU candidate campaign | `NOT_RUN` |
| Exact tokens, Proposal-KV transactions, four-rank correctness, stationarity | dual-verifier candidate manifest | `NOT_RUN` |

### Implementation authority

```text
7076613  docs(performance): design bounded rollback journals
5b29ad4  docs(performance): plan bounded rollback journals
8506673  perf: bound proposal KV rollback state
4463800  perf: bound scheduler rollback state
```

All four commits are on and pushed to:

```text
origin/feat/kv-sparse-attention
```

### RED evidence

The Proposal-KV structural guard failed the old implementation at its complete
`self.blocks` traversal with:

```text
AssertionError: full block iteration is not allowed
```

The scheduler structural guard independently failed the old complete block
tuple for the same reason.

After production journal code was introduced but before AST fixtures were
updated:

```text
python3 -m pytest \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py -q

19 failed, 71 passed
```

The failures were interface-accurate: extracted engine helpers lacked
`SpeculativeKVCommitRollbackError` and
`SchedulerPostprocessRollbackError`, while fake prepared schedulers lacked
`snapshot.extend_speculative_kv_plans(...)`.

Two review-driven RED tests found and fixed additional bounded-journal
defects:

```text
test_prepare_postprocess_does_not_scan_free_blocks:
  AssertionError: free block membership scans are not allowed

test_commit_failure_restores_multiple_hybrid_releases:
  scheduler postprocess rollback failed:
  hybrid free-slot rollback order changed
```

The first removed a hidden O(total KV capacity) `deque.__contains__` scan.
The second changed hybrid lease restoration to exact reverse release order.

### GREEN and static evidence

Fresh pre-commit verification:

```text
Proposal-KV transaction file:
  44 passed

scheduler prepared-postprocess file:
  22 passed in 0.32s

engine speculative files:
  91 passed in 0.68s

four focused transactional files:
  157 passed in 0.99s

dependency-light affected autoregressive-draft suite:
  834 passed in 37.46s

py_compile:
  PASS

git diff --check:
  PASS
```

The four torch-dependent files below remain uncollected under the current Mac
system Python:

```text
tools/test_autoregressive_draft_executor.py
tools/test_autoregressive_draft_model_runner_integration.py
tools/test_autoregressive_draft_registration.py
tools/test_autoregressive_draft_tp.py
```

Their `ModuleNotFoundError: No module named 'torch'` is classified as an
environment setup gap. It is not counted as either GREEN or a code failure.

The forbidden-copy scan still finds three pre-existing operations in
`block_manager.py`: full free-list snapshots in the older sequence-reservation
and speculative-transaction preparation paths, and full block iteration in
explicit `clear_reusable_cache()`. None is inside the two replaced commit or
scheduler journal paths. The only focused `gc.collect()` match is existing
explicit engine shutdown cleanup. No new `.item()`, CUDA synchronization,
measured-path GC control, logging, profiling, acknowledgement, or fence was
introduced.

### Executive matrix update

| Dimension | Current evidence | Classification |
| --- | --- | --- |
| Proposal-KV rollback space/time scaling | plan-local touched blocks and hashes; structural guard | `LOCAL_ESTABLISHED` |
| Scheduler rollback space/time scaling | scheduled/touched state only; block and free-deque guards | `LOCAL_ESTABLISHED` |
| Proposal-KV atomicity | duplicate hashes, release order, injected failures | `LOCAL_ESTABLISHED` |
| Scheduler atomicity | decode, completion, prefill, hooks, hybrid leases, multi-release order | `LOCAL_ESTABLISHED` |
| Rollback failure handling | typed terminal errors and engine runtime poisoning | `LOCAL_ESTABLISHED` |
| Full local dependency-light regression | 834 tests | `PASS` |
| Full torch-dependent local regression | dependency absent | `ENVIRONMENT_BLOCKED_NO_TORCH` |
| TPOT p95/median improvement | candidate paired campaign absent | `NOT_ESTABLISHED` |
| TTFT and throughput protection | candidate paired campaign absent | `NOT_ESTABLISHED` |
| Separate worker/CUDA anomaly | not targeted by this change | `NOT_ESTABLISHED` |
| Python GC as original root cause | not directly measured | `NOT_ESTABLISHED` |

### Reconciled classification

```text
LOCAL_BOUNDED_JOURNAL_CORRECTNESS=ESTABLISHED
PROPOSAL_KV_FULL_CAPACITY_ROLLBACK_SNAPSHOT=REMOVED
SCHEDULER_FULL_CAPACITY_ROLLBACK_SNAPSHOT=REMOVED
SCHEDULER_FREE_DEQUE_LINEAR_MEMBERSHIP_SCAN=REMOVED
MULTI_HYBRID_RELEASE_ROLLBACK_ORDER=ESTABLISHED
ROLLBACK_FAILURE_RUNTIME_POISONING=ESTABLISHED
CURRENT_HEAD_DEPENDENCY_LIGHT_BOUNDED_JOURNAL_REGRESSION=834_PASSED
CURRENT_HEAD_TORCH_DEPENDENT_BOUNDED_JOURNAL_REGRESSION=ENVIRONMENT_BLOCKED_NO_TORCH
TPOT_TAIL_BENEFIT=NOT_ESTABLISHED
TTFT_NON_REGRESSION=NOT_ESTABLISHED_FOR_CANDIDATE
THROUGHPUT_NON_REGRESSION=NOT_ESTABLISHED_FOR_CANDIDATE
SEPARATE_SPECULATIVE_PREPARE_WORKER_CUDA_ANOMALY=NOT_ESTABLISHED
PYTHON_GC_CAUSALITY=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

The next admissible evidence step is a fresh immutable same-protocol paired
four-GPU campaign. It must use the strict admission rule, keep every remote
output beneath
`/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`, and
pass dual verification for exact tokens, Proposal-KV transactions, four-rank
correctness, stationarity, TPOT p95/median, TTFT p95, and throughput. Until
that campaign completes, no performance benefit may be claimed.

## 2026-08-20 Source-Version Paired Gate Readiness and Environment Result

### Prompt-to-artifact checklist

| Requirement | Concrete artifact/evidence | Status |
| --- | --- | --- |
| Compare the pinned pre-journal baseline against the current candidate | baseline `596e724ea87966b2ab3b47cccda08c106f9084bb`; candidate `497421fd4f19199450577139c993113f42c15f62` | `ESTABLISHED` |
| Export both sources from exact Git objects without a checkout or local patch | `export_git_revision_archive()`; live in-memory archive audit | `ESTABLISHED` |
| Preserve the frozen eight-pair eager/graph and source order | `expected_source_pair_schedule()` and schedule contract tests | `ESTABLISHED_LOCALLY` |
| Use 8 epochs, 40 measured repeats, and 160 request samples per source | source-pair constants, aggregation checks, and tests | `ESTABLISHED_LOCALLY` |
| Bind exact output, Proposal-KV transaction, active-transaction, and four-rank correctness | `autoregressive_draft_source_pair_gate.py`; semantic-parity tests | `ESTABLISHED_LOCALLY` |
| Compute request TPOT median/p95, TTFT p95, median batch throughput, regressions, and eager/graph stationarity | pure source-pair artifact builder and threshold/precedence tests | `ESTABLISHED_LOCALLY` |
| Recompute the artifact independently and verify a complete parent manifest | `verify_autoregressive_draft_source_pair_gate.py`; tamper, incomplete-inventory, and symlink rejection tests | `ESTABLISHED_LOCALLY` |
| Interleave all 16 child members, freeze before/after GPU inventory, finalize both children, then build the parent | `run_autoregressive_draft_source_pair_remote.py`; orchestration and parent-pipeline tests | `ESTABLISHED_LOCALLY` |
| Keep every remote cache, scratch, log, receipt, manifest, and run beneath the Sitian task root | source-pair path checks plus inherited SSH TMP/pycache/XDG environment | `ESTABLISHED_LOCALLY` |
| Include the three source-pair production files in the candidate source closure only | baseline archive: 144 members and no source-pair files; candidate archive: 147 members and all three files | `ESTABLISHED` |
| Source-pair and reused command-timeline regression | `306 passed in 30.22s`; py_compile, tabnanny, and `git diff --check` passed | `PASS` |
| Commit and push the complete toolchain | `497421f feat: add source-paired performance gate`; remote branch confirmed by `git ls-remote` | `PASS` |
| Fail before remote mutation when local Kerberos authority is absent | exact preflight JSON returned exit code `2` with `local Kerberos payload is invalid` | `PASS` |
| Admit exactly four clean GPUs | blocked before SSH by Kerberos fail-fast | `NOT_REACHED` |
| Execute the 8-pair/16-member campaign | no remote destination or worker was created | `NOT_RUN` |
| Produce child and parent manifests plus remote/local verifier receipts | campaign did not start | `NOT_RUN` |
| Verify exact tokens, Proposal-KV transactions, four-rank correctness, and paired stationarity on real outputs | campaign did not start | `NOT_RUN` |
| Candidate TPOT p95 `<=105.87 ms` and median `<=85.66 ms` | campaign did not start | `NOT_RUN` |
| TTFT p95 and throughput regressions each `<=3%` | campaign did not start | `NOT_RUN` |

### Source and local-verification authority

The source-pair implementation was committed and pushed as:

```text
497421fd4f19199450577139c993113f42c15f62
feat: add source-paired performance gate
```

The commit contains exactly:

```text
tools/autoregressive_draft_source_pair_gate.py
tools/run_autoregressive_draft_command_timeline_remote.py
tools/run_autoregressive_draft_source_pair_remote.py
tools/test_autoregressive_draft_cuda_graph_gate.py
tools/test_autoregressive_draft_source_pair_gate.py
tools/verify_autoregressive_draft_source_pair_gate.py
```

It has exactly one trailer:

```text
Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

Fresh pre-commit evidence:

```text
source-pair + command-timeline + CUDA-graph runner suites:
  306 passed in 30.22s

py_compile:
  PASS

tabnanny:
  PASS

git diff --check:
  PASS
```

Two final review-driven RED-to-GREEN cases were preserved:

```text
test_source_pair_artifact_uses_embedded_epoch_index_not_mapping_order:
  RED: baseline epoch schedule is invalid
  GREEN: epoch mappings are indexed by unique embedded epoch_index 0..7

test_source_pair_verifier_rejects_parent_manifest_symlink:
  RED: verifier accepted an in-root symlink
  GREEN: manifest, root, bound path components, and inventory reject symlinks
```

The exact-object archive audit produced:

```text
baseline:
  revision: 596e724ea87966b2ab3b47cccda08c106f9084bb
  bytes: 3,133,440
  members: 144
  source-pair production files: absent
  unsafe link members: none

candidate:
  revision: 497421fd4f19199450577139c993113f42c15f62
  bytes: 3,225,600
  members: 147
  source-pair production files: all three present
  unsafe link members: none
```

Local HEAD, `origin/feat/kv-sparse-attention`, and the live remote ref all
resolved to `497421fd4f19199450577139c993113f42c15f62`.

### Strict admission result

The first fresh preflight tag was:

```text
20260820-bounded-journal-tpot-source-pair-r2
```

It returned exit code `2` and:

```json
{"minimum_required_lifetime_seconds":5400,"reason":"local Kerberos payload is invalid","status":"INCONCLUSIVE_ENVIRONMENT"}
```

Read-only diagnosis on Thursday, August 20, 2026 confirmed:

```text
klist --json:
  {"version":1}

klist:
  Cache not found

klist -l:
  no credential caches
```

No manual credential refresh was attempted. A current-Agent local monitor
then used 120 further fresh immutable tags, from `r3` through `r122`, at
60-second intervals. Every attempt returned the same Kerberos fail-fast
result. The monitor ran from `2026-08-20T21:19:37+08:00` through
`2026-08-20T23:19:54+08:00` and ended with:

```text
MONITOR_EXHAUSTED attempts=120
```

Because authentication failed before `_run_remote_command()`:

- no SSH command was issued by the source-pair preflight;
- no remote path was created or overwritten;
- no GPU inventory was queried;
- no worker or background experiment process was launched;
- no unrelated process was signaled, adopted, paused, or modified;
- no child or parent manifest/verifier receipt exists for a real campaign.

All tags from `r2` through `r122` are retired and must not be reused. The next
attempt requires a fresh tag after the local `sitian@BYTEDANCE.COM` TGT exists
with at least 5,400 seconds remaining. Normal implementation/run approval is
already waived; once that external condition is restored, the strict
four-clean-GPU preflight and campaign may proceed automatically.

### Reconciled classification

```text
SOURCE_PAIR_GATE_IMPLEMENTATION=ESTABLISHED_LOCALLY
SOURCE_PAIR_GATE_DUAL_VERIFIER_CONTRACT=ESTABLISHED_LOCALLY
SOURCE_PAIR_GATE_GIT_OBJECT_EXPORT=ESTABLISHED
SOURCE_PAIR_GATE_CANDIDATE_COMMIT=497421fd4f19199450577139c993113f42c15f62
SOURCE_PAIR_GATE_REMOTE_ADMISSION=INCONCLUSIVE_ENVIRONMENT_KERBEROS_CACHE_MISSING
SOURCE_PAIR_GATE_FOUR_GPU_ADMISSION=NOT_REACHED
SOURCE_PAIR_GATE_REAL_8_PAIR_CAMPAIGN=NOT_RUN
SOURCE_PAIR_GATE_CHILD_MANIFESTS=NOT_CREATED
SOURCE_PAIR_GATE_PARENT_MANIFEST=NOT_CREATED
SOURCE_PAIR_GATE_DUAL_VERIFICATION=NOT_RUN
SOURCE_PAIR_GATE_EXACT_CORRECTNESS=NOT_RUN
SOURCE_PAIR_GATE_STATIONARITY=NOT_RUN
TPOT_TAIL_BENEFIT=NOT_ESTABLISHED
TTFT_NON_REGRESSION=NOT_ESTABLISHED_FOR_CANDIDATE
THROUGHPUT_NON_REGRESSION=NOT_ESTABLISHED_FOR_CANDIDATE
SOURCE_PAIR_GATE_CLASSIFICATION=INCONCLUSIVE_ENVIRONMENT
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

This is an environment result, not a correctness or performance failure. No
performance benefit may be claimed from the local toolchain tests or the
failed admission attempts.

## 2026-08-20 Bounded-Journal Source-Pair Terminal Reconciliation

### Prompt-to-artifact checklist

| Requirement | Concrete artifact/evidence | Status |
| --- | --- | --- |
| Compare the pinned pre-journal baseline with the bounded-journal candidate | baseline `596e724ea87966b2ab3b47cccda08c106f9084bb`; candidate `9ea37339859b4c18d54458a2ec0c5bac0fdfc50c` | `PASS` |
| Admit exactly four clean GPUs under the frozen threshold | GPU indices `2,3,5,6`; four frozen UUIDs; every before/after inventory status `0` | `PASS` |
| Keep all remote output beneath the Sitian task root | both child roots, both controller roots, parent root, controller parent, and preserved partial copies are under `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818` | `PASS` |
| Run the exact eight-pair balanced source/CUDA schedule | `source-pair.json.schedule`; four baseline-first and four candidate-first pairs; eager and graph each contain both orders twice | `PASS` |
| Complete 8 epochs, 40 measured repeats, and 160 request samples per source | `sample_counts` in parent artifact; 16/16 normalized child workers and 16/16 after-inventory receipts | `PASS` |
| Preserve exact output, Proposal-KV transaction, active-transaction, and four-rank correctness | parent `correctness.passed=true`, `mismatch_count=0`, and `underlying_command_timeline_passed=true` | `PASS` |
| Produce complete baseline and candidate child manifests | baseline `283` files, manifest `00ea53ca...ec8`; candidate `286` files, manifest `258e00b1...792` | `PASS` |
| Independently verify both child bundles in primary and controller locations | all four child receipts have `verified=true`, complete manifest coverage, matching normalized receipts, and source/raw-input inventories bound | `PASS` |
| Produce and independently verify the parent source-pair artifact | artifact `ed10b16d...a6c`; manifest `58d364ca...f10`, 10 files; primary/controller receipts both `verified=true` and hashes match | `PASS` |
| Candidate TPOT median `<=85.66 ms` | `77.6682515 ms`, 4.8712% below baseline | `PASS` |
| Candidate TPOT p95 `<=105.87 ms` | `107.216581 ms`, 1.346581 ms or 1.2719% above the absolute limit | `FAIL` |
| TTFT p95 regression `<=3%` | `-14.0944%` regression, meaning an improvement | `PASS` |
| Throughput regression `<=3%` | `-12.9961%` regression, meaning a 12.9961% improvement | `PASS` |
| Eager and graph source-ratio stationarity | eager and graph TPOT/throughput ratio checks all pass | `PASS` |
| All sixteen underlying source epochs pass command-timeline stationarity | both child admissions report `stationarity_passed=false`; parent `all_sixteen_source_epochs_passed=false` | `FAIL` |
| Emit one approved terminal classification without overstating benefit | `INCONCLUSIVE_STATIONARITY`; `performance_improvement_established=false` | `PASS` |
| Preserve failed assembly evidence without rewriting frozen source bundles | r910 candidate partial and r911 baseline/candidate failed partial controller copies retained; no frozen `source/` directory was modified | `PASS` |
| Make future child finalization use the committed canonical tooling for both frozen sources | `_child_finalization_remote_arguments()` always launches from the candidate source while targeting the selected child tag; command-path and full campaign tests | `PASS` |

### Immutable campaign and source authority

The terminal campaign tag is:

```text
20260820-bounded-journal-tpot-source-pair-r911
```

Source authority:

```text
baseline:
  596e724ea87966b2ab3b47cccda08c106f9084bb
candidate:
  9ea37339859b4c18d54458a2ec0c5bac0fdfc50c
candidate branch:
  feat/kv-sparse-attention
candidate remote:
  origin/feat/kv-sparse-attention
```

The candidate includes the JSON-safe stationarity fix:

```text
9ea3733 fix: keep zero-median stationarity artifacts finite
```

That fix preserves a failed stationarity decision while representing an
undefined zero-median ratio as JSON `null` rather than `Inf`. The focused RED
case used the exact observed shape `[0, 0, 0, 26430, 0]`; the full affected
local suite passed:

```text
307 passed in 30.87s
py_compile: PASS
tabnanny: PASS
git diff --check: PASS
```

Strict admission selected:

```text
GPU indices:
  2,3,5,6
GPU UUIDs:
  GPU-63c05907-407b-8240-07a0-f38872840867
  GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d
  GPU-687b7858-ca44-98ad-cfba-b6785eaf05e8
  GPU-c27f6fd6-8a66-7935-41fd-bd5ccdaced31
```

All 16 pair members completed. Every worker, invariant, before-inventory, and
after-inventory terminal status was zero.

### Assembly recovery boundary

The earlier r910 campaign completed all 16 members but failed while the
candidate child attempted canonical assembly. The full traceback was:

```text
ValueError: command timeline artifact contains a non-finite number
```

The exact paths were:

```text
epochs.b1-graph-first.stationarity.worker_queue_debt.half_drift
epochs.b1-graph-first.stationarity.queued_behind_prior_command.half_drift
```

r911 then completed all 16 members and reached the same old-schema defect in
the pinned baseline source. The pinned baseline must remain byte-exact, so its
frozen source bundle was not patched. Instead, the candidate's committed
canonical assembler/verifier loaded the baseline raw inputs while the
independent verifier continued to bind and hash the baseline source manifest.
Both original r911 partial controller copies were preserved with
`-failed-partial` suffixes before fresh controller copies were created. This
recovery changes neither measured data nor source identity.

### Metrics and gate result

```text
baseline:
  TPOT p95:                 137.207225 ms
  TPOT median:               81.6453475 ms
  TTFT p95:                 129.584231 ms
  median batch throughput:   39.9216946497 tok/s

candidate:
  TPOT p95:                 107.216581 ms
  TPOT median:               77.6682515 ms
  TTFT p95:                 111.320054 ms
  median batch throughput:   45.1099665366 tok/s

paired direction:
  TPOT p95:                 -21.8579%
  TPOT median:               -4.8712%
  TTFT p95:                 -14.0944%
  throughput:               +12.9961%
```

The directional evidence is favorable, but two mandatory gates remain
unsatisfied:

1. candidate TPOT p95 is `1.346581 ms` above the absolute `105.87 ms` limit;
2. not all sixteen underlying command-timeline epochs pass stationarity.

The eager and graph source-ratio stationarity checks pass, correctness passes,
and TTFT/throughput do not regress. Classification precedence nevertheless
requires the stationarity failure to terminate as
`INCONCLUSIVE_STATIONARITY`, not as a performance GO or NO-GO.

### Manifest and verifier authority

```text
baseline child:
  artifact: dc79b02db22bd6d725b0f5c190865eed905d1a1b7142eb02a1fe231183fda173
  manifest: 00ea53ca3659a36d515075f638ccbfa4afb3f610c1f2b8f8ca636311f8beeec8
  manifest files: 283
  primary/controller verified: true/true

candidate child:
  artifact: bb682b587b59c4e0592201d343c170670cf352bbbafa75170adff2b58caa2105
  manifest: 258e00b175771eb73f22ac43c4e86a8d5011ae9b1780d3b02e0e283bd1179792
  manifest files: 286
  primary/controller verified: true/true

parent:
  artifact: ed10b16df660946c87ab3ff00bf1fadabef8ed50b25faf2d9b009e702907ca6c
  manifest: 58d364ca0c22120b3b5312714714ff0bc5a088301982dc034789ea5a6efdef10
  manifest files: 10
  primary/controller verified: true/true
  verifier source: a8469b1f50d798de1960701acfdad4903b402d6850fc967aa88f1b0d048e0562
```

Primary and controller parent artifacts are byte-identical. Receipt comparison
passed after excluding only the documented location/time fields.

The post-campaign orchestration fix makes the successful recovery path the
default path for future campaigns: runtime workers still execute from their
own frozen baseline/candidate sources, while both child finalizations execute
the committed candidate canonical assembler/verifier against the selected
child's bound raw inputs and source manifest. Its focused RED failed because
the helper did not exist; GREEN evidence is:

```text
source-pair suite:
  35 passed

source-pair + command-timeline + CUDA-graph suites:
  308 passed in 31.44s

py_compile:
  PASS
tabnanny:
  PASS
git diff --check:
  PASS
```

Final direct re-verification ran with bytecode writes disabled and rebuilt all
six verification views in memory:

```text
baseline primary:    verified=true
baseline controller: verified=true
candidate primary:   verified=true
candidate controller: verified=true
parent primary:      verified=true
parent controller:   verified=true
```

One audit command initially omitted `PYTHONDONTWRITEBYTECODE=1` and created
exactly three unmanifested `.pyc` files in the candidate primary source cache.
The three generated cache files were identified by manifest-set difference and
deleted explicitly. A second bytecode-disabled six-view verification passed
with the authoritative artifact and manifest hashes unchanged.

### Executive matrix update

| Dimension | Current evidence | Classification |
| --- | --- | --- |
| Bounded journal implementation and rollback correctness | committed implementation plus focused and broad local regression | `ESTABLISHED_LOCALLY` |
| Real four-GPU source-bound execution | 8 pairs, 16 members, 80 measured repeats total, 320 request samples total | `COMPLETE` |
| Exact correctness and transaction parity | zero mismatches; underlying command-timeline correctness pass | `PASS` |
| Child and parent integrity | complete manifests and primary/controller verification | `PASS` |
| Automated child finalization compatibility | candidate canonical tooling finalizes both frozen sources | `PASS` |
| Candidate TPOT median | `77.6682515 ms <= 85.66 ms` | `PASS` |
| Candidate TPOT p95 | `107.216581 ms > 105.87 ms` | `FAIL_ABSOLUTE_LIMIT` |
| TTFT p95 protection | 14.0944% improvement | `PASS` |
| Throughput protection | 12.9961% improvement | `PASS` |
| Eager/graph paired-ratio stationarity | all four ratio checks pass | `PASS` |
| All sixteen child epoch stationarity | false | `FAIL` |
| TPOT-tail benefit claim | mandatory p95 and stationarity gates are not jointly green | `NOT_ESTABLISHED` |

### Final classification

```text
SOURCE_PAIR_GATE_IMPLEMENTATION=ESTABLISHED
SOURCE_PAIR_GATE_CANDIDATE_COMMIT=9ea37339859b4c18d54458a2ec0c5bac0fdfc50c
SOURCE_PAIR_GATE_FOUR_GPU_ADMISSION=PASS
SOURCE_PAIR_GATE_REAL_8_PAIR_CAMPAIGN=COMPLETE
SOURCE_PAIR_GATE_CHILD_MANIFESTS=PASS
SOURCE_PAIR_GATE_PARENT_MANIFEST=PASS
SOURCE_PAIR_GATE_DUAL_VERIFICATION=PASS
SOURCE_PAIR_GATE_EXACT_CORRECTNESS=PASS
SOURCE_PAIR_GATE_EAGER_GRAPH_RATIO_STATIONARITY=PASS
SOURCE_PAIR_GATE_ALL_SIXTEEN_EPOCH_STATIONARITY=FAIL
CANDIDATE_TPOT_MEDIAN_GATE=PASS
CANDIDATE_TPOT_P95_GATE=FAIL_107_216581_MS_GT_105_87_MS
TTFT_P95_NON_REGRESSION=PASS
THROUGHPUT_NON_REGRESSION=PASS
TPOT_TAIL_BENEFIT=NOT_ESTABLISHED
SOURCE_PAIR_GATE_CLASSIFICATION=INCONCLUSIVE_STATIONARITY
PERFORMANCE_IMPROVEMENT_ESTABLISHED=false
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

No claim that the bounded-journal optimization established TPOT-tail benefit
is admissible from r911. A future attempt must first address the underlying
command-timeline stationarity failure and then beat the absolute TPOT p95
limit under another fresh immutable, source-bound campaign.

## 2026-08-21 500 ms GPU-Sampler Cadence Reconciliation

### Prompt-to-artifact checklist

| Requirement | Concrete artifact/evidence | Status |
| --- | --- | --- |
| Change only the measurement cadence before touching the inference hot path | commit `d1ebb8a19db5746c97ecf64b797e79b567f4fc7f`; only the command-timeline runner and its test changed | `PASS` |
| Preserve all eight NVML fields and 32 field-isolated query processes | generated sampler still uses four GPUs times eight query names; sampler regression group passed | `PASS` |
| Make the cadence explicit and set it to 500 ms | `GPU_SAMPLER_INTERVAL_NS = 500_000_000`; generated script declares `SAMPLE_INTERVAL_NS=500000000` and increments by that name | `PASS` |
| Demonstrate RED before production modification | focused test failed with `AttributeError` because `GPU_SAMPLER_INTERVAL_NS` did not exist | `PASS` |
| Demonstrate focused and affected GREEN | focused `1 passed`; sampler group `12 passed, 99 deselected`; complete command-timeline/CUDA-graph test file `111 passed` | `PASS` |
| Commit and push the exact candidate source used by the source-pair runner | local HEAD, `origin/feat/kv-sparse-attention`, and candidate source identity were `d1ebb8a19db5746c97ecf64b797e79b567f4fc7f` before admission | `PASS` |
| Use a fresh immutable run tag | `20260821-sampler-500ms-tpot-source-pair-r1` | `PASS` |
| Admit exactly four clean GPUs with the frozen thresholds | GPU indices `2,3,5,6`; the same four frozen UUIDs as r911; all 16 before/after admissions completed with status `0` | `PASS` |
| Keep all generated remote files beneath the mounted Sitian task root | child, controller, parent, logs, telemetry, manifests, and receipts are beneath `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818` | `PASS` |
| Run the full balanced 8-pair/16-member source schedule | orchestrator returned `status=PASS` with all 16 schedule members listed | `PASS` |
| Preserve exact correctness and Proposal-KV/transaction parity | parent `correctness.passed=true`, `mismatch_count=0`, and `underlying_command_timeline_passed=true` | `PASS` |
| Produce complete child manifests and verify primary/controller views | baseline manifest `279` files; candidate manifest `282` files; all four child views independently rebuilt with `verified=true` | `PASS` |
| Produce and independently verify the parent artifact and manifest | parent manifest `10` files; primary/controller artifacts share hash `9a67205b...b629`; both fresh verifier runs returned `verified=true` | `PASS` |
| Candidate TPOT median `<=85.66 ms` | `78.2303015 ms` | `PASS` |
| Candidate TPOT p95 `<=105.87 ms` | `122.992254 ms`, which is `17.122254 ms` or `16.1729%` above the absolute limit | `FAIL` |
| TTFT p95 regression `<=3%` | candidate `102.296520 ms` versus baseline `109.264879 ms`, a `6.3775%` improvement | `PASS` |
| Throughput regression `<=3%` | candidate `45.3954218693 tok/s` versus baseline `38.6934990901 tok/s`, a `17.3205%` improvement | `PASS` |
| Eager and graph source-ratio stationarity | all eager/graph TPOT and throughput ratio checks pass | `PASS` |
| All sixteen underlying child epochs pass command-timeline stationarity | parent `all_sixteen_source_epochs_passed=false`; four candidate epochs fail only ACK-wait stationarity and the baseline also remains unstable | `FAIL` |
| Establish that reduced sampling pressure removes the observed CUDA tail | candidate generation median fell from `7.5` to `3` per repeat and query cells fell `58.9214%`, but three eager repeat-2 CUDA verify spikes remained | `FAIL` |
| Emit a terminal claim that respects all gate precedence | `INCONCLUSIVE_STATIONARITY`; `performance_improvement_established=false`; no TPOT-tail benefit claim | `PASS` |
| Update the canonical audit and handoff with the terminal result | this reconciliation and the matching EOF handoff section | `PASS` |

### Candidate and admission authority

```text
candidate commit:
  d1ebb8a19db5746c97ecf64b797e79b567f4fc7f
subject:
  perf(telemetry): reduce GPU sampler cadence
branch:
  feat/kv-sparse-attention
remote:
  origin/feat/kv-sparse-attention

run tag:
  20260821-sampler-500ms-tpot-source-pair-r1

baseline:
  596e724ea87966b2ab3b47cccda08c106f9084bb
candidate:
  d1ebb8a19db5746c97ecf64b797e79b567f4fc7f

GPU indices:
  2,3,5,6
GPU UUIDs:
  GPU-63c05907-407b-8240-07a0-f38872840867
  GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d
  GPU-687b7858-ca44-98ad-cfba-b6785eaf05e8
  GPU-c27f6fd6-8a66-7935-41fd-bd5ccdaced31
```

The read-only preflight on Friday, August 21, 2026 returned `READY`. The local
`sitian@BYTEDANCE.COM` TGT had `32,628` seconds remaining, and all six remote
destinations were absent before the run. The orchestrator then completed all
16 pair members and returned `status=PASS`. No unrelated process was
signaled, adopted, paused, killed, or modified.

### TDD and local verification

The new contract requires both the Python module and the generated remote
sampler to expose the cadence by name:

```text
GPU_SAMPLER_INTERVAL_NS = 500_000_000
SAMPLE_INTERVAL_NS=500000000
next_sample_ns+=SAMPLE_INTERVAL_NS
```

Focused RED:

```text
test_command_timeline_gpu_sampler_uses_500ms_cadence_constant
  AttributeError:
  module ... has no attribute 'GPU_SAMPLER_INTERVAL_NS'
```

GREEN evidence:

```text
focused cadence contract:
  1 passed in 0.12s

sampler lifecycle/schema/isolation group:
  12 passed, 99 deselected in 4.17s

complete affected test file:
  111 passed in 7.87s

final source-pair + command-timeline + CUDA-graph regression:
  329 passed in 32.57s

py_compile:
  PASS
tabnanny:
  PASS
git diff --check:
  PASS
```

The candidate source commit was pushed before remote admission because the
source-pair runner exports exact Git objects and requires
`HEAD == origin/feat/kv-sparse-attention`.

### Gate metrics

```text
baseline:
  TPOT p95:                 126.145358 ms
  TPOT median:               82.8648070 ms
  TTFT p95:                 109.264879 ms
  median batch throughput:   38.6934990901 tok/s

candidate:
  TPOT p95:                 122.992254 ms
  TPOT median:               78.2303015 ms
  TTFT p95:                 102.296520 ms
  median batch throughput:   45.3954218693 tok/s

paired direction:
  TPOT p95:                  -2.4996%
  TPOT median:               -5.5929%
  TTFT p95:                  -6.3775%
  throughput:               +17.3205%

absolute candidate TPOT p95 limit:
  105.870000 ms
candidate excess:
   17.122254 ms
   16.1729%
```

Correctness passes with zero mismatches. The candidate median, TTFT, and
throughput gates pass, and the eager/graph source-ratio stationarity checks
pass. The candidate p95 absolute gate fails by a materially larger margin
than r911, and the required all-sixteen-epoch stationarity gate remains false.

The four candidate command-timeline epoch failures are:

```text
b0-graph-second:
  ACK-wait stationarity only
b1-graph-first:
  ACK-wait stationarity only
b2-graph-first:
  ACK-wait stationarity only
b3-eager-first:
  ACK-wait stationarity only
```

Candidate TPOT stationarity itself passes in every epoch. This does not
override the source-pair contract, which requires every underlying
command-timeline epoch to pass its full stationarity admission.

### Sampler-pressure result and CUDA-tail localization

The cadence change materially reduced query volume:

```text
r911 candidate:
  generations per repeat median: 7.5
  generations per repeat p95:   12
  per-field query observations: 23,808

500 ms candidate:
  generations per repeat median: 3
  generations per repeat p95:    5
  per-field query observations:  9,780

query-observation reduction:
  58.9214%
```

It did not remove the eager CUDA tail. The three new dominant outliers all
occur in measured repeat `2`, at engine step `4`, in the second
`run_spec_verify_batch`, with active batch size `2`:

```text
b0-eager-first:
  max request TPOT: 143.367945 ms
  CUDA event:       720.109863 ms

b1-eager-second:
  max request TPOT: 141.050387 ms
  CUDA event:       609.109131 ms

b2-eager-second:
  max request TPOT: 152.369227 ms
  CUDA event:       779.018738 ms

b3-eager-first repeat 2:
  max request TPOT:  99.562413 ms
  CUDA event:        52.808011 ms
```

The slow block-0 and block-2 CUDA intervals overlap only ordinary
approximately `22 ms` NVML query windows. Block 1 overlaps one approximately
`557 ms` query window, but a normal approximately `53 ms` CUDA repeat also
overlaps an approximately `245 ms` query window. Therefore NVML pressure may
still amplify the environment, but query overlap is neither necessary nor
sufficient for the three repeat-2 CUDA spikes. The evidence does not support
another cadence-only optimization.

The lower-frequency candidate also retained severe per-call NVML latency:

```text
clock_throttle_reasons:
  p95 266.612797 ms
  p99 449.242361 ms
  max 729.581606 ms

other fields:
  p95 approximately 65-111 ms
  p99 approximately 195-306 ms
```

This is lower total query pressure, not a repair of the driver-level long
calls.

### Manifest and fresh six-view verifier authority

```text
baseline child:
  artifact: 9fba430171e84dfab0b23146c69bc413ee8d18cf3b3b61b724cca0097eeccea1
  manifest: a548f11cfde99ceb4172754c7a329ef1b2617d1ca9ee17a46f1e9467901369a3
  manifest files: 279
  primary/controller verified: true/true

candidate child:
  artifact: 3223026c1b58b330aaf2ff08be4946cb67cc155ae5d3412cf665237675ad7737
  manifest: 9bcd902454b123da6307d6c7f8da549eabcf6bf190beb5241c246df233e936c5
  manifest files: 282
  primary/controller verified: true/true

parent:
  artifact: 9a67205bd12ea3bff3126ed053cd785c984fb26fc31b70f384661fbb7eb6b629
  manifest: 7572828ae628c34e43450b985ef266f1f304ecfc8bba97ac19dfd20e454fd610
  manifest files: 10
  primary/controller verified: true/true
```

After the orchestrator completed, a separate bytecode-disabled command
recomputed all six views without passing `--receipt`:

```text
baseline primary:     verified=true
baseline controller:  verified=true
candidate primary:    verified=true
candidate controller: verified=true
parent primary:       verified=true
parent controller:    verified=true
```

Both parent views produced the same artifact, baseline-artifact,
candidate-artifact, manifest, and verifier-source hashes. No verifier run
wrote a new receipt, cache, manifest, or artifact.

### Executive matrix update

| Dimension | Current evidence | Classification |
| --- | --- | --- |
| Explicit 500 ms sampler cadence | committed named constant plus generated-script contract | `ESTABLISHED` |
| Query-pressure reduction | median generations `7.5 -> 3`; observations reduced `58.9214%` | `PASS` |
| Real four-GPU source-bound execution | 8 pairs, 16 members, 80 measured repeats, 320 request samples | `COMPLETE` |
| Exact correctness and transaction parity | zero mismatches | `PASS` |
| Child and parent integrity | six fresh verifier recomputations and complete manifests | `PASS` |
| Candidate TPOT median | `78.2303015 ms <= 85.66 ms` | `PASS` |
| Candidate TPOT p95 | `122.992254 ms > 105.87 ms` | `FAIL_ABSOLUTE_LIMIT` |
| TTFT protection | `6.3775%` improvement | `PASS` |
| Throughput protection | `17.3205%` improvement | `PASS` |
| Eager/graph paired-ratio stationarity | all ratio checks pass | `PASS` |
| All sixteen child epoch stationarity | false | `FAIL` |
| Cadence-only explanation of CUDA spikes | ordinary-query overlaps coexist with severe spikes; long-query overlaps coexist with normal CUDA | `NOT_ESTABLISHED` |
| TPOT-tail benefit claim | mandatory p95 and stationarity gates are not jointly green | `NOT_ESTABLISHED` |

### Final classification

```text
GPU_SAMPLER_500MS_IMPLEMENTATION=ESTABLISHED
GPU_SAMPLER_QUERY_PRESSURE_REDUCTION=PASS_58_9214_PERCENT
SOURCE_PAIR_GATE_CANDIDATE_COMMIT=d1ebb8a19db5746c97ecf64b797e79b567f4fc7f
SOURCE_PAIR_GATE_FOUR_GPU_ADMISSION=PASS
SOURCE_PAIR_GATE_REAL_8_PAIR_CAMPAIGN=COMPLETE
SOURCE_PAIR_GATE_CHILD_MANIFESTS=PASS
SOURCE_PAIR_GATE_PARENT_MANIFEST=PASS
SOURCE_PAIR_GATE_DUAL_VERIFICATION=PASS
SOURCE_PAIR_GATE_EXACT_CORRECTNESS=PASS
SOURCE_PAIR_GATE_EAGER_GRAPH_RATIO_STATIONARITY=PASS
SOURCE_PAIR_GATE_ALL_SIXTEEN_EPOCH_STATIONARITY=FAIL
CANDIDATE_TPOT_MEDIAN_GATE=PASS
CANDIDATE_TPOT_P95_GATE=FAIL_122_992254_MS_GT_105_87_MS
TTFT_P95_NON_REGRESSION=PASS
THROUGHPUT_NON_REGRESSION=PASS
CADENCE_ONLY_CUDA_TAIL_CAUSALITY=NOT_ESTABLISHED
TPOT_TAIL_BENEFIT=NOT_ESTABLISHED
SOURCE_PAIR_GATE_CLASSIFICATION=INCONCLUSIVE_STATIONARITY
PERFORMANCE_IMPROVEMENT_ESTABLISHED=false
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

The 500 ms measurement-cadence experiment is terminal and should not be
repeated under another tag without a new causal change. The next optimization
must target the repeat-2, engine-step-4, second-spec-verify CUDA execution path
or introduce a stronger causal isolation experiment; it must not claim that
NVML cadence alone explains the observed tail.
