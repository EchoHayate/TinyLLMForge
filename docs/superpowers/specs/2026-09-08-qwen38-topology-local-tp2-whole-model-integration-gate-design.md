# Qwen3.8 Topology-Local TP2 Whole-Model Integration Gate Design

**Date:** 2026-09-08

**Status:** Approved design

**Source anchor:** `25d35e3918aba1523406cfc36c2c5668aa6fab91`

**Stage-0 mechanism source:** `83466514cac358061382dcd47b2ceabab71a6f8f`

**Target branch:** `feat/kv-sparse-attention`

**Model:** `Qwen/Qwen3.8-27B`

**Model revision:** `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

**Default:** disabled

**Primary objective:** improve real-request decode TPOT median while protecting
TPOT P99, TTFT, throughput, correctness, and memory

**Promotion boundary:** a pass authorizes a later production-style
continuous-batching gate; it does not authorize production-default enablement

## 1. Decision

Integrate the completed Stage-0 candidate into the real Qwen3.8-27B request
path and qualify it with a source-bound four-GPU paired gate.

The compared arms are:

1. **Baseline:** the current global TP4 Qwen3.8 path.
2. **Candidate:** global TP4 prefill followed by topology-local TP2-island
   decode for the 48 linear-attention layers. The integrated implementation
   retains the short-chunk gated-delta specialization for any eligible
   multi-token segment, but the frozen exact-greedy workloads are expected to
   exercise the recurrent token-one path.
3. **Service-topology control:** two independently scheduled TP2 replicas for
   concurrent workloads only. This control measures the opportunity cost of
   duplicating the same request in both islands, but it is not the
   same-request baseline and cannot determine the candidate's GO result.

The primary gate uses the real `LLMEngine` request lifecycle, all 64 decoder
layers, exact greedy sampling, real checkpoint state, scheduler steps, KV
writes, recurrent-state commits, and the final LM head. It is not a layer
replay or a projection from Stage-0 timings.

The first whole-model gate deliberately uses fixed request cohorts: all
requests in a case are admitted together, complete prefill, cross the
generation-bound TP4-to-TP2 migration boundary, and then decode without new
arrivals joining that cohort. Dynamic arrivals, preemption, prefix restore,
and mixed-layout continuous batching remain a later production-style gate.

## 2. Evidence that authorizes this stage

The predecessor gate returned
`GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE` on the frozen Qwen3.8 checkpoint and
four A100 80 GB PCIe GPUs.

The formal Stage-0 result established:

- exact downstream greedy agreement and all mixer/state tolerance checks;
- active-token 1 median critical-latency improvement of `24.985968%`;
- active-token 4 median improvement of `245.292308%`;
- active-token 8 median improvement of `192.486848%`;
- active-token 4/8 geometric aggregate improvement of `217.794680%`;
- 15 of 15 improving pairs for each measured shape;
- P99 improvements of `17.296137%`, `80.365001%`, and `58.040201%`;
- migration break-even of two, one, and one output tokens;
- projected steady-state increment of `1,845,366,144` bytes, or
  `1,759.878296 MiB`, per rank;
- peak allocated ratio `0.6509444328775996`;
- complete temporary migration release;
- 180 of 180 correctness rows and four of four lifecycle rows passing; and
- matching producer, remote verifier, and local verifier classifications.

Those measurements prove the layer mechanism and its immediate costs. They do
not prove that 48 eligible layers improve a complete request after unchanged
full-attention layers, MLPs, scheduling, sampling, state migration, and the LM
head are included. This design closes that evidence gap.

The predecessor Stage-0 attribution must remain:

> topology-local TP2 islands plus short-chunk gated-delta specialization

The whole-model result instead follows runtime hit evidence. The current
packed layer stack calls linear attention once per request segment. Ordinary
exact-greedy decode contributes one token per request, so concurrency four or
eight creates four or eight token-one segments rather than one token-four or
token-eight segment. The formal whole-model workload therefore expects zero
short-chunk calls. If the evidence confirms that expectation, any measured
whole-model gain is attributed to the topology-local TP2-island path, while
the short-chunk specialization is reported as present but inactive. The
report may name both as the integrated candidate composition only if it also
states which mechanisms were actually exercised.

## 3. Approaches considered

### 3.1 Offline whole-model tensor replay

Capture layer inputs from a baseline request and replay the baseline and
candidate offline.

Advantages:

- lowest implementation and GPU cost;
- deterministic per-layer comparison;
- easy component attribution.

Limitations:

- excludes scheduler, request lifecycle, sampling, KV writes, and state
  publication;
- cannot establish request TPOT, TTFT, QPS, or end-to-end latency;
- risks reproducing the Stage-0 evidence boundary under a larger name.

Decision: reject as the promotion authority. It remains useful only as a
debugging tool.

### 3.2 Real whole-model dual-path A/B gate

Run baseline and candidate through separate real `LLMEngine` campaign epochs
on the same four physical GPUs, same source revision, same model revision,
same requests, and frozen A/B/B/A order.

Advantages:

- directly measures user-visible request behavior;
- preserves a narrow default-off feature boundary;
- separates the candidate layout from baseline state and allocator history;
- permits exact token comparison and complete request-path telemetry.

Costs:

- four model-load epochs are expensive;
- candidate setup and migration require new lifecycle code;
- fixed cohorts do not yet prove arbitrary continuous-batching behavior.

Decision: selected.

### 3.3 Production-style continuous-batching gate

Enable dynamic arrivals, preemption, prefix restore, abort, and mixed cohorts
in the first integration attempt.

Advantages:

- strongest serving realism;
- directly exposes mixed-layout scheduling pressure.

Costs:

- a failure cannot be cleanly attributed to the mixer, migration, scheduler,
  or cache lifecycle;
- substantially enlarges the state machine before whole-model benefit is
  known;
- risks spending a scarce four-GPU window debugging unrelated admission
  behavior.

Decision: defer until this gate returns GO.

## 4. Runtime architecture

### 4.1 Feature boundary

Add one explicit, default-disabled Qwen3.8 integration mode:

```text
qwen38_topology_local_tp2_islands = false
```

The mode is accepted only when all of the following are true:

- the loaded model is the validated Qwen3.8 text adopter;
- checkpoint revision and model manifest match the frozen gate;
- dtype is BF16;
- global tensor-parallel size is four;
- the selected GPUs form two admissible topology-local pairs;
- quantization, CPU offload, speculative decoding, prefix cache, and KV
  offload are disabled;
- the request cohort is fixed before candidate decode begins; and
- the engine is in the explicitly supported eager execution policy.

Unsupported combinations fail closed before request admission. There is no
silent fallback from candidate execution to baseline after a request has
crossed the state-migration boundary.

The implementation must not modify
`tinyvllm/layers/linear.py`. Candidate-specific logical TP views and
collectives belong in new Qwen3.8-specific integration helpers or wrappers.

### 4.2 Process groups and topology

The global TP4 group remains authoritative for:

- embedding;
- all 16 full-attention layers;
- unchanged global operations;
- final LM-head ownership and token agreement; and
- engine command and cleanup coordination.

Every rank creates both pair groups in the same global order:

```text
pair A: global ranks [0, 1]
pair B: global ranks [2, 3]
```

The controller may select a different perfect matching before attempt
creation if current physical topology requires it. The attempt freezes:

- global rank;
- physical GPU index and UUID;
- pair identifier;
- logical TP2 rank;
- pair link class;
- NUMA locality; and
- the evaluated perfect-match alternatives.

The selected matching may not change within an attempt or campaign epoch.

### 4.3 Linear-attention execution

For each of the 48 linear-attention layers:

```text
prefill:
  global TP4 head quarter
  -> existing TP4 state semantics
  -> existing dense-prefill output path

decode after migration:
  logical TP2 head half
  -> pair-local convolution and recurrent delta-rule update
  -> short chunk equal to actual token count for token groups 2..8
  -> logical TP2 FP32 output-projection contribution
  -> pair-local two-rank AllReduce
  -> complete BF16 mixer output in each pair
```

The 16 full-attention layers and replicated MLPs remain unchanged. Pair A and
pair B receive the same hidden state, request order, token IDs, positions,
weights, and logical state. They independently produce the same complete
linear-attention output; no cross-pair activation exchange is added to the
candidate linear-attention timed path.

The candidate must retain the Stage-0 numerical-path decisions:

- Q/K/V use the complete fused baseline-shaped projection before logical TP2
  slicing;
- Z uses the fused contiguous logical-half A/B view;
- token count one uses the recurrent gated-delta path;
- token counts two through eight use `chunk_size=token_count`;
- larger token groups retain the current chunk size;
- output projection uses the logical TP2 half derived from the preserved
  dense checkpoint weight; and
- pair-local accumulation remains FP32 before BF16 materialization.

For the frozen exact-greedy gate, every decode request segment is expected to
have token count one. The recurrent gated-delta call count must therefore
equal the independently derived eligible segment count, and the short-chunk
and ordinary-chunk call counts must both be zero. A nonzero multi-token count
is not silently discarded; it changes the exercised mechanism and makes the
formal workload inventory invalid.

### 4.4 Cross-pair replication invariant

The unchanged global TP4 full-attention path assumes that every rank enters a
layer with the same replicated hidden state. Pair A and pair B therefore
cannot merely be numerically close at a boundary that later feeds global TP4
computation: they must execute deterministically and produce bitwise-identical
BF16 mixer outputs.

The untimed correctness campaign records device-side digests and exact
cross-pair equality for every eligible layer output. No cross-pair
canonicalization collective or host comparison is allowed inside the timed
candidate path. If exact equality cannot be established without such a
collective, this exact integration design returns
`NO_GO_CORRECTNESS_OR_LIFECYCLE`; a later design may evaluate an explicit
reconciliation operation and must measure its cost.

### 4.5 Weight ownership

The integrated candidate must not retain all 48 baseline TP4 FP32
output-projection accumulation shards alongside all 48 candidate TP2 shards.
After candidate construction and before measured warmup:

- each eligible layer retains one candidate logical-half FP32 accumulation
  weight;
- each eligible layer retains the already-required dense BF16 prefill weight;
- the baseline TP4 FP32 decode shard is released;
- fused BF16 A/B logical-half weights are resident and immutable; and
- full-attention and MLP parameters are unchanged.

The worker records object-lifecycle proof and allocator observations for the
released baseline accumulation weights. Calculated tensor bytes and physical
allocated/reserved bytes are reported separately.

## 5. State lifecycle

### 5.1 Two layouts, one generation

Before migration, an active request owns the existing global TP4 state
quarters. After migration, it owns replicated logical TP2 halves:

```text
logical half 0 = TP4 quarters 0 + 1
logical half 1 = TP4 quarters 2 + 3

pair A receives halves 0 and 1
pair B receives halves 0 and 1
```

The state record is bound to:

- request identity;
- generation identity;
- lease and slot identity;
- model revision;
- layer inventory;
- source layout fingerprint;
- destination layout fingerprint; and
- pair map.

A state becomes candidate-visible only after every layer and both pair
replicas complete validation. Publication is one generation-sealed commit.
Partial migration is never visible to decode.

### 5.2 Migration placement

Migration occurs once per request after its final prefill state is committed
and before its first candidate decode step. Migration latency is part of
candidate TTFT and end-to-end request latency. It is also recorded separately
for amortization analysis.

The fixed-cohort gate requires all requests in one case to complete migration
before the first candidate decode scheduler step. This avoids a batch that
mixes TP4-state and TP2-state linear-attention requests.

Temporary gather destinations and conversion buffers must be released before
the first measured decode step. The worker records:

- logical bytes read, transferred, and retained;
- migration duration by request and cohort;
- peak allocated and reserved memory during migration;
- weak-reference or equivalent object-lifecycle release evidence;
- post-release allocated and reserved memory; and
- observed break-even output-token count.

### 5.3 Failure behavior

Before migration publication, a failure leaves the baseline TP4 generation
authoritative and terminates the candidate case.

After migration publication, a failure:

- does not rerun the same step through baseline;
- does not publish a partially updated candidate generation;
- quarantines the affected request generation;
- terminates the case;
- preserves the immutable failure artifacts; and
- cleans only processes and files owned by the attempt.

The first gate does not recover the request in place. Proving recovery,
preemption, prefix restoration, abort reuse, and speculative rollback is part
of the later production-style lifecycle gate.

## 6. Real candidate-hit authority

A feature flag or successful request is not proof that the candidate ran.
Every rank emits structured counters keyed by campaign epoch, case, scheduler
step, rank, and layer:

- TP4 linear-attention prefill calls;
- TP4-to-TP2 migration calls;
- TP2-island linear-attention decode calls;
- recurrent token-one calls;
- short-chunk token-two-through-eight calls;
- ordinary chunk calls above eight tokens;
- pair-local AllReduce calls and bytes;
- global TP4 linear-attention decode AllReduce calls and bytes;
- global TP4 full-attention collective calls and bytes;
- baseline fallback calls and reasons;
- pair-replica comparison failures;
- state publications; and
- post-warmup allocations.

The verifier derives expected counts from the model manifest, request rows,
scheduler-step rows, and token-count rows. Hard-coded producer totals are not
accepted.

Candidate coverage requires:

- all 48 eligible layers use TP2-island decode for every candidate decode
  step;
- recurrent token-one calls equal the derived eligible request-segment count;
- short-chunk and ordinary-chunk calls are zero for the frozen exact-greedy
  workloads;
- zero global TP4 linear-attention decode AllReduce calls;
- all 16 full-attention layers retain the expected global TP4 behavior;
- every request has exactly one successful migration publication;
- zero fallback calls; and
- pair-local call/byte sequences agree inside each pair.

Missing or excess events invalidate the candidate result even when token
outputs match.

## 7. Frozen workload matrix

Use the established Qwen3.8 workloads:

| Workload | Prompt tokens | Output tokens | Concurrency | Role |
|---|---:|---:|---:|---|
| P0 | 256 | 128 | 1 | short-prompt single-request decode |
| P1 | 2,048 | 128 | 1 | long-prompt TTFT protection |
| Q0 | 256 | 128 | 4 | moderate-concurrency decode |
| Q1 | 256 | 128 | 8 | capacity and state-memory pressure |
| Q2 | 2,048 | 128 | 4 | long-prefill concurrent protection |

All cases use:

- BF16;
- global TP size four;
- exact greedy decoding with temperature zero;
- EOS ignored until exactly 128 output tokens;
- deterministic token-ID prompts;
- identical request identities and admission order between arms;
- identical eager/CUDA-Graph policy;
- no profiler during authoritative timing; and
- no threshold or workload changes after measured rows are visible.

### 7.1 A/B/B/A campaign epochs

The authoritative campaign consists of four engine epochs:

```text
epoch 0: baseline
epoch 1: candidate
epoch 2: candidate
epoch 3: baseline
```

Each epoch loads the model once, performs two unmeasured warmups per workload,
then records five measured repetitions per workload. Epochs zero and two use
the workload order `P0, P1, Q0, Q1, Q2`; epochs one and three use the reverse
order. The manifest pairs rows by workload, request-set digest, repetition,
and mirrored epoch position.

This produces ten measured baseline and ten measured candidate observations
per workload while limiting model-load count to four. Model load and initial
construction are reported as startup cost but excluded from TTFT and TPOT.

The candidate's per-request state migration is not startup: it remains inside
the measured request lifecycle and TTFT.

### 7.2 Service-topology control

After the primary A/B/B/A campaign, run Q0, Q1, and Q2 through two independent
TP2 engines, one on each selected pair. Requests are deterministically split
between replicas.

Report:

- aggregate request QPS;
- aggregate output tokens per second;
- TTFT and TPOT distributions;
- per-replica balance;
- peak memory by GPU; and
- exact output parity with the global TP4 baseline.

This control answers whether conventional request sharding is better for
aggregate concurrency. It must be labeled `TP2_X2_SERVICE_CONTROL` and cannot
be used to pass or fail the same-request candidate.

## 8. Correctness authority

### 8.1 Untimed correctness campaign

Before formal timing, run an untimed source-bound correctness campaign over
all five workloads. Each workload contains five deterministic request sets
that are disjoint from timing prompts.

For every request:

- all 128 generated token IDs exactly match baseline;
- stop position and stop reason match;
- every emitted token equals the rank-zero exact greedy argmax;
- all ranks agree on the selected token;
- logits are finite;
- recorded top-logit IDs and values satisfy the frozen numeric tolerance;
- candidate pair replicas have bitwise-identical BF16 mixer outputs and
  canonicalized convolution and recurrent state;
- migrated state reconstructs the baseline TP4 logical full-head order within
  the Stage-0 state tolerance;
- request, generation, lease, slot, and layer identities match; and
- neither arm commits more than one state transition per scheduler step.

State comparisons are required:

- immediately before migration;
- immediately after migration;
- after output tokens 1, 4, 8, 32, and 128; and
- for every one of the 48 linear-attention layers.

The correctness worker may retain additional baseline state and weights that
are prohibited in performance timing, but it must report that overhead and
must not supply performance measurements.

### 8.2 Timed-run correctness

Every authoritative timing row also records:

- generated token IDs and decoded-text hash;
- stop identity;
- rank token agreement;
- candidate-hit coverage;
- migration publication count;
- finite-output checks;
- fallback count; and
- cleanup identity.

An untimed correctness pass cannot excuse a mismatch in a timed row.

## 9. Metrics

### 9.1 Primary benefit

The primary metric is paired request TPOT:

```text
request TPOT =
  (last-token timestamp - first-token timestamp)
  / (generated-token count - 1)
```

Report P50, P95, and P99 by workload and aggregate. Aggregate ratios use the
geometric mean of workload-level candidate-to-baseline ratios. Every
percentage includes absolute baseline and candidate values.

The per-request TPOT values are the median-benefit authority. TPOT P95 and
P99 are reconstructed from all individual post-first-token inter-arrival gaps
within the workload, preserving epoch and request identity. The report also
includes request-level TPOT tails, but the classifier does not pretend that
ten request summaries provide a stable empirical P99.

### 9.2 Protected service metrics

Report:

- TTFT P50/P95/P99;
- end-to-end request latency P50/P95/P99;
- request QPS;
- output tokens per second;
- queueing time;
- scheduler-step duration;
- per-token inter-arrival gaps; and
- completion spread within concurrent cohorts.

QPS and output throughput use the complete cohort makespan, not the sum of
per-request durations.

### 9.3 Mechanism and cost metrics

Report:

- state-migration median/P95/P99 and observed break-even;
- candidate-hit and fallback counts;
- pair-local and global collective counts and bytes;
- eligible-layer coverage;
- per-rank peak allocated and reserved CUDA memory;
- physical GPU-memory telemetry;
- calculated persistent weight and state bytes;
- temporary migration peak and post-release bytes;
- startup/model-load duration;
- host-submission median/P99;
- GPU utilization and power; and
- cleanup duration.

The primary run remains unprofiled. A separate diagnostic subset may measure
component durations, but profiler-derived values cannot replace request
timings in the classifier.

## 10. Formal GO gate

Return `GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE` only if every condition below
passes.

### 10.1 Correctness and execution coverage

- every untimed and timed generated-token sequence exactly matches baseline;
- every rank agrees on exact greedy tokens;
- all numeric and canonical-state checks pass;
- every candidate request migrates exactly once;
- all 48 eligible layers use the candidate on every candidate decode step;
- no candidate decode uses a global TP4 linear-attention AllReduce;
- all full-attention layers retain their expected TP4 path;
- there are zero fallback, retry-after-mutation, or duplicate-commit events;
- there are zero post-warmup request-path allocations; and
- all request, generation, lease, slot, rank, pair, and model identities
  remain valid.

### 10.2 Performance

- aggregate median TPOT improves by at least `5%`;
- at least four of five workloads improve median TPOT;
- no workload's median TPOT regresses;
- no workload's TPOT P99 regresses by more than `2%`;
- no workload's TTFT median or P99 regresses by more than `2%`;
- aggregate Q0/Q1/Q2 request QPS does not regress by more than `2%`;
- no individual online workload's request QPS regresses by more than `2%`;
- aggregate Q0/Q1/Q2 output tokens per second does not regress by more than
  `2%`;
- at least seven of ten paired observations improve TPOT in every workload;
  and
- measured migration break-even is no more than 32 output tokens for every
  workload.

The service-topology control has no promotion threshold. Its benefit and cost
must still be reported.

### 10.3 Memory and lifecycle

- candidate steady-state calculated increment is at most `1,920 MiB/rank` at
  concurrency eight;
- peak allocated memory remains below `98%` of physical memory on every rank;
- no baseline TP4 FP32 linear-attention decode accumulation copy remains
  resident in the candidate performance arm;
- temporary migration objects are proven released before measured decode;
- no leaked state generation, lease, tensor, process group, owned process, or
  task-owned file remains after cleanup; and
- cleanup never signals, terminates, adopts, or modifies a foreign process.

### 10.4 Evidence

- strict-clean admission and launch-time identity checks pass;
- source, model, environment, topology, workload, and rank manifests pass;
- all four epochs and the service control are complete;
- producer, remote independent verifier, and local independent verifier
  independently reconstruct the same terminal classification;
- terminal artifact hashes match after compact download; and
- no required raw row or terminal artifact is missing or extra.

No positive performance metric offsets a correctness, coverage, tail,
throughput, memory, lifecycle, resource, or evidence failure.

## 11. Terminal classifications

The classifier uses this precedence:

1. `NO_GO_CORRECTNESS_OR_LIFECYCLE`
2. `NO_GO_RESOURCE_IDENTITY`
3. `NO_GO_CANDIDATE_NOT_EXERCISED`
4. `NO_GO_MEMORY_OR_ALLOCATION`
5. `INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT`
6. `NO_GO_TAIL_OR_TTFT`
7. `NO_GO_THROUGHPUT`
8. `NO_GO_MIGRATION_AMORTIZATION`
9. `NO_GO_PERFORMANCE`
10. `GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE`

An admission failure before worker launch is
`BLOCKED_ADMISSION`, not a performance result.

A failed immutable attempt is preserved and never repaired or reused. A later
attempt requires a fresh tag. A performance no-go may be revisited only after
a measured root cause motivates a source change; it cannot be rescued by
dropping workloads, changing thresholds, selecting only favorable epochs, or
rerunning for a lucky sample.

## 12. Controller and remote-execution contract

The controller:

- uses `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- never runs `kinit` or `krenew`;
- requires at least 1,800 seconds of Kerberos lifetime at launch;
- uses exactly four strict-clean GPUs;
- defines strict-clean as at most `1,024 MiB` used memory, at most `5%`
  utilization, and an empty compute-process inventory;
- freezes their UUID-to-rank and pair mapping;
- does not kill, pause, adopt, or alter foreign workloads;
- writes remote task-owned data only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`;
- never writes task data to remote `/`, `/tmp`, or an old checkout;
- keeps large raw data and source snapshots remote;
- downloads only the compact sealed final bundle;
- retries only transport failures within the frozen retry budget; and
- preserves every failed attempt.

Admission is checked at controller entry, immediately before each epoch, and
after worker launch. After launch, every GPU process must belong to the
current attempt. Any foreign-process appearance invalidates the epoch and
terminates only task-owned workers.

## 13. Artifact contract

Each formal attempt uses a fresh immutable tag below:

```text
/data00/home/sitian/tinyllmforge-workspaces/
  command-timeline-20260818/attempts/
```

The compact local bundle is stored below:

```text
artifacts/qwen38_topology_local_tp2_whole_model/
  <attempt-tag>/final_bundle/
```

The terminal bundle contains at least:

```text
source_manifest.json
model_manifest.json
environment_manifest.json
gpu_topology.json
gpu_rank_manifest.json
pair_group_manifest.json
workload_manifest.json
campaign_epoch_manifest.json
feature_contract.json
weight_layout_manifest.json
state_layout_manifest.json
migration_rows.jsonl
correctness_rows.jsonl
request_rows.jsonl
scheduler_step_rows.jsonl
candidate_hit_rows.jsonl
collective_rows.jsonl
memory_rows.jsonl
resource_rows.jsonl
service_control_rows.jsonl
cleanup.json
classification.json
remote_independent_verification.json
local_independent_verification.json
report.md
manifest.json
manifest.sha256
```

The manifest rejects missing and extra terminal files. It binds every file by
relative path, size, and SHA-256 digest.

The independent verifier must use only the Python standard library. It may not
import the worker, assembler, controller, TinyLLMForge runtime, or producer
classification code. The local verifier runs from frozen downloaded verifier
source and reconstructs all metrics, thresholds, inventories, and
classification from raw rows.

Remote and local verifier receipts must be byte-identical. Location-specific
metadata belongs outside the signed semantic receipt.

## 14. Prompt-to-artifact checklist

| Requirement | Required evidence |
| --- | --- |
| Real Qwen3.8-27B request path | model manifest plus `LLMEngine` request and scheduler-step rows |
| Frozen checkpoint | repository, revision, shard inventory, and model digest |
| Baseline versus candidate | four-epoch A/B/B/A manifest and paired request rows |
| Exact greedy output identity | all token IDs, stop identity, argmax agreement, decoded-text hashes |
| Whole-model execution | 64-layer inventory and complete per-step layer coverage |
| Candidate actually exercised | independently derived hit, migration, fallback, and collective counts |
| TPOT benefit | raw token timestamps plus recomputed P50/P95/P99 and paired ratios |
| P99 protection | workload-level raw distributions and frozen 2% threshold |
| TTFT protection | admission and first-token timestamps including migration |
| QPS and throughput | cohort admission/completion timestamps and makespan-derived metrics |
| Migration cost | per-request/cohort migration rows and break-even reconstruction |
| Memory cost | calculated tensor inventory, allocator peaks, physical telemetry, release proof |
| TP2 pair locality | topology matrix, perfect matching, UUID-to-rank and pair manifest |
| Full-attention unchanged | TP4 collective sequence/count/byte evidence for 16 layers |
| No hidden TP4 linear collective | zero-count proof joined against all 48 eligible layers |
| Fixed-cohort scope | request admission manifest and no-arrival-during-decode proof |
| TP2 x2 service control | separate control rows and clearly non-authoritative classification field |
| Strict-clean GPUs | entry, pre-epoch, post-launch, and terminal resource inventories |
| No foreign-process action | ownership manifest and cleanup receipt |
| Approved remote storage | path audit covering all task-created remote files |
| Immutable attempts | fresh tag, no-overwrite receipt, source archive, and terminal manifest |
| Dual verification | producer, remote independent verifier, local independent verifier |
| Compact local storage | final-bundle inventory excluding raw remote-only payloads |
| Claim boundary | report classification and explicit production-default prohibition |

## 15. Expected implementation boundaries

The implementation plan should prefer these new boundaries:

- topology-local pair-group identity helper;
- Qwen3.8 logical TP2 projection and parameter-view helper;
- candidate-specific linear-attention wrapper;
- generation-sealed TP4-to-TP2 state migration transaction;
- candidate lifecycle and hit telemetry;
- model/engine configuration plumbing;
- real-request baseline/candidate/service-control worker;
- controller, assembler, manifest builder, and independent verifier;
- focused CPU tests for topology, slicing, state order, lifecycle,
  classification, and evidence integrity; and
- source-bound four-GPU correctness and performance campaigns.

The implementation must preserve existing defaults and avoid changing global
linear-layer behavior.

## 16. Non-goals

This gate does not:

- enable the feature by default;
- establish arbitrary continuous-batching safety;
- support dynamic arrivals during candidate decode;
- prove preemption, prefix restore, abort recovery, speculative rollback, or
  mixed TP4/TP2 state batches;
- optimize prefill;
- alter full-attention or MLP semantics;
- add quantization, KV offload, speculative decoding, or sparse attention;
- compare TinyLLMForge with another inference engine;
- treat the TP2 x2 service control as the same-request candidate;
- infer benefit from Stage-0 numbers, unit tests, process existence, a
  complete manifest, or a verifier receipt alone; or
- claim production readiness from a GO result.

## 17. Result and promotion boundary

If the gate returns `GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE`, the permitted
claim is:

> On the frozen Qwen3.8-27B BF16 four-GPU hardware and fixed-cohort workload,
> the default-off topology-local TP2-island candidate improved measured
> whole-model decode TPOT while satisfying the frozen correctness,
> tail-latency, TTFT, throughput, migration, memory, lifecycle, resource, and
> evidence gates. The integrated short-chunk specialization was present but
> inactive because the exact-greedy request path used token-one segments.

The GO authorizes a separate design for production-style continuous batching,
including dynamic arrivals, layout-aware scheduling, preemption, abort,
prefix restore, speculative rollback, and fallback policy.

It does not authorize:

- production-default enablement;
- results on another model, checkpoint, topology, TP size, or dtype;
- attributing a measured gain to any mechanism not exercised according to the
  candidate-hit rows;
- a universal QPS or latency claim; or
- an originality or first-publication claim.

If the result is any NO_GO or INCONCLUSIVE classification, the terminal report
must still publish both measured benefit and measured cost, preserve the
failed attempt, and state the exact failed gates.
