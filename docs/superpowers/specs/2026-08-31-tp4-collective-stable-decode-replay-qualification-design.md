# TP4 Collective-Stable Decode Replay Qualification Design

## Status

Approved design for a bounded Stage-0 qualification. The user has already
authorized continuing the recommended performance-optimization route without
repeated approval prompts.

This stage does not change the production default. It determines whether the
existing exact multi-sequence CUDA Graph path has enough real Qwen3.8-27B TP4
benefit to justify a later distributed admission contract.

## Objective

Qualify one model-neutral optimization mechanism:

> Replay a fixed-shape, full-batch TP decode step as one CUDA Graph while
> preserving the model computation, KV writes, NCCL collective order, output
> semantics, and scheduler-visible batch.

The first adopter is Qwen3.8-27B BF16 on one node with four A100 80 GB PCIe
GPUs. The gate compares:

- baseline: `enforce_eager=True`;
- candidate: `enforce_eager=False` and
  `multi_sequence_cuda_graphs=True`;
- workloads: Q0, Q1, and Q2;
- decoding: greedy, 128 output tokens;
- topology: one GPU per process, TP4.

Every result reports both benefit and cost. A CUDA Graph microbenchmark or a
successful capture is not an end-to-end performance result.

## Why This Is the Next Candidate

The previous cross-request wavefront experiment split one full batch into
smaller waves. It reduced GEMM efficiency and was terminally classified
`NO_GO_INSUFFICIENT_OVERLAP`. This design keeps the full batch intact.

The current Qwen3.8-27B TP4 eager path exposes repeated host submission work
around a fixed decode shape. CUDA Graph replay can remove launch gaps without
changing the mathematical work or collective sequence. Unlike peer reduction,
synchronous collective reduction, or wavefront overlap, this candidate does
not replace collectives or fragment GEMMs.

The expected ceiling is deliberately modest. Qwen3.8-27B decode is heavily
GPU-compute bound, so a result below the frozen 5% promotion threshold stops
the direction even if graph replay itself works.

## Current Source Findings

The repository already contains the generic mechanism:

- `tinyvllm/engine/model_runner.py`
  - `_build_multi_sequence_graph_identity()` binds active batch, exact page
    table width, FlashAttention split, device properties, and attention shape;
  - `_capture_exact_multi_sequence_graph()` captures
    `self.model(input_ids, positions)` and therefore captures the TP model
    collectives reached by that forward;
  - `_replay_exact_multi_sequence_graph()` copies exact-shaped runtime inputs,
    replays once, and raises on replay failure instead of rerunning eager after
    possible KV mutation;
  - `_multi_sequence_graph_incompatible_reason()` keeps unsupported modes and
    features eager;
  - `capture_cudagraph()` preserves the legacy batch-one startup graph and
    leaves multi-sequence entries to exact lazy capture.
- `tinyvllm/engine/exact_cuda_graph_cache.py`
  - keeps the feature default-off;
  - bounds entries, static bytes, reserved bytes, and capture time;
  - permanently rejects failed identities;
  - exposes capture, hit, miss, and rejection state.
- `tools/multi_sequence_cuda_graph_contract.py`,
  `tools/run_multi_sequence_cuda_graph_production_gate_remote.py`, and
  `tools/verify_multi_sequence_cuda_graph_production.py`
  - provide an older single-GPU Qwen3-0.6B production gate and reusable
    artifact/verifier patterns.

The older gate is not authority for this stage:

- it is fixed to one GPU and Qwen3-0.6B;
- its old remote paths violate the current storage boundary;
- it does not prove that all four ranks capture and replay the same collective
  graph generation;
- it predates the Qwen3.8-27B workloads and memory pressure;
- no retained artifact was found that establishes Qwen3.8-27B TP4 graph
  performance.

## NCCL Constraint

NCCL supports capture of collective operations. A graph launch containing a
captured collective is itself collective: every rank that participated in
capture must launch the corresponding graph. Capture participation must also
be uniform across ranks.

TinyLLMForge uses one GPU per process, which follows NVIDIA's recommended
shape for reliable NCCL graph capture. Stage 0 must nevertheless prove
cross-rank uniformity from per-rank evidence; rank-0 success is insufficient.

Reference:

- NVIDIA NCCL User Guide, “Using NCCL with CUDA Graphs”:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html>

## Considered Approaches

### A. Qualify the Existing Full-Step Graph First

Reuse the default-off exact multi-sequence mechanism and add only a TP4,
source-bound qualification worker, controller, and independent verifiers.

Advantages:

- fastest route to real benefit/cost evidence;
- no production runtime mutation before the performance ceiling is known;
- keeps full-batch GEMMs and current NCCL order;
- reuses the canonical engine entrypoint and exact graph cache.

Costs:

- existing runtime has no explicit distributed admission consensus;
- Stage 0 can prove controlled uniformity, not arbitrary-failure robustness;
- a positive result still requires a later promotion decision.

### B. Add Distributed Graph Admission Consensus Before Measuring

Add a rank-synchronous prepare/commit protocol so an identity becomes replayable
only after every rank reports the same identity, capture result, and generation.

Advantages:

- directly closes asymmetric capture/rejection risk;
- creates a stronger production lifecycle contract.

Costs:

- adds control-plane collectives and state before benefit is established;
- can erase a small launch-overhead gain;
- expands failure, teardown, timeout, and generation semantics prematurely.

### C. Capture Compute Segments Between Eager NCCL Collectives

Leave NCCL launches eager and graph only the compute regions around them.

Advantages:

- avoids collective graph-uniformity requirements;
- isolates a rank-local graph failure.

Costs:

- retains all host-side collective submission gaps;
- requires many graph launches per layer;
- creates invasive model partitioning and model-family leakage;
- has a lower expected ceiling than full-step replay.

## Decision

Use approach A.

Stage 0 is a qualification, not a production rollout. It may add test and gate
tooling, but it must not change the default runtime behavior or add a
distributed admission protocol.

If Stage 0 is positive, a separate Stage-1 design decides whether to:

1. add distributed prepare/commit admission;
2. retain the feature as an explicitly controlled topology mode; or
3. stop because admission cost or lifecycle risk exceeds the measured gain.

If Stage 0 is negative or incomplete, do not implement Stage 1.

## Capability Restatement Without Model Nouns

A tensor-parallel runtime observes a repeated exact decode shape. After a
bounded number of successful eager observations, each rank captures the same
ordered compute-and-collective step into a rank-local graph. Later steps copy
new values into stable input buffers and all ranks replay the same graph
generation once.

The mechanism consumes runtime shape, topology, device, and lifecycle
identity. It does not consume model names, prompt categories, or checkpoint
business semantics.

## Two-Axis Verdict

- Mechanism: `reusable candidate`.
- Integration: `clean` in core, `first adopter only` in this Stage-0 profile.

The core mechanism has an older independent caller/profile, but TP4 collective
stability and large-model benefit are not yet proven. Therefore it is not yet
classified as a generically qualified distributed optimization.

## Layer Map

### Mechanism

- exact graph identity;
- bounded observation and capture admission;
- scratch-KV isolation;
- capture, replay, quarantine, and no-retry-after-replay semantics;
- rank-local dispatch events.

Owned by:

- `tinyvllm/engine/model_runner.py`;
- `tinyvllm/engine/exact_cuda_graph_cache.py`;
- `tinyvllm/engine/flash_attn_split_policy.py`.

### Adapter

No model-specific adapter is required. The canonical `LLMEngine` and
`ModelRunner` paths already translate scheduler state into tensors and graph
identity.

### Policy and Configuration

- feature remains default-off;
- batch allowlist remains `(2, 4, 8)`;
- observation and memory/time ceilings remain configuration policy;
- no Qwen-specific field enters `tinyvllm/`.

### Benchmark and Profile

The following are profile-only:

- Qwen3.8-27B repository and revision;
- BF16 and TP4;
- A100 80 GB PCIe hardware;
- Q0/Q1/Q2 prompt, output, and concurrency values;
- warmups, repetitions, thresholds, and artifact locations.

## Leakage Evidence

No current core symbol consumes a Qwen3.8 name or Q0/Q1/Q2 label.

The new gate must preserve that split:

- generic artifact schemas and verifier logic may use workload IDs but must
  derive workload parameters from a frozen profile;
- checkpoint path and revision belong only in the profile/manifest;
- remote storage paths belong only in the controller;
- core graph identity must not include workload names, prompt length labels,
  or checkpoint names;
- the worker must enter through `LLMEngine`, not call model internals as a
  parallel benchmark path.

Producer-to-consumer flow:

```text
frozen workload profile
  -> canonical LLMEngine request submission
  -> scheduler creates exact runtime tensors
  -> ModelRunner builds generic graph identity
  -> rank-local cache observes/captures/replays
  -> rank-local evidence rows
  -> producer aggregation
  -> remote independent verifier
  -> local frozen-source verifier
```

## Recommended Contribution Split

### Stage 0: Validation Follow-Up

Add only:

- a frozen qualification contract/profile;
- a TP4 worker with one evidence stream per rank;
- a local controller and remote storage discipline;
- a producer classifier;
- a remote independent verifier;
- a local frozen-source verifier;
- a terminal audit and compact immutable bundle.

Do not modify core runtime unless Stage-0 RED tests expose a defect that makes
the existing qualification impossible. Any such change requires a revised
design before implementation.

### Stage 1: Conditional Core RFC

Only after Stage-0 `GO`, define:

- all-rank identity agreement;
- capture prepare/commit/abort;
- graph generation identity;
- asymmetric rank failure and timeout handling;
- teardown and communicator replacement behavior;
- observability cost and control-plane overhead.

### First Adopter

Qwen3.8-27B TP4 is the first large-model collective-capture profile, not part
of the generic mechanism.

### Second Proof Path

The older Qwen3-0.6B single-GPU path is evidence for rank-local reuse only. A
second distributed model/topology is required before claiming broad TP
genericity.

## Stage-0 Data Flow

Each paired repetition runs in a fresh process group:

1. controller freezes source, patch, dependency, model, topology, workload,
   order, seeds, ports, and storage identity;
2. baseline or candidate process group loads the same checkpoint;
3. each rank writes its own environment and graph-dispatch stream;
4. requests enter through the canonical engine;
5. candidate warmup allows exact identities to become ready;
6. measured requests run after warmup;
7. all ranks synchronize and publish terminal cache/lifecycle receipts;
8. rank 0 publishes request-level metrics and outputs;
9. producer classifies only after all four rank streams are complete;
10. remote and local verifiers independently reconstruct the classification.

No rank may infer another rank's success from rank 0.

## Cross-Rank Collective-Stability Contract

For every candidate process group and measured graph replay:

- ranks are exactly `{0, 1, 2, 3}`;
- world size is exactly four;
- all ranks report the same ordered measured step IDs;
- each step has one common graph identity SHA;
- each step has one common dispatch kind;
- capture attempt, terminal cache state, and rejection reason agree;
- graph replay count advances by one on every rank;
- collective inventory and order digest agree across ranks;
- no rank records eager while another records graph for the same step;
- no replay exception, hang, timeout, or rank-local process failure occurs;
- all ranks publish teardown completion.

Any violation is terminal `NO_GO_CORRECTNESS_OR_LIFECYCLE`, not a performance
sample.

The existing runtime's lack of a distributed admission handshake remains an
explicit evidence boundary even when this controlled gate passes.

## Correctness Contract

For every paired workload and repetition:

- output token IDs are exactly equal;
- output lengths and stop reasons are exactly equal;
- request IDs and prompt hashes match;
- no request is missing or duplicated;
- all-rank final live-KV digests match the expected rank-local eager/candidate
  relation defined by the verifier;
- graph replay is observed for candidate measured decode;
- baseline records no graph replay;
- no eager retry occurs after an authoritative replay begins;
- process and communicator exits are clean.

Logit tolerance is not a substitute for exact greedy token equality. If logits
are retained, the verifier reports both maximum absolute and relative
difference, but token/length/stop equality remains mandatory.

## Workloads

Freeze these existing Qwen3.8 profiles:

| ID | Prompt tokens | Output tokens | Concurrency |
|---|---:|---:|---:|
| Q0 | 256 | 128 | 4 |
| Q1 | 256 | 128 | 8 |
| Q2 | 2048 | 128 | 4 |

Candidate measured rows are valid only when:

- active batch is graph-allowlisted;
- the exact identity reaches `ready`;
- steady-state replay is actually observed;
- measured replay coverage is at least 80% of graph-eligible decode steps.

Low replay coverage is `NO_GO_MECHANISM_NOT_EXERCISED`, not an eager
performance result attributed to CUDA Graphs.

## Measurement Protocol

- hardware: one host, four A100 80 GB PCIe GPUs;
- one GPU per process;
- dtype: BF16;
- tensor parallel size: four;
- greedy decoding;
- fixed 128 output tokens;
- paired process order alternates by repetition;
- at least one unmeasured load/capture warmup per exact identity;
- at least five measured paired repetitions per workload;
- fresh process group per arm;
- CUDA synchronization around measured boundaries;
- capture cost excluded from steady-state benefit and reported separately;
- large traces remain remote;
- only compact, verifier-required evidence is downloaded locally.

The controller must record GPU clocks, P-state, utilization, memory, other
compute processes, and host load. It must not terminate external processes.

## Metrics

### Benefit

- output tokens/s;
- aggregate request QPS;
- median TPOT;
- P95 and P99 TPOT;
- median and P99 end-to-end latency;
- host submit gap per decode step when available;
- graph replay coverage.

### Cost

- TTFT;
- cold capture duration per identity and per rank;
- added peak allocated memory per rank;
- added peak reserved memory per rank;
- static graph tensor bytes;
- initialization time;
- cache misses and rejected identities;
- teardown duration;
- verifier and artifact completeness.

## Frozen Gate

Correctness and lifecycle are hard gates.

Performance `GO` requires all of:

- aggregate output throughput improvement across Q0/Q1/Q2 `>= 5%`;
- aggregate median TPOT improvement `>= 5%`;
- each workload output throughput ratio `>= 0.97`;
- each workload median TPOT ratio `<= 1.03`;
- each workload P99 end-to-end latency ratio `<= 1.03`;
- each workload TTFT ratio `<= 1.03`;
- candidate replay coverage `>= 0.80`;
- added peak allocated memory `<= 512 MiB` per rank;
- added peak reserved memory `<= 512 MiB` per rank;
- no workload has a terminal correctness or lifecycle failure.

Ratios use candidate divided by baseline for latency/memory and candidate
divided by baseline for throughput. Aggregate improvements are reconstructed
from paired raw rows, not from producer-written summary values.

Capture cost is reported but not included in steady-state gain. The report
must additionally state the number of output tokens required to amortize the
observed capture cost. A steady-state `GO` does not hide a large cold-start
penalty.

## Classification

The producer and both independent verifiers return exactly one:

- `GO_STAGE1_JUSTIFIED`;
- `NO_GO_PERFORMANCE`;
- `NO_GO_CORRECTNESS_OR_LIFECYCLE`;
- `NO_GO_MECHANISM_NOT_EXERCISED`;
- `INCOMPLETE`.

Precedence:

1. missing or unverifiable mandatory evidence -> `INCOMPLETE`;
2. correctness or lifecycle failure ->
   `NO_GO_CORRECTNESS_OR_LIFECYCLE`;
3. insufficient replay coverage -> `NO_GO_MECHANISM_NOT_EXERCISED`;
4. complete evidence that misses any performance/cost threshold ->
   `NO_GO_PERFORMANCE`;
5. otherwise -> `GO_STAGE1_JUSTIFIED`.

Only `GO_STAGE1_JUSTIFIED` authorizes a Stage-1 design. It does not authorize
default enablement or a broad performance claim.

## Artifact Contract

Remote root:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  tp4-collective-stable-decode-replay/<run-tag>/
```

Required terminal evidence:

- `manifest.json`;
- `source_manifest.json`;
- `source.patch`;
- `environment.json`;
- `gpu_inventory.json`;
- `workload_profile.json`;
- `process_receipts.json`;
- `rank_environment.jsonl`;
- `rank_dispatch_events.jsonl`;
- `rank_collective_events.jsonl`;
- `rank_lifecycle_rows.jsonl`;
- `request_rows.jsonl`;
- `performance_rows.jsonl`;
- `memory_rows.jsonl`;
- `correctness_rows.jsonl`;
- `capture_cost_rows.jsonl`;
- `summary.json`;
- `producer_classification.json`;
- `remote_independent_verification.json`;
- `remote_post_verification_manifest.json`;
- `local_frozen_source_verification.json`;
- `report.md`.

The manifest hashes every verifier input. Verifiers reject partial final
lines, duplicate row IDs, missing ranks, unknown enums, non-finite numbers,
hash drift, source drift, workload drift, or producer-only conclusions.

## Remote and Controller Safety

- All task data, caches, logs, source snapshots, and temporary files stay
  below the frozen `/data00/home/sitian/tinyllmforge-workspaces/` root.
- Do not write task data to remote `/`, `/tmp`, model cache directories, or
  old checkouts.
- Do not run `kinit` or `krenew`.
- Do not terminate, take over, or clean external GPU processes.
- A local controller monitors the remote host and launches immediately only
  when four clean GPUs satisfy the frozen admission rule.
- A clean GPU means no external compute process and no unexplained material
  memory allocation; merely having enough free memory is not equivalent.
- Use fresh ports and a fresh run tag.
- Do not duplicate an active worker for the same run tag.
- Keep large traces remote and download only compact terminal evidence.

## Evidence Boundary

Stage-0 `GO` would prove:

- exact greedy Qwen3.8-27B TP4 parity for Q0/Q1/Q2;
- controlled all-rank capture/replay uniformity;
- bounded steady-state benefit and measured cost on the named A100 topology;
- enough benefit to justify designing distributed admission robustness.

It would not prove:

- safety under arbitrary asymmetric rank failure;
- correctness after communicator replacement;
- broad model or accelerator genericity;
- stochastic sampling;
- prefill graph benefit;
- multi-node behavior;
- H100/NVLink behavior;
- production default readiness.

Those claims require separate evidence.

## Prompt-to-Artifact Checklist

| Requirement | Planned evidence |
|---|---|
| Continue with a real optimization, not another benchmark-only idea | Candidate arm exercises existing full-step graph replay in canonical `LLMEngine` |
| Do not repeat wavefront/peer/synchronous-reduction routes | Design decision and non-goals |
| Preserve full batch and collective order | Per-rank dispatch and collective-order digests |
| Benefit plus cost | Performance, TTFT, capture-cost, memory, initialization, and amortization rows |
| Qwen3.8-27B TP4 first adopter | Frozen model revision, TP4 environment, Q0/Q1/Q2 profile |
| Model-neutral core | Two-axis verdict, layer map, leakage evidence, contribution split |
| Exact correctness | Token/length/stop rows plus all-rank lifecycle evidence |
| No silent replay retry | Runtime source assertion and replay-failure test |
| Real hardware gate | Four-rank A100 process receipts and GPU inventory |
| Dual verifier | Remote independent and local frozen-source verification |
| Immutable evidence | Post-verification manifest with hashes |
| Remote storage boundary | Manifest paths rooted below the approved `/data00/home/sitian/` workspace |
| No external process cleanup | Admission inventory and controller action receipt |
| Commit and push discipline | Exact-path staging, one co-author trailer, remote SHA confirmation |

## Completion Criteria

This design stage is complete when:

1. this spec is self-reviewed with no placeholders or contradictory gates;
2. it is committed and pushed on `origin/feat/kv-sparse-attention`;
3. an implementation plan maps each artifact and gate to strict RED, minimal
   implementation, GREEN, remote execution, verification, audit, commit, and
   push steps.

The optimization itself remains unclassified until the real TP4 terminal
bundle passes both independent verifiers.
