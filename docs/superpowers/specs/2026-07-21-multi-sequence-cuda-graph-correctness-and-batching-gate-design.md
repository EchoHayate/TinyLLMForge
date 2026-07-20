# Multi-Sequence CUDA Graph Correctness and Production Batching Gate Design

Date: 2026-07-21

## Terminology

This document defines a diagnostic-first CUDA Graph experiment for baseline
multi-sequence decode.

In this document:

- **active batch size** is the number of live decode sequences in one model
  step;
- **graph batch size** is the static batch dimension captured by a CUDA Graph;
- **exact replay** means `active_batch_size == graph_batch_size`;
- **rounded replay** means the active rows are copied into the smallest larger
  captured bucket and the remaining rows are inactive padding;
- **live KV slots** are slots owned by real sequences;
- **scratch KV slots** are isolated slots reserved only for inactive graph
  rows;
- **diagnostic phase** is a tool-only experiment that does not alter production
  dispatch;
- **production candidate** is a default-off dispatch policy admitted only after
  the diagnostic result satisfies this document's correctness contract.

The first production candidate, if admitted, is named:

```text
exact_batch_multi_sequence_cuda_graph
```

It is not a general claim that padded CUDA Graph replay is safe.

## Objective

Determine why multi-sequence decode CUDA Graph replay historically corrupted
rows after the first row, then enable only the smallest correctness-proven
production surface.

The work must:

1. Preserve the existing fail-closed eager guard during diagnosis.
2. Reproduce eager, exact-size graph, and rounded-up graph behavior from the
   same model, prompts, initial KV state, and decode inputs.
3. Compare per-step final logits, greedy token IDs, layer-boundary outputs, and
   touched KV slots.
4. Distinguish exact-batch correctness from padded-bucket correctness.
5. Identify the first layer and first decode step where any divergence occurs.
6. Record inactive-row metadata and KV writes so a padded-row failure cannot be
   mistaken for a generic FlashAttention failure.
7. Permit a production code change only when exact replay is independently
   verified correct.
8. Keep non-exact multi-sequence batches eager in the first production phase.
9. Run a separate source-bound production batching gate before making any
   performance claim.
10. Preserve repository-default behavior unless the candidate passes every
    correctness and performance gate in this document.

This work is a structural successor to the stopped P5 scheduler experiment. It
must not resume mixed-prefill workload or SLO-envelope tuning.

## Current Evidence

Current `ModelRunner.run_model()` forces every decode batch larger than one to
eager execution:

```python
multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
```

The guard was added by commit:

```text
f4ab1c7add0c887605257e261e0622271a75d1da
```

The surviving historical statement is:

```text
Multi-sequence captured graphs can corrupt rows after the first one.
```

No independently recoverable artifact records the affected batch shape,
prompt, context lengths, logits, token IDs, KV mutation, or first divergent
layer.

CUDA Graph capture currently creates these graph sizes:

```python
[1, 2, 4, 8] + list(range(16, max_batch_size + 1, 16))
```

The old replay path selected the smallest graph size greater than or equal to
the active batch size, zeroed every static input buffer, copied only active
rows, replayed the graph, and returned only active outputs.

For a rounded batch, inactive rows therefore had zero-valued:

```text
input_ids
positions
slot_mapping
context_lens
block_tables
```

This creates two concrete risks:

1. inactive rows can write K/V to slot zero, which is not an isolated scratch
   slot and may be live;
2. FlashAttention receives rows with zero context length and zero block-table
   metadata that were never defined as a supported inactive-row contract.

Exact replay has neither padding condition. It must be tested separately before
attributing the historical failure to CUDA Graph capture or FlashAttention in
general.

Batch-one CUDA Graph decode is already enabled and documented at approximately
`8-9x` throughput improvement for the validated Qwen3-0.6B environment. This
design does not repeat or re-claim that result.

## Fixed Scope

The first phase covers only:

```text
model                 Qwen3-0.6B
attention             baseline full attention
weight dtype          BF16
tensor parallel       1
sampling              greedy, temperature 0
decode mode           one token per live sequence per step
max_num_seqs          at least 32 for diagnostic capture
remote host           sitian@10.232.195.203
remote Python         /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model path            /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
```

The first phase excludes:

- prefill graph replay;
- mixed prefill-plus-decode;
- speculative verification or drafting;
- Quest;
- Attention Matching;
- KV quantization;
- weight or activation quantization;
- CPU offload or KV offload;
- KV cartridge;
- tensor parallelism;
- non-greedy sampling;
- input embeddings;
- hidden-state return in production;
- other models, GPUs, drivers, or FlashAttention builds.

Every remote model process must use distinct dynamic values for both:

```text
TINYVLLM_DIST_PORT
MASTER_PORT
```

The remote checkout must not be modified or synchronized. The runner must send
an immutable source bundle or individual tool payload using the repository's
existing remote-run pattern, execute in an isolated remote temporary directory,
and download artifacts without deleting shared remote paths.

## Alternatives Considered

### 1. Recommended: Diagnostic First, Then Exact-Bucket-Only Production Replay

Run the full eager/exact/rounded diagnostic. If exact replay is correct for all
required shapes and trajectories, enable graph replay only when an exact
captured graph already exists. Keep all non-exact batches eager.

For the current capture set, the first admitted multi-sequence shapes are:

```text
2, 4, 8, 16
```

Advantages:

- removes inactive padded rows from the production correctness contract;
- reuses already captured graphs and therefore adds no new graph sizes;
- preserves fail-closed eager behavior for ambiguous shapes;
- creates a small change that can be covered by dependency-light dispatch tests
  and real GPU evidence;
- can deliver value for common power-of-two batching without waiting for a
  general padding protocol.

Costs:

- batches such as `3`, `5`, and `9` remain eager;
- production benefit depends on exact-bucket hit rate;
- a later design is required if arbitrary active batch sizes need graph replay.

### 2. Padded Buckets with Explicit Scratch-Row Metadata

Reserve isolated scratch KV blocks and populate every inactive graph row with a
valid token, position, positive context length, block table, and unique scratch
slot. Ignore inactive outputs after replay.

Advantages:

- preserves the compact graph bucket set;
- can accelerate arbitrary active batch sizes;
- directly addresses zero metadata and slot-zero aliasing.

Costs:

- requires lifetime and capacity rules for scratch blocks;
- inactive rows still execute model work;
- scratch rows can silently corrupt live state if isolation is wrong;
- correctness depends on stronger allocator, metadata, and cleanup contracts;
- it is too large for the first production change without diagnostic proof
  that padding, rather than exact replay, is the failure source.

This is a possible follow-up design, not part of first-phase production
enablement.

### 3. Capture Every Active Batch Size

Capture an exact graph for every batch size from `1` through a configured
maximum.

Advantages:

- avoids padded-row semantics;
- supports every batch size in the configured range.

Costs:

- increases initialization time and graph memory;
- scales poorly toward the existing maximum of 512 sequences;
- duplicates graphs for shapes not observed in production;
- changes capacity and startup behavior before exact-bucket hit rate is known.

This is rejected for the first phase. It may be reconsidered only if the
exact-bucket production gate proves material benefit and records a high
non-exact fallback rate.

## Decision

Use Alternative 1.

The first implementation must split into two independently reviewable stages:

1. a diagnostic-only stage that preserves the current production guard;
2. a conditional exact-bucket production stage that is allowed only if the
   diagnostic independent verifier returns `EXACT_REPLAY_CORRECT`.

Rounded replay must remain disabled in production even if one diagnostic run
appears correct. Promoting padded replay requires a separate written design.

## Diagnostic Architecture

### Tool Boundaries

The implementation plan should create focused tools with these
responsibilities:

```text
tools/diagnose_multi_sequence_cuda_graph.py
    Run one local-GPU diagnostic case in an isolated process.

tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
    Validate source identity, allocate dynamic ports, launch the fixed matrix
    remotely, resume completed rows, and download immutable artifacts.

tools/verify_multi_sequence_cuda_graph_diagnostic.py
    Recompute row completeness, comparisons, hashes, and classification from
    raw artifacts without trusting producer labels.

tools/test_multi_sequence_cuda_graph_gate.py
    Dependency-light tests for matrix construction, comparison rules,
    classification, resume behavior, and fail-closed dispatch policy.
```

The diagnostic tool may construct private graph objects and static buffers, or
temporarily invoke private `ModelRunner` methods from the tool. It must not edit
or monkeypatch repository source files on the remote host. It must not remove
the production multi-sequence eager guard.

### Isolation Contract

Each tuple:

```text
(batch_shape, trajectory, execution_mode, repetition)
```

must run in a fresh subprocess with a fresh `LLM`/`ModelRunner`, KV cache, CUDA
context, and dynamic port pair.

Execution modes are exactly:

```text
eager
exact_graph
rounded_graph
```

For every batch shape, `exact_graph` means a tool-owned graph captured at that
exact active size. Shapes `3`, `5`, and `9` therefore use diagnostic-only exact
graphs that are not part of the current production capture set.

For every batch shape, `rounded_graph` means the next strictly larger existing
production graph bucket:

```text
2 -> 4
3 -> 4
4 -> 8
5 -> 8
8 -> 16
9 -> 16
16 -> 32
```

Using a strictly larger bucket for exact production shapes is intentional: it
isolates inactive-row behavior from active-size behavior.

This process isolation is mandatory. Reusing one mutable KV cache for eager and
graph comparisons would allow the first execution to affect the second.

The producer must record:

- source commit and dirty state;
- source-tree SHA256;
- Python, PyTorch, CUDA, driver, GPU, FlashAttention, and Transformers
  identities;
- model path and selected fields from `config.json`;
- process ID and dynamic ports;
- active batch size and graph batch size;
- all active and inactive static-buffer rows before replay;
- prompt IDs, input IDs, positions, context lengths, slot mappings, and block
  tables;
- failure phase, exception type, message, and traceback tail.

## Fixed Diagnostic Matrix

### Batch Shapes

The batch shape set is exactly:

```text
2, 3, 4, 5, 8, 9, 16
```

It covers:

- exact existing production buckets: `2`, `4`, `8`, `16`;
- diagnostic-only exact sizes: `3`, `5`, `9`;
- rounded buckets with one inactive row: `3 -> 4`;
- rounded buckets with two or more inactive rows:
  `2 -> 4`, `4 -> 8`, `5 -> 8`, `8 -> 16`, `9 -> 16`, and `16 -> 32`.

### Trajectories

Each batch shape must run these three deterministic trajectories:

1. **uniform-short**
   - all rows use the same short prompt;
   - verifies row independence when context lengths are equal;
2. **ragged-context**
   - every row uses a different prompt length;
   - actual context lengths must span at least two KV blocks for the largest
     row and include at least one row below one block;
3. **duplicate-and-distinct**
   - at least two rows share identical prompt tokens while another row differs;
   - detects accidental cross-row coupling and slot aliasing.

Prompt text and token IDs must be versioned in the diagnostic manifest before
the first GPU run. Canonical results must not change prompts, decode length, or
thresholds.

For each `(batch_shape, trajectory, repetition)` tuple, the eager process
produces the ordered reference token array first. The exact and rounded graph
processes record their own greedy argmax at every step but feed the eager
reference token into the next step. This teacher-forced continuation keeps
input IDs, positions, and logical KV growth aligned after a mismatch, allowing
the verifier to identify the first divergent step rather than compare two
already different trajectories.

The manifest and raw rows must distinguish:

```text
observed_argmax_token_id
reference_next_input_token_id
```

Missing or mismatched eager reference arrays make the tuple `INCOMPLETE`.

### Decode Length and Repetitions

Each case runs:

```text
warmup decode steps       2
measured decode steps     16
repetitions               3
```

Warmup steps are excluded from timing but included in correctness state
construction. Every measured step must produce a raw comparison row.

The complete fixed matrix is:

```text
7 batch shapes
× 3 trajectories
× 3 execution modes
× 3 repetitions
= 189 isolated processes
```

The remote runner may provide a non-authoritative smoke mode, but smoke results
cannot classify the production candidate.

## Comparison Contract

### Final Logits

For every active row and measured decode step, the producer records:

- shape and dtype;
- finite-value status;
- argmax token ID;
- full-tensor SHA256 over contiguous CPU bytes;
- maximum absolute difference from the paired eager row;
- maximum relative difference from the paired eager row;
- the top 16 `(token_id, logit)` pairs.

The producer must also write complete eager and graph logits as immutable
tensor shards under:

```text
tensors/logits/<matrix-key>.pt
```

Each shard records dtype, shape, ordered step IDs, ordered active row IDs, and
the full logits. `sha256sums.txt` covers every shard. The independent verifier
loads the shards and performs the comparison itself; producer-computed
differences are diagnostic convenience fields only.

The independent verifier requires:

```text
all logits finite
argmax token IDs exactly equal
torch.testing.assert_close(rtol=1e-3, atol=1e-2)
```

The threshold is frozen before execution and must not be widened after reading
results.

### Greedy Token Trajectory

All 16 measured greedy token IDs for every active row must exactly match eager.
The verifier must compare the ordered observed argmax arrays, not generated
text. Teacher forcing is a diagnostic control and does not relax this exact
token requirement.

### Layer Localization

The diagnostic may install tool-owned observation hooks at decoder layer
boundaries. Observation storage must be preallocated before graph capture and
must not allocate tensors inside replay.

For each layer boundary, active row, and measured step, record:

- finite-value status;
- full-tensor SHA256;
- maximum absolute and relative difference from eager.

The complete observed layer-boundary tensors must be written as immutable
shards under:

```text
tensors/layers/<matrix-key>.pt
```

The independent verifier loads paired eager and graph shards and recomputes
finite checks, hashes, and difference metrics.

Layer observation is a localization signal, not the sole correctness gate. A
tool must report `observation_unavailable` rather than silently omit a layer if
the installed model structure cannot expose the required boundary.

All required Qwen3 decoder layers must be observed for the canonical matrix.
Any missing layer makes the canonical diagnostic `INCOMPLETE`.

### KV Slot Integrity

Before and after every measured step, record byte hashes for:

1. every physical KV slot written by an active row, across all layers;
2. slot zero;
3. every inactive row's declared slot;
4. a fixed sentinel set of untouched live slots.

For exact replay:

- active written-slot post-step hashes must equal eager;
- slot zero and sentinel slots must have the same before/after mutation status
  as eager;
- no slot outside the eager touched-slot set may change.

For rounded replay:

- the tool must record whether inactive rows target slot zero, a live slot, a
  duplicate slot, or an isolated scratch slot;
- any mutation outside the eager touched-slot set is a rounded-replay
  correctness failure;
- rounded failure does not invalidate exact replay classification when exact
  replay passes independently.

### First Divergence

For every failed comparison, the verifier must derive:

```text
first divergent decode step
first divergent active row
first divergent layer boundary, if observable
first unexpected KV slot mutation, if any
```

The report must not reduce a failure to a single final generated string.

## Diagnostic Classifications

The independent verifier returns exactly one top-level classification:

### `EXACT_REPLAY_CORRECT`

Required:

- all 189 processes are present or validly resumed;
- all source and environment identities match;
- eager rows are complete and finite;
- exact graph rows for all seven batch shapes satisfy every logit, token,
  layer-observation, and KV integrity rule;
- producer and verifier summaries agree.

Rounded rows may pass or fail. Their result must be reported separately as:

```text
ROUNDED_REPLAY_CORRECT
ROUNDED_REPLAY_CORRUPT
```

Only `EXACT_REPLAY_CORRECT` admits the first production candidate.

### `EXACT_REPLAY_CORRUPT`

Any exact graph case diverges from eager, mutates an unexpected KV slot,
contains non-finite values, crashes, or omits a required observation.

Required action:

- retain the current eager guard;
- do not implement exact-bucket production replay;
- use the first-divergence evidence for a new FlashAttention/capture design;
- do not reinterpret rounded-row failures as the root cause.

### `INCOMPLETE`

The matrix, identities, raw observations, source evidence, or independent
reconstruction is incomplete.

`INCOMPLETE` cannot admit a production code change.

## Conditional Production Candidate

Only after `EXACT_REPLAY_CORRECT`, change dispatch so baseline decode uses a
CUDA Graph when and only when:

```text
multi_sequence_cuda_graph_exact is true
active batch size > 1
active batch size is an exact key in self.graphs
all existing CUDA Graph incompatibility guards are false
```

The first candidate must not choose a larger graph bucket.

Add one default-off configuration field:

```python
multi_sequence_cuda_graph_exact: bool = False
```

The production batching gate enables this field only for
`EXACT_GRAPH_CANDIDATE`. `EAGER_BASELINE` leaves it false. An independently
verified `GO` may justify a later, separately reviewed default-promotion
change; this design does not change the repository default.

Pseudocode:

```python
exact_decode_graph_available = (
    self.config.multi_sequence_cuda_graph_exact
    and
    mode == "decode"
    and input_ids.size(0) > 1
    and input_ids.size(0) in self.graphs
)

unsupported_multi_sequence_decode = (
    mode == "decode"
    and input_ids.size(0) > 1
    and not exact_decode_graph_available
)
```

All existing fail-closed conditions remain authoritative:

- prefill;
- spec verify;
- `enforce_eager`;
- Quest;
- Attention Matching;
- KV quantization;
- CPU offload;
- KV offload;
- input embeddings;
- hidden-state return.

Dependency-light tests must prove:

1. batch `1` still replays graph `1`;
2. exact batches `2`, `4`, `8`, and `16` replay their same-size graph;
3. batches `3`, `5`, and `9` use eager and never replay a larger graph;
4. every unsupported feature still uses eager;
5. missing graph keys fail closed to eager;
6. graph buffers copy exactly the active rows and do not depend on inactive
   padding.

## Production Batching Gate

### Purpose

The diagnostic proves correctness under controlled trajectories. It does not
prove that exact-bucket replay improves a production batching workload.

The production gate compares:

```text
EAGER_BASELINE
EXACT_GRAPH_CANDIDATE
```

from the same clean source and remote environment.

### Workloads

Use a versioned arrival manifest with three workload classes:

1. **stable-exact**
   - sustained decode batches concentrated at `2`, `4`, `8`, and `16`;
2. **ragged-natural**
   - staggered arrivals and prompt lengths that naturally produce exact and
     non-exact active batch sizes;
3. **churn**
   - frequent sequence completion and admission transitions across
     `2/3/4/5/8/9/16`.

Each policy/workload pair runs:

```text
1 warmup repetition
5 measured repetitions
```

Policy order must alternate by repetition. Each repetition uses a fresh model
process and unique dynamic ports.

### Required Evidence

Per decode step, record:

- active batch size;
- selected execution path: `eager`, `graph_exact`, or an existing unrelated
  path;
- exact graph key, if any;
- model-step duration;
- completed token count;
- queue depths and sequence lifecycle transitions;
- output token IDs;
- peak allocated and reserved CUDA memory.

Per run, report:

- request throughput;
- decode token throughput;
- median, p95, and p99 inter-token latency;
- median, p95, and maximum model-step duration;
- exact-bucket graph hit rate;
- non-exact eager fallback rate;
- peak allocated and reserved CUDA memory;
- initialization duration;
- exact-output hash.

### Correctness Gates

Every measured candidate run must satisfy:

- exact generated token arrays equal its paired eager run;
- request counts and lifecycle terminal states equal eager;
- no non-finite logits or runtime errors;
- graph replay occurs only for exact graph keys;
- batches `3`, `5`, and `9` are observed using eager in the churn workload;
- no rounded graph replay event exists;
- no KV allocator or block-accounting invariant fails.

Any correctness failure classifies the production gate `NO_GO`.

### Frozen Performance Gates

The candidate is `GO` only if all conditions hold:

1. median aggregate decode token throughput across all workloads is at least
   `1.15x` eager;
2. `stable-exact` median decode token throughput is at least `1.25x` eager;
3. no workload has median request throughput below `0.95x` eager;
4. no workload has p95 inter-token latency above `1.05x` eager;
5. no workload has p99 inter-token latency above `1.10x` eager;
6. candidate peak reserved CUDA memory is no more than `1.02x` eager;
7. candidate initialization duration is no more than `1.05x` eager;
8. exact-bucket graph hit rate is at least `0.60` in `stable-exact`;
9. all five measured repetitions are complete for every policy/workload pair;
10. an independent verifier recomputes the same result from raw rows.

These thresholds are fixed before canonical execution. They must not be
changed to include observed results.

### Production Gate Classifications

The independent verifier returns:

```text
GO
NO_GO
INCOMPLETE
```

`GO` requires every correctness and performance gate.

`NO_GO` means the candidate is correctly exercised but at least one fixed gate
fails. The production guard remains eager for multi-sequence decode.

`INCOMPLETE` means evidence is missing or cannot be independently reconstructed.

## Artifact Contract

Diagnostic artifacts live under:

```text
experiments/cuda_graph/<run-id>/
```

The canonical diagnostic directory contains:

```text
manifest.json
environment.json
raw_rows.jsonl
layer_observations.jsonl
kv_observations.jsonl
tensors/logits/*.pt
tensors/layers/*.pt
summary.json
report.md
sha256sums.txt
```

Production gate artifacts add:

```text
arrival_manifest.json
step_rows.jsonl
request_rows.jsonl
```

The producer writes raw artifacts first. The independent verifier reads raw
artifacts and writes a separate:

```text
independent_verification.json
```

The verifier must reject:

- missing or duplicate matrix keys;
- source or environment identity drift;
- producer-only aggregate fields without raw evidence;
- changed prompts, thresholds, repetitions, or decode lengths;
- rounded graph replay in a production candidate;
- non-distinct port pairs;
- resumed rows whose source or environment identity differs;
- hash mismatches.

Canonical artifact promotion must use selective staging. Existing unrelated
untracked `experiments/` directories must never be staged with `git add -A`.

## Source and Remote Integrity

Before remote execution, record:

```text
git rev-parse HEAD
git status --porcelain
source-tree SHA256
```

The source-tree hash must cover every file copied into the remote execution
bundle. The remote runner must verify the hash before launching a model
process.

Use:

```text
SSH target        sitian@10.232.195.203
ControlMaster     /tmp/ssh-sitian-10.232.195.203
```

The runner must not:

- use local user `bytedance` for the target or jump proxy;
- modify the remote checkout;
- run `rsync` over the remote repository;
- kill unrelated processes;
- clear shared `/tmp`;
- reuse fixed distributed ports.

Only `EADDRINUSE` permits a controlled retry with a new port pair. Other
failures must be preserved as evidence.

## Claim Boundaries

`EXACT_REPLAY_CORRECT` proves only that exact graph replay matched eager for the
fixed diagnostic matrix and environment.

It does not prove:

- padded replay correctness;
- arbitrary batch-size graph safety;
- performance improvement;
- scheduler or queueing benefit;
- correctness for excluded features;
- correctness on other models, GPUs, drivers, or library versions.

Production `GO` permits this narrow claim:

```text
In the source-bound Qwen3-0.6B BF16 TP=1 greedy baseline gate,
exact-bucket multi-sequence CUDA Graph replay preserved exact generated tokens
and improved aggregate decode throughput while satisfying the frozen latency,
memory, and initialization thresholds.
```

No README performance statement may be added before an independently verified
production `GO`.

## Stop Conditions

Stop and retain eager multi-sequence decode when:

1. any exact replay case is corrupt;
2. the canonical diagnostic is incomplete;
3. exact replay requires relaxing logit, token, layer, or KV integrity rules;
4. the implementation would need rounded replay to show benefit;
5. the production candidate fails exact output or lifecycle correctness;
6. the independent production verifier returns `NO_GO` or `INCOMPLETE`;
7. observed benefit requires changing the frozen workload or thresholds.

If exact replay is correct but the production gate is `NO_GO`, preserve the
diagnostic and gate infrastructure but do not continue tuning the same workload
to increase exact-bucket hit rate. The next decision must be a new written
design choosing between:

1. explicit scratch-row padded replay;
2. demand-driven exact graph capture with a bounded graph-memory budget;
3. a different kernel or quantization bottleneck.

## Acceptance Criteria

The design is complete when:

1. the current eager guard remains unchanged through the diagnostic stage;
2. the fixed 189-process matrix is recoverable from the manifest;
3. eager, exact, and rounded modes run in isolated processes;
4. the verifier reconstructs logits, tokens, layer localization, and KV
   integrity results from raw artifacts;
5. exact and rounded classifications are reported separately;
6. only `EXACT_REPLAY_CORRECT` can admit production implementation;
7. first production dispatch uses exact graph keys only;
8. non-exact batches remain eager and are tested;
9. the source-bound production gate records real graph hit/fallback events;
10. only independent production `GO` permits a performance claim or README
    update.
