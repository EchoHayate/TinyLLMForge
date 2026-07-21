# Fixed-Split Multi-Sequence CUDA Graph Recovery Design

Date: 2026-07-21

## Status

This design replaces the production-admission path in:

```text
docs/superpowers/specs/2026-07-21-multi-sequence-cuda-graph-correctness-and-batching-gate-design.md
```

The prior diagnostic remains authoritative evidence. Its canonical result was:

```text
EXACT_REPLAY_CORRUPT
ROUNDED_REPLAY_CORRUPT
```

Therefore the existing production fail-closed guard must remain in force until
this design's fresh, same-policy canonical verifier returns
`EXACT_REPLAY_CORRECT` and the later production workload gate returns `GO`.

This document is a design only. It does not authorize a production dispatch
change, a remote canonical run, or a README performance claim.

## Terminology

- **auto split** means FlashAttention `num_splits=0`, where the implementation
  selects its split strategy from runtime inputs.
- **fixed split 16** means FlashAttention `num_splits=16`.
- **graph policy** is the attention split policy installed during CUDA Graph
  warmup, capture, and replay.
- **candidate eager** is an eager reference execution using the same fixed
  split policy as the graph candidate.
- **legacy eager** is the unchanged production eager execution using auto
  split.
- **same-policy correctness** compares fixed-split-16 graph replay with
  fixed-split-16 eager.
- **legacy compatibility** compares fixed-split-16 eager with legacy
  auto-split eager.
- **exact replay** has `active_batch_size == graph_batch_size`.
- **rounded replay** uses a larger captured graph and inactive padding rows.

## Objective

Recover a correctness-testable multi-sequence CUDA Graph candidate without
changing the established eager baseline or weakening any gate.

The work must:

1. Make the FlashAttention split strategy explicit and stable across graph
   warmup, capture, replay, and the graph candidate's eager comparator.
2. Prove graph replay correctness only through a same-policy comparison.
3. Measure fixed-split numerical compatibility with legacy auto-split eager as
   a separate contract rather than conflating it with graph replay correctness.
4. Preserve legacy auto split for ordinary eager production decode.
5. Preserve the existing batch-greater-than-one eager guard until a fresh
   independent canonical verifier returns `EXACT_REPLAY_CORRECT`.
6. Keep rounded multi-sequence graph replay disabled in production.
7. Require a separate source-bound production performance gate before any
   default or README claim.
8. Fail closed on missing policy metadata, mixed split policies, incomplete
   evidence, source drift, environment drift, or any correctness failure.

## Current Evidence

### Canonical Failure

The canonical matrix used:

```text
7 batch shapes
× 3 trajectories
× 3 execution modes
× 3 repetitions
= 189 isolated processes
```

It found exact replay divergence at batches:

```text
4, 5, 8, 9, 16
```

The result remains an authoritative `NO_GO` for the implementation and
comparison policy that produced it. It must not be relabeled after the fact.

### Input and Metadata Audit

The diagnostic clears the static replay buffers and then copies every active
row of:

```text
input_ids
positions
slot_mapping
context_lens
block_tables
```

No active-row omission was found in this chain. This does not prove graph
correctness, but it makes a missing replay-row copy an unsupported primary
explanation for the observed exact-replay failures.

### FlashAttention Policy Mismatch

The decode attention call uses:

```python
flash_attn_with_kvcache(
    ...,
    num_splits=context.flash_attn_num_splits,
)
```

The context default is:

```python
flash_attn_num_splits = 0
```

In FlashAttention 2.6.3, `0` delegates split selection to an automatic
heuristic. CUDA Graph capture fixes the kernel path selected from the capture
inputs, while a separately executed eager reference can select from its own
runtime shape and ragged lengths. The old canonical comparison therefore did
not establish that graph replay and eager used the same reduction strategy.

This is a comparator-validity defect even when both executions are individually
legal.

### Fixed-Split Root-Cause Probe

A non-canonical probe compared:

```text
fixed-split-16 CUDA Graph replay
vs
fixed-split-16 eager
```

for the five historically corrupt ragged-context batches:

```text
4, 5, 8, 9, 16
```

Every compared tensor was elementwise equal:

```text
final logits
all observed layer outputs
KV keys before and after
KV values before and after
```

The aggregate artifact reports:

```text
all_equal=true
```

This is strong root-cause evidence, but it is not production admission because
it covers only one trajectory, one repetition, five batches, exact replay, no
production workload, and no performance measurement.

### Existing Fixed-Split Precedent

The speculative verifier already uses an explicit fixed split:

```python
SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS = 16
```

This demonstrates that an explicit split policy is compatible with the current
context mechanism. It does not prove that the same value is correct or
profitable for baseline multi-sequence decode.

## Alternatives Considered

### 1. Recommended: Candidate-Scoped Fixed Split 16

Use fixed split 16 only for the multi-sequence CUDA Graph candidate and its
same-policy eager reference. Keep ordinary eager decode on auto split.

Advantages:

- directly addresses the identified comparator/capture mismatch;
- matches the successful same-policy probe;
- confines behavioral change to the default-off candidate;
- preserves the legacy eager baseline and fallback;
- permits correctness, compatibility, and performance to be evaluated
  independently.

Costs:

- fixed split 16 may be slower for some batch/context distributions;
- fixed split 16 may differ numerically from auto split within floating-point
  tolerance;
- the candidate needs explicit policy plumbing and evidence fields.

### 2. Calibrated Split Table by Graph Bucket

Benchmark several fixed split values and assign a value per graph batch size or
context-length range.

Advantages:

- may outperform one global fixed value;
- can adapt to different graph shapes.

Costs:

- multiplies calibration, correctness, configuration, and artifact state;
- risks tuning to the current model and prompt bank;
- creates more capture identities and more ways to mix comparator policies;
- is not justified before a single fixed policy passes the full gate.

This is deferred. It requires a separate design after the fixed-16 candidate is
fully classified.

### 3. Upgrade FlashAttention

Move to a newer FlashAttention build and retry auto split under graphs.

Advantages:

- a newer implementation may offer graph-stable split selection or better
  kernels.

Costs:

- changes kernels, numerical behavior, build compatibility, and performance at
  once;
- prevents attribution of the recovery to one controlled variable;
- expands the regression surface beyond this candidate.

This is rejected for this recovery. A dependency upgrade must be evaluated
separately.

## Decision

Use Alternative 1.

The candidate policy is:

```text
multi-sequence CUDA Graph warmup/capture/replay: fixed split 16
candidate eager correctness comparator:        fixed split 16
legacy eager baseline and fallback:             auto split 0
batch-one production CUDA Graph path:           unchanged
```

No path may silently inherit a process-global fixed split. Policy selection
must be explicit at the execution boundary and restored after candidate-owned
work.

## Architecture

### Policy Representation

Add one named constant in the CUDA Graph diagnostic/contract layer:

```python
MULTI_SEQUENCE_CUDA_GRAPH_FLASH_ATTN_NUM_SPLITS = 16
```

The production configuration remains default-off and should describe candidate
enablement, not expose arbitrary user tuning:

```python
multi_sequence_cuda_graph_exact: bool = False
```

Do not add a general public `flash_attn_num_splits` tuning option as part of
this work. The value `16` is a source-bound candidate contract, not a broadly
validated user setting.

Every process artifact must record:

```text
flash_attn_version
flash_attn_num_splits
split_policy_name
comparison_policy_name
```

The independent verifier rejects missing, mixed, or unexpected values.

### Execution-Scoped Context

The attention split value must flow through the existing execution context.
Candidate-owned code installs fixed split 16 only around:

1. the graph warmup for a multi-sequence candidate graph;
2. capture of that graph;
3. replay of that graph;
4. candidate eager comparator execution in the diagnostic.

Legacy eager decode continues to install or inherit auto split 0.

Restoration must be exception-safe. A failed capture, replay, or comparator
must not leave fixed split 16 active for later eager work in the same process.

The implementation plan must prefer a small context manager or equivalent
single-purpose boundary over scattered set/reset calls.

### Graph Identity

A graph is reusable only when its recorded identity includes:

```text
graph batch size
execution mode
FlashAttention version
split policy name
split count
model/source identity
```

The runtime must never replay a graph captured under auto split as though it
were a fixed-16 graph, or vice versa.

In-memory graphs do not require a serialized cache in this phase. The identity
contract is still mandatory in diagnostic artifacts and dispatch tests so a
future cache cannot omit it.

### Dispatch Boundary

The current production guard remains:

```python
multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
```

It may be refined only after fresh canonical admission. The eventual
default-off exact-key candidate is allowed when all of these are true:

```text
multi_sequence_cuda_graph_exact is enabled
mode is baseline decode
active batch size is greater than one
active batch size is an exact key in self.graphs
the selected graph has fixed-split-16 identity
all existing graph incompatibility guards are false
```

Non-exact batch sizes remain eager. A larger graph bucket must not be selected.
Rounded replay remains diagnostic-only.

All current fail-closed conditions remain authoritative, including:

```text
prefill
speculative verification
enforce_eager
Quest
Attention Matching
KV quantization
CPU offload
KV offload
input embeddings
hidden-state return
missing graph key
missing or mismatched split identity
```

## Three Independent Gates

The recovery is intentionally split into three gates. Passing one gate does not
waive another.

### Gate A: Same-Policy Replay Correctness

Purpose:

```text
Does fixed-split-16 CUDA Graph replay reproduce fixed-split-16 eager?
```

Use the original frozen canonical matrix without reducing coverage:

```text
batch shapes       2, 3, 4, 5, 8, 9, 16
trajectories       uniform-short, ragged-context, duplicate-and-distinct
modes              candidate_eager, exact_graph_fixed16, rounded_graph_fixed16
repetitions        3
isolated processes 189
```

`candidate_eager` replaces the old auto-split eager comparator for this gate.
Prompts, warmup steps, measured steps, teacher-forced continuation, tensor
shards, layer hooks, KV observations, tolerances, and source/environment rules
remain unchanged from the prior design.

Exact replay must satisfy:

```text
all logits finite
argmax token IDs exactly equal
torch.testing.assert_close(rtol=1e-3, atol=1e-2)
all measured greedy token arrays exactly equal
all required layer observations present and within the frozen tolerance
active KV post-state equals candidate eager
no unexpected KV slot mutates
```

The independent verifier returns one of:

```text
EXACT_REPLAY_CORRECT
EXACT_REPLAY_CORRUPT
INCOMPLETE
```

Rounded replay is classified independently:

```text
ROUNDED_REPLAY_CORRECT
ROUNDED_REPLAY_CORRUPT
ROUNDED_REPLAY_INCOMPLETE
```

Only `EXACT_REPLAY_CORRECT` can unlock implementation of the default-off
production dispatch candidate. Rounded status cannot unlock rounded production
replay.

### Gate B: Legacy Eager Compatibility

Purpose:

```text
Does fixed-split-16 eager preserve the externally relevant behavior of legacy
auto-split eager on the frozen matrix?
```

Run a separate matrix:

```text
7 batch shapes
× 3 trajectories
× 3 repetitions
= 63 logical pairs
= 126 isolated model processes
```

Each pair uses fresh isolated processes:

```text
legacy eager:    auto split 0
candidate eager: fixed split 16
```

Both processes start from the same prompts, initial KV state, and eager
reference token inputs. The gate requires:

```text
all logits finite
greedy argmax token IDs exactly equal at every measured step
ordered generated token arrays exactly equal
torch.testing.assert_close(rtol=1e-3, atol=1e-2)
KV touched-slot ownership sets exactly equal
no unexpected slot mutation in either policy
```

Bitwise equality is recorded but is not required across different reduction
strategies. The frozen tolerance must not be widened after observing results.

The independent verifier returns:

```text
LEGACY_COMPATIBLE
LEGACY_INCOMPATIBLE
INCOMPLETE
```

An incompatibility keeps legacy eager as the fallback and blocks the graph
candidate. It must not be reported as CUDA Graph corruption.

### Gate C: Source-Bound Production Performance

Purpose:

```text
Does the admitted exact-key fixed-16 graph candidate improve a realistic
batched workload without correctness, latency, memory, or initialization
regressions?
```

This gate runs only after:

```text
Gate A == EXACT_REPLAY_CORRECT
Gate B == LEGACY_COMPATIBLE
```

Compare:

```text
EAGER_BASELINE
    ordinary production eager decode with auto split 0

EXACT_GRAPH_FIXED16_CANDIDATE
    fixed-split-16 graph for exact multi-sequence graph keys
    auto-split eager fallback for every non-exact or unsupported step
```

Use the existing frozen workload classes:

```text
stable-exact
ragged-natural
churn
```

Each policy/workload pair runs one warmup repetition and five measured
repetitions, alternating policy order and using fresh processes with unique
dynamic ports.

The correctness requirements and frozen performance thresholds remain:

1. candidate output token arrays exactly equal paired eager;
2. request and lifecycle terminal states exactly equal eager;
3. graph replay occurs only for exact graph keys;
4. no rounded graph replay event exists;
5. batches `3`, `5`, and `9` use auto-split eager in churn;
6. no KV/block invariant fails;
7. aggregate median decode token throughput is at least `1.15x` eager;
8. stable-exact median decode token throughput is at least `1.25x` eager;
9. no workload median request throughput is below `0.95x` eager;
10. no workload p95 inter-token latency exceeds `1.05x` eager;
11. no workload p99 inter-token latency exceeds `1.10x` eager;
12. peak reserved CUDA memory is no more than `1.02x` eager;
13. initialization duration is no more than `1.05x` eager;
14. stable-exact exact-graph hit rate is at least `0.60`;
15. all measured repetitions and independent reconstruction are complete.

The independent verifier returns:

```text
GO
NO_GO
INCOMPLETE
```

Only `GO` permits a later, separately reviewed default-promotion decision.

## Diagnostic and Artifact Changes

The existing tools remain the foundation:

```text
tools/diagnose_multi_sequence_cuda_graph.py
tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
tools/verify_multi_sequence_cuda_graph_diagnostic.py
tools/test_multi_sequence_cuda_graph_gate.py
```

The implementation plan must extend them rather than create a parallel,
incompatible evidence format.

Required additions:

1. explicit split-policy fields in case identity, manifest, environment, raw
   rows, tensor metadata, summary, and report;
2. a candidate-eager mode that executes fixed split 16;
3. an independent 63-pair/126-process legacy-compatibility manifest and
   verifier section;
4. verifier rejection of policy mixing or policy omission;
5. separate classifications for replay correctness and legacy compatibility;
6. preservation of the prior canonical artifacts as immutable negative
   evidence;
7. a fresh run directory for every smoke and canonical execution.

Canonical artifacts remain untracked unless a later explicit documentation
decision says otherwise. They must never be included through `git add -A`.

## Remote Execution Contract

All GPU/model work runs only on:

```text
host          sitian@10.232.195.203
ControlPath   /tmp/ssh-sitian-10.232.195.203
Python        /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model         /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
```

Every model process uses unique dynamic values for:

```text
TINYVLLM_DIST_PORT
MASTER_PORT
```

The runner must:

- upload an immutable source snapshot to an isolated temporary directory;
- never modify or synchronize the remote checkout;
- validate source SHA256 before execution;
- preserve process stdout, stderr, exit code, environment, and port identity;
- retry only explicitly recognized port-collision failures;
- download artifacts without deleting shared remote paths;
- run the independent verifier after download.

## Testing Strategy

Implementation is test-driven.

Dependency-light tests must first fail, then prove:

1. fixed split 16 is the only accepted candidate graph policy;
2. the 189-case same-policy matrix is complete and uniquely keyed;
3. the 63-pair/126-process legacy-compatibility matrix is complete and
   uniquely paired;
4. graph and comparator case IDs include split policy;
5. mixed or missing policy evidence is `INCOMPLETE`;
6. auto-split graph evidence cannot be classified as fixed-16 correctness;
7. exact replay classification ignores rounded status but never missing exact
   rows;
8. legacy compatibility requires exact token equality and frozen logit
   tolerance;
9. graph policy context is restored after success and exceptions;
10. legacy eager fallback observes auto split 0;
11. candidate capture and replay observe fixed split 16;
12. batch one behavior remains unchanged;
13. exact multi-sequence dispatch is default-off;
14. non-exact and unsupported cases fail closed to eager;
15. no larger graph bucket is selected;
16. all production thresholds retain their frozen boundary behavior;
17. artifact finalization is atomic and source-bound;
18. resume accepts only one complete, identity-matching case.

After focused tests, the required repository unit-test workflow, `utree flush`,
and telemetry steps must run as specified by the implementation plan.

## Failure Handling

Any of the following is fail-closed:

- fixed-split graph differs from fixed-split eager;
- fixed-split eager violates legacy compatibility;
- graph/case split metadata is absent or inconsistent;
- a graph captured with one policy is replayed under another identity;
- a required tensor shard, layer observation, KV observation, process row, or
  SHA256 entry is missing;
- source, model, FlashAttention, GPU, driver, prompt, matrix, or threshold
  identity drifts;
- a process exits nonzero outside the allowed port-collision retry;
- a non-exact production batch replays a rounded graph;
- any correctness invariant fails in the performance gate;
- any frozen performance threshold fails.

The response is to retain the batch-greater-than-one eager guard and report the
specific failed gate. Do not weaken coverage, prompts, repetitions, tolerances,
or thresholds.

## Scope Exclusions

This design does not include:

- rounded/padded production graph replay;
- per-bucket split calibration;
- FlashAttention upgrade;
- prefill CUDA Graphs;
- mixed prefill/decode graphs;
- speculative verification or drafting;
- Quest or Attention Matching graph support;
- KV quantization, weight quantization, CPU offload, or KV offload;
- tensor parallelism;
- non-greedy sampling;
- models other than Qwen3-0.6B;
- changing the repository default;
- a README performance claim before an independent production `GO`.

## Admission Sequence

The only valid progression is:

```text
1. design approval
2. TDD implementation plan approval
3. dependency-light RED/GREEN implementation
4. fresh remote smoke
5. fresh 189-process same-policy canonical
6. fresh 63-pair/126-process legacy-compatibility canonical
7. independent Gate A and Gate B verification
8. default-off exact-key production candidate implementation
9. source-bound production workload smoke
10. source-bound production canonical
11. independent performance classification
12. documentation/default-promotion review only after GO
```

If Gate A or Gate B fails, steps 8 through 12 are not authorized.

## Success Criteria

The recovery is technically admitted only when all are true:

```text
Gate A: EXACT_REPLAY_CORRECT
Gate B: LEGACY_COMPATIBLE
Gate C: GO
```

Until then:

- production batch-greater-than-one decode remains eager;
- fixed split 16 remains a candidate-scoped mechanism;
- the historical canonical `EXACT_REPLAY_CORRUPT` result remains valid for its
  auto-policy comparison;
- no multi-sequence CUDA Graph performance improvement may be claimed.

## Claim Boundaries

Even a full `GO` proves only:

```text
Qwen3-0.6B
BF16
TP=1
greedy baseline decode
the recorded FlashAttention 2.6.3 environment
the frozen correctness and production workload matrices
exact-key multi-sequence CUDA Graph replay
```

It does not prove rounded replay, other models, other FlashAttention versions,
sampling, tensor parallelism, quantization, offload, speculative decode, or
general production traffic benefit.
