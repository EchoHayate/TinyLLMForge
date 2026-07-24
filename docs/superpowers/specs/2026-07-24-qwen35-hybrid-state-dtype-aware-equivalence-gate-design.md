# Qwen3.5 Hybrid-State Dtype-Aware Equivalence Gate Design

Date: 2026-07-24

## Status

This document replaces only the numerical-equivalence portion of the
Qwen3.5 hybrid-state compatibility gate. The provenance, architecture,
state-lifecycle, request-isolation, storage-ledger, remote-execution, and
claim-boundary requirements in
`2026-07-23-qwen35-hybrid-state-compatibility-gate-design.md` remain in force.

The approved approach is output-behavior-first and dtype-aware:

1. same-path repeatability remains bitwise exact;
2. cached and one-shot execution must make the same greedy decision at every
   step;
3. FP32 remains subject to strict elementwise comparison;
4. BF16 is judged by decision-preserving evidence rather than a full-vocabulary
   maximum absolute-difference cap;
5. full-vocabulary BF16 drift remains mandatory evidence, but is not by itself
   a compatibility rejection.

This gate establishes reference compatibility only. It does not establish
native TinyLLMForge Qwen3.5 support, model quality, compression, latency,
throughput, speedup, or physical GPU-memory reduction.

## Motivation

The first source-bound Qwen3.5-2B smoke completed all seven required rows and
the two cached repeats produced bitwise-identical full-logit hashes at all
eight decode steps. It nevertheless classified `INCOMPLETE` because the
verifier compared BF16 cached execution against a one-shot oracle using the
FP32-like caps:

```text
atol <= 1e-3
rtol <= 1e-4
```

Layer-local diagnostics then showed:

- the first non-exact BF16 layer is layer 0, a linear-attention layer;
- the difference grows through the network while the greedy token remains
  identical;
- the final BF16 logit maximum absolute difference is `0.125`;
- the final BF16 mean absolute difference is `0.01805786043405533`;
- the FP32 control final maximum absolute difference is
  `1.811981201171875e-05`;
- the FP32 control final mean absolute difference is
  `1.987292534977314e-06`.

The evidence supports a deterministic operation-order difference in the BF16
one-shot and recurrent cached paths. It does not support treating cached
execution as unstable, and it does not justify ignoring output decisions.

The old smoke is preserved as authoritative schema-v1 evidence with
classification `INCOMPLETE`. It cannot be retroactively upgraded because it
does not contain the oracle top-k logits, winner/runner-up margins, allclose
violation counts, or explicit dtype profile required by this design.

## Scope

### In Scope

- add a schema-v2 numerical-equivalence contract;
- record explicit execution dtype and comparison policy;
- separate same-path repeatability from cross-path oracle equivalence;
- add decision-relevant top-k, winner, runner-up, and margin evidence;
- add strict FP32 elementwise control evidence;
- keep BF16 full-vocabulary drift as a diagnostic summary;
- update the independent verifier and dependency-light tests;
- run a new remote smoke using the same immutable model revision;
- allow a new canonical run only if the new smoke independently classifies
  `SMOKE_PASS`.

### Out of Scope

- changing production inference code;
- changing Qwen3.5 model math;
- changing the model revision;
- tuning thresholds against a new smoke or canonical result;
- accepting non-deterministic same-path execution;
- accepting any greedy-token disagreement;
- using perplexity or benchmark accuracy as a substitute for state
  compatibility;
- claiming that BF16 full-vocabulary drift is harmless for arbitrary sampling
  policies;
- running the canonical matrix before the schema-v2 smoke passes;
- deleting or rewriting schema-v1 evidence.

## Approved Acceptance Model

The verifier evaluates four independent layers. A missing or malformed layer
is fail-closed.

### 1. Provenance and Determinism

The existing immutable model, source, environment, process, and artifact
bindings remain mandatory.

For every same-path repeated cached case:

- decoded token IDs must be identical;
- full-logit hashes must be bitwise identical at every step;
- request IDs and request generations must be identical;
- step, sequence-length, and position metadata must be identical;
- execution dtype profiles must be identical;
- state snapshot identities and state component content hashes must be
  identical after normalization.

A same-path mismatch is not evidence that one path is semantically wrong. It
means the environment is not stable enough to judge equivalence and therefore
classifies `INCOMPLETE_NUMERICAL_INSTABILITY`.

### 2. Output Behavior

For every cached, chunked, export/import, interleaved, and slot-reuse
comparison against its one-shot reference:

- the greedy token must match exactly at every decode step;
- the oracle winner must be present in the actual frozen top-20 record;
- the actual winner must be present in the oracle frozen top-20 record;
- the actual and oracle winner margins must both be finite and strictly
  positive;
- the winner-margin sign must remain positive, so neither path crosses the
  zero-margin decision boundary;
- ties at the winning logit are rejected because deterministic tie-breaking
  would otherwise hide a decision boundary.

Any greedy-token mismatch or winner-margin boundary crossing is a semantic
`NO_GO` for a complete canonical run. The same condition in smoke remains
`INCOMPLETE`, because smoke is an admission gate rather than the complete
frozen domain.

### 3. Dtype-Aware Numerical Evidence

#### FP32 Control

The smoke includes a mandatory FP32 control for the 17-token cached path. It
uses strict elementwise comparison:

```text
abs(actual - oracle) <= FP32_ATOL + FP32_RTOL * abs(oracle)
FP32_ATOL = 2e-5
FP32_RTOL = 1e-5
FP32_MEAN_ABS_CAP = 3e-6
```

The worker records the number of violating vocabulary elements and the
maximum normalized allclose error. The verifier requires:

```text
allclose_violation_count == 0
mean_abs_diff <= FP32_MEAN_ABS_CAP
```

These constants are frozen before the schema-v2 smoke. They are decimal
ceilings above the already-preserved FP32 diagnostic values, not values learned
from the new smoke. They apply only to:

- model `Qwen/Qwen3.5-2B`;
- revision `15852e8c16360a2fea060d615a32b45270f8a8fc`;
- the frozen source implementation;
- the recorded A100/CUDA/PyTorch/Transformers environment.

An environment or model-identity change requires a separate approved
recalibration design. The verifier must not silently reuse these limits.

#### BF16 Compatibility Path

The canonical compatibility path uses BF16 model execution with the model's
declared FP32 recurrent-state components. It requires the determinism and
output-behavior layers above.

The following BF16 full-vocabulary statistics are mandatory evidence but are
not hard acceptance thresholds:

- maximum and mean absolute difference;
- maximum and mean relative difference;
- p50, p95, p99, and p99.9 absolute difference;
- cosine similarity;
- actual and oracle full-logit SHA-256;
- top-20 token IDs and logits for both paths;
- top-20 intersection size and oracle-top-20 recall;
- actual and oracle winner/runner-up IDs, logits, and margins;
- winner-logit error, runner-up-logit error, and margin drift.

This design deliberately does not derive a BF16 full-vocabulary cap from the
failing smoke. Doing so would be circular and would make a path pass by
definition. BF16 acceptance is instead tied to exact output decisions,
repeatability, and explicit decision-boundary evidence.

This result applies to deterministic greedy continuation only. It does not
establish equivalence for temperature sampling, top-p sampling, beam search,
log-probability APIs, or downstream quality metrics.

### 4. State Semantics and Accounting

All pre-existing state guards remain mandatory:

- architecture schedule;
- recurrent, convolution, and full-attention state roles;
- fixed recurrent-state shape and growing KV shape;
- export/import continuation;
- interleaved request isolation;
- release and slot reuse;
- logical and unique-storage byte reconstruction;
- CUDA allocated and reserved memory snapshots;
- post-run audit.

The dtype-aware numerical change cannot compensate for a missing state tensor,
state aliasing error, request leak, lifecycle error, or storage-ledger
mismatch.

## Schema V2

Set:

```python
SCHEMA_VERSION = 2
DECISION_TOPK = 20
FP32_ATOL = 2e-5
FP32_RTOL = 1e-5
FP32_MEAN_ABS_CAP = 3e-6
```

### Case Row Additions

Every case row adds:

```python
"execution_dtype"
"comparison_policy"
```

`execution_dtype` is one of:

```text
bfloat16
float32
metadata_only
```

`comparison_policy` is one of:

```text
bf16_decision_preserving
fp32_elementwise
none
```

The existing BF16 case IDs remain stable. Add one smoke-only case:

```text
fp32_path_control__cached_vs_one_shot__p17__r0__c17
```

The FP32 control is not part of the full canonical prompt/chunk matrix. It is a
mandatory attribution guard in both smoke and canonical evidence bundles.

### Logit Record Additions

Each logit record keeps the existing fields and adds:

```python
"actual_topk_token_ids"
"actual_topk_logits"
"oracle_topk_token_ids"
"oracle_topk_logits"
"topk_intersection_size"
"oracle_topk_recall"
"actual_winner_token_id"
"oracle_winner_token_id"
"actual_runner_up_token_id"
"oracle_runner_up_token_id"
"actual_winner_logit"
"oracle_winner_logit"
"actual_runner_up_logit"
"oracle_runner_up_logit"
"actual_winner_margin"
"oracle_winner_margin"
"winner_logit_abs_diff"
"runner_up_logit_abs_diff"
"winner_margin_abs_diff"
"abs_diff_percentiles"
"cosine_similarity"
"allclose_violation_count"
"max_allclose_scaled_error"
```

The old `topk_token_ids` and `topk_logits` fields remain aliases for the actual
path during schema-v2 migration. The verifier requires exact equality between
the aliases and the new actual-path fields. They can be removed only by a
future schema revision.

`abs_diff_percentiles` has exactly these keys:

```text
p50
p95
p99
p99_9
```

All numeric values must be finite. Token IDs must be unique within each top-k
list. Top-k logits must be non-increasing, and each winner must be the first
entry of its corresponding top-k list.

### Dtype Profile

`environment.json` and `model_manifest.json` add a normalized dtype profile:

```python
{
    "requested_model_dtype": "...",
    "dominant_parameter_dtype": "...",
    "logit_dtype_before_comparison": "...",
    "comparison_accumulator_dtype": "float32",
    "recurrent_state_dtypes": ["..."],
    "kv_state_dtypes": ["..."],
}
```

The comparison accumulator is always FP32. The verifier reconstructs the
profile from state components and rejects disagreement with the manifests.

## Data Flow

### Worker

For each comparison step, the worker:

1. computes actual and oracle logits without changing either execution path;
2. preserves the original tensor dtype in the dtype profile;
3. converts detached comparison copies to FP32;
4. computes full-logit hashes from canonical contiguous bytes before summary
   reduction;
5. computes actual and oracle top-20 independently;
6. computes exact winner, runner-up, margin, drift, percentile, cosine, and
   allclose metrics;
7. writes raw records only and does not decide the final classification.

No full logit tensor is persisted. Bounded hashes, top-k values, decision
metrics, and aggregate drift statistics are persisted.

### Independent Verifier

The verifier:

1. validates schema version and exact field sets;
2. validates model, source, environment, process, and artifact identity;
3. reconstructs the frozen case domain;
4. verifies same-path bitwise repeatability independently;
5. verifies top-k ordering, uniqueness, alias consistency, winner identity,
   winner margins, and decoded-token identity;
6. applies the FP32 elementwise-control limits only to the FP32 control case;
7. treats BF16 full-vocabulary drift as report evidence, not a rejection cap;
8. runs all unchanged state, lifecycle, request-isolation, and storage guards;
9. emits the authoritative classification and reason list.

The verifier never trusts worker-computed pass booleans, aggregate case counts,
or final classification.

## Classification

The remote runner continues to expose:

```text
SMOKE_PASS
GO
NO_GO
INCOMPLETE
```

### `SMOKE_PASS`

The schema-v2 smoke is complete and independently passes:

- provenance and environment guards;
- architecture verification;
- two BF16 cached repeats;
- BF16 cached versus one-shot output behavior;
- BF16 state export/import;
- the FP32 strict control;
- smoke state/accounting guards;
- post-run audit.

Only this classification authorizes the canonical command.

### `GO`

The complete schema-v2 canonical domain passes every numerical, state,
lifecycle, isolation, ledger, and audit guard.

### `NO_GO`

Use only for a complete canonical domain with a proved semantic violation, such
as:

- a greedy token differs;
- a winner margin is tied, non-positive, or crosses the zero boundary;
- chunked, cached, or export/import continuation changes output behavior;
- request state leaks across identities or generations;
- required state cannot be represented by the frozen contract.

### `INCOMPLETE`

Use for any inability to make a complete authoritative judgment, including:

- same-path repeatability failure;
- FP32 control failure;
- schema-v1-only evidence;
- missing decision metrics;
- dtype-profile mismatch;
- unsupported or changed environment;
- missing rows or artifacts;
- worker, acquisition, resource, or verifier failure;
- any smoke semantic failure;
- unexplained state or ledger inconsistency.

The existing schema-v1 smoke remains `INCOMPLETE`; it is never relabeled.

## Remote Execution and Evidence Preservation

All existing remote constraints remain unchanged:

- host `sitian@10.232.195.203`;
- SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`;
- remote Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`;
- `CUDA_VISIBLE_DEVICES=0`;
- fresh distinct `TINYVLLM_DIST_PORT` and `MASTER_PORT`;
- at most three attempts, retrying only when stderr contains exact
  `EADDRINUSE`;
- no `rsync`, remote checkout edits, `kill`, `pkill`, cache cleanup, shared
  `/tmp` cleanup, or GPU switching.

The new smoke and canonical runs use new run tags. Existing evidence directories
are immutable inputs and must not be deleted, renamed, or overwritten.

The new smoke must use:

```text
model: Qwen/Qwen3.5-2B
revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
```

The runner must reject `canonical` unless a schema-v2 `SMOKE_PASS` artifact is
bound to the same source commit, contract hash, model revision, model file
hashes, and environment identity.

## Test Strategy

### Contract Tests

- schema-v2 constants and exact field sets;
- the FP32 control case appears only in the required domains;
- dtype and comparison-policy validation;
- finite metric validation;
- top-k uniqueness and order helpers;
- winner/runner-up and margin helper behavior;
- FP32 allclose helper boundary cases.

### Probe Tests

- synthetic logits with matching and mismatching winners;
- ties and zero margins;
- top-k records generated independently for actual and oracle;
- exact percentile keys and monotonic percentile values;
- FP32 violation count at, below, and above the boundary;
- BF16 drift summaries do not emit a pass boolean;
- dtype-profile reconstruction from synthetic state components;
- schema-v1 artifacts are never rewritten in place.

### Verifier Tests

Tamper tests must prove:

- same-path hash drift is `INCOMPLETE`;
- greedy mismatch is canonical `NO_GO` and smoke `INCOMPLETE`;
- winner tie or non-positive margin is canonical `NO_GO`;
- a missing oracle winner from actual top-20 fails;
- top-k duplicates or unsorted logits fail;
- top-k alias disagreement fails;
- FP32 allclose violation fails;
- FP32 mean-absolute cap violation fails;
- a changed dtype profile fails;
- arbitrarily large finite BF16 full-vocabulary max-abs drift does not fail by
  itself when all decision guards still pass;
- non-finite BF16 drift metrics fail;
- missing schema-v2 fields fail;
- schema-v1 evidence remains `INCOMPLETE`;
- all pre-existing lifecycle, isolation, and ledger tamper tests still pass.

### Remote Validation Order

1. run all dependency-light local tests;
2. run Python compilation and `git diff --check`;
3. stage the exact clean source snapshot;
4. run remote CPU/source tests;
5. run the new schema-v2 smoke only;
6. independently verify downloaded smoke evidence;
7. run canonical only after `SMOKE_PASS`;
8. independently verify canonical evidence;
9. update the evidence registry and handoff only if canonical evidence exists.

## Success Criteria

This design is implemented successfully when:

1. schema-v2 contract, probe, verifier, runner, and tests are committed from a
   clean approved source snapshot;
2. old schema-v1 evidence remains unchanged and classified `INCOMPLETE`;
3. a new source-bound remote smoke includes both BF16 decision-preserving
   evidence and the FP32 strict control;
4. the independent verifier, not the worker, determines `SMOKE_PASS` or
   `INCOMPLETE`;
5. canonical execution is blocked unless the new smoke is `SMOKE_PASS`;
6. no production engine or README file changes;
7. no compression, quality, latency, throughput, speedup, or physical-memory
   claim is made;
8. if canonical runs, every frozen correctness, lifecycle, isolation, and
   storage-accounting case is independently verified before compatibility
   `GO`.

## Claim Boundary

Even a canonical `GO` proves only:

- the frozen official Qwen3.5-2B reference has deterministic cached execution
  in the tested environment;
- BF16 cached and one-shot paths preserve deterministic greedy decisions
  across the frozen matrix;
- the mandatory FP32 control remains within the frozen strict tolerance;
- hybrid state can be exported, restored, isolated, released, and accounted
  for under the frozen semantic contract.

It does not prove:

- TinyLLMForge natively serves Qwen3.5;
- sampled generation distributions are equivalent;
- benchmark quality is unchanged;
- recurrent state is compressed;
- GPU memory is reduced;
- any kernel or engine path is faster.
