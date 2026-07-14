# Light Doc Cache Multi-Target Gate Design

Date: 2026-07-14

## Objective

Determine whether the current Light Doc Cache `calibration_holdout` recovery
method generalizes across target prompts well enough to justify attention
hot-path or physical KV-storage integration.

The gate must compare the trained recovery method against inexpensive
non-trained baselines under one fixed calibration bank. It must produce
per-target evidence, aggregate statistics, and an explicit go/no-go decision.

This remains a default-off research workflow. It does not change attention
kernels, KV allocation lifetime, slot mapping, CUDA Graph behavior, or the
serving scheduler.

## Motivation

The existing Qwen3-0.6B result is limited to one 14-token target prompt.
`calibration_holdout` preserved argmax and improved mean and maximum logit
differences relative to `correlated_same_layer_target` on that prompt, but it
was not best on missing-token MSE. A single target cannot establish
generalization and is insufficient evidence for runtime integration.

The next decision should therefore be based on a target set with varied
content and lengths, not another single-prompt selector ablation.

## Scope

### Included

- A versioned JSON target-prompt dataset.
- A batch driver that reuses one model instance and one fixed calibration bank.
- Three required recovery modes:
  - `repeat_last_target`
  - `correlated_same_layer_target`
  - `calibration_holdout`
- Independent per-target artifacts.
- Aggregate CSV, JSON, and Markdown reports.
- Local unit tests for dataset validation and aggregation.
- A minimal remote Qwen3-0.6B GPU smoke followed by the full target matrix.
- Documentation of the decision and claim boundaries.

### Excluded

- Attention hot-path mutation.
- Replacement or shrinking of `ModelRunner.kv_cache`.
- Physical GPU-memory reduction claims.
- CUDA Graph integration.
- New recovery architectures or selector tuning on the target set.
- Qwen3-8B validation in this phase.
- Task-answer quality claims from next-token logit comparison alone.

## Target Dataset

The dataset will contain eight targets. Each target has a stable ID, category,
prompt text, and intended length bucket.

Required categories:

1. Short factual prose.
2. Long document question answering.
3. Source code or code explanation.
4. Mathematical reasoning.
5. Structured text such as JSON or a table.
6. Repetitive text.
7. Cross-paragraph dependency.
8. Out-of-distribution or unrelated prose.

Required approximate tokenizer-length coverage:

- Short: 16-48 tokens.
- Medium: 49-160 tokens.
- Long: 161-384 tokens.

At least two targets must be in each bucket. Actual token counts are recorded
from the remote model tokenizer; the dataset stores only the intended bucket.

Example schema:

```json
{
  "version": 1,
  "targets": [
    {
      "id": "short_fact",
      "category": "short_factual",
      "length_bucket": "short",
      "prompt": "..."
    }
  ]
}
```

Validation rejects duplicate IDs, empty prompts, unknown categories or length
buckets, fewer than eight targets, and missing bucket coverage.

## Calibration Isolation

The calibration prompts and recovery policy are fixed before any target run.
Target prompts must not affect:

- source-head selection;
- affine recovery fitting;
- calibration token selection;
- recovery-bank parameters;
- thresholds or policy budgets.

The batch workflow builds or loads the calibration bank once, records its file
hash and setup metadata, then evaluates every target against that immutable
bank. This avoids target leakage and prevents per-target tuning from being
mistaken for generalization.

Calibration prompt-prefix KV samples must continue to use prefix-only copies.
The workflow must not retain clones of TinyLLM's full preallocated KV cache.

## Components

### 1. Target Dataset

Proposed file:

`experiments/light_doc_cache/read_path_multi_target_prompts_v1.json`

This is human-readable, versioned, and checked into the repository.

### 2. Batch Driver

Proposed file:

`experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py`

Responsibilities:

1. Load and validate the target dataset.
2. Initialize Qwen3-0.6B once.
3. Load or construct one fixed calibration bank.
4. Run each required mode for each target.
5. Write one summary artifact per `(target, mode)`.
6. Continue after a target-level failure, recording the failure explicitly.
7. Invoke aggregation only after all requested rows have been attempted.

The driver should reuse existing functions from
`run_tinyllm_calibrated_kv_smoke.py` and
`run_tinyllm_sidecar_read_path_smoke.py` rather than duplicating recovery or
temporary cache-pointer swap logic.

Directory shape:

```text
<output-dir>/
  manifest.json
  calibration/
    multi_source_recovery_bank.json
  targets/
    short_fact/
      repeat_last_target/summary.json
      correlated_same_layer_target/summary.json
      calibration_holdout/summary.json
    ...
  multi_target_rows.csv
  multi_target_summary.json
  multi_target_report.md
```

### 3. Aggregator

Proposed file:

`experiments/light_doc_cache/make_multi_target_read_path_report.py`

The existing `make_read_path_recovery_matrix.py` treats setup fields from the
first row as common to all rows. That assumption is invalid for targets with
different token counts and missing-token counts. The new aggregator therefore
keeps all setup fields per row and groups only by explicit target and mode
identifiers.

The aggregator accepts a manifest or artifact root and produces deterministic
reports without loading a model.

### 4. Tests

Proposed file:

`tools/test_light_doc_cache_multi_target.py`

Tests cover:

- valid target dataset parsing;
- duplicate and malformed target rejection;
- required category and length-bucket coverage;
- aggregation with different target token counts;
- missing or failed mode rows;
- per-mode mean, median, P90, worst-case, and win-rate calculations;
- go/no-go boundary conditions;
- deterministic output ordering.

## Per-Target Metrics

Every successful row records:

- target ID, category, and intended length bucket;
- actual prompt-token count;
- recovery mode and role;
- calibration-bank hash;
- logical byte-saving fraction;
- missing compact-token count;
- missing-token MSE;
- missing-token MAE;
- missing-token maximum absolute error;
- maximum absolute logit difference;
- mean absolute logit difference;
- original and restored argmax;
- argmax match;
- artifact path;
- run status and error message when applicable.

Logical byte saving is an accounting metric for the prompt-level sidecar. It is
not reported as observed GPU-memory reduction.

## Aggregate Metrics

For each mode, report:

- completed and failed target counts;
- argmax-match count and rate;
- mean, median, P90, and worst mean-logit difference;
- mean, median, P90, and worst maximum-logit difference;
- mean, median, P90, and worst missing-token MSE;
- mean logical byte-saving fraction.

For `calibration_holdout` relative to
`correlated_same_layer_target`, report:

- target-level mean-logit-difference win count and rate;
- geometric or ratio-safe aggregate improvement;
- mean-logit relative change for every target;
- worst relative regression;
- argmax regressions and recoveries;
- targets omitted from paired comparison because either row failed.

Percentile computation must be deterministic and documented in the report.
The implementation will use nearest-rank P90 over sorted successful values.

## Go/No-Go Gate

The decision is `GO` only when all conditions hold:

1. All eight targets complete for both `calibration_holdout` and
   `correlated_same_layer_target`.
2. `calibration_holdout` argmax-match rate is not lower than the correlated
   baseline rate.
3. `calibration_holdout` has lower mean-logit difference on at least 60% of
   paired targets, which means at least five of eight targets.
4. The arithmetic mean of target-level mean-logit differences improves by at
   least 5% relative to the correlated baseline.
5. The worst target-level relative regression in mean-logit difference is no
   more than 25%.
6. No target that matches argmax under the correlated baseline becomes an
   argmax mismatch under `calibration_holdout`.

The decision is `NO_GO` when any condition fails. Failures are not silently
excluded from the gate.

`repeat_last_target` is reported as a lower-cost reference but does not define
the go/no-go threshold.

## Error Handling

- Dataset validation errors fail before model initialization.
- A calibration-bank failure aborts the run because all targets depend on it.
- A target or mode failure is captured as a failed row with the exception type
  and message; remaining targets continue.
- Aggregation fails if artifact identity is ambiguous or duplicate rows exist.
- The final gate becomes `NO_GO` when required paired rows are missing.
- Existing cache pointers must be restored through `try/finally` even when a
  decode comparison fails.

## Remote Execution

Remote environment:

- Host: `sitian@10.232.195.203`
- Repository:
  `/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge`
- Python:
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`
- Model:
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B`

Execution stages:

1. Run local syntax and unit tests.
2. Synchronize only required files with relative paths.
3. Check remote GPU occupancy and choose an available GPU.
4. Allocate a dynamic distributed port and set both
   `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
5. Run a two-target smoke covering different length buckets.
6. Verify artifact schema and immutable calibration-bank hash.
7. Run the full eight-target matrix.
8. Mirror aggregate and per-target artifacts back to the local repository.

Remote failures caused by SSH, occupied ports, or unavailable GPU memory must
be distinguished from model or recovery failures.

## Validation

Local validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 tools/test_light_doc_cache_multi_target.py

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
  experiments/light_doc_cache/make_multi_target_read_path_report.py \
  tools/test_light_doc_cache_multi_target.py

git diff --check
```

Remote validation must establish:

- one model initialization for the batch;
- one immutable calibration-bank hash across all rows;
- distinct target IDs and actual token counts;
- all three required modes attempted for every target;
- pointer restoration after every read-path comparison;
- aggregate files regenerated from mirrored per-target artifacts;
- a reproducible `GO` or `NO_GO` result.

## Documentation and Decision Handling

After validation:

- Add commands, artifact paths, metrics, limitations, and the decision to
  `experiments/light_doc_cache/README.md`.
- Add a concise continuation record to `AGENT_HANDOFF_STATE.md`.
- Preserve negative results. A `NO_GO` decision is a useful outcome and should
  stop further tuning of this recovery selector on the same target set.

If the decision is `GO`, the next design phase may cover a default-off
attention read-path prototype and physical storage accounting. It still must
separate quality validation, observed latency, and observed GPU-memory
reduction.

If the decision is `NO_GO`, the next performance work should prioritize either:

1. controlled automatic prefix-cache and shared-prefix benchmarks; or
2. adaptive speculative decoding with acceptance/cost-based triggering and
   batched target verification.

## Claim Boundary

Passing this gate would show only that `calibration_holdout` generalizes better
than the selected inexpensive baseline on this eight-target Qwen3-0.6B
next-token read-path matrix.

It would not by itself prove:

- end-task quality;
- Qwen3-8B generalization;
- serving-workload latency improvement;
- throughput improvement;
- physical KV-memory reduction;
- CUDA Graph compatibility;
- safe production enablement.
