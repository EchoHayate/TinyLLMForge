# Remaining frozen matrix validation

User continued after accepted D01 and the proposal to validate the remaining
1384 source/case calls. This implements that next stage, not training.
The earlier D01-only scope is completed; its artifacts must remain unchanged.

## Execution choice

Keep existing remote bwrap/worker boundaries. Execute ordered contiguous chunks
`[(start, min(start + 32, 1416)) for start in range(32, 1416, 32)]`.
There are44 new batches:43 of32 and one of8. D01 ordinals0..31 are reused only
from its accepted raw evidence, never executed again. One40-row task crosses a
chunk boundary; acceptance is computed by task over the complete global matrix,
not by assuming every chunk equals one task.

Compared with one oversized connection, bounded chunks retain existing output
and failure limits. Compared with per-task batching, they avoid a special40-row
backend limit. No new isolation backend or generic scheduler is needed.

## Unchanged boundaries

Frozen data/PINS, worker, D01 reporter and old synthetic APIs stay unchanged.
No model loading, generation, training, tokenizer re-audit, dependencies,
telemetry, Git commits/push/branches/worktrees, or remote service changes.
Only main performs remote calls. Correct answers and witness predictions remain
local; requests have exactly request_id/source/entry_point/args.
No aggregate memory/disk quota or automatic cleanup prerequisite.
Current threat scope remains statically reviewed frozen functions.

Per call:2s wall/CPU, stdout65536, stderr4096. Per batch:90s.
SSH:110s, stdout8388608, stderr131072. Sequential remaining stage deadline900s,
checked before each new batch and limiting its outer collector deadline.
No retries or reused claims. Infrastructure/protocol/runtime failure stops all
later batches; missing rows remain missing. Legitimate semantic mismatches do
not truncate the frozen matrix. Any reference/witness disagreement rejects data
acceptance; do not alter answers to obtain a pass.

## Backend

Add explicit `mode="matrix"` with1..32 requests. Like d01 it requires approved
runtime equality before ready or any task. Preserve probe/d01 modes.
Fix ready.bundle_sha256 to hash received raw bytes rather than reserialized JSON.
An artificial formatted-bundle regression must exercise actual run_bundle control
flow with only Linux plumbing/collector replaced; never execute frozen sources.
New final artificial probes must match modified backend bytes.

## Local reporting

New `validation_matrix.py` stays pure and imports no remote executor.
Reuse strict parsing helpers from unchanged validation_real_report where useful.
`batch_ranges()` returns the frozen44 pairs.
`build_batch_report(source_raw, manifest_raw, bundle_raw, transport, *,
start, expected_runtime)` accepts only a scheduled start and exact frozen slice.
Wire row ordinals are local0..count-1; output rows retain global ordinals.
Verify exact fields, run/request/raw-bundle/worker hashes, order, done, normal
receipt flags, runtime equality, and stop semantics. Short/failing streams
retain all planned rows with missing states. Reject excess rows even in final8.
Reports include kind, run_id, start, stop, received, rows, runtime,
batch_complete, bundle/transport digests. Output rows retain same comparison
semantics as D01. Full admission/benefit flags remain false.

`aggregate_matrix(source_raw, manifest_raw, d01_bundle_raw, d01_transport,
batch_records, *, expected_runtime, d01_expected_runtime)` revalidates D01 raw
evidence through build_d01_report and accepts only ordered records with exact
fields start/bundle_raw/transport. Rebuild every batch, reject duplicate run IDs,
gaps, reorderings and records after incomplete batch. Do not trust report booleans.
D01 runtime is bound to its old approved expectation, new batches to new probe;
different backend versions are explicit lineage, not fabricated identical runs.
Require D01 complete and checks satisfied to import its32 rows.

Aggregate always retains1416 rows/44 tasks. Count354 reference observations,
44 initial witnesses plus88 mutant witnesses. Per-task conditions match frozen
November11 protocol: all references match; initial scoring has a rejection;
all3 designated witnesses match prediction and differ from correct expectation.
`matrix_complete` requires all1416 normal observations and complete transports.
`data_checks_satisfied` combines44 task checks;
`full_dynamic_validation_complete = matrix_complete and data_checks_satisfied`.
benchmark_eligible/formal_evaluation_ready/net_benefit_established/isolated/
generated_candidate_backend_ready remain false even if all data checks pass.
This replaces the old cleanup prerequisite only as already waived by the user.

## Driver and evidence

New private matrix.py prepares all44 answer-free bundles, exact launch source
with EXPECTED_RUNTIME, source/worker/backend/reporter/dependency/probe/D01 hashes,
and a fresh stage/run identity. It checks all bytes before the stage claim.
Per-batch claim is exclusive and precedes SSH. Save raw transport before parsing.
Save incremental full-denominator aggregate after each batch; failure preserves
raw observations and errors rather than retries or synthetic results.
Complete stage seals new evidence (0700 dirs/0600 files) independently of D01.
Tests use synthetic receipts/artificial functions; actual task counts come only
from remote receipts. Independent code/driver review and final probes precede launch.
