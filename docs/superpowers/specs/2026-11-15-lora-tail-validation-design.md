# Frozen matrix tail continuation

## Approved scope and choice

The user said “继续” after the explicit proposal to allow resubmitting only the
32 requests whose SSH attempt failed at jump authentication, then finish the
296 never-claimed requests. This is one new bounded stage for [1088,1416), not
permission to replay the old stage, completed rows, or automatically retry again.
The old stage stays incomplete forever. Its failure evidence remains immutable.

Use eleven original-grid batches: ten of32, then8. Keep the existing reviewed
backend, worker, source data and pure matrix reporter unchanged. A separate pure
continuation report records both the failed prior stage and the selected normal
observations across stages. This is preferable to extending the consumed driver
or silently removing the failed attempt from the old report.

No training/model loading/generation, tokenizer rerun, dependencies, telemetry,
Git publication, branches or worktrees. Work in the existing experiment checkout.
Correct answers and witness predictions stay local. Remote request fields remain
exactly request_id/source/entry_point/args. No restored aggregate RAM/disk quota
or automatic-cleanup requirement. Only the main agent performs remote calls.

## Evidence inputs

Previous root: /private/tmp/tlf-matrix-validation-20261115-6PAv0miA.
Previous stage: matrix-2092f8a4d982461ead75c6f97eb8fc66.
Previous seal SHA256:
f6012de5dbbc2c8bdd17042cfc642824fad177ac31f919fd4ac60553e5e1bf59.
Prior stage has34 attempts:33 complete plus one program_error/255 with empty
stdout at start1088. Its aggregate is1088 received,328 missing. The failed
transport hash and run ID must be explicitly bound to the resubmission approval.
All377 previous private artifacts stay unchanged. Workspace hash changes are
limited to new files and current README; old dated result documents stay frozen.

New private root: /private/tmp/tlf-tail-validation-20261115-wh6z70so.
No change to the prior sealed private driver, tests or seal.

## Pure reporter

Create validation_tail.py and test_validation_tail.py only for the repository
implementation. API:

`aggregate_tail(source_raw, manifest_raw, d01_bundle_raw, d01_transport,
previous_records, tail_records, *, expected_runtime, d01_expected_runtime,
authorization)`

Records have the unchanged exact start/bundle_raw/transport schema. Revalidate
all previous raw records with aggregate_matrix; require exactly34 records,33
complete batches and a final start1088 failure with received0, no ready/done,
program_error/255 and empty stdout. Require saved stderr to show the jump
authentication rejection, not a target command launch. This condition is
in addition to, never a replacement for, explicit resubmission approval.

Authorization has exactly: kind=LORA_TAIL_AUTHORIZATION, prior_stage_id,
stage_id, prior_seal_sha256, failed_run_id, failed_transport_sha256,
resubmit_start=1088, resubmit_stop=1120, tail_stop=1416, user_approved=true.
IDs are32 lowercase hex characters; stage differs from prior. Hashes are64
lowercase hex. Failed ID/hash must match the actual last prior record.
The driver pins the historical seal digest and stage ID; the pure function
checks structure and receipt equality but does not authenticate human approval.

Tail has0..11 ordered records starting1088. No duplicate run ID across D01,
all prior attempts and tail; new IDs must differ from stage IDs too. No record
after an incomplete tail batch. The strict existing aggregator can rebuild
observations from previous_records[:-1] + tail_records, but the returned object
MUST explicitly retain the complete old incomplete report, failed attempt,
approval, per-batch stage attribution and combined observation report.

Output: kind=LORA_MATRIX_CONTINUATION_REPORT, authorization, prior, combined,
selected_batches. prior is the entire reconstructed incomplete aggregate.
combined uses kind=LORA_MATRIX_CROSS_STAGE_OBSERVATIONS; it is not a claim that
either individual stage executed1416 calls. selected_batches lists exact
stage_id/start/run_id for33 prior successes and each included tail record.
Keep1416 denominator and all44 task checks, including T06 spanning stages.
All benchmark/formal/net-benefit/isolated/generated-candidate flags stay false.
Full dynamic data validation may be true only when all1416 normal receipts
and all task checks are satisfied. Negative semantic outcomes never cause
selective stopping or source/answer edits.

## One-shot private driver

Create tail.py and test_tail_driver.py in the new evidence root. The old driver
is not imported or modified. Reuse only the unchanged backend collector and
pure report APIs. Prepare snapshots of old sources/D01/raw attempts/approval,
all11 new bundles, exact launch source/SSH argv and all dependency hashes.
Read historical source/data/probe bindings from the verified prior manifest.
Match backend/worker and approved runtime exactly; unchanged bytes reuse the
already accepted artificial probe evidence, with actual runtime checked on each
remote batch. No need to rerun artificial probes or tokenizer for unchanged code.

Verify seal, prior artifact bytes, new source snapshots, exact frozen projections,
unique IDs, approval and command before exclusive stage claim. Keep exclusive
per-batch claims before SSH, revalidate critical source/data/dependency hashes
before each launch. New finite stage limit900s; item2s, batch90s, SSH110s;
stdout8388608/stderr131072, unchanged worker limits. No automatic retry.

Save initial full-denominator report, raw transport/wire/stderr, incremental
report and terminal summary. A collector or parser exception stops the stage,
preserves a bound error and last strictly verified combined report; unvalidated
rows never count. An incomplete batch also stops later launches. Deadline expiry
stops without consuming another batch. Existing stage/run claims forbid reuse.
Failures during preparation or input verification must not launch anything.

## Verification and closeout

Synthetic tests only, no frozen-source execution locally. Cover exact328
projection, full1416 cross-stage result, retained failure, T06 boundary, negative
reference/witness data, missing done/partial/invalid stream, auth mismatch,
reused IDs, gaps and old input drift. Driver tests replace only the collector
with artificial receipts. Independent code review precedes actual launch.
Pure raw reconstruction and independent evidence review follow actual execution;
seal new artifacts independently and verify all old377 files remain unchanged.
