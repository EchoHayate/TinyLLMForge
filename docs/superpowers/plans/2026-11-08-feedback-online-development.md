# Online Fixed Development Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the independent
> model-worker task and read-only reviews. Main owns controller integration,
> actual remote operations and publication. Do not resume historical agents.

**Goal:** Execute and audit one bounded real five-strategy development diagnostic.

**Architecture:** Public-only persistent inference worker over JSONL; local
fixed-program probe executor and private post-seal scorer. Keep existing
synthetic evidence, protocol and historical runners unchanged.

**Tech Stack:** Python 3.12 standard-library controllers/tests; existing pinned
remote torch/Transformers for optional authorized live inference.

## Global Constraints

- Four task families; 24 paired cells; 120 scheduled strategy records.
- At most 144 model calls, 48 tool calls, 96 probe-case executions.
- Max input 4096, max output 32 tokens; greedy, BF16, eager, thinking disabled.
- No retries, downloads, installations, training, generated-code execution,
  final-test access, shared-runtime modification or other-job signals.
- Only model worker receives public messages. Private scoring remains local.
- `benchmark_eligible=false`, `formal_evaluation_ready=false`,
  `net_benefit_established=false`, `isolated=false`.
- User requested continuous implementation on the existing experimental branch.
  Main alone stages explicit files and commits/pushes after verification.
- Use existing unittest style and TDD; no external test-generation bootstrap.

## Task 1: Remote model worker (delegated, independent)

Files: `experiments/feedback_adaptation/development_model_worker.py`,
`test_development_model_worker.py`.

Implement a standalone source file without repository-local imports. Main's
controller will invoke `python -I -B WORKER DIRECTORY`, where DIRECTORY has
`launch.json` containing exact fields:
`run_id` (32 hex), `hostname`, `uid`, `gpu_uuid`, `gpu_index`, `model_dir`,
`model_hashes` (filename -> sha256), `worker_sha256`, `max_calls` (=144).

Pure helpers:

```python
validate_request(request) -> None
validate_gpu(devices, processes, visible, gpu_uuid, gpu_index) -> None
check_model(model_dir, expected_hashes) -> dict
serve(input_stream, generate, emit, max_calls=144) -> int
```

`serve` accepts at most 131072 UTF-8 bytes per line and rejects oversize or
invalid JSON, unknown fields, repeated IDs or >144 requests before generation.
Overlength detection may consume one sentinel byte from binary stdin, or one
text codepoint (at most four UTF-8 bytes) in injected text streams; it does not
parse or generate from that overlength line.
Request: `{"id": "32 hex", "messages": [{"role": "user", "content": "..."}]}`.
Content nonempty, at most 32000 characters. All generated outputs must be
emitted exactly once; no retries. Canonical request SHA uses sorted keys,
compact separators, ensure_ascii=False, allow_nan=False.
Emit `{"event":"output", "id":..., "request_sha256":..., **generated_fields}`.
Generated fields: text, terminated, input_ids, output_ids, generation_ns,
call_ns, prompt_sha256. IDs integers not bool. EOF returns total call count.

Main runtime verifies launch identity/source/model hashes and exact installed
versions before CUDA initialization; GPU admission before model load.
Set threads/seed; local_files_only, trust_remote_code=False, BF16/eager.
Explicit generate overrides `use_model_defaults=False, do_sample=False`;
verify installed effective mode is greedy, with no implicit model defaults.
Validate non-thinking suffix and token budgets; record raw EOS including output
IDs. No additional/warmup generations. Emits exactly one ready event after
loading, output events during serve, one finished event on EOF.
Ready fields: event, run_id, model_hashes, worker_sha256, versions, load_ns,
preparation_ns, generation_config, generation_overrides, effective_mode,
gpu_uuid, pid, hostname, uid. Finished fields: event, calls, worker_wall_ns,
peak_allocated_bytes, peak_reserved_bytes.
Fatal exception emits event=fatal with error type/message, nonzero process exit.
Remote persistent raw capture is main's launcher responsibility.

- [x] Write failing tests for public-only request validation, strict IDs/fields,
  duplicate/replay/call/line caps, zero retries and digest binding.
- [x] Run focused tests and save actual RED output.
- [x] Implement helpers/service and runtime guarded imports.
- [x] Test with StringIO and fake generation; no GPU, network or model loading.
- [x] Report tests, paths and limitations; no commits or remote operations.

## Task 2: Real protocol and post-seal scorer (main)

Files: `development_protocol.py`, `test_development_protocol.py`.

```python
build_manifest(run_id) -> dict
manifest_digest(manifest) -> str
next_step(cell, method, events, run_id) -> dict
model_request(cell, method, events, run_id) -> dict
score(manifest, frozen_digest, records, audit) -> dict
```

Use four fixed task-specific permutations, no outcome-dependent choice.
Manifest is deterministically reconstructed from run_id/current source for
validation. Cells hold private task metadata plus a deep-copied public view.
24 cells exactly; rotated method order; one none-feedback cell per initial.
Pure next-step handles five strategies, validates typed model/tool events
bound to public request digest and a deterministic opaque request ID derived
from run/cell/method/round; no synthetic replay calls or evidence relabel.
Scoring takes actual audited pass maps from raw fixed CPU records, validates
the same-current-source audit, accounts for missing/partial records in all
denominators, and reports paired deltas/strata without significance claims.

- [x] Write failed tests: counts/public projections, active/direct/rethink/keep/
  mandatory tool, rejection of second VERIFY, malformed/truncated/error events,
  wrong tool/cell/request binding and deep-copy isolation.
- [x] Implement and run focused tests to GREEN.
- [x] Add hand-computed score tests for repairs/regressions/fallback/missing
  and mutation/wrong audit rejection, observe RED then GREEN.

## Task 3: Persistent transport and local campaign (main)

Files: `development_online.py`, `test_development_online.py`.

```python
run_campaign(output_dir, manifest, model, audit) -> dict
```

The injected model receives only `{id,messages}`, returns worker output schema.
The controller snapshots/freezes before first call, journals starts/responses,
limits iterations via next_step, invokes development_cpu.run exactly once per
requested tool, preserves raw artifacts. model failure seals partial campaign,
no retries; protocol failures end only their episode. Final scoring happens
after transcript sealing. Per-episode/controller/model-roundtrip times are
measured separately. No overwrite or resume.

Transport uses nonblocking selector reads with capped buffers, bounded writes,
deadline checks and exact line/digest validation. Main creates only a fresh
subprocess group, terminates only that group on cleanup, saves stderr and
partial stdout. No shell evaluation of model data. One ready event, sequential
output events, finished count matches sent count after EOF.

- [x] Write failing control/transport tests, including real local fake worker
  pipes, immediate EOF, timeout, duplicate/oversize replies and partial failures.
- [x] Implement, verify RED/GREEN and real fixed probe use.
- [x] Read-only review protocol/controller/worker integration; fix concrete
  findings with covering tests, preserve all preexisting sources/receipts.

## Task 4: Live validation and publication (main)

- [x] Freeze source hashes, actual model/dependency identities, private schedule,
  public initial messages, exactly bounded launch and exclusive local/remote
  directories. Preflight remote tokenization/configuration on CPU, no generation.
- [x] Recheck single target GPU; launch once with timeout 595+5 seconds.
  Run all 120 scheduled outcomes or retain explicit partial campaign.
- [x] Replay all raw token/request/tool bindings and audit scores; verify
  remote process exit/GPU release and immutable source/model hashes.
- [x] Summarize negative/ceiling results honestly, shared and per-method costs,
  resource failures and evidence limits. No automatic tuning/re-run.
- [x] Run local full regression and isolated publication tests; scan explicit
  file allowlist for private paths/hosts/credentials.

Publication procedure: commit/push only this experimental branch, then verify
the remote SHA and untouched main/original branch. Record the observed release
status separately, rather than pre-asserting a successful push in this commit.

Self-review: ports approved design §§3–9 without extending to training or
arbitrary code. Live operation has separate fresh admission and finite budget.
No final evaluation threshold is invented or inferred after observing outcomes.

## Execution outcome

The sole launch attempt aborted during SSH authentication before readiness:
zero model/tool calls, zero complete records, all 120 scheduled records missing.
The failed transcript and score were sealed and replayed; no token replay or
successful remote-exit/release acceptance exists. Supplementary read-only
observations found no execution lock or worker artifacts and an idle target GPU.
Two transport-only controls succeeded, so the authentication failure's deeper
cause was not established. No relaunch occurred.

Task 4 checkmarks above mean the prescribed attempt and verification procedures
were performed and the failure retained, not that runtime acceptance passed.
See `experiments/feedback_adaptation/ONLINE_RESULTS.md`. Formal evaluation,
model-effect evidence and weight adaptation remain outside this delivery.
