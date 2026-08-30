# Phase-Stitched Exact Graph Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Join the final Exact Prefill CUDA Graph and seven Exact Greedy
decode replays into one bounded, default-disabled K8 transaction while
preserving exact output, first-token host visibility, Scheduler ownership,
and fail-closed semantics.

**Architecture:** Introduce a dependency-light immutable
`PhaseStitchLease`/transaction contract owned by the Scheduler. The
ModelRunner consumes that lease to replay final prefill, publish token 0
through a one-token mailbox, seed retained decode metadata, and launch seven
exact decode replays without a second scheduling pass. `LLMEngine` commits
token 0 and tokens 1..7 as two phases under the same parent identity, then a
four-arm source-bound benchmark and independent verifier decide GO/NO_GO.

**Tech Stack:** Python 3, PyTorch/CUDA Graphs, pinned host mailboxes, CUDA
events/streams, pytest, JSON/JSONL, SHA-256 manifests, remote A100 execution
over SSH.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`; its Desktop path is a
  symlink to the same checkout.
- Do not create a worktree or use a subagent.
- Stage only exact paths; never use broad `git add`, `git reset`, `git clean`,
  or unrelated formatting.
- Keep `phase_stitched_exact_graph_runtime` disabled by default.
- Preserve exact generated token IDs and text.
- Preserve one full target-model forward for every generated token.
- Keep ordinary Scheduler ownership of sequence state and KV allocation.
- Publish token 0 internally without waiting for tokens 1..7; do not add
  ordinary external per-token streaming.
- Reject unsupported identities before final prefill replay and use the
  existing independent Prefill Graph plus ordinary K8 path.
- After stitched prefill replay starts, never retry the request eagerly.
- Quarantine the stitched identity and fail the request on authoritative
  replay, D2H, event, mailbox, validation, or commit failure.
- Support only Qwen3-0.6B BF16, TP1, rank zero, batch one, completion-only,
  exact greedy temperature zero, `ignore_eos=true`, exact prompt shapes 256
  and 2048, and eight remaining output tokens for the first gate.
- Reject EOS-aware, stop-string, callback, per-step-logit, sampled,
  speculative, offload, quantized-KV, sparse/compact-attention, mixed-batch,
  stateful-model, and pending-lease paths.
- Do not implement or claim sentinel-filled prefill graph buckets.
- Keep remote artifacts under
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Reuse `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B` and
  `/data00/home/sitian/tllm/env/bin/python`.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`; never run `kinit` or
  `krenew`.
- Do not terminate or take over external GPU processes.
- Require a strictly clean A100 80GB before launching the real gate.

---

### Task 1: Define the immutable stitch lease and transaction contract

**Files:**

- Create: `tinyvllm/engine/phase_stitched_exact_graph.py`
- Create: `tools/test_phase_stitched_exact_graph.py`

**Interfaces:**

- Produces: `PhaseStitchLease`
- Produces: `PhaseStitchPrefixResult`
- Produces: `PhaseStitchSuffixResult`
- Produces: `PhaseStitchTransaction`
- Produces:
  `build_phase_stitch_lease(...) -> PhaseStitchLease`
- Produces:
  `validate_phase_stitch_prefix(...) -> PhaseStitchPrefixResult`
- Produces:
  `validate_phase_stitch_suffix(...) -> PhaseStitchSuffixResult`

- [ ] **Step 1: Write failing lease-identity tests**

Create tests that construct a valid lease with:

```python
lease = build_phase_stitch_lease(
    sequence_id=7,
    sequence_generation=3,
    schedule_generation=11,
    prefill_graph_identity_sha256="a" * 64,
    prefill_graph_generation=5,
    decode_graph_identity_sha256="b" * 64,
    decode_graph_generation=13,
    prompt_token_count=256,
    final_prefill_first_position=0,
    final_prefill_last_position=255,
    initial_completion_count=0,
    remaining_output_tokens=8,
    decode_first_write_position=256,
    decode_last_write_position=262,
    decode_first_physical_slot=1024,
    decode_last_physical_slot=1030,
    block_table_identity=((64, 9),),
    completion_only=True,
    source_identity_sha256="c" * 64,
)
assert lease.authorized_decode_replay_count == 7
assert lease.parent_token_count == 8
assert len(lease.identity_sha256) == 64
```

Also require that changing any bound generation, graph identity, KV interval,
remaining output count, or visibility policy changes the identity, and reject
booleans-as-integers, malformed digests, non-contiguous ranges, fewer than
eight remaining tokens, and non-completion-only leases.

- [ ] **Step 2: Run the contract tests and confirm RED**

Run:

```bash
python3 -m pytest -q tools/test_phase_stitched_exact_graph.py
```

Expected: collection fails because
`tinyvllm.engine.phase_stitched_exact_graph` does not exist.

- [ ] **Step 3: Implement the dependency-light immutable contract**

Use frozen dataclasses and canonical sorted JSON SHA-256 identities. The lease
must expose:

```python
parent_token_count = 8
authorized_decode_replay_count = 7
first_token_ordinal = 0
suffix_start_ordinal = 1
```

`PhaseStitchTransaction` must allow only:

```text
created -> replay_started -> prefix_ready -> prefix_committed
        -> suffix_ready -> suffix_committed -> closed
```

and terminal transitions:

```text
created -> cancelled
replay_started|prefix_ready -> failed_before_prefix
prefix_committed|suffix_ready -> failed_after_prefix
```

Duplicate prefix/suffix commits and closing with an incomplete phase must
raise.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
python3 -m pytest -q tools/test_phase_stitched_exact_graph.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add -- tinyvllm/engine/phase_stitched_exact_graph.py \
  tools/test_phase_stitched_exact_graph.py
git commit -m "feat(runtime): add phase-stitch lease contract"
```

### Task 2: Add configuration and pure admission policy

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/phase_stitched_exact_graph.py`
- Modify: `tools/test_phase_stitched_exact_graph.py`

**Interfaces:**

- Produces:
  `PhaseStitchDecision(optimized: bool, fallback_reason: str | None)`
- Produces:
  `decide_phase_stitch_admission(...) -> PhaseStitchDecision`

- [ ] **Step 1: Write failing configuration and admission tests**

Require:

```python
assert LLMConfig().phase_stitched_exact_graph_runtime is False
```

Reject a non-boolean config value. Test one accepted tuple and one test for
each frozen rejection family:

```text
disabled
prefill_graph_unavailable
decode_graph_unavailable
prompt_shape_not_allowlisted
sequence_count_unsupported
waiting_request_present
prefilling_request_present
sampling_unsupported
temperature_nonzero
ignore_eos_required
completion_only_required
output_budget_insufficient
decode_kv_capacity_insufficient
tensor_parallel_unsupported
non_root_rank
incompatible_mode:<name>
lease_pending
identity_quarantined
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py::test_phase_stitch_config_defaults_off \
  tools/test_phase_stitched_exact_graph.py::test_phase_stitch_admission_matrix
```

Expected: fail because the config field and decision function are absent.

- [ ] **Step 3: Implement minimal policy**

Add:

```python
phase_stitched_exact_graph_runtime: bool = False
```

Validate exact `bool`. Implement the admission decision as a pure,
dependency-light function. It must not contain model names, prompt labels, GPU
names, or a dynamic tuning policy.

- [ ] **Step 4: Run focused and config tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit configuration and policy**

```bash
git add -- tinyvllm/config.py \
  tinyvllm/engine/phase_stitched_exact_graph.py \
  tools/test_phase_stitched_exact_graph.py
git commit -m "feat(runtime): gate phase-stitched exact graphs"
```

### Task 3: Let the Scheduler preauthorize one parent transaction

**Files:**

- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_phase_stitched_exact_graph.py`

**Interfaces:**

- Produces:
  `Scheduler.prepare_phase_stitch(seqs, ...) -> PhaseStitchLease | None`
- Produces:
  `Scheduler.prepare_phase_stitch_prefix_commit(...)`
- Produces:
  `Scheduler.prepare_phase_stitch_suffix_commit(...)`
- Produces: `Scheduler.cancel_phase_stitch(...) -> None`
- Produces: `Scheduler.fail_phase_stitch(...) -> None`
- Produces: `Scheduler.phase_stitch_summary() -> dict`

- [ ] **Step 1: Write failing Scheduler ownership tests**

Use a fake block manager and one running sequence. Assert that preparation:

- reserves exactly seven post-prefill write positions;
- binds block IDs and generations before replay;
- records one pending parent lease;
- rejects a second lease;
- does not mutate completion tokens.

Assert that prefix commit accepts only token ordinal 0, advances completion
count by one, retains the parent lease, and changes transaction state to
`prefix_committed`. Assert that suffix commit accepts exactly ordinals 1..7,
validates the same parent identity and block generations, closes the lease,
and leaves no pending lease.

- [ ] **Step 2: Run Scheduler tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py -k scheduler
```

Expected: fail because Scheduler has no phase-stitch API.

- [ ] **Step 3: Implement Scheduler preparation and two-phase commit**

Reuse existing block-generation identity helpers. Do not alias the
`ExactGreedyDecodeBurstLease`; the new parent lease must explicitly bind both
prefill and decode graph identities. Reuse `PreparedSchedulerPostprocess` only
as the final mutation carrier.

Before replay, fallback returns `None` and records one exact reason. After
`mark_replay_started`, validation or commit failure is terminal: preserve the
original exception, clear no live KV as reusable, quarantine the parent
identity, and record the last authoritative phase/replay count.

- [ ] **Step 4: Run focused and adjacent Scheduler tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_split_phase.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Scheduler ownership**

```bash
git add -- tinyvllm/engine/scheduler.py \
  tinyvllm/engine/phase_stitched_exact_graph.py \
  tools/test_phase_stitched_exact_graph.py
git commit -m "feat(scheduler): own phase-stitch transactions"
```

### Task 4: Add a one-token prefix mailbox and seven-token suffix mailbox

**Files:**

- Modify: `tinyvllm/engine/phase_stitched_exact_graph.py`
- Modify: `tools/test_phase_stitched_exact_graph.py`

**Interfaces:**

- Produces:
  `PhaseStitchMailboxBackend(copy_stream, first_token_mailbox, suffix_mailbox, ...)`
- Produces:
  `begin_transaction(parent_lease_identity_sha256: str) -> int`
- Produces:
  `enqueue_first_token(...) -> PhaseStitchPrefixResult`
- Produces:
  `enqueue_suffix(...) -> PhaseStitchSuffixResult`
- Produces: `release_transaction(generation: int) -> None`
- Produces: `abort_transaction(generation: int) -> None`

- [ ] **Step 1: Write failing mailbox lifecycle tests**

With fake streams/events/tensors, assert:

- the compute stream records a producer event;
- the copy stream waits on that event;
- token 0 copies to one pinned host slot;
- tokens 1..7 copy to seven pinned host slots;
- waiting for prefix never waits for suffix;
- duplicate enqueue and stale generation fail;
- abort synchronizes owned work before releasing mailbox ownership;
- result byte counts are 8 and 56 for int64 tokens.

- [ ] **Step 2: Run mailbox tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py -k mailbox
```

Expected: fail because `PhaseStitchMailboxBackend` is absent.

- [ ] **Step 3: Implement the mailbox backend**

Follow the existing split-phase backend's event ordering, but use a 1/7
partition and bind both transfers to the new parent lease identity. Cache
host token tuples only after their own completion event synchronizes.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py \
  tools/test_exact_greedy_decode_burst_split_phase.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit mailbox support**

```bash
git add -- tinyvllm/engine/phase_stitched_exact_graph.py \
  tools/test_phase_stitched_exact_graph.py
git commit -m "feat(runtime): add phase-stitch mailboxes"
```

### Task 5: Compose prefill and seven decode replays in ModelRunner

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/phase_stitched_exact_graph.py`
- Modify: `tools/test_phase_stitched_exact_graph.py`

**Interfaces:**

- Produces:
  `ModelRunner.phase_stitch_capability(prompt_token_count: int) -> dict`
- Produces:
  `ModelRunner.run_phase_stitched_exact_graph(seqs, lease, ...)`
- Produces: `ModelRunner.phase_stitch_summary() -> dict`

- [ ] **Step 1: Write failing ModelRunner composition tests**

Use fake prefill/decode graphs and mailboxes. Assert the exact order:

```text
bind prefill live tensors
prefill replay
LM head
float32 argmax token 0
history[0] write
enqueue token-0 D2H and event
seed decode input/position/context/slot/block table
seven decode graph replays
history[1:8] writes
enqueue suffix D2H and event
```

Assert seven, not eight, decode replays; one prefill forward plus seven decode
forwards; no CPU-derived token is rebound between replays; and exact lease,
graph generation, tensor shape/dtype/device, and block identity validation
occurs before the first replay.

- [ ] **Step 2: Run composition tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py -k model_runner
```

Expected: fail because ModelRunner has no stitched capability or execution
method.

- [ ] **Step 3: Implement graph composition with retained tensors**

Reuse the ready exact-prefill entry and existing exact-decode graph. Add only
the adapter needed to seed decode static tensors from token 0 and the
preauthorized lease. Do not capture a new sentinel/padded prefill graph and do
not add model-specific branches.

The runner must return prefix and suffix handles immediately after enqueueing
their transfers. It must not call `.tolist()`, synchronize the suffix event,
or mutate Scheduler state.

- [ ] **Step 4: Implement post-replay quarantine**

Once prefill replay starts, catch only to record the authoritative phase and
replay count, quarantine the joint prefill/decode identity, abort the owned
mailbox, and re-raise the original exception. Never return an eager fallback
after replay begins.

- [ ] **Step 5: Run focused and adjacent graph tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py \
  tools/test_exact_prefill_cuda_graph.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_split_phase.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit ModelRunner composition**

```bash
git add -- tinyvllm/engine/model_runner.py \
  tinyvllm/engine/phase_stitched_exact_graph.py \
  tools/test_phase_stitched_exact_graph.py
git commit -m "feat(runtime): compose exact prefill and decode graphs"
```

### Task 6: Integrate prefix/suffix commit in LLMEngine

**Files:**

- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_phase_stitched_exact_graph.py`

**Interfaces:**

- Consumes all Scheduler and ModelRunner phase-stitch APIs.
- Produces completion-only `step()` behavior with one prefix commit and one
  deferred suffix drain under the same parent transaction.

- [ ] **Step 1: Write failing engine-flow tests**

Create a fake one-sequence final-prefill engine and assert:

- eligible final prefill prepares a stitch lease before ModelRunner dispatch;
- the same step waits only for token 0, commits it once, and stores a pending
  suffix transaction;
- the next `step(completion_only=True)` drains tokens 1..7 without invoking
  `scheduler.schedule()` or a second model dispatch;
- visible outputs equal the ordinary exact path;
- unsupported requests execute the existing independent prefill/K8 path;
- a failure before replay falls back;
- a failure after replay begins quarantines and raises;
- a suffix failure after prefix commit is recorded as partial visibility and
  raises without rolling token 0 back.

- [ ] **Step 2: Run engine tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py -k engine
```

Expected: fail because `LLMEngine.step` has no stitched branch.

- [ ] **Step 3: Implement pre-dispatch admission**

Before final prefill dispatch, query both graph capabilities, ask Scheduler
for one parent lease, and dispatch the stitched runner only when every frozen
condition passes. A rejected lease must leave the existing path unchanged.

- [ ] **Step 4: Implement two-phase host commit**

Wait for and validate token 0, then call the Scheduler prefix commit and
retain the suffix handle. On the next completion-only step, wait for and
validate tokens 1..7, commit them, release the mailbox, and close the parent
transaction without calling ordinary scheduling.

- [ ] **Step 5: Add counters and result metadata**

Report attempts, admissions, fallbacks by reason, prefill replays, decode
replays, first-token/suffix D2H calls and bytes, prefix/suffix commits,
partial-visibility failures, quarantines, pending leases, preauthorized KV
tokens, graph identities, and last authoritative phase.

- [ ] **Step 6: Run engine and regression tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py \
  tools/test_phase_stitch_profile.py \
  tools/test_exact_prefill_cuda_graph.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_split_phase.py \
  tools/test_graph_resident_greedy_tail.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit engine integration**

```bash
git add -- tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/phase_stitched_exact_graph.py \
  tools/test_phase_stitched_exact_graph.py
git commit -m "feat(engine): execute phase-stitched exact graphs"
```

### Task 7: Freeze the four-arm benchmark, producer, and verifier

**Files:**

- Create: `tools/phase_stitched_exact_graph_contract.py`
- Create: `tools/phase_stitched_exact_graph_worker.py`
- Create: `tools/phase_stitched_exact_graph_gate.py`
- Create: `tools/phase_stitched_exact_graph_verify.py`
- Create: `tools/run_phase_stitched_exact_graph_remote.py`
- Create: `tools/test_phase_stitched_exact_graph_benchmark.py`
- Create: `tools/test_run_phase_stitched_exact_graph_remote.py`

**Interfaces:**

- Produces: `build_case_matrix() -> list[dict]`
- Produces: `contract_sha256() -> str`
- Produces one immutable `result.json` per case.
- Produces `summary.json`, `gate.json`, `manifest.json`, producer receipt,
  independent-verifier receipt, and exit receipts.

- [ ] **Step 1: Write failing contract tests**

Freeze:

```python
ARMS = (
    "eager",
    "prefill_only",
    "independent_composition",
    "stitched_composition",
)
PROMPT_TOKEN_COUNTS = (256, 2048)
ROUNDS = 2
WARMUP_REPETITIONS = 2
MEASURED_REPETITIONS = 5
GENERATED_TOKENS = 128
```

Require fresh engines, reversed arm order in round 1, identical prompt hashes,
TP1, batch one, BF16, temperature zero, `ignore_eos=true`,
completion-only execution, and exact config isolation per arm.

- [ ] **Step 2: Run benchmark tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph_benchmark.py \
  tools/test_run_phase_stitched_exact_graph_remote.py
```

Expected: import failure for the new modules.

- [ ] **Step 3: Implement isolated worker and raw schema**

Retain per-request TTFT, token-0-to-token-1 visible gap, TPOT, E2E,
throughput, output token IDs/text hash, capture duration, retained bytes,
allocated/reserved deltas, preauthorized KV capacity, D2H counts/bytes, and
all lease/replay/commit/quarantine/fallback counters. Reject NaN, infinity,
missing metrics, wrong token counts, config drift, or absent graph evidence.

- [ ] **Step 4: Implement producer and independent verifier**

The producer computes D-versus-C primary metrics and A/B attribution. The
verifier must not import producer aggregation code; it must rebuild the case
matrix, raw-pair equality, percentiles, thresholds, source hashes, and
manifest hashes independently.

- [ ] **Step 5: Implement safe remote controller**

The controller must archive only pushed `HEAD`, fail if it changes while
waiting, fail fast when the Kerberos ticket cannot outlive the configured
launch window, wait for one strictly clean A100 without killing any process,
and write only under the approved `/data00` root.

- [ ] **Step 6: Run benchmark/controller tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph_benchmark.py \
  tools/test_run_phase_stitched_exact_graph_remote.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the gate**

```bash
git add -- tools/phase_stitched_exact_graph_contract.py \
  tools/phase_stitched_exact_graph_worker.py \
  tools/phase_stitched_exact_graph_gate.py \
  tools/phase_stitched_exact_graph_verify.py \
  tools/run_phase_stitched_exact_graph_remote.py \
  tools/test_phase_stitched_exact_graph_benchmark.py \
  tools/test_run_phase_stitched_exact_graph_remote.py
git commit -m "test(runtime): gate phase-stitched exact graphs"
```

### Task 8: Run the immutable A100 gate and publish the terminal audit

**Files:**

- Create:
  `artifacts/phase_stitched_exact_graph/<fresh-tag>/`
- Create:
  `docs/superpowers/audits/2026-08-30-phase-stitched-exact-graph-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**

- Consumes pushed source from `origin/feat/kv-sparse-attention`.
- Produces one terminal classification:
  `GO_PHASE_STITCHED_EXACT_GRAPH` or
  `NO_GO_PHASE_STITCHED_EXACT_GRAPH`.

- [ ] **Step 1: Run fresh local verification before push**

Run:

```bash
python3 -m pytest -q \
  tools/test_phase_stitched_exact_graph.py \
  tools/test_phase_stitched_exact_graph_benchmark.py \
  tools/test_run_phase_stitched_exact_graph_remote.py \
  tools/test_phase_stitch_profile.py \
  tools/test_exact_prefill_cuda_graph.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst_split_phase.py \
  tools/test_graph_resident_greedy_tail.py
python3 -m py_compile \
  tinyvllm/engine/phase_stitched_exact_graph.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/phase_stitched_exact_graph_contract.py \
  tools/phase_stitched_exact_graph_worker.py \
  tools/phase_stitched_exact_graph_gate.py \
  tools/phase_stitched_exact_graph_verify.py \
  tools/run_phase_stitched_exact_graph_remote.py
git diff --check
```

Expected: focused tests and compilation pass; diff check is clean.

- [ ] **Step 2: Push source and verify the remote SHA**

Use exact-path staging for any final source changes, commit, push only
`feat/kv-sparse-attention`, then require:

```bash
test "$(git rev-parse HEAD)" = \
  "$(git ls-remote origin refs/heads/feat/kv-sparse-attention | cut -f1)"
```

- [ ] **Step 3: Launch one fresh immutable remote tag**

Run the controller with:

```text
host=sitian@10.232.195.203
python=/data00/home/sitian/tllm/env/bin/python
model=/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B
remote-root=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

Do not overwrite or reinterpret a failed partial tag.

- [ ] **Step 4: Independently verify the downloaded compact bundle**

Run:

```bash
python3 -m tools.phase_stitched_exact_graph_verify \
  --run-dir artifacts/phase_stitched_exact_graph/<fresh-tag>
```

Expected: `verified=true` and the same terminal classification as producer.

- [ ] **Step 5: Apply the frozen runtime gate**

`GO_PHASE_STITCHED_EXACT_GRAPH` requires all of:

```text
exact token IDs and text for every retained pair
zero capture/replay/transaction failure
zero quarantine
D vs C median E2E improvement >= 3% for at least one shape
D vs C aggregate median E2E improvement >= 2%
D vs C token-0-to-token-1 gap improvement >= 10%
TTFT regression <= 2% for each shape
P95 and P99 E2E regression <= 2% for each shape
peak reserved-memory regression <= 3%
complete capture, memory, D2H, lease, visibility, and fallback accounting
independent reconstruction from raw rows and manifest hashes
```

- [ ] **Step 6: Write the completion audit and handoff**

The audit must contain a prompt-to-artifact checklist, source/run identity,
environment, all benefit and cost metrics, gate table, independent-verifier
evidence, failed-attempt reconciliation, and explicit claim boundary.

- [ ] **Step 7: Commit and push exact terminal artifacts**

Stage only the new compact bundle, audit, handoff, and any final source/test
files. Push only `origin/feat/kv-sparse-attention` and verify the remote SHA.
