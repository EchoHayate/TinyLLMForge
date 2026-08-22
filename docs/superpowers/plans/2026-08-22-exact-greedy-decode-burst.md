# Exact Greedy Decode Burst Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce completion-only batch-1 greedy decode TPOT by replaying one exact full target-model step several times before a single host token transfer, while preserving exact output, KV ownership, and Scheduler semantics.

**Architecture:** A dependency-light runtime contract issues immutable Scheduler burst leases and owns lifecycle/accounting. `ModelRunner` reserves one private capture block, captures a complete transformer-to-argmax step with device-resident token feedback and metadata advance, and returns a typed exact-burst result. `LLMEngine` validates the result and commits it through a distinct non-speculative multi-token Scheduler row. A four-arm source-bound gate compares host greedy, full-step graph K1, K4, and K8, with correctness collection isolated from performance timing.

**Tech Stack:** Python 3, PyTorch CUDA Graphs, TinyLLMForge Scheduler/ModelRunner/LLMEngine, dependency-light direct-execution tests, JSON/JSONL evidence, float32 logit sidecars, SSH remote controller.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`, whose resolved target is `/Users/bytedance/dev/TinyLLMForge`.
- Do not create worktrees or use subagents.
- Never modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve all unrelated dirty and untracked files.
- Stage exact paths only; never use broad `git add`, `git reset`, `git clean`, or mass formatting.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- `exact_greedy_decode_burst` defaults to `False`; production width is an integer in `[2, 8]`.
- Width one is accepted only by the gate-only direct causal entrypoint.
- The mechanism layer must not contain Qwen3, checkpoint, tokenizer, prompt, dataset, GPU-model, or benchmark-bucket names.
- Stage 1 is completion-only, Qwen3-0.6B, TP1, rank zero, batch size one, exact numeric temperature zero, and `ignore_eos=True`.
- Every replay executes one complete target-model forward and writes exactly one authorized target-KV position.
- A production burst performs zero intermediate token D2H operations and exactly one final token-vector D2H.
- The correctness-only graph variant performs no intermediate synchronization and may perform one separate bounded sampled-logit D2H after a burst.
- Correctness and performance use separate fresh ModelRunner instances; correctness instrumentation is excluded from performance rows.
- A burst never crosses a physical KV block boundary.
- Capture warmup and capture write only to a private scratch block excluded from Scheduler-visible capacity.
- A pre-replay failure cancels the lease and uses the ordinary one-token path.
- Any failure after the first replay quarantines the burst component, fails the request/step, preserves the original error, and never retries or partially commits.
- Exact burst rows are not speculative rows and never populate `accepted_draft_tokens`.
- Report benefit and cost together: TPOT, P95/P99, TTFT, E2E, throughput, output visibility gap, capture duration, retained bytes, peak allocated/reserved memory, and reserved KV capacity.
- Every remote run tag is immutable.
- All remote task data stays below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Never write remote task data under `/`, `/tmp`, `/private/tmp`, or `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not refresh Kerberos automatically.
- Do not terminate or interfere with unrelated GPU processes.
- GPU admission requires memory used `<=1024 MiB`, utilization `<=5%`, and no compute process.
- A Qwen3-0.6B Stage-1 GO is required before any Qwen3-8B run.
- Unit tests do not prove CUDA Graph viability or speedup; only source-bound GPU evidence may support those claims.

---

## File Structure

- Create `tinyvllm/engine/exact_greedy_decode_burst.py`: generic lease/result identities, eligibility, lifecycle, replay accounting, capture receipt, and JSON-safe summaries.
- Create `tools/test_exact_greedy_decode_burst.py`: dependency-light contract, arithmetic, fallback-order, fake-graph, quarantine, and synthetic-second-caller tests.
- Modify `tinyvllm/config.py`: default-disabled feature flag and strict width validation.
- Modify `tinyvllm/engine/block_manager.py`: exact-burst full-block publication using an explicit materialized-KV end.
- Modify `tinyvllm/engine/scheduler.py`: lease issuance/cancellation, exact-burst row validation, transactional commit, and pending-lease cleanup.
- Modify `tools/test_scheduler_prepared_postprocess.py`: exact-burst row, boundary, publication, completion, rollback, and ordinary/speculative regression tests.
- Modify `tinyvllm/engine/model_runner.py`: private scratch capacity, complete-step capture, production/correctness replay, result summaries, and fail-closed integration.
- Modify `tools/test_model_runner_spec_verify.py`: source-level wiring, fake runner, scratch isolation, replay-count, D2H-count, and fallback tests.
- Modify `tinyvllm/engine/llm_engine.py`: lease dispatch, pre-replay fallback, exact result validation, one transaction commit, token delta, and failure propagation.
- Create `tools/test_llm_engine_exact_greedy_decode_burst.py`: dependency-light engine integration tests.
- Create `tools/profile_exact_greedy_decode_burst.py`: four-arm performance and isolated correctness worker.
- Create `tools/test_profile_exact_greedy_decode_burst.py`: workload order, schema, counters, sampled logits, and metric tests.
- Create `tools/exact_greedy_decode_burst_gate.py`: producer validation, deterministic K4/K8 selection, threshold evaluation, and manifest.
- Create `tools/exact_greedy_decode_burst_verify.py`: independent artifact reconstruction without importing producer classification logic.
- Create `tools/test_exact_greedy_decode_burst_gate.py`: GO and fixed-precedence NO-GO fixtures.
- Create `tools/test_exact_greedy_decode_burst_verify.py`: digest, tamper, row-count, correctness, and classification disagreement fixtures.
- Create `tools/run_exact_greedy_decode_burst_remote.py`: immutable source-bound remote controller.
- Create `tools/test_run_exact_greedy_decode_burst_remote.py`: path, Kerberos, GPU admission, source equality, process cleanup, and complete-download tests.
- Modify `AGENT_HANDOFF_STATE.md`: append the terminal result at true EOF.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`: append prompt-to-artifact checklist and final reconciliation at true EOF.

## Task 1: Generic lease, identity, and accounting contract

**Files:**

- Create: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Create: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- Produces:
  - `ExactGreedyDecodeBurstLease`
  - `ExactGreedyDecodeBurstDecision`
  - `ExactGreedyDecodeBurstResult`
  - `ExactGreedyDecodeBurstCaptureReceipt`
  - `ExactGreedyDecodeBurstStats`
  - `build_exact_greedy_decode_burst_decision(...)`
  - `validate_exact_greedy_decode_burst_result(...)`

- [ ] **Step 1: Write failing lease-arithmetic and fallback-order tests**

Add direct-execution tests that import the module without Torch and assert:

```python
decision = build_exact_greedy_decode_burst_decision(
    enabled=True,
    configured_width=8,
    remaining_output_tokens=6,
    initial_sequence_length=251,
    block_size=256,
    sequence_count=1,
    waiting_count=0,
    prefilling_count=0,
    is_prefill=False,
    do_sample=True,
    batch_kind=None,
    temperatures=(0.0,),
    ignore_eos=(True,),
    completion_only=True,
    tensor_parallel_size=1,
    rank=0,
    graph_available=True,
    incompatible_modes=(),
    pending_lease=False,
    quarantined=False,
)
assert decision.optimized is True
assert decision.authorized_token_count == 6
assert decision.first_write_position == 250
assert decision.last_write_position == 255
```

Cover output-budget clipping, physical-boundary clipping, and fallback when
the final width is one. Assert stable first-failure reasons for disabled,
invalid width, queue contention, prefill/mixed/non-sampling execution,
nonzero or invalid temperature, EOS-sensitive generation, non-completion
visibility, TP/rank mismatch, graph unavailability, incompatible modes,
pending lease, and quarantine.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: import failure because
`tinyvllm.engine.exact_greedy_decode_burst` does not exist.

- [ ] **Step 3: Implement immutable contracts**

Implement these public shapes:

```python
@dataclass(frozen=True)
class ExactGreedyDecodeBurstLease:
    sequence_id: int
    schedule_generation: int
    graph_generation: int
    requested_token_count: int
    authorized_token_count: int
    initial_completion_count: int
    initial_sequence_length: int
    block_table_identity: tuple[tuple[int, int], ...]
    write_block_id: int
    write_block_generation: int
    first_write_position: int
    last_write_position: int
    first_physical_slot: int
    last_physical_slot: int
    remaining_output_tokens: int
    completion_only: bool
    identity_sha256: str


@dataclass(frozen=True)
class ExactGreedyDecodeBurstDecision:
    optimized: bool
    authorized_token_count: int
    first_write_position: int | None
    last_write_position: int | None
    fallback_reason: str | None


@dataclass(frozen=True)
class ExactGreedyDecodeBurstResult:
    lease_identity_sha256: str
    tokens: tuple[int, ...]
    replay_count: int
    final_input_token: int
    final_position: int
    final_context_length: int
    final_physical_slot: int
    graph_identity_sha256: str
    token_d2h_calls: int
    sampled_logit_d2h_calls: int
    sampled_logits: tuple[tuple[int, tuple[float, ...]], ...] = ()
```

Use canonical sorted-key JSON and SHA-256 for lease identity. Reject booleans
where integers are required, non-finite temperatures, duplicate or
non-increasing sampled-logit ordinals, token/replay count mismatch, metadata
advance mismatch, and any physical slot outside the lease.

- [ ] **Step 4: Add stats and lifecycle tests**

Assert exact counters for attempts, acceptances, replay count, target
forwards, intermediate/final token D2H, sampled-logit D2H, clipping,
fallbacks, commits, failures, quarantines, pending leases, capture bytes, and
host-visible gaps. `summary()` must be JSON-safe and sort reason maps.

- [ ] **Step 5: Add a synthetic second caller**

Construct a fake causal decoder caller named only by roles. It must build a
lease, return deterministic tokens, validate the result, and prove the core
module contains none of:

```text
Qwen
checkpoint
tokenizer
prompt
A100
short
medium
long
```

- [ ] **Step 6: Run GREEN and syntax checks**

```bash
python3 tools/test_exact_greedy_decode_burst.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-exact-burst-contract-pycache \
  python3 -m py_compile \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst.py
```

- [ ] **Step 7: Commit and push**

```bash
git add -- \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): add exact decode burst contracts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 2: Scheduler lease and exact multi-token transaction

**Files:**

- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**

- Consumes: Task 1 lease/decision/result types.
- Produces:
  - `ScheduledOutputRow.exact_burst: bool = False`
  - `Scheduler.prepare_exact_greedy_decode_burst(...)`
  - `Scheduler.cancel_exact_greedy_decode_burst(...)`
  - `Scheduler.prepare_exact_greedy_decode_burst_commit(...)`
  - `Scheduler.exact_greedy_decode_burst_summary()`

- [ ] **Step 1: Write failing row-shape tests**

Prove exactly these valid forms:

```python
ScheduledOutputRow(1, (7,), speculative=False)
ScheduledOutputRow(
    1,
    (7, 8, 9, 10),
    speculative=False,
    exact_burst=True,
)
ScheduledOutputRow(
    1,
    (7, 8),
    speculative=True,
    accepted_draft_tokens=(7,),
)
```

Reject exact-burst rows with fewer than two tokens, `speculative=True`,
non-empty `accepted_draft_tokens`, no matching active lease, wrong sequence
ID, wrong token count, or stale lease identity. Preserve all existing
ordinary and speculative validations.

- [ ] **Step 2: Run RED**

```bash
python3 tools/test_scheduler_prepared_postprocess.py
```

Expected: failure because `exact_burst` and lease methods are absent.

- [ ] **Step 3: Implement deterministic lease issuance**

`prepare_exact_greedy_decode_burst(...)` receives the already scheduled
sequence tuple, schedule generation, graph generation/capability, queue
depths, and completion-only authority. It calls the Task 1 policy, obtains
block identities from `BlockManager`, computes:

```python
first_write_position = len(seq) - 1
write_offset = first_write_position % block_size
first_physical_slot = write_block_id * block_size + write_offset
last_physical_slot = first_physical_slot + authorized_count - 1
```

It stores exactly one pending immutable lease only after every validation
passes. Rejection leaves no pending state.

- [ ] **Step 4: Implement cancellation and transactional commit**

`cancel_exact_greedy_decode_burst` accepts only the identical pending lease,
clears it, and records a stable pre-replay fallback reason.

`prepare_exact_greedy_decode_burst_commit` validates the result, constructs
one `ScheduledOutputRow(..., exact_burst=True)`, then delegates to the
existing journal. `_apply_prepared_decode_row` appends tokens in order and
calls:

```python
materialized_end = lease.last_write_position + 1
self.block_manager.publish_full_blocks(
    seq,
    materialized_tokens=materialized_end,
)
```

Only after successful commit does it clear the pending lease and increment
commit counters. Rollback restores host metadata and clears no lease until
the caller records terminal failure.

- [ ] **Step 5: Add block-boundary and lifecycle tests**

With `block_size=4`, cover starts at offsets 0, 1, 2, and 3; prove widths
4, 3, 2, and fallback-one respectively. Prove a width that ends at the
boundary publishes the completed block after KV materialization, a finishing
request releases storage once, a continuing request remains running, and
rollback restores sequence/block/hash/queue state.

- [ ] **Step 6: Run GREEN and regression**

```bash
python3 tools/test_scheduler_prepared_postprocess.py
python3 tools/test_speculative_selection_record.py
python3 tools/test_scheduler_speculative_selection.py
```

- [ ] **Step 7: Commit and push**

```bash
git add -- \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tools/test_scheduler_prepared_postprocess.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): transact exact decode bursts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 3: Strict configuration and private capture capacity

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:
  - `Config.exact_greedy_decode_burst: bool = False`
  - `Config.exact_greedy_decode_burst_tokens: int = 4`
  - `ModelRunner._exact_greedy_burst_scratch_block_ids`

- [ ] **Step 1: Write failing configuration tests**

Use a fake model directory and assert defaults. Reject enable values other
than `bool` and widths that are booleans, non-integers, below two, or above
eight with exact messages:

```text
exact_greedy_decode_burst must be a bool
exact_greedy_decode_burst_tokens must be an integer in [2, 8]
```

- [ ] **Step 2: Write failing capacity tests**

When enabled, reserve one additional physical KV block while leaving
`config.num_kvcache_blocks` equal to Scheduler-visible capacity. Assert the
burst scratch ID is after existing multi-sequence/spec-verify scratch pools,
is absent from visible IDs, and is reported by `capacity_snapshot()`.

- [ ] **Step 3: Run RED**

```bash
python3 tools/test_model_runner_spec_verify.py
```

Expected: assertions fail on missing config fields and scratch inventory.

- [ ] **Step 4: Implement minimal configuration and capacity changes**

Add the two dataclass fields beside the existing greedy flags. In
`allocate_kv_cache()`, add:

```python
burst_scratch_blocks = int(config.exact_greedy_decode_burst)
total_scratch_blocks = (
    decode_scratch_blocks
    + spec_verify_scratch_blocks
    + burst_scratch_blocks
)
burst_start = (
    visible_blocks
    + decode_scratch_blocks
    + spec_verify_scratch_blocks
)
self._exact_greedy_burst_scratch_block_ids = (
    tuple(range(burst_start, burst_start + burst_scratch_blocks))
)
```

Keep KV offload incompatible and assign an empty tuple in that branch.

- [ ] **Step 5: Run GREEN and syntax checks**

```bash
python3 tools/test_model_runner_spec_verify.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-exact-burst-config-pycache \
  python3 -m py_compile tinyvllm/config.py tinyvllm/engine/model_runner.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): reserve exact burst capture capacity" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 4: Complete-step graph capture and replay

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:
  - `ExactGreedyDecodeBurstGraph.capture(...)`
  - `ExactGreedyDecodeBurstGraph.replay(...)`
  - `ModelRunner.exact_greedy_decode_burst_capability()`
  - `ModelRunner.run_exact_greedy_decode_burst(...)`
  - `ModelRunner.exact_greedy_decode_burst_summary()`

- [ ] **Step 1: Add failing fake-graph tests**

Use fake tensors/graph/capture context to prove capture executes this exact
ordered body:

```python
hidden = model(input_token, position)
logits = compute_logits(hidden)
next_token = logits.to(float32_dtype).argmax(dim=-1)
token_history.index_copy_(0, history_index.view(1), next_token)
input_token.copy_(next_token)
position.add_(1)
context_length.add_(1)
slot_mapping.add_(1)
history_index.add_(1)
```

Assert warmup/capture slot mappings point only into the private scratch
block; snapshot live KV before and after and require byte equality.

- [ ] **Step 2: Run RED**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_model_runner_spec_verify.py
```

- [ ] **Step 3: Implement capture**

Create dedicated static tensors for batch one, maximum block-table width,
history capacity eight, and optional correctness sample capacity three.
Capture a production graph with no logit-history write and a gate-only graph
whose fixed device mask selects declared replay ordinals. Use a dedicated
CUDA graph pool, record capture duration and memory deltas, synchronize,
reset every mutable static tensor to sentinels, and retain one immutable
graph identity per variant.

- [ ] **Step 4: Implement replay**

Before replay, copy lease initial token/position/context/slot/block-table
values into static storage and reset history state. Loop exactly
`lease.authorized_token_count` times, recording one target forward and one
graph replay per iteration. Do not inspect tokens between iterations. After
the loop:

```python
tokens = tuple(int(value) for value in token_history[:count].tolist())
```

For correctness mode only, materialize the bounded sampled-logit rows once
after the burst. Build and validate `ExactGreedyDecodeBurstResult`.

- [ ] **Step 5: Add failure tests**

Before-replay source/capability/identity failures return a typed fallback
without replay. First or later replay failure, final D2H failure, and result
construction failure quarantine permanently and re-raise the original
exception. Assert no fallback invokes the target model twice and no
post-replay failure returns tokens.

- [ ] **Step 6: Run GREEN and regressions**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_graph_resident_greedy_tail.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

If the local environment lacks Torch, record the exact import failure and
run the Torch-dependent scripts in the remote source-bound preflight before
claiming them green.

- [ ] **Step 7: Commit and push**

```bash
git add -- \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): replay exact greedy decode bursts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 5: Engine dispatch, fallback, and exact commit

**Files:**

- Modify: `tinyvllm/engine/llm_engine.py`
- Create: `tools/test_llm_engine_exact_greedy_decode_burst.py`

**Interfaces:**

- Consumes: Scheduler lease APIs and ModelRunner capability/result APIs.
- Produces:
  - one exact-burst attempt in the ordinary non-speculative decode branch;
  - one ordered `new_completion_tokens_by_seq` delta per committed burst.

- [ ] **Step 1: Write failing engine tests**

Build fake Scheduler and ModelRunner objects. Prove:

1. eligible completion-only decode issues one lease and one burst call;
2. pre-replay fallback cancels the lease and invokes ordinary `run` once;
3. successful burst validates identity and commits one exact-burst row;
4. K tokens appear once and in order in the engine token delta;
5. completion releases storage once;
6. stale result identity fails before Scheduler commit;
7. post-replay error does not call ordinary `run`;
8. speculative, mixed, prefill, waiting/prefilling contention, and disabled
   cases preserve the existing branch.

- [ ] **Step 2: Run RED**

```bash
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
```

- [ ] **Step 3: Implement the ordinary-branch integration**

Immediately after schedule/partition setup, request ModelRunner capability,
then ask Scheduler for a lease. If no lease exists, execute the unchanged
ordinary branch. If a lease exists, call
`run_exact_greedy_decode_burst(tuple(seqs), lease)`.

On typed pre-replay fallback:

```python
self.scheduler.cancel_exact_greedy_decode_burst(
    lease,
    result.fallback_reason,
)
token_ids = self.model_runner.call("run", ...)
```

On success, validate the result and call the Scheduler exact-burst prepare
and commit methods. Keep speculative runtime/lifecycle code unchanged and
set `token_ids=()` after the dedicated commit to prevent ordinary
postprocess from running twice.

- [ ] **Step 4: Add observation fields**

`last_step_observation` must include burst attempted/accepted/width,
lease/result identities, replay count, token/final-logit D2H counts,
host-visible gap, fallback reason, quarantine reason, and pending lease
count. Existing fields retain their prior meaning.

- [ ] **Step 5: Run GREEN and adjacent regression**

```bash
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_scheduler_prepared_postprocess.py
python3 tools/test_engine_speculative_execution.py
python3 tools/test_chunked_prefill.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/llm_engine.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): integrate exact decode burst execution" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 6: Four-arm profile worker

**Files:**

- Create: `tools/profile_exact_greedy_decode_burst.py`
- Create: `tools/test_profile_exact_greedy_decode_burst.py`

**Interfaces:**

- Produces schemas:
  - `exact-greedy-decode-burst.case.v1`
  - `exact-greedy-decode-burst.correctness.v1`
  - `exact-greedy-decode-burst.summary.v1`
  - `exact-greedy-decode-burst.workload.v1`
  - `exact-greedy-decode-burst.source.v1`

- [ ] **Step 1: Write failing schema and order tests**

Fix:

```python
POLICIES = (
    "host_greedy",
    "full_step_graph_k1",
    "decode_burst_k4",
    "decode_burst_k8",
)
CONTEXT_CASES = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
```

Use a four-order Latin rotation across repetitions, then reverse the order
for odd context indices. Assert two warmups, five measured repetitions, 60
performance rows, and 48 correctness samples
(`4 policies * 3 contexts * 4 sample points`).

- [ ] **Step 2: Run RED**

```bash
python3 tools/test_profile_exact_greedy_decode_burst.py
```

- [ ] **Step 3: Implement isolated correctness collection**

Create fresh runners per arm/context. Preserve exact token IDs, decoded-text
SHA-256, and float32 logits for `prefill-final`, `decode-first`,
`decode-middle`, and `decode-final`. Burst arms use only the gate-only fixed
ordinal graph and report one sampled-logit D2H per containing burst.

- [ ] **Step 4: Implement performance collection**

Create fresh runners with correctness tracing disabled. Record per-token
amortized TPOT, nearest-rank P95/P99, TTFT, E2E, output tokens/s, visible
burst gaps, peak CUDA memory, capture costs, retained bytes, reserved scratch
blocks, lease/replay/D2H inventories, environment, source commit, and source
file hashes. K1 invokes the direct causal entrypoint and cannot be selected.

- [ ] **Step 5: Run GREEN**

```bash
python3 tools/test_profile_exact_greedy_decode_burst.py
```

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tools/profile_exact_greedy_decode_burst.py \
  tools/test_profile_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add exact decode burst profile" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 7: Producer gate and independent verifier

**Files:**

- Create: `tools/exact_greedy_decode_burst_gate.py`
- Create: `tools/exact_greedy_decode_burst_verify.py`
- Create: `tools/test_exact_greedy_decode_burst_gate.py`
- Create: `tools/test_exact_greedy_decode_burst_verify.py`

**Interfaces:**

- Produces:
  - `comparison.json`
  - `gate.json`
  - `manifest.sha256`
  - `independent-verification.json`

- [ ] **Step 1: Write failing producer fixtures**

Cover GO and fixed-precedence failures for correctness, replay inventory,
intermediate/final D2H, lease leak, replay/commit/quarantine event,
host median, host P95, bucket coverage, K1 incremental value, bucket
regression, TTFT/E2E, throughput, memory, visibility gap, cost completeness,
row count, and source/workload evidence.

- [ ] **Step 2: Implement deterministic selection and classification**

Eligible K4/K8 arms must pass correctness/lifecycle/protected metrics.
Select the largest aggregate median TPOT improvement versus host; exact ties
select K4. Apply the spec thresholds exactly:

```text
host aggregate median improvement >= 10%
host aggregate P95 improvement >= 8%
at least 2/3 bucket medians improve >= 8%
K1 aggregate median improvement >= 5%
per-bucket median and P95 regression <= 3%
TTFT and E2E regression <= 3%
throughput regression <= 2%
peak reserved CUDA memory regression <= 3%
maximum host-visible burst gap <= 40 ms
```

- [ ] **Step 3: Write and implement independent verifier**

The verifier must not import producer classification or comparison helpers.
It independently parses raw rows and float32 sidecars, recomputes all
statistics, verifies every source/workload hash, rebuilds the comparison and
manifest digests, and requires exact agreement on classification and selected
arm.

- [ ] **Step 4: Run GREEN**

```bash
python3 tools/test_exact_greedy_decode_burst_gate.py
python3 tools/test_exact_greedy_decode_burst_verify.py
```

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tools/exact_greedy_decode_burst_gate.py \
  tools/exact_greedy_decode_burst_verify.py \
  tools/test_exact_greedy_decode_burst_gate.py \
  tools/test_exact_greedy_decode_burst_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): gate exact decode burst evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 8: Safe source-bound remote controller

**Files:**

- Create: `tools/run_exact_greedy_decode_burst_remote.py`
- Create: `tools/test_run_exact_greedy_decode_burst_remote.py`

**Interfaces:**

- Reuses the canonical SSH, Kerberos, GPU probe, archive, and download
  helpers from `tools/run_staged_inference_benchmark_remote.py`.

- [ ] **Step 1: Write failing safety tests**

Prove all remote paths are descendants of:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/exact-greedy-decode-burst
```

Prove immutable local/remote tags, requested SHA equals pushed branch SHA,
Kerberos TTL is at least 5400 seconds, selected GPU remains the same UUID and
strict-clean immediately before launch, no credential refresh command is
constructed, no process kill command is constructed, and complete manifest
inventory is required before success.

- [ ] **Step 2: Run RED**

```bash
python3 tools/test_run_exact_greedy_decode_burst_remote.py
```

- [ ] **Step 3: Implement controller**

Archive only committed `tinyvllm` and `tools`, upload beneath the approved
staging root, unpack beneath the approved run root, run dependency-light and
Torch-dependent preflight, launch one foreground Stage-1 worker on the
admitted GPU, run producer then independent verifier remotely, download all
manifest-listed files atomically, and independently verify locally.

- [ ] **Step 4: Run GREEN**

```bash
python3 tools/test_run_exact_greedy_decode_burst_remote.py
```

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tools/run_exact_greedy_decode_burst_remote.py \
  tools/test_run_exact_greedy_decode_burst_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): run exact decode burst remotely" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 9: Full verification, hardware gate, and reconciliation

**Files:**

- Create: `artifacts/exact_greedy_decode_burst/<fresh-run-tag>/`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

**Interfaces:**

- Consumes every prior task.
- Produces the final source-bound classification and prompt-to-artifact audit.

- [ ] **Step 1: Run the focused local suite**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_scheduler_prepared_postprocess.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_profile_exact_greedy_decode_burst.py
python3 tools/test_exact_greedy_decode_burst_gate.py
python3 tools/test_exact_greedy_decode_burst_verify.py
python3 tools/test_run_exact_greedy_decode_burst_remote.py
python3 tools/test_graph_resident_greedy_tail.py
python3 tools/test_graph_resident_greedy_tail_gate.py
python3 tools/test_graph_resident_greedy_tail_verify.py
python3 tools/test_zero_temperature_greedy_fast_path_gate.py
python3 tools/test_zero_temperature_greedy_fast_path_verify.py
python3 tools/test_replay_aware_decode_metadata_gate.py
python3 tools/test_replay_aware_decode_metadata_verify.py
python3 tools/test_chunked_prefill.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Run `py_compile` for every changed Python file with
`PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-exact-burst-final-pycache`, then run
`git diff --check` on exact changed paths.

- [ ] **Step 2: Commit and push the source under test**

If Step 1 required fixes, stage only exact source/test paths, commit with the
required single trailer, push, and record the full 40-character SHA. Do not
run hardware evidence from an unpushed or dirty source snapshot.

- [ ] **Step 3: Run fresh Qwen3-0.6B Stage-1**

Use a never-before-used tag:

```bash
PYTHONPATH=. python3 tools/run_exact_greedy_decode_burst_remote.py \
  --run-tag <fresh-run-tag> \
  --source-commit "$(git rev-parse HEAD)" \
  --gpu-wait-timeout-seconds 21600 \
  --gpu-poll-interval-seconds 60
```

Keep all failed/partial artifacts. Do not reuse a tag.

- [ ] **Step 4: Rebuild producer and verifier locally**

```bash
PYTHONPATH=. python3 tools/exact_greedy_decode_burst_gate.py \
  --artifact-dir artifacts/exact_greedy_decode_burst/<fresh-run-tag>
PYTHONPATH=. python3 tools/exact_greedy_decode_burst_verify.py \
  --artifact-dir artifacts/exact_greedy_decode_burst/<fresh-run-tag>
```

Require exact producer/verifier agreement on classification, selected arm,
comparison SHA-256, manifest SHA-256, row counts, and every primary digest.

- [ ] **Step 5: Apply the Stage-2 boundary**

Run Qwen3-8B only if classification is
`GO_EXACT_GREEDY_DECODE_BURST`. Otherwise record the specific NO-GO reason
and keep the feature default-disabled.

- [ ] **Step 6: Append prompt-to-artifact reconciliation**

At true EOF in both canonical documents, map every user requirement and spec
gate to exact files, commits, commands, row counts, digests, and conclusions.
State separately:

- proven by dependency-light tests;
- proven by remote Torch/CUDA preflight;
- proven by Qwen3-0.6B hardware evidence;
- not proven for Qwen3-8B, TP>1, streaming, EOS-aware, multi-sequence, or
  production-default use.

- [ ] **Step 7: Final exact-path commit and push**

```bash
git add -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  artifacts/exact_greedy_decode_burst/<fresh-run-tag>
git -c core.hooksPath=/dev/null commit \
  -m "docs(perf): reconcile exact decode burst gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 8: Final completion audit**

Verify local HEAD equals `origin/feat/kv-sparse-attention`, every intended
file is tracked at HEAD, no unrelated path is staged, both reconciliation
sections are at actual EOF, all manifest entries hash correctly, and the
final claim reports both benefit and cost without exceeding the evidence
boundary.
