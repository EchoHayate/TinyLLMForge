# Split-Phase Ragged-Lease Coalescing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace K8 split-backend rejection churn for authorized widths
`2..7` with K4-capped one-phase exact bursts while preserving full K8
`4 + 4` publication.

**Architecture:** A pure selector computes the scheduler-requested width from
remaining output capacity and current writable KV-block capacity. The
scheduler issues an immutable K8 lease only when split phase is legal;
otherwise it issues a K2–K4 lease. The model runner dispatches K8 leases to
the split backend and smaller leases to the existing one-phase exact graph.

**Tech Stack:** Python, PyTorch CUDA Graphs, TinyLLMForge scheduler/model
runner, JSON/JSONL evidence producers, pytest-style script tests, SSH
source-bound remote controller.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not create a worktree or use subagents.
- Preserve unrelated dirty and untracked files.
- Stage only named task files; never use broad `git add`, `git reset`, or
  `git clean`.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one trailer:
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>`.
- Push only to `origin/feat/kv-sparse-attention`.
- Keep the feature default-disabled until the hardware gate returns
  `GO_EXACT_BURST_RAGGED_COALESCING`.
- Remote task data may be written only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Never refresh Kerberos automatically, kill unrelated GPU processes, or
  reuse a run tag.
- GPU admission requires memory at most `1024 MiB`, utilization at most
  `5%`, and no compute process.
- Every optimization report must include measured benefit and measured cost.

---

### Task 1: Pure ragged-width selector

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- Produces:
  `select_exact_greedy_decode_burst_width(*, configured_width: int,
  remaining_output_tokens: int, initial_sequence_length: int,
  block_size: int, split_phase_enabled: bool,
  ragged_coalescing_enabled: bool) -> int`
- Consumes no runtime objects and performs no mutation.

- [ ] **Step 1: Add RED selector tests**

Add tests that assert:

```python
assert select_width(capacity=8) == 8
assert select_width(capacity=7) == 4
assert select_width(capacity=6) == 4
assert select_width(capacity=5) == 4
assert select_width(capacity=4) == 4
assert select_width(capacity=3) == 3
assert select_width(capacity=2) == 2
assert select_width(capacity=1) == 8
assert select_width(capacity=0) == 8
```

Construct capacity separately through output budget and writable block
positions. Also assert disabled coalescing, disabled split phase, and
configured K4 return the configured width unchanged.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: failure because
`select_exact_greedy_decode_burst_width` is absent.

- [ ] **Step 3: Implement the pure selector**

Validate booleans and integer ranges with the module's existing helpers.
Compute:

```python
first_write_position = initial_sequence_length - 1
writable_positions = block_size - (
    first_write_position % block_size
)
capacity = min(
    configured_width,
    remaining_output_tokens,
    writable_positions,
)
```

Return unchanged width unless all of these are true:

```python
ragged_coalescing_enabled
and split_phase_enabled
and configured_width == 8
and 2 <= capacity < 8
```

For the selected case return `min(4, capacity)`.

- [ ] **Step 4: Run GREEN and adjacent contract tests**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_exact_greedy_decode_burst_split_phase.py
```

Expected: both scripts pass.

- [ ] **Step 5: Commit and push**

Stage only the two files and commit:

```text
feat(runtime): select bounded ragged burst widths
```

---

### Task 2: Strict default-off configuration

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:
  `Config.exact_greedy_decode_burst_ragged_coalescing: bool = False`
- Requires exact burst enabled, split phase enabled, and configured width
  eight whenever true.

- [ ] **Step 1: Add RED configuration tests**

Assert the dataclass default is `False`, reject non-booleans, and reject:

```python
Config(..., exact_greedy_decode_burst_ragged_coalescing=True)
Config(
    ...,
    exact_greedy_decode_burst=True,
    exact_greedy_decode_burst_ragged_coalescing=True,
)
Config(
    ...,
    exact_greedy_decode_burst=True,
    exact_greedy_decode_burst_split_phase=True,
    exact_greedy_decode_burst_tokens=4,
    exact_greedy_decode_burst_ragged_coalescing=True,
)
```

The first two fail because split phase is absent; the third fails because
split phase itself requires K8.

- [ ] **Step 2: Run RED**

Run the focused config test in
`tools/test_model_runner_spec_verify.py`.

Expected: failure because the field is absent.

- [ ] **Step 3: Add field and validation**

Place the new boolean adjacent to the split-phase field. Validate its type,
then require exact burst, split phase, and K8 through explicit error
messages.

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 -m pytest tools/test_model_runner_spec_verify.py \
  -k exact_greedy_decode_burst_config_is_strict_and_default_off -q
```

Expected: selected test passes.

- [ ] **Step 5: Commit and push**

Stage only the two files and commit:

```text
feat(config): gate ragged burst coalescing
```

---

### Task 3: Scheduler-issued effective width

**Files:**

- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`
- Modify: `tools/test_llm_engine_exact_greedy_decode_burst.py`

**Interfaces:**

- `Scheduler.prepare_exact_greedy_decode_burst()` gains keyword arguments
  `split_phase_enabled: bool` and `ragged_coalescing_enabled: bool`.
- The returned lease records the selected width in both
  `requested_token_count` and, after ordinary authorization,
  `authorized_token_count`.

- [ ] **Step 1: Add RED scheduler tests**

Use real scheduler fixtures for remaining output capacities `7`, `3`, and
`1`. Assert:

```python
remaining 7 -> lease requested=4, authorized=4
remaining 3 -> lease requested=3, authorized=3
remaining 1 -> no lease, fallback=insufficient_output_budget
```

Add a block-edge case whose writable capacity is three while output budget
is at least eight; assert a K3 lease and no cross-block authorization.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 tools/test_scheduler_prepared_postprocess.py
```

Expected: new tests fail because the scheduler still requests K8.

- [ ] **Step 3: Select width before the existing decision**

In `prepare_exact_greedy_decode_burst()`, call the pure selector and pass its
return value as `configured_width` to both
`build_exact_greedy_decode_burst_decision()` and
`build_exact_greedy_decode_burst_lease()`.

Do not alter remaining-budget, block-generation, or pending-lease checks.

- [ ] **Step 4: Pass flags from the engine**

At the sole engine call site, pass:

```python
split_phase_enabled=exact_burst_split_phase_enabled
ragged_coalescing_enabled=bool(
    model_runner_config
    .exact_greedy_decode_burst_ragged_coalescing
)
```

Update fake scheduler assertions so the values are visible in event
records.

- [ ] **Step 5: Run GREEN**

Run:

```bash
python3 tools/test_scheduler_prepared_postprocess.py
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
```

Expected: both scripts pass.

- [ ] **Step 6: Commit and push**

Stage only the four files and commit:

```text
feat(scheduler): issue bounded ragged burst leases
```

---

### Task 4: Width-owned model-runner dispatch

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- K8 leases with split phase enabled call `replay_split_phase()`.
- K2–K4 ragged leases call the existing `replay()`.
- The dispatcher never rewrites a lease.

- [ ] **Step 1: Add RED dispatch tests**

Create fake one-phase and split backends with call counters. Under split plus
ragged configuration:

```python
lease(width=8) -> split backend once, one-phase backend zero
lease(width=4) -> split backend zero, one-phase backend once
lease(width=3) -> split backend zero, one-phase backend once
```

Assert the K4/K3 calls preserve `continuation_enabled=False` and produce no
split mailbox lifecycle calls.

- [ ] **Step 2: Run RED**

Run the focused exact-burst model-runner tests.

Expected: K4/K3 currently enter `replay_split_phase()` and fail.

- [ ] **Step 3: Implement immutable-width dispatch**

Change the split branch predicate to:

```python
if (
    self.config.exact_greedy_decode_burst_split_phase
    and lease.authorized_token_count == 8
):
```

Leave the existing `graph.replay()` path unchanged for smaller leases.

- [ ] **Step 4: Run GREEN and adjacent tests**

Run:

```bash
python3 -m pytest tools/test_model_runner_spec_verify.py \
  -k "exact_greedy_decode_burst" -q
python3 tools/test_exact_greedy_decode_burst_split_phase.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit and push**

Stage only the two files and commit:

```text
feat(runtime): route ragged leases to one-phase replay
```

---

### Task 5: End-to-end K8 plus K4/K3 lifecycle

**Files:**

- Modify: `tools/test_llm_engine_exact_greedy_decode_burst.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**

- The integration test observes fifteen split K8 parent leases, followed by
  atomic K4 and K3 commits.
- No pending split transaction exists after either ragged commit.

- [ ] **Step 1: Add a RED lifecycle sequence**

Drive the fake engine through:

```text
K8 prefix -> K8 suffix -> K4 atomic -> K3 atomic -> finished
```

Assert scheduler order, sequence lengths, completion counts, lease release,
and that `split_phase_requires_k8` and `insufficient_output_budget` are both
absent.

- [ ] **Step 2: Run RED**

Run both focused scripts.

Expected: fail at the first ragged dispatch or width assertion until Tasks 3
and 4 are integrated.

- [ ] **Step 3: Make only integration corrections**

Adjust engine observation fields or fake interfaces only where the RED test
demonstrates a real mismatch. Do not add a second state machine.

- [ ] **Step 4: Run GREEN**

Run:

```bash
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_scheduler_prepared_postprocess.py
python3 tools/test_chunked_prefill.py
```

Expected: all scripts pass.

- [ ] **Step 5: Commit and push**

Stage only files changed by this task and commit:

```text
test(runtime): cover ragged burst lifecycle
```

---

### Task 6: Source-bound evidence producer

**Files:**

- Create: `tools/profile_exact_burst_ragged_coalescing.py`
- Create: `tools/test_profile_exact_burst_ragged_coalescing.py`

**Interfaces:**

- Policies:
  `decode_burst_k4`, `decode_burst_k8_split_phase`,
  `decode_burst_k8_split_phase_ragged`.
- Produces 45 performance rows, 36 correctness rows, source manifest,
  workload manifest, and summary.
- Adds `tail_seven_elapsed_ns` to every performance row.

- [ ] **Step 1: Write RED schema and inventory tests**

Freeze schema versions, policy configuration, row identities, source-file
inventory, and expected candidate counters:

```python
requested_width_histogram == {"3": 1, "4": 1, "8": 15}
authorized_width_histogram == {"3": 1, "4": 1, "8": 15}
commits == 17
committed_tokens == 127
final_token_d2h_calls == 2
fallback_counts == {}
prefix_commits == suffix_commits == 15
```

- [ ] **Step 2: Run RED**

Run:

```bash
python3 tools/test_profile_exact_burst_ragged_coalescing.py
```

Expected: import failure because the producer does not exist.

- [ ] **Step 3: Implement producer by composition**

Reuse serialization and runtime helpers from
`profile_exact_burst_split_phase.py`; do not copy runtime algorithms.
Calculate:

```python
tail_seven_elapsed_ns = sum(
    amortized_tpot_samples_ns[-7:]
)
```

Validate all counters and split/ragged ownership before writing a row.

- [ ] **Step 4: Run GREEN**

Run the new producer test plus both existing profiler test scripts.

- [ ] **Step 5: Commit and push**

Stage only the two new files and commit:

```text
feat(benchmark): add ragged coalescing producer
```

---

### Task 7: Gate and independent verifier

**Files:**

- Create: `tools/exact_burst_ragged_coalescing_gate.py`
- Create: `tools/test_exact_burst_ragged_coalescing_gate.py`
- Create: `tools/exact_burst_ragged_coalescing_verify.py`
- Create: `tools/test_exact_burst_ragged_coalescing_verify.py`

**Interfaces:**

- Producer gate emits `gate.json`, `comparison.json`, and
  `manifest.sha256`.
- Independent verifier reads raw rows and sidecars without importing the
  producer or gate.

- [ ] **Step 1: Add RED gate tests**

Test all four classifications, every benefit/cost threshold, missing rows,
duplicate rows, bad candidate ownership, wrong source commit, sidecar hash
failure, and non-finite numbers.

- [ ] **Step 2: Implement producer gate**

Reconstruct paired metrics by repetition and context. Enforce the exact
thresholds from the design spec, including at least 10% paired median
tail-seven improvement.

- [ ] **Step 3: Add RED verifier independence test**

AST-scan the verifier source and reject imports from the producer and gate.
Mutate one derived metric at a time and assert independent rejection.

- [ ] **Step 4: Implement independent verifier**

Duplicate only frozen constants and mathematical reconstruction, not
producer code. Emit `independent-verification.json` and require numeric
agreement at `1e-9`.

- [ ] **Step 5: Run GREEN**

Run all four new tests and the existing split-phase gate/verifier tests.

- [ ] **Step 6: Commit and push**

Stage only the four files and commit:

```text
feat(benchmark): gate ragged burst coalescing
```

---

### Task 8: Remote controller

**Files:**

- Create: `tools/run_exact_burst_ragged_coalescing_remote.py`
- Create: `tools/test_run_exact_burst_ragged_coalescing_remote.py`

**Interfaces:**

- Reuses the existing Kerberos, SSH, GPU admission, source archive, remote
  scratch, download, manifest, and cleanup helpers.
- Launches no worker until a qualifying clean GPU exists.

- [ ] **Step 1: Add RED controller tests**

Freeze:

- approved remote root;
- source commit equals pushed branch head;
- empty source patch;
- no tag reuse;
- GPU admission thresholds;
- Kerberos fail-fast behavior;
- exact producer, gate, and verifier commands;
- local artifact layout and manifest coverage.

- [ ] **Step 2: Implement controller**

Follow `run_exact_burst_split_phase_remote.py`, replacing only campaign
names, source inventory, commands, and artifact expectations.

- [ ] **Step 3: Run GREEN**

Run the new controller test and all adjacent remote-controller tests.

- [ ] **Step 4: Commit and push**

Stage only the two files and commit:

```text
feat(benchmark): run ragged gate remotely
```

---

### Task 9: Full verification, hardware gate, and reconciliation

**Files:**

- Modify:
  `docs/superpowers/plans/2026-08-23-split-phase-ragged-lease-coalescing.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Add downloaded artifacts under:
  `artifacts/exact_burst_ragged_coalescing/<fresh-run-tag>/`

**Interfaces:**

- Final classification is one of the four frozen design values.
- Every reported number is reconstructible from downloaded raw artifacts.

- [ ] **Step 1: Run the full local regression set**

Run all exact-burst, split-phase, ragged producer/gate/verifier/controller,
chunked-prefill, compile, and focused diff checks. Record exact pass counts.

- [ ] **Step 2: Commit and push the verified source**

Confirm local HEAD equals `origin/feat/kv-sparse-attention`, the staged patch
contains only named files, and the source patch is empty.

- [ ] **Step 3: Launch a fresh source-bound hardware run**

Use a never-before-used run tag and the pushed 40-character HEAD. Let the
local controller wait for a clean GPU and launch automatically.

- [ ] **Step 4: Verify artifacts independently**

Require:

- 45 performance rows;
- 36 correctness rows;
- producer gate;
- independent verification;
- source and workload manifests;
- SHA-256 manifest;
- controller preflight, completion, exit code, and runner log.

- [ ] **Step 5: Reconcile benefit and cost**

Report:

- tail-seven latency improvement;
- aggregate and per-bucket TPOT, P95, throughput, TTFT, and E2E;
- maximum host-visible gap against K4;
- peak allocated/reserved and retained static memory;
- graph captures and D2H calls;
- exact counters, fallback ownership, token/logit parity;
- limitations and non-goals.

- [ ] **Step 6: Update plan, handoff, and Phase 1 audit**

Append prompt-to-artifact reconciliation with exact run tag, source commit,
artifact paths, hashes, classification, and unresolved limitations.

- [ ] **Step 7: Final commit and push**

Stage only the plan, handoff, audit, and selected canonical artifacts. Commit:

```text
docs(perf): reconcile ragged coalescing gate
```
