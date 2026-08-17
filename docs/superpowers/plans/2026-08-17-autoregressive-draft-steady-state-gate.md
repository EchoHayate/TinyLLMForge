# Autoregressive Draft Steady-State CUDA Graph Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the TP4/B4/Q4 paired gate measure steady-state CUDA Graph replay rather than charging first capture to the measured graph run.

**Architecture:** Every fresh eager and graph worker runs one unmeasured in-process warmup batch and one measured batch on the same engine. The gate converts both warmup-end and measured-end cumulative graph state into the canonical payload, and schema validation proves that measured graph execution increased replay count without changing capture, quarantine, fallback, or retained-resource state.

**Tech Stack:** Python 3.11, pytest, TinyLLMForge source-bound JSON gate and verifier.

## Global Constraints

- Exact identity remains TP4/B4/Q4 with prompt length 256 and output length 16.
- Do not round or pad shapes.
- Do not change production exact-greedy, Proposal-KV commit/rollback, or TP failure convergence behavior.
- Capture remains admitted only after successful eager execution.
- Pre-replay failure may fall back eager; replay-started failure remains fail-closed and quarantined.
- Pair-level schedule remains two warmup pairs and eight measured balanced pairs.
- Only measured batches contribute timing and performance statistics.
- The canonical gate payload is schema version 2 and records `in_process_warmup_runs = 1`.
- Use explicit file staging; never use `git add -A`.
- A commit must end with exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Require Same-Engine In-Process Warmup

**Files:**
- Modify: `tools/autoregressive_draft_cuda_graph_gate.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `build_worker_command(..., mode, output_path) -> list[str]`
- Produces: worker commands with `--warmup-runs 1 --measured-runs 1`

- [ ] **Step 1: Write the failing command-contract test**

Change the existing assertion to:

```python
assert command[command.index("--warmup-runs") + 1] == "1"
assert command[command.index("--measured-runs") + 1] == "1"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run --offline --python 3.11 --with pytest==8.4.2 \
  pytest -q tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k build_worker_command
```

Expected: failure showing the command still contains `"0"` warmup runs.

- [ ] **Step 3: Make the minimal command change**

In `build_worker_command`, change:

```python
"--warmup-runs",
"0",
```

to:

```python
"--warmup-runs",
"1",
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command. Expected: pass.

### Task 2: Bind Warmup and Measured Graph State into the Payload

**Files:**
- Modify: `tools/autoregressive_draft_cuda_graph_gate.py`
- Modify: `tools/autoregressive_draft_cuda_graph_contract.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: one worker `warmup_runs` row and one `measured_runs` row.
- Produces: mode rows with `warmup_rank_graph_counters`, `rank_graph_counters`, `warmup_rank_graph_resources`, and `rank_graph_resources`.

- [ ] **Step 1: Write failing conversion and schema tests**

Extend worker fixtures with one warmup run and assert:

```python
assert row["warmup_rank_graph_counters"] == warmup[
    "correctness"
]["rank_graph_counters"]
assert row["warmup_rank_graph_resources"] == warmup[
    "correctness"
]["rank_graph_resources"]
```

Add schema mutations proving graph rows are rejected when:

```python
measured_counters[0]["captures"] += 1
measured_counters[0]["replays"] = warmup_counters[0]["replays"]
measured_resources[0]["total_capture_ns"] += 1
measured_resources[0]["reserved_bytes"] += 1
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run --offline --python 3.11 --with pytest==8.4.2 \
  pytest -q tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'converts_worker_evidence or steady_state'
```

Expected: missing warmup fields and no steady-state rejection.

- [ ] **Step 3: Convert exactly one warmup and measured run**

Update `mode_row_from_worker` to require:

```python
warmups = worker.get("warmup_runs")
measured = worker.get("measured_runs")
if not isinstance(warmups, list) or len(warmups) != 1:
    raise ValueError("worker must contain one in-process warmup run")
if not isinstance(measured, list) or len(measured) != 1:
    raise ValueError("worker must contain one measured run")
```

Normalize the warmup correctness graph counters/resources with the same rank
ordering used by the measured row and include both snapshots in the returned
mode row.

- [ ] **Step 4: Validate steady-state transitions in the canonical contract**

Add rank resource normalization for:

```python
("ready_entry_count", "static_bytes", "reserved_bytes", "total_capture_ns")
```

For eager mode, require warmup and measured graph counters/resources to remain
zero. For graph mode, require per rank:

```python
measured["capture_attempts"] == warmup["capture_attempts"]
measured["captures"] == warmup["captures"] == 1
measured["replays"] > warmup["replays"] >= 1
measured["quarantines"] == warmup["quarantines"] == 0
measured["fallback_pre_replay"] == warmup["fallback_pre_replay"] == 0
measured_resource == warmup_resource
measured_resource["ready_entry_count"] == 1
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: pass.

### Task 3: Update Verification Evidence and Documentation

**Files:**
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`
- Modify: `docs/superpowers/audits/2026-08-17-autoregressive-draft-cuda-graph-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: canonical gate schema with warmup/measured state snapshots.
- Produces: local verification evidence and an honest remote rerun boundary.

- [ ] **Step 1: Run the complete focused local suite**

Run:

```bash
uv run --offline --python 3.11 \
  --with pytest==8.4.2 \
  --with torch==2.7.1 \
  --with transformers==4.57.6 \
  --with numpy \
  pytest -q \
    tools/test_autoregressive_draft_cuda_graph_gate.py \
    tools/test_autoregressive_draft_performance_gate.py \
    tools/test_autoregressive_draft_graph.py \
    tools/test_qwen3_draft_cuda_graph_backend.py
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and whitespace verification**

Run:

```bash
uv run --offline --python 3.11 python -m compileall -q \
  tools/autoregressive_draft_cuda_graph_gate.py \
  tools/autoregressive_draft_cuda_graph_contract.py \
  tools/verify_autoregressive_draft_cuda_graph_gate.py
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 3: Record the corrected evidence boundary**

Append to the completion audit and handoff:

```text
The prior 4-of-8 partial run is cold-start-contaminated because each fresh
graph worker used warmup_runs=0. It remains valid correctness/lifecycle
evidence but is not valid steady-state performance evidence. The corrected
schema requires one same-engine warmup and proves no capture/resource growth
during the measured run. A fresh remote 2-warmup/8-measured-pair run is still
required before GO or NO_GO_PERFORMANCE.
```

- [ ] **Step 4: Commit and push the gate correction**

Stage only the plan, spec, gate, contract, tests, audit, and handoff:

```bash
git add \
  docs/superpowers/specs/2026-08-17-autoregressive-draft-exact-cuda-graph-design.md \
  docs/superpowers/plans/2026-08-17-autoregressive-draft-steady-state-gate.md \
  tools/autoregressive_draft_cuda_graph_gate.py \
  tools/autoregressive_draft_cuda_graph_contract.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  docs/superpowers/audits/2026-08-17-autoregressive-draft-cuda-graph-completion-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "fix(cuda-graph): isolate steady-state replay timing" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

Expected: local and remote branch heads match.

### Task 4: Rerun the Source-Bound Remote Gate

**Files:**
- Create: `artifacts/autoregressive_draft_cuda_graph/<new-run-tag>/`
- Modify after evidence exists: `docs/superpowers/audits/2026-08-17-autoregressive-draft-cuda-graph-completion-audit.md`
- Modify after evidence exists: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: four clean GPUs, corrected source-bound gate, Qwen3-8B target, Qwen3-0.6B draft.
- Produces: two warmup pairs, eight measured balanced pairs, dual verifier receipts, and final classification.

- [ ] **Step 1: Require four clean GPUs without terminating external work**

Run the remote preflight. Proceed only when four selected GPUs each have at
most 1024 MiB used, at most 5% utilization, and zero compute processes.

- [ ] **Step 2: Run a new source-bound gate tag in the foreground**

Use `tools/run_autoregressive_draft_cuda_graph_gate_remote.py` with a new
2026-08-17 run tag. Do not reuse or overwrite the interrupted partial tag.

- [ ] **Step 3: Verify the bundle and classify**

Require:

```text
exact eager/graph token, accepted-prefix, transaction, and acceptance parity
captures unchanged between warmup and measured snapshots
replays increased on every rank during measured snapshots
zero fallback/quarantine and zero active transactions
two warmup pairs and eight measured balanced pairs
valid source hashes and both verifier receipts
```

Only then classify `GO`, `NO_GO_CORRECTNESS`, `NO_GO_PERFORMANCE`, or
`INCONCLUSIVE_ENVIRONMENT`.
