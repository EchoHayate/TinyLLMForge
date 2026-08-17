# Autoregressive Draft Controlled Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a source-bound TP4 controlled-performance pilot for the Qwen3 target plus independent Qwen3 draft runtime.

**Architecture:** Reuse the repository's generic synchronized timing aggregation while adding learned-drafter-specific worker, parent gate, verifier, and bounded remote runner. Correctness producers remain unchanged and the artifact always classifies itself as `PILOT_ONLY`.

**Tech Stack:** Python 3.11, pytest, TinyLLMForge Engine API, PyTorch CUDA memory snapshots, Bash, SSH.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify existing TP1/TP4 correctness producers or verifiers.
- Use `sitian@10.232.195.203` and GPUs `3,4,6,7`.
- Keep `max_proposal_tokens=4`, greedy temperature zero, and exact output parity.
- Never synthesize KV movement or memory evidence.
- Do not stage, commit, push, stash, reset, clean, switch branches, or terminate unrelated GPU processes.
- Initial pilot is 256 prompt tokens, 16 output tokens, batch 1/4, one warmup, and three measured runs.

---

### Task 1: Freeze Learned-Drafter Performance Contracts

**Files:**
- Create: `tools/test_autoregressive_draft_performance_gate.py`
- Create: `tools/autoregressive_draft_performance_gate.py`

**Interfaces:**
- Consumes: `build_run_metrics()` and `aggregate_measurements()` from `tools/speculative_runtime_performance_gate.py`.
- Produces: `validate_worker_result()`, `build_performance_artifact()`, and `validate_performance_artifact()`.

- [ ] Write failing tests for four-cell inventory, three measured runs, exact repeat parity, learned acceptance evidence, raw timing distributions, distributed memory rows, Proposal-KV counter rows, `PILOT_ONLY`, and aggregate tamper rejection.
- [ ] Run `/usr/bin/python3 -m pytest -q tools/test_autoregressive_draft_performance_gate.py` and confirm the module import or contract tests fail for missing implementation.
- [ ] Implement the minimal schema, validation, aggregate recomputation, source hashing, and atomic JSON writer.
- [ ] Re-run the focused test and confirm GREEN.

### Task 2: Implement the Isolated Worker

**Files:**
- Create: `tools/autoregressive_draft_performance_worker.py`
- Modify: `tools/test_autoregressive_draft_performance_gate.py`

**Interfaces:**
- Consumes: `_TinyVLLMTP4EngineAdapter`, deterministic prompt rows, `build_run_metrics()`, and Engine memory/authority snapshot APIs.
- Produces: one worker JSON for `(policy, batch_size)`.

- [ ] Write failing fake-engine tests proving one warmup plus exactly three measured runs, step-end token timestamps, exact output counts, peak-memory reset/read ordering, learned acceptance collection, Proposal-KV counter deltas, and guaranteed Engine exit.
- [ ] Run the worker-focused tests and confirm RED for missing worker functions.
- [ ] Implement `run_request_batch()` and `run_policy_campaign()` with no real-GPU dependency in unit tests.
- [ ] Re-run worker and contract tests and confirm GREEN.

### Task 3: Add Parent Gate and Independent Verifier

**Files:**
- Modify: `tools/autoregressive_draft_performance_gate.py`
- Create: `tools/verify_autoregressive_draft_performance_gate.py`
- Modify: `tools/test_autoregressive_draft_performance_gate.py`

**Interfaces:**
- Consumes: four worker JSON files and the frozen source inventory.
- Produces: `result.json`, remote verifier receipt, and local verifier receipt.

- [ ] Write failing tests for isolated subprocess launch, worker failure propagation, exact target/learned repeat parity, aggregate recomputation, source drift rejection, and verifier receipt fields.
- [ ] Run the focused tests and confirm RED.
- [ ] Implement the four-cell launcher and verifier.
- [ ] Re-run all focused tests and `py_compile` the three Python tools.

### Task 4: Add the Bounded Remote Runner

**Files:**
- Create: `tools/run_autoregressive_draft_performance_gate_remote.sh`
- Modify: `tools/test_autoregressive_draft_performance_gate.py`

**Interfaces:**
- Consumes: passing local contracts, target/draft checkpoint paths, GPUs 3/4/6/7, and the remote py311 environment.
- Produces: a local checksum-covered performance bundle.

- [ ] Write a failing source-contract test for the exact remote host, GPU list, fixed ports, hard timeout, source archive, GPU before/after snapshots, and remote plus local verifier calls.
- [ ] Run the focused test and confirm RED.
- [ ] Implement the runner without ControlMaster dependence and without cleanup of unrelated processes.
- [ ] Run `bash -n tools/run_autoregressive_draft_performance_gate_remote.sh` and the focused tests.

### Task 5: Execute and Classify the 256/16 Pilot

**Files:**
- Create: `experiments/autoregressive_draft/<run-tag>/`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`

**Interfaces:**
- Consumes: the source-bound remote runner and passing verifier.
- Produces: raw worker rows, result, receipts, logs, GPU snapshots, source archives, summary, README, and checksum manifest.

- [ ] Run local focused tests and remote dependency-light tests.
- [ ] Verify GPU 3/4/6/7 inventory and free ports without terminating any process.
- [ ] Run the remote pilot and keep polling through completion.
- [ ] Run the independent verifier remotely and locally.
- [ ] Generate and check `manifest.sha256`.
- [ ] Classify direction from raw medians while retaining `PILOT_ONLY`.
- [ ] Record what the pilot proves, what it does not prove, and the next 4K performance step in handoff and audit.
