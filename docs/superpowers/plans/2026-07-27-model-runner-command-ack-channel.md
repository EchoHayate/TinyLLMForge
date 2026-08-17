# ModelRunner Command Acknowledgement Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stdlib-only, fail-closed per-worker command acknowledgement protocol with explicit error, timeout, stale-message, and worker-death handling.

**Architecture:** Preserve the existing shared-memory command broadcast and model the reply path as one unidirectional multiprocessing Pipe per worker. A worker executor emits one rank-tagged acknowledgement; a rank-0 collector waits against one absolute deadline and permanently poisons itself on any uncertain outcome.

**Tech Stack:** Python 3.9, dataclasses, pickle, multiprocessing Pipe/Process, monotonic clocks, dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- CPU-only; do not start local or remote GPU/checkpoint work.
- Do not modify `LLMEngine`, `ModelRunner`, scheduler admission, CUDA, or NCCL in this phase.
- Do not change Qwen3/Qwen3.5 math, attention, RoPE, RMSNorm, or checkpoint semantics.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Use one absolute timeout deadline; never retry a stateful command.
- Poison the collector on any uncertain worker outcome.
- Do not claim latency, throughput, cache, compression, memory, or quality improvement.

---

### Task 1: RED Envelope and Worker Executor

**Files:**
- Create: `tools/test_model_runner_command_ack.py`
- Create after RED: `tinyvllm/engine/model_runner_command_ack.py`

**Interfaces:**
- Produces: `ModelRunnerCommandEnvelope`.
- Produces: `ModelRunnerCommandAck`.
- Produces: `execute_acknowledged_command()`.

- [x] Add validation and pickle round-trip tests.
- [x] Add acknowledged success and bounded exception tests.
- [x] Add fire-and-forget return, exception, and no-ack tests.
- [x] Add acknowledgement send failure test.
- [x] Run focused script and confirm RED because the module does not exist.
- [x] Implement immutable envelope and acknowledgement values.
- [x] Implement acknowledged and fire-and-forget execution semantics.
- [x] Run executor-focused tests and confirm GREEN.

### Task 2: RED Rank-0 Collector

**Files:**
- Modify: `tools/test_model_runner_command_ack.py`
- Modify after RED: `tinyvllm/engine/model_runner_command_ack.py`

**Interfaces:**
- Produces: `ModelRunnerCommandAckCollector`.
- Produces: `ModelRunnerCommandAckCollector.collect()`.
- Produces: `ModelRunnerCommandAckCollector.poisoned`.

- [x] Add two spawned-worker out-of-order success test.
- [x] Add outer worker-error acknowledgement and inner restore-status test.
- [x] Add one-absolute-deadline timeout and dead-worker tests.
- [x] Add stale/future command, wrong-rank, duplicate, malformed, closed-pipe,
  and receive-failure tests.
- [x] Add constructor/input validation and reuse-after-poison tests.
- [x] Run collector-focused tests and confirm RED because collector interfaces
  do not exist.
- [x] Implement receiver/rank validation.
- [x] Implement polling, ordered collection, deadline, and liveness handling.
- [x] Implement fail-closed poison semantics and bounded diagnostics.
- [x] Run focused script and confirm GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run the focused acknowledgement script under Python 3.9 and Python 3.12.
- [x] Run restore-ticket, acquisition/cache, cross-layer transaction,
  hybrid-state, packed-layer-stack, and ModelRunner dependency-light
  regressions.
- [x] Run Python 3.9 and Python 3.12 `py_compile`.
- [x] Run `git diff --check`.
- [x] Confirm `git diff --cached --name-only` is empty.
- [x] Confirm no `experiments/` evidence was removed.
- [x] Build a prompt-to-artifact checklist covering per-rank identity,
  explicit success/error, absolute timeout, death detection, stale protocol
  rejection, poison, pickle transport, CPU-only scope, and no performance
  overclaim.
- [x] Update `AGENT_HANDOFF_STATE.md` with fresh evidence and the live
  `LLMEngine`/`ModelRunner` wiring gate.
- [x] Mark checkboxes complete only from fresh evidence.
