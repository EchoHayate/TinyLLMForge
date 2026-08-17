# ModelRunner Live Acknowledgement Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire per-worker command acknowledgements into the real LLMEngine/ModelRunner Python process lifecycle while preserving existing fire-and-forget calls.

**Architecture:** LLMEngine owns one unidirectional pipe receive endpoint per worker and a collector. ModelRunner shared-memory broadcasts carry immutable command envelopes; workers execute through the acknowledgement executor, while the existing `call()` remains fire-and-forget and a new Engine API performs acknowledged dispatch plus local execution plus worker collection.

**Tech Stack:** Python 3.9, multiprocessing Pipe/Process/Event, shared memory, existing command acknowledgement module, dependency-light AST/method tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- CPU/static tests only; do not start CUDA, NCCL, checkpoint, or remote work.
- Preserve existing `model_runner.call()` return and exception semantics.
- Do not route ordinary `run` through acknowledged collection in this phase.
- Do not enable scheduler hybrid-prefix admission or change `LLMEngine.step()`.
- Preserve Qwen3/Qwen3.5 math, attention, RoPE, RMSNorm, and checkpoint semantics.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Poison acknowledged runtime state on any uncertain outcome.
- Do not claim latency, throughput, cache, compression, memory, or quality improvement.

---

### Task 1: RED ModelRunner Envelope Wiring

**Files:**
- Create: `tools/test_model_runner_live_ack_wiring.py`
- Modify after RED: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Produces: `ModelRunner.dispatch_command()`.
- Modifies: `ModelRunner.call()`.
- Modifies: `ModelRunner.write_shm()`.
- Modifies: `ModelRunner.read_shm()`.
- Modifies: `ModelRunner.loop()`.

- [x] Add fire-and-forget `call()` compatibility test.
- [x] Add acknowledged dispatch and monotonic ID test.
- [x] Add envelope and legacy shared-memory decode tests.
- [x] Add worker-loop ack/no-ack/exit tests.
- [x] Add constructor source-contract tests for `ack_sender`.
- [x] Run focused script and confirm RED because wiring interfaces are absent.
- [x] Import acknowledgement protocol and initialize command IDs/sender.
- [x] Implement envelope serialization and compatibility decoding.
- [x] Implement worker executor loop and dispatch API.
- [x] Preserve existing `call()` behavior.
- [x] Run ModelRunner-focused tests and confirm GREEN.

### Task 2: RED LLMEngine Pipe Lifecycle and Acknowledged Call

**Files:**
- Modify: `tools/test_model_runner_live_ack_wiring.py`
- Modify after RED: `tinyvllm/engine/llm_engine.py`
- Modify if needed: `tinyvllm/engine/model_runner_command_ack.py`

**Interfaces:**
- Produces: `LLMEngine.call_model_runner_acknowledged()`.
- Produces: rank-to-process liveness mapping.
- Modifies: worker process/channel construction and exit cleanup.
- Produces: public `ModelRunnerCommandAckCollector.poison()`.

- [x] Add TP=1 local-only acknowledged call test.
- [x] Add TP>1 local-result plus ordered-worker-ack test.
- [x] Add local-exception poison and collector-failure propagation tests.
- [x] Add rank-to-process liveness mapping test.
- [x] Add pipe endpoint creation/ownership/parent-close test.
- [x] Add exit endpoint cleanup test.
- [x] Run Engine-focused tests and confirm RED.
- [x] Implement public collector poison.
- [x] Create one pipe per worker and pass one sender to ModelRunner.
- [x] Create collector from rank-indexed receivers.
- [x] Implement acknowledged Engine call and liveness callback.
- [x] Close parent endpoint duplicates and Engine-owned receivers.
- [x] Run focused wiring tests and confirm GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run live wiring and command-ack tests under Python 3.9 and 3.12.
- [x] Run chunked-prefill function matrix with only the known Config AST skip.
- [x] Run restore-ticket, acquisition/cache, hybrid-state, packed stack, and
  ModelRunner dependency-light regressions.
- [x] Run Python 3.9 and Python 3.12 `py_compile`.
- [x] Run `git diff --check`.
- [x] Confirm staged files are empty and experiment evidence remains present.
- [x] Build a prompt-to-artifact checklist covering endpoint ownership,
  envelope compatibility, fire-and-forget preservation, local plus worker
  results, liveness, poison, cleanup, CPU/static scope, and no performance
  overclaim.
- [x] Update handoff with fresh evidence and the restore participant-method
  gate.
- [x] Mark checkboxes complete only from fresh evidence.
