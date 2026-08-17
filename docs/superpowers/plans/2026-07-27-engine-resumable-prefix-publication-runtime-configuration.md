# Engine Resumable Prefix Publication Runtime Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit, resumable `LLMEngine` configuration entry point that installs the complete Qwen3.5 hybrid-prefix publication runtime in a strict, conflict-safe order.

**Architecture:** Validate and preflight the complete aggregate configuration before mutation, then compose the existing exact-idempotent restore, identity, publication coordinator, and source-publisher stages. Persist aggregate completion only after every stage succeeds so exact retries can resume partial configuration without claiming premature completion.

**Tech Stack:** Python, `LLMEngine`, existing Qwen3.5 hybrid-prefix runtime components, dependency-light AST method extraction tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `LLMEngine.__init__()` and `LLMEngine.step()` disconnected.
- Preserve Qwen3.5 schema-v2 canonical `NO_GO`.
- Use exact byte-identical sharing only; make no lossy-cache change.
- Do not stage, commit, merge, or clean untracked experiment evidence.
- Do not claim production memory, compression, quality, or speed benefit.

---

### Task 1: Aggregate Contract and Strict Ordering

**Files:**
- Create: `tools/test_engine_resumable_prefix_publication_runtime_configuration.py`
- Modify: `tinyvllm/engine/llm_engine.py`

**Interfaces:**
- Consumes: existing restore, identity, publication-coordinator, and configured-publisher methods.
- Produces: `LLMEngine.configure_qwen35_hybrid_prefix_publication_runtime(*, model_fingerprint, max_entries, max_bytes, timeout_s)`.

- [x] **Step 1: Write failing validation and order tests**

Add dependency-light tests that load the new method from the `LLMEngine` AST,
construct a fake Engine with child-stage recorders, reject each invalid
argument before recorder mutation, and assert the successful call sequence is:

```text
restore -> identity -> coordinator install -> publisher install
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_engine_resumable_prefix_publication_runtime_configuration.py
```

Expected: failure because
`configure_qwen35_hybrid_prefix_publication_runtime` does not exist.

- [x] **Step 3: Implement minimal aggregate configuration**

Add the explicit method to `LLMEngine`. Validate the complete request, preflight
existing aggregate and child configuration slots, invoke child stages in strict
order, and store aggregate configuration plus the returned publisher only
after all stages succeed.

- [x] **Step 4: Run the focused test and verify GREEN**

Run the focused test command and expect all initial tests to pass.

### Task 2: Exact Retry and Conflict Closure

**Files:**
- Modify: `tools/test_engine_resumable_prefix_publication_runtime_configuration.py`
- Modify: `tinyvllm/engine/llm_engine.py`

**Interfaces:**
- Consumes: aggregate method from Task 1.
- Produces: exact completed idempotency and resumability after later-stage failure.

- [x] **Step 1: Write failing retry tests**

Add tests for:

```text
coordinator installation fails once -> exact retry succeeds
publisher installation fails once -> exact retry succeeds
completed exact repeat -> no child method is called again
partial-state conflict -> no child method is called
completed aggregate conflict -> no child method is called
```

- [x] **Step 2: Run the focused test and verify RED**

Expected: at least one retry or preflight assertion fails against the Task 1
minimal implementation.

- [x] **Step 3: Implement minimal resumability and preflight checks**

Reuse an exact installed publication coordinator, reject mismatched child
configuration before child calls, and return the stable completed publisher for
an exact aggregate repeat.

- [x] **Step 4: Run the focused test and verify GREEN**

Run the focused test and expect all tests to pass.

### Task 3: Disconnection, Regression, and Handoff

**Files:**
- Modify: `tools/test_engine_resumable_prefix_publication_runtime_configuration.py`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

**Interfaces:**
- Consumes: completed explicit orchestrator.
- Produces: regression evidence and precise claim boundary.

- [x] **Step 1: Add automatic-wiring audit**

Parse `LLMEngine.step()` and assert it contains no reference to
`configure_qwen35_hybrid_prefix_publication_runtime` or aggregate publication
runtime configuration state.

- [x] **Step 2: Run focused and adjacent regression suites**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_engine_resumable_prefix_publication_runtime_configuration.py
/opt/homebrew/bin/python3.12 tools/test_engine_source_publisher_hook_installation.py
/opt/homebrew/bin/python3.12 tools/test_explicit_prefill_publication_integration.py
/opt/homebrew/bin/python3.12 tools/test_qwen35_runtime_prefix_identity_binding.py
/opt/homebrew/bin/python3.12 tools/test_engine_live_hybrid_prefix_restore_transaction.py
/opt/homebrew/bin/python3.12 tools/test_engine_hybrid_prefix_publication_transaction.py
```

Expected: all pass.

- [x] **Step 3: Run static verification**

Run focused `py_compile`, `git diff --check`, staged-file audit, and AST/grep
audits proving `LLMEngine.step()` remains disconnected.

- [x] **Step 4: Record evidence**

Check every completed plan step and append exact test counts, retry semantics,
default-off boundary, and unproven production-performance claims to
`AGENT_HANDOFF_STATE.md`.
