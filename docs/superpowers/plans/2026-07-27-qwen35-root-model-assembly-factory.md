# Qwen3.5 Root Model Assembly Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble a complete Qwen3.5 root model from config topology, one existing state pool, and injected layer components.

**Architecture:** Validate config/pool layer topology, build one adapter per linear layer, invoke a strict decoder-layer callback in exact layer order, then construct transaction, packed stack, and root model with identity-coherent ownership.

**Tech Stack:** Python 3.9, PyTorch CPU, Qwen3.5 config adapter, hybrid state pool, transactional root shell.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- CPU/static only; no CUDA, NCCL, checkpoint, local GPU, or remote GPU work.
- Never allocate a second state pool.
- Do not change production ModelRunner selection or Engine/Scheduler wiring.
- Preserve schema-v2 canonical `NO_GO`.
- Do not claim performance/cache/memory/compression/quality benefit.

---

### Task 1: RED Config/Pool Topology

**Files:**
- Create: `tools/test_qwen35_root_model_assembly_factory.py`
- Create after RED: `tinyvllm/models/qwen35_factory.py`

- [x] Build a mixed config and matching real pool.
- [x] Assert callback layer order and exact adapter placement.
- [x] Assert exact root/stack/transaction/pool graph and storage identity.
- [x] Reject pool/config missing, extra, and full-layer components.
- [x] Run focused tests and confirm RED because the factory module is absent.
- [x] Implement topology validation and strict assembly.
- [x] Run focused tests and confirm GREEN.

### Task 2: RED Failure Atomicity and Integration

**Files:**
- Modify: `tools/test_qwen35_root_model_assembly_factory.py`

- [x] Reject wrong callback return type and mismatched block type.
- [x] Inject callback failure and prove pool values/ownership unchanged.
- [x] Pass assembled root through `build_qwen35_hybrid_model_owner`.
- [x] Execute one transactional root step with assembled model.
- [x] Run focused tests under Python 3.9 and 3.12.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run root, owner, packed, state, restore, and Qwen3.5 regressions.
- [x] Run 97/1/0 chunked-prefill matrix.
- [x] Run Python 3.9/3.12 compile and `git diff --check`.
- [x] Confirm staged files empty, evidence present, Qwen3 production selection unchanged, and no Engine/Scheduler assembly calls.
- [x] Audit requirements, update handoff, and mark verified checkboxes.

## Completion Audit

Fresh focused results:

```text
Python 3.9 and Python 3.12:
qwen35 root model assembly factory tests passed (5 tests)
```

The tests cover exact callback order, adapter placement, root/stack/
transaction/pool identity, storage reuse, `text_config`, exact pool type,
missing state roles, extra full-layer state, wrong callback output, callback
failure atomicity, owner integration, and one transactional root run.

Compatibility evidence:

```text
CHUNKED_PREFILL_MATRIX passed=97 skipped=1 failed=0 total=98
Python 3.9/3.12 py_compile passed for 62 changed/untracked Python files
git diff --check passed
staged files: 0
Engine assembly references: 0
experiments/qwen35_hybrid_state: present
```

Concrete TP/CUDA components, checkpoint loading, production model selection,
startup binding, and all performance/cache/memory/quality claims remain
blocked.

