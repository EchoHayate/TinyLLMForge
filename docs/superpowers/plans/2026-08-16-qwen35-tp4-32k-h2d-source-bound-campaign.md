# Qwen3.5 TP4/32K Focused-H2D Source-Bound Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan inline. Do not use
> subagents.

**Goal:** Add a local-only deterministic source bundle, inert campaign plan,
and single-use authorization contract for the focused-H2D four-cell campaign.

**Architecture:** Package a conservative explicit producer/verifier closure,
bind it into a canonical immutable campaign plan, and bind a single-use
authorization record to every execution-relevant identity. Add one local
authorization-first callback boundary, but no built-in remote transport.

**Tech Stack:** Python 3, pathlib, hashlib, json, tarfile, tempfile, pytest.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, push, stash, reset, clean, switch branch/worktree, or
  use subagents.
- Do not connect to a remote host or run GPU, CUDA, NCCL, or authority work.
- Do not create built-in SSH, subprocess, GPU, CUDA, or NCCL transport.
- Freeze `sitian@10.232.195.203` and the approved `/dev/shm/sitian` root.
- Treat `/data00` paths as read-only environment/model inputs only.
- Require a second explicit authorization before any real execution.

### Task 1: Source Bundle

**Files:**
- Create: `tools/test_qwen35_tp4_32k_h2d_slot_reuse_campaign.py`
- Create: `tools/qwen35_tp4_32k_h2d_slot_reuse_source_bundle.py`

- [x] Write tests for exact required roots, deterministic inventory/tree/tar
  identities, dynamic dependency inclusion, tar membership, and symlink/path
  rejection.
- [x] Run the focused tests and verify RED because the module is absent.
- [x] Implement the minimal source bundle builder and validator.
- [x] Run the focused tests and verify GREEN.

### Task 2: Inert Campaign Plan

**Files:**
- Modify: `tools/test_qwen35_tp4_32k_h2d_slot_reuse_campaign.py`
- Create: `tools/qwen35_tp4_32k_h2d_slot_reuse_campaign.py`

- [x] Write tests freezing the host/root/cells/repetitions/TP4 GPUs/ports,
  binding all source/checkpoint identities, and rejecting tampering.
- [x] Verify RED because the campaign module is absent.
- [x] Implement canonical plan creation and validation with inert command
  descriptors and all execution flags false.
- [x] Verify GREEN and statically reject execution-capable imports/calls.
- [x] Add tested `prepare` and `validate` CLI subcommands that create no
  authorization and invoke no transport.

### Task 3: Single-Use Authorization

**Files:**
- Modify: `tools/test_qwen35_tp4_32k_h2d_slot_reuse_campaign.py`
- Create:
  `tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_authorization.py`

- [x] Write tests for exact authorization text, plan/source/checkpoint/cell/
  repetition/GPU/port/path binding, unsafe nonce rejection, tamper rejection,
  and rename-first single-use consumption.
- [x] Verify RED because the authorization module is absent.
- [x] Implement atomic production, validation, and consumption.
- [x] Verify GREEN.

### Task 4: Audit and Handoff

**Files:**
- Modify: `tools/test_qwen35_tp4_32k_h2d_slot_reuse_campaign.py`
- Create:
  `tools/qwen35_tp4_32k_h2d_slot_reuse_campaign_executor.py`

- [x] Write tests proving invalid authorization invokes no callback, valid
  authorization is consumed before callback entry, and callback failure
  cannot restore/replay authorization.
- [x] Verify RED because the executor module is absent.
- [x] Implement the minimal injected command-runner boundary with no built-in
  transport.
- [x] Verify GREEN.

### Task 5: Audit and Handoff

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] Run focused pytest, py_compile, static no-transport checks, tar closure
  checks, and `git diff --check`.
- [x] Map every approved local requirement to concrete test or source evidence.
- [x] Update readiness classifications while preserving
  `FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED`,
  `PHASE_1=NOT_ACHIEVED`, and `PROMOTION=NOT_PROMOTABLE`.
- [x] Stop without remote/GPU execution.
