# Qwen3.5 All-Rank Loader Construction Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct TP=1/rank0 and TP=2/rank0-1 manifest-bound loaders from the approved real metadata while invoking zero providers, zero loaders, zero payload reads, and zero CUDA initialization.

**Architecture:** Add a standalone record validator, exact source-closure resolver, remote construction-only worker, deterministic SSH staging, and two-artifact local publisher. The worker uses namespace packages to bypass `tinyvllm.__init__`, reads bounded metadata, constructs loader objects only, and enforces a 512 MiB total-process plus 256 MiB post-Torch-import construction VmHWM ceiling.

**Tech Stack:** Python standard library, AST import discovery, pathlib, hashlib, json, tarfile, subprocess for local SSH transport, remote PyTorch import only.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Use only `sitian@10.232.195.203` through `/tmp/ssh-sitian-10.232.195.203`.
- Read zero tensor-payload bytes and never recompute the full shard SHA256.
- Never call a constructed loader, pool provider, or attention-backend provider.
- Keep CUDA uninitialized and `CUDA_VISIBLE_DEVICES=""`.
- Do not invoke or enable the real checkpoint-load worker `main()`.
- Do not modify Engine, ModelRunner, Scheduler, publication, or schema-v2 evidence.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim performance, cache, memory, compression, or quality improvement.

---

### Task 1: Contract, Closure, and Local Worker TDD

**Files:**
- Create: `tools/qwen35_real_checkpoint_loader_construction_preflight.py`
- Create: `tools/test_qwen35_real_checkpoint_loader_construction_preflight.py`

- [x] Write RED tests for exact schema validation, frozen closure, provider guards, TP row coverage, CUDA state, and VmHWM ceiling.
- [x] Run RED and confirm the module/interfaces are absent.
- [x] Implement closure discovery, namespace-package loading, record validator, and construction-only worker.
- [x] Run focused local GREEN using the retained metadata snapshots and sparse shard fixture.

### Task 2: Deterministic Remote Orchestration

**Files:**
- Modify: `tools/qwen35_real_checkpoint_loader_construction_preflight.py`
- Modify: `tools/test_qwen35_real_checkpoint_loader_construction_preflight.py`

- [x] Add RED tests for deterministic closure tar, fixed SSH command, no-bytecode flags, unique paths, source round trip, and two-artifact publication.
- [x] Run RED.
- [x] Implement staging, remote source verification, worker invocation, round-trip validation, and atomic local publication.
- [x] Run focused GREEN, compile, forbidden-call scan, and worker-rejection audit.

### Task 3: Live Construction-Only Gate

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/loader_construction_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-07-27-qwen35-all-rank-loader-construction-preflight.md`

- [x] Verify the fixed ControlMaster and approved Python.
- [x] Run one unique source-bound construction-only gate.
- [x] Independently validate exact remote inventory, source hashes, TP rows, zero calls/bytes/CUDA, and VmHWM ceiling.
- [x] Run focused/adjacent regressions, compile, static scans, direct worker rejection, `git diff --check`, and staged-file check.
- [x] Check every plan item and append exact evidence, limitations, and the next safe gate.

## Live Result

Authoritative run:

```text
qwen35-loader-construction-preflight-final-20260727-193734
```

Local evidence:

```text
experiments/qwen35_hybrid_state/
  qwen35-loader-construction-preflight-final-20260727-193734/
    loader_construction_preflight.json
    source_manifest.json
```

Status:

```text
PASS
```

All-rank rows:

```text
TP=1 rank 0
TP=2 rank 0
TP=2 rank 1
```

Every row returned:

```text
loader:
  Qwen35ManifestBoundCheckpointCandidateLoader
configuration:
  Qwen35RankCheckpointLoaderConfiguration
plan loads: 320
plan skips: 312
plan payload bytes: 4548144832
```

Hard execution boundaries:

```text
metadata bytes read: 144024
payload bytes read: 0
payload hashes recomputed: false
provider events: []
loader calls: 0
pool creations: 0
backend creations: 0
CUDA initialized before: false
CUDA initialized after: false
```

Memory evidence:

```text
VmHWM before imports: 20560 KiB
VmHWM after Torch import: 365656 KiB
VmHWM after construction: 497460 KiB
total VmHWM increment: 476900 KiB <= 524288 KiB
post-Torch construction increment: 131804 KiB <= 262144 KiB
```

The first formal run:

```text
qwen35-loader-construction-preflight-20260727-193407
```

failed closed and published no local artifact because the original single
256 MiB ceiling measured from Python startup and therefore included the Torch
import. A same-source diagnostic recorded:

```text
VmHWM before imports: 19536 KiB
VmHWM after construction: 496352 KiB
single increment: 476816 KiB
```

The gate was corrected through RED tests to record three memory points and
enforce two separate ceilings: 512 MiB total process and 256 MiB
post-Torch-import construction. The authoritative run passed both.

Remote inventory was exactly 33 staged source files plus:

```text
loader_construction_preflight.json
source_manifest.json
```

All local, remote, record, and source-manifest SHA256 values matched.

Artifact SHA256 values:

```text
loader_construction_preflight.json:
  776ae443e3318bd6ff1d06edfdea17a3b00fb84595ceb21c895191b6b4d7e413
source_manifest.json:
  e1ee806ab5f9b37e96073f1bd50a4594b4084d91b7b523a194c6c078e9ea8d88
```

Exact claim boundary:

```text
approved metadata to TP=1/2 rank loader objects: proven remotely
zero provider/loader call during construction: proven
bounded host-memory construction: proven
real pool/model/attention-backend construction: absent
real tensor payload loading/assignment: not executed
CUDA/Engine/publication/inference: absent
production speed/cache/memory benefit: unmeasured
schema-v2 canonical NO_GO: unchanged
```
