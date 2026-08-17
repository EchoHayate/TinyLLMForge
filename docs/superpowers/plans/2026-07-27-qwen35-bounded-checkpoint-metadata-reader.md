# Qwen3.5 Bounded Checkpoint Metadata Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and parse config, index, and safetensors JSON headers with strict byte budgets and zero tensor-payload reads.

**Architecture:** Validate retained manifest identities first, read bounded config/index bytes, and manually parse only each safetensors 8-byte header prefix plus JSON header. Return a frozen metadata bundle with exact byte accounting for later worker loader construction.

**Tech Stack:** Python pathlib, hashlib, json, MappingProxyType, SimpleNamespace.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Never use safetensors APIs, torch checkpoint APIs, mmap, CUDA, SSH, subprocess, or Engine code.
- Never read a safetensors tensor data-section byte.
- Preserve the existing canonical config/index/shard manifest composite identity.
- Do not modify or execute the real checkpoint worker in Task 1.
- Do not stage, commit, merge, or clean experiment evidence.

---

### Task 1: Bounded Metadata and Header Reader

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_metadata.py`
- Create: `tools/test_qwen35_checkpoint_metadata.py`

- [x] Write RED tests for exact identity, header-only byte accounting, namespace conversion, multi-shard coverage, and fail-closed malformed inputs.
- [x] Run the focused test and confirm the module is missing.
- [x] Implement the minimal bounded reader.
- [x] Run the focused test and confirm GREEN.

### Task 2: Worker Metadata-to-Loader Composition

**Files:**
- Modify: `tools/qwen35_real_checkpoint_load_worker.py`
- Modify: `tools/test_qwen35_real_checkpoint_load_worker.py`

- [x] Add RED tests for a worker helper that consumes an exact metadata bundle and forwards it to the existing construction function.
- [x] Implement the import-only helper without changing `main()`.
- [x] Run focused worker and metadata tests.

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

- [x] Run metadata, worker, configuration/adapter/factory, authorization, and safety-gate regressions.
- [x] Run compile, diff, staged, forbidden API, payload-byte, Engine, `step()`, and execution-rejection audits.
- [x] Check plan items and record exact evidence plus remaining full-shard-hash and execution gaps.

## Completion Evidence

Fresh focused TDD evidence with `/opt/homebrew/bin/python3.12`:

```text
metadata reader:
  passed (4 tests)
worker construction/helper:
  passed (6 tests)
```

The worker helper requires an exact `Qwen35CheckpointMetadataBundle`, rejects
any bundle reporting nonzero payload bytes before construction, and forwards
the parsed config/index/headers plus their verified identities to the existing
rank-loader construction function. `main()` still hard-rejects execution.

Fresh adjacent regression evidence:

```text
manifest-bound loader configuration:
  passed (4 tests)
prepared-target loader adapter:
  passed (5 tests)
prepared-target factory:
  passed (6 tests)
real 24-layer metadata binding:
  passed (1 test)
streamed fresh checkpoint:
  passed (4 tests)
tiled checkpoint loading:
  passed (5 tests)
policy tiled checkpoint loading:
  passed (3 tests)
ModelRunner authorized checkpoint loader:
  passed (7 tests)
ModelRunner candidate publication:
  passed (4 tests)
Engine all-rank candidate binding:
  passed (9 tests)
hybrid model publication:
  passed (2 tests)
native model owner binding:
  passed (13 tests)
real checkpoint authorization:
  passed
real checkpoint safety gate:
  passed (23 tests)
focused py_compile:
  passed
forbidden import/call scan:
  passed
payload-read invariant scan:
  passed
Engine/step isolation:
  passed
direct worker execution rejection:
  passed
git diff --check:
  passed
staged files:
  0
```

Claim boundary:

```text
bounded config/index/header parsing: proven dependency-light
actual config/index SHA256 verification: proven
retained canonical manifest composite verification: proven
exact shard-size verification: proven
safetensors tensor-payload bytes read: zero in the reader contract/tests
metadata-to-loader construction composition: proven dependency-light
full shard SHA256 recomputation: absent
real checkpoint tensor payload loading: not executed
worker execution: unauthorized and hard-rejected
automatic Engine enablement: absent
production CUDA/cache/speed benefit: unmeasured
```
