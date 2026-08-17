# Qwen3.5 Real Checkpoint Metadata-Header Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one source-bound remote preflight that parses the approved Qwen3.5 config/index/safetensors JSON header, builds the checkpoint tensor plan, and reads zero tensor-payload bytes.

**Architecture:** Add one standalone dependency-light module containing the record validator, direct-file remote worker, deterministic source staging, SSH orchestration, and local artifact validation. Keep this evidence namespace separate from the existing stat-only preflight and never invoke the real checkpoint-load worker.

**Tech Stack:** Python standard library, pathlib, hashlib, json, tarfile, importlib, subprocess for local SSH transport.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Use only `sitian@10.232.195.203` through `/tmp/ssh-sitian-10.232.195.203`.
- Read only safetensors prefix/header metadata and zero tensor-payload bytes.
- Do not recompute the full shard SHA256.
- Do not invoke or enable the real checkpoint-load worker.
- Do not modify Engine, ModelRunner, Scheduler, publication, or schema-v2 canonical evidence.
- Do not stage, commit, merge, overwrite, or delete experiment evidence.
- Do not claim speed, memory, cache, compression, or quality improvement.

---

### Task 1: Record Contract and Local Snapshot Worker

**Files:**
- Create: `tools/qwen35_real_checkpoint_metadata_preflight.py`
- Create: `tools/test_qwen35_real_checkpoint_metadata_preflight.py`

**Interfaces:**
- Consumes:
  `read_qwen35_checkpoint_metadata(...)`,
  `build_qwen35_checkpoint_tensor_plan(...)`.
- Produces:
  `validate_metadata_preflight(record)`,
  `run_metadata_worker(...)`.

- [x] Write RED tests for exact PASS validation, identity/count/zero-payload mutations, direct-file imports, and the three retained `/tmp` metadata snapshots.
- [x] Run the focused test and confirm the module/interfaces are absent.
- [x] Implement the exact schema constants, validator, direct-file imports, and metadata worker.
- [x] Run the focused test and require local snapshot PASS with zero payload bytes.

### Task 2: Deterministic Source Staging and SSH Orchestration

**Files:**
- Modify: `tools/qwen35_real_checkpoint_metadata_preflight.py`
- Modify: `tools/test_qwen35_real_checkpoint_metadata_preflight.py`

**Interfaces:**
- Produces:
  `build_source_tar(repo_root)`,
  `stage_source(repo_root, run_tag, command_runner=...)`,
  `run_remote_metadata_preflight(...)`.

- [x] Add RED tests for deterministic tar metadata, fixed SSH identity, unique remote path, source hash equality, exact worker command, and two-artifact persistence.
- [x] Run RED and confirm missing orchestration behavior.
- [x] Implement staging, remote hash verification, worker invocation, artifact round-trip validation, and atomic local publication.
- [x] Run focused GREEN and `py_compile`.

### Task 3: Live Metadata-Only Preflight and Handoff

**Files:**
- Create:
  `experiments/qwen35_hybrid_state/<unique-run-tag>/metadata_preflight.json`
- Create:
  `experiments/qwen35_hybrid_state/<unique-run-tag>/source_manifest.json`
- Modify:
  `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-07-27-qwen35-real-checkpoint-metadata-header-preflight.md`

- [x] Verify the fixed ControlMaster with a non-interactive identity command.
- [x] Run one unique live metadata-header preflight.
- [x] Validate exact remote/local source hashes, two-artifact inventory, topology counts, metadata byte accounting, and zero payload bytes.
- [x] Run focused and adjacent safety regressions, compile, forbidden API scans, worker execution rejection, `git diff --check`, and staged-file check.
- [x] Check all plan items and append exact evidence, limitations, and the next safe gate to the handoff.

## Live Result

Final run tag:

```text
qwen35-metadata-header-preflight-final2-20260727-192004
```

Local evidence:

```text
experiments/qwen35_hybrid_state/
  qwen35-metadata-header-preflight-final2-20260727-192004/
    metadata_preflight.json
    source_manifest.json
```

Classification:

```text
PASS
```

Exact metadata/topology evidence:

```text
metadata_bytes_read: 144024
payload_bytes_read: 0
payload_hashes_recomputed: false
layers: 24
linear-attention layers: 18
full-attention layers: 6
index weights: 632
header tensors: 632
loads: 320
skips: 312
plan/index payload bytes: 4548144832
```

The metadata byte count independently reconstructs as:

```text
config.json: 2908
index JSON: 64460
safetensors prefix: 8
safetensors JSON header: 76648
total: 144024
```

Remote inventory was exactly:

```text
metadata_preflight.json
source_manifest.json
source/tinyvllm/models/qwen35_checkpoint.py
source/tinyvllm/models/qwen35_checkpoint_metadata.py
source/tools/qwen35_real_checkpoint_metadata_preflight.py
```

All three source hashes matched current local files, staged remote files, the
worker record, and the source manifest.

The earlier run:

```text
qwen35-metadata-header-preflight-20260727-191405
```

is retained but superseded. Its metadata result was PASS, but direct-file
imports generated two remote `__pycache__/*.pyc` files and violated the frozen
five-file inventory. A new RED test required
`PYTHONDONTWRITEBYTECODE=1`; the worker command now also uses `python -B`.
The final run contains no bytecode artifacts.

The intermediate run:

```text
qwen35-metadata-header-preflight-final-20260727-191558
```

also passed the metadata and inventory gates, but a later local review found
that the internal worker validated the approved checkpoint path after calling
the reader. A new RED test proved that a manual internal-worker invocation
could read an unapproved directory before rejection. The path validation now
runs before dependency loading or metadata reads. Because that source change
invalidated the earlier source binding, the intermediate run is superseded by
the authoritative `final2` run above.

Exact claim boundary:

```text
approved config/index/header metadata consistency: proven remotely
TinyLLMForge 24-layer tensor-plan construction: proven remotely
safetensors tensor-payload bytes read: zero
retained full-shard SHA256 recomputation: absent
real tensor loading/assignment: not executed
model/CUDA/Engine/inference execution: absent
real checkpoint worker main: still hard-rejected
production CUDA/cache/speed benefit: unmeasured
schema-v2 canonical NO_GO: unchanged
```
